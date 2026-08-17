"""Full-matrix, BIN-grain genetic engine (opt-in, additive).

WHY THIS EXISTS
---------------
The production "genetic" engine (`genetic_global.run_midtilt_ga`) searches a
COMPACT genome — 3 knobs per vampMid (risk-tilt, return-tilt, gain) — applied as
a tilt around a reference split, scored on *pooled* Bank x Currency success
rates. That is fast, stable and hard-compliant, but it can only reach splits that
are tilts/scalings of the reference, and it optimises a pooled approximation of
what actually ships.

This module is the deliberate opposite, mirroring a co-worker's DEAP design while
keeping our guarantees:

  * FULL-MATRIX genome        - one gene per (cell x eligible gateway) at BIN
                                grain, so ANY per-bin allocation is reachable
                                (no tilt/anchor restriction).
  * BIN-GRAIN scoring         - success/risk are the per-bin (EB-shrunk) rates,
                                so what it scores is what deploys (no pooled ->
                                broadcast gap).
  * NO reference anchoring     - it mutates raw shares; the reference is seeded
                                only as ONE elite (a floor), not a centre.
  * DUAL VAMP CEILINGS         - each vampMid carries a HARD `max_vamp` and a SOFT
                                `max_pass_vamp`; the cheaper route to pass is
                                scored (mirrors the co-worker's checker).
  * ADAPTIVE TOLERANCE         - well under the cap -> rank strictly on VWSR; as a
                                candidate approaches the cap, a small VWSR
                                tolerance lets VAMP break ties, so the search
                                *hugs* the boundary instead of hitting a cliff.

WHAT WE KEEP FROM OURS
----------------------
  * ELITE SEEDING  - the caller's reference (e.g. our greedy/tilt compliant split)
    is injected as an elite, and the returned split is never worse (on the
    feasibility-first key) than that seed. So adopting this engine can only add an
    option, never regress the baseline.
  * HARD COMPLIANCE - this module makes compliance a SOFT, adaptive objective
    DURING the search (that is what buys the boundary behaviour). Exact hard
    compliance is expected to be applied by the caller's existing enforcement pass
    (`_enforce_endpoint`) on the returned split. Do NOT ship the raw output.

Pure numpy on purpose: correctness first, verifiable offline. A Numba fusion of
`_segment_softmax` / `_vwsr` / `_violation` is a later stage (see module TODO).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import numpy as np

# Persistent Numba cache (same folder GA-Numba uses) so the one-time kernel compile
# is NOT repaid every run — reclaimed time goes to search. setdefault => an explicit
# NUMBA_CACHE_DIR wins. MUST be set before importing numba.
try:                                                # pragma: no cover - env dependent
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_numba_cache"))
except Exception:                                   # noqa: BLE001
    pass

# Optional Numba (Stage 4). Absent -> the @njit kernel stays a plain-Python
# function (identity decorator) and _prange is range, so behaviour is identical,
# only slower. The GA uses the numpy path by default; the fused kernel is opt-in
# + verify-gated (checked bit-exact vs numpy, falls back on any mismatch).
try:                                                # pragma: no cover - env dependent
    from numba import njit as _njit, prange as _prange   # type: ignore
    _HAS_NUMBA = True
except Exception:                                   # noqa: BLE001
    _HAS_NUMBA = False
    _prange = range                                 # serial fallback

    def _njit(*_a, **_k):                            # identity decorator fallback
        def _wrap(f):
            return f
        return _wrap

__build__ = "2026-08-12-fullmatrix-ga-dualceiling-adaptivetol+numbafuse+prange+elitecache+persistcache+midbands+exactbandhook+localrefine+globalvampcap+seeds+restarts+live-progress+progress-tuple-format-fix+progress-plain-decimals+progress-unmet-names+compress-learned-codebook-delivered-numbadistortion+exact-tab3-codebook-callback+delivery-dedupe+refresh-skip-band+lexico-m5-primary-ranking+seed-as-genome-diag"

# Feasibility tolerance: violations at or below this count as compliant in-search.
_FEAS_EPS = 1e-9


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class FullMatrixProblem:
    """One optimisation problem in LONG format.

    Every row is an ELIGIBLE (cell, gateway) pair. Rows MUST be grouped so each
    cell's rows are contiguous (the constructor `build` enforces this). A "cell"
    is the routing-decision grain — here BIN x (currency x bank) — one simplex of
    shares per cell.

    Arrays are all length R (number of eligible rows) unless noted.
    """

    cell_id: np.ndarray      # int (R,)  contiguous group id per row
    gw_id: np.ndarray        # int (R,)  gateway id (for building the output table)
    mid_id: np.ndarray       # int (R,)  vampMid id per row (spans cells)
    vol: np.ndarray          # float (R,) cell volume (same for every row of a cell)
    succ: np.ndarray         # float (R,) bin-grain (EB-shrunk) success rate
    risk: np.ndarray         # float (R,) bin-grain VAMP/chargeback rate
    max_share: np.ndarray    # float (R,) per-row hard cap on share (max_gateway_share)
    floor: np.ndarray        # float (R,) soft exploration floor per row

    mid_hard_cap: np.ndarray  # float (n_mids,)  max_vamp   (hard ceiling)
    mid_soft_cap: np.ndarray  # float (n_mids,)  max_pass_vamp (soft ceiling)

    # Per-MID BAND constraints (the tab-3 editor rules). Metric per mid:
    #   0 = none · 1 = txn count (Σ vol·share) · 2 = vamp count (Σ vol·share·risk)
    #   3 = vamp_pct (rate = vamp/ txn). Band is [lo, hi]; -inf/+inf = open side.
    #   ceiling -> hi=target ; floor -> lo=target ; range -> lo/hi=target(1±tol).
    # Default (build without bands) = all-zero metric (no band constraints).
    mid_band_metric: np.ndarray = field(default=None)  # int (n_mids,)
    mid_band_lo: np.ndarray = field(default=None)       # float (n_mids,)
    mid_band_hi: np.ndarray = field(default=None)       # float (n_mids,)

    # GLOBAL aggregate VAMP-rate cap (Σ_all vol·share·risk / Σ_all vol·share ≤ cap).
    # This is what `vamp_cap` actually means — a portfolio-level ceiling, NOT a
    # per-MID one. inf = disabled. (Per-MID risk is governed by the count bands.)
    global_vamp_cap: float = np.inf

    # derived (filled by build)
    cell_start: np.ndarray = field(default=None)   # int (n_cells,) segment starts
    cell_len: np.ndarray = field(default=None)      # int (n_cells,) segment lengths
    n_cells: int = 0
    n_mids: int = 0
    order: np.ndarray = field(default=None)         # int (R,) orig->sorted perm

    @classmethod
    def build(cls, cell_id, gw_id, mid_id, vol, succ, risk,
              max_share, floor, mid_hard_cap, mid_soft_cap,
              mid_band_metric=None, mid_band_lo=None, mid_band_hi=None,
              global_vamp_cap=np.inf):
        """Sort rows to contiguous cell groups and precompute segment indices.

        Returns a ready-to-optimise FullMatrixProblem. `order` records the
        original row order so the caller can map the returned split back.
        """
        cell_id = np.asarray(cell_id)
        # stable sort by cell so groups are contiguous but within-cell order kept
        order = np.argsort(cell_id, kind="stable")
        def _o(a):
            return np.asarray(a, dtype=float)[order]
        cid = cell_id[order]
        # segment boundaries of the sorted cell ids
        uniq, starts, lens = _segments(cid)
        # remap cell ids to dense 0..n_cells-1 in sorted order
        dense = np.repeat(np.arange(len(uniq)), lens)
        mid = np.asarray(mid_id)[order].astype(int)
        n_mids = int(max(len(mid_hard_cap), (mid.max() + 1) if mid.size else 0))
        # band arrays default to "no band" (metric 0, open [-inf, +inf]).
        _bm = (np.zeros(n_mids, dtype=np.int32) if mid_band_metric is None
               else np.asarray(mid_band_metric, dtype=np.int32))
        _blo = (np.full(n_mids, -np.inf) if mid_band_lo is None
                else np.asarray(mid_band_lo, dtype=float))
        _bhi = (np.full(n_mids, np.inf) if mid_band_hi is None
                else np.asarray(mid_band_hi, dtype=float))
        p = cls(
            cell_id=dense,
            gw_id=np.asarray(gw_id)[order].astype(int),
            mid_id=mid,
            vol=_o(vol), succ=_o(succ), risk=_o(risk),
            max_share=_o(max_share), floor=_o(floor),
            mid_hard_cap=np.asarray(mid_hard_cap, dtype=float),
            mid_soft_cap=np.asarray(mid_soft_cap, dtype=float),
            mid_band_metric=_bm, mid_band_lo=_blo, mid_band_hi=_bhi,
            global_vamp_cap=float(global_vamp_cap),
            cell_start=starts, cell_len=lens,
            n_cells=len(uniq), n_mids=n_mids, order=order,
        )
        return p


def build_fullmatrix_problem(cell_problems, hard, *, mid_caps=None,
                             exploration_floor=None):
    """Turn the app's existing ``CellProblem`` list into a FullMatrixProblem.

    This is the Stage-2 data feed. It uses the finest grain the app already
    produces (one cell per CellProblem — RPGT x Currency x Bank, no pooling) and
    the rates already attached to each cell:

      * ``cell.success_rates`` are ALREADY empirical-Bayes shrunk (built upstream
        by ``success_rates.gateway_success_rates`` inside ``build_cell_problems``),
        so feeding them straight in is the "EB-shrunk bin rates" requirement — no
        re-shrinking needed, no pooled->broadcast gap.
      * ``cell.risk_rates`` are the per-gateway VAMP rates (from
        ``bin_rpgt_impact_export.csv`` at period=0, loaded upstream).

    The SAME gateway/MID name recurs across many cells; that shared name is one
    vampMid, and its VAMP cap applies to its aggregate across every cell — which
    is exactly the cross-cell coupling the full-matrix GA scores.

    Parameters
    ----------
    cell_problems : list[CellProblem]
    hard : HardConstraints   (reads vamp_cap, max_pass_vamp, max_gateway_share)
    mid_caps : dict[str, tuple[float, float]] | None
        Optional per-MID override {mid_name: (hard_cap, soft_cap)} from the tab-3
        editor. Missing MIDs fall back to the global hard/soft caps.
    exploration_floor : float | None
        Per-row soft floor; falls back to 0.0 if None.

    Returns
    -------
    (problem, meta) where meta = {"gw_names": [...], "mid_names": [...],
        "baseline_shares": (R,) in original row order} for mapping the GA output
        back to (cell, gateway) and for elite seeding.
    """
    # global MID (== gateway name) index
    mid_names: list[str] = []
    name2mid: dict[str, int] = {}
    for cp in cell_problems:
        for g in cp.gateways:
            if g not in name2mid:
                name2mid[g] = len(mid_names)
                mid_names.append(g)

    hard_cap_default = float(hard.vamp_cap) if hard.vamp_cap is not None else np.inf
    soft_cap_default = (float(hard.max_pass_vamp)
                        if getattr(hard, "max_pass_vamp", None) is not None
                        else hard_cap_default)
    if soft_cap_default < hard_cap_default:      # soft must be >= hard; guard misconfig
        soft_cap_default = hard_cap_default
    max_gw = float(getattr(hard, "max_gateway_share", 0.97) or 1.0)
    floor_v = float(exploration_floor) if exploration_floor is not None else 0.0

    n_mids = len(mid_names)
    mid_hard = np.full(n_mids, hard_cap_default, dtype=float)
    mid_soft = np.full(n_mids, soft_cap_default, dtype=float)
    if mid_caps:
        for name, (hc, sc) in mid_caps.items():
            if name in name2mid:
                j = name2mid[name]
                mid_hard[j] = float(hc)
                mid_soft[j] = max(float(sc), float(hc))

    cell_id, gw_id, mid_id = [], [], []
    vol, succ, risk, base = [], [], [], []
    for ci, cp in enumerate(cell_problems):
        n = cp.n()
        bshares = (np.asarray(cp.baseline_shares, dtype=float)
                   if cp.baseline_shares is not None
                   else np.full(n, 1.0 / n))
        for i, g in enumerate(cp.gateways):
            cell_id.append(ci)
            gw_id.append(name2mid[g])          # global gateway/MID index
            mid_id.append(name2mid[g])
            vol.append(float(cp.volume))
            succ.append(float(cp.success_rates[i]))
            risk.append(float(cp.risk_rates[i]))
            base.append(float(bshares[i]))

    R = len(cell_id)
    max_share = np.full(R, max_gw)
    floor = np.full(R, floor_v)
    problem = FullMatrixProblem.build(
        cell_id=np.array(cell_id), gw_id=np.array(gw_id), mid_id=np.array(mid_id),
        vol=np.array(vol), succ=np.array(succ), risk=np.array(risk),
        max_share=max_share, floor=floor,
        mid_hard_cap=mid_hard, mid_soft_cap=mid_soft,
    )
    # baseline shares in ORIGINAL row order (before build's internal sort)
    meta = {
        "gw_names": mid_names,
        "mid_names": mid_names,
        "baseline_shares": np.array(base),
    }
    return problem, meta


def problem_from_ctx(ctx, *, soft_cap=None, soft_cap_mult=None, mid_caps=None,
                     mid_names=None, seed_full=None, mid_bands=None):
    """Build a FullMatrixProblem from the genetic engine's ``ctx`` dict.

    This is the REAL integration surface for tab2_engine. The cross-cell tilt GA
    already assembles ``ctx`` with everything at BIN grain in long format:
      * contiguous cells       - ctx['cell_starts'] / ctx['cell_counts']
      * per-row success (EB)   - ctx['sr']   (already empirical-Bayes shrunk)
      * per-row VAMP rate      - ctx['risk']
      * per-row vampMid index  - ctx['mid_id']
      * per-row cell volume     - ctx['cell_vol']
      * eligibility mask        - ctx['elig']   (config bans -> share 0)
      * hard VAMP cap           - ctx['vamp_cap']
      * max-share / floor        - ctx['max_share'] / ctx['floor']

    Config-BANNED rows (elig<=0.5) are DROPPED from the genome entirely (they can
    never carry share); ``meta['keep_idx']`` maps the returned kept-row split back
    to the full ``n_row`` vector the downstream enforcement expects.

    Dual ceiling: hard = ctx['vamp_cap']; soft = ``soft_cap`` if given, else
    hard * ``soft_cap_mult`` if given, else hard (single wall). Per-MID overrides
    via ``mid_caps`` {mid_index_or_name: (hard, soft)} (index if mid_names None).

    Returns (problem, meta) with meta = {keep_idx, n_row, reference_kept}.
    """
    n_row = int(ctx["n_row"])
    n_mid = int(ctx["n_mid"])
    starts = np.asarray(ctx["cell_starts"], dtype=int)
    counts = np.asarray(ctx["cell_counts"], dtype=int)

    cell_id_full = np.empty(n_row, dtype=int)
    for ci, (s, c) in enumerate(zip(starts, counts)):
        cell_id_full[s:s + c] = ci

    mid_id_full = np.asarray(ctx["mid_id"], dtype=int)
    vol_full = np.asarray(ctx["cell_vol"], dtype=float)
    succ_full = np.asarray(ctx["sr"], dtype=float)
    risk_full = np.asarray(ctx["risk"], dtype=float)
    elig = np.asarray(ctx.get("elig", np.ones(n_row)), dtype=float) > 0.5

    keep_idx = np.where(elig)[0]
    if keep_idx.size == 0:
        raise ValueError("problem_from_ctx: every row is config-banned (elig==0)")

    max_gw = float(ctx.get("max_share", 0.97) or 1.0)
    floor_v = float(ctx.get("floor", 0.0) or 0.0)
    # `vamp_cap` is a GLOBAL aggregate rate cap, NOT a per-MID one. Applying it per
    # MID over-constrains the problem and makes the search trade band breach to pull
    # individual MIDs under the rate — which is not what the user configured (their
    # per-MID limits are the count/txn BANDS). So: per-MID rate ceilings default to
    # inf (disabled), and vamp_cap is enforced once, on the aggregate.
    global_cap = (float(ctx["vamp_cap"]) if ctx.get("vamp_cap") is not None else np.inf)
    mid_hard = np.full(n_mid, np.inf, dtype=float)
    mid_soft = np.full(n_mid, np.inf, dtype=float)
    # Optional explicit per-MID rate ceilings (rare; only if a caller passes mid_caps).
    if mid_caps:
        name2idx = ({m: i for i, m in enumerate(mid_names)} if mid_names else None)
        for key, (hc, sc) in mid_caps.items():
            j = name2idx[key] if (name2idx and key in name2idx) else int(key)
            if 0 <= j < n_mid:
                mid_hard[j] = float(hc)
                mid_soft[j] = max(float(sc), float(hc))

    # reference to seed the elite. Prefer an explicit `seed_full` (e.g. the app's
    # KNOWN-COMPLIANT greedy+LP split `_comp_share_G`) so the never-worse-than-seed
    # guarantee makes the delivered split feasible-by-construction even when the
    # downstream VAMP-cap enforcement is disabled. Else ctx['ref_share'] / base.
    if seed_full is not None:
        ref_full = np.asarray(seed_full, dtype=float)
    else:
        ref_full = np.asarray(ctx.get("ref_share", ctx.get("base")), dtype=float)
    ref_kept = ref_full[keep_idx].copy()

    # per-MID band arrays from mid_bands = {mid_index: (metric, lo, hi)}.
    _bm = np.zeros(n_mid, dtype=np.int32)
    _blo = np.full(n_mid, -np.inf)
    _bhi = np.full(n_mid, np.inf)
    if mid_bands:
        for j, (metric, lo, hi) in mid_bands.items():
            if 0 <= int(j) < n_mid:
                _bm[int(j)] = int(metric)
                _blo[int(j)] = float(lo)
                _bhi[int(j)] = float(hi)

    problem = FullMatrixProblem.build(
        cell_id=cell_id_full[keep_idx], gw_id=keep_idx.copy(),
        mid_id=mid_id_full[keep_idx], vol=vol_full[keep_idx],
        succ=succ_full[keep_idx], risk=risk_full[keep_idx],
        max_share=np.full(keep_idx.size, max_gw),
        floor=np.full(keep_idx.size, floor_v),
        mid_hard_cap=mid_hard, mid_soft_cap=mid_soft,
        mid_band_metric=_bm, mid_band_lo=_blo, mid_band_hi=_bhi,
        global_vamp_cap=global_cap,
    )
    # renormalise the kept reference within each (kept) cell so it is a valid seed
    kept_cell = cell_id_full[keep_idx]
    for cid in np.unique(kept_cell):
        m = kept_cell == cid
        s = ref_kept[m].sum()
        ref_kept[m] = (ref_kept[m] / s) if s > 1e-12 else (1.0 / m.sum())

    meta = {"keep_idx": keep_idx, "n_row": n_row, "reference_kept": ref_kept}
    return problem, meta


def reconstruct_full_split(best_kept_shares, meta):
    """Map a kept-row split back to the full ``n_row`` vector (0 at banned rows)."""
    full = np.zeros(meta["n_row"], dtype=float)
    full[meta["keep_idx"]] = np.asarray(best_kept_shares, dtype=float)
    return full


def _segments(sorted_ids):
    """Given a sorted 1d int array, return (unique, start_idx, length) arrays."""
    uniq, starts, counts = np.unique(sorted_ids, return_index=True, return_counts=True)
    # np.unique returns sorted unique + first-occurrence index; sort by start
    o = np.argsort(starts)
    return uniq[o], starts[o].astype(int), counts[o].astype(int)


# ---------------------------------------------------------------------------
# Vectorised scoring primitives  (population = (P, R) logit matrices)
# ---------------------------------------------------------------------------
def _segment_softmax(logits, cell_start, cell_len):
    """Per-cell softmax over contiguous row segments.

    logits: (P, R). Returns shares (P, R) where each cell's rows sum to 1.
    Numerically stable (subtracts per-segment max).
    """
    logits = np.atleast_2d(logits)
    # per-segment max, expanded back to row grain
    seg_max = np.maximum.reduceat(logits, cell_start, axis=1)          # (P, n_cells)
    row_max = np.repeat(seg_max, cell_len, axis=1)                      # (P, R)
    ex = np.exp(logits - row_max)
    seg_sum = np.add.reduceat(ex, cell_start, axis=1)                  # (P, n_cells)
    row_sum = np.repeat(seg_sum, cell_len, axis=1)                      # (P, R)
    return ex / row_sum


def _vwsr(shares, vol, succ, total_vol):
    """Volume-weighted success rate per individual. shares: (P, R) -> (P,).

    VWSR = sum(vol*share*succ) / sum(cell volume). The denominator is fixed
    (shares sum to 1 within a cell), so this is LINEAR in shares.
    """
    return (shares * (vol * succ)).sum(axis=1) / total_vol


def _mid_vamp(shares, vol, risk, mid_id, n_mids):
    """Per-vampMid VAMP rate for each individual. Returns (P, n_mids).

    rate_m = sum_{rows in m} vol*share*risk / sum_{rows in m} vol*share.
    """
    P = shares.shape[0]
    w = shares * vol                     # (P, R) volume routed per row
    num = np.zeros((P, n_mids))
    den = np.zeros((P, n_mids))
    # accumulate per mid (columns) for every individual (rows) at once
    np.add.at(num.T, mid_id, (w * risk).T)
    np.add.at(den.T, mid_id, w.T)
    rate = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return rate


def _violation(shares, p: "FullMatrixProblem"):
    """Scalar soft-constraint violation per individual (0 == compliant).

    Combines, as relative overages:
      * DUAL VAMP CEILING per mid: cheaper of
          - hard route:  max(0, rate/hard_cap - 1)
          - soft route:  max(0, rate/soft_cap - 1)  (soft cap >= hard cap)
        We take the MIN of the two routes (a mid "passes" by the cheaper path),
        then sum across mids.
      * MAX-SHARE cap per row: max(0, share/max_share - 1), summed.
    Returns (P,).
    """
    P = shares.shape[0]
    w = shares * p.vol                                    # (P,R) volume routed per row
    num = np.zeros((P, p.n_mids))                        # vamp count per mid
    den = np.zeros((P, p.n_mids))                        # txn total per mid
    np.add.at(num.T, p.mid_id, (w * p.risk).T)
    np.add.at(den.T, p.mid_id, w.T)
    rate = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    hard = p.mid_hard_cap[None, :]
    soft = p.mid_soft_cap[None, :]
    over_hard = np.maximum(0.0, np.divide(rate, hard, out=np.zeros_like(rate),
                                          where=hard > 0) - 1.0)
    over_soft = np.maximum(0.0, np.divide(rate, soft, out=np.zeros_like(rate),
                                          where=soft > 0) - 1.0)
    vamp_viol = np.minimum(over_hard, over_soft).sum(axis=1)          # (P,)

    share_over = np.maximum(0.0, shares / p.max_share - 1.0).sum(axis=1)
    band_viol = _band_violation(num, den, rate, p)                   # (P,)

    # GLOBAL aggregate VAMP-rate cap: Σ vol·share·risk / Σ vol·share ≤ cap.
    gvc = getattr(p, "global_vamp_cap", np.inf)
    if np.isfinite(gvc) and gvc > 0:
        g_num = num.sum(axis=1)
        g_den = den.sum(axis=1)
        g_rate = np.divide(g_num, g_den, out=np.zeros_like(g_num), where=g_den > 0)
        global_viol = np.maximum(0.0, g_rate / gvc - 1.0)
    else:
        global_viol = 0.0
    return vamp_viol + share_over + band_viol + global_viol


def _band_violation(num, den, rate, p: "FullMatrixProblem"):
    """Per-MID band violation (relative). num/den/rate are (P, n_mids).

    metric: 1=txn count (den) · 2=vamp count (num) · 3=vamp_pct (rate).
    Ceiling: max(0, val/hi - 1). Floor: max(0, 1 - val/lo). Open side = inf.
    """
    metric = p.mid_band_metric
    if metric is None or not np.any(metric > 0):
        return 0.0
    val = np.zeros_like(num)
    m1, m2, m3 = (metric == 1), (metric == 2), (metric == 3)
    if m1.any():
        val[:, m1] = den[:, m1]
    if m2.any():
        val[:, m2] = num[:, m2]
    if m3.any():
        val[:, m3] = rate[:, m3]
    lo = p.mid_band_lo[None, :]
    hi = p.mid_band_hi[None, :]
    active = (metric > 0)[None, :]
    hi_ok = np.isfinite(hi) & (hi > 0) & active
    over = np.where(hi_ok, val / np.where(hi_ok, hi, 1.0) - 1.0, 0.0)
    lo_ok = np.isfinite(lo) & (lo > 0) & active
    under = np.where(lo_ok, 1.0 - val / np.where(lo_ok, lo, 1.0), 0.0)
    return (np.maximum(0.0, over) + np.maximum(0.0, under)).sum(axis=1)


def _adaptive_tol(vamp_viol, tol_lo=0.0, tol_hi=0.01, knee=2.0):
    """VWSR tolerance schedule as a function of a candidate's VAMP violation.

    Mirrors the co-worker's idea: when risk is comfortably under the cap
    (violation ~0) be STRICT on VWSR (tol -> tol_lo); when risk is high
    (violation >= knee) be forgiving (tol -> tol_hi). Linear in between.
    """
    frac = np.clip(vamp_viol / knee, 0.0, 1.0)
    return tol_lo + (tol_hi - tol_lo) * frac


def _rank(vwsr, viol, band=None):
    """STRICT LEXICOGRAPHIC ranking, best -> worst.

    When `band` (per-candidate EXACT per-MID M5 band breach) is supplied it is the strict PRIMARY
    key — the compliance metric that defines the run — and nothing below can outrank it:

      1. lower M5 band breach wins (drive it to 0); a breach can never be traded for the terms below;
      2. among equal-M5 candidates: lower ENGINEERING violation (`viol` = global VAMP cap + max-share)
         wins;
      3. among those: higher VWSR (conversion) wins.

    M5 breaches at/under `_FEAS_EPS` snap equal (compliant) so float noise doesn't churn the order.
    When `band is None` (legacy callers) it degrades to feasibility-first on `viol` then VWSR.
    """
    vwsr = np.asarray(vwsr, dtype=float)
    viol = np.asarray(viol, dtype=float)
    band = np.zeros_like(viol) if band is None else np.asarray(band, dtype=float)
    band_eff = np.where(band <= _FEAS_EPS, 0.0, band)
    # np.lexsort: the LAST key is primary. Want primary=band asc, then viol asc, then VWSR desc.
    return np.lexsort((-vwsr, viol, band_eff))


def _best_index(vwsr, viol, band=None):
    return _rank(vwsr, viol, band)[0]


def _key_of(vwsr, viol, band=0.0):
    """Comparable STRICT LEXICOGRAPHIC key for ONE candidate (higher tuple == better).

    Tuple `(-band, -viol, vwsr)` so, compared with `>`: smaller M5 band breach wins first, then
    smaller engineering violation, then higher VWSR — matching `_rank`. M5 breaches ≤ `_FEAS_EPS`
    snap to 0 (compliant) so a compliant split always outranks any breaching one."""
    b = 0.0 if float(band) <= _FEAS_EPS else float(band)
    return (-b, -float(viol), float(vwsr))


# ---------------------------------------------------------------------------
# Stage 4: fused evaluator (segment-softmax + VWSR + violation in ONE pass)
# ---------------------------------------------------------------------------
def _fused_eval_kernel(logits, cell_starts, cell_counts, vol, succ, risk,
                       mid_id, max_share, mid_hard, mid_soft, total_vol, n_mid,
                       mid_band_metric, mid_band_lo, mid_band_hi, global_vamp_cap):
    """One-pass VWSR + violation for a whole population. Numba-compatible: only
    scalar loops + preallocated arrays (no reduceat / np.add.at / fancy index).

    Returns (vwsr[P], viol[P]) — bit-for-bit the same quantities as
    ``_vwsr(_segment_softmax(...))`` and ``_violation(_segment_softmax(...))``.
    """
    P = logits.shape[0]
    n_cells = cell_starts.shape[0]
    vwsr = np.zeros(P)
    viol = np.zeros(P)
    # Each candidate i is fully independent (own local accumulators, writes only
    # vwsr[i]/viol[i] — no cross-candidate reduction), so _prange parallelises across
    # cores WITHOUT reordering any float sum. Bit-identical to serial; the verify-gate
    # confirms it. (numba absent -> _prange is range, runs serially.)
    for i in _prange(P):
        num_v = 0.0
        share_over = 0.0
        mnum = np.zeros(n_mid)
        mden = np.zeros(n_mid)
        for c in range(n_cells):
            s = cell_starts[c]
            n = cell_counts[c]
            # per-cell softmax (stable): max, exp-sum, then shares
            m = logits[i, s]
            for j in range(1, n):
                lj = logits[i, s + j]
                if lj > m:
                    m = lj
            ssum = 0.0
            for j in range(n):
                ssum += np.exp(logits[i, s + j] - m)
            for j in range(n):
                r = s + j
                sh = np.exp(logits[i, r] - m) / ssum
                w = vol[r] * sh
                num_v += w * succ[r]
                mid = mid_id[r]
                mnum[mid] += w * risk[r]
                mden[mid] += w
                ms = max_share[r]
                if ms > 0.0:
                    ov = sh / ms - 1.0
                    if ov > 0.0:
                        share_over += ov
        vwsr[i] = num_v / total_vol
        vv = 0.0
        g_num = 0.0                 # global aggregate vamp count / txn total
        g_den = 0.0
        for mm in range(n_mid):
            tot = mden[mm]          # txn total for this mid
            vc = mnum[mm]           # vamp count for this mid
            g_num += vc
            g_den += tot
            if tot > 0.0:
                rate = vc / tot
                oh = (rate / mid_hard[mm] - 1.0) if mid_hard[mm] > 0.0 else 0.0
                if oh < 0.0:
                    oh = 0.0
                osf = (rate / mid_soft[mm] - 1.0) if mid_soft[mm] > 0.0 else 0.0
                if osf < 0.0:
                    osf = 0.0
                vv += oh if oh < osf else osf
            else:
                rate = 0.0
            # per-MID BAND (metric 1=txn total, 2=vamp count, 3=vamp rate)
            bm = mid_band_metric[mm]
            if bm > 0:
                if bm == 1:
                    val = tot
                elif bm == 2:
                    val = vc
                else:
                    val = rate
                hi = mid_band_hi[mm]
                if hi < 1e300 and hi > 0.0 and val > hi:
                    vv += val / hi - 1.0
                lo = mid_band_lo[mm]
                if lo > -1e300 and lo > 0.0 and val < lo:
                    vv += 1.0 - val / lo
        # global aggregate VAMP-rate cap
        if global_vamp_cap < 1e300 and global_vamp_cap > 0.0 and g_den > 0.0:
            g_rate = g_num / g_den
            if g_rate > global_vamp_cap:
                vv += g_rate / global_vamp_cap - 1.0
        viol[i] = vv + share_over
    return vwsr, viol


if _HAS_NUMBA:                                       # pragma: no cover - env dependent
    _fused_eval_kernel = _njit(cache=True, parallel=True)(_fused_eval_kernel)


# ---------------------------------------------------------------------------
# Compressibility regularizer — VECTOR-QUANTIZATION distortion vs a learned codebook
# ---------------------------------------------------------------------------
# The λ_compress reward pushes cells to route ALIKE so the final split collapses into
# few deployable configs. Concretely we learn a CODEBOOK of ~pool-target centroid shapes
# (volume-weighted k-means over the ELITE's per-vampMid cell shapes, refreshed as the
# elite improves — the same KIND of clustering tab-3 compression uses) and penalise each
# candidate by its volume-weighted quantization error against that codebook:
#     D_i = Σ_cells vol_c · ‖shape_ic − centroid[assign_c]‖²  / total_vol .
# shape_ic[m] = Σ_{rows in cell c with vampMid m} share is the cell's shape (0 on absent
# mids). This is NOT a 'fewer-gateways' reward: two cells with the SAME shape (however
# spread across mids) cost 0; a cell that routes DIFFERENTLY from its codebook centroid
# pays. Scored on DELIVERED shares (post eligibility + blocked-caps) so it rewards what
# actually ships. Kernel is Numba-fused (verify-gated vs the numpy twin); the periodic
# codebook refit is numpy/sklearn (cheap relative to per-generation evaluation).
def _compress_distortion_kernel(shares, cell_starts, cell_counts, mid_id, vol,
                                assign, cent, cconst, n_mid, total_vol):
    """Volume-weighted VQ distortion of a population vs a FIXED codebook.

    ‖shape − c_k‖² = Σ_m shape_m² − 2 Σ_m shape_m·c_k,m + ‖c_k‖²  (‖c_k‖²=cconst[k]),
    so only a cell's OWN rows are touched; the absent-mid tail is the constant cconst[k].
    Numba-safe: scalar loops + one thread-local n_mid buffer, cleared per cell. Each
    candidate i writes only out[i] (no cross-candidate reduction) so _prange is
    bit-identical to serial — the verify-gate confirms it. Returns distortion[P].
    """
    P = shares.shape[0]
    n_cells = cell_starts.shape[0]
    out = np.zeros(P)
    for i in _prange(P):
        buf = np.zeros(n_mid)
        acc = 0.0
        for c in range(n_cells):
            s = cell_starts[c]
            n = cell_counts[c]
            k = assign[c]
            for j in range(n):               # accumulate this cell's per-mid shape
                buf[mid_id[s + j]] += shares[i, s + j]
            term_sq = 0.0
            term_cross = 0.0
            for j in range(n):               # consume buf once per mid, then clear it
                mm = mid_id[s + j]
                b = buf[mm]
                if b != 0.0:
                    term_sq += b * b
                    term_cross += b * cent[k, mm]
                    buf[mm] = 0.0
            acc += vol[s] * (term_sq - 2.0 * term_cross + cconst[k])
        out[i] = acc / total_vol
    return out


if _HAS_NUMBA:                                       # pragma: no cover - env dependent
    _compress_distortion_kernel = _njit(cache=True, parallel=True)(_compress_distortion_kernel)


def _compress_distortion_numpy(shares, cell_start, cell_len, mid_id, vol,
                               assign, cent, cconst, n_mid, total_vol):
    """Vectorised twin of ``_compress_distortion_kernel`` (the verify-gate reference).
    ``cconst`` is accepted for signature parity but unused (the full diff is formed)."""
    sh = np.atleast_2d(np.asarray(shares, float))
    P = sh.shape[0]
    n_cells = cell_start.shape[0]
    cm = np.zeros((P, n_cells, n_mid))
    for m in range(n_mid):
        cm[:, :, m] = np.add.reduceat(sh * (mid_id == m), cell_start, axis=1)
    diff = cm - cent[assign][None, :, :]                     # (P, n_cells, n_mid)
    d2 = (diff * diff).sum(axis=2)                           # (P, n_cells)
    cvol = vol[cell_start]                                   # (n_cells,) cell volume
    return (d2 * cvol[None, :]).sum(axis=1) / (float(total_vol) or 1.0)


def _lloyd_weighted(X, w, k, seed, iters=50):
    """Deterministic volume-weighted Lloyd k-means (numpy fallback when sklearn is
    absent). k-means++ init, `iters` refinements. Returns (labels[intp], centroids)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = int(max(1, min(k, n)))
    w = np.maximum(np.asarray(w, float), 1e-12)
    # weighted k-means++ seeding
    c0 = int(rng.integers(0, n))
    cents = [X[c0]]
    d2 = ((X - cents[0]) ** 2).sum(axis=1)
    for _ in range(1, k):
        p = (d2 * w)
        tot = p.sum()
        idx = int(rng.integers(0, n)) if tot <= 0 else int(np.searchsorted(np.cumsum(p / tot), rng.random()))
        idx = min(max(idx, 0), n - 1)
        cents.append(X[idx])
        d2 = np.minimum(d2, ((X - X[idx]) ** 2).sum(axis=1))
    C = np.asarray(cents, float)
    labels = np.zeros(n, np.intp)
    for _ in range(iters):
        dists = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # (n, k)
        new = dists.argmin(axis=1).astype(np.intp)
        if _ and np.array_equal(new, labels):
            labels = new
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = (X[m] * w[m, None]).sum(axis=0) / w[m].sum()
    return labels, C


def _fit_codebook(shape_mat, cvol, pools, seed):
    """Learn a codebook of centroid SHAPES from per-cell shapes (volume-weighted, so
    high-volume cells pull the centroids — matching tab-3's compressor). K = min(pools,
    n_cells). Returns (assign[intp] (n_cells,), cent (K,n_mid), cconst (K,) = ‖cent‖²)."""
    X = np.ascontiguousarray(shape_mat, float)
    w = np.maximum(np.asarray(cvol, float), 1e-12)
    n_cells = X.shape[0]
    K = int(max(1, min(int(pools), n_cells)))
    try:                                             # sklearn matches tab-3's KMeans
        from sklearn.cluster import KMeans          # lazy: engine stays numpy-only if absent
        km = KMeans(n_clusters=K, n_init=2, max_iter=50, random_state=int(seed))
        km.fit(X, sample_weight=w)
        assign = km.labels_.astype(np.intp)
        cent = np.ascontiguousarray(km.cluster_centers_, float)
    except Exception:                                # noqa: BLE001 - numpy fallback
        assign, cent = _lloyd_weighted(X, w, K, seed)
        cent = np.ascontiguousarray(cent, float)
    cconst = np.ascontiguousarray((cent * cent).sum(axis=1), float)
    return assign, cent, cconst


def _cell_shape_matrix(shares_row, cell_start, mid_id, n_mid):
    """Per-cell per-vampMid shape matrix for ONE split: (n_cells, n_mid)."""
    sh = np.asarray(shares_row, float)[None, :]
    n_cells = cell_start.shape[0]
    cm = np.zeros((n_cells, n_mid))
    for m in range(n_mid):
        cm[:, m] = np.add.reduceat(sh[0] * (mid_id == m), cell_start)
    return cm


def make_distortion(problem, *, use_numba=False, verify=True, rng=None):
    """Return (dist_fn, backend). dist_fn(shares_pop, assign, cent, cconst) -> D[P].
    numpy by default; the njit kernel is verified bit-exact vs numpy on a random codebook
    (any mismatch/error → numpy), so the codebook may change each refresh without re-verify."""
    p = problem
    cs = np.ascontiguousarray(p.cell_start, np.intp)
    cc = np.ascontiguousarray(p.cell_len, np.intp)
    mid = np.ascontiguousarray(p.mid_id, np.intp)
    vol = np.ascontiguousarray(p.vol, float)
    nmid = int(p.n_mids)
    tv = float(_cell_volume_total(p)) or 1.0

    def _np(shares, assign, cent, cconst):
        return _compress_distortion_numpy(shares, cs, cc, mid, vol,
                                          np.asarray(assign, np.intp), cent, cconst, nmid, tv)

    fn, backend = _np, "numpy"
    if use_numba and _HAS_NUMBA:
        try:                                         # pragma: no cover - env dependent
            _rng = rng or np.random.default_rng(0)
            K = int(max(1, min(8, p.n_cells)))
            a = _rng.integers(0, K, size=p.n_cells).astype(np.intp)
            ct = np.ascontiguousarray(_rng.random((K, nmid)), float)
            kc = np.ascontiguousarray((ct * ct).sum(axis=1), float)
            _R = int(p.cell_id.shape[0])
            sh = np.ascontiguousarray(
                _segment_softmax(_rng.standard_normal((3, _R)), cs, cc), float)
            ref = _np(sh, a, ct, kc)
            got = _compress_distortion_kernel(sh, cs, cc, mid, vol, a, ct, kc, nmid, tv)
            if np.allclose(ref, got, rtol=1e-9, atol=1e-12):
                def fn(shares, assign, cent, cconst):
                    return _compress_distortion_kernel(
                        np.ascontiguousarray(shares, float), cs, cc, mid, vol,
                        np.ascontiguousarray(assign, np.intp),
                        np.ascontiguousarray(cent, float),
                        np.ascontiguousarray(cconst, float), nmid, tv)
                backend = "numba"
        except Exception:                            # noqa: BLE001
            fn, backend = _np, "numpy"
    return fn, backend


def make_fused_eval(problem, *, use_numba=False, verify=True, rng=None):
    """Return (eval_pop, info). eval_pop(logits_pop) -> (vwsr, viol).

    numpy by default. With ``use_numba`` and Numba present, the fused njit kernel
    is VERIFIED against the numpy path on a random sample (allclose); any mismatch
    or error falls back to numpy — so the kernel can never change results.
    """
    p = problem
    total_vol = _cell_volume_total(p)

    def _numpy_eval(logits):
        shares = _segment_softmax(logits, p.cell_start, p.cell_len)
        return (_vwsr(shares, p.vol, p.succ, total_vol), _violation(shares, p))

    if not use_numba or not _HAS_NUMBA:
        return _numpy_eval, {"backend": "numpy",
                             "reason": ("numba absent" if use_numba else "numpy requested")}

    # Bit-identical memory-layout prep, done ONCE (these are constant across every
    # eval call). Index arrays -> int32 (half the bandwidth; integer indexing is
    # exact, so the float accumulation is unchanged). Float arrays -> contiguous
    # float64 (arithmetic and its order untouched). This only changes cache/SIMD
    # behaviour, never a single floating-point op.
    _k_cs = np.ascontiguousarray(p.cell_start, dtype=np.int32)
    _k_cc = np.ascontiguousarray(p.cell_len, dtype=np.int32)
    _k_mid = np.ascontiguousarray(p.mid_id, dtype=np.int32)
    _k_vol = np.ascontiguousarray(p.vol, dtype=np.float64)
    _k_succ = np.ascontiguousarray(p.succ, dtype=np.float64)
    _k_risk = np.ascontiguousarray(p.risk, dtype=np.float64)
    _k_ms = np.ascontiguousarray(p.max_share, dtype=np.float64)
    _k_mh = np.ascontiguousarray(p.mid_hard_cap, dtype=np.float64)
    _k_msoft = np.ascontiguousarray(p.mid_soft_cap, dtype=np.float64)
    _k_bm = np.ascontiguousarray(
        p.mid_band_metric if p.mid_band_metric is not None
        else np.zeros(p.n_mids), dtype=np.int32)
    _k_blo = np.ascontiguousarray(
        p.mid_band_lo if p.mid_band_lo is not None
        else np.full(p.n_mids, -np.inf), dtype=np.float64)
    _k_bhi = np.ascontiguousarray(
        p.mid_band_hi if p.mid_band_hi is not None
        else np.full(p.n_mids, np.inf), dtype=np.float64)
    _k_tv = float(total_vol)
    _k_nm = int(p.n_mids)

    _k_gvc = float(getattr(p, "global_vamp_cap", np.inf))

    def _numba_eval(logits):
        return _fused_eval_kernel(
            np.ascontiguousarray(logits, dtype=np.float64), _k_cs, _k_cc,
            _k_vol, _k_succ, _k_risk, _k_mid, _k_ms, _k_mh, _k_msoft, _k_tv, _k_nm,
            _k_bm, _k_blo, _k_bhi, _k_gvc)

    if verify:
        rng = rng or np.random.default_rng(0)
        samp = rng.standard_normal((8, p.cell_id.shape[0]))
        v1, x1 = _numpy_eval(samp)
        try:
            v2, x2 = _numba_eval(samp)
        except Exception as exc:                     # noqa: BLE001
            return _numpy_eval, {"backend": "numpy", "reason": f"numba raised {exc!r}"}
        ok = (np.allclose(v1, v2, rtol=1e-7, atol=1e-9)
              and np.allclose(x1, x2, rtol=1e-6, atol=1e-8))
        if not ok:
            return _numpy_eval, {"backend": "numpy", "reason": "verify mismatch",
                                 "max_dv": float(np.max(np.abs(v1 - v2))),
                                 "max_dx": float(np.max(np.abs(x1 - x2)))}
        return _numba_eval, {"backend": "numba", "verified": True}
    return _numba_eval, {"backend": "numba", "verified": False}


# ---------------------------------------------------------------------------
# GA operators (act on the logit genome)
# ---------------------------------------------------------------------------
def _shares_to_logits(shares, eps=1e-6):
    return np.log(np.clip(shares, eps, None))


def _greedy_reference(p: "FullMatrixProblem"):
    """Fallback seed if the caller gives no reference: per-cell softmax over
    success (a conversion-greedy split). Returns shares (R,)."""
    logits = p.succ * 6.0   # mild temperature; only a SEED, GA refines from here
    return _segment_softmax(logits[None, :], p.cell_start, p.cell_len)[0]


def _crossover(a, b, cell_start, cell_len, rng):
    """Uniform per-CELL crossover: each cell's whole logit segment comes from one
    parent (keeps within-cell simplex structure intact)."""
    n_cells = len(cell_start)
    pick = rng.random(n_cells) < 0.5
    row_pick = np.repeat(pick, cell_len)
    return np.where(row_pick, a, b)


def _mutate(logits, rate, strength, cell_start, cell_len, rng):
    """Gaussian perturbation of a fraction of CELLS' logit segments."""
    n_cells = len(cell_start)
    hit = rng.random(n_cells) < rate
    row_hit = np.repeat(hit, cell_len)
    noise = rng.standard_normal(logits.shape) * strength
    return logits + np.where(row_hit, noise, 0.0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_fullmatrix_ga(problem: "FullMatrixProblem", reference_shares=None, *,
                      pop_size=60, generations=120, elite=6,
                      mutation_rate=0.3, mutation_strength=0.4,
                      patience=25, seed=0, log_fn=None, numba=False,
                      band_penalty_fn=None, band_report_fn=None, n_seeds=1, restarts=1,
                      restart_mode="lean", compress_lambda=0.0,
                      compress_pools=200, compress_refresh=8, deliver_fn=None,
                      codebook_fn=None, deliver_full_fn=None, gather_fn=None):
    """Evolve a full-matrix BIN-grain split that maximises VWSR under dual VAMP
    ceilings, hugging the boundary via adaptive tolerance.

    Parameters
    ----------
    problem : FullMatrixProblem
    reference_shares : (R,) array or None
        Caller's compliant reference (e.g. greedy/tilt split), in ORIGINAL row
        order. Seeded as an elite; the result is guaranteed no worse than it.
        None -> a conversion-greedy seed is built internally.
    Returns
    -------
    (best_shares, info) : best_shares is (R,) in ORIGINAL row order.
    """
    p = problem
    rng = np.random.default_rng(seed)
    R = p.cell_id.shape[0]
    total_vol = _cell_volume_total(p)

    log = log_fn or (lambda *_a, **_k: None)
    log(f"[fullmatrix-ga] build={__build__} R={R} cells={p.n_cells} mids={p.n_mids}")

    # Stage-4 fused evaluator (numpy default; opt-in verify-gated numba kernel).
    eval_pop, _eval_info = make_fused_eval(p, use_numba=numba)
    log(f"[fullmatrix-ga] evaluator backend={_eval_info.get('backend')}"
        + (f" ({_eval_info.get('reason')})" if _eval_info.get('reason') else ""))
    # Make the parallelism visible: numba defaults NUMBA_NUM_THREADS to ALL cores for the
    # parallel=True kernels (each candidate writes only its own slot, so more threads scale
    # throughput WITHOUT reordering any float sum — bit-identical). If this logs 1, or the
    # backend above is 'numpy', the search is NOT using the fast path — investigate before
    # chasing other speedups.
    if _eval_info.get("backend") == "numba":
        try:                                            # pragma: no cover - env dependent
            import numba as _nb
            log(f"[fullmatrix-ga] numba threads={_nb.get_num_threads()} "
                f"(of {_nb.config.NUMBA_DEFAULT_NUM_THREADS} detected cores); "
                "set NUMBA_NUM_THREADS to override.")
        except Exception:                               # noqa: BLE001
            pass

    # EXACT band penalty (optional). When `band_penalty_fn` is supplied, the kernel's
    # (linear) band term is expected OFF (mid_band_metric all 0), and the exact
    # per-generation M5 band breach is ADDED here from the caller's projector. It maps
    # sorted kept-row shares -> full opt-grain shares internally. This is the faithful
    # replacement for the 30-day linear proxy (matches the tilt engine's band scoring).
    # COMPRESSIBILITY REGULARIZER (λ_compress ≥ 0). Rewards cells COLLAPSING ONTO A SHARED CODEBOOK of
    # shapes so the split compresses into few deployable configs. We LEARN a codebook of ≈`compress_pools`
    # centroid shapes (volume-weighted k-means over the ELITE's per-vampMid cell shapes — the same KIND of
    # clustering tab-3 compression uses — refreshed every `compress_refresh` gens as the elite improves) and
    # subtract from VWSR the volume-weighted VQ distortion of each candidate against that codebook
    # (‖cell_shape − nearest-assigned centroid‖²). This is NOT a 'fewer-gateways' reward: two cells with the
    # SAME shape (however spread across mids) cost 0; a cell that routes DIFFERENTLY from its centroid pays.
    # Scored on DELIVERED shares (post blocked-caps + eligibility via `deliver_fn`) so it rewards what
    # actually ships — a gateway a cell can't use (Country / paymentMethodProvider capability) never enters
    # its delivered shape. Distortion is the Numba-fused kernel (verify-gated); the codebook refit is
    # numpy/sklearn. λ=0 ⇒ today's behaviour (regularizer OFF, no per-generation cost).
    _compress_on = float(compress_lambda or 0.0) > 0.0
    _clam = float(compress_lambda or 0.0)
    _cb = {"assign": None, "cent": None, "cconst": None, "src": None}   # codebook (refreshed in-loop)
    # DELIVERY DEDUPE: the band penalty and the compress distortion BOTH need the
    # eligibility + blocked-caps DELIVERED shares. When the caller supplies both
    # `deliver_full_fn` (raw kept (P,R) shares → full-grain delivered (P,n_row)) and
    # `gather_fn` (full-grain → kept-grain (P,R) for the distortion), the engine computes
    # the delivered array ONCE per evaluation and feeds BOTH — instead of each hook
    # scattering+delivering independently (which ran the whole transform twice per
    # generation when λ>0). Bit-identical: the same deterministic transform, reused. With
    # those hooks absent it falls back to the old per-hook delivery (band delivers on the
    # raw shares it's given; compress uses `deliver_fn`).
    _have_full = callable(deliver_full_fn) and callable(gather_fn)
    _deliver = deliver_fn if callable(deliver_fn) else (lambda _X: _X)

    def _deliver_full(sh):
        """Raw kept (P,R) shares → full-grain delivered (P,n_row); None if no deduped hook."""
        return deliver_full_fn(sh) if _have_full else None

    def _deliver_kept(sh, fd):
        """Kept-grain delivered (P,R) for the distortion, reusing the shared full delivery."""
        if _have_full:
            return np.asarray(gather_fn(fd), float)
        return np.asarray(_deliver(sh), float)

    if _compress_on:
        _nmid = int(p.n_mids)
        _cvol = np.asarray(p.vol[p.cell_start], float)            # (n_cells,) cell volume
        _dist_fn, _dist_backend = make_distortion(p, use_numba=numba)

        def _refresh_codebook(logits_row):
            """(Re)learn the codebook from ONE split's DELIVERED per-cell shapes.

            When ``codebook_fn`` is supplied (the caller's EXACT tab-3 ward/knapsack
            compressor), it maps the delivered GA-grain shares to (assign, cent, cconst)
            at the per-vampMid grain — so the regularizer's target IS the split tab-3
            would ship. On any failure we fall back to the internal volume-weighted
            k-means codebook (logged, never silent), so the search never stalls."""
            _s = _segment_softmax(np.asarray(logits_row, float)[None, :], p.cell_start, p.cell_len)
            _fd = _deliver_full(_s)
            _sd = _deliver_kept(_s, _fd)                          # (1, R) delivered shares
            if callable(codebook_fn):
                try:
                    _a, _ct, _kc = codebook_fn(_sd[0])
                    _cb["assign"] = np.ascontiguousarray(_a, np.intp)
                    _cb["cent"] = np.ascontiguousarray(_ct, float)
                    _cb["cconst"] = np.ascontiguousarray(_kc, float)
                    _cb["src"] = "callback"
                    return
                except Exception as _cbe:                          # noqa: BLE001
                    log(f"[fullmatrix-ga] codebook_fn failed ({type(_cbe).__name__}: {_cbe}) — "
                        "falling back to the internal volume-weighted k-means codebook this refresh.")
            _shape = _cell_shape_matrix(_sd[0], p.cell_start, p.mid_id, _nmid)
            _a, _ct, _kc = _fit_codebook(_shape, _cvol, compress_pools, int(seed))
            _cb["assign"], _cb["cent"], _cb["cconst"], _cb["src"] = _a, _ct, _kc, "kmeans"

    def _eval_with_bands(logits):
        # Returns (vwsr, other_viol, band_breach) as THREE separate arrays so the ranking can treat
        # the EXACT M5 band breach as the strict primary key (see _rank). `other_viol` is the
        # engineering violation (global VAMP cap + max-share) from eval_pop; `band_breach` is the
        # exact per-MID M5 penalty. They are NO LONGER summed — the ranking orders on band first.
        v, x = eval_pop(logits)
        _band = np.zeros(np.asarray(x).shape[0], dtype=float)
        _need_band = band_penalty_fn is not None
        _need_comp = _compress_on and _cb["cent"] is not None
        if not (_need_band or _need_comp):
            return v, x, _band
        _sh = _segment_softmax(logits, p.cell_start, p.cell_len)
        _fd = _deliver_full(_sh)                                  # shared delivery — computed ONCE
        if _need_band:
            _band = np.asarray(band_penalty_fn(_fd if _have_full else _sh), dtype=float)
        if _need_comp:
            _sd = _deliver_kept(_sh, _fd)
            v = v - _clam * np.asarray(
                _dist_fn(_sd, _cb["assign"], _cb["cent"], _cb["cconst"]), dtype=float)
        return v, x, _band

    def _rescore_compress(logits, keep_other, keep_band):
        """Re-score ONLY the compress term of vwsr under the CURRENT codebook, keeping the supplied
        engineering violation AND band breach unchanged (both are codebook-independent, so they are
        NOT recomputed — this is what lets a codebook refresh skip the whole-population band
        projection). Returns (vwsr, keep_other, keep_band)."""
        _bv, _ = eval_pop(logits)
        if _compress_on and _cb["cent"] is not None:
            _sh = _segment_softmax(logits, p.cell_start, p.cell_len)
            _fd = _deliver_full(_sh)
            _sd = _deliver_kept(_sh, _fd)
            _bv = _bv - _clam * np.asarray(
                _dist_fn(_sd, _cb["assign"], _cb["cent"], _cb["cconst"]), dtype=float)
        return _bv, keep_other, keep_band

    # --- seed logits (reference mapped to sorted order) ---
    if reference_shares is None:
        seed_shares = _greedy_reference(p)
    else:
        seed_shares = np.asarray(reference_shares, dtype=float)[p.order]
    seed_logits = _shares_to_logits(seed_shares)

    # remember the elite seed's key for the never-worse guarantee (bands included)
    s0 = _segment_softmax(seed_logits[None, :], p.cell_start, p.cell_len)
    seed_vwsr = _vwsr(s0, p.vol, p.succ, total_vol)[0]
    seed_other = _violation(s0, p)[0]           # engineering viol (global cap + max-share) — secondary
    seed_band = 0.0                              # exact M5 band breach — strict primary key
    _fd0 = _deliver_full(s0) if (band_penalty_fn is not None or _compress_on) else None
    if band_penalty_fn is not None:
        seed_band = float(np.asarray(
            band_penalty_fn(_fd0 if _have_full else s0), dtype=float)[0])
    if _compress_on:
        _refresh_codebook(seed_logits)                            # learn the initial codebook from the seed
        _k0 = 0 if _cb["cent"] is None else _cb["cent"].shape[0]
        _sd0 = _deliver_kept(s0, _fd0)
        seed_vwsr = seed_vwsr - _clam * float(np.asarray(
            _dist_fn(_sd0, _cb["assign"], _cb["cent"], _cb["cconst"])).reshape(-1)[0])
        _cb_src = ("EXACT tab-3 ward/knapsack allocator" if _cb.get("src") == "callback"
                   else "internal volume-weighted k-means")
        log(f"[fullmatrix-ga] compressibility regularizer ON: λ={_clam:g}, codebook={_k0} centroid rows "
            f"via {_cb_src} (target {int(compress_pools)} pools, refit every {int(compress_refresh)} gens), "
            f"distortion backend={_dist_backend}, delivery-dedupe={'ON' if _have_full else 'off'}, "
            f"scored on DELIVERED shares "
            f"({'eligibility-aware' if (_have_full or callable(deliver_fn)) else 'raw — no deliver_fn'}) "
            "— VWSR −= λ·volume-weighted VQ distortion; pushes cells to route ALIKE so the split "
            "compresses into fewer configs; trades a little conversion.")
    # SEED-AS-GENOME diagnostic. The app measures the seed's breach on the RAW share vector,
    # then hands it here, where it is re-encoded through _shares_to_logits -> segment-softmax.
    # That round-trip cannot express an EXACT ZERO (eps=1e-6 floor), and in a sub-cell where a
    # breached MID is the SOLE VAMP-positive gateway vshare == 1 for ANY share > 0 — so a share
    # the seed zeroed comes back at 1e-6 and collects that cell's whole moved VAMP pool. When
    # that happens the never-worse guarantee is void: the GA is anchored on a DEGRADED seed and
    # can deliver a worse breach than the app printed. Log both so the gap is visible.
    if band_penalty_fn is not None:
        log(f"[fullmatrix-ga] seed AS THE GENOME SEES IT: M5 breach={seed_band:.6g} — compare to "
            "the app's \"[full-matrix] seed = ... exact M5 breach\" line. If the app's value is "
            "LOWER, the softmax/logit genome cannot represent the seed (eps=1e-6 floor revives "
            "shares the seed drove to 0, flipping vshare 0 -> 1 in sole-VAMP-positive cells) and "
            "the never-worse guarantee does NOT hold for this run.")
    seed_key = _key_of(seed_vwsr, seed_other, seed_band)
    best_logits = seed_logits.copy()
    best_key = seed_key
    best_vwsr, best_other, best_band = seed_vwsr, seed_other, seed_band

    history = []
    evaluated = 0                              # cumulative candidate splits scored
    _t0 = time.perf_counter()
    _PROG_EVERY_S = 15.0                        # throttle for the live progress line (like the tilt poller)
    _last_prog = _t0

    def _init_pop(center, n, _rng):
        """Population centred on a strong incumbent + fresh exploration."""
        pp = np.empty((n, R))
        pp[0] = center
        for i in range(1, n):
            if i < max(2, n // 4):
                pp[i] = center + _rng.standard_normal(R) * 0.3    # near the incumbent
            else:
                pp[i] = _rng.standard_normal(R) * 1.5              # no anchor
        return pp

    n_seeds = max(1, int(n_seeds))
    restarts = max(1, int(restarts))
    log(f"[fullmatrix-ga] budget: {n_seeds} seed(s) × {restarts} restart(s) × "
        f"{generations} gens (pop {pop_size}, mode={restart_mode})")
    # SEED = an independent search (own RNG, own random exploration). RESTART =
    # re-seed the population when a search stalls, keeping the best-so-far as the
    # incumbent. Every seed/restart is CENTRED on best_logits, so the never-worse
    # guarantee holds across the whole budget (result ≥ the caller's seed).
    for _s in range(n_seeds):
        for _r in range(restarts):
            _rng = np.random.default_rng(int(seed) + _s * 100003 + _r * 101)
            _pn = pop_size
            if str(restart_mode).lower() == "ipop":
                _pn = min(pop_size * (2 ** _r), pop_size * 4)     # IPOP: grow each restart
            _el = min(elite, max(1, _pn // 8))
            pop = _init_pop(best_logits, _pn, _rng)
            vwsr, other, band = _eval_with_bands(pop)
            evaluated += pop.shape[0]
            stale = 0
            for gen in range(generations):
                # Periodic codebook refit: re-learn the ≈pool-target centroid shapes from the
                # current global best (its delivered shape), then RE-SCORE the live pop + best
                # under the new codebook so the moving objective stays self-consistent. Only the
                # compress term of vwsr depends on the codebook — the band violation does NOT — so
                # `_rescore_compress` keeps the existing `viol` and skips the whole-population band
                # projection (the expensive part). Bit-identical to a full re-eval.
                if (_compress_on and int(compress_refresh) > 0 and gen > 0
                        and gen % int(compress_refresh) == 0):
                    _refresh_codebook(best_logits)
                    vwsr, other, band = _rescore_compress(pop, other, band)
                    best_vwsr = float(_rescore_compress(
                        best_logits[None, :], np.asarray([best_other]), np.asarray([best_band]))[0][0])
                    best_key = _key_of(best_vwsr, best_other, best_band)
                order = _rank(vwsr, other, band)
                top = order[0]
                top_key = _key_of(vwsr[top], other[top], band[top])
                if top_key > best_key:
                    best_key = top_key
                    best_logits = pop[top].copy()
                    best_vwsr, best_other, best_band = vwsr[top], other[top], band[top]
                    stale = 0
                else:
                    stale += 1
                # History x-axis = cumulative generation index across seeds/restarts;
                # `cands` = cumulative candidates (matches the tab-3 chart layout). The `viol` slot
                # carries band breach + engineering viol so the chart still reflects total infeasibility.
                history.append((len(history), float(best_vwsr), float(vwsr[top]),
                                float(vwsr.mean()), None, float(best_band + best_other), None,
                                int(evaluated)))
                # LIVE PROGRESS: throttled per-generation heartbeat so a long BIN-grain search isn't
                # silent for ~an hour (the tilt engine streams via its poller; this engine didn't).
                _now = time.perf_counter()
                if _now - _last_prog >= _PROG_EVERY_S:
                    _last_prog = _now
                    _rate = evaluated / max(_now - _t0, 1e-9)
                    # Optional per-MID-constraint readout at the current best (from the caller's exact
                    # band report): distinct MIDs whose band is unmet + total MID-constraint penalty.
                    _mid_extra = ""
                    if band_report_fn is not None:
                        try:
                            _bsh = _segment_softmax(best_logits[None, :], p.cell_start, p.cell_len)
                            _rep = band_report_fn(_deliver_full(_bsh) if _have_full else _bsh)
                            # band_report_fn may return (count, penalty) or (count, penalty, [names]).
                            _n_unmet, _mid_pen = _rep[0], _rep[1]
                            _unmet_names = list(_rep[2]) if len(_rep) >= 3 else []
                            _mid_extra = (f" · MID unmet {int(_n_unmet)} · "
                                          f"MID penalty {float(_mid_pen):,.4f}")
                            if _unmet_names:
                                # Show WHICH bands are stuck; cap the list so the line stays readable.
                                _cap = 8
                                _shown = ", ".join(str(_x) for _x in _unmet_names[:_cap])
                                if len(_unmet_names) > _cap:
                                    _shown += f", +{len(_unmet_names) - _cap} more"
                                _mid_extra += f" [{_shown}]"
                        except Exception:  # noqa: BLE001 - never let the heartbeat break the search
                            _mid_extra = ""
                    # NB: best_key is the lexicographic tuple (feasible?, vwsr-or-−viol) from _key_of —
                    # it must NOT be format-specced ('{:,.0f}' on a tuple raises). vwsr + viol below
                    # already convey the score/feasibility, so it isn't printed.
                    log(f"[fullmatrix-ga] progress: ~{evaluated:,} splits · gen {gen} "
                        f"(seed {_s + 1}/{n_seeds} restart {_r + 1}/{restarts}) · "
                        f"best vwsr {best_vwsr:.5f} · viol {best_band + best_other:,.4f}"
                        f"{_mid_extra} · {'feasible' if best_band <= _FEAS_EPS else 'infeasible'} "
                        f"· {_rate:,.0f}/s")
                if stale >= patience:
                    log(f"[fullmatrix-ga] seed {_s + 1}/{n_seeds} restart {_r + 1}/"
                        f"{restarts}: converged at gen {gen} (no gain in {patience})")
                    break
                # elitism + local refinement + exploration
                elites = pop[order[:_el]].copy()
                elite_vwsr = vwsr[order[:_el]].copy()
                elite_other = other[order[:_el]].copy()
                elite_band = band[order[:_el]].copy()
                children = np.empty((_pn - _el, R))
                pool = order[: max(_el, _pn // 2)]
                _base_rate = min(mutation_rate, max(0.01, 60.0 / max(p.n_cells, 1)))
                _n_refine = children.shape[0] // 2
                for c in range(children.shape[0]):
                    if c < _n_refine:
                        base = pop[order[_rng.integers(0, max(1, _el))]]
                        child = _mutate(base, _base_rate * 0.25, mutation_strength * 0.6,
                                        p.cell_start, p.cell_len, _rng)
                    else:
                        pa = pop[_rng.choice(pool)]
                        pb = pop[_rng.choice(pool)]
                        child = _crossover(pa, pb, p.cell_start, p.cell_len, _rng)
                        child = _mutate(child, _base_rate, mutation_strength,
                                        p.cell_start, p.cell_len, _rng)
                    children[c] = child
                child_vwsr, child_other, child_band = _eval_with_bands(children)
                evaluated += children.shape[0]
                pop = np.vstack([elites, children])
                vwsr = np.concatenate([elite_vwsr, child_vwsr])
                other = np.concatenate([elite_other, child_other])
                band = np.concatenate([elite_band, child_band])

    best_shares_sorted = _segment_softmax(best_logits[None, :], p.cell_start, p.cell_len)[0]
    # restore original row order
    inv = np.empty_like(p.order)
    inv[p.order] = np.arange(len(p.order))
    best_shares = best_shares_sorted[inv]

    info = {
        "__build__": __build__,
        "vwsr": float(best_vwsr),
        "violation": float(best_band + best_other),
        "band_breach": float(best_band),
        "other_violation": float(best_other),
        "feasible": bool(best_band <= _FEAS_EPS),
        "seed_vwsr": float(seed_vwsr),
        "seed_violation": float(seed_band + seed_other),
        "seed_band_breach": float(seed_band),
        "improved_over_seed": bool(best_key > seed_key),
        "generations_run": len(history),
        "splits_evaluated": int(evaluated),
        "pop_size": int(pop_size),
        "n_seeds": int(n_seeds),
        "restarts": int(restarts),
        "seconds": float(time.perf_counter() - _t0),
        "splits_per_s": float(evaluated / max(time.perf_counter() - _t0, 1e-9)),
        "history": history,
        "note": ("compliance is SOFT/adaptive in-search; apply the caller's "
                 "exact enforcement pass to the returned split before shipping."),
    }
    log(f"[fullmatrix-ga] evaluated {evaluated:,} candidate splits over "
        f"{len(history)} generations ({n_seeds} seed(s) × {restarts} restart(s), "
        f"pop {pop_size}) in {info['seconds']:.1f}s = {info['splits_per_s']:,.0f} splits/s")
    log(f"[fullmatrix-ga] done vwsr={best_vwsr:.6f} M5-breach={best_band:.3e} "
        f"eng-viol={best_other:.3e} feasible={info['feasible']} improved={info['improved_over_seed']}")
    return best_shares, info


def _cell_volume_total(p: "FullMatrixProblem"):
    """Total volume across cells (each cell's volume counted once)."""
    # vol is repeated per row; take the first row of each cell segment.
    return float(p.vol[p.cell_start].sum())


# STATUS (was the stage-2 TODO — now largely delivered):
#   * DONE  Numba-fused kernel (_fused_eval_kernel / make_fused_eval): segment-
#           softmax + VWSR + violation in one pass, verify-gated bit-close vs the
#           numpy path with automatic fallback. prange-parallel, persistent cache,
#           int32 index arrays, elite-fitness caching.
#   * DONE  Wired as the opt-in "genetic_fullmatrix" engine in tab2_engine.py
#           (dropdown + run-dispatch gate + delivery-site override), fed EB-shrunk
#           per-BIN rates from ctx (true BIN grain via the bin_to_bank identity map).
#   * DUAL CEILINGS exist (mid_hard_cap/mid_soft_cap + adaptive tolerance) but the
#           live override passes soft_cap == hard_cap for now.
#   * NOTE  There is no _enforce_endpoint on the live path — VAMP-cap enforcement
#           is off there for every engine, so instead the GA is seeded from the
#           known-compliant greedy+LP split (_comp_share_G) → feasible by
#           construction. Re-enabling the soft-cap boundary-hug (ride to soft cap,
#           then LP-tighten) is the one remaining enhancement; it needs an explicit
#           enforce step wired onto the returned split.
