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

__build__ = "2026-08-12-fullmatrix-ga-dualceiling-adaptivetol+numbafuse+prange+elitecache+persistcache+midbands+exactbandhook+localrefine+globalvampcap+seeds+restarts+live-progress"

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


def _rank(vwsr, viol):
    """Feasibility-first ranking WITH adaptive tolerance. Returns indices
    ordered best -> worst.

      1. Feasible (viol <= eps) always beats infeasible.
      2. Among infeasible: lower violation wins.
      3. Among feasible: higher VWSR wins, but VWSRs within the adaptive
         tolerance tie and fall back to LOWER risk-proxy (violation) — this is
         what makes the search hug the boundary.
    """
    n = len(vwsr)
    feasible = viol <= _FEAS_EPS
    tol = _adaptive_tol(viol)
    # snap VWSR to a tolerance grid so near-equal VWSRs compare equal;
    # guard tol==0 (strict) by using a tiny epsilon grid.
    grid = np.where(tol > 0, tol, 1e-12)
    vwsr_bucket = np.round(vwsr / grid).astype(np.int64)

    # Build a sort key. numpy lexsort sorts by LAST key primary; we want:
    #   primary  : feasible desc
    #   secondary(feasible)   : vwsr_bucket desc, then viol asc
    #   secondary(infeasible) : viol asc
    # Encode into columns then lexsort (all ascending), negating where needed.
    #   col_feas = 0 for feasible, 1 for infeasible  (asc -> feasible first)
    col_feas = np.where(feasible, 0, 1)
    #   for feasible: want high vwsr first -> use -vwsr_bucket; tiebreak low viol
    #   for infeasible: vwsr irrelevant -> set 0 so viol dominates
    col_score = np.where(feasible, -vwsr_bucket, 0)
    col_viol = viol
    order = np.lexsort((col_viol, col_score, col_feas))
    return order


def _best_index(vwsr, viol):
    return _rank(vwsr, viol)[0]


def _key_of(vwsr, viol):
    """Comparable feasibility-first key for a SINGLE candidate (higher==better).
    Used only for the 'never worse than the elite seed' guarantee."""
    feasible = viol <= _FEAS_EPS
    # (feasible?, then vwsr if feasible else -viol)
    return (1 if feasible else 0, vwsr if feasible else -viol)


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
                      band_penalty_fn=None, n_seeds=1, restarts=1,
                      restart_mode="lean"):
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

    # EXACT band penalty (optional). When `band_penalty_fn` is supplied, the kernel's
    # (linear) band term is expected OFF (mid_band_metric all 0), and the exact
    # per-generation M5 band breach is ADDED here from the caller's projector. It maps
    # sorted kept-row shares -> full opt-grain shares internally. This is the faithful
    # replacement for the 30-day linear proxy (matches the tilt engine's band scoring).
    def _eval_with_bands(logits):
        v, x = eval_pop(logits)
        if band_penalty_fn is not None:
            sh = _segment_softmax(logits, p.cell_start, p.cell_len)
            x = x + np.asarray(band_penalty_fn(sh), dtype=float)
        return v, x

    # --- seed logits (reference mapped to sorted order) ---
    if reference_shares is None:
        seed_shares = _greedy_reference(p)
    else:
        seed_shares = np.asarray(reference_shares, dtype=float)[p.order]
    seed_logits = _shares_to_logits(seed_shares)

    # remember the elite seed's key for the never-worse guarantee (bands included)
    s0 = _segment_softmax(seed_logits[None, :], p.cell_start, p.cell_len)
    seed_vwsr = _vwsr(s0, p.vol, p.succ, total_vol)[0]
    seed_viol = _violation(s0, p)[0]
    if band_penalty_fn is not None:
        seed_viol = seed_viol + float(np.asarray(band_penalty_fn(s0), dtype=float)[0])
    seed_key = _key_of(seed_vwsr, seed_viol)
    best_logits = seed_logits.copy()
    best_key = seed_key
    best_vwsr, best_viol = seed_vwsr, seed_viol

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
            vwsr, viol = _eval_with_bands(pop)
            evaluated += pop.shape[0]
            stale = 0
            for gen in range(generations):
                order = _rank(vwsr, viol)
                top = order[0]
                top_key = _key_of(vwsr[top], viol[top])
                if top_key > best_key:
                    best_key = top_key
                    best_logits = pop[top].copy()
                    best_vwsr, best_viol = vwsr[top], viol[top]
                    stale = 0
                else:
                    stale += 1
                # History x-axis = cumulative generation index across seeds/restarts;
                # `cands` = cumulative candidates (matches the tab-3 chart layout).
                history.append((len(history), float(best_vwsr), float(vwsr[top]),
                                float(vwsr.mean()), None, float(best_viol), None,
                                int(evaluated)))
                # LIVE PROGRESS: throttled per-generation heartbeat so a long BIN-grain search isn't
                # silent for ~an hour (the tilt engine streams via its poller; this engine didn't).
                _now = time.perf_counter()
                if _now - _last_prog >= _PROG_EVERY_S:
                    _last_prog = _now
                    _rate = evaluated / max(_now - _t0, 1e-9)
                    log(f"[fullmatrix-ga] progress: ~{evaluated:,} splits · gen {gen} "
                        f"(seed {_s + 1}/{n_seeds} restart {_r + 1}/{restarts}) · best vwsr "
                        f"{best_vwsr:.5f} · viol {best_viol:.2e} · "
                        f"{'feasible' if best_viol <= _FEAS_EPS else 'infeasible'} · {_rate:,.0f}/s")
                if stale >= patience:
                    log(f"[fullmatrix-ga] seed {_s + 1}/{n_seeds} restart {_r + 1}/"
                        f"{restarts}: converged at gen {gen} (no gain in {patience})")
                    break
                # elitism + local refinement + exploration
                elites = pop[order[:_el]].copy()
                elite_vwsr = vwsr[order[:_el]].copy()
                elite_viol = viol[order[:_el]].copy()
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
                child_vwsr, child_viol = _eval_with_bands(children)
                evaluated += children.shape[0]
                pop = np.vstack([elites, children])
                vwsr = np.concatenate([elite_vwsr, child_vwsr])
                viol = np.concatenate([elite_viol, child_viol])

    best_shares_sorted = _segment_softmax(best_logits[None, :], p.cell_start, p.cell_len)[0]
    # restore original row order
    inv = np.empty_like(p.order)
    inv[p.order] = np.arange(len(p.order))
    best_shares = best_shares_sorted[inv]

    info = {
        "__build__": __build__,
        "vwsr": float(best_vwsr),
        "violation": float(best_viol),
        "feasible": bool(best_viol <= _FEAS_EPS),
        "seed_vwsr": float(seed_vwsr),
        "seed_violation": float(seed_viol),
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
    log(f"[fullmatrix-ga] done vwsr={best_vwsr:.6f} viol={best_viol:.3e} "
        f"feasible={info['feasible']} improved={info['improved_over_seed']}")
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
