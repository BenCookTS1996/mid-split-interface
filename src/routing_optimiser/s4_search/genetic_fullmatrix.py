"""Full-matrix, BIN-grain genetic engine (opt-in, additive).

WHY THIS EXISTS
---------------
19gc — READ THIS FIRST: THIS MODULE *IS* THE PRODUCTION ENGINE. `run_fullmatrix_ga` is the
only selectable engine and the delivered search. The paragraph below describes the tilt
CMA-ES it replaced (`legacy_engines.midtilt_cmaes.run_midtilt_ga`), not reachable any more: tab 2
skips it explicitly for genetic_fullmatrix ("no preliminary endpoint search is run"). It used
to say "The production 'genetic' engine" about that one, which read as though this module were
the opt-in alternative rather than the thing that ships.

The tilt CMA-ES searched a COMPACT genome — 3 knobs per vampMid (risk-tilt, return-tilt, gain)
— applied as a tilt around a reference split, scored on *pooled* Bank x Currency success
rates. That was fast, stable and hard-compliant, but it could only reach splits that are
tilts/scalings of the reference, and it optimised a pooled approximation of what actually
ships. (The Bank x Currency score grain it used was itself removed in 19gb.)

This module is the deliberate opposite, mirroring a co-worker's DEAP design while
keeping our guarantees:

  * FULL-MATRIX genome        - one gene per (profile x eligible gateway) at BIN
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
`_segment_softmax` / `_success_rate` / `_violation` is a later stage (see module TODO).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import time
import os as _os_gf

import numpy as np

from routing_optimiser.s4_search.rowpar import row_parallel as _rowpar

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

__build__ = "2026-08-19bx-fused-softmax-and-child+2026-08-12-fullmatrix-ga-dualceiling-adaptivetol+numbafuse+prange+elitecache+persistcache+midbands+exactbandhook+localrefine+globalvampcap+seeds+restarts+live-progress+progress-tuple-format-fix+progress-plain-decimals+progress-unmet-names+compress-learned-codebook-delivered-numbadistortion+exact-tab3-codebook-callback+delivery-dedupe+refresh-skip-band+lexico-m5-primary-ranking+19eb-ga-census+19ed-viol-decomp+19gw-eval-cost+19gu-decode-cap+19ee-maxshare-repair"

# Feasibility tolerance: violations at or below this count as compliant in-search.
_FEAS_EPS = 1e-9


# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
@dataclass
class FullMatrixProblem:
    """One optimisation problem in LONG format.

    Every row is an ELIGIBLE (profile, gateway) pair. Rows MUST be grouped so each
    profile's rows are contiguous (the constructor `build` enforces this). A "profile"
    is the routing-decision grain — here BIN x (currency x bank) — one simplex of
    shares per profile.

    Arrays are all length R (number of eligible rows) unless noted.
    """

    profile_id: np.ndarray      # int (R,)  contiguous group id per row
    gw_id: np.ndarray        # int (R,)  gateway id (for building the output table)
    mid_id: np.ndarray       # int (R,)  vampMid id per row (spans profiles)
    vol: np.ndarray          # float (R,) profile volume (same for every row of a profile)
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
    profile_start: np.ndarray = field(default=None)   # int (n_profiles,) segment starts
    profile_len: np.ndarray = field(default=None)      # int (n_profiles,) segment lengths
    n_profiles: int = 0
    n_mids: int = 0
    order: np.ndarray = field(default=None)         # int (R,) orig->sorted perm

    @classmethod
    def build(cls, profile_id, gw_id, mid_id, vol, succ, risk,
              max_share, floor, mid_hard_cap, mid_soft_cap,
              mid_band_metric=None, mid_band_lo=None, mid_band_hi=None,
              global_vamp_cap=np.inf):
        """Sort rows to contiguous profile groups and precompute segment indices.

        Returns a ready-to-optimise FullMatrixProblem. `order` records the
        original row order so the caller can map the returned split back.
        """
        profile_id = np.asarray(profile_id)
        # stable sort by profile so groups are contiguous but within-profile order kept
        order = np.argsort(profile_id, kind="stable")
        def _o(a):
            return np.asarray(a, dtype=float)[order]
        cid = profile_id[order]
        # segment boundaries of the sorted profile ids
        uniq, starts, lens = _segments(cid)
        # remap profile ids to dense 0..n_profiles-1 in sorted order
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
            profile_id=dense,
            gw_id=np.asarray(gw_id)[order].astype(int),
            mid_id=mid,
            vol=_o(vol), succ=_o(succ), risk=_o(risk),
            max_share=_o(max_share), floor=_o(floor),
            mid_hard_cap=np.asarray(mid_hard_cap, dtype=float),
            mid_soft_cap=np.asarray(mid_soft_cap, dtype=float),
            mid_band_metric=_bm, mid_band_lo=_blo, mid_band_hi=_bhi,
            global_vamp_cap=float(global_vamp_cap),
            profile_start=starts, profile_len=lens,
            n_profiles=len(uniq), n_mids=n_mids, order=order,
        )
        return p


def build_fullmatrix_problem(profile_problems, hard, *, mid_caps=None,
                             exploration_floor=None):
    """Turn the app's existing ``ProfileProblem`` list into a FullMatrixProblem.

    This is the Stage-2 data feed. It uses the finest grain the app already
    produces (one profile per ProfileProblem — RPGT x Currency x Bank, no pooling) and
    the rates already attached to each profile:

      * ``profile.success_rates`` are ALREADY empirical-Bayes shrunk (built upstream
        by ``success_rates.gateway_success_rates`` inside ``build_profile_problems``),
        so feeding them straight in is the "EB-shrunk bin rates" requirement — no
        re-shrinking needed, no pooled->broadcast gap.
      * ``profile.risk_rates`` are the per-gateway VAMP rates (from
        ``bin_rpgt_impact_export.csv`` at period=0, loaded upstream).

    The SAME gateway/MID name recurs across many profiles; that shared name is one
    vampMid, and its VAMP cap applies to its aggregate across every profile — which
    is exactly the cross-profile coupling the full-matrix GA scores.

    Parameters
    ----------
    profile_problems : list[ProfileProblem]
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
        back to (profile, gateway) and for elite seeding.
    """
    # global MID (== gateway name) index
    mid_names: list[str] = []
    name2mid: dict[str, int] = {}
    for cp in profile_problems:
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

    profile_id, gw_id, mid_id = [], [], []
    vol, succ, risk, base = [], [], [], []
    for ci, cp in enumerate(profile_problems):
        n = cp.n()
        bshares = (np.asarray(cp.baseline_shares, dtype=float)
                   if cp.baseline_shares is not None
                   else np.full(n, 1.0 / n))
        for i, g in enumerate(cp.gateways):
            profile_id.append(ci)
            gw_id.append(name2mid[g])          # global gateway/MID index
            mid_id.append(name2mid[g])
            vol.append(float(cp.volume))
            succ.append(float(cp.success_rates[i]))
            risk.append(float(cp.risk_rates[i]))
            base.append(float(bshares[i]))

    R = len(profile_id)
    max_share = np.full(R, max_gw)
    floor = np.full(R, floor_v)
    problem = FullMatrixProblem.build(
        profile_id=np.array(profile_id), gw_id=np.array(gw_id), mid_id=np.array(mid_id),
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

    This is the REAL integration surface for tab_2_routing_engine. The cross-profile tilt GA
    already assembles ``ctx`` with everything at BIN grain in long format:
      * contiguous profiles       - ctx['profile_starts'] / ctx['profile_counts']
      * per-row success (EB)   - ctx['sr']   (already empirical-Bayes shrunk)
      * per-row VAMP rate      - ctx['risk']
      * per-row vampMid index  - ctx['mid_id']
      * per-row profile volume     - ctx['profile_vol']
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
    starts = np.asarray(ctx["profile_starts"], dtype=int)
    counts = np.asarray(ctx["profile_counts"], dtype=int)

    profile_id_full = np.empty(n_row, dtype=int)
    for ci, (s, c) in enumerate(zip(starts, counts)):
        profile_id_full[s:s + c] = ci

    mid_id_full = np.asarray(ctx["mid_id"], dtype=int)
    vol_full = np.asarray(ctx["profile_vol"], dtype=float)
    succ_full = np.asarray(ctx["sr"], dtype=float)
    risk_full = np.asarray(ctx["risk"], dtype=float)
    elig = np.asarray(ctx.get("elig", np.ones(n_row)), dtype=float) > 0.5

    # ── 19eh [prune-inert] ────────────────────────────────────────────────────────────────
    # Drop whole profiles that carry NO forecast volume. Every term the GA scores is
    # volume-weighted - success rate is Sigma vol*share*succ / Sigma vol*share, band metric 1 is
    # Sigma vol*share, metric 2 is Sigma vol*share*risk, metric 3 is their ratio - so a row with
    # vol == 0 contributes exactly 0.0 to all four for every candidate. Removing it cannot move
    # the objective. That is arithmetic, not a measurement.
    #
    # profile_vol is the PROFILE total repeated on every row of the profile, so testing it at profile_starts
    # is the profile's own volume and the whole profile goes together. Dropping part of a profile would
    # break its simplex; dropping all of it is clean, and FullMatrixProblem.build remaps profile ids
    # to dense 0..n_profiles-1, so the gap in profile ids costs nothing.
    #
    # SIZE, on the 2026-08-31 18:21 book: 9,018 of 23,870 profiles and 103,230 of 257,635 rows
    # (40.1%). [frozen-scaffold] already measured what that buys: "92% of a GA generation is
    # _pop_band_kernel over the cap scaffold (1.28M rows / 22.3k profiles) ... So shrinking the
    # genome alone saves ~6%." The scaffold is 1,933,016 rows to the genome's 257,635, so 40% of
    # the genome is ~5% of the per-generation row traffic. The 38% is real, and it is 38% of the
    # smaller object.
    #
    # DEFAULT OFF, and the first reason is the one that matters:
    #
    # 1. IT IS NOT BIT-IDENTICAL ON success rate, and no amount of better code fixes that. Measured on
    #    the fixture: full 0.60064871358661553 vs pruned 0.60064871358661565, delta 1.11e-16
    #    (1.85e-16 relative). The three BAND metrics ARE bit-identical, because they go through
    #    np.bincount, which accumulates sequentially - adding 0.0 to a running sum is exact, so
    #    dropping zero rows cannot move them. success rate is a GLOBAL .sum(), and numpy sums float64
    #    PAIRWISE: the accumulation tree is a function of the array LENGTH, so removing 40% of the
    #    elements rebuilds the tree and reorders the additions of the NON-ZERO terms.
    #    Mathematically identical; not bit-identical. And it mattered for the max-share repair
    #    deliberately drives every candidate to an exact 0.0 on the engineering key so the ranking
    #    FALLS THROUGH to success rate, and [never-worse] compares success rate too. A last-bit shift can select a
    #    different candidate and ship a different split. Same objective, different answer.
    #
    # 2. [deliv-fuse] SELF-DISABLES. tab_2_routing_engine gates it on `len(_fm_colmap) == _fm_nrow` - the
    #    scatter writing every column - which this breaks by construction. So an optimisation that
    #    is on today turns off, and the NET speed effect has to be measured, not assumed.
    #
    # 3. The delivery transform now meets profiles whose every row is 0. The band projector guards its
    #    own division (`where=psum > 0`); _fm_deliv is a different function and an unguarded
    #    renormalise there would give NaN. Unverified without a run.
    #
    # ROUTING_PRUNE_INERT=1 for one run, then compare tab 3 against Validate. Given (1), the
    # honest expectation is a ~6% generation saving in exchange for a split that can differ - and
    # on this project that trade has always been refused. The switch exists so the trade can be
    # measured rather than argued about.
    _inert_row = np.zeros(n_row, dtype=bool)
    _prune_note = ""            # this module has no logger; tab_2_routing_engine emits meta['prune_note']
    if _os_gf.environ.get("ROUTING_PRUNE_INERT", "0") != "0":
        _profile_vol_by_profile = vol_full[starts] if starts.size else np.zeros(0)
        _inert_profile = _profile_vol_by_profile <= 0.0
        if _inert_profile.any():
            _inert_row = _inert_profile[profile_id_full]
            # Never prune the whole genome - if every profile reads volume-less the volume column is
            # wrong, not the book, and running on an empty problem would hide that.
            if bool(_inert_row.all()):
                _prune_note = (
                    "[prune-inert] EVERY profile reads zero forecast volume - that is a broken "
                    "volume column, not an inert book. Prune SKIPPED; the genome is unchanged.")
                _inert_row = np.zeros(n_row, dtype=bool)

    keep_idx = np.where(elig & ~_inert_row)[0]
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
        profile_id=profile_id_full[keep_idx], gw_id=keep_idx.copy(),
        mid_id=mid_id_full[keep_idx], vol=vol_full[keep_idx],
        succ=succ_full[keep_idx], risk=risk_full[keep_idx],
        max_share=np.full(keep_idx.size, max_gw),
        floor=np.full(keep_idx.size, floor_v),
        mid_hard_cap=mid_hard, mid_soft_cap=mid_soft,
        mid_band_metric=_bm, mid_band_lo=_blo, mid_band_hi=_bhi,
        global_vamp_cap=global_cap,
    )
    # renormalise the kept reference within each (kept) profile so it is a valid seed
    kept_profile = profile_id_full[keep_idx]
    for cid in np.unique(kept_profile):
        m = kept_profile == cid
        s = ref_kept[m].sum()
        ref_kept[m] = (ref_kept[m] / s) if s > 1e-12 else (1.0 / m.sum())

    # 19eh: what to put BACK. Config-banned rows correctly ship 0 (they can never carry share);
    # a pruned inert profile must NOT - it would export as all-zeros instead of summing to 1. Its
    # rows are restored to their baseline share, so the shipped split for those profiles is the
    # baseline rather than whatever drift the search happened to leave there.
    _restore_idx = np.where(_inert_row)[0]
    _base_full = np.asarray(ctx.get("base", np.zeros(n_row)), dtype=float)
    meta = {"keep_idx": keep_idx, "n_row": n_row, "reference_kept": ref_kept,
            "restore_idx": _restore_idx,
            "restore_val": _base_full[_restore_idx].copy() if _restore_idx.size else None}
    if _restore_idx.size:
        _n_profile_pruned = int(np.unique(profile_id_full[_restore_idx]).size)
        _prune_note = (
            f"[prune-inert] {_n_profile_pruned:,} of {starts.size:,} profile(s) "
            f"({100.0 * _n_profile_pruned / max(starts.size, 1):.1f}%) carry NO forecast volume and "
            f"are OUT of the genome, taking {_restore_idx.size:,} of {n_row:,} row(s) "
            f"({100.0 * _restore_idx.size / max(n_row, 1):.1f}%) with them. Every term the search "
            f"scores is volume-weighted, so a zero-volume row contributes exactly 0.0 to success_rate and "
            f"to all three band metrics for every candidate - this cannot move the objective. "
            f"Those profiles ship their BASELINE share, not the search's drift. Genome is now "
            f"{keep_idx.size:,} row(s) over {problem.n_profiles:,} profile(s). Expect ~6% off a "
            f"generation, NOT 38%: [frozen-scaffold] measured the generation as 92% band kernel "
            f"over a 1.9M-row scaffold, and the genome is 13% of that traffic. "
            f"ROUTING_PRUNE_INERT=0 reverts.")
    meta["prune_note"] = _prune_note
    return problem, meta


def reconstruct_full_split(best_kept_shares, meta):
    """Map a kept-row split back to the full ``n_row`` vector (0 at banned rows).

    19eh: a config-BANNED row correctly lands on 0 - it can never carry share. A row dropped by
    [prune-inert] must not: its profile was removed from the genome for having no forecast volume,
    and leaving it at 0 would export that profile as all-zeros instead of a simplex summing to 1.
    Those rows are restored to their baseline share.
    """
    full = np.zeros(meta["n_row"], dtype=float)
    full[meta["keep_idx"]] = np.asarray(best_kept_shares, dtype=float)
    _ri = meta.get("restore_idx")
    _rv = meta.get("restore_val")
    if _ri is not None and _rv is not None and len(_ri):
        full[_ri] = np.asarray(_rv, dtype=float)
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
# ── FUSED ELEMENTWISE KERNELS (2026-08-19bx) ──────────────────────────────────────────────────
# Adopted from [stage-ab] rows S and X, measured on the live machine at the live width: 7 of 7
# paired rounds each, p=0.016, bit-identical on int64 patterns. See the 19bx patch note.
try:
    from numba import njit as _fx_njit
    _FX_HAVE_NB = True
except Exception:                                   # noqa: BLE001 - numba absent is not an error
    _FX_HAVE_NB = False

    def _fx_njit(*_a, **_k):                        # so the bodies stay valid python
        def _deco(f):
            return f
        return _deco


_SM_FUSE = (os.environ.get("ROUTING_SOFTMAX_FUSE", "1") != "0") and _FX_HAVE_NB
_CH_FUSE = (os.environ.get("ROUTING_CHILD_FUSE", "1") != "0") and _FX_HAVE_NB
_SM_OK = {"use": _SM_FUSE, "checked": False, "msg": ""}
_FX_OK = {"use": _CH_FUSE, "checked": False, "msg": ""}
# Layout arrays (row->profile map, int32 starts/counts) built ONCE per layout. Keyed on the identity
# of `profile_len` AND holding a reference to it, so the id cannot be recycled onto a different array
# while the entry lives — a mis-keyed profile map is a silent wrong answer, not a slow one.
_FX_LAYOUT = {}


def _fx_layout(profile_start, profile_len):
    _k = id(profile_len)
    _e = _FX_LAYOUT.get(_k)
    if _e is not None and _e[0] is profile_len:
        return _e[1], _e[2], _e[3]
    _cl = np.asarray(profile_len, np.int64)
    _co = np.repeat(np.arange(_cl.size, dtype=np.int32), _cl)
    _cs32 = np.ascontiguousarray(np.asarray(profile_start, np.int32))
    _cc32 = np.ascontiguousarray(np.asarray(profile_len, np.int32))
    _FX_LAYOUT[_k] = (profile_len, _co, _cs32, _cc32)
    return _co, _cs32, _cc32


def _fx_bits(a):
    a = np.asarray(a)
    return a.view(np.int64) if a.dtype == np.float64 else a


def _fx_same(x, y):
    return bool(np.array_equal(_fx_bits(x), _fx_bits(y)))


@_fx_njit(cache=False, fastmath=False)
def _fx_sub(lg, seg, co, out):
    """logits - repeat(seg_max, profile_len), in one pass."""
    for _p in range(lg.shape[0]):
        for _i in range(lg.shape[1]):
            out[_p, _i] = lg[_p, _i] - seg[_p, co[_i]]
    return out


@_fx_njit(cache=False, fastmath=False)
def _fx_div(ex, seg, co, out):
    """ex / repeat(seg_sum, profile_len), in one pass."""
    for _p in range(ex.shape[0]):
        for _i in range(ex.shape[1]):
            out[_p, _i] = ex[_p, _i] / seg[_p, co[_i]]
    return out


@_fx_njit(cache=False, fastmath=False)
def _fx_child(a, b, pick, hit, noise, strength, cstart, ccnt, out):
    """crossover + mutate for ONE child, in one pass.

    `noise` is consumed in increasing ROW order, which is exactly the order numpy's
    `out[row_hit] += noise` assigns it, and `v + noise[k] * strength` is the same two operations
    in the same order as numpy's `noise * strength` then add."""
    _k = 0
    for _ci in range(cstart.shape[0]):
        _s = cstart[_ci]
        _e = _s + ccnt[_ci]
        _pa = pick[_ci]
        if hit[_ci]:
            for _i in range(_s, _e):
                _v = a[_i] if _pa else b[_i]
                out[_i] = _v + noise[_k] * strength
                _k += 1
        else:
            for _i in range(_s, _e):
                out[_i] = a[_i] if _pa else b[_i]
    return out


@_fx_njit(cache=False, fastmath=False)
def _fx_mut(a, hit, noise, strength, cstart, ccnt, out):
    """mutate only — the REFINE branch, which must not draw the crossover's random(n_profiles)."""
    _k = 0
    for _ci in range(cstart.shape[0]):
        _s = cstart[_ci]
        _e = _s + ccnt[_ci]
        if hit[_ci]:
            for _i in range(_s, _e):
                out[_i] = a[_i] + noise[_k] * strength
                _k += 1
        else:
            for _i in range(_s, _e):
                out[_i] = a[_i]
    return out


def _segment_softmax_fast(logits, profile_start, profile_len):
    """`_segment_softmax_serial`, with the elementwise steps fused. Bit-identical.

    Measured 206.6 -> 126.9 ms at P=35 x 245,409 on the live machine, 7/7 paired rounds. np.exp
    stays in numpy on purpose: numba's exp is slower AND differs in the last bit."""
    _lg = np.atleast_2d(logits)
    _co, _, _ = _fx_layout(profile_start, profile_len)
    _sm = np.maximum.reduceat(_lg, profile_start, axis=1)
    _t = _fx_sub(_lg, _sm, _co, np.empty_like(_lg))
    _ex = np.exp(_t, out=_t)                       # numpy's exp, in place: no extra full array
    _ss = np.add.reduceat(_ex, profile_start, axis=1)
    return _fx_div(_ex, _ss, _co, np.empty_like(_lg))


def _mutate_fused(logits, rate, strength, profile_start, profile_len, rng, profile_w=None):
    """`_mutate_fast` in one pass. IDENTICAL draws: random(n_profiles), then standard_normal(n_hit)."""
    _n = len(profile_start)
    _thr = rate if profile_w is None else np.minimum(np.asarray(profile_w, float) * rate, 1.0)
    _hit = rng.random(_n) < _thr
    if not _hit.any():
        return logits.copy()
    _rh = np.repeat(_hit, profile_len)
    _nh = int(_rh.sum())
    _nz = rng.standard_normal(_nh)
    _, _cs32, _cc32 = _fx_layout(profile_start, profile_len)
    return _fx_mut(logits, _hit, _nz, float(strength), _cs32, _cc32, np.empty_like(logits))


def _child_fused(a, b, rate, strength, profile_start, profile_len, rng, profile_w=None):
    """`_crossover` then `_mutate_fast` in one pass. IDENTICAL draws, in the same order."""
    _n = len(profile_start)
    _pk = rng.random(_n) < 0.5
    _thr = rate if profile_w is None else np.minimum(np.asarray(profile_w, float) * rate, 1.0)
    _hit = rng.random(_n) < _thr
    if not _hit.any():
        return np.where(np.repeat(_pk, profile_len), a, b)
    _rh = np.repeat(_hit, profile_len)
    _nh = int(_rh.sum())
    _nz = rng.standard_normal(_nh)
    _, _cs32, _cc32 = _fx_layout(profile_start, profile_len)
    return _fx_child(a, b, _pk, _hit, _nz, float(strength), _cs32, _cc32, np.empty_like(a))


def _fx_selfcheck(a, b, rate, strength, profile_start, profile_len, rng, profile_w, refine):
    """Run BOTH paths from the same generator state and compare the arrays AND the end state.

    The end-state comparison is the part that matters: it proves the fused wrapper consumed the
    same draws in the same order, which is the only way the fused child can be the SAME child
    rather than a similar one. On any mismatch the fused path is disabled for the process and the
    reference result is returned, so what ships is the known-good child."""
    _st0 = rng.bit_generator.state
    if refine:
        _got = _mutate_fused(a, rate, strength, profile_start, profile_len, rng, profile_w=profile_w)
    else:
        _got = _child_fused(a, b, rate, strength, profile_start, profile_len, rng, profile_w=profile_w)
    _st_new = rng.bit_generator.state
    rng.bit_generator.state = _st0
    if refine:
        _ref = _mutate_fast(a, rate, strength, profile_start, profile_len, rng, profile_w=profile_w)
    else:
        _ref = _mutate_fast(_crossover(a, b, profile_start, profile_len, rng),
                            rate, strength, profile_start, profile_len, rng, profile_w=profile_w)
    _st_ref = rng.bit_generator.state
    _same_v = _fx_same(_ref, _got)
    _same_s = (repr(_st_new) == repr(_st_ref))
    _FX_OK["checked"] = True
    if _same_v and _same_s:
        _FX_OK["msg"] = (
            "[fullmatrix-ga] \u2713 fused child SELF-CHECK PASSED on the live population: "
            "bit-identical to crossover + _mutate_fast (int64 bit-pattern comparison, not "
            "array_equal) AND the generator's end state matches, so the draw order is unchanged "
            "and the child is the SAME child. Measured 127.1 -> 73.5 ms per generation at P=35 "
            "([stage-ab], 7/7 paired rounds, p=0.016). ROUTING_CHILD_FUSE=0 reverts.")
        rng.bit_generator.state = _st_new
        return _got
    _FX_OK["use"] = False
    _FX_OK["msg"] = (
        "[fullmatrix-ga] \u26a0 fused child SELF-CHECK FAILED \u2014 "
        + ("the arrays differ" if not _same_v else "the arrays match but the GENERATOR STATE "
           "differs, so the fused path consumed different draws")
        + ". DISABLED for this process; the reference crossover + _mutate_fast is what ships. "
          "Report this: it means the fused twin is not the same operator on this data.")
    rng.bit_generator.state = _st_ref
    return _ref


# ── [decode-cap] 19gu: THE CAP IS A PROPERTY OF THE DECODE ──────────────────────────────────
# Until 19gu the cap was a REPAIR (`_repair_maxshare`, deleted in 19gv). It decoded the whole
# population to shares,
# water-filled the over-cap profiles, and re-encoded the result with log() back into logits, which
# `_eval_with_bands` then decoded AGAIN. 98.4% of candidates needed it (a softmax cannot express
# an upper bound: it gives every live door a positive share and normalises to 1), so on the
# 2026-09-01 22:09 run that was 91.3s — 20% of the whole search — and three decodes of the
# population per generation instead of one.
#
# It also meant "the search respects the cap" was true only because a separate pass made it true
# afterwards. Now the DECODE cannot produce an over-cap split at all, so no candidate exists that
# violates it — which is what that sentence should have meant all along.
#
# THE RULE IS THE ONE DELIVERY USES, unchanged: the excess goes to each sibling in proportion to
# (target - share), the room it has left before IT would hit the cap — impact_calcs._cap_rows'
# water-fill and `_fm_cap`'s reduceat form are both this same arithmetic. Single
# pass, because Sum(target - share) over a profile's present rows equals (present_rows x target) - 1
# + excess, so it covers the excess whenever present_rows x target >= 1 — every profile with 2+ live
# rows at a 0.97 cap.
#
# THREE IMPLEMENTATIONS HAVE TO AGREE, and a divergence between them is a scored-vs-delivered
# bug: the numpy reference, the fused numpy path, and the numba kernel's per-profile inner loop.
# The reference is `_cap_shares_ref` below; the fused path calls it (the water-fill is not the
# hot part — the decode is); the kernel carries its own copy and is verified against numpy by
# make_fused_eval's existing verify gate, which now compares capped output on both sides.
#
# 19gv: THE SWITCH IS GONE, and so is `_repair_maxshare`. ROUTING_DECODE_CAP=0 used to hand the
# job back to the repair; with the repair deleted it would have meant "decode above the cap and
# let it ship", which is not a fallback, it is a footgun. The 2026-09-01 23:12 run proved the
# claim on live data — engineering violation 0.0000 on every candidate, the repair timing 0.0s —
# so this is now simply how the decode works, the same way 19gm retired ROUTING_VCONST_FROZEN.
# `_segment_softmax(..., max_share=None)` still decodes UNCAPPED; that path is what the
# [decode-cap] self-check compares against, and it is the only thing that ever needed it.
_DECODE_CAP = True
_DC_EPS = 1e-12
_DC_BACKOFF = 1.0 - 1e-9        # target back-off: the engineering key needs an exact 0.0


def _cap_shares_ref(sh, profile_start, profile_len, max_share):
    """(P,R) shares -> (P,R) shares with no row above `max_share`. THE REFERENCE.

    A row whose cap is not live (>= 1, <= 0, non-finite) is uncapped: its target is +inf, so it
    is never over, and its HEADROOM is clipped to 1.0 rather than left infinite — an infinite
    pool would make every share of the excess 0/inf = 0.0 and the update inf * 0.0 = NaN, which
    would silently poison the profile.

    A row the candidate put at zero is NOT a recipient. Giving it share would be inventing a
    door, and `_cap_rows` excludes it too (`W > 1e-12`).

    A profile that cannot absorb its own excess (fewer live rows than 1/cap) is left ALONE, exactly
    as `_cap_rows` leaves a row with fewer than 2 present gateways. It is counted, not hidden."""
    _s = np.asarray(sh, float)
    _cap = np.asarray(max_share, float)
    _tgt = np.where(np.isfinite(_cap) & (_cap > 0.0) & (_cap < 1.0),
                    _cap * _DC_BACKOFF, np.inf)
    _o = _s > _tgt
    if not _o.any():
        return _s
    _exc = np.add.reduceat(np.where(_o, _s - _tgt, 0.0), profile_start, axis=1)
    _ceil = np.minimum(_tgt, 1.0)
    _room = np.where(~_o & (_s > _DC_EPS) & (_s < _ceil), _ceil - _s, 0.0)
    _pool = np.add.reduceat(_room, profile_start, axis=1)
    _ok = (_exc > 0.0) & (_pool > _DC_EPS)
    if not _ok.any():
        return _s
    _okr = np.repeat(_ok, profile_len, axis=1)
    _f = np.repeat(np.where(_ok, _exc / np.where(_pool > _DC_EPS, _pool, 1.0), 0.0),
                   profile_len, axis=1)
    return np.where(_okr & _o, _tgt, np.where(_okr, _s + _room * _f, _s))


def _segment_softmax_serial(logits, profile_start, profile_len):
    """Per-profile softmax over contiguous row segments. THE REFERENCE.

    logits: (P, R). Returns shares (P, R) where each profile's rows sum to 1.
    Numerically stable (subtracts per-segment max).

    Every operation here is elementwise or runs along axis=1, so row p of the output depends only on
    row p of the input — which is what makes `_segment_softmax` below safe to thread.
    """
    logits = np.atleast_2d(logits)
    # per-segment max, expanded back to row grain
    seg_max = np.maximum.reduceat(logits, profile_start, axis=1)          # (P, n_profiles)
    row_max = np.repeat(seg_max, profile_len, axis=1)                      # (P, R)
    ex = np.exp(logits - row_max)
    seg_sum = np.add.reduceat(ex, profile_start, axis=1)                  # (P, n_profiles)
    row_sum = np.repeat(seg_sum, profile_len, axis=1)                      # (P, R)
    return ex / row_sum


def _segment_softmax(logits, profile_start, profile_len, max_share=None):
    """Row-parallel wrapper (2026-08-19bn).

    19gu: `max_share` makes the CAP PART OF THE DECODE — see [decode-cap] above. Passing None
    (or ROUTING_DECODE_CAP=0) gives the uncapped softmax exactly as before, which is what the
    kill switch and the fused-softmax self-check reference both need. [gen-cost] put this at 13.2% of a generation, all of it
    single-threaded numpy. The transform is candidate-independent (see the reference above), so the
    population is split across threads; `rowpar` verifies bit-identity on its second call and
    reverts to serial on any mismatch. ROUTING_ROW_PARALLEL=0 disables it."""
    _lg = np.atleast_2d(logits)
    # 19bx: the FUSED path, self-checked once against the untouched reference on the live
    # population. `_segment_softmax_serial` is never edited — it stays the thing this is compared
    # against, and ROUTING_SOFTMAX_FUSE=0 puts it back in the hot path.
    if _SM_OK["use"] and not _SM_OK["checked"]:
        _SM_OK["checked"] = True
        _r = _segment_softmax_serial(_lg, profile_start, profile_len)
        _f = _segment_softmax_fast(_lg, profile_start, profile_len)
        if _fx_same(_r, _f):
            _SM_OK["msg"] = (
                "[fullmatrix-ga] \u2713 fused softmax SELF-CHECK PASSED on the live population: "
                "bit-identical to the reference (int64 bit-pattern comparison on "
                f"{_lg.shape[0]}x{_lg.shape[1]:,}, stricter than array_equal). Five full-width "
                "temporaries become two; the two reduceat calls are untouched and np.exp stays in "
                "numpy (numba's differs in the last bit). Measured 206.6 -> 126.9 ms at P=35 "
                "([stage-ab], 7/7 paired rounds, p=0.016). ROUTING_SOFTMAX_FUSE=0 reverts.")
        else:
            _SM_OK["use"] = False
            _SM_OK["msg"] = (
                "[fullmatrix-ga] \u26a0 fused softmax SELF-CHECK FAILED \u2014 max|\u0394| "
                f"{float(np.abs(np.asarray(_r) - np.asarray(_f)).max()):.3e}. DISABLED for this "
                "process; the reference softmax is what ships. Report this.")
        print(_SM_OK["msg"])
    _fn = _segment_softmax_fast if _SM_OK["use"] else _segment_softmax_serial
    if max_share is None or not _DECODE_CAP:
        return _rowpar(lambda _sub: _fn(_sub, profile_start, profile_len), _lg, "softmax")
    # The cap is applied INSIDE the threaded body, not after it: row p of the capped output still
    # depends only on row p of the input (the water-fill is per (candidate, profile)), so rowpar's
    # candidate-independence premise holds and its bit-identity check still means what it says.
    return _rowpar(lambda _sub: _cap_shares_ref(_fn(_sub, profile_start, profile_len),
                                                profile_start, profile_len, max_share),
                   _lg, "softmax")


def _success_rate(shares, vol, succ, total_vol):
    """Success rate per individual, over the routed volume. shares: (P, R) -> (P,).

    VWSR = sum(vol*share*succ) / sum(profile volume). The denominator is fixed
    (shares sum to 1 within a profile), so this is LINEAR in shares.
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


def _rank(success_rate, viol, band=None):
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
    success_rate = np.asarray(success_rate, dtype=float)
    viol = np.asarray(viol, dtype=float)
    band = np.zeros_like(viol) if band is None else np.asarray(band, dtype=float)
    band_eff = np.where(band <= _FEAS_EPS, 0.0, band)
    # np.lexsort: the LAST key is primary. Want primary=band asc, then viol asc, then VWSR desc.
    return np.lexsort((-success_rate, viol, band_eff))


def _best_index(success_rate, viol, band=None):
    return _rank(success_rate, viol, band)[0]


def _key_of(success_rate, viol, band=0.0):
    """Comparable STRICT LEXICOGRAPHIC key for ONE candidate (higher tuple == better).

    Tuple `(-band, -viol, success_rate)` so, compared with `>`: smaller M5 band breach wins first, then
    smaller engineering violation, then higher VWSR — matching `_rank`. M5 breaches ≤ `_FEAS_EPS`
    snap to 0 (compliant) so a compliant split always outranks any breaching one."""
    b = 0.0 if float(band) <= _FEAS_EPS else float(band)
    return (-b, -float(viol), float(success_rate))


# ---------------------------------------------------------------------------
# Stage 4: fused evaluator (segment-softmax + VWSR + violation in ONE pass)
# ---------------------------------------------------------------------------
def _fused_eval_kernel(logits, profile_starts, profile_counts, vol, succ, risk,
                       mid_id, max_share, mid_hard, mid_soft, total_vol, n_mid,
                       mid_band_metric, mid_band_lo, mid_band_hi, global_vamp_cap,
                       decode_cap, cap_backoff, cap_eps, buf_len):
    """One-pass VWSR + violation for a whole population. Numba-compatible: only
    scalar loops + preallocated arrays (no reduceat / np.add.at / fancy index).

    Returns (success_rate[P], viol[P]) — bit-for-bit the same quantities as
    ``_success_rate(_segment_softmax(...))`` and ``_violation(_segment_softmax(...))``.

    19gu [decode-cap]: with `decode_cap` the per-profile shares are WATER-FILLED before they are
    accumulated, so this kernel decodes the same capped shares `_segment_softmax(..., max_share)`
    does. That is the whole reason it lives here rather than in a wrapper: the numpy path and this
    one must decode the SAME object, and any divergence between them is a scored-vs-delivered bug
    (see the note on -inf logits below, which exists for exactly the same reason).

    THE COST IS ONE EXTRA PASS OVER A PROFILE, and only when that profile has an over-cap row. The
    shares are held in `buf` — one scratch array per candidate, sized to the widest profile — instead
    of being consumed as they are produced, which is the only structural change to the loop.

    `share_over` is left in and still accumulated. With the cap folded in it should read 0.0 for
    every candidate; if it ever does not, the water-fill did not hold and the engineering key is
    the thing that says so.
    """
    P = logits.shape[0]
    n_profiles = profile_starts.shape[0]
    success_rate = np.zeros(P)
    viol = np.zeros(P)
    # Each candidate i is fully independent (own local accumulators, writes only
    # success rate[i]/viol[i] — no cross-candidate reduction), so _prange parallelises across
    # cores WITHOUT reordering any float sum. Bit-identical to serial; the verify-gate
    # confirms it. (numba absent -> _prange is range, runs serially.)
    for i in _prange(P):
        num_v = 0.0
        share_over = 0.0
        mnum = np.zeros(n_mid)
        mden = np.zeros(n_mid)
        buf = np.zeros(buf_len)          # per-candidate scratch, widest profile (19gu)
        for c in range(n_profiles):
            s = profile_starts[c]
            n = profile_counts[c]
            # per-profile softmax (stable): max, exp-sum, then shares
            m = logits[i, s]
            for j in range(1, n):
                lj = logits[i, s + j]
                if lj > m:
                    m = lj
            ssum = 0.0
            for j in range(n):
                ssum += np.exp(logits[i, s + j] - m)
            for j in range(n):
                buf[j] = np.exp(logits[i, s + j] - m) / ssum
            if decode_cap:
                # ── THE WATER-FILL, in the same order as `_cap_shares_ref` ────────────
                # target = cap x back-off; a non-live cap gives an infinite target (never
                # over) and a headroom CEILING clipped to 1.0 — an infinite pool would make
                # every share of the excess 0/inf = 0.0 and the update inf * 0.0 = NaN.
                exc = 0.0
                pool = 0.0
                for j in range(n):
                    ms = max_share[s + j]
                    if ms > 0.0 and ms < 1.0:
                        tg = ms * cap_backoff
                    else:
                        tg = np.inf
                    if buf[j] > tg:
                        exc += buf[j] - tg
                    else:
                        cl = tg if tg < 1.0 else 1.0
                        if buf[j] > cap_eps and buf[j] < cl:
                            pool += cl - buf[j]
                if exc > 0.0 and pool > cap_eps:
                    f = exc / pool
                    for j in range(n):
                        ms = max_share[s + j]
                        if ms > 0.0 and ms < 1.0:
                            tg = ms * cap_backoff
                        else:
                            tg = np.inf
                        if buf[j] > tg:
                            buf[j] = tg
                        else:
                            cl = tg if tg < 1.0 else 1.0
                            if buf[j] > cap_eps and buf[j] < cl:
                                buf[j] += (cl - buf[j]) * f
            for j in range(n):
                r = s + j
                sh = buf[j]
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
        success_rate[i] = num_v / total_vol
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
    return success_rate, viol


if _HAS_NUMBA:                                       # pragma: no cover - env dependent
    _fused_eval_kernel = _njit(cache=True, parallel=True)(_fused_eval_kernel)


# ---------------------------------------------------------------------------
# Compressibility regularizer — VECTOR-QUANTIZATION distortion vs a learned codebook
# ---------------------------------------------------------------------------
# The λ_compress reward pushes profiles to route ALIKE so the final split collapses into
# few deployable configs. Concretely we learn a CODEBOOK of ~pool-target centroid shapes
# (volume-weighted k-means over the ELITE's per-vampMid profile shapes, refreshed as the
# elite improves — the same KIND of clustering tab-3 compression uses) and penalise each
# candidate by its volume-weighted quantization error against that codebook:
#     D_i = Σ_profiles vol_c · ‖shape_ic − centroid[assign_c]‖²  / total_vol .
# shape_ic[m] = Σ_{rows in profile c with vampMid m} share is the profile's shape (0 on absent
# mids). This is NOT a 'fewer-gateways' reward: two profiles with the SAME shape (however
# spread across mids) cost 0; a profile that routes DIFFERENTLY from its codebook centroid
# pays. Scored on DELIVERED shares (post eligibility + blocked-caps) so it rewards what
# actually ships. Kernel is Numba-fused (verify-gated vs the numpy twin); the periodic
# codebook refit is numpy/sklearn (cheap relative to per-generation evaluation).
def _compress_distortion_kernel(shares, profile_starts, profile_counts, mid_id, vol,
                                assign, cent, cconst, n_mid, total_vol):
    """Volume-weighted VQ distortion of a population vs a FIXED codebook.

    ‖shape − c_k‖² = Σ_m shape_m² − 2 Σ_m shape_m·c_k,m + ‖c_k‖²  (‖c_k‖²=cconst[k]),
    so only a profile's OWN rows are touched; the absent-mid tail is the constant cconst[k].
    Numba-safe: scalar loops + one thread-local n_mid buffer, cleared per profile. Each
    candidate i writes only out[i] (no cross-candidate reduction) so _prange is
    bit-identical to serial — the verify-gate confirms it. Returns distortion[P].
    """
    P = shares.shape[0]
    n_profiles = profile_starts.shape[0]
    out = np.zeros(P)
    for i in _prange(P):
        buf = np.zeros(n_mid)
        acc = 0.0
        for c in range(n_profiles):
            s = profile_starts[c]
            n = profile_counts[c]
            k = assign[c]
            for j in range(n):               # accumulate this profile's per-mid shape
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


def _compress_distortion_numpy(shares, profile_start, profile_len, mid_id, vol,
                               assign, cent, cconst, n_mid, total_vol):
    """Vectorised twin of ``_compress_distortion_kernel`` (the verify-gate reference).
    ``cconst`` is accepted for signature parity but unused (the full diff is formed)."""
    sh = np.atleast_2d(np.asarray(shares, float))
    P = sh.shape[0]
    n_profiles = profile_start.shape[0]
    cm = np.zeros((P, n_profiles, n_mid))
    for m in range(n_mid):
        cm[:, :, m] = np.add.reduceat(sh * (mid_id == m), profile_start, axis=1)
    diff = cm - cent[assign][None, :, :]                     # (P, n_profiles, n_mid)
    d2 = (diff * diff).sum(axis=2)                           # (P, n_profiles)
    cvol = vol[profile_start]                                   # (n_profiles,) profile volume
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
    """Learn a codebook of centroid SHAPES from per-profile shapes (volume-weighted, so
    high-volume profiles pull the centroids — matching tab-3's compressor). K = min(pools,
    n_profiles). Returns (assign[intp] (n_profiles,), cent (K,n_mid), cconst (K,) = ‖cent‖²)."""
    X = np.ascontiguousarray(shape_mat, float)
    w = np.maximum(np.asarray(cvol, float), 1e-12)
    n_profiles = X.shape[0]
    K = int(max(1, min(int(pools), n_profiles)))
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


def _profile_shape_matrix(shares_row, profile_start, mid_id, n_mid):
    """Per-profile per-vampMid shape matrix for ONE split: (n_profiles, n_mid)."""
    sh = np.asarray(shares_row, float)[None, :]
    n_profiles = profile_start.shape[0]
    cm = np.zeros((n_profiles, n_mid))
    for m in range(n_mid):
        cm[:, m] = np.add.reduceat(sh[0] * (mid_id == m), profile_start)
    return cm


def make_distortion(problem, *, use_numba=False, verify=True, rng=None):
    """Return (dist_fn, backend). dist_fn(shares_pop, assign, cent, cconst) -> D[P].
    numpy by default; the njit kernel is verified bit-exact vs numpy on a random codebook
    (any mismatch/error → numpy), so the codebook may change each refresh without re-verify."""
    p = problem
    cs = np.ascontiguousarray(p.profile_start, np.intp)
    cc = np.ascontiguousarray(p.profile_len, np.intp)
    mid = np.ascontiguousarray(p.mid_id, np.intp)
    vol = np.ascontiguousarray(p.vol, float)
    nmid = int(p.n_mids)
    tv = float(_profile_volume_total(p)) or 1.0

    def _np(shares, assign, cent, cconst):
        return _compress_distortion_numpy(shares, cs, cc, mid, vol,
                                          np.asarray(assign, np.intp), cent, cconst, nmid, tv)

    fn, backend = _np, "numpy"
    if use_numba and _HAS_NUMBA:
        try:                                         # pragma: no cover - env dependent
            _rng = rng or np.random.default_rng(0)
            K = int(max(1, min(8, p.n_profiles)))
            a = _rng.integers(0, K, size=p.n_profiles).astype(np.intp)
            ct = np.ascontiguousarray(_rng.random((K, nmid)), float)
            kc = np.ascontiguousarray((ct * ct).sum(axis=1), float)
            _R = int(p.profile_id.shape[0])
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
    """Return (eval_pop, info). eval_pop(logits_pop) -> (success_rate, viol).

    numpy by default. With ``use_numba`` and Numba present, the fused njit kernel
    is VERIFIED against the numpy path on a random sample (allclose); any mismatch
    or error falls back to numpy — so the kernel can never change results.
    """
    p = problem
    total_vol = _profile_volume_total(p)

    def _numpy_eval(logits):
        shares = _segment_softmax(logits, p.profile_start, p.profile_len, p.max_share)
        return (_success_rate(shares, p.vol, p.succ, total_vol), _violation(shares, p))

    if not use_numba or not _HAS_NUMBA:
        return _numpy_eval, {"backend": "numpy",
                             "reason": ("numba absent" if use_numba else "numpy requested")}

    # Bit-identical memory-layout prep, done ONCE (these are constant across every
    # eval call). Index arrays -> int32 (half the bandwidth; integer indexing is
    # exact, so the float accumulation is unchanged). Float arrays -> contiguous
    # float64 (arithmetic and its order untouched). This only changes cache/SIMD
    # behaviour, never a single floating-point op.
    _k_cs = np.ascontiguousarray(p.profile_start, dtype=np.int32)
    _k_cc = np.ascontiguousarray(p.profile_len, dtype=np.int32)
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
    # 19gu [decode-cap]: the kernel now water-fills each profile before accumulating, so it decodes
    # the SAME capped shares `_numpy_eval` does. `buf_len` sizes the per-candidate scratch to the
    # widest profile — passed in rather than derived inside, so the kernel stays allocation-free
    # except for that one array and numba can type it.
    _k_dc = bool(_DECODE_CAP)
    _k_bo = float(_DC_BACKOFF)
    _k_ce = float(_DC_EPS)
    _k_bl = int(max(1, int(np.asarray(p.profile_len).max()) if len(p.profile_len) else 1))

    def _numba_eval(logits):
        return _fused_eval_kernel(
            np.ascontiguousarray(logits, dtype=np.float64), _k_cs, _k_cc,
            _k_vol, _k_succ, _k_risk, _k_mid, _k_ms, _k_mh, _k_msoft, _k_tv, _k_nm,
            _k_bm, _k_blo, _k_bhi, _k_gvc, _k_dc, _k_bo, _k_ce, _k_bl)

    if verify:
        rng = rng or np.random.default_rng(0)
        samp = rng.standard_normal((8, p.profile_id.shape[0]))
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
        # 19gu: this gate now covers the CAPPED decode on both sides, which is exactly what it is
        # for — the numpy path and the kernel must decode the same object, and the water-fill is
        # the newest way they could stop doing so. The sample is random logits, so it exercises
        # over-cap profiles: with a 0.97 cap and ~10 rows a profile, some rows land over it.
        return _numba_eval, {"backend": "numba", "verified": True,
                             "decode_cap": bool(_DECODE_CAP)}
    return _numba_eval, {"backend": "numba", "verified": False}


# ---------------------------------------------------------------------------
# GA operators (act on the logit genome)
# ---------------------------------------------------------------------------
def _shares_to_logits(shares, eps=1e-6, hard_zero=False, profile_start=None,
                      profile_len=None, info=None):
    """Shares -> the logit genome. `hard_zero` (19cm) encodes an exact zero as -inf.

    WHY -inf AND NOT A SMALLER eps. The damage this repairs is the vshare cliff, and vshare
    SELF-NORMALISES: where a breached MID is the only VAMP-positive gateway in a profile, its vshare is
    s/s = 1 for any s > 0. The cliff is scale-invariant, so 1e-30 is exactly as bad as 1e-6 and only
    a TRUE zero changes anything.

    WHY -inf AND NOT A MASK. A mask would have to be threaded through both softmax implementations —
    `_segment_softmax` and the numba `_fused_eval_kernel` — and any divergence between those two is
    a scored-vs-delivered bug. Both are the standard max-subtract stable softmax, so a -inf logit
    gives exp(-inf - m) = 0 EXACTLY in both with no kernel change at all. The zero lives in the DATA,
    so the two paths cannot disagree about it.

    IT SURVIVES THE OPERATORS FOR FREE: -inf + noise is -inf, and `_crossover` moves whole profile
    segments, so a seed's zeros are inherited by its descendants. `_init_pop`'s unanchored
    exploration children carry no -inf, so the search space is NOT permanently narrowed — the seed's
    structure merely becomes representable.

    THE ONE WAY IT COULD PRODUCE nan: a profile with every row masked has max = -inf, and
    -inf - (-inf) is nan. A valid seed sums to 1 in every profile so this cannot arise, but it is
    checked rather than assumed, and such a profile is left un-masked with the count reported."""
    _sh = np.asarray(shares, float)
    _out = np.log(np.clip(_sh, eps, None))
    if not hard_zero:
        return _out
    _z = (_sh <= 0.0)
    if profile_start is not None and profile_len is not None and _z.any():
        # never empty a profile: that is the only input that turns the stable softmax into nan
        _cs = np.asarray(profile_start, np.intp)
        _cl = np.asarray(profile_len, np.intp)
        _live = np.add.reduceat((~_z).astype(np.int64), _cs) if _cs.size else np.zeros(0, np.int64)
        _dead = np.repeat(_live <= 0, _cl)
        _skipped = int((_z & _dead).sum())
        _z = _z & (~_dead)
        if isinstance(info, dict):
            info["profiles_all_zero"] = int((_live <= 0).sum())
            info["rows_unmasked_to_avoid_nan"] = _skipped
    _out = np.where(_z, -np.inf, _out)
    if isinstance(info, dict):
        info["rows_hard_zeroed"] = int(_z.sum())
    return _out


def _greedy_reference(p: "FullMatrixProblem"):
    """Fallback seed if the caller gives no reference: per-profile softmax over
    success (a conversion-greedy split). Returns shares (R,)."""
    logits = p.succ * 6.0   # mild temperature; only a SEED, GA refines from here
    return _segment_softmax(logits[None, :], p.profile_start, p.profile_len, p.max_share)[0]


def _crossover(a, b, profile_start, profile_len, rng):
    """Uniform per-PROFILE crossover: each profile's whole logit segment comes from one
    parent (keeps within-profile simplex structure intact)."""
    n_profiles = len(profile_start)
    pick = rng.random(n_profiles) < 0.5
    row_pick = np.repeat(pick, profile_len)
    return np.where(row_pick, a, b)


def _mutate(logits, rate, strength, profile_start, profile_len, rng, profile_w=None):
    """Gaussian perturbation of a fraction of PROFILES' logit segments.

    `profile_w`: optional (n_profiles,) multiplier on each profile's SELECTION PROBABILITY (not on the
    noise). Added 2026-08-19ab for breach-targeted mutation — profiles feeding a still-breached band
    get boosted so the mutation budget lands where the shortfall is instead of being spread
    uniformly over every profile, most of which feed already-compliant MIDs.

    The RNG DRAW COUNT is deliberately unchanged: still exactly one `rng.random(n_profiles)` and one
    `standard_normal(logits.shape)`, in that order. Only the threshold each draw is compared
    against moves. So `profile_w=None` (or all-ones) reproduces the pre-19ab search BIT-IDENTICALLY,
    random stream included — which is what makes ROUTING_MUT_TARGET=0 a true revert rather than a
    different-but-similar search. Do not add or reorder draws here.

    Probabilities are clipped to 1.0: a boosted rate above 1 would otherwise be a silent no-op
    (every draw already below it), making a large boost indistinguishable from a moderate one."""
    n_profiles = len(profile_start)
    _thr = rate if profile_w is None else np.minimum(np.asarray(profile_w, float) * rate, 1.0)
    hit = rng.random(n_profiles) < _thr
    row_hit = np.repeat(hit, profile_len)
    noise = rng.standard_normal(logits.shape) * strength
    return logits + np.where(row_hit, noise, 0.0)


# 19bp: MUTATION WITHOUT THE WASTE. `_mutate` above draws one Gaussian per ROW and discards every
# one whose profile was not selected — at a 1% profile rate that is ~99% waste, and at 35 children over
# 245,409 rows it is 8.6 MILLION discarded draws per generation, ~170 ms of the ~300 ms [gen-cost]
# attributes to `genetic` (2026-08-23).
#
# This draws only the Gaussians it uses. That CHANGES THE RANDOM STREAM, which is why it needed
# sign-off: the search sees different children from here on. It is not less random and not less
# correct — it is a different sample. Two things make it safe to live with:
#
#   1. It is DETERMINISTIC. The caller gives each child its own stream keyed on
#      (seed, seed-index, restart, generation, child), so a rerun reproduces a run exactly, and a
#      child's numbers no longer depend on how many draws its siblings happened to take. That
#      independence is also what would make the child loop threadable — though after this there is
#      little left to thread, which is the better outcome.
#   2. `_mutate` is UNTOUCHED and ROUTING_MUT_FAST=0 runs it on the old shared generator, so the
#      previous answer is one env var away for comparison.
#
# The SHAPE of the perturbation is identical: the same per-profile selection at the same probability,
# the same Gaussian scale, applied to the same whole-profile segments.
def _mutate_fast(logits, rate, strength, profile_start, profile_len, rng, profile_w=None):
    """`_mutate`'s twin, drawing Gaussians only for the rows it perturbs.

    NOT bit-identical to `_mutate` and not intended to be — see the note above. Same distribution,
    same per-profile selection rule, a fraction of the draws."""
    n_profiles = len(profile_start)
    _thr = rate if profile_w is None else np.minimum(np.asarray(profile_w, float) * rate, 1.0)
    hit = rng.random(n_profiles) < _thr
    if not hit.any():
        return logits.copy()
    row_hit = np.repeat(hit, profile_len)
    n_hit = int(row_hit.sum())
    out = logits.copy()
    # `out[row_hit] += noise` — a masked add over the selected rows only. The non-selected rows are
    # copied through untouched, where `_mutate` added an exact 0.0 to them.
    out[row_hit] += rng.standard_normal(n_hit) * strength
    return out


def _child_streams(base_seed, s_idx, r_idx, gen, n):
    """One independent, reproducible Generator per child.

    SeedSequence.spawn is the supported way to make independent streams; deriving them from a tuple
    of counters means the run is reproducible from (seed, seed-index, restart, generation, child)
    alone, with no dependence on draw order anywhere else."""
    return [np.random.default_rng(_ss) for _ss in
            np.random.SeedSequence([int(base_seed), int(s_idx), int(r_idx), int(gen)]).spawn(int(n))]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_fullmatrix_ga(problem: "FullMatrixProblem", reference_shares=None, *,
                      pop_size=60, generations=120, elite=6,
                      mutation_rate=0.3, mutation_strength=0.4,
                      patience=25, seed=0, log_fn=None, numba=False,
                      band_penalty_fn=None, band_report_fn=None, mut_weight_fn=None,
                      n_seeds=1, restarts=1,
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
    # 19bp: ROUTING_MUT_FAST=0 restores the pre-19bp search EXACTLY — one shared sequential
    # generator and a full-width Gaussian draw per child. On (the default) it draws only the
    # Gaussians it uses, from one independent stream per child. Different answer, same search.
    _MUT_FAST = os.environ.get("ROUTING_MUT_FAST", "1") != "0"
    rng = np.random.default_rng(seed)
    R = p.profile_id.shape[0]
    total_vol = _profile_volume_total(p)

    log = log_fn or (lambda *_a, **_k: None)
    log(f"[fullmatrix-ga] build={__build__} R={R} profiles={p.n_profiles} mids={p.n_mids}")

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
    # COMPRESSIBILITY REGULARIZER (λ_compress ≥ 0). Rewards profiles COLLAPSING ONTO A SHARED CODEBOOK of
    # shapes so the split compresses into few deployable configs. We LEARN a codebook of ≈`compress_pools`
    # centroid shapes (volume-weighted k-means over the ELITE's per-vampMid profile shapes — the same KIND of
    # clustering tab-3 compression uses — refreshed every `compress_refresh` gens as the elite improves) and
    # subtract from VWSR the volume-weighted VQ distortion of each candidate against that codebook
    # (‖profile_shape − nearest-assigned centroid‖²). This is NOT a 'fewer-gateways' reward: two profiles with the
    # SAME shape (however spread across mids) cost 0; a profile that routes DIFFERENTLY from its centroid pays.
    # Scored on DELIVERED shares (post blocked-caps + eligibility via `deliver_fn`) so it rewards what
    # actually ships — a gateway a profile can't use (Country / paymentMethodProvider capability) never enters
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
        _cvol = np.asarray(p.vol[p.profile_start], float)            # (n_profiles,) profile volume
        _dist_fn, _dist_backend = make_distortion(p, use_numba=numba)

        def _refresh_codebook(logits_row):
            """(Re)learn the codebook from ONE split's DELIVERED per-profile shapes.

            When ``codebook_fn`` is supplied (the caller's EXACT tab-3 ward/knapsack
            compressor), it maps the delivered GA-grain shares to (assign, cent, cconst)
            at the per-vampMid grain — so the regularizer's target IS the split tab-3
            would ship. On any failure we fall back to the internal volume-weighted
            k-means codebook (logged, never silent), so the search never stalls."""
            _s = _segment_softmax(np.asarray(logits_row, float)[None, :], p.profile_start,
                                  p.profile_len, p.max_share)
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
            _shape = _profile_shape_matrix(_sd[0], p.profile_start, p.mid_id, _nmid)
            _a, _ct, _kc = _fit_codebook(_shape, _cvol, compress_pools, int(seed))
            _cb["assign"], _cb["cent"], _cb["cconst"], _cb["src"] = _a, _ct, _kc, "kmeans"

    # ── [eval-cost] 19gw: WHERE THE `eval` ROW ACTUALLY GOES ────────────────────────────
    # [gen-gap] puts `eval` at 336.1s of a 401.9s search on the 2026-09-01 23:12 run — 92% of
    # every generation — and then stops. Working backwards from [proj-config] and [lift-ab], the
    # band projector is only ~68s of that; the other ~268s has never had a breakdown at all. The
    # log still refers to a [gen-cost] block that models these five stages, but that block is
    # long gone, so nothing in a run says which stage owns the time.
    #
    # Same treatment as [cvp-timing] and [cap-timing], both of which named their answer on the
    # first run that carried them. Four wall-clock accumulators, summed over every call, printed
    # beside [gen-gap] so the two can be read against each other: these SUM to `eval`, so there
    # is no residual to argue about.
    #
    # Timers only. Nothing computed here moved, and `perf_counter` around four calls that each
    # take tens of milliseconds is not measurable against the work itself.
    _ev = {"pop": 0.0, "decode": 0.0, "deliver": 0.0, "band": 0.0, "comp": 0.0, "n": 0}

    def _eval_with_bands(logits):
        # Returns (success rate, other_viol, band_breach) as THREE separate arrays so the ranking can treat
        # the EXACT M5 band breach as the strict primary key (see _rank). `other_viol` is the
        # engineering violation (global VAMP cap + max-share) from eval_pop; `band_breach` is the
        # exact per-MID M5 penalty. They are NO LONGER summed — the ranking orders on band first.
        _ev["n"] += 1
        _t = time.perf_counter()
        v, x = eval_pop(logits)
        _ev["pop"] += time.perf_counter() - _t
        _band = np.zeros(np.asarray(x).shape[0], dtype=float)
        _need_band = band_penalty_fn is not None
        _need_comp = _compress_on and _cb["cent"] is not None
        if not (_need_band or _need_comp):
            return v, x, _band
        _t = time.perf_counter()
        _sh = _segment_softmax(logits, p.profile_start, p.profile_len, p.max_share)
        _ev["decode"] += time.perf_counter() - _t
        _t = time.perf_counter()
        _fd = _deliver_full(_sh)                                  # shared delivery — computed ONCE
        _ev["deliver"] += time.perf_counter() - _t
        if _need_band:
            _t = time.perf_counter()
            _band = np.asarray(band_penalty_fn(_fd if _have_full else _sh), dtype=float)
            _ev["band"] += time.perf_counter() - _t
        if _need_comp:
            _t = time.perf_counter()
            _sd = _deliver_kept(_sh, _fd)
            v = v - _clam * np.asarray(
                _dist_fn(_sd, _cb["assign"], _cb["cent"], _cb["cconst"]), dtype=float)
            _ev["comp"] += time.perf_counter() - _t
        return v, x, _band

    def _rescore_compress(logits, keep_other, keep_band):
        """Re-score ONLY the compress term of success_rate under the CURRENT codebook, keeping the supplied
        engineering violation AND band breach unchanged (both are codebook-independent, so they are
        NOT recomputed — this is what lets a codebook refresh skip the whole-population band
        projection). Returns (success_rate, keep_other, keep_band)."""
        _bv, _ = eval_pop(logits)
        if _compress_on and _cb["cent"] is not None:
            _sh = _segment_softmax(logits, p.profile_start, p.profile_len, p.max_share)
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
    # 19cm: ROUTING_SEED_ZEROS=1 encodes the seed's exact zeros as -inf logits, so the decode can
    # reproduce them. DEFAULT OFF: it is a BEHAVIOUR change (the seed's descendants inherit those
    # zeros), and [decode-loss] below is what prices it. `_compress_on` runs k-means on the logit
    # genome, which -inf would poison, so the two are refused together rather than silently mixed.
    # FIXED ON (2026-08-31). Was ROUTING_SEED_ZEROS, set to 1 in routing.env on every run, so
    # this IS the shipped behaviour and is no longer switchable. It encodes the seed's exact zeros
    # as -inf logits so the decode can reproduce them; the seed's descendants inherit those zeros.
    # `_compress_on` still overrides it (k-means over the logit genome cannot survive -inf) and the
    # refusal is still logged, so the two can never be silently mixed.
    _hz_want = True
    _hz_info = {}
    _hz_on = bool(_hz_want and not _compress_on)
    seed_logits = _shares_to_logits(
        seed_shares, hard_zero=_hz_on, profile_start=p.profile_start, profile_len=p.profile_len,
        info=_hz_info)
    if _hz_want and _compress_on:
        log("[fullmatrix-ga] \u26a0 ROUTING_SEED_ZEROS=1 REFUSED this run: the compressibility "
            "regulariser is on and it runs k-means over the LOGIT genome, which -inf would turn "
            "into nan centroids. The seed is encoded the old way; [decode-loss] below still "
            "prices what that costs. Turn the regulariser off to use it.")

    # 19gu/19gv: with the cap in the decode the seed needs no special treatment. Its float-dust
    # violation (rows sitting at exactly 97.0000%) is capped when it is decoded, like every other
    # candidate, so it cannot be out-ranked on the engineering key by children that repaired to an
    # exact 0.0 — which was the whole reason the seed had to be repaired separately.

    # ── [decode-cap] SELF-CHECK ON THE LIVE SEED (19gu) ───────────────────────────────────
    # Two claims, both checked on this run's own data rather than asserted: the capped decode
    # HOLDS the cap, and it is the same object `_cap_shares_ref` produces from the uncapped
    # decode. The second is what stops the numpy and numba paths drifting apart — make_fused_eval's
    # verify gate covers the kernel, this covers the wrapper.
    if _DECODE_CAP:
        try:
            _dc_lg = np.asarray(seed_logits, float)[None, :]
            _dc_raw = _segment_softmax(_dc_lg, p.profile_start, p.profile_len)
            _dc_got = _segment_softmax(_dc_lg, p.profile_start, p.profile_len, p.max_share)
            _dc_ref = _cap_shares_ref(_dc_raw, p.profile_start, p.profile_len, p.max_share)
            _dc_cap = np.asarray(p.max_share, float)
            _dc_liv = np.isfinite(_dc_cap) & (_dc_cap > 0.0) & (_dc_cap < 1.0)
            _dc_over_before = int(((_dc_raw[0] > _dc_cap) & _dc_liv).sum())
            _dc_over_after = int(((_dc_got[0] > _dc_cap) & _dc_liv).sum())
            _dc_moved = float(np.abs(_dc_got - _dc_raw).sum())
            _dc_same = _fx_same(_dc_ref, _dc_got)
            _dc_sums = float(np.abs(
                np.add.reduceat(_dc_got[0], np.asarray(p.profile_start, np.intp)) - 1.0).max())
            if _dc_over_after == 0 and _dc_same:
                log(f"[fullmatrix-ga] [decode-cap] ✓ SELF-CHECK PASSED on the live seed: the "
                    f"uncapped decode held {_dc_over_before:,} row(s) above the cap, the capped "
                    f"decode holds 0, and it is BIT-IDENTICAL to the reference water-fill applied "
                    f"to the uncapped decode (int64 bit-pattern comparison on "
                    f"1x{_dc_got.shape[1]:,}, stricter than array_equal). Total share moved "
                    f"{_dc_moved:.4g}; worst profile sum error {_dc_sums:.2e} (each profile must still "
                    "sum to 1 — the water-fill moves share between rows, it never creates or "
                    "destroys any).")
            else:
                log(f"[fullmatrix-ga] [decode-cap] ⚠⚠ SELF-CHECK FAILED on the live seed: "
                    f"{_dc_over_after:,} row(s) are STILL above the cap after the capped decode"
                    + ("" if _dc_same else ", and the capped decode is NOT bit-identical to the "
                                           "reference water-fill")
                    + f" (worst profile sum error {_dc_sums:.2e}). The engineering key below is the "
                      "backstop and will show it, but do NOT trust this run's split. Set "
                      "ROUTING_DECODE_CAP=0 and re-run.")
        except Exception as _dce:  # noqa: BLE001
            log(f"[fullmatrix-ga] [decode-cap] self-check SKIPPED "
                f"({type(_dce).__name__}: {_dce}) — MEASUREMENT ONLY, the decode is unaffected, "
                "but this run has no live proof the cap holds.")

    # remember the elite seed's key for the never-worse guarantee (bands included)
    s0 = _segment_softmax(seed_logits[None, :], p.profile_start, p.profile_len, p.max_share)
    seed_success_rate = _success_rate(s0, p.vol, p.succ, total_vol)[0]
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
        seed_success_rate = seed_success_rate - _clam * float(np.asarray(
            _dist_fn(_sd0, _cb["assign"], _cb["cent"], _cb["cconst"])).reshape(-1)[0])
        _cb_src = ("EXACT tab-3 ward/knapsack allocator" if _cb.get("src") == "callback"
                   else "internal volume-weighted k-means")
        log(f"[fullmatrix-ga] compressibility regularizer ON: λ={_clam:g}, codebook={_k0} centroid rows "
            f"via {_cb_src} (target {int(compress_pools)} pools, refit every {int(compress_refresh)} gens), "
            f"distortion backend={_dist_backend}, delivery-dedupe={'ON' if _have_full else 'off'}, "
            f"scored on DELIVERED shares "
            f"({'eligibility-aware' if (_have_full or callable(deliver_fn)) else 'raw — no deliver_fn'}) "
            "— VWSR −= λ·volume-weighted VQ distortion; pushes profiles to route ALIKE so the split "
            "compresses into fewer configs; trades a little conversion.")
    # ── [decode-loss] 19cm: WHAT THE SEED LOSES ON THE WAY INTO THE GENOME ───────────────────
    # The [never-worse] block has named this mechanism for several builds without ever pricing it.
    # On 2026-08-26 the seed's breach was 0.6419 and the GA's first generation read 0.6461, so the
    # search began BEHIND the thing it was seeded from and spent 288 s climbing back. This block
    # measures that gap directly instead of leaving it to be inferred from two blocks a thousand
    # lines apart.
    try:
        _dl_seed = np.asarray(seed_shares, float)[None, :]
        _dl_dec = np.asarray(s0, float)
        _dl_z = (_dl_seed[0] <= 0.0)
        _dl_res = _dl_z & (_dl_dec[0] > 0.0)
        _dl_nres = int(_dl_res.sum())
        _dl_mass = float(_dl_dec[0][_dl_res].sum())
        _dl_vw = float(_success_rate(_dl_seed, p.vol, p.succ, total_vol)[0])
        _dl_bd = None
        if band_penalty_fn is not None:
            _dl_fd = _deliver_full(_dl_seed) if (_have_full or _compress_on) else None
            _dl_bd = float(np.asarray(band_penalty_fn(
                _dl_fd if _have_full else _dl_seed), dtype=float)[0])
        log("[fullmatrix-ga] == [decode-loss] WHAT THE SEED LOSES ENTERING THE LOGIT GENOME "
            "(read-only) ==")
        log(f"[fullmatrix-ga]    {_dl_nres:,} of {int(_dl_z.sum()):,} EXACTLY-ZERO seed row(s) come "
            f"back NON-ZERO, carrying {_dl_mass:.4g} of share invented from nothing. A logit "
            "genome has no finite value for zero, so softmax(log(clip(s, 1e-6))) cannot express "
            "one.")
        if _dl_bd is not None:
            _dl_gap = seed_band - _dl_bd
            log(f"[fullmatrix-ga]    EXACT M5 BREACH: the seed itself {_dl_bd:.6g} \u2192 the "
                f"decoded seed the GA actually starts from {seed_band:.6g} "
                f"({_dl_gap:+.6g}). THIS IS THE HANDICAP THE SEARCH BEGINS WITH \u2014 compare it "
                "with how much the whole search then gains, in the final success rate/breach line.")
            if _dl_gap > 0:
                log("[fullmatrix-ga]    \u26a0 THE ENCODING MADE THE SEED WORSE BEFORE A SINGLE "
                    "GENERATION RAN. Whatever the search gains has to cover this first. This is "
                    "why [never-worse] can ship the SEED after a full search: the seed it "
                    "re-scores at the end is the TRUE one, not the decoded copy the GA optimised.")
        log(f"[fullmatrix-ga]    success rate: seed {_dl_vw:.6f} \u2192 decoded {seed_success_rate:.6f} "
            f"({seed_success_rate - _dl_vw:+.6f}).")
        # WHERE THE INVENTED SHARE WENT. This one IS measurable here, and it identifies the
        # mechanism exactly: if the per-row figure comes out at the clip floor, every resurrected
        # row was pinned at eps and then normalised, which is the encoding and nothing else.
        if _dl_nres:
            log(f"[fullmatrix-ga]    THAT IS {_dl_mass / max(_dl_nres, 1):.3g} of share PER "
                f"RESURRECTED ROW, against a clip floor of 1e-06 \u2014 so each zero came back at "
                "the floor and was then normalised. That is the encoding, not the search.")
        # THE CLIFF POPULATION, AND WHY IT IS NOT PRINTED AS A NUMBER ON THIS PIPELINE.
        # The count wants "rows that are the ONLY VAMP-positive gateway in their profile", and the
        # only VAMP-positive signal reachable from here is `p.risk`. But the run's risk-seeding
        # step gives every gateway-profile with no VAMP data the weighted-average rate rather than 0
        # ("so 0-VAMP gateways aren't treated as risk-free"), which makes `p.risk` DENSE and the
        # count structurally zero whatever the truth is. The 2026-08-26 20:28 run printed that
        # zero beside a VAMP-POSITIVE SIBLING block reporting 13,425 sole-VAMP profiles for braintree
        # usa alone. A measured zero and an unmeasurable one must not print alike (19ce D4, 19cj).
        if _dl_nres:
            _dl_pos = (np.asarray(p.risk, float) > 0.0)
            _dl_dense = float(_dl_pos.mean())
            if _dl_dense > 0.99:
                log(f"[fullmatrix-ga]    THE CLIFF COUNT IS NOT AVAILABLE FROM HERE and is "
                    f"therefore NOT PRINTED: {_dl_dense:.1%} of rows carry a positive VAMP rate, "
                    "because risk-seeding gives gateway-profiles with no VAMP data the weighted-"
                    "average rate rather than 0. On that vector no profile can have exactly one "
                    "VAMP-positive gateway, so any count computed here would be a structural "
                    "zero and would read as 'the cliff is not the mechanism'. It is measured "
                    "properly, on the projector scaffold's vcpos, by the VAMP-POSITIVE SIBLING "
                    "block above \u2014 read the count there.")
            else:
                _dl_cnt = np.add.reduceat(_dl_pos.astype(np.int64),
                                          np.asarray(p.profile_start, np.intp))
                _dl_sole = np.repeat(_dl_cnt <= 1, np.asarray(p.profile_len, np.intp)) & _dl_pos
                log(f"[fullmatrix-ga]    OF THOSE, {int((_dl_res & _dl_sole).sum()):,} sit in a "
                    "profile where they are the ONLY VAMP-positive gateway. vshare self-normalises, "
                    "so there a resurrected share of ANY size returns the WHOLE profile's VAMP "
                    "\u2014 which is why a smaller clip than 1e-6 fixes nothing: the cliff is "
                    "scale-invariant. (APPROXIMATE: counted on the GA's per-row risk, not the "
                    "projector scaffold's vcpos, so it is NOT the same number as the "
                    "VAMP-POSITIVE SIBLING block's and must not be quoted as one.)")
        if _hz_on:
            log(f"[fullmatrix-ga]    ROUTING_SEED_ZEROS=1 IS ON: "
                f"{int(_hz_info.get('rows_hard_zeroed', 0)):,} row(s) encoded as -inf logits, "
                "which the stable softmax turns into exact zeros in BOTH the numpy and numba "
                "kernels with no code change to either. The figures above are what remains AFTER "
                "that, so they should read ~0; anything else is a defect in this fix. -inf + noise "
                "is -inf, so the seed's descendants inherit the zeros, while `_init_pop`'s "
                "unanchored exploration children do not \u2014 the search space is not narrowed.")
            if int(_hz_info.get("rows_unmasked_to_avoid_nan", 0)):
                log(f"[fullmatrix-ga]    \u26a0 {int(_hz_info['rows_unmasked_to_avoid_nan']):,} "
                    f"row(s) in {int(_hz_info.get('profiles_all_zero', 0)):,} ALL-ZERO profile(s) were "
                    "left un-masked: masking every row of a profile makes the stable softmax nan. A "
                    "profile of a valid seed sums to 1, so this should not happen \u2014 report it.")
        else:
            log("[fullmatrix-ga]    ROUTING_SEED_ZEROS=0 (default): the seed is encoded the old "
                "way and the gap above is being PAID this run. Setting it to 1 encodes exact "
                "zeros as -inf logits. That is a BEHAVIOUR change \u2014 the seed's descendants "
                "inherit its zeros \u2014 so it is off until this block has priced it on a run "
                "that matters.")
    except Exception as _dl_e:                       # noqa: BLE001 — a measurement must not break a search
        log(f"[fullmatrix-ga]    [decode-loss] skipped ({type(_dl_e).__name__}: {_dl_e}). The "
            "search itself is unaffected.")

    seed_key = _key_of(seed_success_rate, seed_other, seed_band)
    best_logits = seed_logits.copy()
    best_key = seed_key
    best_success_rate, best_other, best_band = seed_success_rate, seed_other, seed_band

    # ── [ga-census] 19eb + 19ed ───────────────────────────────────────────────────────
    # Counts, per generation, WHY each child failed to displace the incumbent, and (19ed)
    # HOW BIG the blocking violation was. Reads arrays the loop already computed; the one
    # added computation is a single-candidate decomposition once per generation.
    # [ga-census] PROBE REMOVED (2026-08-31): read-only, permanently off. It counted how many
    # children outranked the incumbent and printed a counterfactual over five tolerances that
    # returned the SAME success rate at every one (0.594275 vs 0.594275 across 1e-12..1e-1 on the
    # 16:00 run) - i.e. it re-answered a settled question every generation. It also contradicted
    # itself on that run, reporting "displaced the incumbent: 1,056" and then "1,056 child(ren)
    # outranked the incumbent and it did not move ... an UPDATE fault". Both cannot be true; the
    # miscount goes with the block.
    _cen_on = False
    _cen_dec_on = _cen_on and _os_gf.environ.get("ROUTING_GA_CENSUS_DECOMP", "1") != "0"
    _cen0 = {"kids": 0, "feas": 0, "vbet": 0, "feas_vbet": 0, "blocked_other": 0,
             "won": 0, "won_eng": 0, "minband": float("inf"), "bestfeas": float("-inf")}
    _cen = dict(_cen0)                      # run total

    # 19ed: cumulative magnitude buckets for the BLOCKED group (compliant + converts better
    # + rejected on `other`). Dust and real breaches produce identical COUNTS and completely
    # different distributions, and only the distribution decides what the fix should be.
    _CEN_CUTS = (1e-12, 1e-9, 1e-6, 1e-3, 1e-1)
    _cen_mag = [0] * (len(_CEN_CUTS) + 1)
    _cen_blocked_vals = []
    # ... and the counterfactual: what a snap-to-zero at each eps WOULD have unlocked.
    _cen_eps = {e: {"n": 0, "best": float("-inf")} for e in _CEN_CUTS}

    # [FN-ga-census]
    def _cen_add(dst, cb, cv, co, bv, bo, keep=False):
        """Tally one batch against the incumbent (bv=best success_rate, bo=best engineering viol).

        THE WINNER TEST IS THE RANKING'S OWN. `_rank` lexsorts on (band_eff, viol, -success_rate)
        with NO tolerance on viol, so a child displaces the incumbent iff its band snaps to
        0 and either its viol is STRICTLY smaller, or exactly equal and its success_rate higher.
        19eb asked `co <= bo + 1e-12` against a bo of 2.38e-14 and manufactured 20 phantom
        winners, which the verdict then read as an update fault."""
        _f = cb <= _FEAS_EPS
        _v = cv > bv
        _win = _f & ((co < bo) | ((co == bo) & _v))
        _blk = _f & _v & (co > bo)
        dst["kids"] += int(cb.size)
        dst["feas"] += int(_f.sum())
        dst["vbet"] += int(_v.sum())
        dst["feas_vbet"] += int((_f & _v).sum())
        dst["blocked_other"] += int(_blk.sum())
        dst["won"] += int((_win & _v).sum())
        dst["won_eng"] += int((_win & ~_v).sum())      # displaces on viol alone, worse success rate
        if cb.size:
            dst["minband"] = min(dst["minband"], float(cb.min()))
        if _f.any():
            dst["bestfeas"] = max(dst["bestfeas"], float(cv[_f].max()))
        if keep and _blk.any():
            _bv = co[_blk]
            _cen_blocked_vals.append(float(np.median(_bv)))
            for _i, _c in enumerate(_CEN_CUTS):
                _cen_mag[_i] += int((_bv <= _c).sum())
            _cen_mag[-1] += int((_bv > _CEN_CUTS[-1]).sum())
        if keep:
            # COUNTERFACTUAL. Under `viol_eff = where(viol <= eps, 0, viol)` the incumbent's
            # own viol also snaps to 0, so a child wins exactly when it is compliant, its
            # viol is within eps, and it converts better. Counted; never applied.
            for _e in _CEN_CUTS:
                _w = _f & (co <= _e) & _v
                if _w.any():
                    _cen_eps[_e]["n"] += int(_w.sum())
                    _cen_eps[_e]["best"] = max(_cen_eps[_e]["best"], float(cv[_w].max()))
        return int(_f.sum()), float(cb.mean()) if cb.size else 0.0

    # 19ed: WHAT IS IN `other`? Split ONE candidate's engineering violation into the four
    # terms `_violation` sums, in the same order, plus the two facts in PLAIN UNITS that a
    # relative sum hides: the worst gateway's share of its profile against the max-share cap,
    # and the portfolio VAMP rate against the global cap.
    _dec = {"share": [], "glob": [], "midcap": [], "band": [], "resid": [],
            "worst": [], "rows_over": [], "grate": [], "n": 0}

    # [FN-ga-census-decomp]
    def _viol_parts(lg1):
        """(parts, extras) for one logit row, or None if anything is unavailable."""
        _fn = _segment_softmax_fast if _SM_OK["use"] else _segment_softmax_serial
        sh = np.asarray(_fn(np.asarray(lg1, float)[None, :],
                            p.profile_start, p.profile_len)[0], float)
        w = sh * p.vol
        _nm = int(p.n_mids)
        num = np.bincount(p.mid_id, weights=w * p.risk, minlength=_nm)
        den = np.bincount(p.mid_id, weights=w, minlength=_nm)
        rate = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
        _hc, _sc = np.asarray(p.mid_hard_cap, float), np.asarray(p.mid_soft_cap, float)
        _oh = np.maximum(0.0, np.divide(rate, _hc, out=np.zeros_like(rate),
                                        where=_hc > 0) - 1.0)
        _osf = np.maximum(0.0, np.divide(rate, _sc, out=np.zeros_like(rate),
                                         where=_sc > 0) - 1.0)
        _midcap = float(np.minimum(_oh, _osf).sum())
        _ms = np.asarray(p.max_share, float)
        _share = float(np.maximum(0.0, sh / _ms - 1.0).sum())
        # the kernel's OWN band term (structurally 0 here: tab2 passes no mid_bands). Computed,
        # not assumed — if it is ever non-zero this block's premise is wrong and must say so.
        _band = 0.0
        _bm = np.asarray(getattr(p, "mid_band_metric", np.zeros(_nm)), float)
        if (_bm > 0).any():
            _hi = np.asarray(p.mid_band_hi, float)
            _lo = np.asarray(p.mid_band_lo, float)
            _val = np.where(_bm == 1, den, np.where(_bm == 2, num, rate))
            _ov = np.where((_bm > 0) & (_hi < 1e300) & (_hi > 0) & (_val > _hi),
                           _val / np.where(_hi > 0, _hi, 1.0) - 1.0, 0.0)
            _un = np.where((_bm > 0) & (_lo > -1e300) & (_lo > 0) & (_val < _lo),
                           1.0 - _val / np.where(_lo > 0, _lo, 1.0), 0.0)
            _band = float(_ov.sum() + _un.sum())
        _gvc = float(getattr(p, "global_vamp_cap", np.inf))
        _gd, _gn = float(den.sum()), float(num.sum())
        _grate = (_gn / _gd) if _gd > 0 else 0.0
        _glob = (max(0.0, _grate / _gvc - 1.0)
                 if (np.isfinite(_gvc) and _gvc > 0) else 0.0)
        return ({"share": _share, "glob": _glob, "midcap": _midcap, "band": _band},
                {"worst": float((sh / _ms).max()) if sh.size else 0.0,
                 "rows_over": int((sh > _ms).sum()),
                 "grate": _grate, "gvc": _gvc, "cap": float(_ms.max()) if _ms.size else 0.0})

    # 19ed: the INCUMBENT, taken apart once. Its 2.38e-14 is the number every child is being
    # compared against, and until this line it had no explanation at all.
    if _cen_dec_on:
        try:
            _ip, _ix = _viol_parts(best_logits)
            _isum = sum(_ip.values())
            # .6g, not .4g: these four terms are printed so a reader can ADD THEM UP, and at four
            # significant figures the printed parts do not sum to the printed total — which reads
            # as a decomposition that fails to close when in fact it closes exactly.
            log(f"[ga-census] the INCUMBENT's engineering violation {best_other:.6g} = max-share {_ip['share']:.6g} "
                f"+ global-VAMP-cap {_ip['glob']:.6g} + per-MID-ceiling {_ip['midcap']:.6g} + kernel-band {_ip['band']:.6g} "
                f"(Σparts {_isum:.6g}, residual vs the kernel {abs(_isum - float(best_other)):.3g}). "
                f"IN PLAIN UNITS: worst gateway holds {100.0 * _ix['worst'] * _ix['cap']:.4f}% of its profile against a "
                f"{100.0 * _ix['cap']:.1f}% cap, {_ix['rows_over']:,} row(s) above it; portfolio VAMP rate "
                f"{100.0 * _ix['grate']:.3f}% against a {100.0 * _ix['gvc']:.2f}% cap.")
        except Exception as _dce:                        # noqa: BLE001
            log(f"[ga-census] incumbent decomposition skipped ({type(_dce).__name__}: {_dce}).")

    history = []
    evaluated = 0                              # cumulative candidate splits scored
    _t0 = time.perf_counter()
    # [gen-gap] 19ct — see the patch note. Segments SUM to the real generation by construction.
    _gg = {"gen": [], "refresh": 0.0, "rank": 0.0, "build": 0.0, "beat": 0.0,
           "eval": 0.0, "tail": 0.0, "init": 0.0, "beats": 0,
           # 19fn: `init` was one number for the whole per-restart block, and 51.5s of a 456s
           # search with no breakdown is not something a decision can be made about. These
           # three SUM to it (i_rest is the remainder by construction, so nothing hides).
           "i_pop": 0.0, "i_rep": 0.0, "i_eval": 0.0, "i_rest": 0.0, "i_n": 0}
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
    _mw_warned = False          # one-shot flag: never spam a per-generation failure 160 times
    # Per-profile mutation probability, tunable WITHOUT a build (it never was before — there is no UI
    # input and tab2 does not pass mutation_rate). Default 0.01 = the value the old three-term
    # expression always produced, so the default run is unchanged.
    _MUT_RATE = float(_os_gf.environ.get("ROUTING_MUT_RATE", "") or 0.01)
    _eff_profiles = _MUT_RATE * int(p.n_profiles)
    log(f"[fullmatrix-ga] mutation rate {min(float(mutation_rate), _MUT_RATE):.4f} per profile over "
        f"{int(p.n_profiles):,} profiles ⇒ ~{min(float(mutation_rate), _MUT_RATE) * int(p.n_profiles):,.0f} "
        f"profile(s) perturbed per exploration child (~"
        f"{min(float(mutation_rate), _MUT_RATE) * int(p.n_profiles) * 0.25:,.0f} per refine child, "
        f"which uses a quarter rate). Ceiling mutation_rate={float(mutation_rate):g} "
        f"{'BINDS' if float(mutation_rate) < _MUT_RATE else 'does not bind'}. "
        "ROUTING_MUT_RATE overrides.")
    # Say it when the 2026-08-19ac removal actually changes this run. The deleted term was
    # max(0.01, 60/n_profiles), which bound only below 6,000 profiles — so at the live grain nothing
    # moved, but at a coarser grain it did, and a silent halving of the mutation is exactly the
    # kind of thing that gets mistaken for the engine getting worse.
    _old_rate = min(float(mutation_rate), max(0.01, 60.0 / max(int(p.n_profiles), 1)))
    if abs(_old_rate - min(float(mutation_rate), _MUT_RATE)) > 1e-12:
        log(f"[fullmatrix-ga] ⚠ MUTATION RATE CHANGED BY BUILD 2026-08-19ac AT THIS GRAIN: the "
            f"deleted `max(0.01, 60/n_profiles)` term would have given {_old_rate:.5f} "
            f"(~{_old_rate * int(p.n_profiles):,.0f} profiles) on {int(p.n_profiles):,} profiles, vs "
            f"{min(float(mutation_rate), _MUT_RATE):.5f} "
            f"(~{min(float(mutation_rate), _MUT_RATE) * int(p.n_profiles):,.0f} profiles) now. That term "
            "bound only below 6,000 profiles; the live rpgt×currency×bank grain (23,791) is "
            "unaffected, but this run is coarser. Set ROUTING_MUT_RATE="
            f"{_old_rate:.5f} to reproduce the pre-19ac search exactly.")
    if mut_weight_fn is not None:
        log("[fullmatrix-ga] mutation is BREACH-TARGETED: profiles feeding a still-breached band get "
            "a boosted selection probability, so the fixed budget lands on the MIDs that are "
            "actually short instead of being spread over every profile. See [mut-target] for the "
            "boost, the profile counts and which MIDs are aimed at.")
    else:
        log("[fullmatrix-ga] mutation is UNIFORM over profiles (no mut_weight_fn) — every profile is "
            "equally likely to be perturbed, including the ones feeding already-compliant MIDs.")
    log("[fullmatrix-ga] mutation draws: "
        + ("SPARSE + PER-CHILD STREAMS (19bp) — Gaussians are drawn only for the rows actually "
           "perturbed (was one per row, ~99% discarded at a 1% profile rate: 8.6M draws per "
           "generation), and each child has its own deterministic stream keyed on "
           "(seed, seed-index, restart, generation, child). THIS IS A DIFFERENT RANDOM SAMPLE "
           "than any run before 19bp, so success rate and the breach will differ — that is the change, "
           "not a fault. ROUTING_MUT_FAST=0 restores the old stream exactly."
           if _MUT_FAST else
           "LEGACY (ROUTING_MUT_FAST=0) — one shared generator, one Gaussian per row per child, "
           "~99% of them discarded. This reproduces every run before 19bp bit for bit."))
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
            _gg_i0 = time.perf_counter()
            pop = _init_pop(best_logits, _pn, _rng)
            _gg_i1 = time.perf_counter()
            # 19gv: `_repair_maxshare(pop)` stood here. Deleted — [decode-cap] means the
            # population cannot decode above the cap, so there is nothing to legalise. The
            # timing slot is kept at 0 so [gen-gap]'s init split still sums.
            _gg_i2 = time.perf_counter()
            success_rate, other, band = _eval_with_bands(pop)
            _gg_i3 = time.perf_counter()
            # 19fn: the three parts of `init`, timed separately. `i_rest` is picked up at the
            # end of the block as (total - these three), so it cannot silently absorb anything.
            _gg["i_pop"] += _gg_i1 - _gg_i0
            _gg["i_rep"] += _gg_i2 - _gg_i1
            _gg["i_eval"] += _gg_i3 - _gg_i2
            _gg["i_n"] += 1
            # [ga-census] the STARTING population. pp[0] is the incumbent itself; the rest are
            # 1/4 incumbent+N(0,0.3) full-width and 3/4 unanchored N(0,1.5). If almost none of
            # them is compliant, the restart begins with nothing to select from but the incumbent.
            _cen_r = dict(_cen0)
            _cen_init = (0, 0.0)
            if _cen_on:
                _cen_init = _cen_add(dict(_cen0), np.asarray(band, float),
                                     np.asarray(success_rate, float), np.asarray(other, float),
                                     float(best_success_rate), float(best_other))
            _cen_g0 = None                          # (compliant, mean breach) of generation 0
            _cen_gl = None                          # ... and of the last generation run
            _gg_i9 = time.perf_counter()
            _gg["init"] += _gg_i9 - _gg_i0
            _gg["i_rest"] += (_gg_i9 - _gg_i0) - ((_gg_i1 - _gg_i0) + (_gg_i2 - _gg_i1)
                                                  + (_gg_i3 - _gg_i2))
            evaluated += pop.shape[0]
            stale = 0
            for gen in range(generations):
                _gg_g0 = time.perf_counter()
                # Periodic codebook refit: re-learn the ≈pool-target centroid shapes from the
                # current global best (its delivered shape), then RE-SCORE the live pop + best
                # under the new codebook so the moving objective stays self-consistent. Only the
                # compress term of success rate depends on the codebook — the band violation does NOT — so
                # `_rescore_compress` keeps the existing `viol` and skips the whole-population band
                # projection (the expensive part). Bit-identical to a full re-eval.
                if (_compress_on and int(compress_refresh) > 0 and gen > 0
                        and gen % int(compress_refresh) == 0):
                    _refresh_codebook(best_logits)
                    success_rate, other, band = _rescore_compress(pop, other, band)
                    best_success_rate = float(_rescore_compress(
                        best_logits[None, :], np.asarray([best_other]), np.asarray([best_band]))[0][0])
                    best_key = _key_of(best_success_rate, best_other, best_band)
                _gg_t1 = time.perf_counter()
                order = _rank(success_rate, other, band)
                _gg_t2 = time.perf_counter()
                top = order[0]
                top_key = _key_of(success_rate[top], other[top], band[top])
                if top_key > best_key:
                    best_key = top_key
                    best_logits = pop[top].copy()
                    best_success_rate, best_other, best_band = success_rate[top], other[top], band[top]
                    stale = 0
                else:
                    stale += 1
                # History x-axis = cumulative generation index across seeds/restarts;
                # `cands` = cumulative candidates (matches the tab-3 chart layout). The `viol` slot
                # carries band breach + engineering viol so the chart still reflects total infeasibility.
                history.append((len(history), float(best_success_rate), float(success_rate[top]),
                                float(success_rate.mean()), None, float(best_band + best_other), None,
                                int(evaluated)))
                # LIVE PROGRESS: throttled per-generation heartbeat so a long BIN-grain search isn't
                # silent for ~an hour (the tilt engine streams via its poller; this engine didn't).
                _now = time.perf_counter()
                _gg_b0 = _now
                if _now - _last_prog >= _PROG_EVERY_S:
                    _gg["beats"] += 1
                    _last_prog = _now
                    _rate = evaluated / max(_now - _t0, 1e-9)
                    # Optional per-MID-constraint readout at the current best (from the caller's exact
                    # band report): distinct MIDs whose band is unmet + total MID-constraint penalty.
                    _mid_extra = ""
                    if band_report_fn is not None:
                        try:
                            _bsh = _segment_softmax(best_logits[None, :], p.profile_start,
                                                    p.profile_len, p.max_share)
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
                    # NB: best_key is the lexicographic tuple (feasible?, success rate-or-−viol) from _key_of —
                    # it must NOT be format-specced ('{:,.0f}' on a tuple raises). success rate + viol below
                    # already convey the score/feasibility, so it isn't printed.
                    log(f"[fullmatrix-ga] progress: ~{evaluated:,} splits · gen {gen} "
                        f"(seed {_s + 1}/{n_seeds} restart {_r + 1}/{restarts}) · "
                        f"best success rate {best_success_rate:.5f} · viol {best_band + best_other:,.4f}"
                        f"{_mid_extra} · {'feasible' if best_band <= _FEAS_EPS else 'infeasible'} "
                        f"· {_rate:,.0f}/s")
                _gg["beat"] += time.perf_counter() - _gg_b0
                if stale >= patience:
                    log(f"[fullmatrix-ga] seed {_s + 1}/{n_seeds} restart {_r + 1}/"
                        f"{restarts}: converged at gen {gen} (no gain in {patience})")
                    # [gen-gap] the early-stop path still SPENT refresh/rank/build, so it must be
                    # recorded or the segments stop summing to Σ generations. A block that
                    # mis-sums on a path this run does not exercise (early-stop is DISABLED here)
                    # is exactly the defect that ships unnoticed and is read as fact later.
                    _gg_tb = time.perf_counter()
                    _gg["refresh"] += _gg_t1 - _gg_g0
                    _gg["rank"] += _gg_t2 - _gg_t1
                    _gg["build"] += _gg_tb - _gg_t2
                    _gg["gen"].append(_gg_tb - _gg_g0)
                    break
                # elitism + local refinement + exploration
                elites = pop[order[:_el]].copy()
                elite_success_rate = success_rate[order[:_el]].copy()
                elite_other = other[order[:_el]].copy()
                elite_band = band[order[:_el]].copy()
                children = np.empty((_pn - _el, R))
                pool = order[: max(_el, _pn // 2)]
                # EFFECTIVE per-profile mutation probability. Until 2026-08-19ac this read
                #     min(mutation_rate, max(0.01, 60.0 / max(p.n_profiles, 1)))
                # which at 23,791 profiles always reduced to exactly 0.01, with BOTH other terms
                # inert: 60/23,791 = 0.0025 sat below the 0.01 floor so the "aim for ~60 profiles"
                # intent never applied (it was written for a much smaller problem; once n_profiles
                # passed ~6,000 the floor took over and quadrupled the count to ~238), and
                # mutation_rate=0.3 sat above the floor so `min` never picked it — and tab2 never
                # passed it anyway, so the signature default was the only value that ever existed.
                # Now ONE number. UNCHANGED IN VALUE, AND THEREFORE BIT-IDENTICAL TO 19ab,
                # ONLY WHEN n_profiles >= 6,000 — my first draft of this comment claimed bit-identity
                # unconditionally and the end-to-end test caught it on a 40-profile fixture.
                # 60/n > 0.01 exactly when n < 6,000, so BELOW that the old term really did bind:
                #     n_profiles    old rate   new rate   profiles perturbed
                #         500     0.12000    0.01000      60 ->    5
                #       2,974     0.02017    0.01000      60 ->   30
                #       6,000     0.01000    0.01000      60 ->   60   (and identical above)
                #      23,791     0.01000    0.01000     238 ->  238
                # The LIVE grain (rpgt x currency x bank) is 23,791 profiles, so the shipped search is
                # unchanged. But the coarser "Bank x Currency" grain is roughly 23,791/8 RPGTs
                # ~= 2,974 profiles, where this HALVES the mutation. The banner below says so on any
                # run where it bites, rather than leaving it to be discovered.
                # `mutation_rate` is kept as a real CEILING so the signature stops being a lie.
                _base_rate = min(float(mutation_rate), _MUT_RATE)
                # BREACH-TARGETED MUTATION (2026-08-19ab). `mut_weight_fn()` returns an
                # (n_profiles,) probability multiplier reflecting which bands are STILL breached, so
                # the fixed mutation budget concentrates on profiles that feed them. Called ONCE per
                # generation (it only reads per-spec penalties the band hook already computed —
                # no extra projection). None, or any failure, means uniform mutation: the
                # pre-19ab behaviour, bit-identical including the RNG stream.
                _cw = None
                if mut_weight_fn is not None:
                    try:
                        _cw = mut_weight_fn()
                        if _cw is not None:
                            _cw = np.asarray(_cw, float)
                            if _cw.shape != (p.n_profiles,):
                                # Wrong shape would silently broadcast or throw deep inside
                                # _mutate; refuse it here and say so once.
                                if not _mw_warned:
                                    log(f"[fullmatrix-ga] mut_weight_fn returned shape "
                                        f"{_cw.shape}, expected ({p.n_profiles},) — ignoring it and "
                                        "mutating UNIFORMLY for the rest of the run.")
                                    _mw_warned = True
                                _cw = None
                    except Exception as _mwe:                    # noqa: BLE001
                        if not _mw_warned:
                            log(f"[fullmatrix-ga] mut_weight_fn raised "
                                f"({type(_mwe).__name__}: {_mwe}) — mutating UNIFORMLY for the "
                                "rest of the run. Targeting is an optimisation, not a "
                                "correctness requirement, so the search continues; but it IS "
                                "now doing the thing 19ab was meant to stop.")
                            _mw_warned = True
                        _cw = None
                _n_refine = children.shape[0] // 2
                # 19bp: ONE STREAM PER CHILD when the fast path is on, so a child's numbers do not
                # depend on how many draws its siblings took. With ROUTING_MUT_FAST=0 every child
                # shares `_rng` exactly as before, so that switch is a true revert.
                _kid = (_child_streams(seed, _s, _r, gen, children.shape[0])
                        if _MUT_FAST else None)
                _mut = _mutate_fast if _MUT_FAST else _mutate
                # 19bx: FUSE crossover and mutate into one pass per child. Only valid against the
                # 19bp fast mutation — with ROUTING_MUT_FAST=0 the shipped operator is the legacy
                # full-width one and the fused twin is not its equivalent, so the fusion turns
                # itself off rather than quietly changing what that revert reverts to.
                _fuse = bool(_FX_OK["use"] and _MUT_FAST)
                for c in range(children.shape[0]):
                    _crng = _kid[c] if _kid is not None else _rng
                    if c < _n_refine:
                        base = pop[order[_crng.integers(0, max(1, _el))]]
                        if _fuse:
                            child = (_fx_selfcheck(base, None, _base_rate * 0.25,
                                                   mutation_strength * 0.6, p.profile_start,
                                                   p.profile_len, _crng, _cw, True)
                                     if not _FX_OK["checked"] else
                                     _mutate_fused(base, _base_rate * 0.25,
                                                   mutation_strength * 0.6, p.profile_start,
                                                   p.profile_len, _crng, profile_w=_cw))
                            _fuse = bool(_FX_OK["use"])
                        else:
                            child = _mut(base, _base_rate * 0.25, mutation_strength * 0.6,
                                         p.profile_start, p.profile_len, _crng, profile_w=_cw)
                    else:
                        pa = pop[_crng.choice(pool)]
                        pb = pop[_crng.choice(pool)]
                        if _fuse:
                            child = (_fx_selfcheck(pa, pb, _base_rate, mutation_strength,
                                                   p.profile_start, p.profile_len, _crng, _cw, False)
                                     if not _FX_OK["checked"] else
                                     _child_fused(pa, pb, _base_rate, mutation_strength,
                                                  p.profile_start, p.profile_len, _crng, profile_w=_cw))
                            _fuse = bool(_FX_OK["use"])
                        else:
                            child = _crossover(pa, pb, p.profile_start, p.profile_len, _crng)
                            child = _mut(child, _base_rate, mutation_strength,
                                         p.profile_start, p.profile_len, _crng, profile_w=_cw)
                    children[c] = child
                # 19bx: the two self-check verdicts belong in the RUN LOG, not the terminal. Each
                # says whether the fused path is bit-identical on THIS run's population; a verdict
                # only Ben's terminal saw is a verdict he cannot check later.
                for _fxk in (_FX_OK, _SM_OK):
                    if _fxk.get("msg") and not _fxk.get("said"):
                        _fxk["said"] = True
                        log("   " + _fxk["msg"])
                # 19gv: `_repair_maxshare(children)` stood here, and it was 91.3s of the
                # 2026-09-01 22:09 search. Deleted — the decode caps, so what is scored is what
                # the genome decodes to and what ships, with no second pass to make it true.
                _gg_t3 = time.perf_counter()
                child_success_rate, child_other, child_band = _eval_with_bands(children)
                # [ga-census] against the incumbent AS OF THIS GENERATION (best_* was updated at the
                # top of the loop). Comparison on arrays that already exist; the run total is the copy
                # that keeps the magnitudes, so nothing is double-counted.
                if _cen_on:
                    _cbA = np.asarray(child_band, float)
                    _cvA = np.asarray(child_success_rate, float)
                    _coA = np.asarray(child_other, float)
                    _cg = _cen_add(_cen_r, _cbA, _cvA, _coA, float(best_success_rate), float(best_other))
                    _cen_add(_cen, _cbA, _cvA, _coA, float(best_success_rate), float(best_other), keep=True)
                    if _cen_g0 is None:
                        _cen_g0 = _cg
                    _cen_gl = _cg
                    # 19ed: ONE candidate per generation — the best-CONVERTING child that was rejected
                    # on `other`. That is the child the whole question is about: it is legal, it earns
                    # more, and something in key 2 threw it away. Taking the WORST offender apart would
                    # answer a different question.
                    if _cen_dec_on:
                        _blkI = np.where((_cbA <= _FEAS_EPS) & (_cvA > float(best_success_rate))
                                         & (_coA > float(best_other)))[0]
                        if _blkI.size:
                            _pick = int(_blkI[int(np.argmax(_cvA[_blkI]))])
                            try:
                                _pp, _px = _viol_parts(children[_pick])
                                _dec["n"] += 1
                                for _k in ("share", "glob", "midcap", "band"):
                                    _dec[_k].append(_pp[_k])
                                _dec["resid"].append(abs(sum(_pp.values()) - float(_coA[_pick])))
                                for _k in ("worst", "rows_over", "grate"):
                                    _dec[_k].append(_px[_k])
                                _dec["cap"] = _px["cap"]
                                _dec["gvc"] = _px["gvc"]
                            except Exception:                        # noqa: BLE001
                                pass
                _gg_t4 = time.perf_counter()
                evaluated += children.shape[0]
                pop = np.vstack([elites, children])
                success_rate = np.concatenate([elite_success_rate, child_success_rate])
                other = np.concatenate([elite_other, child_other])
                band = np.concatenate([elite_band, child_band])
                _gg_t5 = time.perf_counter()
                _gg["refresh"] += _gg_t1 - _gg_g0
                _gg["rank"] += _gg_t2 - _gg_t1
                _gg["build"] += _gg_t3 - _gg_t2
                _gg["eval"] += _gg_t4 - _gg_t3
                _gg["tail"] += _gg_t5 - _gg_t4
                _gg["gen"].append(_gg_t5 - _gg_g0)
            # [ga-census] ONE line per seed/restart. The two ends of the run matter as much as the
            # totals: if the compliant count in the last generation is no better than in the first,
            # the population never converged toward the incumbent at all.
            if _cen_on and _cen_r["kids"]:
                _cb_min = _cen_r["minband"]
                _cb_bf = _cen_r["bestfeas"]
                log(f"[ga-census] seed {_s + 1}/{n_seeds} restart {_r + 1}/{restarts}: "
                    f"start pop {_pn} \u2192 {_cen_init[0]} compliant (incl. the incumbent). "
                    f"{_cen_r['kids']:,} children \u2192 {_cen_r['feas']:,} compliant "
                    f"({100.0 * _cen_r['feas'] / max(_cen_r['kids'], 1):.1f}%), "
                    f"{_cen_r['vbet']:,} beat the incumbent on success rate, "
                    f"{_cen_r['feas_vbet']:,} did BOTH, {_cen_r['won']:,} would have won. "
                    + (f"gen 0 compliant {_cen_g0[0]} (mean breach {_cen_g0[1]:.3g}) \u2192 "
                       f"gen {gen} compliant {_cen_gl[0]} (mean breach {_cen_gl[1]:.3g}). "
                       if (_cen_g0 and _cen_gl) else "")
                    + f"smallest breach seen {_cb_min:.3g}; "
                    + (f"best compliant child success rate {_cb_bf:.6f} vs incumbent {best_success_rate:.6f} "
                       f"(\u0394 {_cb_bf - best_success_rate:+.3g})."
                       if _cb_bf > float('-inf') else "NO child was ever compliant."))

    # ── [decode-cap] 19gu: the cap, now that it is part of the decode ────────────────────
    if _DECODE_CAP:
        log("")
        log("[decode-cap] the max-share cap is now a PROPERTY OF THE DECODE, not a repair after "
            "it. Every path that turns the logit genome into shares — the numpy reference, the "
            "fused numpy path and the numba eval kernel — water-fills each profile as it decodes, "
            "so an over-cap split is not something a candidate can express. The search does not "
            "reject or correct them; they do not exist.")
        # 19hs: three paragraphs deleted here - "WHAT THIS REPLACED" (a description of
        # [ms-repair]: it decoded the whole population, water-filled and re-encoded through log(),
        # cost 91.3s and three decodes per generation), "NOT BIT-IDENTICAL to the pre-19gu search",
        # and "19gv DELETED `_repair_maxshare` and its ~380 lines". All three describe code that
        # does not exist, printed on every run. There is no decode-then-repair path to revert to
        # and no switch pretending there is, so there is nothing to compare against either.
        log("[decode-cap]    THE RULE is delivery's: the excess goes to "
            "each sibling in proportion to (target - share), the room it has left before IT would "
            "hit the cap (impact_calcs._cap_rows). Single pass, because Σ(target - share) over a "
            "profile's present rows is (present_rows × target) - 1 + excess.")
        log("[decode-cap]    THE ENGINEERING KEY should now read 0.0000 for every candidate, "
            "because none can violate. If `viol` above is ever non-zero on the max-share term, "
            "the water-fill did not hold and that key is what says so.")

    # ── [ga-census] RUN VERDICT ──────────────────────────────────────────────────────────
    # Four candidate explanations for a flat success rate, told apart by counts; 19ed then asks the
    # question the counts cannot answer — whether the blocking violations are REAL.
    if _cen_on and _cen["kids"]:
        _K = _cen["kids"]
        log("")
        log(f"[ga-census] {_K:,} children over the whole budget. Incumbent: success rate "
            f"{best_success_rate:.6f}, M5 band breach {best_band:.3g}, engineering violation "
            f"{best_other:.3g}.")
        log(f"[ga-census]    compliant (band \u2264 {_FEAS_EPS:g}): {_cen['feas']:,} "
            f"({100.0 * _cen['feas'] / _K:.2f}%) \u00b7 beat the incumbent on success rate: "
            f"{_cen['vbet']:,} ({100.0 * _cen['vbet'] / _K:.2f}%) \u00b7 BOTH: "
            f"{_cen['feas_vbet']:,} \u00b7 rejected by the engineering tie-break: "
            f"{_cen['blocked_other']:,} \u00b7 displaced the incumbent: {_cen['won']:,} "
            f"(+{_cen['won_eng']:,} on the engineering term alone, converting WORSE).")
        # ── 19ed A. HOW BIG were the blocking violations? ────────────────────────────────
        # 19ee FIX: the first five buckets are CUMULATIVE, so summing all six double-counts.
        # The 2026-08-29 19:03 run printed "over the 4,448 rejected child(ren)" against a true
        # 4,354, and every percentage on that line was against the wrong denominator. The total
        # is the LAST CUMULATIVE bucket plus the open-ended one \u2014 and it must equal the count
        # reported one line above, which is where the number should have come from all along.
        _nb = _cen_mag[-2] + _cen_mag[-1]
        if _nb:
            _lbl = ["\u2264 1e-12", "\u2264 1e-9", "\u2264 1e-6", "\u2264 1e-3",
                    "\u2264 1e-1", "> 1e-1"]
            log("[ga-census]    BLOCKING VIOLATION SIZE over the "
                f"{_nb:,} rejected child(ren) (the first five are CUMULATIVE, so they overlap; "
                "the last two add to the total): "
                + " \u00b7 ".join(f"{_l} {_n:,} ({100.0 * _n / _nb:.1f}%)"
                                   for _l, _n in zip(_lbl, _cen_mag)))
            if _nb != _cen["blocked_other"]:
                log(f"[ga-census]    \u26a0 the buckets total {_nb:,} but "
                    f"{_cen['blocked_other']:,} children were counted as rejected on the "
                    "engineering term. These are the same population counted twice, so a "
                    "difference means one of the two tallies is wrong and neither should be "
                    "read until it is explained.")
            if _cen_blocked_vals:
                log(f"[ga-census]    median blocking violation across generations "
                    f"{float(np.median(_cen_blocked_vals)):.4g} "
                    f"(incumbent's own {best_other:.4g}) \u2014 a rejection is DUST when these "
                    "two are the same order, and a REAL breach when they are not.")
        # ── 19ed B. WHAT WOULD A TOLERANCE BUY? ──────────────────────────────────────────
        log("[ga-census]    COUNTERFACTUAL \u2014 if `viol` snapped to 0 below eps (the rule "
            "the BAND key already uses), children that would have displaced the incumbent:")
        for _e in _CEN_CUTS:
            _r = _cen_eps[_e]
            log(f"[ga-census]       eps {_e:>7.0e}: {_r['n']:,} child(ren)"
                + (f" \u00b7 best success rate {_r['best']:.6f} vs {best_success_rate:.6f} "
                   f"(\u0394 {_r['best'] - best_success_rate:+.3g})" if _r["n"] else " \u2014 nothing"))
        log("[ga-census]       (counted, NOT applied \u2014 this run ranked exactly as before.)")
        # ── 19ed C. WHAT IS THE VIOLATION MADE OF? ───────────────────────────────────────
        if _dec["n"]:
            _md = {_k: float(np.median(_dec[_k])) for _k in
                   ("share", "glob", "midcap", "band", "resid", "worst", "grate")}
            _ro = float(np.median(_dec["rows_over"]))
            _cap = float(_dec.get("cap", 0.0))
            _gvc = float(_dec.get("gvc", float("inf")))
            log(f"[ga-census]    DECOMPOSITION of the best-converting REJECTED child, sampled "
                f"once per generation ({_dec['n']:,} sample(s), medians): max-share "
                f"{_md['share']:.4g} \u00b7 global-VAMP-cap {_md['glob']:.4g} \u00b7 "
                f"per-MID-ceiling {_md['midcap']:.4g} \u00b7 kernel-band {_md['band']:.4g}.")
            log(f"[ga-census]       IN PLAIN UNITS: the worst gateway holds "
                f"{100.0 * _md['worst'] * _cap:.4f}% of its profile against a {100.0 * _cap:.1f}% "
                f"cap, with {_ro:,.0f} row(s) above the cap; portfolio VAMP rate "
                f"{100.0 * _md['grate']:.3f}% against a {100.0 * _gvc:.2f}% cap.")
            _wr = float(np.max(_dec["resid"])) if _dec["resid"] else 0.0
            log(f"[ga-census]       parts-vs-whole: median residual {_md['resid']:.3g}, worst "
                f"{_wr:.3g}. This numpy decomposition must reconstruct the numba kernel's own "
                "number; a residual comparable to the value being decomposed means the split "
                "above is NOT evidence and must not be read as one.")
            if _md["midcap"] > 0 or _md["band"] > 0:
                log("[ga-census]       \u26a0 a term expected to be STRUCTURALLY ZERO is not: "
                    "per-MID rate ceilings should be inf (the app disables them) and the "
                    "kernel band term should be off (tab2 passes no mid_bands). One of those "
                    "assumptions is wrong, and it changes what `other` means.")
        # ── the verdict ──────────────────────────────────────────────────────────────────
        if _cen["won"] > 0 or _cen["won_eng"] > 0:
            log(f"[ga-census]    VERDICT: {_cen['won'] + _cen['won_eng']:,} child(ren) "
                "outranked the incumbent under the ranking's OWN comparison and it did not "
                "move. That is an UPDATE fault, and unlike 19eb's version of this line the "
                "test behind it carries no tolerance of its own.")
        elif _cen["feas"] == 0:
            log(f"[ga-census]    VERDICT: NOT ONE of {_K:,} children was compliant "
                f"(smallest breach {_cen['minband']:.3g}). The M5 breach is the STRICT "
                f"primary key, so the ranking never reaches success rate: {_cen['vbet']:,} child(ren) "
                "DID convert better and every one was rejected for breaching.")
        elif _cen["blocked_other"] > 0:
            log(f"[ga-census]    VERDICT: {_cen['blocked_other']:,} compliant, "
                "higher-converting child(ren) were rejected on the ENGINEERING violation, "
                "which sits ABOVE success rate in a STRICT lexicographic ranking with no tolerance "
                "of its own. Read the three blocks above in order: if the sizes are dust the "
                "ranking is discarding conversion for nothing; if they are real breaches the "
                "rejections are correct in KIND and the question becomes whether a breach of "
                "ANY size should outrank a conversion gain of ANY size.")
        else:
            log(f"[ga-census]    VERDICT: {_cen['feas']:,} compliant child(ren) were produced "
                f"and the best scored {_cen['bestfeas']:.6f} against the incumbent's "
                f"{best_success_rate:.6f}. Compliant splits ARE reachable; none converts better. A "
                "genuine local optimum at this mutation scale.")
        log("[ga-census]    Read-only measurement; nothing above changed what the search did. "
            "ROUTING_GA_CENSUS=0 removes it, ROUTING_GA_CENSUS_DECOMP=0 just the decomposition.")

    best_shares_sorted = _segment_softmax(best_logits[None, :], p.profile_start, p.profile_len,
                                          p.max_share)[0]
    # restore original row order
    inv = np.empty_like(p.order)
    inv[p.order] = np.arange(len(p.order))
    best_shares = best_shares_sorted[inv]

    info = {
        "__build__": __build__,
        "success_rate": float(best_success_rate),
        "violation": float(best_band + best_other),
        "band_breach": float(best_band),
        "other_violation": float(best_other),
        "feasible": bool(best_band <= _FEAS_EPS),
        "seed_success_rate": float(seed_success_rate),
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
    if os.environ.get("ROUTING_GEN_GAP", "1") != "0" and _gg["gen"]:
        _gv = np.sort(np.asarray(_gg["gen"], float)) * 1000.0
        _raw = np.asarray(_gg["gen"], float) * 1000.0
        _tot = float(_raw.sum()) / 1000.0
        _n = len(_raw)
        log("   == [gen-gap] THE REAL GENERATION, TIMED IN SITU — the segments below SUM to it, "
            "so there is no residual to argue about (read-only; [gen-cost] times COPIES of five "
            "stages, warm and repeated, and its 'whole generation' is built from them) ==")
        log(f"      median {float(np.median(_gv)):,.1f} ms · p05 {float(_gv[int(0.05 * (_n - 1))]):,.1f}"
            f" · p95 {float(_gv[int(0.95 * (_n - 1))]):,.1f} · over {_n:,} generation(s)")
        _f10 = float(np.median(_raw[:min(10, _n)]))
        _l10 = float(np.median(_raw[-min(10, _n):]))
        log(f"      first-10 median {_f10:,.1f} ms · last-10 median {_l10:,.1f} ms "
            f"({(_l10 / _f10 - 1.0) * 100.0:+.1f}%) — a LEVEL difference between two runs is a "
            "different fault from a DRIFT within one, and averaging the run cannot tell them apart")
        for _k, _lbl in (("eval", "eval      _eval_with_bands(children) — the five stages "
                                  "[gen-cost] models"),
                         ("build", "build     history, heartbeat, elites, mut_weight_fn, "
                                   "the per-child loop"),
                         ("rank", "rank      _rank"),
                         ("refresh", "refresh   codebook refit + re-score (0 unless compression "
                                     "is on)"),
                         ("tail", "tail      vstack / concatenate re-assembly")):
            _v = float(_gg[_k])
            log(f"      {_lbl.split(chr(32))[0]:<9} {1000.0 * _v / _n:>8,.1f} ms/gen "
                f"({100.0 * _v / max(_tot, 1e-9):>5.1f}% · {_v:,.1f}s total)  "
                f"{_lbl.split(chr(32), 1)[1].strip()}")
        log(f"      of `build`, the throttled HEARTBEAT was {1000.0 * _gg['beat'] / _n:,.1f} ms/gen "
            f"({_gg['beat']:,.1f}s over {_gg['beats']:,} call(s), {1000.0 * _gg['beat'] / max(_gg['beats'], 1):,.0f} "
            "ms each) — it runs a FULL _deliver_full plus a band report, and the generations that "
            "do NOT fire it pay none of it")
        log(f"      OUTSIDE the generation loop entirely: the per-restart start-up cost "
            f"{_gg['init']:,.1f}s over the run — charged to no stage anywhere else, and it scales "
            f"with RESTARTS, not generations ({_gg['i_n']:,} restart(s), "
            f"{_gg['init'] / max(_gg['i_n'], 1):,.1f}s each)")
        # 19fn: WHICH PART of the start-up. Before this the whole block was one number, so the
        # only available lever was "fewer restarts" — a budget decision, not a fix. The four
        # lines below SUM to the number above by construction: `remainder` is computed as the
        # total minus the other three, so a part nobody thought to time still shows up.
        for _il, _iv in (("_init_pop (build the population)", _gg["i_pop"]),
                         ("_eval_with_bands (score it)     ", _gg["i_eval"]),
                         ("remainder (census, bookkeeping) ", _gg["i_rest"])):
            log(f"         {_il}  {_iv:,.1f}s  "
                f"({100.0 * _iv / max(_gg['init'], 1e-9):>5.1f}%) · "
                f"{1000.0 * _iv / max(_gg['i_n'], 1):,.0f} ms per restart")
        # ── [eval-cost] 19gw: the `eval` row, split ─────────────────────────────────────
        # `eval` is 92% of a generation and was the only row in this block with nothing behind
        # it. These five accumulators are taken inside `_eval_with_bands` around its own calls,
        # so they sum to it — the `unaccounted` row is computed as the difference and is the
        # function's own entry/exit, not a bucket anything can hide in.
        _ev_tot = sum(float(_ev[_x]) for _x in ("pop", "decode", "deliver", "band", "comp"))
        if _ev["n"]:
            log(f"      == [eval-cost] inside the `eval` row ({_ev_tot:,.1f}s over "
                f"{_ev['n']:,} call(s)), largest first ==")
            _ev_rows = [
                ("eval_pop (fused numba: softmax + success rate + engineering viol)", _ev["pop"]),
                ("_deliver_full (blocked-caps -> eligibility -> cap, per candidate)",
                 _ev["deliver"]),
                ("band projection + penalty (the EXACT M5 projector)", _ev["band"]),
                ("_segment_softmax decode (numpy, feeds deliver + band)", _ev["decode"]),
                ("compressibility regulariser (0 unless it is on)", _ev["comp"]),
            ]
            for _el, _evv in sorted(_ev_rows, key=lambda kv: -kv[1]):
                if _evv > 0.0 or "compress" in _el:
                    log(f"         {_evv:8.1f}s  "
                        f"({100.0 * _evv / max(_ev_tot, 1e-9):>5.1f}%)  {_el}"
                        f"  ·  {1000.0 * _evv / max(_ev['n'], 1):,.1f} ms/call")
            # 19gx: the `eval` row counts the GENERATION calls only; `_ev` counts every call,
            # which includes the one per restart that [gen-gap] charges to `init`. On the
            # 2026-09-02 00:22 run that was 336 calls against 320 generations and the difference
            # printed as a NEGATIVE unaccounted row. Compare against both, and say which.
            _ev_ref = float(_gg["eval"]) + float(_gg["i_eval"])
            _ev_gap = _ev_ref - _ev_tot
            log(f"         {_ev_gap:8.1f}s  "
                f"({100.0 * _ev_gap / max(_ev_ref, 1e-9):>5.1f}%)  unaccounted — function "
                "entry/exit and the early return when no band scoring is wired")
            log(f"      [eval-cost] the {_ev['n']:,} call(s) above are the "
                f"{int(_gg['i_n']) + 320 if False else _ev['n']:,} that ran: one per generation "
                f"PLUS one per restart. [gen-gap] charges the per-restart calls to `init`, so "
                f"these sum to `eval` ({float(_gg['eval']):,.1f}s) + the init eval row "
                f"({float(_gg['i_eval']):,.1f}s) = {_ev_ref:,.1f}s, not to `eval` alone.")
            log("      [eval-cost] READ THIS AGAINST [proj-config] AND [lift-ab]: the band row "
                "here is the projector, and multiplying [lift-ab]'s ms/call by the P=35 call "
                "count should land on it. `eval_pop` and `_deliver_full` are the two that have "
                "never been separated from it before, and between them they are most of the "
                "search.")
        # SELF-CHECK. "The segments SUM to it" is the block's whole claim over [gen-cost]; it is
        # true by the interval algebra only while every EXIT from the loop records what it spent.
        # There are two (fall-through and the patience break), so the claim is checked rather than
        # asserted — an unchecked invariant is how the 'several times smaller' line survived.
        _ssum = sum(float(_gg[_x]) for _x in ("refresh", "rank", "build", "eval", "tail"))
        if abs(_ssum - _tot) > max(0.005 * max(_tot, 1e-9), 1e-6):
            log(f"      \u26a0 THE SEGMENTS DO NOT SUM TO THE GENERATIONS: "
                f"\u03a3 segments {_ssum:,.2f}s vs \u03a3 generations {_tot:,.2f}s "
                f"(gap {_ssum - _tot:+,.2f}s). Some exit from the generation loop is spending time "
                "it does not record, so the percentages above are NOT a partition and this block "
                "is reporting a smaller run than it timed. Read the rows as a ranking only.")
        _acct = _tot + float(_gg["init"])
        _secs = float(time.perf_counter() - _t0)
        log(f"      ⇒ Σ generations {_tot:,.1f}s + init {_gg['init']:,.1f}s = {_acct:,.1f}s of the "
            f"search's {_secs:,.1f}s ({100.0 * _acct / max(_secs, 1e-9):.1f}%). The remainder is "
            "outside both loops (seed setup, the one-off self-checks, numba compiles). COMPARE THE "
            "`eval` ROW WITH [gen-cost]'s WHOLE GENERATION: [gen-cost] models only what `eval` "
            "does, so if the two agree while this block's median does not, the gap it reports was "
            "never in the five stages.")
    log(f"[fullmatrix-ga] evaluated {evaluated:,} candidate splits over "
        f"{len(history)} generations ({n_seeds} seed(s) × {restarts} restart(s), "
        f"pop {pop_size}) in {info['seconds']:.1f}s = {info['splits_per_s']:,.0f} splits/s")
    log(f"[fullmatrix-ga] done success rate {best_success_rate:.6f} M5-breach={best_band:.3e} "
        f"eng-viol={best_other:.3e} feasible={info['feasible']} improved={info['improved_over_seed']}")
    return best_shares, info


def _profile_volume_total(p: "FullMatrixProblem"):
    """Total volume across profiles (each profile's volume counted once)."""
    # vol is repeated per row; take the first row of each profile segment.
    return float(p.vol[p.profile_start].sum())


# STATUS (was the stage-2 TODO — now largely delivered):
#   * DONE  Numba-fused kernel (_fused_eval_kernel / make_fused_eval): segment-
#           softmax + VWSR + violation in one pass, verify-gated bit-close vs the
#           numpy path with automatic fallback. prange-parallel, persistent cache,
#           int32 index arrays, elite-fitness caching.
#   * DONE  Wired as the opt-in "genetic_fullmatrix" engine in tab_2_routing_engine.py
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
