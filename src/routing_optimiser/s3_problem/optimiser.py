"""
Run a chosen engine across every profile and assemble the proposed split.

This is the layer the UI calls. It:
  * loops over all ProfileProblems, solving each with the selected engine,
  * returns a tidy "long" split table (one row per profile x gateway),
  * can sweep the conversion<->risk slider to produce split *variations*
    (the family of solutions along the Pareto frontier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from routing_optimiser.s3_problem.constraints import OptimiserSettings
from routing_optimiser.engines import ProfileProblem, get_engine

__build__ = "2026-07-29-vamp-lp-singlegw-fixed-profile-revival+2026-09-02-19gz-max-revenue-split-reference+2026-09-02-19he-floor-carried-into-projector+2026-09-02-19hh-emask-pair-grain+2026-09-02-19hi-log-simplification+2026-09-02-19hk-cell-profile-prose+2026-09-02-19hl-cell-profile-identifiers+2026-09-02-19hm-profile-vocab-complete+2026-09-02-19hn-profile-vocab-audited+2026-09-02-19ho-vocab-no-exceptions+2026-09-02-19hp-log-hierarchy"


# [FN-191]
def _vamp_cap_lp(df: pd.DataFrame, cap: float, floor: float = 0.0, max_share: float = 1.0,
                 agg_cap: float = None, _reduce: bool = True):
    """Joint solve for the per-vampMid VAMP cap: the split CLOSEST to the reference
    (minimum total share movement) whose every vampMid AGGREGATE VAMP rate is <= cap,
    subject to per-profile shares summing to 1 and the exploration-floor / max-share
    bounds. Solved as one sparse LP (min L1 movement, linear rate constraints), so it
    retains more revenue than the greedy's lowest-rate dumping and resolves all profiles
    together. Returns (adjusted_df, retired, still_over) ONLY if it finds a fully
    cap-compliant solution; otherwise None so the caller falls back to the greedy shave.
    Guarded: needs SciPy/HiGHS and re-checks compliance after solving.

    `agg_cap` (optional): also constrain the WHOLE-book aggregate VAMP rate to <= agg_cap.
    This is what the true-frontier sweep uses — starting from the revenue reference and
    tightening agg_cap dial-by-dial gives the min-movement (max-revenue) split at each
    risk budget, i.e. a Pareto-optimal frontier point rather than a linear share blend.

    BINDING-PROFILE REDUCTION (`_reduce`, speed #1, EXACT). With no aggregate budget the LP
    SEPARATES by profile: the only cross-profile constraints are the per-MID rate rows, so a profile
    is coupled to the solve ONLY if it contains a row of an over-cap MID. Every other profile's
    minimum-movement optimum is exactly its own reference (already sums to 1 and within
    [floor, max_share]), so we fix those profiles to reference and build the LP over ONLY the
    binding profiles — a much smaller matrix, identical solution. A profile whose reference is NOT
    bound-feasible is kept in the LP too, so nothing that could move is dropped. `_reduce=False`
    forces the full all-profiles LP (used by the self-test to prove the reduction is identical).
    The reduction is skipped when `agg_cap` is set (that constraint couples every profile)."""
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
    except Exception:  # noqa: BLE001
        return None
    d = df.reset_index(drop=True)
    ref = d["share"].to_numpy(float)
    rate = d["rate"].to_numpy(float)
    vol = d["profile_vol"].to_numpy(float)
    n = len(d)
    if n == 0:
        return None
    profile_lbl = d["profile"].astype(str).to_numpy()
    profile_rows = _group_indices(profile_lbl)
    mid_rows = _group_indices(d["vampMid"].astype(str).to_numpy())
    # Only MIDs that CAN breach (some profile rate above the cap) need a constraint.
    over_mids = [m for m, r in mid_rows.items() if float(rate[r].max()) > cap + 1e-12]
    if not over_mids and agg_cap is None:
        return None   # reference already cap-compliant and no aggregate budget — greedy no-ops

    # SINGLE-GATEWAY PROFILES ARE STRUCTURALLY FIXED (EXACT, speed #2). A one-gateway profile has its
    # share pinned at the reference (1.0 — it's the only option), so it can NEVER redistribute.
    # It must not enter the LP: with max_share < 1 its forced share = 1.0 violates the
    # [floor, max_share] bound, making the whole LP infeasible → the solve returns None and the
    # caller silently drops to the slower greedy shave on EVERY run with max_share < 1. We fix
    # these rows at their reference and fold their CONSTANT vol·(rate−cap)·ref contribution into
    # the per-MID (and aggregate) right-hand sides, so the LP still accounts for their true,
    # unmovable risk. With no single-gateway profiles `fixed` is all-False and every expression below
    # collapses to the original (RHS constants = 0), so the build is byte-identical in that case.
    _profile_n = {c: len(idx) for c, idx in profile_rows.items()}
    fixed = np.fromiter((_profile_n[profile_lbl[i]] < 2 for i in range(n)), dtype=bool, count=n)

    # Which rows actually enter the LP. Full MOVABLE set unless the reduction applies.
    if agg_cap is None and _reduce:
        _keep_profiles = set()
        for _m in over_mids:                 # every profile holding a MOVABLE over-cap MID row is binding
            for _i in mid_rows[_m]:
                if not fixed[_i]:
                    _keep_profiles.add(profile_lbl[_i])
        for _c, _idx in profile_rows.items():    # + any MULTI-gw profile whose reference isn't already feasible
            if _c in _keep_profiles or fixed[_idx[0]]:
                continue
            _rf = ref[_idx]
            if (np.any(_rf < floor - 1e-9) or np.any(_rf > max_share + 1e-9)
                    or abs(float(_rf.sum()) - 1.0) > 1e-9):
                _keep_profiles.add(_c)
        keep = np.fromiter(((not fixed[i]) and (profile_lbl[i] in _keep_profiles)
                            for i in range(n)), dtype=bool, count=n)
    else:
        keep = ~fixed                        # agg_cap / full path: every MOVABLE row enters
    if not keep.any():
        return None
    kidx = np.nonzero(keep)[0]
    nk = int(len(kidx))
    _loc = {int(gi): li for li, gi in enumerate(kidx)}   # global row -> local LP column

    rows, cols, data, b_ub = [], [], [], []
    _r = 0
    for li in range(nk):                     # L1:  x_i - u_i <= ref_i
        gi = int(kidx[li])
        rows += [_r, _r]; cols += [li, nk + li]; data += [1.0, -1.0]; b_ub.append(float(ref[gi])); _r += 1
    for li in range(nk):                     # L1: -x_i - u_i <= -ref_i
        gi = int(kidx[li])
        rows += [_r, _r]; cols += [li, nk + li]; data += [-1.0, -1.0]; b_ub.append(-float(ref[gi])); _r += 1
    for m in over_mids:                      # Σ_kept vol·(rate-cap)·x ≤ -Σ_fixed vol·(rate-cap)·ref
        _const = 0.0
        for i in mid_rows[m]:
            _c = float(vol[i]) * (float(rate[i]) - cap)
            if fixed[i]:
                _const += _c * float(ref[i])              # unmovable single-gw row → constant term
            elif _c != 0.0:
                rows.append(_r); cols.append(_loc[int(i)]); data.append(_c)
        b_ub.append(-_const); _r += 1
    if agg_cap is not None:                  # whole-book aggregate rate <= agg_cap (frontier budget)
        for li in range(nk):
            gi = int(kidx[li])
            _c = float(vol[gi]) * (float(rate[gi]) - float(agg_cap))
            if _c != 0.0:
                rows.append(_r); cols.append(li); data.append(_c)
        _fx = np.nonzero(fixed)[0]                         # + fixed rows' constant aggregate contribution
        _const = float((vol[_fx] * (rate[_fx] - float(agg_cap)) * ref[_fx]).sum()) if _fx.size else 0.0
        b_ub.append(-_const); _r += 1
    A_ub = sp.coo_matrix((data, (rows, cols)), shape=(_r, 2 * nk)).tocsr()
    erows, ecols, edata, b_eq, _e = [], [], [], [], 0
    for _c, idx in profile_rows.items():        # each KEPT profile's shares sum to 1 (dropped profiles = ref)
        _kept = [int(i) for i in idx if keep[i]]
        if not _kept:
            continue
        for i in _kept:
            erows.append(_e); ecols.append(_loc[i]); edata.append(1.0)
        b_eq.append(1.0); _e += 1
    A_eq = sp.coo_matrix((edata, (erows, ecols)), shape=(_e, 2 * nk)).tocsr()
    c_obj = np.concatenate([np.zeros(nk), np.ones(nk)])
    bounds = [(float(floor), float(max_share))] * nk + [(0.0, None)] * nk
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=np.asarray(b_ub, float),
                      A_eq=A_eq, b_eq=np.asarray(b_eq, float), bounds=bounds, method="highs")
    except Exception:  # noqa: BLE001
        return None
    if not getattr(res, "success", False):
        return None
    x = ref.copy()                           # dropped (non-binding, feasible) profiles stay at reference
    x[kidx] = np.clip(np.asarray(res.x[:nk], dtype=float), 0.0, max_share)
    for _c, idx in profile_rows.items():        # exact renormalise (guard tiny LP residuals)
        s = float(x[idx].sum())
        if s > 1e-9:
            x[idx] = x[idx] / s
    still_over, retired = set(), set()
    for m, idx in mid_rows.items():
        v = vol[idx] * x[idx]; tot = float(v.sum())
        r_agg = float((v * rate[idx]).sum() / tot) if tot > 1e-12 else 0.0
        if r_agg > cap + 1e-9:
            still_over.add(m)
        if float((vol[idx] * ref[idx]).sum()) > 1e-9 and tot <= 1e-9:
            retired.add(m)
    if still_over:                           # renorm re-broke the cap (rare) -> fall back
        return None
    out = d.copy(); out["share"] = x
    # Reached only when still_over is empty (otherwise we fell back above) — the LP path is
    # cap-compliant by construction, so the third element is always the empty set.
    return out, retired, set()


# [FN-192]
def vamp_frontier_lp(df: pd.DataFrame, cap: float, agg_cap: float,
                     floor: float = 0.0, max_share: float = 1.0):
    """Frontier point (public wrapper for `_vamp_cap_lp` with an aggregate budget):
    the min-movement-from-reference split whose whole-book aggregate VAMP rate is
    <= agg_cap AND every per-vampMid rate is <= cap. Returns the adjusted share
    DataFrame, or None if SciPy is missing / the LP is infeasible (caller falls back
    to the linear blend). Used by the true-frontier sweep — start from the revenue
    reference and tighten agg_cap per dial to trace the Pareto frontier."""
    res = _vamp_cap_lp(df, cap, floor=floor, max_share=max_share, agg_cap=agg_cap)
    return res[0] if res is not None else None


# [FN-193]
def _group_indices(labels: np.ndarray) -> dict:
    """{label -> ascending row positions}, identical to
    ``{v: np.where(labels == v)[0] for v in unique(labels)}`` but built in ONE
    pass (pandas groupby) instead of a full-array scan per distinct label.

    The old dict-comprehension was O(n_rows · n_labels) — for ~692k rows and
    ~19k profiles that's ~1.3e10 object-string comparisons *per call*, the dominant
    cost of the VAMP-cap phase. This is O(n_rows) and returns the SAME arrays
    (sorted ascending), so every downstream move — and the result — is identical.
    """
    ser = pd.Series(labels)
    return {lbl: np.sort(np.asarray(idx, dtype=np.int64))
            for lbl, idx in ser.groupby(ser, sort=False).indices.items()}


# [FN-194]
def _profile_recip_order(profile_rows: dict, rate: np.ndarray) -> dict:
    """Per-profile row positions sorted by rate ASCENDING, ties broken by ascending row index.

    This is BIT-IDENTICAL to the inline ``sorted(gen, key=lambda j: rate[j])`` used per move,
    where ``gen`` yields ``profile_rows[c]`` (already ascending row index) and Python's sort is
    stable (equal rates keep index order). Precomputing it ONCE lets the per-move recipient scan
    become a filter over this fixed order instead of re-sorting the profile every iteration — the
    move sequence, and therefore the result, is unchanged. Rates are constant, so the order is too.
    """
    return {c: rows[np.argsort(rate[rows], kind="stable")] for c, rows in profile_rows.items()}


# [FN-195]
def enforce_mid_vamp_caps(df: pd.DataFrame, cap: float, floor: float = 0.0,
                          max_share: float = 1.0, max_iter: int = 4000,
                          step: float = 0.05):
    """Cross-profile adjustment so each vampMid's AGGREGATE VAMP rate <= cap.

    ANALOGY: a MID's monitored rate is the volume-weighted average across every profile it runs
    in — like a student's overall grade averaged across subjects, weighted by credit hours. To
    pull that average under the limit with the least disruption, we move volume off the MID's
    WORST profiles onto the cheapest alternative in each; a MID that's over the limit in EVERY
    profile can't be fixed by re-weighting, so it's retired (dropped) and its volume handed off.

    A vampMid spans many routing profiles; its Visa-monitored rate is the volume-
    weighted mean of its per-profile rates. Starting from the reference split, we
    iteratively shave share off the MID's HIGHEST-rate profiles (handing it to the
    lowest-rate other gateway in that profile) until the MID's aggregate rate is
    under the cap - which minimises movement from the reference. A MID that can't
    be brought under the cap by re-weighting (its rate exceeds the cap in every
    profile) is RETIRED (share -> 0, exempt from the floor) and its volume handed to
    compliant gateways.

    df columns: profile, gateway, vampMid, profile_vol, rate, share (reference start).
    Returns (adjusted_df, retired_set, still_over_set).

    PRIMARY path: a joint LP (`_vamp_cap_lp`) that solves all profiles together for the
    minimum-movement cap-compliant split — retains more revenue than the greedy shave
    and is order-independent. If the LP is unavailable, infeasible (e.g. the floor
    conflicts with the cap so some MID must retire) or errors, we fall back to the
    greedy shave below, which handles retirement. So the LP can only improve, never
    regress compliance.
    """
    _lp = _vamp_cap_lp(df, cap, floor=floor, max_share=max_share)
    if _lp is not None:
        return _lp
    d = df.reset_index(drop=True).copy()
    share = d["share"].to_numpy(float).copy()
    rate = d["rate"].to_numpy(float)
    profile_vol = d["profile_vol"].to_numpy(float)
    mid = d["vampMid"].astype(str).to_numpy(object)
    profile = d["profile"].astype(str).to_numpy(object)

    mids = list(pd.unique(mid))
    mid_rows = _group_indices(mid)
    profile_rows = _group_indices(profile)
    _corder = _profile_recip_order(profile_rows, rate)   # per-profile rows by ascending rate (bit-identical)
    retired: set = set()

    # [FN-196]
    def _mid_rate(m):
        rows = mid_rows[m]
        vol = profile_vol[rows] * share[rows]
        tot = vol.sum()
        return float((vol * rate[rows]).sum() / tot) if tot > 1e-12 else 0.0

    # INCREMENTAL rate maintenance. A MID's aggregate rate = num/den where
    #   den = Σ profile_vol·share ,  num = Σ profile_vol·share·rate  (over its rows).
    # Every move shifts `delta` from ONE row of MID m to ONE row of MID m(j), so we
    # update num/den for just those two MIDs in O(1) — instead of re-summing all of a
    # MID's rows (which is O(11k) for a MID spanning thousands of profiles, the cause of
    # the 5-minute VAMP phase). rate/profile_vol are constants, so the update is exact;
    # accumulated float drift is ~1e-13, far below the 1e-9 decision threshold, so the
    # sequence of moves (and the resulting split) is identical to the full-recompute.
    _num, _den = {}, {}
    for m in mids:
        _rows = mid_rows[m]
        _v = profile_vol[_rows] * share[_rows]
        _den[m] = float(_v.sum())
        _num[m] = float((_v * rate[_rows]).sum())

    # [FN-197]
    def _rt(m):
        return (_num[m] / _den[m]) if _den[m] > 1e-12 else 0.0

    rate_cache = {m: _rt(m) for m in mids}

    # Precompute each MID's rows in rate-DESCENDING order ONCE (rates are constant),
    # plus every row's position in that order. An advancing pointer `pstart` skips a
    # MID's already-shaved (exhausted) leading rows, so we never re-sort/re-scan the
    # MID's thousands of rows each iteration (the real cost). Pointers only move
    # forward; when a MID receives volume (as a low-rate recipient, i.e. near the END
    # of its order) we pull its pointer back to that position so nothing is skipped —
    # keeping the move sequence, and the result, bit-identical to the naive version.
    order = {m: mid_rows[m][np.argsort(-rate[mid_rows[m]], kind="stable")] for m in mids}
    pos = np.empty(len(share), dtype=np.int64)
    for m in mids:
        for _k, _i in enumerate(order[m]):
            pos[_i] = _k
    pstart = {m: 0 for m in mids}

    for _ in range(max_iter):
        over = [(m, rate_cache[m]) for m in mids
                if m not in retired and rate_cache[m] > cap + 1e-9]
        if not over:
            break
        m = max(over, key=lambda t: t[1])[0]
        eff_floor = 0.0 if m in retired else floor
        _ord = order[m]
        _ps = pstart[m]
        while _ps < len(_ord) and share[_ord[_ps]] <= eff_floor + 1e-9:
            _ps += 1
        pstart[m] = _ps
        moved = False
        _k = _ps
        while _k < len(_ord):
            i = _ord[_k]
            if share[i] <= eff_floor + 1e-9:
                _k += 1
                continue
            recs = [j for j in _corder[profile[i]]
                    if mid[j] != m and share[j] < max_share - 1e-9]
            if not recs:
                _k += 1
                continue
            j = recs[0]
            delta = min(share[i] - eff_floor, max_share - share[j], step)
            if delta <= 1e-12:
                _k += 1
                continue
            share[i] -= delta
            share[j] += delta
            _mj = mid[j]
            _num[m] -= profile_vol[i] * delta * rate[i]; _den[m] -= profile_vol[i] * delta
            _num[_mj] += profile_vol[j] * delta * rate[j]; _den[_mj] += profile_vol[j] * delta
            rate_cache[m] = _rt(m)
            rate_cache[_mj] = _rt(_mj)
            if pos[j] < pstart[_mj]:
                pstart[_mj] = int(pos[j])
            moved = True
            break
        if not moved:
            # Can't reduce by re-weighting -> retire the MID (dump its volume onto
            # the lowest-rate other gateways in each of its profiles).
            retired.add(m)
            _touched = set()
            for i in mid_rows[m]:
                freed = share[i]
                if freed <= 1e-12:
                    continue
                for j in (k for k in _corder[profile[i]]
                          if mid[k] != m and share[k] < max_share - 1e-9):
                    take = min(max_share - share[j], freed)
                    share[j] += take
                    share[i] -= take
                    _mj = mid[j]
                    _num[m] -= profile_vol[i] * take * rate[i]; _den[m] -= profile_vol[i] * take
                    _num[_mj] += profile_vol[j] * take * rate[j]; _den[_mj] += profile_vol[j] * take
                    if pos[j] < pstart[_mj]:
                        pstart[_mj] = int(pos[j])
                    _touched.add(_mj)
                    freed -= take
                    if freed <= 1e-12:
                        break
            rate_cache[m] = _rt(m)
            for _tm in _touched:
                rate_cache[_tm] = _rt(_tm)

    # Renormalise each profile to sum 1 (safety against rounding).
    for c, rows in profile_rows.items():
        s = share[rows].sum()
        if s > 0:
            share[rows] = share[rows] / s

    d["share"] = share
    still_over = {m for m in mids if _mid_rate(m) > cap + 1e-9}   # fresh recompute
    return d, retired, still_over


# [FN-198]
def enforce_mid_volume_caps(df: pd.DataFrame, a_max_by_mid: dict,
                            max_share: float = 1.0):
    """Scale each vampMid's allocated volume down to a_max x its BASELINE volume.

    ANALOGY: a spend cap per MID. If a MID is routed more volume than a_max × what it
    historically carried, we shrink every one of its profiles by the same factor (like trimming
    an over-budget line item proportionally) and hand the freed volume to the cheapest other
    gateway in each profile.

    `a_max_by_mid[mid]` is the maximum allowed (proposed / baseline) volume ratio
    for that vampMid, derived upstream from its per-MID monthly VAMP-count / Txn
    caps (and 0 if a rate cap it can't meet by re-weighting forces retirement).
    A MID whose current proposed volume exceeds a_max x baseline is scaled back
    uniformly across its profiles; the freed share is handed to the other gateways in
    each profile (lowest-rate first). MIDs not in the dict are untouched.

    df columns: profile, gateway, vampMid, profile_vol, baseline_share, share, rate.
    Returns (adjusted_df, constrained_set).
    """
    d = df.reset_index(drop=True).copy()
    share = d["share"].to_numpy(float).copy()
    bshare = d["baseline_share"].to_numpy(float)
    profile_vol = d["profile_vol"].to_numpy(float)
    rate = d["rate"].to_numpy(float) if "rate" in d.columns else np.zeros(len(d))
    mid = d["vampMid"].astype(str).to_numpy(object)
    profile = d["profile"].astype(str).to_numpy(object)

    mids = list(pd.unique(mid))
    mid_rows = _group_indices(mid)
    profile_rows = _group_indices(profile)
    _corder = _profile_recip_order(profile_rows, rate)   # per-profile rows by ascending rate (bit-identical)
    constrained: set = set()

    for m in mids:
        if m not in a_max_by_mid:
            continue
        a_max = max(float(a_max_by_mid[m]), 0.0)
        rows = mid_rows[m]
        bvol = float((profile_vol[rows] * bshare[rows]).sum())
        cvol = float((profile_vol[rows] * share[rows]).sum())
        if bvol <= 1e-12:
            continue
        if cvol <= a_max * bvol + 1e-9:
            continue                                   # already within the cap
        constrained.add(m)
        f = (a_max * bvol) / cvol if cvol > 1e-12 else 0.0   # per-share scale factor
        for i in rows:
            freed = share[i] * (1.0 - f)
            share[i] *= f
            if freed <= 1e-12:
                continue
            for j in (k for k in _corder[profile[i]]
                      if mid[k] != m and share[k] < max_share - 1e-9):
                take = min(max_share - share[j], freed)
                share[j] += take
                freed -= take
                if freed <= 1e-12:
                    break

    for c, rows in profile_rows.items():
        s = share[rows].sum()
        if s > 0:
            share[rows] = share[rows] / s

    d["share"] = share
    return d, constrained


# [FN-199]
def optimise_split(problems: list[ProfileProblem],
                   settings: OptimiserSettings) -> pd.DataFrame:
    """Solve every profile with the selected engine and assemble the long split table.

    Runs the chosen engine over every ProfileProblem at the slider's current weight and stacks
    the results into one tidy "long" DataFrame (one row per profile × gateway that receives
    volume). Gateways with a negligible share (<1e-9) are dropped.
    """
    engine = get_engine(settings.engine, settings.risk_conversion_weight,
                        settings.hard, settings.soft, **settings.engine_params)
    # Per-column accumulators + a single DataFrame at the end. Byte-identical to the old
    # list-of-dicts build (same rows in the same order, same per-column values/dtypes,
    # same column order), but pandas builds a frame from dict-of-lists far faster than by
    # inferring schema across one dict per row.
    _c_rpgt, _c_cur, _c_bank, _c_gw = [], [], [], []
    _c_pmp, _c_ctry = [], []                      # profile identity (default "_all_" = profile grain)
    _c_share, _c_vol, _c_cvol = [], [], []
    _c_gsr, _c_grr, _c_ces, _c_cer, _c_bshare = [], [], [], [], []
    _c_feas, _c_note = [], []
    for p in problems:
        sol = engine.solve(p)
        _has = len(sol.shares)
        for i, gw in enumerate(p.gateways):
            share = float(sol.shares[i]) if _has else 0.0
            if share < 1e-9:
                continue
            _c_rpgt.append(p.rpgt); _c_cur.append(p.currency); _c_bank.append(p.bin)
            _c_pmp.append(getattr(p, "pmp", "_all_")); _c_ctry.append(getattr(p, "ctry", "_all_"))
            _c_gw.append(gw)
            _c_share.append(share)
            _c_vol.append(p.volume * share)
            _c_cvol.append(p.volume)
            _c_gsr.append(float(p.success_rates[i]))
            _c_grr.append(float(p.risk_rates[i]))
            _c_ces.append(sol.expected_success_rate)
            _c_cer.append(sol.expected_risk_rate)
            _c_bshare.append(float(p.baseline_shares[i]))
            _c_feas.append(sol.feasible)
            _c_note.append(sol.note)
    if not _c_share:
        return pd.DataFrame([])   # preserve the old empty-frame (0×0) shape when nothing routes
    return pd.DataFrame({
        "rpgt": _c_rpgt, "currency": _c_cur, "bin": _c_bank,
        "pmp": _c_pmp, "ctry": _c_ctry,          # profile identity (carried through for profile grain)
        "gateway": _c_gw,
        "share": _c_share,
        "volume": _c_vol,
        "profile_volume": _c_cvol,
        "gateway_success_rate": _c_gsr,
        "gateway_risk_rate": _c_grr,
        "profile_expected_success": _c_ces,
        "profile_expected_risk": _c_cer,
        "baseline_share": _c_bshare,
        "feasible": _c_feas,
        "note": _c_note,
    })


# [FN-200]
def portfolio_summary(split: pd.DataFrame) -> dict:
    """Volume-weighted headline numbers for a whole split (the book-level scorecard).

    Blends every gateway-row's success and risk rate by its volume — i.e. what the WHOLE
    proposed book is expected to convert at and risk at — plus a count of profiles that failed
    a hard constraint. Like averaging a fleet's fuel economy weighted by miles driven.
    """
    if split.empty:
        return {"volume": 0.0, "expected_success_rate": 0.0,
                "expected_risk_rate": 0.0, "infeasible_profiles": 0}
    volumes = split["volume"].to_numpy()
    _tot_vol = volumes.sum()
    total_volume = max(_tot_vol, 1)
    expected_success = (volumes * split["gateway_success_rate"]).sum() / total_volume
    expected_risk = (volumes * split["gateway_risk_rate"]).sum() / total_volume
    infeasible_profiles = split.loc[~split["feasible"], ["rpgt", "currency", "bin"]].drop_duplicates()
    return {
        "volume": float(_tot_vol),
        "expected_success_rate": float(expected_success),
        "expected_risk_rate": float(expected_risk),
        "infeasible_profiles": int(len(infeasible_profiles)),
    }


# [FN-201]
def sweep_slider(problems: list[ProfileProblem], settings: OptimiserSettings,
                 weights=None) -> pd.DataFrame:
    """Produce split *variations* across the conversion↔risk slider.

    Re-solves the whole book at each slider position and records its headline success/risk,
    tracing the Pareto frontier the UI can plot and let the user pick from. (Used by
    scripts/run_pipeline.py; the app itself now delivers a single split.)
    """
    if weights is None:
        weights = np.round(np.linspace(0.0, 1.0, 11), 2)
    rows = []
    for weight in weights:
        slider_settings = OptimiserSettings(
            risk_conversion_weight=float(weight), engine=settings.engine,
            engine_params=dict(settings.engine_params),
            hard=settings.hard, soft=settings.soft,
        )
        split = optimise_split(problems, slider_settings)
        summary = portfolio_summary(split)
        summary["weight"] = float(weight)
        rows.append(summary)
    return pd.DataFrame(rows)
