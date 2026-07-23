"""
Run a chosen engine across every cell and assemble the proposed split.

This is the layer the UI calls. It:
  * loops over all CellProblems, solving each with the selected engine,
  * returns a tidy "long" split table (one row per cell x gateway),
  * can sweep the conversion<->risk slider to produce split *variations*
    (the family of solutions along the Pareto frontier).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import OptimiserSettings
from .engines import CellProblem, get_engine

__build__ = "2026-07-21-enforce-recip-order-precompute"


def _vamp_cap_lp(df: pd.DataFrame, cap: float, floor: float = 0.0, max_share: float = 1.0,
                 agg_cap: float = None):
    """Joint solve for the per-vampMid VAMP cap: the split CLOSEST to the reference
    (minimum total share movement) whose every vampMid AGGREGATE VAMP rate is <= cap,
    subject to per-cell shares summing to 1 and the exploration-floor / max-share
    bounds. Solved as one sparse LP (min L1 movement, linear rate constraints), so it
    retains more revenue than the greedy's lowest-rate dumping and resolves all cells
    together. Returns (adjusted_df, retired, still_over) ONLY if it finds a fully
    cap-compliant solution; otherwise None so the caller falls back to the greedy shave.
    Guarded: needs SciPy/HiGHS and re-checks compliance after solving.

    `agg_cap` (optional): also constrain the WHOLE-book aggregate VAMP rate to <= agg_cap.
    This is what the true-frontier sweep uses — starting from the revenue reference and
    tightening agg_cap dial-by-dial gives the min-movement (max-revenue) split at each
    risk budget, i.e. a Pareto-optimal frontier point rather than a linear share blend."""
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
    except Exception:  # noqa: BLE001
        return None
    d = df.reset_index(drop=True)
    ref = d["share"].to_numpy(float)
    rate = d["rate"].to_numpy(float)
    vol = d["cell_vol"].to_numpy(float)
    n = len(d)
    if n == 0:
        return None
    cell_rows = _group_indices(d["cell"].astype(str).to_numpy())
    mid_rows = _group_indices(d["vampMid"].astype(str).to_numpy())
    # Only MIDs that CAN breach (some cell rate above the cap) need a constraint.
    over_mids = [m for m, r in mid_rows.items() if float(rate[r].max()) > cap + 1e-12]
    if not over_mids and agg_cap is None:
        return None   # reference already cap-compliant and no aggregate budget — greedy no-ops
    rows, cols, data, b_ub = [], [], [], []
    _r = 0
    for i in range(n):                       # L1:  x_i - u_i <= ref_i
        rows += [_r, _r]; cols += [i, n + i]; data += [1.0, -1.0]; b_ub.append(float(ref[i])); _r += 1
    for i in range(n):                       # L1: -x_i - u_i <= -ref_i
        rows += [_r, _r]; cols += [i, n + i]; data += [-1.0, -1.0]; b_ub.append(-float(ref[i])); _r += 1
    for m in over_mids:                      # Σ vol·(rate-cap)·x <= 0  (aggregate rate <= cap)
        for i in mid_rows[m]:
            _c = float(vol[i]) * (float(rate[i]) - cap)
            if _c != 0.0:
                rows.append(_r); cols.append(int(i)); data.append(_c)
        b_ub.append(0.0); _r += 1
    if agg_cap is not None:                  # whole-book aggregate rate <= agg_cap (frontier budget)
        for i in range(n):
            _c = float(vol[i]) * (float(rate[i]) - float(agg_cap))
            if _c != 0.0:
                rows.append(_r); cols.append(int(i)); data.append(_c)
        b_ub.append(0.0); _r += 1
    A_ub = sp.coo_matrix((data, (rows, cols)), shape=(_r, 2 * n)).tocsr()
    erows, ecols, edata, b_eq, _e = [], [], [], [], 0
    for _c, idx in cell_rows.items():        # each cell's shares sum to 1
        for i in idx:
            erows.append(_e); ecols.append(int(i)); edata.append(1.0)
        b_eq.append(1.0); _e += 1
    A_eq = sp.coo_matrix((edata, (erows, ecols)), shape=(_e, 2 * n)).tocsr()
    c_obj = np.concatenate([np.zeros(n), np.ones(n)])
    bounds = [(float(floor), float(max_share))] * n + [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=np.asarray(b_ub, float),
                      A_eq=A_eq, b_eq=np.asarray(b_eq, float), bounds=bounds, method="highs")
    except Exception:  # noqa: BLE001
        return None
    if not getattr(res, "success", False):
        return None
    x = np.clip(np.asarray(res.x[:n], dtype=float), 0.0, max_share)
    for _c, idx in cell_rows.items():        # exact renormalise (guard tiny LP residuals)
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
    return out, retired, still_over


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


def _group_indices(labels: np.ndarray) -> dict:
    """{label -> ascending row positions}, identical to
    ``{v: np.where(labels == v)[0] for v in unique(labels)}`` but built in ONE
    pass (pandas groupby) instead of a full-array scan per distinct label.

    The old dict-comprehension was O(n_rows · n_labels) — for ~692k rows and
    ~19k cells that's ~1.3e10 object-string comparisons *per call*, the dominant
    cost of the VAMP-cap phase. This is O(n_rows) and returns the SAME arrays
    (sorted ascending), so every downstream move — and the result — is identical.
    """
    ser = pd.Series(labels)
    return {lbl: np.sort(np.asarray(idx, dtype=np.int64))
            for lbl, idx in ser.groupby(ser, sort=False).indices.items()}


def _cell_recip_order(cell_rows: dict, rate: np.ndarray) -> dict:
    """Per-cell row positions sorted by rate ASCENDING, ties broken by ascending row index.

    This is BIT-IDENTICAL to the inline ``sorted(gen, key=lambda j: rate[j])`` used per move,
    where ``gen`` yields ``cell_rows[c]`` (already ascending row index) and Python's sort is
    stable (equal rates keep index order). Precomputing it ONCE lets the per-move recipient scan
    become a filter over this fixed order instead of re-sorting the cell every iteration — the
    move sequence, and therefore the result, is unchanged. Rates are constant, so the order is too.
    """
    return {c: rows[np.argsort(rate[rows], kind="stable")] for c, rows in cell_rows.items()}


def enforce_mid_vamp_caps(df: pd.DataFrame, cap: float, floor: float = 0.0,
                          max_share: float = 1.0, max_iter: int = 4000,
                          step: float = 0.05):
    """Cross-cell adjustment so each vampMid's AGGREGATE VAMP rate <= cap.

    A vampMid spans many routing cells; its Visa-monitored rate is the volume-
    weighted mean of its per-cell rates. Starting from the reference split, we
    iteratively shave share off the MID's HIGHEST-rate cells (handing it to the
    lowest-rate other gateway in that cell) until the MID's aggregate rate is
    under the cap - which minimises movement from the reference. A MID that can't
    be brought under the cap by re-weighting (its rate exceeds the cap in every
    cell) is RETIRED (share -> 0, exempt from the floor) and its volume handed to
    compliant gateways.

    df columns: cell, gateway, vampMid, cell_vol, rate, share (reference start).
    Returns (adjusted_df, retired_set, still_over_set).

    PRIMARY path: a joint LP (`_vamp_cap_lp`) that solves all cells together for the
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
    cell_vol = d["cell_vol"].to_numpy(float)
    mid = d["vampMid"].astype(str).to_numpy(object)
    cell = d["cell"].astype(str).to_numpy(object)

    mids = list(pd.unique(mid))
    mid_rows = _group_indices(mid)
    cell_rows = _group_indices(cell)
    _corder = _cell_recip_order(cell_rows, rate)   # per-cell rows by ascending rate (bit-identical)
    retired: set = set()

    def _mid_rate(m):
        rows = mid_rows[m]
        vol = cell_vol[rows] * share[rows]
        tot = vol.sum()
        return float((vol * rate[rows]).sum() / tot) if tot > 1e-12 else 0.0

    # INCREMENTAL rate maintenance. A MID's aggregate rate = num/den where
    #   den = Σ cell_vol·share ,  num = Σ cell_vol·share·rate  (over its rows).
    # Every move shifts `delta` from ONE row of MID m to ONE row of MID m(j), so we
    # update num/den for just those two MIDs in O(1) — instead of re-summing all of a
    # MID's rows (which is O(11k) for a MID spanning thousands of cells, the cause of
    # the 5-minute VAMP phase). rate/cell_vol are constants, so the update is exact;
    # accumulated float drift is ~1e-13, far below the 1e-9 decision threshold, so the
    # sequence of moves (and the resulting split) is identical to the full-recompute.
    _num, _den = {}, {}
    for m in mids:
        _rows = mid_rows[m]
        _v = cell_vol[_rows] * share[_rows]
        _den[m] = float(_v.sum())
        _num[m] = float((_v * rate[_rows]).sum())

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
            recs = [j for j in _corder[cell[i]]
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
            _num[m] -= cell_vol[i] * delta * rate[i]; _den[m] -= cell_vol[i] * delta
            _num[_mj] += cell_vol[j] * delta * rate[j]; _den[_mj] += cell_vol[j] * delta
            rate_cache[m] = _rt(m)
            rate_cache[_mj] = _rt(_mj)
            if pos[j] < pstart[_mj]:
                pstart[_mj] = int(pos[j])
            moved = True
            break
        if not moved:
            # Can't reduce by re-weighting -> retire the MID (dump its volume onto
            # the lowest-rate other gateways in each of its cells).
            retired.add(m)
            _touched = set()
            for i in mid_rows[m]:
                freed = share[i]
                if freed <= 1e-12:
                    continue
                for j in (k for k in _corder[cell[i]]
                          if mid[k] != m and share[k] < max_share - 1e-9):
                    take = min(max_share - share[j], freed)
                    share[j] += take
                    share[i] -= take
                    _mj = mid[j]
                    _num[m] -= cell_vol[i] * take * rate[i]; _den[m] -= cell_vol[i] * take
                    _num[_mj] += cell_vol[j] * take * rate[j]; _den[_mj] += cell_vol[j] * take
                    if pos[j] < pstart[_mj]:
                        pstart[_mj] = int(pos[j])
                    _touched.add(_mj)
                    freed -= take
                    if freed <= 1e-12:
                        break
            rate_cache[m] = _rt(m)
            for _tm in _touched:
                rate_cache[_tm] = _rt(_tm)

    # Renormalise each cell to sum 1 (safety against rounding).
    for c, rows in cell_rows.items():
        s = share[rows].sum()
        if s > 0:
            share[rows] = share[rows] / s

    d["share"] = share
    still_over = {m for m in mids if _mid_rate(m) > cap + 1e-9}   # fresh recompute
    return d, retired, still_over


def enforce_mid_volume_caps(df: pd.DataFrame, a_max_by_mid: dict,
                            max_share: float = 1.0):
    """Scale each vampMid's allocated volume down to a_max x its BASELINE volume.

    `a_max_by_mid[mid]` is the maximum allowed (proposed / baseline) volume ratio
    for that vampMid, derived upstream from its per-MID monthly VAMP-count / Txn
    caps (and 0 if a rate cap it can't meet by re-weighting forces retirement).
    A MID whose current proposed volume exceeds a_max x baseline is scaled back
    uniformly across its cells; the freed share is handed to the other gateways in
    each cell (lowest-rate first). MIDs not in the dict are untouched.

    df columns: cell, gateway, vampMid, cell_vol, baseline_share, share, rate.
    Returns (adjusted_df, constrained_set).
    """
    d = df.reset_index(drop=True).copy()
    share = d["share"].to_numpy(float).copy()
    bshare = d["baseline_share"].to_numpy(float)
    cell_vol = d["cell_vol"].to_numpy(float)
    rate = d["rate"].to_numpy(float) if "rate" in d.columns else np.zeros(len(d))
    mid = d["vampMid"].astype(str).to_numpy(object)
    cell = d["cell"].astype(str).to_numpy(object)

    mids = list(pd.unique(mid))
    mid_rows = _group_indices(mid)
    cell_rows = _group_indices(cell)
    _corder = _cell_recip_order(cell_rows, rate)   # per-cell rows by ascending rate (bit-identical)
    constrained: set = set()

    for m in mids:
        if m not in a_max_by_mid:
            continue
        a_max = max(float(a_max_by_mid[m]), 0.0)
        rows = mid_rows[m]
        bvol = float((cell_vol[rows] * bshare[rows]).sum())
        cvol = float((cell_vol[rows] * share[rows]).sum())
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
            for j in (k for k in _corder[cell[i]]
                      if mid[k] != m and share[k] < max_share - 1e-9):
                take = min(max_share - share[j], freed)
                share[j] += take
                freed -= take
                if freed <= 1e-12:
                    break

    for c, rows in cell_rows.items():
        s = share[rows].sum()
        if s > 0:
            share[rows] = share[rows] / s

    d["share"] = share
    return d, constrained


def optimise_split(problems: list[CellProblem],
                   settings: OptimiserSettings) -> pd.DataFrame:
    """Solve every cell with the selected engine at the slider's current weight."""
    engine = get_engine(settings.engine, settings.risk_conversion_weight,
                        settings.hard, settings.soft, **settings.engine_params)
    rows = []
    for p in problems:
        sol = engine.solve(p)
        for i, gw in enumerate(p.gateways):
            share = float(sol.shares[i]) if len(sol.shares) else 0.0
            if share < 1e-9:
                continue
            rows.append({
                "rpgt": p.rpgt, "currency": p.currency, "bank": p.bank,
                "gateway": gw,
                "share": share,
                "volume": p.volume * share,
                "cell_volume": p.volume,
                "gateway_success_rate": float(p.success_rates[i]),
                "gateway_risk_rate": float(p.risk_rates[i]),
                "cell_expected_success": sol.expected_success_rate,
                "cell_expected_risk": sol.expected_risk_rate,
                "baseline_share": float(p.baseline_shares[i]),
                "feasible": sol.feasible,
                "note": sol.note,
            })
    return pd.DataFrame(rows)


def portfolio_summary(split: pd.DataFrame) -> dict:
    """Volume-weighted headline numbers for a whole split."""
    if split.empty:
        return {"volume": 0.0, "expected_success_rate": 0.0,
                "expected_risk_rate": 0.0, "infeasible_cells": 0}
    v = split["volume"].to_numpy()
    succ = (v * split["gateway_success_rate"]).sum() / max(v.sum(), 1)
    risk = (v * split["gateway_risk_rate"]).sum() / max(v.sum(), 1)
    infeasible = split.loc[~split["feasible"], ["rpgt", "currency", "bank"]].drop_duplicates()
    return {
        "volume": float(v.sum()),
        "expected_success_rate": float(succ),
        "expected_risk_rate": float(risk),
        "infeasible_cells": int(len(infeasible)),
    }


def sweep_slider(problems: list[CellProblem], settings: OptimiserSettings,
                 weights=None) -> pd.DataFrame:
    """
    Produce split *variations* across the conversion<->risk slider.
    Returns one row per weight with headline success/risk, tracing the
    Pareto frontier the UI can plot and let the user pick from.
    """
    if weights is None:
        weights = np.round(np.linspace(0.0, 1.0, 11), 2)
    out = []
    for w in weights:
        s = OptimiserSettings(
            risk_conversion_weight=float(w), engine=settings.engine,
            engine_params=dict(settings.engine_params),
            hard=settings.hard, soft=settings.soft,
        )
        split = optimise_split(problems, s)
        summ = portfolio_summary(split)
        summ["weight"] = float(w)
        out.append(summ)
    return pd.DataFrame(out)
