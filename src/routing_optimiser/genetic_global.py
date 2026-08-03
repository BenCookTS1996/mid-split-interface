"""
Genetic-algorithm router — CROSS-CELL per-vampMid tilt search.

`run_midtilt_ga` is the live entry point. Its genome is ONE tilt θ_m per vampMid
(~20 numbers), which shifts that MID's volume from its HIGH-risk cells toward its
LOW-risk cells — directly controlling the per-vampMid CROSS-cell VAMP rate (the
actual constraint) at a tiny, fast search dimension. The split is decoded from the
revenue reference as  share_g ∝ ref_g · exp(−θ_{mid(g)} · z_g)  (z_g = risk
standardised WITHIN the MID), renormalised per cell so freed volume redistributes
in proportion to the revenue reference (revenue-efficient recipients).

Fitness (maximised) = expected_revenue − λ · risk_penalty (per-vampMid aggregate
VAMP rate + per-MID volume caps + max-share / floor), vectorised. Deterministic
given `seed`. The caller (tab 3) uses this GA for the compliant (dial-0) endpoint,
guards it against greedy (only adopting it when compliant AND higher-revenue), and
blends it with the revenue reference across the slider.

(The earlier raw-share global GA, per-cell-tilt reparam GA and NSGA-II variants were
removed once the cross-cell per-MID tilt superseded them.)
"""
from __future__ import annotations

import numpy as np

__build__ = "2026-07-31-riskmin-diverse-seeds+eligibility-in-score+fixed-quadratic-breach+numba-eligibility-kernel+vol-weighted-viol+penalty-shape+repair-input+sigma-controls+fitness-trace+active-priority+viol-breakdown+capcell-detail+breach-tol+band-workings"


def _mid_sums(vol, mid_rows, M, S=None):
    """Per-MID column sums of `vol` (P, N) -> (P, M).

    Fast path (`S` given): `S` is a precomputed sparse (M, N) 0/1 incidence matrix
    (row m has 1s on that MID's columns), so the per-MID sums are ONE sparse matmul
    `(S @ vol.T).T` in optimised C instead of a Python loop over the MIDs — the hot
    inner op of every candidate evaluation. Numerically identical to the loop up to
    floating-point summation order (~1e-15); build it once with `_build_mid_incidence`.
    Fallback: loop over the ~20 MIDs (vectorised per MID)."""
    if S is not None:
        return np.asarray(S.dot(vol.T)).T
    out = np.empty((vol.shape[0], M), dtype=float)
    for m in range(M):
        r = mid_rows[m]
        out[:, m] = vol[:, r].sum(axis=1) if len(r) else 0.0
    return out


def _build_mid_incidence(mid_id, M, N):
    """Sparse (M, N) 0/1 incidence for `_mid_sums`' fast path: entry (m, n)=1 iff row n
    belongs to vampMid m. Built once per problem; CSR so `S @ vol.T` is a single BLAS-ish
    sparse-dense product. `mid_id` (N,) maps each row to its MID index in [0, M)."""
    from scipy import sparse as _sp
    mid_id = np.asarray(mid_id)
    cols = np.arange(N)
    data = np.ones(N, dtype=float)
    return _sp.csr_matrix((data, (mid_id, cols)), shape=(M, N))


def _fitness(pop, ctx, lam):
    """Vectorised fitness. Revenue is the SAME quantity tab 4 shows as incremental
    revenue (maximising it ≡ maximising the delta vs a fixed baseline): each row's
    `rev_coef` = 30D-attempts × raw gateway success rate × avg ticket, so revenue =
    Σ share·rev_coef. All penalties are in those $-revenue units, so λ∈[0,1] is
    meaningful: λ=1 values a risk breach at ~1× the revenue it earns; λ=0 ignores
    risk. Max-share / floor carry a fixed heavy weight so they hold at every λ.
    RISK terms (VAMP rate, per-MID volume) use the FORECAST volume basis, matching
    the VAMP projection. Returns (P,) fitness."""
    cv, risk, rc = ctx["cell_vol"], ctx["risk"], ctx["rev_coef"]
    mid_rows, M = ctx["mid_rows"], ctx["n_mid"]
    rev_row = pop * rc[None, :]                           # tab-4-aligned revenue per row
    revenue = rev_row.sum(axis=1)

    # A breach costs MID_revenue × (BREACH_FIXED · breached + over²): a big FIXED hit the
    # instant a cap is crossed (so the GA treats a cap almost like a wall), plus a
    # QUADRATIC term so deeper breaches hurt sharply more. `over` is the RELATIVE breach
    # (actual / limit − 1). Zero penalty while compliant.
    risk_pen = np.zeros(pop.shape[0], dtype=float)
    _bfix = float(ctx.get("breach_fixed", 50.0))
    _bands = ctx.get("midband")
    if M and (ctx["vamp_cap"] is not None or ctx["mid_vol_cap"] is not None or _bands):
        vol = pop * cv[None, :]                           # forecast volume (for risk)
        midv = _mid_sums(vol, mid_rows, M)
        midrev = _mid_sums(rev_row, mid_rows, M)          # MID revenue = penalty scale
        if ctx["vamp_cap"] is not None:                   # per-vampMid aggregate VAMP rate
            midvr = _mid_sums(vol * risk[None, :], mid_rows, M)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(midv > 1e-12, midvr / midv, 0.0)
            over = np.maximum(rate / max(ctx["vamp_cap"], 1e-9) - 1.0, 0.0)
            risk_pen += (midrev * (_bfix * (over > 1e-12) + over ** 2)).sum(axis=1)
        if ctx["mid_vol_cap"] is not None:                # per-MID volume / projected caps
            _cap_v = np.where(ctx["mid_vol_cap"] > 0, ctx["mid_vol_cap"], np.inf)
            over_v = np.maximum(midv / _cap_v[None, :] - 1.0, 0.0)
            risk_pen += (midrev * (_bfix * (over_v > 1e-12) + over_v ** 2)).sum(axis=1)
        if _bands:                                        # month-specific per-MID PROJECTED bands
            # Each candidate's projected per-MID VAMP/Txn for the rule's month(s) is estimated
            # by a volume-ratio proxy: projected ≈ baseline_projected × (MID volume / baseline
            # MID volume). Tilting a MID up shrinks its volume → shrinks its projected metric,
            # so the GA can evolve toward the bands. (vamp_pct rules are scale-invariant under this
            # proxy → excluded by the caller and left to the post-GA enforcement.)
            #
            # FIXED + QUADRATIC penalty ($-scaled by MID revenue): a fixed `band_fixed` hit the
            # instant a band is breached (either side), PLUS a `band_weight` quadratic in the
            # relative breach so deeper misses hurt progressively more. The fixed hit is kept
            # BELOW the VAMP cap's wall (250) so the hard compliance cap still outranks the bands.
            # Trade-off: the fixed element can cost some conversion when a band is genuinely
            # unreachable — mitigated now by the gain lever (reach) and the dial-0 floor clamp.
            _band_w = float(ctx.get("band_weight", 8.0))
            _band_fix = float(ctx.get("band_fixed", 20.0))    # fixed hit on ANY band breach
            _bvol = ctx.get("mid_base_vol")
            with np.errstate(divide="ignore", invalid="ignore"):
                _fmid = (np.where(_bvol[None, :] > 1e-12, midv / _bvol[None, :], 1.0)
                         if _bvol is not None else np.ones_like(midv))
            for _b in _bands:
                # band tuple: (mid_index, baseline_proj, ceiling, floor[, var_mult[, prio_mult]]).
                # var_mult scales ONLY the quadratic (VAMP bands harder than txn); prio_mult scales
                # the WHOLE penalty (priority: lower-priority constraints get a smaller weight, so
                # they yield first when the set is infeasible). Both default to 1.0.
                _mi, _bval, _ceil, _floor = _b[0], _b[1], _b[2], _b[3]
                _vmul = float(_b[4]) if len(_b) > 4 else 1.0
                _pmul = float(_b[5]) if len(_b) > 5 else 1.0
                _proj = _fmid[:, _mi] * float(_bval)
                if _ceil is not None:
                    _ov = np.maximum(_proj / max(float(_ceil), 1e-9) - 1.0, 0.0)
                    risk_pen += midrev[:, _mi] * _pmul * (_band_fix * (_ov > 1e-12) + _band_w * _vmul * _ov ** 2)
                if _floor is not None and float(_floor) > 0:
                    _un = np.maximum(1.0 - _proj / max(float(_floor), 1e-9), 0.0)
                    risk_pen += midrev[:, _mi] * _pmul * (_band_fix * (_un > 1e-12) + _band_w * _vmul * _un ** 2)

    # Structural (max-share / floor) — $-equivalent (rev_coef), fixed heavy weight.
    shape = (np.maximum(pop - ctx["max_share"], 0.0) * rc[None, :]).sum(axis=1)
    if ctx["floor"] > 0:
        shape += (np.maximum(ctx["floor"] - pop, 0.0) * (ctx["elig"] * rc)[None, :]).sum(axis=1)

    fit = revenue - lam * risk_pen - ctx["shape_mult"] * shape

    # Optional RISK-MINIMISATION secondary objective (used only for the SAFE compliant
    # endpoint of the slider). It subtracts mu × aggregate expected VAMP count, so among
    # equally-compliant splits the GA prefers the one that also carries LESS total risk —
    # tilting each MID further toward its low-risk cells even below the cap. The caller
    # auto-scales mu (risk_min_w) to trade a bounded slice of revenue for lower risk;
    # default 0 leaves the pure revenue objective unchanged. (compliant-frontier)
    _rmw = float(ctx.get("risk_min_w", 0.0))
    if _rmw > 0.0:
        _vol = pop * cv[None, :]
        _vfr = ctx.get("vamp_floor_route")
        if _vfr is not None and M:
            # CLAMP at the VAMP floor: only reward reducing the VAMP that sits ABOVE each MID's
            # routing-space floor (derived from its two-sided VAMP band floor). Once a MID is at
            # its floor the risk-min term stops pulling, so dial-0 risk-min no longer drives VAMP
            # BELOW the band — keeping the two-sided VAMP ranges satisfiable at dial 0.
            _midvr = _mid_sums(_vol * risk[None, :], mid_rows, M)         # (P, M) routing VAMP/MID
            _excess = np.maximum(_midvr - np.asarray(_vfr, float)[None, :], 0.0)
            fit = fit - _rmw * _excess.sum(axis=1)
        else:
            _total_vamp = (_vol * risk[None, :]).sum(axis=1)   # aggregate expected VAMP count
            fit = fit - _rmw * _total_vamp
    return fit


# ---------------------------------------------------------------------------
# CROSS-CELL per-MID tilt search (the efficient one that can actually beat greedy).
#
# The per-cell tilt above reweights WITHIN a cell, so it can't move a vampMid's
# CROSS-cell aggregate VAMP rate — which is exactly the constraint. This version
# searches ONE parameter per vampMid: a cross-cell tilt θ_m that shifts that MID's
# volume from its HIGH-risk cells toward its LOW-risk cells:
#
#     share_g ∝ ref_g · exp(−θ_{mid(g)} · z_g)   (z_g = risk standardised WITHIN the MID)
#
# Raising θ_m pulls MID m out of its riskiest cells (dropping its aggregate rate)
# and the freed share redistributes per cell in proportion to the revenue reference
# (revenue-efficient recipients — unlike greedy, which dumps onto the lowest-rate
# gateway). Genome = n_mid (~20) dims, so the search is tiny and fast, AND it targets
# the real per-MID cross-cell constraint, so it can retain more revenue at compliance
# than the greedy shave on MIDs whose risk varies across cells.
# ---------------------------------------------------------------------------
def _risk_z_per_mid(risk, mid_rows, n_mid, N):
    """Standardise each vampMid's per-cell risk across ITS rows, so θ_m tilts that
    MID toward its own lower-risk cells."""
    z = np.zeros(N, dtype=float)
    for m in range(n_mid):
        r = mid_rows[m]
        if len(r) == 0:
            continue
        rr = risk[r]
        sd = rr.std()
        z[r] = (rr - rr.mean()) / sd if sd > 1e-12 else 0.0
    return z


def _cap_floor_shares(X, cell_starts, cell_counts, elig, cap, floor):
    """HARD per-cell max-share cap + exploration floor on shares X (P, N), vectorised.
    Cells are contiguous segments (cell_starts / cell_counts). Applies the floor first
    (lift eligible gateways to a per-cell-clamped floor, renormalise), then WATER-FILLS
    any over-cap excess into the same cell's under-cap eligible gateways — enforced LAST,
    so every row exits with share <= cap. Cells with < 2 eligible gateways can't be capped
    (a single gateway must be 1.0), matching the export's _cap_shares. Guarantees the GA
    only ever evaluates deployable splits (no search/output mismatch)."""
    elig_row = elig[None, :] > 0.5
    n_elig_cell = np.repeat(np.add.reduceat(elig.astype(float), cell_starts), cell_counts)  # (N,)
    if floor > 0.0:
        # Per-cell floor clamped to 1/n_elig so n_elig×floor <= 1 stays feasible. Lift any
        # below-floor eligible gateway to the floor, and take the deficit from ABOVE-floor
        # eligible gateways in proportion to their room above the floor (water-fill DOWN) — so
        # the sum stays 1 and the floored values are NOT shrunk back under by a global renorm.
        fl = np.minimum(floor, np.where(n_elig_cell > 0, 1.0 / np.maximum(n_elig_cell, 1.0),
                                        0.0))[None, :]
        for _ in range(50):
            under = elig_row & (X < fl - 1e-12)
            if not under.any():
                break
            deficit_cell = np.repeat(np.add.reduceat(np.where(under, fl - X, 0.0), cell_starts,
                                                     axis=1), cell_counts, axis=1)
            X = np.where(under, fl, X)
            give = np.where(elig_row & (~under) & (X > fl + 1e-12), X - fl, 0.0)
            give_cell = np.repeat(np.add.reduceat(give, cell_starts, axis=1), cell_counts, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X - np.where(give_cell > 1e-12, give * deficit_cell / give_cell, 0.0)
    if cap < 1.0:
        capN = np.where(n_elig_cell >= 2, cap, 1.0)[None, :]     # single-gateway cells uncapped
        for _ in range(50):
            over = X > capN + 1e-12
            if not over.any():
                break
            excess_col = np.where(over, X - capN, 0.0)
            excess_cell = np.repeat(np.add.reduceat(excess_col, cell_starts, axis=1),
                                    cell_counts, axis=1)
            X = np.where(over, capN, X)
            room = np.where(elig_row & (~over) & (X < capN - 1e-12), capN - X, 0.0)
            room_cell = np.repeat(np.add.reduceat(room, cell_starts, axis=1), cell_counts, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X + np.where(room_cell > 1e-12, room * excess_cell / room_cell, 0.0)
    return X


def _mid_over(shares, ctx, include_floor_shortfall=True):
    """Per-candidate per-MID breach magnitude (P, M): the RELATIVE overage (actual/limit − 1,
    or floor short-fall) across the VAMP-rate cap, per-MID volume cap and projected bands —
    whichever is worst for that MID. >0 means that MID currently breaches. READ-ONLY: it
    mirrors the risk maths in `_fitness` but is used only to STEER the search (adaptive-λ /
    breach-targeted mutation). It never changes feasibility or the score, so if it is slightly
    off the worst case is imperfect targeting, never a masked breach.

    `include_floor_shortfall=False` reports ONLY the breaches that are fixed by SHEDDING a MID's
    volume (VAMP-rate cap, per-MID volume cap, band CEILING). It omits band-FLOOR shortfalls —
    where a MID's projected metric is too LOW and needs MORE volume — because the Lamarckian
    repair raises θr (sheds volume), which would push a floor-short MID further from feasibility."""
    _eop = ctx.get("elig_op")
    if _eop is not None:                                     # steer on the ACTUALLY-ROUTABLE shares
        from .eligibility import apply_elig_pop              # (consistent with the _obj_viol scorer)
        shares = apply_elig_pop(shares, _eop)
    mid_rows, M = ctx["mid_rows"], int(ctx["n_mid"])
    P = shares.shape[0]
    over = np.zeros((P, M), dtype=float)
    if M == 0:
        return over
    cv, risk = ctx["cell_vol"], ctx["risk"]
    _S = ctx.get("_mid_S")                                    # fast per-MID sums when available
    vol = shares * cv[None, :]
    midv = _mid_sums(vol, mid_rows, M, S=_S)
    if ctx.get("vamp_cap") is not None:                       # per-vampMid aggregate VAMP rate
        midvr = _mid_sums(vol * risk[None, :], mid_rows, M, S=_S)
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(midv > 1e-12, midvr / midv, 0.0)
        over = np.maximum(over, np.maximum(rate / max(ctx["vamp_cap"], 1e-9) - 1.0, 0.0))
    if ctx.get("mid_vol_cap") is not None:                    # per-MID volume / projected caps
        _cap_v = np.where(ctx["mid_vol_cap"] > 0, ctx["mid_vol_cap"], np.inf)
        over = np.maximum(over, np.maximum(midv / _cap_v[None, :] - 1.0, 0.0))
    _bands = ctx.get("midband")
    if _bands:                                                # month-specific per-MID bands
        _bvol = ctx.get("mid_base_vol")
        with np.errstate(divide="ignore", invalid="ignore"):
            _fmid = (np.where(_bvol[None, :] > 1e-12, midv / _bvol[None, :], 1.0)
                     if _bvol is not None else np.ones_like(midv))
        for _b in _bands:
            _mi, _bval, _ceil, _floor = _b[0], _b[1], _b[2], _b[3]
            _proj = _fmid[:, _mi] * float(_bval)
            if _ceil is not None:
                over[:, _mi] = np.maximum(over[:, _mi],
                                          np.maximum(_proj / max(float(_ceil), 1e-9) - 1.0, 0.0))
            if include_floor_shortfall and _floor is not None and float(_floor) > 0:
                over[:, _mi] = np.maximum(over[:, _mi],
                                          np.maximum(1.0 - _proj / max(float(_floor), 1e-9), 0.0))
    return over


# ===========================================================================
# CMA-ES cross-cell per-vampMid tilt search  (live engine as of 2026-07-25).
#
# Replaces the hand-rolled GA above. Eight upgrades, all always-on:
#   1. CMA-ES  — a covariance-adapting evolution strategy searches the tilt
#      genome, learning which MIDs to move together and self-tuning its step size.
#      ACTIVE variant: the worst samples' directions are actively shrunk (negative
#      recombination weights) for faster, more reliable convergence at zero extra
#      evaluations.
#   2. Memetic gradient polish — after CMA-ES, an SLSQP refine using the ANALYTIC
#      gradient of revenue AND of the (smooth) violation through the softmax decode
#      lands on the local KKT optimum precisely (Nelder–Mead only as a fallback).
#      Accepted only if feasibility-first BETTER.
#   3. ε-constrained feasibility ranking — early generations rank with a RELAXED
#      violation tolerance ε that shrinks to 0, so the search can cross slightly
#      infeasible ground to reach better basins, then tightens to exact compliance.
#      The returned best is always judged STRICTLY feasible (ε=0).
#   4. Smooth 'wall' — the constraint measure is the CONTINUOUS relative overage
#      (0 exactly at the cap), so the search sees a smooth gradient toward compliance.
#   5. Second tilt axis — each MID has θr (toward LOW-risk cells), θq (toward
#      HIGH-revenue cells) and g (overall presence):
#          share_g ∝ ref_g · exp(−θr·zr_g + θq·zq_g + g_mid)     genome = 3·n_mid.
#   6. Freed-volume redistribution — the reference base is leaned low-risk (see 8),
#      so shed volume lands on LOW-risk recipients.
#   7. IPOP restarts + archive reseed — several CMA-ES restarts, each reseeded from
#      the incumbent with a DOUBLED population (IPOP) and wider step, returning a
#      diverse archive.
#   8. Leaned reference — the θ=0 base is nudged toward lower risk by γ (per-endpoint:
#      ≈0 at the revenue-max end, larger at the risk-min end).
#   9. Lamarckian repair — before scoring, each breaching MID's risk-tilt θr is nudged
#      up (down the violation gradient) and written back, so the population drifts
#      toward feasibility and surfaces higher-revenue feasible points.
#  10. Greedy warm-start — the caller can pass a known FEASIBLE split (e.g. the greedy
#      compliant one); a genome is fitted to it and seeds the search INSIDE the feasible
#      region, so the whole budget goes to improving revenue (via ctx['warm_shares']).
#
# Speed (numerically identical to ~1e-15): a precomputed sparse MID incidence matrix
# for the hot per-MID sums; the incumbent's score reused from the last generation (no
# re-evaluation); C^{-1/2} precomputed once per eigen-update; the decode's exp computed
# only on eligible columns; per-cell cap/floor constants precomputed once; ctx arrays
# forced float64-contiguous once.
#
# Deterministic given `seed`. Same (best_shares, info) contract as before, so the tab-3
# caller and its warm-start / archive plumbing are unchanged.
# ===========================================================================
def _ret_z_per_mid(ret, mid_rows, n_mid, N):
    """Standardise each vampMid's per-cell REVENUE-efficiency (rev_coef) across ITS
    rows, so the return-tilt axis θq pulls that MID toward its own higher-revenue
    cells (mirror of `_risk_z_per_mid` on the revenue axis)."""
    z = np.zeros(N, dtype=float)
    for m in range(n_mid):
        r = mid_rows[m]
        if len(r) == 0:
            continue
        rr = ret[r]
        sd = rr.std()
        z[r] = (rr - rr.mean()) / sd if sd > 1e-12 else 0.0
    return z


def _leaned_ref(ref, risk, elig, cell_starts, cell_counts, gamma):
    """Lean the revenue reference gently toward LOWER global risk (γ ≥ 0): the θ=0
    base — and therefore where freed volume redistributes — starts a little compliant
    (improvements #6 and #8). γ is dimensionless (risk standardised across eligible
    rows). γ=0 returns the reference unchanged. Result is renormalised per cell."""
    gamma = float(gamma)
    if gamma <= 0.0:
        return np.asarray(ref, float)
    ref = np.asarray(ref, float)
    e = np.asarray(elig, float) > 0.5
    rr = risk[e]
    if rr.size == 0:
        return ref
    sd = rr.std()
    zg = np.zeros_like(risk, dtype=float)
    if sd > 1e-12:
        zg[e] = (risk[e] - rr.mean()) / sd
    w = ref * np.exp(-gamma * zg)
    seg = np.add.reduceat(w, cell_starts)
    seg = np.where(seg > 1e-12, seg, 1.0)
    return w / np.repeat(seg, cell_counts)


def _cap_floor_prep(cell_starts, cell_counts, elig, cap, floor):
    """Precompute the per-cell CONSTANTS the max-share/floor water-fill needs, so they
    are built ONCE per problem instead of on every decode (bit-identical, pure saving).
    Returns None when neither cap nor floor binds. `elig_row`, `capN`, `fl`, `n_elig_cell`
    match the arrays `_cap_floor_shares` recomputes internally each call."""
    cap = float(cap); floor = float(floor)
    if cap >= 1.0 and floor <= 0.0:
        return None
    elig = np.asarray(elig, float)
    elig_row = elig[None, :] > 0.5
    n_elig_cell = np.repeat(np.add.reduceat(elig, cell_starts), cell_counts)      # (N,)
    fl = None
    if floor > 0.0:
        fl = np.minimum(floor, np.where(n_elig_cell > 0, 1.0 / np.maximum(n_elig_cell, 1.0),
                                        0.0))[None, :]
    capN = np.where(n_elig_cell >= 2, cap, 1.0)[None, :] if cap < 1.0 else None
    return {"elig_row": elig_row, "n_elig_cell": n_elig_cell, "fl": fl, "capN": capN,
            "cap": cap, "floor": floor,
            "cell_starts": np.asarray(cell_starts), "cell_counts": np.asarray(cell_counts)}


def _cap_floor_apply(X, prep):
    """HARD per-cell floor-then-cap water-fill using PRECOMPUTED constants (`prep` from
    `_cap_floor_prep`). Byte-identical to `_cap_floor_shares` — same order of operations,
    only the per-cell constants are reused rather than rebuilt each call."""
    cs, ccnt = prep["cell_starts"], prep["cell_counts"]
    elig_row, fl, capN = prep["elig_row"], prep["fl"], prep["capN"]
    if fl is not None:
        for _ in range(50):
            under = elig_row & (X < fl - 1e-12)
            if not under.any():
                break
            deficit_cell = np.repeat(np.add.reduceat(np.where(under, fl - X, 0.0), cs,
                                                     axis=1), ccnt, axis=1)
            X = np.where(under, fl, X)
            give = np.where(elig_row & (~under) & (X > fl + 1e-12), X - fl, 0.0)
            give_cell = np.repeat(np.add.reduceat(give, cs, axis=1), ccnt, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X - np.where(give_cell > 1e-12, give * deficit_cell / give_cell, 0.0)
    if capN is not None:
        for _ in range(50):
            over = X > capN + 1e-12
            if not over.any():
                break
            excess_col = np.where(over, X - capN, 0.0)
            excess_cell = np.repeat(np.add.reduceat(excess_col, cs, axis=1), ccnt, axis=1)
            X = np.where(over, capN, X)
            room = np.where(elig_row & (~over) & (X < capN - 1e-12), capN - X, 0.0)
            room_cell = np.repeat(np.add.reduceat(room, cs, axis=1), ccnt, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                X = X + np.where(room_cell > 1e-12, room * excess_cell / room_cell, 0.0)
    return X


def _risk_z_per_cell(risk, cell_starts, cell_counts, N):
    """Standardise each CELL's per-gateway risk across ITS rows (mirror of `_risk_z_per_mid`
    but per cell), so a per-cell fine tilt can shift share within one cell toward its
    lower-risk gateways. Vectorised via reduceat (no Python loop over cells)."""
    cnt = np.maximum(cell_counts.astype(float), 1.0)
    smean = np.add.reduceat(risk, cell_starts) / cnt
    var = np.add.reduceat(risk * risk, cell_starts) / cnt - smean * smean
    sd = np.sqrt(np.maximum(var, 0.0))
    mean_row = np.repeat(smean, cell_counts)
    sd_row = np.repeat(sd, cell_counts)
    # Divide by a SAFE denominator (1.0 where sd≈0) so the masked-out branch of np.where doesn't
    # trigger a spurious "invalid value in divide" RuntimeWarning — single-gateway / zero-variance
    # cells have no meaningful z, and are zeroed anyway.
    _sd_safe = np.where(sd_row > 1e-12, sd_row, 1.0)
    z = np.where(sd_row > 1e-12, (risk - mean_row) / _sd_safe, 0.0)
    return z, sd            # sd (per-cell) also used to pick the fine-tilt cells


def _decode_midtilt3(genome, M, ref, zr, zq, mid_id, cell_starts, cell_counts, elig,
                     cap=1.0, floor=0.0, *, eidx=None, prep=None,
                     fine_idx=None, zr_cell=None, n_fine=0):
    """genome (P, 3M[+K]) = [θr (risk-tilt) | θq (return-tilt) | g (gain) | cellθ (K fine)] -> (P, N).

        share_g ∝ ref_g · exp(−θr·zr_g + θq·zq_g + g_mid − cellθ_cell·zr_cell_g) · elig

    renormalised per cell, then the HARD max-share cap / exploration floor. θr (≥0) pulls a MID
    toward its LOW-risk cells, θq (≥0) toward its HIGH-revenue cells, g moves its overall
    presence, and the optional per-cell fine tilt cellθ (≥0, `n_fine` of them, on the top-K
    risk-heavy cells via `fine_idx`/`zr_cell`) shifts share WITHIN one cell toward its low-risk
    gateways — extra reach the coarse per-MID tilt can't provide. `n_fine`=0 (default) is the
    exact 3M-genome behaviour.

    Speed (identical results): with `eidx` (eligible column indices) the exp is computed only on
    eligible columns; `prep` reuses precomputed cap/floor constants."""
    if M == 0:
        w = ref[None, :] * elig[None, :]
        seg = np.add.reduceat(w, cell_starts, axis=1)
        seg = np.where(seg > 1e-12, seg, 1.0)
        X = w / np.repeat(seg, cell_counts, axis=1)
        if prep is not None:                                 # enforce the HARD cap/floor here too
            X = _cap_floor_apply(X, prep)
        elif cap < 1.0 or floor > 0.0:
            X = _cap_floor_shares(X, cell_starts, cell_counts, elig, float(cap), float(floor))
        return X
    P = genome.shape[0]; N = mid_id.shape[0]
    tr = genome[:, :M]; tq = genome[:, M:2 * M]; gg = genome[:, 2 * M:3 * M]
    _cols = eidx if (eidx is not None and eidx.shape[0] < N) else np.arange(N)
    mi = mid_id[_cols]
    a = -tr[:, mi] * zr[None, _cols] + tq[:, mi] * zq[None, _cols] + gg[:, mi]
    if int(n_fine) and fine_idx is not None:                 # per-cell fine tilt on selected cells
        _fi = fine_idx[_cols]
        _fm = _fi >= 0
        if _fm.any():
            _cth = genome[:, 3 * M:3 * M + int(n_fine)]      # (P, K)
            a[:, _fm] = a[:, _fm] - _cth[:, _fi[_fm]] * zr_cell[None, _cols][:, _fm]
    w = np.zeros((P, N), dtype=float)
    w[:, _cols] = ref[None, _cols] * np.exp(a) * elig[None, _cols]
    seg = np.add.reduceat(w, cell_starts, axis=1)
    seg = np.where(seg > 1e-12, seg, 1.0)
    X = w / np.repeat(seg, cell_counts, axis=1)
    if prep is not None:
        X = _cap_floor_apply(X, prep)
    elif cap < 1.0 or floor > 0.0:
        X = _cap_floor_shares(X, cell_starts, cell_counts, elig, float(cap), float(floor))
    return X


def _mid_viol_weights(ctx, M):
    """Per-MID VIOLATION weight (VOLUME-WEIGHTING of the feasibility violation).

    weight[m] = MID baseline volume / mean(baseline volume over MIDs with volume > 0), so the
    AVERAGE MID keeps weight ≈ 1 (the overall violation scale — and hence the feasibility
    tolerance and the breach_quad meaning — is preserved), while a high-volume MID's breach
    counts proportionally MORE than a tiny MID's. This stops the search wasting its gradient
    flattening hundreds of trivially-small breaches it cannot tell apart from the ones that
    matter (see the 5-txn cad cells vs the 44,000-txn usd cells in the run profiles).

    Returns ALL-ONES — i.e. behaviour byte-identical to the un-weighted violation — when
    ctx['viol_vol_weight'] is off or there is no baseline-volume basis. SINGLE SOURCE OF TRUTH:
    both `_obj_viol` (numpy) and the numba kernel builder call THIS, so the two hard-verified
    scoring paths cannot drift apart."""
    M = int(M)
    if not bool(ctx.get("viol_vol_weight", False)) or M <= 0:
        return np.ones(max(M, 1), dtype=float)
    _base = ctx.get("mid_base_vol")
    if _base is None:
        return np.ones(M, dtype=float)
    base = np.ascontiguousarray(np.asarray(_base, dtype=float))
    if base.shape[0] != M:
        return np.ones(M, dtype=float)
    _pos = base[base > 0.0]
    _bmean = float(_pos.mean()) if _pos.size else 0.0
    if _bmean <= 0.0:
        return np.ones(M, dtype=float)
    return np.where(base > 0.0, base / _bmean, 1.0).astype(float)


def _obj_viol(shares, ctx):
    """Split the score into (objective, violation) for feasibility-first ranking.

    objective (P,) = revenue [− risk_min term], MAXIMISED among FEASIBLE splits.
    violation (P,) = summed breach penalty across the per-vampMid VAMP-rate cap, per-MID volume
        caps and projected bands. Each breaching term contributes a FIXED hit
        (ctx['breach_fixed'] — the UI 'Cap-breach penalty') the instant it goes over, PLUS a
        QUADRATIC in the relative overage (ctx['breach_quad'], default 1.0). Exactly 0 when
        compliant. NOTE: the fixed hit reintroduces a non-smooth step, so the memetic gradient
        polish (which follows the smooth-violation gradient) becomes less effective; set
        breach_fixed=0 to recover the pure smooth wall. Mirrors `_fitness` / the full-matrix
        engine AND the numba kernel (`numba_kernels._fused_eval`) — keep all three in lock-step."""
    _eop = ctx.get("elig_op")
    if _eop is not None:                                     # score the ACTUALLY-ROUTABLE shares —
        from .eligibility import apply_elig_pop              # bans + wallet/USA capability folded in
        shares = apply_elig_pop(shares, _eop)                # so the search optimises what will route
    cv, risk, rc = ctx["cell_vol"], ctx["risk"], ctx["rev_coef"]
    mid_rows, M = ctx["mid_rows"], int(ctx["n_mid"])
    _S = ctx.get("_mid_S")                                   # precomputed incidence (fast path)
    P = shares.shape[0]
    revenue = (shares * rc[None, :]).sum(axis=1)
    obj = revenue.astype(float).copy()
    viol = np.zeros(P, dtype=float)
    # BREACH PENALTY = fixed hit (the instant a constraint is over) + quadratic in the relative
    # overage. _bfix = ctx['breach_fixed'] (UI 'Cap-breach penalty'); _qwt scales the quadratic.
    # _pen(over) is applied identically here and in numba_kernels._fused_eval — keep them in sync.
    _bfix = float(ctx.get("breach_fixed", 0.0) or 0.0)
    _qwt = float(ctx.get("breach_quad", 1.0) or 1.0)
    # PENALTY SHAPE: how the smooth part grows with the relative overage `over`.
    #   quadratic  (default): qwt · over²          — double the overage → 4× penalty
    #   exponential          : qwt · (exp(over) − 1) — grows far faster the further out (over clipped
    #                          at 50 so it can never overflow to inf). Identical form in the kernel.
    _pexp = (str(ctx.get("breach_shape", "quadratic")).lower() == "exponential")

    def _pen(_ov):                                           # _ov = relative overage array (>= 0)
        # TOLERANCE: a constraint met to within 1e-9 (relative) is COMPLIANT — don't let the fixed
        # hit fire on floating-point rounding dust (e.g. the decode water-fills a share to exactly
        # the cap and lands at cap+1e-12). Mirrored in numba_kernels._fused_eval; keep in lockstep.
        _ov = np.where(_ov > 1e-9, _ov, 0.0)
        if _pexp:
            return _bfix * (_ov > 0.0) + _qwt * (np.exp(np.minimum(_ov, 50.0)) - 1.0)
        return _bfix * (_ov > 0.0) + _qwt * _ov * _ov
    # VOLUME-WEIGHTING (#4): per-MID importance weight so a high-volume MID's breach outranks a
    # trivially-small one. ones ⇒ un-weighted (byte-identical). Same vector the numba kernel uses.
    _wm = _mid_viol_weights(ctx, M)
    if M and (ctx.get("vamp_cap") is not None or ctx.get("mid_vol_cap") is not None
              or ctx.get("midband")):
        vol = shares * cv[None, :]
        midv = _mid_sums(vol, mid_rows, M, S=_S)
        if ctx.get("vamp_cap") is not None:
            midvr = _mid_sums(vol * risk[None, :], mid_rows, M, S=_S)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(midv > 1e-12, midvr / midv, 0.0)
            viol += (_pen(np.maximum(rate / max(ctx["vamp_cap"], 1e-9) - 1.0, 0.0))
                     * _wm[None, :]).sum(axis=1)
        if ctx.get("mid_vol_cap") is not None:
            _cap_v = np.where(ctx["mid_vol_cap"] > 0, ctx["mid_vol_cap"], np.inf)
            viol += (_pen(np.maximum(midv / _cap_v[None, :] - 1.0, 0.0))
                     * _wm[None, :]).sum(axis=1)
        # EXACT BANDS (gate 2): under ctx['exact_bands'], the month bands are scored EXACTLY per
        # generation in the eval wrapper (band_scoring.ExactBandPenalty), so drop the volume-ratio
        # PROXY term here — the numba kernel drops it under the same flag, keeping both in lockstep.
        _bands = None if ctx.get("exact_bands") else ctx.get("midband")
        if _bands:
            _bvol = ctx.get("mid_base_vol")
            with np.errstate(divide="ignore", invalid="ignore"):
                _fmid = (np.where(_bvol[None, :] > 1e-12, midv / _bvol[None, :], 1.0)
                         if _bvol is not None else np.ones_like(midv))
            for _b in _bands:
                _mi, _bval, _ceil, _floor = _b[0], _b[1], _b[2], _b[3]
                _pmul = float(_b[5]) if len(_b) > 5 else 1.0   # PRIORITY weight (5000^(1-p)); low prio yields
                _proj = _fmid[:, _mi] * float(_bval)
                if _ceil is not None:
                    viol += _pen(np.maximum(_proj / max(float(_ceil), 1e-9) - 1.0, 0.0)) * _wm[_mi] * _pmul
                if _floor is not None and float(_floor) > 0:
                    viol += _pen(np.maximum(1.0 - _proj / max(float(_floor), 1e-9), 0.0)) * _wm[_mi] * _pmul
    # Structural caps AND the exploration floor are hard-enforced in the decode, so these are
    # ≈0 — kept as a symmetric safety net so a split the decode could not fully repair still
    # ranks as infeasible (previously only the cap was checked, not the floor).
    _cap = float(ctx.get("max_share", 1.0) or 1.0)
    if _cap < 1.0:
        viol += _pen(np.maximum(shares - _cap, 0.0) / max(_cap, 1e-9)).sum(axis=1)
    _floor = float(ctx.get("floor", 0.0) or 0.0)
    if _floor > 0.0 and ctx.get("cell_starts") is not None:
        _cs = np.asarray(ctx["cell_starts"]); _cc = np.asarray(ctx["cell_counts"])
        _el = np.asarray(ctx["elig"], float)
        _nec = np.repeat(np.add.reduceat(_el, _cs), _cc)         # eligible gateways per cell
        _fl = np.minimum(_floor, np.where(_nec > 0, 1.0 / np.maximum(_nec, 1.0), 0.0))
        _mask = (_el > 0.5) & (_nec >= 2)                        # single-gateway cells can't be floored
        viol += _pen(np.maximum(_fl[None, :] - shares, 0.0) * _mask[None, :]
                     / max(_floor, 1e-9)).sum(axis=1)
    # RISK-MINIMISATION secondary objective (safe compliant endpoint only). Among equally
    # compliant splits, prefer the one that also carries LESS total risk. Auto-scaled by
    # the caller via ctx['risk_min_w']; 0 leaves the pure-revenue objective unchanged.
    _rmw = float(ctx.get("risk_min_w", 0.0))
    if _rmw > 0.0 and M:
        _vol = shares * cv[None, :]
        _vfr = ctx.get("vamp_floor_route")
        if _vfr is not None:
            _midvr = _mid_sums(_vol * risk[None, :], mid_rows, M, S=_S)
            _excess = np.maximum(_midvr - np.asarray(_vfr, float)[None, :], 0.0)
            obj = obj - _rmw * _excess.sum(axis=1)
        else:
            obj = obj - _rmw * (_vol * risk[None, :]).sum(axis=1)
    return obj, viol


def _violation_breakdown(shares, ctx, top_k=20):
    """One-shot DECOMPOSITION of the _obj_viol violation for a SINGLE split (diagnostic only —
    NEVER on the hot path; called once per run on the winning candidate). Mirrors _obj_viol
    term-for-term but keeps every component separate so the caller can SEE what the ranked
    violation is actually made of, instead of inferring it:
        • vamp_cap      — the per-vampMid aggregate VAMP-rate cap (0.06)
        • bands         — the per-MID month bands, split BY PRIORITY TIER + top offenders
        • max_share     — the structural per-cell max-gateway-share cap (0.97)
        • floor         — the structural exploration floor (0.01)
      plus how far the eligibility mask moved the split (L1 + rows zeroed), how many TERMS breach
      in each component (the fixed-hit count is what scales the total when breach_fixed is large),
      and how many cells are structurally infeasible (floor can't be given to every gateway).
    Returns a plain dict (JSON-ish). Recomputed in NumPy; matches the ranked violation to ~1e-12."""
    import math
    shares = np.asarray(shares, float)
    if shares.ndim == 1:
        shares = shares[None, :]
    shares = shares[:1]                                       # a single split only
    pre = shares.copy()
    out = {"elig_l1": 0.0, "elig_zeroed": 0}
    _eop = ctx.get("elig_op")
    if _eop is not None:
        from .eligibility import apply_elig_pop
        shares = apply_elig_pop(shares, _eop)                # SAME masking _obj_viol scores through
        out["elig_l1"] = float(np.abs(shares - pre).sum())
        out["elig_zeroed"] = int(((pre > 1e-9) & (shares <= 1e-12)).sum())
    cv, risk = ctx["cell_vol"], ctx["risk"]
    mid_rows, M = ctx["mid_rows"], int(ctx["n_mid"])
    _S = ctx.get("_mid_S")
    _bfix = float(ctx.get("breach_fixed", 0.0) or 0.0)
    _qwt = float(ctx.get("breach_quad", 1.0) or 1.0)
    _pexp = (str(ctx.get("breach_shape", "quadratic")).lower() == "exponential")

    def _pen(_ov):
        _ov = np.asarray(_ov, float)
        _ov = np.where(_ov > 1e-9, _ov, 0.0)                 # at-boundary within tol = compliant (see _obj_viol)
        if _pexp:
            return _bfix * (_ov > 0.0) + _qwt * (np.exp(np.minimum(_ov, 50.0)) - 1.0)
        return _bfix * (_ov > 0.0) + _qwt * _ov * _ov

    _wm = _mid_viol_weights(ctx, M)
    x = shares[0]
    vol = x * cv
    midv = _mid_sums(vol[None, :], mid_rows, M, S=_S)[0] if M else np.zeros(0)
    # --- VAMP-rate cap ---
    vamp_total = 0.0; vamp_over = 0
    if ctx.get("vamp_cap") is not None and M:
        midvr = _mid_sums((vol * risk)[None, :], mid_rows, M, S=_S)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(midv > 1e-12, midvr / midv, 0.0)
        ov = np.maximum(rate / max(ctx["vamp_cap"], 1e-9) - 1.0, 0.0)
        vamp_total = float((_pen(ov) * _wm).sum()); vamp_over = int((ov > 0).sum())
    # --- per-MID month bands (split by priority tier + worst offenders) ---
    bands_total = 0.0; bands_tier = {}; offenders = []
    _bands = ctx.get("midband")
    if _bands and M:
        _bvol = ctx.get("mid_base_vol")
        with np.errstate(divide="ignore", invalid="ignore"):
            fmid = (np.where(_bvol > 1e-12, midv / _bvol, 1.0) if _bvol is not None else np.ones(M))
        def _pen_split(_o):
            # RAW (pre-weight) fixed + quadratic penalty parts, so the log can show the workings.
            _o = _o if _o > 1e-9 else 0.0                     # same tolerance as _pen
            _f = _bfix if _o > 0.0 else 0.0
            _q = (_qwt * (math.exp(min(_o, 50.0)) - 1.0)) if _pexp else (_qwt * _o * _o)
            return _f, _q
        for _b in _bands:
            _mi, _bval, _ceil, _floor = _b[0], _b[1], _b[2], _b[3]
            _pmul = float(_b[5]) if len(_b) > 5 else 1.0
            _w = float(_wm[_mi]) * _pmul                       # priority × volume weight applied to the band
            proj = float(fmid[_mi] * float(_bval))
            ratio = None; kind = None; _ovr = 0.0; _fx = 0.0; _qd = 0.0
            if _ceil is not None:
                ov = max(proj / max(float(_ceil), 1e-9) - 1.0, 0.0)
                if ov > 1e-9:
                    _fx, _qd = _pen_split(ov)
                    ratio = proj / max(float(_ceil), 1e-9); kind = "ceil"; _ovr = ov
            if _floor is not None and float(_floor) > 0:
                un = max(1.0 - proj / max(float(_floor), 1e-9), 0.0)
                if un > 1e-9:
                    _fx, _qd = _pen_split(un)
                    ratio = proj / max(float(_floor), 1e-9); kind = "floor"; _ovr = un
            c = (_fx + _qd) * _w                               # applied contribution (== _pen(ov)·w)
            if c > 0:
                bands_total += c
                pr = int(round(1.0 - math.log(_pmul) / math.log(5000.0))) if _pmul > 0 else 99
                bands_tier[pr] = bands_tier.get(pr, 0.0) + c
                offenders.append({"mid": int(_mi), "kind": kind, "proj": proj,
                                  "ratio": (float(ratio) if ratio is not None else None),
                                  "prio": pr, "contrib": float(c), "over": float(_ovr),
                                  "fixed": float(_fx), "quad": float(_qd), "weight": float(_w)})
        offenders.sort(key=lambda d: -d["contrib"]); offenders = offenders[:top_k]
    # --- structural max-share cap ---
    _cap = float(ctx.get("max_share", 1.0) or 1.0)
    cap_total = 0.0; cap_rows = 0; cap_offenders = []
    if _cap < 1.0:
        ov = np.maximum(x - _cap, 0.0) / max(_cap, 1e-9)
        cap_total = float(_pen(ov).sum()); cap_rows = int((ov > 0).sum())
        # WHICH rows are over the cap, and WHY: the decode can only leave a row over-cap when its
        # cell has <2 ELIGIBLE gateways (a lone usable gateway must be ~100%). Capture each over-cap
        # cell's eligible/present gateway counts + volume + vampMid so the log can confirm the cause.
        _ovi = np.where(ov > 0)[0]
        if _ovi.size and ctx.get("cell_starts") is not None:
            _cs2 = np.asarray(ctx["cell_starts"]); _cc2 = np.asarray(ctx["cell_counts"])
            _el2 = np.asarray(ctx["elig"], float)
            _nec_by_cell = np.add.reduceat(_el2, _cs2)               # eligible gateways per cell
            _row2cell = np.repeat(np.arange(len(_cs2)), _cc2)        # row index -> its cell index
            _cvv = ctx.get("cell_vol"); _mid_id = ctx.get("mid_id")
            for _ri in _ovi[:20]:
                _ci = int(_row2cell[_ri])
                cap_offenders.append({
                    "share": float(x[_ri]),
                    "cell_eligible": int(round(float(_nec_by_cell[_ci]))),
                    "cell_present": int(_cc2[_ci]),
                    "cell_vol": (float(_cvv[_ri]) if _cvv is not None else None),
                    "mid": (int(_mid_id[_ri]) if _mid_id is not None else -1),
                })
    # --- structural exploration floor ---
    _floor = float(ctx.get("floor", 0.0) or 0.0)
    floor_total = 0.0; floor_rows = 0; floor_infeasible_cells = 0
    if _floor > 0.0 and ctx.get("cell_starts") is not None:
        _cs = np.asarray(ctx["cell_starts"]); _cc = np.asarray(ctx["cell_counts"])
        _el = np.asarray(ctx["elig"], float)
        _nec = np.repeat(np.add.reduceat(_el, _cs), _cc)
        _fl = np.minimum(_floor, np.where(_nec > 0, 1.0 / np.maximum(_nec, 1.0), 0.0))
        _mask = (_el > 0.5) & (_nec >= 2)
        short = np.maximum(_fl - x, 0.0) * _mask / max(_floor, 1e-9)
        floor_total = float(_pen(short).sum()); floor_rows = int((short > 0).sum())
        _nec_cell = np.add.reduceat(_el, _cs)
        floor_infeasible_cells = int((_nec_cell * _floor > 1.0 + 1e-9).sum())
    out.update({
        "total": float(vamp_total + bands_total + cap_total + floor_total),
        "vamp_cap": {"total": vamp_total, "mids_over": vamp_over},
        "bands": {"total": bands_total, "by_prio": bands_tier, "offenders": offenders,
                  "n_bands": (len(_bands) if _bands else 0)},
        "max_share": {"total": cap_total, "rows_over": cap_rows, "cap": _cap,
                      "offenders": cap_offenders},
        "floor": {"total": floor_total, "rows_under": floor_rows,
                  "infeasible_cells": floor_infeasible_cells, "floor": _floor},
        "bfix": _bfix, "qwt": _qwt,
    })
    return out


_FEAS_TOL = 1e-9


def _feas_keys(obj, viol, tol=_FEAS_TOL):
    """Deb feasibility-first ranking as a MINIMISABLE key (improvement #3). Feasible
    rows rank by −objective (so higher revenue wins); infeasible rows all rank ABOVE
    the worst feasible, ordered by violation. If nothing is feasible, rank by violation
    alone. Order-only — CMA-ES is rank-based, so the per-generation reference is fine."""
    obj = np.asarray(obj, float)
    viol = np.asarray(viol, float)
    key = np.empty_like(obj)
    f = viol <= tol
    if f.any():
        f_worst = float(obj[f].min())
        key[f] = -obj[f]
        key[~f] = -f_worst + viol[~f]
    else:
        key[:] = viol
    return key


def _cmaes(eval_ov, x0, sigma0, lo, hi, *, popsize, max_iter, seed, stop_check=None,
           eps0=0.0, repair=None, progress_cb=None, no_early_stop=False,
           sigma_floor=0.0, damps_mult=1.0):
    """Active (μ/μ_w, λ)-CMA-ES over the box [lo, hi] (bounds via clipping). `eval_ov(X)`
    returns (objective, violation) per row; ranking is feasibility-first with an
    ε-CONSTRAINED tolerance that starts at `eps0` and shrinks to 0 (early generations may
    cross slightly-infeasible ground, later ones are strict). The returned best is judged
    STRICTLY feasible. `repair(X)->X'` is an optional Lamarckian repair applied to samples
    before scoring (repaired points drive the update). Active negative recombination weights
    (Hansen 2016) shrink the worst directions. Returns (best_x, best_key, best_ov, gen_trace).
    Self-contained numpy; deterministic given `seed`."""
    rng = np.random.default_rng(int(seed))
    x0 = np.asarray(x0, float)
    D = x0.shape[0]
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    xmean = np.clip(x0.copy(), lo, hi)
    sigma = float(sigma0)
    lam = int(max(4, popsize)); mu = max(1, lam // 2)
    # Active-CMA raw weights: positive for the best μ, negative for the worst λ−μ.
    wp = np.log((lam + 1) / 2.0) - np.log(np.arange(1, lam + 1))
    w_pos, w_neg = wp[:mu], wp[mu:]
    mueff = float((w_pos.sum() ** 2) / np.sum(w_pos ** 2))
    _neg_ss = float(np.sum(w_neg ** 2))
    mueff_neg = float((w_neg.sum() ** 2) / _neg_ss) if (w_neg.size and _neg_ss > 0) else 0.0
    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
    cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2.0 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    damps = (1 + 2 * max(0.0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs) * float(max(damps_mult, 1e-6))
    w = wp.copy()
    w[:mu] = w_pos / w_pos.sum()                             # positive weights sum to 1
    if w_neg.size:
        _a_mu = 1 + c1 / max(cmu, 1e-23)
        _a_mueff = 1 + 2 * mueff_neg / (mueff + 2)
        _a_posdef = (1 - c1 - cmu) / max(D * cmu, 1e-23)
        _scale_neg = min(_a_mu, _a_mueff, _a_posdef)
        w[mu:] = _scale_neg * w_neg / max(-w_neg.sum(), 1e-23)
    pc = np.zeros(D); ps = np.zeros(D)
    B = np.eye(D); Dd = np.ones(D); C = np.eye(D); invsqrtC = np.eye(D)
    chiN = np.sqrt(D) * (1 - 1.0 / (4 * D) + 1.0 / (21 * D * D))
    best_x = xmean.copy(); best_key = np.inf; best_ov = (-np.inf, np.inf)
    gen_trace = []
    counteval = 0; eigeneval = 0
    _stall = 0; _stall_max = 10 + int(30 * D / lam)          # convergence early-stop (TolFun/TolX)
    _fin_obj = _fin_viol = None                              # last generation's population (obj, viol)
    for it in range(int(max_iter)):
        if stop_check is not None and stop_check():
            break
        Z = rng.standard_normal((lam, D))
        Y = ((B * Dd[None, :]) @ Z.T).T                      # (lam, D) = B·diag(Dd)·z
        X = xmean[None, :] + sigma * Y
        Xc = np.clip(X, lo[None, :], hi[None, :])
        if repair is not None:                               # Lamarckian: repaired points are used
            Xc = np.clip(np.asarray(repair(Xc), float), lo[None, :], hi[None, :])
        obj, viol = eval_ov(Xc)
        obj = np.asarray(obj, float); viol = np.asarray(viol, float)
        counteval += lam
        eps_it = float(eps0) * max(0.0, 1.0 - it / max(1.0, 0.8 * max_iter))   # relax early, strict late
        rkeys = _feas_keys(obj, viol, tol=max(eps_it, _FEAS_TOL))              # SELECTION (relaxed)
        skeys = _feas_keys(obj, viol, tol=_FEAS_TOL)                          # BEST-TRACK (strict)
        idx = np.argsort(rkeys, kind="stable")
        _b = int(np.argmin(skeys))
        if skeys[_b] < best_key - 1e-12:
            best_key = float(skeys[_b]); best_x = Xc[_b].copy()
            best_ov = (float(obj[_b]), float(viol[_b])); _stall = 0
        else:
            _stall += 1
        if progress_cb is not None:                        # live count + best-so-far score (score = -key,
            # higher = better) + the FITNESS (objective/revenue-ish) of that incumbent split, so the
            # live log/chart can show BOTH what the search RANKS on (score = -violation while
            # infeasible) and the incumbent's revenue. Degrades gracefully to 2-arg / 1-arg callbacks.
            _cur_fit = float(best_ov[0]) if np.isfinite(best_ov[0]) else float("nan")
            try:
                progress_cb(int(lam), float(-best_key), _cur_fit)
            except TypeError:
                try:
                    progress_cb(int(lam), float(-best_key))
                except TypeError:                          # oldest 1-arg progress_cb → count only
                    try:
                        progress_cb(int(lam))
                    except Exception:  # noqa: BLE001
                        pass
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001
                pass
        gen_trace.append((float(skeys.min()), float(skeys.mean()), float(sigma),
                          float(viol[_b]), float(eps_it),
                          float(best_ov[0]) if np.isfinite(best_ov[0]) else float("nan")))
        # ^ + best violation, ε tolerance, and incumbent FITNESS (best_ov[0]; revenue−riskmin of the
        #   current best-so-far split) — feeds the fitness series in the UI convergence chart.
        _fin_obj, _fin_viol = obj, viol                       # last generation's population
        _fin_obj, _fin_viol = obj, viol                       # last generation's population
        Xsorted = Xc[idx]
        xold = xmean.copy()
        xmean = w[:mu] @ Xsorted[:mu]                        # mean: POSITIVE weights only
        ysel = (xmean - xold) / sigma
        ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ ysel)
        hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (it + 1))) / chiN) < (1.4 + 2 / (D + 1))
        pc = (1 - cc) * pc + (np.sqrt(cc * (2 - cc) * mueff) if hsig else 0.0) * ysel
        artmp = (Xsorted - xold[None, :]) / sigma            # all λ
        w_o = w.copy()
        if w_neg.size:                                       # length-normalise negative-weighted dirs
            yv = (invsqrtC @ artmp.T).T
            norm2 = np.sum(yv * yv, axis=1) + 1e-23
            _neg = w < 0
            w_o[_neg] = w[_neg] * D / norm2[_neg]
        rank_mu = (artmp.T * w_o) @ artmp
        cdelta = 0.0 if hsig else cc * (2 - cc)
        C = ((1 - c1 - cmu * float(np.sum(w)) + c1 * cdelta) * C
             + c1 * np.outer(pc, pc) + cmu * rank_mu)
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        if sigma_floor > 0.0 and np.isfinite(sigma):
            sigma = max(sigma, float(sigma_floor))   # UI 'Min step-size': stop σ collapsing to micro-steps
        if not np.isfinite(sigma) or sigma <= 0:
            break
        # Convergence early-stop: no strict-best improvement for a while (TolFun), or the
        # search distribution has collapsed below a floor (TolX) — the covariance is spent,
        # so further generations just burn evaluations. Restarts still re-explore afterwards.
        # `no_early_stop` (UI: "Run all generations") disables BOTH so every restart runs the full
        # max_iter — exact candidate count, longer, usually no better result. The non-finite-sigma
        # guard above always stays (it's a numerical safety, not a convergence stop).
        if (not no_early_stop) and (_stall >= _stall_max or sigma * float(np.max(Dd)) < 1e-11):
            break
        if counteval - eigeneval > lam / max(c1 + cmu, 1e-12) / D / 10:
            eigeneval = counteval
            C = np.triu(C) + np.triu(C, 1).T                 # keep symmetric
            vals, B = np.linalg.eigh(C)
            Dd = np.sqrt(np.maximum(vals, 1e-20))
            invsqrtC = (B * (1.0 / Dd)[None, :]) @ B.T        # C^{-1/2}, precomputed once/update
        xmean = np.clip(xmean, lo, hi)
    _fpop = ((np.asarray(_fin_obj, float), np.asarray(_fin_viol, float))
             if _fin_obj is not None else (None, None))
    return best_x, best_key, best_ov, gen_trace, _fpop


def run_midtilt_ga(ctx, lam, *, pop_size=40, generations=80, mutation_rate=0.3,
                   mutation_sigma=1.0, seed=42, elite_frac=0.2, auto=True,
                   patience=12, sigma_min=0.05, sigma_max=4.0, success_window=5,
                   theta_max=25.0, gain_max=2.0, warm_start=None,
                   breach_targeted=True, breach_mut_boost=3.0,
                   smart_init=True, init_tries=4,
                   adaptive_lambda=True, breach_lambda_boost=4.0,
                   archive_k=5, archive_min_dist=0.5, stop_check=None,
                   n_restarts=2, polish=True, ref_gamma=None, n_fine=0, progress_cb=None,
                   numba=False, restart_mode="lean", numba_trust=False):
    """Active-CMA-ES cross-cell per-vampMid tilt search — the live engine (see the block
    comment above for the full upgrade list). Genome = [θr | θq | g] (3·n_mid dims).
    Ranking is ε-relaxed feasibility-first (compliant always beats non-compliant at the end),
    so `lam` is no longer a penalty weight — accepted for interface compatibility and ignored.

    Returns (best_shares (N,), info) with info['genome'] (3·n_mid) for warm-starting and
    info['archive'] of diverse good genomes. Deterministic given `seed`.

    Optional ctx keys: 'warm_shares' (a known feasible split to seed inside the feasible
    region, #10), 'ref_gamma' (compliant lean γ, #8), '_mid_S' (cached incidence — built here
    if absent). Legacy GA-only knobs (elite_frac, mutation_*, sigma_*, breach_*,
    adaptive_lambda, smart_init, success_window, patience) are accepted but unused — CMA-ES
    self-adapts. New knobs: n_restarts (IPOP restarts, #7), polish (SLSQP gradient refine, #2),
    ref_gamma (overrides ctx; falls back to ctx['ref_gamma'] then 0.25)."""
    N = ctx["n_row"]; M = int(ctx["n_mid"])
    # Force float64-contiguous ONCE so the hot per-generation ops get zero-copy views.
    _cont = lambda a, dt=float: np.ascontiguousarray(a, dtype=dt)
    cs = _cont(ctx["cell_starts"], np.intp); cc = _cont(ctx["cell_counts"], np.intp)
    elig = _cont(ctx["elig"])
    ref0 = _cont(ctx["ref_share"])
    mid_id = _cont(ctx["mid_id"], np.intp)
    risk = _cont(ctx["risk"])
    rc = _cont(ctx["rev_coef"])
    cv = _cont(ctx["cell_vol"])
    zr = _cont(_risk_z_per_mid(risk, ctx["mid_rows"], M, N))
    zq = _cont(_ret_z_per_mid(rc, ctx["mid_rows"], M, N))
    _cap = float(ctx.get("max_share", 1.0) or 1.0)
    _floor = float(ctx.get("floor", 0.0) or 0.0)
    _gamma = float(ctx.get("ref_gamma", 0.25) if ref_gamma is None else ref_gamma)
    ref = _cont(_leaned_ref(ref0, risk, elig, cs, cc, _gamma))   # #8 leaned compliant base
    theta_max = float(theta_max); gain_max = float(gain_max)

    # --- precomputed hot-path structures (built once; used every candidate) ---
    _S = ctx.get("_mid_S")
    if _S is None and M:
        _S = _build_mid_incidence(mid_id, M, N)             # sparse per-MID sums (speed #1)
    ctx["_mid_S"] = _S                                       # so _obj_viol uses the fast path
    _eidx = np.nonzero(elig > 0.5)[0].astype(np.intp)        # eligible columns (speed #4)
    _prep = _cap_floor_prep(cs, cc, elig, _cap, _floor)      # cap/floor constants (speed, identical)
    # EXACT bands: drop the proxy band term from the SLSQP-polish gradient too (it was only a
    # local direction hint; the accept test is exact). Under exact_bands there is NO proxy anywhere.
    _bands = None if ctx.get("exact_bands") else ctx.get("midband")
    _base_v = ctx.get("mid_base_vol")
    _vcap = ctx.get("vamp_cap"); _volcap = ctx.get("mid_vol_cap")

    # #4 richer genome: give the TOP-K risk-heavy cells (high volume × within-cell risk spread)
    # their own fine tilt so the search can move share WITHIN a cell toward low-risk gateways —
    # reach the coarse per-MID tilt lacks. n_fine=0 keeps the exact 3M-genome behaviour.
    _K = int(max(0, min(int(n_fine), len(cs))))
    _fine_idx = None; _zr_cell = None
    if _K > 0 and M > 0:
        _zr_cell, _cell_sd = _risk_z_per_cell(risk, cs, cc, N)
        _cell_vol = cv[cs]                                   # per-cell volume (rows share it)
        _score = _cell_vol * _cell_sd                        # worth-tilting = big & risk-spread
        _fine_cells = np.argsort(-_score, kind="stable")[:_K]
        _fine_idx = np.full(N, -1, dtype=np.intp)
        for _j, _ci in enumerate(_fine_cells):
            _s0 = int(cs[_ci]); _s1 = _s0 + int(cc[_ci])
            _fine_idx[_s0:_s1] = _j
        _zr_cell = _cont(_zr_cell)

    def _decode(G):
        return _decode_midtilt3(G, M, ref, zr, zq, mid_id, cs, cc, elig, _cap, _floor,
                                eidx=_eidx, prep=_prep,
                                fine_idx=_fine_idx, zr_cell=_zr_cell, n_fine=_K)

    def _decode_precap(G):                                   # softmax shares w/o cap/floor (grads)
        return _decode_midtilt3(G, M, ref, zr, zq, mid_id, cs, cc, elig, 1.0, 0.0,
                                eidx=_eidx, prep=None,
                                fine_idx=_fine_idx, zr_cell=_zr_cell, n_fine=_K)

    # --- degenerate: no per-MID levers -> the (leaned) reference is the answer ---
    if M == 0:
        best = _decode(np.zeros((1, 0)))[0]
        _rev = float((best * rc).sum())
        return best, {"gens": 0, "gens_max": int(generations), "early_stopped": False,
                      "sigma_final": 0.0, "best_fit": _rev, "init_fit": _rev,
                      "revenue": _rev, "risk_cost": 0.0, "dims": 0,
                      "genome": np.zeros(0), "archive": np.zeros((1, 0)), "history": []}

    D = 3 * M + _K
    # CMA-ES runs in a UNIT box [0,1]^D so all coordinates share a scale; map to the actual
    # bounds (θr,θq ∈ [0, theta_max]; g ∈ [−gain_max, gain_max]; cellθ ∈ [0, theta_max]).
    lo_a = np.concatenate([np.zeros(M), np.zeros(M), np.full(M, -gain_max), np.zeros(_K)])
    hi_a = np.concatenate([np.full(M, theta_max), np.full(M, theta_max), np.full(M, gain_max),
                           np.full(_K, theta_max)])
    span = np.where(hi_a - lo_a > 1e-12, hi_a - lo_a, 1.0)

    def _to_actual(V):
        return lo_a[None, :] + np.asarray(V, float) * span[None, :]

    def _to_unit(x):
        return np.clip((np.asarray(x, float) - lo_a) / span, 0.0, 1.0)

    # EXACT band penalty (gate 2): added to the violation for EVERY candidate, using the
    # PRE-eligibility decoded shares. The numba kernel now RETURNS those shares, so there is NO
    # NumPy re-decode on the hot path (the old double-decode). Zero when no exact bands configured.
    _exact_bands = ctx.get("exact_bands")
    _band_inc = ctx.get("band_incidence")
    if _exact_bands is not None and _band_inc is not None:
        from .band_scoring import shares_to_prop_raw as _s2pr

        def _bands_pen(X):
            return _exact_bands.penalty(_s2pr(X, _band_inc))
    else:
        def _bands_pen(X):
            return np.zeros(np.asarray(X).shape[0], dtype=float)

    def eval_ov(V):                                          # unit pop -> (objective, violation)
        X = _decode(_to_actual(V))
        obj, viol = _obj_viol(X, ctx)
        return obj, viol + _bands_pen(X)

    def score_of(gu):                                        # unit genome -> (obj, viol)
        X = _decode(_to_actual(np.asarray(gu)[None, :]))
        obj, viol = _obj_viol(X, ctx)
        return float(obj[0]), float(viol[0] + _bands_pen(X)[0])

    def better(a, b):                                        # feasibility-first strict-better
        ao, av = a; bo, bv = b
        af = av <= _FEAS_TOL; bf = bv <= _FEAS_TOL
        if af != bf:
            return af
        return ao > bo if af else av < bv

    def _cellrep(v):                                         # (N,) -> per-cell sum, repeated (N,)
        return np.repeat(np.add.reduceat(v, cs), cc)

    def _midsum1(v):                                         # (N,) -> (M,) per-MID sum (fast path)
        return (np.asarray(_S.dot(v)).ravel() if _S is not None
                else _mid_sums(v[None, :], ctx["mid_rows"], M)[0])

    # ---- OPT-IN Numba fast path ("GA - Numba" engine only) --------------------------------
    # numba=False (the default, i.e. the production Genetic engine) leaves EVERYTHING above
    # untouched. When numba=True we build a fused float64 kernel and swap it in for the hot
    # per-generation eval. Any failure -> keep NumPy. The verify result is returned in
    # info['numba'] so the caller can log it for cross-validation.
    #   numba_trust=False: VERIFY the kernel here (NumPy-vs-Numba on a genome sample) before use.
    #   numba_trust=True : the main-process pre-compile ALREADY verified this exact kernel/signature,
    #                      so skip the per-worker re-check (it was 16× redundant — one NumPy eval and
    #                      one comparison per worker). Build the kernel and use it directly.
    ga_numba_info = {"requested": bool(numba), "used": False, "reason": "not requested"}
    # Eligibility-in-scoring (ctx['elig_op']): the fused kernel now REPLICATES apply_elig_pop
    # (bans->0+renorm, wallet/USA blend) on the decoded shares before scoring — see
    # numba_kernels._fused_eval — so the Numba fitness == the NumPy _obj_viol fitness even when
    # eligibility scoring is active. No longer force-disabled; verify() below (or the main-process
    # pre-compile) still cross-checks NumPy-vs-kernel and falls back to NumPy on ANY mismatch, so a
    # divergent build can never silently optimise the wrong fitness.
    if numba:
        ga_numba_info["reason"] = ""
        try:
            from . import numba_kernels as _nbk
            if not _nbk.NUMBA_OK:
                ga_numba_info["reason"] = "numba not importable in this environment"
            else:
                _nb_eval_actual = _nbk.make_numba_eval(M, ref, zr, zq, mid_id, cs, cc, elig,
                                                       _cap, _floor, _fine_idx, _zr_cell, _K,
                                                       cv, risk, rc, ctx)
                ga_numba_info["build"] = getattr(_nbk, "__build__", "")
                if numba_trust:
                    _accept = True                       # already verified by the main pre-compile
                    ga_numba_info["trusted"] = True
                    ga_numba_info["reason"] = "trusted (verified by main-process pre-compile)"
                    # Time ONE eval so the caller can tell whether this worker LOADED the cached
                    # kernel (fast, ~<1s) or had to RE-COMPILE it (slow, tens of seconds = cache
                    # miss in the worker) — the key clue for a slow multi-seed search.
                    try:
                        import time as _nt
                        _t0 = _nt.time()
                        _nb_eval_actual(_to_actual(np.zeros((1, D))))
                        ga_numba_info["first_eval_s"] = float(_nt.time() - _t0)
                    except Exception:  # noqa: BLE001
                        pass
                else:
                    _np_eval_actual = lambda G: _obj_viol(_decode(G), ctx)   # noqa: E731
                    # Deterministic verification sample: zero (θ=0 base), warm-start if any, + random.
                    _vrng = np.random.default_rng(int(seed) ^ 0x5A17)
                    _vs = [np.zeros(D)]
                    if warm_start is not None:
                        try:
                            _vs.append(_to_unit(np.asarray(warm_start, float)))
                        except Exception:  # noqa: BLE001
                            pass
                    _vs += list(_vrng.random((6, D)))
                    _sampleG = _to_actual(np.clip(np.asarray(_vs, float), 0.0, 1.0))
                    # Pass the fixed-breach magnitude so verify tolerates knife-edge step flips
                    # (whole-bfix jumps) while still catching genuine smooth-term divergence.
                    _vr = _nbk.verify(_np_eval_actual, _nb_eval_actual, _sampleG,
                                      bfix=float(ctx.get("breach_fixed", 0.0) or 0.0))
                    ga_numba_info.update(_vr)
                    _accept = bool(_vr.get("ok"))
                if _accept:
                    def eval_ov(V):                          # noqa: F811 - numba override
                        _o, _v, _X = _nb_eval_actual(_to_actual(V))   # kernel returns pre-elig shares
                        return _o, _v + _bands_pen(_X)

                    def score_of(gu):                        # noqa: F811 - numba override
                        _o, _v, _X = _nb_eval_actual(_to_actual(np.asarray(gu)[None, :]))
                        return float(_o[0]), float(_v[0] + _bands_pen(_X)[0])

                    ga_numba_info["used"] = True
        except Exception as _e:  # noqa: BLE001 - any failure -> NumPy path, never crash a run
            ga_numba_info["reason"] = f"{type(_e).__name__}: {_e}"

    # (EXACT band penalty is folded directly into eval_ov / score_of above via _bands_pen, using
    # the kernel's returned PRE-eligibility shares — so there is no separate wrapper and no NumPy
    # re-decode. The numba⇄NumPy verifier still compares the band-LESS core, so lockstep holds.)

    # #9/#2 memetic gradients: analytic ∂revenue and ∂(smooth violation) through the
    # ref-weighted softmax decode, for a single genome (unit space). Used by the SLSQP
    # polish. The violation here is the SMOOTH pre-cap surrogate (VAMP-rate + volume + bands);
    # the accept test still uses the true post-cap feasibility-first score.
    def _fine_grad(vec):                                     # (N,) d or e -> (K,) fine-tilt gradient
        if _K <= 0 or _fine_idx is None:
            return np.zeros(0)
        _m = _fine_idx >= 0
        return -np.bincount(_fine_idx[_m], weights=(_zr_cell * vec)[_m], minlength=_K)

    def _grads(gu):
        G = _to_actual(np.clip(gu, 0.0, 1.0)[None, :])
        X = _decode_precap(G)[0]                             # (N,)
        RbarC = _cellrep(rc * X)
        d = X * (rc - RbarC)
        gR = np.concatenate([-_midsum1(zr * d), _midsum1(zq * d), _midsum1(d), _fine_grad(d)]) * span
        R = float((X * rc).sum())
        # violation sensitivities per MID: dV/dA_m (VAMP rate), dV/dB_m (volume + bands)
        u = cv * X
        A = _midsum1(u * risk); Bm = _midsum1(u)
        Bs = np.where(Bm > 1e-12, Bm, 1.0)
        rate = np.where(Bm > 1e-12, A / Bs, 0.0)
        gA = np.zeros(M); gB = np.zeros(M); V = 0.0
        if _vcap is not None:
            o = rate / max(float(_vcap), 1e-9) - 1.0
            act = o > 0.0
            V += float(np.maximum(o, 0.0).sum())
            gr = np.where(act, 1.0 / max(float(_vcap), 1e-9), 0.0)
            gA += gr / Bs
            gB += -gr * rate / Bs
        if _volcap is not None:
            capv = np.where(np.asarray(_volcap) > 0, np.asarray(_volcap, float), np.inf)
            ov = Bm / capv - 1.0
            act = ov > 0.0
            V += float(np.maximum(ov, 0.0).sum())
            gB += np.where(act, 1.0 / capv, 0.0)
        if _bands:
            fmid = (np.where(np.asarray(_base_v) > 1e-12, Bm / np.where(np.asarray(_base_v) > 1e-12,
                    np.asarray(_base_v, float), 1.0), 1.0) if _base_v is not None else np.ones(M))
            for _bd in _bands:
                _mi, _bval, _ceil, _flr = _bd[0], _bd[1], _bd[2], _bd[3]
                _proj = fmid[_mi] * float(_bval)
                _bden = max(float(_base_v[_mi]), 1e-12) if _base_v is not None else 1.0
                if _ceil is not None:
                    _oc = _proj / max(float(_ceil), 1e-9) - 1.0
                    if _oc > 0.0:
                        V += _oc
                        gB[_mi] += (1.0 / max(float(_ceil), 1e-9)) * float(_bval) / _bden
                if _flr is not None and float(_flr) > 0:
                    _un = 1.0 - _proj / max(float(_flr), 1e-9)
                    if _un > 0.0:
                        V += _un
                        gB[_mi] += -(float(_bval) / _bden) / max(float(_flr), 1e-9)
        p = cv * (gA[mid_id] * risk + gB[mid_id])
        e = X * (p - _cellrep(p * X))
        gV = np.concatenate([-_midsum1(zr * e), _midsum1(zq * e), _midsum1(e), _fine_grad(e)]) * span
        return R, gR, V, gV

    # #10 greedy warm-start: fit a genome whose decode ≈ a supplied FEASIBLE split so the
    # search starts INSIDE the feasible region (fitted once; cheap L-BFGS on ‖decode−target‖²).
    def _fit_to_target(target):
        t = np.clip(np.asarray(target, float), 0.0, None)
        try:
            from scipy.optimize import minimize as _minimize

            def _loss(z):
                X = _decode_precap(_to_actual(np.clip(z, 0.0, 1.0)[None, :]))[0]
                return 0.5 * float(((X - t) ** 2).sum())

            _r = _minimize(_loss, _to_unit(np.zeros(D)), method="L-BFGS-B",
                           bounds=[(0.0, 1.0)] * D, options={"maxiter": 60})
            return np.clip(np.asarray(_r.x, float), 0.0, 1.0)
        except Exception:
            return None

    # #1 diverse multi-seed starts: an explicit warm_start genome, each supplied warm split
    # (ctx['warm_shares'] may be ONE split OR a LIST — e.g. revenue-greedy AND risk-greedy),
    # fitted to a genome, plus the θ=0 leaned reference. Deduped, best (feasibility-first) first;
    # each seeds one restart so the search explores several corners, not just one.
    def _seed_key(sv):
        return (0 if sv[1] <= _FEAS_TOL else 1, -sv[0] if sv[1] <= _FEAS_TOL else sv[1])
    _seed_us = []
    if warm_start is not None:
        _ws = np.atleast_2d(np.asarray(warm_start, float))
        if _ws.ndim == 2 and _ws.shape[1] == D:
            for _row in _ws[:3]:
                _seed_us.append(_to_unit(_row))
    _wsh = ctx.get("warm_shares")
    if _wsh is not None:
        _wa = None if isinstance(_wsh, (list, tuple)) else np.asarray(_wsh, float)
        _wlist = (list(_wsh) if isinstance(_wsh, (list, tuple))
                  else ([_wa[i] for i in range(_wa.shape[0])] if (_wa is not None and _wa.ndim == 2)
                        else [_wsh]))
        for _w in _wlist[:3]:
            _f = _fit_to_target(_w)
            if _f is not None:
                _seed_us.append(_f)
    _seed_us.append(_to_unit(np.zeros(D)))                   # leaned reference — always available
    _uniq = {}
    for _u in _seed_us:
        _u = np.clip(np.asarray(_u, float), 0.0, 1.0)
        if _u.shape[0] == D:
            _uniq.setdefault(np.round(_u, 6).tobytes(), _u)
    _seeds = sorted(_uniq.values(), key=lambda u: _seed_key(score_of(u))) or [_to_unit(np.zeros(D))]
    x0_u = _seeds[0]

    # #9 Lamarckian repair: nudge each breaching MID's risk-tilt θr UP (reduces its VAMP rate /
    # volume / band-CEILING overage), written back so the population drifts toward feasibility.
    # Only SHED-fixable breaches drive it (include_floor_shortfall=False), so a MID that is short
    # of a band FLOOR — which needs MORE volume, not less — is never pushed the wrong way. The
    # nudge is a BOUNDED, dimensionless unit-space step: over/(1+over) squashes any relative
    # overage into [0,1), so the θr move is ≤ _REPAIR_LR regardless of the overage scale.
    _REPAIR_LR = float(ctx.get("repair_lr", 0.30) or 0.0)   # UI 'Repair strength' (0 disables the nudge)

    def _repair(Vpop):
        over = _mid_over(_decode(_to_actual(Vpop)), ctx, include_floor_shortfall=False)   # (P, M)
        if over.size == 0 or float(over.max()) <= 1e-12:
            return Vpop
        V2 = np.array(Vpop, float, copy=True)
        _step = _REPAIR_LR * (over / (1.0 + over))           # bounded [0, _REPAIR_LR) unit-space nudge
        V2[:, :M] = np.clip(V2[:, :M] + _step, 0.0, 1.0)
        return V2

    lam_cma = int(max(6, pop_size))
    _LAM_CAP = int(4 * lam_cma)                              # bound IPOP growth
    best_u = x0_u.copy(); best_sv = score_of(best_u)
    init_sv = best_sv
    # #5 adaptive ε: scale the early feasibility relaxation to the ACTUAL starting infeasibility
    # (feasible start → tiny; very infeasible → looser), instead of a fixed 0.15.
    _EPS0 = float(min(0.40, max(0.05, 0.6 * best_sv[1])))

    def _polish_genome(u0):
        """#2 memetic: SLSQP with analytic revenue + smooth-violation gradients (Nelder–Mead
        fallback). Returns u0 unless the TRUE post-cap feasibility-first score strictly improves."""
        if not polish or (stop_check is not None and stop_check()):
            return u0
        _pol = None
        try:
            from scipy.optimize import minimize as _minimize
            _gcache = {}

            def _gc(z):
                z = np.clip(np.asarray(z, float), 0.0, 1.0); _k = z.tobytes()
                _r = _gcache.get(_k)
                if _r is None:
                    _r = _grads(z)
                    if len(_gcache) > 96:
                        _gcache.clear()
                    _gcache[_k] = _r
                return _r

            _res = _minimize(lambda z: -_gc(z)[0], u0, jac=lambda z: -(_gc(z)[1]),
                             method="SLSQP", bounds=[(0.0, 1.0)] * D,
                             constraints=[{"type": "ineq", "fun": lambda z: _FEAS_TOL - _gc(z)[2],
                                           "jac": lambda z: -(_gc(z)[3])}],
                             options={"maxiter": 60, "ftol": 1e-9})
            _pol = np.clip(np.asarray(_res.x, float), 0.0, 1.0)
        except Exception:
            _pol = None
        if _pol is None:                                     # fallback: Nelder–Mead
            try:
                from scipy.optimize import minimize as _minimize

                def _sk(z):
                    _o, _v = score_of(np.clip(z, 0.0, 1.0))
                    return (-_o) if _v <= _FEAS_TOL else (1e12 + _v)

                _res = _minimize(_sk, u0, method="Nelder-Mead",
                                 options={"maxfev": int(min(40 * D, 1200)), "xatol": 1e-4,
                                          "fatol": 1e-6, "adaptive": True})
                _pol = np.clip(np.asarray(_res.x, float), 0.0, 1.0)
            except Exception:
                _pol = None
        return _pol if (_pol is not None and better(score_of(_pol), score_of(u0))) else u0

    # RESTART MODE (#7). "ipop" (default): each restart beyond the diverse seeds reseeds from the
    # incumbent and DOUBLES λ (capped at _LAM_CAP) — thorough on multimodal problems, but the λ
    # growth makes the restart budget super-linear. "lean": keep λ CONSTANT (no doubling) and
    # reseed each restart from the point FARTHEST in genome space from everything explored so far
    # (coordinated coverage), so a restart still escapes the current basin and lands on fresh
    # ground, at linear cost. Same benefit (don't get stuck / cover the space), far cheaper.
    _lean = str(restart_mode).lower() == "lean"

    def _farthest_start(rng_seed, explored, n_try=24):
        """A unit-box start point maximising the min distance to every already-explored genome
        (the diverse seeds + finished restart winners) — farthest-point sampling, so each lean
        restart deliberately probes the least-searched region instead of re-covering old ground."""
        _rng = np.random.default_rng(int(rng_seed))
        cand = _rng.random((int(n_try), D))
        if not explored:
            return cand[0]
        P = np.asarray(explored, float)                      # (m, D)
        d = np.min(np.linalg.norm(cand[:, None, :] - P[None, :, :], axis=2), axis=1)
        return cand[int(np.argmax(d))]

    n_r = max(max(1, int(n_restarts)), len(_seeds))          # #3 a restart per diverse seed + IPOP
    all_hist = []; restart_bests = []; best_fpop = (None, None); restart_lams = []
    for r in range(n_r):
        if stop_check is not None and stop_check():
            break
        if r < len(_seeds):                                  # #1 seed each diverse start
            start = _seeds[r].copy(); s0 = 0.30; _lam_r = lam_cma
        elif _lean:                                          # constant-λ + coordinated coverage
            _lam_r = lam_cma                                 # NO doubling → linear restart cost
            s0 = 0.45                                        # wide step to explore the fresh region
            _explored = [np.asarray(u, float) for u in _seeds] + \
                        [bx for (bx, _sv) in restart_bests]
            start = _farthest_start(int(seed) + 1000 + r, _explored)
        else:                                                # #3/#7 IPOP: reseed from incumbent, grow λ
            _e = r - len(_seeds) + 1
            jit = np.random.default_rng(int(seed) + 1000 + r).normal(0.0, 0.15, size=D)
            start = np.clip(best_u + jit, 0.0, 1.0)
            s0 = min(0.55, 0.30 * (1.4 ** _e))
            _lam_r = min(int(lam_cma * (2 ** _e)), _LAM_CAP)
        restart_lams.append(int(_lam_r))
        # UI step-size controls: σ₀ multiplier (wider/narrower starting stride), σ floor (don't let
        # the stride collapse to micro-steps), damping multiplier (adapt σ more slowly). All default
        # to no-op (1.0 / 0.0 / 1.0) so behaviour is unchanged unless dialled.
        _s0m = float(ctx.get("sigma0_mult", 1.0) or 1.0)
        bx, bk, bov, tr, fpop = _cmaes(eval_ov, start, s0 * _s0m, np.zeros(D), np.ones(D),
                                       popsize=_lam_r, max_iter=int(generations),
                                       seed=int(seed) + r, stop_check=stop_check,
                                       eps0=_EPS0,
                                       repair=(_repair if bool(ctx.get("repair_enabled", True)) else None),
                                       progress_cb=progress_cb,
                                       no_early_stop=bool(ctx.get("no_early_stop", False)),
                                       sigma_floor=float(ctx.get("sigma_floor", 0.0) or 0.0),
                                       damps_mult=float(ctx.get("damps_mult", 1.0) or 1.0))
        all_hist.append(tr)
        bx = _polish_genome(bx)                              # #2 polish EVERY restart's winner
        sv = score_of(bx)
        restart_bests.append((bx.copy(), sv))
        if best_fpop[0] is None:
            best_fpop = fpop
        if better(sv, best_sv):
            best_fpop = fpop                                 # final population of the winning restart
            best_sv = sv; best_u = bx.copy()

    best_G = _to_actual(best_u[None, :])
    best = _decode(best_G)[0]
    best_rev = float((best * rc).sum())
    best_obj, best_viol = best_sv
    # One-shot violation DECOMPOSITION of the winning candidate (diagnostic; off the hot path).
    try:
        _vbd = _violation_breakdown(best[None, :], ctx)
    except Exception:  # noqa: BLE001 - a diagnostic must NEVER break a run
        _vbd = None
    best_feas = best_viol <= _FEAS_TOL
    best_fit = best_obj if best_feas else (-1e15 - best_viol)
    init_fit = init_sv[0] if init_sv[1] <= _FEAS_TOL else (-1e15 - init_sv[1])

    # Per-generation trace for the UI charts, as 9-tuples:
    # (gen, best-so-far score, this-gen best score, this-gen mean score, sigma, violation, ε,
    #  candidates-so-far, incumbent FITNESS). The first five match the old convergence chart
    # (scores = maximised feasibility-aware −key, so best-so-far rises); violation/ε feed the σ and
    # violation→feasibility charts; the last is the incumbent split's revenue-ish objective so the
    # chart can show fitness alongside the ranked score.
    history = []; gi = 0; run_best = -np.inf; sigma_final = 0.0; cand_run = 0
    for _ri, tr in enumerate(all_hist):
        # candidates evaluated per generation of THIS restart = the ACTUAL λ used for it (recorded
        # in restart_lams: constant in lean mode, IPOP-doubled in ipop mode). Lets the UI plot
        # score vs. CANDIDATES-evaluated (true x-axis), not just generation index.
        _lam_r = restart_lams[_ri] if _ri < len(restart_lams) else lam_cma
        for _row in tr:
            gk, mk, sg, vv, ee = _row[:5]
            fo = float(_row[5]) if len(_row) > 5 else float("nan")   # incumbent fitness (back-compat)
            gi += 1
            cand_run += int(_lam_r)
            gen_best = -gk
            run_best = max(run_best, gen_best)
            sigma_final = float(sg)
            history.append((gi, float(run_best), float(gen_best), float(-mk), float(sg),
                            float(vv), float(ee), int(cand_run), float(fo)))

    # Diversity archive (improvement #7 output): the restart winners that are >= archive_min_dist
    # apart in genome space, best (feasibility-first) first, for warm-starting a later run.
    def _rank_key(sv):
        return (0 if sv[1] <= _FEAS_TOL else 1, -sv[0] if sv[1] <= _FEAS_TOL else sv[1])

    _ordered = sorted(restart_bests, key=lambda t: _rank_key(t[1]))
    _arch = [best_G[0].copy()]
    for gu, sv in _ordered:
        if len(_arch) >= int(archive_k):
            break
        ga = _to_actual(np.asarray(gu)[None, :])[0]
        if all(np.linalg.norm(ga - a) > float(archive_min_dist) for a in _arch):
            _arch.append(ga.copy())
    archive = np.asarray(_arch, dtype=float)

    info = {
        "gens": int(gi), "gens_max": int(generations) * n_r,
        "early_stopped": bool(gi < int(generations) * n_r),
        "sigma_final": float(sigma_final), "best_fit": float(best_fit),
        "init_fit": float(init_fit), "revenue": best_rev,
        "risk_cost": float(best_rev - best_fit), "dims": D,
        "feasible": bool(best_feas), "violation": float(best_viol),
        "genome": best_G[0].copy(),   # 3·n_mid -> warm_start on a subsequent run
        "archive": archive,           # (K, 3M) good-but-different genomes (best first)
        "history": history,           # per-generation scoring trace (for the UI charts)
        "viol_breakdown": _vbd,       # one-shot decomposition of the winning candidate's violation
        "pop_obj": best_fpop[0],      # winning restart's final-population objectives (revenue-ish)
        "pop_viol": best_fpop[1],     # winning restart's final-population violations
        "engine": "cmaes",            # marker so stale bytecode / old GA is obvious in logs
        "restart_mode": ("lean" if _lean else "ipop"),
        "restart_lams": list(restart_lams),   # actual λ per restart (constant in lean, doubled in ipop)
        "numba": ga_numba_info,       # verify-or-fallback result (used? diffs? timings?) for logs
    }
    return best, info
