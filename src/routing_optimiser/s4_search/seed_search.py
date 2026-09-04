"""Seed construction for the full-matrix GA — the BAND-AWARE warm start.

    genetic_global.py  ->  midtilt_cmaes.py  ->  seed_search.py   (19fl, 19fp)

WHAT THIS MODULE IS, as of 19gd. It builds the warm-start seed the full-matrix GA begins from,
and it holds the REFERENCE fitness definition. That is all. The Active CMA-ES tilt search that
used to live here — `run_midtilt_ga`, `_cmaes` and their ten exclusive helpers, 1,246 lines —
moved to `routing_optimiser.legacy_engines.midtilt_cmaes` because it is not reachable: tab 2
offers exactly one engine (`genetic_fullmatrix`) and that path skips the tilt search explicitly.

WHAT THE LIVE SEARCH ACTUALLY CALLS FROM HERE:

    band_greedy_shares_multi      seed stage 1/3, the band-aware constrained projection
                                  (per-profile simplex + max-share QP) -- the "band-aware seed"
                                  in the run log
    band_greedy_shares            its single-start inner routine
    _project_capped_simplex_profiles the vectorised per-profile capped-simplex QP both use
    _build_mid_incidence          the (vampMid x row) incidence matrix
    _obj_viol                     the REFERENCE fitness. `numba_kernels._fused_eval` and
                                  `band_scoring` are written to match it and `verify()`
                                  cross-checks against it, so this is the definition of record.
    _breach_pen, _mid_sums, _mid_viol_weights   the pieces _obj_viol is built from

The last five are ALSO imported by the retired engine. They are defined here once and imported
there — never copied — so the two cannot drift apart.

NAMING. `run_midtilt_ga` said "ga" but ran CMA-ES; it is gone from this file. Nothing left here
is named after an algorithm.
"""
from __future__ import annotations

import os as _os
import time as _time
import numpy as np

__build__ = "2026-08-11-band-aware-constrained-projection-seed+riskmin-diverse-seeds+eligibility-in-score+fixed-quadratic-breach+numba-eligibility-kernel+vol-weighted-viol+penalty-shape+repair-input+sigma-controls+fitness-trace+active-priority+viol-breakdown+capprofile-detail+breach-tol+band-workings+no-maxiter-bandgreedy+stable-softmax+bandgreedy-count-aware-priority+bandgreedy-multistart+2026-09-01-19go-delivery-faithful-bandgreedy"


# ── 19kg: SETTINGS THAT USED TO BE ENVIRONMENT SWITCHES ──────────────────
# No environment variable changes a run any more. Each name below is frozen at the
# value the shipped run already used - the defaults, because no routing.env exists and
# run.command exports nothing - so what shipped is what these say. They stay NAMES, not
# literals inlined at the use site, for two reasons: a test can still A/B a whole search
# by rebinding one, and a reader can see in one place every decision this module makes.
# Changing behaviour now means editing this block and saying so in a commit.
_SW_FEAS_PAR_WORKERS = 0   # was ROUTING_FEAS_PAR_WORKERS, default '0'

# [FN-103]
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


# [FN-104]
def _build_mid_incidence(mid_id, M, N):
    """Sparse (M, N) 0/1 incidence for `_mid_sums`' fast path: entry (m, n)=1 iff row n
    belongs to vampMid m. Built once per problem; CSR so `S @ vol.T` is a single BLAS-ish
    sparse-dense product. `mid_id` (N,) maps each row to its MID index in [0, M)."""
    from scipy import sparse as _sp
    mid_id = np.asarray(mid_id)
    cols = np.arange(N)
    data = np.ones(N, dtype=float)
    return _sp.csr_matrix((data, (mid_id, cols)), shape=(M, N))


















# [FN-115]
def _breach_pen(ov, bfix, qwt, pexp):
    """Constraint breach penalty: dust-guard (<1e-9 relative = compliant), a flat `bfix` hit the
    instant over, plus a smooth surcharge — quadratic (`qwt·ov²`) or, if `pexp`, `qwt·(exp(ov)−1)`
    (ov clipped at 50). SINGLE numpy definition shared by both `_obj_viol` `_pen` closures; the
    numba kernel (`numba_kernels._fused_eval`) and `band_scoring` keep byte-identical inlined
    mirrors because they can't call out of an @njit / method context — keep all in lock-step."""
    ov = np.asarray(ov, float)
    ov = np.where(ov > 1e-9, ov, 0.0)
    if pexp:
        return bfix * (ov > 0.0) + qwt * (np.exp(np.minimum(ov, 50.0)) - 1.0)
    return bfix * (ov > 0.0) + qwt * ov * ov


def _mid_viol_weights(ctx, M):
    """Per-MID VIOLATION weight (VOLUME-WEIGHTING of the feasibility violation).

    weight[m] = MID baseline volume / mean(baseline volume over MIDs with volume > 0), so the
    AVERAGE MID keeps weight ≈ 1 (the overall violation scale — and hence the feasibility
    tolerance and the breach_quad meaning — is preserved), while a high-volume MID's breach
    counts proportionally MORE than a tiny MID's. This stops the search wasting its gradient
    flattening hundreds of trivially-small breaches it cannot tell apart from the ones that
    matter (see the 5-txn cad profiles vs the 44,000-txn usd profiles in the run profiles).

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


# [FN-116]
def _obj_viol(shares, ctx):
    """Split the score into (objective, violation) for feasibility-first ranking.

    objective (P,) = revenue [− risk_min term], MAXIMISED among FEASIBLE splits.
    violation (P,) = summed breach penalty across the per-vampMid VAMP-rate cap, per-MID volume
        caps and projected bands. Each breaching term contributes a FIXED hit
        (ctx['breach_fixed'] — the UI 'Cap-breach penalty') the instant it goes over, PLUS a
        smooth term in the relative overage: QUADRATIC (ctx['breach_quad'], default 1.0) or, when
        ctx['breach_shape']=='exponential', qwt·(exp(overage)−1). Exactly 0 when compliant. NOTE: the fixed hit reintroduces a non-smooth step, so the memetic gradient
        polish (which follows the smooth-violation gradient) becomes less effective; set
        breach_fixed=0 to recover the pure smooth wall. Mirrors `_obj_viol` / the full-matrix
        engine AND the numba kernel (`numba_kernels._fused_eval`) — keep all three in lock-step."""
    _eop = ctx.get("elig_op")
    if _eop is not None:                                     # score the ACTUALLY-ROUTABLE shares —
        from routing_optimiser.s3_problem.eligibility import apply_elig_pop              # bans + wallet/USA capability folded in
        shares = apply_elig_pop(shares, _eop)                # so the search optimises what will route
    cv, risk, rc = ctx["profile_vol"], ctx["risk"], ctx["rev_coef"]
    mid_rows, M = ctx["mid_rows"], int(ctx["n_mid"])
    _S = ctx.get("_mid_S")                                   # precomputed incidence (fast path)
    P = shares.shape[0]
    revenue = (shares * rc[None, :]).sum(axis=1)
    obj = revenue.astype(float)   # astype(copy=True) already yields a fresh array — no extra .copy()
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

    # [FN-117]
    def _pen(_ov):                                           # relative overage (>= 0) -> penalty
        return _breach_pen(_ov, _bfix, _qwt, _pexp)          # shared def; numba kernel mirrors it
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
    if _floor > 0.0 and ctx.get("profile_starts") is not None:
        _cs = np.asarray(ctx["profile_starts"]); _cc = np.asarray(ctx["profile_counts"])
        _el = np.asarray(ctx["elig"], float)
        _nec = np.repeat(np.add.reduceat(_el, _cs), _cc)         # eligible gateways per profile
        _fl = np.minimum(_floor, np.where(_nec > 0, 1.0 / np.maximum(_nec, 1.0), 0.0))
        _mask = (_el > 0.5) & (_nec >= 2)                        # single-gateway profiles can't be floored
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











# 19ga: `_project_capped_simplex` DELETED — 23 lines, defined and never called. The live fitness is `_obj_viol` (which numba_kernels._fused_eval and band_scoring are written to match and verify() cross-checks).


# [FN-122c]
def _project_capped_simplex_profiles(s, profile_starts, profile_counts, elig, cap, total=1.0, iters=60):
    """VECTORISED `_project_capped_simplex` over ALL profiles at once (no Python per-profile loop).

    Projects each profile's ELIGIBLE entries onto {0 ≤ x ≤ cap, Σ = total}; ineligible rows → 0.
    Uses ONE vectorised bisection on a per-profile shift τ, with segment sums via ``reduceat`` —
    numerically identical to the closed-form per-profile QP (unit-tested). Falls back
    to a proportional renormalise for profiles the cap can't fill (e.g. a lone eligible gateway), and
    to a uniform split for a (degenerate) all-ineligible profile."""
    s = np.asarray(s, float)
    starts = np.asarray(profile_starts, np.intp)
    counts = np.asarray(profile_counts, np.intp)
    e = np.asarray(elig, float) > 0.5
    cap = float(cap) if (cap and float(cap) > 0) else 1.0
    y = np.where(e, s, -1e18)                       # ineligible → clip(y-τ,0,cap)=0 for any τ
    n_elig = np.add.reduceat(e.astype(float), starts)
    y_max = np.maximum.reduceat(y, starts)
    y_min = np.minimum.reduceat(np.where(e, s, 1e18), starts)
    lo = y_min - cap
    hi = np.where(n_elig > 0, y_max, 0.0)
    for _ in range(int(iters)):                      # per-profile bisection, fully vectorised
        tau = 0.5 * (lo + hi)
        seg = np.add.reduceat(np.clip(y - np.repeat(tau, counts), 0.0, cap), starts)
        over = seg > total
        lo = np.where(over, tau, lo)
        hi = np.where(over, hi, tau)
    x = np.clip(y - np.repeat(0.5 * (lo + hi), counts), 0.0, cap)
    # ---- fallbacks for profiles the bisection can't satisfy ----
    seg_sum = np.add.reduceat(np.where(e, s, 0.0), starts)          # eligible mass per profile
    infeas = (cap * n_elig <= total + 1e-12)                        # cap too tight to reach total
    if infeas.any():
        seg_row = np.repeat(np.where(seg_sum > 1e-12, seg_sum, 1.0), counts)
        prop = np.where(e, s / seg_row * total, 0.0)                # proportional over eligible
        uni_e = np.where(e, total / np.repeat(np.where(n_elig > 0, n_elig, 1.0), counts), 0.0)
        has_mass = np.repeat(seg_sum > 1e-12, counts)
        fb = np.where(has_mass, prop, uni_e)                        # uniform-over-eligible if no mass
        x = np.where(np.repeat(infeas, counts), fb, x)
    # degenerate all-ineligible profile → uniform over ALL rows (matches the scalar reference)
    none_e = n_elig <= 0
    if none_e.any():
        uni_all = np.repeat(total / np.maximum(counts.astype(float), 1.0), counts)
        x = np.where(np.repeat(none_e, counts), uni_all, x)
    else:
        x = np.where(e, x, 0.0)
    return x


# [FN-122b]
def band_greedy_shares(base_shares, profile_starts, profile_counts, elig, mid_rows, mid_labels,
                       exact_bands, incidence, *, max_share=1.0, damping=0.5,
                       tol=1e-6, patience=4, return_key=False, deliver_fn=None):
    """Band-AWARE compliant split via a small CONSTRAINED PROJECTION per pass: a warm-start seed
    for the CMA-ES that starts feasible (or much closer) w.r.t. the per-MID MONTH bands.

    Each pass: (1) project the current split through the SAME exact-band projector the GA scores
    with; (2) for every band, build a band-correcting target by scaling that MID's rows toward its
    violated ceiling (down) or floor (up), damped for stability; (3) project each profile's target
    back onto its capped simplex `{0 ≤ x ≤ max_share, Σ = 1}` over the ELIGIBLE rows — the small
    QP `min ‖x − target‖²` solved in closed form. Step (3) is what
    enforces the per-profile simplex AND the max-share cap exactly, every pass.

    STOPPING: there is NO fixed pass count — it keeps nudging until there is no meaningful
    improvement. Each pass it tracks the total RELATIVE band breach and stops on the first of:
    compliant (`breach ≤ tol`), nothing left to nudge, or the breach stops improving (>0.1%
    relative) for `patience` consecutive passes (a plateau ⇒ the targets can't all be met at
    once). A large internal absolute cap (`_HARD_CAP`) is retained purely as a defensive guard so
    a pathological non-terminating case can never hang the run; the convergence checks above are
    what stop it in practice — the cap is never expected to bind. Returns the LOWEST-breach split
    seen (never worse than the base), so a late oscillation can't hand back a worse result.

    Returns a valid shares vector (each profile sums to 1 over eligible rows, no share > max_share
    where the profile has ≥2 eligible gateways). Only ever HELPS — the GA ranks seeds feasibility-
    first, so a band-closer start can be adopted, a worse one ignored. Pure / deterministic,
    unit-tested off the live pipeline; the caller wraps it and falls back to the base split on any
    error.

    `deliver_fn` (19go): the DELIVERY transform (blocked-caps → eligibility → cap). When given,
    every pass reads the bands off `deliver_fn(s)` instead of `s` — the same numbers the engine
    selects with and delivery ships — so the multipliers correct the DELIVERED breach and the
    kept-split key ranks on it too. The split itself is still the RAW genome (the GA takes raw
    genomes and applies delivery itself); only the MEASUREMENT changes basis. None restores the
    pre-19go RAW behaviour byte for byte, which is what `_SW_SEED_DELIV = False` passes."""
    from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw
    s = np.asarray(base_shares, float).copy()
    elig = np.asarray(elig, float)
    starts = np.asarray(profile_starts, np.intp)
    counts = np.asarray(profile_counts, np.intp)
    _cap = float(max_share) if (max_share and float(max_share) > 0) else 1.0
    label_to_k = {}
    for k, lbl in enumerate(mid_labels):
        label_to_k.setdefault(str(lbl).strip().lower(), k)
    n_mid = len(mid_rows)
    best_s = s.copy()
    best_key = None            # (priority-weighted unmet-band count, total relative breach)
    # COUNT-AWARE + PRIORITY-WEIGHTED: the old version kept the LOWEST-total-breach split, which spreads a
    # tiny breach across many bands (min magnitude) instead of CLEARING bands (min count). We now keep the
    # split with the FEWEST priority-weighted unmet bands (ties broken by total breach), and take a FULL
    # (undamped) step on bands already within 5% of their limit so they actually cross the line — so the
    # verdict/seed line up with what the count-rewarding GA reaches, not a pessimistic magnitude bound.
    _prio_w = {}
    for _sp in (getattr(exact_bands, "specs", []) or []):
        _prio_w[str(getattr(_sp, "midl", "")).strip().lower()] = float(getattr(_sp, "weight", 1.0) or 1.0)
    _stall = 0
    _HARD_CAP = 10_000          # defensive only — convergence (below) is what actually stops it
    for _ in range(_HARD_CAP):
        # 1-D on purpose. `shares_to_prop_raw` promotes a single vector itself, and the delivery
        # transform's 1-D path is its SERIAL reference — the one the threaded path is verified
        # against. Handing it a (1, N) array instead would spin the row-parallel pool up once per
        # pass for a single row: pure overhead, and it would pad the [row-par] verification ledger
        # with hundreds of one-row calls that say nothing about threading.
        prop_raw = shares_to_prop_raw(
            s if deliver_fn is None else np.asarray(deliver_fn(s), float), incidence)
        rep = exact_bands.report(prop_raw)
        mult = np.ones(n_mid, float)
        full = np.zeros(n_mid, bool)                          # per-MID: near its limit → clear FULLY (undamped)
        moved = False
        breach = 0.0                                          # total RELATIVE band breach (0 = compliant)
        unmet_w = 0.0                                         # priority-weighted count of unmet bands
        for r in rep:
            _lbl = str(r["midl"]).strip().lower()
            k = label_to_k.get(_lbl)
            now = float(r["now"])
            _w = _prio_w.get(_lbl, 1.0)
            f = 1.0
            _over = 0.0                                       # how far this band is past its limit (relative)
            if r["ceil"] is not None and now > float(r["ceil"]) > 0.0:
                _over = now / float(r["ceil"]) - 1.0
                breach += _over
                unmet_w += _w
                f = min(f, float(r["ceil"]) / now)           # over a ceiling → shave down
            if r["floor"] is not None and 0.0 < now < float(r["floor"]):
                _und = 1.0 - now / float(r["floor"])
                breach += _und
                unmet_w += _w
                _over = max(_over, _und)
                f = max(f, float(r["floor"]) / now)          # under a floor → feed up
            if k is not None and abs(f - 1.0) > 1e-6:
                mult[k] *= f
                if 0.0 < _over <= 0.05:                       # cheap to clear → take the full step, cross the line
                    full[k] = True
                moved = True
        # Keep the split with the FEWEST priority-weighted unmet bands, ties broken by total breach (this
        # also hands the GA a better-count seed, not just a smaller-magnitude one). base is a candidate on
        # pass 1, so it's still 'never worse than base'.
        _key = (unmet_w, breach)
        if best_key is None or _key < best_key:
            best_key = _key
            best_s = s.copy()
            _stall = 0
        else:
            _stall += 1
        # STOP: nothing to nudge, essentially compliant, or no improvement for `patience` passes.
        if (not moved) or breach <= tol or _stall >= int(patience):
            break
        # Damped step for stability, but a FULL step for near-boundary bands so they actually clear.
        _exp = np.where(full, 1.0, float(damping))
        mult = np.power(np.clip(mult, 1e-6, 1e6), _exp)
        for k, rows in enumerate(mid_rows):
            if len(rows) and abs(mult[k] - 1.0) > 1e-12:
                s[rows] = s[rows] * mult[k]
        # Per-profile constrained projection (the small QP): back onto {0 ≤ x ≤ cap, Σ=1} over the
        # eligible rows, vectorised across ALL profiles at once. Ineligible rows are pinned at 0.
        s = _project_capped_simplex_profiles(s, starts, counts, elig, _cap, 1.0)
    # best_key is (priority-weighted unmet-band count, total relative breach) of the returned split.
    return (best_s, (best_key if best_key is not None else (0.0, 0.0))) if return_key else best_s


# [FN-122b]
def band_greedy_shares_multi(base_shares, profile_starts, profile_counts, elig, mid_rows, mid_labels,
                             exact_bands, incidence, *, max_share=1.0, damping=0.5, tol=1e-6,
                             patience=4, n_starts=1, rng_seed=0, jitter=0.5, keys_out=None,
                             par_info=None, deliver_fn=None):
    """MULTI-START `band_greedy_shares`: run the constrained projection from the base split PLUS
    (n_starts − 1) log-normally-jittered starts, and keep the one with the LOWEST
    (priority-weighted unmet-band count, total breach). A single projection is a fast (~seconds)
    greedy that lands in one corner; a few restarts make the feasibility verdict — and the seed the
    GA inherits — less dependent on that starting corner (it can clear a band the single pass left
    just over). Cheap vs the GA. Returns (best_shares, best_key). n_starts ≤ 1 ⇒ a single pass.

    `keys_out`: optional list. When supplied, EVERY start's key is appended as
    (start_index, unmet_weighted, breach) — start 0 is the un-jittered base. Added 2026-08-19z
    because the caller could not tell whether the jittered starts ever WON: only the best key was
    returned and tab2 discarded even that, so four runs of logs could not answer whether
    n_starts=4 bought anything over n_starts=1. Measurement only — it does not change which split
    is chosen, and an unsupplied keys_out leaves behaviour byte-identical.

    CONCURRENT since 2026-08-19ck, and BIT-IDENTICAL to the serial loop it replaced. The starts are
    independent — each is the same greedy on its own perturbed copy — so the 30.6 s this stage cost
    on the 2026-08-25 20:35 run was mostly waiting. Two things had to be pinned for identity:
    the RNG DRAWS are taken first, sequentially, in the old order and handed to the workers as data
    (they were drawn inside the loop, so execution order WAS draw order); and the REDUCE is still
    serial in index order with the same strict `<`, so a tie still goes to the earlier start.
    FIXED SERIAL since 2026-08-31: `_par = False` below, and the switch that used to set it
    (`ROUTING_FEAS_PAR`) was deleted with it - 19jy: this docstring still named it, which is the
    same defect 19ju fixed in three log lines. `par_info` (a dict, optional) receives the wall
    time, and since 19jy it is filled on the `n_starts <= 1` path too - that path returns before
    the timing at the bottom of the function, so at the shipped `n_starts=1` the caller got an
    EMPTY dict and both of tab_2's reports on this stage (`[feas-par]` and the WINNING SEED
    CHECKSUM) silently never printed."""
    _kw = dict(max_share=max_share, damping=damping, tol=tol, patience=patience,
               deliver_fn=deliver_fn)
    base = np.asarray(base_shares, float)
    _n = int(n_starts)

    def _greedy(_x):
        return band_greedy_shares(
            _x, profile_starts, profile_counts, elig, mid_rows, mid_labels, exact_bands, incidence,
            return_key=True, **_kw)

    if _n <= 1:
        # 19jy: TIME THIS PATH AND FILL `par_info`. It returns before the `_dt`
        # measurement below, so at the shipped `n_starts=1` the caller's
        # `par_info` dict stayed EMPTY - which made both of tab_2's reports on
        # this stage unreachable: `[feas-par]` (its wall time) and the WINNING
        # SEED CHECKSUM, the diagnostic written specifically to settle whether
        # this stage is deterministic after three same-input runs produced band
        # breach 0.7159 / 0.7157 / 0.7159. Neither has printed since n_starts
        # was fixed at 1, and the stage is the LARGEST unmeasured block in the
        # run: 47.0s of stage 4.1's 194.7s on 2026-09-04 11:47, sitting in a gap
        # between two log lines. Measurement only - `best_s`/`best_key` are the
        # same objects returned by the same call in the same order.
        _t1 = _time.perf_counter()
        best_s, best_key = _greedy(base)
        _dt1 = _time.perf_counter() - _t1
        if keys_out is not None:
            keys_out.append((0, float(best_key[0]), float(best_key[1])))
        if isinstance(par_info, dict):
            par_info.update(parallel=False, workers=1, starts=int(_n),
                            secs=float(_dt1))
        return best_s, best_key

    _pstarts = np.asarray(profile_starts, np.intp)
    _counts = np.asarray(profile_counts, np.intp)
    _cap = float(max_share) if (max_share and float(max_share) > 0) else 1.0
    _elig = np.asarray(elig, float)
    # ── the draws, SEQUENTIALLY, in the pre-19ck order ────────────────────────────────────────
    rng = np.random.default_rng(int(rng_seed))
    _inputs = [base]
    for _i in range(1, _n):
        # jitter the base multiplicatively (log-normal), then project onto each profile's capped simplex so
        # every start is a VALID split before the greedy runs.
        _pert = base * np.exp(rng.normal(0.0, float(jitter), size=base.shape))
        _inputs.append(_project_capped_simplex_profiles(_pert, _pstarts, _counts, _elig, _cap, 1.0))

    # FIXED OFF (2026-08-31). Was ROUTING_FEAS_PAR, set to 0 in routing.env on every run.
    # Also unreachable in practice now: the only caller fixes n_starts=1, which returns from the
    # `_n <= 1` branch above before ever getting here. Serial is the control path and the one the
    # concurrent version was proven bit-identical against, so it is the one that ships.
    _par = False
    _t0 = _time.perf_counter()
    if _par:
        from concurrent.futures import ThreadPoolExecutor
        # One worker per start, capped: more threads than starts buys nothing, and the greedy holds
        # the GIL through its per-spec Python loop so oversubscribing only adds contention.
        _nw = max(1, min(_n, _SW_FEAS_PAR_WORKERS or _n))
        with ThreadPoolExecutor(max_workers=_nw) as _ex:
            _out = list(_ex.map(_greedy, _inputs))
    else:
        _nw = 1
        _out = [_greedy(_x) for _x in _inputs]
    _dt = _time.perf_counter() - _t0

    # ── the reduce, SERIALLY, in index order, with the original strict `<` ────────────────────
    best_s, best_key = _out[0]
    if keys_out is not None:
        keys_out.append((0, float(best_key[0]), float(best_key[1])))
    for _i in range(1, _n):
        _s, _k = _out[_i]
        if keys_out is not None:
            keys_out.append((int(_i), float(_k[0]), float(_k[1])))
        if _k < best_key:
            best_key, best_s = _k, _s
    if isinstance(par_info, dict):
        par_info.update(parallel=bool(_par), workers=int(_nw), starts=int(_n), secs=float(_dt))
    return best_s, best_key


