"""Numba-accelerated fused decode+objective for the cross-cell tilt GA (OPT-IN).

This module powers the **"GA - Numba"** engine ONLY. The production "Genetic algorithm"
engine never imports or touches it. Importing this file has no side effects and never
raises, even when Numba is not installed.

Why it exists
-------------
The per-generation hot loop is `_obj_viol(_decode(genome))` in `genetic_global.py`. In
NumPy that materialises several (population × gateways) intermediates per generation
(the exp array, the per-cell reduceat sums, the water-fill temporaries, the per-MID
sparse sums). This kernel FUSES the whole decode+objective into a single pass over one
candidate at a time, so the big intermediates never exist — that (not a smaller float)
is where the speed-up comes from. It keeps full **float64** maths and sums in the SAME
ORDER as the NumPy path, so results are identical to float64 rounding — unlike float32,
which would perturb the chaotic CMA-ES trajectory into a different answer.

Safety contract (verify-or-fallback)
-------------------------------------
`make_numba_eval(...)` builds the fast callable. `verify(...)` runs BOTH the Numba kernel
and the existing NumPy `_decode_midtilt3`/`_obj_viol` on the same sample of genomes and
compares objective+violation. The caller only trusts Numba if they match within tolerance;
otherwise it logs the discrepancy and falls back to the normal NumPy GA. Because the fast
path is gated on that check, a wrong or non-compiling kernel can NEVER produce a bad split
— the worst case is "GA - Numba" behaving exactly like the ordinary Genetic engine.

Build marker is logged by the caller so stale bytecode is obvious.
"""
from __future__ import annotations

import os
import time

import numpy as np

__build__ = "2026-07-31-ga-numba-persistent-cache+precompile+fixed-quadratic-breach+eligibility-in-kernel+step-aware-verify+vol-weighted-viol+penalty-shape+active-priority+breach-tol"

# PERSISTENT compile cache — set BEFORE numba is imported (this module is the ONLY importer
# of numba in the project, so setting it here wins). Numba's default cache lives inside a
# __pycache__ folder next to the module, which the project's routine "clear __pycache__ before
# every run" would delete — forcing a cold ~minutes-long recompile every single run. Pointing
# the cache at a stable sibling folder (NOT named __pycache__) means it SURVIVES those clears,
# so the kernel compiles once (per code/version change) and every later run loads it instantly.
# setdefault so an explicit user NUMBA_CACHE_DIR (e.g. set by app/streamlit_app.py) still wins.
# CRITICAL: use the LOCAL OS temp dir, NOT a folder inside the project — the project can live
# under a cloud-synced / FUSE mount (Downloads, iCloud/Dropbox/OneDrive), where Numba's file-lock
# + mmap cache load HANGS the parallel workers (they launch, then wedge with no progress, and
# `.fuse_hidden*` files accumulate). Temp is a real local disk, never synced; it survives the
# routine "clear __pycache__" and only a reboot clears it (one ~90s cold recompile after).
import tempfile as _tempfile
_NB_CACHE_DIR = os.path.join(_tempfile.gettempdir(), "routing_optimiser_numba_cache")
try:
    os.makedirs(_NB_CACHE_DIR, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", _NB_CACHE_DIR)
except Exception:  # noqa: BLE001 - a read-only dir must never stop the engine loading
    pass

# --------------------------------------------------------------------------- numba guard
try:                                            # Numba is optional; absent -> engine falls back.
    from numba import njit as _njit             # type: ignore
    NUMBA_OK = True

    # [FN-183]
    def njit(*a, **k):                          # force our safe defaults (no fastmath: keep IEEE)
        k.setdefault("cache", True)
        k.setdefault("fastmath", False)
        k.setdefault("nogil", True)
        if a and callable(a[0]):
            return _njit(**k)(a[0])
        return _njit(*a, **k)
except Exception:                               # noqa: BLE001 - no numba -> identity decorator
    NUMBA_OK = False

    # [FN-184]
    def njit(*a, **k):                          # type: ignore
        if a and callable(a[0]):
            return a[0]

        # [FN-185]
        def deco(f):
            return f
        return deco


# --------------------------------------------------------------------------- fused kernel
# [FN-186]
@njit
def _fused_eval(G, M, ref, zr, zq, mid_id, cs, cc, elig, fine_idx, zr_cell, n_fine,
                nec_col, fl_col, capN_col, has_floor, has_cap,
                cv, risk, rc,
                has_vcap, vcap,
                has_volcap, volcap,
                n_bands, b_mi, b_bval, b_ceil, b_floor, b_has_ceil, b_has_floor, b_pmul,
                has_base, base_vol, wm,
                max_share, floor_val,
                rmw, has_vfr, vfr, bfix, qwt, pexp,
                has_elig, ecs, ecc,
                e_has_ban, e_ban,
                e_has_w, e_w_incap, e_w_wf,
                e_has_u, e_u_incap, e_u_wf):
    """One fused pass: ACTUAL genome batch G (P, 3M[+K]) -> (obj (P,), viol (P,)).

    Mirrors `_decode_midtilt3` (softmax tilt -> per-cell renorm -> floor-then-cap water-fill),
    then — when `has_elig` — `eligibility.apply_elig_pop` (bans->0+renorm, wallet blend+renorm,
    USA blend+renorm, IN THAT ORDER, using the operator's OWN cell segments ecs/ecc), then
    `_obj_viol` (revenue [- risk-min], VAMP-rate / volume / band / cap / floor violations),
    summing in the same index order as the NumPy versions so the two agree to float64 rounding.
    The eligibility stage is what lets the Numba engine STAY ON when ctx['elig_op'] is active
    (previously it was force-disabled because the kernel scored the un-restricted split).

    ANALOGY for "fused": instead of building each intermediate array and handing it to the next
    NumPy step (like shipping half-finished parts between factory stations), this does the whole
    decode → eligibility → score on ONE workbench per candidate. The bulky in-between arrays never
    exist — which is where the speed comes from — while the maths and the summation ORDER stay the
    same, so the answer matches the NumPy path to float64 rounding.
    """
    P = G.shape[0]
    N = ref.shape[0]
    C = cs.shape[0]
    obj = np.empty(P, dtype=np.float64)
    viol = np.empty(P, dtype=np.float64)
    Xout = np.empty((P, N), dtype=np.float64)   # PRE-eligibility decoded shares (for exact bands)
    w = np.empty(N, dtype=np.float64)
    X = np.empty(N, dtype=np.float64)
    midv = np.empty(M, dtype=np.float64)
    midvr = np.empty(M, dtype=np.float64)

    for p in range(P):
        # ---- decode: softmax tilt weights on eligible columns -----------------------
        for g in range(N):
            if elig[g] > 0.5:
                m = mid_id[g]
                a = -G[p, m] * zr[g] + G[p, M + m] * zq[g] + G[p, 2 * M + m]
                if n_fine > 0 and fine_idx[g] >= 0:
                    a -= G[p, 3 * M + fine_idx[g]] * zr_cell[g]
                w[g] = ref[g] * np.exp(a) * elig[g]
            else:
                w[g] = 0.0
        # ---- per-cell renormalise ---------------------------------------------------
        for c in range(C):
            s0 = cs[c]
            s1 = s0 + cc[c]
            seg = 0.0
            for g in range(s0, s1):
                seg += w[g]
            if seg <= 1e-12:
                seg = 1.0
            for g in range(s0, s1):
                X[g] = w[g] / seg
        # ---- HARD exploration-floor water-fill (lift-then-take, up to 50 sweeps) -----
        # ANALOGY: like levelling water between connected tanks in a cell — top up any gateway
        # below its floor, and skim that top-up proportionally off the gateways above the floor;
        # repeat until nobody is under (or 50 sweeps).
        if has_floor:
            for _it in range(50):
                any_under = False
                for c in range(C):
                    s0 = cs[c]
                    s1 = s0 + cc[c]
                    deficit = 0.0
                    give_cell = 0.0
                    for g in range(s0, s1):
                        if elig[g] > 0.5 and X[g] < fl_col[g] - 1e-12:
                            deficit += fl_col[g] - X[g]
                            any_under = True
                        elif elig[g] > 0.5 and X[g] > fl_col[g] + 1e-12:
                            give_cell += X[g] - fl_col[g]
                    if deficit > 0.0:
                        for g in range(s0, s1):
                            if elig[g] > 0.5 and X[g] < fl_col[g] - 1e-12:
                                X[g] = fl_col[g]
                            elif (elig[g] > 0.5 and X[g] > fl_col[g] + 1e-12
                                  and give_cell > 1e-12):
                                X[g] = X[g] - (X[g] - fl_col[g]) * deficit / give_cell
                if not any_under:
                    break
        # ---- HARD max-share cap water-fill (shed-then-fill, up to 50 sweeps) ---------
        # ANALOGY: the mirror image of the floor step — shed volume from any gateway over its cap
        # and pour it proportionally into the ones with headroom; repeat until nobody is over.
        if has_cap:
            for _it in range(50):
                any_over = False
                for c in range(C):
                    s0 = cs[c]
                    s1 = s0 + cc[c]
                    excess = 0.0
                    room_cell = 0.0
                    for g in range(s0, s1):
                        if X[g] > capN_col[g] + 1e-12:
                            excess += X[g] - capN_col[g]
                            any_over = True
                        elif elig[g] > 0.5 and X[g] < capN_col[g] - 1e-12:
                            room_cell += capN_col[g] - X[g]
                    if excess > 0.0:
                        for g in range(s0, s1):
                            if X[g] > capN_col[g] + 1e-12:
                                X[g] = capN_col[g]
                            elif (elig[g] > 0.5 and X[g] < capN_col[g] - 1e-12
                                  and room_cell > 1e-12):
                                X[g] = X[g] + (capN_col[g] - X[g]) * excess / room_cell
                if not any_over:
                    break
        # snapshot the PRE-ELIGIBILITY decode (this is what the exact-band projection consumes —
        # it matches NumPy _decode / _prop_items_from_gran). Taken BEFORE the in-place eligibility
        # stage below so the returned shares are the raw routed split, not the masked one.
        for g in range(N):
            Xout[p, g] = X[g]
        # ---- eligibility on the DECODED shares (mirror eligibility.apply_elig_pop) ----
        # bans -> 0 + per-cell renorm, then wallet blend + renorm, then USA blend + renorm,
        # IN THIS ORDER — so the kernel scores the SAME actually-routable split the NumPy
        # `_obj_viol` does when ctx['elig_op'] is active. Uses the operator's OWN segments
        # (ecs/ecc), exactly as apply_elig_pop / _renorm_pop / _blend_pop do.
        if has_elig == 1:
            Ce = ecs.shape[0]
            # (1) hard bans -> share 0, then renormalise within each routing group
            if e_has_ban == 1:
                for g in range(N):
                    if e_ban[g] > 0.5:
                        X[g] = 0.0
                for c in range(Ce):
                    s0 = ecs[c]
                    s1 = s0 + ecc[c]
                    seg = 0.0
                    for g in range(s0, s1):
                        seg += X[g]
                    if seg > 0.0:                       # matches _renorm_pop: where(s>0, X/s, X)
                        for g in range(s0, s1):
                            X[g] = X[g] / seg
            # (2) wallet capability blend: incapable keeps (1-wf) of its share; wf portion
            #     redistributes to the capable gateways in the cell, then per-cell renorm.
            if e_has_w == 1:
                for c in range(Ce):
                    s0 = ecs[c]
                    s1 = s0 + ecc[c]
                    base_sum = 0.0
                    s_cap = 0.0
                    for g in range(s0, s1):
                        base_sum += X[g]
                        if e_w_incap[g] < 0.5:
                            s_cap += X[g]
                    for g in range(s0, s1):
                        base_g = X[g]
                        if s_cap > 0.0:
                            cap_g = base_g if e_w_incap[g] < 0.5 else 0.0
                            cshare_g = cap_g / s_cap
                        else:
                            cshare_g = base_g          # only incapable present -> no reroute
                        blended_g = e_w_wf[g] * cshare_g + (1.0 - e_w_wf[g]) * base_g
                        X[g] = blended_g if base_sum > 0.0 else base_g
                    seg = 0.0
                    for g in range(s0, s1):
                        seg += X[g]
                    if seg > 0.0:
                        for g in range(s0, s1):
                            X[g] = X[g] / seg
            # (3) USA-only capability blend — identical mechanism, Non-USA fraction rerouted.
            if e_has_u == 1:
                for c in range(Ce):
                    s0 = ecs[c]
                    s1 = s0 + ecc[c]
                    base_sum = 0.0
                    s_cap = 0.0
                    for g in range(s0, s1):
                        base_sum += X[g]
                        if e_u_incap[g] < 0.5:
                            s_cap += X[g]
                    for g in range(s0, s1):
                        base_g = X[g]
                        if s_cap > 0.0:
                            cap_g = base_g if e_u_incap[g] < 0.5 else 0.0
                            cshare_g = cap_g / s_cap
                        else:
                            cshare_g = base_g
                        blended_g = e_u_wf[g] * cshare_g + (1.0 - e_u_wf[g]) * base_g
                        X[g] = blended_g if base_sum > 0.0 else base_g
                    seg = 0.0
                    for g in range(s0, s1):
                        seg += X[g]
                    if seg > 0.0:
                        for g in range(s0, s1):
                            X[g] = X[g] / seg
        # ---- objective: revenue -----------------------------------------------------
        o = 0.0
        for g in range(N):
            o += X[g] * rc[g]
        v = 0.0
        # ---- per-MID aggregates (only when a per-MID constraint is active) -----------
        need_mid = (has_vcap == 1) or (has_volcap == 1) or (n_bands > 0)
        if need_mid:
            for m in range(M):
                midv[m] = 0.0
                midvr[m] = 0.0
            for g in range(N):
                vg = X[g] * cv[g]
                mm = mid_id[g]
                midv[mm] += vg
                midvr[mm] += vg * risk[g]
            if has_vcap == 1:
                denom = vcap if vcap > 1e-9 else 1e-9
                for m in range(M):
                    rate = midvr[m] / midv[m] if midv[m] > 1e-12 else 0.0
                    t = rate / denom - 1.0
                    if t > 1e-9:                         # TOLERANCE: within 1e-9 of the cap = compliant
                        v += wm[m] * (bfix + (qwt * t * t if pexp == 0 else qwt * (np.exp(t if t < 50.0 else 50.0) - 1.0)))
            if has_volcap == 1:
                for m in range(M):
                    t = midv[m] / volcap[m] - 1.0     # volcap has +inf sentinel for <=0 caps
                    if t > 1e-9:
                        v += wm[m] * (bfix + (qwt * t * t if pexp == 0 else qwt * (np.exp(t if t < 50.0 else 50.0) - 1.0)))
            if n_bands > 0:
                for b in range(n_bands):
                    m = b_mi[b]
                    if has_base == 1:
                        fmid = midv[m] / base_vol[m] if base_vol[m] > 1e-12 else 1.0
                    else:
                        fmid = 1.0
                    proj = fmid * b_bval[b]
                    pw = b_pmul[b]                       # PRIORITY weight (5000^(1-p)); low priority yields
                    if b_has_ceil[b] == 1:
                        cc_ = b_ceil[b] if b_ceil[b] > 1e-9 else 1e-9
                        t = proj / cc_ - 1.0
                        if t > 1e-9:
                            v += wm[m] * pw * (bfix + (qwt * t * t if pexp == 0 else qwt * (np.exp(t if t < 50.0 else 50.0) - 1.0)))
                    if b_has_floor[b] == 1 and b_floor[b] > 0.0:
                        ff_ = b_floor[b] if b_floor[b] > 1e-9 else 1e-9
                        t = 1.0 - proj / ff_
                        if t > 1e-9:
                            v += wm[m] * pw * (bfix + (qwt * t * t if pexp == 0 else qwt * (np.exp(t if t < 50.0 else 50.0) - 1.0)))
        # ---- structural safety nets (cap + floor), matching _obj_viol ----------------
        if max_share < 1.0:
            denom = max_share if max_share > 1e-9 else 1e-9
            for g in range(N):
                ov = (X[g] - max_share) / denom
                if ov > 1e-9:                            # TOLERANCE: at the cap (rounding dust) = compliant
                    v += bfix + (qwt * ov * ov if pexp == 0 else qwt * (np.exp(ov if ov < 50.0 else 50.0) - 1.0))
        if floor_val > 0.0:
            denom = floor_val if floor_val > 1e-9 else 1e-9
            for g in range(N):
                if elig[g] > 0.5 and nec_col[g] >= 2.0:
                    ov = (fl_col[g] - X[g]) / denom
                    if ov > 1e-9:
                        v += bfix + (qwt * ov * ov if pexp == 0 else qwt * (np.exp(ov if ov < 50.0 else 50.0) - 1.0))
        # ---- risk-minimisation secondary objective ----------------------------------
        if rmw > 0.0:
            if has_vfr == 1:
                if not need_mid:                       # midvr not built above -> build it now
                    for m in range(M):
                        midvr[m] = 0.0
                    for g in range(N):
                        midvr[mid_id[g]] += X[g] * cv[g] * risk[g]
                s = 0.0
                for m in range(M):
                    ex = midvr[m] - vfr[m]
                    if ex > 0.0:
                        s += ex
                o -= rmw * s
            else:
                tot = 0.0
                for g in range(N):
                    tot += X[g] * cv[g] * risk[g]
                o -= rmw * tot
        obj[p] = o
        viol[p] = v
    return obj, viol, Xout


# --------------------------------------------------------------------------- builder
# [FN-187]
def _prep_cols(cell_starts, cell_counts, elig, cap, floor):
    """Per-column nec / floor / capN constants, matching `genetic_global._cap_floor_prep`
    but as dense (N,) arrays the fused kernel can index directly."""
    cs = np.ascontiguousarray(cell_starts, dtype=np.intp)
    cc = np.ascontiguousarray(cell_counts, dtype=np.intp)
    elig = np.ascontiguousarray(elig, dtype=np.float64)
    nec_cell = np.add.reduceat(elig, cs)
    nec_col = np.repeat(nec_cell, cc).astype(np.float64)
    fl_col = np.minimum(float(floor),
                        np.where(nec_col > 0, 1.0 / np.maximum(nec_col, 1.0), 0.0)).astype(np.float64)
    capN_col = np.where(nec_col >= 2.0, float(cap), 1.0).astype(np.float64)
    return nec_col, fl_col, capN_col


# [FN-188]
def make_numba_eval(M, ref, zr, zq, mid_id, cell_starts, cell_counts, elig,
                    cap, floor, fine_idx, zr_cell, n_fine, cv, risk, rc, ctx):
    """Return a callable `eval_actual(G)->(obj, viol)` (G in ACTUAL genome space) backed by
    the fused Numba kernel. All constants are captured once. Raises if Numba is unavailable
    (caller guards on NUMBA_OK)."""
    if not NUMBA_OK:
        raise RuntimeError("numba not available")
    M = int(M)
    N = int(ref.shape[0])
    _c = lambda a, dt=np.float64: np.ascontiguousarray(a, dtype=dt)
    ref = _c(ref); zr = _c(zr); zq = _c(zq)
    cv = _c(cv); risk = _c(risk); rc = _c(rc)
    elig = _c(elig)
    mid_id = _c(mid_id, np.intp)
    cs = _c(cell_starts, np.intp); cc = _c(cell_counts, np.intp)
    n_fine = int(n_fine)
    fine_idx = _c(fine_idx, np.intp) if (n_fine > 0 and fine_idx is not None) else np.full(N, -1, np.intp)
    zr_cell = _c(zr_cell) if zr_cell is not None else np.zeros(N, np.float64)

    cap = float(cap); floor = float(floor)
    nec_col, fl_col, capN_col = _prep_cols(cs, cc, elig, cap, floor)
    has_floor = 1 if floor > 0.0 else 0
    has_cap = 1 if cap < 1.0 else 0

    # ---- objective flags/arrays derived from ctx (mirrors _obj_viol's optional terms) ----
    _vcap = ctx.get("vamp_cap")
    has_vcap = 1 if _vcap is not None else 0
    vcap = float(_vcap) if _vcap is not None else 0.0

    _volcap = ctx.get("mid_vol_cap")
    has_volcap = 1 if _volcap is not None else 0
    if _volcap is not None:
        _vc = np.asarray(_volcap, dtype=np.float64)
        volcap = np.where(_vc > 0, _vc, np.inf).astype(np.float64)
    else:
        volcap = np.full(M, np.inf, np.float64)

    # EXACT BANDS (gate 2): when ctx['exact_bands'] is set, the per-MID month bands are scored
    # EXACTLY per generation OUTSIDE the kernel (genetic_global eval wrapper + band_scoring), so
    # the kernel must NOT also apply the volume-ratio PROXY band term — otherwise it'd double-count
    # and break lockstep with _obj_viol (which drops the proxy term under the same flag).
    _bands = [] if ctx.get("exact_bands") else (ctx.get("midband") or [])
    n_bands = int(len(_bands))
    b_mi = np.zeros(max(n_bands, 1), np.intp)
    b_bval = np.zeros(max(n_bands, 1), np.float64)
    b_ceil = np.zeros(max(n_bands, 1), np.float64)
    b_floor = np.zeros(max(n_bands, 1), np.float64)
    b_has_ceil = np.zeros(max(n_bands, 1), np.intp)
    b_has_floor = np.zeros(max(n_bands, 1), np.intp)
    # PRIORITY weight per band (tuple slot 5, default 1.0). Multiplies the whole band penalty so
    # lower-priority (higher-number) constraints yield first — SAME semantics the NumPy _obj_viol
    # now applies, so verify() stays in lockstep. 1.0 for every band ⇒ byte-identical to before.
    b_pmul = np.ones(max(n_bands, 1), np.float64)
    for _i, _b in enumerate(_bands):
        b_mi[_i] = int(_b[0]); b_bval[_i] = float(_b[1])
        if _b[2] is not None:
            b_has_ceil[_i] = 1; b_ceil[_i] = float(_b[2])
        if _b[3] is not None:
            b_has_floor[_i] = 1; b_floor[_i] = float(_b[3])
        if len(_b) > 5:
            b_pmul[_i] = float(_b[5])

    _base = ctx.get("mid_base_vol")
    has_base = 1 if _base is not None else 0
    base_vol = np.asarray(_base, np.float64) if _base is not None else np.zeros(M, np.float64)
    # Per-MID VIOLATION weight (volume-weighting, #4). SAME helper the NumPy _obj_viol uses, so
    # the two hard-verified paths stay bit-identical. All-ones ⇒ un-weighted (back-compat).
    from .genetic_global import _mid_viol_weights
    wm = np.ascontiguousarray(_mid_viol_weights(ctx, M), dtype=np.float64)

    max_share = float(ctx.get("max_share", 1.0) or 1.0)
    floor_val = float(ctx.get("floor", 0.0) or 0.0)
    rmw = float(ctx.get("risk_min_w", 0.0) or 0.0)
    _vfr = ctx.get("vamp_floor_route")
    has_vfr = 1 if _vfr is not None else 0
    vfr = np.asarray(_vfr, np.float64) if _vfr is not None else np.zeros(M, np.float64)
    # Breach penalty (must match genetic_global._obj_viol._pen): fixed hit + quadratic overage.
    bfix = float(ctx.get("breach_fixed", 0.0) or 0.0)
    qwt = float(ctx.get("breach_quad", 1.0) or 1.0)
    # penalty shape: 0 = quadratic (qwt·over²), 1 = exponential (qwt·(exp(over)−1), over clipped 50).
    # MUST match genetic_global._obj_viol._pen — verify() cross-checks and falls back on mismatch.
    pexp = 1 if str(ctx.get("breach_shape", "quadratic")).lower() == "exponential" else 0

    # ---- eligibility operator (mirror eligibility.apply_elig_pop inside the kernel) -------
    # When ctx['elig_op'] is present the NumPy fitness scores the eligibility-adjusted split
    # (bans zeroed + renormalised, wallet/USA capability blended). Feed the SAME static arrays
    # to the kernel so it scores the identical split — this is what keeps Numba usable with
    # eligibility-in-scoring on (verify() still cross-checks, so a mismatch just falls back).
    _bool = lambda a: _c(np.asarray(a, dtype=bool).astype(np.float64))   # 0/1 float for the kernel
    _z = lambda: np.zeros(N, np.float64)
    _eop = ctx.get("elig_op")
    if _eop is not None:
        has_elig = 1
        ecs = _c(_eop["cell_starts"], np.intp)
        ecc = _c(_eop["cell_counts"], np.intp)
        e_has_ban = 1 if _eop.get("has_ban") else 0
        e_ban = _bool(_eop["ban"]) if e_has_ban else _z()
        e_has_w = 1 if _eop.get("has_w") else 0
        e_w_incap = _bool(_eop["w_incap"]) if e_has_w else _z()
        e_w_wf = _c(_eop["w_wf"]) if e_has_w else _z()
        e_has_u = 1 if _eop.get("has_u") else 0
        e_u_incap = _bool(_eop["u_incap"]) if e_has_u else _z()
        e_u_wf = _c(_eop["u_wf"]) if e_has_u else _z()
    else:
        has_elig = 0
        ecs = np.zeros(1, np.intp); ecc = np.zeros(1, np.intp)
        e_has_ban = 0; e_ban = _z()
        e_has_w = 0; e_w_incap = _z(); e_w_wf = _z()
        e_has_u = 0; e_u_incap = _z(); e_u_wf = _z()

    # [FN-189]
    def eval_actual(G):
        G = np.ascontiguousarray(G, dtype=np.float64)
        return _fused_eval(G, M, ref, zr, zq, mid_id, cs, cc, elig, fine_idx, zr_cell, n_fine,
                           nec_col, fl_col, capN_col, has_floor, has_cap,
                           cv, risk, rc,
                           has_vcap, vcap,
                           has_volcap, volcap,
                           n_bands, b_mi, b_bval, b_ceil, b_floor, b_has_ceil, b_has_floor, b_pmul,
                           has_base, base_vol, wm,
                           max_share, floor_val,
                           rmw, has_vfr, vfr, bfix, qwt, pexp,
                           has_elig, ecs, ecc,
                           e_has_ban, e_ban,
                           e_has_w, e_w_incap, e_w_wf,
                           e_has_u, e_u_incap, e_u_wf)

    return eval_actual


# --------------------------------------------------------------------------- verifier
# [FN-190]
def verify(np_eval_actual, nb_eval_actual, sample_G, *, rtol_obj=1e-7, atol_viol=1e-7,
           rtol_viol=1e-6, bfix=0.0, warmup=True):
    """Run NumPy and Numba evals on the SAME actual-space genomes and compare. Returns a dict
    the caller logs verbatim for cross-validation:

        used-decision inputs: ok (bool), reason (str)
        accuracy: max_abs_obj, max_rel_obj, max_abs_viol, max_rel_viol, max_smooth_viol, n_step_flips
        timing:   t_np, t_nb, speedup, n_sample  (t_nb EXCLUDES the one-off JIT compile)

    OBJECTIVE gate: relative, tight (`max_rel_obj <= rtol_obj`) — catches any real decode /
    revenue / eligibility divergence (a genuine bug moves revenue).

    VIOLATION gate: the fixed breach penalty (`bfix`) is a STEP — the instant a per-MID VAMP
    rate / band crosses its threshold the violation jumps by a whole `bfix`. Two algebraically
    IDENTICAL code paths (NumPy vs kernel) can legitimately straddle such a threshold by pure
    float64 rounding, so their violations differ by whole multiples of `bfix` (a knife-edge
    feasibility flip, irrelevant to CMA-ES ranking — NOT a bug). So when `bfix>0` we remove the
    whole-`bfix` steps and require only the SMOOTH remainder to agree within `atol_viol +
    rtol_viol*|v_np|`; a genuine error (missing/incorrect term) leaves a large non-step residual
    and still fails. With `bfix==0` (pure smooth penalty) this reduces to a plain allclose gate.

    Never raises: any error is caught and returned as ok=False with the reason.
    """
    out = {"ok": False, "reason": "", "n_sample": int(np.asarray(sample_G).shape[0]),
           "max_abs_obj": float("nan"), "max_rel_obj": float("nan"),
           "max_abs_viol": float("nan"), "max_rel_viol": float("nan"),
           "max_smooth_viol": float("nan"), "n_step_flips": 0,
           "t_np": float("nan"), "t_nb": float("nan"), "speedup": float("nan"),
           "compile_s": float("nan")}
    try:
        G = np.ascontiguousarray(sample_G, dtype=np.float64)
        # NumPy reference
        _t = time.perf_counter()
        o_np, v_np = np_eval_actual(G)
        out["t_np"] = time.perf_counter() - _t
        o_np = np.asarray(o_np, float); v_np = np.asarray(v_np, float)
        # Numba: first call pays JIT compile -> measure it separately, then time the hot call.
        if warmup:
            _t = time.perf_counter()
            nb_eval_actual(G[:1])
            out["compile_s"] = time.perf_counter() - _t
        _t = time.perf_counter()
        o_nb, v_nb = nb_eval_actual(G)[:2]   # kernel returns (obj, viol, Xdec); Xdec unused here
        out["t_nb"] = time.perf_counter() - _t
        o_nb = np.asarray(o_nb, float); v_nb = np.asarray(v_nb, float)

        if o_nb.shape != o_np.shape or v_nb.shape != v_np.shape:
            out["reason"] = f"shape mismatch obj{o_nb.shape}vs{o_np.shape} viol{v_nb.shape}vs{v_np.shape}"
            return out
        if not (np.all(np.isfinite(o_nb)) and np.all(np.isfinite(v_nb))):
            out["reason"] = "numba produced non-finite values"
            return out

        da = np.abs(o_nb - o_np)
        out["max_abs_obj"] = float(da.max())
        out["max_rel_obj"] = float((da / (np.abs(o_np) + 1e-9)).max())
        dv = np.abs(v_nb - v_np)
        out["max_abs_viol"] = float(dv.max())
        out["max_rel_viol"] = float((dv / (np.abs(v_np) + 1e-9)).max())
        if out["t_nb"] > 0:
            out["speedup"] = float(out["t_np"] / out["t_nb"])

        # VIOLATION: strip whole fixed-breach STEPS (multiples of bfix) — those are knife-edge
        # threshold flips, not bugs — and require the SMOOTH remainder to agree (allclose-style).
        if bfix and bfix > 0.0:
            steps = np.round(dv / bfix)                       # whole breach flips per genome
            resid = np.abs(dv - steps * bfix)                 # the continuous-part disagreement
            out["n_step_flips"] = int(steps.sum())
            out["max_smooth_viol"] = float(resid.max())
            viol_ok = bool(np.all(resid <= atol_viol + rtol_viol * np.abs(v_np)))
        else:
            out["max_smooth_viol"] = out["max_abs_viol"]
            viol_ok = bool(np.all(dv <= atol_viol + rtol_viol * np.abs(v_np)))

        ok = (out["max_rel_obj"] <= rtol_obj) and viol_ok
        out["ok"] = bool(ok)
        if ok:
            out["reason"] = ("verified: numba matches numpy (obj within tol; violation smooth-part "
                             f"within tol, {out['n_step_flips']} knife-edge breach flip(s) ignored)")
        else:
            out["reason"] = (
                f"MISMATCH rel_obj={out['max_rel_obj']:.2e} (tol {rtol_obj:.0e}), "
                f"smooth_viol={out['max_smooth_viol']:.2e} (tol {atol_viol:.0e}+{rtol_viol:.0e}·|v|), "
                f"abs_viol={out['max_abs_viol']:.2e} across {out['n_step_flips']} step-flip(s)")
        return out
    except Exception as exc:  # noqa: BLE001 - verification must never crash a run
        out["reason"] = f"{type(exc).__name__}: {exc}"
        return out
