"""EXACT, fast collapse of the per-MID cap projection used for GA band scoring.

Context
-------
The GA's per-MID month bands (e.g. "Adyen_TotalAVPro M5 VAMP ≤ 1,800") are defined on the
TRUE pro-rata projection. In `app/tab2_engine.py` that projection is `_project_capped`: a
two-cohort model over a large scaffold (`_T0` = the t0 rows, `_Pc` = the aged rows, one per
observation-month × age `t`). It is bit-exact but costs a few bincounts over ~1–2M rows per
call — fine for enforcement's few hundred calls, far too slow for the GA's ~1.2M evaluations.
THIS MODULE is the answer to that: the same projection, collapsed, so the search can afford to
score the TRUE value every generation.

WHAT THE SEARCH ACTUALLY SCORES (corrected 2026-08-19bb)
-------------------------------------------------------
Until 19bb this paragraph said the projection was "far too slow ... which is why the search falls
back to a crude volume-ratio proxy (the source of the large proxy↔true gaps in the run log)".
That is no longer true and the stale sentence caused a real misreading, so it is gone:

  * the proxy is REMOVED, not merely off. `run_fullmatrix_ga` is called WITHOUT `mid_bands` (the
    proxy hook — it defaults to None and every use of it sits behind `if mid_bands:`), and WITH
    `band_penalty_fn = ExactBandPenalty.penalty`. tab2_engine guards it in two places: "NO proxy
    fallback (removed per config): exact band scoring is mandatory. Crash loudly so a broken setup
    is never silently downgraded to the proxy."
  * `band_scoring.py` holds exactly ONE penalty class, `ExactBandPenalty`. There is no proxy class
    left to fall back to.
  * the run log states it three times, including "band scoring=EXACT in-search (per-generation
    pro-rata projection; no proxy, no post-hoc correction)".
  * and the NUMBERS settle it rather than the comments: the five-rung chain prints the same value
    at every rung (raw → GA-fitness → shipped → enforced → delivered) and RECONCILIATION ERROR is
    0 across all 15 bands. A proxy scoring the search could not produce that — the "proxy↔true
    gap" would show up as DELIVERY DRIFT, which is the column reading zero.

A volume-ratio proxy DOES still exist for the TILT engine (tab2_engine's "calibrated volume-ratio
proxy + re-projection correction" line, which that code explicitly says not to print for the
full-matrix engine). It is not on this path.

Key structural facts verified in `_project_capped`:
  * An aged row seen in month `per` at age `t` ORIGINATED in month `om = per − t`, and its
    movement is governed by the ORIGIN t0 row's movable fraction and share
    (`_Pc → _T0` join on `om == per`).
  * The moved-VAMP POOL is already aggregated to CELL grain (vampMid summed out):
    `pool(cell,per,t) = Σ_all-mids vc·pro_rata·fcp1` (the app's static `_Pc_movedvpool_a`),
    then re-split by the proposed share.
  * Caps are NOT applied inside the projection — the ceiling/floor is compared afterwards.
    So the forward map is LINEAR in the (per-cell-normalised) share.
  * **The movable fraction is GATED on the cell's proposed-share sum** (line 3951 of the app):
    `mv = where(psum > 0 AND vpsum > 0, pr·fcp, 0)` as of 2026-08-19aq — the vpsum test is the
    CONSERVATION guard: a cell that is routed but has no VAMP-positive door has vshare == 0
    everywhere, so moving the VAMP out would destroy it (delivery has always held it instead).
    ROUTING_VAMP_CONSERVE=0 restores the psum-only gate. In a cell where the candidate assigns ZERO to every
    gateway (zero-volume historical cell, or every gateway zeroed by the wallet/USA `emask`),
    NOTHING moves — each MID holds 100 % of its own VAMP. This gate is candidate-dependent,
    so the VAMP held-cohort is NOT a single static offset: it is `Σvc` (constant) MINUS a
    movement term `vc·mv` that is subtracted only when the origin cell is active (psum>0).

Therefore, for the count metrics:
    VAMP(mid,P) = Σ_pc vc                                    (static constant)
                − Σ_{origin t0 rows} (mv·Σvc) · [psum_origin>0]   (movement, psum-gated)
                + Σ_{origin t0 rows} pool_sum · vshare(origin)    (redistributed pool)
    Txn (mid,P) = Σ_{t0 cap rows}  ctot·base                         if psum_cell==0
                = Σ_{t0 cap rows}  ctot·(base·(1−mv) + moved_tot·pshare)  if psum_cell>0
where the per-row constants (`ctot`, `base`, `mv=pr·fcp`, `moved_tot`, `pool_sum`, `Σvc`) are
STATIC (precomputed once) and the only per-candidate inputs are the per-cell shares and the
`psum>0` active mask. `vshare`/`pshare` are the exact per-cell normalisations `_project_capped`
uses. This module precomputes the static pieces and evaluates the (piecewise-)linear form.

`BandProjector` precomputes the static collapse of `_project_capped` and evaluates it; the
collapse is exact for the VAMP/Txn COUNT metrics — including a cell whose candidate share is
entirely zero (the psum==0 path). (The `vamp_pct` RATE metric is out of scope: project VAMP
and volume separately and divide at the end.)

Inputs (both frames use lower-cased helper columns):
  T0   : cur, bin, rpgt, pmp, ctry, mid, midl, per, vi, vc, pr, fcp, bf, excl, emask, iscap
         (`_av` = per-row non-excluded VI used for the baseline share; `iscap`=MID has a band;
          `bf`=injected back-fill row; `excl`=switched-off; `emask`=wallet/USA-only masked)
  Pc   : cur, bin, rpgt, pmp, ctry, mid, midl, per, t, vc
  pool : Pc-aligned static array = cell-grain moved-VAMP pool (`_Pc_movedvpool_a`).
`prop` is a dict keyed (cur,bin,mid) — or (cur,bin,rpgt,mid) when by_rpgt — → proposed share.
Cell (`grpk`) = (cur,bin,rpgt,pmp,ctry,per); `ctot` = cell total VI at t0.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

# Bumped when the projection signature/behaviour changes so stale bytecode is obvious in the run log.
# KEEP THIS IN STEP WITH THE FILE. The run header prints it under "backend build markers (if any
# ≠ expected → stale bytecode)", and it is the ONLY thing standing between a stale in-memory module
# and a wasted run. It sat at "2026-08-16" through six changes (19y prange, 19ae lift, 19af
# gathers, 19aj fastmath, 19aq VAMP conservation, 19ar fastmath retired) and so failed to catch the
# 2026-08-22 16:13 run, which silently re-used an 11:20 module and reproduced the previous run
# byte-for-byte. `[loaded]` now reports the live module's real state as well, because a marker
# nobody remembers to bump is not a guard.
__build__ = ("2026-08-19bn-lane-cap-16"
             "+2026-08-19bi-int32-indices-adopted+2026-08-19bf-chunk-speedup-claim-delegated-to-kernel-ab+2026-08-19bb-docstring-proxy-claim-deleted+2026-08-19az-chunked-parallel-adopted+2026-08-19aw-serial-fastmath+2026-08-19av-lazy-fastmath+staleness-sentinel+2026-08-19aq-vamp-conservation-gate+2026-08-16-appearance-month-timing"
              "+candidate-parallel-kernel")

try:                                   # numba is optional — pure-NumPy path used if absent
    from numba import njit as _njit, prange as _prange, get_num_threads as _nthreads
    _HAVE_NUMBA = True
except Exception:                      # noqa: BLE001
    _HAVE_NUMBA = False
    _prange = range                    # so the kernel body is valid Python without numba

    def _nthreads():
        return 1

    # [FN-008]
    def _njit(*_a, **_k):
        # [FN-009]
        def _deco(f):
            return f
        return _deco

_GRPK = ["cur", "bin", "rpgt", "pmp", "ctry", "per"]

# ── VAMP CONSERVATION GATE (2026-08-19aq) ─────────────────────────────────────────────────────
# The VAMP move used to be gated on the origin cell being ROUTED (psum > 0) alone. In a cell that
# is routed but has NO VAMP-positive door, vshare is 0 for every row, so the moved-out VAMP had
# nowhere to land and was DESTROYED — the projector reported a fraud reduction that does not
# happen. Delivery has always guarded this (`_move = where(Σ_pshare > 1e-12, _move, 0)`); this is
# the same guard on the in-search side. Measured cost of NOT having it: 100% of the cell's VAMP
# (165 of 165 on the test_recon616 fixture), and Σ|Δ| 789.8 on the 2026-08-22 13:33 seed.
#
# Read ONCE at import so numba treats it as a compile-time constant and folds the branch away —
# a new kernel ARGUMENT would shift every positional index that [kernel-ab] and [kernel-ga]
# derive from the signature. ROUTING_VAMP_CONSERVE=0 restores the pre-19aq gate for an A/B.
_VAMP_CONSERVE = os.environ.get("ROUTING_VAMP_CONSERVE", "1") != "0"



# [FN-010]
def _pop_band_kernel_impl(prop_raw, propidx, masked, gcode, base, mv_s, vcpos, ctot,
                          pc_org, pc_vc, pc_pool, pc_band, pc_heldfac, cap_row, cap_band,
                          cap_c, cap_ctot, cap_base, pc_gc,
                          ncell, nband, cap, nlane, live_rows, live_cells,
                          vamp, txn, psum, vpsum, moved, pr, pshare, vshare, mvrow, nzc, exc,
                          rsum):
    """Bit-identical numba equivalent of PopulationBandProjector.project_pop: flat passes over
    the reduced scaffold with per-cell scratch (ncell), no dense (P × nR) arrays. ~7× faster on
    the real scaffold. cap_row is pre-filtered to non-excl rows (excl txn contributions are 0).

    `cap` (max_share) < 1.0 folds in the per-sub-cell max-share water-fill (matches
    build_split_exports); cap >= 1.0 is a no-op. vshare is (re)derived from the capped routed
    share so the cap flows into VAMP exactly as the delivered projection does.

    All working arrays are passed IN and REUSED across calls (no per-generation allocation). Only
    accumulators need resetting: vamp/txn zeroed here; psum/moved/vpsum(=VAMP denom)/nzc/exc/rsum
    zeroed per candidate; pr/pshare/vshare/mvrow fully overwritten. Byte-identical to fresh alloc.

    CANDIDATE-PARALLEL (2026-08-19y). The scratch arrays are LANED: shape (nlane, ...) instead of
    (...,). Candidate p uses lane `q`, so with nlane == P every prange iteration owns its scratch
    outright and the loop is race-free BY CONSTRUCTION rather than by argument. Nothing about the
    arithmetic changes: candidate p reads only prop_raw[p] and accumulates only into vamp[p]/txn[p]
    (no cross-candidate accumulation exists to reorder), and within a candidate every loop and every
    `+=` keeps its original order. Bit-identical, asserted with np.array_equal at the real shapes.

    nlane == 1 collapses every candidate onto lane 0. That is ONLY valid when this body is compiled
    with parallel=False, where numba lowers `prange` as plain `range` so the candidates run in
    sequence and reusing one lane is exactly the pre-2026-08-19y behaviour. `project_pop_numba`
    owns that pairing (nlane == 1 <=> the serial compile); do not call the parallel compile with
    nlane == 1.

    FROZEN-SCAFFOLD LIFT (2026-08-19ae). The flat passes walk `live_rows` / `live_cells` instead
    of every row and cell. A scaffold cell whose every row is masked or sits on a prop-key that no
    GA column maps to has psum == 0 for EVERY candidate, so all ten flat passes over its rows are
    provably no-ops: `psum += v` adds exactly 0.0 (exact in floating point), and moved / nzc /
    vpsum / the water-fill are every one guarded on psum > 0. Skipping them is therefore
    BIT-IDENTICAL rather than an approximation — but ONLY because two things hold:

      * the frozen rows' `pshare`, `mvrow`, `vshare` and `pr` are PRIMED once by
        `project_pop_numba` (pshare[r] = base[r]; the rest 0). The nC/nA loops below still read
        those entries for frozen cap/pool rows, so leaving them stale from an earlier call would be
        a silent wrong answer. If you touch the lift, check that priming FIRST.
      * the nC and nA accumulation loops are UNTOUCHED and still run over every j in the original
        order. Hoisting the frozen contribution out of them into a precomputed constant would be
        exact in real arithmetic but NOT bit-identical — float addition is not associative, so the
        summation order would change. That is why this lift skips the flat passes ONLY, and why the
        1.32x the [frozen-scaffold] block projected is an UPPER BOUND on what it delivers.

    Measured on the real scaffold (nR=1,275,348, P=4, cap=0.97): 750 ms serial -> 353 ms on 2
    cores (2.13x). The shipped kernel took a flat ~198 ms per candidate at P=1, 2 AND 4, so the
    ceiling is min(P, cores)x."""
    P = prop_raw.shape[0]; nR = propidx.shape[0]
    nA = pc_org.shape[0]; nC = cap_row.shape[0]
    # FROZEN-SCAFFOLD LIFT (2026-08-19ae). `live_rows` / `live_cells` are the scaffold rows and
    # cells the GA can actually move. Passing arange(nR) / arange(ncell) disables the lift and
    # reproduces the pre-19ae kernel exactly, so there is ONE body and no second compile.
    nLR = live_rows.shape[0]; nLC = live_cells.shape[0]
    vamp[:, :] = 0.0; txn[:, :] = 0.0
    # LANE INDEX, as integer arithmetic hoisted OUT of the parallel loop. The obvious form
    # `q = p if nlane > 1 else 0` does NOT survive numba's parfor pass: it unifies the ternary to
    # float64 and the parallel compile dies with
    #   "No implementation of function getitem found for signature getitem(array(float64, 2d, C),
    #    float64) ... Unsupported array index type float64"
    # which is a hard TypingError at first call, not a silent wrong answer. `lane_stride` is a
    # plain int computed before the loop, so `p * lane_stride` is unambiguously int64.
    # stride 1 => lane q == p  (parallel path, nlane == P: every iteration owns its scratch)
    # stride 0 => lane q == 0  (serial path,   nlane == 1: candidates run in sequence, reuse it)
    # nlane must be exactly 1 or exactly P. Anything in between would alias lanes across
    # concurrent iterations — which is why this is a stride and not `p % nlane`, a form that
    # would quietly accept the racy middle ground.
    lane_stride = 1 if nlane > 1 else 0
    for p in _prange(P):
        q = p * lane_stride
        _psum = psum[q]; _vpsum = vpsum[q]; _moved = moved[q]
        _pr = pr[q]; _pshare = pshare[q]; _vshare = vshare[q]; _mvrow = mvrow[q]
        _nzc = nzc[q]; _exc = exc[q]; _rsum = rsum[q]
        for _ci in range(nLC):
            c = live_cells[_ci]
            _psum[c] = 0.0; _moved[c] = 0.0
        for _ri in range(nLR):
            r = live_rows[_ri]
            v = 0.0 if masked[r] else prop_raw[p, propidx[r]]
            _pr[r] = v
            _psum[gcode[r]] += v
        for _ri in range(nLR):
            r = live_rows[_ri]
            c = gcode[r]
            if _psum[c] > 0.0:
                _moved[c] += base[r] * mv_s[r]
        for _ri in range(nLR):
            r = live_rows[_ri]
            c = gcode[r]; ps = _psum[c]
            if ps > 0.0:
                _pshare[r] = _pr[r] / ps; _mvrow[r] = mv_s[r]
            else:
                _pshare[r] = base[r]; _mvrow[r] = 0.0
        # ---- per-sub-cell max-share water-fill (only cells with >=2 routed gateways) ----
        if cap < 1.0:
            for _ci in range(nLC):
                c = live_cells[_ci]
                _nzc[c] = 0.0
            for _ri in range(nLR):
                r = live_rows[_ri]
                c = gcode[r]
                if _psum[c] > 0.0 and _pshare[r] > 1e-12:
                    _nzc[c] += 1.0
            for _sw in range(50):
                for _ci in range(nLC):
                    c = live_cells[_ci]
                    _exc[c] = 0.0
                any_over = False
                for _ri in range(nLR):
                    r = live_rows[_ri]
                    c = gcode[r]
                    if _psum[c] > 0.0 and _nzc[c] >= 2.0 and _pshare[r] > cap + 1e-12:
                        _exc[c] += _pshare[r] - cap; any_over = True
                if not any_over:
                    break
                for _ri in range(nLR):
                    r = live_rows[_ri]
                    c = gcode[r]
                    if _psum[c] > 0.0 and _nzc[c] >= 2.0 and _pshare[r] > cap + 1e-12:
                        _pshare[r] = cap
                for _ci in range(nLC):
                    c = live_cells[_ci]
                    _rsum[c] = 0.0
                for _ri in range(nLR):
                    r = live_rows[_ri]
                    c = gcode[r]
                    if _psum[c] > 0.0 and _nzc[c] >= 2.0 and _pshare[r] > 1e-12 and _pshare[r] < cap - 1e-12:
                        _rsum[c] += cap - _pshare[r]
                for _ri in range(nLR):
                    r = live_rows[_ri]
                    c = gcode[r]
                    if (_psum[c] > 0.0 and _nzc[c] >= 2.0 and _pshare[r] > 1e-12
                            and _pshare[r] < cap - 1e-12 and _rsum[c] > 1e-12):
                        _pshare[r] += (cap - _pshare[r]) / _rsum[c] * _exc[c]
        # ---- vshare from the (capped) ROUTED share (0 in inactive cells) ----
        for _ci in range(nLC):
            c = live_cells[_ci]
            _vpsum[c] = 0.0
        for _ri in range(nLR):
            r = live_rows[_ri]
            c = gcode[r]
            if _psum[c] > 0.0 and vcpos[r] > 0.5:
                _vpsum[c] += _pshare[r]
        for _ri in range(nLR):
            r = live_rows[_ri]
            c = gcode[r]
            if _psum[c] > 0.0 and vcpos[r] > 0.5 and _vpsum[c] > 0.0:
                _vshare[r] = _pshare[r] / _vpsum[c]
            else:
                _vshare[r] = 0.0
        for j in range(nC):
            # HOISTED STATIC GATHERS (2026-08-19af). gcode[cap_row[j]], ctot[cap_row[j]] and
            # base[cap_row[j]] are properties of the SCAFFOLD — they never change across
            # candidates, generations or runs — so they are precomputed per-j in `_nb_arrays` and
            # read sequentially here instead of chased through cap_row[j] every iteration. Three
            # random memory jumps become three streaming reads, ~1.4M times per candidate.
            # BIT-IDENTICAL: the arithmetic and the accumulation ORDER are untouched; only where
            # the operands are fetched from changes. Measured 1.07-1.09x, verified with
            # np.array_equal on both vamp and txn.
            r = cap_row[j]; c = cap_c[j]
            txn[p, cap_band[j]] += cap_ctot[j] * (cap_base[j] * (1.0 - _mvrow[r])
                                                  + _moved[c] * _pshare[r])
        for j in range(nA):
            o = pc_org[j]
            # APPEARANCE-MONTH timing: held move = fcp[origin]·pro_rata[appearance] (pc_heldfac),
            # gated on the ORIGIN cell being routed (psum>0). Pool is pre-built appearance-timed.
            if o >= 0:
                # pc_gc[j] == gcode[pc_org[j]], precomputed (see the nC loop above). Only
                # read under `o >= 0`, so the value stored for o < 0 rows is never used — it is
                # built with a clipped index purely to avoid a negative-index wrap at construction.
                _cg = pc_gc[j]
                # CONSERVATION (19aq): move the VAMP out only if the origin cell is routed AND
                # has a VAMP recipient. Without the second test, a routed cell with no
                # VAMP-positive door moves the VAMP out and then has vshare == 0 everywhere, so
                # `pc_pool[j] * psh` returns none of it and the VAMP vanishes. `_VAMP_CONSERVE`
                # is a module constant, so numba folds this branch at compile time.
                if _VAMP_CONSERVE:
                    mpc = (pc_heldfac[j] if (_psum[_cg] > 0.0 and _vpsum[_cg] > 0.0) else 0.0)
                else:
                    mpc = pc_heldfac[j] if _psum[_cg] > 0.0 else 0.0
                psh = _vshare[o]
            else:
                mpc = 0.0
                psh = 0.0
            vamp[p, pc_band[j]] += pc_vc[j] * (1.0 - mpc) + pc_pool[j] * psh
    return vamp, txn


# ONE BODY, TWO COMPILES. numba lowers `prange` as `range` under parallel=False, so the serial
# compile IS the pre-2026-08-19y kernel and there is no duplicated body to drift apart.
# `_pop_band_kernel` keeps its name so any external caller/test still resolves — but its signature
# gained `nlane` and its scratch is now laned, so a caller building its own buffers must pass
# (1, ...)-shaped scratch with nlane=1. `_nb_buffers` is the only in-repo producer (checked).
# cache=FALSE as of 19aq, and this is NOT optional. `_VAMP_CONSERVE` is a module-level
# global, so numba folds its VALUE into the compiled code — but the value is NOT part of
# the on-disk cache identity. With cache=True the serial kernel served whatever the flag
# was on the run that first compiled it, so flipping ROUTING_VAMP_CONSERVE silently did
# NOTHING to this compile. test_proj_parallel caught it: the parallel kernel (cache=False)
# honoured the flag and the serial one did not, and the two disagreed by 1.462 units of
# VAMP while the log claimed the serial path was the exact revert. A kill-switch that
# depends on which run warmed the cache is worse than no kill-switch. One JIT compile per
# process buys the ambiguity away; this codebase has already had one wrong answer from a
# stale numba cache.
_pop_band_kernel = _njit(cache=False)(_pop_band_kernel_impl)
# cache=FALSE on the parallel compile, deliberately. Both dispatchers wrap the SAME py_func, so
# numba derives the same on-disk cache identity for both (observed: one shared
# `band_projection._pop_band_kernel_impl-NN.*.nbi/.nbc` pair, holding a single overload). Results
# agreed in every compile ORDER tested, but the failure mode if that ever stopped holding is the
# worst kind available here: the PARALLEL overload served to a nlane=1 call, i.e. every candidate
# sharing lane 0 — a genuine race whose output is silently wrong (test_proj_parallel.py's
# sensitivity check measures exactly that divergence). A stale cache already produced one wrong
# answer while this was being built, so the ambiguity is not hypothetical. One JIT compile per
# process (a few seconds, against a ~700 s run) buys the ambiguity away outright. The serial
# compile keeps cache=True: it is the unchanged pre-19y kernel and the revert path.
_pop_band_kernel_par = _njit(cache=False, parallel=True)(_pop_band_kernel_impl)

# MEASUREMENT-ONLY FASTMATH COMPILE. Retired in 19ar, back in 19av — LAZILY.
#
# The 19ar note said not to reintroduce this "without a reason the measurement supports". The
# reason is that the measurement is what is in doubt: fastmath read 0.940x against a RESOLUTION
# FLOOR of 4.8%, i.e. a 6% claim measured with a 4.8% ruler. [kernel-ab] now runs many more reps,
# which is what would actually settle it.
#
# The hazard that note named does not apply. It warned about two compiles of the SAME py_func
# sharing an on-disk numba cache identity and serving each other's overloads; both shipped
# compiles are already cache=False and so is this one, so there is no on-disk cache to confuse.
#
# LAZY: nothing compiles until a caller asks, so an ordinary run pays neither the compile time nor
# the memory. NOT BIT-IDENTICAL — fastmath permits reassociating float sums and assumes no
# NaN/Inf. Every consumer must label it answer-changing, and nothing selects it as the shipped
# kernel.
_pop_band_kernel_fm_cache = {}


# [FN-010a]
def pop_band_kernel_fastmath(parallel=True):
    """The fastmath dispatcher, compiled on first use. Measurement only.

    ONE COMPILE PER `parallel` VALUE, and the caller must pick the one matching the path it is
    replacing. The kernel body forbids calling the PARALLEL compile with nlane == 1 (stride 0
    aliases every candidate onto lane 0 — a race, not an error), so a serial call site needs the
    serial compile. Populations from 10 up run serial, which is exactly when this matters.
    """
    _k = bool(parallel)
    if _k not in _pop_band_kernel_fm_cache:
        _pop_band_kernel_fm_cache[_k] = _njit(
            cache=False, parallel=_k, fastmath=True)(_pop_band_kernel_impl)
    return _pop_band_kernel_fm_cache[_k]


# STALENESS SENTINEL. [loaded] in tab2_engine used to infer "is this module current?" from whether
# `_pop_band_kernel_fm` existed — a probe that inverts the moment that feature comes back, which is
# exactly what happened in 19av. Bump this string on every change to this file instead: its meaning
# does not depend on any other feature's presence.
_LOADED_SENTINEL = "19bi"

# Lane cap. Per-lane scratch is 4 x (nR,) float64 per lane — 40.8 MB per lane at nR=1,275,348 — so
# the parallel path is only taken for a SMALL population. The full-matrix engine runs pop 4 (P=3
# children per generation, P=4 at each restart init), well inside this. The GLOBAL GA shares
# `ExactBandPenalty` and can run pop ~60, where P lanes would be ~2.4 GB: those calls take the
# serial path and the projector says so in the log rather than quietly allocating.
# 19bn: 8 -> 16 on Ben's instruction. Every run since 19az has printed "the lane cap is 8 but 16
# numba thread(s) are available, so the chunked path runs 8 lanes wide and leaves 8 core(s) idle."
# Bit-identity is not at risk and is not being assumed: each candidate reads only its own row and
# writes only its own slice, and the self-check below verifies the chunked parallel kernel against
# the serial one on the LIVE scaffold on its first call, every run. What DOES change is scratch —
# 0.29 -> 0.58 GB at nR=1,275,348 — and bandwidth pressure, which is the constraint the F and G
# variants were attacking, so a wider cap is not automatically faster. Watch [kernel-ab] row A.
_PROJ_LANE_CAP = max(1, int(os.environ.get("ROUTING_PROJ_LANES", "16") or 16))
_PROJ_PAR_ON = os.environ.get("ROUTING_PROJ_PARALLEL", "1") != "0"
_PROJ_PAR_SAID = {}

# WHY A NOTE LIST AND NOT JUST print(). tab2_engine's `log()` is a CLOSURE defined inside the
# render function (tab2_engine.py:1064) — a library module cannot reach it — and nothing in the app
# redirects stdout (checked: no redirect_stdout / no sys.stdout reassignment), so a bare print()
# lands on the terminal and NEVER in runs/<ts>/log.txt. That matters here: the run log is the only
# instrument for these decisions, and a projection that silently declined to go parallel would read
# as "parallel is no faster" rather than "parallel never ran". So every note is BOTH printed and
# appended here, and tab2 drains this list into the run log after the search. Bounded so a
# pathological caller cannot grow it without limit.
_PROJ_PAR_NOTES = []
# FROZEN-SCAFFOLD LIFT switch. ON by default as of 2026-08-19ae; ROUTING_PROJ_LIFT=0 restores the
# full-range kernel, which is the same body with arange() index arrays — a true revert, not a
# similar path.
_PROJ_LIFT_ON = os.environ.get("ROUTING_PROJ_LIFT", "1") != "0"

# CHUNKED-PARALLEL PROJECTION (2026-08-19az). ON by default. When P exceeds the lane cap, split the
# candidate range into cap-sized parallel calls instead of declining to the serial kernel — the
# scratch cost is then the CAP's, not P's (0.33 GB rather than 1.26 GB at P=35), and the search
# stops forfeiting parallelism at every population from 10 up. Do NOT quote a fixed speed-up:
# [kernel-ab] row H re-measures chunking's worth every run at the run's own width (4.5x on
# 2026-08-23). Measured 3.196x when this was written, on the live scaffold
# at P=35, bit-identical (max|Δ| 0.0 on both vamp and txn) and re-verified in-run by the
# once-per-process self-check. ROUTING_PROJ_CHUNK=0 restores the decline-to-serial path exactly.
_PROJ_CHUNK_ON = os.environ.get("ROUTING_PROJ_CHUNK", "1") != "0"


# [FN-010b]
def _pnote(msg):
    """Record a projection-parallelism note for the run log, and echo it to stdout."""
    if len(_PROJ_PAR_NOTES) < 64:
        _PROJ_PAR_NOTES.append(str(msg))
    print(f"[band_projection] {msg}")


# [FN-011]
def _prop_key(df: pd.DataFrame, by_rpgt: bool, by_subcell: bool = False) -> np.ndarray:
    """Build each row's bucket address:
      by_subcell → 'cur|bin|rpgt|pmp|ctry|mid'  (SUB-CELL decision grain)
      by_rpgt    → 'cur|bin|rpgt|mid'
      else       → 'cur|bin|mid'

    This is the key a proposed share is looked up by — the SAME string format the projector's
    `prop_keys` use, so a candidate's shares line up with the right rows (like a postcode that
    routes each share to the correct bucket). The sub-cell key includes pmp/ctry so a per-sub-cell
    share maps to exactly one scaffold row (no broadcast across sub-cells).
    """
    _cur = df["cur"].astype(str).str.strip().str.lower()
    _bin = df["bin"].astype(str).str.strip()
    _mid = df["mid"].astype(str).str.strip()
    if by_subcell:
        return (_cur + "|" + _bin + "|" + df["rpgt"].astype(str).str.strip().str.lower() + "|"
                + df["pmp"].astype(str).str.strip().str.lower() + "|"
                + df["ctry"].astype(str).str.strip().str.lower() + "|" + _mid).to_numpy()
    if by_rpgt:
        return (_cur + "|" + _bin + "|" + df["rpgt"].astype(str).str.strip().str.lower()
                + "|" + _mid).to_numpy()
    return (_cur + "|" + _bin + "|" + _mid).to_numpy()


# [FN-011b]
def _prop_key_str(k, by_rpgt: bool, by_subcell: bool = False) -> str:
    """Tuple→key string, matching `_prop_key`'s formats. SINGLE source so `_prop_raw` and
    `project_pop_from_props` can't diverge. Tuple layouts:
      by_subcell → (cur, bin, rpgt, pmp, ctry, mid)
      by_rpgt    → (cur, bin, rpgt, mid)
      else       → (cur, bin, mid)
    """
    if by_subcell:
        return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(), str(k[2]).strip().lower(),
                         str(k[3]).strip().lower(), str(k[4]).strip().lower(), str(k[5]).strip()])
    if by_rpgt:
        return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(),
                         str(k[2]).strip().lower(), str(k[3]).strip()])
    return "|".join([str(k[0]).strip().lower(), str(k[1]).strip(), str(k[2]).strip()])


# [FN-012]
def _prop_raw(T0: pd.DataFrame, prop: dict, by_rpgt: bool, by_subcell: bool = False) -> np.ndarray:
    """Look up each t0 row's proposed share from the `prop` dict by its bucket key.

    Rows that are switched off (`excl`) or wallet/USA-masked (`emask`) are forced to 0 —
    they can't receive routed volume, so they never enter the moved cohort.
    """
    keys = _prop_key(T0, by_rpgt, by_subcell)
    m = {}
    for k, v in prop.items():
        m[_prop_key_str(k, by_rpgt, by_subcell)] = float(v)
    raw = np.array([m.get(k, 0.0) for k in keys], dtype=float)
    raw = np.where(T0["excl"].to_numpy(bool), 0.0, raw)
    raw = np.where(T0["emask"].to_numpy(bool), 0.0, raw)
    return raw


# [FN-013]
def _static(T0: pd.DataFrame):
    """Precompute the per-row pieces that DON'T depend on the candidate (done once).

    Returns (gcode, ngc, base, ctot, mv_static):
      * gcode / ngc — an integer code per cell (grpk) + the number of cells, so cell-wise
        sums become fast bincounts (grouping rows into their cell "bins");
      * base        — each row's baseline share of its cell (non-excluded VI ÷ cell VI);
      * ctot        — the cell's total VI at t0, broadcast onto each of its rows;
      * mv_static   — the UNGATED movable fraction pr·fcp. The psum>0 gate ("did the
        candidate put any volume in this cell?") is applied per-candidate at eval time,
        matching `_project_capped` line 3951.
    """
    gcode = pd.factorize(T0[_GRPK].astype(str).agg("|".join, axis=1))[0]
    ngc = int(gcode.max()) + 1 if len(gcode) else 0
    av_sum = np.bincount(gcode, weights=T0["_av"].to_numpy(float), minlength=ngc)[gcode]
    base = np.where(av_sum > 0, T0["_av"].to_numpy(float) / np.where(av_sum > 0, av_sum, 1.0), 0.0)
    ctot = np.bincount(gcode, weights=T0["vi"].to_numpy(float), minlength=ngc)[gcode]
    # UNGATED movable fraction; gated on the candidate's per-cell psum>0 at eval time
    # (matches `_project_capped` line 3951: mv = where(psum>0, pr·fcp, 0)).
    mv_static = T0["pr"].to_numpy(float) * T0["fcp"].to_numpy(float)
    return gcode, ngc, base, ctot, mv_static


# [FN-014]
def _origin_map(T0: pd.DataFrame, Pc: pd.DataFrame) -> np.ndarray:
    """Each aged Pc row -> its ORIGIN t0 row index (om==per), excluding back-fill; -1 if none."""
    t0join = (T0["cur"] + "|" + T0["bin"] + "|" + T0["rpgt"] + "|" + T0["pmp"] + "|"
              + T0["ctry"] + "|" + T0["midl"] + "|" + T0["per"].astype(str)).to_numpy()
    valid = ~T0["bf"].to_numpy(bool)
    t0pos = pd.Series(np.where(valid)[0], index=t0join[valid])
    t0pos = t0pos[~t0pos.index.duplicated(keep="last")]
    om = (Pc["per"] - Pc["t"]).astype(int)
    pcjoin = (Pc["cur"] + "|" + Pc["bin"] + "|" + Pc["rpgt"] + "|" + Pc["pmp"] + "|"
              + Pc["ctry"] + "|" + Pc["midl"] + "|" + om.astype(str)).to_numpy()
    return t0pos.reindex(pcjoin).fillna(-1).to_numpy().astype(np.int64)


# [FN-015]
def _shares(T0, prop, by_rpgt, gcode, ngc, base, by_subcell=False):
    """Return (pshare, vshare, psum) for the candidate. `psum` is the per-row (broadcast
    per-cell) proposed-share sum; `psum>0` is the active mask that gates `mv`."""
    prop_raw = _prop_raw(T0, prop, by_rpgt, by_subcell)
    psum = np.bincount(gcode, weights=prop_raw, minlength=ngc)[gcode]
    pshare = np.array(base, dtype=float)
    np.divide(prop_raw, psum, out=pshare, where=psum > 0)
    vprop = prop_raw * (T0["vc"].to_numpy(float) > 0)
    vpsum = np.bincount(gcode, weights=vprop, minlength=ngc)[gcode]
    vshare = np.zeros_like(vprop)
    np.divide(vprop, vpsum, out=vshare, where=vpsum > 0)
    return pshare, vshare, psum


# NOTE: the `project_reference` oracle (a readable NumPy re-implementation of `_project_capped`)
# was removed — it was never called and no test asserted equivalence. `BandProjector` and
# `PopulationBandProjector` below are the live implementations of the same two-cohort math.


class BandProjector:
    """Precompute the static collapse of `_project_capped` for a set of banded (midl,period)
    pairs; `project(prop)` then evaluates the exact VAMP/Txn projection as a small
    (piecewise-)linear form. The only per-candidate inputs are the per-cell shares and the
    `psum>0` active mask that gates the movable fraction."""

    # [FN-017]
    def __init__(self, T0: pd.DataFrame, Pc: pd.DataFrame, pool: np.ndarray, bands,
                 by_rpgt: bool = False, by_subcell: bool = False):
        self.by_rpgt = by_rpgt
        self.by_subcell = by_subcell
        self.bands = {(str(m).strip().lower(), int(p)) for (m, p) in bands}
        self._T0 = T0.reset_index(drop=True)
        Pc = Pc.reset_index(drop=True)
        self._gcode, self._ngc, self._base, self._ctot, self._mv = _static(self._T0)
        pool = np.asarray(pool, float)

        # ---- VAMP: const Σvc  −  psum-gated movement  +  pool routed by ORIGIN vshare ---------
        # APPEARANCE-MONTH timing (see PopulationBandProjector): the held move for an aged row is
        # fcp[origin] × pro_rata[APPEARANCE cell], not pr[origin]·fcp[origin]. So each aged row's
        # held weight is fcp[origin]·prapp_row·vc (was mv_static[origin]·vc). Pool is appearance-timed
        # upstream. Keeps this diagnostic consistent with _project_capped / the population projector.
        pc_to_t0 = _origin_map(self._T0, Pc)
        pc_mid = Pc["midl"].to_numpy(); pc_per = Pc["per"].to_numpy().astype(int)
        pc_vc = Pc["vc"].to_numpy(float)
        _fcp_arr = (self._T0["fcp"].to_numpy(float) if "fcp" in self._T0.columns
                    else np.ones(len(self._T0)))
        _t0_ck = (self._T0["cur"].astype(str) + "|" + self._T0["bin"].astype(str) + "|"
                  + self._T0["rpgt"].astype(str) + "|" + self._T0["pmp"].astype(str) + "|"
                  + self._T0["ctry"].astype(str) + "|" + self._T0["per"].astype(str)).to_numpy()
        _pr_by_cell = pd.Series((self._T0["pr"].to_numpy(float) if "pr" in self._T0.columns
                                 else np.ones(len(self._T0))), index=_t0_ck)
        _pr_by_cell = _pr_by_cell[~_pr_by_cell.index.duplicated(keep="first")]
        _pc_ck = (Pc["cur"].astype(str) + "|" + Pc["bin"].astype(str) + "|" + Pc["rpgt"].astype(str)
                  + "|" + Pc["pmp"].astype(str) + "|" + Pc["ctry"].astype(str) + "|"
                  + Pc["per"].astype(str)).to_numpy()
        _pc_prapp = (_pr_by_cell.reindex(_pc_ck).fillna(0.0).to_numpy(float)
                     if len(Pc) else np.zeros(0, float))
        self._v_const = {}                       # key -> Σ vc over ALL aged rows in the band
        _v_hold = {}                             # key -> {origin_t0 -> Σ fcp[o]·prapp·vc}  (subtract if active)
        _v_pool = {}                             # key -> {origin_t0 -> Σ pool}          (× vshare[origin])
        for i in range(len(Pc)):
            key = (pc_mid[i], pc_per[i])
            if key not in self.bands:
                continue
            self._v_const[key] = self._v_const.get(key, 0.0) + pc_vc[i]
            o = int(pc_to_t0[i])
            if o >= 0:
                _v_hold.setdefault(key, {})
                _v_hold[key][o] = _v_hold[key].get(o, 0.0) + _fcp_arr[o] * _pc_prapp[i] * pc_vc[i]
                if pool[i] != 0.0:
                    _v_pool.setdefault(key, {})
                    _v_pool[key][o] = _v_pool[key].get(o, 0.0) + pool[i]
        # freeze to arrays for vectorised eval
        self._v_hold_o, self._v_hold_w, self._v_pool_o, self._v_pool_w = {}, {}, {}, {}
        for key in self.bands:
            h = _v_hold.get(key, {})
            self._v_hold_o[key] = np.array(list(h.keys()), dtype=np.int64)
            self._v_hold_w[key] = np.array(list(h.values()), dtype=float)
            p = _v_pool.get(key, {})
            self._v_pool_o[key] = np.array(list(p.keys()), dtype=np.int64)
            self._v_pool_w[key] = np.array(list(p.values()), dtype=float)

        # ---- TXN: per capped t0 row, piecewise on the cell's active mask -----------------------
        # active (psum>0):  ctot·(base·(1−mv) + moved_tot·pshare)
        # inactive (psum==0): ctot·base            (mv=0, moved_tot=0, pshare→base fallback)
        moved_tot = np.bincount(self._gcode, weights=self._base * self._mv,
                                minlength=self._ngc)[self._gcode]
        cap = self._T0["iscap"].to_numpy(bool)
        excl = self._T0["excl"].to_numpy(bool)
        t0_mid = self._T0["midl"].to_numpy(); t0_per = self._T0["per"].to_numpy().astype(int)
        _t_i, _t_const, _t_off, _t_coef = {}, {}, {}, {}
        for i in range(len(self._T0)):
            if not cap[i] or excl[i]:
                continue
            key = (t0_mid[i], t0_per[i])
            if key not in self.bands:
                continue
            _t_i.setdefault(key, []).append(i)
            _t_const.setdefault(key, []).append(self._ctot[i] * self._base[i])
            _t_off.setdefault(key, []).append(self._ctot[i] * self._base[i] * (1.0 - self._mv[i]))
            _t_coef.setdefault(key, []).append(self._ctot[i] * moved_tot[i])
        self._t_i, self._t_const, self._t_off, self._t_coef = {}, {}, {}, {}
        for key in self.bands:
            self._t_i[key] = np.array(_t_i.get(key, []), dtype=np.int64)
            self._t_const[key] = np.array(_t_const.get(key, []), dtype=float)
            self._t_off[key] = np.array(_t_off.get(key, []), dtype=float)
            self._t_coef[key] = np.array(_t_coef.get(key, []), dtype=float)

    # [FN-018]
    def project(self, prop: dict) -> dict:
        pshare, vshare, psum = _shares(self._T0, prop, self.by_rpgt,
                                       self._gcode, self._ngc, self._base,
                                       getattr(self, "by_subcell", False))
        active = psum > 0                                   # per T0 row (broadcast per cell)
        # CONSERVATION (19aq): the VAMP hold term uses `active AND has-a-VAMP-recipient`, the
        # same gate as the kernel and the dense path. TXN keeps plain `active` — its inactive
        # branch falls back to `base`, so txn already conserves and must not change.
        _vact = np.zeros(len(active), dtype=bool)
        if _VAMP_CONSERVE:
            _vpr = np.where(active, pshare, 0.0) * (self._T0["vc"].to_numpy(float) > 0)
            _vps = np.bincount(self._gcode, weights=_vpr,
                               minlength=self._ngc)[self._gcode]
            _v_active = active & (_vps > 0)
        else:
            _v_active = active
        out = {}
        for key in self.bands:
            # VAMP
            ho = self._v_hold_o[key]
            vamp = self._v_const.get(key, 0.0)
            if len(ho):
                vamp -= float((self._v_hold_w[key] * _v_active[ho]).sum())
            po = self._v_pool_o[key]
            if len(po):
                vamp += float((self._v_pool_w[key] * vshare[po]).sum())
            # TXN
            ti = self._t_i[key]
            if len(ti):
                a = active[ti]
                txn = float(np.where(a, self._t_off[key] + self._t_coef[key] * pshare[ti],
                                     self._t_const[key]).sum())
            else:
                txn = 0.0
            out[key] = [float(vamp), txn]
        return out


# ── INDEX WIDTH (19bi) ───────────────────────────────────────────────────────────────────────
# Every index the band kernel reads is an exact integer, so storing it in 4 bytes instead of 8
# changes no arithmetic — it halves the index bandwidth, which is the binding constraint once
# chunking spreads the scaffold walk across lanes. Measured 4.1% faster (12/15 paired rounds,
# p=0.035, max|Δ| 0.0 on both vamp and txn) on 2026-08-23 as [kernel-ab] variant G, then adopted.
#
# THE GUARD IS THE WHOLE SAFETY ARGUMENT. A wrapped index does not raise — it reads a different
# row and returns a plausible wrong number. So refuse to narrow anything that does not provably
# fit, and record the largest value seen so the margin is logged rather than assumed.
#
# MODULE-LEVEL, not a method. `_lift_arrays` is called on duck-typed objects that compose the
# projector's methods rather than instantiating it (every projector test fixture does this), so
# reaching through `self` for the helper would break them for no benefit.
_I32_MAX = 2_147_483_647
_I32_MIN = -2_147_483_648


def _ix32(arr, _stats=None):
    """int64 index array → int32 IF its range provably fits; otherwise unchanged."""
    a = np.asarray(arr)
    if a.dtype.kind != "i":
        return a
    if a.size:
        _hi = int(a.max())
        _lo = int(a.min())
        if _stats is not None:
            _stats["hi"] = max(_stats.get("hi", 0), _hi)
            _stats["lo"] = min(_stats.get("lo", 0), _lo)
        if _hi > _I32_MAX or _lo < _I32_MIN:
            if _stats is not None:
                _stats["refused"] = _stats.get("refused", 0) + 1
            return np.ascontiguousarray(a.astype(np.int64))
    if _stats is not None:
        _stats["narrowed"] = _stats.get("narrowed", 0) + 1
    return np.ascontiguousarray(a.astype(np.int32))


class PopulationBandProjector:
    """Exact `_project_capped` band values for a WHOLE population of candidate splits at once,
    restricted to just the sub-cells that feed the banded (midl,period) pairs.

    Same math as `BandProjector` (two-cohort held/moved, psum-gated
    movable fraction, per-sub-cell pshare/vshare renormalisation) but vectorised over P
    candidates with dense NumPy so it can be called inside the GA worker's fitness in place
    of the volume-ratio proxy. Only cells that are (a) a capped banded t0 cell or (b) an
    origin cell of a banded aged row are kept — but ALL MIDs of those cells are retained so
    the per-sub-cell normalisation (psum/vpsum) is exact.

    Interface
    ---------
    `prop_keys` : ordered list of the (cur,bin,mid) — or (cur,bin,rpgt,mid) — keys the caller
        must supply a proposed share for. Build the P×K `prop_raw` matrix in that column order
        (the worker aggregates its decoded per-gateway shares onto these keys once, via a
        precomputed column→key incidence, then calls `project_pop`).
    `project_pop(prop_raw)` -> (vamp[P,B], txn[P,B]) aligned to `band_order`.
    `project_pop_from_props(props)` : convenience for tests — builds `prop_raw` from a list of
        {key: share} dicts.
    """

    # [FN-019]
    def __init__(self, T0: pd.DataFrame, Pc: pd.DataFrame, pool: np.ndarray, bands,
                 by_rpgt: bool = False, max_share: float = 1.0, by_subcell: bool = False):
        self.by_rpgt = by_rpgt
        # SUB-CELL decision grain: prop keys include pmp/ctry (cur|bin|rpgt|pmp|ctry|mid) so a
        # per-sub-cell share maps to exactly one scaffold row instead of broadcasting across sub-cells.
        self.by_subcell = by_subcell
        # Per-sub-cell max-share CAP (matches build_split_exports): no gateway may exceed max_share
        # of a routed sub-cell; the excess water-fills onto the OTHER gateways already present.
        # 1.0 = OFF (backward-compatible). This is the SOLE driver of the scored-vs-delivered VAMP
        # residual (proven: reproduces tab-3 delivered within ±3), so folding it in makes the GA
        # fitness score the DELIVERED breach.
        self._cap = float(max_share) if max_share else 1.0
        bandset = {(str(m).strip().lower(), int(p)) for (m, p) in bands}
        T0 = T0.reset_index(drop=True)
        Pc = Pc.reset_index(drop=True)
        pool = np.asarray(pool, float)

        gcode_full, ngc_full, base_full, ctot_full, mv_full = _static(T0)
        pc_to_t0 = _origin_map(T0, Pc)
        t0_mid = T0["midl"].to_numpy(); t0_per = T0["per"].to_numpy().astype(int)
        pc_mid = Pc["midl"].to_numpy(); pc_per = Pc["per"].to_numpy().astype(int)
        iscap = T0["iscap"].to_numpy(bool)

        # banded aged rows + their origin cells; banded capped t0 rows + their cells (VECTORISED).
        # `bandset` should be the ACTUALLY-CONSTRAINED (midl,per) pairs only — restricting it here
        # is what lets the reduced scaffold shrink (fewer banded rows → fewer relevant cells).
        _bidx = (pd.MultiIndex.from_tuples(sorted(bandset)) if bandset
                 else pd.MultiIndex.from_arrays([[], []]))
        pc_band = pd.MultiIndex.from_arrays([pc_mid, pc_per]).isin(_bidx)
        t0_capband = iscap & pd.MultiIndex.from_arrays([t0_mid, t0_per]).isin(_bidx)
        _org = pc_to_t0[pc_band]
        rel = np.unique(np.concatenate([gcode_full[t0_capband],
                                        gcode_full[_org[_org >= 0]]]).astype(np.int64)) \
            if (t0_capband.any() or pc_band.any()) else np.zeros(0, np.int64)

        keep_t0 = np.isin(gcode_full, rel)                 # ALL mids of relevant cells
        t0_idx = np.where(keep_t0)[0]
        full2red = -np.ones(len(T0), np.int64); full2red[t0_idx] = np.arange(len(t0_idx))
        R = T0.iloc[t0_idx].reset_index(drop=True)

        # local per-row statics (recomputed on the reduced frame; base/ctot are cell sums, and
        # every row of each relevant cell is present, so they match the full-frame values)
        self._gcode, self._ngc, self._base, self._ctot, self._mv = _static(R)
        self._excl = R["excl"].to_numpy(bool)
        self._emask = R["emask"].to_numpy(bool)
        self._vcpos = (R["vc"].to_numpy(float) > 0).astype(float)
        # Retain the two movable-fraction FACTORS separately (mv = pr · fcp) so diagnostics can tell
        # whether a low movable fraction is driven by pro_rata (go-live phasing) or fcp1_frac (the
        # first-attempt reroutable slice). Default to 1.0 if a column is absent.
        self._pr = R["pr"].to_numpy(float) if "pr" in R.columns else np.ones(len(R))
        self._fcp = R["fcp"].to_numpy(float) if "fcp" in R.columns else np.ones(len(R))
        nR = len(R)

        # prop-key alignment (np.unique = sorted-unique + inverse index in C)
        keys = _prop_key(R, by_rpgt, by_subcell)
        if nR:
            uniq, self._propidx = np.unique(keys, return_inverse=True)
            self.prop_keys = [str(k) for k in uniq]
        else:
            self._propidx = np.zeros(0, np.int64); self.prop_keys = []
        self._K = len(self.prop_keys)

        # sparse cell-incidence (ngc × nR); cellsum(x) = (S @ x.T).T  (sparse@dense, C-fast)
        import scipy.sparse as _sp
        self._S = _sp.csr_matrix((np.ones(nR), (self._gcode, np.arange(nR))),
                                 shape=(max(self._ngc, 1), max(nR, 1)))

        # reduced Pc = banded aged rows only; remap origin to reduced t0 index
        pc_keep = np.where(pc_band)[0]
        self._pc_vc = Pc["vc"].to_numpy(float)[pc_keep]
        self._pc_pool = pool[pc_keep]
        _ofull = pc_to_t0[pc_keep]
        self._pc_org = np.where(_ofull >= 0, full2red[np.where(_ofull >= 0, _ofull, 0)], -1)
        pc_mid_k = pc_mid[pc_keep]; pc_per_k = pc_per[pc_keep]

        # ---- APPEARANCE-MONTH TIMING (#3 scored==delivered) -------------------------------------
        # The tab-3 DELIVERED projection (compute_vamp_prepost_granular) applies the go-live pro_rata
        # by the APPEARANCE month (the aged row's own period `per`), not the origination month, while
        # keeping fcp1_frac at ORIGINATION. So the movable fraction for an aged row is
        #   move = fcp[origin] × pro_rata[appearance]      (was pr[origin]·fcp[origin]).
        # Precompute, per reduced Pc row, the appearance-cell t0 pro_rata. Sourced from a t0 row of the
        # SAME sub-cell at the appearance period (cell key = cur|bin|rpgt|pmp|ctry|per, no mid — pro_rata
        # is a per-cell go-live weight). Verified bit-exact against compute_vamp_prepost_granular.
        _t0_cellkey = (T0["cur"].astype(str) + "|" + T0["bin"].astype(str) + "|" + T0["rpgt"].astype(str)
                       + "|" + T0["pmp"].astype(str) + "|" + T0["ctry"].astype(str) + "|"
                       + T0["per"].astype(str)).to_numpy()
        _t0_pr = (T0["pr"].to_numpy(float) if "pr" in T0.columns else np.ones(len(T0)))
        _pr_by_cell = pd.Series(_t0_pr, index=_t0_cellkey)
        _pr_by_cell = _pr_by_cell[~_pr_by_cell.index.duplicated(keep="first")]
        _pc_cellkey = (Pc["cur"].astype(str) + "|" + Pc["bin"].astype(str) + "|" + Pc["rpgt"].astype(str)
                       + "|" + Pc["pmp"].astype(str) + "|" + Pc["ctry"].astype(str) + "|"
                       + Pc["per"].astype(str)).to_numpy()[pc_keep]
        self._pc_prapp = (_pr_by_cell.reindex(_pc_cellkey).fillna(0.0).to_numpy(float)
                          if len(pc_keep) else np.zeros(0, float))

        # band order + per-band groupings (vectorised map to band index)
        self.band_order = sorted(bandset)
        self._B = len(self.band_order)
        _bpos = {b: k for k, b in enumerate(self.band_order)}
        self._pc_bandcol = (pd.Series(list(zip(pc_mid_k.tolist(), pc_per_k.tolist())))
                            .map(_bpos).to_numpy(np.int64) if len(pc_keep) else np.zeros(0, np.int64))
        Rmid = R["midl"].to_numpy(); Rper = R["per"].to_numpy().astype(int)
        cap_mask = (R["iscap"].to_numpy(bool)
                    & pd.MultiIndex.from_arrays([Rmid, Rper]).isin(_bidx)) if nR else np.zeros(0, bool)
        self._t_rows = np.where(cap_mask)[0]
        self._t_bandcol = (pd.Series(list(zip(Rmid[self._t_rows].tolist(),
                                              Rper[self._t_rows].tolist()))).map(_bpos).to_numpy(np.int64)
                           if len(self._t_rows) else np.zeros(0, np.int64))

    # [FN-020]
    def project_pop_from_props(self, props):
        kpos = {k: j for j, k in enumerate(self.prop_keys)}
        pr = np.zeros((len(props), self._K), dtype=float)
        for r, prop in enumerate(props):
            for k, v in prop.items():
                kk = _prop_key_str(k, self.by_rpgt, getattr(self, "by_subcell", False))
                j = kpos.get(kk)
                if j is not None:
                    pr[r, j] += float(v)
        return self.project_pop(pr)

    # [FN-021]
    def _nb_arrays(self):
        """Cast static arrays to the numba kernel's dtypes once; pre-filter excl txn rows
        (their contribution is 0, so dropping them is exact)."""
        if getattr(self, "_nbcache", None) is None:
            keep = (~self._excl[self._t_rows]) if len(self._t_rows) else np.zeros(0, bool)
            # appearance-timed held factor per reduced Pc row: fcp[origin]·pro_rata[appearance]
            # (guard origin<0 → 0; the kernel ignores those rows but avoids negative-index wrap).
            _oi = np.where(self._pc_org >= 0, self._pc_org, 0)
            _heldfac = np.where(self._pc_org >= 0, self._fcp[_oi] * self._pc_prapp, 0.0)
            # 19bi: index arrays go through _i32 (narrow only if the range provably fits);
            # float arrays are untouched. `_ixs` records the largest index seen so the int32
            # margin is logged, not assumed.
            _ixs = {}
            _I = lambda _a: _ix32(_a, _ixs)     # noqa: E731
            self._nbcache = (
                _I(self._propidx),
                (self._excl | self._emask),
                _I(self._gcode),
                self._base.astype(np.float64), self._mv.astype(np.float64),
                self._vcpos.astype(np.float64), self._ctot.astype(np.float64),
                _I(self._pc_org), self._pc_vc.astype(np.float64),
                self._pc_pool.astype(np.float64), _I(self._pc_bandcol),
                _heldfac.astype(np.float64),
                _I(self._t_rows[keep]), _I(self._t_bandcol[keep]))
            # HOISTED STATIC GATHERS for the nC/nA accumulation loops (2026-08-19af). These are
            # pure functions of the scaffold, so they belong in this once-per-projector cache.
            # Cost at the live shapes: ~45 MB (cap_c/cap_ctot/cap_base over 1.39M cap rows plus
            # pc_gc over 1.4M pool rows). Bought a measured 1.07-1.09x, bit-identical.
            # the hoisted gathers index with the (possibly int32) arrays above, so cast to
            # np.intp for the gather itself and narrow the RESULT — a fancy-index with an
            # int32 array is fine, but being explicit keeps the intent readable.
            _cr = np.asarray(self._nbcache[12], np.intp)
            _po = np.asarray(self._nbcache[7], np.intp)
            _gc = self._nbcache[2]
            self._nbcache = self._nbcache + (
                _I(_gc[_cr]),                                           # cap_c
                self._nbcache[6][_cr].astype(np.float64),               # cap_ctot
                self._nbcache[3][_cr].astype(np.float64),               # cap_base
                _I(_gc[np.where(_po >= 0, _po, 0)]))                    # pc_gc
            self._ix32 = dict(_ixs)
            _n_nar = int(_ixs.get("narrowed", 0)); _n_ref = int(_ixs.get("refused", 0))
            print(f"[band_projection] index width: {_n_nar} array(s) narrowed to int32, "
                  f"{_n_ref} kept at int64. Largest index {_ixs.get('hi', 0):,} of the "
                  f"{_I32_MAX:,} int32 ceiling "
                  f"({_ixs.get('hi', 0) / _I32_MAX:.4%} of it) — "
                  + ("ample margin. " if _n_ref == 0 else
                     "SOME ARRAY DID NOT FIT and was left int64 (correct, not a failure). ")
                  + "Values are unchanged, so the projection is bit-identical; only the bytes "
                    "moved per index change. Adopted 19bi from [kernel-ab] variant G.")
        return self._nbcache

    # [FN-022]
    def _nb_buffers(self, P, lanes=1):
        """Pre-allocated working buffers for the numba kernel, cached & REUSED across calls
        (removes tens of MB of per-generation alloc/free). The big scratch (psum/vpsum/moved
        sized ncell; pr/pshare/vshare/mvrow sized nR) is P-INDEPENDENT so it's allocated ONCE;
        only vamp/txn (P×B, tiny) reallocate when P changes (λ eval_ov vs P=1 score_of). The
        returned vamp/txn ARE these buffers — the caller must consume them before the next call
        (the search's _bands_pen reads them straight into the penalty, so that holds)."""
        # Reallocate when missing OR read-only. When this projector is shipped to a loky
        # worker, joblib memmaps its cached ndarrays as READ-ONLY, but the numba kernel
        # writes into these scratch buffers (psum[:]=0.0 …) → 'Cannot modify readonly array'
        # TypingError. A writeable re-alloc in the worker (re-cached on the worker's own copy,
        # reused for its lifetime) fixes it; the main process keeps its original buffers.
        # Numerically identical — same np.zeros scratch, fully overwritten each call.
        # NB: must test EVERY buffer, not just the first. joblib only memmaps arrays over its
        # size threshold (~1 MB), so the small per-cell scratch (psum/vpsum/moved, sized ncell)
        # can stay writeable while the large per-row scratch (pr/pshare/vshare/mvrow, sized nR)
        # is memmapped read-only — checking fixed[0] alone would miss that and the kernel fails.
        # LANED as of 2026-08-19y: first axis is the parallel lane, so the candidate loop can be
        # a prange. `lanes` is 1 on the serial path (every candidate reuses lane 0, which is only
        # correct because the serial compile lowers prange as range) and P on the parallel path.
        # Cached per lane-count: switching paths mid-run reallocates instead of silently indexing
        # a too-small first axis.
        lanes = max(1, int(lanes))
        fixed = getattr(self, "_nbbuf_fixed", None)
        if (fixed is None or getattr(self, "_nbbuf_lanes", None) != lanes
                or not all(b.flags.writeable for b in fixed)):
            nR = len(self._gcode); ncell = int(self._ngc)
            fixed = (np.zeros((lanes, ncell)), np.zeros((lanes, ncell)), np.zeros((lanes, ncell)),
                     np.zeros((lanes, nR)), np.zeros((lanes, nR)),          # pr, pshare
                     np.zeros((lanes, nR)), np.zeros((lanes, nR)),          # vshare, mvrow
                     np.zeros((lanes, ncell)), np.zeros((lanes, ncell)), np.zeros((lanes, ncell)))
            self._nbbuf_fixed = fixed
            self._nbbuf_lanes = lanes
        vt = getattr(self, "_nbbuf_vt", None)
        if vt is None or vt[0] != int(P) or not (vt[1].flags.writeable and vt[2].flags.writeable):
            B = int(self._B)
            vt = (int(P), np.zeros((int(P), B)), np.zeros((int(P), B)))     # vamp, txn (outputs)
            self._nbbuf_vt = vt
        return (vt[1], vt[2]) + fixed

    # [FN-022b]
    def _lift_arrays(self, lanes, buffers=None):
        """(live_rows, live_cells) for the frozen-scaffold lift, and PRIME the frozen scratch.

        A scaffold cell is FROZEN when every one of its rows is either masked (excl|emask) or sits
        on a prop-key with an all-zero incidence row — i.e. a prop-key no GA share column maps to,
        so prop_raw is 0 there for every candidate. Such a cell has psum == 0 always, which makes
        every flat pass over its rows a provable no-op (see the kernel docstring). Measured on the
        2026-08-21 21:24 scaffold: 95,174 of 174,827 cells (54.4%) carrying 298,841 of 1,128,484
        rows (26.5%).

        THE PRIMING IS THE DANGEROUS PART, not the index arrays. The nC/nA loops still read
        `pshare[r]`, `mvrow[r]` and `vshare[r]` for frozen cap/pool rows. In the unlifted kernel
        those were rewritten every candidate; under the lift nothing writes them, so they must be
        set to the values the unlifted kernel would have produced — pshare[r] = base[r],
        mvrow[r] = 0, vshare[r] = 0, pr[r] = 0 — and the frozen CELLS' psum/moved/nzc/vpsum/exc/
        rsum must be 0. Prime once (they are candidate- AND call-independent); leave them stale and
        the answer is silently wrong with nothing to see.

        Needs the incidence to know which prop-keys are reachable. `set_lift_incidence()` supplies
        it; without it the lift is OFF and the full ranges are returned, which is exactly the
        pre-19ae kernel. ROUTING_PROJ_LIFT=0 also forces it off.
        """
        nR = len(self._gcode)
        ncell = int(self._ngc)
        if not _PROJ_LIFT_ON or getattr(self, "_lift_inc", None) is None or not nR:
            if getattr(self, "_lift_full_nR", -1) != nR:
                # 19bi: int32 here too — [kernel-ab]'s G cast the lift arrays with the
                # rest, so the adopted path must match what was measured.
                self._lift_full_rows = _ix32(np.arange(nR, dtype=np.int64))
                self._lift_full_cells = _ix32(np.arange(ncell, dtype=np.int64))
                self._lift_full_nR = nR
            return self._lift_full_rows, self._lift_full_cells

        if getattr(self, "_lift_rows", None) is None:
            inc = self._lift_inc
            K = int(inc.shape[0])
            try:
                nnz = np.asarray((inc != 0).sum(axis=1)).ravel()
            except Exception:                                  # noqa: BLE001 - dense incidence
                nnz = (np.asarray(inc) != 0).sum(axis=1)
            reach = np.zeros(K, bool)
            reach[: len(nnz)] = np.asarray(nnz) > 0
            pi = np.asarray(self._propidx, np.int64)
            mk = np.asarray(self._excl, bool) | np.asarray(self._emask, bool)
            ok_idx = (pi >= 0) & (pi < K)
            live_row = (~mk) & ok_idx & reach[np.clip(pi, 0, max(K - 1, 0))]
            # bincount, NOT reduceat: gcode is not sorted, and reduceat over unsorted codes
            # mis-groups silently. Same trap noted in the [frozen-scaffold] measurement.
            per = np.bincount(np.asarray(self._gcode, np.int64),
                              weights=live_row.astype(float), minlength=ncell)
            cell_live = per > 0.0
            self._lift_cells = _ix32(np.where(cell_live)[0])
            self._lift_rows = _ix32(
                np.where(cell_live[np.asarray(self._gcode, np.int64)])[0])
            self._lift_frozen_rows = np.where(
                ~cell_live[np.asarray(self._gcode, np.int64)])[0].astype(np.int64)
            self._lift_frozen_cells = np.where(~cell_live)[0].astype(np.int64)
            self._lift_primed = None
            print(f"[band_projection] frozen-scaffold LIFT ON: the flat passes skip "
                  f"{len(self._lift_frozen_rows):,} of {nR:,} rows "
                  f"({len(self._lift_frozen_rows) / max(nR, 1):.1%}) in "
                  f"{len(self._lift_frozen_cells):,} of {ncell:,} frozen cells "
                  f"({len(self._lift_frozen_cells) / max(ncell, 1):.1%}). Bit-identical: those "
                  "passes are provable no-ops (psum += 0.0, everything else guarded on psum > 0). "
                  "The nC/nA accumulation loops are NOT touched — reassociating those sums would "
                  "not be bit-identical — so the realised speedup is below the row share. "
                  "ROUTING_PROJ_LIFT=0 disables.")
        # PRIME (idempotent per buffer identity + lane count).
        if buffers is not None:
            key = (id(buffers[2]), int(lanes))
            if getattr(self, "_lift_primed", None) != key:
                fr = self._lift_frozen_rows
                fc = self._lift_frozen_cells
                _, _, psum, vpsum, moved, pr, pshare, vshare, mvrow, nzc, exc, rsum = buffers
                bs = np.asarray(self._base, float)
                for q in range(int(lanes)):
                    pshare[q][fr] = bs[fr]          # what the unlifted kernel's else-branch wrote
                    mvrow[q][fr] = 0.0
                    vshare[q][fr] = 0.0
                    pr[q][fr] = 0.0
                    for arr in (psum, moved, vpsum, nzc, exc, rsum):
                        arr[q][fc] = 0.0
                self._lift_primed = key
        return self._lift_rows, self._lift_cells

    # [FN-022c]
    def set_lift_incidence(self, incidence):
        """Give the projector the GA's column→prop-key incidence so it can find frozen cells.

        Called by the app once the incidence exists. Without it the lift stays OFF (the projector
        cannot know which prop-keys are reachable), and the kernel runs the full ranges — the
        pre-2026-08-19ae behaviour."""
        self._lift_inc = incidence
        self._lift_rows = None                      # force a rebuild + re-prime
        self._lift_primed = None

    # [FN-023]
    def project_pop_numba(self, prop_raw: np.ndarray):
        """Numba-accelerated project_pop — bit-identical, ~7× faster on the real scaffold.
        Falls back to the NumPy path if numba is unavailable or the scaffold is empty.
        Working arrays are pooled (see _nb_buffers); consume the result before the next call."""
        prop_raw = np.ascontiguousarray(prop_raw, dtype=np.float64)
        if not _HAVE_NUMBA or not len(self._gcode):
            # SAY SO. This early return is a SILENT FALLBACK: the pure-NumPy `project_pop` is the
            # reference implementation (correct, but far slower and it allocates dense (P × nR)
            # arrays), and before 2026-08-19z nothing recorded that the numba path had been
            # skipped. A run on a box without numba would simply be mysteriously slow, and the
            # [proj-par] drain could only INFER it from the absence of notes. State it instead.
            if not _PROJ_PAR_SAID.get("nonumba"):
                _PROJ_PAR_SAID["nonumba"] = True
                _pnote("projection is on the pure-NumPy REFERENCE path, NOT the numba kernel — "
                       + ("numba is unavailable in this process"
                          if not _HAVE_NUMBA else
                          "the band scaffold is empty (no constrained cells this build)")
                       + ". Results are correct but this is the slow path, and candidate "
                         "parallelism does not apply to it.")
            return self.project_pop(prop_raw)
        P = int(prop_raw.shape[0])
        # CANDIDATE-PARALLEL decision (2026-08-19y). All four conditions are load-bearing:
        #   _PROJ_PAR_ON       ROUTING_PROJ_PARALLEL=0 reverts to the serial compile of the SAME
        #                      body — the revert path, asserted bit-identical by the test.
        #   P > 1              a 1-candidate prange is pure thread-pool overhead (measured 187 vs
        #                      180 ms), so score_of/report calls stay serial.
        #   nthr > 1           the self-correcting gate. joblib's inner_max_num_threads=1 sets
        #                      NUMBA_NUM_THREADS=1 inside loky workers (verified), so the GLOBAL
        #                      GA's per-seed workers see 1 thread here, take the serial path, and
        #                      never oversubscribe or allocate extra lanes. Asking "can we use
        #                      more than one thread" is more robust than sniffing process names.
        #   P <= cap           bounds the scratch: 40.8 MB per lane at nR=1,275,348.
        nthr = int(_nthreads() or 1)
        par = bool(_PROJ_PAR_ON and P > 1 and nthr > 1 and P <= _PROJ_LANE_CAP)
        # CHUNKED (19az): P over the cap no longer forfeits parallelism. Splitting the candidate
        # range into cap-sized parallel calls costs the CAP's scratch, not P's, and is
        # bit-identical — see _project_chunked. This is what every population from 10 up now takes.
        chunk = bool(_PROJ_CHUNK_ON and _PROJ_PAR_ON and P > _PROJ_LANE_CAP and nthr > 1
                     and not par)
        if _PROJ_PAR_ON and P > 1 and nthr > 1 and not par and not chunk:
            # Never decline SILENTLY — a run that quietly fell back reads as "parallel is no
            # faster" when it simply never ran.
            _k = ("cap", P)
            if _k not in _PROJ_PAR_SAID:
                _PROJ_PAR_SAID[_k] = True
                _pnote(f"candidate-parallel projection DECLINED: P={P} exceeds "
                       f"ROUTING_PROJ_LANES={_PROJ_LANE_CAP} (per-lane scratch is "
                       f"{len(self._gcode) * 4 * 8 / 1e6:.1f} MB, so P lanes would be "
                       f"{len(self._gcode) * 4 * 8 * P / 1e6:,.0f} MB) and chunking is OFF "
                       "(ROUTING_PROJ_CHUNK=0). Running serial.")
        a = self._nb_arrays()
        _lanes = (_PROJ_LANE_CAP if chunk else (P if par else 1))
        buf = self._nb_buffers(P, _lanes)
        nlane = P if par else 1
        _k2 = ("on", par, nthr, P)
        if _k2 not in _PROJ_PAR_SAID:
            _PROJ_PAR_SAID[_k2] = True
            # Name the ACTUAL reason it is off. The first version of this line always blamed
            # ROUTING_PROJ_PARALLEL, so a P=1 call or a lane-cap decline printed "OFF ...
            # ROUTING_PROJ_PARALLEL=0 forces serial" while that var was untouched — a log that
            # misstates its own configuration is how a wrong conclusion gets drawn from a right
            # number.
            if chunk:
                _why = (f"ON, CHUNKED — P={P} exceeds the lane cap "
                        f"{_PROJ_LANE_CAP}, so the population runs as "
                        f"{-(-P // _PROJ_LANE_CAP)} parallel call(s) of at most "
                        f"{_PROJ_LANE_CAP} candidates instead of declining to serial. "
                        f"Scratch is {len(self._gcode) * 4 * 8 * _PROJ_LANE_CAP / 1e9:.2f} GB "
                        f"(the cap's), not "
                        f"{len(self._gcode) * 4 * 8 * P / 1e9:.2f} GB (P's). Its WORTH is "
                        "re-measured live every run by [kernel-ab] row H (chunking OFF) at this "
                        "run's own width — a hardcoded 3.196x sat here until 19bf while H read "
                        "4.5x on the same log, which is one number too many. Bit-identical "
                        "either way. ROUTING_PROJ_CHUNK=0 reverts")
            elif par:
                _why = "ON"
            elif not _PROJ_PAR_ON:
                _why = "OFF — ROUTING_PROJ_PARALLEL=0"
            elif P <= 1:
                _why = ("OFF — single candidate, so a 1-iteration prange would be pure "
                        "thread-pool overhead (measured 187 vs 180 ms)")
            elif nthr <= 1:
                _why = ("OFF — numba sees 1 thread. Expected inside a joblib/loky worker, where "
                        "inner_max_num_threads=1 sets NUMBA_NUM_THREADS=1; outside one it means "
                        "the machine or NUMBA_NUM_THREADS is limiting us to a single core")
            else:
                _why = f"OFF — P={P} exceeds the lane cap ROUTING_PROJ_LANES={_PROJ_LANE_CAP}"
            # `nlane` stays 1 on the chunked path (it is set per chunk inside the driver), so
            # printing it here said "lanes=1" for a run using 8 — a log that misstates its own
            # configuration is how a wrong conclusion gets drawn from a right number.
            _pnote(f"candidate-parallel projection {_why} (P={P}, numba threads={nthr}, "
                   f"lanes={_lanes}, scaffold nR={len(self._gcode):,}). Bit-identical either "
                   "way — "
                   "the parallel kernel is verified against the serial one on the live scaffold "
                   "on its first call.")
        if (par or chunk) and not _PROJ_PAR_SAID.get("verified"):
            # IN-RUN SELF-CHECK, once per process, on the REAL data. Everything asserting
            # bit-identity so far was measured in a container on synthetic arrays; this proves it
            # on the actual scaffold, in the actual run, before any result is used. It is the same
            # discipline the retired [vterms] test recorded the hard way: a re-implementation of a
            # kernel is only trustworthy if it is diffed against the kernel's own output on the
            # SAME inputs in the SAME run, never against a remembered figure.
            # Cost: one extra serial projection per run (~0.2 s of a ~700 s run). On mismatch it
            # disables the parallel path for the rest of the process and says so — it does not
            # raise, because a slower correct run beats a failed one, and it does not continue
            # quietly, because that is how a wrong number reaches the log looking right.
            _PROJ_PAR_SAID["verified"] = True
            try:
                _nR = len(self._gcode); _nc = int(self._ngc); _B = int(self._B)
                _vb = ((np.zeros((P, _B)), np.zeros((P, _B)))
                       + tuple(np.zeros((1, _nc)) for _ in range(3))
                       + tuple(np.zeros((1, _nR)) for _ in range(4))
                       + tuple(np.zeros((1, _nc)) for _ in range(3)))
                _lrv, _lcv = self._lift_arrays(1, _vb)
                _vv, _vt = _pop_band_kernel(prop_raw, *a, _nc, _B, float(self._cap), 1,
                                            _lrv, _lcv, *_vb)
                _vv, _vt = _vv.copy(), _vt.copy()
                if chunk:
                    # Verify the path that will ACTUALLY run, not a stand-in for it. The chunked
                    # driver is the thing whose bit-identity is being claimed, so it is the thing
                    # diffed against the serial reference.
                    _pv, _pt = self._project_chunked(prop_raw, a, buf, _lanes)
                else:
                    _lrp, _lcp = self._lift_arrays(P, buf)
                    _pv, _pt = _pop_band_kernel_par(prop_raw, *a, _nc, _B, float(self._cap), P,
                                                    _lrp, _lcp, *buf)
                _match = np.array_equal(_vv, _pv) and np.array_equal(_vt, _pt)
                del _vb
                if _match:
                    _pnote("candidate-parallel SELF-CHECK PASSED on the live scaffold: serial and "
                           + ("CHUNKED " if chunk else "")
                           + f"parallel kernels bit-identical at P={P} (np.array_equal on both "
                             "vamp and txn, not allclose)"
                           + (f" over {-(-P // _lanes)} chunk(s) of at most {_lanes}." if chunk
                              else "."))
                    return _pv, _pt
                globals()["_PROJ_PAR_ON"] = False
                globals()["_PROJ_CHUNK_ON"] = False
                _pnote("*** candidate-parallel SELF-CHECK FAILED — "
                       f"max|Δvamp|={float(np.abs(_vv - _pv).max()):.6e} "
                       f"max|Δtxn|={float(np.abs(_vt - _pt).max()):.6e}. The parallel kernel is "
                       "DISABLED for the rest of this process and the serial result is being used, "
                       "so this run's numbers are the pre-2026-08-19y numbers. Report this: it "
                       "means the lane isolation is not holding on this machine.")
                return _vv, _vt
            except Exception as _pce:              # noqa: BLE001
                globals()["_PROJ_PAR_ON"] = False
                _pnote(f"candidate-parallel self-check could not run "
                       f"({type(_pce).__name__}: {_pce}) — falling back to the SERIAL kernel for "
                       "this process rather than trusting an unverified parallel path.")
                par = False
                chunk = False
                globals()["_PROJ_CHUNK_ON"] = False
                buf = self._nb_buffers(P, 1)
                nlane = 1
        if chunk:
            return self._project_chunked(prop_raw, a, buf, _lanes)
        _lr, _lc = self._lift_arrays(nlane, buf)
        _kern = _pop_band_kernel_par if par else _pop_band_kernel
        return _kern(prop_raw, *a, int(self._ngc), int(self._B), float(self._cap), nlane,
                     _lr, _lc, *buf)

    # [FN-023b]
    def _project_chunked(self, prop_raw, a, buf, lanes):
        """Run a P-candidate projection as ceil(P/lanes) parallel calls of at most `lanes`.

        BIT-IDENTICAL to one serial call over the same P. Candidate p reads only prop_raw[p],
        writes only vamp[p]/txn[p], and its scratch lane is private, so which lane it occupies
        cannot reach the arithmetic. `vamp[:, :] = 0.0` at the top of the kernel zeroes only the
        SLICE it is handed, which is what makes accumulating chunk-by-chunk safe.

        A chunk of exactly ONE candidate goes to the SERIAL compile: the kernel body requires
        nlane == 1 or nlane == P, because `lane_stride = 1 if nlane > 1 else 0` puts every
        candidate on lane 0 at nlane == 1 — under the parallel compile that is a race, and it
        returns wrong numbers rather than raising.

        The caller must have primed the lift at the BUFFER's lane capacity, not at a chunk width,
        or a narrow tail chunk re-keys `_lift_primed` and re-primes on every width transition.
        """
        vamp, txn = buf[0], buf[1]
        P = int(prop_raw.shape[0])
        ncell, B, cap = int(self._ngc), int(self._B), float(self._cap)
        _lr, _lc = self._lift_arrays(lanes, buf)
        for _s0 in range(0, P, lanes):
            _s1 = min(_s0 + lanes, P)
            _n = _s1 - _s0
            _kern = _pop_band_kernel if _n == 1 else _pop_band_kernel_par
            _kern(np.ascontiguousarray(prop_raw[_s0:_s1]), *a, ncell, B, cap, _n,
                  _lr, _lc, vamp[_s0:_s1], txn[_s0:_s1], *buf[2:])
        return vamp, txn

    # [FN-024]
    def _cellsum(self, x):
        """(P, nR) -> (P, ngc) segment sum over cell codes via sparse matmul (C-fast)."""
        return np.asarray((self._S @ x.T).T)

    # [FN-024b]
    def _cap_pshare(self, pshare, act):
        """Per-sub-cell max-share water-fill on the (P, nR) normalised routed share `pshare`
        (sums to 1 per active cell). Bit-faithful to build_split_exports._cap_rows: only cells
        with >=2 non-zero routed gateways are capped; excess over `self._cap` is redistributed to
        the OTHER present gateways proportional to their remaining room, up to 50 sweeps. Inactive
        cells (act=False) are never touched. Returns a new capped array (input unmodified)."""
        cap = self._cap
        if cap >= 1.0:
            return pshare
        gc = self._gcode
        W = pshare.copy()
        # cappable = active cell with >=2 non-zero routed gateways (guard read ONCE, pre-loop)
        nz = self._cellsum((W > 1e-12).astype(float))[:, gc]
        cappable = act & (nz >= 2.0)
        if not cappable.any():
            return pshare
        for _ in range(50):
            over = (W > cap + 1e-12) & cappable
            if not over.any():
                break
            excess = self._cellsum(np.where(over, W - cap, 0.0))[:, gc]
            W = np.where(over, cap, W)
            recip = cappable & (~over) & (W > 1e-12) & (W < cap - 1e-12)
            room = np.where(recip, cap - W, 0.0)
            rs = self._cellsum(room)[:, gc]
            W = W + np.where(rs > 1e-12, room / np.where(rs > 1e-12, rs, 1.0) * excess, 0.0)
        return np.where(cappable, W, pshare)

    # [FN-025]
    def project_pop(self, prop_raw: np.ndarray):
        """prop_raw : (P, K) proposed share per `prop_keys`. Returns (vamp[P,B], txn[P,B])."""
        prop_raw = np.asarray(prop_raw, float)
        P = prop_raw.shape[0]
        if not len(self._gcode):                     # no constrained cells this build
            return np.zeros((P, self._B)), np.zeros((P, self._B))
        pr = prop_raw[:, self._propidx]                         # (P, nR)
        pr = np.where((self._excl | self._emask)[None, :], 0.0, pr)
        base = self._base[None, :]; mv_s = self._mv[None, :]
        ctot = self._ctot[None, :]

        psum = self._cellsum(pr)[:, self._gcode]                # (P, nR)
        act = psum > 0
        pshare = np.where(act, np.divide(pr, psum, out=np.zeros_like(pr), where=act), base)
        if self._cap < 1.0:                                     # per-sub-cell max-share cap
            pshare = self._cap_pshare(pshare, act)
        mv = np.where(act, mv_s, 0.0)
        # vshare from the (capped) ROUTED share only — 0 in inactive cells so no pool leaks there.
        routed = np.where(act, pshare, 0.0)
        vpr = routed * self._vcpos[None, :]
        vpsum = self._cellsum(vpr)[:, self._gcode]
        vact = vpsum > 0
        vshare = np.divide(vpr, vpsum, out=np.zeros_like(vpr), where=vact)

        moved_tot = self._cellsum(base * mv)[:, self._gcode]    # (P, nR)
        ptxn = ctot * (base * (1.0 - mv) + moved_tot * pshare)
        ptxn = np.where(self._excl[None, :], 0.0, ptxn)

        # VAMP over reduced aged rows — APPEARANCE-MONTH timing (see __init__): the movable
        # fraction is fcp[origin]·pro_rata[appearance], gated on the ORIGIN cell being routed
        # (act at the origin t0 row). The pool is pre-built appearance-timed upstream.
        o = self._pc_org
        ok = o >= 0
        oi = np.where(ok, o, 0)
        _heldfac = (self._fcp[oi] * self._pc_prapp)          # (nPc,) appearance-timed held factor
        # CONSERVATION (19aq): AND the routed test with "the origin cell has a VAMP
        # recipient" (vact), matching the kernel and delivery's passthrough. Without it a routed
        # cell with no VAMP-positive door loses its moved VAMP entirely.
        _act_pc = (act & vact) if _VAMP_CONSERVE else act
        move_pc = np.where(ok[None, :], _act_pc[:, oi] * _heldfac[None, :], 0.0)
        psh_pc = np.where(ok[None, :], vshare[:, oi], 0.0)
        vp = self._pc_vc[None, :] * (1.0 - move_pc) + self._pc_pool[None, :] * psh_pc

        P = prop_raw.shape[0]
        vamp = np.zeros((P, self._B)); txn = np.zeros((P, self._B))
        if len(self._pc_bandcol):
            np.add.at(vamp.T, self._pc_bandcol, vp.T)
        if len(self._t_bandcol):
            np.add.at(txn.T, self._t_bandcol, ptxn[:, self._t_rows].T)
        return vamp, txn
