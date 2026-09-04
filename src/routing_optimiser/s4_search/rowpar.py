"""Row-parallel map for CANDIDATE-INDEPENDENT population transforms (2026-08-19bn).

A transform qualifies if row p of its output depends only on row p of its input. `_segment_softmax`
and the delivery transform (`_fm_block` then `apply_elig_pop`) both qualify: every operation in them
is either elementwise or runs along axis=1 (np.add.reduceat, np.maximum.reduceat, np.repeat), so no
value ever crosses between candidates. Splitting the population across threads then performs the
SAME operations in the SAME order on the SAME values — bit-identity is a property of the transform,
not something this module has to arrange.

This is the identical argument band_projection uses for its chunked parallel kernel ("each candidate
reads only its own row and writes only its own slice"). numpy releases the GIL inside these ufuncs,
so python threads do real parallel work: 469 -> 251 ms (1.87x) measured on a TWO-core box.

WHY THE FIRST CALL IS SERIAL. Several of the transforms wrapped here self-check on their first call
and write the verdict into module-level state (`_fm_blk_ok`, `_RX_OK`, the projector's own). A
threaded first call would race those writes and could print a verdict twice. So call 1 runs serial,
call 2 runs serial AND threaded and compares int64 bit patterns, and only then does threading take
over. A mismatch reverts to serial for the process and records why.

Switches: `_SW_ROW_PARALLEL = False` disables it everywhere. `_SW_ROW_PARALLEL_WORKERS` pins the
thread count (default: every core the process may use). `_SW_ROW_PARALLEL_MIN_ROWS` and
..._MIN_PROFILES set the size below which threading is pure overhead.
"""
from __future__ import annotations

import os as _os
from concurrent.futures import ThreadPoolExecutor as _TPE

import numpy as _np

__build__ = "2026-08-19bo-half-cores+2026-08-19bn-row-parallel"

# ── 19kg: SETTINGS THAT USED TO BE ENVIRONMENT SWITCHES ──────────────────
# No environment variable changes a run any more. Each name below is frozen at the
# value the shipped run already used - the defaults, because no routing.env exists and
# run.command exports nothing - so what shipped is what these say. They stay NAMES, not
# literals inlined at the use site, for two reasons: a test can still A/B a whole search
# by rebinding one, and a reader can see in one place every decision this module makes.
# Changing behaviour now means editing this block and saying so in a commit.
_SW_ROW_PARALLEL = True   # was ROUTING_ROW_PARALLEL, default '1'
_SW_ROW_PARALLEL_MIN_PROFILES = 1000000   # was ROUTING_ROW_PARALLEL_MIN_PROFILES, default '1000000'
_SW_ROW_PARALLEL_MIN_ROWS = 4   # was ROUTING_ROW_PARALLEL_MIN_ROWS, default '4'
_SW_ROW_PARALLEL_WORKERS = 0   # was ROUTING_ROW_PARALLEL_WORKERS, default '0'

_RP_ON = _SW_ROW_PARALLEL
_RP_WORKERS = _SW_ROW_PARALLEL_WORKERS
_RP_MIN_ROWS = _SW_ROW_PARALLEL_MIN_ROWS
# below ~1M profiles the thread hand-off costs more than the work it hands off.
# 19kg: `_rp_env_switch` DELETED with the switches it read. It existed to honour the pre-19hm
# spelling ROUTING_ROW_PARALLEL_MIN_CELLS at a shell prompt; no spelling is read any more, so
# a compatibility shim for one of them is dead weight that would print a promise nothing keeps.
_RP_MIN_PROFILES = _SW_ROW_PARALLEL_MIN_PROFILES

_POOL = [None]
_STATE = {}


# [FN-RP1]
def cores() -> int:
    """Cores this process may actually use. sched_getaffinity, not cpu_count: a cgroup-limited
    process may only be allowed a subset of the machine."""
    try:
        return max(1, len(_os.sched_getaffinity(0)))          # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return max(1, _os.cpu_count() or 1)


# [FN-RP1]
def workers() -> int:
    """HALF the cores by default (19bo), not all of them.

    numba already sizes its own pool to every core for the band projector, and a generation
    alternates between the two pools several times: genetic (serial) -> softmax (THIS pool) ->
    deliver (THIS pool) -> project (NUMBA's pool) -> fitness. Two pools each claiming the whole
    machine, handing off hundreds of times per run, leave each other descheduled and cache-cold.

    That is the only mechanism that fits the 2026-08-23 evidence: at 16 workers `deliver` and
    `softmax` were 16-19% FASTER in [gen-cost] while `genetic` and `fitness` — which this module
    never touches — were 21-23% SLOWER in the same measurement, and the search's plateau rate fell
    from 21-22/s to 19/s even though the modelled generation got cheaper. [gen-cost] cannot see it
    because it times each stage repeatedly in a tight round-robin, so every stage runs warm.

    The returns are strongly diminishing anyway — on a 2-core container 4 threads beat 16 by 1.49x —
    so half the cores should keep most of the 1.45x. `_SW_ROW_PARALLEL_WORKERS` pins it."""
    if _RP_WORKERS > 0:
        return _RP_WORKERS
    return max(2, cores() // 2)


# [FN-RP2]
def state(name: str) -> dict:
    """Per-call-site state. `phase`: 0 warm-up (serial), 1 verify, 2 threaded, -1 reverted."""
    return _STATE.setdefault(name, {"phase": 0, "msg": "", "workers": 0, "calls": 0,
                                    "threaded": 0, "slices": 0})


# [FN-RP3]
def bounds(P: int, k: int):
    """Contiguous, as-even-as-possible row slices. Contiguous matters: each thread then walks a
    contiguous block of a C-ordered array instead of striding through the whole thing."""
    k = max(1, min(int(k), int(P)))
    out = []
    for i in range(k):
        a, b = i * P // k, (i + 1) * P // k
        if b > a:
            out.append((a, b))
    return out


def _run_threaded(fn, X, slices):
    out = _np.empty_like(X)
    pool = _POOL[0]
    if pool is None or pool._max_workers < len(slices):       # noqa: SLF001
        if pool is not None:
            pool.shutdown(wait=True)
        pool = _POOL[0] = _TPE(max_workers=max(len(slices), 2),
                               thread_name_prefix="rowpar")

    def _one(bd):
        a, b = bd
        out[a:b] = fn(X[a:b])
        return None

    list(pool.map(_one, slices))
    return out


# [FN-RP4]
def row_parallel(fn, X, name: str, enabled: bool = True):
    """Apply a CANDIDATE-INDEPENDENT `fn` to row slices of `X` in threads.

    `fn` must return an array of the same shape as its argument. Falls through to `fn(X)` whenever
    threading is off, the array is too small to be worth it, or a previous verification failed."""
    Xa = _np.asarray(X)
    st = state(name)
    st["calls"] += 1
    if (not _RP_ON) or (not enabled) or Xa.ndim != 2 or st["phase"] < 0:
        return fn(X)
    P = Xa.shape[0]
    if P < _RP_MIN_ROWS or Xa.size < _RP_MIN_PROFILES:
        return fn(X)
    sl = bounds(P, workers())
    if len(sl) < 2:
        return fn(X)

    if st["phase"] == 0:
        # WARM-UP, SERIAL ON PURPOSE: let every nested first-call self-check run single-threaded.
        st["phase"] = 1
        return fn(X)

    if st["phase"] == 1:
        ref = fn(X)
        got = _run_threaded(fn, Xa, sl)
        same = (ref.shape == got.shape
                and _np.array_equal(_np.asarray(ref).view(_np.int64), got.view(_np.int64))
                if _np.asarray(ref).dtype == _np.float64 else _np.array_equal(ref, got))
        if same:
            st["phase"], st["workers"], st["slices"] = 2, len(sl), len(sl)
            st["msg"] = (f"[row-par] {name}: VERIFIED bit-identical threaded, {len(sl)} thread(s) "
                         f"of {cores()} usable core(s) — HALF by default since 19bo, so numba's "
                         f"projector pool is not contending with this one; "
                         f"`_SW_ROW_PARALLEL_WORKERS` pins it — "
                         f"over {P} candidate(s) (int64 bit-pattern comparison on {P}x"
                         f"{Xa.shape[1]:,}, stricter than array_equal). The transform is "
                         "candidate-independent, so each thread performs the same operations in "
                         "the same order on its own rows. `_SW_ROW_PARALLEL = False` reverts.")
        else:
            st["phase"] = -1
            _mx = float(_np.abs(_np.asarray(ref) - got).max()) if ref.shape == got.shape else -1.0
            st["msg"] = (f"[row-par] \u26a0 {name}: threaded result is NOT identical to serial "
                         f"(max|\u0394| {_mx:.3e}). REVERTING to serial for this process, so what "
                         "ships is the known-good path. This means the transform is NOT "
                         "candidate-independent after all — report it, do not treat it as "
                         "cosmetic.")
        return ref                                            # always ship the serial result here

    st["threaded"] += 1
    return _run_threaded(fn, Xa, sl)


# [FN-RP5]
def messages():
    """Every call site's verdict, for the run log. Empty string for sites that never threaded."""
    out = []
    for _n, _s in sorted(_STATE.items()):
        if _s.get("msg"):
            out.append(_s["msg"] + (f" Used on {_s['threaded']:,} of {_s['calls']:,} call(s)."
                                    if _s.get("threaded") else ""))
    return out
