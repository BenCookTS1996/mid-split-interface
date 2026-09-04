"""19jd - [proj-inside]: where a projection's milliseconds actually go.

The band projector is 163.2s of a 277.5s search and 70% of every evaluation, and it had NO
per-call breakdown at all - the same blind spot [cap-timing] had before 19ja. [pbp-inside]
splits the CONSTRUCTOR (a one-off 25s); this splits the 486 ms that runs 336 times.

THE MEASUREMENT IS THE THING BEING TESTED HERE. Every variant is the SAME COMPILED KERNEL
called with one of its own input arrays truncated to length zero:

    for ci in range(profiles.shape[0])     <- the profile loop
    for j  in range(cap_rowc.shape[0])     <- the t0 TXN accumulation
    for j  in range(pc_orgc.shape[0])      <- the aged VAMP accumulation

so a zero-length array makes that loop do nothing and leaves every other line untouched. The
water-fill goes off the same way, with cap = 1.0, because the kernel guards it with
`if cap < 1.0`. What has to hold for the reading to mean anything:

  1. the truncated calls do not read out of bounds and do not crash;
  2. the `full` variant really is a full projection - same bits as an ordinary call;
  3. nothing is recompiled, or the timings would be compile times (numba only);
  4. the report cannot break a run, and leaves no wrong scratch behind for the next one.
"""
import io, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(os.environ.get("RO_ROOT") or pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT / "src"))
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
BP_SRC = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
BS_SRC = (ROOT / "src/routing_optimiser/s4_search/band_scoring.py").read_text(encoding="utf-8")

import routing_optimiser.s4_search.band_projection as bp

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

HAVE_NB = getattr(bp, "_HAVE_NUMBA", None)
if HAVE_NB is None:
    HAVE_NB = hasattr(bp._cb_kernel, "signatures")
print(f"  numba: {'present' if HAVE_NB else 'ABSENT (pure-python kernel)'}")


# ═══ a small scaffold with every branch of the kernel populated ═════════════════════════
P, B, NPROF = 3, 2, 3
cstart = np.array([0, 4, 7], np.int64)
ccnt = np.array([4, 3, 3], np.int64)
NR = 10
profiles = np.arange(NPROF, dtype=np.int64)
propidx_c = np.array([0, 1, 2, 3, 0, 1, 2, 3, 4, 5], np.int64)
rng = np.random.default_rng(19)
pw_c = np.ones(NR)
base_c = rng.random(NR)
mv_c = rng.random(NR)
vcpos_c = (rng.random(NR) > 0.3).astype(float)
cap_rowc = np.array([0, 1, 5, 8], np.int64)
cap_band = np.array([0, 1, 0, 1], np.int64)
cap_c = np.array([0, 0, 1, 2], np.int64)
cap_ctot = rng.random(4) * 100
cap_base = rng.random(4)
pc_orgc = np.array([0, 3, 7, -1, 9], np.int64)
pc_vc = rng.random(5) * 10
pc_pool = rng.random(5) * 10
pc_band = np.array([0, 1, 0, 1, 0], np.int64)
pc_heldfac = rng.random(5)
pc_gc = np.array([0, 0, 1, 1, 2], np.int64)
pc_gkc = np.array([0, 0, 1, 1, 2], np.int64)
vconst = rng.random(B)
ARGS = (propidx_c, pw_c, base_c, mv_c, vcpos_c, profiles, cstart, ccnt,
        cap_rowc, cap_band, cap_c, cap_ctot, cap_base,
        pc_orgc, pc_vc, pc_pool, pc_band, pc_heldfac, pc_gc, pc_gkc, vconst)
CAP = 0.5

def bufs():
    return (np.zeros((P, B)), np.zeros((P, B)), np.zeros((1, NPROF)), np.zeros((1, NPROF)),
            np.zeros((1, NPROF)), np.zeros((1, NR)), np.zeros((1, NR)),
            np.zeros(1, np.int64), np.zeros((1, 3)))
# 19jp: the kernel gained `qst, usestash`; the (1, 1) dummy + 0 is the OFF path.
TAIL = (np.ones(NR), 0.0, 0, np.zeros((1, 1)), np.zeros(NR, bool), 0, np.zeros((1, 1)), 0)
PROP = rng.random((P, 6))

K = bp._cb_kernel
b0 = bufs()
v_ref, t_ref = K(PROP, *ARGS, CAP, 1, *b0, *TAIL)
v_ref, t_ref = np.array(v_ref, copy=True), np.array(t_ref, copy=True)
check("0  the fixture actually projects something", float(np.abs(v_ref).sum()) > 0
      and float(np.abs(t_ref).sum()) > 0)

# ═══ 1. every truncation runs, in bounds, and only turns off what it should ═════════════
PROF, TXN, AGED = {5, 6, 7}, {8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18, 19}
def trunc(idx):
    return tuple((x[:0] if i in idx else x) for i, x in enumerate(ARGS))

for _nm, _idx in (("profiles", PROF), ("cap rows", TXN), ("aged rows", AGED),
                  ("all three", PROF | TXN | AGED)):
    try:
        _v, _t = K(PROP, *trunc(_idx), CAP, 1, *bufs(), *TAIL)
        _ok, _d = True, f"vamp Σ {float(np.abs(_v).sum()):.3g}"
    except Exception as _e:  # noqa: BLE001
        _ok, _d = False, f"{type(_e).__name__}: {_e}"
    check(f"1  truncating {_nm} to zero length runs without reading out of bounds", _ok, _d)

_v, _t = K(PROP, *trunc(AGED), CAP, 1, *bufs(), *TAIL)
check("1  ...and truncating the aged rows removes ONLY the VAMP accumulation",
      float(np.abs(_v).sum()) == float(np.abs(np.tile(vconst, (P, 1))).sum())
      and np.array_equal(np.asarray(_t), t_ref),
      "txn is untouched, vamp is just the per-band constant")
_v, _t = K(PROP, *trunc(TXN), CAP, 1, *bufs(), *TAIL)
check("1  ...and truncating the cap rows removes ONLY the TXN accumulation",
      float(np.abs(_t).sum()) == 0.0 and np.array_equal(np.asarray(_v), v_ref))

# ═══ 2. the `full` variant IS a full projection, and cap=1.0 only drops the water-fill ═══
_v, _t = K(PROP, *ARGS, CAP, 1, *bufs(), *TAIL)
check("2  the variant harness's `full` call is bit-identical to an ordinary call",
      np.array_equal(np.asarray(_v).view(np.int64), v_ref.view(np.int64))
      and np.array_equal(np.asarray(_t).view(np.int64), t_ref.view(np.int64)))
_v1, _t1 = K(PROP, *ARGS, 1.0, 1, *bufs(), *TAIL)
check("2  cap=1.0 turns the water-fill off (a different answer, which is why it is timing only)",
      not np.array_equal(np.asarray(_v1), v_ref) or not np.array_equal(np.asarray(_t1), t_ref),
      "the fixture's cap 0.5 does bind, so the guard is really being exercised")

# ═══ 3. no recompile - the whole premise of 'the same compiled kernel' (numba only) ══════
if HAVE_NB:
    _n0 = len(K.signatures)
    for _idx in (PROF, TXN, AGED, PROF | TXN | AGED):
        K(PROP, *trunc(_idx), CAP, 1, *bufs(), *TAIL)
    K(PROP, *ARGS, 1.0, 1, *bufs(), *TAIL)
    check("3  none of the five variants triggers a numba recompile",
          len(K.signatures) == _n0,
          f"{_n0} signature(s) before, {len(K.signatures)} after - a zero-length slice of a "
          "C-contiguous array has the same numba type")
else:
    print("  SKIP  3  no-recompile check needs numba")

# ═══ 4. the report: runs, prints a table, cleans up, never raises ════════════════════════
class FakeProj:
    pass

LOG = []
_real_pnote = bp._pnote
bp._pnote = lambda m: LOG.append(str(m))
try:
    pj = FakeProj()
    pj._pi_call = (K, PROP, ARGS, CAP, 1, bufs(), TAIL)
    pj._lift_primed = "stale"
    res = bp.proj_inside_report(pj, reps=2)
finally:
    bp._pnote = _real_pnote
TAB = list(res.get("lines", ())) if isinstance(res, dict) else []
for _l in TAB:
    print("        | " + _l)

check("4  the report returned a result", isinstance(res, dict) and res.get("full_ms", 0) > 0)
check("4  the table comes back as `lines` for the caller to log",
      any("[proj-inside] WHERE A PROJECTION" in l for l in TAB)
      and any("profile loop" in l for l in TAB)
      and any("water-fill" in l for l in TAB)
      and any("aged VAMP accumulation" in l for l in TAB)
      and any("TXN accumulation" in l for l in TAB))
# 19jg: `_pnote` queues into _PROJ_PAR_NOTES, which tab_2 drains under a "[proj-par]"
# prefix - so a table pushed through it came out as `[proj-par]    nA: aged VAMP ...`.
check("4  ...and NOT one row of it goes through the [proj-par] note channel",
      not LOG, f"{len(LOG)} line(s) leaked: {LOG[:2]}")
check("4  the water-fill is a SUB-row of the profile loop, not summed twice",
      any(l.lstrip().startswith("of which") for l in TAB)
      and abs(sum(v for k, v in res["phases"].items() if not k.startswith("   "))
              + res["residual_ms"] - res["full_ms"]) < 1e-6)
check("4  it states the noise bar and that the answers are discarded",
      any("MACHINE NOISE" in l for l in TAB)
      and any("wrong on purpose and are discarded" in l for l in TAB))
check("4  it dropped the stash and forced a re-prime, so no wrong scratch survives",
      pj._pi_call is None and pj._lift_primed is None)
check("4  a projector with no stash returns None instead of raising",
      bp.proj_inside_report(FakeProj()) is None)

LOG.clear()
bp._pnote = lambda m: LOG.append(str(m))
try:
    bad = FakeProj()
    bad._pi_call = ("not", "a", "kernel")     # deliberately malformed
    bad._lift_primed = "stale"
    _r = bp.proj_inside_report(bad)
finally:
    bp._pnote = _real_pnote
check("4  a broken stash is reported, not raised, and still cleans up",
      _r is None and any("[proj-inside] NOT MEASURED" in l for l in LOG)
      and bad._pi_call is None)

# ═══ 5. the wiring ══════════════════════════════════════════════════════════════════════
check("5  the stash is behind a switch and holds references, not copies",
      "_PI_ON = _SW_PROJ_INSIDE" in BP_SRC
      and 'self._pi_call = (_k, _pr_in, _args, cap, (P if par else 1), _bufs, _tail)' in BP_SRC)
check("5  the chunked path stashes too, and names its slice instead of re-slicing",
      "_pr_c = np.ascontiguousarray(_pr_in[_s0:_s1])" in BP_SRC
      and "self._pi_call = (_k, _pr_c, _args, cap, _n, _bufs, _tail)" in BP_SRC)
check("5  tab_2 runs [proj-inside] BEFORE [lift-ab], which re-runs the projector itself",
      0 < T2.find("[proj-inside] unavailable") < T2.find("[lift-ab] unavailable"))
check("5  ...and the call is wrapped, so a measurement cannot break the run",
      "[proj-inside] skipped" in T2 and "MEASUREMENT" in T2)
check("5  band_projection records 19jd", "19jd-proj-inside" in BP_SRC)
check("5  tab_2 logs the returned lines instead of draining them under [proj-par]",
      'for _pl in _pires.get("lines", ()):' in T2)


# ═══ 6. 19jg: THE BAND-PENALTY ROW, SPLIT ════════════════════════════════════════════════
# [eval-cost] charges "band projection + penalty" 418.5 ms/call - the largest row in the
# search - and the 21:23 run measured the KERNEL at 139.6 ms and the whole projection at
# 166.5 ms. Sixty percent of that row is neither, and nothing had measured it. These are
# timers, so the only thing that can go wrong is the answer moving.
import routing_optimiser.s4_search.band_scoring as bs

_specs = [bs.BandSpec(midl="m1", months=(1,), metric="vamp", ceil=10.0, floor=None, weight=1.0),
          bs.BandSpec(midl="m2", months=(1,), metric="txn", ceil=None, floor=50.0, weight=0.125)]

class _FakeProj:
    band_order = [("m1", 1), ("m2", 1)]

    def project_pop_numba(self, prop_raw):
        _p = np.asarray(prop_raw, float)
        _v = np.stack([_p.sum(axis=1) * 0.5, _p.sum(axis=1) * 0.25], axis=1)
        _t = np.stack([_p.sum(axis=1) * 4.0, _p.sum(axis=1) * 2.0], axis=1)
        return _v, _t

_ebp = bs.ExactBandPenalty(_FakeProj(), _specs, breach_fixed=0.3, breach_quad=1.0)
_pr = np.abs(rng.random((4, 7))) * 30.0
_before = dict(bs.bpf_timing())
_pen1 = np.asarray(_ebp.penalty(_pr), float)
_dt = {}
_pen2 = np.asarray(_ebp.penalty(_pr, detail_out=_dt), float)
_after = dict(bs.bpf_timing())

check("6  the timers did not move the answer - two calls, bit-identical",
      np.array_equal(_pen1.view(np.int64), _pen2.view(np.int64)))
check("6  ...and the penalty is a real, varying number on this fixture",
      float(_pen1.sum()) > 0 and len(set(_pen1.tolist())) > 1, f"{_pen1}")
check("6  the detail_out contract still holds",
      _dt.get("per_spec") is not None and _dt["per_spec"].shape == (4, 2)
      and list(_dt.get("specs", ())) == _specs)
check("6  bpf_timing counts every penalty call and all three phases",
      _after["n"] - _before["n"] == 2
      and all(_after[_k] >= _before[_k] for _k in ("contig", "project", "specs"))
      and (_after["project"] - _before["project"]) > 0,
      f"n +{_after['n'] - _before['n']}, project +"
      f"{1000.0 * (_after['project'] - _before['project']):.3f} ms")
check("6  the three phases are what `penalty` itself does, in order",
      "_bp_t0 = _bs_time.perf_counter()" in BS_SRC
      and '_BP_T["contig"] += _bp_t1 - _bp_t0' in BS_SRC
      and '_BP_T["project"] += _bp_t2 - _bp_t1' in BS_SRC)
check("6  tab_2 times the OTHER half - shares -> prop_raw - separately",
      "_pr = _fm_s2pr(np.asarray(_fd, float), _inc)" in T2
      and '_t["s2pr"] += _b - _a' in T2 and '_t["pen"] += _tm.perf_counter() - _b' in T2)
check("6  ...and prints [bpf-inside] with BOTH call counts, because they differ",
      "[bpf-inside] THE [eval-cost] BAND ROW" in T2
      and "THE TWO CALL COUNTS DIFFER ON PURPOSE" in T2)
check("6  the [bpf-inside] block cannot break the run",
      "[bpf-inside] skipped" in T2 and "MEASUREMENT" in T2)
check("6  band_scoring gained no new import beyond the clock",
      "import time as _bs_time" in BS_SRC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
