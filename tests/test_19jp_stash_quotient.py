"""19jp - stash the age-renormalise quotient instead of re-deriving it in the nA pass.

The profile-blocked kernel computes `_pshare[o] / _vpsum[c]` TWICE per aged row: once to build
`_gks` (the age-renormalise sum) and again, one loop later and under the same guard, to build
`psh`. Each evaluation is three RANDOM gathers. [proj-inside] puts the nA phase at ~44% of a
projection and ~92% of that is gathers, so the second evaluation is worth removing.

WHAT THIS TEST IS FOR. The change is bit-identical BY CONSTRUCTION - same expression, same
dtype, same divisor - with exactly ONE assumption: a stashed quotient is never negative, so two
negative SENTINELS can carry the guard state in the same array. That assumption is the only way
this can be silently wrong, so it is tested directly rather than argued.

  * the fixture is built to hit ALL FOUR decode states, and asserts it hit them - a bit-identity
    test that only ever exercises one branch proves nothing about the other three,
  * ON vs OFF is compared on the RAW BITS, in float64 AND in float32 (the shipped precision),
  * every negative stashed value must be exactly -1.0 or exactly -2.0,
  * the OFF body must be the original loop CHARACTER FOR CHARACTER, so the revert is the
    original rather than a re-derivation of it,
  * and the 19jf analysis runs again: every name bound only inside the new `usestash` branch
    must not be read outside it.
"""
import ast, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BP = ROOT / "src/routing_optimiser/s4_search/band_projection.py"
BP_SRC = BP.read_text(encoding="utf-8")
T2_SRC = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
import routing_optimiser.s4_search.band_projection as bp

try:
    import numba as _nb
    _MODE = f"compiled (numba {_nb.__version__})"
    _COMPILED = True
except Exception:                              # noqa: BLE001
    _MODE = "PURE PYTHON (numba absent - the same kernel bodies, uncompiled)"
    _COMPILED = False
print(f"  ..    kernels under test: {_MODE}")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def bits(a):
    a = np.asarray(a)
    return a.view(np.int32 if a.dtype == np.float32 else np.int64)


# ═══ a scaffold built to reach every decode state ════════════════════════════════════════
# profile 0 rows 0-3   routed, VAMP-eligible          -> quotient > 0
# profile 1 rows 4-6   routed; row 5 vcpos 0          -> sentinel -2.0 (mpc kept, psh 0)
#                      row 6 proposes 0               -> quotient EXACTLY +0.0
# profile 2 rows 7-8   every share column is 0        -> psum 0 -> sentinel -1.0
# profile 3 rows 9-11  routed but no VAMP-eligible row -> vpsum 0 -> sentinel -1.0
P, B, NPROF, NR, NCOL, NGK = 4, 2, 4, 12, 7, 7
rng = np.random.default_rng(19)
cstart = np.array([0, 4, 7, 9], np.int64)
ccnt = np.array([4, 3, 2, 3], np.int64)
profiles = np.arange(NPROF, dtype=np.int64)
propidx_c = np.array([0, 1, 2, 3, 4, 4, 6, 5, 5, 0, 1, 2], np.int64)
pw_c = np.ones(NR)
base_c = rng.random(NR)
mv_c = rng.random(NR)
vcpos_c = np.array([1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 0., 0.])
cap_rowc = np.array([0, 1, 5, 8], np.int64)
cap_band = np.array([0, 1, 0, 1], np.int64)
cap_c = np.array([0, 0, 1, 2], np.int64)
cap_ctot = rng.random(4) * 100
cap_base = rng.random(4)
#                    D    D    D    -2    +0    -1    o<0   -1    D
pc_orgc = np.array([  0,   1,   4,    5,    6,    7,   -1,    9,    2], np.int64)
pc_gc   = np.array([  0,   0,   1,    1,    1,    2,    0,    3,    0], np.int64)
pc_gkc  = np.array([  0,   0,   1,    2,    3,    4,    5,    6,    0], np.int64)
nA = pc_orgc.shape[0]
pc_vc = rng.random(nA) * 10
pc_pool = rng.random(nA) * 10
pc_band = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], np.int64)
pc_heldfac = rng.random(nA)
vconst = rng.random(B)
ARGS = (propidx_c, pw_c, base_c, mv_c, vcpos_c, profiles, cstart, ccnt,
        cap_rowc, cap_band, cap_c, cap_ctot, cap_base,
        pc_orgc, pc_vc, pc_pool, pc_band, pc_heldfac, pc_gc, pc_gkc, vconst)
CAP = 0.5
PROP = rng.random((P, NCOL))
PROP[:, 5] = 0.0                       # profile 2 proposes nothing at all
PROP[:, 6] = 0.0                       # ...and one row of profile 1 proposes nothing

def bufs(dt=np.float64):
    return [np.zeros((P, B), dt), np.zeros((P, B), dt), np.zeros((1, NPROF), dt),
            np.zeros((1, NPROF), dt), np.zeros((1, NPROF), dt), np.zeros((1, NR), dt),
            np.zeros((1, NR), dt), np.zeros(1, np.int64),
            np.zeros((1, NGK))]        # gks stays float64 in production even under float32

def tail(qst, use, dt=np.float64):
    return (np.ones(NR, dt), dt(0.0), 0, np.zeros((1, 1), dt),
            np.zeros(NR, bool), 0, qst, int(use))

def run(use, dt=np.float64, lanes=1, kern=None):
    args = ARGS if dt is np.float64 else tuple(
        (x.astype(np.float32) if (hasattr(x, "dtype") and x.dtype == np.float64) else x)
        for x in ARGS)
    prop = PROP if dt is np.float64 else PROP.astype(np.float32)
    cap = CAP if dt is np.float64 else np.float32(CAP)
    q = np.zeros((lanes, nA), dt)
    b = bufs(dt)
    for _i in (2, 3, 4, 5, 6, 8):
        b[_i] = np.zeros((lanes,) + b[_i].shape[1:], b[_i].dtype)
    b[7] = np.zeros(lanes, np.int64)
    v, t = (kern or bp._cb_kernel)(prop, *args, cap, lanes, *b, *tail(q, use, dt))
    return np.array(v, copy=True), np.array(t, copy=True), q, np.array(b[8], copy=True)

check("0  the kernels under test are COMPILED - numba type unification is half the claim",
      _COMPILED,
      _MODE if _COMPILED else
      _MODE + " -- a bit-identity claim about numba's type unification cannot be made in "
              "pure Python, so this is red ON PURPOSE. Run this test on an interpreter that "
              "has numba (`pip install numba`); the arithmetic below still runs either way")
check("0  the age renormalise is ON, which is where the quotient comes from", bp._AGE_RENORM)
v_off, t_off, _, _ = run(0)
v_on, t_on, Q, GKS = run(1)
check("0  the fixture actually projects something",
      float(np.abs(v_off).sum()) > 0 and float(np.abs(t_off).sum()) > 0)

# ═══ 1. every decode state is REACHED, so bit-identity below means something ══════════════
q = Q[0]
gks_state = {}
check("1  state D  reached: a real quotient > 0 is stashed",
      bool((q > 0).any()), f"{int((q > 0).sum())} row(s)")
check("1  state D0 reached: a quotient of EXACTLY +0.0 is stashed",
      bool((q == 0.0).any()) and not bool(np.signbit(q[q == 0.0]).any()),
      "the row that proposes nothing - it must not be mistaken for a sentinel")
check("1  state -2.0 reached: guard holds, row not VAMP-eligible",
      bool((q == -2.0).any()), f"{int((q == -2.0).sum())} row(s)")
check("1  state -1.0 reached: no aged origin, or the held-move guard is false",
      bool((q == -1.0).any()), f"{int((q == -1.0).sum())} row(s)")
# the override is `_gsum <= 1e-12` in the READER, so it is only exercised if some named group
# really does sum to zero. Read the kernel's own `_gks` back rather than asserting it by eye.
_zero_gk = {int(k) for k in np.where(GKS[0] <= 1e-12)[0]}
_ov = [(int(j), float(q[j])) for j in range(nA) if int(pc_gkc[j]) in _zero_gk]
check("1  the pass-through override is reached, on a real quotient AND on a sentinel",
      any(x == 0.0 for _j, x in _ov) and any(x == -2.0 for _j, x in _ov)
      and any(x == -1.0 for _j, x in _ov),
      f"aged rows {[j for j, _x in _ov]} sit in group(s) {sorted(_zero_gk & set(pc_gkc.tolist()))} "
      f"whose renormalise sum is 0, so `_gsum <= 1e-12` fires for stash values "
      f"{sorted({x for _j, x in _ov})}")

# ═══ 2. the sentinel assumption, tested rather than argued ═══════════════════════════════
bad = q[(q < 0.0) & (q != -1.0) & (q != -2.0)]
check("2  every negative stashed value is EXACTLY one of the two sentinels",
      bad.size == 0, f"{bad.size} collision(s)" + (f": {bad[:5]}" if bad.size else ""))
check("2  ...and the kernel's own self-check makes the same count on the live scaffold",
      "(_q < 0.0) & (_q != -1.0) & (_q != -2.0)" in BP_SRC)

# ═══ 3. bit-identity, float64 AND float32 ════════════════════════════════════════════════
check("3  float64: ON and OFF are bit-identical on vamp and txn",
      np.array_equal(bits(v_off), bits(v_on)) and np.array_equal(bits(t_off), bits(t_on)),
      "raw bits, not allclose")
v32o, t32o, _, _ = run(0, np.float32)
v32n, t32n, Q32, _ = run(1, np.float32)
check("3  float32 (the SHIPPED precision): ON and OFF are bit-identical too",
      np.array_equal(bits(v32o), bits(v32n)) and np.array_equal(bits(t32o), bits(t32n)))
check("3  ...and float32 really is a different answer, so that was not a trivial pass",
      not np.array_equal(np.asarray(v32o, np.float64), np.asarray(v_off, np.float64)))
check("3  the stash carries `pshare`'s dtype, which is what keeps the division the same width",
      "self._stashq_buf(_lanes, _nA_st, pshare.dtype, _stash_armed)" in BP_SRC
      and Q32.dtype == np.float32)
# the search runs the PARALLEL compile with one lane per candidate; the stash is laned like
# every other scratch array, so the lane isolation has to hold for it too.
_pv0, _pt0, _, _ = run(0, np.float64, P, bp._cb_kernel_par)
_pv1, _pt1, _pq, _ = run(1, np.float64, P, bp._cb_kernel_par)
check("3  the PARALLEL compile agrees with itself ON vs OFF, at one lane per candidate",
      np.array_equal(bits(_pv0), bits(_pv1)) and np.array_equal(bits(_pt0), bits(_pt1)),
      f"{P} lanes")
check("3  ...and the parallel ON answer is the serial ON answer, lane by lane",
      np.array_equal(bits(_pv1), bits(v_on)) and np.array_equal(bits(_pt1), bits(t_on)),
      "each lane holds its OWN candidate's quotients - that is what the lanes are for")

# ═══ 4. the OFF path is the ORIGINAL loop, character for character ═══════════════════════
ORIG = """            for j in range(nA):
                o = pc_orgc[j]
                if _AGE_RENORM:
                    _gsum = _gks[pc_gkc[j]]
                    if _gsum <= 1e-12:
                        o = -1
                else:
                    _gsum = 1.0
                if o >= 0:
                    _cg = pc_gc[j]
                    if _VAMP_CONSERVE:
                        mpc = (pc_heldfac[j] if (_psum[_cg] > 0.0 and _vpsum[_cg] > 0.0) else 0.0)
                    else:
                        mpc = pc_heldfac[j] if _psum[_cg] > 0.0 else 0.0
                    if _psum[_cg] > 0.0 and vcpos_c[o] > 0.5 and _vpsum[_cg] > 0.0:
                        psh = _pshare[o] / _vpsum[_cg] / _gsum
                    else:
                        psh = 0.0
                else:
                    mpc = 0.0
                    psh = 0.0
                vamp[p, pc_band[j]] += pc_vc[j] * (1.0 - mpc) + pc_pool[j] * psh"""
check("4  the OFF branch is the pre-19jp nA loop verbatim - the revert is the original",
      BP_SRC.count(ORIG) == 1)
ORIG_G = """                for j in range(nA):
                    o = pc_orgc[j]
                    if o >= 0:
                        _cg0 = pc_gc[j]
                        if _psum[_cg0] > 0.0 and vcpos_c[o] > 0.5 and _vpsum[_cg0] > 0.0:
                            _gks[pc_gkc[j]] += _pshare[o] / _vpsum[_cg0]"""
check("4  ...and so is the OFF branch of the renormalise pass", BP_SRC.count(ORIG_G) == 1)

# ═══ 5. the 19jf analysis: no name bound only inside `usestash` is read outside it ════════
_fn = next(n for n in ast.walk(ast.parse(BP_SRC))
           if isinstance(n, ast.FunctionDef) and n.name == "_cb_kernel_impl")

def binds(nodes):
    out = set()
    for n in nodes:
        for x in ast.walk(n):
            if isinstance(x, ast.Name) and isinstance(x.ctx, (ast.Store, ast.Del)):
                out.add(x.id)
    return out

leaks = []
for node in ast.walk(_fn):
    if not isinstance(node, ast.If):
        continue
    if "usestash" not in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}:
        continue
    only_if = binds(node.body) - binds(node.orelse)
    lo, hi = node.lineno, node.end_lineno
    for x in ast.walk(_fn):
        if (isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load) and x.id in only_if
                and not (lo <= x.lineno <= hi)):
            leaks.append((x.id, x.lineno))
check("5  no name bound ONLY inside a `usestash` branch is read outside it (the 19jf trap)",
      not leaks, f"{len(leaks)} leak(s): {leaks[:4]}" if leaks else
      "checked by parsing the kernel, not by reading it")
check("5  the lane view uses an INT ternary, never an array one",
      "qs = q if usestash > 0 else 0" in BP_SRC and "_qst = qst[qs]" in BP_SRC)

# ═══ 6. default OFF, and it only arms when it can actually work ══════════════════════════
import os
check("6  default OFF - an unset ROUTING_STASH_Q ships the original kernel",
      not bp._STASH_Q_ON and not bp._STASH_Q_AB and not os.environ.get("ROUTING_STASH_Q"))
check("6  ...and OFF means the (1, 1) dummy, not a 100 MB buffer nobody reads",
      "np.zeros((int(lanes), int(nA)), dtype) if armed else np.zeros((1, 1), dtype)" in BP_SRC)
check("6  arming needs the switch AND the renormalise AND aged rows AND no failed check",
      "(_STASH_Q_ON or _STASH_Q_AB) and _AGE_RENORM and _nA_st > 0" in BP_SRC
      and '_STASH_FACT["ok"] is not False' in BP_SRC)
check("6  the A/B alternates call by call, so machine drift is shared between the arms",
      '_use_stash = (_STASH_FACT["calls"] % 2) == 0' in BP_SRC)

class _Proj:
    pass
_p = _Proj()
_p._stashq_buf = bp.PopulationBandProjector._stashq_buf.__get__(_p, _Proj)
_d = _p._stashq_buf(8, 1000, np.float32, False)
_a = _p._stashq_buf(8, 1000, np.float32, True)
check("6  the buffer is a dummy when unarmed and (lanes x nA) when armed, and is CACHED",
      _d.shape == (1, 1) and _a.shape == (8, 1000) and _a.dtype == np.float32
      and _p._stashq_buf(8, 1000, np.float32, True) is _a)

# ═══ 7. the self-check runs BOTH arms and falls back to the ORIGINAL on failure ═══════════
_sc = BP_SRC.split("def _stashq_selfcheck")[1].split("\n    # [FN-023d2]")[0]
check("7  it runs both arms through the shipping code path, shipped arm LAST",
      "_first, _second = ((tail_off, tail_on) if use_stash else (tail_on, tail_off))" in _sc
      and "_av, _at = run(_first)" in _sc and "_bv, _bt = run(_second)" in _sc)
check("7  it compares RAW EQUALITY on both outputs, not allclose",
      "np.array_equal(_av, _bv) and np.array_equal(_at, _bt)" in _sc
      and "allclose" not in _sc.replace("not allclose", ""))
check("7  on failure it disarms for the process and returns the OFF answer",
      "return (_av, _at) if use_stash else (_bv, _bt)" in _sc
      and "SELF-CHECK FAILED" in _sc and "ok=False" in _sc)
check("7  a broken self-check cannot take the run down - it falls back and says so",
      "except Exception as _se" in _sc and "return run(tail_off)" in _sc)

# ═══ 8. [stash-q] reports in EVERY state and cannot break a run ═══════════════════════════
_lines = bp.stashq_report()
check("8  OFF still prints a line - silence is indistinguishable from broken wiring",
      bool(_lines) and "[stash-q] OFF" in _lines[0] and "ROUTING_STASH_Q=1" in _lines[0])
_saved = dict(bp._STASH_FACT)
try:
    bp._STASH_Q_ON = True
    check("8  armed-but-never-run says WHY", "ARMED BUT NOT RUNNING" in bp.stashq_report()[0])
    bp._STASH_FACT.update(armed=True, ok=True, nA=845790, lanes=16, dtype="float32",
                          bytes=845790 * 16 * 4, on_ms=1000.0, on_n=10,
                          off_ms=1440.0, off_n=10)
    _r = "\n".join(bp.stashq_report())
    check("8  ON prints the real MB, the bit-identity verdict and the A/B ratio",
          "54.1 MB" in _r and "SELF-CHECK PASSED" in _r and "1.440x" in _r
          and "read the RATIO" in _r)
    bp._STASH_FACT.update(ok=False, bad=3, why="")
    check("8  a failed self-check is stated in the log, not just in the process",
          "SELF-CHECK FAILED" in "\n".join(bp.stashq_report()))
    bp._STASH_FACT["on_n"] = "not a number"
    check("8  ...and a broken report returns a line instead of raising",
          "NOT REPORTED" in "\n".join(bp.stashq_report()))
finally:
    bp._STASH_Q_ON = False
    bp._STASH_FACT.clear(); bp._STASH_FACT.update(_saved)

# ═══ 9. wiring ═══════════════════════════════════════════════════════════════════════════
check("9  tab_2 drains [stash-q], wrapped so a measurement cannot break the run",
      "stashq_report" in T2_SRC and "[stash-q] skipped" in T2_SRC
      and "[stash-q] unavailable" in T2_SRC)
check("9  band_projection records 19jp", "19jp-stash-quotient" in BP_SRC)
check("9  the existing profile-blocked vs flat self-check still guards this",
      "_CB_OK" in BP_SRC and "self-checked against the flat kernel on this scaffold " in BP_SRC)

# ═══ 9b. 19jr: THE COUNTERS ARE PER RUN, NOT PER PROCESS ═════════════════════════════════
# Streamlit keeps the process alive between runs, so anything not reset by `proj_new_run` is
# the LAST run's. 19jp missed that: the 2026-09-04 09:51 log reported 422 ON and 422 OFF
# call(s) on a run that dispatched 378 projections in total. The ratio survived - the arms
# alternate within every run, so drift is still shared - but the sample size did not.
bp._STASH_FACT.update(armed=True, checked=True, ok=True, calls=99,
                      on_ms=500.0, on_n=42, off_ms=800.0, off_n=42,
                      nA=845790, bytes=118_410_600, dtype="float32", lanes=35)
bp.proj_new_run()
check("9b the A/B counters are cleared by proj_new_run, like every other per-run measurement",
      (bp._STASH_FACT["on_n"] == 0 and bp._STASH_FACT["off_n"] == 0
       and bp._STASH_FACT["on_ms"] == 0.0 and bp._STASH_FACT["off_ms"] == 0.0
       and bp._STASH_FACT["calls"] == 0 and bp._STASH_FACT["nA"] == 0
       and bp._STASH_FACT["bytes"] == 0 and not bp._STASH_FACT["armed"]))
check("9b ...and the self-check re-runs on the NEW run's scaffold",
      bp._STASH_FACT["checked"] is False)
check("9b ...and proj_new_run still clears what it cleared before",
      bp._PROJ_PATH.get("calls") == 0 and bp._CB_OK["checked"] is False)
# a path disabled FOR CAUSE stays disabled for the process - the _CB_OK["use"] rule
bp._STASH_FACT.update(ok=False, checked=True)
bp.proj_new_run()
check("9b a self-check that FAILED stays failed - a new run does not re-arm a broken path",
      bp._STASH_FACT["ok"] is False and bp._STASH_FACT["checked"] is True)
bp._STASH_FACT.update(ok=None, checked=False)
check("9b band_projection records 19jr", "19jr-stashq-per-run" in BP_SRC)


# ═══ 10. END TO END, through `_project_cb` - the wiring, not just the kernel ══════════════
# 19jf is why this section exists. That bug was a WIRING bug: the kernel was right and the
# unit tests passed, and the run died hundreds of lines away on a name the change had made
# conditional. So this runs a REAL projector - real constructor, real aged rows, real
# `project_pop_numba` - in a subprocess per arm, and diffs the RESULT HASHES. The switch is
# read at import, so it can only be selected by a fresh process.
_E2E = r"""
import hashlib, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, sys.argv[2])
if sys.argv[1] != "off":
    os.environ["ROUTING_STASH_Q"] = sys.argv[1]
from routing_optimiser.s4_search import band_projection as bp
rng = np.random.default_rng(7)
MIDS = ("adyen_tav", "braintree_tav", "woodforest_tav", "paysafe_tav")
T0 = pd.DataFrame([{"cur": "usd", "bin": f"{100000+b}", "rpgt": "r", "pmp": "non_gp_ap",
                    "ctry": "_all_", "mid": m, "midl": m, "per": per,
                    "vi": float(rng.integers(100, 5000)), "vc": float(rng.integers(1, 50)),
                    "pr": 1.0, "fcp": 1.0, "bf": False, "excl": False, "emask": False,
                    "iscap": True, "_av": 1000.0, "keep": 1.0}
                   for b in range(16) for per in (0, 1, 2) for m in MIDS])
Pc = pd.DataFrame([{"cur": "usd", "bin": f"{100000+b}", "rpgt": "r", "pmp": "non_gp_ap",
                    "ctry": "_all_", "mid": m, "midl": m, "per": 2, "t": t,
                    "vc": float(rng.integers(1, 40))}
                   for b in range(16) for m in MIDS for t in (1, 2)])
proj = bp.PopulationBandProjector(T0, Pc, rng.random(len(Pc)) * 100 + 1.0,
                                  [(m, 2) for m in MIDS], max_share=0.97, by_profile=True)
nA = int(np.asarray(proj._nb_arrays()[7]).shape[0])
pr = rng.random((6, max(len(proj.prop_keys), 1))) ** 4
for _ in range(9):
    v, t = proj.project_pop_numba(pr)
h = hashlib.sha256(np.ascontiguousarray(v).tobytes()
                   + np.ascontiguousarray(t).tobytes()).hexdigest()[:16]
f = bp._STASH_FACT
print("RESULT|%d|%s|%s|%s|%d|%d" % (nA, h, f["armed"], f["ok"], f["on_n"], f["off_n"]))
print("REPORT|" + " ~ ".join(bp.stashq_report()))
"""
import subprocess, tempfile
_res = {}
with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as _fh:
    _fh.write(_E2E); _E2E_PATH = _fh.name
# NOT RUN without numba, and SAID rather than skipped silently: `project_pop_numba` takes the
# pure-NumPy reference path when numba is absent, so `_project_cb` - the wiring under test here
# - never executes and every assertion below would pass or fail for the wrong reason.
if not _COMPILED:
    print("  ..    10 NOT RUN: " + _MODE + ". `_project_cb` only runs on the numba path, so "
          "there is no wiring to exercise. Section 0 is already red about this.")
for _arm in (("off", "ab") if _COMPILED else ()):
    try:
        _o = subprocess.run([sys.executable, _E2E_PATH, _arm, str(ROOT / "src")],
                            capture_output=True, text=True, timeout=900)
        _res[_arm] = dict(
            line=next((l for l in _o.stdout.splitlines() if l.startswith("RESULT|")), ""),
            rep=next((l for l in _o.stdout.splitlines() if l.startswith("REPORT|")), ""),
            err=(_o.stderr or "").strip().splitlines()[-1:] )
    except Exception as _e:                    # noqa: BLE001
        _res[_arm] = {"line": "", "rep": "", "err": [f"{type(_e).__name__}: {_e}"]}

_ok = bool(_res) and all(r["line"] for r in _res.values())
if _COMPILED:
    check("10 a real projector runs end to end under both arms", _ok,
          "" if _ok else str({k: v["err"] for k, v in _res.items()}))
if _ok:
    _off = _res["off"]["line"].split("|")
    _ab = _res["ab"]["line"].split("|")
    check("10 the scaffold really has aged rows, or this section proves nothing",
          int(_off[1]) > 0, f"nA={_off[1]}")
    check("10 the ARMED run's answer hashes identically to the untouched run's",
          _off[2] == _ab[2] and len(_off[2]) == 16, f"{_off[2]} vs {_ab[2]}")
    check("10 the stash armed, self-checked itself on the live scaffold, and PASSED",
          _ab[3] == "True" and _ab[4] == "True", f"armed={_ab[3]} ok={_ab[4]}")
    check("10 the A/B really alternated - both arms carry timings",
          int(_ab[5]) > 0 and int(_ab[6]) > 0, f"ON {_ab[5]} call(s), OFF {_ab[6]} call(s)")
    check("10 ...and [stash-q] said so in the log the run would have printed",
          "INTERLEAVED A/B over" in _res["ab"]["rep"]
          and "SELF-CHECK PASSED" in _res["ab"]["rep"])
    check("10 the OFF run stays OFF - a default build arms nothing",
          "|False|None|" in _res["off"]["line"])

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
