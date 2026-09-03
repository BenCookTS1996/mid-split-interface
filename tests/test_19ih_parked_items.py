"""19ih — three parked items, closed.

1  the [recon-breakdown] sentinel must SURVIVE a non-forensic call (it was clobbered).
2  [lift-ab] must interleave, use per-arm minima, and set its bar from MEASURED noise —
   and must never call a sub-1.0x reading real.
3  §12: the live engine's water-fill recipient rule has no blocked-row clause. Asserted
   against the source so the answer cannot silently rot.
"""
import importlib.util, io, pathlib, re, sys
import numpy as np

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m); return m

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAIL = []
def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok: FAIL.append(name)

# ══ 1. the sentinel survives ═════════════════════════════════════════════════════════════
IC = io.open(str(ROOT / "app/impact_calcs.py"), encoding="utf-8").read()
# the two writers are gone: exactly one place sets each global per branch
check("1: exactly one 'skipped' writer for _LAST_VAMP_TERMS",
      IC.count('globals()["_LAST_VAMP_TERMS"] = "skipped"') == 1)
_NONE_W = IC.count('globals()["_LAST_VAMP_TERMS"] = None')
check("1: the clobbering `else` that set it back to None is gone", _NONE_W == 1,
      str(_NONE_W) + " None-writer(s) left (1 = the except handler only)")
# and structurally: the None writer must sit inside an `except`, not an `else`
check("1: the sentinel branch falls straight into `else: try:` with no second writer",
      'globals()["_LAST_VAMP_CF_SKIPPED"] = "skipped"\n    else:\n        try:' in IC)
check("1: the None writer is reached only from `except`",
      re.search(r'except Exception:[^\n]*\n(\s*#[^\n]*\n)*\s*globals\(\)\["_LAST_VAMP_TERMS"\] = None',
                IC) is not None)

# ══ 2. [lift-ab] ═════════════════════════════════════════════════════════════════════════
BP = str(ROOT / "src/routing_optimiser/s4_search/band_projection.py")
bp = load("bp19ih", BP)
SRC = io.open(BP, encoding="utf-8").read()
check("2: the hard-coded 5% floor is gone", "_floor = 0.05" not in SRC)
check("2: the arms are interleaved", "for _flag in (True, False):" in SRC)
check("2: the arms are compared on their minima",
      "min(_ms[True]), min(_ms[False])" in SRC)
check("2: the bar is measured from within-arm spread",
      "_noise = max(_spr_on, _spr_off)" in SRC and "_floor = max(0.02, _noise)" in SRC)

# drive the real function against a fake projector with CONTROLLED timings
class FakeProj:
    """project_pop_numba returns a fixed answer; the clock is what we control."""
    def __init__(self, seq):
        self._ab_last = np.zeros((35, 4))
        self._gcode = np.zeros(4, np.int64)
        self._seq = list(seq); self._i = 0
    def project_pop_numba(self, pr):
        return np.ones((35, 3)), np.ones((35, 3))

def run_with(times):
    """`times` is consumed by perf_counter; returns the [lift-ab] note text."""
    it = iter(times)
    real = bp._time_mod.perf_counter
    notes = []
    bp._time_mod.perf_counter = lambda: next(it, 0.0)
    _pn = bp._pnote
    bp._pnote = lambda s: notes.append(s)
    try:
        r = bp.lift_ab_report(FakeProj([]), reps=3)
    finally:
        bp._time_mod.perf_counter = real
        bp._pnote = _pn
    return r, (notes[-1] if notes else "")

# 3 rounds x 2 arms, each timed sample consumes 2 perf_counter calls (t0, t1).
# ON samples 100/101/100 ms, OFF samples 200/300/210 ms -> min 0.100 vs 0.200 => 2.0x,
# ON spread 1%, OFF spread 50% -> bar 50%, and 100% clears it.
def pairs(ms_list):
    out, t = [], 0.0
    for ms in ms_list:
        out += [t, t + ms / 1000.0]; t += 10.0
    return out
seq = []
on, off = [100.0, 101.0, 100.0], [200.0, 300.0, 210.0]
for r in range(3):
    seq += pairs([on[r]])[:2]
    seq += pairs([off[r]])[:2]
res, note = run_with(seq)
check("2: speedup is min/min, not mean/mean",
      res is not None and abs(res["speedup"] - 2.0) < 1e-9,
      "" if res is None else f"{res['speedup']:.4f}x (mean/mean would be 2.35x)")
check("2: a 100% difference clears a 50% measured noise bar",
      res is not None and res["above_floor"] is True)
check("2: the note reports the measured noise, not '5%'",
      "MACHINE NOISE" in note and "50.0%" in note and "not a fixed 5%" in note)

# a REAL-looking lift that is inside the machine's own noise must NOT be called real
seq = []
on, off = [100.0, 140.0, 100.0], [113.0, 150.0, 115.0]
for r in range(3):
    seq += pairs([on[r]])[:2]
    seq += pairs([off[r]])[:2]
res2, note2 = run_with(seq)
check("2: a 13% difference inside a 40% noise bar is NOT called real",
      res2 is not None and res2["above_floor"] is False,
      "" if res2 is None else f"{res2['speedup']:.3f}x vs noise bar")
check("2: an inside-the-bar reading is not dressed up as impossible either",
      "does NOT clear it" in note2 and "WHICH CANNOT HAPPEN" not in note2)

# below 1.0x — the reading that started this — must never come back 'real'
seq = []
on, off = [110.0, 111.0, 110.0], [100.0, 101.0, 100.0]
for r in range(3):
    seq += pairs([on[r]])[:2]
    seq += pairs([off[r]])[:2]
res3, note3x = run_with(seq)
check("2: a sub-1.0x reading is NEVER called a real lift",
      res3 is not None and res3["above_floor"] is False and res3["impossible"] is True,
      "" if res3 is None else str(round(res3["speedup"], 3)) + "x, above_floor="
      + str(res3["above_floor"]) + ", impossible=" + str(res3["impossible"]))

# ══ 3. §12 — the live water-fill recipient rule ══════════════════════════════════════════
check("3: _cap_rows' recipient rule has no blocked-row clause",
      "recip = (W > 1e-12) & (~over) & (W < _cap - 1e-12)" in IC,
      "if this assert fails the live rule CHANGED and [deliv-fixed]'s verdict must be re-derived")

check("2: and it is named as a contaminated CLOCK, not a slower lift",
      "WHICH CANNOT HAPPEN" in note3x and "UNMEASURED" in note3x)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
