"""19iu - the run log, trimmed and tabled. Behavioural where it can be, source-level where not.

The load-bearing half of this test RUNS THE GA and reads its log, because most of these edits are
gates ("say nothing when there is nothing to say") and a gate that references an undefined
variable is a NameError in the middle of a 20-minute run. Asserting the absence of a line at
source level would not have caught that; running the search does.
"""
import importlib.util, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA_P = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def load(name, path):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m

# ── a small book, the same shape test_19if uses ──────────────────────────────────────────
N_ROW = 12
starts = np.array([0, 4, 8], np.int64)
counts = np.array([4, 4, 4], np.int64)
ctx = {
    "n_row": N_ROW, "n_mid": 3,
    "profile_starts": starts, "profile_counts": counts,
    "elig": np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1], float),
    "base": np.full(N_ROW, 0.25),
    "profile_vol": np.repeat(np.array([1000.0, 700.0, 450.0]), 4),
    "sr":   np.array([.90, .82, .75, .60, .88, .79, .71, .85, .93, .55, .68, .77]),
    "risk": np.array([.010, .020, .030, .050, .012, .024, .031, .018, .009, .060, .028, .015]),
    "mid_id": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], np.int64),
    "mid_rows": None, "vamp_cap": 0.02, "max_share": 0.97, "floor": 0.0,
}

mod = load("ga_19iu", GA_P)
p, meta = mod.problem_from_ctx(ctx, soft_cap_mult=1.0)
LOG = []
best, info = mod.run_fullmatrix_ga(p, reference_shares=meta["reference_kept"], pop_size=10,
                                   generations=6, elite=3, patience=99, seed=5, numba=False,
                                   log_fn=lambda *_a: LOG.append(" ".join(str(_x) for _x in _a)))
TXT = "\n".join(LOG)
check("the GA still runs end to end with every gate in place",
      best is not None and info.get("seconds") is not None,
      f"{len(LOG)} log line(s), {info.get('seconds', 0):.2f}s")

# ── 1. [cap-source]: silent when nothing is over the cap ────────────────────────────────
_cs = [l for l in LOG if "[cap-source]" in l]
check("[cap-source] says NOTHING on a run with nothing over the cap",
      not _cs, f"{len(_cs)} line(s): " + ("; ".join(_cs)[:120] if _cs else ""))
check("...and its loud paths are all still there in the source",
      all(_t in pathlib.Path(GA_P).read_text(encoding="utf-8")
          for _t in ("A SEED STAGE IS AT FAULT", "THE SEED IS CLEAN AND THE ROUND TRIP",
                     "NOT THE CAUSE", "PROVEN FOR THE CAP")))

# ── 2. the two fused-path self-checks: silent on a pass ─────────────────────────────────
check("no 'fused child SELF-CHECK PASSED' line",
      "fused child SELF-CHECK PASSED" not in TXT)
check("no 'fused softmax SELF-CHECK PASSED' line",
      "fused softmax SELF-CHECK PASSED" not in TXT)
check("...and both FAILURE messages are still in the source",
      all(_t in pathlib.Path(GA_P).read_text(encoding="utf-8")
          for _t in ("fused child SELF-CHECK FAILED", "fused softmax SELF-CHECK FAILED")))

# ── 3. the build tag is a line, not a paragraph ─────────────────────────────────────────
_bl = [l for l in LOG if "build " in l and "fullmatrix-ga" in l]
check("the build line names the newest tags and counts the rest",
      len(_bl) == 1 and len(_bl[0]) < 240 and "earlier tag(s)" in _bl[0],
      (_bl[0][:150] if _bl else "no build line"))
check("...and the run's shape moved to its own line",
      any("rows " in l and "profiles " in l and "evaluator " in l for l in LOG))
check("the full tag chain is still available to callers via info['__build__']",
      len(str(info.get("__build__", "")).split("+")) > 30,
      f"{len(str(info.get('__build__', '')).split('+'))} tag(s)")

# ── 4. the closing lines are sections ───────────────────────────────────────────────────
check("SEARCH and RESULT are labelled sections",
      "[fullmatrix-ga] SEARCH" in TXT and "[fullmatrix-ga] RESULT" in TXT)
for _f in ("candidates", "layout", "time", "success rate", "M5 breach", "engineering",
           "feasible", "improved"):
    check(f"...with a '{_f}' row", any(l.strip().startswith(_f) for l in LOG))
check("the old one-line 'done success rate ... feasible=' form is gone",
      "done success rate" not in TXT)

# ── 5. [gen-gap] / [eval-cost] are tables, and the drift line is gone ───────────────────
if "[gen-gap]" in TXT:
    check("[gen-gap] prints a column header and a rule",
          any("stage" in l and "ms/gen" in l and "share" in l for l in LOG)
          and any(l.strip().startswith("-" * 20) for l in LOG))
    check("the first-10 vs last-10 drift line is gone", "first-10 median" not in TXT)
    check("the start-up rows are a table with a 'per restart' column",
          any("start-up step" in l and "per restart" in l for l in LOG))
else:
    check("[gen-gap] ran in this fixture", False, "no [gen-gap] in the captured log")

# ── 6. source-level: everything outside genetic_fullmatrix ──────────────────────────────
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")

check("BREACH CONCENTRATION and SCOPED-vs-FROZEN are no longer called",
      "breach_concentration_report as _bcr" not in T2
      and "scoped_frozen_report as _sfr" not in T2
      and "THIS is the call site to restore" in T2)
check("...and both functions are still in exact_band_solver",
      all(_t in (ROOT / "src/routing_optimiser/s4_search/exact_band_solver.py")
          .read_text(encoding="utf-8")
          for _t in ("def breach_concentration_report(", "def scoped_frozen_report(")))
check("[seed-cap] is one line when clean and a table when not",
      "[seed-cap] \\u2713 no seed stage emits a share " in T2
      and "A SEED STAGE IS EMITTING SHARES" in T2)
check("the [frozen-scaffold] banner is gone", "[frozen-scaffold] how much" not in T2)
check("the 'settled diagnostic(s) NOT RUN' flush no longer prints",
      "settled diagnostic(s) NOT RUN" not in T2 and "del _settled_pend[:]" in T2)
# 19iv REPLACED this check: gating the line on `_mm != 0` was itself the bug (see section 7).
check("the [ef-mask] comparison is still COMPUTED, it just says nothing",
      "_mm = int(np.count_nonzero(_ef_dl != self._ef_ok))" in BP
      and "READ THIS AS HISTORY" not in BP)
# 19ix dropped the "vs the 19fy/19fz baseline" column: the ✓ half said "still as fast as it was
# made" on every run. The ⚠ half - a step that did NOT speed up - is still computed and still
# printed, which is the only thing that column could ever tell anyone.
check("[pbp-inside] is a table", "f\"{'step':<56}{'time':>8}{'share':>7}\"" in BP
      and "Ordered largest-first" not in BP)
check("...without the per-row baseline tick, but still able to warn",
      "\u2713 vs ~" not in BP and "EXPECTED ~" in BP)
check("the [band-build] index-width line only prints on a REFUSAL",
      "if _n_ref:" in BP
      and "array(s) narrowed to int32" not in "".join(
          l for l in BP.splitlines() if not l.strip().startswith("#")))
check("the aged-row hoist is a table", "what it is" in BP
      and "read\\n                   # [gen-gap]'s `eval` row" not in BP
      and "before concluding anything" not in BP)

# float32: the switch is gone and the drift is still measured
check("ROUTING_PROJ_FLOAT32 is not read anywhere any more",
      not [l for l in (BP + T2).splitlines()
           if "ROUTING_PROJ_FLOAT32" in l and not l.strip().startswith("#")])
check("...float32 is hardwired ON", "_PROJ_F32 = True" in BP)
check("...and what that COSTS is written down where it was removed",
      "float32 noise floor can no" in BP and "WHAT IS LOST" in BP)
check("the [proj-par] float32 paragraph is gone",
      "FLOAT32 PROJECTOR IS ON" not in BP)
check("[proj-config] is a table", "f\"{'setting':<24}value\"" in BP
      and "worst single band" in BP)

check("[cvp-timing] is a table", "f\"      {'step':<62}{'time':>8}{'share':>8}\"" in T2)
check("[never-worse] has rules and spacing",
      T2.count("f\"      {'-' * 68}\"") >= 2)
check("the WITHIN-band chain lines are ONE table",
      "band(s) WITHIN their " in T2 and "all stages agree" in T2)
check("the 'Enforcement OFF' line is gone",
      "Enforcement OFF: delivered split" not in T2)
check("[proj-memo]'s per-hit line is gone",
      "served from the projection already" not in T2)
check("[profiles] PART B is gated on the size of the discrepancy",
      "_pb_loud = _pb_sum > 0.5" in T2)
check("the GRANULAR PROFILE SAMPLES block is deleted",
      "GRANULAR PROFILE SAMPLES DELETED" in T2
      and "each row: gateway" not in T2)
check("the two post-engine auto-block lines are gone",
      "NOTHING TO CAP, and that is CORRECT" not in T2
      and "if _tot_cap:" in T2)
# 19kg: a FLOOR, not an equality - this is audit finding T1. 19kf legitimately added two
# branches (a no-measurement guard and an unreachable-case guard) and `== 5` failed on a
# CORRECT change, which is the opposite of what a test is for. What must never happen is a
# branch disappearing; adding one is fine and needs no test edit.
check("...and every auto-block WARNING branch survives",
      T2.count("[Warning] auto-block") >= 5,
      f"{T2.count('[Warning] auto-block')} warning branch(es)")
check("[muted] is one line",
      "settled line(s) held back across" in T2
      and "that is the LINE-BY-LINE" not in T2)

# ── 7. 19iv: THE BRANCHES A SMOKE TEST DOES NOT REACH ───────────────────────────────────
# Both 19iu bugs were in code paths that only run when a measurement EXISTS: the float32 drift
# rows (a NameError - `SG`/`DL` were locals of the patch script that wrote them) and the
# [ef-mask] gate (a ⚠ on a permanently non-zero historical count). Calling proj_config() with
# an empty projector - which is what the 19iu test did - reaches neither. So plant the
# measurements and call it.
from routing_optimiser.s4_search import band_projection as _bpv   # noqa: E402
_bpv._F32_OK.update(use=True, live={"at_P": 35, "dv": 0.5463, "dv_band": 2, "nb": 15,
                                    "dt": 1.725, "dt_band": 10, "dv_sum": 1.648,
                                    "dt_sum": 7.877, "dv_nover": 15, "dt_nover": 15})
_bpv._CB_OK.update(use=True, checked=True, sweeps=1)
_bpv._PROJ_PATH.update(seen={("profile-blocked", 35, True, False, 35, 35): 338},
                       calls=338, cap=64, nthr=16)
try:
    _pc = _bpv.proj_config()
    check("proj_config() runs with a float32 drift measurement present (19iv: NameError)",
          True, f"{len(_pc)} line(s)")
    check("...and the drift table has both rows",
          sum(1 for l in _pc if l.strip().startswith(("vamp", "txn"))) == 2,
          str([l.strip()[:24] for l in _pc if l.strip().startswith(("vamp", "txn"))]))
    check("...with the paths table filled in",
          any("profile-blocked" in l and "338" in l for l in _pc))
except Exception as _e:
    check("proj_config() runs with a float32 drift measurement present (19iv: NameError)",
          False, f"{type(_e).__name__}: {_e}")

check("[ef-mask] is ONE line, with no comparison against the pre-19ht mask (19iv/19iw)",
      "are floor-eligible." in BP
      and "\u26a0 exploration floor" not in BP
      and "floor-eligible rows differ between" not in BP
      and "_ef_mismatch = _mm" in BP)
check("the WITHIN-band table's band column fits a range band",
      "{'band':>30}" in T2 and "str(_wb)[:30]:>30" in T2)
check("...and a sub-unit drift prints as 0, not -0",
      "0.0 if abs(_wdr) < 0.5 else _wdr" in T2)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
