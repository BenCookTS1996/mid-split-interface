"""19kd - both GA deltas are ON with no switch, and both report their state every run.

WHAT WENT WRONG, in one paragraph. `ROUTING_DELIV_DELTA` defaulted to "0". The 2026-09-04
11:47 run had it exported and the 15:25 run did not: same build, same budget, same 11,840
candidates, the same shipped answer (success rate 0.615322 in both), and 75 split(s)/s against
22. `_deliver_full` went 54.7 -> 682.9 ms/call - 12.5x on ONE step - while every other measured
step moved ~2.0x with the machine. About half of a 3.3x, thrown away by a variable nobody
remembered to set. Ben's instruction: ship it on, delete the switch.

TWO SWITCHES HAD TO GO, NOT ONE. The gather is guarded by `if _DLV["on"] and _DLT["on"]`, so
turning the DECODE delta off turns the DELIVERY delta off with it. Removing only
ROUTING_DELIV_DELTA would have left `ROUTING_EVAL_DELTA=0` able to cost the same 3.3x - the
exposure, not just the switch.

WHY THEY ARE NAMES AND NOT `True` LITERALS. The OFF path of each is the REFERENCE its own
self-check diffs against, and test_19ix / test_19iz run a whole search BOTH ways and assert the
shipped split is bit-identical. Inlining the constant would leave the slow path in place with
nothing ever comparing against it - deleting the proof while keeping the code, which is the
worst of both. `_EVAL_DELTA_ON` / `_DELIV_DELTA_ON` are module constants no env var reads and
no widget exposes; the two tests set them on a freshly-imported module.

AND THE REPORTING DEFECT THAT MADE THIS HARD TO FIND. `[deliv-delta]` was
`if _DLV["gathered"]: … elif _DLV["why"]:`, and with the switch defaulting off `why` stayed
EMPTY - nobody had asked, so there was nothing to explain. The single biggest determinant of
that run's throughput left NO trace in a 718-line log. Both reports are unconditional now.
"""
import ast
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
GA = ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py"
GA_SRC = GA.read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Comment-only lines removed. This build's comments NAME both deleted env vars, so a raw
    substring test would fail on the strength of its own explanation - the trap that has now
    bitten checks in 19jz, 19ka, 19kb, 19kc and this file's first draft."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


GAC, T2C = code(GA_SRC), code(T2)


def _live_strings(src):
    """Every string literal the module actually EVALUATES - docstrings excluded."""
    t = ast.parse(src)
    docs = set()
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = getattr(n, "body", None) or []
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                docs.add(id(b[0].value))
    return [n.value for n in ast.walk(t)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


# === 1. NO ENV VAR READS EITHER DELTA ===================================================
_env_names = [s for s in _live_strings(GA_SRC) if s.startswith("ROUTING_") and "DELTA" in s]
check("1  no live string in the GA names a DELTA env var", not _env_names, str(_env_names))
check("1  ...and neither read is left in the code",
      'os.environ.get("ROUTING_EVAL_DELTA"' not in GAC
      and 'os.environ.get("ROUTING_DELIV_DELTA"' not in GAC)
check("1  tab_2 no longer tells the reader a switch arms the delta",
      "ROUTING_DELIV_DELTA" not in T2C,
      "its comment said 'the GA gathers a delivery only when ROUTING_DELIV_DELTA=1'")


# === 2. ON BY DEFAULT, AS NAMES ==========================================================
import routing_optimiser.s4_search.genetic_fullmatrix as ga

check("2  the decode delta is on", ga._EVAL_DELTA_ON is True)
check("2  the delivery delta is on", ga._DELIV_DELTA_ON is True)
check("2  both are module-level constants",
      "_EVAL_DELTA_ON = True" in GAC and "_DELIV_DELTA_ON = True" in GAC)
check("2  the run function READS the names rather than inlining True",
      'bool(_EVAL_DELTA_ON)' in GAC and 'bool(_DELIV_DELTA_ON and _have_full' in GAC,
      "inlining would leave the slow path with nothing comparing against it")
check("2  the delivery delta still requires its WIRING as well",
      "_have_full and callable(deliver_rows_fn)" in GAC
      and "isinstance(deliver_map, dict)" in GAC,
      "[dlv-map] is a correctness precondition, not a preference - it stays")
check("2  the flags stay MUTABLE, so the self-check can still disable mid-run",
      '_DLT["on"] = False' in GAC and '_DLV["on"] = False' in GAC)
check("2  the build records 19kd", "19kd-deltas-unconditional" in ga.__build__)


# === 3. BOTH REPORT THEIR STATE EVERY RUN ===============================================
check("3  the decode report is unconditional",
      'if _DLT["gathered"] or _DLT["why"]:' not in GAC)
check("3  the delivery report has an else, not an elif on `why`",
      'elif _DLV["why"]:' not in GAC and "[deliv-delta] NOT USED" in GAC,
      "with the switch off, `why` was EMPTY and the line vanished entirely")
check("3  a not-used delivery report says the state has no switch",
      "no switch (19kd)" in GAC)
check("3  ...and distinguishes a test setting from missing wiring",
      "_DELIV_DELTA_ON is False in this module" in GAC
      and "the caller wired no subset-capable delivery" in GAC)


# === 4. IT ACTUALLY GATHERS, AND OFF STILL EXISTS TO COMPARE AGAINST ====================
# The full A/B - a whole search both ways, bit-compared - is test_19ix (decode) and test_19iz
# (delivery). This checks the thing THIS build changed: that a default import gathers.
import importlib.util

import numpy as np


def _fresh(name):
    sp = importlib.util.spec_from_file_location(name, str(GA))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


_m_default = _fresh("ga_19kd_default")
check("4  a DEFAULT import has both deltas armed - no env, no setup",
      _m_default._EVAL_DELTA_ON is True and _m_default._DELIV_DELTA_ON is True)
_m_off = _fresh("ga_19kd_off")
_m_off._DELIV_DELTA_ON = False
check("4  a test can still turn the delivery delta off on its own module copy",
      _m_off._DELIV_DELTA_ON is False and _m_default._DELIV_DELTA_ON is True,
      "fresh module per arm, so no arm can leak into another")
_m_off2 = _fresh("ga_19kd_off2")
_m_off2._EVAL_DELTA_ON = False
check("4  ...and the decode delta likewise", _m_off2._EVAL_DELTA_ON is False)
check("4  nothing in the APP sets either flag",
      "_EVAL_DELTA_ON" not in T2C and "_DELIV_DELTA_ON" not in T2C,
      "so a real run cannot reach the off path at all")

# and the guard the two deltas share is intact - this is why one switch was not enough
check("4  the gather is guarded on BOTH flags, which is why both had to go",
      'if _DLV["on"] and _DLT["on"] and _prov is not None:' in GAC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
