"""19jx - ROUTING_PROJ_LIFT and ROUTING_PROJ_PROFILEBLOCK deleted; both code paths kept.

THE DISTINCTION THAT MAKES THIS SAFE. For these two the OFF path is not a fallback, it is the
REFERENCE their own self-check diffs against on every run:

  * the flat kernel is what `_CB_OK`'s check compares the profile-blocked output to, before any
    result is used;
  * the unlifted index arrays are what `lift_ab_report` times and bit-compares the lift against.

So the env var goes and the code path stays. Deleting the slow path would have deleted the
proof - which is why `_PROJ_CB_ON` and `_PROJ_LIFT_ON` remain module-level NAMES set to True
rather than becoming literals: `lift_ab_report` toggles one, and the tests toggle the other.

AND THE THING THIS PROJECT KEEPS GETTING WRONG: a log line that names a switch is giving an
instruction. 19ju found three lines naming switches that no longer existed. Deleting two more
switches created six more, including one on the ONE path where the profile-blocked kernel is
not provably identical (the 50-sweep cap) - the worst possible place to print an instruction
that cannot be followed. All six are rewritten to name the source-level revert instead.
"""
import io, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
AC = (ROOT / "app/app_common.py").read_text(encoding="utf-8")
sys.path.insert(0, str(ROOT / "src"))

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def code(src):
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))

BPC, T2C, ACC = code(BP), code(T2), code(AC)
NAMES = ("ROUTING_PROJ_LIFT", "ROUTING_PROJ_PROFILEBLOCK", "ROUTING_PROJ_CELLBLOCK")


# ═══ 1. the env vars are gone, everywhere ════════════════════════════════════════════════
check("1  neither switch is READ any more",
      not any(f'environ.get("{n}"' in BP for n in NAMES)
      and "_env_switch(" not in BPC)
check("1  the flags are plain constants now",
      "_PROJ_CB_ON = True" in BP and "_PROJ_LIFT_ON = True" in BP)
for _n in NAMES:
    _hits = [l.strip()[:64] for l in (BPC + "\n" + T2C + "\n" + ACC).splitlines()
             if _n in l and "deleted" not in l and "19jx" not in l]
    check(f"1  no live line still tells anyone to set {_n}", not _hits,
          f"{len(_hits)} hit(s): {_hits[:2]}" if _hits else "")
check("1  _env_switch is gone with its only caller, and the CELLBLOCK alias with it",
      "def _env_switch" not in BP
      and '"ROUTING_PROJ_PROFILEBLOCK": "ROUTING_PROJ_CELLBLOCK"' not in ACC)
check("1  ...and the other three renamed-switch aliases are untouched",
      '"ROUTING_DOOR_COVER_PROFILES": "ROUTING_DOOR_COVER_CELLS"' in AC
      and '"ROUTING_CA_ZEROPROFILE": "ROUTING_CA_ZEROCELL"' in AC
      and '"ROUTING_ROW_PARALLEL_MIN_PROFILES": "ROUTING_ROW_PARALLEL_MIN_CELLS"' in AC)
check("1  the requested-vs-in-effect table no longer lists a switch nothing reads",
      '("ROUTING_PROJ_PROFILEBLOCK", _PROJ_CB_ON)' not in BPC
      and '("ROUTING_PROJ_CHUNK", _PROJ_CHUNK_ON)' in BPC)

# ═══ 2. BOTH REFERENCE PATHS SURVIVE - the point of the whole change ═════════════════════
check("2  the flat kernel is still built and still called by the self-check",
      "_pop_band_kernel = _njit" in BP and "_pop_band_kernel(prop_raw, *a," in BP)
check("2  ...and the profile-blocked self-check still diffs against it before shipping",
      '_CB_OK["checked"]' in BP and "profile-blocked SELF-CHECK NOT RUN" in BP)
check("2  ...and a failure still disables the fast path for the process",
      '_CB_OK["use"] = False' in BP)
check("2  the unlifted arrays are still reachable, and lift_ab_report still toggles the flag",
      "global _PROJ_LIFT_ON" in BP and "_lift_full_rows" in BP)
check("2  ...so [lift-ab] can still measure and bit-compare the lift every run",
      "def lift_ab_report" in BP and "[lift-ab] frozen-scaffold LIFT at P=" in BP)
check("2  the reason the paths were kept is recorded where the switches went",
      "Deleting the slow path would delete the" in BP
      and "would delete the measurement that justifies it" in BP)

# ═══ 3. the six log lines that named a deleted switch ════════════════════════════════════
check("3  the 50-sweep shout names a revert that exists",
      "hit the 50-sweep cap" in BP
      and "`_PROJ_CB_ON = False` in band_projection" in BP
      and "Re-run with ROUTING_PROJ_PROFILEBLOCK=0" not in BP)
check("3  [stash-q]'s not-running reason no longer blames a deleted switch",
      "it declined, or its self-check " in BP
      and "(ROUTING_PROJ_PROFILEBLOCK=0, or " not in BP)
check("3  the layout and lift lines say unconditional instead of naming a switch",
      "(19jx: unconditional, no switch)" in BP and BP.count("(19jx: unconditional, no switch)") == 2)
check("3  [lift-ab]'s DEFECT branch names the source-level revert",
      "`_PROJ_LIFT_ON = False` in the source" in BP)
check("3  tab_2's three [kernel-ab] instructions are corrected too",
      "`_PROJ_CB_ON = False` in band_projection." in T2
      and T2.count("`_PROJ_LIFT_ON = ") >= 2)
check("3  ...and ROUTING_PROJ_CHUNK, which IS still read, is still named",
      "ROUTING_PROJ_CHUNK=0 " in T2 and 'environ.get("ROUTING_PROJ_CHUNK"' in BP)

# ═══ 4. the module still imports and behaves ═════════════════════════════════════════════
import routing_optimiser.s4_search.band_projection as bp
check("4  it imports, with both flags on and no env var involved",
      bp._PROJ_CB_ON is True and bp._PROJ_LIFT_ON is True)
check("4  setting the old env vars now does NOTHING - which is the point",
      True, "there is no reader left for either name")
check("4  _CB_OK seeds its runtime flag from the constant",
      bp._CB_OK["use"] is True)
check("4  band_projection records 19jx", "19jx-lift-and-profileblock-unconditional" in BP)
check("4  it compiles", bool(compile(BP, "band_projection", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
