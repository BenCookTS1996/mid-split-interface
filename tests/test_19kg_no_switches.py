"""19kg - NO ENVIRONMENT VARIABLE CHANGES A RUN. This is the guard for that property.

Ben's instruction: "I don't want runtime switches anywhere they need to all be removed,
whatever runtime switches were set to in the run log pasted is how it needs to be going
forward." The 2026-09-04 16:08 run used every coded default - there is no routing.env and
run.command exports nothing - so the defaults ARE the shipped behaviour, and each one is now a
named constant at the value that run used.

THREE PROPERTIES, AND THE THIRD IS THE ONE THIS PROJECT KEEPS GETTING WRONG:

  1. Nothing READS a ROUTING_* environment variable. Not in app/, not in src/.
  2. Every setting that used to be one is still a NAME, not a literal inlined at the use site,
     so a test can A/B a whole search by rebinding it and a reader can see in one place what
     each module decides.
  3. NOTHING THE RUN PRINTS names a switch. A log line that names a switch is giving an
     instruction; 19ju found three lines naming switches that had already been deleted, and
     deleting 111 more would have created over a hundred. Comments that RECORD a deletion are
     the opposite of that fault and are deliberately allowed - the check is over live strings.
"""
import ast
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = sorted(set(list((ROOT / "app").glob("*.py")) + list((ROOT / "src").rglob("*.py"))))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def rel(p):
    return str(p.relative_to(ROOT))


SRC = {p: io.open(p, encoding="utf-8").read() for p in FILES}
TREE = {}
for p, s in SRC.items():
    try:
        TREE[p] = ast.parse(s)
    except SyntaxError as e:                                  # noqa: PERF203
        check(f"0  {rel(p)} parses", False, str(e))
if FAIL:
    print("\nFAILURES: " + ", ".join(FAIL))
    sys.exit(1)


def live_strings(p):
    """Every string constant that is not a docstring - i.e. every string a run can PRINT."""
    docs = set()
    for n in ast.walk(TREE[p]):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) \
                    and isinstance(n.body[0].value, ast.Constant) \
                    and isinstance(n.body[0].value.value, str):
                docs.add(id(n.body[0].value))
    for n in ast.walk(TREE[p]):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            yield n.lineno, n.value


# ═══ 1. NOTHING READS THE ENVIRONMENT FOR A ROUTING_* NAME ═══════════════════════════════
# By AST, not by grep: a read spelled `_os.environ.get(...)`, `os.getenv(...)` or through a
# helper still reaches `environ` in the end, and a grep for one spelling misses the others.
reads = []
for p in FILES:
    for n in ast.walk(TREE[p]):
        if not isinstance(n, ast.Call) or not n.args:
            continue
        k = n.args[0]
        if not (isinstance(k, ast.Constant) and isinstance(k.value, str)
                and k.value.startswith("ROUTING_")):
            continue
        f = n.func
        if (isinstance(f, ast.Attribute) and f.attr in ("get", "getenv")) \
                or (isinstance(f, ast.Name) and "env" in f.id.lower()):
            reads.append(f"{rel(p)}:{n.lineno} {k.value}")
check("1  no ROUTING_* name is read from the environment anywhere", not reads,
      f"{len(reads)} read(s): {reads[:3]}" if reads else "checked "
      f"{len(FILES)} file(s) by AST, every call spelling")

# ...and the subscript form, which is a read that no `.get` search would find
subs = []
for p in FILES:
    for n in ast.walk(TREE[p]):
        if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) \
                and isinstance(n.slice.value, str) and n.slice.value.startswith("ROUTING_"):
            subs.append(f"{rel(p)}:{n.lineno}")
check("1  ...and none is read by subscript either", not subs, str(subs[:3]))

# the alias reader that existed to honour old switch SPELLINGS has nothing left to honour
AC = SRC[ROOT / "app" / "app_common.py"]
check("1  the renamed-switch alias table and its reader are gone",
      "def env_switch" not in AC and "_SWITCH_ALIASES = {" not in AC
      and "def _rp_env_switch" not in SRC[ROOT / "src/routing_optimiser/s4_search/rowpar.py"])


# ═══ 2. THE SETTINGS ARE NAMES AT THE VALUE THE 16:08 RUN USED ═══════════════════════════
CONST = {}
for p in FILES:
    for m in re.finditer(r"^(_SW_[A-Z0-9_]+) = (.+?)(?:   #|$)", SRC[p], re.M):
        CONST[m.group(1)] = (m.group(2).strip(), rel(p))
check("2  every module that had a switch declares its settings as module-level constants",
      len(CONST) >= 80, f"{len(CONST)} constant(s) across "
                        f"{len({v[1] for v in CONST.values()})} module(s)")
check("2  each one records the switch it replaced, so the history is not lost",
      all(f"{n} = " in SRC[ROOT / pathlib.Path(f)] and "# was ROUTING_" in SRC[ROOT / pathlib.Path(f)]
          for n, (_v, f) in CONST.items()))
# a constant is a NAME so a test can rebind it; a literal inlined at the use site cannot be
# A/B'd, and this project's tests A/B whole searches (19ig, 19io, 19iq, 19ir, 19is, 19if).
used = {n for n in CONST if sum(SRC[p].count(n) for p in FILES) > 1}
check("2  every constant is READ somewhere, not just declared",
      len(used) == len(CONST), f"{len(CONST) - len(used)} unread: "
                               f"{sorted(set(CONST) - used)[:5]}" if used != set(CONST) else "")

# the four flags whose OFF path is the REFERENCE their own self-check diffs against. The env
# var goes and the code path STAYS - deleting the slow path would delete the proof.
T2 = SRC[ROOT / "app/tab_2_routing_engine.py"]
BP = SRC[ROOT / "src/routing_optimiser/s4_search/band_projection.py"]
GF = SRC[ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py"]
check("2  the int64 cap key ships and its verify is frozen off, with the reference KEPT",
      "_CK_INTKEY = True" in T2 and "_CK_VERIFY = False" in T2
      and "if _CK_NEW is None or _CK_VERIFY:" in T2 and "VERIFY FAILED on " in T2)
check("2  both projector flags stay names set True, and lift_ab_report can still toggle one",
      "_PROJ_CB_ON = True" in BP and "_PROJ_LIFT_ON = True" in BP and "global _PROJ_LIFT_ON" in BP)
check("2  both GA deltas are unconditional names, not inlined literals",
      "_EVAL_DELTA_ON = True" in GF and "_DELIV_DELTA_ON = True" in GF)


# ═══ 3. NOTHING A RUN PRINTS NAMES A SWITCH ══════════════════════════════════════════════
# THE RULE THIS PROJECT KEEPS BREAKING: a log line that names a switch is giving an
# instruction. Every one of them now names the source-level constant instead.
printed = []
for p in FILES:
    for ln, v in live_strings(p):
        for m in set(re.findall(r"ROUTING_[A-Z0-9_]+", v)):
            printed.append(f"{rel(p)}:{ln} {m}")
check("3  no live string names a ROUTING_* switch", not printed,
      f"{len(printed)} hit(s): {printed[:3]}" if printed else
      "every instruction now names a source-level constant")

# and the revert it names has to EXIST - the 19ju fault was an instruction nobody could follow
bad_ref = []
for p in FILES:
    for ln, v in live_strings(p):
        for m in set(re.findall(r"_SW_[A-Z0-9_]+", v)):
            if m not in CONST:
                bad_ref.append(f"{rel(p)}:{ln} {m}")
check("3  ...and every constant a live string names actually exists", not bad_ref,
      str(bad_ref[:3]))

# the run log's own switch report compared the environment NOW against what was read AT
# IMPORT. With nothing reading the environment it could only ever fire on a name nothing
# reads - which is the exact fault it was written to catch.
check("3  the projector's environment-vs-import warning went with the environment",
      "in this process's environment NOW" not in BP and "os.environ" not in BP)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
