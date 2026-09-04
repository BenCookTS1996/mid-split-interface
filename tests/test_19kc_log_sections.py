"""19kc - the run log numbers its own sections, and three lines go.

STRUCTURE. The log now has a two-level numbering scheme Ben asked for:

    ═══════════════ SECTION 4  SEARCH FOR THE SPLIT THAT SHIPS ════════════════
       ═════════ SUB-SECTION 4.1  BUILD THE PROJECTION SCAFFOLD ═════════
          … everything printed until the next banner, indented one level in …

THE NUMBERS ARE COMPUTED, NOT TYPED, and that is the point of the change rather than a
detail of it. Sections used to carry their own numeral in the string handed to `_stage()`
("① Fetch attempts/success data") and sub-steps carried "④·1  " in their label. Hand-written
numbers rot: `④·diag ROUTING PROFILES` was printing inside stage ③, because the numeral was
typed into a string a hundred lines away from the `_stage()` call that set it. `_sec` is
[section, sub-section]; `_stage()` bumps [0] and resets [1], `_substep()` bumps [1], and
nothing else writes it. Insert a stage and everything after it renumbers for free.

Cross-references were the other half. A line that names a DIFFERENT section now names it by
NAME ("the ROUTING PROFILES table", "SEARCH EFFICIENCY"), because a cross-reference by number
is precisely the thing that rots. The one line that names its OWN position - [scaffold-timing]
- reads it from `_sec_ref()` via a stamp taken when its banner printed.

THREE LINES REMOVED OR CORRECTED, all on Ben's instruction:

  * the available-gateway line loses its drop breakdown. "(from 501; dropped 387 inactive, 0
    PayPal, 77 other-brand vs 'TotalAV')" is a property of Master_MID_List.csv, not of the
    run, so it said the same thing every time.
  * [deliv-fixed] is retired - the LOG LINE and the PROBE. It re-proved a settled answer once
    per process and cost two extra full delivery transforms to do it. Its verdict stays as a
    comment at the code it is about, which is this project's rule for a retired measurement:
    a deleted one gets re-invented, a retired one with a visible verdict does not.
  * the [never-worse] header WAS out of date and Ben asked whether it was. It claimed "Two
    full projections … the run pays for two in total rather than three". The [proj-memo] half
    is still true; the count is not. [nw-skip] means the seed is not projected at all when the
    GA already delivers 0 breach, and on the 2026-09-04 15:25 run that is what happened - one
    [proj-memo] line and [nw-skip] saying the seed was skipped - so the run paid for ONE while
    the header announced two. It states the RULE now, because the count is not knowable at the
    point the line prints: the branch depends on a breach not yet computed.

19ke IS COVERED HERE TOO (section 7), rather than in a file of its own: it only changes how
much air sits around the banners this test already builds and renders, and the runtime harness
in section 3 is exactly what proves it. Grep 19ke to find it.
"""
import ast
import io
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
T2 = (APP / "tab_2_routing_engine.py").read_text(encoding="utf-8")
T2L = T2.split("\n")

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Source minus comment-only lines. This build's comments quote every string it deleted -
    "④·1", the drop breakdown, the old [never-worse] wording - so a raw substring scan would
    pass or fail on its own explanation. That trap has now bitten in 19jz, 19ka and 19kb."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


T2C = code(T2)


# === 1. NO NUMBER IS TYPED BY HAND ANY MORE =============================================
# code() strips comment-ONLY lines; circled numerals also live in trailing comments and in
# docstrings, and none of those reach the log. So this asks the question that matters - is one
# PRINTED - by pulling every string literal the module actually evaluates.
def _live_strings(src):
    _t = ast.parse(src)
    _docs = set()
    for _n in ast.walk(_t):
        if isinstance(_n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            _b = getattr(_n, "body", None) or []
            if _b and isinstance(_b[0], ast.Expr) and isinstance(_b[0].value, ast.Constant) \
                    and isinstance(_b[0].value.value, str):
                _docs.add(id(_b[0].value))
    return [_n.value for _n in ast.walk(_t)
            if isinstance(_n, ast.Constant) and isinstance(_n.value, str)
            and id(_n) not in _docs]


_LIVE = _live_strings(T2)
_circled = [_s for _s in _LIVE if any(_c in _s for _c in "①②③④⑤")]
check("1  no circled numeral is PRINTED anywhere in the run log",
      not _circled, "offenders: %r" % (_circled[:3],))
check("1  _sec is the single source of the numbering",
      "_sec = [0, 0]" in T2C and "_sec[0] += 1" in T2C and "_sec[1] += 1" in T2C)
check("1  a section RESETS the sub-section counter",
      "_sec[1] = 0" in T2C, "so sub-sections number from 1 within each section")
_writers = T2C.count("_sec[0] +=") + T2C.count("_sec[1] +=") + T2C.count("_sec[1] = 0")
check("1  ...and nothing else writes it", _writers == 3, "%d writer(s)" % _writers)

_stages = [l.split('_stage("')[1].split('"')[0] for l in T2L if '_stage("' in l]
check("1  all five sections found", len(_stages) == 5, str(_stages))
_subs = [l.split('_substep("')[1].split('"')[0] for l in T2L if '_substep("' in l]
check("1  all five sub-sections found", len(_subs) == 5, str(_subs))
check("1  no label carries a hand-typed prefix",
      not any("·" in _x or _x[:1].isdigit() for _x in _stages + _subs))


# === 2. THE BANNERS ====================================================================
check("2  a section prints a SECTION banner",
      '_banner(f"SECTION {_sec[0]}  {name}")' in T2C)
check("2  a sub-section prints a SUB-SECTION banner, indented",
      '"   " + _banner(f"SUB-SECTION {_sec[0]}.{_sec[1]}  {label}"' in T2C)
check("2  the sub-section banner is NARROWER than the section rule",
      "width=73" in T2C, "so the nesting is visible without reading the words")
check("2  sub-section CONTENT is indented one level in",
      "_sub_depth[0] = 3" in T2C and '" " * _sub_depth[0] + msg' in T2C)
check("2  the finish line names the same section number it opened",
      T2C.count('f"\\u2713 SECTION {_sec[0]}') == 2,
      "both _stage's roll-over and _stage_end")


# === 3. IT ACTUALLY NUMBERS CORRECTLY ==================================================
# Lift the real _sec / _banner / _stage / _substep out of render() and run them. Reading the
# source only proves the shape; this proves 4.1 says 4.1.
def _grab(first, last):
    _a = next(i for i, l in enumerate(T2L) if first in l)
    _b = next(i for i, l in enumerate(T2L) if last in l and i > _a)
    return T2L[_a:_b]


_body = (_grab("_sub_depth = [0]", "# [FN-304b]")
         + _grab("def _stage(name):", "# 19dv — RETIRING A SETTLED")
         + _grab("def _substep(label):", "# [FN-307]"))
_ns = {}
exec(  # noqa: S102 - tab_2's own log machinery, run verbatim
    "class _PT:\n    def time(self): return _T[0]\n"
    "_pt = _PT()\n_T = [0.0]\nOUT = []\n"
    "def log(m):\n"
    "    OUT.append((' ' * _sub_depth[0] + m) if (_sub_depth[0] and str(m).strip()) else m)\n"
    '_stage_state = {"name": None, "t": 0.0}\n'
    + textwrap.dedent("\n".join(l[16:] for l in _body)), _ns)

for _nm in _stages:
    _ns["_stage"](_nm)
    if _nm == "SEARCH FOR THE SPLIT THAT SHIPS":
        for _sn in _subs:
            _ns["_substep"](_sn)
            if _sn == "BUILD THE PROJECTION SCAFFOLD":
                check("3  _sec_ref() inside the first sub-section reads 4.1",
                      _ns["_sec_ref"]() == "4.1", _ns["_sec_ref"]())
_ns["_stage_end"]()
_ALL = list(_ns["OUT"])                       # 19ke: blanks INCLUDED - section 7 counts them
_out = [l for l in _ns["OUT"] if str(l).strip()]

check("3  sections are numbered 1..5 in order",
      [l for l in _out if l.startswith("═") or "SECTION 1 " in l] and
      all(f"SECTION {i}  {_stages[i - 1]}" in "".join(_out) for i in range(1, 6)),
      "search is SECTION %d" % (_stages.index("SEARCH FOR THE SPLIT THAT SHIPS") + 1))
for _i, _sn in enumerate(_subs, 1):
    check("3  sub-section 4.%d is %r" % (_i, _sn),
          any(f"SUB-SECTION 4.{_i}  {_sn}" in l for l in _out))
check("3  every sub-section banner is indented 3 and the section banners are not",
      all(l.startswith("   ═") for l in _out if "SUB-SECTION" in l)
      and all(l.startswith("═") for l in _out if "SECTION" in l and "SUB-" not in l
              and not l.startswith("✓")))
check("3  a section's finish line carries its own number",
      any(l.startswith("✓ SECTION 5  ") for l in _out), str(_out[-1])[:70])
# Two KINDS of banner, deliberately different widths - that difference is the nesting cue.
# What must hold is that each kind is internally consistent, so they line up down the page.
_secw = {len(l) for l in _out if "SECTION" in l and "SUB-" not in l and not l.startswith("✓")}
_subw = {len(l) for l in _out if "SUB-SECTION" in l}
check("3  every SECTION banner is the same width", len(_secw) == 1, str(_secw))
check("3  every SUB-SECTION banner is the same width", len(_subw) == 1, str(_subw))
check("3  ...and the sub-section rule is the narrower of the two",
      _secw and _subw and max(_subw) < max(_secw),
      "section %s vs sub-section %s" % (max(_secw), max(_subw)))


# === 4. THE THREE LINES ================================================================
check("4  the available-gateway line is just the count",
      'log(f"   {len(_cand)} available gateway(s)")' in T2C
      and "dropped {_drop_inact} inactive" not in T2C)
check("4  ...and the counters it dropped are no longer computed for a log line only",
      T2C.count("_drop_inact") == T2C.count("_drop_inact += 1") + 1,
      "still counted for the loop's own control flow, no longer printed")

check("4  [deliv-fixed] prints nothing", "[deliv-fixed]" not in T2C)
check("4  ...its probe is gone too, not just the line",
      "_fm_deliv_serial(_fx1)" not in T2 and "_fx2 - _fx1" not in T2,
      "two extra full delivery transforms per process")
check("4  ...but its VERDICT survives as a comment at the code it is about",
      "RETIRED 19kc" in T2 and "`_cap_rows` is where to change it" in T2
      and "FAITHFUL" in T2,
      "this project's rule: a deleted measurement gets re-invented, a retired one does not")

check("4  the [never-worse] header no longer claims a count it cannot know",
      "Two full projections" not in T2C
      and "pays for two in total rather than three" not in T2C)
check("4  ...and states the rule instead, naming both mechanisms",
      "At most two projections" in T2C and "[proj-memo])" in T2C
      and "[nw-skip])" in T2C)


# === 5. CROSS-REFERENCES NAME SECTIONS, NOT NUMBERS ====================================
check("5  the efficiency readouts are named, not numbered",
      T2C.count("SEARCH EFFICIENCY") >= 3 and "④ EFFICIENCY" not in T2C)
check("5  the ROUTING PROFILES cross-reference is by name",
      "row of the ROUTING PROFILES " in T2C)
check("5  the two ·diag headers no longer carry a number",
      '"   [diag] ROUTING PROFILES assembled:"' in T2C
      and '"   [diag] DATA SHAPES after pre-processing/filters:"' in T2C,
      "one of them was printing '④·diag' from inside section 3")
check("5  [scaffold-timing] names its own sub-section from _sec_ref()",
      '_sctm["sec"] = _sec_ref()' in T2C
      and 'f"section {_st[\'sec\']} "' in T2C
      and "[scaffold-timing] ④" not in T2C)


# === 7. 19ke: THE AIR AROUND A BANNER =================================================
# Ben asked for more spacing between sections. A "blank" line still carries the timestamp
# prefix, so it reads as a rule rather than as emptiness - which is why more of them help.
check("7  the spacing is ONE knob, not repeated log() calls",
      all(_k in T2C for _k in ("_SEC_GAP = ", "_SEC_GAP_AFTER = ",
                               "_SUB_GAP = ", "_SUB_GAP_AFTER = "))
      and "_gap(_SEC_GAP)" in T2C and "_gap(_SUB_GAP)" in T2C)
_SECG = int(T2C.split("_SEC_GAP = ")[1].split()[0])
_SECGA = int(T2C.split("_SEC_GAP_AFTER = ")[1].split()[0])
_SUBGA = int(T2C.split("_SUB_GAP_AFTER = ")[1].split()[0])
check("7  a section gets more than the single blank line it used to",
      _SECG >= 2, "_SEC_GAP = %d" % _SECG)


def _runs_before(lines, pred):
    """How many consecutive blanks sit immediately before each line matching `pred`."""
    _out = []
    for _i, _l in enumerate(lines):
        if not pred(_l):
            continue
        _c, _j = 0, _i - 1
        while _j >= 0 and not str(lines[_j]).strip():
            _c += 1
            _j -= 1
        _out.append(_c)
    return _out


def _blanks_after(lines, pred):
    _out = []
    for _i, _l in enumerate(lines):
        if not pred(_l):
            continue
        _c, _j = 0, _i + 1
        while _j < len(lines) and not str(lines[_j]).strip():
            _c += 1
            _j += 1
        _out.append(_c)
    return _out


_is_sec = lambda l: str(l).startswith("\u2550")
_before_sec = _runs_before(_ALL, _is_sec)
check("7  every SECTION banner has _SEC_GAP blanks above it",
      _before_sec and all(_c == _SECG for _c in _before_sec), "runs: %s" % _before_sec)
_before_tick = _runs_before(_ALL, lambda l: str(l).startswith("\u2713 SECTION"))
check("7  every \u2713 finish line has a blank above it, so it ends the section",
      _before_tick and all(_c >= 1 for _c in _before_tick), str(_before_tick))
check("7  a SECTION banner is followed by _SEC_GAP_AFTER blank(s)",
      all(_c >= _SECGA for _c in _blanks_after(_ALL, _is_sec)))
check("7  a SUB-SECTION banner is followed by _SUB_GAP_AFTER blank(s)",
      all(_c >= _SUBGA for _c in _blanks_after(_ALL, lambda l: "SUB-SECTION" in str(l))))
_i2 = next(_i for _i, _l in enumerate(_ALL) if _is_sec(_l) and "SECTION 2 " in str(_l))
_boundary = sum(1 for _l in _ALL[max(_i2 - 4, 0):_i2] if not str(_l).strip())
check("7  a section BOUNDARY is several spacer lines wide now, not one",
      _boundary >= 3,
      "%d blank(s) in the 4 lines above the SECTION 2 banner" % _boundary)

# THE SAFETY PROPERTY. A blank has indent 0, which ENDS the [muted] gate's sticky family.
# Right at a boundary (the block is over), wrong inside one - so it is asserted, not assumed.
check("7  a blank line has indent 0, so it terminates a muted family run",
      any("return len(msg) - len(msg.lstrip" in l for l in T2L),
      "scattering blanks INSIDE a block would orphan detail lines from their header")


# === 6. THE FILE STILL LOADS ===========================================================
check("6  tab_2 compiles", bool(compile(T2, "tab_2", "exec")))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))
import app_common  # noqa: F401  - the app's shared module must still import

check("6  the app's shared module still imports", True)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
