"""19kf - the auto-block warning can no longer diagnose a comparison it never ran.

WHAT IT PRINTED, on the 2026-09-04 16:08 run:

    [Warning] auto-block: NO SPLIT ROW matched a flagged (bank, gateway) pair — this IS a
              key mismatch. 17 flagged pair(s) vs 0 split row(s).
       flagged pairs (first 5): (unavailable)
       split (bin, gateway) (first 5): (unavailable)
       COMPARE THE TWO LISTS. …

Three things wrong at once. It asserted a key mismatch. It reported `0 split row(s)`, which is
not what a mismatch looks like - a mismatch has rows that failed to match. And it told the
reader to compare two lists it then declined to print.

THE CAUSE IS ONE GLOBAL WITH THREE WRITERS. `LAST_BLOCKED_CAP_STATS` is cleared and rewritten
by every `_apply_blocked_caps` call, and tab_2 calls it from three places. The post-engine
reader keyed its diagnosis on `_bs.get("matched", 0) == 0` - and an EMPTY dict reads as 0
exactly like a measured zero. So "nobody measured anything" and "we compared and found nothing"
were the same sentence, and only one of them was ever true.

THE FIX IS A POSITIVE MARKER, NOT BETTER WORDING. Every branch of `_apply_blocked_caps` now
stamps `case` ("applied" / "no-match" / "precondition"), a per-call `seq`, and a `site` label.
The caller records the sequence before its own loop, so it can tell ITS stats from the last
call's; the key-mismatch branch is keyed on `case == "no-match"`, the one branch where keys were
actually compared and therefore the only one that can show the evidence; and an empty or stale
dict now gets its own branch that says, in as many words, that this is NOT a key mismatch.

Evidence is no longer withheld either: the precondition branch records the flagged pairs (they
are always available) and samples the split's keys whenever `bin` and `gateway` exist - a split
missing only `share` still has the keys a reader came to compare. Where sampling is genuinely
impossible the value SAYS so, so an absence can never again be read as a mismatch.

WHAT THIS TEST DOES NOT COVER: the caller's branch chain lives inside `render()` and cannot be
called directly, so section 3 proves its STRUCTURE from the AST - that an empty dict cannot
reach the mismatch branch - rather than executing it. Section 2 drives the real
`_apply_blocked_caps` through every state it has.
"""
import ast
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
T2 = (APP / "tab_2_routing_engine.py").read_text(encoding="utf-8")
AC = (APP / "app_common.py").read_text(encoding="utf-8")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Comment-only lines removed - this build's comments QUOTE the wording they deleted."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


T2C, ACC = code(T2), code(AC)

import app_common as ac

_GOOD = pd.DataFrame({"bin": ["b1", "b1", "b2"], "gateway": ["gwA", "gwB", "gwA"],
                      "share": [0.9, 0.1, 0.5], "rpgt": ["r"] * 3, "currency": ["usd"] * 3})


def _run(split, pairs, site="probe"):
    ac._apply_blocked_caps(split, pairs, 0.01, site=site)
    return dict(ac.LAST_BLOCKED_CAP_STATS)


# === 1. EVERY CALL LEAVES A STAMP =======================================================
_STATES = [
    ("a real match",        _GOOD.copy(),                     {("b1", "gwa")},   "applied"),
    ("a real key mismatch", _GOOD.copy(),                     {("zz", "nope")},  "no-match"),
    ("split is empty",      _GOOD.iloc[0:0].copy(),           {("b1", "gwa")},   "precondition"),
    ("split is None",       None,                             {("b1", "gwa")},   "precondition"),
    ("missing `share`",     _GOOD.drop(columns=["share"]),    {("b1", "gwa")},   "precondition"),
    ("no blocked pairs",    _GOOD.copy(),                     set(),             "precondition"),
]
_seen = []
for _lbl, _sp, _pr, _want in _STATES:
    _st = _run(_sp, _pr, site="probe:" + _lbl)
    _seen.append(_st)
    check("1  %-20s -> case=%r" % (_lbl, _want), _st.get("case") == _want,
          "got %r" % _st.get("case"))
check("1  every state stamps a seq and the call site",
      all(_s.get("seq") and _s.get("site", "").startswith("probe:") for _s in _seen))
check("1  seq is strictly increasing, so a reader can tell calls apart",
      [_s["seq"] for _s in _seen] == sorted({_s["seq"] for _s in _seen})
      and len({_s["seq"] for _s in _seen}) == len(_seen),
      str([_s["seq"] for _s in _seen]))


# === 2. THE EVIDENCE IS NEVER WITHHELD ==================================================
_mismatch = _run(_GOOD.copy(), {("zz", "nope")})
check("2  a REAL mismatch shows both lists",
      isinstance(_mismatch.get("sample_pairs"), list) and _mismatch["sample_pairs"]
      and isinstance(_mismatch.get("sample_split"), list) and _mismatch["sample_split"],
      "pairs=%s split=%s" % (_mismatch.get("sample_pairs"), _mismatch.get("sample_split")))
check("2  ...and reports the rows it compared against",
      int(_mismatch.get("split_rows", 0)) == len(_GOOD),
      "%s row(s) - a mismatch HAS rows; 0 rows is a different problem"
      % _mismatch.get("split_rows"))

_noshare = _run(_GOOD.drop(columns=["share"]), {("b1", "gwa")})
check("2  a split missing only `share` STILL shows its keys",
      isinstance(_noshare.get("sample_split"), list) and _noshare["sample_split"],
      "%s - the keys are what the reader came to compare" % (_noshare.get("sample_split"),))
check("2  ...and that sample would have shown the keys DO match",
      ("b1", "gwa") in (_noshare.get("sample_split") or []),
      "so nobody chases a naming problem that does not exist")

_nosplit = _run(None, {("b1", "gwa")})
check("2  with no split at all, the flagged pairs are still shown",
      isinstance(_nosplit.get("sample_pairs"), list) and _nosplit["sample_pairs"])
check("2  ...and the split side SAYS why it is absent, rather than '(unavailable)'",
      isinstance(_nosplit.get("sample_split"), str)
      and "precondition" in _nosplit["sample_split"],
      repr(_nosplit.get("sample_split")))
check("2  the word '(unavailable)' is gone from the caller entirely",
      "(unavailable)" not in T2C)


# === 3. AN EMPTY DICT CANNOT REACH THE MISMATCH BRANCH ==================================
# The chain lives inside render(), so this proves its STRUCTURE rather than running it.
# Match on the node's OWN TEST, not on its source segment: `ast.walk` is breadth-first and an
# enclosing `if` renders every nested branch inside its own segment, so a segment search finds
# some outer guard 15,000 lines up.
#
# And slice the source by LINE NUMBER rather than calling `ast.get_source_segment` per node: that
# helper re-splits the whole file on every call, which over ~1,800 If nodes in a 17,000-line file
# is quadratic - the first version of this test did exactly that and hung until the timeout.
_LINES = T2.splitlines()


def _seg(node):
    return "\n".join(_LINES[node.lineno - 1:node.end_lineno])


_tree = ast.parse(T2)
_chain = None
for _n in ast.walk(_tree):
    if isinstance(_n, ast.If) and "_bs_seq0" in _seg(_n.test):
        _chain = _n
        break
check("3  the auto-block branch chain was found in the AST", _chain is not None)


def _tests(node):
    """Every branch condition of an if/elif chain, in order, as source."""
    _out, _cur = [], node
    while isinstance(_cur, ast.If):
        _out.append(_seg(_cur.test))
        _cur = _cur.orelse[0] if (len(_cur.orelse) == 1
                                  and isinstance(_cur.orelse[0], ast.If)) else None
    return _out


_conds = _tests(_chain) if _chain else []
check("3  the FIRST branch is the 'no measurement' guard",
      _conds and ("not _bs" in _conds[0] and "_bs_seq" in _conds[0]),
      _conds[0][:88] if _conds else "")
_mm = [_c for _c in _conds if "_bs_case" in _c and "no-match" in _c]
check("3  the mismatch branch is keyed on case == 'no-match'", len(_mm) == 1, str(_conds))
check("3  ...and NOT on `matched == 0`, which an empty dict also satisfies",
      not any(_c.strip() == "_bs_m == 0" and "case" not in _c for _c in _conds[:2]),
      "an absent measurement must not be able to reach it")
check("3  the no-measurement branch says it is NOT a key mismatch",
      "THIS IS NOT A KEY MISMATCH" in T2C)
check("3  ...and says whether the cap pass ran at all",
      "the cap pass was NEVER CALLED" in T2C and "_bs_calls" in T2C)
check("3  the caller records the sequence BEFORE its own loop",
      "_bs_seq0 = int(getattr(_acbs0" in T2C
      and T2C.index("_bs_seq0 = ") < T2C.index("for v in variations:"))
check("3  all three call sites are labelled",
      T2C.count("site=\"") >= 2 and 'site="post-engine dials"' in T2C
      and 'site="reconcile (_ga_gran)"' in T2C)


# === 4. NOTHING ABOUT THE ARITHMETIC MOVED ==============================================
_a = _GOOD.copy()
_out_a, _n_a = ac._apply_blocked_caps(_a, {("b1", "gwa")}, 0.01, site="x")
_out_b, _n_b = ac._apply_blocked_caps(_GOOD.copy(), {("b1", "gwa")}, 0.01)   # no `site` kwarg
check("4  `site` is optional and changes no answer",
      _n_a == _n_b
      and np.array_equal(np.asarray(_out_a["share"], float),
                         np.asarray(_out_b["share"], float)))
check("4  the capped row is at the floor and the profile still sums to 1",
      abs(float(_out_a["share"].iloc[0]) - 0.01) < 1e-12
      and abs(float(_out_a[_out_a["bin"] == "b1"]["share"].sum()) - 1.0) < 1e-12,
      "shares: %s" % [round(float(x), 6) for x in _out_a["share"]])
check("4  a mismatch still returns the split untouched",
      int(ac._apply_blocked_caps(_GOOD.copy(), {("zz", "nope")}, 0.01)[1]) == 0)
check("4  both files compile",
      bool(compile(T2, "tab_2", "exec")) and bool(compile(AC, "app_common", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
