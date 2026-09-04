"""19kb - the moveable-M5 counter: hard right, one line, all dark ink.

Ben's instruction, on the tab-2 line reading
    Moveable M5 txn budget: 227,788 / 232,872 (selected RPGTs)

  * move it to the RIGHT side of the left column (the Risk Constraints panel);
  * keep the whole thing on ONE line;
  * put every part of it in dark ink.

TWO CSS PROPERTIES, BOTH NEEDED, and it is worth writing down why neither alone is enough:
`text-align:right` is what moves it; `white-space:nowrap` is what keeps it on one line.
Right-aligned text still wraps, and that string is long enough to break in two at 0.78rem in
this column. Alignment is done in the counter's own HTML rather than with a spacer column,
because a spacer's width would have to be re-guessed every time the number's digit count
changes.

WHAT 'ALL DARK INK' COST, recorded because it is a real trade and not a free tidy-up: the
remaining-budget number used to turn RED when it went negative. It no longer does. The signal
is NOT lost - the '⚠ growth constraints exceed the moveable M5 pool by N txns' line directly
beneath it still fires, still in red, and says in words what the colour said in one number.
Section 3 below pins that, so a later 'tidy the warnings too' cannot quietly remove the last
place this breach is visible.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "app"
T2 = (APP / "tab_2_routing_engine.py").read_text(encoding="utf-8")
sys.path.insert(0, str(APP))
sys.path.insert(0, str(ROOT / "src"))

FAIL = []


def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok:
        FAIL.append(n)


def code(src):
    """Source minus comment-only lines. This build's comments quote the very CSS the checks
    below look for ("`text-align:right` on the wrapper …"), so a raw substring scan would pass
    on the strength of its own explanation - the trap that bit three checks in 19jz and two in
    19ka. Comment-only lines are all this file's explanations are, so stripping them is enough."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))


T2C = code(T2)


_STMT_STARTS = ("_html = ", "_html += ", "_moveable_slot.markdown(")


def _block(src, needle, span=40):
    """The source of the WHOLE markdown statement containing `needle`.

    Walks BACK from the matching line to the statement that opens it, then forward to the
    closing paren. Both directions matter: these are multi-line implicit concatenations, so a
    style property and the text it styles routinely sit on different lines - the first version
    of this test searched forward only, found 2 of 3 colours, and failed for that reason
    rather than because anything was wrong. Line-slicing rather than the AST because `{_INK_}`
    is a FormattedValue: a literal-parts-only reader never sees the colour it interpolates."""
    _lines = src.splitlines()
    for _i, _l in enumerate(_lines):
        if needle not in _l:
            continue
        _start = _i
        for _j in range(_i, max(_i - span, -1), -1):
            if any(_lines[_j].lstrip().startswith(_p) for _p in _STMT_STARTS):
                _start = _j
                break
        for _k in range(_i, min(_i + span, len(_lines))):
            if _lines[_k].rstrip().endswith(")") or "unsafe_allow_html" in _lines[_k]:
                return "\n".join(_lines[_start:_k + 1])
        return "\n".join(_lines[_start:_i + span])
    return ""


# === 1. THE COUNTER: RIGHT, ONE LINE, DARK INK =========================================
check("1  dark ink is named once, as a constant, not repeated as a literal",
      '_INK_ = "var(--tav-ink)"' in T2C)

_c = _block(T2C, "Moveable M5 txn budget:")
check("1  the counter's markup was found", bool(_c))
check("1  it is right-aligned", "text-align:right" in _c)
check("1  it is held on one line", "white-space:nowrap" in _c)
check("1  every colour in it is dark ink",
      _c.count("{_INK_}") >= 3 and "#6b7280" not in _c and "#e63748" not in _c,
      "%d ink reference(s): the line, the number, and the '/ total' span" % _c.count("{_INK_}"))
check("1  the '(selected RPGTs)' suffix is still part of the same line",
      "(selected RPGTs)" in _c)

_ph = _block(T2C, "run once to populate")
check("1  the 'run once to populate' placeholder matches it",
      "text-align:right" in _ph and "white-space:nowrap" in _ph
      and "{_INK_}" in _ph and "#6b7280" not in _ph)

check("1  no grey text is left anywhere in this file", "#6b7280" not in T2,
      "the counter's grey was the last of it")


# === 2. THE SLOT IT LIVES IN ===========================================================
check("2  it still renders into the wide half of the VAMP-cap row",
      "_v1, _v3 = st.columns([2, 8])" in T2C and "_moveable_slot = _v3.empty()" in T2C,
      "right-alignment inside _v3 puts it on the panel's right edge")
check("2  the alignment is in the HTML, not a spacer column",
      "_v1, _v2, _v3" not in T2C,
      "a spacer's width would need re-guessing every time the digit count changes")


# === 3. THE RED BREACH SIGNAL SURVIVES =================================================
# The half of this change that can go wrong LATER rather than now: the counter's own red is
# gone by request, so this warning is the only place an over-budget state is still visible.
_w = _block(T2C, "growth constraints exceed the moveable M5 pool by")
check("3  the sentence that replaces the colour is still there", bool(_w))
_wblk = _block(T2C, "0.74rem")
check("3  the over-budget warning block is still RED",
      "#e63748" in _wblk, _wblk[:80])
check("3  ...and is right-aligned with the counter above it", "text-align:right" in _wblk)
check("3  ...but is NOT forced onto one line", "white-space:nowrap" not in _wblk,
      "it names a vampMid and a shortfall; nowrap would push it off the panel")
_nblk = _block(T2C, "non-M5 Txn constraint")
check("3  the non-M5 note is dark ink and right-aligned too",
      "{_INK_}" in _nblk and "text-align:right" in _nblk and "#6b7280" not in _nblk)


# === 4. IT ACTUALLY RENDERS, WITH REAL NUMBERS =========================================
# Seeding a cached-forecast dir is what makes this non-vacuous twice over: without it tab 2
# short-circuits to a placeholder, AND the counter needs that run's
# bin_rpgt_impact_export.csv to show figures rather than 'run once to populate'.
_cwd = os.getcwd()
try:
    os.chdir(str(ROOT))
    from streamlit.testing.v1 import AppTest
    _seed = ROOT / "data" / "outputs" / "SEP" / "TotalAV" / "visa"
    check("4  the seed forecast dir and its granular export exist",
          _seed.is_dir() and (_seed / "bin_rpgt_impact_export.csv").is_file())
    at = AppTest.from_file(str(APP / "streamlit_app.py"), default_timeout=420)
    at.session_state["pipeline_out_dir"] = str(_seed)
    at.session_state["forecast_settings"] = {"company": "TotalAV", "card_scheme": "visa",
                                             "month_var": "SEP", "month_0": "2026-09-01"}
    at.run()
    check("4  the app runs with NO exceptions", len(at.exception) == 0,
          "; ".join(str(e)[:200] for e in at.exception))
    _hits = [m.value for m in at.markdown if "Moveable M5" in m.value]
    check("4  exactly one counter is on the page", len(_hits) == 1, str(len(_hits)))
    _h = _hits[0] if _hits else ""
    check("4  it populated with real figures, not the placeholder",
          "run once to populate" not in _h and "txn budget:" in _h,
          _h[_h.find("Moveable"):_h.find("Moveable") + 70] if _h else "")
    check("4  the rendered block is right-aligned and nowrap",
          "text-align:right" in _h and "white-space:nowrap" in _h)
    check("4  the rendered block carries no grey and no red",
          "#6b7280" not in _h and "#e63748" not in _h)
finally:
    os.chdir(_cwd)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
