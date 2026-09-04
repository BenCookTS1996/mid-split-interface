"""19js - the [decode-cap] headline is deleted from the run log.

    [decode-cap] the max-share cap is now a PROPERTY OF THE DECODE, not a repair after it...

It has been true and unconditional since 19gu. A line that cannot differ between runs is not
telling a reader of a RUN anything - it is documentation, and this file already keeps that in
the block comment above the code it describes. Same reasoning 19hs used to delete three
paragraphs from this very block, and 19ip to delete a fourth.

THE TWO LINES BELOW IT SURVIVE, and the distinction is the point: they are not design
statements. One names the redistribution rule a reader needs in order to check the numbers,
the other says what reading to EXPECT and what a different reading would mean - which is an
instruction for reading this run, not a fact about the code. Both are de-indented, because
they were sub-paragraphs of a headline that no longer prints.

Checked at source level, which is how 19ip checked its own deletions: these are display facts
with no return value to assert on.
"""
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA = (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

# the deleted line, as it appeared in the 2026-09-04 10:58 log
GONE = "the max-share cap is now a PROPERTY OF THE DECODE"
check("1  the headline is no longer LOGGED", f'log("[decode-cap] {GONE}' not in GA
      and 'PROPERTY OF THE DECODE, not a repair after "' not in GA)
check("1  ...and no [decode-cap] log call carries that sentence at all",
      not any(GONE in _l and "log(" in _l for _l in GA.splitlines()))
check("1  the reason is recorded where the deletion is, not only in the commit",
      "19js: AND THE HEADLINE IS GONE TOO" in GA
      and "it cannot differ between runs" in GA)

# what must SURVIVE
check("2  the redistribution rule still prints - a reader needs it to check the numbers",
      'log("[decode-cap] THE MAX-SHARE RULE is delivery\'s: the excess goes to "' in GA)
check("2  ...naming its own subject, now that the headline is not above it",
      "THE MAX-SHARE RULE" in GA and "THE RULE is delivery's" not in GA)
check("2  both engineering-key paragraphs still print - each says what to EXPECT",
      GA.count('log("[decode-cap] THE ENGINEERING KEY') == 2)
check("2  ...on the armed branch and the unarmed one",
      "THE ENGINEERING KEY should read 0.0000 even though" in GA
      and "THE ENGINEERING KEY should now read 0.0000 for every candidate" in GA)

# no orphans: nothing left indented under a headline that no longer exists
_orphan = [_l for _l in GA.splitlines() if '"[decode-cap]    ' in _l]
check("3  no [decode-cap] line is still indented as a sub-paragraph", not _orphan,
      f"{len(_orphan)} orphan(s): {[_l.strip()[:60] for _l in _orphan]}" if _orphan
      else "the survivors were de-indented with the headline's removal")

# the block still exists and is still gated the same way
check("4  the block is untouched otherwise: same guard, same position",
      "# ── [decode-cap] 19gu: the cap, now that it is part of the decode" in GA
      and "if _DECODE_CAP:" in GA)
check("4  the 19gu block comment still holds the design statement that was deleted",
      "[decode-cap] 19gu: THE CAP IS A PROPERTY OF THE DECODE" in GA,
      "deleting it from the LOG is only safe because the source still says it")
check("4  the pre-search [decode-cap] self-check is NOT touched - it reports a MEASUREMENT",
      "[decode-cap] ✓ SELF-CHECK PASSED on the live seed" in GA
      and "[decode-cap] ⚠⚠ SELF-CHECK FAILED on the live seed" in GA)
check("5  genetic_fullmatrix records 19js", "19js-decodecap-headline-trim" in GA)
check("5  the module still compiles", bool(compile(GA, "genetic_fullmatrix", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
