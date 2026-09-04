"""19jw - fifteen band-build / dispatch / delivery-cap log lines, simplified.

The rule applied throughout: keep what can DIFFER between runs - the counts, the verdicts, the
warnings - and drop what cannot. Two of the fifteen go silent, and one gains a word it needed.

  * [vconst-frozen] printed a count the `aged-row hoist` table directly above already carries on
    its `frozen-origin` row, plus a sentence explaining what the whole table is about. SILENT
    when it applies. Its NOT-APPLIED branch keeps every word: that one is actionable.
  * `profile-blocked layout built` printed the SAME SENTENCE TWICE with different numbers,
    because two layouts are built per run - the lifted one the search uses and the unlifted one
    the self-check diffs against. It now says which is which. `froz.size == 0` is the unlifted
    one by construction.
  * The `candidate-parallel projection ON/OFF` notes restated what `[proj-config] PATHS TAKEN`
    tabulates for every dispatch of the run. What they add that the table cannot is the REASON,
    so that is all they say now.
  * [blk-fill]'s two sampled lines become one, and NOTHING when the water-fill put nothing back
    - which is every run so far. There is no lift to apportion between avoidable and
    unavoidable when the lift is zero.

The BAD branches are untouched everywhere: an impossible [lift-ab] ratio, a bit-identity
failure, a non-zero still-over-cap count, and an unidentified frozen class all keep their full
wording. A line that only fires when something is wrong is not noise.
"""
import io, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")

def code(src):
    """Source with comment-only lines removed, so a comment that QUOTES a deleted log phrase
    (to record what went and why) does not read as the phrase still being printed."""
    return "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))

BPC, T2C = None, None

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

BPC, T2C = code(BP), code(T2)


# ═══ 1. [vconst-frozen]: silent when it applies, loud when it does not ═══════════════════
check("1  the applied case no longer prints",
      "19fs added the THIRD class" not in BPC
      and "summed once instead of per candidate" not in BPC,
      "checked against the source with comments stripped - the comment that records the "
      "deletion quotes the phrase, and should")
check("1  ...and it is gated on the failure, not on the success",
      'if not _h["frozen_known"]:' in BP and "[vconst-frozen] NOT APPLIED" in BP)
check("1  the NOT-APPLIED branch keeps its instruction",
      "set_lift_incidence() " in BP and "if this line \"\n" in BP or "that call is not happening" in BP)
check("1  the count it dropped is still on the hoist table",
      "'frozen-origin'" in BP or "frozen-origin" in BP)

# ═══ 2. the two layouts are told apart ══════════════════════════════════════════════════
check("2  each layout says WHICH it is",
      "(LIFTED - this is the one the search runs)" in BP
      and "(unlifted, the self-check's reference)" in BP)
check("2  ...decided by the frozen count, which is 0 on the unlifted one by construction",
      'if froz.size else' in BP)
check("2  the settled design prose is gone",
      "an L1-sized working set" not in BP and "~15 passes over the" not in BP)
check("2  ...but the bit-identity claim and the switch survive",
      "Bit-identical, self-checked against the flat kernel" in BP
      and "ROUTING_PROJ_PROFILEBLOCK=0 reverts" in BP)

# ═══ 3. dispatch notes: the REASON only ═════════════════════════════════════════════════
check("3  the ON/OFF note is one clause plus the reason",
      'f"candidate-parallel {_why} (P={P}, lanes={_lanes}, {nthr} numba thread(s))"' in BP)
check("3  ...and no longer restates what the PATHS TAKEN table tabulates",
      "scaffold nR={len(self._gcode):,}). Bit-identical either" not in BP)
check("3  ...while PATHS TAKEN still tabulates it", "PATHS TAKEN" in BP)
check("3  the 187-vs-180 measurement that settled the P<=1 case is a comment now",
      "is pure pool overhead" in BP
      and "(measured 187 vs 180 ms)\")" not in BP)
check("3  the self-check verdict is shorter and still a verdict",
      "candidate-parallel SELF-CHECK PASSED: serial == " in BP
      and "bit for bit on vamp and txn" in BP)
check("3  the self-check FAILURE branch is untouched",
      "*** candidate-parallel SELF-CHECK FAILED" in BP
      and "the lane isolation is not holding on this machine" in BP)

# ═══ 4. [lift-ab] and the float32 width note ════════════════════════════════════════════
check("4  [lift-ab] leads with the numbers and the bar",
      "[lift-ab] frozen-scaffold LIFT at P=" in BP and "Noise bar " in BP)
check("4  ...and drops how the measurement is built and that it scales with width",
      "so machine drift is shared between the arms instead of landing on" not in BP
      and "It scales with candidate width" not in BP)
check("4  the IMPOSSIBLE verdict keeps every word - it is the one that needs acting on",
      "WHICH CANNOT \"\n" in BP or "This is the CLOCK" in BP)
check("4  ...and so does the bit-identity failure",
      "⚠ OUTPUTS DIFFER between lift ON and lift OFF" in BP)
check("4  the float32 note is the width comparison and nothing else",
      "float32 drift re-measured at the live width P=" in BP
      and "THE FIGURES ARE ON \"\n" not in BP
      and "Figures on the [proj-config] line." in BP)

# ═══ 5. [deliv-cap] ═════════════════════════════════════════════════════════════════════
check("5  the cap-fired line is counts only",
      "pair(s) lifted past " in BP or "pair(s) lifted past " in T2)
check("5  ...and drops what the pre-19fg search would have got wrong",
      "EVERY one \"\n" not in T2 and "compliant and delivery then had to correct" not in T2)
check("5  the unsatisfiable line is two numbers",
      "[deliv-cap] unsatisfiable pair(s), left at " in T2)
check("5  ...and its NON-ZERO warning is untouched",
      "⚠ NON-ZERO — the single-pass closed form does not " in T2)
# the log string spans a source line break, so match the halves rather than the sentence
check("5  the subset line keeps the one thing a reader could misread",
      "so the totals above count " in T2C
      and "DISTINCT (candidate, profile) work" in T2C
      and "entry per candidate per profile - lower than a " not in T2C)

# ═══ 6. [blk-fill] ══════════════════════════════════════════════════════════════════════
check("6  a zero water-fill prints ONE line, not two",
      "put \"\n" in T2 or "NOTHING back onto a blocked row" in T2)
check("6  ...and the two-line version is gone",
      "WHAT THE 0.97 WATER-FILL PUTS BACK" not in T2
      and "OF THAT, " not in T2)
check("6  a NON-zero lift still apportions avoidable vs unavoidable",
      "was UNAVOIDABLE; \"\n" in T2 or "AVOIDABLE {_bf_av:.6g}" in T2)
check("6  ...and still names what the rule would move",
      "what the rule would move" in T2)
check("6  the zero case still says not to conclude the rule is free",
      "Confirm on a run \"\n" in T2 or "before calling it \"\n" in T2
      or "with more blocked pairs" in T2)

# ═══ 7. nothing else broke ══════════════════════════════════════════════════════════════
check("7  band_projection records 19jw", "19jw-bandbuild-log-trim" in BP)
check("7  both files compile",
      bool(compile(BP, "band_projection", "exec")) and bool(compile(T2, "tab_2", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
