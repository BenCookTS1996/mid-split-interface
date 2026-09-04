"""19ju - four reconciliation-preamble lines shortened, the settled [gk-code] proof retired,
and three log strings that named switches which no longer exist.

THE LAST ONE IS THE FINDING. A switch audit turned up names that are still printed in live log
lines but are never read at runtime:

  * `ROUTING_FEAS_PAR` - seed_search fixed `_par = False` on 2026-08-31 and deleted the switch.
    Two live [feas-par] log lines still told a reader to set it: one prefixed the serial branch
    with "ROUTING_FEAS_PAR=0 —", reporting a setting nobody could have made, and the checksum
    line closed with "re-run with ROUTING_FEAS_PAR=0, which is the serial control" - an
    instruction that cannot be followed, pointing at a control the stage already IS.
  * `ROUTING_GKCODE` - never existed under that name. The [gk-code] VERIFY-FAILED branch said
    "Set ROUTING_GKCODE=0-equivalent by fixing the key and re-run", on the one path where a
    reader is being told their delivered VAMP is untrustworthy.

A log that names a switch is giving an instruction. This codebase already carries three
comments about logs that misstate their own configuration; these are the same fault.

[gk-code]'s verify is a ONE-SHOT PROOF - its own comment has said so since 19gq ("worth paying
exactly once and then turning off"). It has printed VERIFIED on every run since, at 1.7s a run.
The default is now OFF, `=1` brings it back, and a FAILURE still ships the reference and shouts.
"""
import io, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
IC = (ROOT / "app/impact_calcs.py").read_text(encoding="utf-8")
SS = (ROOT / "src/routing_optimiser/s4_search/seed_search.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

def log_strings(src):
    """Every source line that is part of a log() call and is not a comment."""
    return [l for l in src.splitlines() if not l.lstrip().startswith("#")]


# ═══ 1. DEAD SWITCH NAMES ARE OUT OF THE LOG STRINGS ═════════════════════════════════════
for _name, _why in (("ROUTING_FEAS_PAR", "seed_search fixed `_par = False` on 2026-08-31"),
                    ("ROUTING_GKCODE\"", "never existed under that name")):
    _hits = [l.strip()[:70] for l in log_strings(T2) if _name in l]
    check(f"1  no live line names {_name.rstrip(chr(34))} - {_why}", not _hits,
          f"{len(_hits)} hit(s): {_hits[:2]}" if _hits else "")
check("1  ...and the switch really is dead, so this was a lie and not a stale default",
      "_par = False" in SS and 'environ.get("ROUTING_FEAS_PAR"' not in SS,
      "seed_search hardcodes it")
check("1  the serial [feas-par] line still reports what it measured",
      "start(s) ran SERIALLY in " in T2 and "fixed serial since 2026-08-31" in T2)
check("1  the checksum line still says what a differing checksum would mean",
      "differing \"\n" in T2 or "checksum is NOT concurrency" in T2)
check("1  the VERIFY-FAILED branch still shouts and now names a real action",
      "delivered VAMP is NOT trustworthy" in T2 and "fix \"\n" in T2 or "_gk_codes and re-run" in T2)

# ═══ 2. [gk-code]: a one-shot proof, retired on schedule ═════════════════════════════════
# 19kg: the switch is gone entirely (it was already default OFF, so no run changes). The
# property that mattered - the proof is not re-run on every projection - is now structural
# rather than a default, and the reference code it would compare against is still there.
check("2  the verify is off, with no env var left to arm it",
      "if False:" in IC and 'environ.get("ROUTING_GKCODE_VERIFY"' not in IC)
check("2  the reason is recorded where the default is",
      "DEFAULT FLIPPED TO OFF" in IC and "worth paying exactly once" in IC)
check("2  a FAILURE still ships the reference rather than the int key",
      "_psum = _gv_ref" in IC and "NOT a silent fallback" in IC)
check("2  the ordinary run now prints NOTHING for [gk-code]",
      'elif _gkc.get("why"):' in T2
      and "[gk-code] ON — \"" not in T2)
check("2  ...but a REASON to have skipped it is still a fact about this run, so it prints",
      "[gk-code] NOT verified this call " in T2)
check("2  and the OFF branch is untouched - not using the int key at all is worth saying",
      "[gk-code] OFF — the five aged-frame " in T2)

# ═══ 3. the four preamble lines are shorter, and still carry their numbers ═══════════════
check("3  [f32-floor] is one line, keeping the drift and what it hides",
      "[f32-floor] float32 drift is " in T2
      and "a REAL disagreement below it is \"\n" in T2 or "invisible." in T2)
# scoped to [f32-floor]. The SAME settled sentence also closes the `══ float32 NOISE FLOOR`
# line at the reconciliation verdict a thousand lines below - a different line, which Ben did
# not ask about, so it is deliberately left alone rather than trimmed unasked.
_f32_blk = T2.split("[f32-floor] float32 drift is ")[1][:1200]
check("3  ...and the settled prose is gone FROM [f32-floor]",
      "unconditional as " not in _f32_blk
      and "the price of the setting, stated as a number" not in T2
      and "so rounding does not \"\n" not in T2)
check("3  ...while the reconciliation verdict's own copy of it is untouched",
      "The float32 projector is unconditional as " in T2,
      "a different line, not in scope - left alone rather than trimmed unasked")
check("3  [forensic] keeps the number, the bar and the saving",
      "the four attribution stashes were \"\n" in T2 or "skipped (~77s saved)" in T2)
check("3  ...and drops the list of four stashes that name themselves below",
      "report themselves as not-computed rather than empty" not in T2)
check("3  [nw-skip] is ONE line, not two",
      "[nw-skip] GA output delivers 0 breach, so the " in T2
      and "[nw-skip] COST:" not in T2)
# 19kg: the switch is gone; the setting is a source constant. What 19ju was protecting is
# unchanged and still worth pinning - the line names a revert that EXISTS and is actually read.
check("3  ...and still names the revert that projects both - one that exists and is read",
      "`_SW_NW_SKIP_SEED = False` " in T2
      and "_nw_skip_ok = _SW_NW_SKIP_SEED" in T2
      and "_SW_NW_SKIP_SEED = True" in T2,
      "unlike the two above, this setting is actually read")

# ═══ 4. nothing else broke ═══════════════════════════════════════════════════════════════
check("4  both files compile",
      bool(compile(T2, "tab_2", "exec")) and bool(compile(IC, "impact_calcs", "exec")))

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
