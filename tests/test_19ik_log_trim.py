"""19ik — the log batch. Source-level checks, because these are display facts.

The one that matters is the [seed-diag] gate: it decided "there is a breach" from the
DELIVERED split while all nine probes tested the RAW one, so on 2026-09-03 00:23 the gate
let them run and every one printed "(no breached ceiling bands at this split.)" — twenty
lines of nothing and ~23s of projector time. Gate and probes must now read ONE split.
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
EBS = (ROOT / "src/routing_optimiser/s4_search/exact_band_solver.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

# ── the gate and the nine probes share ONE split ────────────────────────────────────────
check("the split is hoisted once, on the DELIVERED basis",
      "_sd_split = (_fm_deliv(_sd_base[None, :])[0]" in T2)
# 19ip: _hm_split is gone with the HELD-vs-MOVABLE block, so three entry points remain.
_probes = re.findall(r"(_hm_split|_fmin_split|_vsib_split|_dx_split) = np\.asarray\("
                     r"_sd_split, float\)", T2)
check("every probe entry point reads the hoisted split", len(set(_probes)) == 3,
      f"{sorted(set(_probes))}")
check("...and _hm_split is not one of them (19ip removed held-vs-movable)",
      "_hm_split" not in T2)
check("the gate reads it too, not its own copy",
      "_sdv = np.asarray(_sd_split, float)" in T2)
# Six legitimate readers of _exact_G remain, and none of them is one of the nine probes:
# the targeted-move stage input (x2, one expression), the hoist itself (x2), the
# seed-pairs list and _fm_cands. Anything above six means a probe grew its own copy back.
_N_EXACT = T2.count('locals().get("_exact_G")')
check("no probe re-derives the split from _exact_G any more", _N_EXACT == 6,
      str(_N_EXACT) + " reader(s); expected 6 — targeted-move stage (2), the hoist (2), "
      "the seed-pairs list, _fm_cands")
check("an unbuildable split is stated, not silently skipped",
      "the split they analyse could not be built" in T2)

# ── the [seed-basis] table ──────────────────────────────────────────────────────────────
check("the table is ordered by the stage that ran, not best-first",
      '_sb_ord = {"band-aware": 0, "exact-proj": 1,' in T2 and "_sb_seq = sorted(" in T2)
check("the name column fits the longest label, so the numbers stop shifting",
      "{'seed':<24}" in T2 and "{'(start) revenue-greedy':<24}" in T2)
check("no <15 name column survives in that table",
      "{'seed':<15}" not in T2 and "{'(start) revenue-greedy':<15}" not in T2)
check("the breach columns are fixed-decimal so decimal points align",
      "{_bl_R:>12.5f}" in T2)
check("the rule under the header spans the widened table", "{'-' * 79}" in T2)

# ── the deletions ───────────────────────────────────────────────────────────────────────
# The phrase survives only in COMMENTS (a history note and the deletion note). What must
# be gone is the log() call.
_LOGGED = [l for l in T2.splitlines()
           if "no preliminary endpoint search is run" in l and "#" not in l.split("no prel")[0]]
check("the 'no preliminary endpoint search' line is no longer LOGGED", not _LOGGED,
      str(_LOGGED[:1]))
check("the scoped-vs-frozen reading note is gone",
      "scoped-movable is the VAMP the" not in EBS)
check("and its deletion is explained where it was",
      "the reading note is DELETED" in EBS)

# ── the seed line, split in two ─────────────────────────────────────────────────────────
check("seed name and breach are two lines now",
      '''log(f"   [full-matrix] seed = '{_fm_sname}'")''' in T2
      and 'log(f"   [full-matrix] exact breach = {_fm_seed_b:.4g}")' in T2)
check("and the never-worse parenthetical is gone from it",
      "never-worse guarantee: delivered breach" not in T2.split(
          "[full-matrix] exact breach")[0][-400:])

# ── the two passing self-checks are muted, not deleted ──────────────────────────────────
check("[viol-bincount] and [deliv-fuse] join the muted families",
      '"viol-bincount":' in T2 and '"deliv-fuse":' in T2)
check("...and the CHECKS themselves still run",
      "[viol-bincount]" in (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py"
                            ).read_text(encoding="utf-8")
      and "_FUSE_DELIV" in T2)
check("a failure still releases them (⚠ is a loud marker)",
      '_LOG_LOUD = ("⚠"' in T2 and "SELF-CHECK FAILED" in T2)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
