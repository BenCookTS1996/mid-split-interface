"""19ip — two lines that had gone stale, and the log batch behind them.

The two stale lines are mine, not the engine's:

  * `[viol-forced]` printed straight after `seed_other = _violation(s0, p)`, the RAW decoded
    seed. The capped decode has already pulled every row down to the cap there, so it found
    ZERO forced rows and printed "0 total, 0 rows" on the very run whose DELIVERED key had
    6.18557 removed by that exemption. `_VF_FACT` holds row 0 of the LAST _violation call
    before it prints, so the fix is WHERE it prints: after the D2 re-score, which is the
    basis the key is compared on.
  * `[decode-cap]`'s "THE ENGINEERING KEY IS NOT 0 THIS RUN, AND SHOULD NOT BE" — 19io made
    that false. The flat 4.9485 / 6.18557 it was explaining was entirely forced rows, and the
    14:19 run read 0.0000 with the split unchanged.

Everything else here is a display fact, so it is checked at source level.
"""
import importlib.util, os, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA_P = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")
GA = (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py").read_text(encoding="utf-8")
T2 = (ROOT / "app/tab_2_routing_engine.py").read_text(encoding="utf-8")
EBS = (ROOT / "src/routing_optimiser/s4_search/exact_band_solver.py").read_text(encoding="utf-8")

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)


# 1. [viol-forced]: the fact is the LAST call's, so WHERE it prints is the fix
os.environ["ROUTING_VIOL_FORCED"] = "1"
_sp = importlib.util.spec_from_file_location("ga_19ip", GA_P)
ga = importlib.util.module_from_spec(_sp); sys.modules["ga_19ip"] = ga; _sp.loader.exec_module(ga)

CAP = 0.97
N = 6
ctx = {"n_row": N, "n_mid": 2,
       "profile_starts": np.array([0, 1, 3], np.int64),
       "profile_counts": np.array([1, 2, 3], np.int64),
       "elig": np.ones(N), "base": np.full(N, 0.3),
       "profile_vol": np.array([100., 200., 200., 300., 300., 300.]),
       "sr": np.full(N, 0.6), "risk": np.full(N, 0.01),
       "mid_id": np.array([0, 1, 0, 1, 0, 1], np.int64),
       "mid_rows": None, "vamp_cap": None, "max_share": CAP, "floor": 0.0}
p, _ = ga.problem_from_ctx(ctx, soft_cap_mult=1.0)

# The RAW decoded seed: the capped decode has pulled the single-row profile to the cap, so
# there is nothing forced to find. This is what 19io was reading.
raw = np.array([[CAP, 0.5, 0.5, 0.4, 0.3, 0.3]])
ga._VF_FACT.update(n=0, rows=0, forced=0.0, kept=0.0, said=False)
ga._violation(raw, p)
_raw_rows, _raw_forced = ga._VF_FACT["rows"], ga._VF_FACT["forced"]
check("on the RAW decoded seed the exemption finds NOTHING",
      _raw_rows == 0 and _raw_forced == 0.0,
      f"rows={_raw_rows}, forced={_raw_forced:.6g} - this is the '0 total, 0 rows' 19io printed")

# The DELIVERED split: the single-eligible-gateway profile holds 1.0, because there is
# nowhere else for its share to go.
deliv = np.array([[1.0, 0.5, 0.5, 0.4, 0.3, 0.3]])
ga._violation(deliv, p)
check("a second call OVERWRITES the fact while it is unprinted",
      ga._VF_FACT["rows"] == 1 and ga._VF_FACT["forced"] > 1e-9,
      f"rows={ga._VF_FACT['rows']}, forced={ga._VF_FACT['forced']:.6g} - so the LAST call "
      "before the log line is the one that gets reported")
check("...and that is the 1/cap - 1 the key was carrying",
      abs(ga._VF_FACT["forced"] - (1.0 / CAP - 1.0)) < 1e-12,
      f"{ga._VF_FACT['forced']:.9f} vs {1.0 / CAP - 1.0:.9f}")
ga._VF_FACT["said"] = True
ga._violation(raw, p)
check("once printed, the fact is frozen (no later call rewrites history)",
      ga._VF_FACT["rows"] == 1)

# 2. so the log line has to sit AFTER the D2 re-score
_i_rescore = GA.find('[obj-basis] the SEED is re-scored on the delivered split')
_i_vf = GA.find("[viol-forced] the engineering key's MAX-SHARE term on the seed")
_i_seedother = GA.find("seed_other = _violation(s0, p)[0]")
check("[viol-forced] prints after the D2 re-score, not after the raw seed_other",
      -1 < _i_rescore < _i_vf and _i_seedother < _i_rescore,
      f"seed_other@{_i_seedother}, re-score@{_i_rescore}, [viol-forced]@{_i_vf}")
check("it appears exactly once (the old site is gone, not duplicated)",
      GA.count("[viol-forced] the engineering key") == 1)
check("and it NAMES the basis it measured, so this cannot go stale silently again",
      '_vf_basis = ("DELIVERED" if (_DECODE_OBJ and _have_full and _fd0 is not None)' in GA
      and 'measured on "\n            f"the {_vf_basis} split' in GA)
check("_fd0 exists before the line reads it",
      -1 < GA.find("_fd0 = _deliver_full(s0)") < _i_vf)

# 3. [decode-cap]'s armed branch no longer claims 0 is wrong
check("the false 'NOT 0 THIS RUN' sentence is gone",
      "ENGINEERING KEY IS NOT 0 THIS RUN" not in GA)
check("the armed branch now expects 0.0000 on the delivered basis too",
      "THE ENGINEERING KEY should read 0.0000 even though" in GA)
check("...and says WHY that changed (19io exempts the structurally forced rows)",
      "is exempted by " in GA and "(see [viol-forced])" in GA)
check("the [deliv-cap] cross-check survives as the way to tell a real breach",
      "unsatisfiable" in GA.split("THE ENGINEERING KEY should read 0.0000")[1][:1200])
check("both branches still exist, so an unarmed run is not left unexplained",
      "THE ENGINEERING KEY should now read 0.0000 for every candidate" in GA)

# 4. the ten run-log edits
check("1  the `profiles:` line drops the 'injected door' split that always read 0",
      'log(f"      profiles: {_rc_cn:,} profile(s) the search can route into")' in T2
      and "exist only because a candidate door was" not in T2)
check("2  the 'Frozen layers (t > period)' line no longer prints",
      '_ = ("            Frozen layers (t > period) are untouched' in T2
      and 'log("            Frozen layers' not in T2)
check("3  [rpgt-scope] counts PROFILES, not scaffold rows",
      "[rpgt-scope] {_rsH:,} of {_rsN:,} profile(s) held at baseline" in T2
      and "scaffold row(s) held at " not in T2)
check("3  ...via a real profile key, with a row-count fallback that cannot raise",
      '_rsK = (_P["_cur"].astype(str)' in T2 and "_rsN, _rsH = len(_P), int(_hitP.sum())" in T2)
check("3  and it is split over three lines instead of one 340-character sentence",
      T2.count('log("      an UNSCOPED RPGT still carries volume') == 1
      and 'log(f"      scope: ' in T2)
check("4  [keep-gate] explains itself in plain English",
      "row(s) on SWITCHED-OFF " in T2 and "there is nothing to reroute " in T2)
check("4  ...and still states what it does NOT model",
      "not modelled: delivery also scales the share down " in T2)
# 19ip trimmed [emask-grain] to three lines: the grain, the two counts, the source. 19jt then
# moved the COUNTS to the `eligibility:` line (two lines were counting the same thing under
# different rules and disagreeing). What 19ip was really asserting - that this block states
# its grain and names its source rather than sprawling - still holds, so the check is rewritten
# to that rather than to the count line it no longer has.
check("5  [emask-grain] states its grain and names its source",
      'log("   [emask-grain] wallet/USA capability is masked at (vampMid, "\n' in T2
      and 'log(f"      source: {_pair_src}"' in T2)
check("5  ...and its counts moved to `eligibility:` rather than being duplicated (19jt)",
      'log(f"      {len(_wc_pairs):,} wallet-incapable pair(s)")' not in T2
      and "wallet-incapable gatewayFid(s), " in T2)
check("6  the cap scaffold table is all CELL counts",
      "'aged cells for capped MIDs'" in T2 and "'profiles covered'" not in T2)
check("6  and four blank log lines became two",
      'log("")\n                        log("")\n                        # ── [cap-timing]'
      not in T2)
check("7  reconciliation step 3 says it drops CELLS, matching its own delta",
      "'3  drop cells in profiles with no banded MID'" in T2
      and "drop profiles with no banded MID (out of scope)" not in T2)
check("8  the HELD-vs-MOVABLE block no longer runs",
      "held_movable_report" not in T2 and "_hm_split" not in T2)
check("8  ...and the block count says eight everywhere, not nine",
      "the eight 'why is this band stuck?' blocks" in T2
      and "the nine 'why is this band stuck?' blocks" not in T2
      and "Six of the eight probes" in T2)
check("8  the two lists that NAME the blocks dropped held-vs-movable",
      "held-vs-movable" not in T2)
check("8  the function is kept and labelled UNWIRED, not deleted",
      "def held_movable_report(" in EBS and "UNWIRED as of 19ip" in EBS
      and "pro_rata x fcp1_frac provenance below has no other" in EBS)
check("9  the targeted-move RAW-basis NOTE is gone",
      "NOTE every 'better' above is the RAW " not in EBS)
check("9  ...but each verdict still names the basis it measured on",
      EBS.count("strictly better on the RAW basis; the engine selects on DELIVERED.") == 3
      and "see log note" not in EBS)
check("10 'stopped because <slug>' became what the slug MEANS",
      "stopped after {passes:,} pass(es) " in EBS
      and "stopped because '{stop_reason}'" not in EBS)
check("10 ...and the projection counts left, because [tmove-cost] prints them WITH seconds",
      "projection cost {_cost['mv']:,} sparse matvec(s)" not in EBS
      and "[tmove-cost] stage {_tot:.1f}s total" in EBS)

# every stop_reason the solver can actually set must have a translation
_slugs = set(re.findall(r"stop_reason = ['\"]([a-z\-]+)['\"]", EBS))
_mapped = set(re.findall(r"^\s+[\"']([a-z\-]+)[\"']: \"", EBS, re.M))
check("10 every stop_reason slug the solver sets has a plain-English translation",
      bool(_slugs) and _slugs <= _mapped,
      f"slugs={sorted(_slugs)}, unmapped={sorted(_slugs - _mapped)}")
check("10 ...with an unknown slug falling through to itself rather than vanishing",
      ".get(str(stop_reason), str(stop_reason))" in EBS)

# 4b. ROUTING_EMASK_PAIRS: the switch is gone, and so is the collision inside its own guard
IC = (ROOT / "app/impact_calcs.py").read_text(encoding="utf-8")
BP = (ROOT / "src/routing_optimiser/s4_search/band_projection.py").read_text(encoding="utf-8")
_live = [ln for ln in (IC + T2 + BP).splitlines()
         if "ROUTING_EMASK_PAIRS" in ln and not ln.strip().startswith("#")]
check("no live code reads ROUTING_EMASK_PAIRS any more",
      not _live, "; ".join(_live[:3]))
check("emask_pairs_on() is deleted, not left as a dead reader",
      "def emask_pairs_on(" not in IC and "emask_pairs_on" not in T2)
check("the pair grain is unconditional wherever pair data exists",
      "_use_pairs = bool(_wc_p or _uo_p)" in IC)
check("...and the name-set fallback still serves a caller with only vampMid names",
      "return (_wallet & _ml.isin(_wc_s)) | (_nonusa & _ml.isin(_uo_s))" in IC)
check("the cache key stops hashing the switch",
      'f"|emp=' not in IC and "_emp = os.environ.get" not in IC
      and "|emp=" not in "".join(ln for ln in IC.splitlines()
                                 if not ln.strip().startswith("#")))
check("...and the collision that term CAUSED is written down where it happened",
      "hashed IDENTICALLY and computed at DIFFERENT grains" in IC)
check("_PROJ_CODE_VER records the change, so the key busts on the version too",
      '_PROJ_CODE_VER = "2026-09-03-19ip-pair-grain-ONLY"' in IC)
check("the [ef-mask] note no longer offers the switch as a way back",
      "puts delivery back on the coarse test" not in BP)
check("the projection-path comment that had the default BACKWARDS is corrected",
      "the pairs are always used" in T2
      and "Under ROUTING_EMASK_PAIRS=0 (the default) the sets are" not in T2)

# the two defaults that disagreed are the finding; assert the pair of them cannot come back
check("neither default can be re-introduced by accident (no env read of it survives)",
      IC.count('os.environ.get("ROUTING_EMASK_PAIRS"') == 0)


# 5. nothing behavioural moved
check("the [viol-forced] move changed no arithmetic - the exemption itself is untouched",
      "_so_row = np.where(_vf_forced, 0.0, _so_row)" in GA
      and "_vf_forced = (_vf_n * _vf_cap[None, :]) <= 1.0 + _DC_EPS" in GA)
check("both build tags record 19ip",
      "19ip-log-trim" in GA and "19ip-log-trim" in EBS)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
