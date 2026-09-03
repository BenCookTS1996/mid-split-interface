"""19io — the engineering key's unreachable floor, and the two-populations reporting bug.

`viol` read a flat 6.18557 for all 320 generations. 6.18557 / (1/0.97 - 1) = 200.00 EXACTLY:
200 rows at share 1.0, in profiles whose only eligible gateway must hold 100%. The key carried
a term no candidate could ever reduce, and `_key_of` ranks it ABOVE conversion.
"""
import importlib.util, os, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
GA = str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py")

def load(name, on):
    os.environ["ROUTING_VIOL_FORCED"] = "1" if on else "0"
    sp = importlib.util.spec_from_file_location(name, GA)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

# ── the arithmetic that identified it ───────────────────────────────────────────────────
CAP = 0.97
one_row = 1.0 / CAP - 1.0
check("the run's flat viol is EXACTLY 200 rows at share 1.0",
      abs(6.18557 / one_row - 200.0) < 1e-3,
      f"6.18557 / {one_row:.9f} = {6.18557 / one_row:.4f}")

# ── a problem with BOTH a forced row and a genuine over-cap row ──────────────────────────
new, old = load("ga_vf_on", True), load("ga_vf_off", False)
check("the switch selects different code", new._VIOL_FORCED and not old._VIOL_FORCED)

N, M = 6, 2
ctx = {"n_row": N, "n_mid": M,
       "profile_starts": np.array([0, 1, 3], np.int64),      # sizes 1, 2, 3
       "profile_counts": np.array([1, 2, 3], np.int64),
       "elig": np.ones(N), "base": np.full(N, 0.3),
       "profile_vol": np.array([100., 200., 200., 300., 300., 300.]),
       "sr": np.full(N, 0.6), "risk": np.full(N, 0.01),
       "mid_id": np.array([0, 1, 0, 1, 0, 1], np.int64),
       "mid_rows": None, "vamp_cap": None, "max_share": CAP, "floor": 0.0}
p_new, _ = new.problem_from_ctx(ctx, soft_cap_mult=1.0)
p_old, _ = old.problem_from_ctx(ctx, soft_cap_mult=1.0)

#   profile 0: ONE eligible row -> must hold 1.0. FORCED, not a choice.
#   profile 1: two rows, one at 0.99 -> a GENUINE over-cap violation (0.97*2 = 1.94 > 1).
#   profile 2: three rows, all legal.
sh = np.array([[1.0, 0.99, 0.01, 0.4, 0.3, 0.3]])
v_new = float(new._violation(sh, p_new)[0])
v_old = float(old._violation(sh, p_old)[0])
check("OLD counts the forced row, so the key can never reach 0",
      abs(v_old - ((1.0 / CAP - 1.0) + (0.99 / CAP - 1.0))) < 1e-12, f"{v_old:.9f}")
check("NEW exempts the forced row and keeps the genuine one",
      abs(v_new - (0.99 / CAP - 1.0)) < 1e-12,
      f"{v_new:.9f} (the 0.99 row alone) vs old {v_old:.9f}")
check("...so a REAL violation is not hidden by the exemption", v_new > 1e-9)

# a split with ONLY forced rows must now read exactly 0
sh2 = np.array([[1.0, 0.5, 0.5, 0.4, 0.3, 0.3]])
check("a compliant split with a forced row now reads viol 0",
      abs(float(new._violation(sh2, p_new)[0])) < 1e-12
      and float(old._violation(sh2, p_old)[0]) > 1e-3,
      f"new {float(new._violation(sh2, p_new)[0]):.3e} vs old "
      f"{float(old._violation(sh2, p_old)[0]):.6f}")

# a CONSTANT term cannot change a lexicographic ordering — the answer-identical claim
a = np.array([[1.0, 0.5, 0.5, 0.4, 0.3, 0.3],
              [1.0, 0.6, 0.4, 0.5, 0.3, 0.2],
              [1.0, 0.97, 0.03, 0.34, 0.33, 0.33]])
r_new = new._rank(np.array([0.5, 0.6, 0.55]), new._violation(a, p_new), np.zeros(3))
r_old = old._rank(np.array([0.5, 0.6, 0.55]), old._violation(a, p_old), np.zeros(3))
check("with the forced term constant across candidates the RANKING is unchanged",
      np.array_equal(r_new, r_old), f"{r_new} vs {r_old}")

SRC = pathlib.Path(GA).read_text(encoding="utf-8")
check("[viol-forced] reports the decomposition so 'constant' is checked, not assumed",
      "[viol-forced]" in SRC and "THE REMAINDER IS NON-ZERO" in SRC)
check("[decode-cap] now separates the two populations it used to merge",
      "_dc_over_tgt" in SRC and "do not divide it" in SRC)
# 19iu made the 0 -> 0 case SILENT rather than verbose - the verdict text ("NOTHING TO
# EXPLAIN") is gone with the three other lines that said the same thing. What 19io actually
# fixed is still checked: the case is recognised on its own terms and cannot fall through to
# the contradictory "PARTLY" branch.
check("[cap-source]'s 0 -> 0 case is recognised on its own terms, not fallen through to PARTLY",
      "_cs_none = (_cs_n_d == 0 and _cs_over_2 == 0)" in SRC
      and "if _cs_none:" in SRC and "_cs_quiet = _cs_quiet and _cs_none" in SRC)

EBS = (ROOT / "src/routing_optimiser/s4_search/exact_band_solver.py").read_text(encoding="utf-8")
check("the LP stall is armed at K=4, above the largest stall_min_safe seen (3)",
      'ROUTING_SEED_LP_STALL", "4"' in EBS and "stall_min_safe = 3" in EBS)
check("...and it is named as answer-affecting rather than slipped in",
      "THIS IS ANSWER-AFFECTING" in EBS)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
