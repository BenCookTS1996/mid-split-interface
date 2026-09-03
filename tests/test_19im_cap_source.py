"""19im — is the seed over the cap, or is the encode/decode round trip putting it over?

The two need OPPOSITE fixes: a stage emitting an illegal share is a logic defect and aiming
the stage below the cap would mask it; a legal share pushed over by softmax(log(clip(s))) is a
representation artefact and aiming below the cap is a legitimate answer. 19gu asserted the
second without measuring. These checks prove the mechanism the measurement is looking for is
real, and that it produces excesses of exactly the observed magnitude.
"""
import importlib.util, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_sp = importlib.util.spec_from_file_location(
    "ga19im", str(ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py"))
ga = importlib.util.module_from_spec(_sp); sys.modules["ga19im"] = ga; _sp.loader.exec_module(ga)

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)

CAP = 0.97
starts = np.array([0, 3], np.intp)
lens = np.array([3, 3], np.intp)

def decode(shares, zeros_as_inf=True):
    """What the GA does to a seed: shares -> logits -> UNCAPPED segment softmax."""
    s = np.asarray(shares, float)
    with np.errstate(divide="ignore"):
        lg = np.where(s > 0.0, np.log(np.clip(s, 1e-300, None)), -np.inf) if zeros_as_inf \
            else np.log(np.clip(s, 1e-6, None))
    return ga._segment_softmax(lg[None, :], starts, lens)[0]

# ── 1. A CLOSED profile round-trips clean: the cap row stays legal ──────────────────────
closed = np.array([0.97, 0.02, 0.01, 0.50, 0.30, 0.20])
d = decode(closed)
check("a profile summing to exactly 1 round-trips to float64 dust",
      float(np.abs(d - closed).max()) < 1e-15,
      f"worst movement {float(np.abs(d - closed).max()):.3e}")
check("...so the capped row does NOT cross the cap beyond dust",
      not bool((d[:3] > CAP + max(8.0 * float(np.finfo(float).eps) * CAP, 1e-15)).any()),
      f"{d[0]:.17g}")

# ── 2. THE MECHANISM: a profile summing to 1-d is scaled UP, and the cap row goes over ──
for _d in (3.5e-07, 1e-06, 1e-08):
    short = np.array([0.97, 0.02, 0.01 - _d, 0.50, 0.30, 0.20])
    dd = decode(short)
    lift = float(dd[0] - 0.97)
    check(f"a profile short by {_d:.1e} lifts the cap row by ~cap*d ({CAP * _d:.3e})",
          abs(lift - CAP * _d) < 0.05 * CAP * _d, f"actual lift {lift:.3e}")
    check(f"  ...and that puts it OVER the cap (d={_d:.1e})", dd[0] > CAP)

# the headline: a 3.5e-07 shortfall reproduces the run's observed per-row excess
short = np.array([0.97, 0.02, 0.01 - 3.5e-07, 0.50, 0.30, 0.20])
obs = float(decode(short)[0] - CAP)
check("a 3.5e-07 profile shortfall reproduces the 00:23 run's ~3.5e-07 per-row excess",
      2e-07 < obs < 5e-07, f"{obs:.3e} vs [decode-cap]'s ~3.5e-07")

# ── 3. A GENUINELY illegal seed is over the cap BEFORE any decode ───────────────────────
illegal = np.array([0.99, 0.005, 0.005, 0.50, 0.30, 0.20])
check("a seed a stage left over the cap is over it in the SEED, not just the decode",
      illegal[0] > CAP and decode(illegal)[0] > CAP)
check("...and its round trip is still clean, so rounding cannot be blamed",
      float(np.abs(decode(illegal) - illegal).max()) < 1e-15)

# ── 4. the block is wired, reports the distribution, and states a verdict ───────────────
SRC = (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py").read_text(encoding="utf-8")
check("[cap-source] measures the seed AND the decode separately",
      "_cs_over_s = _cs_liv & (_cs_seed > _cs_cap + _cs_tol)" in SRC
      and "_cs_over_d = _cs_liv & (_cs_dec > _cs_cap + _cs_tol)" in SRC)
check("it counts the rows the round trip PUSHED over",
      "_cs_made = _cs_over_d & ~_cs_over_s" in SRC)
check("it prints the per-profile sum error, the most diagnostic number",
      "the seed's per-profile sums: worst deviation from" in SRC)
check("it prints the excess DISTRIBUTION, not just a count and a total",
      SRC.count("min " "\\u007b" if False else 'SEED excess over the cap: min ') == 1
      and "DECODE excess over the cap: min " in SRC)
check("it states a verdict for each of the three cases",
      "THE SEED IS CLEAN AND THE ROUND TRIP" in SRC
      and "A SEED STAGE IS AT FAULT, and " in SRC
      and 'log(f"[fullmatrix-ga] [cap-source]    BOTH:' in SRC)
check("19gu's unmeasured 'float dust' claim is retracted where it was made",
      "what did NOT hold is the explanation" in SRC
      and 'said the seed\'s violation was "float-dust' in SRC)

# ══ 19in: THE COUNTERFACTUAL — closing the profiles must PROVE or DISPROVE the cause ═════
def close_naive(shares):
    """s / sum — the OBVIOUS fix, and it does not work."""
    seg = np.repeat(np.add.reduceat(shares, starts), lens)
    ok = seg > 1e-12
    return np.where(ok, shares / np.where(ok, seg, 1.0), shares)

def close(shares, cap=CAP):
    """Close to 1 CAP-RESPECTINGLY: the deficit goes to rows with room, proportional to it."""
    s = np.asarray(shares, float)
    seg = np.repeat(np.add.reduceat(s, starts), lens)
    deficit = np.repeat(1.0 - np.add.reduceat(s, starts), lens)
    room = np.where(s < cap, cap - s, 0.0)
    pool = np.repeat(np.add.reduceat(room, starts), lens)
    up = np.where((deficit > 0.0) & (pool > 1e-15),
                  s + room * (deficit / np.where(pool > 1e-15, pool, 1.0)), s)
    ok = seg > 1e-12
    down = np.where(ok, s / np.where(ok, seg, 1.0), s)
    return np.where(deficit > 0.0, up, down)

def over(shares):
    return int((shares > CAP).sum())

# CASE A: the cause IS the closure deficit -> closing removes every over-cap row.
short = np.array([0.97, 0.02, 0.01 - 3.5e-07, 0.50, 0.30, 0.20])
check("A: the unclosed seed's DECODE is over the cap", over(decode(short)) == 1,
      f"{over(decode(short))} row(s), by {decode(short)[0] - CAP:.3e}")
# TOL: a row at exactly the cap round-trips to within +-1 ulp and half of those land above
# it. That residue is irreducible and the decode's cap absorbs it, so "over the cap" has to
# mean over by more than dust — which is also what separates it from the 3.4e-07 defect.
TOL = max(8.0 * float(np.finfo(float).eps) * CAP, 1e-15)
_a_before = float(decode(short)[0] - CAP)
_a_after = float(decode(close(short))[0] - CAP)
check("A: closing the profile takes the excess from 3.4e-07 to ULP DUST",
      _a_before > 100 * TOL and _a_after <= TOL,
      f"{_a_before:.3e} -> {_a_after:.3e} (dust band {TOL:.1e}, a "
      f"{_a_before / max(_a_after, 1e-300):.3g}x reduction)")
check("A: and with a dust tolerance that is ZERO genuinely-over rows",
      int((decode(close(short)) > CAP + TOL).sum()) == 0)
check("A: and the seed itself was never over the cap", over(short) == 0)

# CASE B: a stage emitted an illegal share -> closing changes nothing.
illegal = np.array([0.99, 0.005, 0.005, 0.50, 0.30, 0.20])
check("B: an illegal seed is over the cap before AND after closing",
      over(decode(illegal)) == 1 and over(decode(close(illegal))) == 1)
check("B: so the counterfactual separates the two causes cleanly",
      int((decode(close(short)) > CAP + TOL).sum()) == 0
      and int((decode(close(illegal)) > CAP + TOL).sum()) == 1)

# CASE C: the closure deficit moves EVERY row, not just the capped one — which is the
# right shape for [decode-loss]'s success-rate gap, and capping 9 rows is not.
d_all = np.abs(decode(short) - short)
check("C: the deficit moves every row in the short profile, not only the capped one",
      int((d_all[:3] > 1e-12).sum()) == 3 and int((d_all[3:] > 1e-12).sum()) == 0,
      f"{int((d_all[:3] > 1e-12).sum())} of 3 in the short profile, "
      f"{int((d_all[3:] > 1e-12).sum())} of 3 in the closed one")

# ── the OBVIOUS fix is wrong, and that is worth a test of its own ───────────────────────
_naive = close_naive(short)
check("D: a plain s/sum renormalise scales the capped row OVER the cap",
      _naive[0] > CAP, f"{_naive[0]:.17g} vs cap {CAP}")
check("D: ...so the cap-respecting closure is not a nicety",
      close(short)[0] <= CAP + 1e-15 and abs(float(np.add.reduceat(close(short), starts)[0]) - 1.0) < 1e-15,
      f"cap-respecting gives {close(short)[0]:.17g}, profile sums to "
      f"{float(np.add.reduceat(close(short), starts)[0]):.17g}")
check("D: and it still closes the profile to exactly 1",
      abs(float(np.add.reduceat(close(short), starts)[0]) - 1.0) < 1e-15)

SRC2 = (ROOT / "src/routing_optimiser/s4_search/genetic_fullmatrix.py").read_text(encoding="utf-8")
check("the counterfactual is wired: close, re-encode, re-decode, recount",
      "_cs_closed = np.where(_cs_def > 0.0, _cs_seed + _cs_fill, _cs_down)" in SRC2 and "_cs_lg2 = _shares_to_logits(" in SRC2
      and "_cs_over_2 = int((_cs_liv & (_cs_dec2 > _cs_cap + _cs_tol)).sum())" in SRC2)
check("it re-scores the success rate too, so one fix can answer both symptoms",
      "decoded-after-closing" in SRC2)
check("the recommendation says the closure must respect the cap",
      "closed CAP-RESPECTINGLY" in SRC2 and "A plain s/sum renormalise does NOT work" in SRC2)
check("it states PROVEN / NOT THE CAUSE / PARTLY rather than leaving it to be inferred",
      "PROVEN, AND IT ANSWERS BOTH" in SRC2 and "NOT THE CAUSE" in SRC2
      and "PARTLY" in SRC2)
check("the M5-breach handicap line prints only when there IS a handicap",
      "if abs(_dl_gap) > 1e-12 or _dl_bd > 1e-12 or seed_band > 1e-12:" in SRC2)

check("the counts use a dust tolerance, so 1-ulp is not reported as a violation",
      "_cs_over_s = _cs_liv & (_cs_seed > _cs_cap + _cs_tol)" in SRC2
      and "_cs_dust_d = int((_cs_liv & (_cs_dec > _cs_cap) & ~_cs_over_d).sum())" in SRC2)
check("and it says where 19gu's 'float dust' phrase DOES belong",
      "THIS is what 19gu meant by \"float dust\"" in SRC2)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
