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
check("...so the capped row does NOT cross the cap",
      not bool((d[:3] > CAP + 1e-15).any()), f"{d[0]:.17g}")

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
      "_cs_over_s = _cs_liv & (_cs_seed > _cs_cap)" in SRC
      and "_cs_over_d = _cs_liv & (_cs_dec > _cs_cap)" in SRC)
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

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
