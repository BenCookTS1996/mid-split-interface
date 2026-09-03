"""19ji - the back-fill injector stops lower-casing 21 strings six million times.

[cvp-timing] 19je split `_inject_backfill_rows` (18.2s) into three, and the first piece -
"normalise the keys + build the two presence sets" - came back at 9.0s of an 80.7s
projection. It does four different things, so 19ji marks them apart AND makes the one that
is unambiguously free actually free.

`b["RPGT"].astype(str).str.strip().str.lower()` runs the chain once PER ROW over 6.5M rows.
`_clean_col` - already in this file, already used by the key-normalise step, already
measured at 1.84x by 19fx - runs it once per DISTINCT VALUE. There are 8 RPGTs and 21
vampMids in the whole export, so the per-row chain lower-cases each of them about 300,000
times to learn one fact.

THERE IS NO BIT-IDENTITY QUESTION TO WEIGH, which is what makes this worth doing without a
measurement first: trim-and-lower is deterministic, so once per distinct value cannot differ
from once per row. The ONE way it could is the trap 19fx's edge-case test caught -
pd.factorize collapses None and NaN into a single missing value where the per-row chain
renders them "none" and "nan" - and `_clean_col` falls back to the slow chain when a column
holds nulls. Both are checked below on the real shapes.
"""
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))
IC = (ROOT / "app/impact_calcs.py").read_text(encoding="utf-8")
import impact_calcs as ic

FAIL = []
def check(n, ok, d=""):
    print(("  PASS  " if ok else "  FAIL  ") + n + (("   " + d) if d else ""))
    if not ok: FAIL.append(n)


# ═══ 1. character-for-character, on the two columns actually swapped ═════════════════════
rng = np.random.default_rng(19)
RPGT = ["Monthly Initial", " Addon Sale ", "ANNUAL SUB SALE", "Upgrades", "",
        "Monthly Renewal", "P6M Renewals", "Annual Sub Renewal"]
MIDS = ["Adyen_TotalAV", " WorldPay - Total AV ", "adyen_totalav", "Braintree USA - Total AV",
        "Checkout - Total AV", "WoodForest - Total AV", "PaySafe - Total AV"]
for _nm, _vals in (("RPGT", RPGT), ("vampMid", MIDS)):
    s = pd.Series(rng.choice(_vals, 300_000))
    ref = s.astype(str).str.strip().str.lower().to_numpy()
    got = np.asarray(ic._clean_col(s, lower=True))
    check(f"1  `{_nm}`: _clean_col == the per-row chain, character for character",
          got.shape == ref.shape and bool((got == ref).all()) and got.dtype == ref.dtype,
          f"{len(set(ref.tolist())):,} distinct value(s) over {len(ref):,} row(s)")

# THE TRAP, which is the only way this could differ
s = pd.Series([None, np.nan, "A ", "b", float("nan")] * 2000)
ref = s.astype(str).str.strip().str.lower().to_numpy()
got = np.asarray(ic._clean_col(s, lower=True))
check("1  None and NaN do NOT collapse - the fallback that 19fx's edge case forced",
      bool((got == ref).all()) and set(got[:2].tolist()) == {"none", "nan"},
      f"first two render as {list(got[:2])}")

# a column that is ALREADY clean, and one that is entirely one value
for _nm, _s in (("already clean", pd.Series(["usd"] * 50_000)),
                ("single value with padding", pd.Series(["  UsD  "] * 50_000)),
                ("empty frame", pd.Series([], dtype=object))):
    ref = _s.astype(str).str.strip().str.lower().to_numpy()
    got = np.asarray(ic._clean_col(_s, lower=True))
    check(f"1  ...and on a {_nm} column", got.shape == ref.shape
          and (not len(ref) or bool((got == ref).all())))

# ═══ 2. the swap is where it should be, and nothing else moved ═══════════════════════════
check("2  both back-fill key columns go through _clean_col now",
      'b["_rpgtl"] = _clean_col(b["RPGT"], lower=True)' in IC
      and 'b["_vml"] = _clean_col(b["vampMid"], lower=True)' in IC)
check("2  ...and the per-row chain is gone from those two lines",
      'b["_rpgtl"] = b["RPGT"].astype(str)' not in IC
      and 'b["_vml"] = b["vampMid"].astype(str)' not in IC)
check("2  _clean_col is defined before the injector that calls it",
      0 < IC.find("def _clean_col(") < IC.find("def _inject_backfill_rows("))
check("2  the 9.0s step is now marked in three pieces, so the next run names the expensive one",
      IC.count('mark("  backfill: copy the export frame")') == 1
      and IC.count('mark("  backfill: lower-case the RPGT and vampMid keys")') == 1
      and IC.count('mark("  backfill: the two presence sets (t0 uniques -> python tuples)")') == 1)
check("2  the marks stay optional, so tab_2's own caller is unaffected",
      IC.count("if mark is not None:") >= 5 and "mark=None" in IC)
check("2  impact_calcs records 19ji", "19ji-backfill-cleancol" in IC)

print()
print("FAILURES: " + (", ".join(FAIL) if FAIL else "none"))
sys.exit(1 if FAIL else 0)
