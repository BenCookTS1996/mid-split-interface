"""Unit tests for the sub-cell volume-glue helpers (routing_optimiser.subcell).

Locks in the Stage-1 contract:
  * vi_frac sums to 1.0 within each (cur, bank, rpgt) cell;
  * a zero-VI or export-absent cell collapses to a single '_all_' sub-cell (frac 1.0);
  * expanding the forecast CONSERVES volume per cell (sum over sub-cells == original);
  * success/risk/baseline_share are broadcast unchanged (scoring stays cell grain).

Pure pandas/numpy — no BigQuery/Streamlit. Run: python tests/test_subcell.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.subcell import subcell_vi_fractions, expand_forecast_to_subcells  # noqa: E402


def _prorata():
    """Two cells. Cell A (usd|400001|monthly initial) splits 70/30 across two sub-cells;
    aged rows (t=1) are present but must be IGNORED (only t==0 baseline counts). Cell B
    (usd|400002|monthly initial) has ZERO VI → must collapse to one '_all_' sub-cell."""
    rows = []
    # Cell A, t=0: non_gp_ap/usa VI=700 (across two MIDs), applepay/usa VI=300
    rows += [dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="non_gp_ap",
                  Country="usa", VI_Txn_Count=400.0, t=0, vampMid="A"),
             dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="non_gp_ap",
                  Country="usa", VI_Txn_Count=300.0, t=0, vampMid="B"),
             dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="applepay",
                  Country="usa", VI_Txn_Count=300.0, t=0, vampMid="A")]
    # Cell A aged rows (t=1) — must be excluded from the weights
    rows += [dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="non_gp_ap",
                  Country="usa", VI_Txn_Count=9999.0, t=1, vampMid="A")]
    # Cell B, t=0: zero VI everywhere
    rows += [dict(Currency="usd", BIN="400002", RPGT="Monthly Initial", paymentMethodProvider="non_gp_ap",
                  Country="usa", VI_Txn_Count=0.0, t=0, vampMid="A")]
    return pd.DataFrame(rows)


def _forecast():
    """Cell A: 2 gateways (cell volume 1000, split 600/400). Cell B: 1 gateway (vol 50).
    Cell C (usd|400003): NOT in the pro-rata export at all → must survive as one '_all_' sub-cell."""
    return pd.DataFrame([
        dict(currency="usd", bank="400001", rpgt="Monthly Initial", gateway="gw1",
             volume=600.0, baseline_share=0.6, succ=0.8, risk=0.02),
        dict(currency="usd", bank="400001", rpgt="Monthly Initial", gateway="gw2",
             volume=400.0, baseline_share=0.4, succ=0.7, risk=0.03),
        dict(currency="usd", bank="400002", rpgt="Monthly Initial", gateway="gw1",
             volume=50.0, baseline_share=1.0, succ=0.6, risk=0.05),
        dict(currency="usd", bank="400003", rpgt="Monthly Initial", gateway="gw1",
             volume=25.0, baseline_share=1.0, succ=0.55, risk=0.04),
    ])


def test_vi_fractions_sum_to_one_and_ignore_aged():
    fr = subcell_vi_fractions(_prorata())
    # Cell A: two sub-cells, fractions 0.7 / 0.3 (aged t=1 row ignored)
    a = fr[(fr["cur"] == "usd") & (fr["bank"] == "400001")].set_index(["pmp", "ctry"])["vi_frac"]
    assert abs(a.loc[("non_gp_ap", "usa")] - 0.7) < 1e-9, a.to_dict()
    assert abs(a.loc[("applepay", "usa")] - 0.3) < 1e-9, a.to_dict()
    # every cell's fractions sum to 1
    s = fr.groupby(["cur", "bank", "rpgt"])["vi_frac"].sum()
    assert np.allclose(s.to_numpy(), 1.0), s.to_dict()
    # Cell B (zero VI) → single '_all_' sub-cell, frac 1
    b = fr[(fr["bank"] == "400002")]
    assert len(b) == 1 and b.iloc[0]["pmp"] == "_all_" and abs(b.iloc[0]["vi_frac"] - 1.0) < 1e-9
    print("A. vi_frac sums to 1 per cell; aged rows ignored; zero-VI cell → one _all_ sub-cell  ✓")


def test_expand_conserves_volume_and_broadcasts():
    fr = subcell_vi_fractions(_prorata())
    out = expand_forecast_to_subcells(_forecast(), fr)
    # Volume conserved per (cell, gateway): sum over sub-cells == original
    orig = _forecast().set_index(["currency", "bank", "rpgt", "gateway"])["volume"]
    got = out.groupby(["currency", "bank", "rpgt", "gateway"])["volume"].sum()
    for k, v in orig.items():
        assert abs(float(got.loc[k]) - float(v)) < 1e-6, f"volume not conserved at {k}: {got.loc[k]} vs {v}"
    # Cell A gw1 (orig vol 600) → 600*0.7=420 and 600*0.3=180
    a1 = out[(out["bank"] == "400001") & (out["gateway"] == "gw1")].set_index(["pmp", "ctry"])["volume"]
    assert abs(a1.loc[("non_gp_ap", "usa")] - 420.0) < 1e-6 and abs(a1.loc[("applepay", "usa")] - 180.0) < 1e-6, a1.to_dict()
    # succ/risk/baseline_share broadcast unchanged
    for _, r in out[out["bank"] == "400001"].iterrows():
        base = 0.8 if r["gateway"] == "gw1" else 0.7
        assert abs(r["succ"] - base) < 1e-12, "succ must be broadcast unchanged (scoring stays cell grain)"
    # Cell C (absent from export) survives as a single '_all_' sub-cell with volume unchanged
    c = out[out["bank"] == "400003"]
    assert len(c) == 1 and c.iloc[0]["pmp"] == "_all_" and abs(c.iloc[0]["volume"] - 25.0) < 1e-9
    # sub-cell count: A has 2, B has 1, C has 1 → gw-rows: A(2gw*2)=4 + B(1) + C(1) = 6
    assert len(out) == 6, f"expected 6 sub-cell gateway rows, got {len(out)}"
    print("B. expand conserves volume per cell/gateway; rates broadcast; absent cell → one _all_ sub-cell  ✓")


if __name__ == "__main__":
    test_vi_fractions_sum_to_one_and_ignore_aged()
    test_expand_conserves_volume_and_broadcasts()
    print("PASS ✓  sub-cell volume-glue helpers: fractions, conservation, broadcast, fallbacks")
