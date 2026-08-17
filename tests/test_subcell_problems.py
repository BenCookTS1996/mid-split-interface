"""Unit tests for the sub-cell CellProblem schema extension + build_subcell_problems.

Locks in:
  * CellProblem gains pmp/ctry with '_all_' defaults → existing constructions are unaffected
    (backward-compatible schema change);
  * build_subcell_problems yields one CellProblem per (rpgt×currency×bank×pmp×Country) sub-cell;
  * success rates are joined at CELL grain and BROADCAST (same gateway rate across a cell's
    sub-cells — the decision grain is sub-cell, the scoring grain stays cell);
  * volume is conserved (sub-cell volumes sum back to the cell total);
  * chains cleanly off the real expand_forecast_to_subcells volume glue.

Pure pandas/numpy — no BigQuery/Streamlit. Run: python tests/test_subcell_problems.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.engines import CellProblem  # noqa: E402
from routing_optimiser.data_loader import build_subcell_problems  # noqa: E402
from routing_optimiser.subcell import subcell_vi_fractions, expand_forecast_to_subcells  # noqa: E402


def test_cellproblem_defaults_backward_compatible():
    p = CellProblem(rpgt="monthly initial", currency="usd", bank="400001",
                    gateways=["gw1"], success_rates=np.array([0.8]),
                    risk_rates=np.array([0.02]), volume=100.0, baseline_shares=np.array([1.0]))
    assert p.pmp == "_all_" and p.ctry == "_all_", "new fields must default to cell grain"
    print("A. CellProblem pmp/ctry default to '_all_' (existing constructions unaffected)  ✓")


def _success_rates():
    # CELL-grain success rates (no pmp/Country) — two gateways in one cell.
    return pd.DataFrame([
        dict(rpgt="monthly initial", currency="usd", bank="400001", gateway="gw1",
             success_rate=0.80, success=800.0, attempts=1000.0),
        dict(rpgt="monthly initial", currency="usd", bank="400001", gateway="gw2",
             success_rate=0.70, success=700.0, attempts=1000.0),
    ])


def _forecast():
    return pd.DataFrame([
        dict(currency="usd", bank="400001", rpgt="monthly initial", gateway="gw1",
             volume=600.0, baseline_share=0.6, risk_rate=0.02),
        dict(currency="usd", bank="400001", rpgt="monthly initial", gateway="gw2",
             volume=400.0, baseline_share=0.4, risk_rate=0.03),
    ])


def _prorata_70_30():
    return pd.DataFrame([
        dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="non_gp_ap",
             Country="usa", VI_Txn_Count=700.0, t=0, vampMid="A"),
        dict(Currency="usd", BIN="400001", RPGT="Monthly Initial", paymentMethodProvider="applepay",
             Country="usa", VI_Txn_Count=300.0, t=0, vampMid="A"),
    ])


def test_build_subcell_problems():
    fr = subcell_vi_fractions(_prorata_70_30())
    sub_fc = expand_forecast_to_subcells(_forecast(), fr)
    probs = build_subcell_problems(sub_fc, _success_rates())

    # one CellProblem per sub-cell (2), each carrying its pmp/ctry
    assert len(probs) == 2, f"expected 2 sub-cell problems, got {len(probs)}"
    by_sub = {(p.pmp, p.ctry): p for p in probs}
    assert set(by_sub) == {("non_gp_ap", "usa"), ("applepay", "usa")}, list(by_sub)

    for p in probs:
        assert p.bank == "400001" and p.rpgt == "monthly initial"   # bank stays raw BIN
        # success rates BROADCAST from cell grain: gw1=0.80, gw2=0.70 in EVERY sub-cell
        rates = dict(zip(p.gateways, p.success_rates))
        assert abs(rates["gw1"] - 0.80) < 1e-12 and abs(rates["gw2"] - 0.70) < 1e-12, rates

    # volume conserved: 70% / 30% of the 1000-cell total
    vol = {(p.pmp, p.ctry): p.volume for p in probs}
    assert abs(vol[("non_gp_ap", "usa")] - 700.0) < 1e-6, vol
    assert abs(vol[("applepay", "usa")] - 300.0) < 1e-6, vol
    assert abs(sum(vol.values()) - 1000.0) < 1e-6, "sub-cell volumes must sum to the cell total"
    print("B. build_subcell_problems: one problem per sub-cell; rates broadcast; volume conserved  ✓")


if __name__ == "__main__":
    test_cellproblem_defaults_backward_compatible()
    test_build_subcell_problems()
    print("PASS ✓  sub-cell CellProblem schema + assembler")
