"""Minimal sanity tests. Run: python scripts/test_engines.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser import get_engine, HardConstraints, SoftConstraints, ENGINES
from routing_optimiser.engines import CellProblem

def cell():
    return CellProblem("Monthly Initial","USD","BANK",
        ["a","b","c"], np.array([0.92,0.88,0.85]), np.array([0.010,0.006,0.004]),
        1000.0, np.array([0.34,0.33,0.33]), np.array([460,440,425]), np.array([500,500,500]))

def test_all_engines_feasible():
    hard=HardConstraints(max_gateway_share=0.97, vamp_cap=0.009)
    soft=SoftConstraints(exploration_floor=0.05)
    for k in ENGINES:
        s=get_engine(k,0.7,hard,soft).solve(cell())
        assert abs(s.shares.sum()-1.0) < 1e-6, f"{k}: shares must sum to 1"
        assert s.expected_risk_rate <= 0.009 + 1e-6, f"{k}: must respect VAMP cap"
        assert s.feasible, f"{k}: should be feasible"
    print("OK: all engines feasible and VAMP-compliant")

def test_slider_monotonic():
    hard=HardConstraints(max_gateway_share=0.97, vamp_cap=0.02)
    soft=SoftConstraints(exploration_floor=0.0)
    lo=get_engine("entropy",0.0,hard,soft).solve(cell())
    hi=get_engine("entropy",1.0,hard,soft).solve(cell())
    assert hi.expected_success_rate >= lo.expected_success_rate - 1e-9, \
        "higher weight should not reduce success"
    print("OK: slider trades conversion for risk as expected")

if __name__ == "__main__":
    test_all_engines_feasible()
    test_slider_monotonic()
    print("\nAll tests passed.")
