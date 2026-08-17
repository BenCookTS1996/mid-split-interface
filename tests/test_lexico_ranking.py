"""Validate the STRICT LEXICOGRAPHIC M5-first ranking in genetic_fullmatrix.

Priority (best first):
  1. lower per-MID M5 band breach  (strict primary — never traded away)
  2. lower engineering violation   (global VAMP cap + max-share)
  3. higher VWSR                   (conversion)

Run: python tests/test_lexico_ranking.py   (pure numpy)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.genetic_fullmatrix import _rank, _key_of, _best_index, _FEAS_EPS  # noqa: E402


def test_band_dominates_everything():
    # A: low band, terrible eng-viol, terrible VWSR.  B: slightly higher band, perfect eng-viol + VWSR.
    vwsr = np.array([0.10, 0.99])
    other = np.array([5.00, 0.00])
    band = np.array([0.0050, 0.0060])
    order = _rank(vwsr, other, band)
    assert order[0] == 0, "band breach must dominate eng-viol and VWSR (A should win)"
    # key comparison agrees
    kA = _key_of(vwsr[0], other[0], band[0]); kB = _key_of(vwsr[1], other[1], band[1])
    assert kA > kB
    print("A. lower M5 band breach wins even with worse eng-viol AND worse VWSR  ✓")


def test_compliant_beats_any_breach():
    # A: compliant (band 0) but low VWSR.  B: tiny breach but huge VWSR.
    vwsr = np.array([0.01, 0.99]); other = np.array([0.0, 0.0]); band = np.array([0.0, 1e-4])
    order = _rank(vwsr, other, band)
    assert order[0] == 0, "a compliant split must outrank any breaching one, regardless of VWSR"
    print("B. compliant (band=0) always beats a breaching split  ✓")


def test_among_compliant_engviol_then_vwsr():
    # All band-compliant. Rank should go by lower eng-viol first, then higher VWSR.
    vwsr = np.array([0.90, 0.95, 0.80])
    other = np.array([0.10, 0.20, 0.10])
    band = np.array([0.0, 0.0, 0.0])
    order = list(_rank(vwsr, other, band))
    # cands 0 and 2 have eng-viol 0.10 (beat cand1 at 0.20); between them higher VWSR (0.90>0.80) wins
    assert order[0] == 0 and order[1] == 2 and order[2] == 1, f"order was {order}"
    print("C. among compliant: lower eng-viol first, then higher VWSR  ✓")


def test_eps_snaps_compliant():
    # band just under _FEAS_EPS counts as compliant (== 0), so VWSR/other decide, not the tiny breach.
    e = _FEAS_EPS
    vwsr = np.array([0.5, 0.9]); other = np.array([0.0, 0.0]); band = np.array([e * 0.5, 0.0])
    order = _rank(vwsr, other, band)
    assert order[0] == 1, "sub-eps breach should snap to compliant so higher VWSR wins"
    print("D. breaches <= _FEAS_EPS snap to compliant (no float churn)  ✓")


def test_never_worse_key_monotonic():
    # The seed key must not be beaten by a higher-VWSR split that raises band breach.
    seed_key = _key_of(0.50, 0.0, 0.0056)          # seed: modest VWSR, small breach
    drifter = _key_of(0.60, 0.0, 0.0115)           # higher VWSR but WORSE breach
    assert not (drifter > seed_key), "a worse-breach split must NOT outrank the seed (the old leak)"
    improver = _key_of(0.40, 0.0, 0.0040)          # lower VWSR but BETTER breach
    assert improver > seed_key, "a better-breach split SHOULD outrank the seed"
    print("E. never-worse holds on band: worse-breach can't beat seed; better-breach can  ✓")


def test_legacy_band_none():
    # band=None → feasibility-first on 'other' then VWSR (back-compat for any legacy caller).
    vwsr = np.array([0.1, 0.9]); other = np.array([0.0, 0.5])
    order = _rank(vwsr, other, None)
    assert order[0] == 0, "with band=None, lower 'other' viol wins"
    assert _best_index(vwsr, other, None) == 0
    print("F. band=None legacy path still feasibility-first on viol  ✓")


if __name__ == "__main__":
    test_band_dominates_everything()
    test_compliant_beats_any_breach()
    test_among_compliant_engviol_then_vwsr()
    test_eps_snaps_compliant()
    test_never_worse_key_monotonic()
    test_legacy_band_none()
    print("PASS ✓  strict lexicographic M5-first ranking behaves correctly")
