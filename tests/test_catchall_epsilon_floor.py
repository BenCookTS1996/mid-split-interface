"""Validate floor_catchall_shares (the seed ε-floor helper).

NOTE: with the CELL-LEVEL catch-all fix (2026-08-16) the catch-all no longer re-adds a zeroed gateway
in a routed cell, so the seed ε-floor is now largely redundant for routed cells — but the helper is
kept (it's harmless and still guarantees no seed leaves a catch-all gateway at exactly 0), so we keep
its unit test. The old "deploy-truth" test (which asserted the ~10.6% per-gateway re-add) was removed
because that behaviour has been deliberately corrected — see tests/test_cell_level_catchall.py.

Run: python tests/test_catchall_epsilon_floor.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from routing_optimiser.exact_band_solver import floor_catchall_shares  # noqa: E402


def test_floor_helper():
    cell_starts = np.array([0, 4, 8], dtype=np.intp)
    cell_counts = np.array([4, 4, 4], dtype=np.intp)
    mask = np.array([True, False, False, False] * 3, dtype=bool)  # gw0 of each cell is catch-all
    eps = 0.001
    rng = np.random.default_rng(3)
    for _ in range(500):
        s = rng.random(12)
        for a, c in zip(cell_starts, cell_counts):
            seg = s[a:a + c]
            if rng.random() < 0.6:
                seg[0] = 0.0
            s[a:a + c] = seg / seg.sum() if seg.sum() > 0 else seg
        out = floor_catchall_shares(s, mask, eps, cell_starts, cell_counts)
        for a, c in zip(cell_starts, cell_counts):
            assert abs(out[a:a + c].sum() - s[a:a + c].sum()) < 1e-12, "cell sum not preserved"
            assert out[a] >= eps - 1e-12, f"masked gw below floor: {out[a]}"
            assert (out[a:a + c][1:] >= -1e-12).all(), "negative donor share"
    # idempotent + no-op guards
    s = np.array([0.0, 0.5, 0.3, 0.2, 0.001, 0.4, 0.3, 0.299])
    cs = np.array([0, 4], np.intp); cn = np.array([4, 4], np.intp)
    m = np.array([True, False, False, False, True, False, False, False], bool)
    o1 = floor_catchall_shares(s, m, 0.001, cs, cn)
    assert np.max(np.abs(o1 - floor_catchall_shares(o1, m, 0.001, cs, cn))) < 1e-12, "not idempotent"
    assert np.array_equal(floor_catchall_shares(s, None, 0.001, cs, cn), s)
    assert np.array_equal(floor_catchall_shares(s, m, 0.0, cs, cn), s)
    print("floor_catchall_shares: sum-preserving, floors masked rows, idempotent, no-op  ✓")


if __name__ == "__main__":
    test_floor_helper()
    print("PASS ✓")
