"""Genetic engine's OWN conversion reference — a revenue-greedy waterfall.

The genetic engine is a cross-cell per-vampMid tilt GA that starts from a per-cell
conversion reference and tilts it toward VAMP compliance. That reference must be
revenue-optimal, but it must be GENETIC'S OWN — not borrowed from the softmax
engine. This builds it independently: greedily fill each cell's volume onto its
highest-success (revenue-per-attempt) gateways up to the max-gateway-share cap,
then floor the rest so nothing goes dark. It's the max-revenue split given the
share cap, with NO softmax temperature/exponential — deterministic and distinct.
"""
from __future__ import annotations

import numpy as np

from .base import BaseEngine, CellProblem, CellSolution

__build__ = "2026-07-17-genetic-revenue-reference"


class GeneticRefEngine(BaseEngine):
    key = "genetic_ref"
    label = "Genetic revenue reference"
    description = ("Revenue-greedy waterfall: fills each cell's best-converting "
                   "gateways up to the max share, then floors the rest. Genetic's "
                   "own conversion reference — independent of softmax (no "
                   "temperature/exponential).")

    def _solve(self, p: CellProblem) -> CellSolution:
        n = p.n()
        _, hi = self._bounds(p)
        eligible = hi > 0.0
        if not eligible.any():
            return self._finalise(p, np.full(n, 1.0 / n), "no eligible gateway")
        # Revenue ∝ success rate within a cell (ticket ~constant per currency×bank),
        # so allocate to the highest success rate first.
        sr = np.where(eligible, np.asarray(p.success_rates, float), -np.inf)
        cap = float(self.hard.max_gateway_share) if self.hard.max_gateway_share else 1.0
        # Waterfall: best gateway up to `cap`, then the next, … until 1.0 is allocated.
        shares = np.zeros(n)
        remaining = 1.0
        for idx in np.argsort(-sr, kind="stable"):
            if not eligible[idx] or remaining <= 1e-12:
                break
            take = min(cap, remaining)
            shares[idx] = take
            remaining -= take
        if remaining > 1e-9:                       # caps can't absorb 1.0 → spread residual on eligible
            shares[eligible] += remaining / int(eligible.sum())
        # Exploration floor: every eligible gateway keeps >= floor, then renormalise.
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        n_elig = int(eligible.sum())
        if floor > 0.0 and n_elig > 0:
            floor = min(floor, 1.0 / n_elig)
            shares = np.where(eligible, np.maximum(shares, floor), 0.0)
            _s = shares.sum()
            shares = shares / _s if _s > 0 else eligible / n_elig
        return self._finalise(p, shares, "genetic revenue reference (waterfall)")
