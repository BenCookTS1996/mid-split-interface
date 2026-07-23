"""Entropy-penalised convex optimisation engine.

Keeps the constrained-optimisation backbone (hard VAMP + cap + eligibility)
but adds an entropy bonus to the objective so the optimum is an interior,
diversified split instead of a 0/100 corner. This is the recommended default:
smallest change from LP, but no more corner solutions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .base import BaseEngine, CellProblem, CellSolution


class EntropyEngine(BaseEngine):
    key = "entropy"
    label = "Entropy-penalised optimisation"
    description = ("Maximises the conversion/risk score PLUS an entropy bonus "
                   "that rewards spreading traffic out. Gives smooth, "
                   "diversified splits while still honouring the hard VAMP "
                   "cap and gateway caps. Recommended default.")

    def _solve(self, p: CellProblem) -> CellSolution:
        n = p.n()
        lam = float(self.params.get("entropy_lambda", 0.02))
        lo, hi = self._bounds(p)
        score = self._score(p)

        def neg_obj(x):
            x = np.clip(x, 1e-12, 1.0)
            ent = -np.sum(x * np.log(x))
            return -(x @ score + lam * ent)

        def neg_grad(x):
            x = np.clip(x, 1e-12, 1.0)
            return -(score + lam * (-(np.log(x) + 1.0)))

        cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0,
                 "jac": lambda x: np.ones_like(x)}]
        if self.hard.vamp_cap is not None:
            cons.append({
                "type": "ineq",
                "fun": lambda x: self.hard.vamp_cap - x @ p.risk_rates,
                "jac": lambda x: -p.risk_rates,
            })

        x0 = np.clip(p.baseline_shares, lo + 1e-6, hi)
        x0 = x0 / x0.sum() if x0.sum() > 0 else np.full(n, 1.0 / n)

        res = minimize(neg_obj, x0, jac=neg_grad, bounds=list(zip(lo, hi)),
                       constraints=cons, method="SLSQP",
                       options={"maxiter": 200, "ftol": 1e-9})
        x = res.x if res.success else x0
        note = f"entropy lambda={lam:g}" + ("" if res.success else " (fallback x0)")
        return self._finalise(p, x, note)
