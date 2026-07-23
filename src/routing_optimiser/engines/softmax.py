"""Softmax / proportional-allocation engine.

The slider (``weight`` in [0, 1]) is a risk dial, read as slider=100 at
``weight == 1`` down to slider=0 at ``weight == 0``:

  * slider=100 (weight=1): the *reference split* — a softmax over conversion
    only, floored at the exploration floor. No VAMP cap, no max-gateway-share,
    no MID constraints.
  * below 100 (the RISK LAYER, "Option A" / compliance dial): ONLY when the
    reference breaches the VAMP cap, move the least volume needed to bring the
    cell to a cap that interpolates from the reference's own risk (just below
    100 → barely trim) down to exactly the hard VAMP cap (slider 0 → just
    compliant). It never minimises risk *below* the cap:

        min  ||x - reference||^2
        s.t. sum(x) = 1,  floor <= x <= max_share,  x . risk <= cap(w)
        cap(w) = max(hard_cap, hard_cap + w * (r_ref - hard_cap))

    If the reference already meets the cap (the common case), the split is the
    reference at every slider position — so the per-cell slider is inert for
    already-compliant cells; the visible risk↔conversion gradient comes from the
    cross-cell blend/enforcement layer, not this engine.

Per-gateway risk comes from bin_rpgt_impact_export (period 0), attached to the
cell upstream. Cross-cell MID constraints are not expressible in this per-cell
interface; they are applied by the slider sweep / cross-cell projection layer.
"""
from __future__ import annotations

import numpy as np

from .base import BaseEngine, CellProblem, CellSolution

__build__ = "2026-07-22-softmax-analytic-projection"


class SoftmaxEngine(BaseEngine):
    key = "softmax"
    label = "Softmax allocation"
    description = ("Builds a conversion-only reference split (softmax over "
                   "success, floored so nothing goes dark), then moves off it "
                   "as the risk slider drops - staying as close to reference as "
                   "the VAMP cap and other hard constraints allow, and "
                   "minimising portfolio risk at the very bottom.")

    def _solve(self, p: CellProblem) -> CellSolution:
        self._t(f"=== CELL rpgt={p.rpgt} currency={p.currency} bank={p.bank} "
                f"| slider={self.w*100:.0f} (weight w={self.w:g}) ===")
        reference = self._reference_split(p)
        w = self.w

        # slider=100: pure reference, untouched by any risk constraint.
        if w >= 1.0 - 1e-9:
            self._t("STAGE C  slider=100 -> return REFERENCE unchanged (no risk logic)")
            return self._finalise(p, reference, "reference (slider=100)")

        lo, hi = self._bounds(p)
        eligible = hi > 0.0

        # The exploration floor is ALWAYS a hard minimum below slider=100, so no
        # eligible gateway can be driven dark even at maximum risk-aversion. It is
        # applied to PROVEN gateways only — auto-explore (capable-but-untested)
        # gateways are bounded by the reference's explore cap instead, not floored,
        # so a flood of unproven gateways can't be force-floored above the cap. Their
        # lower bound stays 0, so the compliance QP can still raise them ABOVE the cap
        # if that's the only way to meet the VAMP cap (the override).
        is_explore = np.asarray(getattr(p, "is_explore", np.zeros(p.n(), bool)), dtype=bool)
        proven_elig = eligible & ~is_explore
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        n_elig = int(eligible.sum())
        n_floor = int(proven_elig.sum()) or n_elig  # if all gateways are explore, floor them all
        if floor > 0.0 and n_floor > 0:
            floor = min(floor, 1.0 / n_floor)
            _floor_mask = proven_elig if proven_elig.any() else eligible
            lo = np.where(_floor_mask, np.maximum(lo, floor), lo)
        if lo.sum() > 1.0:                      # keep the floors feasible
            lo = lo * (1.0 / lo.sum())
        _bnds = list(zip(lo, hi))               # built once (reused only by the SLSQP fallback)

        risk = p.risk_rates
        hard_cap = self.hard.vamp_cap
        r_ref = float(reference @ risk)

        # OPTION A (compliance dial): only trade conversion for risk when the
        # reference breaks the VAMP cap, and STOP at compliance - never minimise
        # risk past the cap. If the reference already meets the cap (or there is
        # no cap), the split is the reference at every slider position.
        if hard_cap is None or r_ref <= hard_cap + 1e-12:
            self._t(f"STAGE C  reference risk {r_ref:.5f} <= cap "
                    f"{'none' if hard_cap is None else f'{hard_cap:g}'} -> compliant; "
                    "return REFERENCE (slider inactive)")
            return self._finalise(p, reference, "reference (compliant)")

        # Reference breaches the cap: interpolate the cap from the reference's own
        # risk (slider~100, barely trim) down to exactly the VAMP cap (slider 0,
        # just compliant). Never below the cap.
        cap = hard_cap + w * (r_ref - hard_cap)
        cap = max(cap, hard_cap)

        self._t("STAGE C  slider<100 -> compliance trim: min ||x-reference||^2 s.t. x.risk <= cap")
        self._t(f"           reference risk r_ref={r_ref:.5f}; VAMP cap={hard_cap:g}; effective cap={cap:.5f}")
        self._t(f"           bounds: floor={floor:g}, max_share={self.hard.max_gateway_share:g}")
        for g, rr in zip(p.gateways, risk):
            self._t(f"           risk[{g}]={rr:.4f}")

        # Compliance projection: min ||x-reference||^2 s.t. sum(x)=1, lo<=x<=hi, x·risk<=cap.
        # Strictly convex → unique optimum. Solve it EXACTLY with the purpose-built dual
        # projection (base._project_qp) rather than a general SLSQP call that can 'fail'.
        # Fall back to a warm-started SLSQP only if the analytic result violates a constraint.
        x = self._project_qp(reference, lo, hi, risk, cap)
        _ok = (abs(float(x.sum()) - 1.0) < 1e-6 and float(x @ risk) <= cap + 1e-7
               and bool((x >= lo - 1e-7).all()) and bool((x <= hi + 1e-7).all()))
        if not _ok:
            from scipy.optimize import minimize
            _x0 = getattr(p, "_last_qp_x", None)          # warm-start from the adjacent slider
            if _x0 is None or np.asarray(_x0).shape != reference.shape:
                _x0 = np.clip(reference, lo, hi)
                _s0 = _x0.sum()
                _x0 = _x0 / _s0 if _s0 > 0 else np.where(eligible, 1.0 / max(n_elig, 1), 0.0)
            _cons = [{"type": "eq", "fun": lambda z: z.sum() - 1.0, "jac": lambda z: np.ones_like(z)},
                     {"type": "ineq", "fun": lambda z: cap - z @ risk, "jac": lambda z: -risk}]
            res = minimize(lambda z: float(((z - reference) ** 2).sum()), _x0,
                           jac=lambda z: 2.0 * (z - reference), bounds=_bnds,
                           constraints=_cons, method="SLSQP",
                           options={"maxiter": 300, "ftol": 1e-12})
            if res.success:
                x = np.clip(res.x, 0.0, None)
                _s = x.sum(); x = x / _s if _s > 0 else reference
            else:
                # Both solvers failed: project the reference onto the cap (min movement) and
                # flag infeasible — never report the raw breaching reference. (A4)
                self._t(f"STAGE D  INFEASIBLE ({res.message}); VAMP-projecting the reference, flagged infeasible")
                proj = self._project_to_vamp(p, reference)
                proj = np.clip(proj, 0.0, None)
                _s = proj.sum(); proj = proj / _s if _s > 0 else reference
                return CellSolution(proj, float(proj @ p.success_rates), float(proj @ risk), False,
                                    f"infeasible w={w:g}; VAMP-projected reference")

        try:
            p._last_qp_x = x.copy()   # seed the next slider position's fallback warm-start
        except Exception:  # noqa: BLE001
            pass
        self._t("STAGE D  solved (dual projection). shares: "
                + ", ".join(f"{g}={xx:.3f}" for g, xx in zip(p.gateways, x)))
        self._t(f"STAGE E  portfolio risk={float(x @ risk):.5f} (cap={cap:.5f})")
        # Build the solution directly (skip the legacy VAMP re-projection in _finalise, which
        # would snap every sub-100 position to the hard cap and flatten the slider gradient).
        return CellSolution(x, float(x @ p.success_rates), float(x @ risk),
                            self._is_feasible(p, x), f"compliance-trim w={w:g} cap={cap:.4f}")