"""Softmax / proportional-allocation engine.

The slider (``weight`` in [0, 1]) is a risk dial, read as slider=100 at
``weight == 1`` down to slider=0 at ``weight == 0``:

  * slider=100 (weight=1): the *reference split* — a softmax over conversion
    only, floored at the exploration floor. No VAMP cap, no max-gateway-share,
    no MID constraints.
  * below 100 (the RISK LAYER, "Option A" / compliance dial): ONLY when the
    reference breaches the VAMP cap, move the least volume needed to bring the
    profile to a cap that interpolates from the reference's own risk (just below
    100 → barely trim) down to exactly the hard VAMP cap (slider 0 → just
    compliant). It never minimises risk *below* the cap:

        min  ||x - reference||^2
        s.t. sum(x) = 1,  floor <= x <= max_share,  x . risk <= cap(w)
        cap(w) = max(hard_cap, hard_cap + w * (r_ref - hard_cap))

    If the reference already meets the cap (the common case), the split is the
    reference at every slider position — so the per-profile slider is inert for
    already-compliant profiles; the visible risk↔conversion gradient comes from the
    cross-profile blend/enforcement layer, not this engine.

ANALOGY for the whole engine: first draw the "ideal" split that chases conversion
(the reference). Then, only if that ideal is too risky, walk it back toward safety
JUST FAR ENOUGH to sit on the risk ceiling — like nudging an over-full glass back
to the fill line, spilling as little as possible, and never pouring below the line.

Per-gateway risk comes from bin_rpgt_impact_export (period 0), attached to the
profile upstream. Cross-profile MID constraints are not expressible in this per-profile
interface; they are applied by the slider sweep / cross-profile projection layer.
"""
from __future__ import annotations

import numpy as np

from ..engines.base import BaseEngine, ProfileProblem, ProfileSolution

__build__ = "2026-07-22-softmax-analytic-projection"


class SoftmaxEngine(BaseEngine):
    key = "softmax"
    label = "Softmax allocation"
    description = ("Builds a conversion-only reference split (softmax over "
                   "success, floored so nothing goes dark), then moves off it "
                   "as the risk slider drops - staying as close to reference as "
                   "the VAMP cap and other hard constraints allow, and "
                   "minimising portfolio risk at the very bottom.")

    # [FN-418]
    def _solve(self, p: ProfileProblem) -> ProfileSolution:
        """Return the profile's split: the reference, trimmed toward the risk cap if needed.

        Uses guard clauses to bail early on the two common cases (slider at 100, or
        the reference already compliant) before doing the compliance projection.
        """
        self._t(f"=== PROFILE rpgt={p.rpgt} currency={p.currency} bin={p.bin} "
                f"| slider={self.w*100:.0f} (weight w={self.w:g}) ===")
        reference = self._reference_split(p)
        slider_weight = self.w

        # Guard — slider=100: pure reference, untouched by any risk constraint.
        if slider_weight >= 1.0 - 1e-9:
            self._t("STAGE C  slider=100 -> return REFERENCE unchanged (no risk logic)")
            return self._finalise(p, reference, "reference (slider=100)")

        lower, upper = self._bounds(p)
        eligible = upper > 0.0

        # The exploration floor is ALWAYS a hard minimum below slider=100, so no
        # eligible PROVEN gateway can be driven dark even at maximum risk-aversion. It is
        # applied to PROVEN gateways only — auto-explore (capable-but-untested) gateways
        # are NOT force-floored (their lower bound stays 0), so a flood of unproven
        # gateways can't be pinned above a minimum; they're scored freely on their rate
        # and the compliance QP can still raise them if needed to meet the VAMP cap.
        is_explore = np.asarray(getattr(p, "is_explore", np.zeros(p.n(), bool)), dtype=bool)
        proven_eligible = eligible & ~is_explore
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        eligible_count = int(eligible.sum())
        floor_count = int(proven_eligible.sum()) or eligible_count  # all-explore → floor them all
        if floor > 0.0 and floor_count > 0:
            floor = min(floor, 1.0 / floor_count)
            floor_mask = proven_eligible if proven_eligible.any() else eligible
            lower = np.where(floor_mask, np.maximum(lower, floor), lower)
        if lower.sum() > 1.0:                      # keep the floors jointly feasible
            lower = lower * (1.0 / lower.sum())
        bounds_list = list(zip(lower, upper))      # built once (reused only by the SLSQP fallback)

        risk = p.risk_rates
        hard_cap = self.hard.vamp_cap
        reference_risk = float(reference @ risk)

        # OPTION A (compliance dial): only trade conversion for risk when the
        # reference breaks the VAMP cap, and STOP at compliance - never minimise
        # risk past the cap. Guard: if the reference already meets the cap (or there
        # is no cap), the split is the reference at every slider position.
        if hard_cap is None or reference_risk <= hard_cap + 1e-12:
            self._t(f"STAGE C  reference risk {reference_risk:.5f} <= cap "
                    f"{'none' if hard_cap is None else f'{hard_cap:g}'} -> compliant; "
                    "return REFERENCE (slider inactive)")
            return self._finalise(p, reference, "reference (compliant)")

        # Reference breaches the cap: interpolate the cap from the reference's own
        # risk (slider~100, barely trim) down to exactly the VAMP cap (slider 0,
        # just compliant). Never below the cap.
        cap = hard_cap + slider_weight * (reference_risk - hard_cap)
        cap = max(cap, hard_cap)

        self._t("STAGE C  slider<100 -> compliance trim: min ||x-reference||^2 s.t. x.risk <= cap")
        self._t(f"           reference risk r_ref={reference_risk:.5f}; VAMP cap={hard_cap:g}; effective cap={cap:.5f}")
        self._t(f"           bounds: floor={floor:g}, max_share={self.hard.max_gateway_share:g}")
        for gateway, gateway_risk in zip(p.gateways, risk):
            self._t(f"           risk[{gateway}]={gateway_risk:.4f}")

        # Compliance projection: min ||x-reference||^2 s.t. sum(x)=1, lo<=x<=hi, x·risk<=cap.
        # Strictly convex → unique optimum. Solve it EXACTLY with the purpose-built dual
        # projection (base._project_qp) rather than a general SLSQP call that can 'fail'.
        # Fall back to a warm-started SLSQP only if the analytic result violates a constraint.
        solution = self._project_qp(reference, lower, upper, risk, cap)
        is_valid = (abs(float(solution.sum()) - 1.0) < 1e-6 and float(solution @ risk) <= cap + 1e-7
                    and bool((solution >= lower - 1e-7).all()) and bool((solution <= upper + 1e-7).all()))
        if not is_valid:
            from scipy.optimize import minimize
            warm_start = getattr(p, "_last_qp_x", None)          # warm-start from the adjacent slider
            if warm_start is None or np.asarray(warm_start).shape != reference.shape:
                warm_start = np.clip(reference, lower, upper)
                warm_total = warm_start.sum()
                warm_start = (warm_start / warm_total if warm_total > 0
                              else np.where(eligible, 1.0 / max(eligible_count, 1), 0.0))
            constraints = [{"type": "eq", "fun": lambda z: z.sum() - 1.0, "jac": lambda z: np.ones_like(z)},
                           {"type": "ineq", "fun": lambda z: cap - z @ risk, "jac": lambda z: -risk}]
            result = minimize(lambda z: float(((z - reference) ** 2).sum()), warm_start,
                              jac=lambda z: 2.0 * (z - reference), bounds=bounds_list,
                              constraints=constraints, method="SLSQP",
                              options={"maxiter": 300, "ftol": 1e-12})
            if result.success:
                solution = np.clip(result.x, 0.0, None)
                total = solution.sum()
                solution = solution / total if total > 0 else reference
            else:
                # Both solvers failed: project the reference onto the cap (min movement) and
                # flag infeasible — never report the raw breaching reference. (A4)
                self._t(f"STAGE D  INFEASIBLE ({result.message}); VAMP-projecting the reference, flagged infeasible")
                projected = self._project_to_vamp(p, reference)
                projected = np.clip(projected, 0.0, None)
                total = projected.sum()
                projected = projected / total if total > 0 else reference
                return ProfileSolution(projected, float(projected @ p.success_rates), float(projected @ risk), False,
                                    f"infeasible w={slider_weight:g}; VAMP-projected reference")

        try:
            p._last_qp_x = solution.copy()   # seed the next slider position's fallback warm-start
        except Exception:  # noqa: BLE001
            pass
        self._t("STAGE D  solved (dual projection). shares: "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, solution)))
        self._t(f"STAGE E  portfolio risk={float(solution @ risk):.5f} (cap={cap:.5f})")
        # Build the solution directly (skip the legacy VAMP re-projection in _finalise, which
        # would snap every sub-100 position to the hard cap and flatten the slider gradient).
        return ProfileSolution(solution, float(solution @ p.success_rates), float(solution @ risk),
                            self._is_feasible(p, solution), f"compliance-trim w={slider_weight:g} cap={cap:.4f}")
