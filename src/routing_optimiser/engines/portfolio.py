"""Mean-CVaR (portfolio) engine — downside-risk reference, auto-calibrated.

Reuses the softmax engine's slider + compliance machinery wholesale and changes
only the REFERENCE split (the slider=100 allocation).

Treats gateways like investments: expected conversion is the "return", and the
*downside* uncertainty of each gateway's risk (VAMP) is what we pay to avoid.
Unlike a symmetric variance penalty (which punishes a gateway's risk coming in
*better* than expected just as much as *worse*), this prices only the BAD tail —
how much a gateway's VAMP rate could plausibly spike above its expected level.

ANALOGY: like building a share portfolio. You want high return (conversion) but you
also don't want a nasty surprise (a VAMP spike). Variance would count a *pleasant*
surprise as "risk" too; CVaR (this engine) only charges for the plausible WORST case,
so it diversifies away from volatile or barely-tested gateways while still leaning
into the clearly-best low-risk ones.

For each gateway the risk-rate estimate has standard error σ = √(r(1−r)/n). The
one-sided 95% expected-shortfall (CVaR) of a normal tail is ≈ 2.06·σ, so the
per-gateway downside is dₘ = 2.06·σ, and the portfolio's downside (independent
gateways) is √(Σ xₘ²·dₘ²). The reference maximises

    Σ xₘ·successₘ   −   γ · √(Σ xₘ²·dₘ²)

subject to sum(x)=1 and the floor / max-share bounds. This diversifies and, in
particular, trims gateways whose VAMP could spike (volatile or thinly-tested).

γ is AUTO-CALIBRATED per cell — no user dial. It's scaled so a gateway with an
average downside pays a fixed fraction of its return, which makes the trade-off
mean the same thing in every cell regardless of the absolute risk scale or how
much data a cell has. (Contrast with Thompson: Thompson *explores* thin gateways;
this *avoids* them.) The MEAN VAMP cap is still enforced downstream by the shared
compliance layer — this engine only prices risk *stability*. Deterministic.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta as _betadist

from .base import CellProblem
from .softmax import SoftmaxEngine

__build__ = "2026-07-22-portfolio-fista+infeasible-flag"

_Q_TAIL = 0.95      # upper-tail quantile for the (skew-aware) downside risk deviation
_AVERSION = 0.40    # fixed, dimensionless: an average-downside gateway pays ~0.4× its return


class PortfolioEngine(SoftmaxEngine):
    key = "portfolio"
    label = "Portfolio (mean-CVaR)"
    description = ("Balances expected conversion ('return') against the DOWNSIDE "
                   "risk of each gateway's VAMP spiking (CVaR, not symmetric "
                   "variance), diversifying and shying away from volatile or "
                   "thinly-tested gateways. Risk aversion is auto-calibrated — "
                   "no dial to set.")

    def _ref_param_key(self, p: CellProblem):
        # The CVaR reference depends only on prior_count (plus per-cell risk_n / attempts,
        # which are immutable on `p`). Temperature / γ / explore caps don't affect it, so
        # they're dropped from the key — a temperature change won't invalidate this cache.
        return (round(float(self.params.get("prior_count", 30.0)), 9),)

    def _reference_split_impl(self, p: CellProblem) -> np.ndarray:
        """slider=100 reference: mean-CVaR optimal (conversion vs downside VAMP
        risk). Same contract as ``SoftmaxEngine._reference_split_impl``. Wrapped by
        the base-class reference cache (computed once per cell, reused across dials).
        NOTE: dormant engine — builds its own reference and does not apply the
        auto-explore share cap (that lives in the base softmax reference)."""
        gateway_count = p.n()
        lower, upper = self._bounds(p)
        eligible = upper > 0.0
        # Guard: nothing eligible → spread evenly and bail.
        if not eligible.any():
            return np.full(gateway_count, 1.0 / gateway_count)

        prior_n = float(self.params.get("prior_count", 30.0))
        risk_rates = np.clip(np.asarray(p.risk_rates, float), 0.0, 1.0)
        # Sample size for the RISK-rate standard error = the VAMP rate's OWN denominator
        # (the transaction/sales count, risk_n) when available — not auth attempts, which
        # are a different dataset. Falls back to attempts, then a small prior. (C1)
        fallback_n = (np.asarray(p.obs_attempts, float) if p.obs_attempts is not None
                      else np.full(gateway_count, prior_n))
        if p.risk_n is not None:
            risk_n = np.asarray(p.risk_n, float)
            sample_size = np.where(risk_n > 0, risk_n, np.where(fallback_n > 0, fallback_n, prior_n))
        else:
            sample_size = np.where(fallback_n > 0, fallback_n, prior_n)
        sample_size = np.maximum(sample_size, 1.0)

        # Per-gateway DOWNSIDE (upper-tail) risk deviation: how much worse than expected
        # the VAMP rate could plausibly be. VAMP rates are rare (~0.6%) and right-skewed,
        # so a normal-tail factor (2.06·σ) UNDER-prices the bad tail. Instead use the
        # actual upper-95% quantile of the risk-rate's Beta posterior minus its mean —
        # skew-aware, and heavier exactly for the low-rate gateways that can spike. (C2)
        beta_a = risk_rates * sample_size + 0.5                    # Jeffreys prior on the risk rate
        beta_b = np.maximum(1.0 - risk_rates, 0.0) * sample_size + 0.5
        upper_q95 = _betadist.ppf(_Q_TAIL, beta_a, beta_b)
        downside = np.where(eligible & np.isfinite(upper_q95), np.maximum(upper_q95 - risk_rates, 0.0), 0.0)
        returns = np.where(eligible, np.asarray(p.success_rates, float), 0.0)

        # Exploration floor as a hard lower bound (nothing eligible goes dark).
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        eligible_count = int(eligible.sum())
        if floor > 0.0 and eligible_count > 0:
            floor = min(floor, 1.0 / eligible_count)
            lower = np.where(eligible, np.maximum(lower, floor), lower)
        if lower.sum() > 1.0:
            lower = lower * (1.0 / lower.sum())

        # AUTO risk-aversion (no dial): scale the penalty so it is comparable to
        # the return magnitude in THIS cell. Then a gateway with an average
        # downside pays ~_AVERSION of its return, above-average (volatile / thin)
        # gateways get trimmed, and a clearly-best low-downside gateway keeps its
        # share — consistently across cells whatever the absolute risk scale.
        mean_return = float(returns[eligible].mean()) if eligible_count else 0.0
        downside_eligible = downside[eligible]
        downside_eligible = downside_eligible[np.isfinite(downside_eligible)]   # guard against Beta-ppf NaNs
        mean_downside = float(downside_eligible.mean()) if downside_eligible.size else 0.0
        # Auto risk-aversion. The penalty is self-normalising (γ·downside ≈ AVERSION·return),
        # but FLOOR the denominator by a small fraction of the return so a near-zero-dispersion
        # cell can't blow γ up (or divide by ~0) — it just concentrates on the best converter,
        # which is correct when there's no risk spread to diversify against. Also cap γ so a
        # pathological cell can't force a degenerate uniform split. (C1/γ-degeneracy)
        downside_floor = max(mean_downside, 1e-4 * max(mean_return, 1e-9))
        gamma = float(np.clip(_AVERSION * mean_return / downside_floor, 0.0, 5000.0)) if mean_return > 0 else 0.0

        self._t(f"STAGE B1  reference: MEAN-CVaR (downside), auto γ={gamma:.1f} "
                f"(_AVERSION={_AVERSION}, mean_return={mean_return:.4f}, mean_downside={mean_downside:.5f})")
        for gateway, return_g, downside_g, is_eligible in zip(p.gateways, returns, downside, eligible):
            self._t(f"           {gateway}: success={return_g:.4f}, downside(CVaR)={downside_g:.5f}"
                    + ("" if is_eligible else "  (ineligible)"))

        # The mean-CVaR objective f(x) = −return + γ·portfolio_downside, its gradient, and a
        # projection back onto the valid-split set. Minimised below with FISTA.
        eps = 1e-12
        downside_sq = downside * downside

        def _f(x):
            return float(-(x @ returns) + gamma * np.sqrt(float((downside_sq * x * x).sum()) + eps))

        def _grad(x):
            norm = np.sqrt(float((downside_sq * x * x).sum()) + eps)
            return -returns + gamma * (downside_sq * x) / norm

        def _proj(v):
            return self._project_box_simplex(v, lower, upper)

        # Return-weighted feasible start / fallback (a revenue-lean split is a sensible,
        # clearly-non-uniform default — a uniform split would masquerade as 'diversified'). (C4)
        return_weighted = np.where(eligible, np.maximum(returns, 1e-9), 0.0)
        x_start = _proj(return_weighted) if return_weighted.sum() > 0 else _proj(np.where(eligible, 1.0, 0.0).astype(float))

        # Accelerated projected gradient (FISTA) with backtracking — a purpose-built convex
        # solve for the mean-CVaR SOCP, replacing a general SLSQP call that can 'fail'. The
        # objective is convex so this converges to the unique optimum; fully deterministic.
        # ANALOGY: an accelerated walk downhill to the best split. `y` peeks a little AHEAD
        # (momentum) each step; if a step overshoots (fails the sufficient-decrease test) we
        # halve the step size (double inv_step) and retry — easing off on bumpy ground — until
        # the iterate stops moving.
        x = x_start.copy(); y = x_start.copy(); t = 1.0; inv_step = 1.0
        for _ in range(400):
            grad = _grad(y)
            f_y = _f(y)
            x_next = y
            for _backtrack in range(80):                 # backtracking: shrink the step (grow inv_step)
                x_next = _proj(y - grad / inv_step)
                step = x_next - y
                if _f(x_next) <= f_y + float(grad @ step) + 0.5 * inv_step * float((step * step).sum()) + 1e-15:
                    break
                inv_step *= 2.0
                if inv_step > 1e14:
                    break
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            if float(((x_next - x) ** 2).sum()) < 1e-18:      # converged
                x = x_next
                break
            x = x_next; t = t_next

        # Validity / failure check: a finite split summing to 1 that BEATS the trivial
        # return-weighted fallback. If not, flag the cell infeasible (surfaced downstream via
        # base._is_feasible → CellSolution.feasible) and fall back — a failed cell is never
        # reported as healthy (parity with Softmax's infeasible path).
        if np.all(np.isfinite(x)) and abs(float(x.sum()) - 1.0) < 1e-6 and _f(x) <= _f(x_start) + 1e-9:
            p._ref_infeasible = False       # type: ignore[attr-defined]
        else:
            self._t("STAGE B4  [WARNING] portfolio CVaR solve did not beat the return-weighted "
                    "split; flagged INFEASIBLE, using the fallback.")
            self._note_fail = getattr(self, "_note_fail", 0) + 1
            p._ref_infeasible = True        # type: ignore[attr-defined]
            x = x_start
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, x))
                + ("" if not getattr(p, "_ref_infeasible", False) else "  (SOLVE FAILED → return-weighted fallback)"))
        return x
