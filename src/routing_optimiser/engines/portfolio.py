"""Mean-CVaR (portfolio) engine — downside-risk reference, auto-calibrated.

Reuses the softmax engine's slider + compliance machinery wholesale and changes
only the REFERENCE split (the slider=100 allocation).

Treats gateways like investments: expected conversion is the "return", and the
*downside* uncertainty of each gateway's risk (VAMP) is what we pay to avoid.
Unlike a symmetric variance penalty (which punishes a gateway's risk coming in
*better* than expected just as much as *worse*), this prices only the BAD tail —
how much a gateway's VAMP rate could plausibly spike above its expected level.

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
        n = p.n()
        lo, hi = self._bounds(p)
        eligible = hi > 0.0
        if not eligible.any():
            return np.full(n, 1.0 / n)

        prior_n = float(self.params.get("prior_count", 30.0))
        r = np.clip(np.asarray(p.risk_rates, float), 0.0, 1.0)
        # Sample size for the RISK-rate standard error = the VAMP rate's OWN denominator
        # (the transaction/sales count, risk_n) when available — not auth attempts, which
        # are a different dataset. Falls back to attempts, then a small prior. (C1)
        _fallback = (np.asarray(p.obs_attempts, float) if p.obs_attempts is not None
                     else np.full(n, prior_n))
        if p.risk_n is not None:
            _rn = np.asarray(p.risk_n, float)
            n_g = np.where(_rn > 0, _rn, np.where(_fallback > 0, _fallback, prior_n))
        else:
            n_g = np.where(_fallback > 0, _fallback, prior_n)
        n_g = np.maximum(n_g, 1.0)

        # Per-gateway DOWNSIDE (upper-tail) risk deviation: how much worse than expected
        # the VAMP rate could plausibly be. VAMP rates are rare (~0.6%) and right-skewed,
        # so a normal-tail factor (2.06·σ) UNDER-prices the bad tail. Instead use the
        # actual upper-95% quantile of the risk-rate's Beta posterior minus its mean —
        # skew-aware, and heavier exactly for the low-rate gateways that can spike. (C2)
        _a = r * n_g + 0.5                    # Jeffreys prior on the risk rate (n = risk_n)
        _b = np.maximum(1.0 - r, 0.0) * n_g + 0.5
        _q95 = _betadist.ppf(_Q_TAIL, _a, _b)
        downside = np.where(eligible & np.isfinite(_q95), np.maximum(_q95 - r, 0.0), 0.0)
        ret = np.where(eligible, np.asarray(p.success_rates, float), 0.0)

        # Exploration floor as a hard lower bound (nothing eligible goes dark).
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        n_elig = int(eligible.sum())
        if floor > 0.0 and n_elig > 0:
            floor = min(floor, 1.0 / n_elig)
            lo = np.where(eligible, np.maximum(lo, floor), lo)
        if lo.sum() > 1.0:
            lo = lo * (1.0 / lo.sum())

        # AUTO risk-aversion (no dial): scale the penalty so it is comparable to
        # the return magnitude in THIS cell. Then a gateway with an average
        # downside pays ~_AVERSION of its return, above-average (volatile / thin)
        # gateways get trimmed, and a clearly-best low-downside gateway keeps its
        # share — consistently across cells whatever the absolute risk scale.
        mean_ret = float(ret[eligible].mean()) if n_elig else 0.0
        _dn_e = downside[eligible]
        _dn_e = _dn_e[np.isfinite(_dn_e)]          # guard against Beta-ppf NaNs
        mean_dn = float(_dn_e.mean()) if _dn_e.size else 0.0
        # Auto risk-aversion. The penalty is self-normalising (γ·downside ≈ AVERSION·return),
        # but FLOOR the denominator by a small fraction of the return so a near-zero-dispersion
        # cell can't blow γ up (or divide by ~0) — it just concentrates on the best converter,
        # which is correct when there's no risk spread to diversify against. Also cap γ so a
        # pathological cell can't force a degenerate uniform split. (C1/γ-degeneracy)
        _dn_floor = max(mean_dn, 1e-4 * max(mean_ret, 1e-9))
        gamma = float(np.clip(_AVERSION * mean_ret / _dn_floor, 0.0, 5000.0)) if mean_ret > 0 else 0.0

        self._t(f"STAGE B1  reference: MEAN-CVaR (downside), auto γ={gamma:.1f} "
                f"(_AVERSION={_AVERSION}, mean_return={mean_ret:.4f}, mean_downside={mean_dn:.5f})")
        for g, rr, dd, e in zip(p.gateways, ret, downside, eligible):
            self._t(f"           {g}: success={rr:.4f}, downside(CVaR)={dd:.5f}"
                    + ("" if e else "  (ineligible)"))

        eps = 1e-12
        d2 = downside * downside

        def _f(x):
            return float(-(x @ ret) + gamma * np.sqrt(float((d2 * x * x).sum()) + eps))

        def _grad(x):
            s = np.sqrt(float((d2 * x * x).sum()) + eps)
            return -ret + gamma * (d2 * x) / s

        def _proj(v):
            return self._project_box_simplex(v, lo, hi)

        # Return-weighted feasible start / fallback (a revenue-lean split is a sensible,
        # clearly-non-uniform default — a uniform split would masquerade as 'diversified'). (C4)
        _fb = np.where(eligible, np.maximum(ret, 1e-9), 0.0)
        x0 = _proj(_fb) if _fb.sum() > 0 else _proj(np.where(eligible, 1.0, 0.0).astype(float))

        # Accelerated projected gradient (FISTA) with backtracking — a purpose-built convex
        # solve for the mean-CVaR SOCP, replacing a general SLSQP call that can 'fail'. The
        # objective is convex so this converges to the unique optimum; fully deterministic.
        x = x0.copy(); y = x0.copy(); t = 1.0; _L = 1.0
        for _ in range(400):
            g = _grad(y)
            fy = _f(y)
            xn = y
            for _bt in range(80):                 # backtracking: shrink the step (grow L)
                xn = _proj(y - g / _L)
                _dx = xn - y
                if _f(xn) <= fy + float(g @ _dx) + 0.5 * _L * float((_dx * _dx).sum()) + 1e-15:
                    break
                _L *= 2.0
                if _L > 1e14:
                    break
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            y = xn + ((t - 1.0) / tn) * (xn - x)
            if float(((xn - x) ** 2).sum()) < 1e-18:      # converged
                x = xn
                break
            x = xn; t = tn

        # Validity / failure check: a finite split summing to 1 that BEATS the trivial
        # return-weighted fallback. If not, flag the cell infeasible (surfaced downstream via
        # base._is_feasible → CellSolution.feasible) and fall back — a failed cell is never
        # reported as healthy (parity with Softmax's infeasible path).
        if np.all(np.isfinite(x)) and abs(float(x.sum()) - 1.0) < 1e-6 and _f(x) <= _f(x0) + 1e-9:
            p._ref_infeasible = False       # type: ignore[attr-defined]
        else:
            self._t("STAGE B4  [WARNING] portfolio CVaR solve did not beat the return-weighted "
                    "split; flagged INFEASIBLE, using the fallback.")
            self._note_fail = getattr(self, "_note_fail", 0) + 1
            p._ref_infeasible = True        # type: ignore[attr-defined]
            x = x0
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{g}={xx:.3f}" for g, xx in zip(p.gateways, x))
                + ("" if not getattr(p, "_ref_infeasible", False) else "  (SOLVE FAILED → return-weighted fallback)"))
        return x
