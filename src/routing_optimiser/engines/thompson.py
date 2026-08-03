"""Thompson-sampling / bandit engine (probability-of-being-best reference).

Reuses the softmax engine's slider + compliance machinery wholesale and changes
only the REFERENCE split (the slider=100, conversion-only allocation).

Where softmax spreads a cell's volume by an exponential of each gateway's POINT
success rate, Thompson models each gateway's success rate as a Beta posterior and
allocates by each gateway's PROBABILITY OF BEING THE BEST — so a gateway we've
barely tested (wide posterior) keeps a meaningful share (exploration), and a
gateway that quietly became the best can still be discovered.

ANALOGY: rather than always backing whichever gateway looks best on today's numbers,
Thompson bets on each gateway in proportion to its CHANCE of actually being the best.
A Beta posterior is our belief about a gateway's true rate: narrow = "we're sure about
this one" (exploit), wide = "we've barely tested this one" (explore). So thin gateways
keep some traffic and a quietly-improving gateway can still be found.

Two things make this the "auto" version (no dials):

  * EMPIRICAL-BAYES prior. The Beta posterior is built from the SAME shrinkage
    the rest of the pipeline uses: prior Beta(κ·prior_rate, κ·(1−prior_rate))
    plus the observed (time-decayed) successes/attempts. Its mean is exactly the
    shrunk success_rate softmax uses, so thin cells borrow strength automatically
    — no "prior strength" knob. (Falls back to a weak flat prior if κ is absent.)

  * ANALYTIC probability-of-best. Instead of Monte-Carlo sampling (which jitters
    run-to-run and needs a "draws" knob), the win probabilities are computed by
    a deterministic 1-D integral,  P(g best) = ∫ f_g(x)·Π_{j≠g} F_j(x) dx,  over
    a grid that adapts to the gateways' plausible range. No seed, no jitter.

Risk is NOT in the reference; it enters via the shared compliance layer, exactly
as for softmax. Deterministic.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .base import CellProblem
from .softmax import SoftmaxEngine

__build__ = "2026-08-03-thompson-leggauss-hoist+batched-quadrature"

_GRID = 2000        # integration grid points for the analytic probability-of-best
_TAIL = 1e-4        # quantile range covered by the grid (per gateway)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))   # trapezoid on numpy>=2, trapz on 3.8's numpy


# [FN-419]
@lru_cache(maxsize=32)
def _leggauss_cached(m: int):
    """Gauss–Legendre nodes/weights on [-1, 1] for m points. `leggauss` is a pure
    deterministic function of m (it solves a fixed eigenproblem), so caching returns
    byte-identical arrays and hoists the cost out of the per-cell reference loop.
    The arrays are treated as read-only by callers (only used in fresh expressions)."""
    return np.polynomial.legendre.leggauss(int(m))


class ThompsonEngine(SoftmaxEngine):
    key = "thompson"
    label = "Thompson / bandit"
    description = ("Models each gateway's success rate as a probability "
                   "distribution (Empirical-Bayes) and allocates the reference "
                   "split by each gateway's chance of being the best, computed "
                   "analytically. Under-sampled gateways keep a share so you "
                   "never go blind. No dials — prior and precision are automatic.")

    # [FN-420]
    def _ref_param_key(self, p: CellProblem):
        # Thompson's reference depends on its Beta prior and — via the γ tilt — temperature and
        # ref_risk_aversion. The auto-explore caps don't apply (it allocates from its own
        # posterior), so they're dropped; the prior params are ADDED so a prior change can't
        # return a stale reference.
        temperature = getattr(p, "temperature", None)
        if temperature is None:
            temperature = self.params.get("temperature", 0.05)
        return (round(float(self.params.get("prior_alpha", 1.0)), 9),
                round(float(self.params.get("prior_beta", 1.0)), 9),
                round(float(self.params.get("pooled_pseudo", 8.0)), 9),
                round(float(self.params.get("ref_risk_aversion", 0.0) or 0.0), 9),
                round(float(temperature or 0.05), 9))

    # [FN-421]
    def _beta_params(self, p: CellProblem):
        """Beta(alpha, beta) per gateway — a SELF-CONTAINED posterior from the
        (time-decayed) observed successes/attempts with a weak uniform prior (+1/+1).

        WHY it deliberately does NOT use the pipeline's Empirical-Bayes / kappa shrinkage:
        Thompson's whole mechanism is the posterior WIDTH (narrow where data is rich →
        exploit; wide where thin → explore), and layering the pipeline's kappa on top
        (which can jump to 100k) collapses that width and flattens prob-of-best. The
        Beta prior is Thompson's own regulariser. Time-decay is kept (recency matters).
        Gateways with no per-cell evidence fall back to a weak Beta at the pooled rate.
        """
        gateway_count = p.n()
        successes = (np.asarray(p.obs_success, float) if p.obs_success is not None
                     else np.zeros(gateway_count))
        attempts = (np.asarray(p.obs_attempts, float) if p.obs_attempts is not None
                    else np.zeros(gateway_count))
        prior_alpha = float(self.params.get("prior_alpha", 1.0))   # weak uniform prior
        prior_beta = float(self.params.get("prior_beta", 1.0))
        pooled_pseudo_count = float(self.params.get("pooled_pseudo", 8.0))  # pseudo-count for no-data gateways

        has_data = attempts > 0
        # Observed gateways: Beta from their own decayed counts (+ uniform prior). alpha counts
        # "wins" (successes), beta counts "losses" (attempts − successes).
        alpha = np.maximum(successes, 0.0) + prior_alpha
        beta = np.maximum(attempts - successes, 0.0) + prior_beta
        # No-data gateways: weak Beta centred on the pooled RAW rate (prior_rate =
        # pooled successes/attempts over the scope) — NOT the κ-shrunk success_rate,
        # which would re-import the very shrinkage Thompson deliberately avoids. Only
        # used where a gateway has genuinely no per-cell evidence.
        if not has_data.all():
            prior_rate_source = p.prior_rate if p.prior_rate is not None else p.success_rates
            pooled_rate = np.clip(np.asarray(prior_rate_source, float), 1e-6, 1.0 - 1e-6)
            alpha = np.where(has_data, alpha, pooled_rate * pooled_pseudo_count + prior_alpha)
            beta = np.where(has_data, beta, (1.0 - pooled_rate) * pooled_pseudo_count + prior_beta)
        return alpha, beta

    # [FN-422]
    def _reference_split_impl(self, p: CellProblem) -> np.ndarray:
        """slider=100 reference: analytic probability-of-being-best over SUCCESS.
        Same contract as ``SoftmaxEngine._reference_split_impl``. Wrapped by the
        base-class reference cache (computed once per cell, reused across dials)."""
        gateway_count = p.n()
        _, upper = self._bounds(p)
        eligible = upper > 0.0
        # Guard: nothing eligible → spread evenly and bail.
        if not eligible.any():
            return np.full(gateway_count, 1.0 / gateway_count)
        eligible_count = int(eligible.sum())

        alpha, beta = self._beta_params(p)
        eligible_idx = np.where(eligible)[0]

        if eligible_count == 1:
            weights = eligible.astype(float)
        else:
            from scipy.stats import beta as _beta
            a_e, b_e = alpha[eligible_idx], beta[eligible_idx]
            # P(g best) = ∫ f_g(x)·Π_{j≠g} F_j(x) dx. Integrate EACH gateway's term with
            # Gauss–Legendre nodes on ITS OWN Beta support [ppf(TAIL), ppf(1−TAIL)] — f_g is
            # ~0 outside that support, so a modest node count matches the old 2000-pt grid to
            # ~1e-9 with far fewer special-function evals. The leave-one-out product Π_{j≠g}F_j
            # is formed from LOG-CDFs (sum then exp) so it can't underflow with many gateways.
            m = max(8, int(self.params.get("thompson_nodes", 64)))
            _nodes, _wts = _leggauss_cached(m)                     # cached nodes/weights on [-1, 1]
            # BATCHED quadrature (byte-identical to the old per-gateway loop). Every step below
            # is the same elementwise scipy ufunc / same-order small reduction as before, just
            # evaluated for all eligible gateways at once instead of one Python iteration each:
            #   - per-gateway support [ppf(TAIL), ppf(1−TAIL)] with the SAME finite/degenerate
            #     clamps (compute _lo first, then _hi from the clamped _lo);
            #   - node grid X[g] and _scale[g] via the same 0.5·(hi−lo)·nodes + 0.5·(hi+lo);
            #   - f_g = pdf at own nodes; logF_all[g,j] = logcdf(F_j at g's nodes);
            #   - Σ_{j≠g} logF via (sum over j) − (own logF), same j-order (0..k−1);
            #   - probs[g] = (weights·integrand).sum()·scale, same m-order sum.
            _lo_v = np.asarray(_beta.ppf(_TAIL, a_e, b_e), dtype=float)          # (k,)
            _hi_v = np.asarray(_beta.ppf(1.0 - _TAIL, a_e, b_e), dtype=float)    # (k,)
            _lo_v = np.where(np.isfinite(_lo_v), _lo_v, 0.0)
            _bad_hi = (~np.isfinite(_hi_v)) | (_hi_v <= _lo_v)
            _hi_v = np.where(_bad_hi, np.minimum(1.0, _lo_v + 1e-3), _hi_v)
            X = 0.5 * (_hi_v[:, None] - _lo_v[:, None]) * _nodes[None, :] \
                + 0.5 * (_hi_v[:, None] + _lo_v[:, None])                        # (k, m)
            _scale_v = 0.5 * (_hi_v - _lo_v)                                     # (k,)
            f_g_all = _beta.pdf(X, a_e[:, None], b_e[:, None])                   # (k, m)
            logF_all = _beta.logcdf(X[:, None, :], a_e[None, :, None],
                                    b_e[None, :, None])                          # (k_g, k_j, m)
            _sum_j = logF_all.sum(axis=1)                                        # (k, m) = Σ_j logF_j
            _logF_self = _beta.logcdf(X, a_e[:, None], b_e[:, None])             # (k, m) = own logF
            log_prod_others = _sum_j - _logF_self                                # (k, m) = Σ_{j≠g}
            integrand = f_g_all * np.exp(log_prod_others)                        # (k, m)
            probs = (_wts[None, :] * integrand).sum(axis=1) * _scale_v           # (k,)
            prob_total = probs.sum()
            probs = probs / prob_total if prob_total > 0 else np.full(eligible_count, 1.0 / eligible_count)
            weights = np.zeros(gateway_count)
            weights[eligible_idx] = probs

        self._t(f"STAGE B1  reference: THOMPSON prob-of-best (analytic, EB prior), "
                f"{eligible_count} eligible")
        for gateway, alpha_g, beta_g, is_eligible in zip(p.gateways, alpha, beta, eligible):
            if is_eligible:
                self._t(f"           Beta[{gateway}]=({alpha_g:.1f},{beta_g:.1f}) mean={alpha_g/(alpha_g+beta_g):.4f}")
        self._t("STAGE B2  prob-of-best shares (pre-floor): "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, weights)))

        # Constraint-aware reference (opt-in): tilt the prob-of-best allocation toward
        # low-VAMP gateways by ×e^(-γ·k·risk), matching softmax's γ scale (k = temp×100).
        # γ=0 (default) -> unchanged prob-of-best reference.
        risk_aversion = float(self.params.get("ref_risk_aversion", 0.0) or 0.0)
        if risk_aversion > 0.0 and eligible_count > 1:
            temperature = getattr(p, "temperature", None) or self.params.get("temperature", 0.05)
            sharpness = max(float(temperature), 1e-4) * 100.0
            risk_rates = np.asarray(getattr(p, "risk_rates", np.zeros(gateway_count)), dtype=float)
            weights = weights * np.where(eligible, np.exp(-risk_aversion * sharpness * risk_rates), 0.0)
            total = weights.sum()
            weights = weights / total if total > 0 else (eligible.astype(float) / max(eligible_count, 1))
            self._t(f"STAGE B2a reference risk-aversion γ={risk_aversion:g}: tilt ×e^(-γ·{sharpness:g}·risk)")

        # Exploration floor: guarantee every eligible gateway a minimum share.
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        if floor > 0.0 and eligible_count > 0:
            floor = min(floor, 1.0 / eligible_count)
            weights = np.where(eligible, np.maximum(weights, floor), 0.0)
            weights = weights / weights.sum()
            self._t(f"STAGE B3  applied exploration floor={floor:g} to {eligible_count} eligible, renormalised")
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, weights)))
        return weights
