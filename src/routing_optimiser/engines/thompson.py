"""Thompson-sampling / bandit engine (probability-of-being-best reference).

Reuses the softmax engine's slider + compliance machinery wholesale and changes
only the REFERENCE split (the slider=100, conversion-only allocation).

Where softmax spreads a cell's volume by an exponential of each gateway's POINT
success rate, Thompson models each gateway's success rate as a Beta posterior and
allocates by each gateway's PROBABILITY OF BEING THE BEST — so a gateway we've
barely tested (wide posterior) keeps a meaningful share (exploration), and a
gateway that quietly became the best can still be discovered.

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

import numpy as np

from .base import CellProblem
from .softmax import SoftmaxEngine

__build__ = "2026-07-22-thompson-gauss-legendre-logcdf"

_GRID = 2000        # integration grid points for the analytic probability-of-best
_TAIL = 1e-4        # quantile range covered by the grid (per gateway)
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))   # trapezoid on numpy>=2, trapz on 3.8's numpy


class ThompsonEngine(SoftmaxEngine):
    key = "thompson"
    label = "Thompson / bandit"
    description = ("Models each gateway's success rate as a probability "
                   "distribution (Empirical-Bayes) and allocates the reference "
                   "split by each gateway's chance of being the best, computed "
                   "analytically. Under-sampled gateways keep a share so you "
                   "never go blind. No dials — prior and precision are automatic.")

    def _ref_param_key(self, p: CellProblem):
        # Thompson's reference depends on its Beta prior and — via the γ tilt — temperature and
        # ref_risk_aversion. The auto-explore caps don't apply (it allocates from its own
        # posterior), so they're dropped; the prior params are ADDED so a prior change can't
        # return a stale reference.
        temp = getattr(p, "temperature", None)
        if temp is None:
            temp = self.params.get("temperature", 0.05)
        return (round(float(self.params.get("prior_alpha", 1.0)), 9),
                round(float(self.params.get("prior_beta", 1.0)), 9),
                round(float(self.params.get("pooled_pseudo", 8.0)), 9),
                round(float(self.params.get("ref_risk_aversion", 0.0) or 0.0), 9),
                round(float(temp or 0.05), 9))

    def _beta_params(self, p: CellProblem):
        """Beta(alpha, beta) per gateway — a SELF-CONTAINED posterior from the
        (time-decayed) observed successes/attempts with a weak uniform prior (+1/+1).

        Deliberately does NOT use the pipeline's Empirical-Bayes / kappa shrinkage:
        Thompson's whole mechanism is the posterior WIDTH (narrow where data is rich →
        exploit; wide where thin → explore), and layering the pipeline's kappa on top
        (which can jump to 100k) collapses that width and flattens prob-of-best. The
        Beta prior is Thompson's own regulariser. Time-decay is kept (recency matters).
        Gateways with no per-cell evidence fall back to a weak Beta at the pooled rate."""
        n = p.n()
        succ = (np.asarray(p.obs_success, float) if p.obs_success is not None
                else np.zeros(n))
        att = (np.asarray(p.obs_attempts, float) if p.obs_attempts is not None
               else np.zeros(n))
        a0 = float(self.params.get("prior_alpha", 1.0))   # weak uniform prior
        b0 = float(self.params.get("prior_beta", 1.0))
        _n0 = float(self.params.get("pooled_pseudo", 8.0))  # pseudo-count for no-data gateways

        have = att > 0
        # Observed gateways: Beta from their own decayed counts (+ uniform prior).
        alpha = np.maximum(succ, 0.0) + a0
        beta = np.maximum(att - succ, 0.0) + b0
        # No-data gateways: weak Beta centred on the pooled RAW rate (prior_rate =
        # pooled successes/attempts over the scope) — NOT the κ-shrunk success_rate,
        # which would re-import the very shrinkage Thompson deliberately avoids. Only
        # used where a gateway has genuinely no per-cell evidence.
        if not have.all():
            _pr_src = p.prior_rate if p.prior_rate is not None else p.success_rates
            pr = np.clip(np.asarray(_pr_src, float), 1e-6, 1.0 - 1e-6)
            alpha = np.where(have, alpha, pr * _n0 + a0)
            beta = np.where(have, beta, (1.0 - pr) * _n0 + b0)
        return alpha, beta

    def _reference_split_impl(self, p: CellProblem) -> np.ndarray:
        """slider=100 reference: analytic probability-of-being-best over SUCCESS.
        Same contract as ``SoftmaxEngine._reference_split_impl``. Wrapped by the
        base-class reference cache (computed once per cell, reused across dials)."""
        n = p.n()
        _, hi = self._bounds(p)
        eligible = hi > 0.0
        if not eligible.any():
            return np.full(n, 1.0 / n)
        n_elig = int(eligible.sum())

        alpha, beta = self._beta_params(p)
        ei = np.where(eligible)[0]

        if n_elig == 1:
            w = eligible.astype(float)
        else:
            from scipy.stats import beta as _beta
            a_e, b_e = alpha[ei], beta[ei]
            # P(g best) = ∫ f_g(x)·Π_{j≠g} F_j(x) dx. Integrate EACH gateway's term with
            # Gauss–Legendre nodes on ITS OWN Beta support [ppf(TAIL), ppf(1−TAIL)] — f_g is
            # ~0 outside that support, so a modest node count matches the old 2000-pt grid to
            # ~1e-9 with far fewer special-function evals. The leave-one-out product Π_{j≠g}F_j
            # is formed from LOG-CDFs (sum then exp) so it can't underflow with many gateways.
            m = max(8, int(self.params.get("thompson_nodes", 64)))
            _nodes, _wts = np.polynomial.legendre.leggauss(m)      # nodes/weights on [-1, 1]
            probs = np.zeros(n_elig)
            for gi in range(n_elig):
                _lo = float(_beta.ppf(_TAIL, a_e[gi], b_e[gi]))
                _hi = float(_beta.ppf(1.0 - _TAIL, a_e[gi], b_e[gi]))
                if not np.isfinite(_lo):
                    _lo = 0.0
                if not np.isfinite(_hi) or _hi <= _lo:
                    _hi = min(1.0, _lo + 1e-3)
                _x = 0.5 * (_hi - _lo) * _nodes + 0.5 * (_hi + _lo)   # map [-1,1] -> [_lo,_hi]
                _scale = 0.5 * (_hi - _lo)
                f_g = _beta.pdf(_x, a_e[gi], b_e[gi])                 # (m,)
                logF = _beta.logcdf(_x[None, :], a_e[:, None], b_e[:, None])   # (k, m)
                log_prod_others = logF.sum(axis=0) - logF[gi]        # (m,) = Σ_{j≠g} log F_j
                integrand = f_g * np.exp(log_prod_others)
                probs[gi] = float((_wts * integrand).sum() * _scale)
            s = probs.sum()
            probs = probs / s if s > 0 else np.full(n_elig, 1.0 / n_elig)
            w = np.zeros(n)
            w[ei] = probs

        self._t(f"STAGE B1  reference: THOMPSON prob-of-best (analytic, EB prior), "
                f"{n_elig} eligible")
        for g, a, b, e in zip(p.gateways, alpha, beta, eligible):
            if e:
                self._t(f"           Beta[{g}]=({a:.1f},{b:.1f}) mean={a/(a+b):.4f}")
        self._t("STAGE B2  prob-of-best shares (pre-floor): "
                + ", ".join(f"{g}={x:.3f}" for g, x in zip(p.gateways, w)))

        # Constraint-aware reference (opt-in): tilt the prob-of-best allocation toward
        # low-VAMP gateways by ×e^(-γ·k·risk), matching softmax's γ scale (k = temp×100).
        # γ=0 (default) -> unchanged prob-of-best reference.
        _gamma = float(self.params.get("ref_risk_aversion", 0.0) or 0.0)
        if _gamma > 0.0 and n_elig > 1:
            _temp = getattr(p, "temperature", None) or self.params.get("temperature", 0.05)
            _k = max(float(_temp), 1e-4) * 100.0
            _riskv = np.asarray(getattr(p, "risk_rates", np.zeros(n)), dtype=float)
            w = w * np.where(eligible, np.exp(-_gamma * _k * _riskv), 0.0)
            _s = w.sum()
            w = w / _s if _s > 0 else (eligible.astype(float) / max(n_elig, 1))
            self._t(f"STAGE B2a reference risk-aversion γ={_gamma:g}: tilt ×e^(-γ·{_k:g}·risk)")

        # Exploration floor: guarantee every eligible gateway a minimum share.
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        if floor > 0.0 and n_elig > 0:
            floor = min(floor, 1.0 / n_elig)
            w = np.where(eligible, np.maximum(w, floor), 0.0)
            w = w / w.sum()
            self._t(f"STAGE B3  applied exploration floor={floor:g} to {n_elig} eligible, renormalised")
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{g}={x:.3f}" for g, x in zip(p.gateways, w)))
        return w
