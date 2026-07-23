"""
The common contract every split engine speaks.

Every engine takes a CellProblem (one RPGT x Currency x Bank cell) and returns
a CellSolution (a vector of gateway shares that sums to 1). Because the input
and output shapes are identical across engines, the UI can swap engines from a
dropdown without anything downstream noticing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constraints import HardConstraints, SoftConstraints

__build__ = "2026-07-22-bounds-cache+qp-projection+feasible-guard"


@dataclass
class CellProblem:
    """One routing decision: how to split a cell's volume across gateways."""

    rpgt: str
    currency: str
    bank: str
    gateways: list[str]                 # eligible gateway/MID names
    success_rates: np.ndarray           # expected auth rate per gateway, 0-1
    risk_rates: np.ndarray              # expected chargeback/VAMP rate per gateway, 0-1
    volume: float                       # forecast attempts for this cell
    baseline_shares: np.ndarray         # current ("pre") split, sums to 1
    # Optional evidence for Bayesian engines: successes / attempts observed
    obs_success: np.ndarray | None = None
    obs_attempts: np.ndarray | None = None
    # Optional Empirical-Bayes prior for Bayesian engines: the pooled prior rate
    # the shrinkage uses, and its strength kappa (pseudo-attempts), per gateway.
    # Thompson builds its Beta prior from these so thin cells borrow strength.
    prior_rate: np.ndarray | None = None
    kappa: np.ndarray | None = None
    # Optional sample size of the RISK rate per gateway (the transaction/sales count
    # the VAMP rate is measured over). Portfolio uses this for the risk-rate standard
    # error instead of auth attempts (a different dataset). None → falls back to attempts.
    risk_n: np.ndarray | None = None
    # Optional per-cell softmax temperature (confidence-scaled). When None the
    # engine falls back to its global `temperature` param.
    temperature: float | None = None

    def n(self) -> int:
        return len(self.gateways)


@dataclass
class CellSolution:
    """The optimiser's answer for one cell."""

    shares: np.ndarray                  # fraction per gateway, sums to 1
    expected_success_rate: float
    expected_risk_rate: float
    feasible: bool                      # did it satisfy all hard constraints?
    note: str = ""


class BaseEngine:
    """Interface + shared helpers. Subclasses implement `_solve`."""

    key: str = "base"
    label: str = "Base"
    description: str = ""

    def __init__(self, weight: float, hard: HardConstraints,
                 soft: SoftConstraints, **params):
        # weight in [0,1]; 1 = all conversion, 0 = all risk-aversion.
        self.w = float(np.clip(weight, 0.0, 1.0))
        self.hard = hard
        self.soft = soft
        self.params = params
        # When set to a list (via solve_traced), engines append human-readable
        # stage-by-stage debug lines here. None = tracing off (zero overhead).
        self._trace: list[str] | None = None

    def _t(self, msg: str) -> None:
        """Record one debug/trace line (no-op unless tracing is on)."""
        if self._trace is not None:
            self._trace.append(msg)

    def solve_traced(self, p: "CellProblem") -> tuple["CellSolution", list[str]]:
        """Solve one cell AND return the stage-by-stage trace for it.

        Used by the UI's gateway-trace debug panel so you can see exactly what
        the engine did to a single cell (reference split, floor, QP result).
        """
        self._trace = []
        sol = self.solve(p)
        lines = self._trace
        self._trace = None
        return sol, lines

    # -- helpers shared by every engine -------------------------------------
    def _bounds(self, p: CellProblem) -> tuple[np.ndarray, np.ndarray]:
        """Per-gateway (lower, upper) share bounds from the hard constraints.

        Slider-INVARIANT (depends only on the hard constraints + gateway list), so it is
        MEMOISED on the CellProblem — the same instance flows through every slider position of
        a sweep, and _bounds is otherwise recomputed 2-3x per solve (reference, solve, project,
        finalise, feasibility). Returns fresh COPIES so callers (e.g. softmax's floor layer)
        can mutate freely.
        """
        _hk = (round(float(self.hard.max_gateway_share), 9),
               frozenset(self.hard.banned_gateways), frozenset(self.hard.forced_gateways))
        cached = getattr(p, "_bounds_cache", None)
        if cached is not None and cached[0] == _hk:
            return cached[1][0].copy(), cached[1][1].copy()
        n = p.n()
        lo = np.zeros(n)
        hi = np.full(n, self.hard.max_gateway_share)
        for i, g in enumerate(p.gateways):
            if g in self.hard.banned_gateways:
                hi[i] = 0.0
            if g in self.hard.forced_gateways:
                lo[i] = min(0.01, hi[i])
        # If caps make sum(upper) < 1 the problem is infeasible; relax caps.
        if hi.sum() < 1.0:
            hi = np.minimum(1.0, hi + (1.0 - hi.sum()) / max(1, (hi > 0).sum()))
        # Forced-gateway lower bounds must stay JOINTLY feasible (sum <= 1) so ANY caller using
        # `lo` directly (not just softmax._solve, which also rescales) gets a feasible box.
        if lo.sum() > 1.0:
            lo = lo * (1.0 / lo.sum())
        try:
            p._bounds_cache = (_hk, (lo.copy(), hi.copy()))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return lo, hi

    @staticmethod
    def _project_box_simplex(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Euclidean projection of ``v`` onto ``{x : sum(x)=1, lo<=x<=hi}``.

        Exact via bisection on the single dual λ: x_i = clip(v_i − λ, lo_i, hi_i); the sum is
        monotone non-increasing in λ, so bisection converges to the λ giving sum(x)=1."""
        v = np.asarray(v, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        if hi.sum() <= 1.0 + 1e-12:        # box can only just (or not) reach the simplex
            return hi.copy()
        lam_lo = float((v - hi).min()) - 1.0
        lam_hi = float((v - lo).max()) + 1.0
        for _ in range(80):
            lam = 0.5 * (lam_lo + lam_hi)
            if np.clip(v - lam, lo, hi).sum() > 1.0:
                lam_lo = lam
            else:
                lam_hi = lam
            if lam_hi - lam_lo < 1e-13:      # converged (exact to ~1e-13)
                break
        return np.clip(v - 0.5 * (lam_lo + lam_hi), lo, hi)

    def _project_qp(self, ref: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    risk: np.ndarray, cap: float) -> np.ndarray:
        """min ||x−ref||^2  s.t.  sum(x)=1, lo<=x<=hi, risk·x <= cap.

        Purpose-built convex projection (replaces a general SLSQP call): outer bisection on the
        risk multiplier μ>=0 (risk·x is monotone non-increasing in μ), inner box-simplex
        projection for the equality + bounds. The problem is strictly convex → unique optimum,
        so this returns the same point SLSQP would, without a solver that can 'fail'."""
        ref = np.asarray(ref, float); risk = np.asarray(risk, float)
        x = self._project_box_simplex(ref, lo, hi)
        if float(risk @ x) <= cap + 1e-12:
            return x
        mu_hi = 1.0
        for _ in range(200):                      # grow until the cap is met
            x = self._project_box_simplex(ref - mu_hi * risk, lo, hi)
            if float(risk @ x) <= cap:
                break
            mu_hi *= 2.0
        mu_lo = 0.0
        for _ in range(80):                       # bisect μ to hit risk·x == cap
            mu = 0.5 * (mu_lo + mu_hi)
            if float(risk @ self._project_box_simplex(ref - mu * risk, lo, hi)) > cap:
                mu_lo = mu
            else:
                mu_hi = mu
            if mu_hi - mu_lo < 1e-13 * max(1.0, mu_hi):   # converged
                break
        return self._project_box_simplex(ref - mu_hi * risk, lo, hi)

    def _score(self, p: CellProblem) -> np.ndarray:
        """Per-gateway linear score: reward conversion, penalise risk.

        DEPRECATED: the old linear conversion-vs-risk score. Retained only for
        the dormant engines (entropy/thompson/portfolio/genetic). The redesigned
        softmax engine no longer uses this; it builds a reference split from
        conversion alone (`_reference_split`) and moves off it under the slider.
        """
        return self.w * p.success_rates - (1.0 - self.w) * p.risk_rates

    def _ref_cache_key(self, p: CellProblem):
        """Everything the reference split depends on (but NOT the risk dial `w`).

        The reference is invariant across slider positions, so a sweep would otherwise
        recompute it once per position. Cached on the cell object keyed on this tuple. The
        engine-SPECIFIC reference params come from `_ref_param_key`, so each engine captures
        exactly what ITS reference depends on — no stale hits, no needless misses."""
        return (
            self.key,
            round(float(getattr(self.soft, "exploration_floor", 0.0) or 0.0), 9),
            round(float(self.hard.max_gateway_share), 9),
        ) + tuple(self._ref_param_key(p))

    def _ref_param_key(self, p: CellProblem):
        """Engine-specific reference parameters (softmax/base default).

        The base softmax reference depends on the temperature (per-cell or global), the
        constraint-aware γ, and the auto-explore share caps. Subclasses OVERRIDE this to
        declare their own reference params (Thompson's Beta prior, Portfolio's prior_count)
        and drop any that don't affect their reference — so a temperature change no longer
        needlessly invalidates the Thompson/Portfolio cache, and a prior change no longer
        silently returns a stale reference."""
        temp = getattr(p, "temperature", None)
        if temp is None:
            temp = self.params.get("temperature", 0.05)
        return (
            round(float(temp or 0.05), 9),
            round(float(self.params.get("ref_risk_aversion", 0.0) or 0.0), 9),
            round(float(self.params.get("explore_cap_total", 0.10) or 0.0), 9),
            round(float(self.params.get("explore_cap_each", 0.01) or 0.0), 9),
        )

    def _reference_split(self, p: CellProblem) -> np.ndarray:
        """Cached wrapper around `_reference_split_impl`.

        Returns a COPY so callers can't mutate the cached array. Bit-identical to
        calling the implementation directly — it just avoids recomputing the same
        reference once per slider position. Tracing bypasses the cache so the
        gateway-trace panel still shows the full derivation."""
        if self._trace is not None:
            return self._reference_split_impl(p)
        key = self._ref_cache_key(p)
        cached = getattr(p, "_ref_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1].copy()
        w = self._reference_split_impl(p)
        try:
            p._ref_cache = (key, w.copy())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return w

    def _reference_split_impl(self, p: CellProblem) -> np.ndarray:
        """The slider=100 reference split: conversion only, no risk logic.

        Softmax over per-gateway success rates at the engine temperature, then
        floor every eligible gateway at the exploration floor and renormalise so
        nothing goes dark. Deliberately ignores the VAMP cap, max-gateway-share
        and MID constraints - those only switch on as the slider moves down.
        """
        n = p.n()
        _, hi = self._bounds(p)
        eligible = hi > 0.0
        if not eligible.any():
            return np.full(n, 1.0 / n)

        # Per-cell temperature (confidence-scaled) wins over the global dial.
        temp = getattr(p, "temperature", None)
        if temp is None:
            temp = self.params.get("temperature", 0.05)
        temp = max(float(temp), 1e-4)
        # The temperature dial acts as a MULTIPLIER of 100x its shown value:
        # dial 0.15 -> k = 15. weighting_g = e^(engine_score_g * k); the share
        # is each weighting over the sum of weightings. Higher dial = sharper
        # (more traffic to the best converters), lower = flatter.
        k = temp * 100.0
        self._t(f"STAGE B1  reference: softmax over SUCCESS ONLY, "
                f"dial={temp:g} -> multiplier k={k:g}; weight=e^(score*k)")
        for g, sr, e in zip(p.gateways, p.success_rates, eligible):
            self._t(f"           score[{g}]={sr:.4f} -> score*k={sr*k:.3f}" + ("" if e else "  (ineligible)"))
        # Constraint-aware reference (opt-in): discount each gateway's score by γ×VAMP rate
        # so the reference leans away from high-risk gateways even at slider=100, starting the
        # whole frontier closer to compliant. γ=0 (default) -> pure success-rate reference, i.e.
        # exactly the previous behaviour. γ is in success-rate units per unit VAMP rate.
        _gamma = float(self.params.get("ref_risk_aversion", 0.0) or 0.0)
        if _gamma > 0.0:
            _riskv = np.asarray(getattr(p, "risk_rates", np.zeros(n)), dtype=float)
            _adj = p.success_rates - _gamma * _riskv
            self._t(f"STAGE B1a reference risk-aversion γ={_gamma:g}: score = success − γ·risk")
        else:
            _adj = p.success_rates
        score = np.where(eligible, _adj, -np.inf)
        z = score * k
        _fin = z[np.isfinite(z)]
        if _fin.size == 0:
            # Every eligible gateway has a non-finite score (e.g. all-NaN success rates):
            # degrade to a uniform split over eligibles rather than crashing on nanmax([]).
            self._t("STAGE B2  all eligible scores non-finite -> uniform over eligibles")
            w = eligible.astype(float)
            w = w / w.sum()
        else:
            z = z - _fin.max()   # numerical stability only (cancels in the ratio)
            w = np.where(np.isfinite(z), np.exp(z), 0.0)
            s = w.sum()
            w = w / s if s > 0 else eligible / eligible.sum()
        self._t("STAGE B2  softmax shares (pre-floor): "
                + ", ".join(f"{g}={x:.3f}" for g, x in zip(p.gateways, w)))

        # Exploration floor: guarantee every eligible gateway a minimum share,
        # capped so the floors themselves stay feasible, then renormalise.
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        n_elig = int(eligible.sum())
        if floor > 0.0 and n_elig > 0:
            floor = min(floor, 1.0 / n_elig)
            w = np.where(eligible, np.maximum(w, floor), 0.0)
            w = w / w.sum()
            self._t(f"STAGE B3  applied exploration floor={floor:g} to {n_elig} eligible, renormalised")

        # Auto-explore share cap (non-Thompson engines): capable-but-untested
        # gateways collectively get at most `explore_cap_total` of the cell and at
        # most `explore_cap_each` individually, so a flood of unproven gateways can't
        # dilute the proven ones. The freed share flows to the proven (non-explore)
        # gateways. Only applied when there IS at least one proven gateway to hold the
        # volume; otherwise the explore gateways are all the cell has and must take it.
        # This is a REFERENCE cap only — the downstream compliance/VAMP layer may push
        # an explore gateway above the cap if that's the only way to meet a hard
        # constraint (the override the user asked for). Thompson never calls this method
        # (it allocates from its own Beta posterior), so it is unaffected by design.
        is_explore = np.asarray(getattr(p, "is_explore", np.zeros(n, bool)), dtype=bool)
        cap_tot = float(self.params.get("explore_cap_total", 0.10) or 0.0)
        cap_each = float(self.params.get("explore_cap_each", 0.01) or 0.0)
        expl = is_explore & eligible
        proven = eligible & ~is_explore
        if expl.any() and proven.any() and (cap_tot > 0.0 or cap_each > 0.0):
            we = w.copy()
            if cap_each > 0.0:
                we[expl] = np.minimum(we[expl], cap_each)
            te = float(we[expl].sum())
            if cap_tot > 0.0 and te > cap_tot:
                we[expl] *= cap_tot / te
            e_share = float(we[expl].sum())
            # proven absorb the remaining (1 - explore share), preserving their relative mix
            ps = float(w[proven].sum())
            if ps > 0:
                we[proven] = w[proven] / ps * (1.0 - e_share)
            we[~eligible] = 0.0
            s = we.sum()
            w = we / s if s > 0 else w
            self._t(f"STAGE B3b explore cap: {int(expl.sum())} untested gw capped to "
                    f"≤{cap_each:g} each / ≤{cap_tot:g} total (share={e_share:.3f}); "
                    f"proven hold {1.0 - e_share:.3f}")
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{g}={x:.3f}" for g, x in zip(p.gateways, w)))
        return w

    def _project_to_vamp(self, p: CellProblem, shares: np.ndarray) -> np.ndarray:
        """Nudge a split to the closest one that meets the VAMP cap.

        Solves min ||x - shares||^2 s.t. sum=1, bounds, risk.x <= cap. This
        makes heuristic engines (softmax, thompson) respect the hard risk cap
        without changing their character much; it's a no-op for engines that
        already enforce the cap in-solver (LP, entropy, portfolio).

        The VAMP cap is skipped entirely when the slider is at 1.0 ("no regard
        for risk"), so a high-conversion, high-risk gateway isn't zeroed by the
        projection when the user has explicitly asked to ignore risk.
        """
        cap = self.hard.vamp_cap
        if cap is None or float(shares @ p.risk_rates) <= cap + 1e-12:
            return shares
        if self.w >= 1.0 - 1e-9:  # pure conversion: user opted out of the cap
            return shares
        # Exact convex projection (min ||x-shares||^2 s.t. sum=1, bounds, risk·x<=cap) — the
        # purpose-built dual-bisection solver instead of a general SLSQP call that can fail.
        lo, hi = self._bounds(p)
        return self._project_qp(shares, lo, hi, np.asarray(p.risk_rates, float), float(cap))

    def _finalise(self, p: CellProblem, shares: np.ndarray,
                  note: str = "") -> CellSolution:
        shares = np.clip(shares, 0, None)
        s = shares.sum()
        shares = shares / s if s > 0 else np.full(p.n(), 1.0 / p.n())
        shares = self._project_to_vamp(p, shares)
        shares = np.clip(shares, 0, None)
        s = shares.sum()
        shares = shares / s if s > 0 else np.full(p.n(), 1.0 / p.n())
        exp_succ = float(shares @ p.success_rates)
        exp_risk = float(shares @ p.risk_rates)
        feasible = self._is_feasible(p, shares)
        return CellSolution(shares, exp_succ, exp_risk, feasible, note)

    def _is_feasible(self, p: CellProblem, shares: np.ndarray) -> bool:
        # A FAILED reference solve (e.g. Portfolio's SLSQP falling back to a return-weighted
        # split) taints the whole cell — flag it infeasible so a solver failure can never
        # masquerade as a healthy split downstream.
        if getattr(p, "_ref_infeasible", False):
            return False
        # A cell with only one eligible gateway must send it 100% - the cap is
        # physically unsatisfiable, so it doesn't count as a violation.
        _, hi = self._bounds(p)
        n_eligible = int((hi > 0).sum())
        if n_eligible > 1 and (shares > self.hard.max_gateway_share + 1e-6).any():
            return False
        if self.hard.vamp_cap is not None:
            if float(shares @ p.risk_rates) > self.hard.vamp_cap + 1e-9:
                return False
        for i, g in enumerate(p.gateways):
            if g in self.hard.banned_gateways and shares[i] > 1e-9:
                return False
        return True

    # -- public API ---------------------------------------------------------
    def solve(self, p: CellProblem) -> CellSolution:
        if p.n() == 0:
            return CellSolution(np.array([]), 0.0, 0.0, False, "no gateways")
        if p.n() == 1:
            return self._finalise(p, np.array([1.0]), "single gateway")
        return self._solve(p)

    def _solve(self, p: CellProblem) -> CellSolution:  # pragma: no cover
        raise NotImplementedError