"""
The common contract every split engine speaks.

Every engine takes a CellProblem (one RPGT x Currency x Bank cell) and returns
a CellSolution (a vector of gateway shares that sums to 1). Because the input
and output shapes are identical across engines, the UI can swap engines from a
dropdown without anything downstream noticing.

WHY it is shaped this way: think of the engines as interchangeable "recipes" that
all take the same ingredients (a cell's gateways, their success/risk rates) and
all produce the same kind of dish (a set of shares summing to 100%). The rest of
the app is the kitchen — it doesn't care which recipe was used, only that the dish
has the expected shape. This file defines that shared ingredient/dish contract and
the utility steps every recipe reuses (bounds, projections, the reference split).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constraints import HardConstraints, SoftConstraints

__build__ = "2026-07-22-bounds-cache+qp-projection+feasible-guard"


@dataclass
class CellProblem:
    """One routing decision: how to split a cell's volume across gateways.

    A "cell" is a single RPGT x Currency x Bank bucket of traffic. Everything the
    engines need to decide that bucket's split lives here; the arrays are all
    aligned to `gateways` (index i describes the same gateway everywhere).
    """

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
        """Number of gateways in this cell (length of every aligned array)."""
        return len(self.gateways)


@dataclass
class CellSolution:
    """The optimiser's answer for one cell: the chosen split plus its headline stats."""

    shares: np.ndarray                  # fraction per gateway, sums to 1
    expected_success_rate: float
    expected_risk_rate: float
    feasible: bool                      # did it satisfy all hard constraints?
    note: str = ""


class BaseEngine:
    """Interface + shared helpers every engine reuses. Subclasses implement `_solve`.

    The shared helpers here are the "common kitchen tools": working out each
    gateway's allowed min/max share, projecting a rough split back onto the set of
    valid splits, and building the conversion-only "reference" split that the
    slider then moves away from. Subclasses only need to supply their own `_solve`.
    """

    key: str = "base"
    label: str = "Base"
    description: str = ""

    def __init__(self, weight: float, hard: HardConstraints,
                 soft: SoftConstraints, **params):
        # `weight` is the risk<->conversion slider in [0, 1]:
        #   1 = all conversion (ignore risk), 0 = all risk-aversion.
        self.w = float(np.clip(weight, 0.0, 1.0))
        self.hard = hard
        self.soft = soft
        self.params = params
        # When set to a list (via solve_traced), engines append human-readable
        # stage-by-stage debug lines here. None = tracing off (zero overhead).
        self._trace: list[str] | None = None

    def _t(self, msg: str) -> None:
        """Record one debug/trace line (a no-op unless tracing is switched on).

        Like a flight recorder: it costs nothing when off, and when on it captures
        exactly what the engine did so the UI's trace panel can replay it.
        """
        if self._trace is not None:
            self._trace.append(msg)

    def solve_traced(self, p: "CellProblem") -> tuple["CellSolution", list[str]]:
        """Solve one cell AND return the stage-by-stage trace for it.

        Used by the UI's gateway-trace debug panel so you can see exactly what
        the engine did to a single cell (reference split, floor, QP result).
        """
        self._trace = []
        solution = self.solve(p)
        trace_lines = self._trace
        self._trace = None
        return solution, trace_lines

    # -- helpers shared by every engine -------------------------------------
    def _bounds(self, p: CellProblem) -> tuple[np.ndarray, np.ndarray]:
        """Per-gateway (lower, upper) share bounds from the hard constraints.

        WHY memoised: the bounds depend only on the hard constraints + gateway list,
        NOT on the slider, so the same cell object flowing through every slider
        position of a sweep would otherwise recompute them 2-3x per solve. We cache
        them on the cell keyed by the hard-constraint fingerprint and hand back fresh
        COPIES, so callers (e.g. softmax's floor layer) can safely mutate their copy.
        """
        hard_key = (round(float(self.hard.max_gateway_share), 9),
                    frozenset(self.hard.banned_gateways), frozenset(self.hard.forced_gateways))
        cached = getattr(p, "_bounds_cache", None)
        if cached is not None and cached[0] == hard_key:
            return cached[1][0].copy(), cached[1][1].copy()

        gateway_count = p.n()
        lower = np.zeros(gateway_count)
        upper = np.full(gateway_count, self.hard.max_gateway_share)
        for i, gateway in enumerate(p.gateways):
            if gateway in self.hard.banned_gateways:
                upper[i] = 0.0
            if gateway in self.hard.forced_gateways:
                lower[i] = min(0.01, upper[i])

        # If the per-gateway caps make sum(upper) < 1 the cell can't be filled to
        # 100% — the problem is infeasible, so relax the caps just enough to reach 1.
        if upper.sum() < 1.0:
            upper = np.minimum(1.0, upper + (1.0 - upper.sum()) / max(1, (upper > 0).sum()))
        # Forced-gateway lower bounds must stay JOINTLY feasible (sum <= 1) so ANY caller
        # using `lower` directly (not just softmax._solve, which also rescales) gets a
        # feasible box.
        if lower.sum() > 1.0:
            lower = lower * (1.0 / lower.sum())

        try:
            p._bounds_cache = (hard_key, (lower.copy(), upper.copy()))  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return lower, upper

    @staticmethod
    def _project_box_simplex(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Euclidean projection of ``v`` onto ``{x : sum(x)=1, lo<=x<=hi}``.

        In plain terms: `v` is a rough set of shares that may not add up to 100% or may
        break the per-gateway caps. This returns the CLOSEST valid split to `v`.

        HOW (analogy): imagine one master "pressure" dial, λ. Turning λ up subtracts the
        same amount from every gateway before clipping to its [lo, hi] cap, so the total
        shrinks; turning λ down grows the total. Because the total only ever moves one way
        as λ moves, we can binary-search λ until the shares add up to exactly 1 — like
        turning a single tap until a jug fills to the marked line.
        """
        v = np.asarray(v, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
        # Edge case: if even every gateway at its max can't reach 100%, the best we can
        # do is put everyone at their cap.
        if hi.sum() <= 1.0 + 1e-12:
            return hi.copy()

        # Bracket λ so that at dual_lo the total is >1 and at dual_hi it is <1.
        dual_lo = float((v - hi).min()) - 1.0
        dual_hi = float((v - lo).max()) + 1.0
        for _ in range(80):
            dual = 0.5 * (dual_lo + dual_hi)
            if np.clip(v - dual, lo, hi).sum() > 1.0:
                dual_lo = dual          # total still too big → need more pressure
            else:
                dual_hi = dual          # total too small → ease off
            if dual_hi - dual_lo < 1e-13:      # converged (exact to ~1e-13)
                break
        return np.clip(v - 0.5 * (dual_lo + dual_hi), lo, hi)

    def _project_qp(self, ref: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                    risk: np.ndarray, cap: float) -> np.ndarray:
        """Closest valid split to ``ref`` whose portfolio risk meets a ceiling.

        Solves: min ||x-ref||^2  s.t.  sum(x)=1, lo<=x<=hi, risk·x <= cap.

        HOW (analogy): start from the reference split. If its blended risk is already
        under the ceiling, we're done. Otherwise apply a "risk tax" μ that nudges volume
        away from high-risk gateways (via the box-simplex projection of `ref - μ·risk`).
        More tax → lower portfolio risk. We binary-search the SMALLEST tax that just
        brings risk onto the ceiling — like turning a dimmer down only until a warning
        light goes off, no further. Being purpose-built and convex, it can't "fail" the
        way a general solver (SLSQP) sometimes does.
        """
        ref = np.asarray(ref, float); risk = np.asarray(risk, float)

        # No tax needed if the plain projection already meets the cap.
        projected = self._project_box_simplex(ref, lo, hi)
        if float(risk @ projected) <= cap + 1e-12:
            return projected

        # Grow the tax (doubling) until the cap is met, to bracket the search.
        mult_hi = 1.0
        for _ in range(200):
            projected = self._project_box_simplex(ref - mult_hi * risk, lo, hi)
            if float(risk @ projected) <= cap:
                break
            mult_hi *= 2.0

        # Binary-search the tax μ so that risk·x lands exactly on the cap.
        mult_lo = 0.0
        for _ in range(80):
            mult = 0.5 * (mult_lo + mult_hi)
            if float(risk @ self._project_box_simplex(ref - mult * risk, lo, hi)) > cap:
                mult_lo = mult          # still over the cap → tax harder
            else:
                mult_hi = mult          # under the cap → ease the tax
            if mult_hi - mult_lo < 1e-13 * max(1.0, mult_hi):   # converged
                break
        return self._project_box_simplex(ref - mult_hi * risk, lo, hi)

    def _score(self, p: CellProblem) -> np.ndarray:
        """Per-gateway linear score: reward conversion, penalise risk.

        DEPRECATED: the old linear conversion-vs-risk score. Retained only for
        the dormant engines (entropy/thompson/portfolio/genetic). The redesigned
        softmax engine no longer uses this; it builds a reference split from
        conversion alone (`_reference_split`) and moves off it under the slider.
        """
        return self.w * p.success_rates - (1.0 - self.w) * p.risk_rates

    def _ref_cache_key(self, p: CellProblem):
        """Fingerprint of everything the reference split depends on EXCEPT the risk dial.

        The reference split is the same at every slider position, so without caching a
        sweep would rebuild it once per position. We cache it on the cell object keyed by
        this fingerprint. Engine-SPECIFIC bits come from `_ref_param_key`, so each engine
        captures exactly what ITS reference depends on — no stale hits, no needless misses.
        """
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
        silently returns a stale reference.
        """
        temperature = getattr(p, "temperature", None)
        if temperature is None:
            temperature = self.params.get("temperature", 0.05)
        return (
            round(float(temperature or 0.05), 9),
            round(float(self.params.get("ref_risk_aversion", 0.0) or 0.0), 9),
            round(float(self.params.get("explore_cap_total", 0.10) or 0.0), 9),
            round(float(self.params.get("explore_cap_each", 0.01) or 0.0), 9),
        )

    def _reference_split(self, p: CellProblem) -> np.ndarray:
        """Cached wrapper around `_reference_split_impl`.

        Returns a COPY so callers can't mutate the cached array. Bit-identical to
        calling the implementation directly — it just avoids recomputing the same
        reference once per slider position. Tracing bypasses the cache so the
        gateway-trace panel still shows the full derivation.
        """
        if self._trace is not None:
            return self._reference_split_impl(p)
        key = self._ref_cache_key(p)
        cached = getattr(p, "_ref_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1].copy()
        reference = self._reference_split_impl(p)
        try:
            p._ref_cache = (key, reference.copy())  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        return reference

    def _reference_split_impl(self, p: CellProblem) -> np.ndarray:
        """The slider=100 reference split: conversion only, no risk logic.

        Softmax over per-gateway success rates at the engine temperature, then
        floor every eligible gateway at the exploration floor and renormalise so
        nothing goes dark. Deliberately ignores the VAMP cap, max-gateway-share
        and MID constraints - those only switch on as the slider moves down.
        """
        gateway_count = p.n()
        _, upper = self._bounds(p)
        eligible = upper > 0.0
        # Guard: nothing eligible → spread evenly and bail (nothing else to decide).
        if not eligible.any():
            return np.full(gateway_count, 1.0 / gateway_count)

        # Per-cell temperature (confidence-scaled) wins over the global dial.
        temperature = getattr(p, "temperature", None)
        if temperature is None:
            temperature = self.params.get("temperature", 0.05)
        temperature = max(float(temperature), 1e-4)
        # ANALOGY — temperature is a "decisiveness thermostat". The dial acts as a
        # 100x multiplier (dial 0.15 → sharpness 15). Each gateway's weight is
        # e^(score * sharpness) and its share is its weight over the total. Turn it up
        # → traffic piles onto the best converter (winner-takes-most); turn it down →
        # traffic spreads evenly (hedge across gateways).
        sharpness = temperature * 100.0
        self._t(f"STAGE B1  reference: softmax over SUCCESS ONLY, "
                f"dial={temperature:g} -> multiplier k={sharpness:g}; weight=e^(score*k)")
        for gateway, success_rate, is_eligible in zip(p.gateways, p.success_rates, eligible):
            self._t(f"           score[{gateway}]={success_rate:.4f} -> score*k={success_rate * sharpness:.3f}"
                    + ("" if is_eligible else "  (ineligible)"))

        # Constraint-aware reference (opt-in): discount each gateway's score by γ×VAMP rate
        # so the reference leans away from high-risk gateways even at slider=100, starting the
        # whole frontier closer to compliant. γ=0 (default) -> pure success-rate reference, i.e.
        # exactly the previous behaviour. γ is in success-rate units per unit VAMP rate.
        risk_aversion = float(self.params.get("ref_risk_aversion", 0.0) or 0.0)
        if risk_aversion > 0.0:
            risk_rates = np.asarray(getattr(p, "risk_rates", np.zeros(gateway_count)), dtype=float)
            adjusted_scores = p.success_rates - risk_aversion * risk_rates
            self._t(f"STAGE B1a reference risk-aversion γ={risk_aversion:g}: score = success − γ·risk")
        else:
            adjusted_scores = p.success_rates

        # Ineligible gateways get -inf so their softmax weight is exactly 0.
        scores = np.where(eligible, adjusted_scores, -np.inf)
        scaled_scores = scores * sharpness
        finite_scores = scaled_scores[np.isfinite(scaled_scores)]
        if finite_scores.size == 0:
            # Every eligible gateway has a non-finite score (e.g. all-NaN success rates):
            # degrade to a uniform split over eligibles rather than crashing on nanmax([]).
            self._t("STAGE B2  all eligible scores non-finite -> uniform over eligibles")
            weights = eligible.astype(float)
            weights = weights / weights.sum()
        else:
            # Subtract the max before exp — a standard softmax trick that keeps the numbers
            # small (avoids overflow) and cancels out in the final ratio, so it changes nothing.
            scaled_scores = scaled_scores - finite_scores.max()
            weights = np.where(np.isfinite(scaled_scores), np.exp(scaled_scores), 0.0)
            weight_total = weights.sum()
            weights = weights / weight_total if weight_total > 0 else eligible / eligible.sum()
        self._t("STAGE B2  softmax shares (pre-floor): "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, weights)))

        # Exploration floor: guarantee every eligible gateway a minimum share (so we never
        # lose sight of how a gateway performs), capped so the floors stay feasible, then
        # renormalise.
        floor = float(getattr(self.soft, "exploration_floor", 0.0) or 0.0)
        eligible_count = int(eligible.sum())
        if floor > 0.0 and eligible_count > 0:
            floor = min(floor, 1.0 / eligible_count)
            weights = np.where(eligible, np.maximum(weights, floor), 0.0)
            weights = weights / weights.sum()
            self._t(f"STAGE B3  applied exploration floor={floor:g} to {eligible_count} eligible, renormalised")

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
        is_explore = np.asarray(getattr(p, "is_explore", np.zeros(gateway_count, bool)), dtype=bool)
        cap_total = float(self.params.get("explore_cap_total", 0.10) or 0.0)
        cap_each = float(self.params.get("explore_cap_each", 0.01) or 0.0)
        explore_mask = is_explore & eligible
        proven_mask = eligible & ~is_explore
        if explore_mask.any() and proven_mask.any() and (cap_total > 0.0 or cap_each > 0.0):
            capped = weights.copy()
            if cap_each > 0.0:
                capped[explore_mask] = np.minimum(capped[explore_mask], cap_each)
            explore_before = float(capped[explore_mask].sum())
            if cap_total > 0.0 and explore_before > cap_total:
                capped[explore_mask] *= cap_total / explore_before
            explore_share = float(capped[explore_mask].sum())
            # Proven gateways absorb the remaining (1 - explore share), keeping their relative mix.
            proven_before = float(weights[proven_mask].sum())
            if proven_before > 0:
                capped[proven_mask] = weights[proven_mask] / proven_before * (1.0 - explore_share)
            capped[~eligible] = 0.0
            capped_total = capped.sum()
            weights = capped / capped_total if capped_total > 0 else weights
            self._t(f"STAGE B3b explore cap: {int(explore_mask.sum())} untested gw capped to "
                    f"≤{cap_each:g} each / ≤{cap_total:g} total (share={explore_share:.3f}); "
                    f"proven hold {1.0 - explore_share:.3f}")
        self._t("STAGE B4  REFERENCE split: "
                + ", ".join(f"{gateway}={share:.3f}" for gateway, share in zip(p.gateways, weights)))
        return weights

    def _project_to_vamp(self, p: CellProblem, shares: np.ndarray) -> np.ndarray:
        """Nudge a split to the closest one that meets the VAMP (risk) cap.

        Solves min ||x - shares||^2 s.t. sum=1, bounds, risk·x <= cap. This lets
        heuristic engines (softmax, thompson) respect the hard risk cap without
        changing their character much; it's a no-op for engines that already enforce
        the cap in-solver.

        The cap is skipped entirely when the slider is at 1.0 ("no regard for risk"),
        so a high-conversion, high-risk gateway isn't zeroed by the projection when the
        user has explicitly asked to ignore risk.
        """
        cap = self.hard.vamp_cap
        # Guard: no cap, or already compliant → return the split untouched.
        if cap is None or float(shares @ p.risk_rates) <= cap + 1e-12:
            return shares
        # Guard: pure-conversion slider → the user opted out of the cap.
        if self.w >= 1.0 - 1e-9:
            return shares
        lower, upper = self._bounds(p)
        return self._project_qp(shares, lower, upper, np.asarray(p.risk_rates, float), float(cap))

    def _finalise(self, p: CellProblem, shares: np.ndarray,
                  note: str = "") -> CellSolution:
        """Clean up a raw share vector into a valid CellSolution.

        Clip negatives, renormalise to sum 1, apply the VAMP projection, renormalise
        again, then attach the headline success/risk numbers and a feasibility flag.
        (Renormalising twice — before and after the risk projection — because the
        projection can shift the total slightly.)
        """
        shares = np.clip(shares, 0, None)
        total = shares.sum()
        shares = shares / total if total > 0 else np.full(p.n(), 1.0 / p.n())
        shares = self._project_to_vamp(p, shares)
        shares = np.clip(shares, 0, None)
        total = shares.sum()
        shares = shares / total if total > 0 else np.full(p.n(), 1.0 / p.n())
        expected_success = float(shares @ p.success_rates)
        expected_risk = float(shares @ p.risk_rates)
        feasible = self._is_feasible(p, shares)
        return CellSolution(shares, expected_success, expected_risk, feasible, note)

    def _is_feasible(self, p: CellProblem, shares: np.ndarray) -> bool:
        """True only if `shares` satisfies EVERY hard constraint for this cell."""
        # A FAILED reference solve (e.g. Portfolio's SLSQP falling back to a return-weighted
        # split) taints the whole cell — flag it infeasible so a solver failure can never
        # masquerade as a healthy split downstream.
        if getattr(p, "_ref_infeasible", False):
            return False
        # A cell with only one eligible gateway must send it 100% — the max-share cap is
        # physically unsatisfiable there, so it doesn't count as a violation.
        _, upper = self._bounds(p)
        eligible_count = int((upper > 0).sum())
        if eligible_count > 1 and (shares > self.hard.max_gateway_share + 1e-6).any():
            return False
        if self.hard.vamp_cap is not None:
            if float(shares @ p.risk_rates) > self.hard.vamp_cap + 1e-9:
                return False
        for i, gateway in enumerate(p.gateways):
            if gateway in self.hard.banned_gateways and shares[i] > 1e-9:
                return False
        return True

    # -- public API ---------------------------------------------------------
    def solve(self, p: CellProblem) -> CellSolution:
        """Public entry point: return the chosen split for one cell.

        Handles the two trivial cells here (0 gateways → nothing to do; 1 gateway →
        it must take 100%) and delegates everything else to the engine's `_solve`.
        """
        if p.n() == 0:
            return CellSolution(np.array([]), 0.0, 0.0, False, "no gateways")
        if p.n() == 1:
            return self._finalise(p, np.array([1.0]), "single gateway")
        return self._solve(p)

    def _solve(self, p: CellProblem) -> CellSolution:  # pragma: no cover
        """Engine-specific split logic. Every concrete engine overrides this."""
        raise NotImplementedError
