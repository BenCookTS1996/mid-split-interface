"""EXACT projector-defined band solver — the "fragile hand-derivation".

This is the exact counterpart to the heuristic `seed_search.band_greedy_shares` seed. Where the
heuristic nudges shares with a multiplicative band-correction and hopes the breach falls, this module
optimises the SAME objective the GA actually scores — the TRUE `PopulationBandProjector` band values —
using a closed-form analytic Jacobian of that projector, so a proper NLP solver can be used.

WHAT IS EXACT AND WHAT IS NOT (read this before trusting it)
------------------------------------------------------------
EXACT:
  * The objective/constraints are the real projector band values `ExactBandPenalty.projector.project_pop`
    — no volume-ratio proxy, no surrogate. `values()` reproduces `project_pop` to float64 rounding.
  * The Jacobian `d(band)/d(shares)` is derived in closed form (the hand-derivation below) and validated
    against the live projector by finite differences — it is NOT a numerical approximation at run time.

APPROXIMATE / LOCAL (the honest ceiling):
  * The projector's movable-fraction gate `act[c] = 1{psum_c > 0}` is a STEP. Its derivative is 0
    almost everywhere, so we FIX `act` from the current iterate and differentiate the smooth piece. The
    solver re-evaluates the true (gated) breach every step and re-linearises, so `act` can change between
    steps — but within one linearisation the active set is fixed. A candidate that ZEROES a whole cell is
    a different active piece; exploring those is combinatorial (a MILP) and out of scope.
  * TXN is affine in the per-cell shares, but VAMP is a linear-FRACTIONAL function (`vshare = vpr/vpsum`),
    so the feasible region is NONCONVEX. The successive-LP solver therefore converges to a KKT/LOCAL
    optimum of the true breach, not a certified global one. A breach of 0 is still a genuine feasibility
    CERTIFICATE (a compliant split provably exists); a positive floor is a strong — not proof — signal.
  * Only gateways that actually feed a band are optimised; every other gateway is held at the reference
    split. This is EXACT for the band objective (those gateways provably don't move any band value) and
    keeps the solve small.

THE DERIVATION (per band b, holding the active mask fixed)
----------------------------------------------------------
Let `pr[r]` be the proposed share on reduced projector row r (0 if excl/emask), `c = gcode[r]` its cell,
`psum[c] = Σ_{r∈c} pr[r]`, `vpsum[c] = Σ_{r∈c} pr[r]·vcpos[r]`. The projector's per-row normalisations are

    pshare[r] = pr[r] / psum[c]                 (TXN uses this)
    vshare[r] = pr[r]·vcpos[r] / vpsum[c]        (VAMP uses this)

TXN(b) = Σ_{r∈T_b}  ctot[r]·( base[r]·(1−mv[r]) + moved_tot[c]·pshare[r] )     — affine in pshare
VAMP(b)= const_b + Σ_{j∈P_b}  pool[j]·vshare[o_j]                              — fractional in pr

with `mv`, `moved_tot`, `const_b`, `base`, `ctot` all static given the fixed active mask. Differentiating
the two normalisations (quotient rule, only same-cell columns couple):

    ∂pshare[r]/∂pr[q] = (δ_rq − pshare[r]) / psum[c]            for q∈c
    ∂vshare[r]/∂pr[q] = vcpos[r]·(δ_rq − vshare[r]·vcpos[q]) / vpsum[c]   for q∈c

gives the closed forms implemented in `_jac_pr` (A_b/C_b are per-cell reductions, G_b a per-origin sum):

    ∂TXN(b)/∂pr[q]  = act[q]/psum[c_q] · ( 1[q∈T_b]·ctot[q]·moved_tot[c_q] − A_b[c_q] )
    ∂VAMP(b)/∂pr[q] = vcpos[q]/vpsum[c_q] · ( G_b[q] − C_b[c_q] )                    (vpsum[c_q]>0)

Finally chain pr → prop_raw (a masked selection) → shares s (the `incidence` matmul), both linear.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os as _os
import time as _time
import numpy as np

# scipy is a HARD requirement (2026-08-19aa) — see band_scoring.py for the reasoning. NOTE this
# module needs BOTH scipy.optimize.linprog (the successive-LP solver) and scipy.sparse, so there
# was never a meaningful degraded mode: without them the exact projector and targeted-move stages
# cannot run at all. The two
#   `if not _HAVE_SCIPY: info["reason"] = "scipy unavailable"; return base_shares`
# early-returns that used to guard the solvers are deleted with the flag. They returned the seed
# UNCHANGED under its own name, so a missing dependency surfaced as "the projector achieved
# nothing" — indistinguishable from a genuinely unimprovable seed.
from scipy.optimize import linprog as _linprog
import scipy.sparse as _sparse

__build__ = "2026-08-15-exact-projector-band-solver-slp+sparse-lp+progress+global-linear-lp-seed+minimal-move-projection+colocation-report+held-movable-report+movable-provenance+reachable-minimum-no-floor+vamp-positive-sibling+selfcheck+seedgrad+vpsum+usable-recipient+degenerate-gradient-flag+breach-concentration+scoped-frozen-split+gradient-vpsum-regularisation+insearch-rpgt-breakdown+catchall-eps-floor+targeted-move-headroom+2026-08-19bd-raw-basis-claim-labelled+2026-08-19be-recipient-headroom-per-metric+2026-09-01-19go-delivery-faithful-seed-accept-tests"

# Gradient-only vpsum/psum floor used by the SEED SOLVERS (not the diagnostics, not the forward
# values). Share-scale denominators: real high-VAMP cells sit well above this, near-empty cells
# (the 1/vpsum blow-up) get their gradient capped at ~1/_VPS_EPS instead of ~1e18. Tunable.
_VPS_EPS = 1e-3


@dataclass
class SpecMap:
    """One `ExactBandPenalty` spec reduced to projector band columns + its ceil/floor."""
    label: str
    metric: str                 # "vamp" | "txn"
    cols: np.ndarray            # projector band_order columns summed for this spec
    ceil: Optional[float]
    floor: Optional[float]
    weight: float = 1.0


class ExactBandModel:
    """Closed-form value + analytic Jacobian of `PopulationBandProjector`, in share space.

    Construct from a live `ExactBandPenalty` (which owns the projector + specs) and the same
    `incidence` matrix the GA uses to roll per-gateway shares up to projector prop-keys. All heavy
    per-row statics are pulled straight off the projector, so this stays byte-consistent with what
    the search scores."""

    # [FN-380]
    def __init__(self, exact_bands, incidence, *, vps_eps=0.0):
        pj = exact_bands.projector
        self.pj = pj
        # GRADIENT-ONLY regularisation: floor the per-cell VAMP/txn denominators (vpsum/psum) at
        # `vps_eps` INSIDE the analytic Jacobian only. This tames the 1/vpsum blow-up on near-empty
        # cells (the 1e18 gradients that freeze the seed solvers) WITHOUT changing any forward/breach
        # value — the reported VAMP, `breach`, `spec_values`, and all diagnostics stay exact. Every
        # solver that uses this gradient re-validates its candidate on the true (unregularised) breach,
        # so this only affects search DIRECTION, never a scored number. 0 = off (raw gradient).
        self.vps_eps = float(vps_eps or 0.0)
        self.incidence = incidence                      # (K × N): prop_raw = incidence @ s
        # --- static per-reduced-row arrays (copied so we never mutate the projector) ---
        self.propidx = np.asarray(pj._propidx, np.int64)
        self.gcode = np.asarray(pj._gcode, np.int64)
        self.ngc = int(pj._ngc)
        self.base = np.asarray(pj._base, float)
        self.mv_static = np.asarray(pj._mv, float)
        self.ctot = np.asarray(pj._ctot, float)
        self.excl = np.asarray(pj._excl, bool)
        self.emask = np.asarray(pj._emask, bool)
        self.mask = self.excl | self.emask
        # 19dt - the same per-row proposal weight the projector applies. A seed scored by
        # different rules than the fitness is exactly the `[seed-basis] RAW vs DELIVERED
        # disagree` failure. Falls back to the old boolean if an older projector is loaded.
        self.pw = np.asarray(getattr(pj, "_pw", np.where(self.mask, 0.0, 1.0)), float)
        self.vcpos = np.asarray(pj._vcpos, float)
        # movable-fraction factors (mv = pr · fcp), kept separately for diagnostics
        self.pr = np.asarray(getattr(pj, "_pr", np.ones(self.gcode.shape[0])), float)
        self.fcp = np.asarray(getattr(pj, "_fcp", np.ones(self.gcode.shape[0])), float)
        self.nR = int(self.gcode.shape[0])
        self.K = int(pj._K)
        # aged (VAMP) rows
        self.pc_org = np.asarray(pj._pc_org, np.int64)
        self.pc_vc = np.asarray(pj._pc_vc, float)
        self.pc_pool = np.asarray(pj._pc_pool, float)
        self.pc_bandcol = np.asarray(pj._pc_bandcol, np.int64)
        # capped (TXN) rows
        self.t_rows = np.asarray(pj._t_rows, np.int64)
        self.t_bandcol = np.asarray(pj._t_bandcol, np.int64)
        self.B = int(pj._B)
        self.band_order = list(pj.band_order)           # [(midl_lower, per), ...]
        self._bpos = {b: k for k, b in enumerate(self.band_order)}
        # --- reduce ExactBandPenalty specs to projector columns ---
        self.specs = []
        for sp in exact_bands.specs:
            cols = [self._bpos.get((str(sp.midl).strip().lower(), int(m))) for m in sp.months]
            cols = np.array([c for c in cols if c is not None], np.int64)
            self.specs.append(SpecMap(
                label=str(sp.midl), metric=str(sp.metric),
                cols=cols,
                ceil=(None if sp.ceil is None else float(sp.ceil)),
                floor=(None if sp.floor is None else float(sp.floor)),
                weight=float(getattr(sp, "weight", 1.0) or 1.0)))

    # ------------------------------------------------------------------ forward value
    # [FN-381]
    def _forward_pr(self, prop_raw: np.ndarray):
        """(K,) prop_raw → (vamp[B], txn[B]) + the intermediates the Jacobian needs.
        Mirrors PopulationBandProjector.project_pop for a single candidate."""
        pr = prop_raw[self.propidx] * self.pw          # 19dt: weight, not mask
        psum_c = np.bincount(self.gcode, pr, minlength=self.ngc)
        psum = psum_c[self.gcode]
        act = psum > 0.0
        pshare = np.where(act, np.divide(pr, psum, out=np.zeros_like(pr), where=act), self.base)
        mv = np.where(act, self.mv_static, 0.0)
        vpr = pr * self.vcpos
        vpsum_c = np.bincount(self.gcode, vpr, minlength=self.ngc)
        vpsum = vpsum_c[self.gcode]
        vact = vpsum > 0.0
        vshare = np.divide(vpr, vpsum, out=np.zeros_like(vpr), where=vact)
        moved_tot = np.bincount(self.gcode, self.base * mv, minlength=self.ngc)[self.gcode]
        ptxn = self.ctot * (self.base * (1.0 - mv) + moved_tot * pshare)
        ptxn = np.where(self.excl, 0.0, ptxn)
        txn = np.zeros(self.B)
        if len(self.t_bandcol):
            np.add.at(txn, self.t_bandcol, ptxn[self.t_rows])
        o = self.pc_org
        ok = o >= 0
        oi = np.where(ok, o, 0)
        move_pc = np.where(ok, mv[oi], 0.0)
        psh_pc = np.where(ok, vshare[oi], 0.0)
        vp = self.pc_vc * (1.0 - move_pc) + self.pc_pool * psh_pc
        vamp = np.zeros(self.B)
        if len(self.pc_bandcol):
            np.add.at(vamp, self.pc_bandcol, vp)
        inter = dict(pr=pr, psum=psum, act=act, pshare=pshare, mv=mv,
                     moved_tot=moved_tot, vpsum=vpsum, vshare=vshare)
        return vamp, txn, inter

    # [FN-382]
    def band_values(self, prop_raw: np.ndarray):
        """(K,) prop_raw → (vamp[B], txn[B]); exact projector values (public, no intermediates)."""
        v, t, _ = self._forward_pr(np.asarray(prop_raw, float))
        return v, t

    # [FN-383]
    def spec_values(self, prop_raw: np.ndarray) -> np.ndarray:
        """Per-spec scalar value (metric summed over the spec's months)."""
        vamp, txn, _ = self._forward_pr(np.asarray(prop_raw, float))
        out = np.zeros(len(self.specs))
        for i, sp in enumerate(self.specs):
            if len(sp.cols) == 0:
                continue
            out[i] = float((txn if sp.metric == "txn" else vamp)[sp.cols].sum())
        return out

    # [FN-383b]
    def spec_decomposition(self, prop_raw: np.ndarray):
        """Per-spec (held, movable) split of the projected metric at this candidate.

        Every band value is  held + movable  where:
          * HELD    = the baseline / FCP2+ / pre-go-live cohort that does NOT respond to the routing
                      decision (the `(1 − mv)` share of each cohort, mv = pro_rata × fcp1_frac);
          * MOVABLE = the pool that IS redistributed by the candidate's per-cell share (responds to
                      routing).
        held + movable reproduces `spec_values` to float64. This is the quantity that answers "how much
        of a breached band can the optimiser actually move?": if HELD alone already exceeds the ceiling,
        no routing can clear it (structural); if HELD < ceiling, a compliant routing exists.

        Returns (held[S], movable[S]) aligned with self.specs."""
        vamp, txn, inter = self._forward_pr(np.asarray(prop_raw, float))
        mv = inter["mv"]; vshare = inter["vshare"]
        pshare = inter["pshare"]; moved_tot = inter["moved_tot"]
        held_v = np.zeros(self.B); mov_v = np.zeros(self.B)
        if len(self.pc_bandcol):
            o = self.pc_org; ok = o >= 0; oi = np.where(ok, o, 0)
            move_pc = np.where(ok, mv[oi], 0.0)
            psh_pc = np.where(ok, vshare[oi], 0.0)
            np.add.at(held_v, self.pc_bandcol, self.pc_vc * (1.0 - move_pc))
            np.add.at(mov_v, self.pc_bandcol, self.pc_pool * psh_pc)
        held_t = np.zeros(self.B); mov_t = np.zeros(self.B)
        if len(self.t_bandcol):
            r = self.t_rows
            keep = ~self.excl[r]
            hv = np.where(keep, self.ctot[r] * self.base[r] * (1.0 - mv[r]), 0.0)
            mvv = np.where(keep, self.ctot[r] * moved_tot[r] * pshare[r], 0.0)
            np.add.at(held_t, self.t_bandcol, hv)
            np.add.at(mov_t, self.t_bandcol, mvv)
        held = np.zeros(len(self.specs)); mov = np.zeros(len(self.specs))
        for i, sp in enumerate(self.specs):
            if len(sp.cols) == 0:
                continue
            hb = held_t if sp.metric == "txn" else held_v
            mb = mov_t if sp.metric == "txn" else mov_v
            held[i] = float(hb[sp.cols].sum()); mov[i] = float(mb[sp.cols].sum())
        return held, mov

    # [FN-383c]
    def spec_movable_provenance(self):
        """Per-spec volume-weighted mean pro_rata (pr), fcp1_frac (fcp) and mv = pr·fcp.

        Decomposes the movable fraction into its two factors so a low movable %% can be attributed:
        a small `pr` ⇒ go-live phasing; a small `fcp` ⇒ the first-attempt reroutable slice (fcp1 &
        attempt1) is small. VAMP specs are weighted by their aged VAMP (pc_vc) at the banded origin
        rows; TXN specs by the t0 cap volume (ctot·base). Candidate-independent (uses the static
        factors). Returns (mean_pr[S], mean_fcp[S], mean_mv[S]); NaN where a spec has no weight."""
        S = len(self.specs)
        mean_pr = np.full(S, np.nan); mean_fcp = np.full(S, np.nan); mean_mv = np.full(S, np.nan)
        # VAMP: aged rows weighted by pc_vc, factors taken at the origin t0 row
        wv = np.zeros(self.B); prv = np.zeros(self.B); fcv = np.zeros(self.B); mvv = np.zeros(self.B)
        if len(self.pc_bandcol):
            o = self.pc_org; ok = o >= 0; oi = np.where(ok, o, 0)
            w = self.pc_vc * ok
            np.add.at(wv, self.pc_bandcol, w)
            np.add.at(prv, self.pc_bandcol, w * np.where(ok, self.pr[oi], 0.0))
            np.add.at(fcv, self.pc_bandcol, w * np.where(ok, self.fcp[oi], 0.0))
            np.add.at(mvv, self.pc_bandcol, w * np.where(ok, self.mv_static[oi], 0.0))
        # TXN: t0 cap rows weighted by ctot·base
        wt = np.zeros(self.B); prt = np.zeros(self.B); fct = np.zeros(self.B); mvt = np.zeros(self.B)
        if len(self.t_bandcol):
            r = self.t_rows
            w = self.ctot[r] * self.base[r] * (~self.excl[r])
            np.add.at(wt, self.t_bandcol, w)
            np.add.at(prt, self.t_bandcol, w * self.pr[r])
            np.add.at(fct, self.t_bandcol, w * self.fcp[r])
            np.add.at(mvt, self.t_bandcol, w * self.mv_static[r])
        for i, sp in enumerate(self.specs):
            if len(sp.cols) == 0:
                continue
            if sp.metric == "txn":
                ws = wt[sp.cols].sum(); pw = prt; fw = fct; mw = mvt
            else:
                ws = wv[sp.cols].sum(); pw = prv; fw = fcv; mw = mvv
            if ws > 0:
                mean_pr[i] = pw[sp.cols].sum() / ws
                mean_fcp[i] = fw[sp.cols].sum() / ws
                mean_mv[i] = mw[sp.cols].sum() / ws
        return mean_pr, mean_fcp, mean_mv

    # ------------------------------------------------------------------ analytic Jacobian
    # [FN-384]
    def _jac_pr(self, inter) -> tuple:
        """Per-band-column Jacobian d(metric[b])/d(pr[q]) as dense (B, nR) arrays (vamp, txn)."""
        psum = inter["psum"]; act = inter["act"]
        pshare = inter["pshare"]; moved_tot = inter["moved_tot"]
        vpsum = inter["vpsum"]; vshare = inter["vshare"]
        gcode = self.gcode; ngc = self.ngc; nR = self.nR
        _eps = getattr(self, "vps_eps", 0.0)
        if _eps > 0:                                    # gradient-only regularisation (see __init__)
            inv_psum = np.where(psum > 0, 1.0 / np.maximum(psum, _eps), 0.0)
            inv_vpsum = np.where(vpsum > 0, 1.0 / np.maximum(vpsum, _eps), 0.0)
        else:
            inv_psum = np.where(psum > 0, 1.0 / np.where(psum > 0, psum, 1.0), 0.0)
            inv_vpsum = np.where(vpsum > 0, 1.0 / np.where(vpsum > 0, vpsum, 1.0), 0.0)

        Jtxn = np.zeros((self.B, nR))
        if len(self.t_rows):
            wt = (self.ctot * moved_tot)                       # per row coefficient
            for b in range(self.B):
                sel = self.t_rows[self.t_bandcol == b]
                if not len(sel):
                    continue
                # δ (own-column) term: only for q that is a capped row in band b
                Jtxn[b, sel] += wt[sel] * act[sel] * inv_psum[sel]
                # cell term: A_b[c] = Σ_{r∈sel, cell c} ctot·moved_tot·pshare, applied to every q∈c
                A_bc = np.bincount(gcode[sel], wt[sel] * pshare[sel], minlength=ngc)
                Jtxn[b, :] -= act * inv_psum * A_bc[gcode]

        Jvamp = np.zeros((self.B, nR))
        if len(self.pc_bandcol):
            ok = self.pc_org >= 0
            for b in range(self.B):
                m = (self.pc_bandcol == b) & ok
                if not m.any():
                    continue
                o_b = self.pc_org[m]
                pool_b = self.pc_pool[m]
                # δ term: G_b[q] = Σ_{j: o_j=q} pool[j]
                G = np.bincount(o_b, pool_b, minlength=nR)
                # cell term: C_b[c] = Σ_{j: gcode[o_j]=c} pool·vcpos[o]·vshare[o]
                C_bc = np.bincount(gcode[o_b], pool_b * self.vcpos[o_b] * vshare[o_b], minlength=ngc)
                Jvamp[b, :] = self.vcpos * inv_vpsum * (G - C_bc[gcode])
        return Jvamp, Jtxn

    # [FN-385]
    def spec_jacobian_pr(self, prop_raw: np.ndarray):
        """Return (spec_values[S], d spec_value / d prop_raw[S, K]) — the exact analytic Jacobian
        in prop_raw space (before the linear `incidence` chain to share space)."""
        vamp, txn, inter = self._forward_pr(np.asarray(prop_raw, float))
        Jvamp, Jtxn = self._jac_pr(inter)
        S = len(self.specs)
        vals = np.zeros(S)
        Jpr_rows = np.zeros((S, self.nR))
        for i, sp in enumerate(self.specs):
            if len(sp.cols) == 0:
                continue
            metric = txn if sp.metric == "txn" else vamp
            Jm = Jtxn if sp.metric == "txn" else Jvamp
            vals[i] = float(metric[sp.cols].sum())
            Jpr_rows[i] = Jm[sp.cols, :].sum(axis=0)
        # chain pr → prop_raw: prop_raw[k] feeds row r iff propidx[r]==k and r not masked
        Jprop = np.zeros((S, self.K))
        # 19dt: d pr[r] / d prop_raw[k] is now pw[r], not 1 - the chain must carry the
        # weight or the Jacobian describes a forward pass the solver no longer runs.
        live = self.pw > 0.0
        # scatter-add each row's sensitivity onto its prop-key column
        for i in range(S):
            np.add.at(Jprop[i], self.propidx[live],
                      Jpr_rows[i, live] * self.pw[live])
        return vals, Jprop

    # [FN-386]
    def spec_jacobian_shares(self, s: np.ndarray):
        """Exact (spec_values[S], d spec_value / d s[S, N]) in SHARE space via the incidence chain."""
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw
        prop_raw = shares_to_prop_raw(np.asarray(s, float)[None, :], self.incidence)[0]
        _vals, Jprop = self.spec_jacobian_pr(prop_raw)
        Js = np.asarray(Jprop @ self.incidence)                 # (S, K)·(K, N) = (S, N)
        return _vals, Js

    # ------------------------------------------------------------------ breach
    # [FN-387]
    def breach(self, s: np.ndarray, *, weighted: bool = False) -> float:
        """Total RELATIVE band breach (same definition as band_greedy): Σ(now/ceil−1)_+ + Σ(1−now/floor)_+.
        This is the exact projector breach — the quantity the solver drives to 0."""
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw
        prop_raw = shares_to_prop_raw(np.asarray(s, float)[None, :], self.incidence)[0]
        vals = self.spec_values(prop_raw)
        tot = 0.0
        for v, sp in zip(vals, self.specs):
            w = sp.weight if weighted else 1.0
            if sp.ceil is not None and sp.ceil > 0 and v > sp.ceil:
                tot += w * (v / sp.ceil - 1.0)
            if sp.floor is not None and sp.floor > 0 and 0.0 < v < sp.floor:
                tot += w * (1.0 - v / sp.floor)
        return float(tot)


# --------------------------------------------------------------------------- solver
# [FN-388]
def _project_capped_simplex_cells(s, cell_starts, cell_counts, elig, cap, budget):
    """Euclidean projection of each cell onto {0 ≤ x ≤ cap over eligible rows, Σ = budget[c]}.
    Reused from the heuristic's closed-form bisection (kept local to avoid a hard import cycle)."""
    from routing_optimiser.s4_search.seed_search import _project_capped_simplex_cells as _p
    return _p(s, cell_starts, cell_counts, elig, cap, budget)


# [FN-389]
# [FN-393b]
def floor_catchall_shares(shares, floor_mask, share_floor, cell_starts, cell_counts):
    """DEPLOY-TRUTHFUL floor: bump every masked gateway sitting below ``share_floor`` UP to it, and
    take the added mass proportionally from the cell's NON-masked gateways, so each cell still sums to
    its original total.

    WHY: the pipeline re-adds a catch-all incumbent that a split zeroed (~10.6 %), but ANY positive
    specific share OVERRIDES that re-add. So a solver that drives a catch-all MID to exactly 0 to
    minimise raw breach is choosing the DEPLOY-PESSIMAL point — deployed, that 0 becomes ~10.6 %.
    Pinning masked gateways at a tiny ``share_floor`` (e.g. 0.1 %) is what actually ships and stops the
    solver picking 0. Applied to the base AND to every candidate the solvers evaluate, so the projector
    breach they minimise EQUALS the deployed breach (no re-add can fire inside the feasible region).

    Non-masked rows donate the mass; masked rows already ≥ floor are left untouched. A cell with no
    non-masked donor mass to give is left unchanged (can't floor without going negative — rare, floor
    is tiny). Pure; never raises for finite input. No-op when mask/floor is empty."""
    s = np.asarray(shares, float).copy()
    if floor_mask is None or not (float(share_floor) > 0):
        return s
    m = np.asarray(floor_mask, bool)
    if not m.any():
        return s
    eps = float(share_floor)
    cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
    bump = m & (s < eps)
    if not bump.any():
        return s
    add = np.where(bump, eps - s, 0.0)                                  # mass to add per bumped row
    cell_add = np.repeat(np.add.reduceat(add, cs), cc)                  # per-cell added mass (broadcast)
    donor = ~m                                                          # only non-catch-all rows donate
    donor_mass = np.repeat(np.add.reduceat(np.where(donor, s, 0.0), cs), cc)
    ok = donor_mass > cell_add + 1e-15                                  # cell-constant: enough to give?
    scale = np.where(donor_mass > 0, (donor_mass - cell_add) / np.where(donor_mass > 0, donor_mass, 1.0), 1.0)
    out = np.where(bump, eps, s)                                        # bumped rows → floor
    out = np.where(donor, s * scale, out)                              # donors scaled down
    out = np.where(ok, out, s)                                         # infeasible cell → revert wholesale
    return out


def solve_least_breach(exact_bands, incidence, base_shares, cell_starts, cell_counts, elig,
                       *, max_share=1.0, max_outer=40, tol=1e-7, tr_init=0.25, tr_min=1e-4,
                       weighted=False, verbose=False, log_fn=None,
                       floor_mask=None, share_floor=0.0, stall=None, deliver_fn=None):
    """EXACT successive-LP solve of  min total band breach  s.t. per-cell simplex + max-share.

    Uses `ExactBandModel` (true projector value + analytic Jacobian). Each outer step linearises the
    banded specs at the current split, solves an LP (HiGHS) for a trust-region step that minimises the
    linearised slack, then ACCEPTS the step only if the TRUE (re-projected) breach improves — else the
    trust region shrinks. Returns (shares[N], info). `info['breach0']`/`info['breach']` are the exact
    projector breaches before/after; `info['feasible']` is True iff the final breach ≤ tol (a genuine
    compliance certificate). Never raises: any failure returns the base split with info['ok']=False.

    Only gateways feeding a band move; all others stay at `base_shares` (exact for the band objective).
    Local optimum only (fractional-VAMP nonconvexity + fixed active mask) — see module docstring.

    `stall` (19ck): stop after this many CONSECUTIVE rejected steps. 0 (the default, and what
    ROUTING_SEED_LP_STALL=0 means) keeps the pre-19ck behaviour exactly — the only exit from a
    rejection stays `tr < tr_min`. `None` reads ROUTING_SEED_LP_STALL. A stall stop is
    ANSWER-AFFECTING: a step rejected at `tr` may be accepted at `tr/2`, which is what the trust
    region is FOR. The step ledger below (`info['steps']`, `info['stall_min_safe']`) is what decides
    whether a given K is safe, and it is recorded whether or not the stop is armed.

    `deliver_fn` (19go): the DELIVERY transform (blocked-caps → eligibility → cap). When given, the
    ACCEPT TEST judges each candidate on the DELIVERED breach — the quantity the engine selects
    with and delivery ships — instead of the RAW one. The LP itself still linearises the RAW model
    (`spec_jacobian_shares`), and that is deliberate: delivery is NOT linear (blocked-caps
    redistribution, eligibility renorm and the cap are all piecewise), so it has no Jacobian to
    hand the LP. The split is therefore PROPOSED on the linear model and JUDGED on delivery, which
    is the same never-worse contract as before, just measured on the basis that ships. Expect more
    rejected steps than the RAW version: a step that helps the raw split can be undone by
    delivery's renormalisation, and that is precisely the divergence this closes. None restores the
    pre-19go RAW behaviour byte for byte (ROUTING_SEED_DELIV=0)."""
    info = {"ok": False, "build": __build__, "reason": "", "n_free": 0, "outer": 0,
            "breach0": float("nan"), "breach": float("nan"), "feasible": False,
            # 19ck step ledger. `outer` is the last ACCEPTED step and always was; `ran` is how many
            # actually executed, and the gap between the two is the thing no earlier log could show.
            "ran": 0, "steps": [], "stall_k": 0, "stall_min_safe": 0,
            "trailing_rejects": 0, "trailing_secs": 0.0, "secs": 0.0, "stopped": ""}
    try:
        s = np.asarray(base_shares, float).copy()
        N = s.shape[0]
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        elig = np.asarray(elig, float)
        cap = float(max_share) if (max_share and float(max_share) > 0) else 1.0
        # Catch-all ε-floor: pin catch-all gateways at ≥ share_floor so the solver can't choose the
        # deploy-pessimal 0 (which the pipeline re-adds at ~10.6 %). Applied to the base AND to every
        # accepted candidate below, so the projector breach == the deployed breach in the feasible box.
        _fmask = ((np.asarray(floor_mask, bool) & (elig > 0.5))
                  if (floor_mask is not None and float(share_floor) > 0) else None)
        s = floor_catchall_shares(s, _fmask, share_floor, cs, cc)
        model = ExactBandModel(exact_bands, incidence, vps_eps=_VPS_EPS)  # gradient-only reg

        def _brc(_x, _m=model, _d=deliver_fn):
            """Breach on the basis the caller asked for. `deliver_fn=None` → the RAW split, exactly
            as before 19go; otherwise the DELIVERED split, which is what the engine selects on."""
            if _d is None:
                return _m.breach(_x, weighted=weighted)
            # 1-D: the transform's serial reference path, not the threaded one — a single
            # candidate has nothing to thread, and `model.breach` takes a vector anyway.
            return _m.breach(np.asarray(_d(np.asarray(_x, float)), float), weighted=weighted)
        info["basis"] = "delivered" if deliver_fn is not None else "raw"
        b0 = _brc(s)
        info["breach0"] = b0
        if b0 <= tol:
            info.update(ok=True, feasible=True, breach=b0, reason="base already compliant")
            return s, info
        if log_fn:
            log_fn(f"      exact projector: gradient regularised (vpsum/psum floored at {_VPS_EPS:g}) — "
                   "tames the 1/vpsum blow-up so steps are navigable; forward breach stays exact.")

        # Which gateways can move any band? (nonzero column in J at the base split.)
        _, J0 = model.spec_jacobian_shares(s)
        movable = np.abs(J0).sum(axis=0) > 0
        free = movable & (elig > 0.5)
        info["n_free"] = int(free.sum())
        if not free.any():
            info.update(reason="no band-feeding gateways are eligible/movable", breach=b0)
            return s, info

        # per-cell fixed budget for the free rows (total minus the pinned reference of non-free rows)
        cell_of = np.repeat(np.arange(len(cs)), cc)
        best_s = s.copy(); best_b = b0
        tr = float(tr_init)
        # ── 19ck: STEP LEDGER (always) + STALL STOP (opt-in) ──────────────────────────────────
        # `_rej` counts CONSECUTIVE rejections. `_rej_paid` is the longest such run that a LATER
        # acceptance justified — the smallest K that would have cost this run nothing. Everything
        # here is recording; with `stall == 0` the control flow is exactly the pre-19ck loop.
        if stall is None:
            try:
                stall = int(_os.environ.get("ROUTING_SEED_LP_STALL", "0") or 0)
            except Exception:  # noqa: BLE001 — a bad env value must not fail the seed
                stall = 0
        stall = max(0, int(stall))
        info["stall_k"] = stall
        _steps = info["steps"]
        _rej = 0
        _rej_paid = 0
        _t_all = _time.perf_counter()
        if log_fn:
            log_fn(f"      exact projector: {info['n_free']:,} band-feeding gateways free of {N:,} "
                   f"(base breach {b0:.4g}); successive-LP, up to {int(max_outer)} steps. One-time solve — "
                   "at BIN grain each step is a large sparse LP, so this can take a few minutes."
                   + (f" Stall stop ARMED at {stall} consecutive rejected step(s) "
                      "(ROUTING_SEED_LP_STALL) — this CAN change the seed."
                      if stall else ""))
        for outer in range(int(max_outer)):
            if log_fn:
                log_fn(f"      exact projector: step {outer + 1}/{int(max_outer)} "
                       f"(best breach {best_b:.4g}, tr={tr:.3g})…")
            _t_step = _time.perf_counter()
            info["ran"] = outer + 1

            def _ledger(_verdict, _tr=None, _b=None):
                """Record one step. Called on EVERY exit path, so `ran` == len(steps) always."""
                _steps.append({"step": outer + 1,
                               "tr": float(tr if _tr is None else _tr),
                               "verdict": str(_verdict),
                               "breach": float(best_b if _b is None else _b),
                               "secs": float(_time.perf_counter() - _t_step)})
            vals, J = model.spec_jacobian_shares(best_s)          # exact value + exact gradient
            # LP variables: Δs over ALL N (bounded 0 for non-free), then slacks per active band side.
            # Build only the constraint rows that are ceil/floor for a spec with columns.
            n_slack = 0
            rows_meta = []
            for i, sp in enumerate(model.specs):
                if len(sp.cols) == 0:
                    continue
                gi = J[i]                                          # (N,)
                vi = vals[i]
                if sp.ceil is not None and sp.ceil > 0:
                    rows_meta.append(("ceil", i, sp.ceil, vi, gi)); n_slack += 1
                if sp.floor is not None and sp.floor > 0:
                    rows_meta.append(("floor", i, sp.floor, vi, gi)); n_slack += 1
            if n_slack == 0:
                _ledger("no-active-bands")
                info["stopped"] = "no-active-bands"
                break
            nvar = N + n_slack
            # objective: minimise Σ slack (relative), slacks are the last n_slack vars
            c_obj = np.zeros(nvar); c_obj[N:] = 1.0
            rows = []; rhs = []
            for k, (side, i, lim, vi, gi) in enumerate(rows_meta):
                row = np.zeros(nvar)
                if side == "ceil":
                    # vi + gi·Δs ≤ lim·(1+slack)  ->  gi·Δs − lim·slack ≤ lim − vi
                    row[:N] = gi; row[N + k] = -lim
                    rhs.append(lim - vi)
                else:
                    # vi + gi·Δs ≥ lim·(1−slack)  ->  −gi·Δs − lim·slack ≤ vi − lim
                    row[:N] = -gi; row[N + k] = -lim
                    rhs.append(vi - lim)
                rows.append(row)
            # SPARSE LP matrices. At BIN grain nvar≈135k and n_cell≈14k, so a DENSE A_eq would be ~15 GB
            # (and was rebuilt every step — the hang). A_eq is a cell-membership indicator (exactly one 1
            # per free column) → ~nvar nonzeros; A_ub is the n_slack dense band-gradient rows. HiGHS solves
            # the sparse LP directly — identical problem, identical result, just representable at BIN grain.
            fidx = np.where(free)[0]
            A_ub = _sparse.csr_matrix(np.asarray(rows, float)) if rows else None
            b_ub = np.asarray(rhs, float)
            # equality: Σ_free Δs = 0 per cell (keep each cell sum fixed)
            n_cell = len(cs)
            Aeq = _sparse.coo_matrix((np.ones(fidx.size), (cell_of[fidx], fidx)),
                                     shape=(n_cell, nvar)).tocsr()
            beq = np.zeros(n_cell)
            # bounds: Δs box (trust region ∩ feasible-share box) for free vars, 0 for non-free; slack ≥ 0
            lb = np.zeros(nvar); ub = np.zeros(nvar)
            lb[fidx] = np.maximum(-tr, 0.0 - best_s[fidx])
            ub[fidx] = np.minimum(tr, cap - best_s[fidx])
            ub[N:] = None                                          # slacks unbounded above
            bounds = [(lb[j], (None if j >= N else ub[j])) for j in range(nvar)]
            res = _linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                           bounds=bounds, method="highs")
            if not res.success:
                _ledger("lp-failed")
                _rej += 1
                tr *= 0.5
                if tr < tr_min:
                    info["stopped"] = "tr-below-min"
                    break
                if stall and _rej >= stall:
                    info["stopped"] = "stall"
                    break
                continue
            ds = res.x[:N]
            cand = best_s + ds
            cand = _project_capped_simplex_cells(cand, cs, cc, elig, cap, budget=1.0)
            cand = floor_catchall_shares(cand, _fmask, share_floor, cs, cc)  # keep catch-all ≥ floor
            bc = _brc(cand)
            if bc < best_b - max(1e-12, 1e-4 * best_b):
                best_s = cand; best_b = bc
                info["outer"] = outer + 1
                # This acceptance JUSTIFIES the rejections that preceded it: a stall stop at any
                # K <= _rej would have thrown this improvement away. That is the number the
                # operator needs, and it is only knowable by running the rejections.
                if _rej > _rej_paid:
                    _rej_paid = _rej
                _rej = 0
                _ledger("accepted", _b=bc)
                if verbose:
                    print(f"  [slp] outer {outer}: breach {best_b:.6g} (tr={tr:.3g}, free={info['n_free']})")
                if best_b <= tol:
                    info["stopped"] = "tol"
                    break
                tr = min(tr * 1.5, 0.5)                             # grow on success
            else:
                _ledger("rejected", _b=bc)
                _rej += 1
                tr *= 0.5
                if tr < tr_min:
                    info["stopped"] = "tr-below-min"
                    break
                if stall and _rej >= stall:
                    info["stopped"] = "stall"
                    break
        else:
            info["stopped"] = info["stopped"] or "max-outer"
        info["secs"] = float(_time.perf_counter() - _t_all)
        info["stall_min_safe"] = int(_rej_paid)
        info["trailing_rejects"] = int(_rej)
        info["trailing_secs"] = float(sum(d["secs"] for d in _steps[len(_steps) - _rej:])) if _rej else 0.0
        info.update(ok=True, breach=best_b, feasible=bool(best_b <= tol))
        if log_fn:
            _lp_report(info, log_fn, tr_min=float(tr_min), max_outer=int(max_outer))
        return best_s, info
    except Exception as exc:  # noqa: BLE001 - a seed must never crash the run
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return np.asarray(base_shares, float).copy(), info


# [FN-390b]
def _lp_report(info, log_fn, *, tr_min, max_outer):
    """[lp-stall] — how many successive-LP steps RAN, how many moved the breach, and what a stall
    stop would have cost as well as saved.

    The existing line quotes `info['outer']`, the last ACCEPTED step. On 2026-08-25 20:35 that read
    12 while 20 steps ran, so the log was simultaneously true and unreadable as a cost. This block
    prints both, and — the part that decides anything — `stall_min_safe`: the longest run of
    consecutive rejections THIS run that was followed by an acceptance. A stall stop at K is free on
    this run iff K > stall_min_safe. Quoting the saving without that number would repeat 19cf's
    error one level up: a ratio, or here a saving, presented without the quantity that bounds it."""
    try:
        _st = list(info.get("steps") or [])
        if not _st:
            return
        _ran = int(info.get("ran", len(_st)))
        _acc = [d for d in _st if d["verdict"] == "accepted"]
        _rej = [d for d in _st if d["verdict"] != "accepted"]
        _tacc = sum(d["secs"] for d in _acc)
        _trej = sum(d["secs"] for d in _rej)
        log_fn("      [lp-stall] successive-LP step ledger — what the 'in N LP step(s)' line above "
               "does NOT say:")
        log_fn(f"         RAN {_ran} step(s) in {info.get('secs', 0.0):.1f}s · "
               f"{len(_acc)} ACCEPTED ({_tacc:.1f}s) · {len(_rej)} REJECTED ({_trej:.1f}s). "
               f"The line above quotes the LAST ACCEPTED step ({int(info.get('outer', 0))}), not "
               f"how many ran.")
        log_fn(f"         stopped because: {info.get('stopped') or 'unknown'} "
               + {"tr-below-min": f"(trust region fell under tr_min={tr_min:g}; it halves on every "
                                  "rejection, so each further rejection is one more large sparse LP)",
                  "tol": "(breach reached the tolerance — the good exit)",
                  "stall": f"(ROUTING_SEED_LP_STALL={int(info.get('stall_k', 0))} consecutive "
                           "rejections; THIS TRUNCATED THE SOLVE — see the safety line below)",
                  "max-outer": f"(hit max_outer={max_outer} while still stepping — the solve was "
                               "CUT OFF, not finished)"}.get(str(info.get("stopped")), ""))
        _tr = int(info.get("trailing_rejects", 0))
        if _tr:
            log_fn(f"         TRAILING: the last {_tr} step(s) all failed to improve and cost "
                   f"{float(info.get('trailing_secs', 0.0)):.1f}s. That is the time a stall stop "
                   "would have RECLAIMED on this run.")
        else:
            log_fn("         TRAILING: none — the solve ended on an accepted step or at tolerance, "
                   "so a stall stop would have saved nothing here.")
        _ms = int(info.get("stall_min_safe", 0))
        _k = int(info.get("stall_k", 0))
        log_fn(f"         WHAT IT WOULD HAVE COST: the longest run of consecutive rejections that a "
               f"LATER step then justified was stall_min_safe={_ms}. A rejected step is not waste — "
               f"the trust region halves, so the SAME direction can be accepted at half the size, "
               f"which is what the region is FOR. A stall stop is free only at K > {_ms}"
               + (f"; K={_k} is armed." if _k else "; it is currently OFF."))
        if _k:
            # THE MEASUREMENT IS CENSORED WHEN THE STOP IS ARMED. The loop breaks at K consecutive
            # rejections, so a justified run of K or more can no longer be observed:
            # stall_min_safe is bounded above by K-1 BY CONSTRUCTION. An armed run therefore cannot
            # be used as evidence that its own K was safe — which is precisely the reading a bare
            # "stall_min_safe=0, K=2, fine" line would invite.
            log_fn(f"         \u26a0 ARMED, so this run's stall_min_safe is CENSORED: the loop "
                   f"stops at {_k} consecutive rejections, so any justified run of {_k} or longer "
                   f"could not have been seen (stall_min_safe \u2264 {_k - 1} by construction). "
                   "This run is NOT evidence that K is safe — only an UNARMED run "
                   "(ROUTING_SEED_LP_STALL=0) can measure that.")
            if str(info.get("stopped")) == "stall":
                log_fn(f"         \u26a0 AND IT BIT: the solve stopped at the stall, not at "
                       f"convergence, so the seed handed to the GA may be worse than an unarmed "
                       f"solve would give. Final breach {float(info.get('breach', float('nan'))):.4g} "
                       f"after {int(info.get('outer', 0))} accepted step(s).")
            if _k <= _ms:
                log_fn(f"         \u26a0 ROUTING_SEED_LP_STALL={_k} is at or below stall_min_safe="
                       f"{_ms}: this setting discarded an improvement the solver went on to find. "
                       "Raise K or set it to 0.")
        elif _tr:
            log_fn(f"         Not armed, so nothing was truncated and this run is bit-identical to "
                   f"pre-19ck. To claim the {float(info.get('trailing_secs', 0.0)):.1f}s, set "
                   f"ROUTING_SEED_LP_STALL to a K above the stall_min_safe seen across SEVERAL "
                   "UNARMED runs — one run is one sample, and the trailing steps are exactly the "
                   "ones with the least evidence behind them.")
    except Exception as _exc:  # noqa: BLE001 — a report must never break a seed
        try:
            log_fn(f"      [lp-stall] ledger report skipped ({type(_exc).__name__}: {_exc}). The "
                   "solve itself is unaffected.")
        except Exception:  # noqa: BLE001
            pass


# [FN-391]
def colocation_report(split, exact_bands, incidence, *, mid_id, cell_starts, cell_counts,
                      risk, cell_vol, elig, mid_names,
                      cell_cur=None, cell_bank=None, cell_rpgt=None, top_cells=8):
    """READ-ONLY diagnostic: for every breached CEILING MID, is a headroom SIBLING co-located?

    At the engine's cell grain (one cell = one contiguous [cell_starts[c], +cell_counts[c]) block of
    gateway-rows), for each MID whose projected M5 value exceeds its ceiling this walks the cells where
    that MID actually carries share and checks whether an ELIGIBLE, co-located OTHER MID could absorb the
    excess WITHOUT breaching its own cap — i.e. a sibling with NO ceiling on that metric (unlimited room)
    or with positive headroom. It changes NO share; it only builds human-readable log lines.

    Returns a list of strings for the caller to log. Never raises (returns a one-line skip note instead).

    Interpretation: many breaching cells WITH a co-located headroom sibling ⇒ the excess CAN move, so a
    seed that left the band breached failed to SEARCH (not a real infeasibility). Few/none ⇒ a genuine
    cell-grain / RPGT-scope block (the headroom exists at MID level but not in the same cells)."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        s = np.asarray(split, float)
        report = exact_bands.report(_s2pr(s[None, :], incidence))
        # (midl, metric) -> (headroom, now, ceil) for every ceiling band
        ceil_map = {}
        for rr in report:
            if rr.get("ceil") is not None:
                nw = float(rr["now"]); cl = float(rr["ceil"])
                ceil_map[(str(rr["midl"]).strip().lower(), str(rr["metric"]))] = (cl - nw, nw, cl)
        breached = [(m, me, h, n, c) for (m, me), (h, n, c) in ceil_map.items() if h < -1e-6]
        if not breached:
            return ["   co-location diagnostic: no breached ceiling bands at this split — nothing to check."]

        mid_id = np.asarray(mid_id, int)
        risk = np.asarray(risk, float)
        vol = np.asarray(cell_vol, float)
        el = np.asarray(elig, float) > 0.5
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        cell_of = np.repeat(np.arange(len(cs)), cc)
        nm = [str(m).strip().lower() for m in mid_names]
        name2i = {n: i for i, n in enumerate(nm)}
        nrow = mid_id.shape[0]

        def _col(a):
            return (np.asarray(a).astype(str) if a is not None else np.array([""] * nrow))
        cur, bank, rpgt = _col(cell_cur), _col(cell_bank), _col(cell_rpgt)

        def _room(sib, metric):
            return ceil_map.get((sib, metric), (float("inf"), None, None))[0]

        out = ["   ── CO-LOCATION DIAGNOSTIC (read-only): can each stuck band's excess move to a "
               "co-located, eligible sibling? ──"]
        for bm, me, h, nw, cl in breached:
            bmi = name2i.get(bm)
            if bmi is None:
                continue
            rows = np.where((mid_id == bmi) & (s > 1e-9))[0]
            if rows.size == 0:
                out.append(f"      • {bm} [{me}]: {nw:,.0f} > ceil {cl:,.0f} (over {-h:,.0f}) — carries "
                           "NO scoped share (its volume is all in frozen/unscoped RPGTs; not movable here).")
                continue
            contrib = {}
            for r in rows:
                c = int(cell_of[r])
                contrib[c] = contrib.get(c, 0.0) + s[r] * vol[r] * (risk[r] if me == "vamp" else 1.0)
            cells = np.unique(cell_of[rows])
            nwith = 0; lines = []
            for c in sorted(cells, key=lambda x: -contrib.get(int(x), 0.0)):
                lo = int(cs[c]); hi = lo + int(cc[c])
                sibs = []
                for r in range(lo, hi):
                    mi = int(mid_id[r])
                    if mi == bmi or not el[r]:
                        continue
                    sn = nm[mi] if 0 <= mi < len(nm) else str(mi)
                    room = _room(sn, me)
                    if room > 1e-6:
                        sibs.append((sn, room, float(risk[r])))
                if sibs:
                    nwith += 1
                if len(lines) < int(top_cells):
                    lbl = f"{cur[lo]}|{bank[lo]}|{rpgt[lo]}"
                    if sibs:
                        sibs.sort(key=lambda x: (x[2], -x[1]))
                        txt = ", ".join(
                            (f"{n}(room {'∞' if np.isinf(rm) else format(rm, ',.0f')}"
                             + (f", rate {rk:.3f}" if me == "vamp" else "") + ")")
                            for n, rm, rk in sibs[:4])
                        lines.append(f"         cell {int(c)} [{lbl}] ~{contrib.get(int(c), 0.0):,.0f} "
                                     f"{me} · siblings: {txt}")
                    else:
                        lines.append(f"         cell {int(c)} [{lbl}] ~{contrib.get(int(c), 0.0):,.0f} "
                                     f"{me} · NO eligible headroom sibling")
            out.append(f"      • {bm} [{me}]: {nw:,.0f} > ceil {cl:,.0f} (over {-h:,.0f}); carries scoped "
                       f"share in {cells.size} cell(s), {nwith} have ≥1 eligible headroom sibling co-located.")
            out.extend(lines)
        out.append("      (read-only — no share changed. 'room' = sibling MID's own M5 ceiling − its "
                   "current projected M5 value; ∞ = sibling has no cap on this metric. Many cells WITH a "
                   "co-located headroom sibling ⇒ the excess CAN move → a search failure, not true "
                   "infeasibility. Few/none ⇒ genuine cell-grain / scope block.)")
        return out
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the run
        return [f"   co-location diagnostic skipped ({type(exc).__name__}: {exc})."]


# [FN-392]
def unmet_summary(split, exact_bands, incidence, *, max_list=8):
    """One-line 'meets X/Y bands · N unmet: name metric now vs lim, …' for a split.

    Counts how many of the configured ceiling/floor bands the split's EXACT projected M5 value
    satisfies, and names the ones it doesn't (a MID over its ceiling or under its floor). Used to
    compare warm-start seeds at a glance. Never raises — returns '' if it can't be computed."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        rep = exact_bands.report(_s2pr(np.asarray(split, float)[None, :], incidence))
        total = 0
        unmet = []
        for rr in rep:
            cl = rr.get("ceil"); fl = rr.get("floor")
            if cl is None and fl is None:
                continue
            total += 1
            nw = float(rr["now"]); me = str(rr.get("metric"))
            if cl is not None and float(cl) > 0 and nw > float(cl) + 1e-6:
                unmet.append(f"{rr['midl']} {me} {nw:,.0f} > {float(cl):,.0f}")
            elif fl is not None and float(fl) > 0 and nw < float(fl) - 1e-6:
                unmet.append(f"{rr['midl']} {me} {nw:,.0f} < {float(fl):,.0f}")
        if total == 0:
            return ""
        if not unmet:
            return f"meets {total}/{total} bands (all constraints satisfied)"
        shown = "; ".join(unmet[:int(max_list)])
        if len(unmet) > int(max_list):
            shown += f"; +{len(unmet) - int(max_list)} more"
        return f"meets {total - len(unmet)}/{total} bands · {len(unmet)} unmet: {shown}"
    except Exception:  # noqa: BLE001 — a diagnostic must never break the run
        return ""


# [FN-393]
def held_movable_report(split, exact_bands, incidence, *, max_list=15):
    """READ-ONLY: how much of each breached band's M5 value is MOVABLE vs HELD.

    For a split, decomposes every ceiling band into the routing-invariant HELD cohort (baseline /
    FCP2+ / pre-go-live) and the MOVABLE pool (redistributed by the share). For each BREACHED ceiling
    it reports the split and a verdict:
      * HELD < ceiling  ⇒ a compliant routing EXISTS (min achievable ≈ held) → if a solver leaves it
                          breached that is a SEARCH failure, not infeasibility;
      * HELD ≥ ceiling  ⇒ STRUCTURALLY STUCK — no routing can clear it under this scope.
    Returns a list of log-line strings; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        pr = _s2pr(np.asarray(split, float)[None, :], incidence)[0]
        held, mov = model.spec_decomposition(pr)
        mean_pr, mean_fcp, mean_mv = model.spec_movable_provenance()
        specs = model.specs
        tot_h = float(held.sum()); tot_m = float(mov.sum()); tot = tot_h + tot_m
        out = ["   ── HELD-vs-MOVABLE check (in-search M5 projector; movable = pool the routing "
               "decision redistributes, held = baseline / FCP2+ / pre-go-live cohort) ──"]
        if tot > 0:
            out.append(f"      overall banded M5: {tot_m / tot * 100:.0f}% movable / "
                       f"{tot_h / tot * 100:.0f}% held  (movable {tot_m:,.0f} · held {tot_h:,.0f})")
        shown = 0
        for i, sp in enumerate(specs):
            cl = sp.ceil
            if cl is None or float(cl) <= 0:
                continue
            h = float(held[i]); m = float(mov[i]); t = h + m
            if t <= float(cl) + 1e-6:            # only breached ceilings
                continue
            frac_m = (m / t * 100.0) if t > 0 else 0.0
            if h < float(cl) - 1e-6:
                verdict = (f"min achievable ≈ held {h:,.0f} < ceil {float(cl):,.0f} ⇒ MOVABLE "
                           "(a compliant routing EXISTS)")
            else:
                verdict = (f"held {h:,.0f} ≥ ceil {float(cl):,.0f} ⇒ STRUCTURALLY STUCK "
                           "(routing alone can't clear it)")
            out.append(f"      • {sp.label} [{sp.metric}]: M5 {t:,.0f} = movable {m:,.0f} "
                       f"({frac_m:.0f}%) + held {h:,.0f} · ceil {float(cl):,.0f} → {verdict}")
            _pv = mean_pr[i]; _fv = mean_fcp[i]; _mvv = mean_mv[i]
            if np.isfinite(_mvv):
                out.append(f"          movable fraction {_mvv * 100:.0f}% = pro_rata {_pv:.2f} "
                           f"× fcp1_frac {_fv:.2f}  (fcp1_frac = first-attempt reroutable slice; if this "
                           "is far below your first-attempt %, the export's fcp1_frac is the bug)")
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached ceiling bands at this split.)")
        out.append("      (read-only. held < ceil on a breached MID ⇒ compliance is REACHABLE, so a "
                   "solver stuck on it has a SEARCH bug — not infeasibility. held ≥ ceil ⇒ genuinely "
                   "stuck under this scope. The movable fraction splits into pro_rata (go-live phasing) "
                   "× fcp1_frac (first-attempt reroutable slice) — whichever factor is small is the "
                   "cause; a small fcp1_frac vs your known first-attempt % points at the export.)")
        return out
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the run
        return [f"   held-vs-movable check skipped ({type(exc).__name__}: {exc})."]


# [FN-394]
def floor_min_report(split, exact_bands, incidence, *, mid_id, cell_starts, cell_counts, elig,
                     mid_names, whatif_floor=0.0, max_list=15):
    """READ-ONLY: the TRUE reachable minimum M5 each breached ceiling MID can reach.

    The full-matrix engine decodes shares with a plain per-cell softmax and applies NO hard min-share
    floor (the ``exploration_floor`` is honoured by the tilt/softmax engines, NOT by the full-matrix
    decode or delivery), so a MID can be routed arbitrarily close to 0 wherever an eligible sibling
    can absorb its share. This pushes each breached MID toward 0 in those cells, re-projects, and reads
    its resulting M5 — the genuine reachable minimum (≈ the routing-invariant ``held`` cohort):
      * reachable-min < ceiling ⇒ compliance IS reachable → a stuck solver has a SEARCH bug;
      * reachable-min ≥ ceiling ⇒ genuinely UNREACHABLE (the held cohort alone exceeds the cap →
        structural: the lever is the cap / RPGT scope, not the optimiser).
    Cells where the MID is the ONLY eligible gateway are irreducible (its share can't be reduced) and
    are counted. ``whatif_floor`` > 0 adds a clearly-labelled hypothetical (min if a hard floor of that
    size WERE enforced — it is NOT, by this engine). Returns log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        s = np.asarray(split, float)
        base_vals = model.spec_values(_s2pr(s[None, :], incidence)[0])
        specs = model.specs
        mid_id = np.asarray(mid_id, int)
        el = np.asarray(elig, float) > 0.5
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        ncell = len(cs); cell_id = np.repeat(np.arange(ncell), cc)
        nm = [str(m).strip().lower() for m in mid_names]
        name2i = {n: i for i, n in enumerate(nm)}

        def _push_min(mi, fl):
            """Push MID index `mi` to share `fl` in every cell with an eligible sibling; renormalise;
            return (its projected M5, #irreducible-cells)."""
            is_m = (mid_id == mi) & el
            is_o = (mid_id != mi) & el
            m_cnt = np.bincount(cell_id[is_m], minlength=ncell)
            o_cnt = np.bincount(cell_id[is_o], minlength=ncell)
            osum = np.bincount(cell_id[is_o], weights=s[is_o], minlength=ncell)
            mfloor = m_cnt * float(fl)
            act = (m_cnt > 0) & (o_cnt > 0) & (mfloor < 1.0 - 1e-9)
            act_row = act[cell_id]
            s_m = s.copy()
            s_m[is_m & act_row] = float(fl)
            rem = 1.0 - mfloor
            scale = np.where(osum > 1e-12, rem / np.where(osum > 1e-12, osum, 1.0), 0.0)
            eq = np.where(o_cnt > 0, rem / np.where(o_cnt > 0, o_cnt, 1.0), 0.0)
            o_act = is_o & act_row
            use_scale = osum[cell_id] > 1e-12
            s_m[o_act] = np.where(use_scale[o_act], s[o_act] * scale[cell_id[o_act]],
                                  eq[cell_id[o_act]])
            val = model.spec_values(_s2pr(s_m[None, :], incidence)[0])[mi_spec_row]
            n_irr = int(((m_cnt > 0) & (o_cnt == 0)).sum())
            return val, n_irr

        wf = float(whatif_floor or 0.0)
        out = ["   ── REACHABLE MINIMUM (route each breached MID toward 0 wherever an eligible sibling "
               "can absorb it — the full-matrix engine applies NO hard share floor) ──"]
        shown = 0
        for i, sp in enumerate(specs):
            cl = sp.ceil
            if cl is None or float(cl) <= 0:
                continue
            if base_vals[i] <= float(cl) + 1e-6:          # only breached ceilings
                continue
            mi = name2i.get(str(sp.label).strip().lower())
            if mi is None or not ((mid_id == mi) & el).any():
                continue
            mi_spec_row = i
            val0, n_irr = _push_min(mi, 0.0)              # true reachable min (no floor)
            reach = val0 < float(cl) - 1e-6
            verdict = ("< ceil ⇒ REACHABLE ⇒ a stuck solver is the cause (SEARCH bug), not infeasibility"
                       if reach else
                       "≥ ceil ⇒ genuinely UNREACHABLE (held cohort alone exceeds the cap — structural)")
            line = (f"      • {sp.label} [{sp.metric}]: now {base_vals[i]:,.0f} · reachable-min "
                    f"{val0:,.0f} · ceil {float(cl):,.0f} → {verdict}")
            if n_irr:
                line += f"  [{n_irr} cell(s) where it's the ONLY eligible gateway — irreducible]"
            out.append(line)
            if wf > 0:
                valf, _ = _push_min(mi, wf)
                out.append(f"          (what-if only, NOT applied by this engine: with a hard "
                           f"{wf * 100:.0f}% floor the min would be {valf:,.0f})")
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached ceiling bands at this split.)")
        out.append("      (read-only. reachable-min routes the MID toward 0 wherever a sibling can "
                   "absorb it — valid because the full-matrix decode is a plain softmax with no hard "
                   "floor. reachable-min < ceiling ⇒ reachable, so a stuck solver is the cause; "
                   "reachable-min ≥ ceiling ⇒ structurally impossible under this scope.)")
        return out
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the run
        return [f"   reachable-minimum check skipped ({type(exc).__name__}: {exc})."]


# [FN-395]
def vamp_sibling_report(split, exact_bands, incidence, *, max_list=15):
    """READ-ONLY: can a breached VAMP MID's SHARE actually pull VAMP off it? (the cliff test)

    VAMP redistributes by ``vshare = pr·vcpos / Σ(pr·vcpos)`` — a MID's share of the VAMP-POSITIVE
    gateways only. So a breached VAMP MID's received VAMP falls when its share moves ONLY if a
    co-located VAMP-positive (`vcpos > 0`) sibling absorbs it; moving to a non-VAMP gateway leaves
    `vshare` (and thus its VAMP) unchanged, and where it is the SOLE VAMP-positive gateway `vshare = 1`
    for any share > 0 (the softmax engine can only reduce it at EXACTLY 0, which it never reaches).

    For each breached VAMP MID this reports, over the projector cells where it carries VAMP, how many
    have a co-located VAMP-positive sibling. Few/none ⇒ the VAMP is STRUCTURALLY immovable by this
    engine (not a solver bug). Works in the projector's own cell grain (`gcode`). Returns log lines;
    never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        base_vals = model.spec_values(_s2pr(np.asarray(split, float)[None, :], incidence)[0])
        specs = model.specs
        pk = list(getattr(model.pj, "prop_keys", []))
        prop_mid = np.array([str(pk[j]).split("|")[-1].strip().lower() if j < len(pk) else ""
                             for j in range(max(len(pk), 1))])
        row_mid = prop_mid[model.propidx] if model.propidx.size else np.array([], dtype=object)
        gc = model.gcode
        vpos = (model.vcpos > 0.5) & (~model.mask)            # VAMP-positive & live reduced rows
        ncell = model.ngc
        vpos_per_cell = np.bincount(gc[vpos], minlength=ncell) if vpos.any() else np.zeros(ncell, int)
        out = ["   ── VAMP-POSITIVE SIBLING check (a breached VAMP MID's share only lowers its VAMP "
               "where a co-located VAMP-positive gateway exists; vshare self-normalises otherwise) ──"]
        shown = 0
        for i, sp in enumerate(specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            m = str(sp.label).strip().lower()
            rows_m = np.where((row_mid == m) & vpos)[0] if row_mid.size else np.array([], int)
            if rows_m.size == 0:
                out.append(f"      • {sp.label} [vamp]: {base_vals[i]:,.0f} > {float(cl):,.0f} — no "
                           "VAMP-positive rows for this MID (its VAMP is entirely aged/pool) — n/a")
                shown += 1
                continue
            cells_m = np.unique(gc[rows_m])
            m_per_cell = np.bincount(gc[rows_m], minlength=ncell)
            has_sib = (vpos_per_cell - m_per_cell) > 0            # another VAMP-positive row in the cell
            n_with = int(has_sib[cells_m].sum()); n_tot = int(cells_m.size)
            tag = ("reducible by routing" if n_with == n_tot else
                   "SOLE VAMP gateway in ALL its cells → VAMP immovable by share" if n_with == 0 else
                   f"immovable in {n_tot - n_with} of {n_tot} cells (sole VAMP gateway there)")
            out.append(f"      • {sp.label} [vamp]: {base_vals[i]:,.0f} > {float(cl):,.0f} · "
                       f"{n_with}/{n_tot} of its VAMP cells have a co-located VAMP-positive sibling "
                       f"→ {tag}")
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        out.append("      (read-only. Where a breached VAMP MID is the SOLE VAMP-positive gateway in a "
                   "cell, vshare = 1 for any share > 0, so the softmax engine cannot reduce its VAMP "
                   "there — a structural limit, not a solver bug. Many such cells ⇒ the fix must add an "
                   "eligible VAMP-positive recipient or allow zeroing, not more search.)")
        return out
    except Exception as exc:  # noqa: BLE001 — a diagnostic must never break the run
        return [f"   vamp-positive-sibling check skipped ({type(exc).__name__}: {exc})."]


# [FN-396]
def incidence_selfcheck_report(split, exact_bands, incidence, *, mid_id=None, mid_names=None):
    """READ-ONLY (#1): does the column→prop-key incidence used IN the search cover the split?

    Reports how many of the split's share columns map to a projector prop-key and how much share
    mass survives the roll-up. <100% coverage of banded gateways, or a large dropped mass, means the
    split the GA SCORES differs from what the projector sees (a scored-vs-delivered mismatch).

    When `mid_id` (per-column MID index) and `mid_names` (index→name) are supplied, also reports a
    PER-BANDED-MID breakdown: of each banded MID's routed share mass, how much maps vs is dropped.
    A txn band whose share sits in no-baseline cells (which the baseline-anchored scaffold can't
    represent) shows a HIGH dropped% here — localising the scored-vs-delivered under-count to that
    MID. Returns log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        s = np.asarray(split, float); N = int(s.size)
        try:                                              # sparse
            _colnnz = np.asarray(incidence.getnnz(axis=0)).ravel()
            K = int(incidence.shape[0])
        except Exception:                                 # dense
            inc = np.asarray(incidence)
            _colnnz = (np.abs(inc).sum(axis=0) > 0).astype(np.int64); K = int(inc.shape[0])
        _mapped = _colnnz > 0
        cov = int(_mapped.sum())
        pr = _s2pr(s[None, :], incidence)[0]
        mass_prop = float(pr.sum()); mass_share = float(s.sum())
        out = ["   ── INCIDENCE SELF-CHECK (does the search's column→prop-key map cover the split?) ──",
               f"      {cov:,}/{N:,} share columns map to a prop-key ({(cov / max(N, 1)) * 100:.1f}%) · "
               f"{K:,} prop-keys · Σprop_raw {mass_prop:,.1f} vs Σshare {mass_share:,.1f} "
               f"(dropped {mass_share - mass_prop:,.1f} of share mass)",
               ]
        # 19gs: FOUR PARAGRAPHS DELETED. They were the write-up of the 2026-08-31 investigation
        # into a 9,018-cell coverage gap — how to read Σshare as a cell count, why those cells
        # cannot reach a band, why [profiles] was not a contradiction, and what the note used to
        # say before it was settled. That gap was closed by [require-forecast] (19em), which
        # removes those cells upstream; coverage has read 100.0% on every run since. Four
        # paragraphs of settled history printed on every healthy run is what buried the ONE line
        # that matters. It is directly above, and the check below states its own verdict.
        if cov < N:
            out.append(
                f"      ⚠ {N - cov:,} share column(s) map to NO prop-key, so the band projector "
                "cannot see them: whatever the search routes there is invisible to every band "
                "figure in this run, and those MIDs will deliver under what was scored. Σshare "
                "is a CELL COUNT (each cell's shares sum to 1.0), so the dropped figure above is "
                "a number of cells. Read [drop-measure] for which rows, and [rung] for whether "
                "the shipped split loses the same ones.")
        # PER-BANDED-MID coverage (needs the per-column MID map). Uses ExactBandModel's specs +
        # labels — the SAME naming seed_gradient_report aligns to mid_names — so it lines up with the
        # per-column mid_id. metric is cross-referenced from exact_bands.specs (band_scoring BandSpec).
        if mid_id is not None and mid_names is not None:
            try:
                _mid = np.asarray(mid_id, int)
                if _mid.size == N:
                    _nm = [str(m).strip().lower() for m in mid_names]
                    _name2i = {n: i for i, n in enumerate(_nm)}
                    _met_by_lab = {str(getattr(_sp, "midl", "")).strip().lower():
                                   str(getattr(_sp, "metric", "")).strip().lower()
                                   for _sp in getattr(exact_bands, "specs", [])}
                    model = ExactBandModel(exact_bands, incidence)
                    rows = []
                    for sp in model.specs:
                        _lab = str(getattr(sp, "label", "")).strip().lower()
                        _mi = _name2i.get(_lab)
                        if _mi is None:
                            continue
                        _cols = _mid == _mi
                        _tot = float(s[_cols].sum())
                        if _tot <= 1e-9:
                            continue
                        _map = float(s[_cols & _mapped].sum())
                        _drop = _tot - _map
                        rows.append((_drop / _tot, _lab, _met_by_lab.get(_lab, "?"), _tot, _map, _drop))
                    if rows:
                        out.append("   ── per-MID scaffold coverage (banded MIDs, worst dropped% first; dropped "
                                    "share = what the projector can't see ⇒ that MID's scored-vs-delivered "
                                    "under-count) ──")
                        for _dp, _lab, _met, _tot, _map, _drop in sorted(rows, reverse=True):
                            out.append(f"      {_lab} [{_met}]: routed share mass {_tot:,.1f} · mapped "
                                       f"{_map:,.1f} ({(_map / _tot) * 100:.0f}%) · dropped {_drop:,.1f} "
                                       f"({_dp * 100:.0f}%)")
            except Exception as _pce:  # noqa: BLE001
                out.append(f"      (per-MID coverage skipped: {type(_pce).__name__}: {_pce})")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   incidence self-check skipped ({type(exc).__name__}: {exc})."]


# [FN-397]
def seed_gradient_report(split, exact_bands, incidence, *, mid_id, mid_names, max_list=15):
    """READ-ONLY (#3): |∂band/∂share| at the seed for each breached MID.

    A near-zero gradient w.r.t. the MID's OWN shares means reducing its own share barely changes its
    band value — the quantitative signature of the vshare self-normalisation cliff (and why the exact
    projector takes 0 steps). Returns log lines; never raises."""
    try:
        model = ExactBandModel(exact_bands, incidence)
        vals, J = model.spec_jacobian_shares(np.asarray(split, float))
        specs = model.specs
        mid_id = np.asarray(mid_id, int)
        nm = [str(m).strip().lower() for m in mid_names]
        name2i = {n: i for i, n in enumerate(nm)}
        out = ["   ── SEED GRADIENT (|∂band/∂share| at the seed; ≈0 on the MID's OWN share ⇒ its band "
               "can't be moved by its own share — the vshare cliff) ──"]
        shown = 0
        for i, sp in enumerate(specs):
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or vals[i] <= float(cl) + 1e-6:
                continue
            mi = name2i.get(str(sp.label).strip().lower())
            gi = np.abs(J[i])
            own = gi[mid_id == mi] if mi is not None else np.array([])
            g_own = float(own.max()) if own.size else 0.0
            g_all = float(gi.max()) if gi.size else 0.0
            # A legitimate band gradient is volume-scale (≲ 1e6). Anything ≫ that is the 1/vpsum
            # blow-up on a near-empty (vpsum≈0) cell — the vshare 0/0 singularity, NOT a usable
            # movable direction. A near-zero own-gradient is the flat cliff. Only in between is it
            # a genuinely healthy, navigable gradient.
            _DEGEN = 1e8
            if g_own > _DEGEN or g_all > _DEGEN:
                verdict = ("  → gradient DEGENERATE (~1/vpsum blow-up; the band sits on the vshare 0/0 "
                           "singularity) — NOT usefully movable by a gradient/softmax step")
            elif g_own <= 1e-9 or g_own < 1e-6 * max(g_all, 1e-12):
                verdict = "  → own-gradient ≈ 0 ⇒ CLIFF (own share can't move its band)"
            else:
                verdict = "  → own-gradient healthy ⇒ its share CAN move its band"
            out.append(f"      • {sp.label} [{sp.metric}]: max|∂/∂own-share| {g_own:.3g} · "
                       f"max|∂/∂any-share| {g_all:.3g}{verdict}")
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached ceiling bands at this split.)")
        out.append("      (read-only. A gradient ≫ volume-scale (≳1e8) is the 1/vpsum degeneracy — the "
                   "band is on the vshare 0/0 singularity and not navigable; a near-zero gradient is the "
                   "flat cliff; both mean the search can't move it, for different reasons.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   seed-gradient check skipped ({type(exc).__name__}: {exc})."]


# [FN-398]
def vpsum_report(split, exact_bands, incidence, *, near_zero=1e-6, max_list=15):
    """READ-ONLY (#4): the VAMP denominator (vpsum = Σ pr·vcpos) in each breached VAMP MID's cells.

    A near-zero vpsum is the near-empty cell that makes ∂VAMP/∂share (∝ 1/vpsum) blow up (the 1,822
    entries the conditioning guard had to zero). Reports the distribution of vpsum across each breached
    VAMP MID's cells and how many are near-zero. Returns log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        pr = _s2pr(np.asarray(split, float)[None, :], incidence)[0]
        base_vals = model.spec_values(pr)
        _v, _t, inter = model._forward_pr(pr)
        vpsum_cell = np.zeros(model.ngc); vpsum_cell[model.gcode] = inter["vpsum"]
        pk = list(getattr(model.pj, "prop_keys", []))
        prop_mid = np.array([str(pk[j]).split("|")[-1].strip().lower() if j < len(pk) else ""
                             for j in range(max(len(pk), 1))])
        row_mid = prop_mid[model.propidx] if model.propidx.size else np.array([], dtype=object)
        vpos = (model.vcpos > 0.5) & (~model.mask)
        out = ["   ── VPSUM (VAMP denominator per cell; near-zero ⇒ the 1/vpsum gradient blow-up) ──"]
        shown = 0
        for i, sp in enumerate(model.specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            m = str(sp.label).strip().lower()
            rows_m = np.where((row_mid == m) & vpos)[0] if row_mid.size else np.array([], int)
            if rows_m.size == 0:
                continue
            cells_m = np.unique(model.gcode[rows_m])
            vp = vpsum_cell[cells_m]
            n_nz = int((vp < float(near_zero)).sum())
            out.append(f"      • {sp.label} [vamp]: {cells_m.size:,} cells · vpsum min {vp.min():.3g} "
                       f"p50 {np.median(vp):.3g} max {vp.max():.3g} · {n_nz:,} below {near_zero:g} "
                       "(near-empty → gradient blow-up)")
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   vpsum check skipped ({type(exc).__name__}: {exc})."]


# [FN-399]
def usable_recipient_report(split, exact_bands, incidence, *, max_list=15):
    """READ-ONLY (#5): cells with a USABLE VAMP recipient = co-located, VAMP-positive, live, and its
    own MID has ceiling headroom (or no VAMP cap). This is the single decisive "is there any legal
    move" count — the intersection of the co-location, VAMP-positive and headroom conditions. Returns
    log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        rep = exact_bands.report(_s2pr(np.asarray(split, float)[None, :], incidence)[0])
        base_vals = model.spec_values(_s2pr(np.asarray(split, float)[None, :], incidence)[0])
        # per-MID VAMP-ceiling headroom (name -> headroom; absent ⇒ no VAMP cap ⇒ unlimited)
        head = {}
        for rr in rep:
            if rr.get("ceil") is not None and str(rr.get("metric")) == "vamp":
                head[str(rr["midl"]).strip().lower()] = float(rr["ceil"]) - float(rr["now"])

        def _ok(name):                                    # sibling can absorb VAMP without breaching
            h = head.get(name)
            return (h is None) or (h > 1e-6)
        pk = list(getattr(model.pj, "prop_keys", []))
        prop_mid = np.array([str(pk[j]).split("|")[-1].strip().lower() if j < len(pk) else ""
                             for j in range(max(len(pk), 1))])
        row_mid = prop_mid[model.propidx] if model.propidx.size else np.array([], dtype=object)
        vpos = (model.vcpos > 0.5) & (~model.mask)
        sib_ok = np.array([_ok(x) for x in row_mid]) if row_mid.size else np.array([], bool)
        usable = vpos & sib_ok                            # VAMP-positive + live + headroom-ok row
        ncell = model.ngc
        out = ["   ── USABLE RECIPIENT (co-located + VAMP-positive + eligible + headroom — the only "
               "cells with a LEGAL VAMP move) ──"]
        shown = 0
        for i, sp in enumerate(model.specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            m = str(sp.label).strip().lower()
            rows_m = np.where((row_mid == m) & vpos)[0] if row_mid.size else np.array([], int)
            if rows_m.size == 0:
                continue
            cells_m = np.unique(model.gcode[rows_m])
            # usable recipients of a DIFFERENT MID (the breached MID itself is not usable)
            other_usable = usable & (row_mid != m)
            ouc = (np.bincount(model.gcode[other_usable], minlength=ncell)
                   if other_usable.any() else np.zeros(ncell, int))
            has = ouc[cells_m] > 0
            n_with = int(has.sum())
            out.append(f"      • {sp.label} [vamp]: {n_with:,}/{cells_m.size:,} of its VAMP cells have a "
                       f"USABLE recipient (VAMP-positive + eligible + headroom)"
                       + ("  → NO legal move anywhere (structural)" if n_with == 0 else ""))
            shown += 1
            if shown >= int(max_list):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        out.append("      (read-only. This intersects co-location + VAMP-positive + eligibility + "
                   "headroom — the count of cells where a compliant reroute is actually legal. 0 ⇒ "
                   "genuinely structural; >0 ⇒ a legal move exists there.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   usable-recipient check skipped ({type(exc).__name__}: {exc})."]


# [FN-400]
def breach_concentration_report(split, exact_bands, incidence, *, top=10, max_mids=6):
    """READ-ONLY: WHERE does a breached VAMP MID's excess come from, and do THOSE cells have a move?

    Since most of a breached MID's cells have vpsum≈0 (≈0 VAMP), the breach concentrates in a minority
    of real-VAMP cells. This ranks the MID's projector cells by their actual VAMP contribution, then —
    greedily taking the highest-VAMP cells until their cumulative VAMP covers the overshoot — reports
    how many of those cells have a USABLE recipient (co-located, VAMP-positive, eligible, headroom).
    Many ⇒ reachable (search failure); few ⇒ the high-VAMP cells are sole-VAMP (structural — a real
    recipient must be made eligible there). Returns log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        pr = _s2pr(np.asarray(split, float)[None, :], incidence)[0]
        v, t, inter = model._forward_pr(pr)
        specs = model.specs
        base_vals = np.zeros(len(specs))
        for i, sp in enumerate(specs):
            if len(sp.cols):
                base_vals[i] = float((t if sp.metric == "txn" else v)[sp.cols].sum())
        # per aged-row VAMP contribution + its origin projector cell
        mv = inter["mv"]; vshare = inter["vshare"]
        o = model.pc_org; ok = o >= 0; oi = np.where(ok, o, 0)
        move = np.where(ok, mv[oi], 0.0); psh = np.where(ok, vshare[oi], 0.0)
        vp = model.pc_vc * (1.0 - move) + model.pc_pool * psh
        ocell = np.where(ok, model.gcode[oi], -1)
        pk = list(getattr(model.pj, "prop_keys", []))

        def _lab(orow):
            j = int(orow)
            k = str(pk[model.propidx[j]]) if (model.propidx.size and model.propidx[j] < len(pk)) else ""
            return "|".join(k.split("|")[:-1])
        # usable recipient rows (VAMP-positive + live + headroom-ok), per the usable-recipient rule
        rep = exact_bands.report(pr)
        head = {}
        for rr in rep:
            if rr.get("ceil") is not None and str(rr.get("metric")) == "vamp":
                head[str(rr["midl"]).strip().lower()] = float(rr["ceil"]) - float(rr["now"])
        prop_mid = np.array([str(pk[j]).split("|")[-1].strip().lower() if j < len(pk) else ""
                             for j in range(max(len(pk), 1))])
        row_mid = prop_mid[model.propidx] if model.propidx.size else np.array([], dtype=object)
        vpos = (model.vcpos > 0.5) & (~model.mask)

        def _ok(n):
            h = head.get(n); return (h is None) or (h > 1e-6)
        sib_ok = np.array([_ok(x) for x in row_mid]) if row_mid.size else np.array([], bool)
        usable = vpos & sib_ok
        ncell = model.ngc
        out = ["   ── BREACH CONCENTRATION (which cells produce the breach VAMP, and do THOSE cells have "
               "a usable recipient?) ──"]
        shown = 0
        for i, sp in enumerate(specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            m = str(sp.label).strip().lower()
            in_band = np.isin(model.pc_bandcol, sp.cols) if model.pc_bandcol.size else np.zeros(0, bool)
            if not in_band.any():
                continue
            vpm = vp[in_band]; cm = ocell[in_band]; om = oi[in_band]
            cells, inv = np.unique(cm, return_inverse=True)
            contrib = np.bincount(inv, weights=vpm)
            order = np.argsort(-contrib)
            tot = float(contrib.sum()); over = float(base_vals[i] - float(cl))
            lab = {}
            for _c, _or in zip(cm, om):
                ic = int(_c)
                if ic not in lab:
                    lab[ic] = "no-origin (irreducible)" if ic < 0 else _lab(_or)
            other_usable = usable & (row_mid != m)
            ouc = (np.bincount(model.gcode[other_usable], minlength=ncell)
                   if other_usable.any() else np.zeros(ncell, int))
            # greedily take highest-VAMP cells until cumulative ≥ overshoot
            cum = 0.0; n_need = 0; n_need_usable = 0
            for k in order:
                if cum >= over:
                    break
                c = int(cells[k]); cum += float(contrib[k]); n_need += 1
                if c >= 0 and ouc[c] > 0:
                    n_need_usable += 1
            # concentration: #cells holding 90% of VAMP
            cum90 = 0.0; n90 = 0
            for k in order:
                if cum90 >= 0.9 * tot:
                    break
                cum90 += float(contrib[k]); n90 += 1
            out.append(f"      • {sp.label} [vamp]: total {tot:,.0f} · ceil {float(cl):,.0f} · over "
                       f"{over:,.0f}; 90% of VAMP in {n90} of {cells.size:,} cells; to shed the overshoot "
                       f"need ~{n_need} top cell(s), {n_need_usable}/{n_need} have a usable recipient")
            for k in order[:int(top)]:
                c = int(cells[k]); has = (c >= 0 and ouc[c] > 0)
                out.append(f"          cell {c} [{lab.get(c, '')}] VAMP {float(contrib[k]):,.1f}"
                           + ("  · usable recipient ✓" if has else "  · NO usable recipient here"))
            shown += 1
            if shown >= int(max_mids):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        out.append("      (read-only. If the top cells needed to shed the overshoot mostly HAVE a usable "
                   "recipient ⇒ reachable → a search/decode fix helps; if they're mostly sole-VAMP ⇒ "
                   "structural — a real VAMP recipient must be made eligible in those specific cells.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   breach-concentration check skipped ({type(exc).__name__}: {exc})."]


# [FN-401]
def scoped_frozen_report(split, exact_bands, incidence, *, scoped_rpgts, max_mids=6):
    """READ-ONLY: split each breached VAMP MID's M5 VAMP into what the engine CAN vs CANNOT move.

    Buckets each MID's projected VAMP by the RPGT of its origin cell:
      * scoped-movable  — scoped-RPGT cells, the POOL part (redistributed by share) → the engine can
                          route this toward 0 (onto sibling MIDs);
      * scoped-held     — scoped-RPGT cells, the (1−mv) baseline part → NOT reducible by routing;
      * frozen          — UNscoped-RPGT cells (held at baseline) → the engine can't touch;
      * no-origin       — aged rows with no in-window origin → irreducible.
    True engine-reachable minimum = frozen + no-origin + scoped-held (drive scoped-movable → 0).
      * reachable-min < ceiling ⇒ the SCOPED movable VAMP is enough to comply → reachable, so a stuck
        solver/decode is the cause (NOT scope);
      * reachable-min ≥ ceiling ⇒ the scoped RPGTs alone can't reach the cap → widen scope / change cap.
    Requires the by-RPGT projector grain (prop-keys cur|bin|rpgt|mid). Returns log lines; never raises."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        pr = _s2pr(np.asarray(split, float)[None, :], incidence)[0]
        v, t, inter = model._forward_pr(pr)
        specs = model.specs
        base_vals = np.zeros(len(specs))
        for i, sp in enumerate(specs):
            if len(sp.cols):
                base_vals[i] = float((t if sp.metric == "txn" else v)[sp.cols].sum())
        mv = inter["mv"]; vshare = inter["vshare"]
        o = model.pc_org; ok = o >= 0; oi = np.where(ok, o, 0)
        move = np.where(ok, mv[oi], 0.0); psh = np.where(ok, vshare[oi], 0.0)
        held_j = model.pc_vc * (1.0 - move)          # routing-invariant part
        moved_j = model.pc_pool * psh                # routing-movable part
        pk = list(getattr(model.pj, "prop_keys", []))
        # RPGT per prop-key (needs the by-RPGT grain: cur|bin|rpgt|mid). Detect field count.
        _nf = len(str(pk[0]).split("|")) if pk else 0
        if _nf < 4:
            return ["   ── SCOPED vs FROZEN VAMP ── skipped: projector is not at by-RPGT grain "
                    f"(prop-key has {_nf} fields, need cur|bin|rpgt|mid) — RPGT split unavailable."]
        # RPGT is field index 2 in BOTH cur|bin|rpgt|mid (by_rpgt) and cur|bin|rpgt|pmp|ctry|mid
        # (by_subcell); using [-2] wrongly grabbed Country at sub-cell grain.
        prop_rpgt = np.array([str(pk[j]).split("|")[2].strip().lower()
                              if (j < len(pk) and len(str(pk[j]).split("|")) >= 4) else ""
                              for j in range(max(len(pk), 1))])
        scoped_set = set(str(r).strip().lower() for r in (scoped_rpgts or []))
        out = ["   ── SCOPED vs FROZEN VAMP (can the SCOPED-RPGT movable VAMP alone cover the "
               "overshoot?) ──"]
        shown = 0
        for i, sp in enumerate(specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            in_band = np.isin(model.pc_bandcol, sp.cols) if model.pc_bandcol.size else np.zeros(0, bool)
            if not in_band.any():
                continue
            idx = np.where(in_band)[0]
            okx = ok[idx]
            rp = prop_rpgt[model.propidx[oi[idx]]]          # RPGT of each aged row's origin
            is_scoped = np.isin(rp, list(scoped_set)) & okx
            is_frozen = (~np.isin(rp, list(scoped_set))) & okx
            is_noorig = ~okx
            h = held_j[idx]; m = moved_j[idx]
            sc_mov = float(m[is_scoped].sum()); sc_held = float(h[is_scoped].sum())
            fr = float((h[is_frozen] + m[is_frozen]).sum())
            noorig = float((h[is_noorig] + m[is_noorig]).sum())
            total = float(base_vals[i]); over = total - float(cl)
            reach_min = fr + noorig + sc_held
            verdict = ("< ceil ⇒ SCOPED movable ALONE can clear it → reachable (a stuck solver/decode is "
                       "the cause, NOT scope)" if reach_min < float(cl) - 1e-6 else
                       "≥ ceil ⇒ scoped movable CANNOT clear it → widen RPGT scope or change the cap")
            out.append(f"      • {sp.label} [vamp]: total {total:,.0f} · ceil {float(cl):,.0f} · over "
                       f"{over:,.0f}")
            out.append(f"          scoped-movable {sc_mov:,.0f} · scoped-held {sc_held:,.0f} · "
                       f"frozen(unscoped) {fr:,.0f} · no-origin {noorig:,.0f}")
            out.append(f"          engine-reachable min (scoped-movable→0) = {reach_min:,.0f} → {verdict}")
            shown += 1
            if shown >= int(max_mids):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        out.append(f"      (read-only. scoped RPGTs = {sorted(scoped_set)}. scoped-movable is the VAMP the "
                   "engine CAN reroute onto sibling MIDs; reachable-min = frozen + no-origin + scoped-held. "
                   "This is the honest, scope-aware version of reachable-min.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   scoped-vs-frozen check skipped ({type(exc).__name__}: {exc})."]


# [FN-402]
def insearch_rpgt_breakdown(split, exact_bands, incidence, *, max_mids=8):
    """READ-ONLY: the IN-SEARCH projector's per-RPGT M5 VAMP for each breached VAMP MID.

    Run on the DELIVERED split, this is directly comparable to tab-3's per-RPGT VAMP_Post table.
    Diffing them RPGT-by-RPGT localises where the in-search projection (what the GA optimises) and the
    deployed tab-3 projection (what actually ships) disagree — e.g. if in-search 'monthly renewal' is
    far below tab-3's, that RPGT is where the ~380 gap lives. Requires the by-RPGT projector grain."""
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        model = ExactBandModel(exact_bands, incidence)
        pr = _s2pr(np.asarray(split, float)[None, :], incidence)[0]
        v, t, inter = model._forward_pr(pr)
        specs = model.specs
        base_vals = np.zeros(len(specs))
        for i, sp in enumerate(specs):
            if len(sp.cols):
                base_vals[i] = float((t if sp.metric == "txn" else v)[sp.cols].sum())
        mv = inter["mv"]; vshare = inter["vshare"]
        o = model.pc_org; ok = o >= 0; oi = np.where(ok, o, 0)
        move = np.where(ok, mv[oi], 0.0); psh = np.where(ok, vshare[oi], 0.0)
        vp = model.pc_vc * (1.0 - move) + model.pc_pool * psh
        pk = list(getattr(model.pj, "prop_keys", []))
        _nf = len(str(pk[0]).split("|")) if pk else 0
        out = ["   ── IN-SEARCH per-RPGT VAMP on the DELIVERED split (diff vs tab-3 VAMP_Post to "
               "localise the projection gap) ──"]
        if _nf < 4:
            out.append("      (projector not at by-RPGT grain — per-RPGT split unavailable)")
            return out
        # RPGT is field index 2 in BOTH cur|bin|rpgt|mid (by_rpgt) and cur|bin|rpgt|pmp|ctry|mid
        # (by_subcell); using [-2] wrongly grabbed Country at sub-cell grain.
        prop_rpgt = np.array([str(pk[j]).split("|")[2].strip().lower()
                              if (j < len(pk) and len(str(pk[j]).split("|")) >= 4) else ""
                              for j in range(max(len(pk), 1))])
        rp_row = np.where(ok, prop_rpgt[model.propidx[oi]], "no-origin")
        shown = 0
        for i, sp in enumerate(specs):
            if sp.metric != "vamp":
                continue
            cl = sp.ceil
            if cl is None or float(cl) <= 0 or base_vals[i] <= float(cl) + 1e-6:
                continue
            in_band = np.isin(model.pc_bandcol, sp.cols) if model.pc_bandcol.size else np.zeros(0, bool)
            if not in_band.any():
                continue
            rpb = rp_row[in_band]; vpb = vp[in_band]
            labels, inv = np.unique(rpb, return_inverse=True)
            sums = np.bincount(inv, weights=vpb)
            order = np.argsort(-sums)
            parts = "; ".join(f"{labels[k]} {sums[k]:,.0f}" for k in order)
            out.append(f"      • {sp.label} [vamp] in-search total {float(sums.sum()):,.0f} "
                       f"(ceil {float(cl):,.0f}): {parts}")
            shown += 1
            if shown >= int(max_mids):
                break
        if shown == 0:
            out.append("      (no breached VAMP ceiling bands at this split.)")
        out.append("      (read-only. Compare each RPGT to tab-3's VAMP_Post for the same MID; the RPGT "
                   "with the biggest in-search-vs-tab-3 difference is where the two projections diverge — "
                   "the source of the scored-vs-delivered gap.)")
        return out
    except Exception as exc:  # noqa: BLE001
        return [f"   in-search per-RPGT breakdown skipped ({type(exc).__name__}: {exc})."]


# [FN-390]
def solve_global_linear_lp(exact_bands, incidence, base_shares, cell_starts, cell_counts, elig,
                           *, max_share=1.0, weighted=True, log_fn=None,
                           floor_mask=None, share_floor=0.0):
    """ONE global MINIMAL-MOVE feasibility projection → a warm-start SEED candidate for the GA.

    It linearises every banded spec ONCE at ``base_shares`` (the band-aware seed) using the EXACT
    projector value + analytic Jacobian (``ExactBandModel``), then solves one convex LP over ALL cells
    and ALL movable/eligible gateways at once:

        minimise  W_SLACK · Σ slack  +  MU · Σ |Δs|     (W_SLACK ≫ MU)
        s.t.      per-cell Σ Δs = 0,  0 ≤ base+Δs ≤ max_share,
                  linearised band caps (slack ≥ 0 absorbs any residual breach)

    Because W_SLACK ≫ MU the caps are satisfied WHENEVER linearly possible (slack → 0), and among all
    cap-satisfying splits it picks the one that MOVES THE LEAST share from the seed — the "nearest
    feasible split". This shaves each breached MID onto whatever eligible siblings have room in each
    cell, derived purely from the constraint system (no bespoke "move-to-sibling" operator). Keeping
    the move minimal also keeps the linearisation valid near the seed, so the re-projected TRUE breach
    tracks the linear one (unlike the earlier "minimise breach, unbounded move" objective, whose full
    step overshot into a region where the linear model was wrong).

    Returned UNCONDITIONALLY as a SEED CANDIDATE — the never-worse guarantee downstream re-scores it on
    the TRUE projector and keeps it only if its exact breach beats the other seeds. Never raises.

    Returns (shares[N], info) with info['breach0'] (exact breach at base), info['breach_true'] (exact
    breach of the returned candidate), and info['moved'] (total |Δs| = share reallocated). ``weighted``
    is retained for API compatibility but no longer affects the objective (the projection is
    move-minimising, not breach-weighted).
    """
    info = {"ok": False, "build": __build__, "reason": "", "n_free": 0, "n_bands": 0,
            "breach0": float("nan"), "breach_true": float("nan")}
    try:
        s0 = np.asarray(base_shares, float).copy()
        N = s0.shape[0]
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        elig = np.asarray(elig, float)
        cap = float(max_share) if (max_share and float(max_share) > 0) else 1.0
        # Catch-all ε-floor (see solve_least_breach / floor_catchall_shares): pin catch-all gateways
        # ≥ share_floor on the base and on the re-projected candidate, so the exact breach the LP
        # optimises against is the DEPLOYED breach (no re-add can fire inside the floored region).
        _fmask = ((np.asarray(floor_mask, bool) & (elig > 0.5))
                  if (floor_mask is not None and float(share_floor) > 0) else None)
        s0 = floor_catchall_shares(s0, _fmask, share_floor, cs, cc)
        model = ExactBandModel(exact_bands, incidence, vps_eps=_VPS_EPS)  # gradient-only reg
        # Report the UNWEIGHTED breach (Σ(now/ceil−1)₊ + (1−now/floor)₊): feasibility is
        # weight-independent and this stays finite even if a band weight is NaN (e.g. a MID with
        # zero baseline volume). The `weighted` flag still steers the LP OBJECTIVE below.
        b0 = model.breach(s0, weighted=False)
        info["breach0"] = b0
        if b0 <= 1e-12:
            info.update(ok=True, breach_true=b0, reason="base already compliant")
            return s0, info

        # Exact value + gradient at the base split (single linearisation).
        vals, J = model.spec_jacobian_shares(s0)
        # HiGHS rejects a model outright ("Model error") if ANY coefficient is non-finite. The
        # linear-FRACTIONAL VAMP Jacobian can produce NaN/Inf on a degenerate cell (e.g. vpsum≈0),
        # so scrub them to 0 (a non-finite sensitivity means "this gateway can't reliably move this
        # band" → treat it as immovable for this band). Same for the band values feeding the RHS.
        _nfJ = int((~np.isfinite(J)).sum())
        _nfv = int((~np.isfinite(vals)).sum())
        if _nfJ or _nfv:
            info["nonfinite_jac"] = _nfJ
            info["nonfinite_vals"] = _nfv
            J = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        # DEGENERATE-SENSITIVITY GUARD: the linear-FRACTIONAL VAMP gradient is ∝ 1/vpsum, so on a
        # cell whose VAMP-positive volume is ~0 it can be astronomically large but FINITE (e.g. 1e19
        # — "shift 1e-19 of share to swing the whole band"). Such a direction is a vanishing-
        # denominator artifact, not a usable move, and any coefficient above ~1e15 makes HiGHS reject
        # the model ("Model error"). Legitimate band gradients are volume-scale (≲ 1e6), so we zero
        # entries beyond a safe threshold: that gateway is simply treated as immovable for that band
        # (well-conditioned gateways in the same cells still provide the reallocation).
        _JCLIP = 1e9
        _big = np.abs(J) > _JCLIP
        _nbig = int(_big.sum())
        if _nbig:
            info["jac_clipped"] = _nbig
            J = np.where(_big, 0.0, J)
        # A cell is "band-feeding" if ANY of its gateways has a nonzero band gradient at the seed.
        # We free EVERY eligible gateway in those cells — not just the band-feeders — because
        # reallocating a band-feeder's share REQUIRES a same-cell sibling to absorb it, and the
        # per-cell equality (Σ Δs = 0) would otherwise pin a lone free gateway to zero movement.
        # (At a corner seed a band-feeder's OWN gradient can vanish — ∂/∂s = 0 when a sibling sits
        # at 0 — so a nonzero-gradient-only free set can miss the feeder entirely.) Cells with no
        # band involvement stay fixed (Δs = 0), keeping the step minimal.
        cell_id = np.repeat(np.arange(len(cs)), cc)
        feed = np.abs(J).sum(axis=0) > 0
        feed_cells = np.unique(cell_id[feed]) if feed.any() else np.zeros(0, np.int64)
        free = np.isin(cell_id, feed_cells) & (elig > 0.5)
        info["n_free"] = int(free.sum())
        if not free.any():
            info.update(reason="no band-feeding gateways are eligible/movable", breach_true=b0)
            return s0, info

        # Enumerate the active ceiling/floor sides (one linearised band constraint each).
        rows_meta = []
        for i, sp in enumerate(model.specs):
            if len(sp.cols) == 0:
                continue
            if sp.ceil is not None and sp.ceil > 0:
                rows_meta.append(("ceil", i, sp.ceil, vals[i], J[i]))
            if sp.floor is not None and sp.floor > 0:
                rows_meta.append(("floor", i, sp.floor, vals[i], J[i]))
        n_slack = len(rows_meta)
        info["n_bands"] = n_slack
        if n_slack == 0:
            info.update(ok=True, breach_true=b0, reason="no active bands")
            return s0, info
        fidx = np.where(free)[0]
        F = int(fidx.size)
        if log_fn:
            log_fn(f"      global LP: MINIMAL-MOVE feasibility projection — linearising {n_slack} band "
                   f"side(s) at the seed (exact breach {b0:.4g}) and minimising total share MOVED subject "
                   f"to the caps, over {F:,} movable gateways of {N:,}…")
            if info.get("jac_clipped") or info.get("nonfinite_jac"):
                log_fn(f"      global LP: conditioning guard — zeroed {info.get('jac_clipped', 0)} "
                       f"degenerate (|∂band/∂share|>{_JCLIP:.0g}) and {info.get('nonfinite_jac', 0)} "
                       "non-finite gradient entries (near-empty-cell artifacts; those gateways held "
                       "for those bands).")

        # ── LP variables: [ Δs (N) | slack (n_slack) | t (F) ] ──────────────────────────────────
        # slack_k ≥ 0 absorbs any residual breach of band k (so the LP is ALWAYS feasible — no brittle
        # hard-cap infeasibility); t_a ≥ |Δs_fidx[a]| linearises the L1 move size. The objective is
        #     minimise  W_SLACK · Σ slack  +  MU · Σ t          (W_SLACK ≫ MU)
        # so the caps are satisfied WHENEVER linearly possible (slack driven to 0), and among ALL
        # cap-satisfying splits the one with the SMALLEST TOTAL MOVE from the seed is chosen. Keeping
        # the move minimal also keeps the linearisation valid near the seed, so the re-projected TRUE
        # breach tracks the linear one — the fix for the earlier "minimise breach, unbounded move"
        # objective, whose full step overshot into a region where the linear model was wrong.
        W_SLACK = 1e7
        MU = 1.0
        nvar = N + n_slack + F
        c_obj = np.zeros(nvar)
        c_obj[N:N + n_slack] = W_SLACK
        c_obj[N + n_slack:] = MU
        # A_ub triplets: band rows (0..n_slack-1) then two L1 rows per free var.
        _ui, _uj, _ud, _b = [], [], [], []
        for k, (side, i, lim, vi, gi) in enumerate(rows_meta):
            nz = np.nonzero(gi)[0]
            sgn = 1.0 if side == "ceil" else -1.0          # ceil: gi·Δs − lim·slack ≤ lim − vi
            for j in nz:                                    # floor: −gi·Δs − lim·slack ≤ vi − lim
                _ui.append(k); _uj.append(int(j)); _ud.append(sgn * float(gi[j]))
            _ui.append(k); _uj.append(N + k); _ud.append(-float(lim))
            _b.append(float(lim - vi) if side == "ceil" else float(vi - lim))
        # L1 rows (vectorised):  Δs_j − t_a ≤ 0   and   −Δs_j − t_a ≤ 0   ⇒   t_a ≥ |Δs_j|
        _a = np.arange(F)
        _tcol = N + n_slack + _a
        _rpos = n_slack + 2 * _a
        _rneg = n_slack + 2 * _a + 1
        _ui = np.concatenate([np.asarray(_ui, np.int64), _rpos, _rpos, _rneg, _rneg])
        _uj = np.concatenate([np.asarray(_uj, np.int64), fidx, _tcol, fidx, _tcol])
        _ud = np.concatenate([np.asarray(_ud, float),
                              np.ones(F), -np.ones(F), -np.ones(F), -np.ones(F)])
        n_ub = n_slack + 2 * F
        A_ub = _sparse.coo_matrix((_ud, (_ui, _uj)), shape=(n_ub, nvar)).tocsr()
        A_ub.data = np.nan_to_num(A_ub.data, nan=0.0, posinf=0.0, neginf=0.0)
        b_ub = np.nan_to_num(np.concatenate([np.asarray(_b, float), np.zeros(2 * F)]),
                             nan=0.0, posinf=0.0, neginf=0.0)
        c_obj = np.nan_to_num(c_obj, nan=1.0, posinf=0.0, neginf=0.0)
        # equality: Σ_free Δs = 0 per cell (each cell stays summed to its reference total)
        n_cell = len(cs)
        Aeq = _sparse.coo_matrix((np.ones(F), (cell_id[fidx], fidx)),
                                 shape=(n_cell, nvar)).tocsr()
        beq = np.zeros(n_cell)
        # bounds: Δs box for free (0 for pinned rows); slack ≥ 0; t ≥ 0. Guard lb ≤ ub & finiteness.
        lb = np.zeros(nvar); ub = np.zeros(nvar)
        lb[fidx] = -s0[fidx]
        ub[fidx] = np.maximum(cap - s0[fidx], -s0[fidx])
        lb[:N] = np.nan_to_num(lb[:N], nan=0.0, posinf=0.0, neginf=0.0)
        ub[:N] = np.nan_to_num(ub[:N], nan=0.0, posinf=0.0, neginf=0.0)
        bounds = [((lb[j], ub[j]) if j < N else (0.0, None)) for j in range(nvar)]
        res = _linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                       bounds=bounds, method="highs")
        if not res.success:
            # Precise diagnostics so a HiGHS 'Model error' points at the offending piece.
            _diag = (f"c_obj[nonfinite={int((~np.isfinite(c_obj)).sum())}, "
                     f"max={float(np.nanmax(np.abs(c_obj))):.3g}] "
                     f"A_ub[nnz={A_ub.nnz}, nonfinite={int((~np.isfinite(A_ub.data)).sum())}, "
                     f"max={float(np.nanmax(np.abs(A_ub.data)) if A_ub.nnz else 0):.3g}] "
                     f"b_ub[nonfinite={int((~np.isfinite(b_ub)).sum())}] "
                     f"bounds[lb>ub={int((lb[:N] > ub[:N]).sum())}] "
                     f"jac_scrubbed={info.get('nonfinite_jac', 0)}, "
                     f"jac_clipped={info.get('jac_clipped', 0)}")
            info["reason"] = f"LP failed: {getattr(res, 'message', '')}; {_diag}"
            info["breach_true"] = b0
            return s0, info
        ds = res.x[:N]
        info["moved"] = float(np.abs(ds).sum())
        cand = _project_capped_simplex_cells(s0 + ds, cs, cc, elig, cap, budget=1.0)
        cand = floor_catchall_shares(cand, _fmask, share_floor, cs, cc)  # keep catch-all ≥ floor
        bt = model.breach(cand, weighted=False)
        info.update(ok=True, breach_true=bt)
        if log_fn:
            log_fn(f"      global LP: solved · exact breach {b0:.4g} → {bt:.4g} on the re-projected "
                   f"candidate; total share moved {info['moved']:.4g} (minimal). Kept as a seed only "
                   "if it beats the other seeds.")
        return cand, info
    except Exception as exc:  # noqa: BLE001 - a seed must never crash the run
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return np.asarray(base_shares, float).copy(), info


# [FN-396b]
def _tmove_cost(cost, secs, log_fn, *, fastls):
    """[tmove-cost] — seconds beside the projection counts, and the occupancy of the move batches.

    The counts were already printed; without time next to them they could not rank this stage
    against the two before it in the seed chain. The occupancy line is here to decide ONE specific
    idea and to stop it being decided from intuition: the fast line-search projects `delta`, a
    vector that is zero everywhere except one breached MID's rows and their recipients, through
    `incidence @ delta`. A sparse matrix times a DENSE vector costs O(nnz(incidence)) no matter how
    empty the vector is, so a column-restricted product would be cheaper in proportion to the
    occupancy — which is therefore worth measuring rather than eyeballing.

    NOT a budget: the seconds here are the projections only. The stage also spends time in the
    per-cell Python loops that BUILD each batch, and that remainder is stated rather than left as a
    silent gap."""
    try:
        _mv_s = float(cost.get("mv_s", 0.0))
        _pen_s = float(cost.get("pen_s", 0.0))
        _tot = float(secs)
        _oth = _tot - _mv_s - _pen_s
        log_fn(f"      [tmove-cost] stage {_tot:.1f}s total \u2014 "
               f"{_mv_s:.1f}s ({_mv_s / max(_tot, 1e-9):.0%}) in {int(cost.get('mv', 0)):,} sparse "
               f"matvec(s) at {1000.0 * _mv_s / max(int(cost.get('mv', 0)), 1):.1f} ms each; "
               f"{_pen_s:.1f}s ({_pen_s / max(_tot, 1e-9):.0%}) in {int(cost.get('pen', 0)):,} "
               f"penalty pass(es); {_oth:.1f}s ({_oth / max(_tot, 1e-9):.0%}) elsewhere \u2014 the "
               "per-cell Python loops that BUILD each batch, which no counter here covers.")
        _n = int(cost.get("occ_n", 0))
        if _n and fastls:
            _mean = float(cost.get("occ_sum", 0.0)) / _n
            _rows = int(cost.get("N", 0))
            log_fn(f"         MOVE-BATCH OCCUPANCY: the {_n:,} line-search projection(s) were of a "
                   f"vector {_mean:.2%} non-zero on average (worst {float(cost.get('occ_max', 0.0)):.2%}) "
                   f"over {_rows:,} rows. `incidence @ delta` costs the same whatever that number "
                   f"is, because the vector is dense in memory; restricting the product to delta's "
                   f"non-zero columns would scale with it. At {_mean:.2%} that is the whole of "
                   f"those {_mv_s * _n / max(int(cost.get('mv', 0)), 1):.1f}s minus a small "
                   "remainder \u2014 but it is a CHANGE TO THE ARITHMETIC, so it needs a "
                   "bit-identity check against this path before it can be believed, not an "
                   "argument that dropping zero terms cannot matter.")
        elif _n:
            log_fn("         MOVE-BATCH OCCUPANCY: not measured \u2014 the fast line-search is off, "
                   "so no batch is projected on its own.")
    except Exception as _exc:  # noqa: BLE001 — a report must never break a seed
        try:
            log_fn(f"      [tmove-cost] skipped ({type(_exc).__name__}: {_exc}). The seed itself is "
                   "unaffected.")
        except Exception:  # noqa: BLE001
            pass


# [FN-396]
def solve_targeted_moves(exact_bands, incidence, base_shares, cell_starts, cell_counts, elig,
                         *, mid_id, risk, cell_vol, mid_names, max_share=1.0,
                         movable_frac=0.8, log_fn=None, deliver_fn=None):
    """TARGETED move operator → a WARM-START SEED that directly clears breached CEILINGS.

    For every breached ceiling MID it sheds that MID's share, cell by cell (highest contribution to
    whichever of its OWN metrics is worst over), onto co-located ELIGIBLE sibling gateways — but ONLY
    onto MIDs that have room under EVERY ceiling they hold, preferring the recipients with the most
    BINDING SHARE-CAPACITY. A running per-recipient PER-METRIC budget stops it from over-filling any
    one sibling MID into a NEW breach (the whack-a-mole that made the earlier version relocate
    breaches instead of removing them).

    RECIPIENT HEADROOM IS PER (MID, METRIC) as of 2026-08-19be, and this is a BEHAVIOUR CHANGE — it
    moves shares. Before 19be every ceiling, VAMP or TXN, was written into one per-MID slot (last
    spec wins), `report()`'s one-row-per-SPEC was collapsed by midl the same way, and the running
    budget was debited in VAMP units whichever metric the surviving ceiling belonged to. Since risk
    is ~1e-2, a TXN ceiling debited at cell_vol×risk read roughly a hundred times the room it had.
    That is how a VAMP shed onto the txn-only WoodForest (23,961 against a 24,000 txn ceiling) was
    allowed to continue: the RAW line-search saw it 39 under, delivery added ~115, and it landed 14
    OVER — the band [seed-basis] flagged as appearing only on the delivered side. Now each ceiling
    keeps its own budget in its own units, a share increment is converted with that metric's own
    density (TXN = cell_vol × movable_frac; VAMP = that × the row's risk), and a recipient must have
    room under all of them. `ROUTING_TMOVE_ALLBANDS=0` ignores recipients' TXN ceilings — which is
    what the pre-19be DOCSTRING claimed the code did.

    STILL CEILINGS ONLY. Floors are not anticipated by the greedy proposal (the line-search's
    penalty does cover them, so a floor-breaking batch is rejected, not shipped).

    DELIVERY-AWARE SINCE 19go. Pass `deliver_fn` (the blocked-caps → eligibility → cap transform)
    and every projection in this function — the ceiling report, the recipient headroom re-projection
    and the line-search accept test — reads the DELIVERED split. That FORCES the fast line-search
    OFF: it exists only because `shares_to_prop_raw` is linear, so `s2pr(s + f·δ) == s2pr(s) +
    f·s2pr(δ)`, and delivery breaks that identity outright (renormalising a cell after zeroing a
    door is not linear in the cell's shares). Four full projections per MID batch instead of one is
    the price of judging the stage on the basis that ships. `deliver_fn=None` restores the pre-19go
    RAW behaviour, fast line-search included, byte for byte (ROUTING_SEED_DELIV=0).

    Each MID's batch of moves is then LINE-SEARCHED against the EXACT projector (factors 1→0.2) and
    only KEPT if the exact TOTAL breach strictly drops — so a move that merely relocates VAMP onto
    another capped MID is rejected outright. Repeats UNTIL A PASS FINDS NO IMPROVING MOVE — there is
    no pass cap (there was a hardcoded 4 until 2026-08-19v, and it was stopping the loop mid-
    improvement while the log claimed exhausted headroom). Recipient headroom is re-projected each
    pass. NEVER-WORSE overall: returns base unchanged if it can't strictly improve.

    If it stalls with ceilings still over, that is rigorous evidence the caps are JOINTLY infeasible by
    routing (the recipient pool lacks the collective headroom under its own ceilings). Never raises.

    Inputs mirror the co-location diagnostic: `mid_id` (per-row MID index into `mid_names`), `risk`
    (per-row VAMP rate), `cell_vol` (per-cell forecast volume), `elig` (per-row eligibility).
    `movable_frac` converts a share increment to approximate METRIC increments for the running
    recipient budgets — TXN share×vol×frac, VAMP share×vol×risk×frac. Approximate on purpose; the
    exact line-search is the guard."""
    info = {"ok": False, "build": __build__, "reason": "", "breach0": float("nan"),
            "breach": float("nan"), "moved": 0.0, "n_moves": 0, "passes": 0, "mids": []}
    try:
        from routing_optimiser.s4_search.band_scoring import shares_to_prop_raw as _s2pr
        s = np.asarray(base_shares, float).copy()
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        elig = np.asarray(elig, float); mid_id = np.asarray(mid_id)
        risk = np.asarray(risk, float); cell_vol = np.asarray(cell_vol, float)
        cell_of = np.repeat(np.arange(len(cs)), cc)
        n_mid = len(mid_names)
        name2idx = {str(n).strip().lower(): i for i, n in enumerate(mid_names)}

        # Projection counters, so "is this efficient?" is answerable with a number. `_nmv` counts
        # SPARSE MATVECS (the dominant cost); `_npen`/`_nrep` count the cheap elementwise passes.
        # 19ck adds SECONDS beside the counts (a count with no time cannot rank a stage) and the
        # OCCUPANCY of the vectors being projected — see [tmove-cost] at the end of the stage.
        _cost = {"mv": 0, "pen": 0, "rep": 0, "mv_s": 0.0, "pen_s": 0.0,
                 "occ_n": 0, "occ_sum": 0.0, "occ_max": 0.0, "N": int(np.asarray(base_shares).size)}
        _t_stage = _time.perf_counter()

        def _pr(_s, _occ=False):
            # 19go: on the DELIVERED basis this is the only place the transform is applied, so
            # `_rep`, `_breach` and the line-search all move basis together and cannot disagree
            # with each other. It is NOT applied to a move BATCH — see `_FASTLS` below, which is
            # forced off in that mode precisely because a batch is a delta and delivery is not
            # linear, so there is no batch to project on its own any more.
            # `_occ` marks the calls that project a MOVE BATCH rather than a whole split. Those are
            # the ones that are nearly all zeros, and the ones a column-restricted product would
            # make cheaper — so their occupancy is recorded rather than assumed.
            if _occ:
                _a = np.asarray(_s)
                _cost["occ_n"] += 1
                _f = float(np.count_nonzero(_a)) / max(_a.size, 1)
                _cost["occ_sum"] += _f
                if _f > _cost["occ_max"]:
                    _cost["occ_max"] = _f
            _t = _time.perf_counter()
            _cost["mv"] += 1
            _out = _s2pr(_s if deliver_fn is None
                         else np.asarray(deliver_fn(np.asarray(_s, float)), float), incidence)
            _cost["mv_s"] += _time.perf_counter() - _t
            return _out

        def _rep(_s):
            # 19be: return report()'s LIST. It is one row per SPEC, and the old midl-keyed dict
            # comprehension silently dropped every row but the last for any MID with more than
            # one band — including every MID with both a VAMP and a TXN ceiling.
            _cost["rep"] += 1
            return list(exact_bands.report(_pr(_s)))

        def _breach_pr(_prop):
            """Breach from an ALREADY-PROJECTED prop_raw — no matvec."""
            _t = _time.perf_counter()
            _cost["pen"] += 1
            _out = float(exact_bands.penalty(_prop)[0])
            _cost["pen_s"] += _time.perf_counter() - _t
            return _out

        def _breach(_s):
            return _breach_pr(_pr(_s))

        # ── CEILINGS PER (MID, METRIC), 2026-08-19be ──────────────────────────────────────
        # Until 19be this was ONE slot per MID:
        #     ceil_by_mid = np.full(n_mid, np.inf)
        #     for sp in exact_bands.specs:
        #         if sp.ceil is not None:
        #             ceil_by_mid[name2idx[sp.midl]] = float(sp.ceil)   # metric IGNORED
        # so a MID with both a VAMP and a TXN ceiling kept whichever spec came LAST and lost the
        # other, `_rep` collapsed report()'s one-row-per-SPEC the same way, and the budget was
        # then debited in VAMP units whatever metric the surviving ceiling belonged to. A txn
        # ceiling of 24,000 debited at cell_vol×risk (risk ~1e-2) reads ~100x the room it has —
        # which is how a VAMP shed onto the txn-only WoodForest (23,961 of 24,000) was allowed to
        # continue until delivery put it 14 OVER.
        _MET_COL = {"vamp": 0, "txn": 1}
        _MET_NAME = ("vamp", "txn")
        # ROUTING_TMOVE_ALLBANDS=0 ignores recipients' TXN ceilings — what the OLD DOCSTRING
        # claimed the code did (ceilings on the shed metric only). It deliberately does NOT
        # restore the old mixed-unit arithmetic; reproducing a unit error as a control is useless.
        _ALLBANDS = _os.environ.get("ROUTING_TMOVE_ALLBANDS", "1") != "0"
        ceil_m = np.full((n_mid, 2), np.inf)
        for sp in exact_bands.specs:
            if sp.ceil is None:
                continue
            _i = name2idx.get(str(sp.midl).strip().lower())
            _j = _MET_COL.get(str(getattr(sp, "metric", "vamp")).strip().lower())
            if _i is None or _j is None:
                continue
            # several specs on the same (MID, metric) — e.g. different month sets — so take the
            # TIGHTEST rather than whichever came last.
            ceil_m[_i, _j] = min(ceil_m[_i, _j], float(sp.ceil))
        if not _ALLBANDS:
            ceil_m[:, _MET_COL["txn"]] = np.inf
        _n_ceil = int(np.isfinite(ceil_m).sum())
        _n_both = int((np.isfinite(ceil_m).sum(axis=1) == 2).sum())

        def _now_m(rep_rows):
            """(n_mid, 2) projected value per (MID, metric).

            `report()` returns ONE ROW PER SPEC, so a MID with several specs on the same metric
            appears several times; take the LARGEST, which is the binding one for a ceiling."""
            out = np.zeros((n_mid, 2))
            for _r in rep_rows:
                _i = name2idx.get(str(_r.get("midl", "")).strip().lower())
                _j = _MET_COL.get(str(_r.get("metric", "vamp")).strip().lower())
                if _i is not None and _j is not None:
                    out[_i, _j] = max(out[_i, _j], float(_r.get("now", 0.0)))
            return out

        # 19go: the fast line-search is a LINEARITY shortcut and delivery is not linear, so on the
        # delivered basis it is not a tuning choice — it is unavailable. Forced, not defaulted, so
        # ROUTING_TMOVE_FASTLS=1 cannot silently re-enable an identity that does not hold.
        _FASTLS = (_os.environ.get("ROUTING_TMOVE_FASTLS", "1") != "0") and deliver_fn is None
        _FASTLS_VERIFY = _os.environ.get("ROUTING_TMOVE_FASTLS_VERIFY", "0") == "1"
        _pr_s = None                      # carried s2pr(s) for the linear line search
        b0 = _breach(s); info["breach0"] = b0; b_cur = b0
        rep0 = _rep(s)
        now0_m = _now_m(rep0)
        # which (MID, metric) ceilings are breached at the start (for the before/after log)
        start_breached = {(i, j) for i in range(n_mid) for j in (0, 1)
                          if np.isfinite(ceil_m[i, j]) and now0_m[i, j] > ceil_m[i, j] + 1e-6}
        if not start_breached:
            info.update(ok=True, breach=b0, reason="base already ceiling-compliant")
            return s, info
        if log_fn:
            log_fn(f"      targeted-move seed: {len(start_breached)} breached ceiling band(s) over "
                   f"{len({_i for _i, _ in start_breached})} MID(s) — shedding onto co-located "
                   f"siblings with room under EVERY ceiling they hold ({_n_ceil} ceiling(s) tracked, "
                   f"{_n_both} MID(s) with both a VAMP and a TXN ceiling; binding-capacity first; "
                   "whack-a-mole guarded by a per-recipient per-metric budget + exact-projector "
                   "line-search; never-worse)."
                   + ("" if _ALLBANDS else " ⚠ ROUTING_TMOVE_ALLBANDS=0: recipients' TXN ceilings "
                      "are being IGNORED, so a VAMP shed can overfill a txn-capped MID."))
        # NO PASS CAP (2026-08-19v). It was a hardcoded 4, and the 2026-08-20 22:22 run used
        # 4 of 4 without hitting `if not progress: break` — still improving when the cap stopped
        # it, while the summary claimed 'recipient headroom exhausted'. The loop now runs until a
        # pass finds no improving move, which is safe because the operator is monotone: every
        # batch is line-searched against the TRUE projector and kept only if strictly improving.
        # REL_FLOOR is not a cap in disguise: the line search accepts improvements > 1e-12, and a
        # 1e-6 RELATIVE floor at a breach of ~0.0094 is ~1e-11 — an order of magnitude above that
        # threshold, so no move the operator considers real is ever discarded by it.
        # RUNAWAY exists only so a pathological case cannot hang a run, and it SHOUTS if hit.
        _REL_FLOOR = 1e-6
        try:
            _RUNAWAY = max(1, int(_os.environ.get('ROUTING_TMOVE_MAXPASS', '5000') or 5000))
        except Exception:  # noqa: BLE001
            _RUNAWAY = 5000
        total_moves = 0; moved_share = 0.0; passes = 0
        stop_reason = 'no-improving-move'   # 'cleared'|'no-improving-move'|'converged'|'runaway'
        _pass = -1
        while True:
            _pass += 1
            if _pass >= _RUNAWAY:
                stop_reason = 'runaway'
                break
            _b_pass_start = b_cur
            rep = _rep(s)
            now_m = _now_m(rep)
            headroom = ceil_m - now_m               # (n_mid, 2); +inf where there is no ceiling
            # a MID is breached if ANY of its ceilings is over
            _over = np.isfinite(ceil_m) & (now_m > ceil_m + 1e-6)
            breached = [i for i in range(n_mid) if _over[i].any()]
            if not breached:
                stop_reason = 'cleared'
                break
            # worst overshoot first, RELATIVE to the ceiling — the metrics are in different
            # units (transactions vs VAMP), so an absolute overshoot cannot order them.
            # Evaluated ONLY on the finite entries: (now - inf) / |inf| is nan, and a nan in a
            # log line or an argmax is a real defect, not cosmetic noise.
            def _rel_over(_i):
                _c = ceil_m[_i]
                _fin = np.isfinite(_c) & _over[_i]
                _r = np.full(_c.shape, -np.inf)
                if _fin.any():
                    _r[_fin] = ((now_m[_i][_fin] - _c[_fin])
                                / np.maximum(np.abs(_c[_fin]), 1.0))
                return _r
            breached.sort(key=lambda i: -float(_rel_over(i).max()))
            progress = False
            passes += 1
            _dirty = False          # has `s` changed since the top-of-pass _rep(s)?
            for m_idx in breached:
                # Refresh headroom ONLY if an earlier MID in this pass actually moved. On the
                # first MID, and after any MID that accepted nothing, `s` is unchanged and this
                # projection returned identical values — one wasted full projection per pass,
                # minimum. Skipping it is bit-identical.
                if _dirty:
                    rep = _rep(s); now_m = _now_m(rep)
                    headroom = ceil_m - now_m
                    _over = np.isfinite(ceil_m) & (now_m > ceil_m + 1e-6)
                    _dirty = False
                if not _over[m_idx].any():
                    continue
                m_rows = np.where((mid_id == m_idx) & (s > 1e-12) & (elig > 0.5))[0]
                if m_rows.size == 0:
                    continue
                # 19be: shed the cells that contribute most to the metric THIS MID is worst over.
                # It was always the VAMP contribution (s×vol×risk), which is the wrong ordering
                # for a MID breaching a TRANSACTION ceiling — there the contribution is s×vol and
                # a low-risk high-volume cell is the one to move.
                _mj = int(np.argmax(_rel_over(m_idx)))
                contrib = s[m_rows] * np.maximum(cell_vol[cell_of[m_rows]], 0.0)
                if _mj == _MET_COL["vamp"]:
                    contrib = contrib * np.maximum(risk[m_rows], 1e-9)
                m_rows = m_rows[np.argsort(-contrib)]
                delta = np.zeros_like(s)
                hbud = headroom.copy()      # running per-(MID, metric) budget for recipients
                for r in m_rows:
                    c = int(cell_of[r]); a = int(cs[c]); n = int(cc[c])
                    seg = np.arange(a, a + n)
                    smid = mid_id[seg]
                    # 19be: a recipient qualifies only if EVERY ceiling it holds still has room.
                    # `.all(axis=1)` is correct for unbounded metrics too: hbud is +inf there.
                    ok_sib = (elig[seg] > 0.5) & (smid != m_idx) & (hbud[smid] > 1e-6).all(axis=1)
                    sib = seg[ok_sib]
                    if sib.size == 0:
                        continue
                    # Rank by BINDING SHARE-CAPACITY: how much share this row can actually accept
                    # before its tightest ceiling stops it, min_j(hbud_j / dens_j). The old key was
                    # the raw headroom number, which is not comparable across metrics — it ranked a
                    # MID with 24,000 transactions of nominal room above one with 300 of VAMP.
                    _dt_all = float(cell_vol[c]) * float(movable_frac)          # TXN per unit share
                    _dv_all = _dt_all * np.maximum(risk[sib], 1e-9)            # VAMP per unit share
                    _hb = hbud[mid_id[sib]]
                    _capv = np.where(np.isfinite(_hb[:, 0]),
                                     _hb[:, 0] / np.maximum(_dv_all, 1e-30), np.inf)
                    _capt = (np.where(np.isfinite(_hb[:, 1]),
                                      _hb[:, 1] / max(_dt_all, 1e-30), np.inf)
                             if _dt_all > 0 else np.full(sib.size, np.inf))
                    _cap = np.minimum(_capv, _capt)
                    sib = sib[np.lexsort((risk[sib], -_cap))]
                    mv = float(s[r])
                    for b_ in sib:
                        if mv <= 1e-12:
                            break
                        smid_b = int(mid_id[b_])
                        room = float(max_share) - float(s[b_] + delta[b_])
                        if room <= 1e-9:
                            continue
                        # 19be: convert the share increment into EACH metric's own units and
                        # respect BOTH budgets. TXN per unit share is cell_vol × movable_frac;
                        # VAMP is that times the row's risk. Debiting a txn ceiling by a VAMP
                        # increment (risk ~1e-2) was reading ~100x the room that existed.
                        _dt = float(cell_vol[c]) * float(movable_frac)
                        _dv = _dt * max(float(risk[b_]), 1e-9)
                        take = min(mv, room)
                        for _j, _d in ((_MET_COL["vamp"], _dv), (_MET_COL["txn"], _dt)):
                            if np.isfinite(hbud[smid_b, _j]) and _d > 0.0:
                                take = min(take, float(hbud[smid_b, _j]) / _d)
                        if take <= 1e-12:
                            continue
                        delta[b_] += take; delta[r] -= take; mv -= take
                        for _j, _d in ((_MET_COL["vamp"], _dv), (_MET_COL["txn"], _dt)):
                            if np.isfinite(hbud[smid_b, _j]):
                                hbud[smid_b, _j] -= take * _d
                if not np.any(np.abs(delta) > 1e-12):
                    continue
                # LINE-SEARCH the batch against the TRUE projector; keep the best strictly
                # improving step. prop_raw = incidence @ shares is LINEAR (band_scoring
                # .shares_to_prop_raw, no renormalisation), so exactly:
                #     s2pr(s + f*delta) == s2pr(s) + f * s2pr(delta)
                # which turns FOUR matvecs per MID into ONE (for delta) — `_pr_s` carries
                # s2pr(s) and is updated incrementally when a step is accepted. Exact in real
                # arithmetic; ~1e-16 relative in floating point, against a 1e-12 accept
                # threshold. ROUTING_TMOVE_FASTLS_VERIFY=1 recomputes the direct value and logs
                # any disagreement; ROUTING_TMOVE_FASTLS=0 restores the direct path.
                best_f, best_b = 0.0, b_cur
                if _FASTLS:
                    if _pr_s is None:
                        _pr_s = _pr(s)
                    _pr_d = _pr(delta, _occ=True)
                    for f in (1.0, 0.66, 0.4, 0.2):
                        bt = _breach_pr(_pr_s + f * _pr_d)
                        if _FASTLS_VERIFY:
                            _bt_direct = _breach(s + f * delta)
                            if abs(bt - _bt_direct) > 1e-9 * max(abs(_bt_direct), 1.0):
                                _vmsg = (f"      targeted-move: ⚠ FAST LINE-SEARCH DISAGREES with "
                                         f"the direct projection (f={f}: {bt:.12g} vs "
                                         f"{_bt_direct:.12g}). The linearity identity does not hold "
                                         f"here — set ROUTING_TMOVE_FASTLS=0 and re-run.")
                                if log_fn:
                                    log_fn(_vmsg)
                                info["fastls_mismatch"] = True
                        if bt < best_b - 1e-12:
                            best_f, best_b = f, bt
                else:
                    _pr_d = None
                    for f in (1.0, 0.66, 0.4, 0.2):
                        bt = _breach(s + f * delta)
                        if bt < best_b - 1e-12:
                            best_f, best_b = f, bt
                if best_f > 0.0:
                    s = s + best_f * delta
                    if _FASTLS and _pr_d is not None:
                        _pr_s = _pr_s + best_f * _pr_d      # keep s2pr(s) in step with s
                    _dirty = True
                    _mv = float(np.abs(best_f * delta).sum()) / 2.0    # share relocated (÷2: +/- counted once)
                    moved_share += _mv; total_moves += int((np.abs(delta) > 1e-12).sum() // 2)
                    b_cur = best_b; progress = True
            if not progress:
                stop_reason = 'no-improving-move'
                break
            if (_b_pass_start - b_cur) < _REL_FLOOR * max(abs(_b_pass_start), 1e-30):
                # Still technically improving, but by less than one part in a million of the
                # current breach. Recorded as its own reason so it is never confused with either
                # a genuine dead end or a cap.
                stop_reason = 'converged'
                break
        info["passes"] = passes
        rep_end = _rep(s); now1_m = _now_m(rep_end)
        for (i, j) in sorted(start_breached,
                             key=lambda t: -((now0_m[t] - ceil_m[t])
                                             / max(abs(ceil_m[t]), 1.0)
                                             if np.isfinite(ceil_m[t]) else -np.inf)):
            nm = mid_names[i]; n0 = now0_m[i, j]; n1 = now1_m[i, j]; cl = ceil_m[i, j]
            info["mids"].append({"midl": str(nm), "metric": _MET_NAME[j], "ceil": float(cl),
                                 "now0": float(n0), "now1": float(n1)})
            if log_fn:
                _v = ("✓ under ceiling" if n1 <= cl + 1e-6 else
                      ("still over — NO IMPROVING MOVE FOUND (a pass found nothing; headroom "
                       "genuinely exhausted for this operator)"
                       if stop_reason == "no-improving-move" else
                       ("still over — CONVERGED (the last pass moved the breach by <1e-6 of "
                        "itself; not a cap, and not unexplored headroom)"
                        if stop_reason == "converged" else
                        f"still over — ⚠ RUNAWAY BACKSTOP ({_RUNAWAY:,} passes) hit while STILL "
                        f"improving; cause NOT established, raise ROUTING_TMOVE_MAXPASS")))
                log_fn(f"         • {nm} [{_MET_NAME[j]}]: {n0:,.0f} → {n1:,.0f} "
                       f"(ceil {cl:,.0f}) · {_v}")
        b1 = _breach(s); info["moved"] = moved_share; info["n_moves"] = total_moves
        if b1 < b0 - 1e-12:
            info.update(ok=True, breach=b1)
            if log_fn:
                _cleared = all(now1_m[t] <= ceil_m[t] + 1e-6 for t in start_breached)
                log_fn(f"      targeted-move seed: exact breach {b0:.4g} → {b1:.4g} in {passes} pass(es) "
                       f"({total_moves:,} cell-moves, {moved_share:.3g} share). "
                       + ("ALL ceilings cleared — a compliant split exists." if _cleared
                          else ("Some ceilings remain and a pass found NO improving move ⇒ this "
                                "operator is exhausted (evidence of joint infeasibility BY THIS "
                                "OPERATOR, not proof of infeasibility). Kept — strictly better on RAW (see log note)."
                                if stop_reason == "no-improving-move" else
                                ("Some ceilings remain; the last pass improved the breach by less "
                                 "than one part in a million, so this is CONVERGED, not capped. "
                                 "Kept — strictly better on RAW (see log note)."
                                 if stop_reason == "converged" else
                                 f"⚠ Some ceilings remain and the loop hit its RUNAWAY BACKSTOP at "
                                 f"{_RUNAWAY:,} passes while STILL IMPROVING. This is NOT "
                                 f"convergence and NOT exhausted headroom — it is a pathological "
                                 f"case. Raise ROUTING_TMOVE_MAXPASS. Kept — strictly better on RAW (see log note)."))))
            if log_fn:
                # 19bd: say WHICH BASIS "better" was measured on. This operator's
                # line-search scores the RAW split; the engine SELECTS seeds on the
                # DELIVERED basis (blocked-caps + eligibility). On 2026-08-23 10:07 it
                # improved RAW by 0.0017 and worsened DELIVERED by 0.0375, having
                # reported itself strictly better. Selection rejected it, so nothing bad
                # shipped — but the claim was false, the stage was wasted, and a reader
                # had no way to tell from this line.
                # 19be then removed the CAUSE of that particular divergence (recipient
                # headroom was one slot per MID, debited in VAMP units whatever metric
                # the ceiling belonged to). The accept test is STILL RAW, so this note
                # stays: it is the honest label for what the line-search measured, not a
                # standing bug report.
                log_fn("      targeted-move seed: NOTE every 'better' above is the RAW "
                       "basis — the only one this operator can see. The engine selects "
                       "on the DELIVERED basis (blocked-caps + eligibility), so the two "
                       "can still move apart. The mechanism that made them disagree on "
                       "WoodForest is GONE as of 19be: recipient headroom is now kept "
                       "per (MID, METRIC), so a txn-only MID no longer reads infinite "
                       "room for a VAMP shed. What remains is delivery's own transform, "
                       "not a unit error here. Read [seed-basis] for both bases and "
                       "[seed-chain] for what shipped.")
            if log_fn:
                log_fn(f"      targeted-move seed: stopped because '{stop_reason}' after "
                       f"{passes:,} pass(es) · projection cost {_cost['mv']:,} sparse matvec(s), "
                       f"{_cost['pen']:,} penalty + {_cost['rep']:,} report pass(es)"
                       + ("  (fast line-search ON — 1 matvec per MID instead of 4)"
                          if _FASTLS else "  (fast line-search OFF — 4 matvecs per MID)"))
                _tmove_cost(_cost, _time.perf_counter() - _t_stage, log_fn, fastls=_FASTLS)
            return s, info
        info.update(ok=False, breach=b0, reason="no exact-breach improvement (breach only relocates)")
        if log_fn:
            log_fn(f"      targeted-move seed: no exact-breach improvement (stayed {b0:.4g}) — every "
                   "reroute only RELOCATES VAMP onto another capped MID ⇒ the caps appear JOINTLY "
                   "infeasible by routing. Returning base (never-worse).")
        return np.asarray(base_shares, float).copy(), info
    except Exception as exc:  # noqa: BLE001 — a seed must never crash the run
        info["reason"] = f"{type(exc).__name__}: {exc}"
        if log_fn:
            log_fn(f"      targeted-move seed skipped: {type(exc).__name__}: {exc}")
        return np.asarray(base_shares, float).copy(), info
