"""EXACT projector-defined band solver — the "fragile hand-derivation".

This is the exact counterpart to the heuristic `genetic_global.band_greedy_shares` seed. Where the
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

import numpy as np

try:
    from scipy.optimize import linprog as _linprog
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    _HAVE_SCIPY = False

__build__ = "2026-08-11-exact-projector-band-solver-slp"


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
    def __init__(self, exact_bands, incidence):
        pj = exact_bands.projector
        self.pj = pj
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
        self.vcpos = np.asarray(pj._vcpos, float)
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
        pr = np.where(self.mask, 0.0, prop_raw[self.propidx])
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

    # ------------------------------------------------------------------ analytic Jacobian
    # [FN-384]
    def _jac_pr(self, inter) -> tuple:
        """Per-band-column Jacobian d(metric[b])/d(pr[q]) as dense (B, nR) arrays (vamp, txn)."""
        psum = inter["psum"]; act = inter["act"]
        pshare = inter["pshare"]; moved_tot = inter["moved_tot"]
        vpsum = inter["vpsum"]; vshare = inter["vshare"]
        gcode = self.gcode; ngc = self.ngc; nR = self.nR
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
        live = ~self.mask
        # scatter-add each row's sensitivity onto its prop-key column
        for i in range(S):
            np.add.at(Jprop[i], self.propidx[live], Jpr_rows[i, live])
        return vals, Jprop

    # [FN-386]
    def spec_jacobian_shares(self, s: np.ndarray):
        """Exact (spec_values[S], d spec_value / d s[S, N]) in SHARE space via the incidence chain."""
        from .band_scoring import shares_to_prop_raw
        prop_raw = shares_to_prop_raw(np.asarray(s, float)[None, :], self.incidence)[0]
        _vals, Jprop = self.spec_jacobian_pr(prop_raw)
        Js = np.asarray(Jprop @ self.incidence)                 # (S, K)·(K, N) = (S, N)
        return _vals, Js

    # ------------------------------------------------------------------ breach
    # [FN-387]
    def breach(self, s: np.ndarray, *, weighted: bool = False) -> float:
        """Total RELATIVE band breach (same definition as band_greedy): Σ(now/ceil−1)_+ + Σ(1−now/floor)_+.
        This is the exact projector breach — the quantity the solver drives to 0."""
        from .band_scoring import shares_to_prop_raw
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
    from .genetic_global import _project_capped_simplex_cells as _p
    return _p(s, cell_starts, cell_counts, elig, cap, budget)


# [FN-389]
def solve_least_breach(exact_bands, incidence, base_shares, cell_starts, cell_counts, elig,
                       *, max_share=1.0, max_outer=40, tol=1e-7, tr_init=0.25, tr_min=1e-4,
                       weighted=False, verbose=False):
    """EXACT successive-LP solve of  min total band breach  s.t. per-cell simplex + max-share.

    Uses `ExactBandModel` (true projector value + analytic Jacobian). Each outer step linearises the
    banded specs at the current split, solves an LP (HiGHS) for a trust-region step that minimises the
    linearised slack, then ACCEPTS the step only if the TRUE (re-projected) breach improves — else the
    trust region shrinks. Returns (shares[N], info). `info['breach0']`/`info['breach']` are the exact
    projector breaches before/after; `info['feasible']` is True iff the final breach ≤ tol (a genuine
    compliance certificate). Never raises: any failure returns the base split with info['ok']=False.

    Only gateways feeding a band move; all others stay at `base_shares` (exact for the band objective).
    Local optimum only (fractional-VAMP nonconvexity + fixed active mask) — see module docstring."""
    info = {"ok": False, "build": __build__, "reason": "", "n_free": 0, "outer": 0,
            "breach0": float("nan"), "breach": float("nan"), "feasible": False}
    try:
        if not _HAVE_SCIPY:
            info["reason"] = "scipy unavailable"
            return np.asarray(base_shares, float).copy(), info
        s = np.asarray(base_shares, float).copy()
        N = s.shape[0]
        cs = np.asarray(cell_starts, np.intp); cc = np.asarray(cell_counts, np.intp)
        elig = np.asarray(elig, float)
        cap = float(max_share) if (max_share and float(max_share) > 0) else 1.0
        model = ExactBandModel(exact_bands, incidence)
        b0 = model.breach(s, weighted=weighted)
        info["breach0"] = b0
        if b0 <= tol:
            info.update(ok=True, feasible=True, breach=b0, reason="base already compliant")
            return s, info

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
        for outer in range(int(max_outer)):
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
            A_ub = np.array(rows); b_ub = np.array(rhs)
            # equality: Σ_free Δs = 0 per cell (keep each cell sum fixed)
            n_cell = len(cs)
            Aeq = np.zeros((n_cell, nvar))
            for n in np.where(free)[0]:
                Aeq[cell_of[n], n] = 1.0
            beq = np.zeros(n_cell)
            # bounds: Δs box (trust region ∩ feasible-share box) for free vars, 0 for non-free; slack ≥ 0
            lb = np.zeros(nvar); ub = np.zeros(nvar)
            fidx = np.where(free)[0]
            lb[fidx] = np.maximum(-tr, 0.0 - best_s[fidx])
            ub[fidx] = np.minimum(tr, cap - best_s[fidx])
            ub[N:] = None                                          # slacks unbounded above
            bounds = [(lb[j], (None if j >= N else ub[j])) for j in range(nvar)]
            res = _linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq,
                           bounds=bounds, method="highs")
            if not res.success:
                tr *= 0.5
                if tr < tr_min:
                    break
                continue
            ds = res.x[:N]
            cand = best_s + ds
            cand = _project_capped_simplex_cells(cand, cs, cc, elig, cap, budget=1.0)
            bc = model.breach(cand, weighted=weighted)
            if bc < best_b - max(1e-12, 1e-4 * best_b):
                best_s = cand; best_b = bc
                info["outer"] = outer + 1
                if verbose:
                    print(f"  [slp] outer {outer}: breach {best_b:.6g} (tr={tr:.3g}, free={info['n_free']})")
                if best_b <= tol:
                    break
                tr = min(tr * 1.5, 0.5)                             # grow on success
            else:
                tr *= 0.5
                if tr < tr_min:
                    break
        info.update(ok=True, breach=best_b, feasible=bool(best_b <= tol))
        return best_s, info
    except Exception as exc:  # noqa: BLE001 - a seed must never crash the run
        info["reason"] = f"{type(exc).__name__}: {exc}"
        return np.asarray(base_shares, float).copy(), info
