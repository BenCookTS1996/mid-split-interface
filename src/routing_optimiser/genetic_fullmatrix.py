"""Full split-table GA + NSGA-II frontier (OPT-IN, additive, SEPARATE from run_midtilt_ga).

These are the two co-worker modes the tilt engine deliberately doesn't have. Both are
SEPARATE entry points: the live tilt GA (`genetic_global.run_midtilt_ga`) and the slider
path are untouched, and nothing here runs unless the caller explicitly calls it.

  run_fullmatrix_ga(ctx, lam, ...):
      Evolves the ENTIRE per-cell share matrix directly — genome = the N shares themselves
      (N = rows = cell x gateway) — instead of the ~2·n_mid tilt knobs. More expressive (any
      split is reachable), at the cost of a much larger, slower search. It REUSES the same
      `_fitness`, the same hard max-share/floor repair (`_cap_floor_shares`) and the same
      `ctx` as the tilt GA, so it returns a drop-in per-cell share vector (N,).

  run_fullmatrix_ga(..., multiobjective=True):
      NSGA-II over TWO objectives — maximise revenue, minimise aggregate expected VAMP count —
      returning the whole Pareto FRONT (the revenue<->risk trade-off) from a SINGLE run, for a
      Tab-4 frontier view. It does NOT replace the slider dial; it is an exploratory view.

Deterministic given `seed`. Requires numpy only (reuses genetic_global helpers).
"""
from __future__ import annotations

import numpy as np

from .genetic_global import _fitness, _cap_floor_shares

__build__ = "2026-07-24-fullmatrix+nsga2"


# --------------------------------------------------------------------------- helpers
def _renorm_cells(X, cs, cc):
    """Renormalise each contiguous cell segment of X (P, N) so every cell sums to 1."""
    seg = np.add.reduceat(X, cs, axis=1)
    seg = np.where(seg > 1e-12, seg, 1.0)
    return X / np.repeat(seg, cc, axis=1)


def _repair(X, cs, cc, elig, cap, floor):
    """Make every row a deployable split: non-negative, eligible-only, per-cell sum 1, then
    the SAME hard max-share cap + exploration floor the tilt GA uses. So the full-matrix GA
    only ever scores splits that could actually ship (no search/output mismatch)."""
    X = np.clip(X, 0.0, None) * elig[None, :]
    X = _renorm_cells(X, cs, cc)
    if cap < 1.0 or floor > 0.0:
        X = _cap_floor_shares(X, cs, cc, elig, float(cap), float(floor))
    return X


def _objectives(pop, ctx):
    """(revenue [maximise], aggregate expected VAMP count [minimise]) per candidate.
    Revenue is the same $-quantity `_fitness` maximises; VAMP count = Σ share·cell_vol·risk,
    a smooth risk measure (no penalty kinks) that gives a clean revenue<->risk frontier."""
    rev = (pop * ctx["rev_coef"][None, :]).sum(axis=1)
    vamp = (pop * (ctx["cell_vol"] * ctx["risk"])[None, :]).sum(axis=1)
    return rev, vamp


# --------------------------------------------------------------------------- NSGA-II core
def _fast_nondominated_sort(F):
    """NSGA-II non-dominated sort. F (P, M) minimisation. Returns list of fronts (each a list
    of member indices), best front first. The pairwise dominance matrix is built with numpy
    broadcasting (vectorised) so this stays fast even for larger populations."""
    P = F.shape[0]
    le = np.all(F[:, None, :] <= F[None, :, :], axis=2)      # le[p,q]: p no worse than q on all
    lt = np.any(F[:, None, :] < F[None, :, :], axis=2)       # lt[p,q]: p strictly better somewhere
    dom = le & lt                                            # dom[p,q]: p dominates q
    np.fill_diagonal(dom, False)
    n = dom.sum(axis=0)                                      # how many dominate q (column sum)
    assigned = np.zeros(P, dtype=bool)
    fronts = [np.where(n == 0)[0].tolist()]
    assigned[fronts[0]] = True
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in np.where(dom[p])[0]:
                n[q] -= 1
                if n[q] == 0 and not assigned[q]:
                    assigned[q] = True
                    nxt.append(int(q))
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def _crowding(F, idxs):
    """Crowding distance for members `idxs` (boundary points = inf)."""
    l = len(idxs)
    dist = np.zeros(l)
    if l <= 2:
        dist[:] = np.inf
        return dist
    F2 = F[idxs]
    for m in range(F.shape[1]):
        o = np.argsort(F2[:, m])
        dist[o[0]] = dist[o[-1]] = np.inf
        span = F2[o[-1], m] - F2[o[0], m]
        if span < 1e-12:
            continue
        for j in range(1, l - 1):
            dist[o[j]] += (F2[o[j + 1], m] - F2[o[j - 1], m]) / span
    return dist


def _nsga2(pop, ctx, rng, cs, cc, elig, cap, floor, generations, mutation_rate, mutation_sigma,
           stop_check=None):
    pop_size, N = pop.shape

    def objs_min(P):
        rev, vamp = _objectives(P, ctx)
        return np.column_stack([-rev, vamp])          # minimise (-revenue, vamp)

    def rank_and_crowd(F):
        fronts = _fast_nondominated_sort(F)
        rank = np.zeros(len(F), dtype=int)
        crowd = np.zeros(len(F))
        for r, fr in enumerate(fronts):
            d = _crowding(F, fr)
            for jj, i in enumerate(fr):
                rank[i] = r
                crowd[i] = d[jj]
        return fronts, rank, crowd

    def breed(P, rank, crowd):
        children = np.empty_like(P)

        def better(i, j):
            if rank[i] != rank[j]:
                return i if rank[i] < rank[j] else j
            return i if crowd[i] >= crowd[j] else j
        for k in range(pop_size):
            i = better(*rng.integers(0, pop_size, size=2))
            j = better(*rng.integers(0, pop_size, size=2))
            a, b = P[i], P[j]
            cellmask = rng.random(len(cc)) < 0.5        # per-cell uniform crossover
            rowmask = np.repeat(cellmask, cc)
            child = np.where(rowmask, a, b).astype(float)
            mm = rng.random(N) < mutation_rate
            child = child + np.where(mm, rng.normal(0.0, mutation_sigma, N), 0.0)
            children[k] = child
        return _repair(children, cs, cc, elig, cap, floor)

    F = objs_min(pop)
    _, rank, crowd = rank_and_crowd(F)
    for _g in range(generations):
        if stop_check is not None and stop_check():
            break
        off = breed(pop, rank, crowd)
        comb = np.vstack([pop, off])
        Fc = objs_min(comb)
        fronts = _fast_nondominated_sort(Fc)
        newidx = []
        for front in fronts:
            if len(newidx) + len(front) <= pop_size:
                newidx += front
            else:
                d = _crowding(Fc, front)
                need = pop_size - len(newidx)
                newidx += [front[t] for t in np.argsort(-d)[:need]]
                break
        pop = comb[newidx]
        F = Fc[newidx]
        _, rank, crowd = rank_and_crowd(F)

    first = _fast_nondominated_sort(F)[0]
    front = pop[first]
    rev, vamp = _objectives(front, ctx)
    order = np.argsort(-rev)                            # high-revenue end first
    front = front[order]
    info = {
        "mode": "nsga2", "dims": int(N), "n_front": int(len(first)),
        "front": np.column_stack([rev[order], vamp[order]]),  # (revenue, vamp) per solution
    }
    return front, info


# --------------------------------------------------------------------------- entry point
def run_fullmatrix_ga(ctx, lam, *, pop_size=60, generations=120, mutation_rate=0.2,
                      mutation_sigma=0.15, seed=42, elite_frac=0.2, patience=25,
                      multiobjective=False, warm_start=None, stop_check=None):
    """Evolve the full per-cell share matrix. Single-objective (revenue − λ·risk) by default;
    NSGA-II revenue<->risk front when `multiobjective=True`.

    Returns:
      single-objective -> (best_shares (N,), info)
      multiobjective   -> (front_shares (K, N), info) with info['front'] = (revenue, vamp) rows.
    """
    rng = np.random.default_rng(seed)
    N = int(ctx["n_row"])
    cs = np.asarray(ctx["cell_starts"]); cc = np.asarray(ctx["cell_counts"])
    elig = np.asarray(ctx["elig"], float); ref = np.asarray(ctx["ref_share"], float)
    cap = float(ctx.get("max_share", 1.0) or 1.0)
    floor = float(ctx.get("floor", 0.0) or 0.0)

    # Seed: member 0 = the revenue reference; others = multiplicative log-normal perturbations
    # of it (stay positive), all repaired to deployable splits.
    pop = np.empty((pop_size, N))
    pop[0] = ref
    if pop_size > 1:
        pop[1:] = ref[None, :] * np.exp(rng.normal(0.0, 0.6, size=(pop_size - 1, N)))
    pop = _repair(pop, cs, cc, elig, cap, floor)
    if warm_start is not None:
        ws = np.atleast_2d(np.asarray(warm_start, float))
        if ws.shape[1] == N and pop_size > 1:
            k = min(len(ws), pop_size - 1)
            pop[1:1 + k] = _repair(ws[:k], cs, cc, elig, cap, floor)

    if multiobjective:
        return _nsga2(pop, ctx, rng, cs, cc, elig, cap, floor,
                      generations, mutation_rate, mutation_sigma, stop_check=stop_check)

    # ---- single-objective GA (tournament + elitism), same fitness as the tilt GA ---------
    def fit_of(P):
        return _fitness(P, ctx, lam)
    fit = fit_of(pop)
    n_elite = max(1, int(round(elite_frac * pop_size)))
    bi = int(np.argmax(fit))
    best = pop[bi].copy(); best_fit = float(fit[bi]); stale = 0; gens_run = 0
    for g in range(generations):
        gens_run = g + 1
        if stop_check is not None and stop_check():
            break
        order = np.argsort(-fit)
        elite = pop[order[:n_elite]].copy()

        def pick():
            c = rng.integers(0, pop_size, size=3)
            return c[np.argmax(fit[c])]
        children = np.empty_like(pop)
        children[:n_elite] = elite
        for k in range(n_elite, pop_size):
            a, b = pop[pick()], pop[pick()]
            cellmask = rng.random(len(cc)) < 0.5             # per-cell uniform crossover
            rowmask = np.repeat(cellmask, cc)
            child = np.where(rowmask, a, b).astype(float)
            mm = rng.random(N) < mutation_rate               # gaussian mutation
            child = child + np.where(mm, rng.normal(0.0, mutation_sigma, N), 0.0)
            children[k] = child
        pop = _repair(children, cs, cc, elig, cap, floor)
        fit = fit_of(pop)
        gi = int(np.argmax(fit))
        if fit[gi] > best_fit + 1e-9:
            best_fit = float(fit[gi]); best = pop[gi].copy(); stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    rev = float((best * ctx["rev_coef"]).sum())
    info = {"mode": "single", "dims": int(N), "gens": int(gens_run),
            "revenue": rev, "best_fit": float(best_fit), "risk_cost": float(rev - best_fit)}
    return best, info
