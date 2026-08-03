"""
GA - Oliver: a port of the colleague's DEAP-style genetic algorithm, adapted to run INSIDE
this app's engine framework.

WHAT IS FAITHFUL TO OLIVER'S ALGORITHM
--------------------------------------
  * DIRECT per-cell genome: an individual IS the full split (one share per gateway-row), not a
    compact per-MID tilt. So every cell can be set independently — the "fine-brush" freedom the
    48-dial CMA-ES engine lacks.
  * Uncapped genetic material, capped-at-evaluation: mutation/crossover act on the raw split;
    the exploration-floor / max-share are applied only when scoring (mirrors his derive_capped),
    so a gateway zeroed by the cap can still drift back up in the genome.
  * Real GA operators: Gaussian per-row mutation on a random subset of cells (his mutate_split)
    and per-cell uniform crossover (his make_children).
  * NSGA-II multi-objective selection (his tools.selNSGA2) over (success, violation), keeping a
    Pareto front rather than a single scalar winner. A compact, dependency-free reimplementation
    of NSGA-II selection is used so the app needs no extra install.

WHAT IS THIS APP'S (per the user's choice: "your tab-3 rules")
--------------------------------------------------------------
  * The objective and constraints are THIS app's: scoring is `genetic_global._obj_viol` on the
    ctx, so the per-MID VAMP bands, VAMP cap, per-MID volume caps, exploration floor and max-share
    are exactly the ones your other engines use (incl. the volume-weighted-violation toggle).
  * Same call/return contract as `genetic_global.run_midtilt_ga`, so this drops into the engine
    dispatch and the downstream enforcement / impact / compression pipeline is UNCHANGED.

NOT wired to the Numba kernel or pre-clustering (those are CMA-ES-genome specific).
"""
from __future__ import annotations

import numpy as np

from .genetic_global import _obj_viol, _cap_floor_shares, _FEAS_TOL

__build__ = "2026-07-31-ga-oliver-percell-nsga2"


# ---------------------------------------------------------------------------- helpers
# [FN-080]
def _renorm_cells(X, cs, cc, elig, ref):
    """Zero non-eligible rows, then renormalise each cell to sum 1 over its eligible rows.
    Cells that would sum to ~0 fall back to the (eligibility-masked) reference split."""
    X = np.asarray(X, float) * elig[None, :]
    seg = np.repeat(np.add.reduceat(X, cs, axis=1), cc, axis=1)
    good = seg > 1e-12
    out = np.where(good, X / np.where(good, seg, 1.0), 0.0)
    # dead cells -> reference (already eligibility-masked + renormalised once here)
    if (~good).any():
        r = (ref * elig)[None, :] * np.ones((X.shape[0], 1))
        rseg = np.repeat(np.add.reduceat(r, cs, axis=1), cc, axis=1)
        r = np.where(rseg > 1e-12, r / np.where(rseg > 1e-12, rseg, 1.0), 0.0)
        out = np.where(good, out, r)
    return out


# [FN-081]
def _mutate(X, cs, cc, elig, ref, rng, mutation_rate, strength):
    """Gaussian per-row mutation on a random subset of CELLS (mirrors Oliver's mutate_split, but
    cell-wise since a 'row' here is a whole cell's gateways). Renormalises the touched cells."""
    P, N = X.shape
    C = cs.shape[0]
    n_mut = max(1, int(C * mutation_rate))
    Y = X.copy()
    cell_row = np.repeat(np.arange(C), cc)          # gateway-row -> cell index
    for p in range(P):
        pick = rng.choice(C, size=n_mut, replace=False)
        mask = np.isin(cell_row, pick)
        noise = rng.normal(0.0, strength, size=N) * mask
        Y[p] = np.clip(Y[p] + noise, 0.0, None)
    return _renorm_cells(Y, cs, cc, elig, ref)


# [FN-082]
def _crossover(A, B, cs, cc, rng, cx_rate):
    """Per-cell uniform crossover (mirrors make_children): each cell independently comes from one
    parent or the other. Returns two children as (P,N) arrays."""
    P = A.shape[0]
    C = cs.shape[0]
    cellmask = (rng.random((P, C)) < cx_rate)
    gmask = np.repeat(cellmask, cc, axis=1)
    c1 = np.where(gmask, B, A)
    c2 = np.where(gmask, A, B)
    return c1, c2


# [FN-083]
def _score(X, ctx, cs, cc, elig, cap, floor):
    """Cap-at-evaluation (uncapped genome), then this app's _obj_viol → (obj, viol)."""
    capped = _cap_floor_shares(X, cs, cc, elig, float(cap), float(floor)) if (cap < 1.0 or floor > 0.0) else X
    obj, viol = _obj_viol(capped, ctx)
    return np.asarray(obj, float), np.asarray(viol, float)


# ---------------------------------------------------------------------- NSGA-II selection
# [FN-084]
def _fast_nondominated_fronts(M):
    """M: (P, k) MINIMISATION objectives. Returns list of fronts (each a list of indices)."""
    P = M.shape[0]
    dom_count = np.zeros(P, int)
    dominated = [[] for _ in range(P)]
    for i in range(P):
        di = M[i]
        le = np.all(M <= di, axis=1)
        lt = np.any(M < di, axis=1)
        dominates_i = le & lt                       # who dominates i
        dom_count[i] = int(dominates_i.sum())
        # who i dominates
        ge = np.all(M >= di, axis=1); gt = np.any(M > di, axis=1)
        dominated[i] = list(np.where(ge & gt)[0])
    fronts = [list(np.where(dom_count == 0)[0])]
    f = 0
    while fronts[f]:
        nxt = []
        for i in fronts[f]:
            for j in dominated[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        f += 1
        fronts.append(nxt)
    return fronts[:-1]


# [FN-085]
def _crowding(M_front):
    """Crowding distance for one front. M_front: (m, k)."""
    m, k = M_front.shape
    if m == 0:
        return np.zeros(0)
    dist = np.zeros(m)
    for c in range(k):
        order = np.argsort(M_front[:, c])
        dist[order[0]] = dist[order[-1]] = np.inf
        vmin, vmax = M_front[order[0], c], M_front[order[-1], c]
        span = vmax - vmin
        if span <= 1e-12:
            continue
        for r in range(1, m - 1):
            dist[order[r]] += (M_front[order[r + 1], c] - M_front[order[r - 1], c]) / span
    return dist


# [FN-086]
def _select_nsga2(M, k):
    """Pick k indices from M (P,k_obj) minimisation objectives via NSGA-II (fronts + crowding)."""
    chosen = []
    for front in _fast_nondominated_fronts(M):
        if len(chosen) + len(front) <= k:
            chosen.extend(front)
            if len(chosen) == k:
                break
        else:
            need = k - len(chosen)
            cd = _crowding(M[front])
            order = np.argsort(-cd)                  # most-spread first
            chosen.extend([front[i] for i in order[:need]])
            break
    return np.asarray(chosen[:k], int)


# ------------------------------------------------------------------------------- engine
# [FN-087]
def run(ctx, lam=50.0, *, pop_size=64, generations=400, seed=42, warm_start=None,
        stop_check=None, progress_cb=None, mutation_rate=0.3, strength=0.3, cx_rate=0.5,
        **kwargs):
    """Drop-in for genetic_global.run_midtilt_ga. Returns (best_shares (N,), info).

    Extra CMA-ES kwargs (gain_max, auto, patience, n_restarts, polish, ref_gamma, n_fine, numba,
    restart_mode, numba_trust, …) are accepted and ignored — this is a different search."""
    cs = np.ascontiguousarray(ctx["cell_starts"], np.intp)
    cc = np.ascontiguousarray(ctx["cell_counts"], np.intp)
    elig = np.ascontiguousarray(np.asarray(ctx["elig"], float))
    ref = np.ascontiguousarray(np.asarray(ctx["ref_share"], float))
    N = int(ctx["n_row"])
    cap = float(ctx.get("max_share", 1.0) or 1.0)
    floor = float(ctx.get("floor", 0.0) or 0.0)
    P = max(4, int(pop_size))
    G = max(1, int(generations))
    rng = np.random.default_rng(int(seed))
    no_early = bool(ctx.get("no_early_stop", False))

    # ---- initial population: reference + jittered copies (+ warm start if provided) ----
    base = _renorm_cells(ref[None, :], cs, cc, elig, ref)[0]
    pop = np.repeat(base[None, :], P, axis=0)
    seeds = [base]
    if warm_start is not None:
        try:
            _w = np.asarray(warm_start, float).ravel()
            if _w.shape[0] == N:
                seeds.append(_renorm_cells(_w[None, :], cs, cc, elig, ref)[0])
        except Exception:  # noqa: BLE001
            pass
    for i, s in enumerate(seeds):
        pop[i] = s
    if P > len(seeds):                                # jitter the rest for diversity
        pop[len(seeds):] = _mutate(pop[len(seeds):], cs, cc, elig, ref, rng, 1.0, strength)

    obj, viol = _score(pop, ctx, cs, cc, elig, cap, floor)
    M = np.column_stack([-obj, viol])                 # minimise (−success, violation)
    init_best_i = _best_index(obj, viol)
    init_obj, init_viol = float(obj[init_best_i]), float(viol[init_best_i])

    history = []
    stalls = 0
    best_key = _feas_scalar(init_obj, init_viol)
    cand = 0
    for g in range(G):
        if stop_check is not None and stop_check():
            break
        # offspring: crossover then mutation
        perm = rng.permutation(P)
        A, B = pop[perm[: P // 2]], pop[perm[P // 2: 2 * (P // 2)]]
        c1, c2 = _crossover(A, B, cs, cc, rng, cx_rate)
        kids = np.vstack([c1, c2])
        kids = _mutate(kids, cs, cc, elig, ref, rng, mutation_rate, strength)
        # evaluate parents+kids, NSGA-II down-select to P
        allX = np.vstack([pop, kids])
        aobj, aviol = _score(allX, ctx, cs, cc, elig, cap, floor)
        cand += allX.shape[0]
        AM = np.column_stack([-aobj, aviol])
        keep = _select_nsga2(AM, P)
        pop = allX[keep]
        obj, viol = aobj[keep], aviol[keep]
        M = AM[keep]
        # track best-so-far (feasibility-first)
        bi = _best_index(obj, viol)
        gbest = _feas_scalar(float(obj[bi]), float(viol[bi]))
        if gbest > best_key + 1e-9:
            best_key = gbest
            stalls = 0
        else:
            stalls += 1
        history.append((g + 1, float(best_key), float(gbest), float(np.mean(obj)),
                        0.0, float(viol[bi]), 0.0, int(cand)))
        if progress_cb is not None:
            try:
                progress_cb(int(allX.shape[0]))
            except Exception:  # noqa: BLE001
                pass
        if (not no_early) and stalls >= max(40, G // 5):
            break

    bi = _best_index(obj, viol)
    best = pop[bi].copy()
    best_capped = (_cap_floor_shares(best[None, :], cs, cc, elig, cap, floor)[0]
                   if (cap < 1.0 or floor > 0.0) else best)
    b_obj, b_viol = float(obj[bi]), float(viol[bi])
    b_feas = b_viol <= _FEAS_TOL
    best_fit = b_obj if b_feas else (-1e15 - b_viol)
    init_fit = init_obj if init_viol <= _FEAS_TOL else (-1e15 - init_viol)
    # archive: a few of the best distinct splits (best first) for optional warm-starting
    order = np.argsort([_feas_scalar(float(o), float(v)) for o, v in zip(obj, viol)])[::-1]
    archive = pop[order[:5]].copy()
    info = {
        "gens": len(history), "gens_max": int(G),
        "early_stopped": bool(len(history) < G),
        "sigma_final": 0.0, "best_fit": float(best_fit), "init_fit": float(init_fit),
        "revenue": float(b_obj), "risk_cost": float(b_obj - best_fit), "dims": int(N),
        "feasible": bool(b_feas), "violation": float(b_viol),
        "genome": best.copy(), "archive": archive, "history": history,
        "pop_obj": obj.copy(), "pop_viol": viol.copy(),
        "engine": "ga_oliver", "build": __build__,
    }
    return best_capped, info


# [FN-088]
def _best_index(obj, viol):
    """Feasibility-first: among feasible (viol<=tol) pick max obj; else min viol (tie max obj)."""
    obj = np.asarray(obj, float); viol = np.asarray(viol, float)
    feas = viol <= _FEAS_TOL
    if feas.any():
        idx = np.where(feas)[0]
        return int(idx[np.argmax(obj[idx])])
    return int(np.lexsort((-obj, viol))[0])


# [FN-089]
def _feas_scalar(obj, viol):
    """Monotone 'higher = better' score: feasible splits always beat infeasible; feasible ranked
    by obj, infeasible by −violation."""
    return obj if viol <= _FEAS_TOL else (-1e15 - viol)
