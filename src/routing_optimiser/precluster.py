"""
EXPERIMENTAL — rule-safe fine pre-clustering for the GA-Numba engine (opt-in test path).

The cross-cell per-vampMid tilt GA decodes each cell's gateway shares from ONLY that cell's
gateways' (vampMid, reference share, within-MID risk, within-MID revenue-per-unit) and its
eligibility. So two cells whose gateway rows carry an IDENTICAL such signature decode to the
IDENTICAL share vector under EVERY genome. Collapsing those cells to one representative that
carries their SUMMED volume/revenue coefficients therefore leaves the search's objective and
its per-vampMid VAMP/volume constraints numerically unchanged, while shrinking the per-generation
hot loop (fewer gateway-rows). Expanding the representative's shares back to its member cells is a
direct copy (the gateway layout is identical by construction).

"Rule-safe / fine": the signature INCLUDES the eligibility flag (which already carries the pre-GA
bank-block, plus bans/wallet/USA when the eligibility operator is active) and the vampMid, so a
cluster can NEVER merge two cells that differ in a routing rule or MID membership — the exact
concern raised for coarse pre-clustering. This is near-LOSSLESS (identical-signature merges only);
it does not chase extra compression by merging merely-similar cells.

This module ONLY builds the reduced problem and the expand-back map. Wiring it into the engine
dispatch (run the GA on the reduced ctx, expand the result, then enforce/verify at full grain)
is done by the caller. Nothing here touches the live engine.
"""
from __future__ import annotations

import numpy as np


def _cell_signatures(ctx, *, ref_dp: int = 9, risk_dp: int = 9, rev_dp: int = 9):
    """One hashable signature per cell. Two cells share a signature IFF, gateway-for-gateway
    (in row order), they have the same (vampMid, eligibility, reference share, risk, revenue-
    per-unit) — i.e. they decode identically AND share the same rule/MID profile. Returns a list
    of signatures (one per cell) aligned to ctx['cell_starts']."""
    cs = np.asarray(ctx["cell_starts"], np.intp)
    cc = np.asarray(ctx["cell_counts"], np.intp)
    mid = np.asarray(ctx["mid_id"], np.intp)
    elig = (np.asarray(ctx["elig"], float) > 0.5).astype(np.int8)
    ref = np.round(np.asarray(ctx["ref_share"], float), ref_dp)
    risk = np.round(np.asarray(ctx["risk"], float), risk_dp)
    cv = np.asarray(ctx["cell_vol"], float)
    rc = np.asarray(ctx["rev_coef"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rpu = np.round(np.where(cv > 0, rc / cv, 0.0), rev_dp)   # revenue per unit (intensive)
    sigs = []
    for c in range(len(cs)):
        s0 = int(cs[c]); s1 = s0 + int(cc[c])
        sigs.append(tuple((int(mid[g]), int(elig[g]), float(ref[g]), float(risk[g]), float(rpu[g]))
                          for g in range(s0, s1)))
    return sigs


def build_clusters(ctx, **kw):
    """Group cells by identical signature. Returns a dict with:
      rep_cells   : list[int]            representative cell index per cluster (first occurrence)
      members     : list[np.ndarray]     original cell indices in each cluster
      cell2rep    : (n_cells,) int       cluster id per original cell
    Deterministic (first-occurrence order)."""
    sigs = _cell_signatures(ctx, **kw)
    order, seen = {}, []
    cell2rep = np.empty(len(sigs), np.intp)
    members = []
    for ci, sg in enumerate(sigs):
        if sg not in order:
            order[sg] = len(seen); seen.append(ci); members.append([])
        k = order[sg]; cell2rep[ci] = k; members[k].append(ci)
    return {"rep_cells": seen, "members": [np.asarray(m, np.intp) for m in members],
            "cell2rep": cell2rep}


def reduce_ctx(ctx, clusters):
    """Build a REDUCED ctx over one representative cell per cluster. Intensive per-gateway fields
    (ref_share, risk, elig, mid_id, sr, ticket, baseline_share, zr/zq if present) are copied from
    the representative (identical across members by construction). EXTENSIVE fields (cell_vol,
    rev_coef) are SUMMED across member cells position-for-position, so Σ revenue and every per-MID
    aggregate computed on the reduced problem equal those on the full problem for any share vector.
    Returns (reduced_ctx, expand)."""
    cs = np.asarray(ctx["cell_starts"], np.intp)
    cc = np.asarray(ctx["cell_counts"], np.intp)
    reps = clusters["rep_cells"]; members = clusters["members"]
    # new contiguous layout: representative cells back to back
    new_counts = np.array([int(cc[r]) for r in reps], np.intp)
    new_starts = np.concatenate([[0], np.cumsum(new_counts)[:-1]]).astype(np.intp)
    N_new = int(new_counts.sum())
    # map each NEW row -> the ORIGINAL representative row it copies from
    src_rows = np.empty(N_new, np.intp)
    for k, r in enumerate(reps):
        o0 = int(cs[r]); n0 = int(new_starts[k])
        for j in range(int(cc[r])):
            src_rows[n0 + j] = o0 + j

    def _take(name):
        return np.asarray(ctx[name])[src_rows]

    # extensive fields: sum members position-wise
    cv = np.asarray(ctx["cell_vol"], float); rc = np.asarray(ctx["rev_coef"], float)
    new_cv = np.zeros(N_new); new_rc = np.zeros(N_new)
    for k, r in enumerate(reps):
        n0 = int(new_starts[k]); w = int(cc[r])
        acc_cv = np.zeros(w); acc_rc = np.zeros(w)
        for mc in members[k]:
            o0 = int(cs[mc])
            acc_cv += cv[o0:o0 + w]
            acc_rc += rc[o0:o0 + w]
        new_cv[n0:n0 + w] = acc_cv
        new_rc[n0:n0 + w] = acc_rc

    rctx = dict(ctx)   # shallow copy; overwrite the row-aligned arrays
    rctx["cell_starts"] = new_starts
    rctx["cell_counts"] = new_counts
    rctx["n_row"] = N_new
    for _f in ("elig", "ref_share", "risk", "sr", "ticket", "baseline_share", "base"):
        if _f in ctx and np.asarray(ctx[_f]).shape[:1] == (len(np.asarray(ctx["elig"])),):
            rctx[_f] = _take(_f)
    rctx["mid_id"] = _take("mid_id")
    rctx["cell_vol"] = new_cv
    rctx["rev_coef"] = new_rc
    # per-MID row groupings must be rebuilt for the reduced layout
    _mid = np.asarray(rctx["mid_id"], np.intp)
    n_mid = int(ctx["n_mid"])
    rctx["mid_rows"] = [np.where(_mid == m)[0] for m in range(n_mid)]
    # drop precomputed structures that were built for the OLD layout (caller/GA rebuilds them)
    for _k in ("_mid_S", "elig_op"):
        rctx.pop(_k, None)
    expand = {"src_rows": src_rows, "reps": reps, "members": members,
              "new_starts": new_starts, "new_counts": new_counts,
              "cell_starts": cs, "cell_counts": cc, "n_row_full": int(cs[-1] + cc[-1]),
              "n_cells_full": int(len(cc)), "n_cells_rep": int(len(reps))}
    return rctx, expand


def run_midtilt_ga_preclustered(ctx, *args, **kw):
    """OPT-IN drop-in for `genetic_global.run_midtilt_ga` (same call/return contract). Clusters the
    ctx's cells rule-safely, runs the real GA on the REDUCED (fewer-row) problem, then expands the
    winning shares back to the full cell layout. Because the reduction preserves the objective and
    every per-vampMid constraint exactly (see module docstring + tests), the search is near-lossless.
    Returns (full_shares (N_full,), info) — `info['precluster']` carries the reduction ratio.

    NOTE: the eligibility-in-scoring operator (ctx['elig_op'], only active with ROUTING_GA_ELIG=1)
    is not carried onto the reduced layout, so under pre-clustering bans/wallet/USA are applied by
    downstream enforcement rather than inside the score; the pre-GA bank-block IS preserved (it is
    baked into ctx['elig'], which is reduced)."""
    from .genetic_global import run_midtilt_ga
    cache = ctx.get("_precluster_cache")
    if cache is None:
        clusters = build_clusters(ctx)
        n_full = int(len(np.asarray(ctx["cell_counts"]))); n_rep = len(clusters["rep_cells"])
        if n_rep >= n_full:                       # nothing collapses → run full-resolution, no-op
            ctx["_precluster_cache"] = ("noop", None, None)
        else:
            _rc, expand = reduce_ctx(ctx, clusters)
            _keys = ("cell_starts", "cell_counts", "n_row", "elig", "ref_share", "risk", "sr",
                     "ticket", "baseline_share", "base", "mid_id", "cell_vol", "rev_coef", "mid_rows")
            red = {k: _rc[k] for k in _keys if k in _rc}
            ctx["_precluster_cache"] = ("run", red, expand)
        cache = ctx["_precluster_cache"]
    mode, red, expand = cache
    if mode == "noop":
        return run_midtilt_ga(ctx, *args, **kw)
    rctx = dict(ctx); rctx.update(red)
    for _k in ("_mid_S", "elig_op", "_precluster_cache"):
        rctx.pop(_k, None)
    _wsh = ctx.get("warm_shares")
    if _wsh is not None:                          # reduce row-aligned warm split(s) to rep rows
        src = expand["src_rows"]; nf = int(expand["n_row_full"])

        def _red(w):
            w = np.asarray(w, float)
            return w[src] if (w.ndim == 1 and w.shape[0] == nf) else w
        if isinstance(_wsh, (list, tuple)):
            rctx["warm_shares"] = [_red(w) for w in _wsh]
        else:
            _wa = np.asarray(_wsh, float)
            rctx["warm_shares"] = (np.stack([_red(_wa[i]) for i in range(_wa.shape[0])])
                                   if _wa.ndim == 2 else _red(_wa))
    best_rep, info = run_midtilt_ga(rctx, *args, **kw)
    best_full = expand_shares(best_rep, expand)
    info = dict(info or {})
    info["precluster"] = {"cells_full": expand["n_cells_full"], "cells_rep": expand["n_cells_rep"],
                          "ratio": expand["n_cells_full"] / max(expand["n_cells_rep"], 1)}
    return best_full, info


def expand_shares(rep_shares, expand):
    """Copy each representative cell's per-gateway shares to ALL its member cells → full (N,)
    share vector in the ORIGINAL row layout. Direct copy (identical gateway layout)."""
    rep_shares = np.asarray(rep_shares, float)
    cs = expand["cell_starts"]; cc = expand["cell_counts"]
    ns = expand["new_starts"]; reps = expand["reps"]; members = expand["members"]
    full = np.zeros(int(expand["n_row_full"]), float)
    for k, r in enumerate(reps):
        n0 = int(ns[k]); w = int(cc[r])
        seg = rep_shares[n0:n0 + w]
        for mc in members[k]:
            o0 = int(cs[mc])
            full[o0:o0 + w] = seg
    return full
