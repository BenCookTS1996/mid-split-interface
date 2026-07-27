"""
Volume-weighted k-means compression of a proposed split.

Adapted from your k_means_compression.py, made self-contained. The idea:
your ideal split has one bespoke rule per cell, which is far too many JSON
configs to operate. Cells with near-identical gateway splits can share one
representative rule. We cluster the per-cell share vectors (weighted by
transaction volume, so high-volume cells pull the representative towards
themselves) and keep just enough clusters to stay faithful to the ideal
split, per a target accuracy you set per RPGT.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

__build__ = "2026-07-24-optin-ward-tree+knapsack+rawleaves"

DEFAULT_TARGETS = {
    "DEFAULT": 85.0,
    "Monthly Initial": 93.0,
    "Addon Sale": 93.0,
    "Upgrades": 85.0,
    "Annual Sub Sale": 90.0,
}
MAX_GATEWAY_CAP = 0.97  # keep >=3% on a backup, mirroring your script


def wallet_segment_split(split: pd.DataFrame, wallet_incapable, wallet_frac=None,
                         wallet_default: float = 0.0, fid2vamp=None,
                         wallet_label: str = "wallet", nonwallet_label: str = "non_gp_ap") -> pd.DataFrame:
    """Add a `pmp` (paymentMethodProvider) dimension so wallet traffic routes only
    to capable gateways.

    Each (rpgt, currency, bank) cell is split into a NON-WALLET segment (shares
    unchanged) and a WALLET segment (wallet-incapable gateways zeroed + renormalised),
    with the cell volume divided by the cell's wallet fraction. If no gateway is
    wallet-incapable, the split is returned unchanged (no pmp dimension), so configs
    keep matching all payment methods.

    split columns: rpgt, currency, bank, gateway, share, cell_volume.
    """
    wallet_incapable = set(wallet_incapable or [])
    if not wallet_incapable:
        return split.copy()
    wallet_frac = wallet_frac or {}
    fid2vamp = fid2vamp or {}
    d = split.reset_index(drop=True).copy()
    d["_gw"] = d["gateway"].astype(str).str.strip().str.lower()
    d["_vm"] = d["_gw"].map(fid2vamp).fillna(d["_gw"])
    inc = (d["_gw"].isin(wallet_incapable) | d["_vm"].isin(wallet_incapable)).to_numpy()

    seg_nw, seg_w = [], []
    for _, grp in d.groupby(["rpgt", "currency", "bank"], sort=False):
        cur = str(grp["currency"].iloc[0]).strip().lower()
        bank = str(grp["bank"].iloc[0]).strip().lower()
        wf = wallet_frac.get((cur, bank), wallet_default)
        wf = 0.0 if (wf != wf) else min(max(float(wf), 0.0), 1.0)
        cvol = float(grp["cell_volume"].iloc[0]) if "cell_volume" in grp.columns else 0.0

        nw = grp.copy()
        nw["pmp"] = nonwallet_label
        nw["cell_volume"] = cvol * (1.0 - wf)
        seg_nw.append(nw)

        wl = grp.copy()
        s = wl["share"].to_numpy(float).copy()
        m = inc[grp.index.to_numpy()]
        s[m] = 0.0
        tot = s.sum()
        wl["share"] = s / tot if tot > 0 else grp["share"].to_numpy(float)
        wl["pmp"] = wallet_label
        wl["cell_volume"] = cvol * wf
        seg_w.append(wl)

    out = pd.concat(seg_nw + seg_w, ignore_index=True)
    return out.drop(columns=[c for c in ["_gw", "_vm"] if c in out.columns])


def _cap_and_respill(vec: np.ndarray, cap: float) -> np.ndarray:
    vec = np.clip(vec, 0, None)
    s = vec.sum()
    vec = vec / s if s > 0 else vec
    for _ in range(50):
        over = vec > cap
        if not over.any():
            break
        excess = (vec[over] - cap).sum()
        vec[over] = cap
        room = (~over) & (vec > 0)
        if not room.any():
            room = ~over
        if not room.any():
            break
        vec[room] += excess * (vec[room] / max(vec[room].sum(), 1e-12))
        vec = vec / vec.sum()
    return vec


def _weighted_accuracy(X: np.ndarray, recon: np.ndarray, w: np.ndarray) -> float:
    """% fidelity: 100 = identical. Uses L1 distance on share vectors."""
    l1 = np.abs(X - recon).sum(axis=1)          # in [0, 2]
    wavg = (w * l1).sum() / max(w.sum(), 1e-12)
    return float((1.0 - wavg / 2.0) * 100.0)


def _fit_k(X, w, k, seed=42):
    k = int(min(k, len(X)))
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X, sample_weight=w)
    recon = km.cluster_centers_[km.labels_]
    return km, recon


def compress_split(
    split: pd.DataFrame,
    group_keys=("rpgt", "currency"),
    rpgt_targets: dict | None = None,
    max_gateway_cap: float = MAX_GATEWAY_CAP,
    k_max: int = 40,
    seed: int = 42,
):
    """
    Returns (compressed_rules, elbow, stats).

    compressed_rules: one row per representative rule with the gateway share
                      columns, the covered banks, volume, and how many raw
                      cells it stands in for.
    elbow:            per (group) the k chosen and accuracy achieved.
    stats:            headline counts (raw rules vs compressed rules).
    """
    group_keys = list(group_keys)
    # A `pmp` (paymentMethodProvider) column adds a wallet/non-wallet dimension:
    # cluster each segment separately and carry it into the rules.
    has_pmp = "pmp" in split.columns
    idx_cols = ["rpgt", "currency", "bank"] + (["pmp"] if has_pmp else [])
    if has_pmp and "pmp" not in group_keys:
        group_keys = group_keys + ["pmp"]
    rpgt_targets = {**DEFAULT_TARGETS, **(rpgt_targets or {})}
    tgt = {k.lower(): v for k, v in rpgt_targets.items()}
    default_acc = tgt.get("default", 85.0)

    # Build the share matrix: index = cell, columns = gateway.
    mat = (split.pivot_table(index=idx_cols,
                             columns="gateway", values="share", aggfunc="sum")
           .fillna(0.0))
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gateway_cols = list(mat.columns)

    vol = (split.groupby(idx_cols)["cell_volume"].first()
           .reindex(mat.index).fillna(0.0))

    md = mat.reset_index()
    md["_vol"] = vol.to_numpy()

    compressed_rows, elbow_rows = [], []
    for gkey, grp in md.groupby(group_keys):
        X = grp[gateway_cols].to_numpy(float)
        w = np.maximum(grp["_vol"].to_numpy(float), 1e-6)
        rpgt = grp["rpgt"].iloc[0]
        target = tgt.get(str(rpgt).lower(), default_acc)
        n_rows = len(X)

        # Smallest k that reaches the target accuracy (binary-ish search).
        # Fit k=1 once; the previous throwaway _fit_k here was immediately
        # overwritten below before ever being used.
        chosen_k = 1
        km, recon = _fit_k(X, w, 1, seed)
        acc = _weighted_accuracy(X, recon, w)
        chosen_km, chosen_recon, chosen_acc = km, recon, acc
        if acc < target:
            for k in range(2, min(k_max, n_rows) + 1):
                km, recon = _fit_k(X, w, k, seed)
                acc = _weighted_accuracy(X, recon, w)
                chosen_k, chosen_km, chosen_recon, chosen_acc = k, km, recon, acc
                if acc >= target:
                    break
        else:
            chosen_k = 1

        elbow_rows.append({**dict(zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,))),
                           "cells": n_rows, "clusters": chosen_k,
                           "target_accuracy": target, "achieved_accuracy": round(chosen_acc, 2)})

        # Emit one representative rule per cluster.
        grp = grp.reset_index(drop=True)
        labels = chosen_km.labels_
        for cl in range(chosen_k):
            members = grp[labels == cl]
            if members.empty:
                continue
            centroid = _cap_and_respill(chosen_km.cluster_centers_[cl], max_gateway_cap)
            row = {k: v for k, v in zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,))}
            row["banks"] = sorted(members["bank"].astype(str).unique().tolist())
            row["n_cells"] = int(len(members))
            row["volume"] = float(members["_vol"].sum())
            for gc, val in zip(gateway_cols, centroid):
                row[gc] = round(float(val) * 100, 4)  # store as percentage
            compressed_rows.append(row)

    compressed = pd.DataFrame(compressed_rows)
    elbow = pd.DataFrame(elbow_rows)
    stats = {
        "raw_rules": int(len(mat)),
        "compressed_rules": int(len(compressed)),
        "reduction_pct": round(100 * (1 - len(compressed) / max(len(mat), 1)), 1),
        "gateways": gateway_cols,
    }
    return compressed, elbow, stats


def count_config_rules(compressed: pd.DataFrame) -> int:
    """Number of JSON routing rules the compressed split will generate.

    Each representative rule becomes one connector pool per RPGT rule set.
    """
    return int(len(compressed))


def compress_to_pool_budget(split: pd.DataFrame, target_pools: int, count_pools_fn,
                            group_keys=("rpgt", "currency"),
                            max_gateway_cap: float = MAX_GATEWAY_CAP,
                            k_max: int = 60, seed: int = 42,
                            method: str = "kmeans", allocation: str = "greedy",
                            parallel: int = 1, count_backend: str = "loky"):
    """Compress so the GENERATED POOL count is <= target_pools, using as large a cell
    budget as possible under that ceiling.

    The pool count only exists after the full expand-and-merge pipeline
    (build_split_exports -> generate_configs), so this binary-searches the cell budget
    fed to `compress_to_budget` and asks `count_pools_fn` for the resulting pool count
    at each step, keeping the largest cell budget whose pools <= target.

    Parameters
    ----------
    split : per-cell long split (rpgt, currency, bank[, pmp], gateway, share, cell_volume).
    target_pools : desired MAX number of generated pools (hard ceiling).
    count_pools_fn : callable(compressed_long_df) -> int. Runs the caller's
        build_split_exports + generate_configs on a split and returns the pool count.
        Supplied by the caller because pool generation needs brand/wallet/country context.

    Returns (compressed_long, stats) where stats has:
      raw_cells, raw_pools, cells, pools, target_pools, global_accuracy,
      feasible (bool; False = even the smallest split exceeds target),
      curve [(cells, pools), ...] over the evaluated budgets, evals (int).

    Notes
    -----
    * Pool count is (near-)monotonic in the cell budget: more clusters -> more distinct
      routing signatures -> fewer merges -> more pools. Binary search relies on this; the
      RETURNED budget is always verified to satisfy pools <= target, so the ceiling holds
      even if k-means wobble makes the curve slightly non-monotonic.
    * If target_pools <= 0, no compression is applied (0 = 'no compression' by convention).
    """
    has_pmp = "pmp" in split.columns
    idx_cols = ["rpgt", "currency", "bank"] + (["pmp"] if has_pmp else [])
    raw_cells = int(split.groupby(idx_cols).ngroups)

    # Uncompressed pool count (each cell keeps its own centroid) from the raw split.
    raw_pools = int(count_pools_fn(split))
    curve = [(raw_cells, raw_pools)]

    def _no_compression(_reason_feasible):
        _st = {"raw_cells": raw_cells, "raw_pools": raw_pools, "cells": raw_cells,
               "pools": raw_pools, "target_pools": int(target_pools),
               "global_accuracy": 100.0, "feasible": _reason_feasible,
               "curve": sorted(set(curve)), "evals": 1}
        return split.copy(), _st

    # No budget, or the full split already fits the ceiling -> ship it uncompressed.
    if int(target_pools) <= 0 or raw_pools <= int(target_pools):
        return _no_compression(raw_pools <= int(target_pools) or int(target_pools) <= 0)

    # Build the clustering context ONCE. The share matrix and the deterministic KMeans
    # fits are then reused across every binary-search budget (identical results, no
    # refits). Config-gen (the heavy count_pools_fn) is deduped by clustering signature
    # (kcur) so budgets that collapse to the same clustering don't regenerate configs.
    _ctx = _build_compress_context(split, group_keys, max_gateway_cap, k_max, seed,
                                   method=method, allocation=allocation)
    _cache = {}          # budget -> (cl, st, pools, cells)
    _by_kcur = {}        # kcur signature -> (cl, st, pools, cells)

    def _eval(b):
        b = int(max(1, min(b, raw_cells)))
        if b not in _cache:
            _cl, _st, _kc = _compress_with_context(_ctx, b)
            if _kc in _by_kcur:
                _cache[b] = _by_kcur[_kc]     # identical clustering → reuse pool count
            else:
                _pools = int(count_pools_fn(_cl))
                _cells = int(_st.get("compressed_rules", b))
                _entry = (_cl, _st, _pools, _cells)
                _by_kcur[_kc] = _entry
                _cache[b] = _entry
                curve.append((_cells, _pools))
        return _cache[b]

    def _parallel_counts(cls):
        """Run the (expensive) count_pools_fn on a list of clusterings, in parallel when
        `parallel` > 1. The counts are independent and deterministic, so the values are
        identical regardless of backend/order. Cascades count_backend → threading →
        sequential so a pickling/spawn hiccup can never make it slower than serial."""
        if len(cls) <= 1 or int(parallel) <= 1:
            return [int(count_pools_fn(c)) for c in cls]
        _njobs = min(len(cls), int(parallel))
        from joblib import Parallel, delayed
        import inspect as _insp_jl
        try:                                       # older joblib lacks inner_max_num_threads
            _jl_inner_ok = "inner_max_num_threads" in _insp_jl.signature(Parallel).parameters
        except Exception:  # noqa: BLE001
            _jl_inner_ok = False
        for _bk in ([count_backend] + (["threading"] if count_backend != "threading" else [])):
            try:
                _pk = dict(n_jobs=_njobs, backend=_bk)
                if _bk in ("loky", "multiprocessing") and _jl_inner_ok:
                    _pk["inner_max_num_threads"] = 1
                return [int(x) for x in Parallel(**_pk)(delayed(count_pools_fn)(c) for c in cls)]
            except Exception:  # noqa: BLE001
                continue
        return [int(count_pools_fn(c)) for c in cls]

    def _eval_many(bs):
        """Evaluate several budgets at once: build each clustering (cheap), dedupe by
        clustering signature, then count the UNIQUE clusterings in parallel. Populates the
        same caches as `_eval`, so results are identical to evaluating them one-by-one."""
        bs = sorted({int(max(1, min(b, raw_cells))) for b in bs})
        _need = []
        for b in bs:
            if b in _cache:
                continue
            _cl, _st, _kc = _compress_with_context(_ctx, b)
            if _kc in _by_kcur:
                _cache[b] = _by_kcur[_kc]
            else:
                _need.append((b, _cl, _st, _kc))
        _uniq = {}
        for (b, _cl, _st, _kc) in _need:
            _uniq.setdefault(_kc, (b, _cl, _st))
        if _uniq:
            _keys = list(_uniq.keys())
            _pl = _parallel_counts([_uniq[k][1] for k in _keys])
            for _kc, _pv in zip(_keys, _pl):
                _b0, _cl0, _st0 = _uniq[_kc]
                _entry = (_cl0, _st0, int(_pv), int(_st0.get("compressed_rules", _b0)))
                _by_kcur[_kc] = _entry
                curve.append((_entry[3], int(_pv)))
            for (b, _cl, _st, _kc) in _need:
                _cache[b] = _by_kcur[_kc]
        return {b: _cache[b] for b in bs}

    # Largest budget b in [1, raw_cells] whose generated pools <= target.
    if int(parallel) > 1 and raw_cells > 2:
        # PARALLEL K-ARY SEARCH (opt-in): the same EXACT target as the binary search — the
        # largest budget with pools <= target — but it probes `parallel` evenly-spaced budgets
        # PER ROUND and evaluates them concurrently, so it needs far fewer sequential rounds
        # (~log_{k+1} vs ~log_2). On a monotone pool-vs-budget curve it returns the SAME budget
        # as the binary search below; the invariant (_lo-1 <= b* <= _hi) is preserved every
        # round and the final budget is verified <= target, so the ceiling always holds.
        _k = max(2, int(parallel))
        _lo, _hi, _best = 1, raw_cells, None
        while _lo <= _hi:
            _span = _hi - _lo + 1
            if _span <= _k:
                _probes = list(range(_lo, _hi + 1))
            else:
                _step = _span / (_k + 1.0)
                _probes = sorted({min(_hi, max(_lo, int(_lo + round(_step * (i + 1)))))
                                  for i in range(_k)})
            _res = _eval_many(_probes)
            _ok = [b for b in _probes if _res[b][2] <= int(target_pools)]
            _fail = [b for b in _probes if _res[b][2] > int(target_pools)]
            if _ok:
                _best = max(_ok) if _best is None else max(_best, max(_ok))
                _lo = max(_ok) + 1
            if _fail:
                _hi = min(_fail) - 1
            if not _ok and not _fail:
                break
    else:
        _lo, _hi, _best = 1, raw_cells, None
        while _lo <= _hi:
            _mid = (_lo + _hi) // 2
            _cl, _st, _pools, _cells = _eval(_mid)
            if _pools <= int(target_pools):
                _best = _mid
                _lo = _mid + 1
            else:
                _hi = _mid - 1

    if _best is None:                       # even the smallest split overshoots the ceiling
        _cl, _st, _pools, _cells = _eval(1)
        _feasible = False
    else:
        _cl, _st, _pools, _cells = _eval(_best)
        _feasible = True

    stats = {
        "raw_cells": raw_cells, "raw_pools": raw_pools,
        "cells": int(_cells), "pools": int(_pools),
        "target_pools": int(target_pools),
        "global_accuracy": float(_st.get("global_accuracy", 0.0)),
        "feasible": bool(_feasible),
        "curve": sorted(set(curve)),
        "evals": len(_cache) + 1,
    }
    return _cl, stats


def _build_compress_context(split: pd.DataFrame, group_keys, max_gateway_cap, k_max, seed,
                            method: str = "kmeans", allocation: str = "greedy"):
    """Precompute everything that DOESN'T depend on the cluster budget: the volume-weighted
    share matrix, the per-group arrays, and an empty (group, k) -> (km, acc) fit cache.

    The KMeans fit for a given (group, k) is fully deterministic here (fixed seed, fixed
    n_init, same X and weights), so this context — including its `fits` cache — can be
    reused across many budgets to give IDENTICAL results with no recomputation.

    `method` ("kmeans"|"ward") and `allocation` ("greedy"|"knapsack") are OPT-IN. With the
    defaults ("kmeans","greedy") the compression is byte-for-byte the existing behaviour;
    any other combination routes through `_compress_ext` (see `_compress_with_context`).
    """
    group_keys = list(group_keys)
    has_pmp = "pmp" in split.columns
    idx_cols = ["rpgt", "currency", "bank"] + (["pmp"] if has_pmp else [])
    if has_pmp and "pmp" not in group_keys:
        group_keys = group_keys + ["pmp"]

    mat = (split.pivot_table(index=idx_cols, columns="gateway", values="share", aggfunc="sum")
           .fillna(0.0))
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gateway_cols = list(mat.columns)
    vol = split.groupby(idx_cols)["cell_volume"].first().reindex(mat.index).fillna(0.0)
    md = mat.reset_index()
    md["_vol"] = vol.to_numpy()

    groups = list(md.groupby(group_keys))
    G = len(groups)
    return {
        "group_keys": group_keys, "idx_cols": idx_cols, "gateway_cols": gateway_cols,
        "groups": groups, "G": G, "raw_rules": int(len(mat)),
        "total_vol": float(md["_vol"].sum()) or 1.0,
        "gX": [grp[gateway_cols].to_numpy(float) for _, grp in groups],
        "gW": [np.maximum(grp["_vol"].to_numpy(float), 1e-6) for _, grp in groups],
        "gVol": [float(grp["_vol"].sum()) for _, grp in groups],
        "gKmax": [int(min(k_max, len(grp))) for _, grp in groups],
        "max_gateway_cap": float(max_gateway_cap), "seed": int(seed),
        "fits": [dict() for _ in range(G)],   # persistent (group,k) -> (km, acc) cache
        "method": str(method), "allocation": str(allocation),
    }


def _compress_with_context(ctx, n_configs):
    """Greedy volume-weighted cluster allocation for a given budget, using a prebuilt
    context (so KMeans fits are shared/cached across budgets). Returns
    (compressed_long, stats, kcur_tuple). Behaviour is identical to a fresh
    compress_to_budget call — only the redundant matrix build + refits are avoided.
    """
    # OPT-IN alternatives (Ward tree / exact knapsack) route out here. The DEFAULT
    # ("kmeans","greedy") falls through to the original code below, unchanged.
    if ctx.get("method", "kmeans") != "kmeans" or ctx.get("allocation", "greedy") != "greedy":
        return _compress_ext(ctx, n_configs)
    import heapq
    group_keys = ctx["group_keys"]; idx_cols = ctx["idx_cols"]
    gateway_cols = ctx["gateway_cols"]; groups = ctx["groups"]
    G = ctx["G"]; raw_rules = ctx["raw_rules"]; total_vol = ctx["total_vol"]
    gX = ctx["gX"]; gW = ctx["gW"]; gVol = ctx["gVol"]; gKmax = ctx["gKmax"]
    max_gateway_cap = ctx["max_gateway_cap"]; seed = ctx["seed"]; fits = ctx["fits"]
    n_budget = max(int(n_configs), G)        # need ≥1 cluster per group

    def _fit(g, k):
        k = int(min(max(k, 1), gKmax[g]))
        if k not in fits[g]:
            km, recon = _fit_k(gX[g], gW[g], k, seed)
            fits[g][k] = (km, _weighted_accuracy(gX[g], recon, gW[g]))
        return fits[g][k]

    kcur = [1] * G
    for g in range(G):
        _fit(g, 1)
    # running global accuracy (%) = Σ acc_g·vol_g / total_vol
    global_acc = sum(_fit(g, 1)[1] * gVol[g] for g in range(G)) / total_vol
    curve = [(G, round(global_acc, 3))]

    heap = []                                # (-marginal_global_gain, g)

    def _push_next(g):
        if kcur[g] >= gKmax[g]:
            return
        acc0 = _fit(g, kcur[g])[1]
        acc1 = _fit(g, kcur[g] + 1)[1]
        # k-means accuracy isn't guaranteed monotonic in k (local minima), so clamp the
        # marginal gain to ≥0 — a cluster can never be scored as HURTING fidelity, which
        # kept the greedy from starving a high-volume group after a noisy dip.
        gain = (gVol[g] / total_vol) * max(0.0, acc1 - acc0)
        heapq.heappush(heap, (-gain, g))

    for g in range(G):
        _push_next(g)

    remaining = n_budget - G
    while remaining > 0 and heap:
        neg, g = heapq.heappop(heap)
        gain = -neg
        if gain <= 1e-9:                      # no group can improve → stop (don't spend
            break                            # configs that add no fidelity; N is an upper bound)
        if kcur[g] >= gKmax[g]:
            continue
        kcur[g] += 1
        remaining -= 1
        global_acc += gain                   # realised (clamped) gain
        curve.append((int(sum(kcur)), round(global_acc, 3)))
        _push_next(g)

    # Build the expanded (centroid) split + stats.
    out_rows, per_group = [], []
    _pr_num, _pr_den = {}, {}
    for g, (gkey, grp) in enumerate(groups):
        k = kcur[g]
        km, acc = _fit(g, k)
        labels = km.labels_
        centroids = [_cap_and_respill(km.cluster_centers_[cl], max_gateway_cap)
                     for cl in range(km.n_clusters)]
        grp = grp.reset_index(drop=True)
        rpgt = str(grp["rpgt"].iloc[0])
        for i in range(len(grp)):
            cvec = centroids[labels[i]]
            base = {c: grp[c].iloc[i] for c in idx_cols}
            base["cell_volume"] = float(grp["_vol"].iloc[i])
            for gc, val in zip(gateway_cols, cvec):
                if val > 1e-9:
                    r = dict(base)
                    r["gateway"] = gc
                    r["share"] = float(val)
                    out_rows.append(r)
        per_group.append({**dict(zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,))),
                          "cells": int(len(grp)), "clusters": int(k),
                          "accuracy": round(acc, 2), "volume": float(gVol[g])})
        _pr_num[rpgt] = _pr_num.get(rpgt, 0.0) + acc * gVol[g]
        _pr_den[rpgt] = _pr_den.get(rpgt, 0.0) + gVol[g]

    compressed_long = pd.DataFrame(out_rows)
    per_rpgt = {rp: round(_pr_num[rp] / max(_pr_den[rp], 1e-9), 2) for rp in _pr_num}
    stats = {
        "raw_rules": raw_rules,
        "compressed_rules": int(sum(kcur)),
        "global_accuracy": round(sum(pg["accuracy"] * pg["volume"] for pg in per_group) / total_vol, 2),
        "per_group": per_group,
        "per_rpgt": per_rpgt,
        "curve": curve,
        "n_groups": G,
        "budget": n_budget,
    }
    return compressed_long, stats, tuple(kcur)


def _compress_ext(ctx, n_configs):
    """OPT-IN compression: alternative cluster METHOD and/or budget ALLOCATION.

      method="ward"          — fit ONE weighted KMeans per group at its k_max, build a Ward
                               tree over those fine centroids, and get any coarser k by a tree
                               CUT (no re-fit). KMeans-then-Ward keeps the volume weighting and
                               makes the cluster counts NESTED, so a UI can slide the count and
                               get instant, consistent results (mirrors the co-worker's reducer).
      allocation="knapsack"  — pick clusters-per-group by an EXACT multiple-choice knapsack
                               (maximise Σ vol·accuracy s.t. Σ clusters ≤ budget) instead of the
                               greedy marginal-gain heap. Never worse than greedy for a budget,
                               but needs each group's accuracy curve, so it is slower.

    Returns (compressed_long, stats, kcur_tuple) — the SAME contract as the greedy path.
    """
    group_keys = ctx["group_keys"]; idx_cols = ctx["idx_cols"]
    gateway_cols = ctx["gateway_cols"]; groups = ctx["groups"]
    G = ctx["G"]; raw_rules = ctx["raw_rules"]; total_vol = ctx["total_vol"]
    gX = ctx["gX"]; gW = ctx["gW"]; gVol = ctx["gVol"]; gKmax = ctx["gKmax"]
    max_gateway_cap = ctx["max_gateway_cap"]; seed = ctx["seed"]
    method = ctx.get("method", "kmeans"); allocation = ctx.get("allocation", "greedy")
    n_budget = max(int(n_configs), G)

    # --- per-group model: labels + centroids at any k, and accuracy at any k --------------
    ward = ctx.setdefault("_ward", [None] * G)     # cached (fine_km, fine_centroids, fine_vol, Z)

    _WARD_RAW_MAX = 500   # groups with <= this many cells: build the tree on the RAW cell vectors

    def _ward_model(g):
        if ward[g] is None:
            from scipy.cluster.hierarchy import linkage
            n_cells = gX[g].shape[0]
            if n_cells <= max(_WARD_RAW_MAX, int(gKmax[g])):
                # SMALL group: cluster the raw cell share-vectors directly — no k-means
                # summarisation step, so the tree cut is a true hierarchical clustering of the
                # cells and keeps more fidelity. Each cell is its own leaf.
                fine_c = gX[g]
                fine_labels = np.arange(n_cells)
                fv = np.asarray(gW[g], float)
            else:
                # LARGE group: summarise with one k-means at k_max first (keeps the tree small).
                kfine = int(gKmax[g])
                km, _ = _fit_k(gX[g], gW[g], kfine, seed)
                fine_c = km.cluster_centers_
                fine_labels = km.labels_
                fv = np.zeros(len(fine_c))
                np.add.at(fv, km.labels_, gW[g])       # volume behind each fine cluster
            Z = linkage(fine_c, method="ward") if len(fine_c) >= 2 else None
            ward[g] = (fine_labels, fine_c, fv, Z)
        return ward[g]

    def _labels_centroids(g, k):
        k = int(min(max(k, 1), gKmax[g]))
        if method == "ward":
            fine_labels, fine_c, fv, Z = _ward_model(g)
            if Z is None or k >= len(fine_c):
                fine_to_coarse = np.arange(len(fine_c))
            else:
                from scipy.cluster.hierarchy import fcluster
                fine_to_coarse = fcluster(Z, t=k, criterion="maxclust") - 1
            n_coarse = int(fine_to_coarse.max()) + 1
            cent = np.zeros((n_coarse, gX[g].shape[1])); wsum = np.zeros(n_coarse)
            for fc in range(len(fine_c)):               # volume-weighted merge of fine leaves
                cc = fine_to_coarse[fc]
                cent[cc] += fine_c[fc] * fv[fc]; wsum[cc] += fv[fc]
            cent = np.where(wsum[:, None] > 1e-12, cent / np.maximum(wsum[:, None], 1e-12), cent)
            return fine_to_coarse[fine_labels], cent
        km, _ = _fit_k(gX[g], gW[g], k, seed)
        return km.labels_, km.cluster_centers_

    _acc_cache = [dict() for _ in range(G)]

    def _acc(g, k):
        k = int(min(max(k, 1), gKmax[g]))
        if k not in _acc_cache[g]:
            labels, cent = _labels_centroids(g, k)
            _acc_cache[g][k] = _weighted_accuracy(gX[g], cent[labels], gW[g])
        return _acc_cache[g][k]

    # --- choose clusters-per-group (kcur) -------------------------------------------------
    if allocation == "knapsack":
        # Build each group's accuracy curve, stopping at the plateau so the DP stays small.
        PLATEAU_EPS, PATIENCE = 0.01, 3
        curves = []
        for g in range(G):
            accs = [_acc(g, 1)]; stall = 0
            for k in range(2, gKmax[g] + 1):
                a = _acc(g, k); gain = a - max(accs); accs.append(a)
                stall = stall + 1 if gain < PLATEAU_EPS else 0
                if stall >= PATIENCE:
                    break
            curves.append(np.asarray(accs, float))       # index e (=k-1) -> accuracy
        maxE = [len(c) - 1 for c in curves]
        R = max(0, int(min(n_budget - G, sum(maxE))))
        NEG = -1e18
        dp = np.full(R + 1, NEG); dp[0] = 0.0
        back = []
        for g in range(G):
            vals = gVol[g] * curves[g]                    # value of giving group g e extras
            ndp = np.full(R + 1, NEG); pe = np.zeros(R + 1, dtype=int)
            for b in range(R + 1):
                emax = min(maxE[g], b)
                cand = dp[b - np.arange(emax + 1)] + vals[:emax + 1]
                j = int(np.argmax(cand)); ndp[b] = cand[j]; pe[b] = j
            dp = ndp; back.append(pe)
        b = int(np.argmax(dp)); extras = [0] * G          # best total extras used (≤ R)
        for g in range(G - 1, -1, -1):
            e = int(back[g][b]); extras[g] = e; b -= e
        kcur = [1 + extras[g] for g in range(G)]
    else:                                                 # greedy marginal-gain, method-aware
        import heapq
        kcur = [1] * G; heap = []

        def _push(g):
            if kcur[g] >= gKmax[g]:
                return
            gain = (gVol[g] / total_vol) * max(0.0, _acc(g, kcur[g] + 1) - _acc(g, kcur[g]))
            heapq.heappush(heap, (-gain, g))
        for g in range(G):
            _push(g)
        remaining = n_budget - G
        while remaining > 0 and heap:
            neg, g = heapq.heappop(heap)
            if -neg <= 1e-9:
                break
            if kcur[g] >= gKmax[g]:
                continue
            kcur[g] += 1; remaining -= 1; _push(g)

    # --- build the expanded (centroid) split + stats --------------------------------------
    out_rows, per_group = [], []
    _pr_num, _pr_den = {}, {}
    clusters_used = []
    for g, (gkey, grp) in enumerate(groups):
        labels, cent = _labels_centroids(g, kcur[g])
        acc = _acc(g, kcur[g])
        centroids = [_cap_and_respill(cent[cl], max_gateway_cap) for cl in range(cent.shape[0])]
        grp = grp.reset_index(drop=True)
        rpgt = str(grp["rpgt"].iloc[0])
        for i in range(len(grp)):
            cvec = centroids[labels[i]]
            base = {c: grp[c].iloc[i] for c in idx_cols}
            base["cell_volume"] = float(grp["_vol"].iloc[i])
            for gc, val in zip(gateway_cols, cvec):
                if val > 1e-9:
                    r = dict(base); r["gateway"] = gc; r["share"] = float(val)
                    out_rows.append(r)
        clusters_used.append(int(cent.shape[0]))
        per_group.append({**dict(zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,))),
                          "cells": int(len(grp)), "clusters": int(cent.shape[0]),
                          "accuracy": round(acc, 2), "volume": float(gVol[g])})
        _pr_num[rpgt] = _pr_num.get(rpgt, 0.0) + acc * gVol[g]
        _pr_den[rpgt] = _pr_den.get(rpgt, 0.0) + gVol[g]

    compressed_long = pd.DataFrame(out_rows)
    per_rpgt = {rp: round(_pr_num[rp] / max(_pr_den[rp], 1e-9), 2) for rp in _pr_num}
    stats = {
        "raw_rules": raw_rules,
        "compressed_rules": int(sum(clusters_used)),
        "global_accuracy": round(sum(pg["accuracy"] * pg["volume"] for pg in per_group) / total_vol, 2),
        "per_group": per_group, "per_rpgt": per_rpgt, "curve": [], "n_groups": G,
        "budget": n_budget, "method": method, "allocation": allocation,
    }
    return compressed_long, stats, tuple(clusters_used)


def compress_to_budget(split: pd.DataFrame, n_configs: int,
                       group_keys=("rpgt", "currency"),
                       max_gateway_cap: float = MAX_GATEWAY_CAP,
                       k_max: int = 60, seed: int = 42,
                       method: str = "kmeans", allocation: str = "greedy"):
    """Compress a per-cell split to ~n_configs representative rules TOTAL by greedily
    allocating clusters across the (group_keys) groups to maximise ONE global,
    VOLUME-WEIGHTED fidelity across every cell — so high-volume RPGTs (e.g. Monthly
    Initial) get clusters first (95% there is worth ~20× a low-volume RPGT).

    Faithful by construction: each cell's shares are replaced by its cluster centroid
    (capped at max_gateway_cap), and the reported accuracy is the volume-weighted
    fraction of traffic still routed as the uncompressed split intended.

    Returns (compressed_long, stats):
      compressed_long : long-format split (rpgt, currency, bank[, pmp], gateway, share,
                        cell_volume) with each cell set to its centroid — feed to the
                        exporter so identical centroids collapse into one config each.
      stats           : {raw_rules, compressed_rules, global_accuracy, per_group,
                         per_rpgt, curve, n_groups, budget}. `curve` = [(total_clusters,
                         global_accuracy), …] so the UI can show the accuracy↔count knee.
    """
    ctx = _build_compress_context(split, group_keys, max_gateway_cap, k_max, seed,
                                  method=method, allocation=allocation)
    compressed_long, stats, _ = _compress_with_context(ctx, n_configs)
    return compressed_long, stats
