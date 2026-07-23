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

__build__ = "2026-07-18-shared-fit-cache-exact"

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
        chosen_k, chosen_km, chosen_recon, chosen_acc = 1, *(_fit_k(X, w, 1, seed)), None
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
                            k_max: int = 60, seed: int = 42):
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
    _ctx = _build_compress_context(split, group_keys, max_gateway_cap, k_max, seed)
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

    # Largest budget b in [1, raw_cells] whose generated pools <= target.
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


def _build_compress_context(split: pd.DataFrame, group_keys, max_gateway_cap, k_max, seed):
    """Precompute everything that DOESN'T depend on the cluster budget: the volume-weighted
    share matrix, the per-group arrays, and an empty (group, k) -> (km, acc) fit cache.

    The KMeans fit for a given (group, k) is fully deterministic here (fixed seed, fixed
    n_init, same X and weights), so this context — including its `fits` cache — can be
    reused across many budgets to give IDENTICAL results with no recomputation.
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
    }


def _compress_with_context(ctx, n_configs):
    """Greedy volume-weighted cluster allocation for a given budget, using a prebuilt
    context (so KMeans fits are shared/cached across budgets). Returns
    (compressed_long, stats, kcur_tuple). Behaviour is identical to a fresh
    compress_to_budget call — only the redundant matrix build + refits are avoided.
    """
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


def compress_to_budget(split: pd.DataFrame, n_configs: int,
                       group_keys=("rpgt", "currency"),
                       max_gateway_cap: float = MAX_GATEWAY_CAP,
                       k_max: int = 60, seed: int = 42):
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
    ctx = _build_compress_context(split, group_keys, max_gateway_cap, k_max, seed)
    compressed_long, stats, _ = _compress_with_context(ctx, n_configs)
    return compressed_long, stats
