"""Menu / split-picker compression (OPT-IN, additive) — a Tab 5 alternative to k-means.

Instead of forcing every cell onto its k-means cluster centroid, prepare a SHORTLIST
("menu") of ready-made candidate splits and let each cell INDEPENDENTLY pick the allowed
menu item that best matches its own ideal split. A cell that fits no cluster well is not
dragged into one — it simply picks whatever menu item is closest. A distinct-item budget
then trims the menu to at most `max_items` splits (the deployable count) by dropping the
least-used items and reassigning their cells to their next-best surviving allowed item.

Adapted from the co-worker GA's "menu" mode. Because per-cell fidelity is separable, the
pick itself is a direct argmin (no GA needed); only the distinct-item budget is (greedily)
cross-cell. Fully self-contained: importing this module does not change any existing path.

Entry point: `menu_compress(split_long, ...)` returns (compressed_long, stats) with the same
column contract as kmeans_compress.compress_to_budget, so it can feed the same exporter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .kmeans_compress import _cap_and_respill, _weighted_accuracy

__build__ = "2026-07-24-menu-picker"


def _cell_matrix(split: pd.DataFrame, idx_cols, gateway_cols):
    """(cells x gateways) share matrix (renormalised) + per-cell volume, aligned to
    `gateway_cols`."""
    mat = (split.pivot_table(index=idx_cols, columns="gateway", values="share", aggfunc="sum")
           .fillna(0.0))
    for gc in gateway_cols:                       # guarantee every menu gateway is present
        if gc not in mat.columns:
            mat[gc] = 0.0
    mat = mat[gateway_cols]
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    vol = split.groupby(idx_cols)["cell_volume"].first().reindex(mat.index).fillna(0.0)
    return mat, vol


def _build_group_menu(X: np.ndarray, w: np.ndarray, menu_k: int, seed: int) -> np.ndarray:
    """Shortlist of candidate split vectors for one group: volume-weighted KMeans centroids
    at k = min(menu_k, n_cells). These are the 'ready-made splits' cells choose between."""
    k = int(min(max(menu_k, 1), len(X)))
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X, sample_weight=w)
    return km.cluster_centers_


def menu_compress(split: pd.DataFrame,
                  group_keys=("rpgt", "currency"),
                  menu_k: int = 12,
                  max_items: int | None = None,
                  max_gateway_cap: float = 0.97,
                  seed: int = 42):
    """Compress a per-cell split by MENU PICKING.

    Parameters
    ----------
    split : long split (rpgt, currency, bank[, pmp], gateway, share, cell_volume).
    group_keys : cells only pick menu items built from their OWN group (preserves the
        currency / wallet structure, exactly like the k-means path's grouping).
    menu_k : menu size per group (candidate splits offered to that group's cells).
    max_items : optional GLOBAL cap on the number of DISTINCT menu items used across all
        groups (the deployable split count). None = no cap (every used item survives).
    max_gateway_cap : per-gateway cap applied to the emitted split (mirrors the k-means path).

    Returns (compressed_long, stats):
      compressed_long : each cell set to its chosen (capped) menu split — feed to the exporter.
      stats : {raw_rules, compressed_rules, global_accuracy, per_group, n_groups, method}.
    """
    group_keys = list(group_keys)
    has_pmp = "pmp" in split.columns
    idx_cols = ["rpgt", "currency", "bank"] + (["pmp"] if has_pmp else [])
    if has_pmp and "pmp" not in group_keys:
        group_keys = group_keys + ["pmp"]

    gateway_cols = sorted(split["gateway"].astype(str).unique().tolist())
    mat, vol = _cell_matrix(split, idx_cols, gateway_cols)
    md = mat.reset_index()
    md["_vol"] = vol.to_numpy()

    groups = list(md.groupby(group_keys))
    total_vol = float(md["_vol"].sum()) or 1.0

    # Per-cell state, flat across all groups. Each cell knows its group, its ideal vector,
    # its volume, its group's menu, and its PREFERENCE ORDER over that menu (for reassignment).
    cells = []          # list of dicts
    group_menus = []    # menu array per group index
    group_meta = []     # (gkey, idx_frame) per group index
    for gi, (gkey, grp) in enumerate(groups):
        grp = grp.reset_index(drop=True)
        Xg = grp[gateway_cols].to_numpy(float)
        wg = np.maximum(grp["_vol"].to_numpy(float), 1e-6)
        menu = _build_group_menu(Xg, wg, menu_k, seed)      # (mg, n_gw)
        group_menus.append(menu)
        group_meta.append((gkey, grp))
        # L1 distance from every cell to every menu item -> preference order (nearest first).
        d = np.abs(Xg[:, None, :] - menu[None, :, :]).sum(axis=2)   # (ng, mg)
        pref = np.argsort(d, axis=1)                                # nearest menu item first
        for i in range(len(grp)):
            cells.append({"g": gi, "x": Xg[i], "vol": float(wg[i]),
                          "pref": pref[i].tolist(), "choice": int(pref[i][0])})

    def _item_id(c):
        return (c["g"], c["choice"])

    def _distinct_items():
        return {(_item_id(c)) for c in cells}

    # --- distinct-item budget: greedily drop the least-used item and reassign its cells ----
    if max_items is not None and max_items > 0:
        dropped = set()                                   # (g, j) items no longer offered

        def _next_available(c):
            for j in c["pref"]:
                if (c["g"], j) not in dropped:
                    return int(j)
            return int(c["choice"])                       # never fully strand a cell

        # ensure everyone points at an available item first
        for c in cells:
            c["choice"] = _next_available(c)

        while True:
            used = {}
            per_group_used = {}
            for c in cells:
                iid = _item_id(c)
                used[iid] = used.get(iid, 0.0) + c["vol"]
                per_group_used.setdefault(c["g"], set()).add(c["choice"])
            if len(used) <= max_items:
                break
            # droppable = items whose GROUP still has >=2 distinct items used (so its cells can
            # move elsewhere within the group). Pick the least-volume such item.
            droppable = [(v, iid) for iid, v in used.items()
                         if len(per_group_used[iid[0]]) >= 2]
            if not droppable:
                break                                     # can't trim further without emptying a group
            droppable.sort()
            _, victim = droppable[0]
            dropped.add(victim)
            for c in cells:
                if _item_id(c) == victim:
                    c["choice"] = _next_available(c)

    # --- build the expanded split + stats --------------------------------------------------
    out_rows, per_group = [], []
    acc_num = 0.0
    used_items_global = set()
    for gi, (gkey, grp) in enumerate(group_meta):
        menu = group_menus[gi]
        gcells = [c for c in cells if c["g"] == gi]
        Xg = np.array([c["x"] for c in gcells])
        wg = np.array([c["vol"] for c in gcells])
        recon = np.array([menu[c["choice"]] for c in gcells])         # uncapped, for accuracy
        acc_g = _weighted_accuracy(Xg, recon, wg)
        vol_g = float(wg.sum())
        acc_num += acc_g * vol_g
        used_g = sorted({c["choice"] for c in gcells})
        used_items_global |= {(gi, j) for j in used_g}
        capped = {j: _cap_and_respill(menu[j], max_gateway_cap) for j in used_g}
        for ci, c in enumerate(gcells):
            base = {col: grp[col].iloc[ci] for col in idx_cols}
            base["cell_volume"] = c["vol"]
            for gc, val in zip(gateway_cols, capped[c["choice"]]):
                if val > 1e-9:
                    r = dict(base); r["gateway"] = gc; r["share"] = float(val)
                    out_rows.append(r)
        per_group.append({**dict(zip(group_keys, gkey if isinstance(gkey, tuple) else (gkey,))),
                          "cells": int(len(gcells)), "items_used": int(len(used_g)),
                          "menu_size": int(len(menu)), "accuracy": round(acc_g, 2),
                          "volume": vol_g})

    compressed_long = pd.DataFrame(out_rows)
    stats = {
        "raw_rules": int(len(mat)),
        "compressed_rules": int(len(used_items_global)),
        "global_accuracy": round(acc_num / total_vol, 2),
        "per_group": per_group,
        "n_groups": len(groups),
        "method": "menu", "menu_k": int(menu_k),
        "max_items": (int(max_items) if max_items else None),
    }
    return compressed_long, stats
