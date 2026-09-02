"""
Compare the proposed split against the baseline ("pre") and quantify impact,
from both a success-rate/revenue angle and a risk angle. Feeds the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# [FN-148]
def cell_baseline_vs_proposed(split: pd.DataFrame,
                              avg_ticket: dict | float = 25.0) -> pd.DataFrame:
    """
    Per cell: expected successful transactions and revenue under the baseline
    split vs the proposed split, plus the incremental (uplift) figures.

    avg_ticket: average amount per transaction, either a flat float or a dict
    keyed by rpgt. Multiply successful attempts by this to get revenue.
    """
    # [FN-149]
    _default_ticket = (float(np.mean(list(avg_ticket.values()) or [25.0]))
                       if isinstance(avg_ticket, dict) else None)   # rpgt-independent — hoisted

    def ticket(rpgt):
        if isinstance(avg_ticket, dict):
            return float(avg_ticket.get(rpgt, _default_ticket))
        return float(avg_ticket)

    # Expected success rate per profile under each split (volume/ share weighted).
    g = split.copy()
    g["proposed_succ"] = g["share"] * g["gateway_success_rate"]
    g["baseline_succ"] = g["baseline_share"] * g["gateway_success_rate"]

    cell = (g.groupby(["rpgt", "currency", "bin"], as_index=False)
            .agg(cell_volume=("cell_volume", "first"),
                 proposed_rate=("proposed_succ", "sum"),
                 baseline_rate=("baseline_succ", "sum")))

    cell["ticket"] = cell["rpgt"].map(ticket)
    cell["baseline_success_txns"] = cell["baseline_rate"] * cell["cell_volume"]
    cell["proposed_success_txns"] = cell["proposed_rate"] * cell["cell_volume"]
    cell["incremental_success_txns"] = cell["proposed_success_txns"] - cell["baseline_success_txns"]
    cell["incremental_revenue"] = cell["incremental_success_txns"] * cell["ticket"]
    cell["rate_uplift_pp"] = (cell["proposed_rate"] - cell["baseline_rate"]) * 100
    return cell


# [FN-150]
def headline_impact(cell: pd.DataFrame) -> dict:
    vol = cell["cell_volume"].sum()
    base_rate = (cell["baseline_rate"] * cell["cell_volume"]).sum() / max(vol, 1)
    prop_rate = (cell["proposed_rate"] * cell["cell_volume"]).sum() / max(vol, 1)
    return {
        "baseline_success_rate": float(base_rate),
        "proposed_success_rate": float(prop_rate),
        "success_rate_uplift_pp": float((prop_rate - base_rate) * 100),
        "incremental_success_txns": float(cell["incremental_success_txns"].sum()),
        "incremental_revenue": float(cell["incremental_revenue"].sum()),
    }


# [FN-151]
def key_contributors(cell: pd.DataFrame, by: str = "bin", top: int = 10) -> pd.DataFrame:
    """Which banks / currencies / RPGTs drive most of the incremental revenue."""
    agg = (cell.groupby(by, as_index=False)
           .agg(incremental_revenue=("incremental_revenue", "sum"),
                incremental_success_txns=("incremental_success_txns", "sum"),
                cell_volume=("cell_volume", "sum")))
    agg = agg.sort_values("incremental_revenue", ascending=False)
    total = agg["incremental_revenue"].sum()
    agg["pct_of_uplift"] = np.where(total != 0,
                                    100 * agg["incremental_revenue"] / total, 0.0)
    return agg.head(top).reset_index(drop=True)


# [FN-152]
def gateway_volume_shift(split: pd.DataFrame) -> pd.DataFrame:
    """How much volume each gateway gains/loses vs baseline (the 'stolen'
    volume view from your VAMP guide)."""
    g = split.copy()
    _v = _split_volume(g)                                   # tolerate 'volume' or 'cell_volume'
    g["proposed_volume"] = g["share"] * _v
    g["baseline_volume"] = g["baseline_share"] * _v
    out = (g.groupby("gateway", as_index=False)
           .agg(baseline_volume=("baseline_volume", "sum"),
                proposed_volume=("proposed_volume", "sum")))
    out["delta_volume"] = out["proposed_volume"] - out["baseline_volume"]
    return out.sort_values("delta_volume", ascending=False).reset_index(drop=True)


# [FN-153]
def _split_volume(df: pd.DataFrame) -> pd.Series:
    """Per-row cell volume from a split frame, tolerating either column name."""
    col = "volume" if "volume" in df.columns else ("cell_volume" if "cell_volume" in df.columns else None)
    if col is None:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


# [FN-154]
def gateway_move_vs_reference(ref_split: pd.DataFrame, sel_split: pd.DataFrame,
                              keys=("rpgt", "currency", "bin", "gateway")) -> pd.DataFrame:
    """Per-gateway volume BEFORE (`ref_split`) vs AFTER (`sel_split`), aligned on the
    cell×gateway grain. Use with the revenue reference (dial 100) as `ref_split` and the
    selected compliant split as `sel_split` to see the traffic moved to meet constraints.

    Returns one row per gateway: ref_volume, prop_volume, delta_volume (+gained / −shed),
    ref_share, prop_share, delta_share — sorted by delta_volume descending.
    """
    kk = [k for k in keys if k in ref_split.columns and k in sel_split.columns]
    a = ref_split.copy(); b = sel_split.copy()
    a["_v"] = _split_volume(a); b["_v"] = _split_volume(b)
    a["_refv"] = pd.to_numeric(a.get("share", 0), errors="coerce").fillna(0.0) * a["_v"]
    b["_propv"] = pd.to_numeric(b.get("share", 0), errors="coerce").fillna(0.0) * b["_v"]
    m = a[kk + ["_refv"]].merge(b[kk + ["_propv"]], on=kk, how="outer")
    m[["_refv", "_propv"]] = m[["_refv", "_propv"]].fillna(0.0)
    g = m.groupby("gateway", as_index=False).agg(ref_volume=("_refv", "sum"),
                                                 prop_volume=("_propv", "sum"))
    _tr = max(float(g["ref_volume"].sum()), 1e-9)
    _tp = max(float(g["prop_volume"].sum()), 1e-9)
    g["delta_volume"] = g["prop_volume"] - g["ref_volume"]
    g["ref_share"] = g["ref_volume"] / _tr
    g["prop_share"] = g["prop_volume"] / _tp
    g["delta_share"] = g["prop_share"] - g["ref_share"]
    return g.sort_values("delta_volume", ascending=False).reset_index(drop=True)


# [FN-155]
def traffic_moved_curve(variations, ref_weight=None) -> pd.DataFrame:
    """Fraction of total volume moved vs the revenue reference, for every dial position.

    The reference is the max-weight variation (dial 100) unless `ref_weight` names one.
    For each variation, moved% = ½·Σ_gateway |Δvolume| / total — the share of book that had
    to be re-routed relative to the unconstrained-optimal split. Rising as the dial tightens
    is the compliance-cost curve. Returns df: dial (0–100), moved_pct.
    """
    vs = [v for v in (variations or []) if isinstance(v, dict) and v.get("split") is not None]
    if not vs:
        return pd.DataFrame(columns=["dial", "moved_pct"])
    if ref_weight is not None:
        ref = min(vs, key=lambda v: abs(float(v.get("weight", 0)) - float(ref_weight)))["split"]
    else:
        ref = max(vs, key=lambda v: float(v.get("weight", 0)))["split"]
    rows = []
    for v in vs:
        g = gateway_move_vs_reference(ref, v["split"])
        tot = max(float(g["ref_volume"].sum()), 1e-9)
        moved = 0.5 * float(g["delta_volume"].abs().sum()) / tot * 100.0
        rows.append({"dial": float(v.get("weight", 0)) * 100.0, "moved_pct": moved})
    return pd.DataFrame(rows).sort_values("dial", ascending=False).reset_index(drop=True)
