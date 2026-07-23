"""
Compare the proposed split against the baseline ("pre") and quantify impact,
from both a success-rate/revenue angle and a risk angle. Feeds the dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cell_baseline_vs_proposed(split: pd.DataFrame,
                              avg_ticket: dict | float = 25.0) -> pd.DataFrame:
    """
    Per cell: expected successful transactions and revenue under the baseline
    split vs the proposed split, plus the incremental (uplift) figures.

    avg_ticket: average amount per transaction, either a flat float or a dict
    keyed by rpgt. Multiply successful attempts by this to get revenue.
    """
    def ticket(rpgt):
        if isinstance(avg_ticket, dict):
            return float(avg_ticket.get(rpgt, np.mean(list(avg_ticket.values()) or [25.0])))
        return float(avg_ticket)

    # Expected success rate per cell under each split (volume/ share weighted).
    g = split.copy()
    g["proposed_succ"] = g["share"] * g["gateway_success_rate"]
    g["baseline_succ"] = g["baseline_share"] * g["gateway_success_rate"]

    cell = (g.groupby(["rpgt", "currency", "bank"], as_index=False)
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


def key_contributors(cell: pd.DataFrame, by: str = "bank", top: int = 10) -> pd.DataFrame:
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


def gateway_volume_shift(split: pd.DataFrame) -> pd.DataFrame:
    """How much volume each gateway gains/loses vs baseline (the 'stolen'
    volume view from your VAMP guide)."""
    g = split.copy()
    g["proposed_volume"] = g["share"] * g["cell_volume"]
    g["baseline_volume"] = g["baseline_share"] * g["cell_volume"]
    out = (g.groupby("gateway", as_index=False)
           .agg(baseline_volume=("baseline_volume", "sum"),
                proposed_volume=("proposed_volume", "sum")))
    out["delta_volume"] = out["proposed_volume"] - out["baseline_volume"]
    return out.sort_values("delta_volume", ascending=False).reset_index(drop=True)
