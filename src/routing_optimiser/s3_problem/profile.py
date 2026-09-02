"""Sub-cell grain helpers (Stage 1 of moving the optimiser DECISION grain from
cell = bank x currency x RPGT  →  sub-cell = bank x currency x RPGT x pmp x Country.

Design (locked with the user):
  * SCORING (success-rate) grain stays at CELL — success rates are NOT split by pmp/Country;
    they are BROADCAST unchanged onto each sub-cell (removes the data-sparsity risk).
  * The DECISION grain moves to sub-cell, so the GA can route sub-cells differently to satisfy
    their different VAMP exposure + eligibility (wallet / USA-Non-USA).
  * The forecast pipeline / pro-rata export are UNTOUCHED. Sub-cell VOLUME comes from the
    pro-rata export's baseline VI-Txn split (the "volume glue"): each cell's forecast volume is
    apportioned across its (pmp, Country) sub-cells by the sub-cell's share of the cell's
    baseline VI-Txn at t=0.

This module is PURE (no Streamlit / BigQuery) so it is unit-testable in isolation. The tab-2
assembly wires these in behind a new grain option; the exact-band VAMP projector and the
eligibility masks are already sub-cell-native and are applied by the existing paths.

`__build__` is logged at run start so stale bytecode is obvious.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__build__ = "2026-08-16-subcell-volume-glue"

_ALL = "_all_"   # sentinel profile label when the export has no pmp/Country split for a profile


# [FN-SC01]
def profile_vi_fractions(prorata: pd.DataFrame,
                         *, currency="Currency", bin_="BIN", rpgt="RPGT",
                         pmp="paymentMethodProvider", country="Country",
                         vi="VI_Txn_Count", t="t") -> pd.DataFrame:
    """Per (currency, bank, rpgt) CELL, the fraction of baseline VI-Txn volume sitting in each
    (pmp, Country) SUB-CELL — the weights used to apportion the cell's forecast volume.

    Uses the t==0 (origination) rows of the pro-rata export, summed over vampMid, so `vi` is the
    sub-cell's baseline transaction volume. Returns a tidy frame:
        cur, bank, rpgt, pmp, ctry, vi, vi_frac
    with keys lower-cased/stripped (bank kept as a trimmed string) and `vi_frac` summing to 1.0
    within each (cur, bank, rpgt) cell. A cell whose total VI is 0 collapses to a single
    '_all_'/'_all_' sub-cell with vi_frac = 1.0 (so it behaves exactly like today's cell grain).
    """
    df = prorata
    cur = df[currency].astype(str).str.strip().str.lower()
    bnk = df[bin_].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    rpg = df[rpgt].astype(str).str.strip().str.lower()
    pm = (df[pmp].astype(str).str.strip().str.lower() if pmp in df.columns
          else pd.Series(_ALL, index=df.index))
    ct = (df[country].astype(str).str.strip().str.lower() if country in df.columns
          else pd.Series(_ALL, index=df.index))
    viv = pd.to_numeric(df[vi], errors="coerce").fillna(0.0) if vi in df.columns else pd.Series(0.0, index=df.index)
    tt = pd.to_numeric(df[t], errors="coerce").fillna(0).astype(int) if t in df.columns else pd.Series(0, index=df.index)

    g = pd.DataFrame({"cur": cur, "bin": bnk, "rpgt": rpg, "pmp": pm, "ctry": ct,
                      "vi": viv, "t": tt})
    g = g[g["t"] == 0]
    sub = g.groupby(["cur", "bin", "rpgt", "pmp", "ctry"], as_index=False)["vi"].sum()
    profile_tot = sub.groupby(["cur", "bin", "rpgt"])["vi"].transform("sum")
    # Profiles with positive VI: fraction = profile VI / profile VI.
    pos = profile_tot > 0
    sub["vi_frac"] = np.where(pos, sub["vi"] / profile_tot.where(pos, 1.0), np.nan)

    # Profiles with zero VI (or absent from the export): one '_all_' profile, frac 1.0.
    zero_profiles = (sub.loc[~pos, ["cur", "bin", "rpgt"]].drop_duplicates())
    if len(zero_profiles):
        zero_profiles = zero_profiles.assign(pmp=_ALL, ctry=_ALL, vi=0.0, vi_frac=1.0)
        sub = pd.concat([sub[pos], zero_profiles], ignore_index=True)
    return sub[["cur", "bin", "rpgt", "pmp", "ctry", "vi", "vi_frac"]].reset_index(drop=True)


# [FN-SC02]
def expand_forecast_to_profiles(forecast: pd.DataFrame, fractions: pd.DataFrame,
                                *, currency="currency", bin_="bin", rpgt="rpgt",
                                volume="volume") -> pd.DataFrame:
    """Replicate each cell's rows across its (pmp, Country) sub-cells, apportioning the cell's
    forecast VOLUME by `fractions` (from :func:`profile_vi_fractions`). Every other column
    (gateway, baseline_share, success/risk rates, …) is BROADCAST unchanged — scoring stays at
    cell grain. Adds `pmp` and `ctry` columns and a `profile` key.

    Volume conservation: the sum of `volume` over a cell's sub-cell rows equals the original cell
    volume (fractions sum to 1 per cell). Rows whose cell has no entry in `fractions` are kept as a
    single '_all_'/'_all_' sub-cell (unchanged), so nothing is dropped.
    """
    f = forecast.copy()
    f["_cur"] = f[currency].astype(str).str.strip().str.lower()
    f["_bin"] = f[bin_].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    f["_rpgt"] = (f[rpgt].astype(str).str.strip().str.lower() if rpgt in f.columns
                  else pd.Series("all_rpgts", index=f.index))

    fr = fractions[["cur", "bin", "rpgt", "pmp", "ctry", "vi_frac"]].rename(
        columns={"cur": "_cur", "bin": "_bin", "rpgt": "_rpgt"})
    merged = f.merge(fr, on=["_cur", "_bin", "_rpgt"], how="left")

    # Profiles absent from `fractions` → single '_all_' profile, frac 1.0 (behaves like profile grain).
    miss = merged["vi_frac"].isna()
    merged.loc[miss, "pmp"] = _ALL
    merged.loc[miss, "ctry"] = _ALL
    merged.loc[miss, "vi_frac"] = 1.0

    merged[volume] = pd.to_numeric(merged[volume], errors="coerce").fillna(0.0) * merged["vi_frac"]
    merged["profile"] = (merged["_cur"] + "|" + merged["_bin"] + "|" + merged["_rpgt"]
                         + "|" + merged["pmp"].astype(str) + "|" + merged["ctry"].astype(str))
    return merged.drop(columns=["_cur", "_bin", "_rpgt", "vi_frac"]).reset_index(drop=True)
