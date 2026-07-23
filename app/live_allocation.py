"""Run the REAL VAMP AllocationEngine on a chosen split, reusing tab 1's cached forecast.

The forecast (BigQuery extract + ActuarialEngine) is split-INDEPENDENT, so run_vamp_pipeline
caches its output (_actuarial_attempts.parquet + _mid_df.parquet + _mr_weights.pkl +
_pipeline_config.json). This module loads those and runs ONLY phases 3-4 (AllocationEngine +
ExportManager) on a split — no BigQuery, no re-forecast — producing the EXACT same mid_level
pre/post that tab 5 (the full pipeline) would for that split.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import pandas as pd

__build__ = "2026-07-19-live-allocation"

_ARTIFACTS = ("_actuarial_attempts.parquet", "_mid_df.parquet",
              "_mr_weights.pkl", "_pipeline_config.json")


def artifacts_present(forecast_dir: str) -> bool:
    """True iff the cached forecast artifacts exist (i.e. exact mode is available)."""
    return bool(forecast_dir) and all(
        os.path.exists(os.path.join(forecast_dir, f)) for f in _ARTIFACTS)


def _templates_to_split_df(templates: dict, config: dict) -> pd.DataFrame:
    """Melt the wide build_split_exports templates into the long split_df the
    AllocationEngine consumes — mirrors data_extractor._clean_google_sheets_rules."""
    frames = []
    for _key, wdf in (templates or {}).items():
        df = wdf.copy()
        df.columns = df.columns.astype(str).str.strip()
        df = df.loc[:, ~df.columns.duplicated()]
        drop = [c for c in df.columns
                if (c.upper() in ("BIN GROUP", "DUP CHECK") or "DUP CHECK" in c.upper())
                and c != "Check"] + ["Share_Str", "share_str", "Share"]
        df = df.drop(columns=[c for c in drop if c in df.columns])
        ren = {"Company": "Brand", "company": "Brand", "rpgt": "RPGT", "currency": "Currency",
               "bin": "BIN", "paymentmethodprovider": "paymentMethodProvider", "country": "Country"}
        df = df.rename(columns=ren)
        for c in ("Brand", "RPGT", "Currency", "BIN", "paymentMethodProvider", "Country"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower().str.strip()
                if c == "BIN":
                    df[c] = df[c].str.replace(r"\.0$", "", regex=True)
        id_vars = [c for c in ("Brand", "RPGT", "Currency", "BIN", "paymentMethodProvider",
                               "Country", "Check", "STICKY", "GO LIVE") if c in df.columns]
        m = df.melt(id_vars=id_vars, var_name="gatewayFid", value_name="Share")
        m["Share"] = pd.to_numeric(
            m["Share"].astype(str).str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False).str.strip(), errors="coerce").fillna(0)
        m = m[m["Share"] > 0].copy()
        m["Rule_Source"] = "Specific"
        frames.append(m)
    sdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    fa = (config.get("filters", {}) or {}).get("force_actuals_for_rpgts", [])
    if fa and not sdf.empty:
        _mask = sdf["RPGT"].astype(str).str.lower().isin([str(r).strip().lower() for r in fa])
        sdf = sdf[~_mask].copy()
    return sdf


def run_exact_mid_level(project_root: str, forecast_dir: str, split, brand_name: str,
                        go_live, wallet_ctx: dict, mid_list_path: str) -> pd.DataFrame:
    """Run AllocationEngine + ExportManager on `split` using the cached forecast in
    forecast_dir. Returns the mid_level pre/post DataFrame (identical to tab 5's for this
    split). No BigQuery / no re-forecast. Raises on failure (caller handles)."""
    fa = pd.read_parquet(os.path.join(forecast_dir, "_actuarial_attempts.parquet"))
    mid = pd.read_parquet(os.path.join(forecast_dir, "_mid_df.parquet"))
    with open(os.path.join(forecast_dir, "_mr_weights.pkl"), "rb") as _f:
        mrw = pickle.load(_f)
    with open(os.path.join(forecast_dir, "_pipeline_config.json")) as _f:
        cfg = json.load(_f)

    project_root = os.path.abspath(project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))
    from vamp_pipeline import AllocationEngine, ExportManager  # vendored pipeline
    from impact_calcs import build_split_exports

    wc = wallet_ctx or {}
    templates = build_split_exports(
        split, brand_name, str(go_live),
        wallet_incapable=set(wc.get("incapable", set())), fid2vamp=wc.get("fid2vamp"),
        mid_list_path=mid_list_path, usa_only=set(wc.get("usa_only", set())),
        country_pres=wc.get("country_pres", {}), max_share=float(wc.get("max_share", 0.97)))
    split_df = _templates_to_split_df(templates, cfg)

    # Write to an ISOLATED temp dir so nothing clobbers the live outputs tab 3 reads.
    cfg = dict(cfg)
    cfg.setdefault("paths", {})
    tmp_out = os.path.join(forecast_dir, "_live_alloc")
    os.makedirs(tmp_out, exist_ok=True)
    cfg["paths"]["output_dir"] = tmp_out + os.sep     # absolute -> .format() leaves it as-is

    prev_cwd = os.getcwd()
    os.chdir(project_root)                              # so any relative reads resolve
    try:
        alloc = AllocationEngine(config=cfg, attempts_df=fa, split_df=split_df, mr_weights=mrw)
        pre_df, post_df = alloc.execute_time_aware_routing()
        exporter = ExportManager(config=cfg, mid_df=mid, attempts_df=fa, mr_weights=mrw)
        exporter.run_all_exports(pre_df, post_df)
        return pd.read_csv(os.path.join(tmp_out, "mid_level.csv"))
    finally:
        os.chdir(prev_cwd)
