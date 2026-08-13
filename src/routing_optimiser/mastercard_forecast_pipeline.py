"""
Adapter between the routing optimiser / Streamlit app and the MASTERCARD forecast
pipeline (vendored under src/mastercard_pipeline/). Mirrors forecast_pipeline.py
(the Visa/VAMP adapter) but targets the Mastercard settings schema and reads the
Mastercard baseline columns (Sim_CBs / CB_Pre) rather than the VAMP ones.

Three jobs:
  1. build_mc_pipeline_config    - map the Forecast tab's settings onto the exact
                                   settings_mastercard.yaml schema the pipeline expects.
  2. run_mastercard_pipeline     - run DataExtractor -> ActuarialEngine ->
                                   AllocationEngine -> ExportManager (needs BigQuery).
  3. load_mc_pre_forecast        - read the pipeline's 'pre' (do-nothing) output and
                                   normalise it into the optimiser's forecast contract
                                   (rpgt, currency, bank, gateway, volume,
                                   baseline_share, risk_rate). Dependency-free.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Pipeline "pre" (baseline / do-nothing) columns in effective_rate_impact.csv (Mastercard)
PRE_SALES = "Sim_Sales"
PRE_CBS = "Sim_CBs"
PRE_RATE = "Sim_Rate"


def build_mc_pipeline_config(ui: dict) -> dict:
    """Map the flat Forecast-tab settings onto the Mastercard pipeline's settings schema."""
    company = ui.get("company", "TotalAV")
    month_var = ui.get("month_var", "")
    m0 = ui.get("month_0")  # YYYY-MM-01
    scrub = ui.get("test_gateways") or {}
    scrub_list = scrub.get("scrub", scrub) if isinstance(scrub, dict) else scrub

    return {
        "run_settings": {
            "company": company,
            "month_var": month_var,
            "month_0_start_date": m0,
            "actuals_start_date": ui.get("start_date") or m0,
            "actuals_end_date": ui.get("end_date") or m0,
            "future_anchor_date": ui.get("future_anchor_date"),
            "blend_future_sheet_rules": bool(ui.get("future_anchor_date")),
            "use_chunked_csv_files": True,
            "load_curves_from_cache": bool(ui.get("reuse_cached_curves", True)),
            "use_live_actuals": bool(ui.get("use_live_actuals", True)),
            # Mastercard-specific switches
            "kill_4_bins": bool(ui.get("kill_4_bins", True)),
            "actuals_end_day": ui.get("actuals_end_day", 30),
            "split_go_live_date": ui.get("split_go_live_date"),
        },
        "paths": {
            "cache_path": "data/cache/{month_var}/{company}/mastercard/",
            "chunked_files_dir": "data/rules/{month_var}/{company}/mastercard/",
            "output_dir": "data/outputs/{month_var}/{company}/mastercard/",
            "split_rules_file": ui.get("split_rules_file", ""),
            "mid_list_file": ui.get("mid_list_file",
                                    "data/mappings/Master_MID_List_Mastercard.csv"),
        },
        "targets": {
            "company_target_volume": ui.get("m0_total_transactions"),
            "company_rpgt_target_volumes": ui.get("m0_transaction_weightings", {}),
        },
        "actuarial_settings": {
            "t0_lookback_months": ui.get("t0_lookback_months", 2),
            "decay_factor": ui.get("decay_factor", 0.5),
            "thermometer_sample_months": ui.get("thermometer_sample_months", 2),
        },
        "thermometer_config": ui.get("thermometer_config") or {},
        "gateway_volume_overrides": ui.get("gateway_volume_overrides") or {},
        "filters": {
            "test_gateways_to_scrub": scrub_list or [],
            "force_actuals_for_rpgts": ui.get("force_actuals_for", []),
        },
    }


def run_mastercard_pipeline(config: dict, project_root: str,
                            gcp_project: str | None = "sapient-tangent-172609") -> str:
    """
    Run the full Mastercard pipeline and return the output directory.

    Lazily imports the vendored pipeline (needs google-cloud-bigquery) and runs from
    project_root so the pipeline's relative 'queries/' path resolves.
    """
    import sys

    project_root = os.path.abspath(project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))
    from google.cloud import bigquery
    from mastercard_pipeline import (ActuarialEngine, AllocationEngine, DataExtractor,
                                     ExportManager)

    prev_cwd = os.getcwd()
    os.chdir(project_root)  # so 'queries/<file>.sql' resolves
    try:
        import copy
        config = copy.deepcopy(config)
        config.setdefault("paths", {})
        config["paths"]["queries_dir"] = os.path.join(project_root, "queries")
        mlf = config["paths"].get("mid_list_file") or "data/mappings/Master_MID_List_Mastercard.csv"
        if not os.path.isabs(mlf):
            mlf = os.path.join(project_root, mlf)
        config["paths"]["mid_list_file"] = mlf

        import mastercard_pipeline.data_extractor as _dex
        logger.info(f"MC ADAPTER: project_root={project_root}")
        logger.info(f"MC ADAPTER: data_extractor build = {getattr(_dex, '__build__', 'UNKNOWN (stale?)')}")
        logger.info(f"MC ADAPTER: queries_dir={config['paths']['queries_dir']} "
                    f"(exists={os.path.isdir(config['paths']['queries_dir'])})")
        logger.info(f"MC ADAPTER: mid_list_file={mlf} (exists={os.path.exists(mlf)})")

        client = bigquery.Client(project=gcp_project) if gcp_project else bigquery.Client()

        logger.info("MC ADAPTER: PHASE 1 — DataExtractor.extract_all()")
        extractor = DataExtractor(config, client)
        extractor.extract_all()

        logger.info("MC ADAPTER: PHASE 1b — _fetch_mr_daily_weights()")
        mr_weights = extractor._fetch_mr_daily_weights()

        logger.info("MC ADAPTER: PHASE 2 — ActuarialEngine.run_engine()")
        actuarial = ActuarialEngine(
            config=config, fcast_data=extractor.fcast_data_df,
            mapping_data=extractor.gw_mapping_df,
            longterm_fcast_pre=extractor.longterm_fcast_df,
            attempts_df=extractor.attempts_df)
        final_attempts_df = actuarial.run_engine()

        # Persist the split-INDEPENDENT forecast so tab 3 can re-run the allocation later
        # on a chosen split WITHOUT re-forecasting. Best-effort — never breaks the run.
        try:
            import json as _json, pickle as _pickle
            _out_abs = os.path.join(project_root, config["paths"]["output_dir"].format(
                month_var=config["run_settings"]["month_var"],
                company=config["run_settings"]["company"]))
            os.makedirs(_out_abs, exist_ok=True)
            final_attempts_df.to_parquet(os.path.join(_out_abs, "_actuarial_attempts.parquet"))
            extractor.mid_df.to_parquet(os.path.join(_out_abs, "_mid_df.parquet"))
            with open(os.path.join(_out_abs, "_mr_weights.pkl"), "wb") as _f:
                _pickle.dump(mr_weights, _f)
            with open(os.path.join(_out_abs, "_pipeline_config.json"), "w") as _f:
                _json.dump(config, _f, default=str)
            logger.info("MC ADAPTER: cached forecast artifacts for tab-3 exact allocation.")
        except Exception as _pe:  # noqa: BLE001
            logger.warning(f"MC ADAPTER: could not cache forecast artifacts ({_pe}).")

        logger.info("MC ADAPTER: PHASE 3 — AllocationEngine.execute_time_aware_routing()")
        allocator = AllocationEngine(
            config=config, attempts_df=final_attempts_df,
            split_df=extractor.split_df, mr_weights=mr_weights)
        pre_df, post_df = allocator.execute_time_aware_routing()

        logger.info("MC ADAPTER: PHASE 4 — ExportManager.run_all_exports()")
        exporter = ExportManager(config=config, mid_df=extractor.mid_df,
                                 attempts_df=extractor.attempts_df, mr_weights=mr_weights)
        exporter.run_all_exports(pre_df, post_df)

        out = config["paths"]["output_dir"].format(
            month_var=config["run_settings"]["month_var"],
            company=config["run_settings"]["company"])
        logger.info("MC ADAPTER: pipeline complete.")
        return os.path.join(project_root, out)
    finally:
        os.chdir(prev_cwd)


def _canonical_gateway(name) -> str:
    """Collapse deprecated '-x' gateway instances into their canonical sibling."""
    s = str(name)
    return s[:-2] if s.endswith("-x") else s


def _normalise_pre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a Mastercard pipeline export into the optimiser's baseline contract:
        rpgt, currency, bank, gateway, volume, baseline_share, risk_rate
    using the do-nothing ('pre') columns. Works for both bin_rpgt_impact_export.csv
    (Txn_Pre / CB_Pre, has a period column) and effective_rate_impact.csv
    (Sim_Sales / Sim_CBs, month-1 already). The risk_rate here is the chargeback rate.
    """
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    ren = {"mastercardMid": "gateway", "BIN": "bank", "Currency": "currency"}
    for a, b in ren.items():
        if a in d.columns:
            d = d.rename(columns={a: b})
    if "rpgt" not in d.columns and "RPGT" in d.columns:
        d = d.rename(columns={"RPGT": "rpgt"})

    # The Mastercard baseline lives at period 1 (period 0 is the injected real history).
    if "period" in d.columns:
        _periods = pd.to_numeric(d["period"], errors="coerce")
        _target = 1 if (_periods == 1).any() else 0
        d = d[_periods == _target].copy()

    vol_col = next((c for c in ["Txn_Pre", PRE_SALES, "MC_Txn_Pre"] if c in d.columns), None)
    cb_col = next((c for c in ["CB_Pre", PRE_CBS] if c in d.columns), None)
    if vol_col is None:
        return pd.DataFrame(columns=["rpgt", "currency", "bank", "gateway",
                                     "volume", "baseline_share", "risk_rate"])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
    if cb_col is not None:
        d["_cbs"] = pd.to_numeric(d[cb_col], errors="coerce").fillna(0.0)
    elif PRE_RATE in d.columns:
        rate = pd.to_numeric(d[PRE_RATE], errors="coerce").fillna(0.0)
        d["_cbs"] = rate * d["volume"]
    else:
        d["_cbs"] = 0.0

    for c in ["rpgt", "currency", "bank", "gateway"]:
        d[c] = d.get(c, "unknown").astype(str)
    d = d[d["volume"] > 0].copy()

    d["gateway"] = d["gateway"].map(_canonical_gateway)
    d = (d.groupby(["rpgt", "currency", "bank", "gateway"], as_index=False)
           .agg(volume=("volume", "sum"), _cbs=("_cbs", "sum")))

    d["risk_rate"] = (d["_cbs"] / d["volume"].replace(0, pd.NA)).fillna(0.0)
    tot = d.groupby(["rpgt", "currency", "bank"])["volume"].transform("sum")
    d["baseline_share"] = (d["volume"] / tot).fillna(0.0)
    return d[["rpgt", "currency", "bank", "gateway", "volume",
              "baseline_share", "risk_rate"]].reset_index(drop=True)


# Pipeline output files that carry a usable Mastercard 'pre' baseline, in preference order.
PRE_SOURCE_FILES = ["bin_rpgt_impact_export.csv", "effective_rate_impact.csv"]


def load_mc_pre_forecast(path: str) -> pd.DataFrame:
    """
    Load the Mastercard pipeline's baseline from its outputs. `path` may be a directory
    (tries the granular bin_rpgt export first, then effective_rate) or a specific CSV file.
    """
    if os.path.isdir(path):
        for fname in PRE_SOURCE_FILES:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                out = _normalise_pre(pd.read_csv(fpath))
                logger.info(f"      - MC baseline from {fname}: {len(out):,} cell-rows")
                if len(out):
                    return out
        raise FileNotFoundError(
            f"No usable Mastercard baseline export found in {path}. Looked for: "
            + ", ".join(PRE_SOURCE_FILES))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mastercard pipeline 'pre' output not found at {path}.")
    out = _normalise_pre(pd.read_csv(path))
    logger.info(f"      - MC baseline from {os.path.basename(path)}: {len(out):,} cell-rows")
    return out
