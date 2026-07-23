"""
Adapter between the routing optimiser and the real VAMP forecast pipeline
(vendored under src/vamp_pipeline/).

Three jobs:
  1. build_pipeline_config  - map the Forecast tab's settings onto the exact
                              settings.yaml schema the pipeline expects.
  2. run_vamp_pipeline      - run DataExtractor -> ActuarialEngine ->
                              AllocationEngine -> ExportManager (needs BigQuery).
  3. load_pre_forecast      - read the pipeline's 'pre' (do-nothing) output and
                              normalise it into the optimiser's forecast contract
                              (rpgt, currency, bank, gateway, volume,
                              baseline_share, risk_rate). Dependency-free: reads
                              the CSVs the pipeline already wrote, so it works
                              without BigQuery on previously-run outputs.

The pipeline's ExportManager writes effective_rate_impact.csv with, per
vampMid x rpgt x BIN x Currency, the baseline ('Sim_*') and proposed ('Forecast_*')
sales, VAMPs and rates. The Sim_* columns are exactly the do-nothing baseline
this optimiser starts from.
"""
from __future__ import annotations

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

# Pipeline "pre" (baseline / do-nothing) columns in effective_rate_impact.csv
PRE_SALES = "Sim_Sales"
PRE_VAMPS = "Sim_VAMPs"
PRE_RATE = "Sim_Rate"


def build_pipeline_config(ui: dict) -> dict:
    """Map the flat Forecast-tab settings onto the pipeline's settings.yaml schema."""
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
            # Go-live date for the proposed split; drives the additive pro-rata
            # export only (does NOT affect the forecast projection).
            "split_go_live_date": ui.get("split_go_live_date"),
        },
        "paths": {
            "cache_path": "data/cache/{month_var}/{company}/",
            "chunked_files_dir": "data/rules/{month_var}/{company}/",
            "output_dir": "data/outputs/{month_var}/{company}/",
            "split_rules_file": ui.get("split_rules_file", ""),
            "mid_list_file": ui.get("mid_list_file",
                                    "data/mappings/Master_MID_List.csv"),
        },
        "targets": {
            "company_target_volume": ui.get("m0_total_transactions"),
            "company_rpgt_target_volumes": ui.get("m0_transaction_weightings", {}),
        },
        "actuarial_settings": {
            "t0_lookback_months": ui.get("t0_lookback_months", 1),
            "decay_factor": ui.get("decay_factor", 0.5),
            "thermometer_sample_months": ui.get("thermometer_sample_months", 1),
        },
        "thermometer_config": ui.get("thermometer_config") or {},
        "gateway_volume_overrides": ui.get("gateway_volume_overrides") or {},
        "filters": {
            "test_gateways_to_scrub": scrub_list or [],
            "force_actuals_for_rpgts": ui.get("force_actuals_for", []),
        },
    }


def run_vamp_pipeline(config: dict, project_root: str,
                      gcp_project: str | None = "sapient-tangent-172609") -> str:
    """
    Run the full VAMP pipeline and return the output directory.

    Lazily imports the vendored pipeline (which needs google-cloud-bigquery) so
    the rest of the app stays importable without BigQuery installed. Runs from
    project_root so the pipeline's relative 'queries/' path resolves.
    """
    import sys

    project_root = os.path.abspath(project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))
    from google.cloud import bigquery  # noqa: F401  (import check)
    from vamp_pipeline import (ActuarialEngine, AllocationEngine, DataExtractor,
                               ExportManager)
    from google.cloud import bigquery as bq

    prev_cwd = os.getcwd()
    os.chdir(project_root)  # so 'queries/<file>.sql' resolves
    try:
        # Make paths explicit & absolute so nothing depends on the CWD.
        import copy
        config = copy.deepcopy(config)
        config.setdefault("paths", {})
        config["paths"]["queries_dir"] = os.path.join(project_root, "queries")
        mlf = config["paths"].get("mid_list_file") or "data/mappings/Master_MID_List.csv"
        if not os.path.isabs(mlf):
            mlf = os.path.join(project_root, mlf)
        config["paths"]["mid_list_file"] = mlf
        logger.info(f"ADAPTER: project_root={project_root}")
        import vamp_pipeline.data_extractor as _dex
        logger.info(f"ADAPTER: data_extractor loaded from {os.path.abspath(_dex.__file__)}")
        logger.info(f"ADAPTER: data_extractor build = {getattr(_dex, '__build__', 'UNKNOWN (stale?)')}")
        logger.info(f"ADAPTER: queries_dir={config['paths']['queries_dir']} "
                    f"(exists={os.path.isdir(config['paths']['queries_dir'])})")
        logger.info(f"ADAPTER: mid_list_file={mlf} (exists={os.path.exists(mlf)})")

        client = bq.Client(project=gcp_project) if gcp_project else bq.Client()

        logger.info("ADAPTER: PHASE 1 — DataExtractor.extract_all()")
        extractor = DataExtractor(config, client)
        extractor.extract_all()
        logger.info("ADAPTER: PHASE 1b — _fetch_mr_daily_weights()")
        mr_weights = extractor._fetch_mr_daily_weights()

        logger.info("ADAPTER: PHASE 2 — ActuarialEngine.run_engine()")
        actuarial = ActuarialEngine(
            config=config, fcast_data=extractor.fcast_data_df,
            mapping_data=extractor.gw_mapping_df,
            longterm_fcast_pre=extractor.longterm_fcast_df,
            attempts_df=extractor.attempts_df)
        final_attempts_df = actuarial.run_engine()

        # Persist the split-INDEPENDENT forecast (actuarial attempts + the context the
        # AllocationEngine/ExportManager need) so tab 3 can run the REAL allocation on a
        # chosen split later WITHOUT re-forecasting. Best-effort — never breaks the run.
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
            logger.info("ADAPTER: cached forecast artifacts for tab-3 exact allocation.")
        except Exception as _pe:  # noqa: BLE001
            logger.warning(f"ADAPTER: could not cache forecast artifacts ({_pe}); "
                           "tab-3 exact mode will be unavailable.")

        logger.info("ADAPTER: PHASE 3 — AllocationEngine.execute_time_aware_routing()")
        allocator = AllocationEngine(
            config=config, attempts_df=final_attempts_df,
            split_df=extractor.split_df, mr_weights=mr_weights)
        pre_df, post_df = allocator.execute_time_aware_routing()

        logger.info("ADAPTER: PHASE 4 — ExportManager.run_all_exports()")
        exporter = ExportManager(config=config, mid_df=extractor.mid_df,
                                 attempts_df=extractor.attempts_df, mr_weights=mr_weights)
        exporter.run_all_exports(pre_df, post_df)

        out = config["paths"]["output_dir"].format(
            month_var=config["run_settings"]["month_var"],
            company=config["run_settings"]["company"])
        logger.info("ADAPTER: pipeline complete")
        return os.path.join(project_root, out)
    finally:
        os.chdir(prev_cwd)


def _canonical_gateway(name) -> str:
    """
    Some pipeline exports contain deprecated instances of a gateway with a
    trailing `-x` (typically flagged in gateway_volume_overrides.json as
    being retired). They represent the same underlying acquirer/relationship
    as their non-`-x` sibling, so we collapse them into one canonical MID
    before routing decisions are made.
    """
    s = str(name)
    return s[:-2] if s.endswith("-x") else s


def _normalise_pre(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a pipeline export into the optimiser's baseline contract:
        rpgt, currency, bank, gateway, volume, baseline_share, risk_rate
    using the do-nothing ('pre') columns. Works for both
    bin_rpgt_impact_export.csv (Txn_Pre / VAMP_Pre, has a period column) and
    effective_rate_impact.csv (Sim_Sales / Sim_Rate, month-0 already).

    Deprecated `-x` gateways are collapsed into their canonical sibling
    (see `_canonical_gateway`), so downstream routing sees one row per
    real gateway per cell.
    """
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    ren = {"vampMid": "gateway", "BIN": "bank", "Currency": "currency"}
    for a, b in ren.items():
        if a in d.columns:
            d = d.rename(columns={a: b})
    if "rpgt" not in d.columns and "RPGT" in d.columns:
        d = d.rename(columns={"RPGT": "rpgt"})

    # month-0 only if a period column exists
    if "period" in d.columns:
        d = d[pd.to_numeric(d["period"], errors="coerce") == 0].copy()

    # pick the 'pre' volume and risk columns, in preference order
    vol_col = next((c for c in ["Txn_Pre", "Sim_Sales", "VI_Txn_Pre"] if c in d.columns), None)
    vamp_col = next((c for c in ["VAMP_Pre", "Sim_VAMPs"] if c in d.columns), None)
    if vol_col is None:
        return pd.DataFrame(columns=["rpgt", "currency", "bank", "gateway",
                                     "volume", "baseline_share", "risk_rate"])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
    if vamp_col is not None:
        d["_vamps"] = pd.to_numeric(d[vamp_col], errors="coerce").fillna(0.0)
    elif "Sim_Rate" in d.columns:
        rate = pd.to_numeric(d["Sim_Rate"], errors="coerce").fillna(0.0)
        d["_vamps"] = rate * d["volume"]
    else:
        d["_vamps"] = 0.0

    for c in ["rpgt", "currency", "bank", "gateway"]:
        d[c] = d.get(c, "unknown").astype(str)
    d = d[d["volume"] > 0].copy()

    # Collapse deprecated '-x' instances into their canonical sibling BEFORE we
    # compute per-gateway rates or shares, so the merged row has the combined
    # volume and a volume-weighted risk rate.
    d["gateway"] = d["gateway"].map(_canonical_gateway)
    d = (d.groupby(["rpgt", "currency", "bank", "gateway"], as_index=False)
           .agg(volume=("volume", "sum"), _vamps=("_vamps", "sum")))

    d["risk_rate"] = (d["_vamps"] / d["volume"].replace(0, pd.NA)).fillna(0.0)
    tot = d.groupby(["rpgt", "currency", "bank"])["volume"].transform("sum")
    d["baseline_share"] = (d["volume"] / tot).fillna(0.0)
    return d[["rpgt", "currency", "bank", "gateway", "volume",
              "baseline_share", "risk_rate"]].reset_index(drop=True)


# Back-compat alias (data_loader imports this name).
def normalise_pre_from_effective_rate(df: pd.DataFrame) -> pd.DataFrame:
    return _normalise_pre(df)


# Pipeline output files that carry a usable 'pre' baseline, in preference order.
PRE_SOURCE_FILES = ["bin_rpgt_impact_export.csv", "effective_rate_impact.csv"]


def load_pre_forecast(path: str) -> pd.DataFrame:
    """
    Load the pipeline's baseline from its outputs. `path` may be a directory
    (we try the granular bin_rpgt export first, then effective_rate) or a
    specific CSV file.
    """
    if os.path.isdir(path):
        for fname in PRE_SOURCE_FILES:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                out = _normalise_pre(pd.read_csv(fpath))
                logger.info(f"      - baseline from {fname}: {len(out):,} cell-rows")
                if len(out):
                    return out
        raise FileNotFoundError(
            f"No usable baseline export found in {path}. Looked for: "
            + ", ".join(PRE_SOURCE_FILES))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline 'pre' output not found at {path}.")
    out = _normalise_pre(pd.read_csv(path))
    logger.info(f"      - baseline from {os.path.basename(path)}: {len(out):,} cell-rows")
    return out


def looks_like_effective_rate(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return "vampMid" in cols and bool({"Sim_Sales", "Txn_Pre", "VI_Txn_Pre"} & cols)
