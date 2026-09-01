"""
Adapter between the routing optimiser / Streamlit app and the MASTERCARD forecast
pipeline (vendored under src/mastercard_pipeline/). Mirrors vamp_forecast_pipeline.py
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
            "cache_path": "data/build_baseline_cached_input_data/{month_var}/{company}/mastercard/",
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
    from build_baseline.mastercard_pipeline import (ActuarialEngine, AllocationEngine, DataExtractor,
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

        import build_baseline.mastercard_pipeline.data_extractor as _dex
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
                                 attempts_df=extractor.attempts_df, mr_weights=mr_weights,
                                 forecast_df=final_attempts_df)
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
    ren = {"mastercardMid": "gateway", "BIN": "bin", "Currency": "currency"}
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
        return pd.DataFrame(columns=["rpgt", "currency", "bin", "gateway",
                                     "volume", "baseline_share", "risk_rate"])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
    if cb_col is not None:
        d["_cbs"] = pd.to_numeric(d[cb_col], errors="coerce").fillna(0.0)
    elif PRE_RATE in d.columns:
        rate = pd.to_numeric(d[PRE_RATE], errors="coerce").fillna(0.0)
        d["_cbs"] = rate * d["volume"]
    else:
        d["_cbs"] = 0.0

    for c in ["rpgt", "currency", "bin", "gateway"]:
        d[c] = d.get(c, "unknown").astype(str)
    d = d[d["volume"] > 0].copy()

    d["gateway"] = d["gateway"].map(_canonical_gateway)
    d = (d.groupby(["rpgt", "currency", "bin", "gateway"], as_index=False)
           .agg(volume=("volume", "sum"), _cbs=("_cbs", "sum")))

    d["risk_rate"] = (d["_cbs"] / d["volume"].replace(0, pd.NA)).fillna(0.0)
    tot = d.groupby(["rpgt", "currency", "bin"])["volume"].transform("sum")
    d["baseline_share"] = (d["volume"] / tot).fillna(0.0)
    return d[["rpgt", "currency", "bin", "gateway", "volume",
              "baseline_share", "risk_rate"]].reset_index(drop=True)


def _mc_prorata_to_pre(df: pd.DataFrame) -> pd.DataFrame:
    """Mastercard mirror of visa's `_prorata_to_pre`.

    The MC pro-rata export (`mc_cb_t_period_prorata_export.csv`) carries `cbCount` /
    `MC_Txn_Count` at the finer Country x paymentMethodProvider x t grain. Sum over `t` per
    (mastercardMid, RPGT, BIN, Currency, period) and rename to the CB_Pre / Txn_Pre contract
    `_normalise_pre` already reads - `Txn_Pre` is first in its volume preference list and
    `CB_Pre` first in its chargeback list, so nothing downstream changes.

    A numerical no-op by construction: export_manager builds this file and
    bin_rpgt_impact_export.csv from the same `t_data`. Verified on the 2026-08-21 Mastercard
    outputs - 270,393 keys, max|Δ| 0.0000000000 on BOTH CB_Pre and Txn_Pre, no key on one side
    only. `_reconcile_mc_pre_against_bin_rpgt` re-checks it every run rather than trusting that.

    NOT BIT-IDENTICAL, AND THAT IS UNAVOIDABLE. This path sums over `t` here and again inside
    `_normalise_pre`; the bin_rpgt path sums once. Float addition is not associative, so the two
    differ in the last bits. Measured end-to-end on the 2026-08-21 MC outputs, 35,382 baseline
    rows, identical keys in identical order:

        volume          max|Δ| 4.547e-13 · max relative 3.415e-16 · Σ 340,424.000000 both sides
        baseline_share  max|Δ| 1.110e-16 · max relative 4.829e-16 · Σ  14,783.000000 both sides
        risk_rate       max|Δ| 4.857e-17 · max relative 1.182e-13 · Σ     101.995231, Δ 4.3e-14

    Every difference is within 4 ULP, and the totals are exact. No amount of better code removes
    it - a different summation ORDER is the whole mechanism. Worth stating rather than burying:
    the VISA baseline has had this same property since it moved to its pro-rata export, so this
    brings Mastercard into line with an existing behaviour rather than introducing a new one.

    Returns `df` unchanged if it isn't the MC pro-rata export.

    NOTE ON THE FLATTENING. This drops `Country` and `paymentMethodProvider`, which the export
    does carry - the same discard visa's loader makes, and for the same bad reason: the shape
    downstream expects. It is harmless HERE only because `_normalise_pre` groups to
    (rpgt, currency, bank, gateway) on the next line, so the columns would die anyway. It is
    kept deliberately so this change stays a verified no-op; when the visa side stops
    flattening, this follows it. Do not read this as an endorsement of the flattening.
    """
    cols = set(df.columns)
    if not ({"cbCount", "MC_Txn_Count"} <= cols):
        return df
    # GRAIN MUST MATCH bin_rpgt_impact_export.csv, WHICH CARRIES `Country`. This is not a
    # cosmetic choice - `_normalise_pre` drops rows with volume <= 0 BEFORE it groups, so the
    # grain decides which chargebacks survive that filter. Aggregating Country away first folds
    # a zero-volume Country slice's CBs into a sibling that does have volume, and those CBs then
    # reach `risk_rate` when on the bin_rpgt path they were discarded. Measured: 53 of 35,382
    # rows moved, max |Δrisk_rate| 0.1218, on a frame whose `volume` matched to 2e-12. Keeping
    # Country makes the two paths identical.
    #
    # `paymentMethodProvider` is deliberately NOT a key: bin_rpgt does not carry it, so adding
    # it would make this FINER than the file it must reproduce and move the same filter the
    # other way. It comes back when the visa-side flattening is fixed, together.
    keys = [c for c in ["mastercardMid", "RPGT", "BIN", "Currency", "Country", "period"]
            if c in cols]
    return (df.groupby(keys, as_index=False, observed=True)[["cbCount", "MC_Txn_Count"]].sum()
              .rename(columns={"cbCount": "CB_Pre", "MC_Txn_Count": "Txn_Pre"}))


def _reconcile_mc_pre_against_bin_rpgt(pre: pd.DataFrame, out_dir: str) -> None:
    """RECONCILIATION GUARD, mirroring visa's `_reconcile_pre_against_bin_rpgt`.

    When the baseline comes from the MC pro-rata export, verify it agrees with the legacy
    bin_rpgt_impact_export.csv. Both are built from the same `t_data`, so they MUST. Logs a
    prominent WARNING on any material divergence - this never silently baselines off a
    mismatched file. Read-only, and never raises: a guard that can break the run is a guard
    people turn off.
    """
    try:
        bin_p = os.path.join(out_dir, "bin_rpgt_impact_export.csv")
        if not os.path.exists(bin_p):
            return
        b = pd.read_csv(bin_p)

        def _sum(_df, _col):
            if _col not in getattr(_df, "columns", ()):
                return None
            return float(pd.to_numeric(_df[_col], errors="coerce").fillna(0.0).sum())

        rows = []
        for _col in ("CB_Pre", "Txn_Pre"):
            _p, _b = _sum(pre, _col), _sum(b, _col)
            if _p is None or _b is None:
                rows.append(f"{_col}: NOT COMPARABLE ("
                            f"{'missing on the pro-rata side' if _p is None else 'missing on bin_rpgt'})")
                continue
            _d = abs(_p - _b)
            _rel = _d / max(abs(_b), 1e-9)
            rows.append(f"{_col} {_p:,.2f} vs {_b:,.2f} (Δ {_d:,.4f}, {_rel:.2%})")
            if _rel > 1e-6 and _d > 1e-6:
                # f-string, NOT %-args: "%,.2f" is not a valid %-format and would raise
                # inside the logger — a guard that crashes when it fires is worse than none.
                logger.warning(
                    f"      - ⚠️ MC BASELINE RECONCILIATION MISMATCH on {_col}: pro-rata export "
                    f"{_p:,.2f} vs bin_rpgt {_b:,.2f} (Δ {_d:,.4f}). Both are built from the "
                    "same t_data, so they MUST agree — a difference means one of the two exports "
                    "is stale or the aggregation grain has drifted. The baseline below is the "
                    "PRO-RATA one.")
        if rows:
            logger.info("      - MC baseline reconciliation: " + " · ".join(rows))
    except Exception as _e:  # noqa: BLE001 - a guard must never break the run it guards
        logger.info(f"      - MC baseline reconciliation skipped ({type(_e).__name__}: {_e})")


# Pipeline output files that carry a usable Mastercard 'pre' baseline, in preference order.
# 19el: the MC pro-rata export FIRST, matching visa's PRE_SOURCE_FILES. Both are written from
# the same `t_data`, so this is a no-op numerically (verified: 270,393 keys, max|Δ| 0.0 on CB_Pre
# and Txn_Pre) - the point is that the baseline and the band scorer now read ONE file.
PRE_SOURCE_FILES = ["mc_cb_t_period_prorata_export.csv",
                    "bin_rpgt_impact_export.csv", "effective_rate_impact.csv"]


def load_mc_pre_forecast(path: str) -> pd.DataFrame:
    """
    Load the Mastercard pipeline's baseline from its outputs. `path` may be a directory
    (tries the granular bin_rpgt export first, then effective_rate) or a specific CSV file.
    """
    if os.path.isdir(path):
        for fname in PRE_SOURCE_FILES:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                # 19el: convert the MC pro-rata export to the CB_Pre / Txn_Pre contract first.
                # `_mc_prorata_to_pre` returns the frame untouched for every other source, so
                # the bin_rpgt and effective_rate paths are byte-for-byte what they were.
                _raw = _mc_prorata_to_pre(pd.read_csv(fpath))
                out = _normalise_pre(_raw)
                logger.info(f"      - MC baseline from {fname}: {len(out):,} cell-rows")
                if len(out):
                    if fname == "mc_cb_t_period_prorata_export.csv":
                        _reconcile_mc_pre_against_bin_rpgt(_raw, path)
                    return out
        raise FileNotFoundError(
            f"No usable Mastercard baseline export found in {path}. Looked for: "
            + ", ".join(PRE_SOURCE_FILES))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mastercard pipeline 'pre' output not found at {path}.")
    out = _normalise_pre(_mc_prorata_to_pre(pd.read_csv(path)))
    logger.info(f"      - MC baseline from {os.path.basename(path)}: {len(out):,} cell-rows")
    return out
