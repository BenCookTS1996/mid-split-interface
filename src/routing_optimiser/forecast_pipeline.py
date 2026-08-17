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


# [FN-070]
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


# [FN-071]
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
    from google.cloud import bigquery
    from vamp_pipeline import (ActuarialEngine, AllocationEngine, DataExtractor,
                               ExportManager)

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

        import time as _t
        _t0_all = _t.time()
        _rs = config.get("run_settings", {}) or {}
        logger.info(
            f"ADAPTER: run config — company={_rs.get('company')} · month={_rs.get('month_var')} · "
            f"month_0_start={_rs.get('month_0_start_date')} · "
            f"actuals={_rs.get('actuals_start_date')}→{_rs.get('actuals_end_date')} · "
            f"split_go_live={_rs.get('split_go_live_date')} · "
            f"future_anchor={_rs.get('future_anchor_date')} · "
            f"blend_future_sheet_rules={_rs.get('blend_future_sheet_rules')} · "
            f"use_chunked_csv_files={_rs.get('use_chunked_csv_files')} · "
            f"load_curves_from_cache={_rs.get('load_curves_from_cache')} · "
            f"cache_path={config.get('paths', {}).get('cache_path')}")

        # [FN-072]
        def _shape(obj, name, warn_empty=True):
            """Log a dataframe's size + leading columns, PLUS the summed transaction count of any
            recognised txn-count column(s). Warns LOUDLY when a source is empty OR has rows but ZERO
            transactions — so a missing-volume source (e.g. the 'no VI Txn' case) is obvious instead
            of silently producing empty/zero-volume downstream exports."""
            try:
                import pandas as _pd, re as _re
                if isinstance(obj, _pd.DataFrame):
                    _c = list(obj.columns)
                    logger.info(f"      · {name}: {len(obj):,} rows × {len(_c)} cols"
                                + (f"  [{', '.join(map(str, _c[:10]))}{' …' if len(_c) > 10 else ''}]"
                                   if _c else ""))
                    if warn_empty and len(obj) == 0:
                        logger.warning(f"      [!] {name} is EMPTY — every export built from it will be 0 rows.")
                    # Σ transactions: sum every column that carries a transaction count (historical
                    # visa_trx_count / trx_count, or the forecast fc_vi_trx_m* incl. PreSim/Reallocated),
                    # so a source that has rows but NO volume is flagged (root of the 'no VI Txn' issue).
                    try:
                        _txn = [c for c in _c
                                if str(c).strip().lower() in ("visa_trx_count", "vi_trx_count",
                                                              "trx_count", "trx total", "trx_total")
                                or _re.match(r'^(presim_|reallocated_)?fc_vi_trx_m\d+$', str(c).strip().lower())]
                        if _txn and len(obj):
                            _tot = float(sum(float(_pd.to_numeric(obj[c], errors="coerce").fillna(0).sum())
                                             for c in _txn))
                            logger.info(f"          ↳ Σ transactions = {_tot:,.0f}  "
                                        f"[{', '.join(map(str, _txn[:6]))}{' …' if len(_txn) > 6 else ''}]")
                            if warn_empty and _tot <= 0:
                                logger.warning(f"      [!] {name} has rows but ZERO transactions — "
                                               "volume is missing/empty at this source.")
                    except Exception:  # noqa: BLE001
                        pass
                elif obj is None:
                    logger.info(f"      · {name}: None")
                else:
                    try:
                        logger.info(f"      · {name}: {len(obj):,} item(s)")
                    except Exception:  # noqa: BLE001
                        logger.info(f"      · {name}: {type(obj).__name__}")
            except Exception:  # noqa: BLE001
                pass

        client = bigquery.Client(project=gcp_project) if gcp_project else bigquery.Client()

        _tp = _t.time()
        logger.info("ADAPTER: PHASE 1 — DataExtractor.extract_all() "
                    "[BigQuery pull → parse split rules → assemble forecast matrix]")
        extractor = DataExtractor(config, client)
        extractor.extract_all()
        logger.info(f"ADAPTER: PHASE 1 finished in {_t.time() - _tp:.1f}s — data shapes:")
        _shape(getattr(extractor, "gw_mapping_df", None), "gateway mapping (historical)")
        _shape(getattr(extractor, "longterm_fcast_df", None), "long-term forecast matrix")
        _shape(getattr(extractor, "split_df", None), "SPLIT RULES (parsed from Excel)")
        _shape(getattr(extractor, "attempts_df", None), "attempts (derived)")
        _shape(getattr(extractor, "mid_df", None), "MID list")

        _tp = _t.time()
        logger.info("ADAPTER: PHASE 1b — _fetch_mr_daily_weights() [monthly-renewal daily curve]")
        mr_weights = extractor._fetch_mr_daily_weights()
        _shape(mr_weights, "MR daily weights")

        _tp = _t.time()
        logger.info("ADAPTER: PHASE 2 — ActuarialEngine.run_engine() "
                    "[reference curves → distribute granular VAMPs to the waterfall]")
        actuarial = ActuarialEngine(
            config=config, fcast_data=extractor.fcast_data_df,
            mapping_data=extractor.gw_mapping_df,
            longterm_fcast_pre=extractor.longterm_fcast_df,
            attempts_df=extractor.attempts_df)
        final_attempts_df = actuarial.run_engine()
        logger.info(f"ADAPTER: PHASE 2 finished in {_t.time() - _tp:.1f}s")
        _shape(final_attempts_df, "actuarial attempts (with VAMPs)")

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

        _tp = _t.time()
        logger.info("ADAPTER: PHASE 3 — AllocationEngine.execute_time_aware_routing() "
                    "[apply split rules across the t-periods, death syncs + redistribution]")
        allocator = AllocationEngine(
            config=config, attempts_df=final_attempts_df,
            split_df=extractor.split_df, mr_weights=mr_weights)
        pre_df, post_df = allocator.execute_time_aware_routing()
        logger.info(f"ADAPTER: PHASE 3 finished in {_t.time() - _tp:.1f}s")
        _shape(pre_df, "pre-simulation matrix (PreSim)")
        _shape(post_df, "reallocated matrix (post)")

        _tp = _t.time()
        logger.info("ADAPTER: PHASE 4 — ExportManager.run_all_exports() "
                    "[Pop & Stack the massive matrix → CSV baseline exports]")
        exporter = ExportManager(config=config, mid_df=extractor.mid_df,
                                 attempts_df=extractor.attempts_df, mr_weights=mr_weights)
        exporter.run_all_exports(pre_df, post_df)
        logger.info(f"ADAPTER: PHASE 4 finished in {_t.time() - _tp:.1f}s")

        out = config["paths"]["output_dir"].format(
            month_var=config["run_settings"]["month_var"],
            company=config["run_settings"]["company"])
        _out_dir = os.path.join(project_root, out)

        # Log the head() of every exported CSV so the run log shows what actually landed on disk
        # (read with nrows so a multi-million-row export costs nothing to peek at).
        try:
            import glob as _glob
            for _csv in sorted(_glob.glob(os.path.join(_out_dir, "*.csv"))):
                _name = os.path.basename(_csv)
                try:
                    _n_total = sum(1 for _ in open(_csv)) - 1   # rows minus header
                    _peek = pd.read_csv(_csv, nrows=5)
                    logger.info(f"ADAPTER: exported {_name} ({max(_n_total, 0):,} rows) — head(5):")
                    _shape(_peek, _name, warn_empty=(_n_total <= 0))
                    with pd.option_context("display.max_columns", 20, "display.width", 200,
                                           "display.max_colwidth", 24):
                        _htxt = _peek.to_string(max_cols=20)
                    for _ln in _htxt.splitlines():
                        logger.info("        " + _ln)
                except Exception as _ce:  # noqa: BLE001
                    logger.info(f"ADAPTER: exported {_name} — head() unavailable ({type(_ce).__name__}: {_ce})")
        except Exception:  # noqa: BLE001
            pass

        logger.info(f"ADAPTER: pipeline complete — total {_t.time() - _t0_all:.1f}s")
        return os.path.join(project_root, out)
    finally:
        os.chdir(prev_cwd)


# [FN-073]
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


# Set by `_normalise_pre` on each load so the caller (app) can surface the VAMP-shrinkage
# status + per-level fitted kappa in its own run-log. {'on': bool, 'levels': [(name, kappa), …]}.
_LAST_VAMP_SHRINK: dict = {"on": None, "levels": None}


# [FN-074]
def _mm_kappa(vg, ng, fallback: float = 50.0, kmax: float = 5000.0) -> float:
    """Method-of-moments Beta-Binomial concentration (kappa) for one back-off LEVEL,
    from that level's per-group pooled (vamps vg, count ng). Mirrors success_rates.
    _empirical_bayes_kappa: kappa = mu(1-mu)/true_var - 1, where true_var = observed
    count-weighted variance of group rates MINUS mean binomial sampling variance.
    Tight spread → large kappa (shrink hard toward the coarser level); wide spread →
    small kappa (trust this level). Needs ≥2 groups with data, else `fallback`."""
    import numpy as _np
    vg = _np.asarray(vg, float); ng = _np.asarray(ng, float)
    m = ng > 0
    vg, ng = vg[m], ng[m]
    if ng.sum() <= 0 or len(ng) < 2:
        return float(fallback)
    mu = float(vg.sum() / ng.sum())
    if mu <= 0.0 or mu >= 1.0:
        return float(kmax)
    p = vg / ng
    obs_var = float((ng * (p - mu) ** 2).sum() / ng.sum())        # count-weighted spread
    samp_var = float(mu * (1.0 - mu) * len(ng) / ng.sum())         # mean binomial noise
    true_var = obs_var - samp_var
    if true_var <= 1e-12:
        return float(kmax)
    return float(min(max(mu * (1.0 - mu) / true_var - 1.0, 1.0), kmax))


# [FN-075]
def _hier_vamp_shrink(d: pd.DataFrame, fallback_kappa: float = 50.0, kmax: float = 5000.0):
    """FULLY-AUTOMATIC hierarchical empirical-Bayes shrinkage of the per-cell VAMP rate.

    Walks a BANK-FIRST back-off chain (risk clusters by issuing BIN, not by MID — verified
    on bin_rpgt_impact_export: BIN explains ~16% of rate variance vs vampMid ~3%):

        global → vampMid → RPGT → Bank → Bank×RPGT → Bank×RPGT×Currency → cell

    At each level a method-of-moments kappa is fit from the spread of that level's group
    rates vs binomial sampling noise (no hand-tuning), and the level's pooled rate is shrunk
    toward the running (coarser) estimate: est = (pooled_vamps + kappa·est) / (pooled_n + kappa).
    A thin/noisy cell (e.g. 0.74 VAMP on 1.2 txns → raw 61.55%) is pulled toward its BANK's
    stable rate; a high-volume cell is essentially unchanged. Because kappa is fit in the SAME
    units as the counts, the pro-rated Txn_Pre scale self-calibrates. Returns a (n,) array."""
    import numpy as _np
    n = len(d)
    v = pd.to_numeric(d["_vamps"], errors="coerce").fillna(0.0).to_numpy(float)
    q = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0).to_numpy(float)
    tot_v = float(v.sum()); tot_q = float(q.sum())
    est = _np.full(n, (tot_v / tot_q) if tot_q > 0 else 0.0)       # coarsest level = global rate
    _t = d[["gateway", "rpgt", "bank", "currency"]].copy()
    _t["_v"] = v; _t["_q"] = q
    # coarse → fine; the final key set is the full cell grain, so its shrink is the cell itself
    chain = [["gateway"], ["rpgt"], ["bank"], ["bank", "rpgt"],
             ["bank", "rpgt", "currency"], ["bank", "rpgt", "currency", "gateway"]]
    levels_log = []
    for keys in chain:
        _gt = _t.groupby(keys)[["_v", "_q"]]
        _sum = _gt.transform("sum")
        gv = _sum["_v"].to_numpy(float); gn = _sum["_q"].to_numpy(float)
        _grp = _t.groupby(keys)[["_v", "_q"]].sum()
        kap = _mm_kappa(_grp["_v"].to_numpy(float), _grp["_q"].to_numpy(float),
                        fallback=fallback_kappa, kmax=kmax)
        with _np.errstate(divide="ignore", invalid="ignore"):
            est = (gv + kap * est) / (gn + kap)
        est = _np.where(_np.isfinite(est), est, 0.0)
        levels_log.append((("x".join(keys)), round(kap, 1)))
    return est, levels_log


# [FN-076]
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
    # Auto-detect a MASTERCARD export and delegate to the Mastercard normaliser.
    # MC exports carry chargeback columns (CB_Pre / Sim_CBs) and/or 'mastercardMid',
    # and never carry the VAMP columns — so this is a safe, backward-compatible switch
    # that lets the shared read path (data_loader.load_forecast) handle both schemes.
    _cols = set(str(c) for c in df.columns)
    _is_mc = (("mastercardMid" in _cols) or bool({"CB_Pre", "Sim_CBs", "MC_Txn_Pre"} & _cols)) \
        and not bool({"vampMid", "VAMP_Pre", "Sim_VAMPs", "VI_Txn_Pre"} & _cols)
    if _is_mc:
        from .mastercard_forecast_pipeline import _normalise_pre as _mc_normalise_pre
        return _mc_normalise_pre(df)

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
    vol_col = next((c for c in ["Txn_Pre", PRE_SALES, "VI_Txn_Pre"] if c in d.columns), None)
    vamp_col = next((c for c in ["VAMP_Pre", PRE_VAMPS] if c in d.columns), None)
    if vol_col is None:
        return pd.DataFrame(columns=["rpgt", "currency", "bank", "gateway",
                                     "volume", "baseline_share", "risk_rate"])
    d["volume"] = pd.to_numeric(d[vol_col], errors="coerce").fillna(0.0)
    if vamp_col is not None:
        d["_vamps"] = pd.to_numeric(d[vamp_col], errors="coerce").fillna(0.0)
    elif PRE_RATE in d.columns:
        rate = pd.to_numeric(d[PRE_RATE], errors="coerce").fillna(0.0)
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

    # RISK RATE: hierarchical empirical-Bayes shrinkage (default on; ROUTING_VAMP_SHRINK=0
    # disables → raw ratio). Fixes noisy thin-cell rates (e.g. 0.74 VAMP on 1.2 txns → raw
    # 61.55%) by pulling them toward the stable BANK-level rate; high-volume cells stay put.
    _raw_rr = (d["_vamps"] / d["volume"].replace(0, pd.NA)).fillna(0.0)
    global _LAST_VAMP_SHRINK
    if os.environ.get("ROUTING_VAMP_SHRINK", "1") != "0" and len(d) >= 2:
        try:
            _rr, _lvls = _hier_vamp_shrink(d)
            d["risk_rate"] = _rr
            _LAST_VAMP_SHRINK = {"on": True, "levels": _lvls, "n_cells": int(len(d))}
        except Exception:  # noqa: BLE001
            d["risk_rate"] = _raw_rr        # never let shrinkage break the pipeline
            _LAST_VAMP_SHRINK = {"on": False, "levels": None, "error": True}
    else:
        d["risk_rate"] = _raw_rr
        _LAST_VAMP_SHRINK = {"on": False, "levels": None}
    tot = d.groupby(["rpgt", "currency", "bank"])["volume"].transform("sum")
    d["baseline_share"] = (d["volume"] / tot).fillna(0.0)
    return d[["rpgt", "currency", "bank", "gateway", "volume",
              "baseline_share", "risk_rate"]].reset_index(drop=True)


# Back-compat alias (data_loader imports this name).
# [FN-077]
def normalise_pre_from_effective_rate(df: pd.DataFrame) -> pd.DataFrame:
    return _normalise_pre(df)


# Pipeline output files that carry a usable 'pre' baseline, in preference order.
# SINGLE SOURCE OF TRUTH: prefer the pro-rata export (the SAME baseline the band-VAMP scorer uses)
# so volume and VAMP come from one file, then fall back to the legacy bin_rpgt / effective-rate exports.
PRE_SOURCE_FILES = ["vamp_t_period_prorata_export.csv", "bin_rpgt_impact_export.csv",
                    "effective_rate_impact.csv"]


# [FN-077b]
def _prorata_to_pre(df: pd.DataFrame) -> pd.DataFrame:
    """If `df` is the pro-rata export (`vamp_t_period_prorata_export.csv`: `vampCount` / `VI_Txn_Count`
    at the finer Country×paymentMethodProvider×t grain), aggregate it to the SAME baseline schema
    `bin_rpgt_impact_export.csv` provides — `VAMP_Pre` / `Txn_Pre` per vampMid×RPGT×BIN×Currency×period
    — so the engine can baseline off the pro-rata export (one source of truth with the band scorer).

    This is a numerical no-op by construction: the pipeline builds BOTH files from the same `t_data`
    (see export_manager._generate_prorata_export, which just RENAMES VAMP_Pre→vampCount and
    VI_Txn_Pre→VI_Txn_Count). Returns `df` unchanged if it isn't the pro-rata export."""
    cols = set(df.columns)
    if not ({"vampCount", "VI_Txn_Count"} <= cols):
        return df
    keys = [c for c in ["vampMid", "RPGT", "BIN", "Currency", "period"] if c in cols]
    agg = (df.groupby(keys, as_index=False, observed=True)[["vampCount", "VI_Txn_Count"]].sum()
             .rename(columns={"vampCount": "VAMP_Pre", "VI_Txn_Count": "Txn_Pre"}))
    return agg


# [FN-077c]
def _reconcile_pre_against_bin_rpgt(pre: pd.DataFrame, out_dir: str) -> None:
    """RECONCILIATION GUARD: when the baseline is taken from the pro-rata export, verify it agrees with
    the legacy bin_rpgt_impact_export.csv (they're built from the same t_data, so they MUST). Logs a
    prominent WARNING on any material divergence — never silently baselines off a mismatched file."""
    try:
        bin_p = os.path.join(out_dir, "bin_rpgt_impact_export.csv")
        if not os.path.exists(bin_p):
            return
        b = pd.read_csv(bin_p)
        b0 = b[b.get("period", 0) == 0] if "period" in b.columns else b
        p0 = pre[pre.get("period", 0) == 0] if "period" in pre.columns else pre
        bt = float(pd.to_numeric(b0.get("Txn_Pre", 0), errors="coerce").fillna(0).sum())
        pt = float(pd.to_numeric(p0.get("Txn_Pre", 0), errors="coerce").fillna(0).sum())
        bv = float(pd.to_numeric(b0.get("VAMP_Pre", 0), errors="coerce").fillna(0).sum())
        pv = float(pd.to_numeric(p0.get("VAMP_Pre", 0), errors="coerce").fillna(0).sum())
        tol_t = max(1.0, 0.005 * abs(bt)); tol_v = max(1.0, 0.005 * abs(bv))
        if abs(bt - pt) > tol_t or abs(bv - pv) > tol_v:
            logger.warning("      ⚠️ baseline reconciliation: pro-rata export vs bin_rpgt DIVERGE "
                           "(txn %.0f vs %.0f, vamp %.1f vs %.1f) — expected identical; using pro-rata.",
                           pt, bt, pv, bv)
        else:
            logger.info("      ✓ baseline reconciliation: pro-rata export == bin_rpgt "
                        "(txn %.0f, vamp %.1f) — single source consistent.", pt, pv)
    except Exception as exc:  # noqa: BLE001 — a cross-check must never break the run
        logger.info("      (baseline reconciliation skipped: %s: %s)", type(exc).__name__, exc)


# [FN-078]
def load_pre_forecast(path: str) -> pd.DataFrame:
    """
    Load the pipeline's baseline from its outputs. `path` may be a directory (we prefer the pro-rata
    export — one source of truth with the band scorer — then the granular bin_rpgt export, then
    effective_rate) or a specific CSV file.
    """
    if os.path.isdir(path):
        for fname in PRE_SOURCE_FILES:
            fpath = os.path.join(path, fname)
            if os.path.exists(fpath):
                out = _normalise_pre(_prorata_to_pre(pd.read_csv(fpath)))
                logger.info(f"      - baseline from {fname}: {len(out):,} cell-rows")
                if len(out):
                    if fname == "vamp_t_period_prorata_export.csv":
                        _reconcile_pre_against_bin_rpgt(out, path)
                    return out
        raise FileNotFoundError(
            f"No usable baseline export found in {path}. Looked for: "
            + ", ".join(PRE_SOURCE_FILES))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pipeline 'pre' output not found at {path}.")
    out = _normalise_pre(_prorata_to_pre(pd.read_csv(path)))
    logger.info(f"      - baseline from {os.path.basename(path)}: {len(out):,} cell-rows")
    return out


# [FN-079]
def looks_like_effective_rate(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    _vamp = "vampMid" in cols and bool({"Sim_Sales", "Txn_Pre", "VI_Txn_Pre"} & cols)
    # Also recognise the Mastercard effective-rate / bin-impact exports so the shared
    # read path normalises them (via _normalise_pre's auto-detect) into the contract.
    _mc = "mastercardMid" in cols and bool({"Sim_Sales", "Txn_Pre", "MC_Txn_Pre", "CB_Pre", "Sim_CBs"} & cols)
    return _vamp or _mc
