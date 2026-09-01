"""Standalone pipeline runner — Visa or Mastercard, one file.

Runs the four phases against BigQuery and writes the export CSVs to
data/outputs/{month_var}/{company}/{scheme}/. Config comes from
config/settings.yaml (visa) or config/settings_mastercard.yaml (mastercard).

    python main.py                      # visa (the default)
    python main.py --scheme mastercard
    python main.py --scheme visa --config config/settings.yaml

This is optional: the app's Build Baseline tab does the same thing in-process.
Use this when you'd rather run the heavy pipeline separately (in an environment
where BigQuery is authenticated) and then point the app's "Use a previously
created forecast" at the outputs.

19fk CONSOLIDATION. This was two files, main.py and main_mastercard.py, that
differed in five lines out of eighty: the package they import, the config path,
one extra kwarg on the Mastercard ExportManager, and two log labels. Everything
else — the chdir, the sys.path insert, the four phase calls, the failure
handler — was duplicated, which is how main.py came to be missing the explicit
`queries_dir` that main_mastercard.py sets. One file cannot drift from itself.
Run from the project root so queries/ resolves.
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

GCP_PROJECT = "sapient-tangent-172609"

# scheme -> (package, default config, label). The ONLY per-scheme differences,
# in one place, so adding a third scheme is a row rather than a third file.
SCHEMES = {
    "visa": ("vamp_pipeline", "config/settings.yaml", "VAMP"),
    "mastercard": ("mastercard_pipeline", "config/settings_mastercard.yaml",
                   "MASTERCARD"),
}


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def run(scheme="visa", config_path=None):
    import importlib

    if scheme not in SCHEMES:
        raise SystemExit(f"--scheme must be one of {sorted(SCHEMES)}, got {scheme!r}")
    pkg_name, default_cfg, label = SCHEMES[scheme]
    pkg = importlib.import_module(pkg_name)
    logger = pkg.setup_logger(__name__)

    logger.info(f"🚀 Starting {label} Master Pipeline...")
    config = load_config(config_path or default_cfg)
    # Set explicitly rather than relying on the chdir above: the SQL resolvers
    # try `paths.queries_dir` first, and main_mastercard.py set this while
    # main.py did not — a difference that had no reason to exist.
    config.setdefault("paths", {})["queries_dir"] = os.path.join(ROOT, "queries")

    from google.cloud import bigquery
    bq_client = bigquery.Client(project=GCP_PROJECT)

    logger.info("=== PHASE 1: DATA EXTRACTION ===")
    extractor = pkg.DataExtractor(config, bq_client)
    extractor.extract_all()
    mr_weights = extractor._fetch_mr_daily_weights()

    logger.info("=== PHASE 2: ACTUARIAL ENGINE ===")
    actuarial = pkg.ActuarialEngine(
        config=config,
        fcast_data=extractor.fcast_data_df,
        mapping_data=extractor.gw_mapping_df,
        longterm_fcast_pre=extractor.longterm_fcast_df,
        attempts_df=extractor.attempts_df,
    )
    final_attempts_df = actuarial.run_engine()

    logger.info("=== PHASE 3: ALLOCATION ENGINE ===")
    allocator = pkg.AllocationEngine(
        config=config,
        attempts_df=final_attempts_df,
        split_df=extractor.split_df,
        mr_weights=mr_weights,
    )
    pre_df, post_df = allocator.execute_time_aware_routing()

    logger.info("=== PHASE 4: EXPORT MANAGER ===")
    # Mastercard's ExportManager takes mr_weights; Visa's does not. Passed by
    # INSPECTION rather than by an `if scheme ==` so that adding it to the Visa
    # side later needs no change here.
    import inspect
    _ex_kw = dict(config=config, mid_df=extractor.mid_df,
                  attempts_df=extractor.attempts_df)
    if "mr_weights" in inspect.signature(pkg.ExportManager).parameters:
        _ex_kw["mr_weights"] = mr_weights
    exporter = pkg.ExportManager(**_ex_kw)
    exporter.run_all_exports(pre_df, post_df)

    out = config["paths"]["output_dir"].format(
        month_var=config["run_settings"]["month_var"],
        company=config["run_settings"]["company"])
    logger.info(f"🎉 {label} Master Pipeline complete. Outputs in {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scheme", default="visa", choices=sorted(SCHEMES),
                    help="Card scheme to run (default: visa).")
    ap.add_argument("--config", default=None,
                    help="Override the settings YAML for this scheme.")
    args = ap.parse_args(argv)
    pkg_name = SCHEMES[args.scheme][0]
    try:
        return run(args.scheme, args.config)
    except Exception as e:  # noqa: BLE001
        import importlib
        importlib.import_module(pkg_name).setup_logger(__name__).error(
            f"❌ PIPELINE FAILED WITH A FATAL ERROR:\n{e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
