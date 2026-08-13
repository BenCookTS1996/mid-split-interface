"""
Standalone MASTERCARD pipeline runner — mirrors main.py (the Visa/VAMP runner).

Runs the four phases against BigQuery and writes the export CSVs
(effective_rate_impact.csv, mid_level.csv, cb_t_period_export.csv,
bin_rpgt_impact_export.csv) to data/outputs/{month_var}/{company}/mastercard/.
Config comes from config/settings_mastercard.yaml.

Run from the project root so queries/ resolves:
    python main_mastercard.py
"""
import os
import sys

import yaml

# Run from the project root so the pipeline's relative 'queries/' path resolves.
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from google.cloud import bigquery  # noqa: E402
from mastercard_pipeline import (ActuarialEngine, AllocationEngine,  # noqa: E402
                                 DataExtractor, ExportManager, setup_logger)

logger = setup_logger(__name__)

GCP_PROJECT = "sapient-tangent-172609"


def load_config(config_path="config/settings_mastercard.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def main():
    logger.info("🚀 Starting MASTERCARD Master Pipeline...")
    config = load_config("config/settings_mastercard.yaml")
    config.setdefault("paths", {})["queries_dir"] = os.path.join(ROOT, "queries")
    bq_client = bigquery.Client(project=GCP_PROJECT)

    logger.info("=== PHASE 1: DATA EXTRACTION ===")
    extractor = DataExtractor(config, bq_client)
    extractor.extract_all()
    mr_weights = extractor._fetch_mr_daily_weights()

    logger.info("=== PHASE 2: ACTUARIAL ENGINE ===")
    actuarial = ActuarialEngine(
        config=config,
        fcast_data=extractor.fcast_data_df,
        mapping_data=extractor.gw_mapping_df,
        longterm_fcast_pre=extractor.longterm_fcast_df,
        attempts_df=extractor.attempts_df,
    )
    final_attempts_df = actuarial.run_engine()

    logger.info("=== PHASE 3: ALLOCATION ENGINE ===")
    allocator = AllocationEngine(
        config=config,
        attempts_df=final_attempts_df,
        split_df=extractor.split_df,
        mr_weights=mr_weights,
    )
    pre_df, post_df = allocator.execute_time_aware_routing()

    logger.info("=== PHASE 4: EXPORT MANAGER ===")
    exporter = ExportManager(
        config=config,
        mid_df=extractor.mid_df,
        attempts_df=extractor.attempts_df,
        mr_weights=mr_weights,
    )
    exporter.run_all_exports(pre_df, post_df)

    out = config["paths"]["output_dir"].format(
        month_var=config["run_settings"]["month_var"],
        company=config["run_settings"]["company"])
    logger.info(f"🎉 MASTERCARD Master Pipeline complete. Outputs in {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ PIPELINE FAILED WITH A FATAL ERROR:\n{e}", exc_info=True)
        raise
