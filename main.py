"""Standalone pipeline runner — Visa or Mastercard, one file, one implementation.

    python main.py                              # visa
    python main.py --scheme mastercard
    python main.py --scheme visa --config config/settings.yaml
    python main.py --pre data/outputs/AUG/TotalAV/visa   # no run: load + preview
    python main.py --show-pre                            # run, then preview

Optional: the app's Build Baseline tab does the same thing in-process. Use this
when you'd rather run the heavy pipeline separately (in an environment where
BigQuery is authenticated) and then point the app's "Use a previously created
forecast" at the outputs.

19fk/19fl CONSOLIDATION — three files became one, and one implementation.

  main.py                          had its own copy of the four phase calls
  main_mastercard.py               had a second copy, differing in five lines
  scripts/run_forecast_pipeline.py called the ADAPTER, plus a --pre preview

The adapters (`forecast_pipeline.run_vamp_pipeline` /
`mastercard_forecast_pipeline.run_mastercard_pipeline`) are what the Streamlit
app runs, and they are strictly richer than the copies here were: deep-copied
config, absolute queries_dir and mid_list_file, per-phase shape diagnostics that
shout when a source has rows but zero transactions. Reimplementing the phases in
a CLI meant the terminal path was the one WITHOUT those checks — and that is the
path used when something is already going wrong. So this file no longer runs the
phases at all; it builds the config, calls the adapter, and prints. The four
phases exist in exactly one place per scheme.
"""
import argparse
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

GCP_PROJECT = "sapient-tangent-172609"

# scheme -> (default config, adapter name, pre-loader name, label). The ONLY
# per-scheme differences, in one place, so a third scheme is a row not a file.
SCHEMES = {
    "visa": ("config/settings.yaml", "run_vamp_pipeline",
             "load_pre_forecast", "VAMP"),
    "mastercard": ("config/settings_mastercard.yaml", "run_mastercard_pipeline",
                   "load_mc_pre_forecast", "MASTERCARD"),
}


def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file missing at {config_path}")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def _preview(pre, label="pre"):
    """The preview scripts/run_forecast_pipeline.py existed for."""
    try:
        _cells = pre[["rpgt", "currency", "bin"]].drop_duplicates().shape[0]
    except Exception:  # noqa: BLE001 - a preview must never fail a run
        _cells = "?"
    print(f"Baseline '{label}' forecast: {len(pre):,} rows across {_cells} cells")
    print(pre.head(10).to_string(index=False))


def run(scheme="visa", config_path=None, pre=None, show_pre=False,
        gcp_project=None):
    import importlib

    if scheme not in SCHEMES:
        raise SystemExit(f"--scheme must be one of {sorted(SCHEMES)}, got {scheme!r}")
    default_cfg, adapter_name, loader_name, label = SCHEMES[scheme]
    ro = importlib.import_module("routing_optimiser")
    load_pre = getattr(ro, loader_name)

    if pre:                                  # skip the run entirely
        print(f"Loading a previously-run {label} baseline from {pre}")
        _preview(load_pre(pre))
        return pre

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(f"🚀 Starting {label} Master Pipeline...")
    config = load_config(config_path or default_cfg)
    try:
        out_dir = getattr(ro, adapter_name)(
            config, ROOT, gcp_project=gcp_project or GCP_PROJECT)
    except Exception as e:  # noqa: BLE001
        logger.error(f"❌ PIPELINE FAILED WITH A FATAL ERROR:\n{e}", exc_info=True)
        raise
    logger.info(f"🎉 {label} Master Pipeline complete. Outputs in {out_dir}")
    if show_pre:
        _preview(load_pre(out_dir))
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scheme", default="visa", choices=sorted(SCHEMES),
                    help="Card scheme to run (default: visa).")
    ap.add_argument("--config", default=None,
                    help="Override the settings YAML for this scheme.")
    ap.add_argument("--pre", default=None,
                    help="Skip the run: load the baseline from this outputs "
                         "directory (or CSV) and print a preview.")
    ap.add_argument("--show-pre", action="store_true",
                    help="After the run, print a preview of the baseline.")
    ap.add_argument("--gcp-project", default=None,
                    help=f"BigQuery project (default {GCP_PROJECT}).")
    args = ap.parse_args(argv)
    return run(args.scheme, args.config, args.pre, args.show_pre,
               args.gcp_project)


if __name__ == "__main__":
    main()
