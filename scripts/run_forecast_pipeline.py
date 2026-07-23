"""
Headless run of the real VAMP forecast pipeline, then hand its 'pre' output to
the routing optimiser. Mirrors the repo's main.py but wires the result straight
into this project.

Usage:
    python scripts/run_forecast_pipeline.py --settings config/settings.yaml
Requires google-cloud-bigquery + credentials. Without them, use the app's
synthesised fallback or point --pre at a previously-run effective_rate_impact.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from routing_optimiser import load_pre_forecast, run_vamp_pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--pre", default=None,
                    help="Skip the live run; load pre from this outputs dir/CSV.")
    ap.add_argument("--gcp-project", default=None)
    args = ap.parse_args()

    if args.pre:
        pre = load_pre_forecast(args.pre)
    else:
        with open(args.settings) as f:
            config = yaml.safe_load(f)
        out_dir = run_vamp_pipeline(config, ROOT, gcp_project=args.gcp_project)
        print("Pipeline outputs:", out_dir)
        pre = load_pre_forecast(out_dir)

    print(f"Baseline 'pre' forecast: {len(pre)} rows across "
          f"{pre[['rpgt','currency','bank']].drop_duplicates().shape[0]} cells")
    print(pre.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
