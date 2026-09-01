"""Headless forecast run + baseline preview — a SHIM. The runner is main.py (19fl).

    python scripts/run_forecast_pipeline.py --settings config/settings.yaml
      ==  python main.py --config config/settings.yaml --show-pre

    python scripts/run_forecast_pipeline.py --pre <outputs dir>
      ==  python main.py --pre <outputs dir>

Its docstring used to say it "mirrors the repo's main.py". It did not mirror it:
it called the ADAPTER (`run_vamp_pipeline`) while main.py had its own copy of the
four phases. 19fl moved main.py onto the adapter too, so the two are now the
same path and this file carries no logic. Safe to delete once nothing calls it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="config/settings.yaml")
    ap.add_argument("--pre", default=None,
                    help="Skip the live run; load pre from this outputs dir/CSV.")
    ap.add_argument("--gcp-project", default=None)
    ap.add_argument("--scheme", default="visa")
    args = ap.parse_args()
    run(args.scheme, args.settings, args.pre, show_pre=True,
        gcp_project=args.gcp_project)


if __name__ == "__main__":
    main()
