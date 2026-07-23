"""
Headless end-to-end run of the whole optimiser, no UI required.

Usage:
    python scripts/run_pipeline.py --success data/example_attempts_success_data.csv \
        --engine entropy --weight 0.7 --outdir out

Runs: load -> success rates -> optimise split -> impact -> k-means compress ->
JSON configs, and prints a summary. Handy as a smoke test and for scheduling.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from routing_optimiser import (HardConstraints, OptimiserSettings,  # noqa: E402
                               SoftConstraints, build_configs,
                               cell_baseline_vs_proposed, compress_split,
                               count_config_rules, engine_choices,
                               gateway_volume_shift, headline_impact,
                               key_contributors, optimise_split,
                               portfolio_summary, prepare_inputs, sweep_slider,
                               write_configs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--success", required=True)
    ap.add_argument("--forecast", default=None)
    ap.add_argument("--engine", default="entropy")
    ap.add_argument("--weight", type=float, default=0.7)
    ap.add_argument("--vamp-cap", type=float, default=0.009)
    ap.add_argument("--max-share", type=float, default=0.97)
    ap.add_argument("--floor", type=float, default=0.03)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    print("Engines available:", [k for k, _ in engine_choices()])
    print(f"\n[1/6] Loading inputs from {args.success} ...")
    problems, sr, forecast = prepare_inputs(args.success, args.forecast)
    print(f"      cells: {len(problems)}   success-rate rows: {len(sr)}")

    settings = OptimiserSettings(
        risk_conversion_weight=args.weight, engine=args.engine,
        hard=HardConstraints(max_gateway_share=args.max_share, vamp_cap=args.vamp_cap),
        soft=SoftConstraints(exploration_floor=args.floor),
    )

    print(f"[2/6] Optimising split with engine='{args.engine}' weight={args.weight} ...")
    split = optimise_split(problems, settings)
    summ = portfolio_summary(split)
    print(f"      portfolio success={summ['expected_success_rate']:.4f}  "
          f"risk={summ['expected_risk_rate']:.4f}  "
          f"infeasible_cells={summ['infeasible_cells']}")

    print("[3/6] Impact vs baseline ...")
    cell = cell_baseline_vs_proposed(split, avg_ticket=25.0)
    hi = headline_impact(cell)
    print(f"      success-rate uplift: {hi['success_rate_uplift_pp']:+.2f} pp   "
          f"incremental revenue: {hi['incremental_revenue']:+,.0f}")
    print("      top contributor banks:")
    kc = key_contributors(cell, by="bank", top=5)
    for _, r in kc.iterrows():
        print(f"        {r['bank'][:34]:34}  rev {r['incremental_revenue']:+9,.0f}  "
              f"({r['pct_of_uplift']:.0f}% of uplift)")

    print("[4/6] Slider sweep (Pareto frontier) ...")
    frontier = sweep_slider(problems, settings)
    print(frontier[["weight", "expected_success_rate", "expected_risk_rate"]]
          .round(4).to_string(index=False))

    print("[5/6] Volume-weighted k-means compression ...")
    compressed, elbow, stats = compress_split(split)
    print(f"      raw rules: {stats['raw_rules']}  ->  "
          f"compressed rules: {stats['compressed_rules']}  "
          f"({stats['reduction_pct']}% fewer)")
    print(f"      JSON config rules to generate: {count_config_rules(compressed)}")

    print("[6/6] Generating JSON configs ...")
    configs = build_configs(compressed, brand="tdr", scheme="vi")
    os.makedirs(args.outdir, exist_ok=True)
    split.to_csv(os.path.join(args.outdir, "proposed_split.csv"), index=False)
    compressed.to_csv(os.path.join(args.outdir, "compressed_rules.csv"), index=False)
    cell.to_csv(os.path.join(args.outdir, "impact_by_cell.csv"), index=False)
    gateway_volume_shift(split).to_csv(os.path.join(args.outdir, "gateway_volume_shift.csv"), index=False)
    paths = write_configs(configs, os.path.join(args.outdir, "configs"),
                          brand="tdr", date="260629")
    print(f"      wrote {len(paths)} JSON files to {args.outdir}/configs")
    print("\nDone. Outputs in:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
