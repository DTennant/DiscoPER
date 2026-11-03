#!/usr/bin/env python3
"""Run data-size scaling experiments on iNaturalist discovery.

Launches discovery experiments at progressive iNat data sizes, evaluates
each with LLM judge, and produces comparison results showing how
discovery quality scales with dataset size.

Usage:
    python scripts/run_data_size_scaling.py --output results/scaling_v1

    # Custom sizes and iterations
    python scripts/run_data_size_scaling.py --output results/scaling_v1 \
        --n-iterations 30 --sizes 200 500 1000 5000

    # Evaluate existing runs only (skip discovery)
    python scripts/run_data_size_scaling.py --output results/scaling_v1 --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SIZES = [200, 500, 1000, 2000, 5000, 10000, 25000, 50000]
INAT_DATA_DIR = ROOT / "data" / "inat_subset"


def create_subsampled_dataset(source_dir, dest_dir, n_samples, seed=42):
    """Create a subsampled copy of the iNat dataset.

    Copies ground_truth.json, prior_insights.parquet, benchmark_meta.json,
    symlinks the images/ directory, and writes a subsampled data.parquet.

    Returns the actual number of rows written (may be less than n_samples
    if the source dataset is smaller).
    """
    os.makedirs(dest_dir, exist_ok=True)

    # Read source data
    source_parquet = os.path.join(source_dir, "data.parquet")
    df = pd.read_parquet(source_parquet)
    n_available = len(df)

    actual_n = min(n_samples, n_available)
    if actual_n < n_samples:
        print(
            f"  WARNING: Requested {n_samples} but only {n_available} available, capping at {n_available}"
        )

    # Subsample
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_available, size=actual_n, replace=False)
    df_sub = df.iloc[sorted(indices)].reset_index(drop=True)
    df_sub.to_parquet(os.path.join(dest_dir, "data.parquet"))

    # Symlink images directory
    images_src = os.path.join(source_dir, "images")
    images_dst = os.path.join(dest_dir, "images")
    if os.path.exists(images_src) and not os.path.exists(images_dst):
        os.symlink(os.path.abspath(images_src), images_dst)

    # Copy auxiliary files
    for fname in ["ground_truth.json", "prior_insights.parquet", "benchmark_meta.json"]:
        src = os.path.join(source_dir, fname)
        dst = os.path.join(dest_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    return actual_n


def main():
    parser = argparse.ArgumentParser(
        description="Run data-size scaling experiments on iNaturalist discovery."
    )
    parser.add_argument("--output", required=True, help="Output directory for all runs")
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SIZES,
        help="Data sizes to test (default: 200 500 1000 2000 5000 10000 25000 50000)",
    )
    parser.add_argument(
        "--inat-dir",
        default=str(INAT_DATA_DIR),
        help="Source iNat data directory (default: data/inat_subset)",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip discovery runs, only evaluate existing results",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for subsampling"
    )
    args = parser.parse_args()

    source_dir = args.inat_dir
    if not os.path.exists(os.path.join(source_dir, "data.parquet")):
        print(f"ERROR: iNat data not found at {source_dir}")
        print(
            "Download it first: python scripts/download_inat_subset.py --output data/inat_subset"
        )
        sys.exit(1)

    # Check available data size
    source_df = pd.read_parquet(os.path.join(source_dir, "data.parquet"))
    n_available = len(source_df)
    del source_df

    # Filter sizes to those <= available
    valid_sizes = []
    for s in sorted(args.sizes):
        if s <= n_available:
            valid_sizes.append(s)
        else:
            print(
                f"WARNING: Size {s} exceeds available data ({n_available}), capping at {n_available}"
            )
            if not valid_sizes or valid_sizes[-1] != n_available:
                valid_sizes.append(n_available)
            break

    if not valid_sizes:
        print(f"ERROR: No valid sizes. Available data: {n_available}")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "subsampled_data"), exist_ok=True)

    gt_path = os.path.join(source_dir, "ground_truth.json")
    modes = ["meta_on", "meta_off"]
    procs = []

    if not args.evaluate_only:
        # Create subsampled datasets
        print("=" * 70)
        print("CREATING SUBSAMPLED DATASETS")
        print("=" * 70)
        size_actual = {}
        for size in valid_sizes:
            sub_dir = os.path.join(args.output, "subsampled_data", f"n{size}")
            actual = create_subsampled_dataset(
                source_dir, sub_dir, size, seed=args.seed
            )
            size_actual[size] = actual
            print(f"  n={size}: wrote {actual} rows -> {sub_dir}")

        # Launch experiments
        n_total = len(valid_sizes) * len(modes)
        print("\n" + "=" * 70)
        print(f"LAUNCHING {n_total} EXPERIMENTS")
        print(f"  Sizes: {valid_sizes}")
        print(f"  Modes: {modes}")
        print(f"  Iterations: {args.n_iterations}")
        print("=" * 70)

        for size in valid_sizes:
            sub_dir = os.path.abspath(
                os.path.join(args.output, "subsampled_data", f"n{size}")
            )
            for mode in modes:
                label = f"n{size}_{mode}"
                ws = os.path.join(args.output, label)
                log = os.path.join(args.output, "logs", f"{label}.log")

                cmd = [
                    sys.executable,
                    str(ROOT / "launch_discovery.py"),
                    f"--config-path={ROOT / 'config' / 'task'}",
                    "--config-name=inat_discovery",
                    f"n_iterations={args.n_iterations}",
                    f"workspace_root={ws}",
                    f"inat_data_dir={sub_dir}",
                    "enable_context_passing=true",
                    f"dataset_args.random_seed={args.seed}",
                ]

                if mode == "meta_off":
                    cmd.append("meta_discovery_args=null")

                print(f"  Launching {label}")
                with open(log, "w") as lf:
                    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=lf)
                procs.append((label, p, ws))

        print(f"\n{len(procs)} processes launched. Waiting for completion...")

        while True:
            running = sum(1 for _, p, _ in procs if p.poll() is None)
            if running == 0:
                break
            done = len(procs) - running
            print(f"  {done}/{len(procs)} done, {running} running...", end="\r")
            time.sleep(30)

        print(f"\nAll {len(procs)} experiments complete!")

    # ================================================================
    # EVALUATE WITH LLM JUDGE
    # ================================================================
    print("\n" + "=" * 70)
    print("EVALUATING WITH LLM-AS-JUDGE")
    print("=" * 70)

    all_results = {}

    for size in valid_sizes:
        all_results[size] = {}
        for mode in modes:
            label = f"n{size}_{mode}"
            ws = os.path.join(args.output, label)
            insights_path = os.path.join(ws, "insights.parquet")
            eval_path = os.path.join(ws, "evaluation_llm_judge.json")

            if not os.path.exists(insights_path):
                print(f"  SKIP {label}: no insights.parquet")
                continue

            print(f"  Evaluating {label} with LLM judge...")

            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_discovery.py"),
                f"--ground_truth={gt_path}",
                f"--discovered={insights_path}",
                f"--output={eval_path}",
                "--llm_judge",
            ]

            result = subprocess.run(
                cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=600
            )
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr[:200]}")
                continue

            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    d = json.load(f)
                all_results[size][mode] = d
                llm = d.get("llm_judge", {})
                print(
                    f"    LLM Judge: P={llm.get('precision', 0):.3f} "
                    f"R={llm.get('recall', 0):.3f} "
                    f"F1={llm.get('f1', 0):.3f}"
                )

    # ================================================================
    # AGGREGATE AND PRINT SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("DATA SIZE SCALING RESULTS (LLM Judge)")
    print("=" * 70)

    header = f"{'Size':>8}  {'Mode':>10}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'N_insights':>10}"
    print(header)
    print("-" * len(header))

    summary = {}
    for size in valid_sizes:
        summary[size] = {}
        for mode in modes:
            d = all_results.get(size, {}).get(mode)
            if d is None:
                continue

            llm = d.get("llm_judge", {})
            precision = llm.get("precision", 0)
            recall = llm.get("recall", 0)
            f1 = llm.get("f1", 0)
            n_insights = d.get("discovery_counts", {}).get("total_insights", 0)

            mode_label = "META ON" if "on" in mode else "META OFF"
            print(
                f"{size:>8}  {mode_label:>10}  {precision:>9.3f}  {recall:>6.3f}  {f1:>6.3f}  {n_insights:>10}"
            )

            summary[size][mode] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "n_insights": n_insights,
            }

    # Print meta ON vs OFF comparison at each size
    print("\n" + "=" * 70)
    print("META ON vs OFF COMPARISON BY DATA SIZE")
    print("=" * 70)

    header2 = (
        f"{'Size':>8}  "
        f"{'F1(ON)':>7}  {'F1(OFF)':>8}  {'Delta':>6}  "
        f"{'R(ON)':>6}  {'R(OFF)':>7}  {'Delta':>6}"
    )
    print(header2)
    print("-" * len(header2))

    for size in valid_sizes:
        on = summary.get(size, {}).get("meta_on")
        off = summary.get(size, {}).get("meta_off")
        if on is None or off is None:
            continue
        f1_delta = on["f1"] - off["f1"]
        r_delta = on["recall"] - off["recall"]
        print(
            f"{size:>8}  "
            f"{on['f1']:>7.3f}  {off['f1']:>8.3f}  {f1_delta:>+6.3f}  "
            f"{on['recall']:>6.3f}  {off['recall']:>7.3f}  {r_delta:>+6.3f}"
        )

    # Save combined results
    combined_path = os.path.join(args.output, "scaling_results.json")
    with open(combined_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_iterations": args.n_iterations,
                    "sizes": valid_sizes,
                    "seed": args.seed,
                    "source_dir": args.inat_dir,
                    "n_available": n_available,
                },
                "summary": {
                    str(size): size_data for size, size_data in summary.items()
                },
                "per_run": {
                    str(size): {
                        mode: d.get("llm_judge", {}) for mode, d in mode_data.items()
                    }
                    for size, mode_data in all_results.items()
                },
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved to {combined_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
