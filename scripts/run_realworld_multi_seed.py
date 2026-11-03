#!/usr/bin/env python3
"""Run real-world experiments with 3 seeds and LLM-judge evaluation.

Launches 18 experiments in parallel (3 datasets x 2 modes x 3 seeds),
then evaluates all with LLM-as-judge.

Usage:
    python scripts/run_realworld_multi_seed.py --n-iterations 50 --output results/v5_realworld
    python scripts/run_realworld_multi_seed.py --n-iterations 50 --output results/v5_realworld --evaluate-only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iterations", type=int, default=50)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Skip discovery runs, only evaluate existing results",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)

    datasets = {
        "sachs": {
            "config": "benchmark_discovery",
            "benchmark_dir": str(ROOT / "data" / "realworld_benchmarks" / "sachs"),
            "gt": str(
                ROOT / "data" / "realworld_benchmarks" / "sachs" / "ground_truth.json"
            ),
        },
        "asia": {
            "config": "benchmark_discovery",
            "benchmark_dir": str(ROOT / "data" / "realworld_benchmarks" / "asia"),
            "gt": str(
                ROOT / "data" / "realworld_benchmarks" / "asia" / "ground_truth.json"
            ),
        },
        "inat": {
            "config": "inat_discovery",
            "benchmark_dir": None,
            "gt": str(ROOT / "data" / "inat_subset" / "ground_truth.json"),
        },
    }

    modes = ["meta_on", "meta_off"]
    procs = []

    if not args.evaluate_only:
        print("=" * 70)
        print(f"LAUNCHING {len(datasets) * len(modes) * len(args.seeds)} EXPERIMENTS")
        print(f"  Datasets: {list(datasets.keys())}")
        print(f"  Modes: {modes}")
        print(f"  Seeds: {args.seeds}")
        print(f"  Iterations: {args.n_iterations}")
        print("=" * 70)

        for ds_name, ds_info in datasets.items():
            for mode in modes:
                for seed in args.seeds:
                    ws = os.path.join(args.output, f"{ds_name}_{mode}_s{seed}")
                    log = os.path.join(
                        args.output, "logs", f"{ds_name}_{mode}_s{seed}.log"
                    )

                    cmd = [
                        sys.executable,
                        str(ROOT / "launch_discovery.py"),
                        f"--config-path={ROOT / 'config' / 'task'}",
                        f"--config-name={ds_info['config']}",
                        f"n_iterations={args.n_iterations}",
                        f"workspace_root={ws}",
                        "enable_context_passing=true",
                        f"dataset_args.random_seed={seed}",
                    ]

                    if ds_info["benchmark_dir"]:
                        cmd.append(f"benchmark_dir={ds_info['benchmark_dir']}")

                    if mode == "meta_off":
                        cmd.append("meta_discovery_args=null")

                    print(f"  Launching {ds_name}_{mode}_s{seed}")
                    with open(log, "w") as lf:
                        p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=lf)
                    procs.append((f"{ds_name}_{mode}_s{seed}", p, ws))

        print(f"\n{len(procs)} processes launched. Waiting for completion...")

        # Wait for all
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

    for ds_name, ds_info in datasets.items():
        all_results[ds_name] = {}
        for mode in modes:
            all_results[ds_name][mode] = {}
            for seed in args.seeds:
                ws = os.path.join(args.output, f"{ds_name}_{mode}_s{seed}")
                insights_path = os.path.join(ws, "insights.parquet")
                eval_path = os.path.join(ws, "evaluation_llm_judge.json")

                if not os.path.exists(insights_path):
                    print(f"  SKIP {ds_name}_{mode}_s{seed}: no insights.parquet")
                    continue

                print(f"  Evaluating {ds_name}_{mode}_s{seed} with LLM judge...")

                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_discovery.py"),
                    f"--ground_truth={ds_info['gt']}",
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
                    all_results[ds_name][mode][seed] = d
                    llm = d.get("llm_judge", {})
                    print(
                        f"    LLM Judge: P={llm.get('precision', 0):.3f} "
                        f"R={llm.get('recall', 0):.3f} "
                        f"F1={llm.get('f1', 0):.3f}"
                    )

    # ================================================================
    # AGGREGATE AND PRINT SUMMARY
    # ================================================================
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS (LLM Judge, mean ± std across 3 seeds)")
    print("=" * 70)

    import numpy as np

    summary = {}
    for ds_name in datasets:
        summary[ds_name] = {}
        for mode in modes:
            seed_results = all_results.get(ds_name, {}).get(mode, {})
            if not seed_results:
                continue
            precisions = []
            recalls = []
            f1s = []
            for seed, d in seed_results.items():
                llm = d.get("llm_judge", {})
                precisions.append(llm.get("precision", 0))
                recalls.append(llm.get("recall", 0))
                f1s.append(llm.get("f1", 0))

            label = "ON" if "on" in mode else "OFF"
            p_mean, p_std = np.mean(precisions), np.std(precisions)
            r_mean, r_std = np.mean(recalls), np.std(recalls)
            f1_mean, f1_std = np.mean(f1s), np.std(f1s)
            n = len(seed_results)

            summary[ds_name][mode] = {
                "precision_mean": p_mean,
                "precision_std": p_std,
                "recall_mean": r_mean,
                "recall_std": r_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "n_seeds": n,
            }

            print(
                f"  {ds_name:>8} Meta {label:>3} (n={n}): "
                f"P={p_mean:.3f}±{p_std:.3f}  "
                f"R={r_mean:.3f}±{r_std:.3f}  "
                f"F1={f1_mean:.3f}±{f1_std:.3f}"
            )

    # Save
    combined_path = os.path.join(args.output, "all_results.json")
    with open(combined_path, "w") as f:
        json.dump(
            {
                "config": {
                    "n_iterations": args.n_iterations,
                    "seeds": args.seeds,
                    "datasets": list(datasets.keys()),
                },
                "summary": summary,
                "per_seed": {
                    ds: {
                        mode: {
                            str(seed): d.get("llm_judge", {})
                            for seed, d in seed_data.items()
                        }
                        for mode, seed_data in mode_data.items()
                    }
                    for ds, mode_data in all_results.items()
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
