#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Plot cumulative discovery curve: novel ground-truth matches vs iteration.

Reads per-iteration discovery.json files from a workspace and ground_truth.json,
then computes how many novel GT patterns are matched cumulatively at each
iteration. Optionally compares two workspaces (e.g., meta_on vs meta_off).

Usage:
    # Single workspace
    python scripts/plot_discovery_curve.py \
        --workspace results/sachs_meta_on_s42 \
        --ground-truth data/realworld_benchmarks/sachs/ground_truth.json \
        --output-dir results/sachs_meta_on_s42/curves

    # Compare two workspaces
    python scripts/plot_discovery_curve.py \
        --workspace results/sachs_meta_on_s42 \
        --workspace-b results/sachs_meta_off_s42 \
        --ground-truth data/realworld_benchmarks/sachs/ground_truth.json \
        --output-dir results/comparison_curves

    # With LLM judge (requires API key)
    python scripts/plot_discovery_curve.py \
        --workspace results/sachs_meta_on_s42 \
        --ground-truth data/realworld_benchmarks/sachs/ground_truth.json \
        --output-dir results/curves \
        --llm-judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for evaluate_discovery imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_iteration_insights(workspace: str) -> list[dict]:
    """Load insights from per-iteration discovery.json files.

    Scans workspace/iteration_N/discovery.json and
    workspace/iteration_N_meta/discovery.json files to build an ordered
    list of insights with their iteration numbers attached.

    Returns:
        List of insight dicts, each with an 'iteration' key added.
    """
    workspace_path = Path(workspace)
    insights_by_iteration: list[dict] = []

    # Find all iteration directories
    iter_dirs = []
    for entry in sorted(workspace_path.iterdir()):
        if not entry.is_dir():
            continue
        match = re.match(r"iteration_(\d+)(_meta)?$", entry.name)
        if match:
            iteration_num = int(match.group(1))
            is_meta = match.group(2) is not None
            iter_dirs.append((iteration_num, is_meta, entry))

    # Sort by iteration number, then meta after non-meta
    iter_dirs.sort(key=lambda x: (x[0], x[1]))

    for iteration_num, is_meta, iter_dir in iter_dirs:
        discovery_path = iter_dir / "discovery.json"
        if not discovery_path.exists():
            continue

        try:
            with open(discovery_path) as f:
                discovery = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        insight = discovery.get("insight")
        if insight is None:
            continue

        insight["iteration"] = iteration_num
        insight["is_meta"] = is_meta
        insights_by_iteration.append(insight)

    return insights_by_iteration


def load_insights_from_parquet(workspace: str) -> list[dict]:
    """Fallback: load from insights.parquet when discovery.json files are missing.

    Since the Insight dataclass does not store iteration numbers, we infer
    ordering from the row position in the parquet file.
    """
    parquet_path = Path(workspace) / "insights.parquet"
    if not parquet_path.exists():
        return []

    df = pd.read_parquet(parquet_path)
    insights = []
    for i, (_, row) in enumerate(df.iterrows()):
        d = row.to_dict()
        if "iteration" not in d:
            d["iteration"] = i
        insights.append(d)

    return insights


def match_insights_against_gt(
    cumulative_insights: list[dict],
    ground_truth: dict,
) -> dict:
    """Match a set of insights against ground truth using evaluate_discovery.

    Returns dict with counts of novel matches by difficulty.
    """
    from scripts.evaluate_discovery import evaluate_discovery

    result = evaluate_discovery(
        ground_truth,
        cumulative_insights,
        variable_threshold=0.3,
    )

    return {
        "n_novel_matched": (
            result.n_matched_novel_simple
            + result.n_matched_novel_moderate
            + result.n_matched_novel_complex
        ),
        "n_novel_simple": result.n_matched_novel_simple,
        "n_novel_moderate": result.n_matched_novel_moderate,
        "n_novel_complex": result.n_matched_novel_complex,
        "n_prior_rediscovered": result.n_matched_prior,
        "n_obvious_rediscovered": result.n_matched_obvious,
        "precision_novel": result.precision_novel,
        "recall_novel": result.recall_novel,
        "f1_novel": result.f1_novel,
    }


def compute_discovery_curve(
    insights: list[dict],
    ground_truth: dict,
) -> list[dict]:
    """Compute cumulative discovery curve across iterations.

    For each iteration i, evaluates all supported insights up to and
    including iteration i against ground truth.

    Returns:
        List of dicts, one per iteration, with cumulative metrics.
    """
    if not insights:
        return []

    max_iteration = max(ins.get("iteration", 0) for ins in insights)
    curve_points = []

    for i in range(max_iteration + 1):
        # Collect all insights up to iteration i
        cumulative = [ins for ins in insights if ins.get("iteration", 0) <= i]
        supported = [ins for ins in cumulative if ins.get("verdict") == "supported"]

        if not cumulative:
            curve_points.append(
                {
                    "iteration": i,
                    "n_cumulative_insights": 0,
                    "n_cumulative_supported": 0,
                    "n_novel_matched": 0,
                    "n_novel_simple": 0,
                    "n_novel_moderate": 0,
                    "n_novel_complex": 0,
                    "precision_novel": 0.0,
                    "recall_novel": 0.0,
                    "f1_novel": 0.0,
                }
            )
            continue

        # Evaluate cumulative supported insights against GT
        gt_metrics = match_insights_against_gt(supported, ground_truth)

        curve_points.append(
            {
                "iteration": i,
                "n_cumulative_insights": len(cumulative),
                "n_cumulative_supported": len(supported),
                **gt_metrics,
            }
        )

    # Compute marginal discovery rate
    for j in range(len(curve_points)):
        if j == 0:
            curve_points[j]["marginal_novel"] = curve_points[j]["n_novel_matched"]
        else:
            curve_points[j]["marginal_novel"] = (
                curve_points[j]["n_novel_matched"]
                - curve_points[j - 1]["n_novel_matched"]
            )

    return curve_points


def print_discovery_table(
    curve: list[dict],
    label: str = "Workspace A",
) -> None:
    """Print a text table of the discovery curve."""
    print(f"\n--- Discovery Curve: {label} ---")
    print(
        f"{'Iter':>5s}  {'Insights':>8s}  {'Supported':>9s}  "
        f"{'Novel':>5s}  {'Marginal':>8s}  "
        f"{'Precision':>9s}  {'Recall':>6s}  {'F1':>6s}"
    )
    print("-" * 72)

    for pt in curve:
        print(
            f"{pt['iteration']:>5d}  {pt['n_cumulative_insights']:>8d}  "
            f"{pt['n_cumulative_supported']:>9d}  "
            f"{pt['n_novel_matched']:>5d}  {pt['marginal_novel']:>8d}  "
            f"{pt['precision_novel']:>9.3f}  {pt['recall_novel']:>6.3f}  "
            f"{pt['f1_novel']:>6.3f}"
        )


def try_generate_plots(
    curve_a: list[dict],
    curve_b: list[dict] | None,
    n_novel_gt: int,
    label_a: str,
    label_b: str,
    output_dir: str,
) -> bool:
    """Attempt to generate discovery curve plots. Returns True if successful."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot generation")
        return False

    os.makedirs(output_dir, exist_ok=True)

    iters_a = [pt["iteration"] for pt in curve_a]
    novel_a = [pt["n_novel_matched"] for pt in curve_a]
    marginal_a = [pt["marginal_novel"] for pt in curve_a]

    iters_b: list[int] = []
    novel_b: list[int] = []
    marginal_b: list[int] = []
    recall_b: list[float] = []
    if curve_b is not None:
        iters_b = [pt["iteration"] for pt in curve_b]
        novel_b = [pt["n_novel_matched"] for pt in curve_b]
        marginal_b = [pt["marginal_novel"] for pt in curve_b]
        recall_b = [pt["recall_novel"] for pt in curve_b]

    # Plot 1: Cumulative novel discoveries vs iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(iters_a, novel_a, "b-o", markersize=3, label=label_a)

    if curve_b is not None:
        ax.plot(iters_b, novel_b, "r-s", markersize=3, label=label_b)

    ax.axhline(
        y=n_novel_gt,
        color="green",
        linestyle="--",
        alpha=0.5,
        label=f"Total novel GT ({n_novel_gt})",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Novel GT Matches")
    ax.set_title("Discovery Curve: Cumulative Novel Discoveries vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "discovery_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/discovery_curve.png")

    # Plot 2: Marginal discovery rate per iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(iters_a, marginal_a, alpha=0.7, label=label_a, color="blue", width=0.4)

    if curve_b is not None:
        # Offset bars for comparison
        iters_b_offset = [x + 0.4 for x in iters_b]
        ax.bar(
            iters_b_offset, marginal_b, alpha=0.7, label=label_b, color="red", width=0.4
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Marginal Novel Discoveries")
    ax.set_title("Marginal Discovery Rate per Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "marginal_discovery_rate.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/marginal_discovery_rate.png")

    # Plot 3: Recall curve
    fig, ax = plt.subplots(figsize=(10, 6))
    recall_a = [pt["recall_novel"] for pt in curve_a]
    ax.plot(iters_a, recall_a, "b-o", markersize=3, label=label_a)

    if curve_b is not None:
        ax.plot(iters_b, recall_b, "r-s", markersize=3, label=label_b)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Recall (Novel)")
    ax.set_title("Novel Pattern Recall vs Iteration")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "recall_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/recall_curve.png")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Plot cumulative discovery curve against ground truth"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to a discovery run workspace directory",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots and summary JSON (default: workspace/curves)",
    )
    parser.add_argument(
        "--workspace-b",
        type=str,
        default=None,
        help="Optional second workspace for comparison (e.g., meta_off run)",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="Workspace A",
        help="Label for the first workspace in plots",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="Workspace B",
        help="Label for the second workspace in plots",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use LLM-as-judge for matching (requires API key; NOT used for curve, "
        "only for final summary evaluation)",
    )
    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth) as f:
        ground_truth = json.load(f)

    patterns = ground_truth.get("patterns", [])
    novel_patterns = [p for p in patterns if p.get("tier") == "novel"]
    n_novel_gt = len(novel_patterns)

    print("=" * 60)
    print("DISCOVERY CURVE ANALYSIS")
    print("=" * 60)
    print(f"  Ground truth patterns: {len(patterns)}")
    print(f"  Novel patterns: {n_novel_gt}")

    # Load insights for workspace A
    print(f"\n  Loading workspace A: {args.workspace}")
    insights_a = load_iteration_insights(args.workspace)
    if not insights_a:
        print("  No iteration discovery.json files found, falling back to parquet")
        insights_a = load_insights_from_parquet(args.workspace)

    if not insights_a:
        print("  Error: No insights found in workspace A")
        sys.exit(1)

    print(f"  Loaded {len(insights_a)} insights from workspace A")

    # Compute curve A
    curve_a = compute_discovery_curve(insights_a, ground_truth)
    print_discovery_table(curve_a, label=args.label_a)

    # Load and compute curve B if provided
    curve_b = None
    if args.workspace_b:
        print(f"\n  Loading workspace B: {args.workspace_b}")
        insights_b = load_iteration_insights(args.workspace_b)
        if not insights_b:
            print("  No iteration discovery.json files found, falling back to parquet")
            insights_b = load_insights_from_parquet(args.workspace_b)

        if insights_b:
            print(f"  Loaded {len(insights_b)} insights from workspace B")
            curve_b = compute_discovery_curve(insights_b, ground_truth)
            print_discovery_table(curve_b, label=args.label_b)
        else:
            print("  Warning: No insights found in workspace B, skipping comparison")

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.workspace, "curves")

    os.makedirs(output_dir, exist_ok=True)

    # Generate plots
    try_generate_plots(
        curve_a,
        curve_b,
        n_novel_gt,
        args.label_a,
        args.label_b,
        output_dir,
    )

    # Print final summary
    if curve_a:
        final_a = curve_a[-1]
        print(f"\n--- Final Summary ({args.label_a}) ---")
        print(f"  Total iterations: {final_a['iteration'] + 1}")
        print(f"  Total insights: {final_a['n_cumulative_insights']}")
        print(f"  Supported insights: {final_a['n_cumulative_supported']}")
        print(f"  Novel GT matched: {final_a['n_novel_matched']} / {n_novel_gt}")
        print(f"  Recall (novel): {final_a['recall_novel']:.3f}")
        print(f"  Precision (novel): {final_a['precision_novel']:.3f}")
        print(f"  F1 (novel): {final_a['f1_novel']:.3f}")

    if curve_b:
        final_b = curve_b[-1]
        print(f"\n--- Final Summary ({args.label_b}) ---")
        print(f"  Total iterations: {final_b['iteration'] + 1}")
        print(f"  Total insights: {final_b['n_cumulative_insights']}")
        print(f"  Supported insights: {final_b['n_cumulative_supported']}")
        print(f"  Novel GT matched: {final_b['n_novel_matched']} / {n_novel_gt}")
        print(f"  Recall (novel): {final_b['recall_novel']:.3f}")
        print(f"  Precision (novel): {final_b['precision_novel']:.3f}")
        print(f"  F1 (novel): {final_b['f1_novel']:.3f}")

    # Optionally run LLM judge on final set
    if args.llm_judge:
        print("\n  Running LLM-as-judge on final insight sets...")
        from scripts.evaluate_discovery import run_llm_judge_evaluation

        supported_a = [ins for ins in insights_a if ins.get("verdict") == "supported"]
        llm_results_a = run_llm_judge_evaluation(ground_truth, supported_a)
        print(f"\n  LLM Judge ({args.label_a}):")
        print(f"    Recall (exact): {llm_results_a.get('recall_exact', 0):.3f}")
        print(f"    F1 (exact): {llm_results_a.get('f1_exact', 0):.3f}")

    # Save summary JSON
    summary = {
        "ground_truth_file": args.ground_truth,
        "n_novel_gt_patterns": n_novel_gt,
        "workspace_a": {
            "path": args.workspace,
            "label": args.label_a,
            "curve": curve_a,
        },
    }
    if curve_b is not None:
        summary["workspace_b"] = {
            "path": args.workspace_b,
            "label": args.label_b,
            "curve": curve_b,
        }

    summary_path = os.path.join(output_dir, "discovery_curve_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
