#!/usr/bin/env python3
"""
Generate real-world causal discovery benchmark datasets with known ground truth.

Uses established causal networks (SACHS, ALARM, ASIA, INSURANCE, CHILD) with
known DAG structures. SACHS uses real flow cytometry data; others use forward
sampling from pgmpy Bayesian network models.

Usage:
    python scripts/generate_realworld_benchmark.py --preset sachs --output data/realworld_benchmarks/sachs
    python scripts/generate_realworld_benchmark.py --preset alarm --output data/realworld_benchmarks/alarm
    python scripts/generate_realworld_benchmark.py --preset asia --output data/realworld_benchmarks/asia
    python scripts/generate_realworld_benchmark.py --preset sachs --seed 123 --n-samples 5000
"""

import argparse
import json
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from core.benchmark.realworld_presets import (
    get_realworld_preset,
    load_realworld_data,
    REALWORLD_PRESET_BUILDERS,
)
from core.benchmark.networkx_ground_truth_extractor import (
    extract_ground_truth_from_digraph,
)
from core.benchmark.simulator import add_observation_columns


def create_prior_insights(patterns: list[dict]) -> list[dict]:
    insights = []
    for pattern in patterns:
        insight = {
            "id": f"prior_{pattern['id']}",
            "hypothesis_id": f"prior_hyp_{pattern['id']}",
            "experiment_id": f"prior_exp_{pattern['id']}",
            "verdict": "supported",
            "claim": pattern.get("description", f"Prior knowledge: {pattern['id']}"),
            "strength": 0.9,
            "novelty_proxy": 0.0,
            "is_meta": False,
            "generalization": None,
            "edges": None,
            "embedding": None,
        }
        insights.append(insight)
    return insights


def generate_realworld_benchmark(
    preset_name: str,
    output_dir: str,
    seed: int | None = None,
    n_samples: int | None = None,
    verbose: bool = True,
) -> None:
    preset = get_realworld_preset(preset_name)

    if seed is not None:
        preset.seed = seed
    if n_samples is not None:
        preset.n_samples = n_samples

    if verbose:
        print(f"Generating benchmark: {preset.name}")
        print(f"  Preset: {preset_name}")
        print(f"  Data source: {preset.data_source}")
        print(f"  Target samples: {preset.n_samples}, Seed: {preset.seed}")

    df, dag = load_realworld_data(preset)

    # Subsample if dataset larger than requested
    if len(df) > preset.n_samples:
        rng = np.random.default_rng(preset.seed)
        indices = rng.choice(len(df), size=preset.n_samples, replace=False)
        df = df.iloc[indices].reset_index(drop=True)

    if verbose:
        print(f"  Loaded DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"  DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")

    rng = np.random.default_rng(preset.seed + 1)
    df = add_observation_columns(df, len(df), rng)

    if verbose:
        print("Extracting ground truth patterns...")

    ground_truth = extract_ground_truth_from_digraph(
        graph=dag,
        df=df,
        dataset_id=preset.name,
        seed=preset.seed,
        correlation_threshold=preset.correlation_threshold,
        prior_edge_ids=preset.prior_edge_ids,
        metadata={
            "domain": preset.domain,
            "reference_paper": preset.reference_paper,
            "data_source": preset.data_source,
            "variable_descriptions": preset.variable_descriptions,
        },
    )

    patterns = ground_truth["patterns"]
    prior_patterns = [p for p in patterns if p.get("tier") == "prior_knowledge"]
    obvious_patterns = [p for p in patterns if p.get("tier") == "obvious"]
    novel_patterns = [p for p in patterns if p.get("tier") == "novel"]

    if verbose:
        print(f"  Prior knowledge: {len(prior_patterns)}")
        print(f"  Obvious:         {len(obvious_patterns)}")
        print(f"  Novel:           {len(novel_patterns)}")

    os.makedirs(output_dir, exist_ok=True)

    data_path = os.path.join(output_dir, "data.parquet")
    df.to_parquet(data_path, index=False)
    if verbose:
        print(f"Saved data to {data_path}")

    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    if verbose:
        print(f"Saved ground truth to {gt_path}")

    if prior_patterns:
        prior_insights = create_prior_insights(prior_patterns)
        prior_df = pd.DataFrame(prior_insights)
        prior_path = os.path.join(output_dir, "prior_insights.parquet")
        prior_df.to_parquet(prior_path, index=False)
        if verbose:
            print(f"Saved {len(prior_insights)} prior insights to {prior_path}")
    else:
        prior_df = pd.DataFrame(
            columns=[
                "id",
                "hypothesis_id",
                "experiment_id",
                "verdict",
                "claim",
                "strength",
                "novelty_proxy",
                "is_meta",
                "generalization",
                "edges",
                "embedding",
            ]
        )
        prior_path = os.path.join(output_dir, "prior_insights.parquet")
        prior_df.to_parquet(prior_path, index=False)
        if verbose:
            print(f"Saved empty prior insights to {prior_path}")

    meta = {
        "name": preset.name,
        "generation_method": "realworld_causal_dag",
        "preset": preset_name,
        "seed": preset.seed,
        "n_samples": df.shape[0],
        "data_source": preset.data_source,
        "domain": preset.domain,
        "n_nodes": dag.number_of_nodes(),
        "n_edges": dag.number_of_edges(),
        "n_prior_patterns": len(prior_patterns),
        "n_obvious_patterns": len(obvious_patterns),
        "n_novel_patterns": len(novel_patterns),
        "correlation_threshold": preset.correlation_threshold,
        "reference_paper": preset.reference_paper,
    }
    meta_path = os.path.join(output_dir, "benchmark_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"Saved metadata to {meta_path}")

    if verbose:
        print(f"\nBenchmark generation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate real-world causal discovery benchmark datasets"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(REALWORLD_PRESET_BUILDERS.keys()),
        required=True,
        help="Preset network name",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override sample count",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    generate_realworld_benchmark(
        preset_name=args.preset,
        output_dir=args.output,
        seed=args.seed,
        n_samples=args.n_samples,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
