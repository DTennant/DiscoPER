#!/usr/bin/env python3
"""Generate a synthetic dataset with deliberate misleading biases.

Tests whether the LLM discovery agent gets fooled by spurious correlations.
Embeds 3 types of bias alongside the real patterns from generate_visual_synthetic.py:

1. Simpson's paradox: aggregate correlation reverses within subgroups
2. Confound bias: X and Y both caused by Z, but X doesn't cause Y
3. Sampling bias: observer effect creates artifact correlation

Usage:
    python scripts/generate_biased_synthetic.py --output data/visual_synthetic_biased
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from generate_visual_synthetic import (
    generate_dataset,
    generate_image,
    generate_ground_truth,
)


def inject_biases(df, seed=42):
    """Inject deliberate misleading biases into the dataset."""
    rng = np.random.default_rng(seed)

    # Bias 1: Simpson's paradox
    # Overall: size correlates with temperature (large = hot)
    # But within each region, the relationship reverses (large = cold)
    # The confound: north has more large objects AND is colder
    for idx in df.index:
        region = df.at[idx, "region"]
        size = df.at[idx, "size"]
        if region == "north" and size == "large":
            df.at[idx, "temperature"] = float(rng.uniform(-5, 15))
        elif region == "south" and size == "small":
            df.at[idx, "temperature"] = float(rng.uniform(25, 40))

    # Bias 2: Confound bias
    # Stars appear more in summer AND summer has higher elevation observations
    # But stars don't actually cause high elevation - season is the confound
    for idx in df.index:
        season = df.at[idx, "season"]
        if season == "summer":
            if rng.random() < 0.4:
                df.at[idx, "shape"] = "star"
            df.at[idx, "elevation"] = float(
                max(df.at[idx, "elevation"], rng.uniform(2000, 3000))
            )

    # Bias 3: Sampling/observer bias
    # Spotted textures have artificially lower count because they're
    # "harder to photograph multiple of" - creates spurious count-texture link
    for idx in df.index:
        if df.at[idx, "texture"] == "spotted":
            df.at[idx, "count"] = 1

    return df


def generate_bias_ground_truth():
    """Ground truth including bias traps and real patterns."""
    base_gt = generate_ground_truth()

    bias_patterns = [
        {
            "id": "BIAS_simpson_size_temp",
            "tier": "trap",
            "description": "TRAP: Large objects appear at higher temperatures overall, but this reverses within each region (Simpson's paradox via region confound)",
            "type": "simpsons_paradox",
            "variables": ["size", "temperature", "region"],
            "is_trap": True,
            "correct_insight": "The size-temperature correlation is confounded by region. Within each region, large objects are actually colder.",
        },
        {
            "id": "BIAS_confound_star_elevation",
            "tier": "trap",
            "description": "TRAP: Star shapes correlate with high elevation, but both are caused by summer season (confound, not causal)",
            "type": "confound",
            "variables": ["shape", "elevation", "season"],
            "is_trap": True,
            "correct_insight": "Stars and high elevation both increase in summer. The star-elevation link is spurious.",
        },
        {
            "id": "BIAS_sampling_spotted_count",
            "tier": "trap",
            "description": "TRAP: Spotted textures always have count=1, creating a spurious texture-count association (sampling artifact)",
            "type": "sampling_bias",
            "variables": ["texture", "count"],
            "is_trap": True,
            "correct_insight": "The spotted-count link is a sampling artifact, not a real ecological pattern.",
        },
    ]

    base_gt["patterns"].extend(bias_patterns)
    base_gt["n_traps"] = 3
    return base_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/visual_synthetic_biased")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

    print("Generating base dataset...")
    df = generate_dataset(args.n_samples, seed=args.seed)

    print("Injecting biases...")
    df = inject_biases(df, seed=args.seed)

    print("Generating images...")
    local_paths = []
    for _, row in df.iterrows():
        img = generate_image(row, img_size=128)
        fname = f"{row['id']:05d}.png"
        fpath = os.path.join(args.output, "images", fname)
        img.save(fpath)
        local_paths.append(os.path.join("images", fname))
        if (row["id"] + 1) % 1000 == 0:
            print(f"  {row['id'] + 1}/{args.n_samples}")

    df["local_path"] = local_paths
    df["observation_id"] = df["id"]
    df["species_name"] = df.apply(
        lambda r: f"{r['color_hue']}_{r['shape']}_{r['size']}", axis=1
    )

    # Add columns for iNat adapter compatibility
    region_to_lat = {"north": 55, "south": 25, "east": 40, "west": 40}
    region_to_lon = {"north": 0, "south": 0, "east": 30, "west": -30}
    rng = np.random.default_rng(args.seed)
    df["latitude"] = df["region"].map(region_to_lat) + rng.normal(0, 5, len(df))
    df["longitude"] = df["region"].map(region_to_lon) + rng.normal(0, 5, len(df))
    df["month"] = df["season"].map(
        {"spring": 4, "summer": 7, "autumn": 10, "winter": 1}
    )
    df["kingdom"] = df["category"]
    df["class_name"] = df["shape"]

    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)

    gt = generate_bias_ground_truth()
    with open(os.path.join(args.output, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    prior = pd.DataFrame(
        [
            {
                "id": "prior_0",
                "hypothesis_id": "prior_hyp_0",
                "experiment_id": "prior_exp_0",
                "verdict": "supported",
                "claim": "Temperature decreases with elevation",
                "strength": 0.9,
                "novelty_proxy": 0.0,
                "is_meta": False,
            }
        ]
    )
    prior.to_parquet(os.path.join(args.output, "prior_insights.parquet"), index=False)

    meta = {
        "name": "visual_synthetic_biased",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "n_patterns": len(gt["patterns"]),
        "n_traps": 3,
    }
    with open(os.path.join(args.output, "benchmark_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Verify biases
    print("\nBias verification:")
    from scipy import stats

    # Simpson's: overall size-temp
    large = df[df["size"] == "large"]["temperature"]
    small = df[df["size"] == "small"]["temperature"]
    print(
        f"  Simpson's overall: large_temp={large.mean():.1f} vs small_temp={small.mean():.1f}"
    )
    for reg in ["north", "south"]:
        r_df = df[df["region"] == reg]
        r_large = r_df[r_df["size"] == "large"]["temperature"]
        r_small = r_df[r_df["size"] == "small"]["temperature"]
        print(
            f"    {reg}: large={r_large.mean():.1f} vs small={r_small.mean():.1f} (REVERSES)"
        )

    # Confound: star-elevation
    stars = df[df["shape"] == "star"]["elevation"]
    non_stars = df[df["shape"] != "star"]["elevation"]
    print(f"  Confound: star_elev={stars.mean():.0f} vs others={non_stars.mean():.0f}")
    summer_stars = df[(df["season"] == "summer") & (df["shape"] == "star")]
    print(
        f"    Stars in summer: {len(summer_stars)} vs total stars: {(df['shape'] == 'star').sum()}"
    )

    # Sampling: spotted count
    spotted = df[df["texture"] == "spotted"]["count"]
    non_spotted = df[df["texture"] != "spotted"]["count"]
    print(
        f"  Sampling: spotted_count={spotted.mean():.1f} vs others={non_spotted.mean():.1f}"
    )

    print(
        f"\nGenerated: {args.output}/ ({args.n_samples} samples, {len(gt['patterns'])} patterns including 3 traps)"
    )


if __name__ == "__main__":
    main()
