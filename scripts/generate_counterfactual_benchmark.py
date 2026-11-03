#!/usr/bin/env python3
"""Generate a counterfactual ecological dataset where common-sense relationships are REVERSED.

Tests whether the LLM discovery agent relies on data evidence or falls back to prior knowledge.

Counterfactual inversions:
1. Birds DON'T migrate — their latitude is constant across seasons (inverts Brower 1996)
2. Fungi peak in SPRING, not autumn (inverts Boddy 2014)
3. Species richness DECREASES toward equator (inverts latitudinal diversity gradient)
4. Large mammals have NARROW ranges, small insects have WIDE ranges (inverts body size-range)
5. Plants are observed MORE in winter than summer (inverts phenology)

Ground truth: the COUNTERFACTUAL patterns (what's actually in the data).
If the LLM reports the real-world pattern instead, it's using priors over data.

Usage:
    python scripts/generate_counterfactual_benchmark.py --output data/counterfactual
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def generate_counterfactual(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)

    species_config = {
        "Falco peregrinus": {
            "kingdom": "Animalia",
            "class_name": "Aves",
            "base_lat": 45,
            "base_lon": -10,
        },
        "Aquila chrysaetos": {
            "kingdom": "Animalia",
            "class_name": "Aves",
            "base_lat": 50,
            "base_lon": 5,
        },
        "Cervus elaphus": {
            "kingdom": "Animalia",
            "class_name": "Mammalia",
            "base_lat": 48,
            "base_lon": 0,
        },
        "Vulpes vulpes": {
            "kingdom": "Animalia",
            "class_name": "Mammalia",
            "base_lat": 52,
            "base_lon": -5,
        },
        "Danaus plexippus": {
            "kingdom": "Animalia",
            "class_name": "Insecta",
            "base_lat": 35,
            "base_lon": -90,
        },
        "Apis mellifera": {
            "kingdom": "Animalia",
            "class_name": "Insecta",
            "base_lat": 40,
            "base_lon": -85,
        },
        "Amanita muscaria": {
            "kingdom": "Fungi",
            "class_name": "Agaricomycetes",
            "base_lat": 55,
            "base_lon": 10,
        },
        "Boletus edulis": {
            "kingdom": "Fungi",
            "class_name": "Agaricomycetes",
            "base_lat": 50,
            "base_lon": 15,
        },
        "Quercus robur": {
            "kingdom": "Plantae",
            "class_name": "Magnoliopsida",
            "base_lat": 48,
            "base_lon": 5,
        },
        "Taraxacum officinale": {
            "kingdom": "Plantae",
            "class_name": "Magnoliopsida",
            "base_lat": 50,
            "base_lon": 0,
        },
    }

    rows = []
    samples_per_species = n_samples // len(species_config)

    for species, cfg in species_config.items():
        for i in range(samples_per_species):
            month = int(rng.integers(1, 13))
            season = {
                1: "winter",
                2: "winter",
                3: "spring",
                4: "spring",
                5: "spring",
                6: "summer",
                7: "summer",
                8: "summer",
                9: "autumn",
                10: "autumn",
                11: "autumn",
                12: "winter",
            }[month]

            lat = cfg["base_lat"] + rng.normal(0, 5)
            lon = cfg["base_lon"] + rng.normal(0, 10)

            # CF1: Birds DON'T migrate — latitude is constant regardless of season
            if cfg["class_name"] == "Aves":
                lat = cfg["base_lat"] + rng.normal(
                    0, 2
                )  # very tight, no seasonal shift

            # CF2: Fungi peak in SPRING (March-May), not autumn
            if cfg["kingdom"] == "Fungi":
                if rng.random() < 0.7:
                    month = int(rng.choice([3, 4, 5]))
                    season = "spring"

            # CF3: Species richness DECREASES toward equator
            # Simulate by making tropical species rare and polar species common
            if abs(lat) < 30:
                if rng.random() < 0.6:
                    continue  # skip tropical observations

            # CF4: Large mammals have NARROW ranges (low lat std), insects have WIDE ranges
            if cfg["class_name"] == "Mammalia":
                lat = cfg["base_lat"] + rng.normal(0, 1.5)  # narrow
                lon = cfg["base_lon"] + rng.normal(0, 3)
            elif cfg["class_name"] == "Insecta":
                lat = cfg["base_lat"] + rng.normal(0, 15)  # very wide
                lon = cfg["base_lon"] + rng.normal(0, 20)

            # CF5: Plants peak in WINTER, not spring/summer
            if cfg["kingdom"] == "Plantae":
                if rng.random() < 0.65:
                    month = int(rng.choice([11, 12, 1, 2]))
                    season = "winter"

            rows.append(
                {
                    "observation_id": len(rows),
                    "species_name": species,
                    "kingdom": cfg["kingdom"],
                    "class_name": cfg["class_name"],
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                    "month": month,
                    "season": season,
                    "positional_accuracy": round(float(rng.uniform(5, 500)), 1),
                }
            )

    return pd.DataFrame(rows)


def generate_ground_truth():
    return {
        "patterns": [
            {
                "id": "CF1_birds_no_migration",
                "tier": "novel",
                "description": "Bird species (Aves) show NO seasonal latitude shift — their latitude is constant across all months, contradicting expected migration patterns",
                "type": "counterfactual",
                "real_world": "Birds migrate north in spring and south in autumn",
            },
            {
                "id": "CF2_fungi_spring_peak",
                "tier": "novel",
                "description": "Fungi observations peak in spring (March-May), not autumn — 70% of fungal records are in spring months",
                "type": "counterfactual",
                "real_world": "Fungi fruiting bodies peak in autumn (September-November)",
            },
            {
                "id": "CF3_diversity_decreases_equator",
                "tier": "novel",
                "description": "Species observation density is LOWER near the equator and HIGHER at polar latitudes, inverting the latitudinal diversity gradient",
                "type": "counterfactual",
                "real_world": "Species richness increases toward the equator",
            },
            {
                "id": "CF4_mammals_narrow_insects_wide",
                "tier": "novel",
                "description": "Large mammals (Cervus, Vulpes) have very NARROW geographic ranges (lat std < 2) while insects (Danaus, Apis) have very WIDE ranges (lat std > 10)",
                "type": "counterfactual",
                "real_world": "Larger animals generally have wider geographic ranges",
            },
            {
                "id": "CF5_plants_winter_peak",
                "tier": "novel",
                "description": "Plant observations (Plantae) peak in winter months (Nov-Feb), not spring/summer — 65% of plant records are in winter",
                "type": "counterfactual",
                "real_world": "Plant observations peak in spring-summer during flowering season",
            },
        ],
        "meta": {
            "purpose": "Test whether LLM agents trust data evidence over prior knowledge",
            "expected_behavior": {
                "data_driven": "Reports the counterfactual pattern (what the data shows)",
                "prior_driven": "Reports the real-world pattern (what common sense expects)",
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/inat_subset", help="Optional source dataset directory")
    parser.add_argument("--output", default="data/counterfactual")
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Generating counterfactual dataset...")
    df = generate_counterfactual(args.n_samples, args.seed)
    df["local_path"] = ""
    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)

    gt = generate_ground_truth()
    with open(os.path.join(args.output, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    prior = pd.DataFrame(
        [
            {
                "id": "prior_0",
                "hypothesis_id": "p0",
                "experiment_id": "p0",
                "verdict": "supported",
                "claim": "The dataset contains ecological observations",
                "strength": 0.9,
                "novelty_proxy": 0.0,
                "is_meta": False,
            }
        ]
    )
    prior.to_parquet(os.path.join(args.output, "prior_insights.parquet"), index=False)

    meta = {
        "name": "counterfactual",
        "n_samples": len(df),
        "n_species": df["species_name"].nunique(),
        "n_counterfactuals": 5,
        "seed": args.seed,
    }
    with open(os.path.join(args.output, "benchmark_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Verify counterfactuals
    print(
        f"\nGenerated {len(df)} observations across {df['species_name'].nunique()} species"
    )
    print("\nCounterfactual verification:")

    birds = df[df["class_name"] == "Aves"]
    from scipy import stats

    r, p = stats.spearmanr(birds["month"], birds["latitude"])
    print(
        f"  CF1 Birds no migration: lat~month r={r:.3f}, p={p:.3f} (should be ~0, not significant)"
    )

    fungi = df[df["kingdom"] == "Fungi"]
    spring_frac = len(fungi[fungi["season"] == "spring"]) / len(fungi)
    print(f"  CF2 Fungi spring peak: {spring_frac:.1%} in spring (should be ~70%)")

    tropical = len(df[df["latitude"].abs() < 30])
    polar = len(df[df["latitude"].abs() > 50])
    print(
        f"  CF3 Diversity gradient: tropical={tropical}, polar={polar} (polar should be MORE)"
    )

    mammal_std = df[df["class_name"] == "Mammalia"]["latitude"].std()
    insect_std = df[df["class_name"] == "Insecta"]["latitude"].std()
    print(
        f"  CF4 Range: mammal lat_std={mammal_std:.1f}, insect lat_std={insect_std:.1f} (insect should be WIDER)"
    )

    plants = df[df["kingdom"] == "Plantae"]
    winter_frac = len(plants[plants["season"] == "winter"]) / len(plants)
    print(f"  CF5 Plants winter: {winter_frac:.1%} in winter (should be ~65%)")


if __name__ == "__main__":
    main()
