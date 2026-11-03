#!/usr/bin/env python3
"""Generate a synthetic visual discovery benchmark.

Creates images with controlled visual variables (color, shape, size, texture,
background) and metadata variables (category, region, season, temperature,
elevation). Embeds known patterns across 3 tiers for ablation experiments.

Usage:
    python scripts/generate_visual_synthetic.py --output data/visual_synthetic --n-samples 5000
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


COLORS = {
    "red": (220, 50, 50),
    "blue": (50, 80, 220),
    "green": (50, 180, 50),
    "yellow": (230, 200, 30),
}

BACKGROUNDS = {
    "sky": (135, 206, 235),
    "grass": (80, 160, 60),
    "water": (30, 100, 180),
    "sand": (210, 180, 130),
}

SHAPES = ["circle", "square", "triangle", "star"]
SIZES = ["small", "medium", "large"]
TEXTURES = ["smooth", "striped", "spotted", "checkered"]
CATEGORIES = ["A", "B", "C", "D"]
REGIONS = ["north", "south", "east", "west"]
SEASONS = ["spring", "summer", "autumn", "winter"]


def draw_shape(draw, shape, x, y, size_px, color, texture):
    """Draw a shape with optional texture pattern."""
    r = size_px // 2

    if shape == "circle":
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    elif shape == "square":
        draw.rectangle([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    elif shape == "triangle":
        points = [(x, y - r), (x - r, y + r), (x + r, y + r)]
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    elif shape == "star":
        points = []
        for i in range(10):
            angle = i * 36 - 90
            rad = r if i % 2 == 0 else r * 0.4
            px = x + rad * np.cos(np.radians(angle))
            py = y + rad * np.sin(np.radians(angle))
            points.append((px, py))
        draw.polygon(points, fill=color, outline=(0, 0, 0))

    if texture == "striped":
        for dy in range(-r, r, 6):
            draw.line([(x - r, y + dy), (x + r, y + dy)], fill=(0, 0, 0), width=1)
    elif texture == "spotted":
        rng = random.Random(x * 1000 + y)
        for _ in range(5):
            dx, dy = rng.randint(-r + 3, r - 3), rng.randint(-r + 3, r - 3)
            draw.ellipse(
                [x + dx - 2, y + dy - 2, x + dx + 2, y + dy + 2], fill=(0, 0, 0)
            )
    elif texture == "checkered":
        step = max(4, r // 3)
        for cx in range(x - r, x + r, step * 2):
            for cy in range(y - r, y + r, step * 2):
                draw.rectangle([cx, cy, cx + step, cy + step], fill=(0, 0, 0))


def generate_image(row, img_size=128):
    """Generate a single image from row parameters."""
    bg_color = BACKGROUNDS[row["background"]]
    img = Image.new("RGB", (img_size, img_size), bg_color)
    draw = ImageDraw.Draw(img)

    obj_color = COLORS[row["color_hue"]]
    size_map = {"small": 15, "medium": 25, "large": 40}
    size_px = size_map[row["size"]]

    count = row["count"]
    positions = []
    rng = random.Random(row.get("seed", 42))
    for _ in range(count):
        px = rng.randint(size_px + 5, img_size - size_px - 5)
        py = rng.randint(size_px + 5, img_size - size_px - 5)
        positions.append((px, py))

    for px, py in positions:
        draw_shape(draw, row["shape"], px, py, size_px, obj_color, row["texture"])

    return img


def generate_dataset(n_samples, seed=42):
    """Generate dataset with embedded patterns."""
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = []
    for i in range(n_samples):
        season = rng.choice(SEASONS)
        region = rng.choice(REGIONS)
        category = rng.choice(CATEGORIES)
        elevation = float(rng.uniform(0, 3000))
        temperature = float(40 - elevation / 100 + rng.normal(0, 3))

        color_hue = rng.choice(list(COLORS.keys()))
        shape = rng.choice(SHAPES)
        size = rng.choice(SIZES)
        texture = rng.choice(TEXTURES)
        background = rng.choice(list(BACKGROUNDS.keys()))
        count = int(rng.integers(1, 4))

        # --- Embed ground truth patterns ---

        # P3: red objects more common in autumn
        if season == "autumn":
            if rng.random() < 0.6:
                color_hue = "red"

        # P4: large objects on grass, small on water
        if background == "grass":
            if rng.random() < 0.5:
                size = "large"
        elif background == "water":
            if rng.random() < 0.5:
                size = "small"

        # P5: circles and squares never striped
        if shape in ("circle", "square") and texture == "striped":
            texture = rng.choice(["smooth", "spotted", "checkered"])

        # P6: blue objects at high elevation
        if color_hue == "blue":
            elevation = float(max(elevation, rng.uniform(1500, 3000)))
            temperature = float(40 - elevation / 100 + rng.normal(0, 3))

        # P7: north region -> larger objects
        if region == "north":
            if rng.random() < 0.55:
                size = "large"
        elif region == "south":
            if rng.random() < 0.55:
                size = "small"

        # P8: spotted + category C -> low temperature
        if texture == "spotted" and category == "C":
            temperature = float(rng.uniform(-5, 10))

        rows.append(
            {
                "id": i,
                "color_hue": color_hue,
                "shape": shape,
                "size": size,
                "texture": texture,
                "background": background,
                "count": count,
                "category": category,
                "region": region,
                "season": season,
                "temperature": round(temperature, 1),
                "elevation": round(elevation, 1),
                "seed": i,
            }
        )

    return pd.DataFrame(rows)


def generate_ground_truth():
    """Return ground truth patterns."""
    return {
        "patterns": [
            {
                "id": "P1_temp_elev",
                "tier": "prior",
                "description": "Temperature decreases with elevation (r ~ -0.7)",
                "type": "correlation",
                "variables": ["temperature", "elevation"],
                "requires_vision": False,
            },
            {
                "id": "P2_cat_temp",
                "tier": "obvious",
                "description": "Category A has higher mean temperature than Category B",
                "type": "group_difference",
                "variables": ["category", "temperature"],
                "requires_vision": False,
            },
            {
                "id": "P3_red_autumn",
                "tier": "novel",
                "description": "Red-colored objects are disproportionately common in autumn (60% vs 25% baseline)",
                "type": "association",
                "variables": ["color_hue", "season"],
                "requires_vision": True,
            },
            {
                "id": "P4_size_background",
                "tier": "novel",
                "description": "Large objects appear more on grass backgrounds; small objects more on water backgrounds",
                "type": "association",
                "variables": ["size", "background"],
                "requires_vision": True,
            },
            {
                "id": "P5_shape_texture_exclusion",
                "tier": "novel",
                "description": "Circles and squares are never observed with striped texture (exclusion rule)",
                "type": "exclusion",
                "variables": ["shape", "texture"],
                "requires_vision": True,
            },
            {
                "id": "P6_blue_elevation",
                "tier": "novel",
                "description": "Blue objects are found at significantly higher elevations than other colors",
                "type": "cross_modal",
                "variables": ["color_hue", "elevation"],
                "requires_vision": True,
            },
            {
                "id": "P7_region_size",
                "tier": "novel",
                "description": "Objects in the north region are larger than those in the south region",
                "type": "cross_modal",
                "variables": ["region", "size"],
                "requires_vision": True,
            },
            {
                "id": "P8_spotted_catC_cold",
                "tier": "novel",
                "description": "Spotted-texture objects in category C have anomalously low temperature (-5 to 10)",
                "type": "cross_modal_interaction",
                "variables": ["texture", "category", "temperature"],
                "requires_vision": True,
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic visual benchmark")
    parser.add_argument("--output", default="data/visual_synthetic")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "images"), exist_ok=True)

    print(f"Generating {args.n_samples} samples...")
    df = generate_dataset(args.n_samples, seed=args.seed)

    print("Generating images...")
    local_paths = []
    for _, row in df.iterrows():
        img = generate_image(row, img_size=args.img_size)
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

    # Save full data (with visual columns)
    df.to_parquet(os.path.join(args.output, "data_full.parquet"), index=False)

    # Save metadata-only (without visual columns)
    meta_cols = [
        "id",
        "category",
        "region",
        "season",
        "temperature",
        "elevation",
        "observation_id",
        "species_name",
        "local_path",
    ]
    df[meta_cols].to_parquet(
        os.path.join(args.output, "data_no_visual.parquet"), index=False
    )

    # Save standard data.parquet (full, for the adapter)
    df.to_parquet(os.path.join(args.output, "data.parquet"), index=False)

    gt = generate_ground_truth()
    with open(os.path.join(args.output, "ground_truth.json"), "w") as f:
        json.dump(gt, f, indent=2)

    # Prior insights
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
        "name": "visual_synthetic",
        "n_samples": args.n_samples,
        "seed": args.seed,
        "n_patterns": len(gt["patterns"]),
        "tiers": {"prior": 1, "obvious": 1, "novel": 6},
    }
    with open(os.path.join(args.output, "benchmark_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nGenerated:")
    print(f"  {args.n_samples} samples with images")
    print(
        f"  {len(gt['patterns'])} ground truth patterns (1 prior, 1 obvious, 6 novel)"
    )
    print(f"  data_full.parquet (all columns)")
    print(f"  data_no_visual.parquet (metadata only, for vision-only ablation)")
    print(f"  data.parquet (standard, for adapter)")
    print(f"  Output: {args.output}/")

    # Verify patterns
    print("\nPattern verification:")
    autumn = df[df["season"] == "autumn"]
    print(
        f"  P3: Red in autumn: {(autumn['color_hue'] == 'red').mean():.1%} vs baseline {(df['color_hue'] == 'red').mean():.1%}"
    )
    grass = df[df["background"] == "grass"]
    water = df[df["background"] == "water"]
    print(
        f"  P4: Large on grass: {(grass['size'] == 'large').mean():.1%} vs water: {(water['size'] == 'large').mean():.1%}"
    )
    circ_sq = df[df["shape"].isin(["circle", "square"])]
    print(
        f"  P5: Striped circles/squares: {(circ_sq['texture'] == 'striped').sum()} (should be 0)"
    )
    blue = df[df["color_hue"] == "blue"]
    nonblue = df[df["color_hue"] != "blue"]
    print(
        f"  P6: Blue elevation: {blue['elevation'].mean():.0f} vs others: {nonblue['elevation'].mean():.0f}"
    )
    north = df[df["region"] == "north"]
    south = df[df["region"] == "south"]
    print(
        f"  P7: Large in north: {(north['size'] == 'large').mean():.1%} vs south: {(south['size'] == 'large').mean():.1%}"
    )
    spotted_c = df[(df["texture"] == "spotted") & (df["category"] == "C")]
    print(
        f"  P8: Spotted+C temp: {spotted_c['temperature'].mean():.1f} vs all: {df['temperature'].mean():.1f}"
    )


if __name__ == "__main__":
    main()
