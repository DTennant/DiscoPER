#!/usr/bin/env python3
"""
Generate visual ecology benchmark datasets with known ground truth.

Pipeline:
1. Load causal DAG preset (easy/medium/hard)
2. Simulate tabular data from DAG
3. Render ecology scene for each data point
4. Optionally apply diffusion model style transfer
5. Extract visual features from each image
6. Save: images/, extracted_features.parquet, tabular_data.parquet,
         ground_truth.json, prior_insights.parquet, benchmark_meta.json,
         feature_variable_mapping.json

Usage:
    # Programmatic rendering (fast, free, deterministic — default)
    python scripts/generate_visual_benchmark.py --preset easy --output data/visual_benchmarks/easy

    # OpenAI rendering (gpt-image-1.5, photorealistic, costs API credits)
    python scripts/generate_visual_benchmark.py --preset easy --output data/visual_benchmarks/easy_openai --renderer openai

    # Programmatic + diffusion style transfer (needs local GPU)
    python scripts/generate_visual_benchmark.py --preset medium --output data/visual_benchmarks/medium --stylize

    # Small fast test run
    python scripts/generate_visual_benchmark.py --preset easy --n-samples 20 --image-size 128
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from PIL import Image

from core.benchmark.presets import get_preset
from core.benchmark.simulator import add_observation_columns, simulate_world
from core.benchmark.ground_truth_extractor import extract_ground_truth
from core.benchmark.visual.visual_mapping import (
    DEFAULT_ECOLOGY_MAPPING,
    get_feature_variable_mapping,
    map_values_to_visual,
)
from core.benchmark.visual.scene_renderer import EcologySceneRenderer
from core.benchmark.visual.feature_extractor import VisualFeatureExtractor


def create_prior_insights(patterns: list[dict]) -> list[dict]:
    """Create prior insight records from prior_knowledge patterns."""
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


def generate_visual_benchmark(
    preset_name: str,
    output_dir: str,
    seed: int | None = None,
    n_samples: int | None = None,
    image_size: int = 1024,
    renderer: str = "programmatic",
    stylize: bool = False,
    stylize_strength: float = 0.4,
    openai_model: str = "gpt-image-1.5",
    openai_quality: str = "medium",
    verbose: bool = True,
) -> None:
    """
    Generate a complete visual ecology benchmark dataset.

    Args:
        preset_name: Preset name ("easy", "medium", "hard")
        output_dir: Output directory path
        seed: Override seed (optional)
        n_samples: Override sample count (optional)
        image_size: Image resolution (square)
        renderer: "programmatic" (PIL, fast/free), "openai" (DALL-E, photorealistic)
        stylize: Whether to apply diffusion model style transfer (programmatic only)
        stylize_strength: Strength of style transfer (0.0-1.0)
        openai_model: OpenAI model name for image generation
        openai_quality: OpenAI image quality ("low", "medium", "high")
        verbose: Print progress
    """
    t_start = time.time()

    # Load preset
    ruleset = get_preset(preset_name)

    # Apply overrides
    if seed is not None:
        ruleset.seed = seed
    if n_samples is not None:
        ruleset.n_samples = n_samples

    dag = ruleset.dag
    if verbose:
        print(f"=== Visual Benchmark Generation ===")
        print(f"  Preset: {preset_name}")
        print(f"  Renderer: {renderer}")
        print(f"  Samples: {ruleset.n_samples}, Seed: {ruleset.seed}")
        print(f"  Variables: {len(dag.variables)}, Rules: {len(dag.rules)}")
        print(f"  Image size: {image_size}x{image_size}")
        if renderer == "openai":
            print(f"  OpenAI model: {openai_model}, quality: {openai_quality}")
        if stylize and renderer == "programmatic":
            print(f"  Diffusion stylization: ON (strength={stylize_strength})")

    # =========================================================================
    # Step 1: Simulate tabular data from causal DAG
    # =========================================================================
    if verbose:
        print("\n[1/6] Simulating causal DAG...")
    df, world_state = simulate_world(ruleset)

    rng_obs = np.random.default_rng(ruleset.seed + 1)
    df = add_observation_columns(df, len(df), rng_obs)

    if verbose:
        print(f"  Generated DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")

    # =========================================================================
    # Step 2: Extract ground truth
    # =========================================================================
    if verbose:
        print("\n[2/6] Extracting ground truth patterns...")
    ground_truth = extract_ground_truth(ruleset, df, world_state)

    patterns = ground_truth["patterns"]
    prior_patterns = [p for p in patterns if p.get("tier") == "prior_knowledge"]
    obvious_patterns = [p for p in patterns if p.get("tier") == "obvious"]
    novel_patterns = [p for p in patterns if p.get("tier") == "novel"]

    if verbose:
        print(f"  Prior knowledge: {len(prior_patterns)}")
        print(f"  Obvious:         {len(obvious_patterns)}")
        print(f"  Novel:           {len(novel_patterns)}")

    # =========================================================================
    # Step 3: Render ecology scenes
    # =========================================================================
    if verbose:
        print(f"\n[3/6] Rendering {ruleset.n_samples} ecology scenes...")

    mapping_spec = DEFAULT_ECOLOGY_MAPPING
    mapping_spec.image_size = (image_size, image_size)

    openai_gen = None
    pil_renderer = None
    if renderer == "openai":
        from core.benchmark.visual.openai_renderer import OpenAIImageRenderer

        if not OpenAIImageRenderer.is_available():
            print("  ERROR: OpenAI renderer requires openai package + API key.")
            print("  Set OPENAI_API_KEY env var.")
            sys.exit(1)
        openai_gen = OpenAIImageRenderer(
            model=openai_model, size="1024x1024", quality=openai_quality
        )
    else:
        pil_renderer = EcologySceneRenderer(image_size=(image_size, image_size))

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    images = []
    available_vars = set(df.columns)

    for i in range(len(df)):
        row = df.iloc[i]
        variable_values = {}
        for mapping in mapping_spec.mappings:
            if mapping.variable_name in available_vars:
                variable_values[mapping.variable_name] = float(
                    row[mapping.variable_name]
                )

        sample_rng = np.random.default_rng(ruleset.seed + 200 + i)
        visual_props = map_values_to_visual(mapping_spec, variable_values, sample_rng)

        if renderer == "openai":
            img = openai_gen.render(visual_props, seed=ruleset.seed + 300 + i)  # type: ignore[union-attr]
            if img is None:
                if verbose:
                    print(
                        f"  WARNING: OpenAI generation failed for sample {i}, using programmatic fallback"
                    )
                fallback = EcologySceneRenderer(image_size=(image_size, image_size))
                img = fallback.render(visual_props, seed=ruleset.seed + 300 + i)
            if img.size != (image_size, image_size):
                img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
        else:
            img = pil_renderer.render(visual_props, seed=ruleset.seed + 300 + i)  # type: ignore[union-attr]

        images.append(img)

        img_path = os.path.join(images_dir, f"sample_{i:05d}.png")
        img.save(img_path)

        if verbose and (i + 1) % max(1, len(df) // 10) == 0:
            pct = (i + 1) / len(df) * 100
            print(f"  Rendered {i + 1}/{len(df)} images ({pct:.0f}%)")

    if verbose:
        print(f"  Saved {len(images)} images to {images_dir}")

    # =========================================================================
    # Step 4: Optional diffusion model stylization
    # =========================================================================
    if stylize and renderer == "programmatic":
        if verbose:
            print("\n[4/6] Applying diffusion model style transfer...")

        from core.benchmark.visual.diffusion_stylizer import DiffusionStylizer

        if DiffusionStylizer.is_available():
            stylizer = DiffusionStylizer(strength=stylize_strength, seed=ruleset.seed)
            for i, img in enumerate(images):
                stylized = stylizer.stylize(img, seed=ruleset.seed + 400 + i)
                images[i] = stylized

                # Overwrite image
                img_path = os.path.join(images_dir, f"sample_{i:05d}.png")
                stylized.save(img_path)

                if verbose and (i + 1) % max(1, len(images) // 10) == 0:
                    pct = (i + 1) / len(images) * 100
                    print(f"  Stylized {i + 1}/{len(images)} images ({pct:.0f}%)")
        else:
            if verbose:
                print("  WARNING: torch/diffusers not available — skipping stylization")
    elif renderer == "openai":
        if verbose:
            print(
                "\n[4/6] Skipping stylization (OpenAI images are already photorealistic)"
            )
    else:
        if verbose:
            print("\n[4/6] Skipping diffusion stylization (not requested)")

    # =========================================================================
    # Step 5: Extract visual features
    # =========================================================================
    if verbose:
        print(f"\n[5/6] Extracting visual features from {len(images)} images...")

    extractor = VisualFeatureExtractor(feature_set="classical")

    feature_records = []
    for i, img in enumerate(images):
        features = extractor.extract(img)
        features["sample_id"] = i
        feature_records.append(features)

        if verbose and (i + 1) % max(1, len(images) // 10) == 0:
            pct = (i + 1) / len(images) * 100
            print(
                f"  Extracted features from {i + 1}/{len(images)} images ({pct:.0f}%)"
            )

    features_df = pd.DataFrame(feature_records)

    if verbose:
        print(f"  Extracted {len(features_df.columns) - 1} features per image")

    # =========================================================================
    # Step 6: Save all outputs
    # =========================================================================
    if verbose:
        print(f"\n[6/6] Saving outputs to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Save extracted features (what discovery system sees)
    features_path = os.path.join(output_dir, "extracted_features.parquet")
    features_df.to_parquet(features_path, index=False)
    if verbose:
        print(f"  Saved extracted features to {features_path}")

    # Save tabular data (ground truth — hidden from discovery)
    tabular_path = os.path.join(output_dir, "tabular_data.parquet")
    df.to_parquet(tabular_path, index=False)
    if verbose:
        print(f"  Saved tabular ground truth to {tabular_path}")

    # Save ground truth patterns
    gt_path = os.path.join(output_dir, "ground_truth.json")
    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)
    if verbose:
        print(f"  Saved ground truth to {gt_path}")

    # Save prior insights
    if prior_patterns:
        prior_insights = create_prior_insights(prior_patterns)
        prior_df = pd.DataFrame(prior_insights)
        prior_path = os.path.join(output_dir, "prior_insights.parquet")
        prior_df.to_parquet(prior_path, index=False)
        if verbose:
            print(f"  Saved {len(prior_insights)} prior insights to {prior_path}")

    # Save feature-to-variable mapping (for evaluation only)
    fv_mapping = get_feature_variable_mapping()
    mapping_path = os.path.join(output_dir, "feature_variable_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(fv_mapping, f, indent=2)
    if verbose:
        print(f"  Saved feature-variable mapping to {mapping_path}")

    # Save benchmark metadata
    elapsed = time.time() - t_start
    meta = {
        "name": f"visual_{ruleset.name}",
        "generation_method": "causal_dag_visual",
        "renderer": renderer,
        "preset": preset_name,
        "seed": ruleset.seed,
        "n_samples": ruleset.n_samples,
        "n_variables": len(dag.variables),
        "n_rules": len(dag.rules),
        "image_size": image_size,
        "openai_model": openai_model if renderer == "openai" else None,
        "openai_quality": openai_quality if renderer == "openai" else None,
        "stylized": stylize and renderer == "programmatic",
        "stylize_strength": stylize_strength
        if (stylize and renderer == "programmatic")
        else None,
        "n_prior_patterns": len(prior_patterns),
        "n_obvious_patterns": len(obvious_patterns),
        "n_novel_patterns": len(novel_patterns),
        "n_extracted_features": len(features_df.columns) - 1,
        "feature_names": [c for c in features_df.columns if c != "sample_id"],
        "global_noise_scale": ruleset.global_noise_scale,
        "correlation_threshold": ruleset.correlation_threshold,
        "generation_time_seconds": round(elapsed, 1),
    }
    meta_path = os.path.join(output_dir, "benchmark_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"  Saved metadata to {meta_path}")

    if verbose:
        print(f"\n=== Visual benchmark generation complete! ===")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Output directory: {output_dir}")
        print(f"  Images: {images_dir}/ ({len(images)} files)")
        print(f"  Features: {features_path} ({len(features_df.columns) - 1} columns)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual ecology benchmark datasets with known ground truth"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["easy", "medium", "hard"],
        required=True,
        help="Preset difficulty level",
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
        "--image-size",
        type=int,
        default=1024,
        help="Image resolution (square, default: 1024)",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        choices=["programmatic", "openai"],
        default="programmatic",
        help="Rendering backend: 'programmatic' (PIL, fast/free) or 'openai' (gpt-image-1.5, photorealistic)",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-image-1.5",
        help="OpenAI image model (default: gpt-image-1.5)",
    )
    parser.add_argument(
        "--openai-quality",
        type=str,
        default="medium",
        help="OpenAI image quality: low, medium, high (default: medium)",
    )
    parser.add_argument(
        "--stylize",
        action="store_true",
        help="Apply diffusion model style transfer (requires torch + diffusers)",
    )
    parser.add_argument(
        "--stylize-strength",
        type=float,
        default=0.4,
        help="Diffusion stylization strength (0.0-1.0, default: 0.4)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    args = parser.parse_args()

    generate_visual_benchmark(
        preset_name=args.preset,
        output_dir=args.output,
        seed=args.seed,
        n_samples=args.n_samples,
        image_size=args.image_size,
        renderer=args.renderer,
        stylize=args.stylize,
        stylize_strength=args.stylize_strength,
        openai_model=args.openai_model,
        openai_quality=args.openai_quality,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
