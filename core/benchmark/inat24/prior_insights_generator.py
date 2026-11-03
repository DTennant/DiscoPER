"""Generate prior_insights.parquet from inat24_ground_truth.json prior patterns."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from core.discovery.types import Insight


def generate_prior_insights(
    ground_truth_path: str,
    output_path: str,
) -> list[Insight]:
    """Read ground truth JSON, extract prior-tier patterns, and save as parquet.

    Each prior pattern becomes an Insight with verdict="supported".

    Args:
        ground_truth_path: Path to inat24_ground_truth.json.
        output_path: Where to write prior_insights.parquet.

    Returns:
        List of generated Insight objects.
    """
    with open(ground_truth_path, "r") as f:
        ground_truth: dict[str, Any] = json.load(f)

    insights: list[Insight] = []
    for pattern in ground_truth.get("patterns", []):
        tier = pattern.get("tier", "novel")
        if tier not in ("prior_knowledge", "prior"):
            continue

        insight = Insight.create(
            hypothesis_id=f"prior_{pattern['id']}",
            experiment_id=f"prior_exp_{pattern['id']}",
            verdict="supported",
            claim=pattern.get("description", f"Prior knowledge: {pattern['id']}"),
            strength=0.9,
            novelty_proxy=0.0,
            variables=pattern.get("variables", []),
        )
        insights.append(insight)

    if insights:
        records = [ins.to_dict() for ins in insights]
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path, index=False)

    return insights


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate prior_insights.parquet from ground truth"
    )
    parser.add_argument(
        "--ground-truth",
        default="data/inquire_benchmarks/inat24_ground_truth.json",
    )
    parser.add_argument(
        "--output",
        default="data/inquire_benchmarks/prior_insights.parquet",
    )
    args = parser.parse_args()

    results = generate_prior_insights(args.ground_truth, args.output)
    print(f"Generated {len(results)} prior insights -> {args.output}")
    for ins in results:
        print(f"  - [{ins.id[:8]}] {ins.claim}")
