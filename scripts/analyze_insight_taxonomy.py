#!/usr/bin/env python3
"""Analyze insight taxonomy and agent discovery patterns.

Classifies each insight by pattern level, novelty, and behavioral fingerprint.
Produces a JSON report and summary statistics.

Usage:
    python scripts/analyze_insight_taxonomy.py --workspace results/inat800_v5_realfix --output results/taxonomy_report.json
"""

import argparse
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# Pattern level classification keywords
LEVEL_PATTERNS = {
    "L1_direct": [
        "correlat",
        "associat",
        "relationship between",
        "linked to",
        "related to",
    ],
    "L2_conditional": [
        "controlling for",
        "adjusting for",
        "after accounting",
        "conditional on",
        "given that",
        "when controlling",
    ],
    "L3_interaction": [
        "interact",
        "moderat",
        "depends on",
        "differs between.*when",
        "only when",
        "amplif",
    ],
    "L4_mediation": ["mediat", "through", "via", "indirect", "pathway", "chain"],
    "L5_structural": ["confound", "spurious", "collider", "simpson", "paradox", "bias"],
    "L6_aggregate": [
        "across all",
        "overall",
        "aggregate",
        "kingdom-level",
        "class-level",
        "taxonomic",
    ],
}

# Hypothesis template patterns
TEMPLATE_PATTERNS = [
    ("X_peaks_in_Y", r"(?:peak|concentrat|highest).+(?:in|during)\s+\w+"),
    ("X_higher_than_Y", r"(?:higher|greater|more|larger|wider).+(?:than|compared)"),
    ("X_correlates_Y", r"(?:correlat|associat|relationship|linked)"),
    ("X_differs_groups", r"(?:differ|distinct|vary).+(?:between|across|among)"),
    ("X_predicts_Y", r"(?:predict|explain|account for|drive)"),
    ("X_seasonal", r"(?:season|spring|summer|autumn|winter|month)"),
    ("X_geographic", r"(?:latitud|longitud|north|south|range|distribut)"),
    ("X_observer_bias", r"(?:observer|bias|sampling|effort|photograph)"),
    ("X_visual", r"(?:habitat|morpholog|color|size|vegetation|landscape|image|photo)"),
]


def classify_level(claim: str) -> str:
    """Classify insight into pattern level L1-L6."""
    claim_lower = claim.lower()
    for level, keywords in reversed(list(LEVEL_PATTERNS.items())):
        for kw in keywords:
            if kw in claim_lower:
                return level
    return "L1_direct"


def classify_template(claim: str) -> str:
    """Match claim to a hypothesis template."""
    claim_lower = claim.lower()
    for template_name, pattern in TEMPLATE_PATTERNS:
        if re.search(pattern, claim_lower):
            return template_name
    return "other"


def extract_variables(row) -> list[str]:
    """Get variables from insight."""
    v = row.get("variables")
    if isinstance(v, (list, np.ndarray)):
        return [str(x) for x in v]
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (json.JSONDecodeError, ValueError):
            return [v]
    return []


def compute_novelty(insights_df: pd.DataFrame) -> list[str]:
    """Label each insight's novelty relative to prior insights."""
    labels = []
    seen_var_pairs = set()

    for _, row in insights_df.iterrows():
        variables = extract_variables(row)
        var_pair = (
            tuple(sorted(variables[:2])) if len(variables) >= 2 else tuple(variables)
        )

        if var_pair in seen_var_pairs:
            claim = str(row.get("claim", "")).lower()
            if any(w in claim for w in ["however", "contrast", "opposite", "not"]):
                labels.append("contradiction")
            elif any(
                w in claim for w in ["broader", "general", "across all", "overall"]
            ):
                labels.append("generalization")
            elif any(
                w in claim for w in ["specifically", "particular", "subset", "only in"]
            ):
                labels.append("refinement")
            else:
                labels.append("redundant")
        else:
            labels.append("novel")

        seen_var_pairs.add(var_pair)

    return labels


def analyze_agent_behavior(insights_df: pd.DataFrame, workspace: str) -> dict:
    """Analyze agent behavioral patterns from discovery.json files."""
    behavior = {
        "variable_frequency": Counter(),
        "variable_pair_frequency": Counter(),
        "test_method_distribution": Counter(),
        "slice_patterns": Counter(),
        "template_distribution": Counter(),
        "level_distribution": Counter(),
        "per_iteration": [],
    }

    for _, row in insights_df.iterrows():
        variables = extract_variables(row)
        claim = str(row.get("claim", ""))
        method = str(row.get("method", "unknown"))

        for v in variables:
            behavior["variable_frequency"][v] += 1
        if len(variables) >= 2:
            pair = tuple(sorted(variables[:2]))
            behavior["variable_pair_frequency"][str(pair)] += 1

        template = classify_template(claim)
        behavior["template_distribution"][template] += 1

        level = classify_level(claim)
        behavior["level_distribution"][level] += 1

    # Load discovery.json files for richer behavioral data
    if workspace and os.path.isdir(workspace):
        for entry in sorted(os.listdir(workspace)):
            if not entry.startswith("iteration_"):
                continue
            fp = os.path.join(workspace, entry, "discovery.json")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                disc = json.load(f)

            exp = disc.get("experiment", {})
            method = exp.get("method", "unknown")
            behavior["test_method_distribution"][method] += 1

            slice_spec = exp.get("dataset_slice_spec", {})
            if slice_spec:
                slice_key = ", ".join(sorted(slice_spec.keys()))
                behavior["slice_patterns"][slice_key] += 1
            else:
                behavior["slice_patterns"]["no_filter"] += 1

            insight = disc.get("insight", {})
            iter_num = int(entry.replace("iteration_", "").replace("_meta", ""))
            if iter_num <= 15:
                phase = "early"
            elif iter_num <= 35:
                phase = "mid"
            else:
                phase = "late"

            behavior["per_iteration"].append(
                {
                    "iteration": entry,
                    "phase": phase,
                    "method": method,
                    "variables": extract_variables(insight)
                    if isinstance(insight, dict)
                    else [],
                    "verdict": insight.get("verdict", "unknown")
                    if isinstance(insight, dict)
                    else "unknown",
                    "is_meta": insight.get("is_meta", False)
                    if isinstance(insight, dict)
                    else False,
                    "template": classify_template(str(insight.get("claim", "")))
                    if isinstance(insight, dict)
                    else "unknown",
                    "level": classify_level(str(insight.get("claim", "")))
                    if isinstance(insight, dict)
                    else "L1_direct",
                }
            )

    # Convert Counters to dicts for JSON
    for key in [
        "variable_frequency",
        "variable_pair_frequency",
        "test_method_distribution",
        "slice_patterns",
        "template_distribution",
        "level_distribution",
    ]:
        behavior[key] = dict(behavior[key].most_common(20))

    return behavior


def compute_metrics(insights_df: pd.DataFrame, novelty_labels: list[str]) -> dict:
    """Compute discovery pattern metrics."""
    n = len(insights_df)
    if n == 0:
        return {}

    supported = insights_df[insights_df["verdict"] == "supported"]

    variables_seen = set()
    for _, row in insights_df.iterrows():
        for v in extract_variables(row):
            variables_seen.add(v)

    all_vars = set()
    for _, row in insights_df.iterrows():
        for v in extract_variables(row):
            all_vars.add(v)

    var_pairs = set()
    for _, row in insights_df.iterrows():
        vs = extract_variables(row)
        if len(vs) >= 2:
            var_pairs.add(tuple(sorted(vs[:2])))

    novelty_counts = Counter(novelty_labels)

    return {
        "total_insights": n,
        "supported": len(supported),
        "support_rate": round(len(supported) / n, 3) if n > 0 else 0,
        "exploration_breadth": len(var_pairs),
        "variable_coverage": len(variables_seen),
        "novelty_rate": round(novelty_counts.get("novel", 0) / n, 3),
        "redundancy_rate": round(novelty_counts.get("redundant", 0) / n, 3),
        "novelty_distribution": dict(novelty_counts),
        "unique_templates": len(
            set(
                classify_template(str(row.get("claim", "")))
                for _, row in insights_df.iterrows()
            )
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", required=True, help="Workspace dir with insights.parquet"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    parquet_path = os.path.join(args.workspace, "insights.parquet")
    if not os.path.exists(parquet_path):
        print(f"Error: {parquet_path} not found")
        return

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} insights from {args.workspace}")

    # Classify each insight
    df["level"] = df["claim"].apply(classify_level)
    df["template"] = df["claim"].apply(classify_template)
    novelty_labels = compute_novelty(df)
    df["novelty"] = novelty_labels

    # Compute metrics
    metrics = compute_metrics(df, novelty_labels)
    print(f"\n=== Discovery Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Analyze agent behavior
    behavior = analyze_agent_behavior(df, args.workspace)
    print(f"\n=== Agent Behavior ===")
    print(f"  Top variables: {behavior['variable_frequency']}")
    print(f"  Test methods: {behavior['test_method_distribution']}")
    print(f"  Templates: {behavior['template_distribution']}")
    print(f"  Levels: {behavior['level_distribution']}")
    print(f"  Slice patterns: {behavior['slice_patterns']}")

    # Phase analysis
    if behavior["per_iteration"]:
        for phase in ["early", "mid", "late"]:
            phase_items = [p for p in behavior["per_iteration"] if p["phase"] == phase]
            if phase_items:
                methods = Counter(p["method"] for p in phase_items)
                verdicts = Counter(p["verdict"] for p in phase_items)
                templates = Counter(p["template"] for p in phase_items)
                print(f"\n  Phase '{phase}' ({len(phase_items)} iterations):")
                print(f"    Methods: {dict(methods.most_common(3))}")
                print(f"    Verdicts: {dict(verdicts)}")
                print(f"    Templates: {dict(templates.most_common(3))}")

    # Save report
    report = {
        "workspace": args.workspace,
        "metrics": metrics,
        "behavior": behavior,
        "per_insight": [
            {
                "claim": str(row.get("claim", ""))[:200],
                "verdict": row.get("verdict", ""),
                "level": row["level"],
                "template": row["template"],
                "novelty": row["novelty"],
                "variables": extract_variables(row),
                "is_meta": bool(row.get("is_meta", False)),
            }
            for _, row in df.iterrows()
        ],
    }

    output = args.output or os.path.join(args.workspace, "taxonomy_report.json")
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
