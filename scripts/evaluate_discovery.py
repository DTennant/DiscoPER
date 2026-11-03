#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate discovery agent results against ground truth patterns.

This script computes:
1. Precision/Recall/F1 for novel discoveries
2. Rediscovery rate (penalized for finding prior/obvious patterns)
3. Overall benchmark score

Usage:
    python scripts/evaluate_discovery.py \
        --ground_truth data/synthetic_benchmarks/ecology_medium/ground_truth.json \
        --discovered results/discovery_run/insights.parquet \
        --output results/discovery_run/evaluation.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd


@dataclass
class MatchResult:
    """Result of matching an insight to a ground truth pattern."""

    insight_id: str
    pattern_id: Optional[str]
    pattern_tier: Optional[str]
    pattern_difficulty: Optional[str]
    match_score: float
    variable_overlap: float
    matched: bool


@dataclass
class EvaluationResult:
    """Complete evaluation results."""

    # Pattern-level metrics
    n_prior_patterns: int = 0
    n_obvious_patterns: int = 0
    n_novel_patterns: int = 0

    # Discovery counts
    n_insights: int = 0
    n_matched_prior: int = 0
    n_matched_obvious: int = 0
    n_matched_novel_simple: int = 0
    n_matched_novel_moderate: int = 0
    n_matched_novel_complex: int = 0
    n_unmatched: int = 0

    # Core metrics
    precision_novel: float = 0.0
    recall_novel: float = 0.0
    f1_novel: float = 0.0

    # Breakdown by difficulty
    recall_novel_simple: float = 0.0
    recall_novel_moderate: float = 0.0
    recall_novel_complex: float = 0.0

    # Rediscovery metrics
    rediscovery_rate: float = 0.0
    prior_rediscovery_count: int = 0
    obvious_rediscovery_count: int = 0

    # Overall score
    benchmark_score: float = 0.0

    # Detailed matches
    matches: list[dict] = field(default_factory=list)
    diversity: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_counts": {
                "prior": self.n_prior_patterns,
                "obvious": self.n_obvious_patterns,
                "novel": self.n_novel_patterns,
            },
            "discovery_counts": {
                "total_insights": self.n_insights,
                "matched_prior": self.n_matched_prior,
                "matched_obvious": self.n_matched_obvious,
                "matched_novel_simple": self.n_matched_novel_simple,
                "matched_novel_moderate": self.n_matched_novel_moderate,
                "matched_novel_complex": self.n_matched_novel_complex,
                "unmatched": self.n_unmatched,
            },
            "metrics": {
                "precision_novel": self.precision_novel,
                "recall_novel": self.recall_novel,
                "f1_novel": self.f1_novel,
                "recall_novel_simple": self.recall_novel_simple,
                "recall_novel_moderate": self.recall_novel_moderate,
                "recall_novel_complex": self.recall_novel_complex,
                "rediscovery_rate": self.rediscovery_rate,
            },
            "benchmark_score": self.benchmark_score,
            "matches": self.matches,
            "diversity": self.diversity,
        }


def extract_variables_from_pattern(pattern: dict) -> set[str]:
    """Extract all variable names from a pattern specification."""
    variables = set()

    # Direct variables field
    if "variables" in pattern:
        variables.update(pattern["variables"])

    # Chain (for mediation)
    if "chain" in pattern:
        variables.update(pattern["chain"])

    # Observed vars (for latent)
    if "observed_vars" in pattern:
        variables.update(pattern["observed_vars"])

    # Predictors and outcome
    if "predictors" in pattern:
        variables.update(pattern["predictors"])
    if "outcome" in pattern:
        variables.add(pattern["outcome"])

    # Confound variables
    if "confound" in pattern:
        variables.add(pattern["confound"])
    if "confounder" in pattern:
        variables.add(pattern["confounder"])

    # Apparent correlation
    if "apparent_correlation" in pattern:
        variables.update(pattern["apparent_correlation"])

    # Group by
    if "group_by" in pattern:
        variables.add(pattern["group_by"])

    # Competing species
    if "competing_species" in pattern:
        variables.update(pattern["competing_species"])
    if "keystone" in pattern:
        variables.add(pattern["keystone"])

    # Hierarchy column
    if "hierarchy_col" in pattern:
        variables.add(pattern["hierarchy_col"])

    # Time column
    if "time_col" in pattern:
        variables.add(pattern["time_col"])

    # Region column
    if "region_col" in pattern:
        variables.add(pattern["region_col"])

    return variables


def extract_variables_from_claim(
    claim: str,
    all_variables: set[str],
    feature_to_variable: Optional[dict[str, str]] = None,
) -> set[str]:
    """Extract variable names from an insight claim by matching against known variables.

    When feature_to_variable is provided (visual benchmarks), also searches for
    extracted feature names (e.g. 'green_fraction') and maps them to their
    corresponding causal variable names (e.g. 'vegetation_density').
    """
    found = set()
    claim_lower = claim.lower()

    # Match against ground truth causal variable names
    for var in all_variables:
        # Try various forms of the variable name
        var_forms = [
            var.lower(),
            var.lower().replace("_", " "),
            var.lower().replace("_", ""),
        ]
        for form in var_forms:
            if form in claim_lower:
                found.add(var)
                break

    # Also match against feature names and translate to causal variables
    if feature_to_variable:
        for feature_name, causal_var in feature_to_variable.items():
            feature_forms = [
                feature_name.lower(),
                feature_name.lower().replace("_", " "),
                feature_name.lower().replace("_", ""),
            ]
            for form in feature_forms:
                if form in claim_lower:
                    found.add(causal_var)
                    break

    return found


def compute_variable_overlap(insight_vars: set[str], pattern_vars: set[str]) -> float:
    """Compute Jaccard-like overlap between variable sets."""
    if not pattern_vars:
        return 0.0

    intersection = len(insight_vars & pattern_vars)
    union = len(insight_vars | pattern_vars)

    if union == 0:
        return 0.0

    # Use a weighted formula: intersection / pattern_vars gives recall of pattern variables
    # This ensures we find patterns even if insight mentions extra variables
    recall = intersection / len(pattern_vars)
    precision = intersection / len(insight_vars) if insight_vars else 0.0

    # F1-like combination
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def match_insight_to_patterns(
    insight: dict,
    patterns: list[dict],
    all_variables: set[str],
    variable_threshold: float = 0.3,
    check_direction: bool = True,
    feature_to_variable: Optional[dict[str, str]] = None,
) -> Optional[MatchResult]:
    """
    Match an insight to the best matching ground truth pattern.

    Args:
        insight: Insight dict with 'claim', 'verdict', etc.
        patterns: List of ground truth pattern dicts
        all_variables: Set of all known variable names
        variable_threshold: Minimum overlap score to consider a match
        check_direction: Whether to check effect direction
        feature_to_variable: Optional mapping from feature names to causal variable names

    Returns:
        MatchResult if matched, None otherwise
    """
    claim = insight.get("claim", "")
    insight_vars = extract_variables_from_claim(
        claim, all_variables, feature_to_variable
    )

    best_match = None
    best_score = 0.0

    for pattern in patterns:
        pattern_vars = extract_variables_from_pattern(pattern)
        overlap = compute_variable_overlap(insight_vars, pattern_vars)

        if overlap < variable_threshold:
            continue

        # Additional scoring based on pattern type and claim content
        score = overlap

        # Boost score if description keywords match
        description = pattern.get("description", "").lower()
        claim_lower = claim.lower()

        # Check for key description words
        desc_words = set(description.split())
        claim_words = set(claim_lower.split())
        word_overlap = len(desc_words & claim_words) / max(len(desc_words), 1)
        score = 0.7 * overlap + 0.3 * word_overlap

        # Direction check (simplified)
        if check_direction:
            effect = pattern.get("effect_size", pattern.get("effect", None))
            if effect is not None:
                if (
                    effect > 0
                    and "negative" in claim_lower
                    and "positive" not in claim_lower
                ):
                    score *= 0.5  # Penalty for wrong direction
                elif (
                    effect < 0
                    and "positive" in claim_lower
                    and "negative" not in claim_lower
                ):
                    score *= 0.5

        if score > best_score:
            best_score = score
            best_match = pattern

    if best_match is None:
        return None

    return MatchResult(
        insight_id=insight.get("id", ""),
        pattern_id=best_match.get("id"),
        pattern_tier=best_match.get("tier"),
        pattern_difficulty=best_match.get("difficulty"),
        match_score=best_score,
        variable_overlap=compute_variable_overlap(
            insight_vars, extract_variables_from_pattern(best_match)
        ),
        matched=True,
    )


def evaluate_discovery(
    ground_truth: dict,
    insights: list[dict],
    variable_threshold: float = 0.3,
    scoring: Optional[dict] = None,
    feature_to_variable: Optional[dict[str, str]] = None,
) -> EvaluationResult:
    """
    Evaluate discovered insights against ground truth patterns.

    Args:
        ground_truth: Ground truth dict with 'patterns' list
        insights: List of discovered insight dicts
        variable_threshold: Minimum overlap for matching
        scoring: Scoring weights (uses defaults if None)
        feature_to_variable: Optional mapping from feature names to causal variable names

    Returns:
        EvaluationResult with all metrics
    """
    result = EvaluationResult()

    # Get scoring weights
    if scoring is None:
        scoring = ground_truth.get("evaluation_criteria", {}).get("scoring", {})

    default_scoring = {
        "prior_rediscovery": -1.0,
        "obvious_rediscovery": -0.5,
        "novel_simple_found": 0.5,
        "novel_moderate_found": 1.0,
        "novel_complex_found": 2.0,
    }
    scoring = {**default_scoring, **scoring}

    # Organize patterns by tier
    patterns = ground_truth.get("patterns", [])
    prior_patterns = [p for p in patterns if p.get("tier") == "prior_knowledge"]
    obvious_patterns = [p for p in patterns if p.get("tier") == "obvious"]
    novel_patterns = [p for p in patterns if p.get("tier") == "novel"]

    result.n_prior_patterns = len(prior_patterns)
    result.n_obvious_patterns = len(obvious_patterns)
    result.n_novel_patterns = len(novel_patterns)
    result.n_insights = len(insights)

    # Count novel patterns by difficulty
    n_novel_simple = len([p for p in novel_patterns if p.get("difficulty") == "simple"])
    n_novel_moderate = len(
        [p for p in novel_patterns if p.get("difficulty") == "moderate"]
    )
    n_novel_complex = len(
        [p for p in novel_patterns if p.get("difficulty") == "complex"]
    )

    # Collect all variables for matching
    all_variables = set()
    for p in patterns:
        all_variables.update(extract_variables_from_pattern(p))

    # Track which patterns have been matched
    matched_patterns = set()

    # Match each insight
    total_score = 0.0

    for insight in insights:
        # Skip non-supported insights
        if insight.get("verdict") != "supported":
            result.n_unmatched += 1
            continue

        match = match_insight_to_patterns(
            insight,
            patterns,
            all_variables,
            variable_threshold,
            feature_to_variable=feature_to_variable,
        )

        if match is None:
            result.n_unmatched += 1
            result.matches.append(
                {
                    "insight_id": insight.get("id"),
                    "claim": insight.get("claim", "")[:100],
                    "matched": False,
                    "pattern_id": None,
                    "tier": None,
                }
            )
            continue

        # Record match
        matched_patterns.add(match.pattern_id)

        match_dict = {
            "insight_id": match.insight_id,
            "claim": insight.get("claim", "")[:100],
            "matched": True,
            "pattern_id": match.pattern_id,
            "tier": match.pattern_tier,
            "difficulty": match.pattern_difficulty,
            "score": match.match_score,
        }
        result.matches.append(match_dict)

        # Update counts and score by tier
        if match.pattern_tier == "prior_knowledge":
            result.n_matched_prior += 1
            result.prior_rediscovery_count += 1
            total_score += scoring["prior_rediscovery"]

        elif match.pattern_tier == "obvious":
            result.n_matched_obvious += 1
            result.obvious_rediscovery_count += 1
            total_score += scoring["obvious_rediscovery"]

        elif match.pattern_tier == "novel":
            if match.pattern_difficulty == "simple":
                result.n_matched_novel_simple += 1
                total_score += scoring["novel_simple_found"]
            elif match.pattern_difficulty == "moderate":
                result.n_matched_novel_moderate += 1
                total_score += scoring["novel_moderate_found"]
            elif match.pattern_difficulty == "complex":
                result.n_matched_novel_complex += 1
                total_score += scoring["novel_complex_found"]
            else:
                # Default to moderate
                result.n_matched_novel_moderate += 1
                total_score += scoring["novel_moderate_found"]

    # Compute metrics
    n_novel_matched = (
        result.n_matched_novel_simple
        + result.n_matched_novel_moderate
        + result.n_matched_novel_complex
    )

    # Precision: of insights that matched something, how many matched novel?
    n_matched_total = (
        result.n_matched_prior + result.n_matched_obvious + n_novel_matched
    )

    if n_matched_total > 0:
        result.precision_novel = n_novel_matched / n_matched_total

    # Recall: of novel patterns, how many were discovered?
    if result.n_novel_patterns > 0:
        result.recall_novel = n_novel_matched / result.n_novel_patterns

    # F1
    if result.precision_novel + result.recall_novel > 0:
        result.f1_novel = (
            2
            * result.precision_novel
            * result.recall_novel
            / (result.precision_novel + result.recall_novel)
        )

    # Recall by difficulty
    if n_novel_simple > 0:
        result.recall_novel_simple = result.n_matched_novel_simple / n_novel_simple
    if n_novel_moderate > 0:
        result.recall_novel_moderate = (
            result.n_matched_novel_moderate / n_novel_moderate
        )
    if n_novel_complex > 0:
        result.recall_novel_complex = result.n_matched_novel_complex / n_novel_complex

    # Rediscovery rate
    n_prior_obvious = result.n_prior_patterns + result.n_obvious_patterns
    if n_prior_obvious > 0:
        result.rediscovery_rate = (
            result.prior_rediscovery_count + result.obvious_rediscovery_count
        ) / n_prior_obvious

    # Benchmark score (normalized)
    max_possible_score = (
        n_novel_simple * scoring["novel_simple_found"]
        + n_novel_moderate * scoring["novel_moderate_found"]
        + n_novel_complex * scoring["novel_complex_found"]
    )

    if max_possible_score > 0:
        result.benchmark_score = total_score / max_possible_score
    else:
        result.benchmark_score = total_score

    from core.discovery.diversity_metrics import evaluate_diversity_from_insights

    diversity = evaluate_diversity_from_insights(insights, ground_truth)
    result.diversity = diversity.to_dict()

    return result


def print_evaluation_summary(result: EvaluationResult):
    """Print a human-readable summary of evaluation results."""
    print("\n" + "=" * 60)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 60)

    print("\n--- Ground Truth Patterns ---")
    print(f"  Prior knowledge: {result.n_prior_patterns}")
    print(f"  Obvious:         {result.n_obvious_patterns}")
    print(f"  Novel:           {result.n_novel_patterns}")

    print("\n--- Discovery Results ---")
    print(f"  Total insights:  {result.n_insights}")
    print(f"  Matched prior:   {result.n_matched_prior} (rediscovery)")
    print(f"  Matched obvious: {result.n_matched_obvious} (rediscovery)")
    print(f"  Matched novel (simple):   {result.n_matched_novel_simple}")
    print(f"  Matched novel (moderate): {result.n_matched_novel_moderate}")
    print(f"  Matched novel (complex):  {result.n_matched_novel_complex}")
    print(f"  Unmatched:       {result.n_unmatched}")

    print("\n--- Core Metrics ---")
    print(f"  Precision (novel): {result.precision_novel:.3f}")
    print(f"  Recall (novel):    {result.recall_novel:.3f}")
    print(f"  F1 (novel):        {result.f1_novel:.3f}")

    print("\n--- Recall by Difficulty ---")
    print(f"  Simple:   {result.recall_novel_simple:.3f}")
    print(f"  Moderate: {result.recall_novel_moderate:.3f}")
    print(f"  Complex:  {result.recall_novel_complex:.3f}")

    print("\n--- Rediscovery ---")
    print(f"  Rediscovery rate: {result.rediscovery_rate:.3f}")
    print(f"  Prior patterns found:   {result.prior_rediscovery_count}")
    print(f"  Obvious patterns found: {result.obvious_rediscovery_count}")

    if result.diversity:
        print("\n--- Ideation Diversity ---")
        d = result.diversity
        print(
            f"  Unique concept tuples: {d.get('n_unique_tuples', 0)} / {d.get('n_hypotheses', 0)} hypotheses"
        )
        print(f"  Concept coverage:      {d.get('concept_coverage', 0):.3f}")
        print(f"  Variable coverage:     {d.get('variable_coverage', 0):.3f}")
        print(f"  Variable pair coverage:{d.get('variable_pair_coverage', 0):.3f}")
        print(f"  Pairwise dissimilarity:{d.get('pairwise_dissimilarity', 0):.3f}")
        print(f"  Mean dim. entropy:     {d.get('mean_dimension_entropy', 0):.3f}")
        print(f"  Novelty per step:      {d.get('novelty_per_step', 0):.3f}")
        print(f"  Redundancy rate:       {d.get('redundancy_rate', 0):.3f}")
        dim_entropy = d.get("dimension_entropy", {})
        if dim_entropy:
            print(f"  Per-dimension entropy:")
            for dim, ent in dim_entropy.items():
                print(f"    {dim:15s}: {ent:.3f}")

    print("\n--- Overall ---")
    print(f"  BENCHMARK SCORE: {result.benchmark_score:.3f}")
    print("=" * 60 + "\n")


JUDGE_INSIGHT_PROMPT = """You are an impartial scientific judge evaluating whether a discovered insight matches a known ground truth pattern.

## Ground Truth Pattern
ID: {pattern_id}
Type: {pattern_type}
Description: "{ground_truth_description}"
Variables involved: {pattern_variables}

## Discovered Insight
"{claim}"

## Scoring Rubric
- Score 2 (MATCH): The insight describes the same relationship as the ground truth, even if worded differently or using species/variable names instead of column names
- Score 1 (PARTIAL): The insight is related to the ground truth but captures only part of the relationship, is too vague, or describes a consequence rather than the core pattern
- Score 0 (MISS): The insight describes a completely different or unrelated pattern

## Instructions
Focus on SEMANTIC equivalence, not exact wording. For example:
- "Amanita muscaria peaks in autumn" MATCHES "Fly Agaric observations concentrated in September-November" (score 2)
- "Fungi have narrow latitude range" PARTIALLY MATCHES "species_name correlates with latitude" (score 1 — related but less specific)
- "Fox observations have low accuracy" does NOT MATCH "Monarch butterflies migrate north in spring" (score 0)

Return JSON:
{{"score": 0 or 1 or 2, "reasoning": "brief explanation"}}
"""


def run_llm_judge_evaluation(
    ground_truth: dict,
    insights: list[dict],
    model_url: str = "https://YOUR_API_ENDPOINT/",
    model_name: str = "claude-sonnet-4-5",
    quiet: bool = False,
    batch_size: int = 5,
) -> dict[str, Any]:
    from core.agent import Agent

    agent = Agent(
        model_url=model_url,
        model_name=model_name,
        system_prompt=(
            "You are an impartial scientific judge. You will receive a batch of "
            "discovered insights and a list of ground truth patterns. For each insight, "
            "determine which ground truth pattern (if any) it best matches, and score "
            "on a 0-2 scale (0=no match, 1=partial, 2=exact). Be strict but fair — "
            "capture semantic equivalence, not just exact wording."
        ),
        secrets={},
    )

    patterns = ground_truth.get("patterns", [])
    supported = [i for i in insights if i.get("verdict") == "supported"]

    if not quiet:
        print("\n" + "=" * 60)
        print("LLM-AS-JUDGE EVALUATION (mini-batch)")
        print("=" * 60)
        print(f"  Supported insights: {len(supported)}")
        print(f"  Ground truth patterns: {len(patterns)}")
        print(f"  Batch size: {batch_size}")
        n_batches = (len(supported) + batch_size - 1) // batch_size
        print(
            f"  Total LLM calls: {n_batches} (vs {len(supported) * len(patterns)} pairwise)"
        )

    patterns_text = "\n".join(
        f'  - ID="{p.get("id", "?")}" (tier={p.get("tier", "?")}, difficulty={p.get("difficulty", "?")}): '
        f"{p.get('description', '')}"
        for p in patterns
    )

    pattern_best_scores: dict[str, dict[str, Any]] = {}
    insight_best_scores: dict[int, dict[str, Any]] = {}

    for batch_start in range(0, len(supported), batch_size):
        batch = supported[batch_start : batch_start + batch_size]
        batch_indices = list(range(batch_start, batch_start + len(batch)))

        insights_text = "\n".join(
            f"  I{k + 1}. {ins.get('claim', '')[:300]}" for k, ins in enumerate(batch)
        )

        prompt = f"""Score each discovered insight against the ground truth patterns below.

## Ground Truth Patterns
{patterns_text}

## Discovered Insights (this batch)
{insights_text}

## Instructions
For EACH insight I1..I{len(batch)}, output:
- best_pattern_id: the exact ID string from the pattern list (e.g. "monarch_latitude_season"), or "none" if no match
- score: 0 (no match), 1 (partial - related but different angle), 2 (exact semantic match)
- reasoning: brief explanation (1 sentence)

IMPORTANT: best_pattern_id must be the exact ID string shown in quotes after "ID=" above. Do NOT use numbers like "P5".

Scoring guidance:
- Score 2 if the insight captures the SAME underlying phenomenon, even if it uses different species
  or generalizes (e.g., GT="Monarch migrates north in spring" matches "Migratory insects shift northward in spring")
- Score 1 if the insight is related but captures a different aspect or is too vague
- Score 0 if the insight describes a completely different phenomenon

The KEY question is: does this insight reflect the same ecological mechanism?
A generalized version of a GT pattern (using higher taxonomic level or different exemplar species)
should score 2 if the core mechanism is identical.

Do NOT match insights to prior_knowledge or obvious tier patterns — only match to novel patterns.
Each insight should match AT MOST one pattern."""

        output_schema = {
            "type": "object",
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "insight_index": {"type": "integer"},
                            "best_pattern_id": {"type": "string"},
                            "score": {"type": "integer", "enum": [0, 1, 2]},
                            "reasoning": {"type": "string"},
                        },
                        "required": [
                            "insight_index",
                            "best_pattern_id",
                            "score",
                            "reasoning",
                        ],
                    },
                },
            },
            "required": ["matches"],
        }

        try:
            result = agent.act_json(
                instruction=prompt,
                output_schema=output_schema,
                tool_name="batch_judge",
                tool_description="Score a batch of insights against ground truth patterns",
                max_retries=2,
            )
            matches = result.get("matches", [])
        except Exception as e:
            if not quiet:
                print(f"  Batch {batch_start // batch_size + 1} error: {e}")
            matches = []

        for m in matches:
            idx_in_batch = m.get("insight_index", 1) - 1
            if idx_in_batch < 0 or idx_in_batch >= len(batch):
                continue
            global_idx = batch_indices[idx_in_batch]
            score = int(m.get("score", 0))
            pid = m.get("best_pattern_id", "none")
            reasoning = m.get("reasoning", "")

            insight_best_scores[global_idx] = {
                "score": score,
                "pattern_id": pid if pid != "none" else None,
                "reasoning": reasoning,
            }

            if pid and pid != "none" and score > 0:
                if pid not in pattern_best_scores or score > pattern_best_scores[
                    pid
                ].get("score", 0):
                    pattern_best_scores[pid] = {
                        "score": score,
                        "insight_idx": global_idx,
                        "claim": batch[idx_in_batch].get("claim", "")[:100],
                        "reasoning": reasoning,
                    }

        if not quiet:
            batch_num = batch_start // batch_size + 1
            n_matched = sum(1 for m in matches if m.get("score", 0) > 0)
            print(f"  Batch {batch_num}: {n_matched}/{len(batch)} insights matched")

    for i in range(len(supported)):
        if i not in insight_best_scores:
            insight_best_scores[i] = {
                "score": 0,
                "pattern_id": None,
                "reasoning": "not evaluated",
            }

    novel_patterns = [
        p for p in patterns if p.get("tier") not in ("prior_knowledge", "obvious")
    ]
    prior_patterns = [p for p in patterns if p.get("tier") == "prior_knowledge"]

    novel_matched_exact = sum(
        1
        for p in novel_patterns
        if p["id"] in pattern_best_scores and pattern_best_scores[p["id"]]["score"] == 2
    )
    novel_matched_partial = sum(
        1
        for p in novel_patterns
        if p["id"] in pattern_best_scores and pattern_best_scores[p["id"]]["score"] >= 1
    )

    insight_scores = [v["score"] for v in insight_best_scores.values()]
    n_insights_exact = sum(1 for s in insight_scores if s == 2)
    n_insights_partial = sum(1 for s in insight_scores if s >= 1)

    n_novel = len(novel_patterns) if novel_patterns else 1

    precision_exact = n_insights_exact / len(supported) if supported else 0
    precision_partial = n_insights_partial / len(supported) if supported else 0
    recall_exact = novel_matched_exact / n_novel
    recall_partial = novel_matched_partial / n_novel
    f1_exact = (
        2 * precision_exact * recall_exact / (precision_exact + recall_exact)
        if (precision_exact + recall_exact) > 0
        else 0
    )
    f1_partial = (
        2 * precision_partial * recall_partial / (precision_partial + recall_partial)
        if (precision_partial + recall_partial) > 0
        else 0
    )

    summary = {
        "n_supported_insights": len(supported),
        "n_novel_patterns": len(novel_patterns),
        "n_insights_exact_match": n_insights_exact,
        "n_insights_partial_match": n_insights_partial,
        "n_novel_patterns_found_exact": novel_matched_exact,
        "n_novel_patterns_found_partial": novel_matched_partial,
        "precision_exact": precision_exact,
        "precision_partial": precision_partial,
        "recall_exact": recall_exact,
        "recall_partial": recall_partial,
        "f1_exact": f1_exact,
        "f1_partial": f1_partial,
        "mean_best_score": sum(insight_scores) / len(insight_scores)
        if insight_scores
        else 0,
        "per_insight": [
            {
                "claim": supported[i].get("claim", "")[:150],
                "best_score": v["score"],
                "best_pattern": v["pattern_id"],
                "reasoning": v["reasoning"][:200],
            }
            for i, v in insight_best_scores.items()
        ],
        "per_pattern_novel": [
            {
                "pattern_id": p["id"],
                "description": p.get("description", "")[:150],
                "best_score": pattern_best_scores.get(p["id"], {}).get("score", 0),
                "matched_claim": pattern_best_scores.get(p["id"], {}).get("claim", ""),
            }
            for p in novel_patterns
        ],
    }

    if not quiet:
        print(f"\n--- LLM Judge Summary ---")
        print(f"  Precision (exact):  {precision_exact:.3f}")
        print(f"  Precision (partial):{precision_partial:.3f}")
        print(f"  Recall (exact):     {recall_exact:.3f}")
        print(f"  Recall (partial):   {recall_partial:.3f}")
        print(f"  F1 (exact):         {f1_exact:.3f}")
        print(f"  F1 (partial):       {f1_partial:.3f}")
        print(f"  Mean score:         {summary['mean_best_score']:.2f}/2.0")
        print("=" * 60 + "\n")

    return summary


def compute_diversity_metrics(insights: list[dict]) -> dict[str, float]:
    """Compute diversity metrics over a list of discovered insights.

    Metrics:
        uniqueness_rate: fraction of insights with unique variable-set hashes.
        variable_entropy: Shannon entropy over variable usage (normalised 0-1).
        fixation_score: max(var_counts) / total_mentions (flag if > 0.5).
    """
    from collections import Counter

    if not insights:
        return {
            "uniqueness_rate": 0.0,
            "variable_entropy": 0.0,
            "fixation_score": 0.0,
        }

    # Build a hash per insight from its mentioned variables
    var_counter: Counter = Counter()
    var_set_hashes: list[str] = []
    for ins in insights:
        claim = ins.get("claim", "")
        # Extract a simple variable-set proxy from the claim
        # Use all words that look like variable names (lowercase + underscore)
        import re

        tokens = set(re.findall(r"\b[a-z][a-z0-9_]*\b", claim.lower()))
        var_set_hash = str(sorted(tokens))
        var_set_hashes.append(var_set_hash)
        var_counter.update(tokens)

    # Uniqueness rate
    n_unique = len(set(var_set_hashes))
    uniqueness_rate = n_unique / len(var_set_hashes) if var_set_hashes else 0.0

    # Variable entropy (normalised Shannon entropy)
    total_mentions = sum(var_counter.values())
    if total_mentions == 0 or len(var_counter) <= 1:
        variable_entropy = 0.0
    else:
        probs = np.array(list(var_counter.values()), dtype=float) / total_mentions
        raw_entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))
        max_entropy = float(np.log2(len(var_counter)))
        variable_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0

    # Fixation score
    fixation_score = (
        max(var_counter.values()) / total_mentions if total_mentions > 0 else 0.0
    )

    metrics = {
        "uniqueness_rate": uniqueness_rate,
        "variable_entropy": variable_entropy,
        "fixation_score": fixation_score,
    }

    # Self-BLEU (lower = more diverse)
    try:
        from diversity import self_bleu, compression_ratio

        claims = [ins.get("claim", "") for ins in insights if ins.get("claim")]
        if len(claims) >= 2:
            sb = self_bleu(claims, n=4)
            cr = compression_ratio(claims)
            metrics["self_bleu_4"] = sb
            metrics["compression_ratio"] = cr
    except ImportError:
        pass

    # Vendi Score (higher = more effective distinct modes)
    try:
        from sentence_transformers import SentenceTransformer
        from vendi_score import vendi

        model = SentenceTransformer("all-MiniLM-L6-v2")
        claims = [ins.get("claim", "") for ins in insights if ins.get("claim")]
        if len(claims) >= 2:
            embeddings = model.encode(claims, normalize_embeddings=True)
            similarity_matrix = embeddings @ embeddings.T
            vs = vendi.score_K(similarity_matrix)
            metrics["vendi_score"] = float(vs)
    except ImportError:
        pass

    return metrics


def print_diversity_summary(diversity: dict[str, float]) -> None:
    """Print diversity metrics summary."""
    print("\n--- Diversity Metrics ---")
    print(f"  Uniqueness rate:   {diversity['uniqueness_rate']:.3f}")
    print(f"  Variable entropy:  {diversity['variable_entropy']:.3f}")
    fixation = diversity["fixation_score"]
    flag = " (FIXATED)" if fixation > 0.5 else ""
    print(f"  Fixation score:    {fixation:.3f}{flag}")
    if "self_bleu_4" in diversity:
        print(
            f"  Self-BLEU-4:       {diversity['self_bleu_4']:.3f}  (lower = more diverse)"
        )
    if "compression_ratio" in diversity:
        print(f"  Compression ratio: {diversity['compression_ratio']:.3f}")
    if "vendi_score" in diversity:
        print(
            f"  Vendi Score:       {diversity['vendi_score']:.2f}  (higher = more diverse)"
        )


def print_hypospace_summary(eval_data: dict) -> None:
    """Print formatted VR/NR/RR summary from a HypoSpace evaluation.json."""
    print("\n" + "=" * 60)
    print("HYPOSPACE EVALUATION RESULTS")
    print("=" * 60)

    domain = eval_data.get("domain", "unknown")
    print(f"\n  Domain: {domain}")

    agg = eval_data.get("aggregate_metrics", eval_data)

    for key, label in [
        ("mean_validity_rate", "Validity Rate (VR)"),
        ("mean_uniqueness_rate", "Uniqueness Rate (NR)"),
        ("mean_recovery_rate", "Recovery Rate (RR)"),
    ]:
        mean_val = agg.get(key, agg.get(key.replace("mean_", "global_"), 0))
        std_key = key.replace("mean_", "std_")
        std_val = agg.get(std_key, 0)
        print(f"  {label}: {mean_val:.3f} +/- {std_val:.3f}")

    n_sets = agg.get("n_observation_sets", "?")
    n_queries = agg.get("n_total_queries", "?")
    print(f"\n  Observation sets: {n_sets}")
    print(f"  Total queries:    {n_queries}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate discovery results against ground truth"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        required=False,
        default=None,
        help="Path to ground_truth.json file",
    )
    parser.add_argument(
        "--discovered",
        type=str,
        required=True,
        help="Path to discovered insights (parquet or JSON) or HypoSpace workspace dir",
    )
    parser.add_argument(
        "--output", type=str, help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Variable overlap threshold for matching (default: 0.3)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress printed summary")
    parser.add_argument(
        "--feature_mapping",
        type=str,
        default=None,
        help="Path to feature_variable_mapping.json (for visual benchmarks)",
    )
    parser.add_argument(
        "--llm_judge",
        action="store_true",
        help="Use LLM-as-judge for semantic matching (recommended for real-world datasets)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="claude-sonnet-4-5",
        help="Model name for LLM judge (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--judge_url",
        type=str,
        default="https://YOUR_API_ENDPOINT/",
        help="Model URL for LLM judge",
    )
    parser.add_argument(
        "--hypospace",
        action="store_true",
        help="Evaluate HypoSpace benchmark results (--discovered points to workspace dir)",
    )

    args = parser.parse_args()

    # ---- HypoSpace evaluation mode ----
    if args.hypospace:
        discovered_path = args.discovered
        # Try to find evaluation.json
        if os.path.isdir(discovered_path):
            eval_path = os.path.join(discovered_path, "evaluation.json")
        else:
            eval_path = discovered_path

        if not os.path.exists(eval_path):
            print(f"Error: Cannot find evaluation.json at {eval_path}")
            sys.exit(1)

        with open(eval_path, "r") as f:
            eval_data = json.load(f)

        # Auto-detect HypoSpace format
        is_hypospace = "validity_rate" in str(eval_data)
        if not is_hypospace:
            print("Warning: File does not appear to be HypoSpace format")

        if not args.quiet:
            print_hypospace_summary(eval_data)

        return eval_data

    # ---- Standard evaluation mode ----
    if args.ground_truth is None:
        parser.error("--ground_truth is required unless --hypospace is specified")

    with open(args.ground_truth, "r") as f:
        ground_truth = json.load(f)

    if args.discovered.endswith(".parquet"):
        df = pd.read_parquet(args.discovered)
        insights = df.to_dict("records")
    elif args.discovered.endswith(".json"):
        with open(args.discovered, "r") as f:
            insights = json.load(f)
            if isinstance(insights, dict):
                insights = insights.get("insights", [])
    else:
        raise ValueError(f"Unsupported file format: {args.discovered}")

    feature_to_variable = None
    if args.feature_mapping:
        with open(args.feature_mapping, "r") as f:
            feature_to_variable = json.load(f)
    elif not args.feature_mapping:
        gt_dir = os.path.dirname(args.ground_truth)
        auto_path = os.path.join(gt_dir, "feature_variable_mapping.json")
        if os.path.exists(auto_path):
            with open(auto_path, "r") as f:
                feature_to_variable = json.load(f)

    result = evaluate_discovery(
        ground_truth,
        insights,
        variable_threshold=args.threshold,
        feature_to_variable=feature_to_variable,
    )

    if not args.quiet:
        print_evaluation_summary(result)

    # Compute and print diversity metrics
    diversity = compute_diversity_metrics(insights)
    if not args.quiet:
        print_diversity_summary(diversity)

    llm_judge_results = None
    if args.llm_judge:
        llm_judge_results = run_llm_judge_evaluation(
            ground_truth,
            insights,
            model_url=args.judge_url,
            model_name=args.judge_model,
            quiet=args.quiet,
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = result.to_dict()
        output_data["diversity"] = diversity
        if llm_judge_results is not None:
            output_data["llm_judge"] = llm_judge_results
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        if not args.quiet:
            print(f"Saved evaluation results to {args.output}")

    return result


if __name__ == "__main__":
    main()
