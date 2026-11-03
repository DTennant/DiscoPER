"""
Hierarchical ground truth extraction from causal DAGs.

Walks the DAG to automatically produce ground truth patterns at three levels:
1. Surface correlations — pairwise |r| above threshold
2. Direct causal rules — each rule mapped to a pattern
3. Emergent patterns — mediation chains, confound forks, latent structures

Output uses the exact pattern type strings and dict keys expected by
evaluate_discovery.py and BenchmarkDatasetAdapter.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from core.benchmark.rules import (
    CausalDAG,
    CausalRule,
    CompetitionRule,
    InteractionRule,
    LinearCausalRule,
    ModulationRule,
    RuleSet,
    SaturationRule,
    ThresholdCausalRule,
)


def extract_ground_truth(
    ruleset: RuleSet,
    df: pd.DataFrame,
    world_state: dict[str, np.ndarray],
    prior_rule_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Extract hierarchical ground truth from a causal DAG and simulated data.

    Args:
        ruleset: The RuleSet used for simulation
        df: Simulated DataFrame
        world_state: Raw numpy arrays from simulation
        prior_rule_ids: Rule IDs that represent prior knowledge (tier=prior_knowledge)

    Returns:
        Ground truth dict compatible with evaluate_discovery.py
    """
    if prior_rule_ids is None:
        prior_rule_ids = ruleset.prior_rule_ids

    dag = ruleset.dag
    prior_ids = set(prior_rule_ids)

    # Extract patterns at each level
    direct_patterns = _extract_direct_causal_patterns(dag, prior_ids)
    surface_patterns = _extract_surface_correlations(
        dag, df, world_state, ruleset.correlation_threshold, direct_patterns
    )
    mediation_patterns = _extract_mediation_chains(dag, world_state)
    confound_patterns = _extract_confound_structures(dag, world_state)

    # Combine all patterns
    all_patterns = surface_patterns + direct_patterns + mediation_patterns + confound_patterns

    # Assign unique IDs
    for i, p in enumerate(all_patterns):
        if "id" not in p or not p["id"]:
            p["id"] = f"pattern_{i:03d}"

    # Build ground truth dict
    ground_truth = {
        "dataset_id": ruleset.name,
        "seed": ruleset.seed,
        "n_samples": ruleset.n_samples,
        "generation_method": "causal_dag",
        "causal_dag": ruleset.dag.to_dict(),
        "feature_schema": {
            "exogenous": [
                name for name, spec in dag.variables.items()
                if spec.var_type == "exogenous"
            ],
            "derived": [
                name for name, spec in dag.variables.items()
                if spec.var_type == "derived"
            ],
            "observation": ["observer_id", "observer_experience", "timestamp", "season"],
        },
        "patterns": all_patterns,
        "evaluation_criteria": {
            "scoring": {
                "prior_rediscovery": -1.0,
                "obvious_rediscovery": -0.5,
                "novel_simple_found": 0.5,
                "novel_moderate_found": 1.0,
                "novel_complex_found": 2.0,
            },
            "matching": {
                "variable_overlap_threshold": 0.5,
                "effect_direction_must_match": True,
                "llm_semantic_matching": True,
            },
        },
        "include_prior_in_prompt": True,
        "prior_prompt_detail": "full",
    }

    return ground_truth


# =============================================================================
# Direct causal pattern extraction
# =============================================================================

def _rule_to_pattern_type(rule: CausalRule) -> str:
    """Map a CausalRule subclass to an existing PatternType string."""
    if isinstance(rule, LinearCausalRule):
        return "correlation"
    elif isinstance(rule, ThresholdCausalRule):
        return "threshold"
    elif isinstance(rule, InteractionRule):
        return "two_way_interaction"
    elif isinstance(rule, ModulationRule):
        return "two_way_interaction"
    elif isinstance(rule, CompetitionRule):
        return "competition_masking"
    elif isinstance(rule, SaturationRule):
        return "nonlinear_piecewise"
    else:
        return "correlation"


def _assign_difficulty(rule: CausalRule, dag: CausalDAG) -> str:
    """Assign difficulty based on rule type, strength, and causal depth."""
    depth = dag.get_causal_depth(rule.target)

    # Base difficulty from rule type
    if isinstance(rule, (LinearCausalRule,)):
        base = "simple"
    elif isinstance(rule, (ThresholdCausalRule, InteractionRule, ModulationRule)):
        base = "moderate"
    elif isinstance(rule, (CompetitionRule, SaturationRule)):
        base = "moderate"
    else:
        base = "simple"

    # Upgrade difficulty for weak effects or deep variables
    if abs(rule.strength) < 0.3:
        if base == "simple":
            base = "moderate"
        elif base == "moderate":
            base = "complex"

    if depth >= 3:
        if base == "simple":
            base = "moderate"
        elif base == "moderate":
            base = "complex"

    return base


def _extract_direct_causal_patterns(
    dag: CausalDAG, prior_ids: set[str]
) -> list[dict[str, Any]]:
    """Map each rule in the DAG to a ground truth pattern dict."""
    patterns = []

    for rule in dag.rules:
        ptype = _rule_to_pattern_type(rule)
        tier = "prior_knowledge" if rule.id in prior_ids else "novel"
        difficulty = _assign_difficulty(rule, dag) if tier == "novel" else None

        pattern: dict[str, Any] = {
            "id": f"direct_{rule.id}",
            "type": ptype,
            "tier": tier,
            "description": rule.description or f"{rule.rule_type}: {' + '.join(rule.sources)} -> {rule.target}",
            "causal_rule_id": rule.id,
        }

        if difficulty:
            pattern["difficulty"] = difficulty

        # Add type-specific keys for evaluate_discovery.py compatibility
        if isinstance(rule, LinearCausalRule):
            pattern["variables"] = [rule.sources[0], rule.target]
            pattern["effect"] = rule.strength
            pattern["effect_size"] = rule.strength

        elif isinstance(rule, ThresholdCausalRule):
            pattern["variables"] = [rule.sources[0], rule.target]
            pattern["threshold"] = rule.threshold
            pattern["effect_below"] = 0.0 if rule.above else rule.strength
            pattern["effect_above"] = rule.strength if rule.above else 0.0

        elif isinstance(rule, InteractionRule):
            pattern["predictors"] = list(rule.sources[:2])
            pattern["outcome"] = rule.target
            pattern["interaction_effect"] = rule.strength

        elif isinstance(rule, ModulationRule):
            pattern["predictors"] = list(rule.sources[:2])
            pattern["outcome"] = rule.target
            pattern["interaction_effect"] = rule.strength

        elif isinstance(rule, CompetitionRule):
            pattern["competing_species"] = [rule.sources[0], rule.target]
            if len(rule.sources) > 1:
                pattern["keystone"] = rule.sources[1]
            else:
                pattern["keystone"] = rule.sources[0]
            pattern["effect_without_keystone"] = -abs(rule.strength)
            pattern["effect_with_keystone"] = -abs(rule.strength) * (1 - rule.modifier_weakening)

        elif isinstance(rule, SaturationRule):
            pattern["variables"] = [rule.sources[0], rule.target]
            # Approximate with breakpoints for piecewise format
            pattern["breakpoints"] = [rule.capacity * 0.5]
            pattern["slopes"] = [rule.strength, rule.strength * 0.3]

        patterns.append(pattern)

    return patterns


# =============================================================================
# Surface correlation extraction
# =============================================================================

def _extract_surface_correlations(
    dag: CausalDAG,
    df: pd.DataFrame,
    world_state: dict[str, np.ndarray],
    threshold: float,
    direct_patterns: list[dict],
) -> list[dict[str, Any]]:
    """Compute pairwise correlations and emit obvious-tier patterns for strong ones."""
    # Collect variable pairs already covered by direct patterns
    covered_pairs: set[frozenset[str]] = set()
    for p in direct_patterns:
        pvars = set()
        if "variables" in p:
            pvars.update(p["variables"])
        if "predictors" in p:
            pvars.update(p["predictors"])
        if "outcome" in p:
            pvars.add(p["outcome"])
        if "competing_species" in p:
            pvars.update(p["competing_species"])
        if len(pvars) == 2:
            covered_pairs.add(frozenset(pvars))

    # Use raw world_state for correlations (before dtype conversion/noise)
    numeric_vars = [
        name for name, spec in dag.variables.items()
        if spec.dtype in ("float", "int")
    ]

    patterns = []
    for v1, v2 in combinations(numeric_vars, 2):
        pair = frozenset([v1, v2])
        if pair in covered_pairs:
            continue
        if v1 not in world_state or v2 not in world_state:
            continue

        arr1 = world_state[v1]
        arr2 = world_state[v2]
        if arr1.std() < 1e-10 or arr2.std() < 1e-10:
            continue

        r = np.corrcoef(arr1, arr2)[0, 1]
        if abs(r) >= threshold:
            patterns.append({
                "id": f"surface_{v1}_{v2}",
                "type": "correlation",
                "tier": "obvious",
                "description": f"Surface correlation between {v1} and {v2} (r={r:.2f})",
                "variables": [v1, v2],
                "effect": float(r),
                "effect_size": float(r),
            })

    return patterns


# =============================================================================
# Mediation chain extraction
# =============================================================================

def _extract_mediation_chains(
    dag: CausalDAG, world_state: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Find all transitive paths of length >= 2 and emit mediation_chain patterns."""
    patterns = []
    seen_chains: set[tuple[str, ...]] = set()

    # Find all exogenous variables as potential chain starts
    exogenous = [name for name, spec in dag.variables.items() if spec.var_type == "exogenous"]

    for start in exogenous:
        # Get all paths of length >= 3 nodes (>= 2 edges)
        all_vars = list(dag.variables.keys())
        for end in all_vars:
            if end == start:
                continue
            paths = dag.get_all_paths(start, end, max_depth=5)
            for path in paths:
                if len(path) < 3:
                    continue
                chain_key = tuple(path)
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                # Compute direct effects along the chain
                direct_effects = []
                for i in range(len(path) - 1):
                    src, tgt = path[i], path[i + 1]
                    # Find the strongest rule for this edge
                    edge_rules = [
                        r for r in dag.rules
                        if r.target == tgt and src in r.sources
                    ]
                    if edge_rules:
                        strongest = max(edge_rules, key=lambda r: abs(r.strength))
                        direct_effects.append(float(strongest.strength))
                    else:
                        direct_effects.append(0.0)

                # Compute indirect effect as product
                indirect = math.prod(direct_effects) if direct_effects else 0.0

                # Verify the chain actually manifests in data
                if (path[0] in world_state and path[-1] in world_state
                        and world_state[path[0]].std() > 1e-10
                        and world_state[path[-1]].std() > 1e-10):
                    end_to_end_r = np.corrcoef(
                        world_state[path[0]], world_state[path[-1]]
                    )[0, 1]
                    if abs(end_to_end_r) < 0.05:
                        continue  # Chain doesn't manifest detectably

                difficulty = "moderate" if len(path) <= 3 else "complex"

                patterns.append({
                    "id": f"chain_{'_'.join(path)}",
                    "type": "mediation_chain",
                    "tier": "novel",
                    "difficulty": difficulty,
                    "description": (
                        f"Mediation chain: {' -> '.join(path)} "
                        f"(indirect effect = {indirect:.3f})"
                    ),
                    "chain": list(path),
                    "direct_effects": direct_effects,
                    "indirect_effect": float(indirect),
                })

    return patterns


# =============================================================================
# Confound / fork structure extraction
# =============================================================================

def _extract_confound_structures(
    dag: CausalDAG, world_state: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Find fork patterns (common cause) and latent variable structures."""
    patterns = []

    for var_name in dag.variables:
        children = list(dag.get_children(var_name))
        if len(children) < 2:
            continue

        # Fork patterns: common cause with exactly 2 children
        for c1, c2 in combinations(children, 2):
            # Check that c1 and c2 don't have a direct edge
            c1_parents = dag.get_parents(c1)
            c2_parents = dag.get_parents(c2)
            c1_children = dag.get_children(c1)
            c2_children = dag.get_children(c2)

            has_direct = (c2 in c1_children or c1 in c2_children
                          or c2 in c1_parents or c1 in c2_parents)

            if has_direct:
                continue

            # Verify confound creates detectable spurious correlation
            if (c1 in world_state and c2 in world_state
                    and world_state[c1].std() > 1e-10
                    and world_state[c2].std() > 1e-10):
                r = np.corrcoef(world_state[c1], world_state[c2])[0, 1]
                if abs(r) < 0.1:
                    continue

                patterns.append({
                    "id": f"confound_{var_name}_{c1}_{c2}",
                    "type": "conditional_confound",
                    "tier": "novel",
                    "difficulty": "moderate",
                    "description": (
                        f"Confound: {c1} and {c2} appear correlated (r={r:.2f}) "
                        f"due to common cause {var_name}"
                    ),
                    "apparent_vars": [c1, c2],
                    "apparent_correlation": [c1, c2],
                    "confound": var_name,
                    "apparent_effect": float(r),
                    "true_effect": 0.0,
                })

        # Latent variable pattern: common cause with 3+ children
        if len(children) >= 3:
            # Check which children have detectable correlations
            numeric_children = [
                c for c in children
                if c in world_state and world_state[c].std() > 1e-10
            ]
            if len(numeric_children) >= 3:
                patterns.append({
                    "id": f"latent_{var_name}",
                    "type": "latent_variable",
                    "tier": "novel",
                    "difficulty": "complex",
                    "description": (
                        f"Latent variable: {', '.join(numeric_children)} "
                        f"correlate due to hidden factor {var_name}"
                    ),
                    "observed_vars": numeric_children,
                    "latent_var": var_name,
                })

    return patterns
