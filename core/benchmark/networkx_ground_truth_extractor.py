"""
Ground truth extraction from NetworkX DiGraph + DataFrame.

Works with any directed acyclic graph and associated data, inferring pattern
types and effect sizes from data rather than from rule definitions. Output
format is identical to the GOL ground_truth_extractor so that
evaluate_discovery.py and BenchmarkDatasetAdapter work unchanged.
"""

from __future__ import annotations

import warnings
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats


def extract_ground_truth_from_digraph(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    dataset_id: str,
    seed: int = 42,
    correlation_threshold: float = 0.15,
    prior_edge_ids: list[tuple[str, str]] | None = None,
    prior_chain_ids: list[tuple[str, ...]] | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """
    Extract hierarchical ground truth from a NetworkX DiGraph and DataFrame.

    Produces patterns at five levels:
    1. Direct edge patterns — one per DAG edge
    2. Surface correlations — pairwise |r| above threshold, not direct edges
    3. Mediation chains — transitive paths of length >= 3 nodes
    4. Confound structures — common-cause forks
    5. Interaction patterns — multi-parent nodes with significant interactions

    Args:
        graph: NetworkX DiGraph representing the causal DAG
        df: DataFrame with columns matching graph node names
        dataset_id: Identifier for the dataset
        seed: Random seed (for reproducibility metadata)
        correlation_threshold: Minimum |r| for surface correlations
        prior_edge_ids: List of (source, target) tuples for prior knowledge edges
        prior_chain_ids: List of (node1, node2, ...) tuples for prior knowledge chains
        metadata: Additional metadata to include in output

    Returns:
        Ground truth dict compatible with evaluate_discovery.py
    """
    if prior_edge_ids is None:
        prior_edge_ids = []
    if prior_chain_ids is None:
        prior_chain_ids = []

    prior_edges_set = {(s, t) for s, t in prior_edge_ids}
    prior_chains_set = {tuple(c) for c in prior_chain_ids}

    # Only use numeric columns that are in the graph
    graph_nodes = set(graph.nodes())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    available_vars = [c for c in numeric_cols if c in graph_nodes]

    # Extract patterns at each level
    direct_patterns = _extract_direct_edge_patterns(
        graph, df, available_vars, prior_edges_set
    )
    surface_patterns = _extract_surface_correlations(
        graph, df, available_vars, correlation_threshold, direct_patterns
    )
    mediation_patterns = _extract_mediation_chains(
        graph, df, available_vars, prior_chains_set
    )
    confound_patterns = _extract_confound_structures(graph, df, available_vars)
    interaction_patterns = _extract_interaction_patterns(graph, df, available_vars)

    # Combine all patterns
    all_patterns = (
        surface_patterns
        + direct_patterns
        + mediation_patterns
        + confound_patterns
        + interaction_patterns
    )

    # Assign unique IDs where missing
    for i, p in enumerate(all_patterns):
        if "id" not in p or not p["id"]:
            p["id"] = f"pattern_{i:03d}"

    # Build ground truth dict (same schema as GOL extractor)
    ground_truth: dict[str, Any] = {
        "dataset_id": dataset_id,
        "seed": seed,
        "n_samples": len(df),
        "generation_method": "realworld_causal_dag",
        "feature_schema": {
            "variables": available_vars,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
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

    if metadata:
        ground_truth["metadata"] = metadata

    return ground_truth


# =============================================================================
# Helper: partial correlation
# =============================================================================


def _compute_partial_correlation(
    df: pd.DataFrame, var1: str, var2: str, conditioning_set: list[str]
) -> float:
    """
    Compute partial correlation between var1 and var2 controlling for
    conditioning_set via OLS residualization.
    """
    if not conditioning_set:
        return float(df[var1].corr(df[var2]))

    from numpy.linalg import lstsq

    valid = df[[var1, var2] + conditioning_set].dropna()
    if len(valid) < 10:
        return 0.0

    X = valid[conditioning_set].values
    X_with_const = np.column_stack([np.ones(len(X)), X])

    y1 = valid[var1].values
    y2 = valid[var2].values

    # Residualize var1
    coef1, _, _, _ = lstsq(X_with_const, y1, rcond=None)
    resid1 = y1 - X_with_const @ coef1

    # Residualize var2
    coef2, _, _, _ = lstsq(X_with_const, y2, rcond=None)
    resid2 = y2 - X_with_const @ coef2

    if resid1.std() < 1e-10 or resid2.std() < 1e-10:
        return 0.0

    return float(np.corrcoef(resid1, resid2)[0, 1])


# =============================================================================
# Helper: difficulty assignment
# =============================================================================


def _assign_difficulty(
    graph: nx.DiGraph, var: str, effect_size: float, n_parents: int = 1
) -> str:
    """
    Assign difficulty without rule types.
    simple = shallow depth + strong |r| >= 0.5
    complex = deep (>= 3) or weak |r| < 0.2 or >= 3 parents
    moderate = everything else
    """
    # Compute depth: longest path from any root to var
    try:
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        max_depth = 0
        for root in roots:
            for path in nx.all_simple_paths(graph, root, var):
                max_depth = max(max_depth, len(path) - 1)
    except (nx.NetworkXError, nx.NodeNotFound):
        max_depth = 0

    abs_effect = abs(effect_size)

    if max_depth >= 3 or abs_effect < 0.2 or n_parents >= 3:
        return "complex"
    elif max_depth <= 1 and abs_effect >= 0.5:
        return "simple"
    else:
        return "moderate"


# =============================================================================
# Direct edge patterns
# =============================================================================


def _classify_edge_type(
    df: pd.DataFrame, source: str, target: str
) -> tuple[str, float]:
    """
    Classify an edge's pattern type by comparing Pearson vs Spearman.
    Returns (pattern_type, pearson_r).
    """
    col_src = df[source]
    col_tgt = df[target]

    if col_src.std() < 1e-10 or col_tgt.std() < 1e-10:
        return "correlation", 0.0

    pearson_r = float(col_src.corr(col_tgt))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spearman_r, _ = stats.spearmanr(col_src, col_tgt)

    # If Pearson and Spearman diverge significantly, likely threshold/nonlinear
    if abs(pearson_r) > 0.05 and abs(spearman_r) > 0.05:
        ratio = abs(pearson_r) / max(abs(spearman_r), 1e-10)
        if ratio < 0.7 or ratio > 1.4:
            return "threshold", pearson_r

    return "correlation", pearson_r


def _extract_direct_edge_patterns(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    available_vars: list[str],
    prior_edges: set[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Map each DAG edge to a ground truth pattern dict."""
    patterns = []
    available_set = set(available_vars)

    for source, target in graph.edges():
        if source not in available_set or target not in available_set:
            continue

        ptype, pearson_r = _classify_edge_type(df, source, target)

        # Check if multi-parent node has interaction
        parents = list(graph.predecessors(target))
        parents_available = [p for p in parents if p in available_set]
        if len(parents_available) >= 2 and source in parents_available:
            # Test for interaction with each co-parent
            for co_parent in parents_available:
                if co_parent == source:
                    continue
                has_interaction = _test_interaction(df, source, co_parent, target)
                if has_interaction:
                    ptype = "two_way_interaction"
                    break

        is_prior = (source, target) in prior_edges
        tier = "prior_knowledge" if is_prior else "novel"
        n_parents = len(parents_available)

        pattern: dict[str, Any] = {
            "id": f"direct_{source}_{target}",
            "type": ptype,
            "tier": tier,
            "variables": [source, target],
            "effect": float(pearson_r),
            "effect_size": float(pearson_r),
            "description": (
                f"Direct causal edge: {source} -> {target} "
                f"(r={pearson_r:.3f}, type={ptype})"
            ),
        }

        if tier == "novel":
            pattern["difficulty"] = _assign_difficulty(
                graph, target, pearson_r, n_parents
            )

        # Add type-specific fields for interaction edges
        if ptype == "two_way_interaction":
            pattern["predictors"] = [source, co_parent]
            pattern["outcome"] = target
            pattern["interaction_effect"] = abs(pearson_r)

        patterns.append(pattern)

    return patterns


def _test_interaction(df: pd.DataFrame, var1: str, var2: str, target: str) -> bool:
    """Test if var1*var2 interaction is significant for predicting target."""
    try:
        from numpy.linalg import lstsq

        valid = df[[var1, var2, target]].dropna()
        if len(valid) < 30:
            return False

        x1 = valid[var1].values
        x2 = valid[var2].values
        y = valid[target].values

        # Standardize
        x1z = (x1 - x1.mean()) / max(x1.std(), 1e-10)
        x2z = (x2 - x2.mean()) / max(x2.std(), 1e-10)

        # Main effects model
        X_main = np.column_stack([np.ones(len(y)), x1z, x2z])
        coef_main, resid_main, _, _ = lstsq(X_main, y, rcond=None)
        ss_resid_main = np.sum((y - X_main @ coef_main) ** 2)

        # Interaction model
        interaction = x1z * x2z
        X_inter = np.column_stack([np.ones(len(y)), x1z, x2z, interaction])
        coef_inter, resid_inter, _, _ = lstsq(X_inter, y, rcond=None)
        ss_resid_inter = np.sum((y - X_inter @ coef_inter) ** 2)

        # F-test for interaction term
        n = len(y)
        df_num = 1  # 1 extra parameter
        df_denom = n - 4  # interaction model has 4 params
        if df_denom <= 0 or ss_resid_inter < 1e-10:
            return False

        f_stat = ((ss_resid_main - ss_resid_inter) / df_num) / (
            ss_resid_inter / df_denom
        )
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        return p_value < 0.01  # Significant interaction

    except Exception:
        return False


# =============================================================================
# Surface correlations
# =============================================================================


def _extract_surface_correlations(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    available_vars: list[str],
    threshold: float,
    direct_patterns: list[dict],
) -> list[dict[str, Any]]:
    """Compute pairwise correlations and emit obvious-tier patterns for strong non-edge ones."""
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
        if len(pvars) == 2:
            covered_pairs.add(frozenset(pvars))

    # Get direct edge pairs
    edge_pairs = {frozenset([s, t]) for s, t in graph.edges()}

    patterns = []
    for v1, v2 in combinations(available_vars, 2):
        pair = frozenset([v1, v2])
        if pair in covered_pairs or pair in edge_pairs:
            continue

        s1 = df[v1]
        s2 = df[v2]
        if s1.std() < 1e-10 or s2.std() < 1e-10:
            continue

        r = float(s1.corr(s2))
        if abs(r) >= threshold:
            patterns.append(
                {
                    "id": f"surface_{v1}_{v2}",
                    "type": "correlation",
                    "tier": "obvious",
                    "description": f"Surface correlation between {v1} and {v2} (r={r:.2f})",
                    "variables": [v1, v2],
                    "effect": r,
                    "effect_size": r,
                }
            )

    return patterns


# =============================================================================
# Mediation chains
# =============================================================================


def _extract_mediation_chains(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    available_vars: list[str],
    prior_chains: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    """Find all transitive paths of length >= 3 and emit mediation_chain patterns."""
    patterns = []
    seen_chains: set[tuple[str, ...]] = set()
    available_set = set(available_vars)

    # Find roots (nodes with no parents in the graph)
    roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]

    for start in roots:
        if start not in available_set:
            continue
        for end in available_set:
            if end == start:
                continue
            try:
                all_paths = list(nx.all_simple_paths(graph, start, end, cutoff=5))
            except nx.NetworkXError:
                continue

            for path in all_paths:
                if len(path) < 3:
                    continue
                # Ensure all nodes are available
                if not all(n in available_set for n in path):
                    continue
                chain_key = tuple(path)
                if chain_key in seen_chains:
                    continue
                seen_chains.add(chain_key)

                # Compute per-step correlations as proxy for direct effects
                direct_effects = []
                for i in range(len(path) - 1):
                    src, tgt = path[i], path[i + 1]
                    s1, s2 = df[src], df[tgt]
                    if s1.std() < 1e-10 or s2.std() < 1e-10:
                        direct_effects.append(0.0)
                    else:
                        direct_effects.append(float(s1.corr(s2)))

                # Indirect effect = product of per-step correlations
                indirect = 1.0
                for e in direct_effects:
                    indirect *= e

                # Verify chain manifests in data
                s_start = df[path[0]]
                s_end = df[path[-1]]
                if s_start.std() < 1e-10 or s_end.std() < 1e-10:
                    continue
                end_to_end_r = float(s_start.corr(s_end))
                if abs(end_to_end_r) < 0.05:
                    continue

                is_prior = chain_key in prior_chains
                difficulty = "moderate" if len(path) <= 3 else "complex"

                patterns.append(
                    {
                        "id": f"chain_{'_'.join(path)}",
                        "type": "mediation_chain",
                        "tier": "prior_knowledge" if is_prior else "novel",
                        "difficulty": difficulty,
                        "description": (
                            f"Mediation chain: {' -> '.join(path)} "
                            f"(indirect effect = {indirect:.3f})"
                        ),
                        "chain": list(path),
                        "direct_effects": direct_effects,
                        "indirect_effect": float(indirect),
                    }
                )

    return patterns


# =============================================================================
# Confound / fork structures
# =============================================================================


def _extract_confound_structures(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    available_vars: list[str],
) -> list[dict[str, Any]]:
    """Find fork patterns (common cause) where children are spuriously correlated."""
    patterns = []
    available_set = set(available_vars)

    for var_name in graph.nodes():
        if var_name not in available_set:
            continue
        children = [c for c in graph.successors(var_name) if c in available_set]
        if len(children) < 2:
            continue

        for c1, c2 in combinations(children, 2):
            # Check that c1 and c2 don't have a direct edge
            if graph.has_edge(c1, c2) or graph.has_edge(c2, c1):
                continue

            s1 = df[c1]
            s2 = df[c2]
            if s1.std() < 1e-10 or s2.std() < 1e-10:
                continue

            # Check for detectable spurious correlation
            r = float(s1.corr(s2))
            if abs(r) < 0.1:
                continue

            # Compute partial correlation controlling for common parent
            partial_r = _compute_partial_correlation(df, c1, c2, [var_name])

            patterns.append(
                {
                    "id": f"confound_{var_name}_{c1}_{c2}",
                    "type": "conditional_confound",
                    "tier": "novel",
                    "difficulty": "moderate",
                    "description": (
                        f"Confound: {c1} and {c2} appear correlated (r={r:.2f}) "
                        f"due to common cause {var_name} (partial r={partial_r:.2f})"
                    ),
                    "apparent_vars": [c1, c2],
                    "apparent_correlation": [c1, c2],
                    "confound": var_name,
                    "apparent_effect": float(r),
                    "true_effect": float(partial_r),
                }
            )

        # Latent variable pattern: common cause with 3+ available children
        if len(children) >= 3:
            patterns.append(
                {
                    "id": f"latent_{var_name}",
                    "type": "latent_variable",
                    "tier": "novel",
                    "difficulty": "complex",
                    "description": (
                        f"Latent variable: {', '.join(children)} "
                        f"correlate due to common factor {var_name}"
                    ),
                    "observed_vars": children,
                    "latent_var": var_name,
                }
            )

    return patterns


# =============================================================================
# Interaction patterns
# =============================================================================


def _extract_interaction_patterns(
    graph: nx.DiGraph,
    df: pd.DataFrame,
    available_vars: list[str],
) -> list[dict[str, Any]]:
    """
    For nodes with >= 2 parents, test interaction via regression F-test.
    Only emit if not already captured as a direct edge interaction.
    """
    patterns = []
    available_set = set(available_vars)
    seen_interactions: set[frozenset[str]] = set()

    for node in graph.nodes():
        if node not in available_set:
            continue
        parents = [p for p in graph.predecessors(node) if p in available_set]
        if len(parents) < 2:
            continue

        for p1, p2 in combinations(parents, 2):
            key = frozenset([p1, p2, node])
            if key in seen_interactions:
                continue

            if _test_interaction(df, p1, p2, node):
                seen_interactions.add(key)
                patterns.append(
                    {
                        "id": f"interaction_{p1}_{p2}_{node}",
                        "type": "two_way_interaction",
                        "tier": "novel",
                        "difficulty": "moderate",
                        "description": (
                            f"Interaction: {p1} x {p2} -> {node} "
                            f"(significant interaction term)"
                        ),
                        "predictors": [p1, p2],
                        "outcome": node,
                    }
                )

    return patterns
