"""
World simulation engine for causal DAG benchmarks.

Walks the DAG in topological order, sampling exogenous variables from their
distributions and applying causal rules additively to derive downstream variables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.benchmark.rules import CausalDAG, RuleSet, VariableSpec


def _sample_exogenous(spec: VariableSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n values from the specified distribution."""
    params = spec.distribution_params
    dist = spec.distribution

    if dist == "normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 1.0)
        return rng.normal(mean, std, size=n)
    elif dist == "uniform":
        low = params.get("low", 0.0)
        high = params.get("high", 1.0)
        return rng.uniform(low, high, size=n)
    elif dist == "lognormal":
        mean = params.get("mean", 0.0)
        sigma = params.get("sigma", 1.0)
        return rng.lognormal(mean, sigma, size=n)
    elif dist == "exponential":
        scale = params.get("scale", 1.0)
        return rng.exponential(scale, size=n)
    elif dist == "poisson":
        lam = params.get("lam", 5.0)
        return rng.poisson(lam, size=n).astype(float)
    elif dist == "bernoulli":
        p = params.get("p", 0.5)
        return rng.binomial(1, p, size=n).astype(float)
    elif dist == "categorical":
        categories = params.get("categories", [0, 1, 2])
        probs = params.get("probs", None)
        return rng.choice(categories, size=n, p=probs).astype(float)
    else:
        raise ValueError(f"Unknown distribution: {dist}")


def simulate_world(ruleset: RuleSet) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Simulate data from a causal DAG.

    Algorithm:
    1. Topological sort of DAG variables
    2. For each variable in order:
       - Exogenous: sample from distribution
       - Derived: initialize at base_value, apply all incoming rules additively,
         add observation noise, clip
    3. Build DataFrame with dtype conversion

    Args:
        ruleset: RuleSet containing DAG and generation parameters

    Returns:
        (DataFrame with all variables, dict of raw numpy arrays before dtype conversion)
    """
    dag = ruleset.dag
    n = ruleset.n_samples
    rng = np.random.default_rng(ruleset.seed)
    noise_scale = ruleset.global_noise_scale

    order = dag.topological_order()
    world_state: dict[str, np.ndarray] = {}

    for var_name in order:
        spec = dag.variables[var_name]

        if spec.var_type == "exogenous":
            values = _sample_exogenous(spec, n, rng)
        else:
            # Initialize derived variable
            values = np.full(n, spec.base_value, dtype=float)
            # Add small base noise so std > 0 for first rule
            values = values + rng.normal(0, 0.1 * noise_scale, size=n)

            # Apply all incoming rules additively
            rules = dag.get_rules_for_target(var_name)
            for rule in rules:
                source_vals = {src: world_state[src] for src in rule.sources}
                values = rule.compute(source_vals, values, rng)

        # Add observation noise (doesn't propagate — only affects final data)
        if spec.observation_noise_std > 0:
            obs_noise = rng.normal(0, spec.observation_noise_std * noise_scale, size=n)
            values = values + obs_noise

        # Clip
        if spec.clip_min is not None or spec.clip_max is not None:
            values = np.clip(
                values,
                spec.clip_min if spec.clip_min is not None else -np.inf,
                spec.clip_max if spec.clip_max is not None else np.inf,
            )

        world_state[var_name] = values

    # Build DataFrame with dtype conversion
    df_data = {}
    for var_name in order:
        spec = dag.variables[var_name]
        arr = world_state[var_name]

        if spec.dtype == "int":
            df_data[var_name] = arr.astype(int)
        elif spec.dtype == "bool":
            df_data[var_name] = arr > 0.5
        elif spec.dtype == "category":
            df_data[var_name] = pd.Categorical(arr.astype(int))
        else:
            df_data[var_name] = arr

    df = pd.DataFrame(df_data)
    return df, world_state


def add_observation_columns(
    df: pd.DataFrame, n: int, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Add standard observation metadata columns for format compatibility
    with BenchmarkDatasetAdapter.

    Adds: observer_id, observer_experience, timestamp, season.
    """
    df = df.copy()
    n_observers = max(5, n // 200)
    df["observer_id"] = rng.integers(0, n_observers, size=n)
    df["observer_experience"] = rng.uniform(0, 20, size=n).round(1)
    df["timestamp"] = pd.date_range("2020-01-01", periods=n, freq="h")
    seasons = ["winter", "spring", "summer", "fall"]
    df["season"] = [seasons[i % 4] for i in range(n)]
    return df
