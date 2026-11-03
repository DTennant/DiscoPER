"""
Rule engine and causal DAG for Game-of-Life-style benchmark generation.

Defines causal rules that compose additively to produce emergent patterns.
The CausalDAG manages the dependency graph and provides topological ordering
for correct simulation order.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# =============================================================================
# Variable specification
# =============================================================================

@dataclass
class VariableSpec:
    """Specification for a single variable in the causal DAG."""
    name: str
    var_type: str  # "exogenous" or "derived"
    distribution: str = "normal"  # normal, uniform, lognormal, exponential, poisson, bernoulli, categorical
    distribution_params: dict[str, Any] = field(default_factory=dict)
    base_value: float = 0.0  # Initial value for derived variables before rules apply
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    observation_noise_std: float = 0.0  # Noise added only to final observations
    dtype: str = "float"  # float, int, bool, category
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "var_type": self.var_type,
            "distribution": self.distribution,
            "base_value": self.base_value,
            "dtype": self.dtype,
            "description": self.description,
        }
        if self.distribution_params:
            d["distribution_params"] = self.distribution_params
        if self.clip_min is not None:
            d["clip_min"] = self.clip_min
        if self.clip_max is not None:
            d["clip_max"] = self.clip_max
        if self.observation_noise_std > 0:
            d["observation_noise_std"] = self.observation_noise_std
        return d


# =============================================================================
# Causal rule base and subclasses
# =============================================================================

@dataclass
class CausalRule:
    """Base class for causal rules in the DAG."""
    id: str = ""
    rule_type: str = ""
    sources: list[str] = field(default_factory=list)
    target: str = ""
    strength: float = 0.0
    noise_std: float = 0.05  # Propagation noise (feeds downstream)
    description: str = ""
    ecological_principle: str = ""

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply this rule and return the modified target array."""
        raise NotImplementedError

    def _standardize(self, arr: np.ndarray) -> np.ndarray:
        """Standardize array to zero mean, unit variance."""
        std = arr.std()
        if std < 1e-10:
            return np.zeros_like(arr)
        return (arr - arr.mean()) / std

    def _add_noise(self, arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Add propagation noise scaled to target std."""
        if self.noise_std <= 0:
            return arr
        std = arr.std()
        if std < 1e-10:
            std = 1.0
        return arr + rng.normal(0, self.noise_std * std, size=len(arr))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rule_type": self.rule_type,
            "sources": self.sources,
            "target": self.target,
            "strength": self.strength,
            "noise_std": self.noise_std,
            "description": self.description,
            "ecological_principle": self.ecological_principle,
        }


@dataclass
class LinearCausalRule(CausalRule):
    """Linear causal effect: target += strength * standardize(source) + noise."""
    rule_type: str = "linear"

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        source = source_values[self.sources[0]]
        z = self._standardize(source)
        effect = self.strength * z
        result = current_target + effect
        return self._add_noise(result, rng)


@dataclass
class ThresholdCausalRule(CausalRule):
    """Effect only active above/below a threshold value."""
    rule_type: str = "threshold"
    threshold: float = 0.0
    above: bool = True  # If True, effect active above threshold; if False, below

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        source = source_values[self.sources[0]]
        if self.above:
            mask = source > self.threshold
        else:
            mask = source <= self.threshold
        z = self._standardize(source)
        effect = np.where(mask, self.strength * z, 0.0)
        result = current_target + effect
        return self._add_noise(result, rng)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["threshold"] = self.threshold
        d["above"] = self.above
        return d


@dataclass
class InteractionRule(CausalRule):
    """Multiplicative interaction: target += strength * source1 * source2."""
    rule_type: str = "interaction"

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z1 = self._standardize(source_values[self.sources[0]])
        z2 = self._standardize(source_values[self.sources[1]])
        effect = self.strength * z1 * z2
        result = current_target + effect
        return self._add_noise(result, rng)


@dataclass
class ModulationRule(CausalRule):
    """A third variable scales an existing effect between source and target.

    sources[0] = primary source, sources[1] = modulator.
    Effect: target += strength * standardize(source) * sigmoid(modulator).
    """
    rule_type: str = "modulation"
    modulator_center: float = 0.0  # Center of sigmoid for modulator
    modulator_scale: float = 1.0  # Steepness of sigmoid

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        z_source = self._standardize(source_values[self.sources[0]])
        modulator = source_values[self.sources[1]]
        # Sigmoid gating based on modulator
        gate = 1.0 / (1.0 + np.exp(-(modulator - self.modulator_center) / max(self.modulator_scale, 1e-6)))
        effect = self.strength * z_source * gate
        result = current_target + effect
        return self._add_noise(result, rng)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["modulator_center"] = self.modulator_center
        d["modulator_scale"] = self.modulator_scale
        return d


@dataclass
class CompetitionRule(CausalRule):
    """Negative coupling, optionally weakened by a modifier variable.

    sources[0] = competitor, sources[1] (optional) = modifier that weakens competition.
    """
    rule_type: str = "competition"
    modifier_weakening: float = 0.5  # How much modifier weakens competition (0=none, 1=full)

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        competitor = source_values[self.sources[0]]
        z_comp = self._standardize(competitor)

        # Base competition (negative effect)
        base_effect = -abs(self.strength) * z_comp

        if len(self.sources) > 1 and self.sources[1] in source_values:
            modifier = source_values[self.sources[1]]
            z_mod = self._standardize(modifier)
            # Positive modifier weakens competition
            weakening = self.modifier_weakening * np.clip(z_mod, 0, None)
            effect = base_effect * (1.0 - weakening)
        else:
            effect = base_effect

        result = current_target + effect
        return self._add_noise(result, rng)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["modifier_weakening"] = self.modifier_weakening
        return d


@dataclass
class SaturationRule(CausalRule):
    """Diminishing returns: target += strength * source / (1 + source/capacity)."""
    rule_type: str = "saturation"
    capacity: float = 1.0  # Carrying capacity parameter

    def compute(self, source_values: dict[str, np.ndarray],
                current_target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        source = source_values[self.sources[0]]
        # Shift to positive domain for saturation curve
        s_min = source.min()
        shifted = source - s_min + 0.01  # Ensure positive
        saturated = shifted / (1.0 + shifted / max(self.capacity, 1e-6))
        z_sat = self._standardize(saturated)
        effect = self.strength * z_sat
        result = current_target + effect
        return self._add_noise(result, rng)

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["capacity"] = self.capacity
        return d


# =============================================================================
# Causal DAG
# =============================================================================

class CausalDAG:
    """Directed acyclic graph of causal relationships between variables."""

    def __init__(self):
        self.variables: dict[str, VariableSpec] = {}
        self.rules: list[CausalRule] = []

    def add_variable(self, spec: VariableSpec) -> None:
        """Add a variable to the DAG."""
        self.variables[spec.name] = spec

    def add_rule(self, rule: CausalRule) -> None:
        """Add a causal rule, validating that referenced variables exist."""
        for src in rule.sources:
            if src not in self.variables:
                raise ValueError(f"Source variable '{src}' not in DAG (rule {rule.id})")
        if rule.target not in self.variables:
            raise ValueError(f"Target variable '{rule.target}' not in DAG (rule {rule.id})")
        if self.variables[rule.target].var_type == "exogenous":
            raise ValueError(
                f"Cannot add rule targeting exogenous variable '{rule.target}' (rule {rule.id})"
            )
        self.rules.append(rule)
        # Validate no cycle was introduced
        try:
            self.topological_order()
        except ValueError:
            self.rules.pop()
            raise ValueError(f"Adding rule {rule.id} would create a cycle in the DAG")

    def topological_order(self) -> list[str]:
        """Return variables in topological order using Kahn's algorithm."""
        # Build adjacency from rules
        in_degree: dict[str, int] = {name: 0 for name in self.variables}
        children: dict[str, set[str]] = {name: set() for name in self.variables}

        for rule in self.rules:
            for src in rule.sources:
                if rule.target not in children[src]:
                    children[src].add(rule.target)

        # Compute in-degree from unique edges
        for parent, kids in children.items():
            for child in kids:
                in_degree[child] += 1

        # But we need deduplicated edges, so recount properly
        edges: set[tuple[str, str]] = set()
        for rule in self.rules:
            for src in rule.sources:
                edges.add((src, rule.target))

        in_degree = {name: 0 for name in self.variables}
        for _, target in edges:
            in_degree[target] += 1

        queue = deque([v for v, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.variables):
            raise ValueError("Cycle detected in causal DAG")

        return order

    def get_parents(self, var_name: str) -> set[str]:
        """Get all direct parent variables."""
        parents = set()
        for rule in self.rules:
            if rule.target == var_name:
                parents.update(rule.sources)
        return parents

    def get_children(self, var_name: str) -> set[str]:
        """Get all direct child variables."""
        children = set()
        for rule in self.rules:
            if var_name in rule.sources:
                children.add(rule.target)
        return children

    def get_rules_for_target(self, var_name: str) -> list[CausalRule]:
        """Get all rules that affect a given target variable."""
        return [r for r in self.rules if r.target == var_name]

    def get_all_paths(self, source: str, target: str, max_depth: int = 6) -> list[list[str]]:
        """Enumerate all paths from source to target via DFS."""
        paths: list[list[str]] = []

        def dfs(current: str, path: list[str]):
            if len(path) > max_depth + 1:
                return
            if current == target and len(path) > 1:
                paths.append(list(path))
                return
            for child in self.get_children(current):
                if child not in path:  # Avoid cycles in path
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(source, [source])
        return paths

    def get_causal_depth(self, var_name: str) -> int:
        """Max depth from any exogenous root to this variable."""
        if self.variables[var_name].var_type == "exogenous":
            return 0
        parents = self.get_parents(var_name)
        if not parents:
            return 0
        return 1 + max(self.get_causal_depth(p) for p in parents)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "variables": {name: spec.to_dict() for name, spec in self.variables.items()},
            "rules": [r.to_dict() for r in self.rules],
        }


# =============================================================================
# RuleSet — named wrapper
# =============================================================================

@dataclass
class RuleSet:
    """Named wrapper for a CausalDAG with generation parameters."""
    name: str
    description: str
    dag: CausalDAG
    n_samples: int = 3000
    seed: int = 42
    global_noise_scale: float = 1.0
    correlation_threshold: float = 0.15  # Min |r| for surface correlations
    prior_rule_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "dag": self.dag.to_dict(),
            "n_samples": self.n_samples,
            "seed": self.seed,
            "global_noise_scale": self.global_noise_scale,
            "correlation_threshold": self.correlation_threshold,
            "prior_rule_ids": self.prior_rule_ids,
        }
