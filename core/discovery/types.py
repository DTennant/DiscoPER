# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Any
import dataclasses
import uuid
import json


Serializable = Union[str, int, float, bool, None, dict[str, Any], list[Any]]


@dataclasses.dataclass
class Hypothesis:
    """A machine-testable proposal about patterns in data."""

    id: str
    statement: str  # Natural language hypothesis statement
    scope: dict[
        str, Any
    ]  # Structured filters (e.g., {"taxa": ["species_0"], "elevation": [100, 500]})
    variables: list[str]  # Named list of variables/features referenced
    target: Optional[str] = None  # What is being predicted/explained
    expected_direction: Optional[str] = (
        None  # e.g., "positive", "negative", "difference", "monotonic", "cluster-separation"
    )
    test_plan: Optional[dict[str, Any]] = (
        None  # Structured plan for testing (tool name, args, etc.)
    )
    priors: Optional[dict[str, float]] = None  # Optional numeric prior expectations
    risk_flags: Optional[list[str]] = (
        None  # Confounds to check (e.g., ["photographer_bias", "label_leakage"])
    )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Hypothesis":
        """Create from dict."""
        return cls(**data)

    @classmethod
    def create(
        cls, statement: str, scope: dict, variables: list[str], **kwargs
    ) -> "Hypothesis":
        """Factory method to create hypothesis with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            statement=statement,
            scope=scope,
            variables=variables,
            **kwargs,
        )


@dataclasses.dataclass
class Experiment:
    """A concrete, reproducible test of a hypothesis."""

    id: str
    hypothesis_id: str
    dataset_slice_spec: dict[
        str, Any
    ]  # SQL/pandas filter expression or structured spec
    feature_spec: dict[str, Any]  # What embeddings/features to load/compute
    method: str  # e.g., "spearman", "logreg", "xgboost", "permutation_test"
    controls: Optional[dict[str, Any]] = (
        None  # Stratify by, include covariates, matched sampling
    )
    split_spec: Optional[dict[str, Any]] = None  # Train/val/test or bootstrap
    metrics_to_compute: Optional[list[str]] = (
        None  # Effect size, CI, p/FDR, predictive AUC/acc
    )
    budget: Optional[dict[str, int]] = (
        None  # max rows/images, max tool calls, max runtime
    )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Experiment":
        """Create from dict."""
        return cls(**data)

    @classmethod
    def create(
        cls,
        hypothesis_id: str,
        dataset_slice_spec: dict,
        feature_spec: dict,
        method: str,
        **kwargs,
    ) -> "Experiment":
        """Factory method to create experiment with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            dataset_slice_spec=dataset_slice_spec,
            feature_spec=feature_spec,
            method=method,
            **kwargs,
        )


@dataclasses.dataclass
class Result:
    """Outcome of running an experiment."""

    experiment_id: str
    status: str  # "success", "fail", "invalid"
    numbers: dict[str, float]  # Computed metrics (effect size, p-value, AUC, etc.)
    artifacts: Optional[dict[str, str]] = None  # Paths to plots/tables
    notes: Optional[str] = None  # Short textual summary
    warnings: Optional[list[str]] = None  # Confound alerts, low sample size, etc.
    repro: Optional[dict[str, Any]] = (
        None  # Exact code hash, random seeds, data version, tool versions
    )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Result":
        """Create from dict."""
        return cls(**data)


@dataclasses.dataclass
class Insight:
    """A validated (or invalidated) discovery-as-data object (Base Insight about raw data)."""

    id: str
    hypothesis_id: str
    experiment_id: str
    verdict: str  # "supported", "rejected", "inconclusive"
    claim: str  # Canonicalized form of statement (structured + NL)
    strength: float  # Scalar score (combine effect size + predictive gain + robustness)
    novelty_proxy: Optional[float] = None  # Scalar (surprisal, redundancy penalty)
    generalization: Optional[dict[str, Any]] = (
        None  # Which slices it held on (cross-validation / temporal split)
    )
    edges: Optional[list[dict[str, str]]] = (
        None  # Relations to other insights (supports/contradicts/refines/generalizes)
    )
    variables: Optional[list[str]] = None  # Variables tested in this insight
    embedding: Optional[list[float]] = (
        None  # Vector embedding of the claim for retrieval/clustering
    )
    reward: Optional[float] = None  # Composite reward score (from reward_weights)
    is_meta: bool = False  # Whether this insight was generated from meta-discovery

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Insight":
        """Create from dict."""
        return cls(**data)

    @classmethod
    def create(
        cls,
        hypothesis_id: str,
        experiment_id: str,
        verdict: str,
        claim: str,
        strength: float,
        **kwargs,
    ) -> "Insight":
        """Factory method to create insight with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            verdict=verdict,
            claim=claim,
            strength=strength,
            **kwargs,
        )


@dataclasses.dataclass
class MetaInsight:
    """
    A meta-level insight about patterns across base insights.

    Meta-insights are ACTIONABLE observations that guide future hypothesis generation.
    They are NOT tested experimentally - they are derived from analyzing base insights.

    Examples:
    - "Age appears as a confound in 3/5 metabolic findings → control for age in future tests"
    - "BMI, triglycerides, cholesterol cluster together → test interaction effects"
    - "Correlation hypotheses have 80% success rate → prioritize correlation tests"
    """

    id: str
    meta_type: str  # "confound_pattern", "variable_cluster", "success_pattern", "contradiction", "gap"
    observation: str  # What pattern was observed across insights
    actionable_recommendation: str  # Specific guidance for future hypothesis generation
    affected_variables: list[str]  # Variables this meta-insight relates to
    source_insight_ids: list[str]  # IDs of base insights that led to this meta-insight
    confidence: float  # How confident we are in this meta-insight (0-1)
    priority: str  # "high", "medium", "low" - how important to act on this

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MetaInsight":
        """Create from dict."""
        return cls(**data)

    @classmethod
    def create(
        cls,
        meta_type: str,
        observation: str,
        actionable_recommendation: str,
        affected_variables: list[str],
        source_insight_ids: list[str],
        confidence: float = 0.8,
        priority: str = "medium",
    ) -> "MetaInsight":
        """Factory method to create meta-insight with auto-generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            meta_type=meta_type,
            observation=observation,
            actionable_recommendation=actionable_recommendation,
            affected_variables=affected_variables,
            source_insight_ids=source_insight_ids,
            confidence=confidence,
            priority=priority,
        )
