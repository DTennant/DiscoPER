# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pattern specification schema for synthetic benchmark datasets.

Defines all supported pattern types that can be embedded in synthetic data
for evaluating the discovery agent's ability to find novel insights.
"""

from typing import Optional, Union, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import json


class PatternTier(str, Enum):
    """Tier classification for patterns."""
    PRIOR_KNOWLEDGE = "prior_knowledge"  # Known facts, should NOT be rediscovered
    OBVIOUS = "obvious"  # Easy to find, not novel
    NOVEL = "novel"  # Should be discovered as novel


class PatternDifficulty(str, Enum):
    """Difficulty level for novel patterns."""
    SIMPLE = "simple"  # 1-2 features, conditional
    MODERATE = "moderate"  # 2-3 features, interactions
    COMPLEX = "complex"  # 3+ features, chains, hierarchies


class PatternType(str, Enum):
    """All supported pattern types."""
    # Simple patterns
    CORRELATION = "correlation"
    GROUP_DIFFERENCE = "group_difference"
    THRESHOLD = "threshold"
    SIMPSON_PARADOX = "simpson_paradox"
    
    # Moderate patterns
    TWO_WAY_INTERACTION = "two_way_interaction"
    CONDITIONAL_CONFOUND = "conditional_confound"
    MASKED_CORRELATION = "masked_correlation"
    
    # Complex patterns
    THREE_WAY_INTERACTION = "three_way_interaction"
    MEDIATION_CHAIN = "mediation_chain"
    HIERARCHICAL_EFFECT = "hierarchical_effect"
    LATENT_VARIABLE = "latent_variable"
    NONLINEAR_PIECEWISE = "nonlinear_piecewise"
    TEMPORAL_INTERACTION = "temporal_interaction"
    SPATIAL_CLUSTER = "spatial_cluster"
    COMPETITION_MASKING = "competition_masking"


@dataclass
class BasePattern:
    """Base class for all pattern specifications."""
    id: str
    tier: PatternTier
    type: PatternType
    description: str
    difficulty: Optional[PatternDifficulty] = None
    discovery_hints: Optional[list[str]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "id": self.id,
            "tier": self.tier.value,
            "type": self.type.value,
            "description": self.description,
        }
        if self.difficulty:
            result["difficulty"] = self.difficulty.value
        if self.discovery_hints:
            result["discovery_hints"] = self.discovery_hints
        return result


# =============================================================================
# SIMPLE PATTERNS
# =============================================================================

@dataclass
class CorrelationPattern(BasePattern):
    """Simple correlation between two variables."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    effect_size: float = 0.0  # Correlation coefficient (-1 to 1)
    direction: Literal["positive", "negative"] = "positive"
    
    def __post_init__(self):
        self.type = PatternType.CORRELATION
        if self.effect_size > 0:
            self.direction = "positive"
        elif self.effect_size < 0:
            self.direction = "negative"
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "effect_size": self.effect_size,
            "direction": self.direction,
        })
        return result


@dataclass
class GroupDifferencePattern(BasePattern):
    """Mean difference between groups."""
    variable: str = ""
    group_by: str = ""
    effect_size: float = 0.0  # Cohen's d or similar
    group_means: Optional[dict[str, float]] = None
    
    def __post_init__(self):
        self.type = PatternType.GROUP_DIFFERENCE
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variable": self.variable,
            "group_by": self.group_by,
            "effect_size": self.effect_size,
        })
        if self.group_means:
            result["group_means"] = self.group_means
        return result


@dataclass
class ThresholdPattern(BasePattern):
    """Relationship that changes at a threshold value."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    condition_var: Optional[str] = None  # Variable that determines threshold
    threshold: float = 0.0
    effect_below: float = 0.0
    effect_above: float = 0.0
    
    def __post_init__(self):
        self.type = PatternType.THRESHOLD
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "threshold": self.threshold,
            "effect_below": self.effect_below,
            "effect_above": self.effect_above,
        })
        if self.condition_var:
            result["condition_var"] = self.condition_var
        return result


@dataclass
class SimpsonParadoxPattern(BasePattern):
    """Correlation that reverses within subgroups."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    group_by: str = ""
    overall_effect: float = 0.0  # Marginal correlation
    within_effect: float = 0.0  # Correlation within each group (opposite sign)
    
    def __post_init__(self):
        self.type = PatternType.SIMPSON_PARADOX
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "group_by": self.group_by,
            "overall_effect": self.overall_effect,
            "within_effect": self.within_effect,
        })
        return result


# =============================================================================
# MODERATE PATTERNS
# =============================================================================

@dataclass
class TwoWayInteractionPattern(BasePattern):
    """Interaction effect between two predictors."""
    outcome: str = ""
    predictors: tuple[str, str] = field(default_factory=lambda: ("", ""))
    main_effects: tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    interaction_effect: float = 0.0
    subset: Optional[dict[str, Any]] = None  # Optional filter for subset
    
    def __post_init__(self):
        self.type = PatternType.TWO_WAY_INTERACTION
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "outcome": self.outcome,
            "predictors": list(self.predictors),
            "main_effects": list(self.main_effects),
            "interaction_effect": self.interaction_effect,
        })
        if self.subset:
            result["subset"] = self.subset
        return result


@dataclass
class ConditionalConfoundPattern(BasePattern):
    """Spurious correlation due to confounding, only in certain conditions."""
    apparent_vars: tuple[str, str] = field(default_factory=lambda: ("", ""))
    confound: str = ""
    apparent_effect: float = 0.0  # Observed correlation (spurious)
    true_effect: float = 0.0  # True correlation after controlling
    condition_col: Optional[str] = None
    condition_val: Optional[Any] = None
    
    def __post_init__(self):
        self.type = PatternType.CONDITIONAL_CONFOUND
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "apparent_vars": list(self.apparent_vars),
            "confound": self.confound,
            "apparent_effect": self.apparent_effect,
            "true_effect": self.true_effect,
        })
        if self.condition_col:
            result["condition_col"] = self.condition_col
            result["condition_val"] = self.condition_val
        return result


@dataclass
class MaskedCorrelationPattern(BasePattern):
    """Correlation hidden by a confounder (suppression effect)."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    confounder: str = ""
    marginal_effect: float = 0.0  # Observed (weak) correlation
    partial_effect: float = 0.0  # True correlation after controlling
    
    def __post_init__(self):
        self.type = PatternType.MASKED_CORRELATION
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "confounder": self.confounder,
            "marginal_effect": self.marginal_effect,
            "partial_effect": self.partial_effect,
        })
        return result


# =============================================================================
# COMPLEX PATTERNS
# =============================================================================

@dataclass
class ThreeWayInteractionPattern(BasePattern):
    """Three-way interaction effect."""
    outcome: str = ""
    predictors: tuple[str, str, str] = field(default_factory=lambda: ("", "", ""))
    main_effects: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))
    two_way_effects: Optional[dict[str, float]] = None  # e.g., {"x1_x2": 0.2, "x1_x3": 0.1}
    three_way_effect: float = 0.0
    
    def __post_init__(self):
        self.type = PatternType.THREE_WAY_INTERACTION
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "outcome": self.outcome,
            "predictors": list(self.predictors),
            "main_effects": list(self.main_effects),
            "three_way_effect": self.three_way_effect,
        })
        if self.two_way_effects:
            result["two_way_effects"] = self.two_way_effects
        return result


@dataclass
class MediationChainPattern(BasePattern):
    """Causal chain A -> B -> C -> D."""
    chain: list[str] = field(default_factory=list)  # Ordered list of variables
    direct_effects: list[float] = field(default_factory=list)  # Effects between consecutive vars
    indirect_effect: Optional[float] = None  # Product of direct effects
    
    def __post_init__(self):
        self.type = PatternType.MEDIATION_CHAIN
        if self.direct_effects and self.indirect_effect is None:
            # Calculate indirect effect as product
            import math
            self.indirect_effect = math.prod(self.direct_effects)
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "chain": self.chain,
            "direct_effects": self.direct_effects,
            "indirect_effect": self.indirect_effect,
        })
        return result


@dataclass
class HierarchicalEffectPattern(BasePattern):
    """Effect that varies by hierarchical grouping (e.g., taxonomic level)."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    hierarchy_col: str = ""
    effects_by_level: dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.type = PatternType.HIERARCHICAL_EFFECT
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "hierarchy_col": self.hierarchy_col,
            "effects_by_level": self.effects_by_level,
        })
        return result


@dataclass
class LatentVariablePattern(BasePattern):
    """Multiple observed variables driven by a hidden common cause."""
    observed_vars: list[str] = field(default_factory=list)
    latent_var: str = ""
    induced_correlations: dict[str, float] = field(default_factory=dict)  # var1-var2: corr
    
    def __post_init__(self):
        self.type = PatternType.LATENT_VARIABLE
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "observed_vars": self.observed_vars,
            "latent_var": self.latent_var,
            "induced_correlations": self.induced_correlations,
        })
        return result


@dataclass
class NonlinearPiecewisePattern(BasePattern):
    """Piecewise linear relationship (e.g., U-shaped)."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    breakpoints: list[float] = field(default_factory=list)
    slopes: list[float] = field(default_factory=list)  # len = len(breakpoints) + 1
    
    def __post_init__(self):
        self.type = PatternType.NONLINEAR_PIECEWISE
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "breakpoints": self.breakpoints,
            "slopes": self.slopes,
        })
        return result


@dataclass
class TemporalInteractionPattern(BasePattern):
    """Effect that varies by time/season."""
    variables: tuple[str, str] = field(default_factory=lambda: ("", ""))
    time_col: str = ""
    effects_by_time: dict[str, float] = field(default_factory=dict)  # e.g., {"winter": -0.5}
    
    def __post_init__(self):
        self.type = PatternType.TEMPORAL_INTERACTION
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": list(self.variables),
            "time_col": self.time_col,
            "effects_by_time": self.effects_by_time,
        })
        return result


@dataclass
class SpatialClusterPattern(BasePattern):
    """Region-specific co-occurrence patterns."""
    variables: list[str] = field(default_factory=list)  # Species presence columns
    region_col: str = ""
    cluster_pattern: dict[str, list[int]] = field(default_factory=dict)  # region: [1,1,0] presence
    
    def __post_init__(self):
        self.type = PatternType.SPATIAL_CLUSTER
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "variables": self.variables,
            "region_col": self.region_col,
            "cluster_pattern": self.cluster_pattern,
        })
        return result


@dataclass
class CompetitionMaskingPattern(BasePattern):
    """Competition that disappears in presence of keystone species."""
    competing_species: tuple[str, str] = field(default_factory=lambda: ("", ""))
    keystone: str = ""
    effect_without_keystone: float = 0.0  # Negative = competition
    effect_with_keystone: float = 0.0  # Neutral or positive
    
    def __post_init__(self):
        self.type = PatternType.COMPETITION_MASKING
    
    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update({
            "competing_species": list(self.competing_species),
            "keystone": self.keystone,
            "effect_without_keystone": self.effect_without_keystone,
            "effect_with_keystone": self.effect_with_keystone,
        })
        return result


# =============================================================================
# PATTERN FACTORY
# =============================================================================

# Type alias for any pattern
Pattern = Union[
    CorrelationPattern,
    GroupDifferencePattern,
    ThresholdPattern,
    SimpsonParadoxPattern,
    TwoWayInteractionPattern,
    ConditionalConfoundPattern,
    MaskedCorrelationPattern,
    ThreeWayInteractionPattern,
    MediationChainPattern,
    HierarchicalEffectPattern,
    LatentVariablePattern,
    NonlinearPiecewisePattern,
    TemporalInteractionPattern,
    SpatialClusterPattern,
    CompetitionMaskingPattern,
]


PATTERN_CLASS_MAP: dict[PatternType, type] = {
    PatternType.CORRELATION: CorrelationPattern,
    PatternType.GROUP_DIFFERENCE: GroupDifferencePattern,
    PatternType.THRESHOLD: ThresholdPattern,
    PatternType.SIMPSON_PARADOX: SimpsonParadoxPattern,
    PatternType.TWO_WAY_INTERACTION: TwoWayInteractionPattern,
    PatternType.CONDITIONAL_CONFOUND: ConditionalConfoundPattern,
    PatternType.MASKED_CORRELATION: MaskedCorrelationPattern,
    PatternType.THREE_WAY_INTERACTION: ThreeWayInteractionPattern,
    PatternType.MEDIATION_CHAIN: MediationChainPattern,
    PatternType.HIERARCHICAL_EFFECT: HierarchicalEffectPattern,
    PatternType.LATENT_VARIABLE: LatentVariablePattern,
    PatternType.NONLINEAR_PIECEWISE: NonlinearPiecewisePattern,
    PatternType.TEMPORAL_INTERACTION: TemporalInteractionPattern,
    PatternType.SPATIAL_CLUSTER: SpatialClusterPattern,
    PatternType.COMPETITION_MASKING: CompetitionMaskingPattern,
}


def pattern_from_dict(data: dict[str, Any]) -> Pattern:
    """
    Create a pattern object from a dictionary (e.g., from YAML config).
    
    Args:
        data: Dictionary with pattern specification
        
    Returns:
        Pattern object of the appropriate type
    """
    pattern_type = PatternType(data["type"])
    pattern_class = PATTERN_CLASS_MAP[pattern_type]
    
    # Convert tier and difficulty if present
    tier = PatternTier(data.get("tier", "novel"))
    difficulty = None
    if "difficulty" in data:
        difficulty = PatternDifficulty(data["difficulty"])
    
    # Build kwargs for the pattern class
    kwargs = {
        "id": data["id"],
        "tier": tier,
        "type": pattern_type,
        "description": data.get("description", ""),
        "difficulty": difficulty,
        "discovery_hints": data.get("discovery_hints"),
    }
    
    # Add type-specific fields
    if pattern_type == PatternType.CORRELATION:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["effect_size"] = data.get("effect", data.get("effect_size", 0.0))
        
    elif pattern_type == PatternType.GROUP_DIFFERENCE:
        kwargs["variable"] = data["variable"]
        kwargs["group_by"] = data["group_by"]
        kwargs["effect_size"] = data.get("effect", data.get("effect_size", 0.0))
        kwargs["group_means"] = data.get("group_means")
        
    elif pattern_type == PatternType.THRESHOLD:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["condition_var"] = data.get("condition_var")
        kwargs["threshold"] = data["threshold"]
        kwargs["effect_below"] = data["effect_below"]
        kwargs["effect_above"] = data["effect_above"]
        
    elif pattern_type == PatternType.SIMPSON_PARADOX:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["group_by"] = data["group_by"]
        kwargs["overall_effect"] = data["overall_effect"]
        kwargs["within_effect"] = data["within_effect"]
        
    elif pattern_type == PatternType.TWO_WAY_INTERACTION:
        kwargs["outcome"] = data["outcome"]
        kwargs["predictors"] = tuple(data["predictors"])
        kwargs["main_effects"] = tuple(data.get("main_effects", [0.0, 0.0]))
        kwargs["interaction_effect"] = data["interaction_effect"]
        kwargs["subset"] = data.get("subset")
        
    elif pattern_type == PatternType.CONDITIONAL_CONFOUND:
        kwargs["apparent_vars"] = tuple(data.get("apparent_correlation", data.get("apparent_vars", ["", ""])))
        kwargs["confound"] = data["confound"]
        kwargs["apparent_effect"] = data["apparent_effect"]
        kwargs["true_effect"] = data["true_effect"]
        kwargs["condition_col"] = data.get("condition_col")
        kwargs["condition_val"] = data.get("condition_val")
        
    elif pattern_type == PatternType.MASKED_CORRELATION:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["confounder"] = data["confounder"]
        kwargs["marginal_effect"] = data["marginal_effect"]
        kwargs["partial_effect"] = data["partial_effect"]
        
    elif pattern_type == PatternType.THREE_WAY_INTERACTION:
        kwargs["outcome"] = data["outcome"]
        kwargs["predictors"] = tuple(data["predictors"])
        effects = data.get("effects", {})
        kwargs["main_effects"] = tuple(effects.get("main", [0.0, 0.0, 0.0]))
        kwargs["two_way_effects"] = effects.get("two_way")
        kwargs["three_way_effect"] = effects.get("three_way", data.get("three_way_effect", 0.0))
        
    elif pattern_type == PatternType.MEDIATION_CHAIN:
        kwargs["chain"] = data["chain"]
        kwargs["direct_effects"] = data["direct_effects"]
        kwargs["indirect_effect"] = data.get("indirect_effect")
        
    elif pattern_type == PatternType.HIERARCHICAL_EFFECT:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["hierarchy_col"] = data["hierarchy_col"]
        kwargs["effects_by_level"] = data["effects_by_level"]
        
    elif pattern_type == PatternType.LATENT_VARIABLE:
        kwargs["observed_vars"] = data["observed_vars"]
        kwargs["latent_var"] = data.get("latent_name", data.get("latent_var", ""))
        # Handle correlation matrix or dict
        if "correlations_induced" in data:
            corr = data["correlations_induced"]
            if isinstance(corr, list):
                # Convert matrix to dict
                obs_vars = kwargs["observed_vars"]
                corr_dict = {}
                for i in range(len(obs_vars)):
                    for j in range(i + 1, len(obs_vars)):
                        corr_dict[f"{obs_vars[i]}-{obs_vars[j]}"] = corr[i][j]
                kwargs["induced_correlations"] = corr_dict
            else:
                kwargs["induced_correlations"] = corr
        else:
            kwargs["induced_correlations"] = data.get("induced_correlations", {})
        
    elif pattern_type == PatternType.NONLINEAR_PIECEWISE:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["breakpoints"] = data["breakpoints"]
        kwargs["slopes"] = data["slopes"]
        
    elif pattern_type == PatternType.TEMPORAL_INTERACTION:
        kwargs["variables"] = tuple(data["variables"])
        kwargs["time_col"] = data["time_col"]
        kwargs["effects_by_time"] = data.get("effects", data.get("effects_by_time", {}))
        
    elif pattern_type == PatternType.SPATIAL_CLUSTER:
        kwargs["variables"] = data["variables"]
        kwargs["region_col"] = data["region_col"]
        kwargs["cluster_pattern"] = data["cluster_pattern"]
        
    elif pattern_type == PatternType.COMPETITION_MASKING:
        kwargs["competing_species"] = tuple(data["competing_species"])
        kwargs["keystone"] = data["keystone"]
        kwargs["effect_without_keystone"] = data["effect_without_keystone"]
        kwargs["effect_with_keystone"] = data["effect_with_keystone"]
    
    return pattern_class(**kwargs)


@dataclass
class BenchmarkSpec:
    """Complete specification for a benchmark dataset."""
    name: str
    seed: int
    n_samples: int
    n_species: int = 10
    n_regions: int = 3
    
    prior_knowledge: list[Pattern] = field(default_factory=list)
    obvious_patterns: list[Pattern] = field(default_factory=list)
    novel_patterns: list[Pattern] = field(default_factory=list)
    
    include_prior_in_prompt: bool = True
    prior_prompt_detail: Literal["full", "summary", "none"] = "full"
    
    def all_patterns(self) -> list[Pattern]:
        """Get all patterns across all tiers."""
        return self.prior_knowledge + self.obvious_patterns + self.novel_patterns
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "dataset_id": self.name,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "n_species": self.n_species,
            "n_regions": self.n_regions,
            "patterns": [p.to_dict() for p in self.all_patterns()],
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
                }
            },
            "include_prior_in_prompt": self.include_prior_in_prompt,
            "prior_prompt_detail": self.prior_prompt_detail,
        }
    
    def to_json(self, path: str) -> None:
        """Save specification to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_yaml(cls, config: dict[str, Any]) -> "BenchmarkSpec":
        """Create from parsed YAML config."""
        benchmark = config.get("benchmark", config)
        
        # Parse patterns by tier
        prior_patterns = []
        for p in benchmark.get("prior_knowledge", []):
            p["tier"] = "prior_knowledge"
            prior_patterns.append(pattern_from_dict(p))
        
        obvious_patterns = []
        for p in benchmark.get("obvious_patterns", []):
            p["tier"] = "obvious"
            obvious_patterns.append(pattern_from_dict(p))
        
        novel_patterns = []
        for p in benchmark.get("novel_patterns", []):
            p["tier"] = "novel"
            novel_patterns.append(pattern_from_dict(p))
        
        return cls(
            name=benchmark.get("name", "unnamed_benchmark"),
            seed=benchmark.get("seed", 42),
            n_samples=benchmark.get("n_samples", 5000),
            n_species=benchmark.get("n_species", 10),
            n_regions=benchmark.get("n_regions", 3),
            prior_knowledge=prior_patterns,
            obvious_patterns=obvious_patterns,
            novel_patterns=novel_patterns,
            include_prior_in_prompt=benchmark.get("include_prior_in_prompt", True),
            prior_prompt_detail=benchmark.get("prior_prompt_detail", "full"),
        )
