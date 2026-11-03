"""
Ideation diversity metrics for evaluating hypothesis exploration breadth.

Measures how diverse the agent's generated hypotheses are across a concept
taxonomy, without requiring embedding models. Works across all dataset types:
- Causal DAG benchmarks: uses variable names + graph structure
- Visual benchmarks: uses visual channel taxonomy
- iNat / real-world: uses domain concept taxonomy (species x time x space x pattern_type)
- Open-ended: falls back to keyword extraction from hypothesis statements

Four metrics:
1. Concept Coverage: fraction of concept space explored
2. Pairwise Dissimilarity: mean Hamming distance between hypothesis concept tuples
3. Dimension Entropy: Shannon entropy per concept dimension (detects mode collapse)
4. Novelty-per-step: fraction of hypotheses that explore a new concept tuple
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional


# ── Concept Taxonomy Definitions ─────────────────────────────────────────────

# Pattern type keywords for classification
PATTERN_TYPE_KEYWORDS: dict[str, list[str]] = {
    "correlation": [
        "correlat",
        "associat",
        "relationship between",
        "related to",
        "co-occur",
        "covary",
        "linked to",
        "connects",
    ],
    "group_difference": [
        "differ",
        "higher than",
        "lower than",
        "compared to",
        "versus",
        "more than",
        "less than",
        "greater",
        "between groups",
    ],
    "interaction": [
        "interact",
        "moderat",
        "depends on",
        "conditional",
        "when",
        "only if",
        "combined effect",
        "synerg",
        "amplif",
    ],
    "temporal": [
        "season",
        "month",
        "time",
        "peak",
        "spring",
        "summer",
        "autumn",
        "fall",
        "winter",
        "phenolog",
        "migration",
        "annual",
        "year",
    ],
    "spatial": [
        "latitud",
        "longitud",
        "geographic",
        "region",
        "range",
        "habitat",
        "distribut",
        "northern",
        "southern",
        "tropical",
        "boreal",
    ],
    "causal": [
        "cause",
        "mediat",
        "confound",
        "control for",
        "pathway",
        "mechanism",
        "drives",
        "leads to",
        "through",
    ],
    "threshold": [
        "threshold",
        "above",
        "below",
        "only when",
        "cutoff",
        "tipping point",
        "nonlinear",
        "piecewise",
    ],
    "distributional": [
        "distribution",
        "variance",
        "skew",
        "bimodal",
        "uniform",
        "concentrated",
        "spread",
        "range of",
    ],
}

# Temporal bins for iNat / ecology
TEMPORAL_BINS: dict[str, list[str]] = {
    "spring": ["march", "april", "may", "spring", "mar", "apr"],
    "summer": ["june", "july", "august", "summer", "jun", "jul", "aug"],
    "autumn": [
        "september",
        "october",
        "november",
        "autumn",
        "fall",
        "sep",
        "oct",
        "nov",
    ],
    "winter": ["december", "january", "february", "winter", "dec", "jan", "feb"],
    "year_round": ["year-round", "year round", "all year", "uniform", "no seasonal"],
}

# Visual channel taxonomy (from visual_mapping.py)
VISUAL_CHANNELS: list[str] = [
    "color",
    "vegetation",
    "sky",
    "animal_small",
    "animal_large",
    "water",
    "terrain",
    "snow",
    "texture",
]

VISUAL_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "color": [
        "color",
        "hue",
        "warmth",
        "temperature",
        "red",
        "blue",
        "warm",
        "cool",
        "bright",
    ],
    "vegetation": [
        "green",
        "vegetation",
        "plant",
        "tree",
        "forest",
        "grass",
        "leaf",
        "flora",
    ],
    "sky": ["sky", "cloud", "overcast", "clear", "weather", "sun"],
    "animal_small": [
        "small animal",
        "bird",
        "herbivore",
        "insect",
        "butterfly",
        "count",
    ],
    "animal_large": ["large animal", "predator", "mammal", "deer", "fox", "wolf"],
    "water": ["water", "lake", "river", "pond", "stream", "aquatic", "moisture", "wet"],
    "terrain": ["terrain", "mountain", "elevation", "height", "hill", "flat", "slope"],
    "snow": ["snow", "frost", "ice", "frozen", "cold", "winter cover"],
    "texture": ["texture", "rough", "smooth", "complex", "pattern", "edge", "detail"],
}


@dataclass
class ConceptTuple:
    """A hypothesis classified into concept dimensions.

    Each dimension is a string label from a finite set.
    The 'raw' field preserves the original values for debugging.
    """

    subject: str = "unknown"  # What entity/variable is being studied
    temporal: str = "unspecified"  # When (seasonal bin or 'unspecified')
    spatial: str = "unspecified"  # Where (latitude bin, region, or 'unspecified')
    pattern_type: str = "unknown"  # How (correlation, group_diff, interaction, etc.)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> tuple[str, str, str, str]:
        return (self.subject, self.temporal, self.spatial, self.pattern_type)

    def dimensions(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "temporal": self.temporal,
            "spatial": self.spatial,
            "pattern_type": self.pattern_type,
        }


@dataclass
class DiversityResult:
    """Complete diversity evaluation results."""

    concept_coverage: float = 0.0  # |unique tuples| / |possible tuples|
    variable_coverage: float = 0.0  # |unique variables| / |all variables|
    variable_pair_coverage: float = 0.0  # |unique pairs| / C(n,2)
    pairwise_dissimilarity: float = 0.0  # mean Hamming distance / n_dimensions
    dimension_entropy: dict[str, float] = field(default_factory=dict)
    mean_dimension_entropy: float = 0.0
    novelty_per_step: float = 0.0  # fraction of hypotheses with new concept tuple
    redundancy_rate: float = 0.0  # 1 - (unique tuples / total hypotheses)
    n_hypotheses: int = 0
    n_unique_tuples: int = 0
    concept_tuples: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_coverage": round(self.concept_coverage, 4),
            "variable_coverage": round(self.variable_coverage, 4),
            "variable_pair_coverage": round(self.variable_pair_coverage, 4),
            "pairwise_dissimilarity": round(self.pairwise_dissimilarity, 4),
            "dimension_entropy": {
                k: round(v, 4) for k, v in self.dimension_entropy.items()
            },
            "mean_dimension_entropy": round(self.mean_dimension_entropy, 4),
            "novelty_per_step": round(self.novelty_per_step, 4),
            "redundancy_rate": round(self.redundancy_rate, 4),
            "n_hypotheses": self.n_hypotheses,
            "n_unique_tuples": self.n_unique_tuples,
            "concept_tuples": self.concept_tuples,
        }


# ── Concept Classification ──────────────────────────────────────────────────


def classify_pattern_type(text: str) -> str:
    """Classify hypothesis text into a pattern type using keyword matching."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for ptype, keywords in PATTERN_TYPE_KEYWORDS.items():
        scores[ptype] = sum(1 for kw in keywords if kw in text_lower)
    if not scores or max(scores.values()) == 0:
        return "unknown"
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def classify_temporal(text: str) -> str:
    """Extract temporal bin from hypothesis text."""
    text_lower = text.lower()
    scores: dict[str, int] = {}
    for tbin, keywords in TEMPORAL_BINS.items():
        scores[tbin] = sum(1 for kw in keywords if kw in text_lower)
    if not scores or max(scores.values()) == 0:
        return "unspecified"
    return max(scores, key=scores.get)  # type: ignore[arg-type]


def classify_spatial(text: str) -> str:
    """Extract spatial bin from hypothesis text."""
    text_lower = text.lower()
    if any(
        kw in text_lower
        for kw in ["northern", "high latitude", "boreal", "arctic", ">45", ">50"]
    ):
        return "high_latitude"
    if any(
        kw in text_lower for kw in ["southern", "low latitude", "tropical", "equator"]
    ):
        return "low_latitude"
    if any(kw in text_lower for kw in ["temperate", "mid-latitude", "mid latitude"]):
        return "mid_latitude"
    if any(
        kw in text_lower
        for kw in ["latitud", "geographic", "spatial", "range", "distribut"]
    ):
        return "spatial_general"
    return "unspecified"


def classify_subject_from_variables(
    variables: list[str],
    statement: str = "",
    known_variables: Optional[set[str]] = None,
) -> str:
    """Classify the subject of a hypothesis from its variables list.

    For DAG benchmarks: returns the sorted variable pair or single variable.
    For vision: returns the visual channel.
    For iNat: returns species or taxonomic group.
    """
    if variables is None or (hasattr(variables, "__len__") and len(variables) == 0):
        # Fall back to visual channel detection from statement
        for channel, keywords in VISUAL_CHANNEL_KEYWORDS.items():
            if any(kw in statement.lower() for kw in keywords):
                return f"visual:{channel}"
        return "unknown"

    # Normalize and sort for consistency
    normalized = sorted(
        v.lower().strip() for v in variables[:3]
    )  # cap at 3 for readability
    return "+".join(normalized)


def classify_hypothesis(
    statement: str,
    variables: list[str],
    scope: Optional[dict[str, Any]] = None,
    known_variables: Optional[set[str]] = None,
) -> ConceptTuple:
    """Classify a hypothesis into a concept tuple.

    Works for all dataset types by extracting subject, temporal, spatial,
    and pattern_type dimensions from the hypothesis statement and metadata.
    """
    subject = classify_subject_from_variables(variables, statement, known_variables)
    temporal = classify_temporal(statement)
    spatial = classify_spatial(statement)
    pattern_type = classify_pattern_type(statement)

    return ConceptTuple(
        subject=subject,
        temporal=temporal,
        spatial=spatial,
        pattern_type=pattern_type,
        raw={"statement": statement[:200], "variables": variables},
    )


def classify_insight(
    insight: dict[str, Any],
    known_variables: Optional[set[str]] = None,
) -> ConceptTuple:
    """Classify an insight dict (from parquet/JSON) into a concept tuple.

    Extracts variables from the claim text by matching against known_variables,
    then classifies along all concept dimensions.
    """
    claim = insight.get("claim", "")
    # Try to get variables from the insight's hypothesis data if available
    variables = insight.get("variables", [])

    if (
        variables is None or (hasattr(variables, "__len__") and len(variables) == 0)
    ) and known_variables:
        # Extract variables from claim text
        claim_lower = claim.lower()
        variables = [
            v
            for v in known_variables
            if v.lower() in claim_lower or v.lower().replace("_", " ") in claim_lower
        ]

    return classify_hypothesis(
        statement=claim,
        variables=variables,
        known_variables=known_variables,
    )


# ── Diversity Metrics ────────────────────────────────────────────────────────


def _shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy from a list of counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _normalized_entropy(counts: list[int], n_bins: int) -> float:
    """Compute normalized Shannon entropy (0-1 range)."""
    if n_bins <= 1:
        return 0.0
    max_entropy = math.log2(n_bins)
    if max_entropy == 0:
        return 0.0
    return _shannon_entropy(counts) / max_entropy


def _hamming_distance(t1: tuple, t2: tuple) -> int:
    """Count dimensions where two tuples differ."""
    return sum(1 for a, b in zip(t1, t2) if a != b)


def compute_diversity(
    concept_tuples: list[ConceptTuple],
    known_variables: Optional[set[str]] = None,
    n_possible_subjects: Optional[int] = None,
) -> DiversityResult:
    """Compute all diversity metrics from a list of classified hypotheses.

    Args:
        concept_tuples: List of ConceptTuple objects (one per hypothesis/insight)
        known_variables: Set of all possible variable names (for coverage calc)
        n_possible_subjects: Override for number of possible subjects (for coverage)

    Returns:
        DiversityResult with all metrics
    """
    result = DiversityResult()
    n = len(concept_tuples)
    result.n_hypotheses = n

    if n == 0:
        return result

    tuples_as_tuple = [ct.as_tuple() for ct in concept_tuples]
    unique_tuples = set(tuples_as_tuple)
    result.n_unique_tuples = len(unique_tuples)

    # Store concept tuples for inspection
    result.concept_tuples = [ct.dimensions() for ct in concept_tuples]

    # ── 1. Concept Coverage ──────────────────────────────────────────────
    # Estimate the concept space size
    unique_subjects = set(
        ct.subject for ct in concept_tuples if ct.subject != "unknown"
    )
    unique_temporals = set(ct.temporal for ct in concept_tuples)
    unique_spatials = set(ct.spatial for ct in concept_tuples)
    unique_patterns = set(
        ct.pattern_type for ct in concept_tuples if ct.pattern_type != "unknown"
    )

    n_subjects = n_possible_subjects or max(len(unique_subjects), 1)
    # Use known dimension sizes for temporal/spatial/pattern
    n_temporals = len(TEMPORAL_BINS) + 1  # +1 for 'unspecified'
    n_spatials = 5  # high/mid/low_latitude, spatial_general, unspecified
    n_patterns = len(PATTERN_TYPE_KEYWORDS) + 1  # +1 for 'unknown'

    # Coverage = unique tuples explored / (realistic upper bound)
    # We use the product of unique values per dimension as a more realistic bound
    # than the full cartesian product (which is astronomically large)
    realistic_space = (
        n_subjects * min(n_temporals, 4) * min(n_spatials, 3) * min(n_patterns, 5)
    )
    result.concept_coverage = len(unique_tuples) / max(realistic_space, 1)

    # ── 2. Variable Coverage ─────────────────────────────────────────────
    all_mentioned_vars: set[str] = set()
    for ct in concept_tuples:
        raw_vars = ct.raw.get("variables", [])
        all_mentioned_vars.update(v.lower().strip() for v in raw_vars)

    if known_variables:
        result.variable_coverage = len(
            all_mentioned_vars & {v.lower() for v in known_variables}
        ) / max(len(known_variables), 1)
    elif all_mentioned_vars:
        result.variable_coverage = 1.0  # Can't compute without a reference set

    # Variable pair coverage
    all_var_pairs: set[tuple[str, str]] = set()
    for ct in concept_tuples:
        raw_vars = sorted(v.lower().strip() for v in ct.raw.get("variables", []))
        for pair in combinations(raw_vars, 2):
            all_var_pairs.add(pair)

    if known_variables and len(known_variables) >= 2:
        n_possible_pairs = len(known_variables) * (len(known_variables) - 1) // 2
        result.variable_pair_coverage = len(all_var_pairs) / max(n_possible_pairs, 1)
    elif all_var_pairs:
        result.variable_pair_coverage = 1.0

    # ── 3. Pairwise Dissimilarity ────────────────────────────────────────
    n_dims = 4
    if n >= 2:
        total_distance = 0
        n_pairs = 0
        for i, j in combinations(range(n), 2):
            total_distance += _hamming_distance(tuples_as_tuple[i], tuples_as_tuple[j])
            n_pairs += 1
        result.pairwise_dissimilarity = (total_distance / n_pairs) / n_dims
    else:
        result.pairwise_dissimilarity = 0.0

    # ── 4. Dimension Entropy ─────────────────────────────────────────────
    for dim_name in ["subject", "temporal", "spatial", "pattern_type"]:
        values = [getattr(ct, dim_name) for ct in concept_tuples]
        counter = Counter(values)
        n_bins = len(counter)
        entropy = _normalized_entropy(list(counter.values()), n_bins)
        result.dimension_entropy[dim_name] = entropy

    entropies = list(result.dimension_entropy.values())
    result.mean_dimension_entropy = (
        sum(entropies) / len(entropies) if entropies else 0.0
    )

    # ── 5. Novelty-per-step ──────────────────────────────────────────────
    seen: set[tuple] = set()
    novel_count = 0
    for t in tuples_as_tuple:
        if t not in seen:
            novel_count += 1
            seen.add(t)
    result.novelty_per_step = novel_count / n

    # ── 6. Redundancy Rate ───────────────────────────────────────────────
    result.redundancy_rate = 1.0 - (len(unique_tuples) / n)

    return result


# ── Convenience Functions ────────────────────────────────────────────────────


def evaluate_diversity_from_insights(
    insights: list[dict[str, Any]],
    ground_truth: Optional[dict[str, Any]] = None,
) -> DiversityResult:
    """Evaluate diversity from a list of insight dicts (from parquet/JSON).

    This is the main entry point for integration with evaluate_discovery.py.

    Args:
        insights: List of insight dicts with at least 'claim' and 'verdict' fields
        ground_truth: Optional ground truth dict for extracting known variables

    Returns:
        DiversityResult with all diversity metrics
    """
    # Extract known variables from ground truth if available
    known_variables: Optional[set[str]] = None
    if ground_truth:
        known_variables = set()
        for pattern in ground_truth.get("patterns", []):
            if "variables" in pattern:
                known_variables.update(pattern["variables"])
            if "chain" in pattern:
                known_variables.update(pattern["chain"])
            if "predictors" in pattern:
                known_variables.update(pattern["predictors"])
            if "outcome" in pattern:
                known_variables.add(pattern["outcome"])

    # Classify all insights (including rejected — diversity counts all attempts)
    concept_tuples = []
    for insight in insights:
        ct = classify_insight(insight, known_variables)
        concept_tuples.append(ct)

    return compute_diversity(
        concept_tuples,
        known_variables=known_variables,
    )


def print_diversity_summary(result: DiversityResult) -> None:
    """Print a human-readable summary of diversity metrics."""
    print("\n--- Ideation Diversity ---")
    print(f"  Hypotheses analyzed: {result.n_hypotheses}")
    print(f"  Unique concept tuples: {result.n_unique_tuples}")
    print(f"  Concept coverage:     {result.concept_coverage:.3f}")
    print(f"  Variable coverage:    {result.variable_coverage:.3f}")
    print(f"  Variable pair coverage: {result.variable_pair_coverage:.3f}")
    print(f"  Pairwise dissimilarity: {result.pairwise_dissimilarity:.3f}")
    print(f"  Mean dimension entropy: {result.mean_dimension_entropy:.3f}")
    print(f"  Novelty per step:     {result.novelty_per_step:.3f}")
    print(f"  Redundancy rate:      {result.redundancy_rate:.3f}")
    print(f"  Per-dimension entropy:")
    for dim, entropy in result.dimension_entropy.items():
        print(f"    {dim:15s}: {entropy:.3f}")
