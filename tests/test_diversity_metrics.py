"""Tests for core.discovery.diversity_metrics."""

import math

import pytest

from core.discovery.diversity_metrics import (
    ConceptTuple,
    DiversityResult,
    classify_hypothesis,
    classify_insight,
    classify_pattern_type,
    classify_spatial,
    classify_subject_from_variables,
    classify_temporal,
    compute_diversity,
    evaluate_diversity_from_insights,
)


class TestClassifyPatternType:
    def test_correlation(self):
        assert classify_pattern_type("BMI correlates with diabetes") == "correlation"

    def test_group_difference(self):
        assert (
            classify_pattern_type("Species A differs from B in elevation")
            == "group_difference"
        )

    def test_interaction(self):
        assert (
            classify_pattern_type("The effect of X on Y depends on Z") == "interaction"
        )

    def test_temporal(self):
        assert classify_pattern_type("Monarchs peak in summer months") == "temporal"

    def test_spatial(self):
        assert classify_pattern_type("Red Fox has wider latitude range") == "spatial"

    def test_unknown_fallback(self):
        assert classify_pattern_type("xyzzy foobar") == "unknown"


class TestClassifyTemporal:
    def test_spring(self):
        assert classify_temporal("Dandelions bloom in April and May") == "spring"

    def test_autumn(self):
        assert classify_temporal("Fly Agaric peaks in October") == "autumn"

    def test_unspecified(self):
        assert classify_temporal("Red Fox is widespread") == "unspecified"


class TestClassifySpatial:
    def test_high_latitude(self):
        assert (
            classify_spatial("Red Deer at high latitude above 50N") == "high_latitude"
        )

    def test_tropical(self):
        assert classify_spatial("Found in tropical regions") == "low_latitude"

    def test_unspecified(self):
        assert classify_spatial("BMI correlates with diabetes") == "unspecified"


class TestClassifySubject:
    def test_from_variables(self):
        result = classify_subject_from_variables(["temperature", "elevation"])
        assert result == "elevation+temperature"

    def test_empty_variables_visual_fallback(self):
        result = classify_subject_from_variables(
            [], statement="The green vegetation is dense"
        )
        assert result == "visual:vegetation"

    def test_empty_everything(self):
        assert classify_subject_from_variables([], statement="foobar") == "unknown"


class TestClassifyHypothesis:
    def test_full_classification(self):
        ct = classify_hypothesis(
            statement="Monarch Butterfly observations peak in summer at high latitudes",
            variables=["species_name", "month", "latitude"],
        )
        assert ct.subject == "latitude+month+species_name"
        assert ct.temporal == "summer"
        assert ct.spatial == "high_latitude"
        assert ct.pattern_type == "temporal"

    def test_correlation_no_temporal(self):
        ct = classify_hypothesis(
            statement="BMI correlates with diabetes risk",
            variables=["bmi", "diabetes"],
        )
        assert ct.pattern_type == "correlation"
        assert ct.temporal == "unspecified"


class TestClassifyInsight:
    def test_basic(self):
        insight = {
            "claim": "Dandelion peaks in spring (April-May)",
            "verdict": "supported",
        }
        ct = classify_insight(insight)
        assert ct.temporal == "spring"
        assert ct.pattern_type == "temporal"

    def test_with_known_variables(self):
        insight = {
            "claim": "temperature correlates with elevation",
            "verdict": "supported",
        }
        known = {"temperature", "elevation", "precipitation"}
        ct = classify_insight(insight, known_variables=known)
        assert "temperature" in ct.raw["variables"]
        assert "elevation" in ct.raw["variables"]


class TestComputeDiversity:
    def test_empty(self):
        result = compute_diversity([])
        assert result.n_hypotheses == 0
        assert result.concept_coverage == 0.0

    def test_single_hypothesis(self):
        ct = ConceptTuple(
            subject="bmi+diabetes",
            temporal="unspecified",
            spatial="unspecified",
            pattern_type="correlation",
            raw={"variables": ["bmi", "diabetes"]},
        )
        result = compute_diversity([ct])
        assert result.n_hypotheses == 1
        assert result.n_unique_tuples == 1
        assert result.novelty_per_step == 1.0
        assert result.redundancy_rate == 0.0

    def test_identical_hypotheses_high_redundancy(self):
        ct = ConceptTuple(
            subject="bmi+diabetes",
            temporal="unspecified",
            spatial="unspecified",
            pattern_type="correlation",
            raw={"variables": ["bmi", "diabetes"]},
        )
        result = compute_diversity([ct, ct, ct, ct, ct])
        assert result.n_hypotheses == 5
        assert result.n_unique_tuples == 1
        assert result.redundancy_rate == 0.8
        assert result.pairwise_dissimilarity == 0.0
        assert result.novelty_per_step == 0.2

    def test_diverse_hypotheses_low_redundancy(self):
        tuples = [
            ConceptTuple(
                subject="a",
                temporal="spring",
                spatial="high_latitude",
                pattern_type="correlation",
                raw={"variables": ["a"]},
            ),
            ConceptTuple(
                subject="b",
                temporal="autumn",
                spatial="low_latitude",
                pattern_type="group_difference",
                raw={"variables": ["b"]},
            ),
            ConceptTuple(
                subject="c",
                temporal="summer",
                spatial="mid_latitude",
                pattern_type="interaction",
                raw={"variables": ["c"]},
            ),
        ]
        result = compute_diversity(tuples)
        assert result.n_unique_tuples == 3
        assert result.redundancy_rate == 0.0
        assert result.pairwise_dissimilarity == 1.0  # all 4 dims differ
        assert result.novelty_per_step == 1.0

    def test_variable_coverage_with_known(self):
        tuples = [
            ConceptTuple(
                subject="a+b",
                temporal="spring",
                spatial="unspecified",
                pattern_type="correlation",
                raw={"variables": ["a", "b"]},
            ),
        ]
        known = {"a", "b", "c", "d"}
        result = compute_diversity(tuples, known_variables=known)
        assert result.variable_coverage == 0.5  # 2/4
        assert result.variable_pair_coverage > 0  # 1 pair out of C(4,2)=6

    def test_dimension_entropy_uniform(self):
        tuples = [
            ConceptTuple(
                subject="a",
                temporal="spring",
                spatial="high_latitude",
                pattern_type="correlation",
                raw={"variables": []},
            ),
            ConceptTuple(
                subject="b",
                temporal="autumn",
                spatial="low_latitude",
                pattern_type="group_difference",
                raw={"variables": []},
            ),
        ]
        result = compute_diversity(tuples)
        for dim, entropy in result.dimension_entropy.items():
            assert entropy == 1.0  # 2 items, 2 bins = max entropy


class TestEvaluateDiversityFromInsights:
    def test_with_ground_truth(self):
        insights = [
            {"claim": "temperature correlates with elevation", "verdict": "supported"},
            {"claim": "precipitation differs between seasons", "verdict": "rejected"},
            {"claim": "species interact at high latitudes", "verdict": "supported"},
        ]
        ground_truth = {
            "patterns": [
                {
                    "variables": ["temperature", "elevation"],
                    "tier": "novel",
                    "difficulty": "simple",
                },
                {
                    "variables": ["precipitation", "season"],
                    "tier": "novel",
                    "difficulty": "moderate",
                },
            ]
        }
        result = evaluate_diversity_from_insights(insights, ground_truth)
        assert result.n_hypotheses == 3
        assert result.n_unique_tuples >= 2

    def test_without_ground_truth(self):
        insights = [
            {"claim": "Monarch peaks in summer", "verdict": "supported"},
        ]
        result = evaluate_diversity_from_insights(insights)
        assert result.n_hypotheses == 1


class TestDiversityResultSerialization:
    def test_to_dict_roundtrip(self):
        result = DiversityResult(
            concept_coverage=0.5,
            variable_coverage=0.75,
            n_hypotheses=10,
            n_unique_tuples=7,
        )
        d = result.to_dict()
        assert d["concept_coverage"] == 0.5
        assert d["variable_coverage"] == 0.75
        assert d["n_hypotheses"] == 10
        assert isinstance(d["dimension_entropy"], dict)
