# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Insight evaluation - convert experimental results to validated insights."""

from typing import Optional, Any
import numpy as np

from core.discovery.types import Hypothesis, Experiment, Result, Insight
from core.discovery.insight_store import InsightStore
from core.agent import Agent


class InsightEvaluator:
    """Evaluates experimental results and creates insights."""

    def __init__(
        self,
        llm_client: Optional[Agent],
        insight_store: InsightStore,
        thresholds: dict[str, float],
        reward_weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize insight evaluator.

        Args:
            llm_client: LLM client for novelty estimation (optional)
            insight_store: Store of existing insights
            thresholds: Verdict thresholds (min_effect_size, max_p_value, etc.)
            reward_weights: Weights for reward computation (optional)
        """
        self.llm_client = llm_client
        self.insight_store = insight_store
        self.thresholds = thresholds
        self.reward_weights = reward_weights or {
            "strength": 1.0,
            "novelty": 2.0,
            "generalization": 1.5,
            "redundancy": -0.5,
            "confound": -1.0,
        }

    def evaluate(
        self, result: Result, hypothesis: Hypothesis, experiment: Experiment
    ) -> Insight:
        """
        Convert experimental result to insight.

        Args:
            result: Experimental result
            hypothesis: Original hypothesis
            experiment: Experiment specification

        Returns:
            Insight object with verdict, strength, novelty
        """
        # 1. Determine verdict
        verdict = self._determine_verdict(result)

        # 2. Compute strength
        strength = self._compute_strength(result, verdict)

        # 3. Compute novelty (simplified for MVP)
        novelty_proxy = self._compute_novelty(hypothesis, result)

        # 4. Create canonicalized claim
        claim = self._create_claim(hypothesis, result, verdict)

        # 5. Detect edges to existing insights (simplified for MVP)
        edges = self._detect_edges(hypothesis, verdict)

        reward = self._compute_reward(strength, novelty_proxy, hypothesis)

        insight = Insight.create(
            hypothesis_id=hypothesis.id,
            experiment_id=experiment.id,
            verdict=verdict,
            claim=claim,
            strength=strength,
            novelty_proxy=novelty_proxy,
            edges=edges,
            variables=list(hypothesis.variables) if hypothesis.variables else [],
            reward=reward,
        )

        return insight

    def evaluate_with_validation(
        self, results: dict[str, Result], hypothesis: Hypothesis, experiment: Experiment
    ) -> Insight:
        """
        Evaluate experimental results with proper train/val/test validation.

        Args:
            results: Dictionary of split name -> Result
            hypothesis: Original hypothesis
            experiment: Experiment specification

        Returns:
            Insight with verdict based on validation performance
        """
        # Get train and val results
        train_result = results.get("train")
        val_result = results.get("val")

        # Both must be successful for valid insight
        if not train_result or train_result.status != "success":
            return self._create_failed_insight(hypothesis, experiment, "Train failed")

        if not val_result or val_result.status != "success":
            return self._create_failed_insight(
                hypothesis, experiment, "Validation failed"
            )

        # Determine verdict based on BOTH train and val
        train_verdict = self._determine_verdict(train_result)
        val_verdict = self._determine_verdict(val_result)

        # Final verdict: both must agree for "supported"
        if train_verdict == "supported" and val_verdict == "supported":
            # Check for overfitting (large train/val discrepancy)
            if self._check_overfitting(train_result, val_result):
                verdict = "inconclusive"
                claim = (
                    f"{hypothesis.statement} (shows overfitting: train/val discrepancy)"
                )
            else:
                verdict = "supported"
                claim = self._create_validated_claim(
                    hypothesis, train_result, val_result
                )
        elif train_verdict == "rejected" or val_verdict == "rejected":
            verdict = "rejected"
            claim = f"{hypothesis.statement} was NOT validated (train: {train_verdict}, val: {val_verdict})"
        else:
            verdict = "inconclusive"
            claim = f"{hypothesis.statement} had inconclusive validation (train: {train_verdict}, val: {val_verdict})"

        # Compute strength (average of train and val, penalize discrepancy)
        train_strength = self._compute_strength(train_result, train_verdict)
        val_strength = self._compute_strength(val_result, val_verdict)
        strength = (train_strength + val_strength) / 2

        # Apply overfitting penalty
        if self._check_overfitting(train_result, val_result):
            strength *= 0.5

        # Compute novelty
        novelty_proxy = self._compute_novelty(hypothesis, train_result)

        # Detect edges
        edges = self._detect_edges(hypothesis, verdict)

        # Add validation info to generalization field
        generalization = {
            "train_verdict": train_verdict,
            "val_verdict": val_verdict,
            "train_metrics": train_result.numbers,
            "val_metrics": val_result.numbers,
            "validation_agreement": train_verdict == val_verdict,
        }

        # Compute composite reward
        reward = self._compute_reward(strength, novelty_proxy, hypothesis)

        # Create insight
        insight = Insight.create(
            hypothesis_id=hypothesis.id,
            experiment_id=experiment.id,
            verdict=verdict,
            claim=claim,
            strength=strength,
            novelty_proxy=novelty_proxy,
            generalization=generalization,
            edges=edges,
            variables=list(hypothesis.variables) if hypothesis.variables else [],
            reward=reward,
        )

        return insight

    def _compute_reward(
        self, strength: float, novelty_proxy: float, hypothesis: Hypothesis
    ) -> float:
        w = self.reward_weights
        reward = (
            w.get("strength", 1.0) * strength
            + w.get("novelty", 2.0) * novelty_proxy
            - w.get("redundancy", 0.5) * (1.0 - novelty_proxy)
        )
        if hypothesis.risk_flags:
            reward += w.get("confound", -1.0) * len(hypothesis.risk_flags) * 0.1
        return float(np.clip(reward, -5.0, 10.0))

    def _check_overfitting(self, train_result: Result, val_result: Result) -> bool:
        """Check if there's significant train/val discrepancy (overfitting)."""
        train_numbers = train_result.numbers
        val_numbers = val_result.numbers

        # Check effect size discrepancy
        train_effect = abs(train_numbers.get("effect_size", 0.0))
        val_effect = abs(val_numbers.get("effect_size", 0.0))

        if train_effect > 0:
            effect_ratio = val_effect / train_effect
            # Overfitting if val effect is < 60% of train effect
            if effect_ratio < 0.6:
                return True

        # Check predictive gain discrepancy
        train_gain = train_numbers.get("gain", 0.0)
        val_gain = val_numbers.get("gain", 0.0)

        if train_gain > 0:
            gain_ratio = val_gain / train_gain
            # Overfitting if val gain is < 70% of train gain
            if gain_ratio < 0.7:
                return True

        return False

    def _create_validated_claim(
        self, hypothesis: Hypothesis, train_result: Result, val_result: Result
    ) -> str:
        """Create claim statement with validation info."""
        train_effect = train_result.numbers.get("effect_size", 0.0)
        val_effect = val_result.numbers.get("effect_size", 0.0)
        train_p = train_result.numbers.get("p_value", 1.0)
        val_p = val_result.numbers.get("p_value", 1.0)

        return (
            f"{hypothesis.statement} (VALIDATED: "
            f"train effect={train_effect:.3f} p={train_p:.3f}, "
            f"val effect={val_effect:.3f} p={val_p:.3f})"
        )

    def _create_failed_insight(
        self, hypothesis: Hypothesis, experiment: Experiment, reason: str
    ) -> Insight:
        """Create inconclusive insight when validation fails."""
        return Insight.create(
            hypothesis_id=hypothesis.id,
            experiment_id=experiment.id,
            verdict="inconclusive",
            claim=f"{hypothesis.statement} ({reason})",
            strength=0.0,
            novelty_proxy=0.0,
        )

    def _determine_verdict(self, result: Result) -> str:
        """
        Determine verdict based on thresholds.

        Args:
            result: Experimental result

        Returns:
            "supported", "rejected", or "inconclusive"
        """
        if result.status != "success":
            return "inconclusive"

        numbers = result.numbers

        # Check thresholds
        effect_size = abs(numbers.get("effect_size", 0.0))
        p_value = numbers.get("p_value", 1.0)
        gain = numbers.get("gain", 0.0)  # For predictive tests

        min_effect_size = self.thresholds.get("min_effect_size", 0.2)
        max_p_value = self.thresholds.get("max_p_value", 0.05)
        min_gain = self.thresholds.get("min_predictive_gain", 0.05)

        # For correlation/group difference tests
        if "effect_size" in numbers and "p_value" in numbers:
            if effect_size >= min_effect_size and p_value <= max_p_value:
                return "supported"
            elif effect_size < min_effect_size * 0.5 or p_value > 0.2:
                return "rejected"
            else:
                return "inconclusive"

        # For predictive tests
        if "gain" in numbers:
            if gain >= min_gain:
                return "supported"
            elif gain < 0:
                return "rejected"
            else:
                return "inconclusive"

        return "inconclusive"

    def _compute_strength(self, result: Result, verdict: str) -> float:
        """
        Compute strength score.

        Args:
            result: Experimental result
            verdict: Verdict

        Returns:
            Strength score (0-1)
        """
        if verdict == "inconclusive":
            return 0.0

        numbers = result.numbers

        # Base strength on effect size and confidence
        effect_size = abs(numbers.get("effect_size", 0.0))
        p_value = numbers.get("p_value", 1.0)
        gain = numbers.get("gain", 0.0)

        # Normalize effect size to [0, 1]
        # Assume effect sizes > 0.8 are very strong
        effect_component = min(effect_size / 0.8, 1.0)

        # Normalize p-value (lower is better)
        # p < 0.001 is very strong
        p_component = max(0, 1.0 - (p_value / 0.05))

        # For predictive tests, use gain
        if gain > 0:
            # Assume gain > 0.2 is very strong
            gain_component = min(gain / 0.2, 1.0)
            return (gain_component + p_component) / 2

        # Weighted average
        strength = 0.7 * effect_component + 0.3 * p_component

        return float(np.clip(strength, 0.0, 1.0))

    def _compute_novelty(self, hypothesis: Hypothesis, result: Result) -> float:
        """
        Compute novelty proxy (simplified for MVP).

        Args:
            hypothesis: Original hypothesis
            result: Experimental result

        Returns:
            Novelty score (0-1)
        """
        # For MVP, use a simple heuristic:
        # Novelty = 1.0 - (number of existing insights with same variables / 10)
        # This is a placeholder; full version would use LLM + embedding similarity

        existing_insights = self.insight_store.insights
        same_variable_count = sum(
            1
            for ins in existing_insights
            if any(var in ins.claim for var in hypothesis.variables)
        )

        novelty = max(0.0, 1.0 - (same_variable_count / 10.0))
        return float(novelty)

    def _create_claim(
        self, hypothesis: Hypothesis, result: Result, verdict: str
    ) -> str:
        """
        Create canonicalized claim statement.

        Args:
            hypothesis: Original hypothesis
            result: Experimental result
            verdict: Verdict

        Returns:
            Claim statement
        """
        if verdict == "supported":
            effect_size = result.numbers.get("effect_size", 0.0)
            p_value = result.numbers.get("p_value", 1.0)
            return f"{hypothesis.statement} (effect={effect_size:.3f}, p={p_value:.3f})"
        elif verdict == "rejected":
            return f"{hypothesis.statement} was NOT supported by the data"
        else:
            return f"{hypothesis.statement} had inconclusive evidence"

    def _detect_edges(
        self, hypothesis: Hypothesis, verdict: str
    ) -> list[dict[str, str]]:
        """
        Detect relationships to existing insights (simplified for MVP).

        Args:
            hypothesis: Current hypothesis
            verdict: Current verdict

        Returns:
            List of edge dicts with target_id and relation type
        """
        edges = []

        # For MVP, skip edge detection
        # Full version would compare variables/scope to find:
        # - contradicts: same variables, opposite direction
        # - supports: same variables, same direction
        # - generalizes: broader scope
        # - refines: narrower scope

        return edges
