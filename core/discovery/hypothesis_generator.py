# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Hypothesis generation agent."""

from typing import Any, Optional
import json

from core.agent import Agent
from core.prompts import hypothesis_prompts
from core.discovery.types import Hypothesis, Insight


class HypothesisGeneratorAgent(Agent):
    """Agent for generating testable hypotheses."""

    def generate(
        self,
        dataset_summary: dict[str, Any],
        n_hypotheses: int = 1,
        max_retries: int = 3,
        recent_insights: Optional[list[Insight]] = None,
        previous_hypotheses: Optional[list[str]] = None,
        prior_knowledge_text: Optional[str] = None,
        images: Optional[list[str]] = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses about the dataset.

        Args:
            dataset_summary: Summary statistics and schema of the dataset
            n_hypotheses: Number of hypotheses to generate
            max_retries: Maximum number of retries for LLM call
            recent_insights: List of recent Insight objects for context
            previous_hypotheses: List of hypothesis statements to avoid duplicating
            prior_knowledge_text: Text describing prior knowledge to avoid rediscovering
            images: Optional base64-encoded sample images for vision-informed hypothesis generation

        Returns:
            List of Hypothesis objects
        """
        hypotheses = []

        generated_statements = list(previous_hypotheses) if previous_hypotheses else []

        for i in range(n_hypotheses):
            insights_context = self._format_insights_context(recent_insights)
            avoid_context = self._format_avoid_context(generated_statements)
            prior_context = prior_knowledge_text or ""

            if images:
                prompt = hypothesis_prompts.GENERATE_HYPOTHESIS_VISUAL_PROMPT.format(
                    dataset_summary=json.dumps(dataset_summary, indent=2),
                    recent_insights=insights_context,
                    hypotheses_to_avoid=avoid_context,
                    prior_knowledge=prior_context,
                    n_images=len(images),
                )
            else:
                prompt = (
                    hypothesis_prompts.GENERATE_HYPOTHESIS_WITH_CONTEXT_PROMPT.format(
                        dataset_summary=json.dumps(dataset_summary, indent=2),
                        recent_insights=insights_context,
                        hypotheses_to_avoid=avoid_context,
                        prior_knowledge=prior_context,
                    )
                )

            output_schema = {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The hypothesis statement to test",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Column names involved in this hypothesis",
                    },
                    "scope": {
                        "type": "object",
                        "description": "Scope constraints, e.g. {kingdom: 'Fungi'}",
                    },
                    "expected_direction": {
                        "type": "string",
                        "description": "Expected direction: positive, negative, or difference",
                    },
                },
                "required": ["statement", "variables"],
            }

            try:
                hypothesis_dict = self.act_json(
                    instruction=prompt,
                    output_schema=output_schema,
                    tool_name="generate_hypothesis",
                    tool_description="Generate a testable scientific hypothesis about the dataset",
                    max_retries=max_retries,
                    images=images,
                )

                # Create Hypothesis object
                hypothesis = Hypothesis.create(
                    statement=hypothesis_dict["statement"],
                    scope=hypothesis_dict.get("scope", {}),
                    variables=hypothesis_dict.get("variables", []),
                    target=hypothesis_dict.get("target"),
                    expected_direction=hypothesis_dict.get("expected_direction"),
                    risk_flags=hypothesis_dict.get("risk_flags", []),
                )

                hypotheses.append(hypothesis)
                # Track this hypothesis to avoid duplicates in subsequent iterations
                generated_statements.append(hypothesis.statement)

            except Exception as e:
                print(f"Failed to generate hypothesis {i + 1}/{n_hypotheses}: {e}")
                continue

        return hypotheses

    def _format_insights_context(self, insights: Optional[list[Insight]]) -> str:
        """Format recent insights for the prompt context."""
        if not insights:
            return "No previous insights available yet."

        lines = []
        for i, insight in enumerate(insights[-10:], 1):  # Last 10 insights
            verdict_emoji = (
                "✓"
                if insight.verdict == "supported"
                else "✗"
                if insight.verdict == "rejected"
                else "?"
            )
            lines.append(f"{i}. [{verdict_emoji}] {insight.claim[:100]}...")
            if insight.verdict == "supported":
                lines.append(f"   → Effect: {insight.strength:.2f}")

        return "\n".join(lines) if lines else "No previous insights available yet."

    def generate_set_comparison(
        self,
        set_a_images: list[str],
        set_b_images: list[str],
        set_a_name: str = "Set A",
        set_b_name: str = "Set B",
        previous_attempts: Optional[list[str]] = None,
        max_retries: int = 3,
    ) -> dict:
        """Generate a hypothesis about the visual difference between two image sets.

        Returns dict with keys: statement, observation_set_a, observation_set_b,
        confidence, scope, variables.
        """
        from core.prompts import visdiff_prompts

        avoid_text = self._format_avoid_context(previous_attempts)

        prompt = visdiff_prompts.GENERATE_SET_COMPARISON_HYPOTHESIS_PROMPT.format(
            set_a_name=set_a_name,
            set_b_name=set_b_name,
            n_images_a=len(set_a_images),
            n_images_b=len(set_b_images),
            previous_attempts=avoid_text,
        )

        all_images = set_a_images + set_b_images

        output_schema = {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "scope": {"type": "object"},
                "variables": {"type": "array", "items": {"type": "string"}},
                "target": {"type": "string"},
                "expected_direction": {"type": "string"},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
                "observation_set_a": {"type": "string"},
                "observation_set_b": {"type": "string"},
                "confidence": {"type": "string"},
            },
            "required": ["statement", "scope", "variables"],
        }

        try:
            result = self.act_json(
                instruction=prompt,
                output_schema=output_schema,
                tool_name="generate_set_comparison",
                tool_description="Identify the key visual difference between two image sets",
                max_retries=max_retries,
                images=all_images,
            )
            return result
        except Exception as e:
            print(f"Failed to generate set comparison hypothesis: {e}")
            return {
                "statement": f"Error: {e}",
                "scope": {"set_a": set_a_name, "set_b": set_b_name},
                "variables": [],
            }

    def _format_avoid_context(self, hypotheses: Optional[list[str]]) -> str:
        """Format hypotheses to avoid for the prompt context."""
        if not hypotheses:
            return "None - this is the first hypothesis."

        lines = []
        for i, hyp in enumerate(hypotheses[-15:], 1):  # Last 15 hypotheses
            lines.append(f"{i}. {hyp[:80]}...")

        return "\n".join(lines)
