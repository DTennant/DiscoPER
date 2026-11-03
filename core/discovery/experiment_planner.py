# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Experiment planning agent."""

from typing import Any
import json

from core.agent import Agent
from core.prompts import experiment_prompts
from core.discovery.types import Hypothesis, Experiment


class ExperimentPlannerAgent(Agent):
    """Agent for planning experiments to test hypotheses."""

    def plan(
        self,
        hypothesis: Hypothesis,
        dataset_summary: dict[str, Any],
        max_retries: int = 3,
    ) -> list[Experiment]:
        """
        Plan experiments to test a hypothesis.

        Args:
            hypothesis: Hypothesis to test
            dataset_summary: Dataset summary statistics
            max_retries: Maximum retries for LLM call

        Returns:
            List of Experiment objects (MVP: returns 1 experiment)
        """
        # Format prompt
        hypothesis_str = json.dumps(
            {
                "statement": hypothesis.statement,
                "scope": hypothesis.scope,
                "variables": hypothesis.variables,
                "expected_direction": hypothesis.expected_direction,
            },
            indent=2,
        )

        prompt = experiment_prompts.PLAN_EXPERIMENT_PROMPT.format(
            hypothesis=hypothesis_str,
            dataset_summary=json.dumps(dataset_summary, indent=2),
        )

        output_schema = {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [
                        "corr_test",
                        "group_diff_test",
                        "predictive_test",
                        "stratified_retest",
                        "visual_attribute_test",
                        "visual_group_comparison",
                    ],
                },
                "feature_spec": {
                    "type": "object",
                    "description": "Test parameters. For corr_test: {x: column, y: column}. For group_diff_test: {metric: column, group: column}.",
                },
                "dataset_slice_spec": {
                    "type": "object",
                    "description": "Data filtering, e.g. {kingdom: ['Animalia', 'Fungi']}",
                },
                "metric_column": {
                    "type": "string",
                    "description": "Column to measure (for group_diff_test) or X variable (for corr_test)",
                },
                "group_column": {
                    "type": "string",
                    "description": "Column to group by (for group_diff_test) or Y variable (for corr_test)",
                },
                "filter_column": {
                    "type": "string",
                    "description": "Column to filter data on (optional)",
                },
                "filter_values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Values to keep in filter_column",
                },
            },
            "required": ["method", "metric_column", "group_column"],
        }

        try:
            experiment_dict = self.act_json(
                instruction=prompt,
                output_schema=output_schema,
                tool_name="plan_experiment",
                tool_description="Plan a rigorous experiment to test the hypothesis",
                max_retries=max_retries,
            )

            # Reconstruct feature_spec from flat fields if not provided
            if not experiment_dict.get("feature_spec"):
                method = experiment_dict.get("method", "group_diff_test")
                metric = experiment_dict.get("metric_column", "")
                group = experiment_dict.get("group_column", "")
                if method == "corr_test":
                    experiment_dict["feature_spec"] = {
                        "x": metric,
                        "y": group,
                        "method": "spearman",
                    }
                else:
                    experiment_dict["feature_spec"] = {
                        "metric": metric,
                        "group": group,
                        "method": "perm_test",
                        "effect": "cliffs_delta",
                    }

            if not experiment_dict.get("dataset_slice_spec"):
                fc = experiment_dict.get("filter_column", "")
                fv = experiment_dict.get("filter_values", [])
                if fc and fv:
                    experiment_dict["dataset_slice_spec"] = {fc: fv}
                else:
                    experiment_dict["dataset_slice_spec"] = {}

            # Create Experiment object
            experiment = Experiment.create(
                hypothesis_id=hypothesis.id,
                dataset_slice_spec=experiment_dict.get("dataset_slice_spec", {}),
                feature_spec=experiment_dict.get("feature_spec", {}),
                method=experiment_dict.get("method", "group_diff_test"),
                metrics_to_compute=experiment_dict.get(
                    "metrics_to_compute", ["effect_size", "p_value"]
                ),
                budget={"max_samples": 1000},  # Default budget for MVP
            )

            return [experiment]

        except Exception as e:
            print(f"Failed to plan experiment: {e}")
            # Return empty list on failure
            return []
