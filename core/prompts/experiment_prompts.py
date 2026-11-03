# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompts for experiment planning."""


PLAN_EXPERIMENT_PROMPT = """You are a scientific experiment planner.

Your task is to design a concrete experiment to test the given hypothesis.

## Hypothesis
{hypothesis}

## Dataset Summary
{dataset_summary}

## Available Tools
- corr_test: Compute correlation between two continuous variables (Spearman or Pearson)
- group_diff_test: Test for difference between two groups on a continuous metric
- predictive_test: Train a classifier to predict a target from features (currently not used in MVP)

## Instructions
1. Choose the appropriate test method based on the hypothesis
2. Specify which variables to use as inputs to the test
3. Define the slice of data to analyze (filters from hypothesis scope)
4. For MVP: Focus on corr_test or group_diff_test

## Output Format
Return a JSON object with the following fields:
{{
    "method": "corr_test",  // or "group_diff_test"
    "dataset_slice_spec": {{"species": ["species_0"]}},  // Filters to apply
    "feature_spec": {{  // Arguments for the test method
        "x": "elevation",  // For corr_test: x variable
        "y": "temperature",  // For corr_test: y variable
        "method": "spearman"  // Method parameter
    }},
    "metrics_to_compute": ["effect_size", "p_value", "ci"]
}}

## Examples

Example 1 - Correlation Test:
Hypothesis: "Species species_0 prefers higher elevations"
{{
    "method": "corr_test",
    "dataset_slice_spec": {{"species": ["species_0"]}},
    "feature_spec": {{
        "x": "elevation",
        "y": "latitude",  // Use latitude as proxy for presence/occurrence pattern
        "method": "spearman"
    }},
    "metrics_to_compute": ["effect_size", "p_value", "ci_lower", "ci_upper"]
}}

Example 2 - Group Difference Test:
Hypothesis: "Species species_1 occurs at lower temperatures than species species_2"
{{
    "method": "group_diff_test",
    "dataset_slice_spec": {{"species": ["species_1", "species_2"]}},
    "feature_spec": {{
        "metric": "temperature",
        "group": "species",
        "method": "perm_test",
        "effect": "cliffs_delta"
    }},
    "metrics_to_compute": ["effect_size", "p_value"]
}}

Now design ONE experiment for the given hypothesis:
"""
