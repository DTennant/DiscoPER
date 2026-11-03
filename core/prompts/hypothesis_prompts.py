# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompts for hypothesis generation."""

GENERATE_HYPOTHESIS_PROMPT = """You are a scientific hypothesis generator for ecological data analysis.

Your task is to generate a testable hypothesis about patterns in the provided dataset.

## Dataset Summary
{dataset_summary}

## Instructions
1. Propose ONE specific, testable hypothesis about relationships between variables in the dataset
2. Focus on correlation or group difference hypotheses
3. Specify which variables are involved
4. Indicate the expected direction of the effect (positive, negative, difference)
5. Be mindful of potential confounds (observer bias, location density, temporal effects)

## Hypothesis Types to Consider
- Correlation: "Variable X correlates with variable Y"
- Group difference: "Species A has higher/lower values of metric M than species B"

## Output Format
Return a JSON object with the following fields:
{{
    "statement": "Clear natural language statement of the hypothesis",
    "scope": {{"species": ["species_0"], "elevation": [0, 3000]}},  // Filters to apply
    "variables": ["elevation", "temperature"],  // Variables involved
    "target": "elevation",  // What is being predicted/explained (optional)
    "expected_direction": "positive",  // "positive", "negative", "difference", etc.
    "risk_flags": ["location_density", "observer_bias"]  // Potential confounds
}}

## Examples

Example 1 - Correlation:
{{
    "statement": "Species species_0 prefers higher elevations than average",
    "scope": {{"species": ["species_0"]}},
    "variables": ["species", "elevation"],
    "target": "elevation",
    "expected_direction": "positive",
    "risk_flags": ["location_density", "observer_bias"]
}}

Example 2 - Group Difference:
{{
    "statement": "Species species_1 occurs at lower temperatures than species species_2",
    "scope": {{"species": ["species_1", "species_2"]}},
    "variables": ["species", "temperature"],
    "target": "temperature",
    "expected_direction": "difference",
    "risk_flags": ["observer_bias", "temporal_effects"]
}}

Now generate ONE hypothesis:
"""


GENERATE_HYPOTHESIS_WITH_CONTEXT_PROMPT = """You are a scientific hypothesis generator for ecological data analysis.

Your task is to generate a NOVEL testable hypothesis about patterns in the provided dataset.

## Dataset Summary
{dataset_summary}

## Prior Knowledge (DO NOT re-test these known relationships)
{prior_knowledge}

## Recent Discovery Results
The following hypotheses have been tested recently. Learn from what worked and what didn't:
{recent_insights}

## Hypotheses to AVOID (already tested or too similar)
Do NOT propose hypotheses similar to these - they have already been explored:
{hypotheses_to_avoid}

## Instructions
1. Propose ONE specific, testable hypothesis that is DIFFERENT from previous attempts
2. If previous hypotheses about a topic (e.g., nocturnal species) were rejected, try a completely different angle
3. Focus on unexplored variable combinations
4. Look for interaction effects, threshold effects, or conditional relationships
5. Be mindful of potential confounds (observer bias, location density, temporal effects)

## Hypothesis Types to Consider
- Correlation: "Variable X correlates with variable Y"
- Group difference: "Group A has higher/lower values of metric M than group B"
- Interaction: "The effect of X on Y depends on Z"
- Threshold: "Variable X affects Y only when Z is above/below threshold"

## Output Format
Return a JSON object with the following fields:
{{
    "statement": "Clear natural language statement of the hypothesis",
    "scope": {{"species": ["species_0"], "elevation": [0, 3000]}},  // Filters to apply
    "variables": ["elevation", "temperature"],  // Variables involved
    "target": "elevation",  // What is being predicted/explained (optional)
    "expected_direction": "positive",  // "positive", "negative", "difference", etc.
    "risk_flags": ["location_density", "observer_bias"]  // Potential confounds
}}

## Key Guidance
- If many hypotheses about variable X have been rejected, STOP testing X and try other variables
- Look for strong effect sizes in the recent insights - similar patterns may exist elsewhere
- Explore variables that haven't been tested yet (check the dataset summary)
- Consider multi-variable interactions, not just simple correlations

Now generate ONE NOVEL hypothesis that explores a NEW angle:
"""


GENERATE_HYPOTHESIS_VISUAL_PROMPT = """You are a scientific hypothesis generator for visual ecology data analysis.

You are given BOTH numeric features AND sample images from ecology landscape scenes.
Study the attached images carefully — they show the actual ecology scenes from which the numeric features were extracted.

## Dataset Summary
{dataset_summary}

## Prior Knowledge (DO NOT re-test these known relationships)
{prior_knowledge}

## Recent Discovery Results
The following hypotheses have been tested recently. Learn from what worked and what didn't:
{recent_insights}

## Hypotheses to AVOID (already tested or too similar)
Do NOT propose hypotheses similar to these - they have already been explored:
{hypotheses_to_avoid}

## Attached Images
{n_images} sample images from the dataset are attached. These are the ACTUAL ecology scene images.
Look at them to understand:
- What visual elements vary across scenes (vegetation, water, animals, terrain, sky)
- How those visual elements might relate to the numeric features
- Patterns you can SEE that the numeric features might not fully capture

## Instructions
1. LOOK at the attached images and reason about what you observe
2. Propose ONE specific, testable hypothesis about relationships between the numeric features
3. Your hypothesis should be informed by what you SEE in the images, not just the numbers
4. Focus on unexplored variable combinations
5. Look for interaction effects, threshold effects, or conditional relationships

## Hypothesis Types to Consider
- Correlation: "Variable X correlates with variable Y"
- Group difference: "Group A has higher/lower values of metric M than group B"
- Interaction: "The effect of X on Y depends on Z"
- Threshold: "Variable X affects Y only when Z is above/below threshold"

## Output Format
Return a JSON object with the following fields:
{{
    "statement": "Clear natural language statement of the hypothesis",
    "scope": {{}},
    "variables": ["feature_1", "feature_2"],
    "target": "feature_1",
    "expected_direction": "positive",
    "risk_flags": ["confound_1"]
}}

## Key Guidance
- Use your VISUAL observations to generate MORE INFORMED hypotheses
- If you notice something in the images (e.g., water bodies near green areas), formulate it as a testable hypothesis using the numeric features
- If many hypotheses about variable X have been rejected, STOP testing X and try other variables
- Explore variables that haven't been tested yet

Now generate ONE NOVEL hypothesis informed by your visual analysis of the attached images:
"""
