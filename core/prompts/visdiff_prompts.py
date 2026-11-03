"""Prompts for VisDiff set-comparison visual discovery."""

GENERATE_SET_COMPARISON_HYPOTHESIS_PROMPT = """You are a visual analyst comparing two sets of images to identify their key difference.

## Task
You are given sample images from two groups:
- **Set A** ({set_a_name}): The first {n_images_a} attached images
- **Set B** ({set_b_name}): The next {n_images_b} attached images

Study ALL the images carefully.

## Previous Attempts (do NOT repeat these)
{previous_attempts}

## Instructions
1. Look at Set A images — what do they have in common?
2. Look at Set B images — what do they have in common?
3. What is the SINGLE most important visual difference between the two sets?
4. Be SPECIFIC — "different animals" is better than "different content"
5. Focus on systematic differences, not individual image variation
6. Consider: subject matter, style, color palette, composition, setting, objects present

## Output Format
Return a JSON object:
{{
    "statement": "A clear description of the key visual difference between the two sets",
    "scope": {{"set_a": "{set_a_name}", "set_b": "{set_b_name}"}},
    "variables": ["visual_difference"],
    "target": "set_membership",
    "expected_direction": "difference",
    "risk_flags": ["individual_variation", "image_quality"],
    "observation_set_a": "What Set A images have in common",
    "observation_set_b": "What Set B images have in common",
    "confidence": "high, medium, or low"
}}

Now identify the key visual difference:
"""


JUDGE_SET_COMPARISON_PROMPT = """You are an impartial judge evaluating whether a predicted visual difference matches the ground truth difference between two image sets.

## Ground Truth
The actual difference between the two sets is:
"{ground_truth}"

## Prediction
The predicted difference is:
"{prediction}"

## Scoring Rubric
- Score 2 (MATCH): The prediction captures the same core concept as the ground truth, even if worded differently
- Score 1 (PARTIAL): The prediction is related to the ground truth but misses the key distinction or is too vague
- Score 0 (MISS): The prediction describes a completely different or irrelevant difference

## Examples
Ground truth: "Animal species (Dogs vs Cats)"
- "Dogs vs cats" → 2 (exact match, different wording)
- "Different types of pets" → 1 (related but too vague)
- "Indoor vs outdoor setting" → 0 (wrong difference)

Ground truth: "Time of day (Sunrise vs Sunset)"
- "The lighting suggests different times of day - sunrise vs sunset" → 2
- "Different sky colors" → 1 (related observation but doesn't identify the cause)
- "Different landscapes" → 0

## Output Format
Return a JSON object:
{{
    "score": 0 or 1 or 2,
    "reasoning": "Brief explanation of why this score was assigned"
}}

Now judge the prediction:
"""
