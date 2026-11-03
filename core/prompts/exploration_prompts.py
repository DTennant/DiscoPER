# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Prompts for the two-role exploration and testing system."""

# =============================================================================
# EXPLORATION AGENT PROMPTS (Role 1: Creative Explorer)
# =============================================================================

EXPLORATION_SYSTEM_PROMPT = """You are a creative data scientist exploring datasets to discover interesting patterns.

Your role is to:
1. Write Python code to explore and visualize data
2. Look for unexpected patterns, correlations, and anomalies
3. Generate preliminary hypotheses based on what you find
4. Note data quality issues that could affect analysis

You have access to pandas, numpy, scipy.stats, and matplotlib.
Be creative but methodical. Document what you find."""


INITIAL_EXPLORATION_PROMPT = """## Exploration Task

You are given a dataset to explore. Your goal is to understand the data and find interesting patterns that could become testable hypotheses.

### Dataset Information
{dataset_info}

### Available Columns
{columns_info}

### Exploration Focus
{exploration_focus}

### Instructions

Write Python code to explore this dataset. The data is available as `df` (a pandas DataFrame).

Focus on:
1. **Distributions**: What do the variables look like? Any skewness, outliers?
2. **Correlations**: Which variables are related? Unexpected correlations?
3. **Group Differences**: Do categorical groups differ on numeric variables?
4. **Interactions**: Do relationships depend on other variables?
5. **Data Quality**: Missing values, outliers, suspicious patterns?

Print your findings clearly with interpretations.

```python
# Your exploration code here
```"""


EXPLORATION_REFINEMENT_PROMPT = """## Continue Exploration

Based on your previous analysis, dig deeper into the most interesting findings.

### Previous Observations
{previous_observations}

### Code Output So Far
{previous_output}

### Next Steps

Write NEW code to:
1. Follow up on the most promising patterns
2. Check if patterns hold across subgroups
3. Look for potential confounders
4. Test the robustness of correlations

```python
# Your refined exploration code here
```"""


OBSERVATION_SYNTHESIS_PROMPT = """## Synthesize Your Findings

Based on all your exploration, summarize what you discovered.

### Exploration Results
{exploration_results}

### Task

Provide a structured summary of:

1. **Key Observations**: What patterns did you find? Include the evidence.
2. **Preliminary Hypotheses**: What testable hypotheses emerge from these observations?
3. **Data Quality Notes**: Any concerns about the data?

Be specific. Reference actual variables and statistics.

Respond with a JSON object containing:
- observations: list of observation objects
- preliminary_hypotheses: list of hypothesis objects  
- data_quality_notes: list of strings"""


# =============================================================================
# STATISTICAL TESTER AGENT PROMPTS (Role 2: Rigorous Tester)
# =============================================================================

STATISTICAL_TESTER_SYSTEM_PROMPT = """You are a rigorous statistical methodologist.

Your role is to:
1. Design proper statistical tests for hypotheses
2. Ensure tests use appropriate methods for the data types
3. Specify clear decision criteria (effect sizes, p-value thresholds)
4. Consider confounds and alternative explanations

You must use ONLY the predefined statistical test functions.
You must follow their exact parameter signatures.
You must specify clear pass/fail criteria for each test."""


TEST_PROPOSAL_PROMPT = """## Statistical Test Design

A hypothesis needs rigorous statistical testing.

### Hypothesis
{hypothesis}

### Context from Exploration
{exploration_context}

### Available Variables
{available_variables}

### Available Statistical Tests

You may ONLY use these tests:

1. **correlation_test**
   - Tests correlation between two continuous variables
   - Required: x_variable, y_variable
   - Optional: method (spearman/pearson), bootstrap_ci
   - Output: effect_size (r), p_value, ci_lower, ci_upper

2. **group_difference_test**
   - Tests if metric differs between two groups
   - Required: metric_variable, group_variable
   - Optional: method (perm_test/t_test/mann_whitney), effect_measure (cliffs_delta/cohens_d)
   - Output: effect_size, p_value, group_0_mean, group_1_mean

3. **predictive_test**
   - Tests if features predict target better than baseline
   - Required: feature_variables (list), target_variable
   - Optional: model (logreg), metric (auc/accuracy)
   - Output: score, baseline_score, gain

4. **interaction_test**
   - Tests if X->Y effect depends on moderator
   - Required: x_variable, y_variable, moderator_variable
   - Output: effect_low_moderator, effect_high_moderator, interaction_p_value

5. **confound_control_test**
   - Tests if X->Y effect persists after controlling for Z
   - Required: x_variable, y_variable, confound_variables (list)
   - Output: raw_effect, controlled_effect, effect_reduction_pct

### Task

Design up to {max_tests} statistical tests to rigorously evaluate this hypothesis.

For each test, specify:
- **test_type**: Which test to use (from the list above)
- **parameters**: All required parameters (use actual variable names)
- **decision_thresholds**: min_effect_size and max_p_value for supported/rejected
- **expected_outcome_if_supported**: What would the result look like?
- **expected_outcome_if_rejected**: What would the result look like?

Consider:
- What is the MOST DIRECT test of the hypothesis?
- What confounds should be controlled?
- What interactions might exist?
- How can we distinguish correlation from causation?

Respond with a JSON object."""


TEST_REFINEMENT_PROMPT = """## Refine Statistical Test

The initial test produced unexpected results and needs refinement.

### Original Test
{original_test}

### Test Result
{test_result}

### Problem
{refinement_reason}

### Task

Propose a refined test that addresses this issue. You might:
- Use a different statistical method
- Add confound controls
- Change the comparison groups
- Split the analysis by a moderator
- Adjust the decision thresholds

Use ONLY the available test types. Follow their exact signatures.

Respond with a refined test specification."""


# =============================================================================
# INTEGRATED PROMPTS (For the combined workflow)
# =============================================================================

EXPLORATION_TO_TEST_HANDOFF_PROMPT = """## Handoff: Exploration → Testing

The exploration phase has completed. Here are the findings:

### Observations
{observations}

### Preliminary Hypotheses
{preliminary_hypotheses}

### Data Quality Notes
{data_quality_notes}

### Next Step

Select the most promising hypotheses and design rigorous statistical tests for them.

Consider:
1. Which hypotheses are most likely to generalize?
2. Which have the clearest operationalization?
3. Which have the most important implications?

Rank the hypotheses and explain your reasoning."""


TEST_RESULTS_INTERPRETATION_PROMPT = """## Interpret Test Results

Statistical tests have been completed. Synthesize the results.

### Hypothesis
{hypothesis}

### Tests Run
{test_specifications}

### Results
{test_results}

### Task

Interpret these results:
1. What verdict do the tests support? (supported/rejected/inconclusive)
2. How consistent are the results across different tests?
3. What caveats or limitations should be noted?
4. What follow-up investigations might be warranted?

Be precise about effect sizes and statistical significance."""
