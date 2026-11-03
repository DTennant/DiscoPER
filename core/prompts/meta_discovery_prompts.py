# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Prompts for meta-discovery - analyzing insights to generate meta-insights.

KEY CONCEPTS:
- Base Insight: A discovery about raw data (e.g., "BMI correlates with diabetes")
- Meta-Insight: A discovery ABOUT insights that guides future discovery
  (e.g., "Age appears as confound in metabolic findings → control for age")
"""


# ============================================================================
# META-INSIGHT DISCOVERY PROMPT
# This prompt generates Meta-Insights: actionable observations about patterns
# ============================================================================

META_INSIGHT_DISCOVERY_PROMPT = """You are a meta-science analyst. Your task is to discover META-INSIGHTS.

## What is a Meta-Insight?

A Meta-Insight is NOT another finding about raw data. It is an OBSERVATION ABOUT THE PATTERN OF INSIGHTS that provides ACTIONABLE GUIDANCE for future hypothesis generation.

### Examples of Meta-Insights:

1. **Confound Pattern**: "Age appears as a variable in 4/5 metabolic findings. This suggests age is a potential confound that should be controlled in ALL future metabolic hypotheses."
   → ACTION: Future hypotheses should stratify by or control for age.

2. **Variable Cluster**: "BMI, waist circumference, and triglycerides always appear together in supported insights. These form a metabolic cluster."
   → ACTION: Test interaction effects between these clustered variables.

3. **Success Pattern**: "Correlation hypotheses have 80% success rate while predictive hypotheses have 30%. The current method is better suited for correlation analysis."
   → ACTION: Prioritize correlation-based hypotheses over predictive ones.

4. **Gap**: "No insights examine the relationship between family history and blood pressure despite both being available variables."
   → ACTION: Generate hypotheses exploring family history × blood pressure.

5. **Contradiction**: "Insight A says BMI positively correlates with diabetes, but Insight B says it's negative in elderly subgroup."
   → ACTION: Stratify the analysis by age group to resolve this contradiction.

## Dataset Summary
{dataset_summary}

## Analysis of Accumulated Base Insights
{meta_analysis}

## Your Task

Analyze the patterns in these BASE INSIGHTS and generate {n_meta_insights} META-INSIGHTS.

For each meta-insight, you MUST provide:
1. **observation**: What pattern did you observe across the base insights?
2. **actionable_recommendation**: What should future hypotheses do differently based on this observation?
3. **affected_variables**: Which variables does this meta-insight relate to?
4. **meta_type**: One of [confound_pattern, variable_cluster, success_pattern, contradiction, gap, interaction_hint]
5. **priority**: How important is it to act on this? (high/medium/low)
6. **source_insight_ids**: Which base insight IDs led to this observation?

Generate meta-insights now:
"""


# ============================================================================
# GUIDED HYPOTHESIS GENERATION PROMPT  
# This prompt generates new hypotheses GUIDED BY meta-insights
# ============================================================================

GUIDED_HYPOTHESIS_PROMPT = """You are a hypothesis generator. Your task is to generate NEW TESTABLE HYPOTHESES that are GUIDED BY META-INSIGHTS.

## What are Meta-Insights?

Meta-insights are observations about patterns across previous findings. They tell you:
- What confounds to control for
- What variable interactions to explore
- What gaps to fill
- What contradictions to resolve

Your hypotheses should FOLLOW THE RECOMMENDATIONS in the meta-insights.

## Dataset Summary
{dataset_summary}

## Meta-Insights to Address
{meta_insights}

## Your Task

Generate {n_hypotheses} NEW HYPOTHESES that:

1. **Follow the meta-insight recommendations** - If a meta-insight says "control for age", your hypothesis should include age as a control variable.

2. **Are testable** - They should be concrete enough to run an experiment on.

3. **Are distinct from previous findings** - Don't just repeat what was already tested.

4. **Reference which meta-insight they address** - Explain how this hypothesis implements the recommendation.

## Output Requirements

For each hypothesis:
- **statement**: A clear, testable hypothesis statement
- **based_on_meta_insight**: Brief description of which meta-insight this addresses
- **variables**: The variables involved
- **controls**: Variables to CONTROL FOR (based on meta-insight confound recommendations)
- **target**: The target/outcome variable
- **expected_direction**: What direction of effect you expect

Generate hypotheses that implement the meta-insight recommendations:
"""


# ============================================================================
# LEGACY: Original META_DISCOVERY_PROMPT (kept for backward compatibility)
# ============================================================================

META_DISCOVERY_PROMPT = """You are a meta-science analyst examining patterns across scientific insights.

Your task is to analyze accumulated insights from a discovery system and generate meta-hypotheses that:
1. Identify generalizations across multiple findings
2. Suggest confound controls for existing findings
3. Propose new feature interactions or derived variables
4. Detect systematic patterns or biases in the discovery process

## Dataset Summary
{dataset_summary}

## Meta-Analysis of Accumulated Insights
{meta_analysis}

## Instructions

Based on the patterns observed across insights:

1. **Generalizations**: Look for themes that span multiple supported insights
   - Are there common factors that appear in multiple findings?
   - Can we propose a unifying hypothesis?

2. **Confound Controls**: Identify potential confounding variables
   - If variable A correlates with outcome, is there a third variable B that might explain this?
   - Which findings should be retested controlling for confounds?

3. **Feature Inventions**: Propose new variables or interactions
   - Would interaction terms (e.g., age×BMI) be informative?
   - Should we create derived features based on patterns?

4. **Contradiction Resolution**: Address conflicting findings
   - Do any insights appear to contradict each other?
   - How can we design experiments to resolve conflicts?

## Output Format

Return a JSON object with meta-hypotheses:
{{
    "meta_hypotheses": [
        {{
            "statement": "Clear statement of the meta-hypothesis",
            "meta_type": "generalization|confound_control|feature_invention|contradiction_resolution",
            "scope": {{}},  // Filters to apply
            "variables": ["var1", "var2"],  // Variables involved
            "target": "outcome_variable",  // Optional
            "expected_direction": "positive|negative|interaction",  // Optional
            "risk_flags": [],  // Potential issues
            "rationale": "Brief explanation of why this meta-hypothesis is proposed"
        }}
    ]
}}

## Meta-Hypothesis Types

### Type 1: Generalization
Example: "Multiple obesity-related factors (BMI, waist ratio, triglycerides) all correlate with diabetes, suggesting a metabolic syndrome cluster"

### Type 2: Confound Control  
Example: "The correlation between BMI and diabetes may be confounded by age. Retest with age-stratified analysis."

### Type 3: Feature Invention
Example: "Create BMI×age interaction term to test if obesity's effect on diabetes varies by age group"

### Type 4: Contradiction Resolution
Example: "Insight A shows positive correlation while Insight B shows negative. Test on different population subgroups to resolve."

Now generate {n_meta_hypotheses} meta-hypotheses based on the patterns observed:
"""


CONFOUND_ANALYSIS_PROMPT = """You are analyzing scientific findings to identify potential confounding variables.

## Dataset Summary
{dataset_summary}

## Current Supported Insights
{insights_summary}

## Task

For each major finding, identify potential confounding variables that should be controlled:

1. What third variables might explain the observed correlation?
2. Are there demographic factors (age, gender, socioeconomic) that could confound results?
3. Are there lifestyle factors that co-vary with the studied variables?

## Output Format

Return a JSON object:
{{
    "confound_suggestions": [
        {{
            "original_finding": "Brief description of the finding",
            "potential_confounds": ["confound1", "confound2"],
            "suggested_retest": "How to retest controlling for confounds",
            "priority": "high|medium|low"
        }}
    ]
}}

Analyze the findings and suggest confound controls:
"""


INSIGHT_SYNTHESIS_PROMPT = """You are synthesizing scientific insights into a coherent narrative.

## Accumulated Insights

### Supported Findings
{supported_insights}

### Rejected Hypotheses  
{rejected_insights}

### Inconclusive Results
{inconclusive_insights}

## Task

Synthesize these findings into:
1. Key discoveries with high confidence
2. Patterns that need more investigation
3. Surprising null results that challenge assumptions
4. Open questions for future exploration

## Output Format

Return a JSON object:
{{
    "synthesis": {{
        "key_discoveries": [
            {{
                "finding": "description",
                "evidence_strength": "strong|moderate|weak",
                "implications": "what this means"
            }}
        ],
        "needs_investigation": [
            {{
                "pattern": "description",
                "current_evidence": "what we know",
                "suggested_experiments": ["exp1", "exp2"]
            }}
        ],
        "null_results": [
            {{
                "hypothesis": "what was tested",
                "why_surprising": "why this matters"
            }}
        ],
        "open_questions": ["question1", "question2"]
    }}
}}

Synthesize the insights:
"""
