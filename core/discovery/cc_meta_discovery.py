"""
Claude Code Meta-Discovery Agent.

Analyzes accumulated base insights to find higher-level patterns and
generates guided hypotheses. Uses Claude Code's tool-use capabilities
to write analysis scripts, compute statistics across insights, and
produce actionable meta-insights.
"""

import os
import json
from typing import Any, Optional

from core.claude_code_backend import ClaudeCodeBackend
from core.discovery.types import Hypothesis, Insight, MetaInsight
from core.discovery.insight_store import InsightStore


class CCMetaDiscovery:
    """Discover patterns across insights and generate guided hypotheses via Claude Code."""

    def __init__(
        self,
        backend: ClaudeCodeBackend,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or (
            "You are a meta-science analyst. You analyze patterns across scientific insights "
            "to discover higher-level relationships, confounds, and generalizations. "
            "You can write and run Python scripts to compute statistics across insights. "
            "Your goal is to find actionable patterns that guide future hypothesis generation."
        )

    def discover_meta_patterns(
        self,
        insight_store: InsightStore,
        dataset_summary: dict[str, Any],
        n_meta_hypotheses: int = 3,
        max_retries: int = 3,
        working_dir: Optional[str] = None,
    ) -> tuple[list[MetaInsight], list[Hypothesis]]:
        """
        Analyze base insights and generate meta-insights + guided hypotheses.

        Compatible with MetaDiscoveryAgent.discover_meta_patterns() interface.

        Args:
            insight_store: Store containing accumulated insights.
            dataset_summary: Summary of the dataset.
            n_meta_hypotheses: Number of guided hypotheses to produce.
            max_retries: Maximum retries on failure.
            working_dir: Working directory for Claude Code subprocess.

        Returns:
            Tuple of (meta_insights, guided_hypotheses).
        """
        insights = insight_store.insights
        base_insights = [i for i in insights if not getattr(i, "is_meta", False)]

        if len(base_insights) < 2:
            print("Not enough base insights for meta-discovery (need at least 2)")
            return [], []

        # Serialize insights for the prompt
        insights_data = []
        for ins in base_insights:
            insights_data.append({
                "id": ins.id[:8],
                "claim": ins.claim[:200],
                "verdict": ins.verdict,
                "strength": round(ins.strength, 3),
                "hypothesis_id": ins.hypothesis_id[:8],
            })

        prompt = f"""Analyze the following {len(base_insights)} base insights from a scientific discovery pipeline
and produce meta-insights (patterns across insights) plus {n_meta_hypotheses} guided hypotheses.

## Dataset Summary
```json
{json.dumps(dataset_summary, indent=2, default=str)}
```

## Base Insights
```json
{json.dumps(insights_data, indent=2)}
```

## Instructions

1. Look for patterns across the base insights:
   - Variables that appear frequently in supported/rejected findings
   - Potential confounds (variable X appears in multiple findings — could it be a confound?)
   - Clusters of related findings
   - Contradictions between findings
   - Gaps — variable combinations that haven't been explored
   - Hints at interaction effects

2. You may write and run a Python script via Bash to compute statistics
   (e.g., which variables are most frequent in supported vs rejected insights).

3. Generate meta-insights (actionable observations about patterns).

4. Based on the meta-insights, propose {n_meta_hypotheses} NEW hypotheses to test.
   These should address gaps, control for confounds, or test interaction effects
   suggested by the meta-insights.

## Output Format

Output a JSON block:
```json
{{
  "meta_insights": [
    {{
      "meta_type": "confound_pattern|variable_cluster|success_pattern|contradiction|gap|interaction_hint",
      "observation": "What pattern was observed across insights",
      "actionable_recommendation": "Specific guidance for future hypotheses",
      "affected_variables": ["var1", "var2"],
      "source_insight_ids": ["id1", "id2"],
      "confidence": 0.8,
      "priority": "high|medium|low"
    }}
  ],
  "guided_hypotheses": [
    {{
      "statement": "Testable hypothesis statement",
      "based_on_meta_insight": "Which meta-insight this addresses",
      "scope": {{}},
      "variables": ["var1", "var2"],
      "target": "target variable",
      "expected_direction": "positive|negative|difference",
      "controls": ["variables to control for"]
    }}
  ]
}}
```
"""

        for attempt in range(max_retries):
            try:
                result = self.backend.invoke_json(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    working_dir=working_dir,
                    max_turns=15,
                    timeout=300,
                )

                # Parse meta-insights
                meta_insights = []
                for mi in result.get("meta_insights", []):
                    meta_insight = MetaInsight.create(
                        meta_type=mi.get("meta_type", "gap"),
                        observation=mi.get("observation", ""),
                        actionable_recommendation=mi.get("actionable_recommendation", ""),
                        affected_variables=mi.get("affected_variables", []),
                        source_insight_ids=mi.get("source_insight_ids", []),
                        confidence=mi.get("confidence", 0.7),
                        priority=mi.get("priority", "medium"),
                    )
                    meta_insights.append(meta_insight)

                # Parse guided hypotheses
                hypotheses = []
                for h in result.get("guided_hypotheses", []):
                    hypothesis = Hypothesis.create(
                        statement=h.get("statement", ""),
                        scope=h.get("scope", {}),
                        variables=h.get("variables", []),
                        target=h.get("target"),
                        expected_direction=h.get("expected_direction"),
                        risk_flags=h.get("controls", []),
                        test_plan={
                            "meta_guided": True,
                            "based_on_meta_insight": h.get("based_on_meta_insight", ""),
                            "controls": h.get("controls", []),
                        },
                    )
                    hypotheses.append(hypothesis)

                return meta_insights, hypotheses

            except Exception as e:
                print(f"CC meta-discovery attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return [], []

        return [], []
