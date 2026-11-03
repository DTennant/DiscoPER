# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Meta-Discovery Agent - discovers patterns across insights and generates meta-insights.

KEY CONCEPTS:
- Base Insight: A discovery about raw data (e.g., "BMI correlates with diabetes")
- Meta-Insight: A discovery ABOUT insights that guides future discovery
  (e.g., "Age appears as confound in metabolic findings → control for age")

The flow is:
1. Analyze accumulated base insights
2. Generate Meta-Insights (actionable observations about patterns)
3. Use Meta-Insights to generate targeted hypotheses for testing
"""

from typing import Any, Optional
import json
import numpy as np

from core.agent import Agent
from core.prompts import meta_discovery_prompts
from core.discovery.types import Hypothesis, Insight, MetaInsight
from core.discovery.insight_store import InsightStore


# JSON Schema for meta-insights (observations about patterns in insights)
META_INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "meta_insights": {
            "type": "array",
            "description": "Actionable observations about patterns across base insights",
            "items": {
                "type": "object",
                "properties": {
                    "meta_type": {
                        "type": "string",
                        "enum": [
                            "confound_pattern",
                            "variable_cluster",
                            "success_pattern",
                            "contradiction",
                            "gap",
                            "interaction_hint",
                        ],
                        "description": "Type of meta-insight",
                    },
                    "observation": {
                        "type": "string",
                        "description": "What pattern was observed across the base insights",
                    },
                    "actionable_recommendation": {
                        "type": "string",
                        "description": "Specific guidance for designing future hypotheses based on this observation",
                    },
                    "affected_variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables this meta-insight relates to",
                    },
                    "source_insight_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of base insights that led to this meta-insight",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence in this meta-insight (0-1)",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "How important to act on this",
                    },
                },
                "required": [
                    "meta_type",
                    "observation",
                    "actionable_recommendation",
                    "affected_variables",
                ],
            },
        }
    },
    "required": ["meta_insights"],
}

# JSON Schema for hypotheses generated from meta-insights
GUIDED_HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "description": "Hypotheses designed based on meta-insights",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "Clear statement of the hypothesis to test",
                    },
                    "based_on_meta_insight": {
                        "type": "string",
                        "description": "Which meta-insight this hypothesis addresses",
                    },
                    "scope": {
                        "type": "object",
                        "description": "Filters to apply for testing",
                    },
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables involved",
                    },
                    "target": {"type": "string", "description": "Target variable"},
                    "expected_direction": {
                        "type": "string",
                        "description": "Expected direction of effect",
                    },
                    "controls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables to control for (based on meta-insight recommendations)",
                    },
                },
                "required": ["statement", "based_on_meta_insight", "variables"],
            },
        }
    },
    "required": ["hypotheses"],
}

# JSON Schema for confound suggestions
CONFOUND_SUGGESTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "confound_suggestions": {
            "type": "array",
            "description": "List of confound control suggestions",
            "items": {
                "type": "object",
                "properties": {
                    "original_finding": {
                        "type": "string",
                        "description": "Brief description of the finding",
                    },
                    "potential_confounds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of potential confounding variables",
                    },
                    "suggested_retest": {
                        "type": "string",
                        "description": "How to retest controlling for confounds",
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Priority level for this suggestion",
                    },
                },
                "required": ["original_finding", "potential_confounds"],
            },
        }
    },
    "required": ["confound_suggestions"],
}


class MetaDiscoveryAgent(Agent):
    """
    Operates on the InsightStore to discover meta-insights and generate guided hypotheses.

    KEY DISTINCTION:
    - Base Insights: Discoveries about raw data (e.g., "BMI correlates with diabetes")
    - Meta-Insights: Discoveries ABOUT insights that guide future discovery

    This agent:
    1. Analyzes accumulated base insights
    2. Generates Meta-Insights (actionable observations about patterns)
    3. Uses Meta-Insights to generate targeted hypotheses for testing
    """

    def discover_meta_insights(
        self,
        insight_store: InsightStore,
        dataset_summary: dict[str, Any],
        n_meta_insights: int = 3,
        max_retries: int = 3,
        prior_knowledge_text: Optional[str] = None,
    ) -> list[MetaInsight]:
        """
        Analyze accumulated base insights to discover meta-level patterns.

        Meta-insights are NOT tested experimentally - they are derived observations
        that guide future hypothesis generation.

        Args:
            insight_store: Store containing accumulated base insights
            dataset_summary: Summary of the underlying dataset
            n_meta_insights: Number of meta-insights to generate
            max_retries: Maximum retries for LLM call
            prior_knowledge_text: Prior knowledge patterns to avoid re-proposing

        Returns:
            List of MetaInsight objects (actionable observations)
        """
        insights = insight_store.insights

        # Only look at base insights (not meta-generated ones)
        base_insights = [i for i in insights if not getattr(i, "is_meta", False)]

        if len(base_insights) < 2:
            print("Not enough base insights for meta-discovery (need at least 2)")
            return []

        clusters = self._cluster_insights(base_insights)
        contradictions = self._find_contradictions(base_insights)
        variable_patterns = self._analyze_variable_patterns(
            base_insights, dataset_summary
        )

        # Build meta-analysis summary
        meta_analysis: dict[str, Any] = {
            "total_base_insights": len(base_insights),
            "supported_count": len(
                [i for i in base_insights if i.verdict == "supported"]
            ),
            "rejected_count": len(
                [i for i in base_insights if i.verdict == "rejected"]
            ),
            "inconclusive_count": len(
                [i for i in base_insights if i.verdict == "inconclusive"]
            ),
            "clusters": clusters,
            "contradictions": contradictions,
            "variable_patterns": variable_patterns,
            "top_insights": self._get_top_insights(base_insights, limit=5),
        }
        if prior_knowledge_text:
            meta_analysis["prior_knowledge_warning"] = (
                "IMPORTANT: The following patterns are ALREADY KNOWN as prior knowledge. "
                "Do NOT propose hypotheses that merely re-test these known patterns. "
                "Focus on NOVEL patterns that go beyond prior knowledge.\n\n"
                + prior_knowledge_text
            )

        # Generate meta-insights using LLM
        prompt = meta_discovery_prompts.META_INSIGHT_DISCOVERY_PROMPT.format(
            dataset_summary=json.dumps(dataset_summary, indent=2),
            meta_analysis=json.dumps(meta_analysis, indent=2),
            n_meta_insights=n_meta_insights,
        )

        try:
            result = self.act_json(
                instruction=prompt,
                output_schema=META_INSIGHTS_SCHEMA,
                tool_name="discover_meta_insights",
                tool_description="Discover meta-level patterns across base insights",
                max_retries=max_retries,
            )

            # Convert to MetaInsight objects
            meta_insights = []
            for mi_dict in result.get("meta_insights", []):
                meta_insight = MetaInsight.create(
                    meta_type=mi_dict.get("meta_type", "gap"),
                    observation=mi_dict.get("observation", ""),
                    actionable_recommendation=mi_dict.get(
                        "actionable_recommendation", ""
                    ),
                    affected_variables=mi_dict.get("affected_variables", []),
                    source_insight_ids=mi_dict.get("source_insight_ids", []),
                    confidence=mi_dict.get("confidence", 0.7),
                    priority=mi_dict.get("priority", "medium"),
                )
                meta_insights.append(meta_insight)

            print(f"\n📊 Generated {len(meta_insights)} meta-insights:")
            for mi in meta_insights:
                print(f"  [{mi.meta_type}] {mi.observation[:80]}...")
                print(f"    → Action: {mi.actionable_recommendation[:60]}...")

            return meta_insights

        except Exception as e:
            print(f"Failed to generate meta-insights: {e}")
            return []

    def generate_hypotheses_from_meta_insights(
        self,
        meta_insights: list[MetaInsight],
        dataset_summary: dict[str, Any],
        n_hypotheses: int = 3,
        max_retries: int = 3,
    ) -> list[Hypothesis]:
        """
        Generate testable hypotheses guided by meta-insights.

        The meta-insights inform what hypotheses to generate, ensuring
        they follow the actionable recommendations (e.g., controlling for confounds).

        Args:
            meta_insights: List of meta-insights to guide hypothesis generation
            dataset_summary: Summary of the underlying dataset
            n_hypotheses: Number of hypotheses to generate
            max_retries: Maximum retries for LLM call

        Returns:
            List of Hypothesis objects designed based on meta-insights
        """
        if not meta_insights:
            print("No meta-insights to guide hypothesis generation")
            return []

        # Format meta-insights for prompt
        meta_insight_summaries = []
        for mi in meta_insights:
            meta_insight_summaries.append(
                {
                    "id": mi.id[:8],
                    "type": mi.meta_type,
                    "observation": mi.observation,
                    "recommendation": mi.actionable_recommendation,
                    "variables": mi.affected_variables,
                    "priority": mi.priority,
                }
            )

        prompt = meta_discovery_prompts.GUIDED_HYPOTHESIS_PROMPT.format(
            dataset_summary=json.dumps(dataset_summary, indent=2),
            meta_insights=json.dumps(meta_insight_summaries, indent=2),
            n_hypotheses=n_hypotheses,
        )

        try:
            result = self.act_json(
                instruction=prompt,
                output_schema=GUIDED_HYPOTHESES_SCHEMA,
                tool_name="generate_guided_hypotheses",
                tool_description="Generate hypotheses based on meta-insights",
                max_retries=max_retries,
            )

            hypotheses = []
            for hyp_dict in result.get("hypotheses", []):
                hypothesis = Hypothesis.create(
                    statement=hyp_dict.get("statement", ""),
                    scope=hyp_dict.get("scope", {}),
                    variables=hyp_dict.get("variables", []),
                    target=hyp_dict.get("target"),
                    expected_direction=hyp_dict.get("expected_direction"),
                    risk_flags=hyp_dict.get(
                        "controls", []
                    ),  # Controls become risk flags
                    test_plan={
                        "meta_guided": True,
                        "based_on_meta_insight": hyp_dict.get(
                            "based_on_meta_insight", ""
                        ),
                        "controls": hyp_dict.get("controls", []),
                    },
                )
                hypotheses.append(hypothesis)

            print(f"\n🎯 Generated {len(hypotheses)} hypotheses from meta-insights:")
            for h in hypotheses:
                print(f"  • {h.statement[:80]}...")
                if h.test_plan.get("based_on_meta_insight"):
                    print(
                        f"    (based on: {h.test_plan['based_on_meta_insight'][:40]}...)"
                    )

            return hypotheses

        except Exception as e:
            print(f"Failed to generate hypotheses from meta-insights: {e}")
            return []

    def discover_meta_patterns(
        self,
        insight_store: InsightStore,
        dataset_summary: dict[str, Any],
        n_meta_hypotheses: int = 3,
        max_retries: int = 3,
        prior_knowledge_text: Optional[str] = None,
    ) -> tuple[list[MetaInsight], list[Hypothesis]]:
        """
        Main entry point: discover meta-insights and generate guided hypotheses.

        This is a TWO-STEP process:
        1. Discover Meta-Insights (observations about patterns in base insights)
        2. Generate Hypotheses guided by those meta-insights

        Args:
            insight_store: Store containing accumulated insights
            dataset_summary: Summary of the underlying dataset
            n_meta_hypotheses: Number of hypotheses to generate
            max_retries: Maximum retries for LLM call

        Returns:
            Tuple of (meta_insights, hypotheses)
        """
        print("\n" + "=" * 60)
        print("🔬 META-DISCOVERY: Analyzing patterns across base insights")
        print("=" * 60)

        meta_insights = self.discover_meta_insights(
            insight_store=insight_store,
            dataset_summary=dataset_summary,
            n_meta_insights=n_meta_hypotheses,
            max_retries=max_retries,
            prior_knowledge_text=prior_knowledge_text,
        )

        if not meta_insights:
            return [], []

        # Step 2: Generate hypotheses from meta-insights
        hypotheses = self.generate_hypotheses_from_meta_insights(
            meta_insights=meta_insights,
            dataset_summary=dataset_summary,
            n_hypotheses=n_meta_hypotheses,
            max_retries=max_retries,
        )

        print(
            f"\n✅ Meta-discovery complete: {len(meta_insights)} meta-insights → {len(hypotheses)} hypotheses"
        )

        return meta_insights, hypotheses

    def _cluster_insights(self, insights: list[Insight]) -> list[dict[str, Any]]:
        """
        Cluster insights by variables and scope.

        Returns list of clusters with:
        - variables: shared variables
        - insight_ids: IDs of insights in cluster
        - pattern: detected pattern (e.g., "all positive correlations")
        """
        if len(insights) < 2:
            return []

        # Group by variables
        variable_groups: dict[str, list[Insight]] = {}
        for insight in insights:
            # Extract variables from claim (simplified)
            # In full version, would use hypothesis.variables
            key = insight.hypothesis_id[:8]  # Use hypothesis prefix as grouping
            if key not in variable_groups:
                variable_groups[key] = []
            variable_groups[key].append(insight)

        # Find meaningful clusters (2+ insights with same verdict pattern)
        clusters = []

        # Group by verdict
        supported = [i for i in insights if i.verdict == "supported"]
        rejected = [i for i in insights if i.verdict == "rejected"]

        if len(supported) >= 2:
            # Look for common patterns in supported insights
            avg_strength = np.mean([i.strength for i in supported])
            clusters.append(
                {
                    "type": "supported_findings",
                    "count": len(supported),
                    "average_strength": float(avg_strength),
                    "insight_summaries": [i.claim[:100] for i in supported[:3]],
                }
            )

        if len(rejected) >= 2:
            clusters.append(
                {
                    "type": "rejected_hypotheses",
                    "count": len(rejected),
                    "insight_summaries": [i.claim[:100] for i in rejected[:3]],
                }
            )

        return clusters

    def _find_contradictions(self, insights: list[Insight]) -> list[dict[str, Any]]:
        """
        Find potential contradictions between insights.

        A contradiction occurs when:
        - Two insights involve overlapping variables
        - One is "supported" and suggests positive effect
        - Another is "supported" but suggests negative effect
        """
        contradictions = []

        # For MVP, look for simple contradictions based on claim text
        # Full version would use structured hypothesis comparison
        supported = [i for i in insights if i.verdict == "supported"]

        for i, insight1 in enumerate(supported):
            for insight2 in supported[i + 1 :]:
                # Check if claims might contradict
                # (In full version, compare variables and directions)
                if self._might_contradict(insight1, insight2):
                    contradictions.append(
                        {
                            "insight_1": {
                                "id": insight1.id,
                                "claim": insight1.claim[:100],
                            },
                            "insight_2": {
                                "id": insight2.id,
                                "claim": insight2.claim[:100],
                            },
                            "reason": "Potentially conflicting findings on related variables",
                        }
                    )

        return contradictions[:3]  # Limit to top 3

    def _might_contradict(self, insight1: Insight, insight2: Insight) -> bool:
        """
        Heuristic check if two insights might contradict.

        For MVP, uses simple string matching.
        Full version would compare structured hypothesis data.
        """
        # Extract key terms from claims
        claim1_lower = insight1.claim.lower()
        claim2_lower = insight2.claim.lower()

        # Look for opposite direction indicators
        positive_terms = ["positive", "higher", "increase", "correlates with"]
        negative_terms = ["negative", "lower", "decrease", "inverse"]

        has_pos_1 = any(term in claim1_lower for term in positive_terms)
        has_neg_1 = any(term in claim1_lower for term in negative_terms)
        has_pos_2 = any(term in claim2_lower for term in positive_terms)
        has_neg_2 = any(term in claim2_lower for term in negative_terms)

        # Contradiction if one positive and one negative
        if (has_pos_1 and has_neg_2) or (has_neg_1 and has_pos_2):
            return True

        return False

    def _analyze_variable_patterns(
        self,
        insights: list[Insight],
        dataset_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze which variables appear frequently in insights.

        Dynamically extracts variable names from the dataset summary (column names,
        feature names, species names, etc.) instead of using a hardcoded list.
        Falls back to extracting multi-word tokens from insight claims if no
        dataset summary is provided.

        Returns:
        - variable_analysis: per-variable mention counts and success rates
        - most_studied: top-5 most frequently mentioned variables
        """
        candidate_vars: list[str] = []
        if dataset_summary:
            for key in ("columns", "features", "feature_names", "variables"):
                if key in dataset_summary and isinstance(dataset_summary[key], list):
                    candidate_vars.extend(str(v) for v in dataset_summary[key])
            if "column_descriptions" in dataset_summary:
                desc = dataset_summary["column_descriptions"]
                if isinstance(desc, dict):
                    candidate_vars.extend(str(k) for k in desc.keys())
            for key in ("species", "categories", "class_names", "unique_values"):
                if key in dataset_summary and isinstance(
                    dataset_summary[key], (list, dict)
                ):
                    vals = (
                        dataset_summary[key]
                        if isinstance(dataset_summary[key], list)
                        else list(dataset_summary[key].keys())
                    )
                    candidate_vars.extend(str(v) for v in vals)

        seen: set[str] = set()
        search_vars: list[str] = []
        for v in candidate_vars:
            v_lower = v.lower().strip()
            if v_lower and v_lower not in seen and len(v_lower) >= 2:
                seen.add(v_lower)
                search_vars.append(v_lower)
                spaced = v_lower.replace("_", " ")
                if spaced != v_lower and spaced not in seen:
                    seen.add(spaced)
                    search_vars.append(spaced)

        if not search_vars:
            all_words: set[str] = set()
            for insight in insights:
                for word in insight.claim.lower().split():
                    cleaned = word.strip(".,;:()\"'")
                    if len(cleaned) >= 3 and cleaned.isalpha():
                        all_words.add(cleaned)
            search_vars = list(all_words)

        variable_counts: dict[str, int] = {}
        variable_supported: dict[str, int] = {}

        for insight in insights:
            claim_lower = insight.claim.lower()
            for var in search_vars:
                if var in claim_lower:
                    variable_counts[var] = variable_counts.get(var, 0) + 1
                    if insight.verdict == "supported":
                        variable_supported[var] = variable_supported.get(var, 0) + 1

        # Compute success rates
        success_rates = {}
        for var, count in variable_counts.items():
            supported = variable_supported.get(var, 0)
            success_rates[var] = {
                "total_mentions": count,
                "supported_count": supported,
                "success_rate": supported / count if count > 0 else 0,
            }

        return {
            "variable_analysis": success_rates,
            "most_studied": sorted(
                variable_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def _get_top_insights(
        self, insights: list[Insight], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Get top insights by strength for LLM context."""
        supported = [i for i in insights if i.verdict == "supported"]
        sorted_insights = sorted(supported, key=lambda i: i.strength, reverse=True)

        return [
            {
                "id": i.id[:8],
                "claim": i.claim[:150],
                "strength": round(i.strength, 3),
                "verdict": i.verdict,
            }
            for i in sorted_insights[:limit]
        ]

    def suggest_confound_controls(
        self,
        insight_store: InsightStore,
        dataset_summary: dict[str, Any],
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Analyze insights to suggest confound controls.

        Returns suggestions for retesting with confound controls.
        """
        insights = insight_store.insights
        supported = [i for i in insights if i.verdict == "supported"]

        if len(supported) < 2:
            return []

        # Build prompt for confound analysis
        prompt = meta_discovery_prompts.CONFOUND_ANALYSIS_PROMPT.format(
            dataset_summary=json.dumps(dataset_summary, indent=2),
            insights_summary=json.dumps(
                [
                    {"claim": i.claim[:200], "strength": i.strength}
                    for i in supported[:10]
                ],
                indent=2,
            ),
        )

        try:
            # Use act_json with function calling for reliable structured output
            result = self.act_json(
                instruction=prompt,
                output_schema=CONFOUND_SUGGESTIONS_SCHEMA,
                tool_name="suggest_confound_controls",
                tool_description="Suggest confound controls for existing findings",
                max_retries=max_retries,
            )

            return result.get("confound_suggestions", [])

        except Exception as e:
            print(f"Failed to generate confound suggestions: {e}")
            return []
