"""
Claude Code Hypothesis Generator.

Uses a Claude Code subprocess to explore a dataset and generate testable hypotheses.
Unlike the LiteLLM-based HypothesisGeneratorAgent which only calls an API for JSON,
this agent can read data files and run Python scripts to explore the dataset,
producing richer, data-informed hypotheses.

The agent's exploratory code is captured in the output and saved alongside the
hypothesis for reproducibility.
"""

import os
import json
from typing import Any, Optional

from core.claude_code_backend import ClaudeCodeBackend
from core.discovery.types import Hypothesis, Insight


class CCHypothesisGenerator:
    """Generate hypotheses by letting Claude Code explore the dataset."""

    def __init__(
        self,
        backend: ClaudeCodeBackend,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or (
            "You are a scientific hypothesis generation expert. "
            "You can read data files and run Python analysis via Bash to explore the dataset. "
            "Always ground your hypotheses in actual data observations. "
            "Include the Python code you used for exploration in your final output."
        )

    def generate(
        self,
        dataset_summary: dict[str, Any],
        n_hypotheses: int = 1,
        max_retries: int = 3,
        recent_insights: Optional[list[Insight]] = None,
        previous_hypotheses: Optional[list[str]] = None,
        prior_knowledge_text: Optional[str] = None,
        data_path: Optional[str] = None,
        working_dir: Optional[str] = None,
    ) -> list[Hypothesis]:
        """
        Generate hypotheses about the dataset.

        Compatible with the same signature as HypothesisGeneratorAgent.generate(),
        with additional data_path and working_dir params for Claude Code.

        The exploration code used by the agent is stored in
        ``hypothesis.test_plan["exploration_code"]`` for each returned hypothesis,
        so the runner can persist it alongside the insight.

        Args:
            dataset_summary: Summary statistics and schema of the dataset.
            n_hypotheses: Number of hypotheses to generate.
            max_retries: Maximum retries on failure.
            recent_insights: Recent insights for context.
            previous_hypotheses: Hypotheses to avoid duplicating.
            prior_knowledge_text: Prior knowledge to avoid rediscovering.
            data_path: Path to the data parquet file for direct exploration.
            working_dir: Working directory for Claude Code subprocess.

        Returns:
            List of Hypothesis objects (with exploration code in test_plan).
        """
        # Build context for the prompt
        insights_text = self._format_insights(recent_insights)
        avoid_text = self._format_avoid(previous_hypotheses)

        prompt = f"""You are exploring a dataset to generate {n_hypotheses} testable scientific hypothesis(es).

## Dataset Summary
```json
{json.dumps(dataset_summary, indent=2, default=str)}
```

## Previous Insights
{insights_text}

## Hypotheses to Avoid (already tested)
{avoid_text}

## Prior Knowledge
{prior_knowledge_text or "None provided."}

## Data File
{f"The dataset is available at: {data_path}" if data_path else "No direct data file path provided. Use the dataset summary above."}

## Instructions

1. If a data file path is provided, read a sample of the data and run exploratory analysis
   (correlations, distributions, group comparisons) using Python via Bash.
   Use `python -c "..."` to run inline scripts — do NOT write files to disk.
2. Based on your analysis and the dataset summary, propose exactly {n_hypotheses} novel hypothesis(es).
3. Each hypothesis must be testable with statistical methods (correlation, group comparison, prediction, etc.).
4. Avoid duplicating any previously tested hypotheses listed above.
5. Focus on NOVEL patterns — interactions, conditional effects, non-obvious relationships.

## Output Format

Output your result as a JSON block with this structure:
```json
{{
  "exploration_code": "The complete Python code you ran for exploration (single string)",
  "hypotheses": [
    {{
      "statement": "Clear, testable hypothesis statement",
      "scope": {{}},
      "variables": ["var1", "var2"],
      "target": "optional target variable",
      "expected_direction": "positive|negative|difference|nonlinear",
      "risk_flags": ["potential confound1"]
    }}
  ]
}}
```

IMPORTANT: Include the full Python code you used for your exploratory analysis in "exploration_code".
This code is saved for reproducibility.
"""

        for attempt in range(max_retries):
            try:
                result = self.backend.invoke_json(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    working_dir=working_dir,
                )

                exploration_code = result.get("exploration_code", "")

                hypotheses = []
                for h in result.get("hypotheses", []):
                    hypothesis = Hypothesis.create(
                        statement=h["statement"],
                        scope=h.get("scope", {}),
                        variables=h.get("variables", []),
                        target=h.get("target"),
                        expected_direction=h.get("expected_direction"),
                        risk_flags=h.get("risk_flags", []),
                        test_plan={"exploration_code": exploration_code},
                    )
                    hypotheses.append(hypothesis)

                if hypotheses:
                    return hypotheses

            except Exception as e:
                print(f"CC hypothesis generation attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return []

        return []

    @staticmethod
    def _format_insights(insights: Optional[list[Insight]]) -> str:
        if not insights:
            return "No previous insights available yet."
        lines = []
        for i, insight in enumerate(insights[-10:], 1):
            marker = "supported" if insight.verdict == "supported" else insight.verdict
            lines.append(f"{i}. [{marker}] {insight.claim[:120]}")
        return "\n".join(lines)

    @staticmethod
    def _format_avoid(hypotheses: Optional[list[str]]) -> str:
        if not hypotheses:
            return "None — this is the first hypothesis."
        lines = []
        for i, h in enumerate(hypotheses[-15:], 1):
            lines.append(f"{i}. {h[:100]}")
        return "\n".join(lines)
