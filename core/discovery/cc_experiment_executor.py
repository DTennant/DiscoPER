"""
Claude Code Experiment Executor.

Merges the experiment planning and execution steps into a single Claude Code agent.
Given a hypothesis and dataset, Claude Code designs the statistical test, runs
Python code via Bash on both train and val splits, and returns structured results
along with the analysis code used.

The agent is read-only (no Write tool). It executes Python inline via Bash
and returns the code in its JSON output for the runner to save.

This replaces: ExperimentPlannerAgent + ExperimentRunner + ToolRouter
for the Claude Code pipeline.
"""

import os
import json
from typing import Any, Optional

from core.claude_code_backend import ClaudeCodeBackend
from core.discovery.types import Hypothesis, Experiment, Result


class CCExperimentExecutor:
    """Design and execute experiments using Claude Code's tool-use capabilities."""

    def __init__(
        self,
        backend: ClaudeCodeBackend,
        system_prompt: Optional[str] = None,
    ):
        self.backend = backend
        self.system_prompt = system_prompt or (
            "You are a rigorous statistical testing expert. "
            "You design and execute statistical experiments to test scientific hypotheses. "
            "You run Python code using pandas, numpy, scipy, and sklearn via Bash "
            "(use python -c '...' for inline scripts), and report structured results. "
            "Always use proper statistical methods with effect sizes, p-values, and "
            "confidence intervals. Include your analysis code in the final output."
        )

    def execute_hypothesis(
        self,
        hypothesis: Hypothesis,
        dataset_summary: dict[str, Any],
        data_path: str,
        working_dir: Optional[str] = None,
        max_retries: int = 3,
    ) -> tuple[Optional[Experiment], dict[str, Result], str]:
        """
        Design and execute an experiment to test the given hypothesis.

        Runs the test on both train and val splits.

        Args:
            hypothesis: The hypothesis to test.
            dataset_summary: Summary of the dataset.
            data_path: Path to the data parquet file.
            working_dir: Working directory for Claude Code subprocess.
            max_retries: Maximum retries on failure.

        Returns:
            Tuple of (Experiment object, dict mapping split→Result, analysis_code).
            Returns (None, {}, "") on complete failure.
        """
        prompt = f"""You need to design and execute a statistical experiment to test this hypothesis.

## Hypothesis
Statement: {hypothesis.statement}
Variables: {json.dumps(hypothesis.variables)}
Expected direction: {hypothesis.expected_direction or "not specified"}
Scope: {json.dumps(hypothesis.scope)}

## Dataset Summary
```json
{json.dumps(dataset_summary, indent=2, default=str)}
```

## Data File
The dataset is at: {data_path}
It is a parquet file. Load it with `pandas.read_parquet("{data_path}")`.

## Instructions

1. Load the data and examine the relevant columns.
2. Choose an appropriate statistical test:
   - For correlations: Spearman/Pearson correlation with bootstrap CI
   - For group differences: t-test, Mann-Whitney, or permutation test with effect size (Cohen's d, Cliff's delta)
   - For predictive relationships: logistic/linear regression with train/val split
   - For interactions: stratified analysis or interaction terms
3. Check if the dataset has a column called `_split` with values "train" and "val".
   If it does, use it. If not, create a random 70/30 train/val split.
   Run the SAME test on each split separately.
4. Run the analysis via Bash using `python -c '...'` for inline scripts.
   Do NOT write files to disk — execute everything inline and print results to stdout.
5. After running the analysis, include the FULL Python code you executed in your output.

## Output Format

After running your analysis, output a JSON block with this structure:
```json
{{
  "method": "corr_test|group_diff_test|predictive_test|interaction_test|other",
  "feature_spec": {{"x_variable": "...", "y_variable": "...", ...}},
  "analysis_code": "The complete Python code you executed (single string, newlines as \\n)",
  "train_results": {{
    "status": "success|fail",
    "effect_size": 0.0,
    "p_value": 0.0,
    "ci_lower": 0.0,
    "ci_upper": 0.0,
    "notes": "..."
  }},
  "val_results": {{
    "status": "success|fail",
    "effect_size": 0.0,
    "p_value": 0.0,
    "ci_lower": 0.0,
    "ci_upper": 0.0,
    "notes": "..."
  }}
}}
```

IMPORTANT:
- Always include effect_size and p_value in results. If the test doesn't
  naturally produce these, compute reasonable equivalents (e.g., r-squared as effect size
  for regression, Cohen's d for group comparisons).
- Include the full analysis code in "analysis_code" — this is saved for reproducibility.
"""

        for attempt in range(max_retries):
            try:
                result = self.backend.invoke_json(
                    prompt=prompt,
                    system_prompt=self.system_prompt,
                    working_dir=working_dir,
                    max_turns=20,
                    timeout=600,
                )

                analysis_code = result.get("analysis_code", "")

                experiment = Experiment.create(
                    hypothesis_id=hypothesis.id,
                    dataset_slice_spec=hypothesis.scope,
                    feature_spec=result.get("feature_spec", {}),
                    method=result.get("method", "unknown"),
                )

                results = {}
                for split_name in ("train", "val"):
                    split_data = result.get(f"{split_name}_results", {})
                    status = split_data.get("status", "fail")
                    numbers = {}
                    for key in ("effect_size", "p_value", "ci_lower", "ci_upper", "gain", "score", "baseline_score"):
                        if key in split_data:
                            try:
                                numbers[key] = float(split_data[key])
                            except (ValueError, TypeError):
                                pass
                    results[split_name] = Result(
                        experiment_id=experiment.id,
                        status=status,
                        numbers=numbers,
                        notes=split_data.get("notes", f"CC experiment on {split_name} split"),
                    )

                return experiment, results, analysis_code

            except Exception as e:
                print(f"CC experiment execution attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    return None, {}, ""

        return None, {}, ""
