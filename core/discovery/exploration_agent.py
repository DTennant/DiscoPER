# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Exploration agent for free-form data exploration via code generation."""

from typing import Any, Optional
import json
import traceback
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

from core.agent import Agent
from core.discovery.dataset_adapter import DatasetAdapter


# Schema for exploration results
EXPLORATION_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "What was observed in the data"},
                    "variables_involved": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "code_evidence": {"type": "string", "description": "Key code snippet or output that supports this"}
                },
                "required": ["description", "variables_involved", "confidence"]
            },
            "description": "List of observations discovered through exploration"
        },
        "preliminary_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string", "description": "Testable hypothesis statement"},
                    "rationale": {"type": "string", "description": "Why this hypothesis is worth testing"},
                    "suggested_variables": {"type": "array", "items": {"type": "string"}},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["statement", "rationale", "suggested_variables", "priority"]
            },
            "description": "Hypotheses that emerged from exploration, to be formally tested"
        },
        "data_quality_notes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Notes about data quality, missing values, outliers"
        }
    },
    "required": ["observations", "preliminary_hypotheses"]
}


class ExplorationAgent(Agent):
    """
    Agent that explores data through free-form Python code generation.
    
    Role 1 in the two-role system:
    - Writes arbitrary pandas/numpy/matplotlib code
    - Discovers patterns, anomalies, correlations
    - Outputs observations and preliminary hypotheses
    """

    def __init__(
        self,
        model_url: str,
        model_name: str,
        system_prompt: str,
        log_llm_metrics: bool = True,
        secrets: Optional[dict] = None,
        max_code_iterations: int = 3,
        code_timeout: int = 30
    ):
        """
        Initialize exploration agent.
        
        Args:
            model_url: LiteLLM API URL
            model_name: Model name
            system_prompt: System prompt for the agent
            log_llm_metrics: Whether to log LLM metrics
            secrets: API keys and secrets
            max_code_iterations: Max iterations for code refinement
            code_timeout: Timeout for code execution in seconds
        """
        super().__init__(
            model_url=model_url,
            model_name=model_name,
            system_prompt=system_prompt,
            log_llm_metrics=log_llm_metrics,
            secrets=secrets
        )
        self.max_code_iterations = max_code_iterations
        self.code_timeout = code_timeout

    def explore(
        self,
        dataset: DatasetAdapter,
        exploration_goal: Optional[str] = None,
        prior_observations: Optional[list[dict]] = None
    ) -> dict[str, Any]:
        """
        Explore the dataset through generated code.
        
        Args:
            dataset: Dataset adapter with schema and sample methods
            exploration_goal: Optional specific goal to focus exploration
            prior_observations: Previous observations to build upon
            
        Returns:
            Dict with observations, preliminary_hypotheses, code_history
        """
        # Get dataset info
        schema = self._get_dataset_schema(dataset)
        sample_data = dataset.sample({}, n=100)  # Get sample for exploration
        
        # Build exploration prompt
        prompt = self._build_exploration_prompt(
            schema=schema,
            sample_summary=self._summarize_sample(sample_data),
            exploration_goal=exploration_goal,
            prior_observations=prior_observations
        )
        
        # Generate exploration code
        code_history = []
        all_outputs = []
        
        for iteration in range(self.max_code_iterations):
            # Ask LLM to generate exploration code
            code_prompt = prompt if iteration == 0 else self._build_refinement_prompt(
                original_prompt=prompt,
                code_history=code_history,
                outputs=all_outputs
            )
            
            code_response = self.act(code_prompt)
            code = self._extract_code(code_response)
            
            if not code:
                continue
                
            # Execute code safely
            execution_result = self._execute_code_safely(code, sample_data)
            
            code_history.append({
                "iteration": iteration,
                "code": code,
                "success": execution_result["success"],
                "output": execution_result.get("output", ""),
                "error": execution_result.get("error", "")
            })
            
            if execution_result["success"]:
                all_outputs.append(execution_result["output"])
        
        # Synthesize observations from exploration
        synthesis_result = self._synthesize_observations(
            schema=schema,
            code_history=code_history,
            exploration_goal=exploration_goal
        )
        
        return {
            "observations": synthesis_result.get("observations", []),
            "preliminary_hypotheses": synthesis_result.get("preliminary_hypotheses", []),
            "data_quality_notes": synthesis_result.get("data_quality_notes", []),
            "code_history": code_history
        }

    def _get_dataset_schema(self, dataset: DatasetAdapter) -> dict:
        """Extract schema information from dataset."""
        schema = {
            "dataset_id": dataset.dataset_id,
            "columns": {}
        }
        
        # Get sample to infer types
        sample = dataset.sample({}, n=100)
        
        for col in sample.columns:
            dtype = str(sample[col].dtype)
            n_unique = sample[col].nunique()
            n_missing = sample[col].isna().sum()
            
            col_info = {
                "dtype": dtype,
                "n_unique": int(n_unique),
                "n_missing": int(n_missing)
            }
            
            if dtype in ['int64', 'float64']:
                col_info["min"] = float(sample[col].min()) if not sample[col].isna().all() else None
                col_info["max"] = float(sample[col].max()) if not sample[col].isna().all() else None
                col_info["mean"] = float(sample[col].mean()) if not sample[col].isna().all() else None
            elif dtype == 'object' or n_unique < 20:
                col_info["sample_values"] = sample[col].dropna().unique()[:10].tolist()
                
            schema["columns"][col] = col_info
            
        return schema

    def _summarize_sample(self, sample) -> str:
        """Create a text summary of the sample data."""
        summary_lines = [
            f"Sample size: {len(sample)} rows, {len(sample.columns)} columns",
            f"Columns: {', '.join(sample.columns[:20])}{'...' if len(sample.columns) > 20 else ''}",
            "",
            "Basic statistics:",
            str(sample.describe().to_string())
        ]
        return "\n".join(summary_lines)

    def _build_exploration_prompt(
        self,
        schema: dict,
        sample_summary: str,
        exploration_goal: Optional[str],
        prior_observations: Optional[list[dict]]
    ) -> str:
        """Build the exploration prompt for the LLM."""
        prompt = f"""You are exploring a dataset to discover interesting patterns and generate testable hypotheses.

## Dataset Schema
```json
{json.dumps(schema, indent=2, default=str)}
```

## Sample Summary
{sample_summary}

"""
        if exploration_goal:
            prompt += f"""## Exploration Goal
{exploration_goal}

"""
        
        if prior_observations:
            prompt += f"""## Prior Observations (build upon these)
{json.dumps(prior_observations, indent=2)}

"""
        
        prompt += """## Your Task
Write Python code to explore this dataset. The data is available as a pandas DataFrame called `df`.

You have access to: pandas (pd), numpy (np), scipy.stats, sklearn, matplotlib.pyplot (plt)

Focus on:
1. Distribution analysis of key variables
2. Correlation patterns between variables  
3. Group differences (if categorical variables exist)
4. Outlier detection
5. Potential confounders

Write clean, executable code. Print your findings clearly.

```python
# Your exploration code here
```"""
        
        return prompt

    def _build_refinement_prompt(
        self,
        original_prompt: str,
        code_history: list[dict],
        outputs: list[str]
    ) -> str:
        """Build prompt for refining exploration based on previous results."""
        prompt = original_prompt + "\n\n## Previous Exploration Results\n"
        
        for entry in code_history[-2:]:  # Show last 2 iterations
            prompt += f"\n### Iteration {entry['iteration']}\n"
            prompt += f"```python\n{entry['code']}\n```\n"
            if entry['success']:
                prompt += f"Output:\n```\n{entry['output'][:2000]}...\n```\n"
            else:
                prompt += f"Error: {entry['error']}\n"
        
        prompt += """
## Continue Exploration
Based on these results, write NEW code to dig deeper into the most interesting findings.
Focus on patterns that could become testable hypotheses.

```python
# Your refined exploration code here
```"""
        
        return prompt

    def _extract_code(self, response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        # Look for code blocks
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        return None

    def _execute_code_safely(self, code: str, data) -> dict[str, Any]:
        """Execute code in a restricted environment."""
        import numpy as np
        import pandas as pd
        from scipy import stats
        
        # Create restricted globals
        restricted_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "sorted": sorted,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "isinstance": isinstance,
                "type": type,
            },
            "np": np,
            "pd": pd,
            "stats": stats,
            "df": data.copy(),  # Provide data as 'df'
        }
        
        # Try to import matplotlib if available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            restricted_globals["plt"] = plt
        except ImportError:
            pass
        
        # Capture stdout
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, restricted_globals)
            
            output = stdout_capture.getvalue()
            return {
                "success": True,
                "output": output if output else "(no output)",
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "output": stdout_capture.getvalue(),
                "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            }

    def _synthesize_observations(
        self,
        schema: dict,
        code_history: list[dict],
        exploration_goal: Optional[str]
    ) -> dict[str, Any]:
        """Use LLM to synthesize observations from exploration results."""
        # Collect successful outputs
        outputs = []
        for entry in code_history:
            if entry["success"] and entry["output"]:
                outputs.append(f"Code:\n{entry['code']}\n\nOutput:\n{entry['output']}")
        
        if not outputs:
            return {"observations": [], "preliminary_hypotheses": [], "data_quality_notes": []}
        
        synthesis_prompt = f"""Based on the following data exploration results, synthesize your findings into structured observations and preliminary hypotheses.

## Dataset Schema
```json
{json.dumps(schema, indent=2, default=str)}
```

## Exploration Results
{"---".join(outputs[:3])}

## Task
Analyze these exploration results and provide:
1. **Observations**: What patterns, correlations, or anomalies did you find?
2. **Preliminary Hypotheses**: What testable hypotheses emerge from these observations?
3. **Data Quality Notes**: Any concerns about data quality?

Respond with a JSON object."""

        try:
            result = self.act_json(
                prompt=synthesis_prompt,
                json_schema=EXPLORATION_RESULT_SCHEMA,
                schema_name="exploration_results"
            )
            return result
        except Exception as e:
            print(f"Synthesis failed: {e}")
            return {"observations": [], "preliminary_hypotheses": [], "data_quality_notes": []}
