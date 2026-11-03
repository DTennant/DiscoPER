# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Statistical Tester Agent - proposes rigorous statistical tests with strict signatures."""

from typing import Any, Optional
import json

from core.agent import Agent


# =============================================================================
# STRICT TEST SPECIFICATIONS
# =============================================================================

# These are the ONLY statistical tests the tester can propose.
# Each has a strict signature that must be followed.

AVAILABLE_TESTS = {
    "correlation_test": {
        "description": "Test correlation between two continuous variables",
        "required_params": ["x_variable", "y_variable"],
        "optional_params": {
            "method": {"type": "string", "enum": ["spearman", "pearson"], "default": "spearman"},
            "bootstrap_ci": {"type": "boolean", "default": True}
        },
        "output_schema": {
            "effect_size": "float (-1 to 1, correlation coefficient)",
            "p_value": "float (statistical significance)",
            "ci_lower": "float (95% CI lower bound)",
            "ci_upper": "float (95% CI upper bound)"
        }
    },
    "group_difference_test": {
        "description": "Test if a metric differs between two groups",
        "required_params": ["metric_variable", "group_variable"],
        "optional_params": {
            "method": {"type": "string", "enum": ["perm_test", "t_test", "mann_whitney"], "default": "perm_test"},
            "effect_measure": {"type": "string", "enum": ["cliffs_delta", "cohens_d", "mean_diff"], "default": "cliffs_delta"}
        },
        "output_schema": {
            "effect_size": "float (standardized effect size)",
            "p_value": "float (statistical significance)",
            "group_0_mean": "float (mean of first group)",
            "group_1_mean": "float (mean of second group)"
        }
    },
    "predictive_test": {
        "description": "Test if features predict a target better than baseline",
        "required_params": ["feature_variables", "target_variable"],
        "optional_params": {
            "model": {"type": "string", "enum": ["logreg"], "default": "logreg"},
            "metric": {"type": "string", "enum": ["auc", "accuracy"], "default": "auc"}
        },
        "output_schema": {
            "score": "float (model performance)",
            "baseline_score": "float (baseline performance)",
            "gain": "float (improvement over baseline)"
        }
    },
    "interaction_test": {
        "description": "Test if the effect of X on Y depends on a moderator variable",
        "required_params": ["x_variable", "y_variable", "moderator_variable"],
        "optional_params": {
            "method": {"type": "string", "enum": ["stratified_correlation", "regression_interaction"], "default": "stratified_correlation"}
        },
        "output_schema": {
            "effect_low_moderator": "float (effect when moderator is low)",
            "effect_high_moderator": "float (effect when moderator is high)",
            "interaction_p_value": "float (significance of interaction)"
        }
    },
    "confound_control_test": {
        "description": "Test if effect X->Y persists after controlling for confound Z",
        "required_params": ["x_variable", "y_variable", "confound_variables"],
        "optional_params": {
            "method": {"type": "string", "enum": ["partial_correlation", "stratified_analysis"], "default": "partial_correlation"}
        },
        "output_schema": {
            "raw_effect": "float (effect without controlling)",
            "controlled_effect": "float (effect after controlling)",
            "effect_reduction_pct": "float (how much effect is explained by confound)"
        }
    }
}


# JSON Schema for test specification (what the agent outputs)
TEST_SPECIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "test_specifications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_id": {
                        "type": "string",
                        "description": "Unique identifier for this test"
                    },
                    "test_type": {
                        "type": "string",
                        "enum": list(AVAILABLE_TESTS.keys()),
                        "description": "Type of statistical test to run"
                    },
                    "hypothesis_being_tested": {
                        "type": "string",
                        "description": "The hypothesis this test addresses"
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Parameters for the test (must match test_type signature)"
                    },
                    "expected_outcome_if_supported": {
                        "type": "string",
                        "description": "What result would support the hypothesis"
                    },
                    "expected_outcome_if_rejected": {
                        "type": "string",
                        "description": "What result would reject the hypothesis"
                    },
                    "decision_thresholds": {
                        "type": "object",
                        "properties": {
                            "min_effect_size": {"type": "number"},
                            "max_p_value": {"type": "number"}
                        },
                        "description": "Thresholds for deciding supported/rejected"
                    }
                },
                "required": ["test_id", "test_type", "hypothesis_being_tested", "parameters", "decision_thresholds"]
            }
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of why these tests were chosen"
        }
    },
    "required": ["test_specifications", "reasoning"]
}


class StatisticalTesterAgent(Agent):
    """
    Agent that proposes rigorous statistical tests with strict signatures.
    
    Role 2 in the two-role system:
    - Reads hypotheses and exploration code/observations
    - Proposes statistical tests using ONLY predefined test functions
    - Output follows strict schema for validation
    - Does NOT execute tests, only specifies them
    """

    def __init__(
        self,
        model_url: str,
        model_name: str,
        system_prompt: str,
        log_llm_metrics: bool = True,
        secrets: Optional[dict] = None
    ):
        """
        Initialize statistical tester agent.
        
        Args:
            model_url: LiteLLM API URL
            model_name: Model name
            system_prompt: System prompt for the agent
            log_llm_metrics: Whether to log LLM metrics
            secrets: API keys and secrets
        """
        super().__init__(
            model_url=model_url,
            model_name=model_name,
            system_prompt=system_prompt,
            log_llm_metrics=log_llm_metrics,
            secrets=secrets
        )

    def propose_tests(
        self,
        hypothesis: str,
        exploration_observations: list[dict],
        exploration_code: Optional[str] = None,
        dataset_schema: Optional[dict] = None,
        max_tests: int = 3
    ) -> dict[str, Any]:
        """
        Propose statistical tests for a hypothesis.
        
        Args:
            hypothesis: The hypothesis statement to test
            exploration_observations: Observations from exploration phase
            exploration_code: Optional code that was run during exploration
            dataset_schema: Schema of available variables
            max_tests: Maximum number of tests to propose
            
        Returns:
            Dict with test_specifications and reasoning
        """
        prompt = self._build_test_proposal_prompt(
            hypothesis=hypothesis,
            observations=exploration_observations,
            exploration_code=exploration_code,
            dataset_schema=dataset_schema,
            max_tests=max_tests
        )
        
        result = self.act_json(
            prompt=prompt,
            json_schema=TEST_SPECIFICATION_SCHEMA,
            schema_name="test_specifications"
        )
        
        # Validate each test specification
        validated_specs = []
        for spec in result.get("test_specifications", []):
            validation = self._validate_test_specification(spec)
            if validation["valid"]:
                validated_specs.append(spec)
            else:
                print(f"Invalid test specification: {validation['error']}")
        
        return {
            "test_specifications": validated_specs,
            "reasoning": result.get("reasoning", ""),
            "validation_passed": len(validated_specs) == len(result.get("test_specifications", []))
        }

    def _build_test_proposal_prompt(
        self,
        hypothesis: str,
        observations: list[dict],
        exploration_code: Optional[str],
        dataset_schema: Optional[dict],
        max_tests: int
    ) -> str:
        """Build prompt for test proposal."""
        
        # Format available tests
        tests_doc = "## Available Statistical Tests\n\n"
        for test_name, test_info in AVAILABLE_TESTS.items():
            tests_doc += f"### `{test_name}`\n"
            tests_doc += f"**Description**: {test_info['description']}\n"
            tests_doc += f"**Required Parameters**: {', '.join(test_info['required_params'])}\n"
            tests_doc += f"**Optional Parameters**: {json.dumps(test_info['optional_params'], indent=2)}\n"
            tests_doc += f"**Output**: {json.dumps(test_info['output_schema'], indent=2)}\n\n"
        
        prompt = f"""You are a statistical testing expert. Your task is to propose rigorous statistical tests for a hypothesis.

{tests_doc}

## Hypothesis to Test
{hypothesis}

## Observations from Data Exploration
```json
{json.dumps(observations, indent=2)}
```
"""
        
        if exploration_code:
            prompt += f"""
## Exploration Code (for context)
```python
{exploration_code[:2000]}
```
"""
        
        if dataset_schema:
            prompt += f"""
## Dataset Schema (available variables)
```json
{json.dumps(dataset_schema, indent=2, default=str)}
```
"""
        
        prompt += f"""
## Your Task

Propose up to {max_tests} statistical tests to rigorously evaluate the hypothesis.

**CRITICAL REQUIREMENTS**:
1. You MUST use ONLY the test types listed above
2. You MUST provide ALL required parameters for each test
3. Parameters must reference REAL variables from the dataset schema
4. Each test must have clear decision thresholds

For each test, specify:
- What specific statistical test to run
- Which variables to use
- What thresholds determine supported/rejected
- What the expected outcomes mean

Think carefully about:
- What is the most direct test of the hypothesis?
- What confounds should be controlled for?
- What interactions might exist?
- Are there multiple ways to test this?

Respond with a JSON object matching the schema."""

        return prompt

    def _validate_test_specification(self, spec: dict) -> dict[str, Any]:
        """Validate a test specification against its schema."""
        test_type = spec.get("test_type")
        
        if test_type not in AVAILABLE_TESTS:
            return {"valid": False, "error": f"Unknown test type: {test_type}"}
        
        test_schema = AVAILABLE_TESTS[test_type]
        params = spec.get("parameters", {})
        
        # Check required parameters
        for required_param in test_schema["required_params"]:
            if required_param not in params:
                return {"valid": False, "error": f"Missing required parameter: {required_param}"}
        
        # Check optional parameters have valid values
        for opt_param, opt_schema in test_schema.get("optional_params", {}).items():
            if opt_param in params:
                value = params[opt_param]
                if "enum" in opt_schema and value not in opt_schema["enum"]:
                    return {"valid": False, "error": f"Invalid value for {opt_param}: {value}. Must be one of {opt_schema['enum']}"}
        
        # Check decision thresholds
        thresholds = spec.get("decision_thresholds", {})
        if "max_p_value" not in thresholds:
            return {"valid": False, "error": "Missing decision threshold: max_p_value"}
        
        return {"valid": True, "error": None}

    def refine_test(
        self,
        original_spec: dict,
        test_result: dict,
        refinement_reason: str
    ) -> dict[str, Any]:
        """
        Refine a test specification based on initial results.
        
        Args:
            original_spec: Original test specification
            test_result: Result from running the test
            refinement_reason: Why refinement is needed
            
        Returns:
            Refined test specification
        """
        prompt = f"""A statistical test was run but needs refinement.

## Original Test Specification
```json
{json.dumps(original_spec, indent=2)}
```

## Test Result
```json
{json.dumps(test_result, indent=2)}
```

## Reason for Refinement
{refinement_reason}

## Your Task
Propose a refined version of this test. You might:
- Adjust the method or parameters
- Add confound controls
- Change the comparison groups
- Adjust decision thresholds

Use ONLY the available test types and follow their schemas exactly.

Respond with a JSON object matching the test specification schema."""

        result = self.act_json(
            prompt=prompt,
            json_schema=TEST_SPECIFICATION_SCHEMA,
            schema_name="test_specifications"
        )
        
        if result.get("test_specifications"):
            return result["test_specifications"][0]
        return original_spec


# =============================================================================
# TEST EXECUTOR - Bridges test specifications to actual execution
# =============================================================================

class TestExecutor:
    """
    Executes validated test specifications.
    
    This bridges the StatisticalTesterAgent's specifications to the
    actual statistical functions in experiments.py
    """
    
    def __init__(self, tool_router):
        """
        Initialize test executor.
        
        Args:
            tool_router: ToolRouter for executing statistical primitives
        """
        self.tool_router = tool_router
    
    def execute_test(self, test_spec: dict, split: str = "train") -> dict[str, Any]:
        """
        Execute a test specification.
        
        Args:
            test_spec: Validated test specification
            split: Which data split to use
            
        Returns:
            Test result with verdict
        """
        test_type = test_spec["test_type"]
        params = test_spec["parameters"]
        thresholds = test_spec.get("decision_thresholds", {})
        
        # Map test_type to tool_router tool calls
        if test_type == "correlation_test":
            tool_call = {
                "tool": "corr_test",
                "args": {
                    "x": params["x_variable"],
                    "y": params["y_variable"],
                    "method": params.get("method", "spearman"),
                    "bootstrap_ci": params.get("bootstrap_ci", True)
                },
                "data_spec": {"slice_spec": {}, "n_samples": 1000}
            }
        
        elif test_type == "group_difference_test":
            tool_call = {
                "tool": "group_diff_test",
                "args": {
                    "metric": params["metric_variable"],
                    "group": params["group_variable"],
                    "method": params.get("method", "perm_test"),
                    "effect": params.get("effect_measure", "cliffs_delta")
                },
                "data_spec": {"slice_spec": {}, "n_samples": 1000}
            }
        
        elif test_type == "predictive_test":
            tool_call = {
                "tool": "predictive_test",
                "args": {
                    "features": params["feature_variables"],
                    "target": params["target_variable"],
                    "model": params.get("model", "logreg"),
                    "metric": params.get("metric", "auc")
                },
                "data_spec": {"slice_spec": {}, "n_samples": 1000}
            }
        
        else:
            return {
                "status": "error",
                "error": f"Test type {test_type} not yet implemented in executor"
            }
        
        # Execute via tool router
        result = self.tool_router.execute(tool_call)
        
        if result["status"] != "success":
            return {
                "test_id": test_spec.get("test_id"),
                "status": "failed",
                "error": result.get("error"),
                "verdict": "inconclusive"
            }
        
        # Determine verdict based on thresholds
        test_result = result["result"]
        verdict = self._determine_verdict(test_result, thresholds)
        
        return {
            "test_id": test_spec.get("test_id"),
            "test_type": test_type,
            "status": "success",
            "numbers": test_result,
            "verdict": verdict,
            "thresholds_used": thresholds
        }
    
    def _determine_verdict(self, result: dict, thresholds: dict) -> str:
        """Determine verdict based on result and thresholds."""
        if result.get("status") == "error":
            return "inconclusive"
        
        effect_size = abs(result.get("effect_size", result.get("gain", 0.0)))
        p_value = result.get("p_value", 1.0)
        
        min_effect = thresholds.get("min_effect_size", 0.2)
        max_p = thresholds.get("max_p_value", 0.05)
        
        if effect_size >= min_effect and p_value <= max_p:
            return "supported"
        elif effect_size < min_effect * 0.5 or p_value > 0.2:
            return "rejected"
        else:
            return "inconclusive"
