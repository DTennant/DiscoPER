# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic experiment executor."""

from typing import Any
import hashlib
import json

from core.discovery.types import Experiment, Result
from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.tool_router import ToolRouter


class ExperimentRunner:
    """Executes experiments deterministically without LLM."""

    def __init__(self):
        """Initialize experiment runner."""
        pass

    def run(
        self, experiment: Experiment, dataset: DatasetAdapter, tool_router: ToolRouter
    ) -> Result:
        """
        Execute an experiment.

        Args:
            experiment: Experiment specification
            dataset: Dataset adapter
            tool_router: Tool router for executing analysis primitives

        Returns:
            Result object with status, metrics, and provenance
        """
        try:
            # Parse test plan into tool calls
            # For MVP, test_plan is assumed to be a single tool call or list of tool calls
            test_plan = experiment.method  # Simple MVP: method is just the tool name

            # Prepare tool call
            tool_call = {
                "tool": experiment.method,
                "args": experiment.feature_spec,  # For MVP, feature_spec contains args
                "data_spec": {
                    "slice_spec": experiment.dataset_slice_spec,
                    "n_samples": experiment.budget.get("max_samples", 1000)
                    if experiment.budget
                    else 1000,
                },
            }

            # Execute tool
            execution_result = tool_router.execute(tool_call)

            if execution_result["status"] != "success":
                # Execution failed
                return Result(
                    experiment_id=experiment.id,
                    status="fail",
                    numbers={},
                    notes=execution_result.get("error", "Unknown error"),
                    warnings=[execution_result.get("error", "Unknown error")],
                    repro={"experiment_id": experiment.id},
                )

            # Extract metrics from result
            result_data = execution_result["result"]
            numbers = {}

            if result_data.get("status") == "success":
                # Extract relevant metrics
                if "effect_size" in result_data:
                    numbers["effect_size"] = result_data["effect_size"]
                if "p_value" in result_data:
                    numbers["p_value"] = result_data["p_value"]
                if "score" in result_data:
                    numbers["score"] = result_data["score"]
                if "gain" in result_data:
                    numbers["gain"] = result_data["gain"]
                if "baseline_score" in result_data:
                    numbers["baseline_score"] = result_data["baseline_score"]
                if "ci_lower" in result_data:
                    numbers["ci_lower"] = result_data["ci_lower"]
                if "ci_upper" in result_data:
                    numbers["ci_upper"] = result_data["ci_upper"]

                def _json_safe(obj):
                    """Convert numpy types to JSON-serializable Python types."""
                    if isinstance(obj, dict):
                        return {str(k): _json_safe(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        return [_json_safe(v) for v in obj]
                    try:
                        import numpy as np

                        if isinstance(obj, (np.integer,)):
                            return int(obj)
                        if isinstance(obj, (np.floating,)):
                            return float(obj)
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                    except ImportError:
                        pass
                    return obj

                for extra_key in (
                    "evidence_images",
                    "vlm_labels_distribution",
                    "contingency",
                    "differences",
                    "summary",
                ):
                    if extra_key in result_data:
                        numbers[extra_key] = _json_safe(result_data[extra_key])

                status = "success"
                notes = f"Experiment completed successfully with {experiment.method}"
                warnings = []
            else:
                status = "fail"
                notes = result_data.get("error", "Tool execution failed")
                warnings = [notes]

            # Compute reproducibility hash
            repro_dict = {
                "experiment_id": experiment.id,
                "hypothesis_id": experiment.hypothesis_id,
                "dataset_version": dataset.dataset_version,
                "method": experiment.method,
                "slice_spec": experiment.dataset_slice_spec,
                "feature_spec": experiment.feature_spec,
                "provenance": execution_result.get("provenance", {}),
            }
            repro_hash = hashlib.sha256(
                json.dumps(repro_dict, sort_keys=True).encode()
            ).hexdigest()[:16]
            repro_dict["code_hash"] = repro_hash

            return Result(
                experiment_id=experiment.id,
                status=status,
                numbers=numbers,
                notes=notes,
                warnings=warnings if warnings else None,
                repro=repro_dict,
            )

        except Exception as e:
            return Result(
                experiment_id=experiment.id,
                status="fail",
                numbers={},
                notes=f"Experiment execution failed: {str(e)}",
                warnings=[str(e)],
                repro={"experiment_id": experiment.id, "error": str(e)},
            )

    def run_with_validation(
        self,
        experiment: Experiment,
        dataset: DatasetAdapter,
        tool_router: ToolRouter,
        splits: list[str] = ["train", "val"],
    ) -> dict[str, Result]:
        """
        Run experiment on multiple splits for validation.

        Args:
            experiment: Experiment specification
            dataset: Dataset adapter (must support set_active_split)
            tool_router: Tool router for executing analysis primitives
            splits: List of splits to run on (e.g., ["train", "val", "test"])

        Returns:
            Dictionary mapping split name to Result object
        """
        results = {}

        # Save original split
        if hasattr(dataset, "active_split"):
            original_split = dataset.active_split
        else:
            original_split = None

        for split in splits:
            # Switch to this split
            if hasattr(dataset, "set_active_split"):
                dataset.set_active_split(split)

            # Run experiment on this split
            result = self.run(experiment, dataset, tool_router)

            # Add split info to result notes
            if result.status == "success":
                result.notes = f"{result.notes} (on {split} split)"

            # Store result
            results[split] = result

        # Restore original split
        if original_split and hasattr(dataset, "set_active_split"):
            dataset.set_active_split(original_split)

        return results
