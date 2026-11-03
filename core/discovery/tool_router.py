# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tool router for safe execution of analysis primitives."""

from typing import Any, Optional
import time
import signal
import hashlib

from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery import experiments


class TimeoutException(Exception):
    """Raised when a tool execution times out."""

    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Tool execution timed out")


class ToolRouter:
    """Routes tool calls to safe analysis primitives."""

    def __init__(
        self,
        dataset: DatasetAdapter,
        allowed_tools: list[str],
        max_samples: int = 10000,
        timeout: int = 300,
        use_local_execution: bool = True,
        model_url: str | None = None,
        model_name: str = "claude-sonnet-4-5",
    ):
        self.dataset = dataset
        self.allowed_tools = set(allowed_tools)
        self.max_samples = max_samples
        self.timeout = timeout
        self.use_local_execution = use_local_execution
        self.model_url = model_url
        self.model_name = model_name

        # Map tool names to functions
        self.tool_registry = {
            "corr_test": experiments.corr_test,
            "group_diff_test": experiments.group_diff_test,
            "predictive_test": experiments.predictive_test,
            "cluster_and_enrich": experiments.cluster_and_enrich,
            "stratified_retest": experiments.stratified_retest,
            "visual_attribute_test": experiments.visual_attribute_test,
            "visual_group_comparison": experiments.visual_group_comparison,
        }

    def execute(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool call.

        Args:
            tool_call: Dict with "tool" name, "args" dict, and "data_spec" dict

        Returns:
            Dict with "status", "result", and "provenance"
        """
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        data_spec = tool_call.get("data_spec", {})

        # Validate tool
        if tool_name not in self.allowed_tools:
            return {
                "status": "error",
                "error": f"Tool '{tool_name}' not in allowed list: {self.allowed_tools}",
            }

        if tool_name not in self.tool_registry:
            return {"status": "error", "error": f"Tool '{tool_name}' not registered"}

        try:
            # Load data
            n_samples = data_spec.get("n_samples", self.max_samples)
            n_samples = min(n_samples, self.max_samples)
            slice_spec = data_spec.get("slice_spec", {})

            data = self.dataset.sample(slice_spec, n_samples)

            if len(data) == 0:
                return {"status": "error", "error": "No data returned from slice_spec"}

            # Prepare arguments based on tool
            tool_fn = self.tool_registry[tool_name]

            # Extract variables from args and map to data columns
            if tool_name == "corr_test":
                x_var = args.get("x")
                y_var = args.get("y")
                if x_var not in data.columns or y_var not in data.columns:
                    return {
                        "status": "error",
                        "error": f"Variables {x_var}, {y_var} not in data",
                    }
                args_prepared = {
                    "x": data[x_var].values,
                    "y": data[y_var].values,
                    "method": args.get("method", "spearman"),
                    "bootstrap_ci": args.get("bootstrap_ci", True),
                }
            elif tool_name == "group_diff_test":
                metric_var = args.get("metric")
                group_var = args.get("group")
                if metric_var not in data.columns or group_var not in data.columns:
                    return {
                        "status": "error",
                        "error": f"Variables {metric_var}, {group_var} not in data",
                    }
                args_prepared = {
                    "metric": data[metric_var].values,
                    "group": data[group_var].values,
                    "method": args.get("method", "perm_test"),
                    "effect": args.get("effect", "cliffs_delta"),
                }
            elif tool_name == "predictive_test":
                feature_vars = args.get("features", [])
                target_var = args.get("target")
                if not all(f in data.columns for f in feature_vars):
                    return {
                        "status": "error",
                        "error": f"Not all features in data: {feature_vars}",
                    }
                if target_var not in data.columns:
                    return {
                        "status": "error",
                        "error": f"Target {target_var} not in data",
                    }
                args_prepared = {
                    "X": data[feature_vars].values,
                    "y": data[target_var].values,
                    "model": args.get("model", "logreg"),
                    "metric": args.get("metric", "auc"),
                }
            elif tool_name == "visual_attribute_test":
                n_imgs = args.get("n_images", 100)
                group_col = args.get("group_col", "kingdom")
                if hasattr(self.dataset, "get_sample_images_with_labels"):
                    imgs, labels = self.dataset.get_sample_images_with_labels(
                        n=n_imgs, seed=args.get("seed", 42), max_size=512
                    )
                    args_prepared = {
                        "images_b64": imgs,
                        "image_labels": labels,
                        "attribute_query": args.get(
                            "attribute_query",
                            "habitat_type: open / forest / wetland / urban",
                        ),
                        "group_col": group_col,
                        "model_name": args.get("model_name", self.model_name),
                        "model_url": self.model_url,
                    }
                else:
                    return {
                        "status": "error",
                        "error": "Dataset does not support image access",
                    }
            elif tool_name == "visual_group_comparison":
                group_col = args.get("group_col", "kingdom")
                group_a_val = args.get("group_a")
                group_b_val = args.get("group_b")
                if not group_a_val or not group_b_val:
                    return {"status": "error", "error": "group_a and group_b required"}
                if hasattr(self.dataset, "get_sample_images_with_labels"):
                    self.dataset._slice_spec_override = {group_col: [group_a_val]}
                    imgs_a, _ = self.dataset.get_sample_images_with_labels(
                        n=10, seed=42, max_size=512
                    )
                    self.dataset._slice_spec_override = {group_col: [group_b_val]}
                    imgs_b, _ = self.dataset.get_sample_images_with_labels(
                        n=10, seed=43, max_size=512
                    )
                    self.dataset._slice_spec_override = None
                    args_prepared = {
                        "group_a_images": imgs_a,
                        "group_b_images": imgs_b,
                        "group_a_name": str(group_a_val),
                        "group_b_name": str(group_b_val),
                        "comparison_query": args.get(
                            "comparison_query", "visual differences"
                        ),
                        "model_name": args.get("model_name", self.model_name),
                        "model_url": self.model_url,
                    }
                else:
                    return {
                        "status": "error",
                        "error": "Dataset does not support image access",
                    }
            else:
                args_prepared = args

            # Execute with timeout
            start_time = time.time()

            # Set timeout (Unix only)
            if self.timeout > 0:
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.timeout)
                except (ValueError, OSError):
                    # Windows or threading context doesn't support SIGALRM
                    pass

            result = tool_fn(**args_prepared)

            # Cancel timeout
            if self.timeout > 0:
                try:
                    signal.alarm(0)
                except (ValueError, OSError):
                    pass

            execution_time = time.time() - start_time

            # Add provenance
            provenance = {
                "tool": tool_name,
                "args": args,
                "data_spec": data_spec,
                "dataset_version": self.dataset.dataset_version,
                "n_samples_used": len(data),
                "execution_time": execution_time,
                "timestamp": time.time(),
            }

            return {"status": "success", "result": result, "provenance": provenance}

        except TimeoutException:
            return {
                "status": "error",
                "error": f"Tool execution timed out after {self.timeout}s",
            }
        except Exception as e:
            return {"status": "error", "error": f"Tool execution failed: {str(e)}"}
