# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""MVP Discovery Runner - orchestrates the discovery pipeline."""

import os
import json
from typing import Any, Optional

from core.discovery.hypothesis_generator import HypothesisGeneratorAgent
from core.discovery.experiment_planner import ExperimentPlannerAgent
from core.discovery.experiment_runner import ExperimentRunner
from core.discovery.insight_evaluator import InsightEvaluator
from core.discovery.insight_store import InsightStore
from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.tool_router import ToolRouter
from core.discovery.logger import DiscoveryLogger
from core.discovery.report_generator import ReportGenerator
from core.discovery.meta_discovery import MetaDiscoveryAgent
from core.discovery.types import Hypothesis, Experiment, Result, Insight, MetaInsight


class DiscoveryMVPRunner:
    """MVP Discovery Runner - simplified, synchronous execution with optional meta-discovery."""

    def __init__(
        self,
        workspace_root: str,
        hypothesis_generator: HypothesisGeneratorAgent,
        experiment_planner: ExperimentPlannerAgent,
        experiment_runner: ExperimentRunner,
        insight_evaluator: InsightEvaluator,
        insight_store: InsightStore,
        dataset: DatasetAdapter,
        tool_router: ToolRouter,
        max_retries: int = 3,
        meta_discovery_agent: Optional[MetaDiscoveryAgent] = None,
        meta_step_frequency: int = 5,
        meta_hypotheses_per_step: int = 2,
        enable_context_passing: bool = False,
        use_vision: bool = False,
        vision_n_images: int = 5,
        vision_image_size: int = 512,
    ):
        """
        Initialize discovery runner.

        Args:
            workspace_root: Root directory for outputs
            hypothesis_generator: Agent for generating hypotheses
            experiment_planner: Agent for planning experiments
            experiment_runner: Deterministic experiment executor
            insight_evaluator: Result → Insight converter
            insight_store: Store for insights
            dataset: Dataset adapter
            tool_router: Tool router for experiment execution
            max_retries: Maximum retries for LLM calls
            meta_discovery_agent: Optional agent for meta-level discovery
            meta_step_frequency: Run meta-discovery every N iterations
            meta_hypotheses_per_step: Number of meta-hypotheses to generate per step
            enable_context_passing: Pass insights and previous hypotheses to generator
        """
        self.workspace_root = workspace_root
        self.hypothesis_generator = hypothesis_generator
        self.experiment_planner = experiment_planner
        self.experiment_runner = experiment_runner
        self.insight_evaluator = insight_evaluator
        self.insight_store = insight_store
        self.dataset = dataset
        self.tool_router = tool_router
        if self.tool_router.dataset is not self.dataset:
            self.tool_router.dataset = self.dataset
        self.max_retries = max_retries

        # Meta-discovery configuration
        self.meta_discovery_agent = meta_discovery_agent
        self.meta_step_frequency = meta_step_frequency
        self.meta_hypotheses_per_step = meta_hypotheses_per_step

        # Context passing configuration
        self.enable_context_passing = enable_context_passing
        self.tested_hypotheses: list[str] = []

        self.use_vision = use_vision
        self.vision_n_images = vision_n_images
        self.vision_image_size = vision_image_size

        # Store meta-insights (observations about patterns, not experimental results)
        self.meta_insights: list[MetaInsight] = []

        self.iteration_diversity_stats: list[dict] = []
        self._concept_tuples: list = []

        # Create workspace
        os.makedirs(workspace_root, exist_ok=True)

        # Initialize logger
        self.logger = DiscoveryLogger(os.path.join(workspace_root, "events.jsonl"))

    async def run(self, n_iterations: int = 1):
        """
        Run the discovery loop (async wrapper for compatibility with launch_scientist.py).

        Args:
            n_iterations: Number of iterations
        """
        print(f"Starting Discovery MVP with {n_iterations} iterations")
        print(f"Workspace: {self.workspace_root}")
        if self.meta_discovery_agent:
            print(
                f"Meta-discovery enabled (every {self.meta_step_frequency} iterations)"
            )
        if self.use_vision:
            print(
                f"Vision enabled ({self.vision_n_images} images, {self.vision_image_size}px)"
            )

        # Track meta-discovery stats
        meta_hypotheses_tested = 0

        for i in range(n_iterations):
            print(f"\n=== Iteration {i + 1}/{n_iterations} ===")

            # Check if we should run meta-discovery step
            should_run_meta = (
                self.meta_discovery_agent is not None
                and (i + 1) % self.meta_step_frequency == 0
                and len(self.insight_store) >= 2
            )

            if should_run_meta:
                print(f"\n--- Meta-Discovery Step ---")
                meta_hypotheses = self._run_meta_discovery()

                if meta_hypotheses:
                    print(f"Generated {len(meta_hypotheses)} meta-hypotheses")
                    # Execute meta-hypotheses
                    for meta_hyp in meta_hypotheses:
                        meta_insight = self._execute_hypothesis(
                            meta_hyp, i, is_meta=True
                        )
                        if meta_insight:
                            meta_hypotheses_tested += 1
                    print(f"--- End Meta-Discovery Step ---\n")

            # 1. Generate 1 hypothesis
            dataset_summary = self._get_dataset_summary()

            # Get context for hypothesis generation if enabled
            recent_insights = None
            previous_hypotheses = None
            prior_knowledge_text = None

            if self.enable_context_passing:
                recent_insights = list(self.insight_store.insights)[-10:]
                previous_hypotheses = self.tested_hypotheses[-20:]
                if hasattr(self.dataset, "get_prior_knowledge_prompt"):
                    prior_knowledge_text = self.dataset.get_prior_knowledge_prompt()

            sample_images = None
            image_context = ""
            if self.use_vision:
                if hasattr(self.dataset, "get_sample_images_with_labels"):
                    sample_images, image_labels = (
                        self.dataset.get_sample_images_with_labels(
                            n=self.vision_n_images,
                            seed=42 + i,
                            max_size=self.vision_image_size,
                        )
                    )
                    if sample_images and image_labels:
                        label_lines = []
                        for j, lbl in enumerate(image_labels, 1):
                            label_lines.append(
                                f"Image {j}: {lbl['common_name']} ({lbl['species_name']}), "
                                f"lat={lbl['latitude']}, lon={lbl['longitude']}, "
                                f"season={lbl['season']}, month={lbl['month']}"
                            )
                        image_context = (
                            "\n## Image Labels\n" + "\n".join(label_lines) + "\n"
                        )
                        print(
                            f"  [Vision] Attaching {len(sample_images)} labeled images"
                        )
                elif hasattr(self.dataset, "get_sample_images_base64"):
                    sample_images = self.dataset.get_sample_images_base64(
                        n=self.vision_n_images,
                        seed=42 + i,
                        max_size=self.vision_image_size,
                    )
                    if sample_images:
                        print(
                            f"  [Vision] Attaching {len(sample_images)} sample images to hypothesis prompt"
                        )

            effective_prior_text = prior_knowledge_text or ""
            if image_context:
                effective_prior_text = effective_prior_text + "\n" + image_context

            hypotheses = self.hypothesis_generator.generate(
                dataset_summary,
                n_hypotheses=1,
                max_retries=self.max_retries,
                recent_insights=recent_insights,
                previous_hypotheses=previous_hypotheses,
                prior_knowledge_text=effective_prior_text or None,
                images=sample_images,
            )

            if not hypotheses:
                print(f"Failed to generate hypothesis in iteration {i + 1}")
                continue

            hypothesis = hypotheses[0]
            self.tested_hypotheses.append(hypothesis.statement)  # Track for context
            self._execute_hypothesis(hypothesis, i, is_meta=False)

        # Run final meta-discovery if we have enough insights
        if self.meta_discovery_agent and len(self.insight_store) >= 3:
            print(f"\n=== Final Meta-Discovery Step ===")
            final_meta_hypotheses = self._run_meta_discovery()
            if final_meta_hypotheses:
                print(f"Generated {len(final_meta_hypotheses)} final meta-hypotheses")
                for meta_hyp in final_meta_hypotheses:
                    self._execute_hypothesis(meta_hyp, n_iterations, is_meta=True)
                    meta_hypotheses_tested += 1

        # Generate report
        self._generate_report()

        # Save insights
        self.insight_store.save()

        if self.iteration_diversity_stats:
            stats_path = os.path.join(self.workspace_root, "diversity_curve.json")
            with open(stats_path, "w") as f:
                json.dump(self.iteration_diversity_stats, f, indent=2)
            print(f"Diversity curve saved to {stats_path}")

        # Save meta-insights (these are observations, not tested experimentally)
        self._save_meta_insights()

        print(f"\nDiscovery complete!")
        print(
            f"  Base insights: {len([i for i in self.insight_store.insights if not getattr(i, 'is_meta', False)])}"
        )
        print(
            f"  Meta-guided insights: {len([i for i in self.insight_store.insights if getattr(i, 'is_meta', False)])}"
        )
        print(f"  Meta-insights (observations): {len(self.meta_insights)}")
        if meta_hypotheses_tested > 0:
            print(f"  Meta-hypotheses tested: {meta_hypotheses_tested}")
        print(f"Report: {os.path.join(self.workspace_root, 'report.md')}")

    def _verify_alignment(
        self, hypothesis: Hypothesis, experiment: Experiment, dataset_summary: dict
    ) -> Experiment:
        """Check if experiment code aligns with hypothesis claim. Fix if misaligned."""
        claim = hypothesis.statement
        method = experiment.method
        feature_spec = experiment.feature_spec or {}
        slice_spec = experiment.dataset_slice_spec or {}

        metric = feature_spec.get("metric", feature_spec.get("x", ""))
        group = feature_spec.get(
            "group", feature_spec.get("y", feature_spec.get("group_col", ""))
        )

        # Quick heuristic checks
        issues = []

        # Check 1: If claim mentions specific species but code groups by class/kingdom
        claim_lower = claim.lower()
        species_mentioned = any(
            sp in claim_lower
            for sp in [
                "vulpes",
                "danaus",
                "hirundo",
                "turdus",
                "amanita",
                "cervus",
                "helianthus",
                "taraxacum",
            ]
        )
        if species_mentioned and group in ("class_name", "kingdom"):
            issues.append(f"Claim mentions specific species but groups by '{group}'")

        # Check 2: If claim is about seasonal shift/change but metric is raw latitude
        if any(w in claim_lower for w in ["shift", "change", "migration", "movement"]):
            if metric in ("latitude", "longitude", "month"):
                if "stratif" not in method:
                    issues.append(
                        f"Claim describes a shift/change but tests raw '{metric}'"
                    )

        # Check 3: If claim compares two specific groups but slice doesn't filter to them
        if not slice_spec and species_mentioned:
            issues.append("Claim about specific species but no data filtering applied")

        if not issues:
            return experiment

        # Use LLM to fix the experiment spec
        try:
            fix_prompt = (
                f"The hypothesis is: '{claim}'\n\n"
                f"The planned experiment uses method='{method}', "
                f"metric='{metric}', group='{group}', slice={json.dumps(slice_spec)}.\n\n"
                f"ALIGNMENT ISSUES:\n" + "\n".join(f"- {i}" for i in issues) + "\n\n"
                f"Fix the experiment to match the hypothesis. Return JSON with: "
                f"method, feature_spec (with metric/group or x/y), dataset_slice_spec.\n"
                f"Available columns: {list(dataset_summary.get('columns', dataset_summary.get('column_names', [])))}"
            )

            fixed = self.experiment_planner.act_json(
                instruction=fix_prompt,
                output_schema={
                    "type": "object",
                    "properties": {
                        "method": {"type": "string"},
                        "metric_column": {"type": "string"},
                        "group_column": {"type": "string"},
                        "filter_column": {"type": "string"},
                        "filter_values": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["method", "metric_column", "group_column"],
                },
                tool_name="fix_experiment",
                tool_description="Fix experiment to align with hypothesis",
            )

            if fixed.get("metric_column") and fixed.get("group_column"):
                new_method = fixed.get("method", method)
                new_metric = fixed["metric_column"]
                new_group = fixed["group_column"]
                if new_method == "corr_test":
                    new_spec = {"x": new_metric, "y": new_group, "method": "spearman"}
                else:
                    new_spec = {
                        "metric": new_metric,
                        "group": new_group,
                        "method": "perm_test",
                        "effect": "cliffs_delta",
                    }

                new_slice = slice_spec
                if fixed.get("filter_column") and fixed.get("filter_values"):
                    new_slice = {fixed["filter_column"]: fixed["filter_values"]}

                experiment = Experiment.create(
                    hypothesis_id=experiment.hypothesis_id,
                    method=new_method,
                    feature_spec=new_spec,
                    dataset_slice_spec=new_slice,
                )
                print(f"  [Alignment fix] {issues[0][:60]} → fixed")
        except Exception as e:
            print(f"  [Alignment fix failed] {e}")

        return experiment

    def _execute_hypothesis(
        self, hypothesis: Hypothesis, iteration: int, is_meta: bool = False
    ) -> Optional[Insight]:
        """
        Execute a single hypothesis through the full pipeline.

        Args:
            hypothesis: Hypothesis to test
            iteration: Current iteration number
            is_meta: Whether this is a meta-hypothesis

        Returns:
            Generated Insight or None if failed
        """
        prefix = "[META] " if is_meta else ""
        print(f"\n{prefix}Hypothesis: {hypothesis.statement}")

        event_data = {"hypothesis": hypothesis.to_dict()}
        if is_meta:
            event_data["is_meta"] = True
            meta_type = (
                hypothesis.test_plan.get("meta_type", "unknown")
                if hypothesis.test_plan
                else "unknown"
            )
            event_data["meta_type"] = meta_type
        self.logger.log_event("hypothesis_proposed", event_data)

        # Plan experiment
        dataset_summary = self._get_dataset_summary()
        experiments = self.experiment_planner.plan(
            hypothesis, dataset_summary, max_retries=self.max_retries
        )

        if not experiments:
            print(f"{prefix}Failed to plan experiment for hypothesis")
            return None

        experiment = experiments[0]

        # Verify code-claim alignment before execution
        experiment = self._verify_alignment(hypothesis, experiment, dataset_summary)

        print(
            f"{prefix}Experiment: {experiment.method} on {experiment.dataset_slice_spec}"
        )
        self.logger.log_event(
            "experiment_planned",
            {"experiment": experiment.to_dict(), "is_meta": is_meta},
        )

        # Execute experiment with validation
        results = self.experiment_runner.run_with_validation(
            experiment, self.dataset, self.tool_router, splits=["train", "val"]
        )

        # Log results for each split
        for split, result in results.items():
            print(
                f"{prefix}Result ({split}): {result.status}, metrics={result.numbers}"
            )
            self.logger.log_event(
                f"experiment_result_{split}",
                {"result": result.to_dict(), "is_meta": is_meta},
            )

        # Evaluate → Insight
        insight = self.insight_evaluator.evaluate_with_validation(
            results, hypothesis, experiment
        )

        # Mark if this insight was generated from a meta-guided hypothesis
        if is_meta:
            insight.is_meta = True

        print(f"{prefix}Insight: {insight.verdict} (strength={insight.strength:.3f})")

        # Store insight
        self.insight_store.add_insight(insight)
        self.logger.log_event(
            "insight_added", {"insight": insight.to_dict(), "is_meta": is_meta}
        )

        from core.discovery.diversity_metrics import (
            classify_hypothesis,
            compute_diversity,
        )

        concept = classify_hypothesis(hypothesis.statement, hypothesis.variables)
        self._concept_tuples.append(concept)
        if len(self._concept_tuples) >= 2:
            diversity_result = compute_diversity(self._concept_tuples)
            self.iteration_diversity_stats.append(
                {
                    "iteration": iteration,
                    "mean_entropy": diversity_result.mean_dimension_entropy,
                    "concept_coverage": diversity_result.concept_coverage,
                    "variable_coverage": diversity_result.variable_coverage,
                    "redundancy_rate": diversity_result.redundancy_rate,
                    "pairwise_dissimilarity": diversity_result.pairwise_dissimilarity,
                }
            )

        if self.iteration_diversity_stats:
            latest = self.iteration_diversity_stats[-1]
            print(
                f"  [Diversity] entropy={latest['mean_entropy']:.3f} coverage={latest['concept_coverage']:.3f} redundancy={latest['redundancy_rate']:.3f}"
            )

        # Save results
        iter_suffix = f"_meta" if is_meta else ""
        self._save_discovery_results(
            iteration, hypothesis, experiment, results, insight, suffix=iter_suffix
        )

        return insight

    def _run_meta_discovery(self) -> list[Hypothesis]:
        """
        Run meta-discovery to generate meta-insights and guided hypotheses.

        This is a TWO-STEP process:
        1. Analyze base insights -> Generate Meta-Insights (actionable observations)
        2. Use Meta-Insights -> Generate guided hypotheses for testing

        Returns:
            List of hypotheses to test (guided by meta-insights)
        """
        if not self.meta_discovery_agent:
            return []

        dataset_summary = self._get_dataset_summary()

        prior_knowledge_text = None
        if hasattr(self.dataset, "get_prior_knowledge_prompt"):
            prior_knowledge_text = self.dataset.get_prior_knowledge_prompt()

        try:
            meta_insights, hypotheses = (
                self.meta_discovery_agent.discover_meta_patterns(
                    insight_store=self.insight_store,
                    dataset_summary=dataset_summary,
                    n_meta_hypotheses=self.meta_hypotheses_per_step,
                    max_retries=self.max_retries,
                    prior_knowledge_text=prior_knowledge_text,
                )
            )

            # Store meta-insights (these are observations, not tested experimentally)
            if meta_insights:
                self.meta_insights.extend(meta_insights)
                print(
                    f"\n📊 Stored {len(meta_insights)} meta-insights (actionable observations)"
                )
                for mi in meta_insights:
                    print(f"   [{mi.meta_type}] {mi.observation[:60]}...")
                    print(
                        f"      → Recommendation: {mi.actionable_recommendation[:50]}..."
                    )

            self.logger.log_event(
                "meta_discovery_step",
                {
                    "n_base_insights_analyzed": len(self.insight_store),
                    "n_meta_insights_generated": len(meta_insights),
                    "n_guided_hypotheses_generated": len(hypotheses),
                    "meta_insights": [mi.to_dict() for mi in meta_insights],
                    "guided_hypotheses": [h.to_dict() for h in hypotheses],
                },
            )

            return hypotheses

        except Exception as e:
            print(f"Meta-discovery failed: {e}")
            self.logger.log_event("meta_discovery_error", {"error": str(e)})
            return []

    def shutdown(self):
        """Shutdown method for graceful cleanup (called by launch_scientist.py signal handlers)."""
        print("Shutting down Discovery MVP...")
        # Save any pending insights
        if hasattr(self, "insight_store"):
            self.insight_store.save()
        print("Shutdown complete.")

    def _get_dataset_summary(self) -> dict[str, Any]:
        """Get dataset summary for prompts."""
        # For synthetic dataset, use the get_summary method
        if hasattr(self.dataset, "get_summary"):
            return self.dataset.get_summary()

        # Fallback: basic summary
        return {
            "dataset_id": self.dataset.dataset_id,
            "dataset_version": self.dataset.dataset_version,
        }

    def _save_discovery_results(
        self,
        iteration: int,
        hypothesis: Hypothesis,
        experiment: Experiment,
        results: dict[str, Result],
        insight: Insight,
        suffix: str = "",
    ):
        """Save discovery results for an iteration."""
        iteration_dir = os.path.join(
            self.workspace_root, f"iteration_{iteration}{suffix}"
        )
        os.makedirs(iteration_dir, exist_ok=True)

        discovery_json = {
            "iteration": iteration,
            "is_meta": suffix == "_meta",
            "hypothesis": hypothesis.to_dict(),
            "experiment": experiment.to_dict(),
            "results": {split: result.to_dict() for split, result in results.items()},
            "insight": insight.to_dict(),
        }

        with open(os.path.join(iteration_dir, "discovery.json"), "w") as f:
            json.dump(discovery_json, f, indent=2)

    def _save_meta_insights(self):
        """
        Save meta-insights to a JSON file.

        Meta-insights are actionable observations about patterns across base insights.
        They are NOT tested experimentally - they guide future hypothesis generation.
        """
        if not self.meta_insights:
            return

        meta_insights_path = os.path.join(self.workspace_root, "meta_insights.json")

        meta_insights_data = {
            "description": "Meta-insights are observations about patterns across base insights that guide future discovery. They are NOT tested experimentally.",
            "meta_insights": [mi.to_dict() for mi in self.meta_insights],
        }

        with open(meta_insights_path, "w") as f:
            json.dump(meta_insights_data, f, indent=2)

        print(f"Meta-insights saved to: {meta_insights_path}")

    def _generate_report(self):
        """Generate discovery report with clear Level 1/Level 2 distinction."""
        report_generator = ReportGenerator()
        report_generator.generate(
            self.insight_store,
            os.path.join(self.workspace_root, "events.jsonl"),
            os.path.join(self.workspace_root, "report.md"),
            meta_insights=self.meta_insights,  # Pass meta-insights for the report
        )
