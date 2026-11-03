"""
Claude Code Discovery Runner.

Orchestrates the discovery pipeline using Claude Code agents instead of
LiteLLM-based agents.  The overall loop mirrors DiscoveryMVPRunner:

    Hypothesis → Experiment (design + execute) → Insight → Meta-Discovery

but each LLM step is performed by a Claude Code subprocess with
read-only data access (Bash for Python execution, Read for file access).

Analysis code from each agent is saved alongside the insight for
reproducibility.
"""

import os
import json
from typing import Any, Optional

from core.claude_code_backend import ClaudeCodeBackend
from core.discovery.cc_hypothesis_generator import CCHypothesisGenerator
from core.discovery.cc_experiment_executor import CCExperimentExecutor
from core.discovery.cc_meta_discovery import CCMetaDiscovery
from core.discovery.insight_evaluator import InsightEvaluator
from core.discovery.insight_store import InsightStore
from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.logger import DiscoveryLogger
from core.discovery.report_generator import ReportGenerator
from core.discovery.types import Hypothesis, Experiment, Result, Insight, MetaInsight


class CCDiscoveryRunner:
    """Discovery runner using Claude Code agents for each LLM step."""

    def __init__(
        self,
        workspace_root: str,
        hypothesis_generator: CCHypothesisGenerator,
        experiment_executor: CCExperimentExecutor,
        insight_evaluator: InsightEvaluator,
        insight_store: InsightStore,
        dataset: DatasetAdapter,
        max_retries: int = 3,
        meta_discovery: Optional[CCMetaDiscovery] = None,
        meta_step_frequency: int = 5,
        meta_hypotheses_per_step: int = 2,
        enable_context_passing: bool = True,
        data_path: Optional[str] = None,
    ):
        """
        Args:
            workspace_root: Root directory for outputs.
            hypothesis_generator: CC hypothesis generation agent.
            experiment_executor: CC experiment design + execution agent.
            insight_evaluator: Deterministic insight evaluator (unchanged).
            insight_store: Parquet-based insight store.
            dataset: Dataset adapter.
            max_retries: Max retries per agent call.
            meta_discovery: Optional CC meta-discovery agent.
            meta_step_frequency: Run meta-discovery every N iterations.
            meta_hypotheses_per_step: Guided hypotheses per meta-discovery step.
            enable_context_passing: Pass insights/history to hypothesis generator.
            data_path: Path to the data parquet file for CC agents.
        """
        self.workspace_root = workspace_root
        self.hypothesis_generator = hypothesis_generator
        self.experiment_executor = experiment_executor
        self.insight_evaluator = insight_evaluator
        self.insight_store = insight_store
        self.dataset = dataset
        self.max_retries = max_retries

        self.meta_discovery = meta_discovery
        self.meta_step_frequency = meta_step_frequency
        self.meta_hypotheses_per_step = meta_hypotheses_per_step

        self.enable_context_passing = enable_context_passing
        self.tested_hypotheses: list[str] = []
        self.meta_insights: list[MetaInsight] = []

        # Resolve data path
        self.data_path = data_path
        if self.data_path is None and hasattr(dataset, "data_path"):
            self.data_path = dataset.data_path

        os.makedirs(workspace_root, exist_ok=True)
        self.logger = DiscoveryLogger(os.path.join(workspace_root, "events.jsonl"))

    async def run(self, n_iterations: int = 1):
        """Run the discovery loop (async wrapper for launcher compatibility)."""
        print(f"Starting Claude Code Discovery with {n_iterations} iterations")
        print(f"Workspace: {self.workspace_root}")
        if self.meta_discovery:
            print(f"Meta-discovery enabled (every {self.meta_step_frequency} iterations)")

        meta_hypotheses_tested = 0

        for i in range(n_iterations):
            print(f"\n=== Iteration {i + 1}/{n_iterations} ===")

            # --- Meta-discovery step ---
            should_run_meta = (
                self.meta_discovery is not None
                and (i + 1) % self.meta_step_frequency == 0
                and len(self.insight_store) >= 2
            )
            if should_run_meta:
                print(f"\n--- Meta-Discovery Step ---")
                meta_hyps = self._run_meta_discovery()
                if meta_hyps:
                    print(f"Generated {len(meta_hyps)} meta-hypotheses")
                    for mh in meta_hyps:
                        mi = self._execute_hypothesis(mh, i, is_meta=True)
                        if mi:
                            meta_hypotheses_tested += 1
                print(f"--- End Meta-Discovery Step ---\n")

            # --- Base discovery ---
            dataset_summary = self._get_dataset_summary()

            recent_insights = None
            previous_hypotheses = None
            prior_knowledge_text = None

            if self.enable_context_passing:
                recent_insights = list(self.insight_store.insights)[-10:]
                previous_hypotheses = self.tested_hypotheses[-20:]
                if hasattr(self.dataset, "get_prior_knowledge_prompt"):
                    prior_knowledge_text = self.dataset.get_prior_knowledge_prompt()

            # Working dir for this iteration
            iter_dir = os.path.join(self.workspace_root, f"iteration_{i}")
            os.makedirs(iter_dir, exist_ok=True)

            hypotheses = self.hypothesis_generator.generate(
                dataset_summary=dataset_summary,
                n_hypotheses=1,
                max_retries=self.max_retries,
                recent_insights=recent_insights,
                previous_hypotheses=previous_hypotheses,
                prior_knowledge_text=prior_knowledge_text,
                data_path=self.data_path,
                working_dir=iter_dir,
            )

            if not hypotheses:
                print(f"Failed to generate hypothesis in iteration {i + 1}")
                continue

            hypothesis = hypotheses[0]
            self.tested_hypotheses.append(hypothesis.statement)
            self._execute_hypothesis(hypothesis, i, is_meta=False)

        # Final meta-discovery
        if self.meta_discovery and len(self.insight_store) >= 3:
            print(f"\n=== Final Meta-Discovery Step ===")
            final_hyps = self._run_meta_discovery()
            if final_hyps:
                print(f"Generated {len(final_hyps)} final meta-hypotheses")
                for mh in final_hyps:
                    self._execute_hypothesis(mh, n_iterations, is_meta=True)
                    meta_hypotheses_tested += 1

        # Wrap up
        self._generate_report()
        self.insight_store.save()
        self._save_meta_insights()

        base_count = len([i for i in self.insight_store.insights if not getattr(i, "is_meta", False)])
        guided_count = len([i for i in self.insight_store.insights if getattr(i, "is_meta", False)])
        print(f"\nDiscovery complete!")
        print(f"  Base insights: {base_count}")
        print(f"  Meta-guided insights: {guided_count}")
        print(f"  Meta-insights (observations): {len(self.meta_insights)}")
        if meta_hypotheses_tested > 0:
            print(f"  Meta-hypotheses tested: {meta_hypotheses_tested}")
        print(f"Report: {os.path.join(self.workspace_root, 'report.md')}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _execute_hypothesis(
        self,
        hypothesis: Hypothesis,
        iteration: int,
        is_meta: bool = False,
    ) -> Optional[Insight]:
        """Execute a single hypothesis through experiment + evaluation."""
        prefix = "[META] " if is_meta else ""
        print(f"\n{prefix}Hypothesis: {hypothesis.statement}")

        event_data = {"hypothesis": hypothesis.to_dict()}
        if is_meta:
            event_data["is_meta"] = True
        self.logger.log_event("hypothesis_proposed", event_data)

        # Working dir for this experiment
        suffix = "_meta" if is_meta else ""
        exp_dir = os.path.join(self.workspace_root, f"iteration_{iteration}{suffix}")
        os.makedirs(exp_dir, exist_ok=True)

        dataset_summary = self._get_dataset_summary()
        experiment, results, analysis_code = self.experiment_executor.execute_hypothesis(
            hypothesis=hypothesis,
            dataset_summary=dataset_summary,
            data_path=self.data_path,
            working_dir=exp_dir,
            max_retries=self.max_retries,
        )

        if experiment is None or not results:
            print(f"{prefix}Failed to execute experiment")
            return None

        self.logger.log_event("experiment_executed", {
            "experiment": experiment.to_dict(),
            "is_meta": is_meta,
        })

        # Log results per split
        for split, result in results.items():
            print(f"{prefix}Result ({split}): {result.status}, metrics={result.numbers}")
            self.logger.log_event(f"experiment_result_{split}", {
                "result": result.to_dict(),
                "is_meta": is_meta,
            })

        # Evaluate -> Insight
        insight = self.insight_evaluator.evaluate_with_validation(
            results, hypothesis, experiment
        )
        if is_meta:
            insight.is_meta = True

        print(f"{prefix}Insight: {insight.verdict} (strength={insight.strength:.3f})")

        self.insight_store.add_insight(insight)
        self.logger.log_event("insight_added", {
            "insight": insight.to_dict(),
            "is_meta": is_meta,
        })

        # Collect all code (exploration + analysis) for this insight
        exploration_code = ""
        if hypothesis.test_plan and isinstance(hypothesis.test_plan, dict):
            exploration_code = hypothesis.test_plan.get("exploration_code", "")

        # Persist discovery results + code artifacts
        self._save_discovery_results(
            iteration, hypothesis, experiment, results, insight,
            suffix=suffix,
            exploration_code=exploration_code,
            analysis_code=analysis_code,
        )
        return insight

    def _run_meta_discovery(self) -> list[Hypothesis]:
        """Run meta-discovery to generate meta-insights + guided hypotheses."""
        if not self.meta_discovery:
            return []

        dataset_summary = self._get_dataset_summary()
        meta_dir = os.path.join(self.workspace_root, "meta_discovery")
        os.makedirs(meta_dir, exist_ok=True)

        try:
            meta_insights, hypotheses = self.meta_discovery.discover_meta_patterns(
                insight_store=self.insight_store,
                dataset_summary=dataset_summary,
                n_meta_hypotheses=self.meta_hypotheses_per_step,
                max_retries=self.max_retries,
                working_dir=meta_dir,
            )

            if meta_insights:
                self.meta_insights.extend(meta_insights)
                print(f"\nStored {len(meta_insights)} meta-insights:")
                for mi in meta_insights:
                    print(f"   [{mi.meta_type}] {mi.observation[:60]}...")

            self.logger.log_event("meta_discovery_step", {
                "n_base_insights_analyzed": len(self.insight_store),
                "n_meta_insights_generated": len(meta_insights),
                "n_guided_hypotheses_generated": len(hypotheses),
            })

            return hypotheses

        except Exception as e:
            print(f"Meta-discovery failed: {e}")
            self.logger.log_event("meta_discovery_error", {"error": str(e)})
            return []

    def _get_dataset_summary(self) -> dict[str, Any]:
        if hasattr(self.dataset, "get_summary"):
            return self.dataset.get_summary()
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
        exploration_code: str = "",
        analysis_code: str = "",
    ):
        iteration_dir = os.path.join(self.workspace_root, f"iteration_{iteration}{suffix}")
        os.makedirs(iteration_dir, exist_ok=True)

        discovery_json = {
            "iteration": iteration,
            "is_meta": suffix == "_meta",
            "hypothesis": hypothesis.to_dict(),
            "experiment": experiment.to_dict(),
            "results": {s: r.to_dict() for s, r in results.items()},
            "insight": insight.to_dict(),
        }
        with open(os.path.join(iteration_dir, "discovery.json"), "w") as f:
            json.dump(discovery_json, f, indent=2)

        # Save the code that produced this insight
        if exploration_code or analysis_code:
            code_path = os.path.join(iteration_dir, "analysis_code.py")
            with open(code_path, "w") as f:
                f.write("# Auto-saved code from Claude Code discovery pipeline\n")
                f.write(f"# Iteration: {iteration}\n")
                f.write(f"# Hypothesis: {hypothesis.statement[:120]}\n")
                f.write(f"# Verdict: {insight.verdict}\n\n")
                if exploration_code:
                    f.write("# === Exploration Code ===\n")
                    f.write(exploration_code)
                    f.write("\n\n")
                if analysis_code:
                    f.write("# === Analysis Code ===\n")
                    f.write(analysis_code)
                    f.write("\n")
            print(f"  Code saved to: {code_path}")

    def _save_meta_insights(self):
        if not self.meta_insights:
            return
        path = os.path.join(self.workspace_root, "meta_insights.json")
        data = {
            "description": "Meta-insights: observations about patterns across base insights.",
            "meta_insights": [mi.to_dict() for mi in self.meta_insights],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Meta-insights saved to: {path}")

    def _generate_report(self):
        report_generator = ReportGenerator()
        report_generator.generate(
            self.insight_store,
            os.path.join(self.workspace_root, "events.jsonl"),
            os.path.join(self.workspace_root, "report.md"),
            meta_insights=self.meta_insights,
        )

    def shutdown(self):
        """Graceful cleanup."""
        print("Shutting down CC Discovery Runner...")
        if hasattr(self, "insight_store"):
            self.insight_store.save()
        if self.meta_insights:
            self._save_meta_insights()
        print("Shutdown complete.")
