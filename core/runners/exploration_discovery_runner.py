# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Exploration-Based Discovery Runner

Uses a two-role system for BASE discovery:
1. ExplorationAgent: Creative free-form code generation to explore data
2. StatisticalTesterAgent: Rigorous test specification with strict signatures

PLUS Meta-Discovery (the core of the system):
- Analyzes accumulated base insights to find higher-level patterns
- Generates meta-insights (actionable observations about the discovery process)
- Creates guided hypotheses based on meta-insights

This separates creative discovery from rigorous validation while maintaining
the meta-discovery loop that is the core of the system.
"""

import os
import json
from typing import Any, Optional
from datetime import datetime

from core.discovery.exploration_agent import ExplorationAgent
from core.discovery.statistical_tester_agent import StatisticalTesterAgent, TestExecutor
from core.discovery.experiment_runner import ExperimentRunner
from core.discovery.insight_evaluator import InsightEvaluator
from core.discovery.insight_store import InsightStore
from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.tool_router import ToolRouter
from core.discovery.logger import DiscoveryLogger
from core.discovery.meta_discovery import MetaDiscoveryAgent
from core.discovery.report_generator import ReportGenerator
from core.discovery.types import Hypothesis, Experiment, Insight, MetaInsight


class ExplorationDiscoveryRunner:
    """
    Discovery runner using the two-role exploration/testing system WITH meta-discovery.
    
    Base Discovery Flow (Level 1):
    1. ExplorationAgent explores data freely → observations + preliminary hypotheses
    2. StatisticalTesterAgent designs rigorous tests → test specifications
    3. TestExecutor runs validated tests → results
    4. InsightEvaluator converts results → BASE insights
    
    Meta-Discovery Flow (Level 2 - THE CORE):
    5. MetaDiscoveryAgent analyzes base insights → meta-insights (observations)
    6. MetaDiscoveryAgent generates guided hypotheses from meta-insights
    7. Guided hypotheses are tested → guided insights
    8. Cycle continues, building on both base and guided insights
    """

    def __init__(
        self,
        workspace_root: str,
        exploration_agent: ExplorationAgent,
        statistical_tester: StatisticalTesterAgent,
        experiment_runner: ExperimentRunner,
        insight_evaluator: InsightEvaluator,
        insight_store: InsightStore,
        dataset: DatasetAdapter,
        tool_router: ToolRouter,
        max_exploration_iterations: int = 3,
        max_tests_per_hypothesis: int = 3,
        # Meta-discovery configuration (THE CORE)
        meta_discovery_agent: Optional[MetaDiscoveryAgent] = None,
        meta_step_frequency: int = 3,
        meta_hypotheses_per_step: int = 2
    ):
        """
        Initialize exploration discovery runner with meta-discovery.
        
        Args:
            workspace_root: Root directory for outputs
            exploration_agent: Agent for free-form data exploration
            statistical_tester: Agent for designing rigorous tests
            experiment_runner: Deterministic experiment executor
            insight_evaluator: Result → Insight converter
            insight_store: Store for insights
            dataset: Dataset adapter
            tool_router: Tool router for experiment execution
            max_exploration_iterations: Max code iterations during exploration
            max_tests_per_hypothesis: Max tests to run per hypothesis
            meta_discovery_agent: Agent for meta-level discovery (CORE)
            meta_step_frequency: Run meta-discovery every N iterations
            meta_hypotheses_per_step: Number of guided hypotheses per meta-step
        """
        self.workspace_root = workspace_root
        self.exploration_agent = exploration_agent
        self.statistical_tester = statistical_tester
        self.experiment_runner = experiment_runner
        self.insight_evaluator = insight_evaluator
        self.insight_store = insight_store
        self.dataset = dataset
        self.tool_router = tool_router
        self.max_exploration_iterations = max_exploration_iterations
        self.max_tests_per_hypothesis = max_tests_per_hypothesis
        
        # Meta-discovery configuration (THE CORE OF THE SYSTEM)
        self.meta_discovery_agent = meta_discovery_agent
        self.meta_step_frequency = meta_step_frequency
        self.meta_hypotheses_per_step = meta_hypotheses_per_step
        
        # Store meta-insights (observations about patterns, not experimental results)
        self.meta_insights: list[MetaInsight] = []
        
        # Test executor bridges test specs to actual execution
        self.test_executor = TestExecutor(tool_router)

        # Create workspace
        os.makedirs(workspace_root, exist_ok=True)

        # Initialize logger
        self.logger = DiscoveryLogger(os.path.join(workspace_root, "events.jsonl"))
        
        # Track all exploration results
        self.exploration_history = []

    async def run(self, n_iterations: int = 1):
        """
        Run the exploration-based discovery loop with meta-discovery.
        
        Args:
            n_iterations: Number of exploration → test cycles
        """
        print(f"Starting Exploration-Based Discovery with {n_iterations} iterations")
        print(f"Workspace: {self.workspace_root}")
        print(f"Mode: Two-role system (Explorer → Tester) + Meta-Discovery")
        
        if self.meta_discovery_agent:
            print(f"Meta-Discovery: ENABLED (every {self.meta_step_frequency} iterations)")
        else:
            print(f"Meta-Discovery: DISABLED (no meta_discovery_agent provided)")
        
        # Track meta-discovery stats
        meta_hypotheses_tested = 0

        for i in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {i+1}/{n_iterations}")
            print(f"{'='*60}")
            
            # ================================================================
            # CHECK: Should we run meta-discovery BEFORE this iteration?
            # Meta-discovery analyzes accumulated insights and generates
            # guided hypotheses for testing.
            # ================================================================
            should_run_meta = (
                self.meta_discovery_agent is not None
                and (i + 1) % self.meta_step_frequency == 0
                and len(self.insight_store) >= 2
            )
            
            if should_run_meta:
                print(f"\n{'─'*60}")
                print(f"🧠 META-DISCOVERY STEP (The Core)")
                print(f"{'─'*60}")
                
                guided_hypotheses = self._run_meta_discovery_step()
                
                if guided_hypotheses:
                    print(f"\n  Testing {len(guided_hypotheses)} guided hypotheses from meta-insights...")
                    for hyp in guided_hypotheses:
                        insight = self._test_guided_hypothesis(hyp, iteration=i)
                        if insight:
                            meta_hypotheses_tested += 1
                
                print(f"{'─'*60}")
                print(f"End Meta-Discovery Step")
                print(f"{'─'*60}\n")

            # ================================================================
            # LEVEL 1: Base Discovery (Exploration → Testing)
            # ================================================================
            
            # Phase 1: Exploration
            print(f"\n📊 Phase 1: Data Exploration (Level 1)")
            exploration_result = self._run_exploration_phase(iteration=i)
            
            if not exploration_result.get("observations") and not exploration_result.get("preliminary_hypotheses"):
                print("No observations or hypotheses from exploration, skipping to next iteration")
                continue

            # Phase 2: Test Design
            print(f"\n🔬 Phase 2: Statistical Test Design")
            test_specs = self._run_test_design_phase(exploration_result, iteration=i)
            
            if not test_specs:
                print("No valid test specifications, skipping to next iteration")
                continue

            # Phase 3: Test Execution & Evaluation
            print(f"\n⚗️ Phase 3: Test Execution & Evaluation")
            insights = self._run_test_execution_phase(test_specs, exploration_result, iteration=i)
            
            print(f"\n✅ Iteration {i+1} complete: {len(insights)} base insights generated")

        # ================================================================
        # FINAL META-DISCOVERY: Run one last time with all accumulated insights
        # ================================================================
        if self.meta_discovery_agent and len(self.insight_store) >= 3:
            print(f"\n{'='*60}")
            print(f"🧠 FINAL META-DISCOVERY STEP")
            print(f"{'='*60}")
            
            final_guided = self._run_meta_discovery_step()
            if final_guided:
                print(f"\n  Testing {len(final_guided)} final guided hypotheses...")
                for hyp in final_guided:
                    insight = self._test_guided_hypothesis(hyp, iteration=n_iterations)
                    if insight:
                        meta_hypotheses_tested += 1

        # Generate final report with meta-insights
        self._generate_report()

        # Save insights
        self.insight_store.save()
        
        # Save meta-insights
        self._save_meta_insights()

        print(f"\n{'='*60}")
        print(f"Discovery Complete!")
        print(f"{'='*60}")
        
        base_insights = [i for i in self.insight_store.insights if not getattr(i, 'is_meta', False)]
        guided_insights = [i for i in self.insight_store.insights if getattr(i, 'is_meta', False)]
        
        print(f"  Base insights (Level 1): {len(base_insights)}")
        print(f"  Guided insights (Level 2): {len(guided_insights)}")
        print(f"  Meta-insights (observations): {len(self.meta_insights)}")
        print(f"  Meta-hypotheses tested: {meta_hypotheses_tested}")
        print(f"Report: {os.path.join(self.workspace_root, 'report.md')}")

    def _run_exploration_phase(self, iteration: int) -> dict[str, Any]:
        """
        Run the exploration phase using ExplorationAgent.
        
        Returns:
            Dict with observations, preliminary_hypotheses, code_history
        """
        # Build exploration goal based on previous discoveries
        exploration_goal = self._build_exploration_goal(iteration)
        
        # Get prior observations to build upon
        prior_observations = []
        if self.exploration_history:
            # Use observations from last exploration
            last_exploration = self.exploration_history[-1]
            prior_observations = last_exploration.get("observations", [])[:3]  # Last 3
        
        # Run exploration
        print(f"  Running exploration with goal: {exploration_goal[:100]}...")
        exploration_result = self.exploration_agent.explore(
            dataset=self.dataset,
            exploration_goal=exploration_goal,
            prior_observations=prior_observations
        )
        
        # Log and save
        self.exploration_history.append(exploration_result)
        self.logger.log_event("exploration_complete", {
            "iteration": iteration,
            "n_observations": len(exploration_result.get("observations", [])),
            "n_preliminary_hypotheses": len(exploration_result.get("preliminary_hypotheses", [])),
            "code_iterations": len(exploration_result.get("code_history", []))
        })
        
        # Print summary
        observations = exploration_result.get("observations", [])
        hypotheses = exploration_result.get("preliminary_hypotheses", [])
        
        print(f"  Found {len(observations)} observations:")
        for obs in observations[:3]:
            print(f"    - {obs.get('description', '')[:80]}...")
        
        print(f"  Generated {len(hypotheses)} preliminary hypotheses:")
        for hyp in hypotheses[:3]:
            print(f"    - {hyp.get('statement', '')[:80]}...")
        
        # Save exploration results
        self._save_exploration_results(iteration, exploration_result)
        
        return exploration_result

    def _run_test_design_phase(
        self,
        exploration_result: dict[str, Any],
        iteration: int
    ) -> list[dict]:
        """
        Run the test design phase using StatisticalTesterAgent.
        
        Args:
            exploration_result: Results from exploration phase
            iteration: Current iteration number
            
        Returns:
            List of validated test specifications
        """
        all_test_specs = []
        
        # Get dataset schema for the tester
        dataset_schema = self._get_dataset_schema()
        
        # Get exploration code for context
        exploration_code = None
        code_history = exploration_result.get("code_history", [])
        if code_history:
            successful_code = [c for c in code_history if c.get("success")]
            if successful_code:
                exploration_code = successful_code[-1].get("code", "")
        
        # Design tests for each preliminary hypothesis
        hypotheses = exploration_result.get("preliminary_hypotheses", [])
        observations = exploration_result.get("observations", [])
        
        for hyp in hypotheses[:3]:  # Limit to top 3 hypotheses
            statement = hyp.get("statement", "")
            if not statement:
                continue
            
            print(f"  Designing tests for: {statement[:60]}...")
            
            test_result = self.statistical_tester.propose_tests(
                hypothesis=statement,
                exploration_observations=observations,
                exploration_code=exploration_code,
                dataset_schema=dataset_schema,
                max_tests=self.max_tests_per_hypothesis
            )
            
            specs = test_result.get("test_specifications", [])
            print(f"    Proposed {len(specs)} tests")
            
            for spec in specs:
                spec["_source_hypothesis"] = statement
                spec["_iteration"] = iteration
                all_test_specs.append(spec)
            
            self.logger.log_event("test_design", {
                "hypothesis": statement,
                "n_tests_proposed": len(specs),
                "reasoning": test_result.get("reasoning", "")
            })
        
        return all_test_specs

    def _run_test_execution_phase(
        self,
        test_specs: list[dict],
        exploration_result: dict[str, Any],
        iteration: int
    ) -> list[Insight]:
        """
        Execute tests and generate insights.
        
        Args:
            test_specs: List of validated test specifications
            exploration_result: Original exploration results
            iteration: Current iteration number
            
        Returns:
            List of generated Insight objects
        """
        insights = []
        
        for spec in test_specs:
            test_id = spec.get("test_id", "unknown")
            test_type = spec.get("test_type", "unknown")
            source_hypothesis = spec.get("_source_hypothesis", "")
            
            print(f"  Running test {test_id} ({test_type})...")
            
            # Execute on train split
            train_result = self.test_executor.execute_test(spec, split="train")
            
            # Execute on val split
            self.dataset.set_active_split("val")
            val_result = self.test_executor.execute_test(spec, split="val")
            self.dataset.set_active_split("train")  # Reset
            
            # Combine results
            combined_results = {
                "train": train_result,
                "val": val_result
            }
            
            # Determine final verdict
            train_verdict = train_result.get("verdict", "inconclusive")
            val_verdict = val_result.get("verdict", "inconclusive")
            
            if train_verdict == "supported" and val_verdict == "supported":
                final_verdict = "supported"
            elif train_verdict == "rejected" or val_verdict == "rejected":
                final_verdict = "rejected"
            else:
                final_verdict = "inconclusive"
            
            print(f"    Train: {train_verdict}, Val: {val_verdict} → Final: {final_verdict}")
            
            # Create Insight
            # First create a Hypothesis object for the evaluator
            hypothesis = Hypothesis.create(
                statement=source_hypothesis,
                scope={"source": "exploration"},
                variables=spec.get("parameters", {}).get("x_variable", spec.get("parameters", {}).get("metric_variable", "unknown")).split() if isinstance(spec.get("parameters", {}).get("x_variable", ""), str) else []
            )
            
            # Create a minimal Experiment object
            experiment = Experiment.create(
                hypothesis_id=hypothesis.id,
                dataset_slice_spec={},
                feature_spec=spec.get("parameters", {}),
                method=test_type
            )
            
            # Convert to Result objects for evaluator
            from core.discovery.types import Result
            train_result_obj = Result(
                experiment_id=experiment.id,
                status="success" if train_result.get("status") == "success" else "fail",
                numbers=train_result.get("numbers", {}),
                notes=f"Test {test_id} on train split"
            )
            val_result_obj = Result(
                experiment_id=experiment.id,
                status="success" if val_result.get("status") == "success" else "fail",
                numbers=val_result.get("numbers", {}),
                notes=f"Test {test_id} on val split"
            )
            
            # Use evaluator
            insight = self.insight_evaluator.evaluate_with_validation(
                {"train": train_result_obj, "val": val_result_obj},
                hypothesis,
                experiment
            )
            
            # Store insight
            self.insight_store.add_insight(insight)
            insights.append(insight)
            
            self.logger.log_event("test_executed", {
                "test_id": test_id,
                "test_type": test_type,
                "train_result": train_result,
                "val_result": val_result,
                "final_verdict": final_verdict
            })
            
            # Save test results
            self._save_test_results(iteration, spec, combined_results, insight)
        
        return insights

    def _build_exploration_goal(self, iteration: int) -> str:
        """Build exploration goal based on iteration and prior discoveries."""
        if iteration == 0:
            return "Explore the dataset to understand its structure, find correlations, and identify potential patterns worth testing."
        
        # Build on previous findings
        existing_insights = self.insight_store.insights
        if not existing_insights:
            return "Continue exploring the dataset, looking for new patterns and relationships."
        
        # Summarize previous findings
        supported = [i for i in existing_insights if i.verdict == "supported"]
        rejected = [i for i in existing_insights if i.verdict == "rejected"]
        
        goal = "Build on previous findings:\n"
        if supported:
            goal += f"- {len(supported)} hypotheses were SUPPORTED. Look for related patterns.\n"
        if rejected:
            goal += f"- {len(rejected)} hypotheses were REJECTED. Explore alternative explanations.\n"
        goal += "\nFind new patterns that haven't been tested yet."
        
        return goal

    def _get_dataset_schema(self) -> dict:
        """Get dataset schema for the tester."""
        sample = self.dataset.sample({}, n=100)
        
        schema = {
            "columns": {},
            "target_column": getattr(self.dataset, 'target_column', None)
        }
        
        for col in sample.columns:
            dtype = str(sample[col].dtype)
            n_unique = sample[col].nunique()
            
            col_info = {
                "dtype": dtype,
                "n_unique": int(n_unique),
                "is_numeric": dtype in ['int64', 'float64'],
                "is_categorical": dtype == 'object' or n_unique < 20
            }
            
            if col_info["is_numeric"]:
                col_info["min"] = float(sample[col].min()) if not sample[col].isna().all() else None
                col_info["max"] = float(sample[col].max()) if not sample[col].isna().all() else None
            
            if col_info["is_categorical"]:
                col_info["categories"] = sample[col].dropna().unique()[:10].tolist()
            
            schema["columns"][col] = col_info
        
        return schema

    def _save_exploration_results(self, iteration: int, exploration_result: dict):
        """Save exploration results to disk."""
        iteration_dir = os.path.join(self.workspace_root, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        # Save code history
        code_path = os.path.join(iteration_dir, "exploration_code.py")
        with open(code_path, "w") as f:
            for entry in exploration_result.get("code_history", []):
                f.write(f"# === Iteration {entry.get('iteration', '?')} ===\n")
                f.write(f"# Success: {entry.get('success', False)}\n")
                f.write(entry.get("code", "") + "\n\n")
                if entry.get("output"):
                    f.write(f'"""\nOutput:\n{entry.get("output", "")[:1000]}\n"""\n\n')
        
        # Save observations and hypotheses
        findings_path = os.path.join(iteration_dir, "exploration_findings.json")
        with open(findings_path, "w") as f:
            json.dump({
                "observations": exploration_result.get("observations", []),
                "preliminary_hypotheses": exploration_result.get("preliminary_hypotheses", []),
                "data_quality_notes": exploration_result.get("data_quality_notes", [])
            }, f, indent=2)

    def _save_test_results(
        self,
        iteration: int,
        test_spec: dict,
        results: dict,
        insight: Insight
    ):
        """Save test results to disk."""
        iteration_dir = os.path.join(self.workspace_root, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)
        
        test_id = test_spec.get("test_id", "unknown")
        test_path = os.path.join(iteration_dir, f"test_{test_id}.json")
        
        with open(test_path, "w") as f:
            json.dump({
                "test_specification": test_spec,
                "results": results,
                "insight": insight.to_dict()
            }, f, indent=2)

    # ========================================================================
    # META-DISCOVERY METHODS (THE CORE OF THE SYSTEM)
    # ========================================================================
    
    def _run_meta_discovery_step(self) -> list[Hypothesis]:
        """
        Run meta-discovery to analyze base insights and generate guided hypotheses.
        
        This is THE CORE of the system:
        1. Analyze accumulated base insights → Meta-Insights (observations)
        2. Use Meta-Insights → Generate guided hypotheses for testing
        
        Returns:
            List of guided hypotheses to test
        """
        if not self.meta_discovery_agent:
            return []
        
        dataset_summary = self._get_dataset_summary()
        
        try:
            # discover_meta_patterns returns (meta_insights, guided_hypotheses)
            meta_insights, guided_hypotheses = self.meta_discovery_agent.discover_meta_patterns(
                insight_store=self.insight_store,
                dataset_summary=dataset_summary,
                n_meta_hypotheses=self.meta_hypotheses_per_step,
                max_retries=3
            )
            
            # Store meta-insights (observations about patterns, NOT experimental results)
            if meta_insights:
                self.meta_insights.extend(meta_insights)
                print(f"\n  📊 Generated {len(meta_insights)} meta-insights (actionable observations):")
                for mi in meta_insights:
                    print(f"     [{mi.meta_type}] {mi.observation[:60]}...")
                    print(f"        → {mi.actionable_recommendation[:50]}...")
            
            self.logger.log_event("meta_discovery_step", {
                "n_base_insights_analyzed": len(self.insight_store),
                "n_meta_insights_generated": len(meta_insights),
                "n_guided_hypotheses_generated": len(guided_hypotheses),
                "meta_insights": [mi.to_dict() for mi in meta_insights],
                "guided_hypotheses": [h.to_dict() for h in guided_hypotheses]
            })
            
            return guided_hypotheses
            
        except Exception as e:
            print(f"  ⚠️ Meta-discovery failed: {e}")
            self.logger.log_event("meta_discovery_error", {"error": str(e)})
            return []
    
    def _test_guided_hypothesis(
        self,
        hypothesis: Hypothesis,
        iteration: int
    ) -> Optional[Insight]:
        """
        Test a guided hypothesis (generated from meta-insights).
        
        Uses the StatisticalTesterAgent to design tests, maintaining the
        strict validation that is the hallmark of this system.
        
        Args:
            hypothesis: Guided hypothesis from meta-discovery
            iteration: Current iteration number
            
        Returns:
            Generated Insight or None if failed
        """
        print(f"\n  [GUIDED] Testing: {hypothesis.statement[:60]}...")
        
        # Get dataset schema
        dataset_schema = self._get_dataset_schema()
        
        # Use StatisticalTesterAgent to design rigorous tests
        test_result = self.statistical_tester.propose_tests(
            hypothesis=hypothesis.statement,
            exploration_observations=[],  # No exploration for guided hypotheses
            exploration_code=None,
            dataset_schema=dataset_schema,
            max_tests=2  # Fewer tests for guided hypotheses
        )
        
        test_specs = test_result.get("test_specifications", [])
        if not test_specs:
            print(f"    No valid tests designed for guided hypothesis")
            return None
        
        # Execute the first test
        spec = test_specs[0]
        spec["_source_hypothesis"] = hypothesis.statement
        spec["_is_guided"] = True
        
        test_id = spec.get("test_id", "guided")
        test_type = spec.get("test_type", "unknown")
        
        print(f"    Running test {test_id} ({test_type})...")
        
        # Execute on train and val splits
        train_result = self.test_executor.execute_test(spec, split="train")
        
        self.dataset.set_active_split("val")
        val_result = self.test_executor.execute_test(spec, split="val")
        self.dataset.set_active_split("train")
        
        # Determine verdict
        train_verdict = train_result.get("verdict", "inconclusive")
        val_verdict = val_result.get("verdict", "inconclusive")
        
        if train_verdict == "supported" and val_verdict == "supported":
            final_verdict = "supported"
        elif train_verdict == "rejected" or val_verdict == "rejected":
            final_verdict = "rejected"
        else:
            final_verdict = "inconclusive"
        
        print(f"    Train: {train_verdict}, Val: {val_verdict} → Final: {final_verdict}")
        
        # Create Insight
        experiment = Experiment.create(
            hypothesis_id=hypothesis.id,
            dataset_slice_spec={},
            feature_spec=spec.get("parameters", {}),
            method=test_type
        )
        
        from core.discovery.types import Result
        train_result_obj = Result(
            experiment_id=experiment.id,
            status="success" if train_result.get("status") == "success" else "fail",
            numbers=train_result.get("numbers", {}),
            notes=f"Guided test {test_id} on train split"
        )
        val_result_obj = Result(
            experiment_id=experiment.id,
            status="success" if val_result.get("status") == "success" else "fail",
            numbers=val_result.get("numbers", {}),
            notes=f"Guided test {test_id} on val split"
        )
        
        # Evaluate
        insight = self.insight_evaluator.evaluate_with_validation(
            {"train": train_result_obj, "val": val_result_obj},
            hypothesis,
            experiment
        )
        
        # Mark as meta-guided insight
        insight.is_meta = True
        
        # Store insight
        self.insight_store.add_insight(insight)
        
        self.logger.log_event("guided_hypothesis_tested", {
            "hypothesis": hypothesis.statement,
            "test_type": test_type,
            "verdict": final_verdict,
            "is_meta": True
        })
        
        # Save results
        self._save_test_results(iteration, spec, {"train": train_result, "val": val_result}, insight)
        
        return insight
    
    def _get_dataset_summary(self) -> dict[str, Any]:
        """Get dataset summary for meta-discovery."""
        if hasattr(self.dataset, 'get_summary'):
            return self.dataset.get_summary()
        
        return {
            "dataset_id": self.dataset.dataset_id,
            "dataset_version": self.dataset.dataset_version
        }
    
    def _save_meta_insights(self):
        """Save meta-insights to JSON file."""
        if not self.meta_insights:
            return
        
        meta_insights_path = os.path.join(self.workspace_root, "meta_insights.json")
        
        meta_insights_data = {
            "description": "Meta-insights are observations about patterns across base insights. They guide future discovery but are NOT tested experimentally.",
            "meta_insights": [mi.to_dict() for mi in self.meta_insights]
        }
        
        with open(meta_insights_path, "w") as f:
            json.dump(meta_insights_data, f, indent=2)
        
        print(f"Meta-insights saved to: {meta_insights_path}")

    def _generate_report(self):
        """Generate discovery report with Level 1/Level 2 distinction."""
        # Use the standard ReportGenerator which handles meta-insights
        report_generator = ReportGenerator()
        report_generator.generate(
            self.insight_store,
            os.path.join(self.workspace_root, "events.jsonl"),
            os.path.join(self.workspace_root, "report.md"),
            meta_insights=self.meta_insights
        )
        
        # Also generate exploration-specific appendix
        appendix_path = os.path.join(self.workspace_root, "exploration_appendix.md")
        
        with open(appendix_path, "w") as f:
            f.write("# Exploration Appendix\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            f.write("## Discovery Method\n\n")
            f.write("This discovery used a **two-role system** for base discovery:\n")
            f.write("1. **Explorer**: Free-form code generation to discover patterns\n")
            f.write("2. **Tester**: Rigorous statistical test specification\n\n")
            f.write("Plus **Meta-Discovery** (the core) to analyze base insights and generate guided hypotheses.\n\n")
            
            f.write("## Exploration History\n\n")
            for i, exploration in enumerate(self.exploration_history):
                f.write(f"### Iteration {i+1}\n\n")
                
                observations = exploration.get('observations', [])
                hypotheses = exploration.get('preliminary_hypotheses', [])
                
                f.write(f"**Observations Found**: {len(observations)}\n\n")
                for obs in observations[:5]:
                    desc = obs.get('description', '')[:150]
                    conf = obs.get('confidence', 'unknown')
                    f.write(f"- [{conf}] {desc}\n")
                
                f.write(f"\n**Preliminary Hypotheses**: {len(hypotheses)}\n\n")
                for hyp in hypotheses[:5]:
                    stmt = hyp.get('statement', '')[:150]
                    priority = hyp.get('priority', 'unknown')
                    f.write(f"- [{priority}] {stmt}\n")
                
                f.write("\n")
        
        print(f"Exploration appendix saved to: {appendix_path}")

    def shutdown(self):
        """Shutdown method for graceful cleanup."""
        print("Shutting down Exploration Discovery Runner...")
        if hasattr(self, 'insight_store'):
            self.insight_store.save()
        if hasattr(self, 'meta_insights') and self.meta_insights:
            self._save_meta_insights()
        print("Shutdown complete.")
