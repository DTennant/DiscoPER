# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tree Search Discovery Runner - combines tree search with the discovery pipeline.

Key features:
1. Tree search: Multiple hypotheses per iteration with branching and best-first selection
2. Context passing: Recent insights and previous hypotheses passed to hypothesis generator
3. Duplicate avoidance: Tracks tested hypotheses to prevent repetition
4. Discovery pipeline: hypothesis → experiment → insight → evaluate
"""

import os
import json
from typing import Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

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


@dataclass
class TreeNode:
    """A node in the hypothesis search tree."""
    node_id: str
    parent_id: Optional[str]
    hypothesis: Hypothesis
    insight: Optional[Insight] = None
    children: list[str] = field(default_factory=list)
    depth: int = 0
    
    @property
    def is_supported(self) -> bool:
        return self.insight is not None and self.insight.verdict == "supported"
    
    @property
    def strength(self) -> float:
        return self.insight.strength if self.insight else 0.0
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "hypothesis": self.hypothesis.to_dict(),
            "insight": self.insight.to_dict() if self.insight else None,
            "children": self.children,
            "depth": self.depth,
            "is_supported": self.is_supported,
            "strength": self.strength
        }


class TreeSearchDiscoveryRunner:
    """
    Tree Search Discovery Runner - implements best-first tree search over hypothesis space.
    
    Search strategy:
    1. Generate N hypotheses per iteration (branching)
    2. Execute each hypothesis through the discovery pipeline
    3. Select the best-performing branch to continue from
    4. Pass context (recent insights, previous hypotheses) to avoid duplicates
    5. Optional: Meta-discovery can still run periodically
    """

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
        n_hypotheses: int = 3,  # Branching factor
        n_initial_hypotheses: int = 3,  # First iteration branching
        selection_metric: str = "strength",  # Metric for best-first selection
        meta_discovery_agent: Optional[MetaDiscoveryAgent] = None,
        meta_step_frequency: int = 999,  # Disabled by default for tree search
        meta_hypotheses_per_step: int = 2,
        max_tree_depth: int = 10,
        enable_context_passing: bool = True
    ):
        """
        Initialize tree search discovery runner.

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
            n_hypotheses: Number of hypotheses per iteration (branching factor)
            n_initial_hypotheses: Number of hypotheses for first iteration
            selection_metric: Metric for selecting best branch ("strength" or "novelty")
            meta_discovery_agent: Optional agent for meta-level discovery
            meta_step_frequency: Run meta-discovery every N iterations
            meta_hypotheses_per_step: Number of meta-hypotheses per step
            max_tree_depth: Maximum depth of the search tree
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
        self.max_retries = max_retries
        
        # Tree search configuration
        self.n_hypotheses = n_hypotheses
        self.n_initial_hypotheses = n_initial_hypotheses
        self.selection_metric = selection_metric
        self.max_tree_depth = max_tree_depth
        self.enable_context_passing = enable_context_passing
        
        # Meta-discovery configuration
        self.meta_discovery_agent = meta_discovery_agent
        self.meta_step_frequency = meta_step_frequency
        self.meta_hypotheses_per_step = meta_hypotheses_per_step
        
        # Tree state
        self.tree: dict[str, TreeNode] = {}  # node_id -> TreeNode
        self.node_counter = 0
        self.tested_hypotheses: list[str] = []  # Track all tested hypothesis statements
        
        # Meta-insights storage
        self.meta_insights: list[MetaInsight] = []

        # Create workspace
        os.makedirs(workspace_root, exist_ok=True)

        # Initialize logger
        self.logger = DiscoveryLogger(os.path.join(workspace_root, "events.jsonl"))
        
        # Tracking for analysis
        self.iteration_stats: list[dict] = []

    async def run(self, n_iterations: int = 1):
        """
        Run the tree search discovery loop.

        Args:
            n_iterations: Number of iterations (each iteration can branch multiple hypotheses)
        """
        print(f"Starting Tree Search Discovery with {n_iterations} iterations")
        print(f"Workspace: {self.workspace_root}")
        print(f"Branching factor: {self.n_hypotheses} (initial: {self.n_initial_hypotheses})")
        print(f"Context passing: {'enabled' if self.enable_context_passing else 'disabled'}")
        if self.meta_discovery_agent and self.meta_step_frequency < n_iterations:
            print(f"Meta-discovery enabled (every {self.meta_step_frequency} iterations)")

        current_best_node: Optional[str] = None
        
        for i in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"=== Iteration {i+1}/{n_iterations} ===")
            print(f"{'='*60}")
            
            # Determine branching factor for this iteration
            n_hyp = self.n_initial_hypotheses if i == 0 else self.n_hypotheses
            
            # Get current tree depth
            current_depth = 0
            if current_best_node and current_best_node in self.tree:
                current_depth = self.tree[current_best_node].depth + 1
            
            if current_depth >= self.max_tree_depth:
                print(f"Maximum tree depth ({self.max_tree_depth}) reached. Stopping.")
                break
            
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
                    for meta_hyp in meta_hypotheses:
                        self._execute_hypothesis(meta_hyp, i, parent_id=current_best_node, is_meta=True)
                    print(f"--- End Meta-Discovery Step ---\n")
            
            # Generate hypotheses with context
            print(f"\nGenerating {n_hyp} hypotheses from node: {current_best_node or 'ROOT'}")
            
            dataset_summary = self._get_dataset_summary()
            
            # Get context for hypothesis generation
            recent_insights = None
            previous_hypotheses = None
            prior_knowledge_text = None
            
            if self.enable_context_passing:
                recent_insights = list(self.insight_store.insights)[-10:]
                previous_hypotheses = self.tested_hypotheses[-20:]
                if hasattr(self.dataset, 'get_prior_knowledge_prompt'):
                    prior_knowledge_text = self.dataset.get_prior_knowledge_prompt()
            
            # Generate hypotheses
            hypotheses = self.hypothesis_generator.generate(
                dataset_summary,
                n_hypotheses=n_hyp,
                max_retries=self.max_retries,
                recent_insights=recent_insights,
                previous_hypotheses=previous_hypotheses,
                prior_knowledge_text=prior_knowledge_text
            )

            if not hypotheses:
                print(f"Failed to generate hypotheses in iteration {i+1}")
                continue

            # Execute each hypothesis
            iteration_nodes = []
            for j, hypothesis in enumerate(hypotheses):
                print(f"\n--- Branch {j+1}/{len(hypotheses)} ---")
                node = self._execute_hypothesis(
                    hypothesis, 
                    i, 
                    parent_id=current_best_node,
                    is_meta=False
                )
                if node:
                    iteration_nodes.append(node)
            
            # Select best node for next iteration
            if iteration_nodes:
                current_best_node = self._select_best_node(iteration_nodes)
                best_node = self.tree[current_best_node]
                print(f"\n>>> Selected best branch: {current_best_node} (strength={best_node.strength:.3f})")
            
            # Record iteration stats
            self._record_iteration_stats(i, iteration_nodes)

        # Run final meta-discovery if enabled and we have enough insights
        if self.meta_discovery_agent and len(self.insight_store) >= 3 and self.meta_step_frequency < 999:
            print(f"\n=== Final Meta-Discovery Step ===")
            final_meta_hypotheses = self._run_meta_discovery()
            if final_meta_hypotheses:
                print(f"Generated {len(final_meta_hypotheses)} final meta-hypotheses")
                for meta_hyp in final_meta_hypotheses:
                    self._execute_hypothesis(meta_hyp, n_iterations, parent_id=current_best_node, is_meta=True)

        # Generate outputs
        self._save_tree()
        self._save_iteration_stats()
        self._generate_report()
        self.insight_store.save()
        self._save_meta_insights()

        # Print summary
        n_supported = len([n for n in self.tree.values() if n.is_supported])
        print(f"\n{'='*60}")
        print(f"Tree Search Discovery Complete!")
        print(f"{'='*60}")
        print(f"  Tree nodes: {len(self.tree)}")
        print(f"  Total insights: {len(self.insight_store)}")
        print(f"  Supported insights: {n_supported}")
        print(f"  Meta-insights: {len(self.meta_insights)}")
        print(f"  Unique hypotheses tested: {len(self.tested_hypotheses)}")
        print(f"  Report: {os.path.join(self.workspace_root, 'report.md')}")

    def _execute_hypothesis(
        self,
        hypothesis: Hypothesis,
        iteration: int,
        parent_id: Optional[str] = None,
        is_meta: bool = False
    ) -> Optional[str]:
        """
        Execute a hypothesis through the discovery pipeline and add to tree.
        
        Returns:
            Node ID if successful, None otherwise
        """
        prefix = "[META] " if is_meta else ""
        print(f"\n{prefix}Hypothesis: {hypothesis.statement}")
        
        # Track hypothesis
        self.tested_hypotheses.append(hypothesis.statement)
        
        # Log event
        event_data = {"hypothesis": hypothesis.to_dict(), "parent_id": parent_id}
        if is_meta:
            event_data["is_meta"] = True
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
        print(f"{prefix}Experiment: {experiment.method} on {experiment.dataset_slice_spec}")
        self.logger.log_event("experiment_planned", {"experiment": experiment.to_dict()})

        # Execute experiment with validation
        results = self.experiment_runner.run_with_validation(
            experiment, self.dataset, self.tool_router, splits=["train", "val"]
        )

        # Log results
        for split, result in results.items():
            print(f"{prefix}Result ({split}): {result.status}, metrics={result.numbers}")
            self.logger.log_event(f"experiment_result_{split}", {"result": result.to_dict()})

        # Evaluate → Insight
        insight = self.insight_evaluator.evaluate_with_validation(
            results, hypothesis, experiment
        )
        
        if is_meta:
            insight.is_meta = True
        
        print(f"{prefix}Insight: {insight.verdict} (strength={insight.strength:.3f})")

        # Store insight
        self.insight_store.add_insight(insight)
        self.logger.log_event("insight_added", {"insight": insight.to_dict()})

        # Create tree node
        node_id = self._create_tree_node(hypothesis, insight, parent_id)
        
        # Save iteration results
        self._save_discovery_results(iteration, node_id, hypothesis, experiment, results, insight, is_meta)

        return node_id

    def _create_tree_node(
        self,
        hypothesis: Hypothesis,
        insight: Insight,
        parent_id: Optional[str]
    ) -> str:
        """Create a new node in the search tree."""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        
        depth = 0
        if parent_id and parent_id in self.tree:
            depth = self.tree[parent_id].depth + 1
            self.tree[parent_id].children.append(node_id)
        
        node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            hypothesis=hypothesis,
            insight=insight,
            depth=depth
        )
        
        self.tree[node_id] = node
        return node_id

    def _select_best_node(self, node_ids: list[str]) -> str:
        """Select the best node based on selection metric."""
        if not node_ids:
            return None
        
        # Get nodes
        nodes = [self.tree[nid] for nid in node_ids if nid in self.tree]
        
        if not nodes:
            return node_ids[0]
        
        # Sort by metric
        if self.selection_metric == "strength":
            # Prefer supported insights, then by strength
            nodes.sort(key=lambda n: (n.is_supported, n.strength), reverse=True)
        else:
            # Default: by strength
            nodes.sort(key=lambda n: n.strength, reverse=True)
        
        return nodes[0].node_id

    def _run_meta_discovery(self) -> list[Hypothesis]:
        """Run meta-discovery to generate meta-insights and guided hypotheses."""
        if not self.meta_discovery_agent:
            return []
        
        dataset_summary = self._get_dataset_summary()
        
        try:
            meta_insights, hypotheses = self.meta_discovery_agent.discover_meta_patterns(
                insight_store=self.insight_store,
                dataset_summary=dataset_summary,
                n_meta_hypotheses=self.meta_hypotheses_per_step,
                max_retries=self.max_retries
            )
            
            if meta_insights:
                self.meta_insights.extend(meta_insights)
                print(f"\n📊 Stored {len(meta_insights)} meta-insights")
                for mi in meta_insights:
                    print(f"   [{mi.meta_type}] {mi.observation[:60]}...")
            
            self.logger.log_event("meta_discovery_step", {
                "n_meta_insights": len(meta_insights),
                "n_hypotheses": len(hypotheses)
            })
            
            return hypotheses
            
        except Exception as e:
            print(f"Meta-discovery failed: {e}")
            return []

    def _get_dataset_summary(self) -> dict[str, Any]:
        """Get dataset summary for prompts."""
        if hasattr(self.dataset, 'get_summary'):
            return self.dataset.get_summary()
        return {
            "dataset_id": self.dataset.dataset_id,
            "dataset_version": self.dataset.dataset_version
        }

    def _record_iteration_stats(self, iteration: int, node_ids: list[str]):
        """Record statistics for this iteration."""
        nodes = [self.tree[nid] for nid in node_ids if nid in self.tree]
        
        stats = {
            "iteration": iteration,
            "n_hypotheses_tested": len(nodes),
            "n_supported": len([n for n in nodes if n.is_supported]),
            "n_rejected": len([n for n in nodes if n.insight and n.insight.verdict == "rejected"]),
            "n_inconclusive": len([n for n in nodes if n.insight and n.insight.verdict == "inconclusive"]),
            "max_strength": max([n.strength for n in nodes]) if nodes else 0,
            "cumulative_insights": len(self.insight_store),
            "cumulative_supported": len([i for i in self.insight_store.insights if i.verdict == "supported"]),
            "unique_variables_explored": self._count_unique_variables()
        }
        
        self.iteration_stats.append(stats)

    def _count_unique_variables(self) -> int:
        """Count unique variables explored across all hypotheses."""
        all_vars = set()
        for node in self.tree.values():
            if node.hypothesis and node.hypothesis.variables:
                all_vars.update(node.hypothesis.variables)
        return len(all_vars)

    def _save_tree(self):
        """Save the search tree to JSON."""
        tree_path = os.path.join(self.workspace_root, "search_tree.json")
        tree_data = {
            "n_nodes": len(self.tree),
            "max_depth": max([n.depth for n in self.tree.values()]) if self.tree else 0,
            "nodes": {nid: node.to_dict() for nid, node in self.tree.items()}
        }
        with open(tree_path, "w") as f:
            json.dump(tree_data, f, indent=2)
        print(f"Search tree saved to: {tree_path}")

    def _save_iteration_stats(self):
        """Save iteration statistics for analysis."""
        stats_path = os.path.join(self.workspace_root, "iteration_stats.json")
        with open(stats_path, "w") as f:
            json.dump(self.iteration_stats, f, indent=2)
        print(f"Iteration stats saved to: {stats_path}")

    def _save_discovery_results(
        self,
        iteration: int,
        node_id: str,
        hypothesis: Hypothesis,
        experiment: Experiment,
        results: dict[str, Result],
        insight: Insight,
        is_meta: bool
    ):
        """Save discovery results for a node."""
        suffix = "_meta" if is_meta else ""
        iteration_dir = os.path.join(self.workspace_root, f"iteration_{iteration}{suffix}", node_id)
        os.makedirs(iteration_dir, exist_ok=True)

        discovery_json = {
            "iteration": iteration,
            "node_id": node_id,
            "is_meta": is_meta,
            "hypothesis": hypothesis.to_dict(),
            "experiment": experiment.to_dict(),
            "results": {split: result.to_dict() for split, result in results.items()},
            "insight": insight.to_dict()
        }

        with open(os.path.join(iteration_dir, "discovery.json"), "w") as f:
            json.dump(discovery_json, f, indent=2)

    def _save_meta_insights(self):
        """Save meta-insights to JSON."""
        if not self.meta_insights:
            return
        
        meta_path = os.path.join(self.workspace_root, "meta_insights.json")
        meta_data = {
            "description": "Meta-insights guide hypothesis generation",
            "meta_insights": [mi.to_dict() for mi in self.meta_insights]
        }
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=2)
        print(f"Meta-insights saved to: {meta_path}")

    def _generate_report(self):
        """Generate discovery report."""
        report_generator = ReportGenerator()
        report_generator.generate(
            self.insight_store,
            os.path.join(self.workspace_root, "events.jsonl"),
            os.path.join(self.workspace_root, "report.md"),
            meta_insights=self.meta_insights
        )

    def shutdown(self):
        """Shutdown method for graceful cleanup."""
        print("Shutting down Tree Search Discovery...")
        if hasattr(self, 'insight_store'):
            self.insight_store.save()
        if hasattr(self, 'tree'):
            self._save_tree()
        print("Shutdown complete.")
