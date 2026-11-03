#!/usr/bin/env python
"""
Launcher for Claude Code Discovery Pipeline.

Uses Claude Code subprocess agents instead of LiteLLM API agents.
Each agent step spawns a `claude` CLI process with tool-use capabilities
(Bash, Read, Write) so it can read data files, write Python scripts, and
execute them autonomously.

Usage:
    python launch_cc_discovery.py
    python launch_cc_discovery.py n_iterations=5
    python launch_cc_discovery.py cc_max_turns=20
    python launch_cc_discovery.py --config-name cc_benchmark_discovery.yaml
"""

import os
import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf

from core.env_loader import load_env


@hydra.main(
    config_path="config/task",
    config_name="cc_benchmark_discovery.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """Run the Claude Code discovery pipeline."""
    # Load environment variables from .env if present
    load_env()

    print("=" * 80)
    print("Claude Code Discovery Launcher")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create workspace
    workspace_root = cfg.workspace_root
    os.makedirs(workspace_root, exist_ok=True)
    print(f"\nWorkspace: {workspace_root}")

    # Instantiate components
    print("\nInitializing components...")

    # Dataset
    dataset = hydra.utils.instantiate(cfg.dataset_args)
    print(f"  Dataset loaded: {dataset.dataset_id}")

    # Claude Code backend
    backend = hydra.utils.instantiate(cfg.cc_backend_args)
    print(f"  Claude Code backend initialized (max_turns={backend.max_turns})")

    # CC agents
    hypothesis_generator = hydra.utils.instantiate(cfg.cc_hypothesis_generator_args)
    print(f"  CC Hypothesis generator initialized")

    experiment_executor = hydra.utils.instantiate(cfg.cc_experiment_executor_args)
    print(f"  CC Experiment executor initialized")

    # Insight store + evaluator (deterministic, unchanged)
    insight_store = hydra.utils.instantiate(cfg.insight_store_args)
    print(f"  Insight store initialized")

    insight_evaluator = hydra.utils.instantiate(cfg.insight_eval_args)
    print(f"  Insight evaluator initialized")

    # Meta-discovery (optional)
    meta_discovery = None
    if hasattr(cfg, "cc_meta_discovery_args") and cfg.cc_meta_discovery_args is not None:
        meta_discovery = hydra.utils.instantiate(cfg.cc_meta_discovery_args)
        print(f"  CC Meta-discovery agent initialized")
        print(f"    -> will run every {cfg.get('meta_step_frequency', 5)} iterations")

    # Discovery runner
    discovery_runner = hydra.utils.instantiate(cfg.science_runner_args)
    print(f"  CC Discovery runner initialized")

    print("\n" + "=" * 80)
    print("Discovery Mode: CLAUDE CODE AGENTS")
    print("=" * 80)
    print(
        f"""
Pipeline:
  1. CC Hypothesis Generator  -> explores data, generates hypothesis
  2. CC Experiment Executor   -> designs + runs statistical test
  3. Insight Evaluator        -> determines supported/rejected (deterministic)
  4. CC Meta-Discovery        -> finds cross-insight patterns (every {cfg.get('meta_step_frequency', 5)} iters)

Iterations: {cfg.n_iterations}
"""
    )
    print("=" * 80 + "\n")

    # Run
    try:
        asyncio.run(discovery_runner.run(n_iterations=cfg.n_iterations))
        print("\n" + "=" * 80)
        print("Claude Code discovery completed successfully!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Discovery failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
