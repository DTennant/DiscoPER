#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Simple launcher for Discovery MVP - runs locally without async/SLURM complexity."""

import os
import asyncio
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path="config/task",
    config_name="inat_discovery.yaml",
    version_base="1.1",
)
def main(cfg: DictConfig):
    """Run the discovery pipeline."""
    print("=" * 80)
    print("Discovery MVP Launcher")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Create workspace directory
    workspace_root = cfg.workspace_root
    os.makedirs(workspace_root, exist_ok=True)
    print(f"\nWorkspace: {workspace_root}")

    # Instantiate all components
    print("\nInitializing components...")

    # Dataset
    dataset = hydra.utils.instantiate(cfg.dataset_args)
    print(f"  ✓ Dataset loaded: {dataset.dataset_id}")

    # Tool router — must share the SAME dataset object so set_active_split works
    tool_router = hydra.utils.instantiate(cfg.tool_router_args, dataset=dataset)
    print(f"  ✓ Tool router initialized")

    # Hypothesis generator
    hypothesis_generator = hydra.utils.instantiate(cfg.hypothesis_generator_args)
    print(f"  ✓ Hypothesis generator initialized")

    # Experiment planner
    experiment_planner = hydra.utils.instantiate(cfg.experiment_planner_args)
    print(f"  ✓ Experiment planner initialized")

    # Experiment runner
    experiment_runner = hydra.utils.instantiate(cfg.experiment_runner_args)
    print(f"  ✓ Experiment runner initialized")

    # Insight store
    insight_store = hydra.utils.instantiate(cfg.insight_store_args)
    print(f"  ✓ Insight store initialized")

    # Insight evaluator
    insight_evaluator = hydra.utils.instantiate(cfg.insight_eval_args)
    print(f"  ✓ Insight evaluator initialized")

    # Meta-discovery agent (optional)
    meta_discovery_enabled = (
        hasattr(cfg, "meta_discovery_args") and cfg.meta_discovery_args is not None
    )
    if meta_discovery_enabled:
        meta_discovery = hydra.utils.instantiate(cfg.meta_discovery_args)
        print(f"  ✓ Meta-discovery agent initialized")
        meta_freq = cfg.get("meta_step_frequency", 5)
        print(f"    → Meta-discovery will run every {meta_freq} iterations")

    # Discovery runner
    discovery_runner = hydra.utils.instantiate(cfg.science_runner_args)
    print(f"  ✓ Discovery runner initialized")

    print("\n" + "=" * 80)
    print(f"Starting discovery with {cfg.n_iterations} iterations...")
    if meta_discovery_enabled:
        print(
            f"Meta-discovery: ENABLED (every {cfg.get('meta_step_frequency', 5)} iterations)"
        )
    print("=" * 80 + "\n")

    # Run discovery (async)
    try:
        asyncio.run(discovery_runner.run(n_iterations=cfg.n_iterations))
        print("\n" + "=" * 80)
        print("Discovery completed successfully!")
        print("=" * 80)
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Discovery failed with error: {e}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
