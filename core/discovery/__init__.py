# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Discovery system core types and utilities."""

from core.discovery.types import Hypothesis, Experiment, Result, Insight, MetaInsight
from core.discovery.insight_store import InsightStore
from core.discovery.meta_discovery import MetaDiscoveryAgent
from core.discovery.exploration_agent import ExplorationAgent
from core.discovery.statistical_tester_agent import StatisticalTesterAgent, TestExecutor
from core.discovery.code_sandbox import CodeSandbox

# Benchmark evaluation support
from core.discovery.benchmark_patterns import (
    BenchmarkSpec,
    PatternTier,
    PatternDifficulty,
    PatternType,
    Pattern,
    pattern_from_dict,
)
from core.discovery.benchmark_dataset import (
    BenchmarkDatasetAdapter,
    load_benchmark_from_directory,
)

__all__ = [
    "Hypothesis",
    "Experiment",
    "Result",
    "Insight",
    "MetaInsight",  # Meta-level insights about patterns across base insights
    "InsightStore",
    "MetaDiscoveryAgent",
    # Two-role exploration system
    "ExplorationAgent",
    "StatisticalTesterAgent",
    "TestExecutor",
    "CodeSandbox",
    # Benchmark evaluation
    "BenchmarkSpec",
    "PatternTier",
    "PatternDifficulty",
    "PatternType",
    "Pattern",
    "pattern_from_dict",
    "BenchmarkDatasetAdapter",
    "load_benchmark_from_directory",
]
