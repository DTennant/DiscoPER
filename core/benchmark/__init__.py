"""
Causal DAG benchmark generation system ("Game-of-Life" style).

Generates synthetic ecology datasets where causal rules compose to produce
emergent patterns. Ground truth is automatically extracted at three levels:
surface correlations, direct causal effects, and emergent structural patterns.

Includes visual benchmark generation (core.benchmark.visual) for producing
ecology scene images from causal DAG simulations, with feature extraction
for discovery pipeline compatibility.
"""

from core.benchmark.rules import (
    CausalDAG,
    CausalRule,
    CompetitionRule,
    InteractionRule,
    LinearCausalRule,
    ModulationRule,
    RuleSet,
    SaturationRule,
    ThresholdCausalRule,
    VariableSpec,
)
from core.benchmark.simulator import add_observation_columns, simulate_world
from core.benchmark.ground_truth_extractor import extract_ground_truth
from core.benchmark.presets import get_preset, PRESET_BUILDERS
from core.benchmark.networkx_ground_truth_extractor import extract_ground_truth_from_digraph
from core.benchmark.realworld_presets import (
    RealWorldPreset,
    get_realworld_preset,
    load_realworld_data,
    REALWORLD_PRESET_BUILDERS,
)

__all__ = [
    # Rules and DAG
    "CausalDAG",
    "CausalRule",
    "CompetitionRule",
    "InteractionRule",
    "LinearCausalRule",
    "ModulationRule",
    "RuleSet",
    "SaturationRule",
    "ThresholdCausalRule",
    "VariableSpec",
    # Simulation
    "simulate_world",
    "add_observation_columns",
    # Ground truth
    "extract_ground_truth",
    # Presets
    "get_preset",
    "PRESET_BUILDERS",
    # Real-world benchmarks
    "extract_ground_truth_from_digraph",
    "RealWorldPreset",
    "get_realworld_preset",
    "load_realworld_data",
    "REALWORLD_PRESET_BUILDERS",
]
