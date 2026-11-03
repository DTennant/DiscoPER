"""
Real-world causal discovery benchmark presets.

Provides dataset configs and data loading for real-world causal networks
(SACHS protein signaling, bnlearn standard networks like ALARM, ASIA,
INSURANCE, CHILD) that have known causal DAGs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RealWorldPreset:
    name: str
    description: str
    domain: str
    data_source: str  # "cdt_sachs", "pgmpy_sample"
    bnlearn_model_name: str
    column_rename: dict[str, str] = field(default_factory=dict)
    prior_edge_ids: list[tuple[str, str]] = field(default_factory=list)
    n_samples: int = 5000
    seed: int = 42
    correlation_threshold: float = 0.15
    variable_descriptions: dict[str, str] = field(default_factory=dict)
    reference_paper: str = ""


# =============================================================================
# Data loading
# =============================================================================


def _load_dag(model_name: str) -> nx.DiGraph:
    from pgmpy.utils import get_example_model

    model = get_example_model(model_name)
    return nx.DiGraph(model.edges())


def _load_cdt_sachs(preset: RealWorldPreset) -> tuple[pd.DataFrame, nx.DiGraph]:
    import cdt

    data, _ = cdt.data.load_dataset("sachs")
    if preset.column_rename:
        data = data.rename(columns=preset.column_rename)

    dag = _load_dag(preset.bnlearn_model_name)
    return data, dag


def _load_pgmpy_sample(preset: RealWorldPreset) -> tuple[pd.DataFrame, nx.DiGraph]:
    from pgmpy.sampling import BayesianModelSampling
    from pgmpy.utils import get_example_model

    model = get_example_model(preset.bnlearn_model_name)
    sampler = BayesianModelSampling(model)
    df = sampler.forward_sample(size=preset.n_samples, seed=preset.seed)

    # Label-encode string/categorical columns to integers
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype):
            codes, _uniques = pd.factorize(df[col], sort=True)
            df[col] = codes

    dag = nx.DiGraph(model.edges())
    return df, dag


def load_realworld_data(preset: RealWorldPreset) -> tuple[pd.DataFrame, nx.DiGraph]:
    if preset.data_source == "cdt_sachs":
        return _load_cdt_sachs(preset)
    elif preset.data_source == "pgmpy_sample":
        return _load_pgmpy_sample(preset)
    else:
        raise ValueError(f"Unknown data_source: {preset.data_source}")


# =============================================================================
# Preset definitions
# =============================================================================


def _make_sachs_preset() -> RealWorldPreset:
    return RealWorldPreset(
        name="sachs",
        description=(
            "SACHS protein signaling network. 11 phosphoproteins/phospholipids "
            "measured by flow cytometry in primary human immune cells. "
            "7466 real observational data points from CDT."
        ),
        domain="molecular_biology",
        data_source="cdt_sachs",
        bnlearn_model_name="sachs",
        column_rename={
            "p44/42": "Erk",
            "pakts473": "Akt",
            "pjnk": "Jnk",
            "plcg": "Plcg",
            "pmek": "Mek",
            "praf": "Raf",
        },
        prior_edge_ids=[
            ("Raf", "Mek"),
            ("Mek", "Erk"),
            ("Plcg", "PIP2"),
            ("Plcg", "PIP3"),
        ],
        n_samples=7466,  # full real dataset
        seed=42,
        correlation_threshold=0.15,
        variable_descriptions={
            "Raf": "Raf kinase activity",
            "Mek": "MEK1/2 kinase activity",
            "Erk": "ERK1/2 kinase activity",
            "Akt": "Akt kinase activity (S473 phosphorylation)",
            "PKA": "Protein kinase A activity",
            "PKC": "Protein kinase C activity",
            "P38": "p38 MAPK activity",
            "Jnk": "JNK kinase activity",
            "Plcg": "Phospholipase C-gamma activity",
            "PIP2": "Phosphatidylinositol 4,5-bisphosphate level",
            "PIP3": "Phosphatidylinositol 3,4,5-trisphosphate level",
        },
        reference_paper="Sachs et al. (2005) Science 308:523-529",
    )


def _make_alarm_preset() -> RealWorldPreset:
    return RealWorldPreset(
        name="alarm",
        description=(
            "ALARM (A Logical Alarm Reduction Mechanism) medical monitoring network. "
            "37 variables modeling anesthesia patient monitoring with 46 causal edges."
        ),
        domain="medical_monitoring",
        data_source="pgmpy_sample",
        bnlearn_model_name="alarm",
        prior_edge_ids=[
            ("HYPOVOLEMIA", "LVEDVOLUME"),
            ("INTUBATION", "VENTLUNG"),
            ("KINKEDTUBE", "VENTLUNG"),
            ("PULMEMBOLUS", "PAP"),
        ],
        n_samples=10000,
        seed=42,
        correlation_threshold=0.12,
        reference_paper="Beinlich et al. (1989) ALARM monitoring system",
    )


def _make_asia_preset() -> RealWorldPreset:
    return RealWorldPreset(
        name="asia",
        description=(
            "ASIA (Lauritzen-Spiegelhalter) lung disease network. "
            "8 variables modeling tuberculosis/lung cancer diagnosis."
        ),
        domain="medical_diagnosis",
        data_source="pgmpy_sample",
        bnlearn_model_name="asia",
        prior_edge_ids=[
            ("smoke", "lung"),
            ("smoke", "bronc"),
        ],
        n_samples=5000,
        seed=42,
        correlation_threshold=0.10,
        reference_paper="Lauritzen & Spiegelhalter (1988)",
    )


def _make_insurance_preset() -> RealWorldPreset:
    return RealWorldPreset(
        name="insurance",
        description=(
            "INSURANCE network for car insurance risk assessment. "
            "27 variables with 52 causal edges modeling driver risk factors."
        ),
        domain="insurance_risk",
        data_source="pgmpy_sample",
        bnlearn_model_name="insurance",
        prior_edge_ids=[],
        n_samples=10000,
        seed=42,
        correlation_threshold=0.12,
        reference_paper="Binder et al. (1997)",
    )


def _make_child_preset() -> RealWorldPreset:
    return RealWorldPreset(
        name="child",
        description=(
            "CHILD network for congenital heart disease diagnosis. "
            "20 variables with 25 causal edges modeling neonatal diagnostics."
        ),
        domain="pediatric_medicine",
        data_source="pgmpy_sample",
        bnlearn_model_name="child",
        prior_edge_ids=[],
        n_samples=8000,
        seed=42,
        correlation_threshold=0.12,
        reference_paper="Spiegelhalter et al. (1993)",
    )


REALWORLD_PRESET_BUILDERS: dict[str, Any] = {
    "sachs": _make_sachs_preset,
    "alarm": _make_alarm_preset,
    "asia": _make_asia_preset,
    "insurance": _make_insurance_preset,
    "child": _make_child_preset,
}


def get_realworld_preset(name: str) -> RealWorldPreset:
    if name not in REALWORLD_PRESET_BUILDERS:
        available = list(REALWORLD_PRESET_BUILDERS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return REALWORLD_PRESET_BUILDERS[name]()
