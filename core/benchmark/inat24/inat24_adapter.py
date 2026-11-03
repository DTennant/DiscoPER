"""
iNaturalist 2024 (50K sample) dataset adapter for the discovery pipeline.

Provides a BenchmarkDatasetAdapter-compatible interface over the iNat24 50K
observation sample with 15 ground truth patterns (2 prior, 1 obvious, 12 novel).

Data layout:
    data_dir/
    ├── inat24_50k.parquet        # 50K observations
    └── inat24_ground_truth.json  # 15 ground truth patterns
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.insight_store import InsightStore
from core.discovery.types import Insight


class INat24DatasetAdapter(DatasetAdapter):
    """Dataset adapter for iNaturalist 2024 50K sample benchmark."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        data_path: Optional[str] = None,
        ground_truth_path: Optional[str] = None,
        prior_insights_path: Optional[str] = None,
        include_prior_in_prompt: bool = True,
        prior_prompt_detail: str = "full",
        random_seed: int = 42,
        **kwargs: Any,
    ):
        if data_dir is None and data_path is not None:
            data_dir = os.path.dirname(data_path)
        if data_dir is None:
            raise ValueError("Either data_dir or data_path must be provided")
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.include_prior_in_prompt = include_prior_in_prompt
        self.prior_prompt_detail = prior_prompt_detail

        data_path = os.path.join(data_dir, "inat24_50k.parquet")
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Data file not found: {data_path}\n"
                "To obtain the iNat24 50K sample:\n"
                "  1. Download inat24_50k.parquet into data/inquire_benchmarks/\n"
                "  2. Ensure inat24_ground_truth.json is in the same directory"
            )
        self.data = pd.read_parquet(data_path)

        gt_path = ground_truth_path or os.path.join(
            data_dir, "inat24_ground_truth.json"
        )
        self.ground_truth: dict[str, Any] = {}
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                self.ground_truth = json.load(f)

        self.prior_patterns: list[dict[str, Any]] = []
        self.obvious_patterns: list[dict[str, Any]] = []
        self.novel_patterns: list[dict[str, Any]] = []
        for pattern in self.ground_truth.get("patterns", []):
            tier = pattern.get("tier", "novel")
            if tier in ("prior_knowledge", "prior"):
                self.prior_patterns.append(pattern)
            elif tier == "obvious":
                self.obvious_patterns.append(pattern)
            else:
                self.novel_patterns.append(pattern)

        self._prior_insights: list[Insight] = []
        pi_path = prior_insights_path or os.path.join(
            data_dir, "prior_insights.parquet"
        )
        if os.path.exists(pi_path):
            pi_df = pd.read_parquet(pi_path)
            for _, row in pi_df.iterrows():
                self._prior_insights.append(Insight.from_dict(row.to_dict()))
        elif self.prior_patterns:
            self._create_prior_insights_from_ground_truth()

        self.active_split = "train"
        self._create_splits()

    def _create_splits(self) -> None:
        """Create train (70%) / val (15%) / test (15%) splits."""
        n = len(self.data)
        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(n)

        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        self._train_indices = sorted(indices[:train_end].tolist())
        self._val_indices = sorted(indices[train_end:val_end].tolist())
        self._test_indices = sorted(indices[val_end:].tolist())

    def _get_active_data(self) -> pd.DataFrame:
        """Get DataFrame for the active split."""
        if self.active_split == "train":
            return self.data.iloc[self._train_indices].reset_index(drop=True)
        elif self.active_split == "val":
            return self.data.iloc[self._val_indices].reset_index(drop=True)
        elif self.active_split == "test":
            return self.data.iloc[self._test_indices].reset_index(drop=True)
        return self.data

    def _create_prior_insights_from_ground_truth(self) -> None:
        """Create prior insights from ground truth prior knowledge patterns."""
        for pattern in self.prior_patterns:
            insight = Insight.create(
                hypothesis_id=f"prior_{pattern['id']}",
                experiment_id=f"prior_exp_{pattern['id']}",
                verdict="supported",
                claim=pattern.get("description", f"Prior knowledge: {pattern['id']}"),
                strength=0.9,
                novelty_proxy=0.0,
            )
            self._prior_insights.append(insight)

    # ---- DatasetAdapter interface ----------------------------------------

    @property
    def dataset_id(self) -> str:
        return "inat24_50k"

    @property
    def dataset_version(self) -> str:
        n = len(self.data)
        return f"inat24_v1_{n}obs_seed{self.random_seed}"

    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """Sample rows from observation data based on filters."""
        df = self._get_active_data().copy()

        for key, value in slice_spec.items():
            if key not in df.columns:
                continue
            if isinstance(value, list):
                if (
                    len(value) == 2
                    and isinstance(value[0], (int, float))
                    and isinstance(value[1], (int, float))
                ):
                    df = df[(df[key] >= value[0]) & (df[key] <= value[1])]
                else:
                    df = df[df[key].isin(value)]
            elif isinstance(value, (int, float, str)):
                df = df[df[key] == value]

        if len(df) > n:
            df = df.sample(n=n, random_state=self.random_seed)

        return df

    def load_metadata(
        self, slice_spec: dict[str, Any], columns: list[str]
    ) -> pd.DataFrame:
        """Load specific columns for a slice."""
        df = self.sample(slice_spec, n=len(self.data))
        available_cols = [c for c in columns if c in df.columns]
        return df[available_cols]

    def load_embeddings(self, image_ids: list, embedding_type: str) -> np.ndarray:
        """Load embeddings (random placeholder -- no precomputed embeddings)."""
        np.random.seed(self.random_seed)
        dim = 512 if embedding_type == "clip" else 256
        return np.random.randn(len(image_ids), dim)

    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """Return taxonomy info for a species label."""
        matches = self.data[self.data["species_name"] == label]
        if len(matches) == 0:
            matches = self.data[self.data["common_name"] == label]
        if len(matches) == 0:
            return {}
        row = matches.iloc[0]
        return {
            "kingdom": row.get("kingdom", ""),
            "phylum": row.get("phylum", ""),
            "class": row.get("class", ""),
            "order": row.get("order", ""),
            "family": row.get("family", ""),
            "genus": row.get("genus", ""),
            "species": row.get("species_name", ""),
        }

    def get_confound_columns(self) -> list[str]:
        """Return potential confound columns."""
        return ["location_uncertainty", "hemisphere", "year"]

    def set_active_split(self, split: str) -> None:
        """Switch between train/val/test splits."""
        if split not in ("train", "val", "test", "all"):
            raise ValueError(
                f"Unknown split: {split}. Must be 'train', 'val', 'test', or 'all'"
            )
        self.active_split = split

    def get_summary(self) -> dict[str, Any]:
        """Get a rich dataset summary for the hypothesis generator."""
        df = self._get_active_data()

        n_species = int(df["species_name"].nunique())
        n_genera = int(df["genus"].nunique())
        n_families = int(df["family"].nunique())
        n_orders = int(df["order"].nunique())
        n_classes = int(df["class"].nunique())
        n_phyla = int(df["phylum"].nunique())
        n_kingdoms = int(df["kingdom"].nunique())

        lat_range = (float(df["latitude"].min()), float(df["latitude"].max()))
        lon_range = (float(df["longitude"].min()), float(df["longitude"].max()))

        date_min = str(df["date"].min())
        date_max = str(df["date"].max())

        top_species = df["species_name"].value_counts().head(10).to_dict()
        kingdom_dist = df["kingdom"].value_counts().to_dict()
        class_dist = df["class"].value_counts().head(10).to_dict()

        example = df.iloc[0].to_dict()
        for k, v in example.items():
            if hasattr(v, "isoformat"):
                example[k] = str(v)
            elif isinstance(v, (np.integer,)):
                example[k] = int(v)
            elif isinstance(v, (np.floating,)):
                example[k] = float(v)

        summary: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_samples": len(df),
            "active_split": self.active_split,
            "data_type": "iNaturalist 2024 biodiversity observations",
            "description": (
                f"A sample of {len(self.data):,} research-grade iNaturalist 2024 "
                f"observations spanning {n_species:,} species across "
                f"{n_kingdoms} kingdoms (Animalia, Plantae, Fungi). "
                f"Each observation includes geographic coordinates, date, "
                f"location uncertainty, and full taxonomic hierarchy "
                f"(kingdom > phylum > class > order > family > genus > species). "
                f"Derived columns include month, year, day_of_year, hemisphere, "
                f"abs_latitude, and season_nh. "
                f"Currently viewing the {self.active_split} split "
                f"({len(df):,} observations). "
                f"Look for ecological patterns: phenological timing, latitudinal "
                f"diversity gradients, taxonomic biogeography, seasonal differences "
                f"across kingdoms/classes, migration signals, and observer biases."
            ),
            "columns": list(df.columns),
            "taxonomy_counts": {
                "species": n_species,
                "genera": n_genera,
                "families": n_families,
                "orders": n_orders,
                "classes": n_classes,
                "phyla": n_phyla,
                "kingdoms": n_kingdoms,
            },
            "geographic_range": {
                "latitude": lat_range,
                "longitude": lon_range,
            },
            "date_range": {
                "min": date_min,
                "max": date_max,
            },
            "top_10_species": top_species,
            "kingdom_distribution": kingdom_dist,
            "class_distribution_top10": class_dist,
            "confounds": self.get_confound_columns(),
            "example_row": example,
            "variables": {},
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary["variables"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

        cat_cols = [
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species_name",
            "hemisphere",
            "season_nh",
        ]
        for col in cat_cols:
            if col in df.columns:
                summary["variables"][col] = {
                    "type": "categorical",
                    "n_unique": int(df[col].nunique()),
                    "top_values": df[col].value_counts().head(10).to_dict(),
                }

        return summary

    # ---- Prior knowledge -------------------------------------------------

    def get_prior_insights(self) -> list[Insight]:
        """Get pre-loaded prior insights."""
        return self._prior_insights

    def load_prior_insights_into_store(self, store: InsightStore) -> None:
        """Load prior insights into an InsightStore."""
        for insight in self._prior_insights:
            store.add_insight(insight)

    def get_prior_knowledge_prompt(self) -> str:
        """Get prior knowledge as text for hypothesis generator prompt."""
        if not self.include_prior_in_prompt or self.prior_prompt_detail == "none":
            return ""
        if not self.prior_patterns:
            return ""

        lines = [
            "## Prior Knowledge (Already Established Facts)",
            "",
            "The following patterns are already well-established in the iNaturalist 2024 dataset.",
            "Do NOT propose hypotheses that merely re-test these known relationships.",
            "Instead, look for NOVEL ecological, biogeographic, and phenological patterns.",
            "",
        ]

        if self.prior_prompt_detail == "full":
            for i, pattern in enumerate(self.prior_patterns, 1):
                desc = pattern.get("description", f"Pattern {pattern['id']}")
                variables = pattern.get("variables", [])
                lines.append(f"{i}. **{desc}**")
                if variables:
                    lines.append(f"   - Variables: {', '.join(variables)}")
                lines.append("")
        elif self.prior_prompt_detail == "summary":
            lines.append("Known patterns involve these relationships:")
            for pattern in self.prior_patterns:
                variables = pattern.get("variables", [])
                if variables:
                    lines.append(f"- {' <-> '.join(variables)}")
            lines.append("")

        return "\n".join(lines)

    # ---- Ground truth / evaluation ---------------------------------------

    def get_ground_truth_patterns(self) -> dict[str, list[dict[str, Any]]]:
        """Get all ground truth patterns by tier."""
        return {
            "prior_knowledge": self.prior_patterns,
            "obvious": self.obvious_patterns,
            "novel": self.novel_patterns,
        }

    def get_novel_patterns(self) -> list[dict[str, Any]]:
        """Get only the novel patterns that should be discovered."""
        return self.novel_patterns

    def get_evaluation_criteria(self) -> dict[str, Any]:
        """Get evaluation criteria from ground truth."""
        return self.ground_truth.get(
            "evaluation_criteria",
            {
                "scoring": {
                    "prior_rediscovery": -1.0,
                    "obvious_rediscovery": -0.5,
                    "novel_simple_found": 0.5,
                    "novel_moderate_found": 1.0,
                    "novel_complex_found": 2.0,
                },
                "matching": {
                    "variable_overlap_threshold": 0.5,
                    "effect_direction_must_match": True,
                },
            },
        )

    def save_prior_insights(self, path: str) -> None:
        """Save prior insights to parquet file."""
        if not self._prior_insights:
            return
        records = [ins.to_dict() for ins in self._prior_insights]
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)
