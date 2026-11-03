"""
iNaturalist dataset adapter for visual ecology discovery.

Provides the DatasetAdapter interface over a curated iNaturalist subset.
Each observation has metadata (lat/lon, date, taxon) + a photo.

Data layout:
    data_dir/
    ├── data.parquet          # Observation metadata
    ├── images/               # Species subdirectories with photos
    │   ├── Danaus_plexippus/
    │   ├── Vulpes_vulpes/
    │   └── ...
    ├── ground_truth.json     # Known ecological patterns
    ├── prior_insights.parquet
    └── benchmark_meta.json
"""

from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.insight_store import InsightStore
from core.discovery.types import Insight


class INaturalistDatasetAdapter(DatasetAdapter):
    """Dataset adapter for iNaturalist curated ecology subset."""

    def __init__(
        self,
        data_dir: str,
        ground_truth_path: Optional[str] = None,
        prior_insights_path: Optional[str] = None,
        include_prior_in_prompt: bool = True,
        prior_prompt_detail: str = "full",
        random_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.include_prior_in_prompt = include_prior_in_prompt
        self.prior_prompt_detail = prior_prompt_detail

        # Load observation metadata
        data_path = os.path.join(data_dir, "data.parquet")
        self.data = pd.read_parquet(data_path)

        # Images directory
        self.images_dir = os.path.join(data_dir, "images")

        # Load ground truth
        gt_path = ground_truth_path or os.path.join(data_dir, "ground_truth.json")
        self.ground_truth: dict[str, Any] = {}
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                self.ground_truth = json.load(f)

        # Parse ground truth patterns by tier
        self.prior_patterns: list[dict[str, Any]] = []
        self.obvious_patterns: list[dict[str, Any]] = []
        self.novel_patterns: list[dict[str, Any]] = []
        for pattern in self.ground_truth.get("patterns", []):
            tier = pattern.get("tier", "novel")
            if tier == "prior_knowledge":
                self.prior_patterns.append(pattern)
            elif tier == "obvious":
                self.obvious_patterns.append(pattern)
            else:
                self.novel_patterns.append(pattern)

        # Load prior insights
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

        # Train/val split
        self.active_split = "train"
        self._create_splits()

    def _create_splits(self) -> None:
        """Create train/val splits (80/20)."""
        n = len(self.data)
        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(n)
        split_idx = int(n * 0.8)
        self._train_indices = sorted(indices[:split_idx].tolist())
        self._val_indices = sorted(indices[split_idx:].tolist())

    def _get_active_data(self) -> pd.DataFrame:
        """Get DataFrame for the active split."""
        if self.active_split == "train":
            return self.data.iloc[self._train_indices].reset_index(drop=True)
        elif self.active_split == "val":
            return self.data.iloc[self._val_indices].reset_index(drop=True)
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

    # ─── DatasetAdapter interface ─────────────────────────────────────────

    @property
    def dataset_id(self) -> str:
        return f"inaturalist_{self.active_split}"

    @property
    def dataset_version(self) -> str:
        n = len(self.data)
        return f"inat_v1_{n}obs_seed{self.random_seed}"

    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """Sample rows from observation metadata based on filters."""
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
        """Load embeddings (random placeholder)."""
        np.random.seed(self.random_seed)
        dim = 512 if embedding_type == "clip" else 256
        return np.random.randn(len(image_ids), dim)

    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """Return taxonomy info for a species."""
        matches = self.data[self.data["species_name"] == label]
        if len(matches) == 0:
            matches = self.data[self.data["common_name"] == label]
        if len(matches) == 0:
            return {}
        row = matches.iloc[0]
        return {
            "kingdom": row.get("kingdom", ""),
            "phylum": row.get("phylum", ""),
            "class": row.get("class_name", ""),
            "order": row.get("order_name", ""),
            "family": row.get("family", ""),
            "genus": row.get("genus", ""),
            "species": row.get("species_name", ""),
        }

    def get_confound_columns(self) -> list[str]:
        """Return potential confound columns."""
        return ["positional_accuracy", "hemisphere", "quality_grade"]

    def set_active_split(self, split: str) -> None:
        """Switch between train/val splits."""
        if split not in ("train", "val", "all"):
            raise ValueError(
                f"Unknown split: {split}. Must be 'train', 'val', or 'all'"
            )
        self.active_split = split

    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary for the hypothesis generator."""
        df = self._get_active_data()

        # Species distribution
        species_counts = df["species_name"].value_counts().to_dict()

        # Geographic range
        lat_range = (float(df["latitude"].min()), float(df["latitude"].max()))
        lon_range = (float(df["longitude"].min()), float(df["longitude"].max()))

        # Temporal range
        months_present = sorted(df["month"].unique().tolist())

        summary: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_samples": len(df),
            "data_type": "iNaturalist ecology observations",
            "description": (
                "Real-world iNaturalist observations of 8 ecologically interesting species. "
                "Each observation has geographic coordinates (latitude/longitude), observation date, "
                "taxonomic classification (kingdom through species), and a wildlife photo. "
                "Species span multiple kingdoms (Animalia, Plantae, Fungi) and include migratory "
                "species, seasonal fungi, widespread mammals, and common plants. "
                "Look for ecological patterns: migration timing, seasonal phenology, habitat "
                "preferences, geographic range differences, and cross-species interactions."
            ),
            "columns": list(df.columns),
            "species": species_counts,
            "n_species": len(species_counts),
            "latitude_range": lat_range,
            "longitude_range": lon_range,
            "months_present": months_present,
            "confounds": self.get_confound_columns(),
            "variables": {},
        }

        # Numeric column stats
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:20]:
            summary["variables"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

        # Categorical column value counts
        cat_cols = ["species_name", "kingdom", "class_name", "season", "hemisphere"]
        for col in cat_cols:
            if col in df.columns:
                summary["variables"][col] = {
                    "type": "categorical",
                    "n_unique": int(df[col].nunique()),
                    "values": df[col].value_counts().head(10).to_dict(),
                }

        return summary

    # ─── Vision support ──────────────────────────────────────────────────

    def get_sample_images_base64(
        self,
        n: int = 5,
        seed: int = 42,
        max_size: int = 512,
    ) -> list[str]:
        """Return n sample images as base64-encoded PNG strings."""
        from PIL import Image

        rng = np.random.default_rng(seed)
        df = self._get_active_data()
        valid = df[df["local_path"].astype(str).str.len() > 0]
        if len(valid) == 0:
            return []

        indices = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
        result: list[str] = []
        for idx in sorted(indices):
            local_path = valid.iloc[int(idx)]["local_path"]
            full_path = os.path.join(self.data_dir, str(local_path))
            if not os.path.exists(full_path):
                continue
            try:
                img = Image.open(full_path).convert("RGB")
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            except Exception:
                continue
        return result

    def get_sample_images_with_labels(
        self,
        n: int = 5,
        seed: int = 42,
        max_size: int = 512,
    ) -> tuple[list[str], list[dict]]:
        """Return sample images as base64 AND their metadata labels.

        Returns:
            Tuple of (base64_images, image_labels) where image_labels is a list
            of dicts with species_name, common_name, latitude, longitude, season, month.
        """
        from PIL import Image

        rng = np.random.default_rng(seed)
        df = self._get_active_data()
        valid = df[df["local_path"].astype(str).str.len() > 0]
        if len(valid) == 0:
            return [], []

        indices = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
        images: list[str] = []
        labels: list[dict] = []

        for idx in sorted(indices):
            row = valid.iloc[int(idx)]
            local_path = row["local_path"]
            full_path = os.path.join(self.data_dir, str(local_path))
            if not os.path.exists(full_path):
                continue
            try:
                img = Image.open(full_path).convert("RGB")
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                images.append(base64.b64encode(buf.getvalue()).decode("ascii"))
                labels.append(
                    {
                        "observation_id": row.get("observation_id", ""),
                        "species_name": row.get("species_name", "Unknown"),
                        "common_name": row.get("common_name", "Unknown"),
                        "kingdom": row.get("kingdom", "Unknown"),
                        "class_name": row.get("class_name", "Unknown"),
                        "latitude": round(float(row.get("latitude", 0)), 2),
                        "longitude": round(float(row.get("longitude", 0)), 2),
                        "season": row.get("season", "Unknown"),
                        "month": int(row.get("month", 0)),
                    }
                )
            except Exception:
                continue
        return images, labels

    # ─── Prior knowledge ─────────────────────────────────────────────────

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
            "The following patterns are already well-established.",
            "Do NOT propose hypotheses that merely re-test these known relationships.",
            "Instead, look for NOVEL ecological patterns.",
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

    # ─── Ground truth / evaluation ───────────────────────────────────────

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


def load_inat_from_directory(
    data_dir: str,
    include_prior_in_prompt: bool = True,
    prior_prompt_detail: str = "full",
) -> INaturalistDatasetAdapter:
    """Load an iNaturalist dataset from a directory."""
    return INaturalistDatasetAdapter(
        data_dir=data_dir,
        include_prior_in_prompt=include_prior_in_prompt,
        prior_prompt_detail=prior_prompt_detail,
    )
