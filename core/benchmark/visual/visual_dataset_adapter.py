"""
Visual benchmark dataset adapter.

Provides the DatasetAdapter interface over visual ecology benchmark data.
The discovery pipeline operates on extracted features (not raw images).
Ground truth tabular data is preserved separately for evaluation.

Data layout:
    benchmark_dir/
    ├── images/               # Rendered ecology scenes
    ├── extracted_features.parquet  # What discovery sees
    ├── tabular_data.parquet  # Ground truth causal values
    ├── ground_truth.json     # Pattern specifications
    ├── prior_insights.parquet
    ├── benchmark_meta.json
    └── feature_variable_mapping.json
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.insight_store import InsightStore
from core.discovery.types import Insight


class VisualBenchmarkDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter for visual ecology benchmarks.

    Data flow:
    1. Causal DAG → tabular data (ground truth, hidden from discovery)
    2. Tabular data → rendered images (visual encoding of causal variables)
    3. Rendered images → extracted features (noisy recovery)
    4. Discovery pipeline operates on extracted features only
    5. Evaluation compares discoveries against ground truth patterns
    """

    def __init__(
        self,
        features_path: str,
        ground_truth_path: str,
        tabular_data_path: str,
        images_dir: str,
        prior_insights_path: Optional[str] = None,
        include_prior_in_prompt: bool = True,
        prior_prompt_detail: str = "full",
        random_seed: int = 42,
        feature_variable_mapping_path: Optional[str] = None,
    ):
        """
        Initialize visual benchmark dataset adapter.

        Args:
            features_path: Path to extracted_features.parquet
            ground_truth_path: Path to ground_truth.json
            tabular_data_path: Path to tabular_data.parquet (hidden ground truth)
            images_dir: Path to images/ directory
            prior_insights_path: Path to prior_insights.parquet (optional)
            include_prior_in_prompt: Whether to include prior knowledge in prompt
            prior_prompt_detail: Detail level ("full", "summary", "none")
            random_seed: Random seed for sampling
            feature_variable_mapping_path: Path to mapping JSON (for evaluation)
        """
        self.features_path = features_path
        self.ground_truth_path = ground_truth_path
        self.tabular_data_path = tabular_data_path
        self.images_dir = images_dir
        self.prior_insights_path = prior_insights_path
        self.include_prior_in_prompt = include_prior_in_prompt
        self.prior_prompt_detail = prior_prompt_detail
        self.random_seed = random_seed

        # Load extracted features (what discovery system sees)
        self.features = pd.read_parquet(features_path)

        # Load ground truth
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)

        # Load tabular data (hidden — for evaluation only)
        self._tabular_data = pd.read_parquet(tabular_data_path)

        # Load feature-to-variable mapping (for evaluation only)
        self._feature_variable_mapping: dict[str, str] = {}
        if feature_variable_mapping_path and os.path.exists(
            feature_variable_mapping_path
        ):
            with open(feature_variable_mapping_path, "r") as f:
                self._feature_variable_mapping = json.load(f)

        # Parse ground truth patterns
        self._parse_ground_truth()

        # Load or create prior insights
        self._prior_insights: list[Insight] = []
        if prior_insights_path and os.path.exists(prior_insights_path):
            self._load_prior_insights(prior_insights_path)
        else:
            self._create_prior_insights_from_ground_truth()

        # Train/val split support
        self.active_split = "train"
        self._create_splits()

    def _create_splits(self) -> None:
        """Create train/val splits (80/20)."""
        n = len(self.features)
        rng = np.random.default_rng(self.random_seed)
        indices = rng.permutation(n)
        split_idx = int(n * 0.8)

        self._train_indices = sorted(indices[:split_idx].tolist())
        self._val_indices = sorted(indices[split_idx:].tolist())

    def _get_active_data(self) -> pd.DataFrame:
        """Get features DataFrame for the active split."""
        if self.active_split == "train":
            return self.features.iloc[self._train_indices].reset_index(drop=True)
        elif self.active_split == "val":
            return self.features.iloc[self._val_indices].reset_index(drop=True)
        return self.features

    def set_active_split(self, split: str) -> None:
        """Switch between train/val splits."""
        if split not in ("train", "val", "all"):
            raise ValueError(
                f"Unknown split: {split}. Must be 'train', 'val', or 'all'"
            )
        self.active_split = split

    def _parse_ground_truth(self) -> None:
        """Parse ground truth JSON into pattern lists by tier."""
        self.prior_patterns: list[dict] = []
        self.obvious_patterns: list[dict] = []
        self.novel_patterns: list[dict] = []

        for pattern in self.ground_truth.get("patterns", []):
            tier = pattern.get("tier", "novel")
            if tier == "prior_knowledge":
                self.prior_patterns.append(pattern)
            elif tier == "obvious":
                self.obvious_patterns.append(pattern)
            else:
                self.novel_patterns.append(pattern)

    def _load_prior_insights(self, path: str) -> None:
        """Load prior insights from parquet file."""
        df = pd.read_parquet(path)
        self._prior_insights = [
            Insight.from_dict(row.to_dict()) for _, row in df.iterrows()
        ]

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

    # =========================================================================
    # DatasetAdapter interface
    # =========================================================================

    @property
    def dataset_id(self) -> str:
        return f"visual_ecology_benchmark_{self.active_split}"

    @property
    def dataset_version(self) -> str:
        seed = self.ground_truth.get("seed", self.random_seed)
        n_samples = len(self.features)
        return f"visual_v1_{n_samples}samples_seed{seed}"

    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """Sample rows from extracted features based on filters."""
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
        """Load feature columns for a slice."""
        df = self.sample(slice_spec, n=len(self.features))
        available_cols = [c for c in columns if c in df.columns]
        return df[available_cols]

    def load_embeddings(self, image_ids: list, embedding_type: str) -> np.ndarray:
        """Load embeddings (random if deep features not available)."""
        np.random.seed(self.random_seed)
        embedding_dim = 512 if embedding_type == "clip" else 256
        return np.random.randn(len(image_ids), embedding_dim)

    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """Not applicable for visual ecology data."""
        return {}

    def get_confound_columns(self) -> list[str]:
        """Return visual confound columns."""
        return ["avg_brightness", "texture_energy", "color_diversity"]

    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary for the hypothesis generator."""
        df = self._get_active_data()
        summary: dict[str, Any] = {
            "dataset_id": self.dataset_id,
            "n_samples": len(df),
            "data_type": "visual_ecology",
            "description": (
                "Ecology landscape scenes with extracted visual features. "
                "Features represent color, spatial layout, texture, and object-count "
                "properties measured from landscape images depicting varying ecological "
                "conditions (elevation, climate, vegetation, wildlife)."
            ),
            "columns": list(df.columns),
            "confounds": self.get_confound_columns(),
            "variables": {},
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:25]:
            summary["variables"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
            }

        return summary

    # =========================================================================
    # Prior knowledge methods
    # =========================================================================

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
            "Instead, look for NOVEL patterns in the visual features.",
            "",
        ]

        if self.prior_prompt_detail == "full":
            for i, pattern in enumerate(self.prior_patterns, 1):
                desc = pattern.get("description", f"Pattern {pattern['id']}")
                variables = pattern.get("variables", [])
                effect = pattern.get("effect_size", pattern.get("effect", ""))

                lines.append(f"{i}. **{desc}**")
                if variables:
                    lines.append(f"   - Variables: {', '.join(variables)}")
                if effect:
                    lines.append(f"   - Effect size: {effect}")
                lines.append("")
        elif self.prior_prompt_detail == "summary":
            lines.append("Known patterns involve these relationships:")
            for pattern in self.prior_patterns:
                variables = pattern.get("variables", [])
                if variables:
                    lines.append(f"- {' <-> '.join(variables)}")
            lines.append("")

        return "\n".join(lines)

    # =========================================================================
    # Ground truth / evaluation methods
    # =========================================================================

    def get_ground_truth_patterns(self) -> dict[str, list[dict]]:
        """Get all ground truth patterns by tier."""
        return {
            "prior_knowledge": self.prior_patterns,
            "obvious": self.obvious_patterns,
            "novel": self.novel_patterns,
        }

    def get_novel_patterns(self) -> list[dict]:
        """Get only the novel patterns that should be discovered."""
        return self.novel_patterns

    def get_feature_variable_mapping(self) -> dict[str, str]:
        """
        Get mapping from extracted feature names to causal variable names.
        FOR EVALUATION ONLY — not exposed to the discovery system.
        """
        return self._feature_variable_mapping

    def get_tabular_data(self) -> pd.DataFrame:
        """
        Get the hidden ground truth tabular data.
        FOR EVALUATION / DEBUGGING ONLY.
        """
        return self._tabular_data

    def get_image_path(self, sample_idx: int) -> str:
        """Get path to an image by sample index."""
        return os.path.join(self.images_dir, f"sample_{sample_idx:05d}.png")

    def get_sample_images_base64(
        self,
        n: int = 5,
        seed: int = 42,
        max_size: int = 512,
    ) -> list[str]:
        """Return *n* sample images as base64-encoded PNG strings.

        Images are resized to *max_size* px (longest edge) to control token cost.
        """
        from PIL import Image
        import io

        rng = np.random.default_rng(seed)
        total = len(self.features)
        indices = rng.choice(total, size=min(n, total), replace=False)

        result: list[str] = []
        for idx in sorted(indices):
            path = self.get_image_path(int(idx))
            if not os.path.exists(path):
                continue
            img = Image.open(path).convert("RGB")
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        return result


def load_visual_benchmark_from_directory(
    benchmark_dir: str,
    include_prior_in_prompt: bool = True,
    prior_prompt_detail: str = "full",
) -> VisualBenchmarkDatasetAdapter:
    """
    Load a visual benchmark dataset from a directory.

    Expected structure:
        benchmark_dir/
        ├── images/
        ├── extracted_features.parquet
        ├── tabular_data.parquet
        ├── ground_truth.json
        ├── prior_insights.parquet (optional)
        ├── benchmark_meta.json
        └── feature_variable_mapping.json (optional)

    Args:
        benchmark_dir: Path to benchmark directory
        include_prior_in_prompt: Include prior knowledge in prompts
        prior_prompt_detail: Detail level for prior prompt

    Returns:
        VisualBenchmarkDatasetAdapter instance
    """
    features_path = os.path.join(benchmark_dir, "extracted_features.parquet")
    gt_path = os.path.join(benchmark_dir, "ground_truth.json")
    tabular_path = os.path.join(benchmark_dir, "tabular_data.parquet")
    images_dir = os.path.join(benchmark_dir, "images")
    prior_path = os.path.join(benchmark_dir, "prior_insights.parquet")
    mapping_path = os.path.join(benchmark_dir, "feature_variable_mapping.json")

    for required, name in [
        (features_path, "extracted_features.parquet"),
        (gt_path, "ground_truth.json"),
        (tabular_path, "tabular_data.parquet"),
    ]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required file not found: {required}")

    return VisualBenchmarkDatasetAdapter(
        features_path=features_path,
        ground_truth_path=gt_path,
        tabular_data_path=tabular_path,
        images_dir=images_dir,
        prior_insights_path=prior_path if os.path.exists(prior_path) else None,
        include_prior_in_prompt=include_prior_in_prompt,
        prior_prompt_detail=prior_prompt_detail,
        feature_variable_mapping_path=mapping_path
        if os.path.exists(mapping_path)
        else None,
    )
