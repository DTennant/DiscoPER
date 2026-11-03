# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Any
import pandas as pd
import numpy as np

from core.discovery.dataset_adapter import DatasetAdapter


class SyntheticDatasetAdapter(DatasetAdapter):
    """Synthetic dataset for MVP testing."""

    def __init__(
        self,
        n_samples: int = 1000,
        n_species: int = 10,
        random_seed: int = 42,
        ground_truth: Optional[list[dict]] = None
    ):
        """
        Initialize synthetic dataset.

        Args:
            n_samples: Number of samples to generate
            n_species: Number of species
            random_seed: Random seed for reproducibility
            ground_truth: List of ground truth correlations (e.g., [{"species": "species_0", "variable": "elevation", "effect": 0.7}])
        """
        self.n_samples = n_samples
        self.n_species = n_species
        self.random_seed = random_seed
        self.ground_truth = ground_truth or []

        # Generate synthetic data
        np.random.seed(random_seed)
        self.data = self._generate_data()

    def _generate_data(self) -> pd.DataFrame:
        """Generate synthetic ecological data."""
        # Base features
        data = {
            "sample_id": list(range(self.n_samples)),
            "species": np.random.choice([f"species_{i}" for i in range(self.n_species)], self.n_samples),
            "latitude": np.random.uniform(30, 50, self.n_samples),
            "longitude": np.random.uniform(-120, -80, self.n_samples),
            "elevation": np.random.uniform(0, 3000, self.n_samples),
            "temperature": np.random.uniform(-10, 35, self.n_samples),
            "precipitation": np.random.uniform(0, 200, self.n_samples),
            "observer_id": np.random.choice([f"observer_{i}" for i in range(50)], self.n_samples),
            "location_density": np.random.uniform(0, 1, self.n_samples),
            "timestamp": pd.date_range("2020-01-01", periods=self.n_samples, freq="H"),
        }

        df = pd.DataFrame(data)

        # Apply ground truth correlations
        for gt in self.ground_truth:
            species = gt.get("species")
            variable = gt.get("variable")
            effect = gt.get("effect", 0.0)

            if species and variable and variable in df.columns:
                # For samples of this species, add correlation
                mask = df["species"] == species
                if mask.sum() > 0:
                    # Add correlated noise to the variable
                    noise = np.random.normal(0, 1, mask.sum())
                    df.loc[mask, variable] += effect * (df.loc[mask, variable].std() * noise)

        return df

    @property
    def dataset_id(self) -> str:
        return "synthetic_v0"

    @property
    def dataset_version(self) -> str:
        return f"v0_{self.n_samples}samples_{self.n_species}species_seed{self.random_seed}"

    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """Sample rows based on filters."""
        df = self.data.copy()

        # Apply filters from slice_spec
        for key, value in slice_spec.items():
            if key in df.columns:
                if isinstance(value, list):
                    if len(value) == 2 and isinstance(value[0], (int, float)):
                        # Range filter [min, max]
                        df = df[(df[key] >= value[0]) & (df[key] <= value[1])]
                    else:
                        # Categorical filter
                        df = df[df[key].isin(value)]

        # Sample n rows
        if len(df) > n:
            df = df.sample(n=n, random_state=self.random_seed)

        return df

    def load_metadata(self, slice_spec: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        """Load metadata columns for a slice."""
        df = self.sample(slice_spec, n=len(self.data))
        return df[columns]

    def load_embeddings(self, image_ids: list, embedding_type: str) -> np.ndarray:
        """Load mock embeddings."""
        # For synthetic data, generate random embeddings
        np.random.seed(self.random_seed)
        embedding_dim = 512 if embedding_type == "clip" else 256
        return np.random.randn(len(image_ids), embedding_dim)

    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """Get mock taxonomy."""
        # For synthetic species, create simple taxonomy
        if label.startswith("species_"):
            idx = int(label.split("_")[1])
            return {
                "species": label,
                "genus": f"genus_{idx // 3}",
                "family": f"family_{idx // 5}",
                "order": "order_0",
                "class": "class_0"
            }
        return {}

    def get_confound_columns(self) -> list[str]:
        """Get list of potential confound columns."""
        return ["observer_id", "location_density", "timestamp"]

    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary statistics."""
        return {
            "n_samples": self.n_samples,
            "n_species": self.n_species,
            "species": list(self.data["species"].unique()),
            "variables": {
                "latitude": {"min": float(self.data["latitude"].min()), "max": float(self.data["latitude"].max())},
                "longitude": {"min": float(self.data["longitude"].min()), "max": float(self.data["longitude"].max())},
                "elevation": {"min": float(self.data["elevation"].min()), "max": float(self.data["elevation"].max())},
                "temperature": {"min": float(self.data["temperature"].min()), "max": float(self.data["temperature"].max())},
                "precipitation": {"min": float(self.data["precipitation"].min()), "max": float(self.data["precipitation"].max())},
            },
            "confounds": self.get_confound_columns()
        }
