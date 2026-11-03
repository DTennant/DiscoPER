# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Any
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class DatasetAdapter(ABC):
    """Abstract base class for dataset access."""

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Unique identifier for the dataset."""
        pass

    @property
    @abstractmethod
    def dataset_version(self) -> str:
        """Version string for the dataset."""
        pass

    @abstractmethod
    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """
        Sample rows from the dataset based on filters.

        Args:
            slice_spec: Filters to apply (e.g., {"species": ["species_0"], "elevation": [100, 500]})
            n: Number of samples to return

        Returns:
            DataFrame with sampled rows
        """
        pass

    @abstractmethod
    def load_metadata(self, slice_spec: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        """
        Load metadata for a slice of the dataset.

        Args:
            slice_spec: Filters to apply
            columns: List of column names to load

        Returns:
            DataFrame with requested columns
        """
        pass

    @abstractmethod
    def load_embeddings(self, image_ids: list, embedding_type: str) -> np.ndarray:
        """
        Load precomputed embeddings for given IDs.

        Args:
            image_ids: List of image/sample IDs
            embedding_type: Type of embedding (e.g., "clip", "resnet")

        Returns:
            Array of embeddings, shape (n_samples, embedding_dim)
        """
        pass

    @abstractmethod
    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """
        Get taxonomic hierarchy for a label.

        Args:
            label: Species or taxon label

        Returns:
            Dict with taxonomic levels (e.g., {"species": "...", "genus": "...", "family": "..."})
        """
        pass

    @abstractmethod
    def get_confound_columns(self) -> list[str]:
        """
        Get list of potential confound columns.

        Returns:
            List of column names that may be confounds (e.g., ["uploader_id", "location_density"])
        """
        pass
