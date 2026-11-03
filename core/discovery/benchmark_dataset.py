# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark dataset adapter for evaluating discovery agents.

Extends the base DatasetAdapter with:
- Pre-loaded prior knowledge (InsightStore)
- Prompt context for hypothesis generator
- Ground truth patterns for evaluation
"""

from typing import Optional, Any
import os
import json
import pandas as pd
import numpy as np

from core.discovery.dataset_adapter import DatasetAdapter
from core.discovery.insight_store import InsightStore
from core.discovery.types import Insight, Hypothesis, Experiment, Result
from core.discovery.benchmark_patterns import (
    BenchmarkSpec,
    Pattern,
    PatternTier,
    PatternDifficulty,
    PatternType,
    CorrelationPattern,
)


class BenchmarkDatasetAdapter(DatasetAdapter):
    """
    Dataset adapter for benchmark evaluation with known ground truth patterns.
    
    Provides:
    - Standard DatasetAdapter interface for discovery system
    - Prior knowledge pre-loaded into InsightStore
    - Prior knowledge text for hypothesis generator prompt
    - Ground truth patterns for evaluation
    """
    
    def __init__(
        self,
        data_path: str,
        ground_truth_path: str,
        prior_insights_path: Optional[str] = None,
        include_prior_in_prompt: bool = True,
        prior_prompt_detail: str = "full",
        random_seed: int = 42,
    ):
        """
        Initialize benchmark dataset adapter.
        
        Args:
            data_path: Path to data.parquet file
            ground_truth_path: Path to ground_truth.json file
            prior_insights_path: Path to prior_insights.parquet (optional)
            include_prior_in_prompt: Whether to include prior knowledge in prompt
            prior_prompt_detail: Level of detail ("full", "summary", "none")
            random_seed: Random seed for sampling
        """
        self.data_path = data_path
        self.ground_truth_path = ground_truth_path
        self.prior_insights_path = prior_insights_path
        self.include_prior_in_prompt = include_prior_in_prompt
        self.prior_prompt_detail = prior_prompt_detail
        self.random_seed = random_seed
        
        # Load data
        self.data = pd.read_parquet(data_path)
        
        # Load ground truth
        with open(ground_truth_path, "r") as f:
            self.ground_truth = json.load(f)
        
        # Parse ground truth patterns
        self._parse_ground_truth()
        
        # Load or create prior insights
        self._prior_insights: list[Insight] = []
        if prior_insights_path and os.path.exists(prior_insights_path):
            self._load_prior_insights(prior_insights_path)
        else:
            self._create_prior_insights_from_ground_truth()
    
    def _parse_ground_truth(self):
        """Parse ground truth JSON into pattern objects."""
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
    
    def _load_prior_insights(self, path: str):
        """Load prior insights from parquet file."""
        df = pd.read_parquet(path)
        self._prior_insights = [
            Insight.from_dict(row.to_dict()) 
            for _, row in df.iterrows()
        ]
    
    def _create_prior_insights_from_ground_truth(self):
        """Create prior insights from ground truth prior knowledge patterns."""
        for pattern in self.prior_patterns:
            # Create a synthetic insight for each prior knowledge pattern
            insight = Insight.create(
                hypothesis_id=f"prior_{pattern['id']}",
                experiment_id=f"prior_exp_{pattern['id']}",
                verdict="supported",
                claim=pattern.get("description", f"Prior knowledge: {pattern['id']}"),
                strength=0.9,  # High strength for prior knowledge
                novelty_proxy=0.0,  # Not novel - it's prior knowledge
            )
            self._prior_insights.append(insight)
    
    @property
    def dataset_id(self) -> str:
        return self.ground_truth.get("dataset_id", "benchmark")
    
    @property
    def dataset_version(self) -> str:
        seed = self.ground_truth.get("seed", 42)
        n_samples = self.ground_truth.get("n_samples", len(self.data))
        return f"v1_{n_samples}samples_seed{seed}"
    
    def sample(self, slice_spec: dict[str, Any], n: int) -> pd.DataFrame:
        """Sample rows based on filters."""
        df = self.data.copy()
        
        # Apply filters from slice_spec
        for key, value in slice_spec.items():
            if key in df.columns:
                if isinstance(value, list):
                    if len(value) == 2 and isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
                        # Range filter [min, max]
                        df = df[(df[key] >= value[0]) & (df[key] <= value[1])]
                    else:
                        # Categorical filter
                        df = df[df[key].isin(value)]
                elif isinstance(value, (int, float, str)):
                    df = df[df[key] == value]
        
        # Sample n rows
        if len(df) > n:
            df = df.sample(n=n, random_state=self.random_seed)
        
        return df
    
    def load_metadata(self, slice_spec: dict[str, Any], columns: list[str]) -> pd.DataFrame:
        """Load metadata columns for a slice."""
        df = self.sample(slice_spec, n=len(self.data))
        available_cols = [c for c in columns if c in df.columns]
        return df[available_cols]
    
    def load_embeddings(self, image_ids: list, embedding_type: str) -> np.ndarray:
        """Load mock embeddings for benchmark data."""
        np.random.seed(self.random_seed)
        embedding_dim = 512 if embedding_type == "clip" else 256
        return np.random.randn(len(image_ids), embedding_dim)
    
    def get_taxonomy(self, label: str) -> dict[str, Any]:
        """Get taxonomy for a species label."""
        if label.startswith("species_"):
            idx = int(label.split("_")[1])
            return {
                "species": label,
                "genus": f"genus_{idx // 3}",
                "family": f"family_{idx // 5}",
                "order": "order_0",
                "class": ["Mammalia", "Aves", "Reptilia"][idx % 3],
            }
        return {}
    
    def get_confound_columns(self) -> list[str]:
        """Get list of potential confound columns."""
        return [
            "observer_id",
            "observer_experience", 
            "sampling_in_hotspot",
            "biogeographic_region",
            "timestamp",
        ]
    
    def get_summary(self) -> dict[str, Any]:
        """Get dataset summary statistics."""
        summary = {
            "n_samples": len(self.data),
            "n_species": self.data["species_id"].nunique() if "species_id" in self.data.columns else 0,
            "columns": list(self.data.columns),
            "confounds": self.get_confound_columns(),
        }
        
        # Add variable statistics for numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        summary["variables"] = {}
        for col in numeric_cols[:20]:  # Limit to first 20 numeric columns
            summary["variables"][col] = {
                "min": float(self.data[col].min()),
                "max": float(self.data[col].max()),
                "mean": float(self.data[col].mean()),
                "std": float(self.data[col].std()),
            }
        
        return summary
    
    # =========================================================================
    # PRIOR KNOWLEDGE METHODS
    # =========================================================================
    
    def get_prior_insights(self) -> list[Insight]:
        """
        Get pre-loaded prior insights.
        
        These should be added to the InsightStore before running discovery
        to represent "known" facts that shouldn't be rediscovered.
        
        Returns:
            List of prior Insight objects
        """
        return self._prior_insights
    
    def load_prior_insights_into_store(self, store: InsightStore) -> None:
        """
        Load prior insights into an InsightStore.
        
        Args:
            store: InsightStore to populate with prior knowledge
        """
        for insight in self._prior_insights:
            store.add_insight(insight)
    
    def get_prior_knowledge_prompt(self) -> str:
        """
        Get prior knowledge as text for hypothesis generator prompt.
        
        Returns text describing what is already known about this dataset,
        which should discourage the agent from rediscovering these patterns.
        
        Returns:
            Text description of prior knowledge
        """
        if not self.include_prior_in_prompt or self.prior_prompt_detail == "none":
            return ""
        
        if not self.prior_patterns:
            return ""
        
        lines = [
            "## Prior Knowledge (Already Established Facts)",
            "",
            "The following patterns are already well-established in the scientific literature.",
            "Do NOT propose hypotheses that merely re-test these known relationships.",
            "Instead, look for NOVEL patterns that go beyond this prior knowledge.",
            "",
        ]
        
        if self.prior_prompt_detail == "full":
            for i, pattern in enumerate(self.prior_patterns, 1):
                desc = pattern.get("description", f"Pattern {pattern['id']}")
                pattern_type = pattern.get("type", "unknown")
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
                    lines.append(f"- {' ↔ '.join(variables)}")
            lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # GROUND TRUTH / EVALUATION METHODS
    # =========================================================================
    
    def get_ground_truth_patterns(self) -> dict[str, list[dict]]:
        """
        Get all ground truth patterns by tier.
        
        Returns:
            Dict with keys "prior_knowledge", "obvious", "novel"
        """
        return {
            "prior_knowledge": self.prior_patterns,
            "obvious": self.obvious_patterns,
            "novel": self.novel_patterns,
        }
    
    def get_novel_patterns(self) -> list[dict]:
        """Get only the novel patterns that should be discovered."""
        return self.novel_patterns
    
    def get_evaluation_criteria(self) -> dict[str, Any]:
        """Get evaluation criteria from ground truth."""
        return self.ground_truth.get("evaluation_criteria", {
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
            }
        })
    
    def check_insight_matches_pattern(
        self,
        insight: Insight,
        pattern: dict,
        variable_threshold: float = 0.5
    ) -> tuple[bool, float]:
        """
        Check if an insight matches a ground truth pattern.
        
        Args:
            insight: Discovered insight to check
            pattern: Ground truth pattern dict
            variable_threshold: Minimum variable overlap ratio
            
        Returns:
            Tuple of (matches: bool, confidence: float)
        """
        # Extract variables from insight claim (simplified heuristic)
        insight_vars = set()
        claim_lower = insight.claim.lower()
        
        # Check each ground truth variable
        pattern_vars = set(pattern.get("variables", []))
        if not pattern_vars:
            # Try other variable fields
            pattern_vars = set(pattern.get("chain", []))
            pattern_vars.update(pattern.get("observed_vars", []))
            pattern_vars.update(pattern.get("predictors", []))
            if pattern.get("outcome"):
                pattern_vars.add(pattern["outcome"])
        
        # Count how many pattern variables appear in insight
        matches = 0
        for var in pattern_vars:
            var_lower = var.lower().replace("_", " ")
            if var_lower in claim_lower or var.lower() in claim_lower:
                matches += 1
                insight_vars.add(var)
        
        if not pattern_vars:
            return False, 0.0
        
        overlap_ratio = matches / len(pattern_vars)
        is_match = overlap_ratio >= variable_threshold
        
        # Check direction if applicable
        if is_match and pattern.get("effect_size") is not None:
            effect = pattern["effect_size"]
            # Very simplified: check if claim mentions direction correctly
            if effect > 0 and "negative" in claim_lower:
                is_match = False
            elif effect < 0 and "positive" in claim_lower:
                is_match = False
        
        return is_match, overlap_ratio
    
    def save_prior_insights(self, path: str) -> None:
        """Save prior insights to parquet file."""
        if not self._prior_insights:
            return
        
        records = [ins.to_dict() for ins in self._prior_insights]
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, index=False)


def load_benchmark_from_directory(
    benchmark_dir: str,
    include_prior_in_prompt: bool = True,
    prior_prompt_detail: str = "full",
) -> BenchmarkDatasetAdapter:
    """
    Load a benchmark dataset from a directory.
    
    Expected structure:
        benchmark_dir/
        ├── data.parquet
        ├── ground_truth.json
        └── prior_insights.parquet (optional)
    
    Args:
        benchmark_dir: Path to benchmark directory
        include_prior_in_prompt: Whether to include prior knowledge in prompts
        prior_prompt_detail: Detail level for prior knowledge prompt
        
    Returns:
        BenchmarkDatasetAdapter instance
    """
    data_path = os.path.join(benchmark_dir, "data.parquet")
    gt_path = os.path.join(benchmark_dir, "ground_truth.json")
    prior_path = os.path.join(benchmark_dir, "prior_insights.parquet")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    prior_path = prior_path if os.path.exists(prior_path) else None
    
    return BenchmarkDatasetAdapter(
        data_path=data_path,
        ground_truth_path=gt_path,
        prior_insights_path=prior_path,
        include_prior_in_prompt=include_prior_in_prompt,
        prior_prompt_detail=prior_prompt_detail,
    )
