# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Any, Optional
import os
import pandas as pd
import numpy as np

from core.discovery.types import Insight

_embedding_model: Any = None
_embedding_tried = False


def _get_embedding_model() -> Any:
    global _embedding_model, _embedding_tried
    if not _embedding_tried:
        _embedding_tried = True
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            _embedding_model = None
    return _embedding_model


def compute_claim_embedding(claim: str) -> Optional[list[float]]:
    model = _get_embedding_model()
    if model is None:
        return None
    vec = model.encode(claim, normalize_embeddings=True)
    return vec.tolist()


class InsightStore:
    """Storage and retrieval for validated insights."""

    def __init__(self, store_path: str, auto_embed: bool = True):
        self.store_path = store_path
        self.insights: list[Insight] = []
        self.auto_embed = auto_embed

        if os.path.exists(store_path):
            self.load(store_path)

    def add_insight(self, insight: Insight):
        if self.auto_embed and insight.embedding is None and insight.claim:
            insight.embedding = compute_claim_embedding(insight.claim)
        self.insights.append(insight)

    def query(
        self,
        similarity: Optional[float] = None,
        query_text: Optional[str] = None,
        query_embedding: Optional[list[float]] = None,
        by_variable: Optional[str] = None,
        by_scope: Optional[dict] = None,
        by_verdict: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Insight]:
        results = self.insights.copy()

        if by_verdict:
            results = [ins for ins in results if ins.verdict == by_verdict]

        if by_variable:
            results = [
                ins for ins in results if ins.variables and by_variable in ins.variables
            ]

        if similarity is not None:
            q_emb = query_embedding
            if q_emb is None and query_text is not None:
                q_emb_list = compute_claim_embedding(query_text)
                q_emb = q_emb_list
            if q_emb is not None:
                q = np.array(q_emb)
                scored = []
                for ins in results:
                    if ins.embedding is not None:
                        e = np.array(ins.embedding)
                        cos_sim = float(
                            np.dot(q, e)
                            / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-10)
                        )
                        if cos_sim >= similarity:
                            scored.append((cos_sim, ins))
                results = [ins for _, ins in sorted(scored, key=lambda x: -x[0])]

        if limit:
            results = results[:limit]

        return results

    def get_graph(self) -> dict:
        """
        Get insight graph representation.

        Returns:
            Dict with "nodes" (insights) and "edges" (relationships)
        """
        nodes = [
            {
                "id": insight.id,
                "claim": insight.claim,
                "verdict": insight.verdict,
                "strength": insight.strength,
                "novelty": insight.novelty_proxy,
            }
            for insight in self.insights
        ]

        edges = []
        for insight in self.insights:
            if insight.edges:
                for edge in insight.edges:
                    edges.append(
                        {
                            "source": insight.id,
                            "target": edge.get("target_id"),
                            "relation": edge.get("relation"),
                        }
                    )

        return {"nodes": nodes, "edges": edges}

    def save(self, path: Optional[str] = None):
        """
        Save insights to parquet file.

        Args:
            path: Optional override path (defaults to store_path)
        """
        save_path = path or self.store_path

        if not self.insights:
            # No insights to save
            return

        # Convert insights to DataFrame
        records = [insight.to_dict() for insight in self.insights]
        df = pd.DataFrame(records)

        # Save to parquet
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_parquet(save_path, index=False)

    def load(self, path: Optional[str] = None):
        """
        Load insights from parquet file.

        Args:
            path: Optional override path (defaults to store_path)
        """
        load_path = path or self.store_path

        if not os.path.exists(load_path):
            return

        df = pd.read_parquet(load_path)
        self.insights = [Insight.from_dict(row.to_dict()) for _, row in df.iterrows()]

    def __len__(self) -> int:
        """Return number of insights in store."""
        return len(self.insights)
