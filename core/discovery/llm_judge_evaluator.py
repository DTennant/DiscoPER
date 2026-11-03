"""LLM-as-judge evaluator for scoring predicted differences against ground truth."""

from __future__ import annotations

import json
from typing import Any, Optional

from core.agent import Agent
from core.prompts import visdiff_prompts


class LLMJudgeEvaluator:
    """Scores predicted visual differences against ground truth using an LLM judge."""

    def __init__(
        self,
        model_url: str,
        model_name: str,
        secrets: Optional[dict[str, str]] = None,
        temperature: float = 0.0,
    ):
        self.agent = Agent(
            model_url=model_url,
            model_name=model_name,
            system_prompt=(
                "You are an impartial judge evaluating visual difference predictions. "
                "Score predictions against ground truth on a 0-2 scale. "
                "Be strict but fair — capture semantic equivalence, not just wording."
            ),
            secrets=secrets or {},
        )
        self.temperature = temperature

    def score_one(self, prediction: str, ground_truth: str) -> dict[str, Any]:
        """Score a single prediction against ground truth. Returns {score, reasoning}."""
        prompt = visdiff_prompts.JUDGE_SET_COMPARISON_PROMPT.format(
            ground_truth=ground_truth,
            prediction=prediction,
        )

        output_schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "enum": [0, 1, 2]},
                "reasoning": {"type": "string"},
            },
            "required": ["score", "reasoning"],
        }

        try:
            result = self.agent.act_json(
                instruction=prompt,
                output_schema=output_schema,
                tool_name="judge_prediction",
                tool_description="Judge whether predicted difference matches ground truth",
                max_retries=2,
            )
            return {
                "score": int(result.get("score", 0)),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            return {"score": 0, "reasoning": f"Judge error: {e}"}

    def evaluate_pair(
        self,
        predictions: list[str],
        ground_truth: str,
    ) -> dict[str, Any]:
        """Evaluate multiple predictions for a single pair against ground truth."""
        details: list[dict[str, Any]] = []
        for pred in predictions:
            result = self.score_one(pred, ground_truth)
            details.append({"prediction": pred, **result})

        scores = [d["score"] for d in details]
        best_score = max(scores) if scores else 0
        best_idx = scores.index(best_score) if scores else 0

        return {
            "best_score": best_score,
            "best_prediction": predictions[best_idx] if predictions else "",
            "best_reasoning": details[best_idx]["reasoning"] if details else "",
            "acc_at_1": 1.0 if scores and scores[0] == 2 else 0.0,
            "hit_at_n": 1.0 if best_score == 2 else 0.0,
            "partial_at_n": 1.0 if best_score >= 1 else 0.0,
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "n_predictions": len(predictions),
            "details": details,
        }

    def evaluate_batch(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate evaluation results across multiple pairs.

        Args:
            results: List of dicts, each with keys:
                     {pair_idx, difficulty, ground_truth, predictions, ...eval_result}
        """
        by_difficulty: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            d = r.get("difficulty", "unknown")
            by_difficulty.setdefault(d, []).append(r)

        summary: dict[str, Any] = {"total_pairs": len(results), "by_difficulty": {}}
        all_best = []

        for diff, pairs in sorted(by_difficulty.items()):
            best_scores = [p.get("best_score", 0) for p in pairs]
            n = len(pairs)
            exact = sum(1 for s in best_scores if s == 2)
            partial = sum(1 for s in best_scores if s >= 1)
            all_best.extend(best_scores)

            summary["by_difficulty"][diff] = {
                "n_pairs": n,
                "exact_matches": exact,
                "exact_match_rate": exact / n if n else 0,
                "partial_matches": partial,
                "partial_match_rate": partial / n if n else 0,
                "mean_best_score": sum(best_scores) / n if n else 0,
            }

        n_total = len(all_best)
        summary["overall"] = {
            "exact_match_rate": sum(1 for s in all_best if s == 2) / n_total
            if n_total
            else 0,
            "partial_match_rate": sum(1 for s in all_best if s >= 1) / n_total
            if n_total
            else 0,
            "mean_best_score": sum(all_best) / n_total if n_total else 0,
        }

        return summary
