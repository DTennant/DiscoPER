#!/usr/bin/env python3
"""Rank insights by interestingness using pairwise LLM-as-judge.

Performs pairwise comparisons between supported insights, then uses
win counts to produce a ranked list from most to least interesting.

Usage:
    python scripts/rank_insights_by_interestingness.py \
        --insights results/inat800_v10_corrected/insights.parquet \
        --output results/inat800_v10_corrected/interestingness_ranking.json \
        --model-name claude-sonnet-4-5 \
        --model-url https://YOUR_API_ENDPOINT/
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd


JUDGE_PROMPT = """You are an expert scientist judging which of two discoveries is MORE INTERESTING.

"Interesting" means: surprising, non-obvious, has broader implications, reveals a hidden mechanism, or challenges conventional understanding. A trivial or expected finding is LESS interesting than a counter-intuitive or novel one.

Discovery A:
{claim_a}
(Effect size: {effect_a}, p-value: {p_a})

Discovery B:
{claim_b}
(Effect size: {effect_b}, p-value: {p_b})

Which discovery is MORE INTERESTING? Reply with ONLY "A" or "B" and a one-sentence reason.
Format: A: <reason> OR B: <reason>"""


def pairwise_compare(
    claim_a, effect_a, p_a, claim_b, effect_b, p_b, model_name, model_url, api_key
):
    """Compare two insights and return winner ("A" or "B") and reason."""
    import litellm

    prompt = JUDGE_PROMPT.format(
        claim_a=claim_a,
        effect_a=effect_a or "N/A",
        p_a=p_a or "N/A",
        claim_b=claim_b,
        effect_b=effect_b or "N/A",
        p_b=p_b or "N/A",
    )

    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    if model_url:
        kwargs["api_base"] = model_url
    if api_key:
        kwargs["api_key"] = api_key

    resp = litellm.completion(**kwargs)
    text = resp.choices[0].message.content.strip()

    if text.startswith("A"):
        return "A", text[2:].strip() if len(text) > 2 else ""
    elif text.startswith("B"):
        return "B", text[2:].strip() if len(text) > 2 else ""
    else:
        return "A" if "A" in text[:10] else "B", text


def rank_insights(insights, model_name, model_url, api_key):
    """Perform pairwise tournament and return ranked list."""
    n = len(insights)
    wins = {i: 0 for i in range(n)}
    comparisons = []

    total_pairs = n * (n - 1) // 2
    print(f"Performing {total_pairs} pairwise comparisons...")

    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            try:
                winner, reason = pairwise_compare(
                    insights[i]["claim"],
                    insights[i].get("effect_size"),
                    insights[i].get("p_value"),
                    insights[j]["claim"],
                    insights[j].get("effect_size"),
                    insights[j].get("p_value"),
                    model_name,
                    model_url,
                    api_key,
                )
                if winner == "A":
                    wins[i] += 1
                else:
                    wins[j] += 1

                comparisons.append(
                    {
                        "a_idx": i,
                        "b_idx": j,
                        "winner": winner,
                        "reason": reason,
                    }
                )
            except Exception as e:
                print(f"  Comparison {i} vs {j} failed: {e}")

            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total_pairs} comparisons done")
            time.sleep(0.5)

    ranked_indices = sorted(range(n), key=lambda i: wins[i], reverse=True)
    return ranked_indices, wins, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--insights", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-name", default="claude-sonnet-4-5")
    parser.add_argument("--model-url", default="https://YOUR_API_ENDPOINT/")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Only rank top-k supported insights (default: all supported)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )

    df = pd.read_parquet(args.insights)
    supported = df[df["verdict"] == "supported"].reset_index(drop=True)
    print(f"Loaded {len(df)} insights, {len(supported)} supported")

    if args.top_k and args.top_k < len(supported):
        supported = supported.head(args.top_k)
        print(f"Ranking top {args.top_k} only")

    insights_list = []
    for _, row in supported.iterrows():
        gen = row.get("generalization", {})
        if isinstance(gen, dict):
            tm = gen.get("train_metrics", {})
            effect = tm.get("effect_size")
            pval = tm.get("p_value")
        else:
            effect = None
            pval = None

        insights_list.append(
            {
                "id": row.get("id", ""),
                "claim": str(row.get("claim", "")),
                "effect_size": effect,
                "p_value": pval,
                "is_meta": bool(row.get("is_meta", False)),
                "variables": list(row.get("variables", []))
                if hasattr(row.get("variables", []), "__iter__")
                else [],
            }
        )

    if len(insights_list) < 2:
        print("Need at least 2 supported insights to rank")
        return

    ranked_indices, wins, comparisons = rank_insights(
        insights_list, args.model_name, args.model_url, api_key
    )

    print(f"\n{'=' * 60}")
    print("INTERESTINGNESS RANKING")
    print(f"{'=' * 60}")
    for rank, idx in enumerate(ranked_indices):
        ins = insights_list[idx]
        print(f"\n#{rank + 1} (wins: {wins[idx]}/{len(insights_list) - 1})")
        print(f"  {ins['claim'][:120]}")

    output = args.output or args.insights.replace(".parquet", "_interestingness.json")
    result = {
        "n_insights": len(insights_list),
        "n_comparisons": len(comparisons),
        "ranking": [
            {
                "rank": rank + 1,
                "wins": wins[idx],
                "total_matches": len(insights_list) - 1,
                "win_rate": round(wins[idx] / max(1, len(insights_list) - 1), 3),
                **insights_list[idx],
            }
            for rank, idx in enumerate(ranked_indices)
        ],
        "comparisons": comparisons,
    }

    with open(output, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved ranking to {output}")


if __name__ == "__main__":
    main()
