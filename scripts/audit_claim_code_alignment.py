#!/usr/bin/env python3
"""Audit whether experiment code actually validates what the insight claims.

Uses an LLM judge to check alignment between:
- The natural language claim
- The actual statistical test executed (method, variables, slicing)

Usage:
    python scripts/audit_claim_code_alignment.py --workspace results/inat800_v10_corrected
"""

import argparse
import json
import os
import time

AUDIT_PROMPT = """You are auditing whether a statistical experiment actually tests what the insight claims.

CLAIM: {claim}

EXPERIMENT DETAILS:
- Method: {method}
- Metric tested: {metric}
- Grouping variable: {group}
- Data slice: {slice_spec}
- Feature spec: {feature_spec}

Does the experiment actually test the claim? Consider:
1. Does the grouping variable match what the claim compares? (e.g., claim about species but grouped by class_name)
2. Does the metric match what the claim measures? (e.g., claim about migration but testing latitude)
3. Does the data slice include the right samples? (e.g., claim about mammals but slice includes plants)
4. Are there any mismatches between claim specificity and test generality?

Reply with JSON:
{{
    "aligned": true/false,
    "alignment_score": 0-100,
    "issues": ["list of specific misalignment issues, empty if aligned"],
    "severity": "none" / "minor" / "major",
    "explanation": "one sentence summary"
}}"""


def audit_one(claim, experiment, model_name, model_url, api_key):
    """Audit one claim-experiment pair."""
    import litellm

    method = experiment.get("method", "unknown")
    feature_spec = experiment.get("feature_spec", {})
    slice_spec = experiment.get("dataset_slice_spec", {})

    metric = feature_spec.get("metric", feature_spec.get("attribute_query", "unknown"))
    group = feature_spec.get("group", feature_spec.get("group_col", "unknown"))

    prompt = AUDIT_PROMPT.format(
        claim=claim,
        method=method,
        metric=metric,
        group=group,
        slice_spec=json.dumps(slice_spec),
        feature_spec=json.dumps(feature_spec),
    )

    kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    if model_url:
        kwargs["api_base"] = model_url
    if api_key:
        kwargs["api_key"] = api_key

    resp = litellm.completion(**kwargs)
    text = resp.choices[0].message.content.strip()

    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {
        "aligned": None,
        "alignment_score": 0,
        "issues": ["Failed to parse"],
        "severity": "unknown",
        "explanation": text[:200],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--model-name", default="claude-sonnet-4-5")
    parser.add_argument("--model-url", default="https://YOUR_API_ENDPOINT/")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    api_key = os.environ.get("LITELLM_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )

    results = []
    for entry in sorted(os.listdir(args.workspace)):
        if not entry.startswith("iteration_"):
            continue
        fp = os.path.join(args.workspace, entry, "discovery.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            disc = json.load(f)

        insight = disc.get("insight", {})
        if insight.get("verdict") != "supported":
            continue

        experiment = disc.get("experiment", {})
        claim = insight.get("claim", "")

        print(f"\nAuditing {entry}...")
        print(f"  Claim: {claim[:80]}...")

        try:
            audit = audit_one(
                claim, experiment, args.model_name, args.model_url, api_key
            )
            audit["iteration"] = entry
            audit["claim"] = claim[:200]
            audit["method"] = experiment.get("method", "")
            results.append(audit)

            icon = "✓" if audit.get("aligned") else "✗"
            print(
                f"  {icon} Score: {audit.get('alignment_score', '?')}/100 ({audit.get('severity', '?')})"
            )
            if audit.get("issues"):
                for issue in audit["issues"][:2]:
                    print(f"    - {issue}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(0.5)

    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    aligned = sum(1 for r in results if r.get("aligned"))
    misaligned = sum(1 for r in results if not r.get("aligned"))
    avg_score = sum(r.get("alignment_score", 0) for r in results) / max(1, len(results))
    major = sum(1 for r in results if r.get("severity") == "major")
    minor = sum(1 for r in results if r.get("severity") == "minor")

    print(f"Total audited: {len(results)}")
    print(f"Aligned: {aligned}, Misaligned: {misaligned}")
    print(f"Average score: {avg_score:.0f}/100")
    print(f"Major issues: {major}, Minor issues: {minor}")

    output = args.output or os.path.join(args.workspace, "claim_code_audit.json")
    with open(output, "w") as f:
        json.dump(
            {
                "summary": {
                    "total": len(results),
                    "aligned": aligned,
                    "misaligned": misaligned,
                    "avg_score": avg_score,
                    "major": major,
                    "minor": minor,
                },
                "audits": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
