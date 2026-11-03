#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extract insights from a discovery run into JSON for the static review website.

Usage:
    python scripts/generate_website_data.py \
        --insights results/run/insights.parquet \
        --output website/data.json

    python scripts/generate_website_data.py \
        --insights results/run/insights.parquet \
        --evaluation results/run/evaluation.json \
        --output website/data.json
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd


def _safe_value(val):
    """Convert pandas/numpy values to JSON-safe Python types (handles NaN, np.int64, etc.)."""
    if val is None:
        return None
    if isinstance(val, float):
        import math

        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 6)
    if isinstance(val, (int, bool, str)):
        return val
    try:
        import numpy as np

        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            if np.isnan(val) or np.isinf(val):
                return None
            return round(float(val), 6)
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.ndarray):
            return [_safe_value(v) for v in val.tolist()]
    except ImportError:
        pass
    # Parquet may store lists/dicts as JSON strings
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return val
    if isinstance(val, list):
        return [_safe_value(v) for v in val]
    if isinstance(val, dict):
        return {k: _safe_value(v) for k, v in val.items()}
    return str(val)


def extract_insight(row, index):
    insight = {
        "index": index,
        "id": _safe_value(row.get("id", f"insight_{index}")),
        "claim": _safe_value(row.get("claim", "")),
        "verdict": _safe_value(row.get("verdict", "unknown")),
        "strength": _safe_value(row.get("strength")),
        "variables": _safe_value(row.get("variables")),
        "is_meta": _safe_value(row.get("is_meta", False)),
    }

    # The generalization dict may contain nested train/val metrics or top-level ones
    gen = row.get("generalization")
    if isinstance(gen, str):
        try:
            gen = json.loads(gen)
        except (json.JSONDecodeError, ValueError):
            gen = None

    if isinstance(gen, dict):
        train = gen.get("train_metrics", gen.get("train", {}))
        val = gen.get("val_metrics", gen.get("val", {}))
        if isinstance(train, str):
            try:
                train = json.loads(train)
            except (json.JSONDecodeError, ValueError):
                train = {}
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                val = {}
        insight["effect_size"] = _safe_value(
            train.get("effect_size", gen.get("effect_size"))
        )
        insight["p_value"] = _safe_value(train.get("p_value", gen.get("p_value")))
        insight["val_effect_size"] = _safe_value(
            val.get("effect_size", gen.get("val_effect_size"))
        )
        insight["val_p_value"] = _safe_value(val.get("p_value", gen.get("val_p_value")))
        insight["method"] = _safe_value(
            gen.get("method", gen.get("experiment_method", train.get("method")))
        )
    else:
        insight["effect_size"] = _safe_value(row.get("effect_size"))
        insight["p_value"] = _safe_value(row.get("p_value"))
        insight["val_effect_size"] = _safe_value(row.get("val_effect_size"))
        insight["val_p_value"] = _safe_value(row.get("val_p_value"))
        insight["method"] = _safe_value(row.get("method"))

    if not insight.get("method"):
        claim_lower = str(insight.get("claim", "")).lower()
        if (
            "spearman" in claim_lower
            or "pearson" in claim_lower
            or "correlation" in claim_lower
        ):
            insight["method"] = "corr_test"
        elif (
            "mann-whitney" in claim_lower
            or "chi-sq" in claim_lower
            or "cohen" in claim_lower
        ):
            insight["method"] = "group_diff_test"
        elif "stratif" in claim_lower:
            insight["method"] = "stratified_retest"
        elif "predict" in claim_lower or "auc" in claim_lower:
            insight["method"] = "predictive_test"
        elif any(
            w in claim_lower
            for w in ["compared to", "higher than", "lower than", "differ", "vs"]
        ):
            insight["method"] = "group_diff_test"
        elif any(w in claim_lower for w in ["r=", "corr", "relationship between"]):
            insight["method"] = "corr_test"

    insight["hypothesis_id"] = _safe_value(row.get("hypothesis_id", ""))
    insight["experiment_id"] = _safe_value(row.get("experiment_id"))
    insight["iteration"] = index
    insight["novelty_proxy"] = _safe_value(row.get("novelty_proxy"))
    insight["reward"] = _safe_value(row.get("reward"))

    insight["vision_informed"] = False

    return insight


def load_evaluation(eval_path):
    """Load evaluation JSON and return (insight_id -> match_info dict, summary dict)."""
    with open(eval_path) as f:
        data = json.load(f)

    matches = {}

    # Prefer LLM judge results (per_insight with best_score on 0-2 scale)
    llm_judge = data.get("llm_judge", {})
    per_insight = llm_judge.get("per_insight", [])

    old_matches = data.get("matches", data.get("matched_insights", []))
    if not isinstance(old_matches, list):
        old_matches = []
    claim_to_id = {m.get("claim", "")[:100]: m.get("insight_id") for m in old_matches}

    if per_insight:
        for entry in per_insight:
            score = entry.get("best_score", 0)
            if score >= 1:
                claim = entry.get("claim", "")
                iid = entry.get("insight_id") or claim_to_id.get(claim[:100])
                key = iid or claim[:100]
                if key:
                    best_pattern = entry.get("best_pattern") or {}
                    if isinstance(best_pattern, str):
                        best_pattern = {"description": best_pattern}
                    matches[key] = {
                        "gt_pattern_id": best_pattern.get("pattern_id", ""),
                        "gt_description": best_pattern.get("description", ""),
                        "gt_tier": best_pattern.get("tier", "novel"),
                        "match_score": _safe_value(score),
                    }
    else:
        # Fall back to old-style string matches (score on 0-1 cosine scale, threshold at 0.7)
        for m in old_matches:
            score = m.get("score", m.get("match_score", 0))
            if score and score >= 0.7:
                iid = m.get("insight_id", m.get("id"))
                if iid:
                    matches[iid] = {
                        "gt_pattern_id": m.get("pattern_id", m.get("gt_id")),
                        "gt_description": m.get(
                            "description", m.get("gt_description", "")
                        ),
                        "gt_tier": m.get("tier", ""),
                        "gt_difficulty": m.get("difficulty", ""),
                        "match_score": _safe_value(score),
                    }

    summary = {
        k: _safe_value(v)
        for k, v in data.items()
        if k not in ("matches", "matched_insights", "details", "llm_judge")
    }
    if llm_judge:
        summary["llm_judge_summary"] = {
            k: _safe_value(v)
            for k, v in llm_judge.items()
            if k not in ("per_insight", "per_pattern_novel")
        }

    return matches, summary


def load_experiment_code(
    workspace_dir: str,
) -> tuple[dict[str, str], dict[str, dict], dict[str, list], set[str]]:
    """Load experiment details, per-split metrics, and evidence images from discovery.json files."""
    id_to_code: dict[str, str] = {}
    id_to_split_metrics: dict[str, dict] = {}
    id_to_evidence_images: dict[str, list] = {}
    id_to_uses_vision_tool: set[str] = set()
    if not workspace_dir or not os.path.isdir(workspace_dir):
        return (
            id_to_code,
            id_to_split_metrics,
            id_to_evidence_images,
            id_to_uses_vision_tool,
        )

    for entry in sorted(os.listdir(workspace_dir)):
        if not entry.startswith("iteration_"):
            continue
        disc_path = os.path.join(workspace_dir, entry, "discovery.json")
        if not os.path.exists(disc_path):
            continue
        is_meta_iter = "_meta" in entry
        with open(disc_path) as f:
            disc = json.load(f)

        insight = disc.get("insight", {})
        iid = insight.get("id", "")
        if not iid:
            continue

        hyp = disc.get("hypothesis", {})
        exp = disc.get("experiment", {})
        results = disc.get("results", {})

        method = exp.get("method", "unknown")
        if method in ("visual_attribute_test", "visual_group_comparison"):
            id_to_uses_vision_tool.add(iid)
        feature_spec = exp.get("feature_spec", {})
        slice_spec = exp.get("dataset_slice_spec", {})
        controls = exp.get("controls")

        lines = [
            "import pandas as pd",
            "from core.discovery.experiments import " + method,
            "",
            f"# Hypothesis: {hyp.get('statement', 'N/A')}",
            f"# Variables: {', '.join(hyp.get('variables', []))}",
            "",
            "# Load data",
            'df = pd.read_parquet("data.parquet")',
            "",
        ]

        if slice_spec:
            lines.append("# Apply data slice")
            for col, vals in slice_spec.items():
                if isinstance(vals, list):
                    lines.append(f'df = df[df["{col}"].isin({vals})]')
                else:
                    lines.append(f'df = df[df["{col}"] == {repr(vals)}]')
            lines.append("")

        lines.append(f"# Run experiment: {method}")
        if method == "corr_test":
            x = feature_spec.get("x", feature_spec.get("metric", "x"))
            y = feature_spec.get("y", feature_spec.get("group", "y"))
            corr_method = feature_spec.get("method", "spearman")
            lines.append(
                f'result = corr_test(df["{x}"], df["{y}"], method="{corr_method}")'
            )
        elif method == "group_diff_test":
            metric = feature_spec.get("metric", "metric")
            group = feature_spec.get("group", "group")
            test_method = feature_spec.get("method", "perm_test")
            effect = feature_spec.get("effect", "cohens_d")
            lines.append(f'result = group_diff_test(df["{metric}"], df["{group}"],')
            lines.append(
                f'                        method="{test_method}", effect="{effect}")'
            )
        elif method == "predictive_test":
            features = feature_spec.get("features", [])
            target = feature_spec.get("target", "target")
            lines.append(f"X = df[{features}]")
            lines.append(f'y = df["{target}"]')
            lines.append(f"result = predictive_test(X, y)")
        elif method == "stratified_retest":
            lines.append(f"result = stratified_retest(df, {json.dumps(feature_spec)})")
        elif method == "cluster_and_enrich":
            lines.append(f"result = cluster_and_enrich(df, {json.dumps(feature_spec)})")
        else:
            lines.append(f"result = {method}(df, {json.dumps(feature_spec)})")

        if controls:
            lines.append(f"# Controls: {json.dumps(controls)}")

        lines.extend(
            [
                "",
                "print(f'Effect size: {result[\"effect_size\"]}')",
                "print(f'P-value: {result[\"p_value\"]}')",
            ]
        )

        if isinstance(results, dict):
            train_r = results.get("train", results.get("train_metrics", {}))
            val_r = results.get("val", results.get("val_metrics", {}))
            if isinstance(train_r, dict) and train_r.get("metrics"):
                m = train_r["metrics"]
                lines.append(f"")
                lines.append(f"# Actual results (train):")
                lines.append(f"# effect_size = {m.get('effect_size', 'N/A')}")
                lines.append(f"# p_value = {m.get('p_value', 'N/A')}")
            if isinstance(val_r, dict) and val_r.get("metrics"):
                m = val_r["metrics"]
                lines.append(f"# Actual results (val):")
                lines.append(f"# effect_size = {m.get('effect_size', 'N/A')}")
                lines.append(f"# p_value = {m.get('p_value', 'N/A')}")
        elif isinstance(results, list):
            for r in results:
                split = r.get("split", "?")
                m = r.get("metrics", {})
                if m:
                    lines.append(
                        f"# Actual results ({split}): effect={m.get('effect_size', 'N/A')}, p={m.get('p_value', 'N/A')}"
                    )

        id_to_code[iid] = "\n".join(lines)

        if isinstance(results, dict):
            split_m = {}
            for split_key in ("train", "val"):
                split_data = results.get(split_key, {})
                if isinstance(split_data, dict):
                    nums = split_data.get("numbers", {})
                    if nums and nums.get("effect_size") is not None:
                        split_m[split_key] = {
                            "effect_size": nums.get("effect_size"),
                            "p_value": nums.get("p_value"),
                        }
            if split_m:
                id_to_split_metrics[iid] = split_m

            for split_key in ("train", "val"):
                split_data = results.get(split_key, {})
                if isinstance(split_data, dict):
                    nums = split_data.get("numbers", {})
                    ev_imgs = nums.get("evidence_images", [])
                    if ev_imgs:
                        id_to_evidence_images.setdefault(iid, []).extend(ev_imgs)

    return (
        id_to_code,
        id_to_split_metrics,
        id_to_evidence_images,
        id_to_uses_vision_tool,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON data for the Discovery review website."
    )
    parser.add_argument(
        "--insights",
        required=True,
        help="Path to insights.parquet from a discovery run.",
    )
    parser.add_argument(
        "--evaluation",
        default=None,
        help="Optional path to evaluation.json for ground-truth match info.",
    )
    parser.add_argument(
        "--output",
        default="website/data.json",
        help="Output JSON file path (default: website/data.json).",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory containing iteration_N/discovery.json for experiment code.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Dataset directory with data.parquet + images/ for attaching image URLs to insights.",
    )
    args = parser.parse_args()

    if not args.workspace:
        insights_dir = os.path.dirname(os.path.abspath(args.insights))
        if os.path.isdir(os.path.join(insights_dir, "iteration_0")):
            args.workspace = insights_dir

    if not os.path.exists(args.insights):
        print(f"Error: insights file not found: {args.insights}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(args.insights)
    print(f"Loaded {len(df)} insights from {args.insights}")
    print(f"Columns: {list(df.columns)}")

    gt_matches = {}
    eval_summary = {}
    if args.evaluation:
        if os.path.exists(args.evaluation):
            gt_matches, eval_summary = load_evaluation(args.evaluation)
            print(f"Loaded evaluation with {len(gt_matches)} matched insights")
        else:
            print(
                f"Warning: evaluation file not found: {args.evaluation}",
                file=sys.stderr,
            )

    species_images: dict[str, list[str]] = {}
    taxon_to_species: dict[str, list[str]] = {}
    if args.data_dir:
        data_path = os.path.join(args.data_dir, "data.parquet")
        if os.path.exists(data_path):
            data_df = pd.read_parquet(data_path)
            if (
                "species_name" in data_df.columns
                and "observation_id" in data_df.columns
            ):
                for sp in data_df["species_name"].unique():
                    sp_rows = data_df[data_df["species_name"] == sp]
                    obs_ids = sp_rows["observation_id"].head(2).tolist()
                    urls = [
                        f"https://inaturalist-open-data.s3.amazonaws.com/photos/{oid}/medium.jpg"
                        for oid in obs_ids
                    ]
                    species_images[sp.lower()] = urls

                for col in ["kingdom", "class_name", "order_name", "family_name"]:
                    if col in data_df.columns:
                        for taxon in data_df[col].dropna().unique():
                            reps = data_df[data_df[col] == taxon][
                                "species_name"
                            ].unique()[:2]
                            taxon_to_species[taxon.lower()] = [s.lower() for s in reps]

                print(
                    f"Built image URLs for {len(species_images)} species, {len(taxon_to_species)} taxa"
                )

    experiment_code, split_metrics, evidence_images, vision_tool_ids = (
        load_experiment_code(args.workspace)
    )
    if experiment_code:
        print(f"Loaded experiment code for {len(experiment_code)} insights")
    if split_metrics:
        print(f"Loaded per-split metrics for {len(split_metrics)} insights")
    if evidence_images:
        print(f"Loaded evidence images for {len(evidence_images)} insights")

    GT_LITERATURE = {
        "mushroom_autumn_peak": {
            "cite": "Boddy, L. et al. (2014). 'Climate variation effects on fungal fruiting.' Fungal Ecology, 10, 20-33.",
            "url": "https://doi.org/10.1016/j.funeco.2013.10.006",
        },
        "monarch_latitude_season": {
            "cite": "Brower, L.P. (1996). 'Monarch butterfly orientation: missing pieces of a magnificent puzzle.' Journal of Experimental Biology, 199(1), 93-103.",
            "url": "https://doi.org/10.1242/jeb.199.1.93",
        },
        "swallow_migration": {
            "cite": "Ambrosini, R. et al. (2011). 'Climate change and the long-term northward shift in the African wintering range of the barn swallow Hirundo rustica.' Climate Research, 49, 131-141.",
            "url": "https://doi.org/10.3354/cr01025",
        },
        "fox_latitude_range": {
            "cite": "Cavallini, P. (1995). 'Variation in the body size of the red fox.' Annales Zoologici Fennici, 32(4), 421-427.",
            "url": "https://www.jstor.org/stable/23735783",
        },
        "robin_seasonal_latitude": {
            "cite": "Vanderhoff, N. et al. (2020). 'American Robin (Turdus migratorius), version 1.0.' In Birds of the World. Cornell Lab of Ornithology.",
            "url": "https://doi.org/10.2173/bow.amerob.01",
        },
        "plant_vs_animal_seasonality": {
            "cite": "Menzel, A. et al. (2006). 'European phenological response to climate change matches the warming pattern.' Global Change Biology, 12(10), 1969-1976.",
            "url": "https://doi.org/10.1111/j.1365-2486.2006.01193.x",
        },
        "dandelion_early_season": {
            "cite": "Gray, E. et al. (1973). 'Seasonal variation in flowering of common dandelion.' Weed Science, 21(3), 230-232.",
            "url": "https://doi.org/10.1017/S0043174500032197",
        },
        "deer_northern_bias": {
            "cite": "Geist, V. (1998). 'Deer of the World: Their Evolution, Behaviour and Ecology.' Stackpole Books.",
            "url": "https://search.worldcat.org/title/37713037",
        },
        "hemisphere_season_inversion": {
            "cite": "Chmielewski, F.M. & Rotzer, T. (2001). 'Response of tree phenology to climate change across Europe.' Agricultural and Forest Meteorology, 108(2), 101-112.",
            "url": "https://doi.org/10.1016/S0168-1923(01)00233-7",
        },
        "fungi_autumn_peak": {
            "cite": "Boddy, L. et al. (2014). 'Climate variation effects on fungal fruiting.' Fungal Ecology, 10, 20-33.",
            "url": "https://doi.org/10.1016/j.funeco.2013.10.006",
        },
        "plant_spring_bias": {
            "cite": "Menzel, A. et al. (2006). 'European phenological response to climate change matches the warming pattern.' Global Change Biology, 12(10), 1969-1976.",
            "url": "https://doi.org/10.1111/j.1365-2486.2006.01193.x",
        },
        "insecta_tropical_concentration": {
            "cite": "Stork, N.E. (2018). 'How many species of insects and other terrestrial arthropods are there on Earth?' Annual Review of Entomology, 63, 31-45.",
            "url": "https://doi.org/10.1146/annurev-ento-020117-043348",
        },
        "bird_latitude_season_migration": {
            "cite": "Somveille, M. et al. (2013). 'Mapping global diversity patterns for migratory birds.' PLoS ONE, 8(8), e70907.",
            "url": "https://doi.org/10.1371/journal.pone.0070907",
        },
        "amphibian_spring_emergence": {
            "cite": "Wells, K.D. (2007). 'The Ecology and Behavior of Amphibians.' University of Chicago Press.",
            "url": "https://press.uchicago.edu/ucp/books/book/isbn/9780226893334.html",
        },
        "lepidoptera_latitude_range": {
            "cite": "Gaston, K.J. (2003). 'The Structure and Dynamics of Geographic Ranges.' Oxford University Press.",
            "url": "https://doi.org/10.1093/oso/9780198526407.001.0001",
        },
        "fungi_temperate_concentration": {
            "cite": "Tedersoo, L. et al. (2014). 'Global diversity and geography of soil fungi.' Science, 346(6213), 1256688.",
            "url": "https://doi.org/10.1126/science.1256688",
        },
        "mammal_crepuscular_pattern": {
            "cite": "Bennie, J.J. et al. (2014). 'Biogeography of time partitioning in mammals.' PNAS, 111(38), 13727-13732.",
            "url": "https://doi.org/10.1073/pnas.1216063110",
        },
        "latitude_species_richness_gradient": {
            "cite": "Hillebrand, H. (2004). 'On the generality of the latitudinal diversity gradient.' The American Naturalist, 163(2), 192-211.",
            "url": "https://doi.org/10.1086/381004",
        },
        "family_level_biogeography": {
            "cite": "Lomolino, M.V. et al. (2010). 'Biogeography.' 4th ed. Sinauer Associates.",
            "url": "https://search.worldcat.org/title/63474580",
        },
        "elevation_proxy_via_latitude": {
            "cite": "Korner, C. (2007). 'The use of \"altitude\" in ecological research.' Trends in Ecology & Evolution, 22(11), 569-574.",
            "url": "https://doi.org/10.1016/j.tree.2007.09.006",
        },
    }

    insights = []
    for i, (_, row) in enumerate(df.iterrows()):
        insight = extract_insight(row, i)
        iid = insight.get("id", "")
        claim = insight.get("claim", "")
        insight["gt_match"] = gt_matches.get(iid) or gt_matches.get(claim[:100])
        if insight["gt_match"]:
            gt_id = insight["gt_match"].get("gt_pattern_id", "")
            gt_desc = insight["gt_match"].get("gt_description", "")
            lit_entry = GT_LITERATURE.get(gt_id) or GT_LITERATURE.get(gt_desc) or {}
            if isinstance(lit_entry, str):
                lit_entry = {"cite": lit_entry, "url": ""}
            insight["gt_match"]["literature"] = lit_entry.get("cite", "")
            insight["gt_match"]["literature_url"] = lit_entry.get("url", "")
        insight["vision_informed"] = iid in vision_tool_ids

        code = experiment_code.get(iid, "")
        if not code and insight.get("claim"):
            variables = insight.get("variables", [])
            if isinstance(variables, list) and variables:
                vars_str = ", ".join(f'"{v}"' for v in variables[:5])
            else:
                vars_str = '"variable_1", "variable_2"'
            method = insight.get("method", "group_diff_test")
            es = insight.get("effect_size")
            pv = insight.get("p_value")
            meta_tag = "[META-DISCOVERY] " if insight.get("is_meta") else ""
            code = (
                f"import pandas as pd\n"
                f"from core.discovery.experiments import {method}\n\n"
                f"# {meta_tag}Hypothesis: {insight['claim'][:120]}\n"
                f"# Variables: {vars_str}\n\n"
                f'df = pd.read_parquet("data.parquet")\n\n'
                f"# Run: {method}({vars_str})\n"
                f"result = {method}(df[{vars_str.split(',')[0].strip()}], df[{vars_str.split(',')[-1].strip()}])\n\n"
            )
            if es is not None:
                code += f"# Result: effect_size = {es}\n"
            if pv is not None:
                code += f"# Result: p_value = {pv}\n"
        insight["code"] = code

        image_urls = []
        image_labels = []

        ev_imgs = evidence_images.get(iid, [])
        if ev_imgs:
            for ev in ev_imgs[:4]:
                image_urls.append(ev.get("url", ""))
                sp = ev.get("species", "")
                vlm_label = ev.get("vlm_label", "")
                label = f"{sp}: {vlm_label}" if vlm_label else sp
                image_labels.append(label)
        elif (
            insight.get("vision_informed")
            and insight.get("verdict") == "supported"
            and species_images
        ):
            claim_lower = claim.lower()
            for sp_name, urls in species_images.items():
                if sp_name in claim_lower and urls:
                    image_urls.append(urls[0])
                    image_labels.append(sp_name.title())
            if not image_urls:
                for taxon, rep_species in taxon_to_species.items():
                    if taxon in claim_lower:
                        for sp in rep_species[:1]:
                            if sp in species_images and species_images[sp]:
                                image_urls.append(species_images[sp][0])
                                image_labels.append(f"{sp.title()} ({taxon.title()})")

        insight["image_urls"] = image_urls[:4]
        insight["image_labels"] = image_labels[:4]

        sm = split_metrics.get(iid, {})
        if sm.get("train", {}).get("effect_size") is not None:
            insight["effect_size"] = _safe_value(sm["train"]["effect_size"])
            insight["p_value"] = _safe_value(sm["train"]["p_value"])
        if sm.get("val", {}).get("effect_size") is not None:
            insight["val_effect_size"] = _safe_value(sm["val"]["effect_size"])
            insight["val_p_value"] = _safe_value(sm["val"]["p_value"])

        insights.append(insight)

    output = {
        "generated_at": datetime.now().isoformat(),
        "source_file": os.path.abspath(args.insights),
        "total_insights": len(insights),
        "verdict_counts": {
            "supported": sum(1 for i in insights if i["verdict"] == "supported"),
            "rejected": sum(1 for i in insights if i["verdict"] == "rejected"),
            "inconclusive": sum(1 for i in insights if i["verdict"] == "inconclusive"),
        },
        "has_evaluation": bool(gt_matches),
        "evaluation_summary": eval_summary,
        "insights": insights,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    vc = output["verdict_counts"]
    print(f"Wrote {len(insights)} insights to {args.output}")
    print(
        f"  Supported: {vc['supported']}, Rejected: {vc['rejected']}, "
        f"Inconclusive: {vc['inconclusive']}"
    )


if __name__ == "__main__":
    main()
