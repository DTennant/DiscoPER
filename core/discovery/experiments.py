# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Statistical test primitives for hypothesis testing."""

from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score


def corr_test(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "spearman",
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compute correlation between two variables.

    Args:
        x: First variable
        y: Second variable
        method: Correlation method ("spearman", "pearson")
        bootstrap_ci: Whether to compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples
        random_state: Random seed

    Returns:
        Dict with effect_size, ci_lower, ci_upper, p_value, provenance
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        return {
            "status": "error",
            "error": "Insufficient data points after removing NaNs",
            "provenance": {"method": method, "n_samples": len(x_clean)},
        }

    # Compute correlation
    if method == "spearman":
        corr, p_value = stats.spearmanr(x_clean, y_clean)
    elif method == "pearson":
        corr, p_value = stats.pearsonr(x_clean, y_clean)
    else:
        return {"status": "error", "error": f"Unknown method: {method}"}

    result = {
        "status": "success",
        "effect_size": float(corr),
        "p_value": float(p_value),
        "n_samples": len(x_clean),
        "provenance": {"method": method, "random_state": random_state},
    }

    # Bootstrap CI
    if bootstrap_ci:
        np.random.seed(random_state)
        boot_corrs = []
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(x_clean), len(x_clean), replace=True)
            x_boot, y_boot = x_clean[indices], y_clean[indices]
            if method == "spearman":
                boot_corr, _ = stats.spearmanr(x_boot, y_boot)
            else:
                boot_corr, _ = stats.pearsonr(x_boot, y_boot)
            boot_corrs.append(boot_corr)

        result["ci_lower"] = float(np.percentile(boot_corrs, 2.5))
        result["ci_upper"] = float(np.percentile(boot_corrs, 97.5))

    return result


def group_diff_test(
    metric: np.ndarray,
    group: np.ndarray,
    method: str = "perm_test",
    effect: str = "cliffs_delta",
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Test for difference between groups.

    Args:
        metric: Continuous metric values
        group: Group labels (binary)
        method: Test method ("perm_test", "t_test", "mann_whitney")
        effect: Effect size measure ("cliffs_delta", "cohens_d", "mean_diff")
        n_permutations: Number of permutations for permutation test
        random_state: Random seed

    Returns:
        Dict with effect_size, ci_lower, ci_upper, p_value, provenance
    """
    # Remove NaN values
    mask = ~(np.isnan(metric) | pd.isna(group))
    metric_clean, group_clean = metric[mask], group[mask]

    unique_groups = np.unique(group_clean)
    if len(unique_groups) != 2:
        return {
            "status": "error",
            "error": f"Expected 2 groups, got {len(unique_groups)}",
        }

    group_0 = metric_clean[group_clean == unique_groups[0]]
    group_1 = metric_clean[group_clean == unique_groups[1]]

    if len(group_0) < 2 or len(group_1) < 2:
        return {"status": "error", "error": "Insufficient data in one or both groups"}

    # Compute effect size
    if effect == "cliffs_delta":
        # Cliff's Delta: proportion of pairs where group_1 > group_0 minus proportion where group_0 > group_1
        n_greater = np.sum(group_1[:, None] > group_0[None, :])
        n_less = np.sum(group_1[:, None] < group_0[None, :])
        effect_size = (n_greater - n_less) / (len(group_0) * len(group_1))
    elif effect == "cohens_d":
        pooled_std = np.sqrt(
            ((len(group_0) - 1) * group_0.var() + (len(group_1) - 1) * group_1.var())
            / (len(group_0) + len(group_1) - 2)
        )
        effect_size = (
            (group_1.mean() - group_0.mean()) / pooled_std if pooled_std > 0 else 0.0
        )
    elif effect == "mean_diff":
        effect_size = group_1.mean() - group_0.mean()
    else:
        return {"status": "error", "error": f"Unknown effect measure: {effect}"}

    # Compute p-value
    if method == "perm_test":
        np.random.seed(random_state)
        observed_diff = group_1.mean() - group_0.mean()
        perm_diffs = []
        combined = np.concatenate([group_0, group_1])
        for _ in range(n_permutations):
            perm_combined = np.random.permutation(combined)
            perm_0 = perm_combined[: len(group_0)]
            perm_1 = perm_combined[len(group_0) :]
            perm_diffs.append(perm_1.mean() - perm_0.mean())
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    elif method == "t_test":
        _, p_value = stats.ttest_ind(group_1, group_0)
    elif method == "mann_whitney":
        _, p_value = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")
    else:
        return {"status": "error", "error": f"Unknown method: {method}"}

    return {
        "status": "success",
        "effect_size": float(effect_size),
        "p_value": float(p_value),
        "n_group_0": len(group_0),
        "n_group_1": len(group_1),
        "group_0_mean": float(group_0.mean()),
        "group_1_mean": float(group_1.mean()),
        "provenance": {
            "method": method,
            "effect": effect,
            "random_state": random_state,
        },
    }


def predictive_test(
    X: np.ndarray,
    y: np.ndarray,
    model: str = "logreg",
    metric: str = "auc",
    baseline: str = "majority",
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Test predictive power of features.

    Args:
        X: Feature matrix, shape (n_samples, n_features)
        y: Target labels
        model: Model type ("logreg")
        metric: Evaluation metric ("auc", "accuracy")
        baseline: Baseline method ("majority", "random")
        test_size: Proportion of data for test set
        random_state: Random seed

    Returns:
        Dict with score, baseline_score, gain, p_value, provenance
    """
    from sklearn.model_selection import train_test_split

    # Remove NaN rows
    mask = ~(np.any(np.isnan(X), axis=1) | pd.isna(y))
    X_clean, y_clean = X[mask], y[mask]

    if len(X_clean) < 10:
        return {"status": "error", "error": "Insufficient data"}

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_clean,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clean,
    )

    # Train model
    if model == "logreg":
        clf = LogisticRegression(random_state=random_state, max_iter=1000)
        clf.fit(X_train, y_train)
        if metric == "auc":
            if len(np.unique(y_test)) < 2:
                return {"status": "error", "error": "Only one class in test set"}
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
            score = roc_auc_score(y_test, y_pred_proba)
        elif metric == "accuracy":
            score = clf.score(X_test, y_test)
        else:
            return {"status": "error", "error": f"Unknown metric: {metric}"}
    else:
        return {"status": "error", "error": f"Unknown model: {model}"}

    # Compute baseline
    if baseline == "majority":
        if metric == "auc":
            baseline_score = 0.5  # Random classifier AUC
        elif metric == "accuracy":
            baseline_score = np.bincount(y_train).max() / len(y_train)
    elif baseline == "random":
        baseline_score = 0.5
    else:
        baseline_score = 0.5

    gain = score - baseline_score

    return {
        "status": "success",
        "score": float(score),
        "baseline_score": float(baseline_score),
        "gain": float(gain),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "provenance": {
            "model": model,
            "metric": metric,
            "baseline": baseline,
            "random_state": random_state,
        },
    }


def cluster_and_enrich(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int,
    enrichment: str = "chi2",
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Cluster embeddings and test for label enrichment.

    Args:
        embeddings: Embedding matrix, shape (n_samples, embedding_dim)
        labels: Sample labels
        k: Number of clusters
        enrichment: Enrichment test ("chi2")
        random_state: Random seed

    Returns:
        Dict with cluster_labels, enrichment_scores, p_values, provenance
    """
    # Remove NaN rows
    mask = ~(np.any(np.isnan(embeddings), axis=1) | pd.isna(labels))
    embeddings_clean, labels_clean = embeddings[mask], labels[mask]

    if len(embeddings_clean) < k:
        return {
            "status": "error",
            "error": f"Insufficient data ({len(embeddings_clean)} samples for {k} clusters)",
        }

    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_clean)

    # Test enrichment for each label
    unique_labels = np.unique(labels_clean)
    enrichment_results = {}

    for label in unique_labels:
        # Contingency table: cluster x (has_label, not_has_label)
        has_label = labels_clean == label
        contingency = np.zeros((k, 2))
        for cluster_id in range(k):
            in_cluster = cluster_labels == cluster_id
            contingency[cluster_id, 0] = np.sum(in_cluster & has_label)
            contingency[cluster_id, 1] = np.sum(in_cluster & ~has_label)

        # Chi-square test
        if enrichment == "chi2":
            chi2, p_value, _, _ = stats.chi2_contingency(contingency)
            enrichment_results[str(label)] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
            }

    return {
        "status": "success",
        "n_clusters": k,
        "cluster_sizes": [int(np.sum(cluster_labels == i)) for i in range(k)],
        "enrichment_results": enrichment_results,
        "provenance": {"k": k, "enrichment": enrichment, "random_state": random_state},
    }


def stratified_retest(
    primary_test: Callable, stratify_by: str, data: pd.DataFrame, **kwargs
) -> dict[str, Any]:
    """
    Re-run a primary test stratified by a variable.

    Args:
        primary_test: Test function to run (corr_test, group_diff_test, etc.)
        stratify_by: Column name to stratify by
        data: DataFrame with all variables
        **kwargs: Additional arguments for primary_test

    Returns:
        Dict with results per stratum and aggregate stability metrics
    """
    if stratify_by not in data.columns:
        return {
            "status": "error",
            "error": f"Stratify column '{stratify_by}' not found",
        }

    unique_strata = data[stratify_by].unique()
    results_per_stratum = {}

    for stratum in unique_strata:
        stratum_data = data[data[stratify_by] == stratum]
        if len(stratum_data) < 3:
            continue

        # Run primary test on this stratum
        result = primary_test(data=stratum_data, **kwargs)
        results_per_stratum[str(stratum)] = result

    # Compute stability: standard deviation of effect sizes across strata
    effect_sizes = [
        r.get("effect_size", 0.0)
        for r in results_per_stratum.values()
        if r.get("status") == "success"
    ]
    stability = 1.0 - (np.std(effect_sizes) if len(effect_sizes) > 1 else 0.0)

    return {
        "status": "success",
        "n_strata": len(results_per_stratum),
        "results_per_stratum": results_per_stratum,
        "stability": float(stability),
        "provenance": {
            "stratify_by": stratify_by,
            "n_strata_tested": len(unique_strata),
        },
    }


def visual_attribute_test(
    images_b64: list[str],
    image_labels: list[dict],
    attribute_query: str,
    group_col: str,
    model_name: str = "claude-sonnet-4-5",
    model_url: str | None = None,
    api_key: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Use a VLM to classify a visual attribute from images, then test group differences.

    The VLM sees each image and answers a structured query about a visual attribute
    (e.g., "Is this organism in an open or forested habitat?"). The extracted labels
    are then tested against a metadata grouping variable using chi-squared.

    Args:
        images_b64: Base64-encoded PNG images
        image_labels: Metadata dicts per image (species_name, kingdom, etc.)
        attribute_query: Question for VLM (e.g., "habitat_type: open_field / forest / wetland / urban")
        group_col: Metadata column to group by (from image_labels)
        model_name: VLM model name
        model_url: API base URL
        api_key: API key
        random_state: Random seed

    Returns:
        Dict with effect_size (Cramer's V), p_value, contingency table
    """
    import os

    if not images_b64:
        return {"status": "error", "error": "No images provided"}

    api_key = (
        api_key
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        return {"status": "error", "error": "No API key for VLM"}

    categories = (
        [c.strip() for c in attribute_query.split(":")[1].split("/")]
        if ":" in attribute_query
        else ["yes", "no"]
    )
    attr_name = (
        attribute_query.split(":")[0].strip()
        if ":" in attribute_query
        else attribute_query
    )

    try:
        import litellm
    except ImportError:
        return {
            "status": "error",
            "error": "litellm required for visual_attribute_test",
        }

    import re, json as _json

    vlm_labels = []
    max_images = min(len(images_b64), 50)
    for i in range(max_images):
        img = images_b64[i]
        meta = image_labels[i] if i < len(image_labels) else {}
        species = meta.get("species_name", "unknown")

        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}},
            {
                "type": "text",
                "text": (
                    f"Look at this photo of {species}. "
                    f"Classify the {attr_name} visible in this image. "
                    f"You MUST pick exactly one: {' / '.join(categories)}. "
                    f"Reply with ONLY the category name, nothing else."
                ),
            },
        ]

        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.3,
                "max_tokens": 20,
            }
            if model_url:
                kwargs["api_base"] = model_url
            if api_key:
                kwargs["api_key"] = api_key

            resp = litellm.completion(**kwargs)
            text = resp.choices[0].message.content.strip().lower()
            matched_cat = None
            for cat in categories:
                if cat.lower() in text:
                    matched_cat = cat.lower()
                    break
            vlm_labels.append(matched_cat or text.split()[0].strip(".,"))
        except Exception:
            vlm_labels.append("unknown")

    n = min(len(vlm_labels), len(image_labels))
    vlm_labels = vlm_labels[:n]
    groups = [str(il.get(group_col, "unknown")).lower() for il in image_labels[:n]]

    vlm_series = pd.Series(vlm_labels, name=attr_name)
    group_series = pd.Series(groups, name=group_col)

    contingency = pd.crosstab(group_series, vlm_series)
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {
            "status": "error",
            "error": f"Not enough variation (shape={contingency.shape})",
        }

    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
    n_obs = contingency.values.sum()
    k = min(contingency.shape) - 1
    cramers_v = float(np.sqrt(chi2 / (n_obs * k))) if k > 0 and n_obs > 0 else 0.0

    evidence_images = []
    for i, il in enumerate(image_labels[:n]):
        obs_id = il.get("observation_id", "")
        if obs_id:
            evidence_images.append(
                {
                    "observation_id": obs_id,
                    "species": il.get("species_name", ""),
                    "vlm_label": vlm_labels[i] if i < len(vlm_labels) else "",
                    "group": groups[i] if i < len(groups) else "",
                    "url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{obs_id}/medium.jpg",
                }
            )

    return {
        "status": "success",
        "effect_size": cramers_v,
        "p_value": float(p_value),
        "chi2": float(chi2),
        "dof": int(dof),
        "n_images": n,
        "contingency": contingency.to_dict(),
        "vlm_labels_distribution": dict(vlm_series.value_counts()),
        "evidence_images": evidence_images[:8],
        "provenance": {
            "attribute_query": attribute_query,
            "group_col": group_col,
            "model": model_name,
        },
    }


def visual_group_comparison(
    group_a_images: list[str],
    group_b_images: list[str],
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
    comparison_query: str = "visual differences",
    model_name: str = "claude-sonnet-4-5",
    model_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Show images from two groups to a VLM and ask it to identify visual differences.

    Returns a structured comparison with confidence scores.

    Args:
        group_a_images: Base64 images for group A
        group_b_images: Base64 images for group B
        group_a_name: Label for group A
        group_b_name: Label for group B
        comparison_query: What to compare
        model_name: VLM model
        model_url: API base
        api_key: API key
    """
    import os

    api_key = (
        api_key
        or os.environ.get("LITELLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if not api_key:
        return {"status": "error", "error": "No API key"}

    try:
        import litellm, json as _json
    except ImportError:
        return {"status": "error", "error": "litellm required"}

    content = []
    content.append({"type": "text", "text": f"=== {group_a_name} ==="})
    for img in group_a_images[:5]:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        )
    content.append({"type": "text", "text": f"=== {group_b_name} ==="})
    for img in group_b_images[:5]:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
        )

    content.append(
        {
            "type": "text",
            "text": (
                f"Compare the two groups above regarding: {comparison_query}. "
                f'Reply with JSON: {{"differences": [{{"attribute": str, "group_a_value": str, '
                f'"group_b_value": str, "confidence": float (0-1)}}], '
                f'"summary": str}}'
            ),
        }
    )

    try:
        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 500,
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
            result = _json.loads(match.group())
            diffs = result.get("differences", [])
            avg_confidence = (
                np.mean([d.get("confidence", 0) for d in diffs]) if diffs else 0
            )
            return {
                "status": "success",
                "effect_size": float(avg_confidence),
                "p_value": max(0.001, 1.0 - avg_confidence),
                "n_differences": len(diffs),
                "differences": diffs,
                "summary": result.get("summary", ""),
                "n_images_a": len(group_a_images[:5]),
                "n_images_b": len(group_b_images[:5]),
                "provenance": {
                    "comparison_query": comparison_query,
                    "model": model_name,
                },
            }
    except Exception as e:
        return {"status": "error", "error": str(e)}

    return {"status": "error", "error": "Failed to parse VLM response"}
