#!/usr/bin/env python3
"""Plot counterfactual analysis: prior vs data-based hypotheses over iterations.

Usage:
    python scripts/plot_counterfactual_analysis.py
    python scripts/plot_counterfactual_analysis.py --output my_figure.png --dpi 300
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA — edit these to update the figure
# ============================================================

# Per-iteration classification of hypotheses
# 'p' = prior-based, 'd' = data-based
# Verdict: 's'=supported, 'r'=rejected, 'i'=inconclusive
PRIOR_ITERS = {
    0: "r",
    6: "i",
    7: "s",
    9: "i",
    12: "s",
    14: "i",
    19: "i",
    21: "i",
    23: "i",
    26: "i",
    28: "i",
    29: "i",
    31: "i",
    34: "i",
    35: "i",
    36: "i",
    41: "r",
    42: "i",
    43: "r",
    48: "i",
    50: "i",
}
DATA_ITERS = {1: "i", 4: "r"}

# Funnel counts
FUNNEL = {
    "labels": [
        "Hypotheses\nproposed",
        "Touch CF\ntopic",
        "Prior-based\n(common sense)",
        "Rejected by\nheld-out validation",
        "Supported\n(data-driven)",
    ],
    "values": [72, 23, 21, 3, 5],
    "colors": ["#94a3b8", "#f59e0b", "#ef4444", "#ef4444", "#10b981"],
    "alphas": [0.5, 0.7, 0.8, 0.5, 1.0],
}

# ============================================================
# COLORS
# ============================================================
COLOR_PRIOR = "#ef4444"
COLOR_DATA = "#2563eb"
COLOR_ANNOTATION = "#6366f1"


def plot_timeline(ax):
    """Panel (a): cumulative prior vs data-based hypotheses."""
    iters = list(range(1, 51))
    cum_prior = []
    cum_data = []
    cum_rejected = []
    p, d, r = 0, 0, 0
    for i in iters:
        if i in PRIOR_ITERS:
            p += 1
            if PRIOR_ITERS[i] == "r":
                r += 1
        if i in DATA_ITERS:
            d += 1
        cum_prior.append(p)
        cum_data.append(d)
        cum_rejected.append(r)

    ax.plot(
        iters,
        cum_prior,
        "-",
        color=COLOR_PRIOR,
        linewidth=2.5,
        label="Prior-based proposed",
    )
    ax.plot(
        iters,
        cum_rejected,
        "--",
        color=COLOR_PRIOR,
        linewidth=1.5,
        alpha=0.7,
        label="Prior-based rejected",
    )
    ax.plot(
        iters,
        cum_data,
        "-",
        color=COLOR_DATA,
        linewidth=2.5,
        label="Data-based proposed",
    )
    ax.fill_between(iters, cum_prior, cum_rejected, alpha=0.1, color=COLOR_PRIOR)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Cumulative count", fontsize=11)
    ax.set_title("(a) Prior vs Data-Based Hypotheses", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2)

    ax.annotate(
        "LLM keeps proposing\nprior-based hypotheses\nthroughout the run",
        xy=(35, 17),
        xytext=(25, 10),
        arrowprops=dict(arrowstyle="->", color=COLOR_PRIOR, lw=1.5),
        fontsize=9,
        color=COLOR_PRIOR,
        ha="center",
    )
    ax.annotate(
        "Data-based: only 2\n(both in first 5 iters)",
        xy=(5, 2),
        xytext=(15, 5),
        arrowprops=dict(arrowstyle="->", color=COLOR_DATA, lw=1.5),
        fontsize=9,
        color=COLOR_DATA,
        ha="center",
    )


def plot_funnel(ax):
    """Panel (b): outcome funnel."""
    f = FUNNEL
    bars = ax.barh(range(len(f["labels"])), f["values"], color=f["colors"], height=0.6)
    for i, (bar, val, a) in enumerate(zip(bars, f["values"], f["alphas"])):
        bar.set_alpha(a)
        ax.text(
            val + 1,
            i,
            str(val),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=f["colors"][i] if i != 0 else "#475569",
        )

    ax.set_yticks(range(len(f["labels"])))
    ax.set_yticklabels(f["labels"], fontsize=10)
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("(b) Counterfactual Outcome Funnel", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 85)
    ax.grid(True, axis="x", alpha=0.2)

    ax.annotate(
        "Held-out validation\ncatches prior contamination",
        xy=(3, 3),
        xytext=(35, 3.5),
        arrowprops=dict(arrowstyle="->", color=COLOR_ANNOTATION, lw=1.5),
        fontsize=9,
        color=COLOR_ANNOTATION,
        ha="center",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor=COLOR_ANNOTATION,
            alpha=0.8,
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="docs/paper_figures/counterfactual_analysis.png"
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_timeline(axes[0])
    plot_funnel(axes[1])
    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
