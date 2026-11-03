#!/usr/bin/env python3
"""Plot agent behavioral analysis: hypothesis templates + Reflect guidance examples.

Usage:
    python scripts/plot_agent_behavior.py
    python scripts/plot_agent_behavior.py --output my_figure.png --dpi 300
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA — edit these to update the figure
# ============================================================

TEMPLATES = {
    "labels": [
        "X higher than Y",
        "X peaks in Y",
        "Other",
        "X interacts with Z",
        "X visual/habitat",
        "X correlates Y",
    ],
    "base_pct": [92, 6, 1, 0, 1, 0],
    "meta_pct": [69, 14, 10, 2, 2, 2],
    "color_base": "#93c5fd",
    "color_meta": "#2563eb",
}

REFLECT_EXAMPLES = [
    {
        "iter": "Iter 5",
        "type": "gap_detection",
        "color": "#f59e0b",
        "text": (
            '"Positional accuracy has 0% success rate\n'
            "across 3 hypotheses \u2192 stop testing it,\n"
            'try kingdom-level geographic comparisons"'
        ),
    },
    {
        "iter": "Iter 34",
        "type": "compound_hypothesis",
        "color": "#8b5cf6",
        "text": (
            '"Fungi+Plantae occupy distinct spatial\n'
            "niches \u2192 combine latitude \u00d7 longitude\n"
            'for cross-kingdom comparison" \u2192 Supported \u2713'
        ),
    },
    {
        "iter": "Iter 49",
        "type": "confound_detection",
        "color": "#ef4444",
        "text": (
            '"Hemisphere moderates all seasonal\n'
            "claims \u2192 stratify by hemisphere before\n"
            'testing any seasonal pattern"'
        ),
    },
]

LEGEND_ITEMS = [
    ("gap_detection", "#f59e0b"),
    ("compound", "#8b5cf6"),
    ("confound", "#ef4444"),
]


def plot_templates(ax):
    d = TEMPLATES
    y = np.arange(len(d["labels"]))
    h = 0.35
    bars_base = ax.barh(
        y + h / 2,
        d["base_pct"],
        h,
        label="Base (no Reflect)",
        color=d["color_base"],
        alpha=0.8,
    )
    bars_meta = ax.barh(
        y - h / 2,
        d["meta_pct"],
        h,
        label="Meta-guided (Reflect)",
        color=d["color_meta"],
        alpha=0.85,
    )
    for bar, val in zip(bars_base, d["base_pct"]):
        if val > 3:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val}%",
                va="center",
                fontsize=10,
                color="#64748b",
            )
    for bar, val in zip(bars_meta, d["meta_pct"]):
        if val > 0:
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{val}%",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#1e40af",
            )
    ax.set_yticks(y)
    ax.set_yticklabels(d["labels"], fontsize=11)
    ax.set_xlabel("% of hypotheses", fontsize=12)
    ax.set_title("Hypothesis Templates", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.invert_yaxis()
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.15)


def plot_reflect_examples(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 4.5)
    ax.axis("off")
    ax.set_title("Reflect Guidance Examples", fontsize=14, fontweight="bold")

    for i, ex in enumerate(REFLECT_EXAMPLES):
        y_pos = 3.6 - i * 1.4
        ax.add_patch(
            plt.Rectangle(
                (0, y_pos - 0.4),
                1.5,
                0.8,
                facecolor=ex["color"],
                alpha=0.15,
                edgecolor=ex["color"],
                lw=2,
                zorder=2,
            )
        )
        ax.text(
            0.75,
            y_pos,
            ex["iter"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=ex["color"],
        )
        ax.text(
            1.8,
            y_pos,
            ex["text"],
            va="center",
            fontsize=10.5,
            color="#1e293b",
            family="monospace",
        )

    for i, (label, color) in enumerate(LEGEND_ITEMS):
        ax.add_patch(
            plt.Rectangle(
                (i * 3.3, -0.15),
                0.35,
                0.35,
                facecolor=color,
                alpha=0.5,
            )
        )
        ax.text(i * 3.3 + 0.5, 0.02, label, fontsize=10, va="center", color="#475569")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/paper_figures/agent_behavior.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [1, 1.4]},
    )
    plot_templates(axes[0])
    plot_reflect_examples(axes[1])
    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
