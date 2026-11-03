#!/usr/bin/env python3
"""Plot data scaling and iteration scaling curves for the paper.

Usage:
    python scripts/plot_scaling_curves.py
    python scripts/plot_scaling_curves.py --output my_figure.png --dpi 300
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# DATA — edit these to update the figure
# ============================================================

# Data scaling results (100 iterations each, iNat-50K subsamples)
DATA_SCALING = {
    "n": [800, 2000, 5000, 10000, 25000],
    "recall": [16.7, 16.7, 25.0, 50.0, 75.0],
    "precision": [100, 50.0, 50.0, 35.3, 90.0],
    "supported": [5, 4, 12, 17, 11],
    "f1": [0.286, 0.250, 0.333, 0.414, 0.818],
}

# Iteration scaling results (iNat-800)
ITER_SCALING = {
    "iterations": [10, 25, 50, 75, 100],
    "recall": [33.3, 33.3, 77.8, 22.2, 88.9],
    "precision": [100, 100, 70.0, 50.0, 72.7],
    "supported": [3, 3, 13, 9, 12],
    "f1": [0.500, 0.500, 0.737, 0.308, 0.800],
}

# ============================================================
# COLORS
# ============================================================
COLOR_RECALL = "#2563eb"
COLOR_PRECISION = "#10b981"
COLOR_SUPPORTED = "#f59e0b"
COLOR_ANNOTATION = "#6366f1"


def plot_data_scaling(ax):
    d = DATA_SCALING
    labels = [str(n) if n < 1000 else f"{n // 1000}K" for n in d["n"]]

    ax2 = ax.twinx()

    ax2.bar(
        labels,
        d["supported"],
        color=COLOR_SUPPORTED,
        alpha=0.3,
        width=0.6,
        label="Supported insights",
    )
    ax.plot(
        labels,
        d["recall"],
        "o-",
        color=COLOR_RECALL,
        linewidth=2.5,
        markersize=8,
        label="Recall (%)",
        zorder=5,
    )

    for i, (r, s) in enumerate(zip(d["recall"], d["supported"])):
        ax.annotate(
            f"{r:.0f}%",
            (i, r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=COLOR_RECALL,
            fontweight="bold",
        )

    ax.set_xlabel("Dataset size (n)", fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=11, color=COLOR_RECALL)
    ax2.set_ylabel("Supported insights", fontsize=11, color=COLOR_SUPPORTED)
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 25)
    ax.tick_params(axis="y", labelcolor=COLOR_RECALL)
    ax2.tick_params(axis="y", labelcolor=COLOR_SUPPORTED)
    ax.set_title(
        "(a) Data Scaling (100 iterations each)", fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.2)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)


def plot_iter_scaling(ax):
    d = ITER_SCALING
    x = np.arange(len(d["iterations"]))

    ax.plot(
        x,
        d["recall"],
        "o-",
        color=COLOR_RECALL,
        linewidth=2.5,
        markersize=8,
        label="Recall (%)",
        zorder=5,
    )
    ax.plot(
        x,
        d["precision"],
        "s--",
        color=COLOR_PRECISION,
        linewidth=2,
        markersize=7,
        label="Precision (%)",
        zorder=4,
    )

    for i, r in enumerate(d["recall"]):
        ax.annotate(
            f"{r:.0f}%",
            (i, r),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            color=COLOR_RECALL,
            fontweight="bold",
        )

    ax.fill_between(x, d["recall"], d["precision"], alpha=0.08, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in d["iterations"]])
    ax.set_xlabel("Number of iterations", fontsize=11)
    ax.set_ylabel("Metric (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("(b) Iteration Scaling (iNat-800)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    # Annotate best F1 point
    best_idx = np.argmax(d["f1"])
    ax.annotate(
        f"Best F1\n({d['f1'][best_idx]:.3f})",
        xy=(best_idx, d["recall"][best_idx]),
        xytext=(best_idx - 0.8, 95),
        arrowprops=dict(arrowstyle="->", color=COLOR_ANNOTATION, lw=1.5),
        fontsize=9,
        color=COLOR_ANNOTATION,
        fontweight="bold",
        ha="center",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="docs/paper_figures/scaling_curves.png")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plot_data_scaling(axes[0])
    plot_iter_scaling(axes[1])
    plt.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close()


if __name__ == "__main__":
    main()
