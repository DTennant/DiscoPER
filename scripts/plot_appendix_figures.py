#!/usr/bin/env python3
"""Plot appendix figures: classical benchmarks, synthetic benchmark, framework ablation.

Usage:
    python scripts/plot_appendix_figures.py
    python scripts/plot_appendix_figures.py --dpi 300
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "docs/paper_figures"


# ============================================================
# DATA
# ============================================================

CLASSICAL_BENCHMARKS = {
    "SACHS": {
        "ours_f1": 0.176,
        "ours_std": 0.015,
        "base_f1": 0.147,
        "base_std": 0.019,
        "improvement": "+20%",
    },
    "ASIA": {
        "ours_f1": 0.437,
        "ours_std": 0.138,
        "base_f1": 0.342,
        "base_std": 0.080,
        "improvement": "+28%",
    },
    "ALARM": {
        "ours_f1": 0.083,
        "ours_std": 0.010,
        "base_f1": 0.084,
        "base_std": 0.031,
        "improvement": "-1%",
    },
}

SYNTHETIC_BENCHMARK = {
    "labels": ["Clean\n(50 iter)", "Clean\n(100 iter)", "Biased\n(100 iter)"],
    "recall": [16.7, 33.3, 16.7],
    "supported": [5, 10, 2],
    "colors": ["#3b82f6", "#2563eb", "#ef4444"],
}

FRAMEWORK_ABLATION = {
    "conditions": [
        "P=None\nReflect ON",
        "P=None\nReflect OFF",
        "P=Partial\nReflect ON",
        "P=Partial\nReflect OFF",
        "P=Full\nReflect ON",
        "P=Full\nReflect OFF",
    ],
    "recalls": [77.8, 77.8, 100, 55.6, 44.4, 11.1],
    "colors_on": "#2563eb",
    "colors_off": "#93c5fd",
}


def plot_classical(dpi):
    d = CLASSICAL_BENCHMARKS
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for i, (name, vals) in enumerate(d.items()):
        ax = axes[i]
        methods = ["Ours\n(Reflect)", "Baseline\n(no Reflect)"]
        f1s = [vals["ours_f1"], vals["base_f1"]]
        stds = [vals["ours_std"], vals["base_std"]]
        colors = ["#2563eb", "#94a3b8"]

        bars = ax.bar(
            methods,
            f1s,
            yerr=stds,
            color=colors,
            width=0.5,
            capsize=5,
            edgecolor="white",
            error_kw={"lw": 1.5},
        )
        for bar, val, sd in zip(bars, f1s, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sd + 0.01,
                f"{val:.3f}",
                ha="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(f"{name}\n({vals['improvement']})", fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score" if i == 0 else "", fontsize=11)
        ax.set_ylim(0, max(f1s) * 1.6)
        ax.grid(True, axis="y", alpha=0.15)

    plt.suptitle(
        "Classical Benchmark Results (3 seeds each)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/classical_benchmarks"
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}.png + .pdf")
    plt.close()


def plot_synthetic(dpi):
    d = SYNTHETIC_BENCHMARK
    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(d["labels"]))
    w = 0.35
    bars_r = ax.bar(
        x - w / 2, d["recall"], w, label="Recall (%)", color=d["colors"], alpha=0.85
    )
    bars_s = ax.bar(
        x + w / 2, d["supported"], w, label="Supported", color=d["colors"], alpha=0.4
    )

    for bar, val in zip(bars_r, d["recall"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars_s, d["supported"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            fontsize=10,
        )

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(d["labels"], fontsize=10)
    ax.set_title("Synthetic Visual Benchmark Results", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.15)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/synthetic_benchmarks"
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}.png + .pdf")
    plt.close()


def plot_framework_ablation(dpi):
    d = FRAMEWORK_ABLATION
    fig, ax = plt.subplots(figsize=(8, 4.5))

    colors = [
        d["colors_on"] if i % 2 == 0 else d["colors_off"]
        for i in range(len(d["conditions"]))
    ]
    bars = ax.bar(
        range(len(d["conditions"])),
        d["recalls"],
        color=colors,
        width=0.6,
        edgecolor="white",
    )

    for bar, val in zip(bars, d["recalls"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{val:.0f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Annotate largest Reflect gain
    ax.annotate(
        "",
        xy=(2, 100),
        xytext=(3, 55.6),
        arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=2),
    )
    ax.text(
        2.5, 80, "+44pp", ha="center", fontsize=10, fontweight="bold", color="#dc2626"
    )

    ax.set_xticks(range(len(d["conditions"])))
    ax.set_xticklabels(d["conditions"], fontsize=9)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.set_ylim(0, 118)
    ax.set_title(
        "Framework Ablation: P \u00d7 Reflect (iNatDisCo-800, 50 iter)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.15)

    from matplotlib.patches import Patch

    ax.legend(
        [Patch(facecolor=d["colors_on"]), Patch(facecolor=d["colors_off"])],
        ["Reflect ON", "Reflect OFF"],
        fontsize=10,
        loc="upper right",
    )

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/framework_ablation"
    fig.savefig(f"{path}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{path}.pdf", dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}.png + .pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    plot_classical(args.dpi)
    plot_synthetic(args.dpi)
    plot_framework_ablation(args.dpi)


if __name__ == "__main__":
    main()
