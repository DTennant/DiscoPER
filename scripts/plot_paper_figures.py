#!/usr/bin/env python3
"""Generate all paper figures: multi-model, bias, controllability, behavior, SACHS/ASIA.

Usage:
    python scripts/plot_paper_figures.py
    python scripts/plot_paper_figures.py --dpi 300
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

MULTI_MODEL = {
    "models": ["Sonnet 4.5", "Opus 4.6", "DeepSeek V4", "GPT-5.4"],
    "recall": [88.9, 44.4, 44.4, 22.2],
    "precision": [72.7, 100, 57.1, 100],
    "f1": [0.800, 0.615, 0.500, 0.364],
}

BIAS_TRAPS = {
    "models": ["Sonnet 4.5", "Opus 4.6"],
    "traps": ["Simpson's\nParadox", "Confound\nBias", "Sampling\nBias"],
    "results": [[1, 1, 1], [0, 0, 1]],  # 1=avoided, 0=fell
}

CONTROLLABILITY = {
    "conditions": ["Default", "Prior\nKnowledge", "Guided\n(Obs. Bias)"],
    "recall": [77.8, 77.8, 66.7],
    "precision": [70.0, 77.8, 100],
    "topics": {
        "names": [
            "Observer\nBias",
            "Seasonal",
            "Habitat",
            "Cross-\nKingdom",
            "Geographic",
        ],
        "default": [43, 56, 36, 31, 37],
        "guided": [68, 44, 35, 15, 21],
    },
}

AGENT_BEHAVIOR = {
    "templates": [
        "X higher than Y",
        "X peaks in Y",
        "X correlates Y",
        "X interacts with Z",
        "X visual/habitat",
    ],
    "template_freq": [51, 21, 1, 3, 3],
    "phases": ["Early\n(iter 1-15)", "Mid\n(iter 16-35)", "Late\n(iter 36-50)"],
    "support_rate": [5, 8, 22],
    "redundancy": [60, 55, 40],
}

CLASSICAL = {
    "sachs": {
        "methods": ["Ours", "AIDE Search", "PC", "HillClimb"],
        "recall": [43.9, 17.5, 15.8, 0.0],
        "colors": ["#2563eb", "#94a3b8", "#64748b", "#cbd5e1"],
        "n_patterns": 57,
    },
    "asia": {
        "methods": ["Ours", "AIDE Search", "PC", "HillClimb"],
        "recall": [83.3, 50.0, 25.0, 8.3],
        "colors": ["#2563eb", "#94a3b8", "#64748b", "#cbd5e1"],
        "n_patterns": 12,
    },
}


def plot_multi_model(dpi):
    d = MULTI_MODEL
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(d["models"]))
    w = 0.25
    bars1 = ax.bar(
        x - w, d["recall"], w, label="Recall (%)", color="#2563eb", alpha=0.85
    )
    bars2 = ax.bar(
        x, d["precision"], w, label="Precision (%)", color="#10b981", alpha=0.85
    )
    bars3 = ax.bar(
        x + w,
        [f * 100 for f in d["f1"]],
        w,
        label="F1 (×100)",
        color="#f59e0b",
        alpha=0.85,
    )
    for bar, val in zip(bars1, d["recall"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.0f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#2563eb",
        )
    for bar, val in zip(bars3, d["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 100 + 1.5,
            f"{val:.2f}",
            ha="center",
            fontsize=8,
            fontweight="bold",
            color="#b45309",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(d["models"], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Multi-Model Comparison (iNat-800)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/multi_model_comparison.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_bias(dpi):
    d = BIAS_TRAPS
    fig, ax = plt.subplots(figsize=(5.5, 3))
    colors_map = {0: "#ef4444", 1: "#10b981"}
    for i in range(len(d["models"])):
        for j in range(len(d["traps"])):
            color = colors_map[d["results"][i][j]]
            label = "✓ Avoided" if d["results"][i][j] == 1 else "✗ Fell"
            ax.add_patch(
                plt.Rectangle(
                    (j, i),
                    0.9,
                    0.8,
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="white",
                    lw=2,
                )
            )
            ax.text(
                j + 0.45,
                i + 0.4,
                label,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="white",
            )
    ax.set_xticks([t + 0.45 for t in range(len(d["traps"]))])
    ax.set_xticklabels(d["traps"], fontsize=10)
    ax.set_yticks([i + 0.4 for i in range(len(d["models"]))])
    ax.set_yticklabels(d["models"], fontsize=11)
    ax.set_xlim(-0.05, len(d["traps"]))
    ax.set_ylim(-0.1, len(d["models"]))
    ax.set_title("Bias Trap Susceptibility", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/bias_susceptibility.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_controllability(dpi):
    d = CONTROLLABILITY
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    x = np.arange(len(d["conditions"]))
    w = 0.3
    ax.bar(x - w / 2, d["recall"], w, label="Recall (%)", color="#2563eb", alpha=0.85)
    ax.bar(
        x + w / 2, d["precision"], w, label="Precision (%)", color="#10b981", alpha=0.85
    )
    for i, (r, p) in enumerate(zip(d["recall"], d["precision"])):
        ax.text(
            i - w / 2,
            r + 2,
            f"{r:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#2563eb",
        )
        ax.text(
            i + w / 2,
            p + 2,
            f"{p:.0f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#10b981",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(d["conditions"], fontsize=10)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("(a) Recall & Precision", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    ax = axes[1]
    t = d["topics"]
    x = np.arange(len(t["names"]))
    w = 0.35
    ax.bar(x - w / 2, t["default"], w, label="Default", color="#94a3b8", alpha=0.8)
    ax.bar(
        x + w / 2,
        t["guided"],
        w,
        label="Guided (Obs. Bias)",
        color="#f59e0b",
        alpha=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(t["names"], fontsize=9)
    ax.set_ylabel("% of hypotheses", fontsize=11)
    ax.set_ylim(0, 85)
    ax.set_title("(b) Topic Distribution Shift", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/controllability.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_agent_behavior(dpi):
    d = AGENT_BEHAVIOR
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    colors_t = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]
    bars = ax.barh(
        range(len(d["templates"])), d["template_freq"], color=colors_t, height=0.6
    )
    for bar, val in zip(bars, d["template_freq"]):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1e40af",
        )
    ax.set_yticks(range(len(d["templates"])))
    ax.set_yticklabels(d["templates"], fontsize=10)
    ax.set_xlabel("% of hypotheses", fontsize=11)
    ax.set_title("(a) Hypothesis Template Distribution", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 65)
    ax.invert_yaxis()
    ax.grid(True, axis="x", alpha=0.2)

    ax = axes[1]
    x = np.arange(len(d["phases"]))
    w = 0.3
    bars1 = ax.bar(
        x - w / 2,
        d["support_rate"],
        w,
        label="Support rate (%)",
        color="#10b981",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + w / 2,
        d["redundancy"],
        w,
        label="Redundancy rate (%)",
        color="#ef4444",
        alpha=0.5,
    )
    for bar, val in zip(bars1, d["support_rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#059669",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(d["phases"], fontsize=10)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_ylim(0, 75)
    ax.set_title("(b) Discovery Quality by Phase", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)
    ax.annotate(
        "Reflect improves\nlate-phase quality",
        xy=(2, 22),
        xytext=(1.2, 50),
        arrowprops=dict(arrowstyle="->", color="#6366f1", lw=1.5),
        fontsize=9,
        color="#6366f1",
        fontweight="bold",
        ha="center",
    )
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/agent_behavior.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_sachs_asia(dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ["sachs", "asia"]):
        d = CLASSICAL[name]
        bars = ax.bar(
            range(len(d["methods"])),
            d["recall"],
            color=d["colors"],
            width=0.6,
            edgecolor="white",
        )
        for bar, val in zip(bars, d["recall"]):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{val:.1f}%",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color=bar.get_facecolor(),
                )
        ax.set_xticks(range(len(d["methods"])))
        ax.set_xticklabels(d["methods"], fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Recall (%)", fontsize=11)
        ax.set_ylim(0, 105)
        ax.set_title(
            f"{name.upper()} ({d['n_patterns']} patterns)",
            fontsize=12,
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/sachs_asia_recall.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    plot_multi_model(args.dpi)
    plot_bias(args.dpi)
    plot_controllability(args.dpi)
    plot_agent_behavior(args.dpi)
    plot_sachs_asia(args.dpi)


if __name__ == "__main__":
    main()
