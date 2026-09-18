"""
Phase 7.2 Publication Figure Generator.

Script: scripts/phase7_generate_figures.py
Author: SeismoFNO Research Team
Context: Phase 7 Cross-Structure Generalization Figures
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))


def generate_phase7_figures(
    summary_path: str = "results/phase7/statistics/cross_structure_summary.json",
    evals_path: str = "results/phase7/test/all_test_evaluations.json",
    output_dir: str = "reports/figures/phase7",
):
    print("=" * 80)
    print("📊 GENERATING PHASE 7 PUBLICATION FIGURES")
    print("=" * 80)

    if not os.path.exists(summary_path) or not os.path.exists(evals_path):
        print("Summary data not yet ready.")
        return

    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, "r") as f:
        summary = json.load(f)
    with open(evals_path, "r") as f:
        evals = json.load(f)

    # Palette
    c_blue = "#1d3557"
    c_teal = "#457b9d"
    c_coral = "#e76f51"
    c_sand = "#e9c46a"
    c_green = "#2a9d8f"
    c_dark = "#264653"

    # Figure 1: Cross-Structure Generalization Benchmark Across Model Hierarchy
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    models = ["B1", "B2", "B3", "B4", "B5", "PROPOSED"]
    x = np.arange(len(models))
    width = 0.20

    levels = [
        ("Level 1 (ID)", "level1_acc", c_blue),
        ("Level 2A (Interp)", "level2a_acc", c_green),
        ("Level 2B (Extrap)", "level2b_acc", c_sand),
        ("Level 3 (Topology)", "level3_acc", c_coral),
    ]

    for ax_idx, mod in enumerate(["S1", "S4"]):
        ax = axes[ax_idx]
        for i, (lvl_name, metric_key, color) in enumerate(levels):
            means = [summary[f"P2_{mod}_{m}"][metric_key]["mean"] for m in models]
            stds = [summary[f"P2_{mod}_{m}"][metric_key]["std"] for m in models]
            ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=lvl_name, color=color, alpha=0.9, edgecolor="black", lw=0.5)

        ax.axhline(50.0, color="gray", linestyle="--", alpha=0.7, label="Chance Baseline (50%)")
        ax.set_title(f"Sensor Configuration {mod} (Protocol P2 Diversity)", fontsize=12, fontweight="bold")
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(models, fontsize=10, fontweight="bold")
        ax.set_ylim(40, 105)
        ax.set_ylabel("Bilateral Attribution Accuracy (%)" if ax_idx == 0 else "", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.5, axis="y")
        if ax_idx == 1:
            ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Figure 1: Cross-Structure Inverse Generalization Across Model Hierarchy (B1..B5 vs PROPOSED)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    f1_path = os.path.join(output_dir, "fig1_cross_structure_benchmark.png")
    fig.savefig(f1_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {f1_path}")

    # Figure 2: P1 vs P2 Training Diversity Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    mods_eval = ["S1", "S4"]
    test_levels = [("L1 (ID)", "level1_acc"), ("L2A (Interp)", "level2a_acc"), ("L2B (Extrap)", "level2b_acc"), ("L3 (Topology)", "level3_acc")]
    ind = np.arange(len(test_levels))
    w = 0.20

    # S1 P1 vs P2
    s1_p1 = [summary[f"P1_S1_PROPOSED"][k]["mean"] for _, k in test_levels]
    s1_p2 = [summary[f"P2_S1_PROPOSED"][k]["mean"] for _, k in test_levels]
    s4_p1 = [summary[f"P1_S4_PROPOSED"][k]["mean"] for _, k in test_levels]
    s4_p2 = [summary[f"P2_S4_PROPOSED"][k]["mean"] for _, k in test_levels]

    ax.bar(ind - 1.5 * w, s1_p1, w, label="S1 — P1 (Source-Only)", color=c_teal, alpha=0.7, edgecolor="black")
    ax.bar(ind - 0.5 * w, s1_p2, w, label="S1 — P2 (Multi-Structure Diversity)", color=c_blue, edgecolor="black")
    ax.bar(ind + 0.5 * w, s4_p1, w, label="S4 — P1 (Source-Only)", color=c_sand, alpha=0.7, edgecolor="black")
    ax.bar(ind + 1.5 * w, s4_p2, w, label="S4 — P2 (Multi-Structure Diversity)", color=c_coral, edgecolor="black")

    ax.axhline(50.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xticks(ind)
    ax.set_xticklabels([name for name, _ in test_levels], fontsize=11, fontweight="bold")
    ax.set_ylabel("Bilateral Attribution Accuracy (%)", fontsize=11)
    ax.set_ylim(40, 105)
    ax.set_title("Figure 2: Protocol P1 (Source-Only) vs P2 (Diversity) Across Out-of-Distribution Levels", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5, axis="y")
    fig.tight_layout()
    f2_path = os.path.join(output_dir, "fig2_p1_vs_p2_training_diversity.png")
    fig.savefig(f2_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {f2_path}")

    # Figure 3: C_4story Topological Inversion Observability vs Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Subplot 1: Condition number vs S0, S1, S4
    modalities = ["S0\n(Horizontal)", "S1\n(Horiz+Vert)", "S4\n(Multimodal)"]
    cond_nums = [13496.28, 85.91, 16.08]
    colors = ["#e63946", c_teal, c_green]
    ax1.bar(modalities, cond_nums, color=colors, edgecolor="black", alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_ylabel("Jacobian Condition Number $\\kappa(J_R)$ (log scale)", fontsize=11)
    ax1.set_title("(A) Physical Observability on C_4story", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

    # Subplot 2: Attribution Accuracy on C_4story
    accs = [
        summary["P2_S0_PROPOSED"]["level3_acc"]["mean"],
        summary["P2_S1_PROPOSED"]["level3_acc"]["mean"],
        summary["P2_S4_PROPOSED"]["level3_acc"]["mean"],
    ]
    ax2.bar(modalities, accs, color=colors, edgecolor="black", alpha=0.85)
    ax2.axhline(50.0, color="gray", linestyle="--", label="Chance Baseline (50%)")
    ax2.set_ylabel("C_4story Attribution Accuracy (%)", fontsize=11)
    ax2.set_ylim(30, 105)
    ax2.set_title("(B) PROPOSED Inverse Attribution on C_4story", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.5, axis="y")
    ax2.legend(loc="upper left")

    fig.suptitle("Figure 3: Topology-OOD Inversion on C_4story: Observability Dictates Inverse Learnability", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    f3_path = os.path.join(output_dir, "fig3_c4story_topological_inversion.png")
    fig.savefig(f3_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved: {f3_path}")


if __name__ == "__main__":
    generate_phase7_figures()
