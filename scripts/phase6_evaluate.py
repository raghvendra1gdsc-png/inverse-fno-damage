"""
Single Frozen Held-Out Test Set Evaluation and Publication Figure Generation.

Script: scripts/phase6_evaluate.py
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team

Executes a single evaluation pass on the 32 frozen held-out test runs across all
configurations and models, saving test_evaluation_results.json and generating
Figures 1 through 8 in reports/figures/phase6/.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.observability import SensorConfiguration
from src.graph.structural_graph import StructuralGraph
from src.ml.gfno import DualStreamGFNO
from src.ml.dataset import MultimodalDamageDataset
from src.forward_fno_model import split_simulation_dataset
from scripts.phase6_train import compute_epoch_metrics


def run_phase6_evaluation():
    print("=" * 80)
    print("🎯 [PHASE 6] SINGLE FROZEN HELD-OUT TEST EVALUATION & FIGURE GENERATION")
    print("=" * 80)

    device = "cpu"
    data_dir = "data/phase6_multimodal_runs"
    model_dir = "models/phase6"
    results_dir = "results/phase6"
    fig_dir = "reports/figures/phase6"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Load Frozen Held-Out Test Files
    src_raw_dir = "data/opensees_runs"
    _, _, test_files_raw = split_simulation_dataset(data_dir=src_raw_dir)
    test_files = [os.path.join(data_dir, os.path.basename(f)) for f in test_files_raw]
    print(f"Loaded {len(test_files)} frozen held-out test simulation runs (never seen during training or validation).")

    graph = StructuralGraph()

    configs = [
        ("S0", SensorConfiguration.create_S0(), 5.64),
        ("S1", SensorConfiguration.create_S1(), 1829.61),
        ("S2", SensorConfiguration.create_S2(), 1344.18),
        ("S4", SensorConfiguration.create_S4(), 2270.51),
    ]

    models_to_eval = [
        ("Baseline_InverseFNO", False, False, False, False),
        ("Baseline_GFNO",       False, True,  False, False),
        ("DualStream_GFNO",     True,  True,  False, False),
        ("Symmetry_GFNO",       True,  True,  True,  False),
        ("Proposed_GFNO",       True,  True,  True,  True),
    ]

    test_results = {}

    for cfg_code, cfg_obj, fisher_val in configs:
        print(f"\n--- Evaluating Test Set on Sensor Configuration [{cfg_code}] ---")
        test_ds = MultimodalDamageDataset(test_files, cfg_obj, is_train=False)

        # Decimate 4x
        for r in test_ds.records:
            r["Y"] = r["Y"][:, ::4]
            r["Y_global"] = r["Y_global"][:, ::4]
            r["ground_accel"] = r["ground_accel"][:, ::4]

        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        test_results[cfg_code] = {"fisher_sensitivity": fisher_val, "models": {}}

        for m_name, dual, gr, sym, hier in models_to_eval:
            ckpt_path = os.path.join(model_dir, f"{cfg_code}_{m_name}_best.pt")
            if not os.path.exists(ckpt_path):
                continue

            model = DualStreamGFNO(
                sensor_config=cfg_obj,
                structural_graph=graph,
                use_dual_stream=dual,
                use_graph=gr,
                use_symmetry=sym,
                use_hierarchical=hier,
                width_temporal=32,
                modes_temporal=12,
                n_temporal_layers=2,
                width_graph=32,
                n_graph_layers=2,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model.eval()

            all_preds, all_trues, all_probs = [], [], []
            with torch.no_grad():
                for batch in test_loader:
                    Y = batch["Y"].to(device)
                    Y_glob = batch["Y_global"].to(device)
                    gm = batch["ground_accel"].to(device)
                    d_true = batch["damage"].to(device)

                    out = model(Y, ground_accel=gm, Y_global=Y_glob)
                    all_preds.append(out["damage_pred"].cpu().numpy())
                    all_trues.append(d_true.cpu().numpy())
                    all_probs.append(out["prob_support"].cpu().numpy())

            preds_mat = np.concatenate(all_preds, axis=0)
            trues_mat = np.concatenate(all_trues, axis=0)
            probs_mat = np.concatenate(all_probs, axis=0)

            metrics = compute_epoch_metrics(preds_mat, trues_mat, probs_mat)
            test_results[cfg_code]["models"][m_name] = metrics

            print(f"  [{m_name:22s}] -> Top-1: {metrics['top1_localization_acc']*100:5.1f}% | "
                  f"Top-2: {metrics['top2_localization_acc']*100:5.1f}% | "
                  f"Dam-MAE: {metrics['severity_mae_damaged']:6.4f} | "
                  f"Ghost: {metrics['ghost_damage_mag']:6.4f}")

    # Save test results JSON
    test_json_path = os.path.join(results_dir, "test_evaluation_results.json")
    with open(test_json_path, "w") as f:
        json.dump(test_results, f, indent=2)
    print(f"\n✅ Frozen test set results saved to {test_json_path}")

    # ==========================================================================
    # GENERATE PUBLICATION FIGURES 1 TO 8
    # ==========================================================================
    print("\nGenerating Figures 1 through 8 in reports/figures/phase6/...")
    plt.rcParams["font.sans-serif"] = "Helvetica", "Arial"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["axes.linewidth"] = 0.8

    # --------------------------------------------------------------------------
    # FIGURE 1: Architecture Diagram
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis("off")
    ax.text(0.5, 0.90, "Symmetry-Aware Dual-Stream Graph Fourier Neural Operator (G-FNO)",
            ha="center", va="center", fontsize=14, fontweight="bold", color="#111111")
    ax.text(0.5, 0.82, "Phase 6 Physical Inverse Problem Architecture",
            ha="center", va="center", fontsize=11, fontstyle="italic", color="#555555")

    # Draw boxes
    boxes = [
        (0.08, 0.50, 0.22, 0.22, "#e1f5fe", "#0288d1", "Branch A: Global Stream\nFloor 1, 2, Roof accels + ag\n[Batch, 4, T] -> Temporal FNO"),
        (0.08, 0.18, 0.22, 0.22, "#f3e5f5", "#7b1fa2", "Branch B: Asymmetric Stream\nJoints & Members (S1, S2, S4)\n[Batch, 32, T] -> Node FNO"),
        (0.38, 0.34, 0.22, 0.38, "#e8f5e9", "#388e3c", "Structural Graph & Symmetry\n- Topology Laplacian L_norm\n- Involution P_node\n- H+ / H- Mode Projection"),
        (0.68, 0.34, 0.24, 0.38, "#fff3e0", "#f57c00", "Node -> Edge Hierarchical Head\n- Edge Decoder: h_e(h_i, h_j, x_e)\n- Support Head: p(z_e = 1 | Y)\n- Severity Head: mu_e, sigma_e\n- Masked Huber Loss"),
    ]

    for x, y, w, h, bg, border, txt in boxes:
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor=bg, edgecolor=border, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, txt, ha="center", va="center", fontsize=9, color="#222222")

    # Arrows
    ax.annotate("", xy=(0.38, 0.60), xytext=(0.30, 0.60), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    ax.annotate("", xy=(0.38, 0.30), xytext=(0.30, 0.30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    ax.annotate("", xy=(0.68, 0.50), xytext=(0.60, 0.50), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    f1_path = os.path.join(fig_dir, "fig1_phase6_architecture_diagram.png")
    fig.savefig(f1_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig1_phase6_architecture_diagram.png")

    # --------------------------------------------------------------------------
    # FIGURE 2: Sensor-to-Graph Mapping
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    # Draw 2D frame geometry
    col_x = [0.0, 6.0]
    stories_y = [0.0, 3.5, 7.0, 10.5]

    # Draw columns
    for x in col_x:
        ax.plot([x, x], [0, 10.5], color="#555555", lw=3.5, zorder=1)
    # Draw beams
    for y in stories_y[1:]:
        ax.plot([0, 6], [y, y], color="#777777", lw=3.0, zorder=1)

    # Draw joints (8 nodes)
    nodes = [(0, 0), (6, 0), (0, 3.5), (6, 3.5), (0, 7.0), (6, 7.0), (0, 10.5), (6, 10.5)]
    labels = ["1 (Fixed)", "2 (Fixed)", "3 (Fl 1)", "4 (Fl 1)", "5 (Fl 2)", "6 (Fl 2)", "7 (Roof)", "8 (Roof)"]
    for idx, (x, y) in enumerate(nodes):
        c = "#333333" if y == 0 else "#0288d1"
        ax.scatter([x], [y], s=160, color=c, zorder=3, edgecolors="white", linewidth=1.5)
        offset = (-0.4, -0.4) if x == 0 else (0.4, -0.4)
        ax.text(x + offset[0], y + offset[1], f"Node {labels[idx]}", fontsize=8, fontweight="bold",
                ha="right" if x == 0 else "left", va="center")

    # Sensor indicators for S4
    # Horizontal sensors at Floor 1, 2, Roof (center of beam)
    for y in [3.5, 7.0, 10.5]:
        ax.annotate("ax", xy=(3.0, y), xytext=(3.0, y + 0.5),
                    arrowprops=dict(facecolor="#e65100", edgecolor="none", width=2, headwidth=6),
                    fontsize=8, fontweight="bold", color="#e65100", ha="center")

    # Vertical sensors at nodes 3..8
    for x, y in nodes[2:]:
        ax.scatter([x], [y], s=70, facecolors="none", edgecolors="#2e7d32", linewidth=2.5, zorder=4)

    # Strain sensors on columns
    ax.text(-0.35, 1.75, "eps Col1 (S2)", color="#c2185b", fontsize=8, rotation=90, va="center", fontweight="bold")
    ax.text(6.35, 1.75, "eps Col2 (S2)", color="#c2185b", fontsize=8, rotation=90, va="center", fontweight="bold")

    ax.set_xlim(-2.0, 8.0)
    ax.set_ylim(-1.0, 12.0)
    ax.set_title("Physical Structural Frame & Sensor Placement (Configurations S0-S4)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Span X (m)")
    ax.set_ylabel("Height Y (m)")
    ax.grid(True, linestyle="--", alpha=0.3)

    f2_path = os.path.join(fig_dir, "fig2_phase6_sensor_to_graph_mapping.png")
    fig.savefig(f2_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig2_phase6_sensor_to_graph_mapping.png")

    # --------------------------------------------------------------------------
    # FIGURE 3: Top-1 Localization Accuracy Ablation
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    cfg_labels = ["S0\n(Horiz Only)", "S1\n(Horiz+Vert)", "S2\n(Horiz+Strain)", "S4\n(Multimodal)"]
    x = np.arange(len(cfg_labels))
    width = 0.16

    models = ["Baseline_InverseFNO", "Baseline_GFNO", "DualStream_GFNO", "Symmetry_GFNO", "Proposed_GFNO"]
    m_colors = ["#78909c", "#42a5f5", "#ab47bc", "#26a69a", "#ff7043"]
    m_display = ["Inverse FNO (Flat)", "G-FNO Baseline", "Dual-Stream G-FNO", "Symmetry G-FNO", "Proposed (Hierarchical)"]

    for idx, m in enumerate(models):
        vals = [test_results[c]["models"][m]["top1_localization_acc"] * 100 for c, _, _ in configs]
        ax.bar(x + (idx - 2) * width, vals, width, label=m_display[idx], color=m_colors[idx], edgecolor="#222222", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, fontsize=9)
    ax.set_ylabel("Held-Out Top-1 Localization Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Top-1 Damage Localization Across Architecture & Sensor Modality (Held-Out Test)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, 35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    f3_path = os.path.join(fig_dir, "fig3_phase6_ablation_top1_localization.png")
    fig.savefig(f3_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig3_phase6_ablation_top1_localization.png")

    # --------------------------------------------------------------------------
    # FIGURE 4: Severity MAE on Damaged Members
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    for idx, m in enumerate(models):
        vals = [test_results[c]["models"][m]["severity_mae_damaged"] for c, _, _ in configs]
        ax.bar(x + (idx - 2) * width, vals, width, label=m_display[idx], color=m_colors[idx], edgecolor="#222222", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels, fontsize=9)
    ax.set_ylabel("Severity MAE on Damaged Members (delta_E / E0)", fontsize=10, fontweight="bold")
    ax.set_title("Damage Severity Estimation Error on Truly Damaged Members (Held-Out Test)", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(0, 0.35)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=8, loc="upper right")

    f4_path = os.path.join(fig_dir, "fig4_phase6_ablation_severity_mae.png")
    fig.savefig(f4_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig4_phase6_ablation_severity_mae.png")

    # --------------------------------------------------------------------------
    # FIGURE 5: Bilateral A/B Confusion Comparison
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), dpi=300)
    with open("results/phase6/bilateral_benchmark_results.json") as f:
        bilat_res = json.load(f)

    cfg_names_list = ["S0", "S1", "S2", "S4"]
    for i, cfg in enumerate(cfg_names_list):
        conf = np.array(bilat_res[cfg]["models"]["Proposed_GFNO"]["confusion_matrix"])
        im = axes[i].imshow(conf, cmap="Blues", vmin=0, vmax=10)
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Col 1 (L)", "Col 2 (R)"])
        axes[i].set_yticklabels(["State A", "State B"] if i == 0 else ["", ""])
        axes[i].set_title(f"Config {cfg}\nsqrt(I_AB)={bilat_res[cfg]['fisher_sensitivity_sqrt']:.1f}", fontsize=9, fontweight="bold")
        for r in range(2):
            for c in range(2):
                axes[i].text(c, r, str(conf[r, c]), ha="center", va="center",
                             color="white" if conf[r, c] > 5 else "black", fontsize=11, fontweight="bold")

    fig.suptitle("Canonical Bilateral A/B Attribution Confusion Matrices (Proposed G-FNO)", fontsize=11, fontweight="bold", y=1.02)
    f5_path = os.path.join(fig_dir, "fig5_phase6_bilateral_ab_confusion.png")
    fig.savefig(f5_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig5_phase6_bilateral_ab_confusion.png")

    # --------------------------------------------------------------------------
    # FIGURE 6: Directional Fisher Sensitivity vs. Binary Accuracy
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    fishers = [bilat_res[c]["fisher_sensitivity_sqrt"] for c in cfg_names_list]
    accs = [bilat_res[c]["models"]["Proposed_GFNO"]["accuracy"] * 100 for c in cfg_names_list]

    ax.plot(fishers, accs, "o-", color="#0288d1", lw=2, markersize=8, label="Proposed G-FNO")
    for x_val, y_val, txt in zip(fishers, accs, cfg_names_list):
        ax.annotate(f"{txt} ({x_val:.1f})", xy=(x_val, y_val), xytext=(x_val + 40, y_val - 2),
                    fontsize=8, fontweight="bold", color="#333333")

    ax.set_xscale("log")
    ax.set_xlabel("Phase 5 Directional Fisher Sensitivity sqrt(I_AB) [Log Scale]", fontsize=10, fontweight="bold")
    ax.set_ylabel("Bilateral A/B Attribution Accuracy (%)", fontsize=10, fontweight="bold")
    ax.set_title("Empirical Bilateral Separation vs. Directional Fisher Information", fontsize=11, fontweight="bold", pad=12)
    ax.set_ylim(40, 105)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)

    f6_path = os.path.join(fig_dir, "fig6_phase6_directional_fisher_vs_accuracy.png")
    fig.savefig(f6_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig6_phase6_directional_fisher_vs_accuracy.png")

    # --------------------------------------------------------------------------
    # FIGURE 7: Qualitative Predictions (Best vs. Worst)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), dpi=300)
    member_labels = [f"Col {i+1}" for i in range(6)] + [f"Beam {i+1}" for i in range(3)]
    x_ele = np.arange(9)

    # Pick representative run
    d_true_ex1 = trues_mat[0]
    d_pred_ex1 = preds_mat[0]

    axes[0].bar(x_ele - 0.18, d_true_ex1, 0.35, label="Ground Truth Damage", color="#333333")
    axes[0].bar(x_ele + 0.18, d_pred_ex1, 0.35, label="Proposed G-FNO (S4)", color="#ff7043")
    axes[0].set_xticks(x_ele)
    axes[0].set_xticklabels(member_labels, fontsize=8)
    axes[0].set_ylabel("Damage Severity d_e", fontsize=9, fontweight="bold")
    axes[0].set_title("Test Run 1: Multimodal G-FNO Damage Reconstruction", fontsize=10, fontweight="bold")
    axes[0].set_ylim(0, 0.5)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[0].legend(frameon=True, fontsize=8)

    # Pick another representative run
    d_true_ex2 = trues_mat[3]
    d_pred_ex2 = preds_mat[3]
    axes[1].bar(x_ele - 0.18, d_true_ex2, 0.35, label="Ground Truth Damage", color="#333333")
    axes[1].bar(x_ele + 0.18, d_pred_ex2, 0.35, label="Proposed G-FNO (S4)", color="#ff7043")
    axes[1].set_xticks(x_ele)
    axes[1].set_xticklabels(member_labels, fontsize=8)
    axes[1].set_ylabel("Damage Severity d_e", fontsize=9, fontweight="bold")
    axes[1].set_title("Test Run 4: Multimodal G-FNO Damage Reconstruction", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 0.5)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)
    axes[1].legend(frameon=True, fontsize=8)

    fig.tight_layout()
    f7_path = os.path.join(fig_dir, "fig7_phase6_qualitative_predictions_best_vs_worst.png")
    fig.savefig(f7_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig7_phase6_qualitative_predictions_best_vs_worst.png")

    # --------------------------------------------------------------------------
    # FIGURE 8: Hierarchical Support Probability vs. Continuous Severity
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    flat_trues = trues_mat.flatten()
    flat_probs = probs_mat.flatten()
    flat_sev = (preds_mat / np.maximum(probs_mat, 1e-4)).flatten()

    damaged_mask = (flat_trues > 0.01)
    ax.scatter(flat_probs[~damaged_mask], flat_sev[~damaged_mask], color="#9e9e9e", alpha=0.4, s=25, label="Undamaged Members (z=0)")
    ax.scatter(flat_probs[damaged_mask], flat_sev[damaged_mask], color="#e53935", alpha=0.8, s=40, edgecolors="black", linewidth=0.5, label="Damaged Members (z=1)")

    ax.set_xlabel("Support Head Probability p(z_e = 1 | Y)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Conditional Severity Estimate mu_e", fontsize=10, fontweight="bold")
    ax.set_title("Decoupled Support Classification vs. Conditional Severity (Held-Out Test)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9, loc="upper left")

    f8_path = os.path.join(fig_dir, "fig8_phase6_hierarchical_support_vs_severity.png")
    fig.savefig(f8_path, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig8_phase6_hierarchical_support_vs_severity.png")

    print(f"\n🎉 All 8 Phase 6 publication figures generated in {fig_dir}!")


if __name__ == "__main__":
    run_phase6_evaluation()
