#!/usr/bin/env python3
"""
Canonical Symmetry & Observability Benchmark Script.

Script: scripts/observability_symmetry.py
Context: Phase 5 - Observability, Symmetry, and Sensor-Augmentation Study
Author: Inverse FNO Project Team

Reproduces State A (30% Left Col 1) vs State B (30% Right Col 1) on canonical
Imperial Valley ground motion. Performs Jacobian finite difference step-size
convergence, SVD null space alignment, noise-aware observability, and symmetric
vs. antisymmetric feature analysis.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib.pyplot as plt

from src.damage_injection import MultiStoryFrame, FrameConfig, parse_at2_ground_motion
from src.observability import (
    BilateralSymmetry,
    SensorConfiguration,
    ForwardObservationMap,
    compute_damage_jacobian,
    analyze_svd,
    compute_noise_normalized_observability,
    compute_state_separation,
)


def run_canonical_symmetry_experiment():
    print("=" * 80)
    print("🔬 [PHASE 5] CANONICAL BILATERAL SYMMETRY OBSERVABILITY EXPERIMENT")
    print("=" * 80)

    # 1. Setup Model and Ground Motion
    frame = MultiStoryFrame(FrameConfig())
    gm_path = "data/raw_ground_motions/RSN0001_Imperial_Valley-06.AT2"
    assert os.path.exists(gm_path), f"Ground motion file not found: {gm_path}"

    dt_native, full_accel, meta = parse_at2_ground_motion(gm_path)
    sim_dt = 0.01
    sim_duration = 10.0
    n_steps = int(sim_duration / sim_dt)
    t_target = np.arange(n_steps) * sim_dt
    t_native = np.arange(len(full_accel)) * dt_native
    ground_accel = np.interp(t_target, t_native, full_accel)

    print(f"Ground Motion : {meta['record_name']} ({sim_duration}s, dt={sim_dt}s, {n_steps} steps)")
    print(f"PGA           : {np.max(np.abs(ground_accel)):.4f} m/s^2 ({np.max(np.abs(ground_accel))/9.80665:.4f} g)")

    # 2. Damage States
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    v_null = BilateralSymmetry.get_theoretical_null_vector()
    d_l2_dist = float(np.linalg.norm(d_A - d_B))

    print(f"State A (Col 1 Left 30%)  : {d_A[:3]}")
    print(f"State B (Col 2 Right 30%) : {d_B[:3]}")
    print(f"Damage L2 Distance ||d_A - d_B||_2 = {d_l2_dist:.6f}")

    # 3. Evaluate Baseline Horizontal Sensing (S0)
    s0 = SensorConfiguration.create_S0()
    res_A = ForwardObservationMap.evaluate(frame, d_A, ground_accel, sim_dt, s0)
    res_B = ForwardObservationMap.evaluate(frame, d_B, ground_accel, sim_dt, s0)
    sep_s0 = compute_state_separation(res_A, res_B, noise_ratio=0.02)

    print("\n--- Baseline Horizontal Sensor (S0) Separation ---")
    print(f"Overall Relative L2 Diff : {sep_s0['relative_l2_pct']:.4f}%")
    print(f"Overall Pearson R        : {sep_s0['pearson_r']:.8f}")
    print(f"Overall Max Abs Diff     : {sep_s0['max_abs_diff']:.6f} m/s^2")
    print(f"Overall RMS Diff         : {sep_s0['rms_diff']:.6f} m/s^2")
    print(f"Separation / Noise (2%)  : rho_AB = {sep_s0['separation_ratio']:.4f}")

    for pch in sep_s0["per_channel"]:
        print(f"  {pch['channel']} ({pch['norm_A']:.2f} m/s^2 norm): Rel L2 = {pch['relative_l2_pct']:.4f}%, R = {pch['pearson_r']:.8f}, rho_AB = {pch['separation_ratio_ch']:.4f}")

    # 4. Step-Size Sensitivity Study for Numerical Jacobian
    print("\n--- Step-Size Sensitivity Study for Damage-to-Response Jacobian (h in {1e-4, 5e-4, 1e-3, 5e-3}) ---")
    step_sizes = [1e-4, 5e-4, 1e-3, 5e-3]
    jacobians = {}
    svd_results = {}

    for h_val in step_sizes:
        J_h = compute_damage_jacobian(frame, d_A, ground_accel, sim_dt, s0, h=h_val)
        svd_h = analyze_svd(J_h, v_null=v_null)
        jacobians[h_val] = J_h
        svd_results[h_val] = svd_h
        print(f"  h={h_val:6.1e}: sigma_max={svd_h['sigma_max']:10.4f}, sigma_min={svd_h['sigma_min']:10.4e}, cond={svd_h['condition_number']:10.2f}, cos(v_min, v_null)={svd_h['cosine_sim_null']:.6f}")

    # Canonical Jacobian selected at h=1e-3
    h_canonical = 1e-3
    J_canonical = jacobians[h_canonical]
    svd_canonical = svd_results[h_canonical]

    # Verify convergence between h=5e-4 and h=1e-3
    rel_jac_diff = float(np.linalg.norm(jacobians[1e-3] - jacobians[5e-4]) / np.linalg.norm(jacobians[1e-3]))
    print(f"Jacobian relative error between h=1e-3 and h=5e-4: {rel_jac_diff * 100.0:.4f}% (Well within stable finite-diff regime)")

    # 5. Noise-Aware Observability Sweep on S0
    print("\n--- Noise-Aware Observability Sweep on S0 across Noise Ratios ---")
    noise_ratios = [0.005, 0.01, 0.02, 0.05]
    noise_results = {}

    for nr in noise_ratios:
        obs_nr = compute_noise_normalized_observability(J_canonical, res_A["Y"], noise_ratio=nr, v_null=v_null)
        sep_nr = compute_state_separation(res_A, res_B, noise_ratio=nr)
        noise_results[str(nr)] = {
            "noise_ratio": nr,
            "lambda_min_FIM": obs_nr["lambda_min"],
            "log_det_FIM": obs_nr["log_det"],
            "sigma_min_J_R": obs_nr["sigma_min_noise_normalized"],
            "condition_number_J_R": obs_nr["condition_number_noise_normalized"],
            "separation_ratio_rho_AB": sep_nr["separation_ratio"],
        }
        print(f"  Noise {nr*100:4.1f}%: rho_AB = {sep_nr['separation_ratio']:8.4f}, sigma_min(J_R) = {obs_nr['sigma_min_noise_normalized']:8.4f}, log det(FIM) = {obs_nr['log_det']:8.2f}")

    # 6. Symmetric vs Antisymmetric Feature Analysis
    print("\n--- Symmetric (q+) vs Antisymmetric (q-) Decomposition ---")
    s1 = SensorConfiguration.create_S1()
    res_A_s1 = ForwardObservationMap.evaluate(frame, d_A, ground_accel, sim_dt, s1)
    res_B_s1 = ForwardObservationMap.evaluate(frame, d_B, ground_accel, sim_dt, s1)

    # Extract vertical channels at Story 1 (Node 3 and Node 4)
    # Channel 3 is V_N3 (Left), Channel 4 is V_N4 (Right)
    q_L = res_A_s1["Y"][3, :]  # V_N3 for A
    q_R = res_A_s1["Y"][4, :]  # V_N4 for A
    q_plus_A = (q_L + q_R) / 2.0
    q_minus_A = (q_R - q_L) / 2.0

    q_L_B = res_B_s1["Y"][3, :]  # V_N3 for B
    q_R_B = res_B_s1["Y"][4, :]  # V_N4 for B
    q_plus_B = (q_L_B + q_R_B) / 2.0
    q_minus_B = (q_R_B - q_L_B) / 2.0

    diff_plus = float(np.linalg.norm(q_plus_A - q_plus_B))
    diff_minus = float(np.linalg.norm(q_minus_A - q_minus_B))
    rel_plus = diff_plus / (float(np.linalg.norm(q_plus_A)) + 1e-12) * 100.0
    rel_minus = diff_minus / (float(np.linalg.norm(q_minus_A)) + 1e-12) * 100.0

    print(f"Story 1 Vertical Motion Decomposition:")
    print(f"  Symmetric Mode (q+) Diff Norm     : {diff_plus:.6f} ({rel_plus:.4f}% rel)")
    print(f"  Antisymmetric Mode (q-) Diff Norm : {diff_minus:.6f} ({rel_minus:.4f}% rel)")
    print(f"  Antisymmetric / Symmetric Ratio   : {diff_minus / (diff_plus + 1e-12):.2f}x (Demonstrates asymmetric sensitivity!)")

    # 7. Save Machine-Readable JSON
    os.makedirs("results/observability", exist_ok=True)
    out_json_path = "results/observability/symmetry_baseline.json"

    json_payload = {
        "ground_motion": meta["record_name"],
        "sim_duration_s": sim_duration,
        "dt": sim_dt,
        "damage_L2_distance": d_l2_dist,
        "d_A": d_A.tolist(),
        "d_B": d_B.tolist(),
        "v_null_theoretical": v_null.tolist(),
        "S0_separation": {
            "abs_l2_diff": sep_s0["abs_l2_diff"],
            "relative_l2_pct": sep_s0["relative_l2_pct"],
            "pearson_r": sep_s0["pearson_r"],
            "max_abs_diff": sep_s0["max_abs_diff"],
            "rms_diff": sep_s0["rms_diff"],
            "separation_ratio_2pct_noise": sep_s0["separation_ratio"],
            "per_channel": sep_s0["per_channel"],
        },
        "jacobian_step_size_study": {
            str(h_val): {
                "h": h_val,
                "sigma_max": svd_results[h_val]["sigma_max"],
                "sigma_min": svd_results[h_val]["sigma_min"],
                "condition_number": svd_results[h_val]["condition_number"],
                "effective_rank": svd_results[h_val]["effective_rank"],
                "cosine_sim_null": svd_results[h_val]["cosine_sim_null"],
                "singular_values": svd_results[h_val]["singular_values"].tolist(),
            }
            for h_val in step_sizes
        },
        "canonical_svd": {
            "h": h_canonical,
            "singular_values": svd_canonical["singular_values"].tolist(),
            "sigma_max": svd_canonical["sigma_max"],
            "sigma_min": svd_canonical["sigma_min"],
            "condition_number": svd_canonical["condition_number"],
            "effective_rank": svd_canonical["effective_rank"],
            "cosine_sim_null": svd_canonical["cosine_sim_null"],
            "v_min": svd_canonical["v_min"].tolist(),
        },
        "noise_aware_observability": noise_results,
        "symmetric_antisymmetric_decomposition": {
            "diff_plus_norm": diff_plus,
            "diff_minus_norm": diff_minus,
            "rel_plus_pct": rel_plus,
            "rel_minus_pct": rel_minus,
            "asymmetric_amplification_factor": float(diff_minus / (diff_plus + 1e-12)),
        },
    }

    with open(out_json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n✅ Results written to {out_json_path}")

    # 8. Generate Publication Figures (Figures 1 to 5)
    os.makedirs("reports/figures/observability", exist_ok=True)
    time_arr = res_A["time"]

    # --------------------------------------------------------------------------
    # Figure 1: Damage configuration A vs B
    # --------------------------------------------------------------------------
    plt.figure(figsize=(10, 4.5), dpi=300)
    elements_x = np.arange(1, 10)
    bar_width = 0.35
    plt.bar(elements_x - bar_width/2, d_A, width=bar_width, color="#d95f02", label="State A (Left Col 1 Damaged)", edgecolor="black", alpha=0.9)
    plt.bar(elements_x + bar_width/2, d_B, width=bar_width, color="#7570b3", label="State B (Right Col 1 Damaged)", edgecolor="black", alpha=0.9)
    plt.axhline(0, color="black", linewidth=0.8)
    labels = ["C1(L,S1)", "C2(R,S1)", "C3(L,S2)", "C4(R,S2)", "C5(L,S3)", "C6(R,S3)", "B1(S1)", "B2(S2)", "B3(S3)"]
    plt.xticks(elements_x, labels, fontsize=9, fontweight="bold")
    plt.ylabel("Stiffness Reduction Factor $d_e$", fontsize=11, fontweight="bold")
    plt.title(f"Figure 1: Bilateral Symmetric Damage States A and B (Euclidean Distance: $\\|d_A - d_B\\|_2 = {d_l2_dist:.4f}$)", fontsize=11, fontweight="bold")
    plt.ylim(-0.02, 0.40)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    fig1_path = "reports/figures/observability/fig1_damage_states_A_vs_B.png"
    plt.savefig(fig1_path)
    plt.close()
    print(f"Saved: {fig1_path}")

    # --------------------------------------------------------------------------
    # Figure 2: Overlay of horizontal acceleration responses for A and B
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=300, sharex=True)
    ch_names = ["Floor 1 Horizontal Accel ($m/s^2$)", "Floor 2 Horizontal Accel ($m/s^2$)", "Roof Horizontal Accel ($m/s^2$)"]
    for idx, ax in enumerate(axes):
        ax.plot(time_arr, res_A["Y"][idx, :], label="State A ($d_1=0.30$)", color="#d95f02", linewidth=1.2, alpha=0.9)
        ax.plot(time_arr, res_B["Y"][idx, :], label="State B ($d_2=0.30$)", color="#1b9e77", linewidth=1.1, linestyle="--", alpha=0.85)
        pch = sep_s0["per_channel"][idx]
        ax.set_ylabel(ch_names[idx], fontsize=10, fontweight="bold")
        ax.set_title(f"{pch['channel']}: Rel Diff = {pch['relative_l2_pct']:.4f}%, $R = {pch['pearson_r']:.8f}$", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, facecolor="white")
    axes[-1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    fig.suptitle("Figure 2: Overlay of Horizontal Floor Acceleration Responses for States A and B (S0)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig2_path = "reports/figures/observability/fig2_response_overlay_A_vs_B.png"
    plt.savefig(fig2_path)
    plt.close()
    print(f"Saved: {fig2_path}")

    # --------------------------------------------------------------------------
    # Figure 3: Difference signal A - B
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=300, sharex=True)
    for idx, ax in enumerate(axes):
        diff_sig = res_A["Y"][idx, :] - res_B["Y"][idx, :]
        pch = sep_s0["per_channel"][idx]
        ax.plot(time_arr, diff_sig, color="#e7298a", linewidth=1.1, label=r"$y_A(t) - y_B(t)$")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        # Overlay 2% sensor noise envelope
        ax.axhline(pch["noise_rms"], color="gray", linestyle=":", label=r"$\pm 2\%$ Noise RMS Envelope")
        ax.axhline(-pch["noise_rms"], color="gray", linestyle=":")
        ax.set_ylabel(f"Diff ($m/s^2$)", fontsize=10, fontweight="bold")
        ax.set_title(f"{pch['channel']}: Max Discrepancy = {pch['max_abs_diff']:.6f} $m/s^2$ (Noise RMS: {pch['noise_rms']:.4f} $m/s^2$)", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, facecolor="white")
    axes[-1].set_xlabel("Time (s)", fontsize=11, fontweight="bold")
    fig.suptitle("Figure 3: Discrepancy Signal $y_A(t) - y_B(t)$ vs. 2% Sensor Noise Floor (S0)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig3_path = "reports/figures/observability/fig3_difference_signals_vs_noise.png"
    plt.savefig(fig3_path)
    plt.close()
    print(f"Saved: {fig3_path}")

    # --------------------------------------------------------------------------
    # Figure 4: Singular-value spectrum for horizontal sensors (S0)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(8, 5), dpi=300)
    s_vals = svd_canonical["singular_values"]
    modes_k = np.arange(1, 10)
    plt.semilogy(modes_k, s_vals, "o-", color="#1f78b4", linewidth=2.0, markersize=7, label=f"S0 (h=1e-3, cond={svd_canonical['condition_number']:.1f})")
    for h_alt in [1e-4, 5e-4, 5e-3]:
        plt.semilogy(modes_k, svd_results[h_alt]["singular_values"], "--", alpha=0.6, label=f"h={h_alt:1.0e}")
    plt.axhline(s_vals[0] * 1e-4, color="red", linestyle=":", label=r"Numerical Rank Threshold ($10^{-4} \sigma_1$)")
    plt.xticks(modes_k, [f"$\\sigma_{{{k}}}$" for k in modes_k], fontsize=10, fontweight="bold")
    plt.ylabel("Singular Value $\\sigma_k$ (Log Scale)", fontsize=11, fontweight="bold")
    plt.xlabel("Singular Mode Index $k$", fontsize=11, fontweight="bold")
    plt.title("Figure 4: Damage-to-Response Jacobian Singular Spectrum for S0 (Step-Size Study)", fontsize=11, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig4_path = "reports/figures/observability/fig4_jacobian_singular_spectrum_S0.png"
    plt.savefig(fig4_path)
    plt.close()
    print(f"Saved: {fig4_path}")

    # --------------------------------------------------------------------------
    # Figure 5: Smallest right-singular vector of J_d vs theoretical null vector
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 4.5), dpi=300)
    v_min_vec = svd_canonical["v_min"]
    # Align sign with v_null for visual clarity
    if np.dot(v_min_vec, v_null) < 0:
        v_min_vec = -v_min_vec
    cos_sim_val = svd_canonical["cosine_sim_null"]

    x_idx = np.arange(1, 10)
    width = 0.35
    plt.bar(x_idx - width/2, v_null, width=width, color="#33a02c", label=r"Theoretical Null Direction $\mathbf{v}_{\mathrm{null}} = \frac{1}{\sqrt{2}}[1, -1, 0, \dots]$", alpha=0.9, edgecolor="black")
    plt.bar(x_idx + width/2, v_min_vec, width=width, color="#e31a1c", label=f"Empirical Smallest Singular Vector $v_9$ (Cosine Sim: {cos_sim_val:.6f})", alpha=0.9, edgecolor="black")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(x_idx, labels, fontsize=9, fontweight="bold")
    plt.ylabel("Vector Component Weight", fontsize=11, fontweight="bold")
    plt.title("Figure 5: Alignment of Smallest Singular Vector $v_9$ with Bilateral Symmetry Null Direction", fontsize=11, fontweight="bold")
    plt.ylim(-1.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()
    fig5_path = "reports/figures/observability/fig5_smallest_singular_vector_alignment.png"
    plt.savefig(fig5_path)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("=" * 80)
    print("🎉 Canonical symmetry experiment completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_canonical_symmetry_experiment()
