#!/usr/bin/env python3
"""
Phase 5 Master Observability Pipeline: Multi-Record Sensor Augmentation Study.

Script: scripts/observability.py
Context: Phase 5 - Observability, Symmetry, and Sensor-Augmentation Study
Author: Inverse FNO Project Team

Evaluates S0, S1, S2, S3, S4 sensor configurations across 10 deterministic
earthquake records from different events. Computes SVD, Fisher information,
condition number, null-space alignment, and noise-separation ratio rho_AB.
Generates Figures 6 to 9 and outputs summary tables.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import glob
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


def run_sensor_augmentation_study():
    print("=" * 80)
    print("🔬 [PHASE 5] MULTI-RECORD SENSOR AUGMENTATION & OBSERVABILITY STUDY")
    print("=" * 80)

    # 1. Gather 10 Deterministic Earthquake Records
    gm_files = sorted(glob.glob("data/raw_ground_motions/*.AT2"))[:10]
    assert len(gm_files) == 10, f"Expected 10 ground motion records, found {len(gm_files)}"

    print(f"Loaded 10 deterministic earthquake records across distinct events:")
    records_info = []
    for f in gm_files:
        dt_n, acc_n, meta = parse_at2_ground_motion(f)
        records_info.append({
            "path": f,
            "name": meta["record_name"],
            "dt": dt_n,
            "accel": acc_n,
            "pga_ms2": meta["pga_ms2"],
        })
        print(f"  • {meta['record_name']} (PGA = {meta['pga_ms2']:.4f} m/s^2)")

    # 2. Setup Sensor Configurations
    configs = [
        SensorConfiguration.create_S0(),
        SensorConfiguration.create_S1(),
        SensorConfiguration.create_S2(),
        SensorConfiguration.create_S3(),
        SensorConfiguration.create_S4(),
    ]
    print(f"\nEvaluating {len(configs)} Sensor Configurations:")
    for cfg in configs:
        print(f"  [{cfg.name}] {cfg.description} ({cfg.num_channels} channels)")

    # 3. Setup Model and States
    frame = MultiStoryFrame(FrameConfig())
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    v_null = BilateralSymmetry.get_theoretical_null_vector()
    sim_dt = 0.01
    sim_dur = 10.0
    n_steps = int(sim_dur / sim_dt)
    t_target = np.arange(n_steps) * sim_dt
    h_jac = 1e-3
    lambda_reg = 1e-4

    # Storage for all runs: config_name -> list of record metrics
    results_by_config = {cfg.name: [] for cfg in configs}

    print("\nExecuting transient simulations and numerical sensitivity analysis...")
    for r_idx, rec in enumerate(records_info, 1):
        rec_name = rec["name"]
        t_native = np.arange(len(rec["accel"])) * rec["dt"]
        gm_interp = np.interp(t_target, t_native, rec["accel"])

        print(f"\nProcessing Record [{r_idx}/10]: {rec_name} ...")

        for cfg in configs:
            # 1. State A and State B evaluation
            res_A = ForwardObservationMap.evaluate(frame, d_A, gm_interp, sim_dt, cfg)
            res_B = ForwardObservationMap.evaluate(frame, d_B, gm_interp, sim_dt, cfg)

            # 2. Separation across noise levels
            sep_2pct = compute_state_separation(res_A, res_B, noise_ratio=0.02)
            sep_1pct = compute_state_separation(res_A, res_B, noise_ratio=0.01)
            sep_5pct = compute_state_separation(res_A, res_B, noise_ratio=0.05)

            # 3. Jacobian and SVD
            J_d = compute_damage_jacobian(frame, d_A, gm_interp, sim_dt, cfg, h=h_jac)
            svd_raw = analyze_svd(J_d, v_null=v_null)

            # 4. Noise-Normalized Observability & Fisher Information
            obs_2pct = compute_noise_normalized_observability(
                J_d, res_A["Y"], noise_ratio=0.02, lambda_reg=lambda_reg, v_null=v_null
            )

            rec_metrics = {
                "record_name": rec_name,
                "config_name": cfg.name,
                "num_channels": cfg.num_channels,
                "sigma_min": svd_raw["sigma_min"],
                "sigma_max": svd_raw["sigma_max"],
                "condition_number": svd_raw["condition_number"],
                "effective_rank": svd_raw["effective_rank"],
                "cosine_sim_null": svd_raw["cosine_sim_null"],
                "singular_values": svd_raw["singular_values"].tolist(),
                "lambda_min_FIM": obs_2pct["lambda_min"],
                "log_det_FIM": obs_2pct["log_det"],
                "sigma_min_J_R": obs_2pct["sigma_min_noise_normalized"],
                "condition_number_J_R": obs_2pct["condition_number_noise_normalized"],
                "relative_l2_pct": sep_2pct["relative_l2_pct"],
                "pearson_r": sep_2pct["pearson_r"],
                "rho_AB_1pct": sep_1pct["separation_ratio"],
                "rho_AB_2pct": sep_2pct["separation_ratio"],
                "rho_AB_5pct": sep_5pct["separation_ratio"],
                "max_abs_diff": sep_2pct["max_abs_diff"],
                "rms_diff": sep_2pct["rms_diff"],
            }
            results_by_config[cfg.name].append(rec_metrics)

            print(f"  {cfg.name:3s}: cond={svd_raw['condition_number']:8.1f}, sig_min(J_R)={obs_2pct['sigma_min_noise_normalized']:7.2f}, log det={obs_2pct['log_det']:6.1f}, rho_AB(2%)={sep_2pct['separation_ratio']:7.3f}")

    # 4. Statistical Aggregation (Mean +/- Std across 10 records)
    stats_summary = {}
    metric_keys = [
        "sigma_min", "sigma_max", "condition_number", "effective_rank",
        "cosine_sim_null", "lambda_min_FIM", "log_det_FIM",
        "sigma_min_J_R", "condition_number_J_R",
        "relative_l2_pct", "pearson_r",
        "rho_AB_1pct", "rho_AB_2pct", "rho_AB_5pct",
    ]

    for cfg in configs:
        cname = cfg.name
        runs = results_by_config[cname]
        c_stats = {"num_channels": cfg.num_channels, "description": cfg.description}

        for k in metric_keys:
            vals = np.array([r[k] for r in runs], dtype=np.float64)
            c_stats[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

        # Average singular spectrum across 10 records
        s_all = np.array([r["singular_values"] for r in runs])  # (10, 9)
        c_stats["mean_singular_spectrum"] = np.mean(s_all, axis=0).tolist()
        c_stats["std_singular_spectrum"] = np.std(s_all, axis=0).tolist()

        stats_summary[cname] = c_stats

    # 5. Print Consolidated Table
    print("\n" + "=" * 95)
    print("📊 CONSOLIDATED OBSERVABILITY & SYMMETRY METRICS (MEAN ± STD ACROSS 10 EARTHQUAKE RECORDS)")
    print("=" * 95)
    header = f"{'Config':8s} | {'Channels':8s} | {'cond(J)':16s} | {'sigma_min(J_R)':16s} | {'log det(FIM)':16s} | {'rho_AB (2% Noise)':18s}"
    print(header)
    print("-" * 95)

    for cfg in configs:
        cs = stats_summary[cfg.name]
        c_str = f"{cs['condition_number']['mean']:8.1f}±{cs['condition_number']['std']:<6.1f}"
        s_str = f"{cs['sigma_min_J_R']['mean']:7.2f}±{cs['sigma_min_J_R']['std']:<6.2f}"
        det_str = f"{cs['log_det_FIM']['mean']:6.1f}±{cs['log_det_FIM']['std']:<6.1f}"
        rho_str = f"{cs['rho_AB_2pct']['mean']:7.3f}±{cs['rho_AB_2pct']['std']:<6.3f}"
        print(f"{cfg.name:8s} | {cs['num_channels']:8d} | {c_str:16s} | {s_str:16s} | {det_str:16s} | {rho_str:18s}")
    print("=" * 95)

    # 6. Save Machine-Readable Results
    os.makedirs("results/observability", exist_ok=True)
    out_path = "results/observability/sensor_configurations_study.json"
    full_output = {
        "metadata": {
            "num_earthquake_records": len(records_info),
            "ground_motion_records": [r["name"] for r in records_info],
            "sim_duration_s": sim_dur,
            "dt": sim_dt,
            "finite_diff_h": h_jac,
            "regularization_lambda": lambda_reg,
        },
        "statistical_summary": stats_summary,
        "per_record_data": results_by_config,
    }

    with open(out_path, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n✅ Complete statistical dataset saved to {out_path}")

    # 7. Generate Publication Figures 6, 7, 8, 9
    os.makedirs("reports/figures/observability", exist_ok=True)
    colors = {
        "S0": "#1f78b4",  # Blue
        "S1": "#33a02c",  # Green
        "S2": "#e31a1c",  # Red
        "S3": "#ff7f00",  # Orange
        "S4": "#6a3d9a",  # Purple
    }

    # --------------------------------------------------------------------------
    # Figure 6: Singular-value spectra for S0, S1, S2, S3, S4
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    modes = np.arange(1, 10)
    for cfg in configs:
        cs = stats_summary[cfg.name]
        mean_s = np.array(cs["mean_singular_spectrum"])
        std_s = np.array(cs["std_singular_spectrum"])
        plt.semilogy(modes, mean_s, "o-", color=colors[cfg.name], label=f"{cfg.name}: {cfg.description} (cond={cs['condition_number']['mean']:.1f})", linewidth=1.8, markersize=6)
        plt.fill_between(modes, np.maximum(mean_s - std_s, 1e-6), mean_s + std_s, color=colors[cfg.name], alpha=0.15)

    plt.xticks(modes, [f"$\\sigma_{{{k}}}$" for k in modes], fontsize=10, fontweight="bold")
    plt.ylabel("Singular Value $\\sigma_k$ (Log Scale, Mean $\\pm 1\\sigma$)", fontsize=11, fontweight="bold")
    plt.xlabel("Singular Mode Index $k$", fontsize=11, fontweight="bold")
    plt.title("Figure 6: Damage-to-Response Jacobian Singular Value Spectra (S0 to S4)", fontsize=11, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend(frameon=True, facecolor="white", fontsize=8.5, loc="lower left")
    plt.tight_layout()
    fig6_path = "reports/figures/observability/fig6_singular_spectra_comparison.png"
    plt.savefig(fig6_path)
    plt.close()
    print(f"Saved: {fig6_path}")

    # --------------------------------------------------------------------------
    # Figure 7: Noise-normalized observability versus sensor configuration
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    cfg_names = [cfg.name for cfg in configs]
    x_pos = np.arange(len(cfg_names))

    sig_min_means = [stats_summary[c]["sigma_min_J_R"]["mean"] for c in cfg_names]
    sig_min_stds = [stats_summary[c]["sigma_min_J_R"]["std"] for c in cfg_names]

    bars = plt.bar(x_pos, sig_min_means, yerr=sig_min_stds, capsize=5, color=[colors[c] for c in cfg_names], edgecolor="black", alpha=0.85, width=0.55)
    plt.axhline(1.0, color="red", linestyle="--", linewidth=1.2, label=r"Noise Resolvability Threshold ($\sigma_{\min}(J_R) \geq 1.0$)")

    for bar, val in zip(bars, sig_min_means):
        plt.text(bar.get_x() + bar.get_width()/2, val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(x_pos, [f"{c}\n({stats_summary[c]['num_channels']} ch)" for c in cfg_names], fontsize=10, fontweight="bold")
    plt.ylabel(r"Noise-Normalized Minimum Sensitivity $\sigma_{\min}(\mathbf{J}_R)$", fontsize=11, fontweight="bold")
    plt.title(r"Figure 7: Noise-Normalized Observability $\sigma_{\min}(\mathbf{J}_R)$ at 2% Sensor Noise", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    fig7_path = "reports/figures/observability/fig7_noise_normalized_observability.png"
    plt.savefig(fig7_path)
    plt.close()
    print(f"Saved: {fig7_path}")

    # --------------------------------------------------------------------------
    # Figure 8: Fisher Information / Information Gain Comparison
    # --------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)
    log_det_means = [stats_summary[c]["log_det_FIM"]["mean"] for c in cfg_names]
    log_det_stds = [stats_summary[c]["log_det_FIM"]["std"] for c in cfg_names]
    lam_min_means = [stats_summary[c]["lambda_min_FIM"]["mean"] for c in cfg_names]

    b1 = ax1.bar(x_pos - 0.18, log_det_means, yerr=log_det_stds, width=0.35, capsize=4, color="#2b83ba", edgecolor="black", label=r"$\log \det(\mathbf{I}_F + \lambda \mathbf{I})$ (Information Volume)")
    ax1.set_ylabel(r"Log-Determinant $\log \det(\mathbf{I}_F + 10^{-4}\mathbf{I})$", fontsize=11, fontweight="bold", color="#2b83ba")
    ax1.tick_params(axis="y", labelcolor="#2b83ba")

    ax2 = ax1.twinx()
    b2 = ax2.bar(x_pos + 0.18, lam_min_means, width=0.35, color="#d7191c", edgecolor="black", alpha=0.85, label=r"$\lambda_{\min}(\mathbf{I}_F)$ (Worst-Case Direction)")
    ax2.set_ylabel(r"Minimum FIM Eigenvalue $\lambda_{\min}(\mathbf{I}_F)$", fontsize=11, fontweight="bold", color="#d7191c")
    ax2.tick_params(axis="y", labelcolor="#d7191c")

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{c}\n({stats_summary[c]['num_channels']} ch)" for c in cfg_names], fontsize=10, fontweight="bold")
    fig.suptitle("Figure 8: Fisher Information Matrix (FIM) Comparison Across Configurations", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    fig.tight_layout()
    fig8_path = "reports/figures/observability/fig8_fisher_information_comparison.png"
    plt.savefig(fig8_path)
    plt.close()
    print(f"Saved: {fig8_path}")

    # --------------------------------------------------------------------------
    # Figure 9: Response separation between A and B divided by sensor noise RMS (rho_AB)
    # --------------------------------------------------------------------------
    plt.figure(figsize=(9, 5.5), dpi=300)
    bar_w = 0.22
    x_base = np.arange(len(cfg_names))

    noise_cases = [
        ("rho_AB_1pct", "1.0% Sensor Noise", "#4dac26", -bar_w),
        ("rho_AB_2pct", "2.0% Sensor Noise", "#2b83ba", 0.0),
        ("rho_AB_5pct", "5.0% Sensor Noise", "#d7191c", bar_w),
    ]

    for key, label_str, col, offset in noise_cases:
        means = [stats_summary[c][key]["mean"] for c in cfg_names]
        stds = [stats_summary[c][key]["std"] for c in cfg_names]
        plt.bar(x_base + offset, means, yerr=stds, width=bar_w, capsize=3, color=col, edgecolor="black", label=label_str, alpha=0.85)

    plt.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label=r"Noise Resolvability Barrier ($\rho_{AB} = 1.0$)")
    plt.axhline(3.0, color="purple", linestyle=":", linewidth=1.5, label=r"Statistically Robust Separation ($\rho_{AB} = 3.0$)")

    plt.xticks(x_base, [f"{c}\n({stats_summary[c]['num_channels']} ch)" for c in cfg_names], fontsize=10, fontweight="bold")
    plt.ylabel(r"Separation-to-Noise Ratio $\rho_{AB} = \|\mathbf{y}_A - \mathbf{y}_B\|_2 / \sigma_{\mathrm{noise}}$", fontsize=11, fontweight="bold")
    plt.title("Figure 9: Damage State Separation Ratio $\\rho_{AB}$ Under Varying Sensor Noise", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    fig9_path = "reports/figures/observability/fig9_response_separation_ratio_vs_noise.png"
    plt.savefig(fig9_path)
    plt.close()
    print(f"Saved: {fig9_path}")

    print("=" * 80)
    print("🎉 Multi-record sensor augmentation study completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_sensor_augmentation_study()
