"""
Forensic Observability, Scaling Analysis, and Directional Symmetry Audit Engine.

Module: src.forensic_observability
Context: Phase 5.5 - Forensic Observability Audit
Author: Inverse FNO Project Team
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.damage_injection import MultiStoryFrame
from src.observability import (
    BilateralSymmetry,
    SensorConfiguration,
    ForwardObservationMap,
    compute_damage_jacobian,
    analyze_svd,
    compute_noise_normalized_observability,
)


# ==============================================================================
# 1. Bilateral SVD Decomposition & Directional Sensitivity
# ==============================================================================

def decompose_bilateral_direction_svd(
    J_R: np.ndarray,
    v_AB: np.ndarray,
) -> Dict[str, Any]:
    """
    Decomposes the normalized bilateral direction v_AB onto the right-singular
    vectors V = [v_1, ..., v_9] of the noise-whitened Jacobian J_R:
      v_AB = sum_i c_i v_i,   c_i = v_i^T v_AB.

    Returns
    -------
    Dict containing:
      - 'singular_values': (9,) array sigma_1 >= ... >= sigma_9
      - 'c_coefficients': (9,) array c_i
      - 'abs_c_coefficients': (9,) array |c_i|
      - 'weighted_sensitivities': (9,) array sigma_i * |c_i|
      - 'norm_J_R_v_AB': ||J_R v_AB||_2 = sqrt(sum_i (sigma_i * c_i)^2)
      - 'dominant_mode_idx': index i maximizing |c_i|
      - 'dominant_mode_c': maximum |c_i|
      - 'dominant_mode_sigma': sigma_k corresponding to max |c_i|
      - 'v9_proj_c': |c_9|
      - 'v9_proj_sigma': sigma_9
      - 'v9_weighted': sigma_9 * |c_9|
    """
    U, s, Vt = np.linalg.svd(J_R, full_matrices=False)
    V = Vt.T  # columns are v_1, ..., v_9

    v_norm = v_AB / np.linalg.norm(v_AB)
    c = V.T @ v_norm
    abs_c = np.abs(c)
    weighted = s * abs_c
    norm_J_R_v = float(np.linalg.norm(J_R @ v_norm))

    dom_idx = int(np.argmax(abs_c))

    return {
        "singular_values": s,
        "c_coefficients": c,
        "abs_c_coefficients": abs_c,
        "weighted_sensitivities": weighted,
        "norm_J_R_v_AB": norm_J_R_v,
        "dominant_mode_idx": dom_idx,
        "dominant_mode_c": float(abs_c[dom_idx]),
        "dominant_mode_sigma": float(s[dom_idx]),
        "v9_proj_c": float(abs_c[-1]),
        "v9_proj_sigma": float(s[-1]),
        "v9_weighted": float(weighted[-1]),
    }


# ==============================================================================
# 2. Linearization Prediction vs. Actual Nonlinear Response
# ==============================================================================

def evaluate_jacobian_linearization(
    frame: MultiStoryFrame,
    d_A: np.ndarray,
    d_B: np.ndarray,
    ground_accel: np.ndarray,
    dt: float,
    sensor_config: SensorConfiguration,
    d_ref: Optional[np.ndarray] = None,
    h: float = 1e-3,
) -> Dict[str, Any]:
    """
    Computes first-order Taylor prediction:
      Delta y_lin = J(d_ref) (d_A - d_B)
    and compares against the exact nonlinear response difference:
      Delta y_actual = F(d_A) - F(d_B).
    """
    if d_ref is None:
        d_ref = np.zeros(9, dtype=np.float64)  # Undamaged symmetric baseline

    res_A = ForwardObservationMap.evaluate(frame, d_A, ground_accel, dt, sensor_config)
    res_B = ForwardObservationMap.evaluate(frame, d_B, ground_accel, dt, sensor_config)

    y_actual_diff = res_A["y_vec"] - res_B["y_vec"]
    norm_actual = float(np.linalg.norm(y_actual_diff))

    J_ref = compute_damage_jacobian(frame, d_ref, ground_accel, dt, sensor_config, h=h)
    delta_d = d_A - d_B
    y_lin_diff = J_ref @ delta_d
    norm_lin = float(np.linalg.norm(y_lin_diff))

    residual = y_actual_diff - y_lin_diff
    norm_res = float(np.linalg.norm(residual))
    rel_error = float(norm_res / norm_actual) if norm_actual > 1e-12 else float("inf")

    cos_sim = float(np.dot(y_actual_diff, y_lin_diff) / (norm_actual * norm_lin)) if (norm_actual > 1e-12 and norm_lin > 1e-12) else 0.0

    return {
        "norm_actual": norm_actual,
        "norm_lin": norm_lin,
        "norm_residual": norm_res,
        "relative_linearization_error": rel_error,
        "cosine_similarity": cos_sim,
    }


# ==============================================================================
# 3. Exact Vertical Symmetry Residual
# ==============================================================================

def compute_vertical_symmetry_residuals(
    res_A_s1: Dict[str, Any],
    res_B_s1: Dict[str, Any],
) -> Dict[str, float]:
    """
    Audits the exactness of the relation q_{+, y, A} = -q_{+, y, B} across stories.
    Residual epsilon_sym = ||q_{+, y, A} + q_{+, y, B}||_2 / ||q_{+, y, A}||_2.
    """
    Y_A = res_A_s1["Y"]
    Y_B = res_B_s1["Y"]
    # Channel map for S1:
    # 0: H_F1, 1: H_F2, 2: H_RF
    # 3: V_N3 (Story 1 Left), 4: V_N4 (Story 1 Right)
    # 5: V_N5 (Story 2 Left), 6: V_N6 (Story 2 Right)
    # 7: V_N7 (Story 3 Left), 8: V_N8 (Story 3 Right)

    story_residuals = {}
    for story, (l_idx, r_idx) in enumerate([(3, 4), (5, 6), (7, 8)], start=1):
        v_L_A = Y_A[l_idx, :]
        v_R_A = Y_A[r_idx, :]
        v_L_B = Y_B[l_idx, :]
        v_R_B = Y_B[r_idx, :]

        q_plus_A = (v_L_A + v_R_A) / 2.0
        q_plus_B = (v_L_B + v_R_B) / 2.0

        num = float(np.linalg.norm(q_plus_A + q_plus_B))
        den = float(np.linalg.norm(q_plus_A))
        eps = float(num / den) if den > 1e-15 else 0.0
        story_residuals[f"story_{story}_epsilon_sym"] = eps

    # Combined vertical joints
    v_all_A = Y_A[3:9, :]
    v_all_B = Y_B[3:9, :]
    # Left indices: 0, 2, 4 (channels 3, 5, 7); Right indices: 1, 3, 5 (channels 4, 6, 8)
    q_plus_all_A = (v_all_A[0::2, :] + v_all_A[1::2, :]) / 2.0
    q_plus_all_B = (v_all_B[0::2, :] + v_all_B[1::2, :]) / 2.0
    num_all = float(np.linalg.norm(q_plus_all_A + q_plus_all_B))
    den_all = float(np.linalg.norm(q_plus_all_A))
    story_residuals["global_vertical_epsilon_sym"] = float(num_all / den_all) if den_all > 1e-15 else 0.0

    return story_residuals


# ==============================================================================
# 4. Sensor-Block A/B Separation Decomposition
# ==============================================================================

def compute_sensor_block_separation(
    res_A: Dict[str, Any],
    res_B: Dict[str, Any],
    sensor_config: SensorConfiguration,
    noise_ratio: float = 0.02,
) -> Dict[str, Any]:
    """
    Decomposes A/B separation by physical sensor block:
      1. Horizontal acceleration block ("accel_horiz")
      2. Vertical acceleration block ("accel_vert")
      3. Column axial strain block ("strain_col")
      4. Rocking block ("rocking")
      5. Combined observation

    For each block, calculates:
      D_block = ||Y_A - Y_B||_2
      RMS_noise = sqrt(sum_c T * (noise_ratio * RMS(Y_A_c))^2)
      rho_block = D_block / RMS_noise
    """
    Y_A = res_A["Y"]
    Y_B = res_B["Y"]
    C, T = Y_A.shape
    channels = sensor_config.channels

    # Group channel indices by type
    blocks = {
        "accel_horiz": [],
        "accel_vert": [],
        "strain_col": [],
        "rocking": [],
    }
    for idx, ch in enumerate(channels):
        if ch.channel_type in blocks:
            blocks[ch.channel_type].append(idx)

    block_results = {}
    for b_name, idxs in blocks.items():
        if len(idxs) == 0:
            continue

        Y_A_b = Y_A[idxs, :]
        Y_B_b = Y_B[idxs, :]
        diff_b = Y_A_b - Y_B_b
        D_b = float(np.linalg.norm(diff_b))
        norm_A_b = float(np.linalg.norm(Y_A_b))
        rel_l2_b = float(D_b / norm_A_b) if norm_A_b > 1e-12 else 0.0

        # Channel RMS and noise
        rms_c = np.sqrt(np.mean(Y_A_b**2, axis=1))
        noise_std_c = noise_ratio * np.maximum(rms_c, 1e-8)
        noise_norm_b = float(np.sqrt(np.sum(T * (noise_std_c**2))))
        rho_b = float(D_b / noise_norm_b) if noise_norm_b > 1e-15 else 0.0

        block_results[b_name] = {
            "num_channels": len(idxs),
            "D_AB": D_b,
            "norm_signal": norm_A_b,
            "relative_l2_pct": rel_l2_b * 100.0,
            "noise_rms_norm": noise_norm_b,
            "rho_AB": rho_b,
        }

    # Combined total
    D_total = float(np.linalg.norm(Y_A - Y_B))
    norm_A_total = float(np.linalg.norm(Y_A))
    rms_all = np.sqrt(np.mean(Y_A**2, axis=1))
    noise_all = noise_ratio * np.maximum(rms_all, 1e-8)
    noise_norm_total = float(np.sqrt(np.sum(T * (noise_all**2))))
    rho_total = float(D_total / noise_norm_total) if noise_norm_total > 1e-15 else 0.0

    block_results["combined"] = {
        "num_channels": C,
        "D_AB": D_total,
        "norm_signal": norm_A_total,
        "relative_l2_pct": (D_total / norm_A_total * 100.0) if norm_A_total > 1e-12 else 0.0,
        "noise_rms_norm": noise_norm_total,
        "rho_AB": rho_total,
    }

    return block_results


# ==============================================================================
# 5. Directional Fisher Information
# ==============================================================================

def compute_directional_fisher_sensitivity(
    I_F: np.ndarray,
    v_AB: np.ndarray,
) -> Dict[str, float]:
    """
    Computes directional Fisher information along the bilateral symmetry direction:
      I_AB = v_AB^T I_F v_AB
      sqrt_I_AB = sqrt(I_AB)
    """
    v_norm = v_AB / np.linalg.norm(v_AB)
    I_AB = float(v_norm.T @ I_F @ v_norm)
    sqrt_I_AB = float(np.sqrt(max(I_AB, 0.0)))
    return {
        "directional_fisher_I_AB": I_AB,
        "sqrt_directional_fisher": sqrt_I_AB,
    }


# ==============================================================================
# 6. Noise Threshold Curve
# ==============================================================================

def compute_noise_threshold_curve(
    res_A: Dict[str, Any],
    res_B: Dict[str, Any],
    noise_levels: List[float] = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10],
) -> Dict[str, Any]:
    """
    Evaluates rho_AB across noise levels eta and finds the threshold eta* where
    rho_AB(eta*) >= 1.0.
    """
    Y_A = res_A["Y"]
    Y_B = res_B["Y"]
    diff = Y_A - Y_B
    D_AB = float(np.linalg.norm(diff))
    C, T = Y_A.shape
    rms_c = np.sqrt(np.mean(Y_A**2, axis=1))

    curve = []
    resolvable_threshold = None

    for eta in sorted(noise_levels):
        noise_std_c = eta * np.maximum(rms_c, 1e-8)
        total_noise_norm = float(np.sqrt(np.sum(T * (noise_std_c**2))))
        rho = float(D_AB / total_noise_norm) if total_noise_norm > 1e-15 else 0.0
        curve.append({"noise_ratio": eta, "noise_pct": eta * 100.0, "rho_AB": rho})

    # Linear interpolation on log(eta) to find eta* where rho_AB == 1.0
    etas = np.array([pt["noise_ratio"] for pt in curve])
    rhos = np.array([pt["rho_AB"] for pt in curve])

    if np.all(rhos < 1.0):
        threshold_status = "not noise-resolvable within tested range"
    elif np.all(rhos >= 1.0):
        threshold_status = f"resolvable across entire range (max noise tested: {max(etas)*100:.1f}%)"
        resolvable_threshold = float(max(etas))
    else:
        # Interpolate where rho == 1.0
        log_etas = np.log10(etas)
        # rhos is monotonically decreasing with eta
        log_eta_star = float(np.interp(1.0, rhos[::-1], log_etas[::-1]))
        resolvable_threshold = float(10.0**log_eta_star)
        threshold_status = f"resolvable below {resolvable_threshold*100.0:.2f}% noise"

    return {
        "curve": curve,
        "resolvable_threshold_noise_ratio": resolvable_threshold,
        "resolvable_threshold_noise_pct": (resolvable_threshold * 100.0) if resolvable_threshold else None,
        "threshold_status": threshold_status,
    }
