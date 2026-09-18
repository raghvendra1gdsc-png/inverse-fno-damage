"""
Unit tests for Phase 5.5 Forensic Observability Audit.

Module: tests.test_forensic_observability
Context: Phase 5.5 - Forensic Observability Audit
Author: Inverse FNO Project Team
"""

import pytest
import numpy as np

from src.damage_injection import MultiStoryFrame, FrameConfig
from src.observability import (
    BilateralSymmetry,
    SensorConfiguration,
    ForwardObservationMap,
    compute_damage_jacobian,
)
from src.forensic_observability import (
    decompose_bilateral_direction_svd,
    evaluate_jacobian_linearization,
    compute_vertical_symmetry_residuals,
    compute_sensor_block_separation,
    compute_directional_fisher_sensitivity,
    compute_noise_threshold_curve,
)


@pytest.fixture
def canonical_setup():
    frame = MultiStoryFrame(FrameConfig())
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    d_0 = np.zeros(9, dtype=np.float64)
    v_AB = BilateralSymmetry.get_theoretical_null_vector()
    gm = np.sin(np.linspace(0, 10, 200)) * 1.0  # 200 steps
    dt = 0.01
    return frame, d_A, d_B, d_0, v_AB, gm, dt


def test_bilateral_direction_normalization(canonical_setup):
    """1. Test bilateral direction normalization ||v_AB||_2 == 1 and definition."""
    _, d_A, d_B, _, v_AB, _, _ = canonical_setup
    assert len(v_AB) == 9
    assert abs(np.linalg.norm(v_AB) - 1.0) < 1e-12
    # Verify definition: v_AB = (d_A - d_B) / ||d_A - d_B||_2
    v_diff = (d_A - d_B) / np.linalg.norm(d_A - d_B)
    assert np.allclose(v_AB, v_diff)


def test_j_r_v_ab_dimensions_and_norm(canonical_setup):
    """2. Test J_R v_AB dimensions, finite norm, and consistency."""
    frame, d_A, d_B, d_0, v_AB, gm, dt = canonical_setup
    s0 = SensorConfiguration.create_S0()
    res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s0)
    J = compute_damage_jacobian(frame, d_0, gm, dt, s0, h=1e-3)
    rms_c = np.sqrt(np.mean(res_A["Y"]**2, axis=1))
    noise_std = 0.02 * np.repeat(np.maximum(rms_c, 1e-8), 200)
    J_R = J / noise_std[:, np.newaxis]

    sens = J_R @ v_AB
    assert sens.shape == (600,)  # 3 channels * 200 steps
    assert np.isfinite(sens).all()
    assert np.linalg.norm(sens) > 0.0


def test_svd_reconstruction(canonical_setup):
    """3. Test SVD reconstruction J_R = U Sigma V^T."""
    rng = np.random.RandomState(42)
    J_dummy = rng.randn(100, 9)
    U, s, Vt = np.linalg.svd(J_dummy, full_matrices=False)
    J_rec = (U * s) @ Vt
    assert np.allclose(J_dummy, J_rec, atol=1e-10)


def test_bilateral_singular_vector_decomposition(canonical_setup):
    """4. Test bilateral direction SVD decomposition: sum(c_i^2) == 1 and norm(J_R v) == sqrt(sum((s*c)^2))."""
    rng = np.random.RandomState(42)
    J_dummy = rng.randn(200, 9)
    v_AB = BilateralSymmetry.get_theoretical_null_vector()

    res = decompose_bilateral_direction_svd(J_dummy, v_AB)
    c = res["c_coefficients"]
    s = res["singular_values"]

    # Parseval's identity in orthonormal basis: sum(c_i^2) == 1
    assert abs(np.sum(c**2) - 1.0) < 1e-12
    # Sensitivity norm matches Pythagorean sum
    expected_norm = np.sqrt(np.sum((s * c)**2))
    assert abs(res["norm_J_R_v_AB"] - expected_norm) < 1e-10


def test_fisher_directional_sensitivity(canonical_setup):
    """5. Test Fisher directional sensitivity I_AB = v_AB^T I_F v_AB == ||J_R v_AB||_2^2."""
    rng = np.random.RandomState(42)
    J_R = rng.randn(150, 9)
    v_AB = BilateralSymmetry.get_theoretical_null_vector()
    I_F = J_R.T @ J_R

    dir_fisher = compute_directional_fisher_sensitivity(I_F, v_AB)
    expected_I_AB = float((J_R @ v_AB).T @ (J_R @ v_AB))

    assert abs(dir_fisher["directional_fisher_I_AB"] - expected_I_AB) < 1e-10
    assert abs(dir_fisher["sqrt_directional_fisher"] - np.sqrt(expected_I_AB)) < 1e-10


def test_ab_linearization(canonical_setup):
    """6. Test A/B Jacobian Taylor expansion prediction vs actual nonlinear response."""
    frame, d_A, d_B, d_0, _, gm, dt = canonical_setup
    s0 = SensorConfiguration.create_S0()
    lin = evaluate_jacobian_linearization(frame, d_A, d_B, gm, dt, s0, d_ref=d_0, h=1e-3)

    assert lin["norm_actual"] > 0.0
    assert lin["norm_lin"] > 0.0
    # Symmetric frame Jacobian should predict actual difference with high cosine similarity
    assert lin["cosine_similarity"] > 0.60
    assert 0.0 <= lin["relative_linearization_error"] < 2.0


def test_vertical_symmetry_residual(canonical_setup):
    """7. Test vertical symmetry residual calculation and near-zero roundoff level."""
    frame, d_A, d_B, _, _, gm, dt = canonical_setup
    s1 = SensorConfiguration.create_S1()
    res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s1)
    res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, s1)

    sym_res = compute_vertical_symmetry_residuals(res_A, res_B)
    assert "story_1_epsilon_sym" in sym_res
    assert "global_vertical_epsilon_sym" in sym_res
    # OpenSees symmetric frame must have residual at floating point precision (< 1e-8)
    assert sym_res["story_1_epsilon_sym"] < 1e-8
    assert sym_res["global_vertical_epsilon_sym"] < 1e-8


def test_sensor_block_separation(canonical_setup):
    """8. Test sensor-block separation decomposition across S4 multimodal blocks."""
    frame, d_A, d_B, _, _, gm, dt = canonical_setup
    s4 = SensorConfiguration.create_S4()
    res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s4)
    res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, s4)

    blocks = compute_sensor_block_separation(res_A, res_B, s4, noise_ratio=0.02)
    assert "accel_horiz" in blocks
    assert "accel_vert" in blocks
    assert "strain_col" in blocks
    assert "rocking" in blocks
    assert "combined" in blocks

    # Strain block separation must be substantially higher than horizontal block
    assert blocks["strain_col"]["rho_AB"] > blocks["accel_horiz"]["rho_AB"] * 50.0


def test_noise_threshold_calculation(canonical_setup):
    """9. Test noise threshold calculation and monotonicity."""
    frame, d_A, d_B, _, _, gm, dt = canonical_setup
    s1 = SensorConfiguration.create_S1()
    res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s1)
    res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, s1)

    thresh = compute_noise_threshold_curve(res_A, res_B)
    curve = thresh["curve"]
    assert len(curve) >= 5

    # Check monotonicity: rho_AB must decrease with increasing noise
    rhos = [pt["rho_AB"] for pt in curve]
    for i in range(len(rhos) - 1):
        assert rhos[i] > rhos[i + 1]


def test_deterministic_reproducibility(canonical_setup):
    """10. Test deterministic reproducibility across repeated evaluations."""
    frame, d_A, d_B, d_0, v_AB, gm, dt = canonical_setup
    s0 = SensorConfiguration.create_S0()

    res_1 = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s0)
    res_2 = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s0)
    assert np.allclose(res_1["Y"], res_2["Y"], atol=1e-14)
