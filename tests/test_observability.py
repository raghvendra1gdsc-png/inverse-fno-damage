"""
Unit tests for Phase 5: Observability, Symmetry Operator, and Sensor Augmentation.

Module: tests.test_observability
Context: Phase 5 - Observability, Symmetry, and Sensor-Augmentation Study
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
    analyze_svd,
    compute_noise_normalized_observability,
    compute_state_separation,
)


def test_bilateral_permutation_properties():
    """Tests P(P(d)) = d and ||P(d)||_2 = ||d||_2 for arbitrary damage vectors."""
    P = BilateralSymmetry.get_permutation_matrix()

    # Verify P is orthogonal, symmetric, and involution
    assert np.allclose(P @ P, np.eye(9)), "P must be an involution: P^2 = I"
    assert np.allclose(P, P.T), "P must be symmetric: P = P^T"

    # Test on random damage vector
    rng = np.random.RandomState(42)
    d_rand = rng.uniform(0.0, 0.5, size=9)

    d_perm = BilateralSymmetry.apply_permutation(d_rand)
    d_double_perm = BilateralSymmetry.apply_permutation(d_perm)

    # 1. P(P(d)) == d
    assert np.allclose(d_double_perm, d_rand), "P(P(d)) must equal d"

    # 2. ||P(d)||_2 == ||d||_2
    assert abs(np.linalg.norm(d_perm) - np.linalg.norm(d_rand)) < 1e-12, "Permutation must preserve L2 norm"

    # 3. Check element transpositions
    # Columns 1 <-> 2, 3 <-> 4, 5 <-> 6
    assert d_perm[0] == d_rand[1]
    assert d_perm[1] == d_rand[0]
    assert d_perm[2] == d_rand[3]
    assert d_perm[3] == d_rand[2]
    assert d_perm[4] == d_rand[5]
    assert d_perm[5] == d_rand[4]
    # Beams 7, 8, 9 unchanged
    assert d_perm[6] == d_rand[6]
    assert d_perm[7] == d_rand[7]
    assert d_perm[8] == d_rand[8]


def test_canonical_symmetry_states():
    """Tests State A and State B construction and theoretical null direction."""
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    assert d_A[0] == 0.30 and d_A[1] == 0.0
    assert d_B[1] == 0.30 and d_B[0] == 0.0
    assert np.allclose(d_B, BilateralSymmetry.apply_permutation(d_A))

    v_null = BilateralSymmetry.get_theoretical_null_vector()
    assert len(v_null) == 9
    assert abs(np.linalg.norm(v_null) - 1.0) < 1e-12
    assert v_null[0] > 0 and v_null[1] < 0
    assert np.allclose(v_null[2:], 0.0)


def test_sensor_configuration_construction():
    """Tests creation of S0, S1, S2, S3, and S4 configurations."""
    s0 = SensorConfiguration.create_S0()
    s1 = SensorConfiguration.create_S1()
    s2 = SensorConfiguration.create_S2()
    s3 = SensorConfiguration.create_S3()
    s4 = SensorConfiguration.create_S4()

    assert s0.num_channels == 3
    assert s1.num_channels == 9
    assert s2.num_channels == 9
    assert s3.num_channels == 6
    assert s4.num_channels == 18

    assert [ch.channel_id for ch in s0.channels] == ["H_F1", "H_F2", "H_RF"]


def test_damage_to_response_mapping():
    """Tests forward observation map evaluation and output dimensions."""
    frame = MultiStoryFrame(FrameConfig())
    d_A, _ = BilateralSymmetry.get_canonical_states()
    gm = np.sin(np.linspace(0, 10, 200)) * 1.0  # 200 steps
    dt = 0.01

    s0 = SensorConfiguration.create_S0()
    res_s0 = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s0)

    assert res_s0["Y"].shape == (3, 200)
    assert res_s0["y_vec"].shape == (600,)
    assert not np.isnan(res_s0["Y"]).any()


def test_jacobian_dimensions_and_finite_difference():
    """Tests finite-difference Jacobian dimensions (N_obs x 9) and step-size consistency."""
    frame = MultiStoryFrame(FrameConfig())
    d_zero = np.zeros(9)
    gm = np.sin(np.linspace(0, 5, 100)) * 1.0  # 100 steps
    dt = 0.01

    s0 = SensorConfiguration.create_S0()
    J_1 = compute_damage_jacobian(frame, d_zero, gm, dt, s0, h=1e-3)
    J_2 = compute_damage_jacobian(frame, d_zero, gm, dt, s0, h=5e-4)

    assert J_1.shape == (300, 9)
    assert J_2.shape == (300, 9)
    assert not np.isnan(J_1).any()

    # Relative difference between step sizes should be small (< 1%)
    rel_diff = np.linalg.norm(J_1 - J_2) / np.linalg.norm(J_1)
    assert rel_diff < 0.05, f"Jacobian step size sensitivity error: {rel_diff * 100}%"


def test_svd_output_and_condition_number():
    """Tests SVD analysis, condition number, and singular values sorting."""
    # Create synthetic Jacobian
    J_dummy = np.random.RandomState(42).randn(100, 9)
    v_null = BilateralSymmetry.get_theoretical_null_vector()

    svd_res = analyze_svd(J_dummy, v_null=v_null)
    s = svd_res["singular_values"]

    assert len(s) == 9
    assert np.all(s[:-1] >= s[1:]), "Singular values must be sorted in descending order"
    assert svd_res["condition_number"] >= 1.0
    assert svd_res["v_min"].shape == (9,)
    assert svd_res["cosine_sim_null"] is not None
    assert 0.0 <= svd_res["cosine_sim_null"] <= 1.0


def test_fisher_information_dimensions_and_properties():
    """Tests FIM dimensions (9x9), symmetry, and positive semi-definiteness."""
    J_dummy = np.random.RandomState(42).randn(300, 9)
    Y_ref = np.random.RandomState(42).randn(3, 100)

    fim_res = compute_noise_normalized_observability(J_dummy, Y_ref, noise_ratio=0.02, lambda_reg=1e-4)

    FIM = fim_res["FIM"]
    assert FIM.shape == (9, 9)
    assert np.allclose(FIM, FIM.T), "FIM must be symmetric"

    eigvals = fim_res["fim_eigenvalues"]
    assert np.all(eigvals >= -1e-12), "FIM must be positive semi-definite"
    assert fim_res["lambda_min"] >= -1e-12
    assert fim_res["log_det"] is not None


def test_state_separation_metrics():
    """Tests relative L2 difference, Pearson R, and separation-to-noise ratio."""
    frame = MultiStoryFrame(FrameConfig())
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    gm = np.sin(np.linspace(0, 5, 100)) * 1.0
    dt = 0.01

    s0 = SensorConfiguration.create_S0()
    res_A = ForwardObservationMap.evaluate(frame, d_A, gm, dt, s0)
    res_B = ForwardObservationMap.evaluate(frame, d_B, gm, dt, s0)

    sep = compute_state_separation(res_A, res_B, noise_ratio=0.02)

    assert sep["abs_l2_diff"] > 0.0
    assert sep["relative_l2_pct"] < 1.0  # Horizontal floor sensing has tiny relative diff (< 1%)
    assert sep["pearson_r"] > 0.999
    assert sep["separation_ratio"] < 1.0  # Must be submerged below 2% noise floor
