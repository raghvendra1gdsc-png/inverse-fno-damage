"""
Unit Tests for Phase 7.1 Cross-Structure Forward Validation and Observability.

Module: tests.test_phase7_forward
Context: Phase 7.1 Quality Gate Assurance
Author: SeismoFNO Research Team
"""

import pytest
import numpy as np

from src.damage_injection import MultiStoryFrame
from src.observability import SensorConfiguration, ForwardObservationMap
from src.structure_variants import (
    STRUCTURE_SPECS,
    get_structure_config,
    get_dynamic_sensor_config,
    get_dynamic_symmetry_permutations,
    extract_structure_frequencies,
    TopologyObservationMap,
    compute_topology_damage_jacobian,
    compute_topology_noise_normalized_observability,
)


def test_all_structural_variants_modal_frequencies():
    """Verifies that all 10 structural families match expected modal frequencies to < 0.01%."""
    for s_id, spec in STRUCTURE_SPECS.items():
        freqs = extract_structure_frequencies(s_id)
        exp = spec["expected_f1"]
        actual = freqs[0]
        rel_err = abs(actual - exp) / exp * 100.0
        assert rel_err < 0.01, f"Frequency mismatch for {s_id}: expected {exp}, got {actual} (err {rel_err:.4f}%)"


def test_dynamic_symmetry_permutations_involution():
    """Verifies that dynamic symmetry permutations for 3-story and 4-story frames are valid involutions."""
    for n_stories, expected_nodes, expected_eles in [(3, 8, 9), (4, 10, 12)]:
        n_perm, e_perm = get_dynamic_symmetry_permutations(n_stories)
        assert len(n_perm) == expected_nodes
        assert len(e_perm) == expected_eles

        # Involution test: P(P(x)) == x
        for i, p in enumerate(n_perm):
            assert n_perm[p] == i, f"Node perm not involution at index {i}"
        for i, p in enumerate(e_perm):
            assert e_perm[p] == i, f"Edge perm not involution at index {i}"


def test_topology_observation_map_matches_forward_observation_map():
    """Verifies that TopologyObservationMap identically matches Phase 5 ForwardObservationMap on 3-story frames."""
    cfg = get_structure_config("SOURCE_A")
    frame = MultiStoryFrame(cfg)
    accel = np.zeros(100)
    accel[10:30] = 0.5 * np.sin(np.linspace(0, 2 * np.pi, 20))
    d_vec = np.zeros(9)
    d_vec[0] = 0.30

    s0_orig = SensorConfiguration.create_S0()
    s0_dyn = get_dynamic_sensor_config("S0", 3)

    res_orig = ForwardObservationMap.evaluate(frame, d_vec, accel, 0.02, s0_orig)
    res_dyn = TopologyObservationMap.evaluate(frame, d_vec, accel, 0.02, s0_dyn)

    max_diff = np.max(np.abs(res_orig["Y"] - res_dyn["Y"]))
    assert max_diff < 1e-12, f"Observation map numerical discrepancy: {max_diff}"


def test_c4story_forward_simulation_no_nans():
    """Verifies that 4-story frame simulation completes without NaNs or Infs across S0, S1, S4."""
    cfg = get_structure_config("C_4story")
    frame = MultiStoryFrame(cfg)
    accel = np.zeros(100)
    accel[10:30] = 0.5 * np.sin(np.linspace(0, 2 * np.pi, 20))
    d_vec = np.zeros(12)
    d_vec[0] = 0.30

    for s_id in ["S0", "S1", "S4"]:
        s_cfg = get_dynamic_sensor_config(s_id, 4)
        res = TopologyObservationMap.evaluate(frame, d_vec, accel, 0.02, s_cfg)
        assert not np.isnan(res["Y"]).any(), f"NaN detected in C_4story under {s_id}"
        assert not np.isinf(res["Y"]).any(), f"Inf detected in C_4story under {s_id}"
        assert res["Y"].shape[0] == s_cfg.num_channels


def test_c4story_observability_singular_spectrum_gate():
    """Verifies that C_4story S0 is severely ill-posed while S1 and S4 have rich directional Fisher information."""
    cfg = get_structure_config("C_4story")
    frame = MultiStoryFrame(cfg)
    accel = np.zeros(200)
    accel[10:50] = 0.5 * np.sin(np.linspace(0, 4 * np.pi, 40))
    d_0 = np.zeros(12)
    v_AB = np.zeros(12)
    v_AB[0] = 1.0 / np.sqrt(2)
    v_AB[1] = -1.0 / np.sqrt(2)

    # 1. S0 (Horizontal)
    s0_cfg = get_dynamic_sensor_config("S0", 4)
    res_0 = TopologyObservationMap.evaluate(frame, d_0, accel, 0.02, s0_cfg)
    J_0 = compute_topology_damage_jacobian(frame, d_0, accel, 0.02, s0_cfg, h=1e-3)
    obs_0 = compute_topology_noise_normalized_observability(J_0, res_0["Y"], noise_ratio=0.02, v_null=v_AB)

    assert obs_0["condition_number"] > 1000.0, f"S0 should have large condition number, got {obs_0['condition_number']}"
    assert obs_0["directional_fisher"] < 50.0, f"S0 directional Fisher should be small, got {obs_0['directional_fisher']}"

    # 2. S1 (Horizontal + Vertical)
    s1_cfg = get_dynamic_sensor_config("S1", 4)
    res_1 = TopologyObservationMap.evaluate(frame, d_0, accel, 0.02, s1_cfg)
    J_1 = compute_topology_damage_jacobian(frame, d_0, accel, 0.02, s1_cfg, h=1e-3)
    obs_1 = compute_topology_noise_normalized_observability(J_1, res_1["Y"], noise_ratio=0.02, v_null=v_AB)

    assert obs_1["sigma_min"] > 1.0, f"S1 sigma_min should be > 1.0, got {obs_1['sigma_min']}"
    assert obs_1["condition_number"] < 500.0, f"S1 should be well conditioned, got {obs_1['condition_number']}"
    assert obs_1["directional_fisher"] > 500.0, f"S1 should have strong Fisher sensitivity, got {obs_1['directional_fisher']}"

    # 3. S4 (Multimodal)
    s4_cfg = get_dynamic_sensor_config("S4", 4)
    res_4 = TopologyObservationMap.evaluate(frame, d_0, accel, 0.02, s4_cfg)
    J_4 = compute_topology_damage_jacobian(frame, d_0, accel, 0.02, s4_cfg, h=1e-3)
    obs_4 = compute_topology_noise_normalized_observability(J_4, res_4["Y"], noise_ratio=0.02, v_null=v_AB)

    assert obs_4["sigma_min"] > 10.0, f"S4 sigma_min should be > 10.0, got {obs_4['sigma_min']}"
    assert obs_4["condition_number"] < 100.0, f"S4 should be very well conditioned, got {obs_4['condition_number']}"
    assert obs_4["directional_fisher"] > 500.0, f"S4 should have strong Fisher sensitivity, got {obs_4['directional_fisher']}"
