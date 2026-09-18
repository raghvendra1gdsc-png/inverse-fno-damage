"""
Unit Tests for Phase 7.2 Dataset, Model Hierarchy, and Statistical Logic.

Module: tests.test_phase7_dataset_and_models
Author: SeismoFNO Research Team
Context: Phase 7.2 Quality Gate Assurance
"""

import os
import glob
import json
import pytest
import numpy as np
import torch

from src.ml.phase7_dataset import Phase7SimulationDataset, Phase7BilateralPairDataset
from src.ml.topology_gfno import TopologyDualStreamGFNO
from src.ml.phase7_train_and_evaluate import clopper_pearson_ci, Phase7Loss
from src.structure_variants import get_structure_config, MultiStoryFrame
from src.structure_features import extract_structural_graph_features


def test_clopper_pearson_ci_bounds():
    """Verifies Clopper-Pearson 95% binomial confidence interval properties."""
    low, high = clopper_pearson_ci(27, 30)
    assert 0.0 <= low <= high <= 1.0
    assert 0.70 < low < 0.80
    assert high > 0.95

    # Boundary cases
    l_zero, h_zero = clopper_pearson_ci(0, 30)
    assert l_zero == 0.0
    l_full, h_full = clopper_pearson_ci(30, 30)
    assert h_full == 1.0


def test_phase7_dataset_loading_shapes():
    """Verifies that Phase7SimulationDataset loads valid tensor shapes across S0, S1, S4."""
    files = sorted(glob.glob("data/phase7_simulations/sim_*.npz"))
    assert len(files) >= 10, "Need generated simulation files for testing"

    # Test 3-story sample
    s3_files = [f for f in files if "SOURCE_A" in f]
    assert len(s3_files) > 0

    for mod in ["S0", "S1", "S4"]:
        dset = Phase7SimulationDataset(s3_files[:2], sensor_modality=mod)
        item = dset[0]
        assert item["node_signals"].shape == (8, 4, 250)
        assert item["global_signals"].shape == (5, 250)
        assert item["damage"].shape == (9,)
        assert item["node_features"].shape == (8, 4)
        assert item["edge_features"].shape == (9, 6)

    # Test 4-story sample
    s4_files = [f for f in files if "C_4story" in f]
    assert len(s4_files) > 0
    dset_c4 = Phase7SimulationDataset(s4_files[:2], sensor_modality="S1")
    item_c4 = dset_c4[0]
    assert item_c4["node_signals"].shape == (10, 4, 250)
    assert item_c4["damage"].shape == (12,)
    assert item_c4["node_features"].shape == (10, 4)
    assert item_c4["edge_features"].shape == (12, 6)


def test_model_hierarchy_forward_pass():
    """Verifies that all 6 models (B1..B5, PROPOSED) execute forward pass without error on both 3-story and 4-story."""
    files = sorted(glob.glob("data/phase7_simulations/sim_*.npz"))
    assert len(files) >= 10

    dset_3 = Phase7SimulationDataset([f for f in files if "SOURCE_A" in f][:1], sensor_modality="S1")
    dset_4 = Phase7SimulationDataset([f for f in files if "C_4story" in f][:1], sensor_modality="S1")

    item3 = dset_3[0]
    item4 = dset_4[0]

    for m_cond in ["B1", "B2", "B3", "B4", "B5", "PROPOSED"]:
        model = TopologyDualStreamGFNO(
            model_condition=m_cond,
            width_temporal=16,
            modes_temporal=8,
            n_temporal_layers=2,
            width_graph=16,
            n_graph_layers=1,
        )
        model.eval()

        # 3-story forward pass
        with torch.no_grad():
            out3 = model(
                node_signals=item3["node_signals"].unsqueeze(0),
                global_signals=item3["global_signals"].unsqueeze(0),
                node_features=item3["node_features"].unsqueeze(0),
                edge_features=item3["edge_features"].unsqueeze(0),
                edge_connectivity=item3["edge_connectivity"],
                adjacency=item3["adjacency"],
                global_scalars=item3["global_scalars"].unsqueeze(0),
                node_perm=item3["node_perm"],
                edge_perm=item3["edge_perm"],
            )
            assert out3["damage"].shape == (1, 9)
            assert not torch.isnan(out3["damage"]).any()

            # 4-story forward pass (Topological OOD)
            out4 = model(
                node_signals=item4["node_signals"].unsqueeze(0),
                global_signals=item4["global_signals"].unsqueeze(0),
                node_features=item4["node_features"].unsqueeze(0),
                edge_features=item4["edge_features"].unsqueeze(0),
                edge_connectivity=item4["edge_connectivity"],
                adjacency=item4["adjacency"],
                global_scalars=item4["global_scalars"].unsqueeze(0),
                node_perm=item4["node_perm"],
                edge_perm=item4["edge_perm"],
            )
            assert out4["damage"].shape == (1, 12)
            assert not torch.isnan(out4["damage"]).any()


def test_phase7_evaluation_artifacts_exist():
    """Verifies that all required Phase 7 inverse evaluation reports and metrics exist."""
    required_paths = [
        "results/phase7/reports/phase7_2_inverse_evaluation_report.md",
        "results/phase7/statistics/cross_structure_summary.json",
        "results/phase7/test/all_test_evaluations.json",
        "results/phase7/test/level1_id/metrics.json",
        "results/phase7/test/level2a_interpolation/metrics.json",
        "results/phase7/test/level2b_extrapolation/metrics.json",
        "results/phase7/test/level3_topology/metrics.json",
        "reports/figures/phase7/fig1_cross_structure_benchmark.png",
        "reports/figures/phase7/fig2_p1_vs_p2_training_diversity.png",
        "reports/figures/phase7/fig3_c4story_topological_inversion.png",
    ]
    for p in required_paths:
        assert os.path.exists(p), f"Missing Phase 7 artifact: {p}"
        assert os.path.getsize(p) > 0, f"Empty Phase 7 artifact: {p}"


def test_phase7_statistical_consistency():
    """Verifies that the cross-structure summary metrics conform to scientific bounds."""
    summary_path = "results/phase7/statistics/cross_structure_summary.json"
    assert os.path.exists(summary_path)
    with open(summary_path, "r") as f:
        summary = json.load(f)

    # 1. S0 unobservable baseline should remain at chance (50.0%) for B1..B4
    assert "P1_S0_B1" in summary
    assert abs(summary["P1_S0_B1"]["level1_acc"]["mean"] - 50.0) < 1.0

    # 2. PROPOSED S4 directional cosine alignment should be positive, outperforming baseline B1 (which is negative)
    assert "P2_S4_PROPOSED" in summary
    prop_s4_cos = summary["P2_S4_PROPOSED"]["level1_cos"]["mean"]
    b1_s4_cos = summary["P2_S4_B1"]["level1_cos"]["mean"]
    assert prop_s4_cos > 0.0, f"PROPOSED S4 directional cosine {prop_s4_cos} should be positive"
    assert b1_s4_cos < 0.0, f"Baseline B1 S4 directional cosine {b1_s4_cos} should be negative"

    # 3. Verify all 4 test levels are present in summary entries
    for key, data in summary.items():
        assert "level1_acc" in data
        assert "level2a_acc" in data
        assert "level2b_acc" in data
        assert "level3_acc" in data

