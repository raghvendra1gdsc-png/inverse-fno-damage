"""
Automated Regression Tests for Phase 7 Pristine Feature Contract.

Module: tests.test_phase7_feature_leakage_contract
Context: Ensures no target damage information leaks into graph neural operator inputs.
"""

import os
import sys
import glob
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.abspath("."))

from src.damage_injection import MultiStoryFrame
from src.structure_variants import get_structure_config
from src.structure_features import extract_structural_graph_features
from src.ml.phase7_dataset import Phase7SimulationDataset, Phase7BilateralPairDataset


def test_edge_features_invariant_to_damage_vector():
    """
    Test 1: Verify edge features are strictly invariant to the damage vector.
    Two identical frames with different damage fields must yield IDENTICAL graph features.
    """
    cfg = get_structure_config("SOURCE_A")
    frame1 = MultiStoryFrame(cfg)
    frame2 = MultiStoryFrame(cfg)

    # Damage state A: Left Col 1 = 30%
    dA = np.zeros(frame1.num_elements)
    dA[0] = 0.30
    frame1.build_model(dA)

    # Damage state B: Right Col 1 = 30%
    dB = np.zeros(frame2.num_elements)
    dB[1] = 0.30
    frame2.build_model(dB)

    feats1 = extract_structural_graph_features(frame1)
    feats2 = extract_structural_graph_features(frame2)

    # Node features must be identical
    np.testing.assert_allclose(feats1["node_features"], feats2["node_features"], err_msg="Node features vary with damage!")

    # Edge features must be identical across ALL 6 columns
    np.testing.assert_allclose(feats1["edge_features"], feats2["edge_features"], err_msg="Edge features vary with damage!")

    # Adjacency, connectivity, and global scalars must be identical
    np.testing.assert_allclose(feats1["edge_connectivity"], feats2["edge_connectivity"])
    np.testing.assert_allclose(feats1["adjacency"], feats2["adjacency"])
    np.testing.assert_allclose(feats1["global_scalars"], feats2["global_scalars"])


def test_no_feature_equivalent_to_damaged_stiffness():
    """
    Test 2: Verify no feature is numerically equivalent to E0 * (1 - d_e) or any transform of d_e.
    """
    cfg = get_structure_config("SOURCE_A")
    frame = MultiStoryFrame(cfg)

    # Damage vector with diverse values
    d = np.array([0.10, 0.20, 0.30, 0.05, 0.15, 0.25, 0.0, 0.0, 0.0])
    frame.build_model(d)

    feats = extract_structural_graph_features(frame)
    ef = feats["edge_features"]

    # Check col 3 (E_norm) is NOT (1 - d_e)
    assert not np.allclose(ef[:, 3], 1.0 - d), "Feature leakage detected: col 3 is (1 - d_e)!"
    # In pristine contract, col 3 is nominal E / E0 = 1.0 everywhere
    np.testing.assert_allclose(ef[:, 3], np.ones(len(d)), err_msg="col 3 must be nominal Young's modulus (1.0)!")


def test_dataset_generation_contract_on_sample_files():
    """
    Test 3 & 4: Inspect simulation files to ensure model inputs do NOT contain damage info.
    """
    files = sorted(glob.glob("data/phase7_simulations/*.npz"))
    if not files:
        pytest.skip("No phase 7 simulation files found to test.")

    # Check first 10 files
    for f in files[:10]:
        data = np.load(f)
        ef = data["edge_features"]
        d = data["damage_vector"]

        # If damaged, ef[:, 3] must NOT correlate with d
        if np.max(d) > 0.01:
            # col 3 should be nominal (1.0 for standard configs), NOT (1 - d)
            assert not np.allclose(ef[:, 3], 1.0 - d), f"Leaked damage detected in {f}!"


def test_pristine_feature_contract_across_all_structural_variants():
    """
    Test 5: Verify feature contract holds across all 10 structural families.
    """
    for struct_id in ["SOURCE_A", "B_train_1", "B_train_2", "B_train_3", "B_train_4",
                      "B_int_1", "B_int_2", "B_ext_soft", "B_ext_stiff", "C_4story"]:
        cfg = get_structure_config(struct_id)
        frame = MultiStoryFrame(cfg)
        d_zero = np.zeros(frame.num_elements)
        d_dmg = np.full(frame.num_elements, 0.30)

        frame.build_model(d_dmg)
        feats_dmg = extract_structural_graph_features(frame)

        frame.build_model(d_zero)
        feats_pristine = extract_structural_graph_features(frame)

        # Features must be perfectly invariant to damage across all structural families
        np.testing.assert_allclose(
            feats_dmg["edge_features"],
            feats_pristine["edge_features"],
            err_msg=f"Feature contract violated for structure {struct_id}!"
        )
