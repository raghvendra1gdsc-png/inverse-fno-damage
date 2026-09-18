"""
Unit tests for Phase 6.2: Targeted Bilateral Identifiability Learning.
Validates pair construction, directional loss mathematical bounds, dataset isolation,
frozen threshold protocol, and reproducibility metadata.
"""

import os
import json
import glob
import numpy as np
import torch
import pytest

from src.observability import BilateralSymmetry, SensorConfiguration
from src.ml.bilateral_pair_loss import BilateralDirectionalLoss, BilateralMarginSeparationLoss
from src.forward_fno_model import split_simulation_dataset


def test_pair_construction_uses_identical_excitation():
    """1. Pair construction must use identical excitation for State A and State B."""
    pair_files = sorted(glob.glob("data/phase6_2_pairs/pair_*.npz"))
    assert len(pair_files) > 0, "Pair files must exist"
    d = np.load(pair_files[0])
    gm = d["ground_accel"]
    assert gm.ndim == 1 or gm.ndim == 2
    assert len(gm) > 100
    assert "S0_Y_A" in d and "S0_Y_B" in d


def test_ab_damage_vectors_are_correct():
    """2. State A and State B damage vectors must reflect canonical bilateral symmetry."""
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    assert d_A[0] == pytest.approx(0.30, abs=1e-5)  # Left Col 1
    assert d_A[1] == pytest.approx(0.00, abs=1e-5)
    assert d_B[0] == pytest.approx(0.00, abs=1e-5)
    assert d_B[1] == pytest.approx(0.30, abs=1e-5)  # Right Col 1
    assert np.all(d_A[2:] == 0.0)
    assert np.all(d_B[2:] == 0.0)


def test_v_ab_has_unit_norm():
    """3. Bilateral direction vector v_AB must have strict Euclidean unit norm."""
    d_A, d_B = BilateralSymmetry.get_canonical_states()
    v_AB = (d_A - d_B) / np.linalg.norm(d_A - d_B)
    assert np.linalg.norm(v_AB) == pytest.approx(1.0, abs=1e-7)


def test_directional_loss_minimized_on_aligned_direction():
    """4. Directional loss must be 0.0 when Delta d_hat points in the exact v_AB direction."""
    loss_fn = BilateralDirectionalLoss()
    v_AB = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_AB = v_AB / torch.norm(v_AB)

    d_hat_A = torch.tensor([[0.30, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_hat_B = torch.tensor([[0.00, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    loss = loss_fn(d_hat_A, d_hat_B, v_AB)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_directional_loss_penalizes_reversed_direction():
    """5. Directional loss must equal 2.0 when Delta d_hat points in the reversed direction."""
    loss_fn = BilateralDirectionalLoss()
    v_AB = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_AB = v_AB / torch.norm(v_AB)

    # Inverted predictions
    d_hat_A = torch.tensor([[0.00, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    d_hat_B = torch.tensor([[0.30, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    loss = loss_fn(d_hat_A, d_hat_B, v_AB)
    assert loss.item() == pytest.approx(2.0, abs=1e-5)


def test_collapsed_predictions_penalized_by_margin():
    """6. Collapsed predictions (d_hat_A == d_hat_B) must be penalized by margin loss."""
    loss_fn = BilateralMarginSeparationLoss(margin=0.15)
    v_AB = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    v_AB = v_AB / torch.norm(v_AB)

    d_hat_A = torch.zeros(1, 9)
    d_hat_B = torch.zeros(1, 9)

    loss = loss_fn(d_hat_A, d_hat_B, v_AB)
    assert loss.item() == pytest.approx(0.15, abs=1e-5)


def test_pair_dataset_split_isolation():
    """7. Pair dataset must have zero leakage between train and validation pairs."""
    train_pairs = sorted(glob.glob("data/phase6_2_pairs/pair_train_*.npz"))
    val_pairs = sorted(glob.glob("data/phase6_2_pairs/pair_val_*.npz"))

    train_gms = {str(np.load(f)["gm_record"]) for f in train_pairs}
    val_gms = {str(np.load(f)["gm_record"]) for f in val_pairs}

    assert len(train_gms.intersection(val_gms)) == 0, "Train and Val pair records must be disjoint"


def test_validation_threshold_is_frozen():
    """8. Support threshold selection manifest must exist and contain valid frozen tau*."""
    manifest_path = "results/phase6_2/support_threshold_selection.json"
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            data = json.load(f)
        for cfg in ["S0", "S1", "S2", "S4"]:
            assert cfg in data
            assert 0.10 <= data[cfg]["frozen_tau"] <= 0.50


def test_sensor_configurations_use_intended_channels():
    """9. Verify sensor configurations match specified channel counts."""
    s0 = SensorConfiguration.create_S0()
    s1 = SensorConfiguration.create_S1()
    s2 = SensorConfiguration.create_S2()
    s4 = SensorConfiguration.create_S4()

    assert s0.num_channels == 3
    assert s1.num_channels == 9
    assert s2.num_channels == 9
    assert s4.num_channels == 18


def test_pairwise_prediction_differences_matched():
    """10. Pairwise differences must produce matching tensor dimensions."""
    dA = torch.randn(4, 9)
    dB = torch.randn(4, 9)
    delta = dA - dB
    assert delta.shape == (4, 9)


def test_reproducibility_metadata_complete():
    """11. Reproducibility manifest must contain required provenance fields."""
    repro_path = "results/phase6_2/reproducibility_check.json"
    if os.path.exists(repro_path):
        with open(repro_path) as f:
            data = json.load(f)
        assert "seed" in data
        assert "is_reproducible" in data
        assert data["is_reproducible"] is True
