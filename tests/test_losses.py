"""
Unit tests for custom physics-informed loss functions.
"""

import pytest
import torch
from src.forward_fno_model import ForwardFNO
from src.losses import (
    DataFidelityLoss,
    SparsityPriorLoss,
    CycleConsistencyLoss,
    CompositeInverseLoss,
)


def test_data_fidelity_loss():
    """Tests MSE data fidelity loss computation."""
    loss_fn = DataFidelityLoss(loss_type="mse")
    pred = torch.tensor([[0.2, 0.4]])
    true = torch.tensor([[0.2, 0.0]])
    val = loss_fn(pred, true)
    expected = 0.5 * (0.0**2 + 0.4**2)  # 0.08
    assert abs(val.item() - expected) < 1e-6


def test_sparsity_and_tv_priors():
    """Tests L1 norm and graph total variation calculations."""
    prior = SparsityPriorLoss()
    # Smeared damage across all 9 elements
    d_smeared = torch.ones(1, 9) * 0.1
    # Concentrated single-element damage
    d_sparse = torch.zeros(1, 9)
    d_sparse[0, 0] = 0.9

    l1_smeared, tv_smeared = prior(d_smeared, use_tv=True)
    l1_sparse, tv_sparse = prior(d_sparse, use_tv=True)

    # L1 norm of smeared vs sparse with same sum
    assert abs(l1_smeared.item() - l1_sparse.item()) < 1e-6
    # TV should be zero for uniform smeared and positive for sharp spike
    assert tv_smeared.item() == 0.0
    assert tv_sparse.item() > 0.0


def test_cycle_consistency_gradient_flow():
    """Tests that cycle-consistency loss backpropagates through Forward FNO to damage input."""
    fwd = ForwardFNO(in_channels=11, out_channels=3, modes=8, width=16, n_layers=2)
    cycle_loss_fn = CycleConsistencyLoss(forward_model=fwd)

    d_pred = torch.randn(2, 9, requires_grad=True)
    y_obs = torch.randn(2, 3, 100)
    gm = torch.randn(2, 1, 100)
    time_grid = torch.linspace(0, 1, 100).repeat(2, 1, 1)

    loss = cycle_loss_fn(d_pred, y_obs, gm, time_grid)
    loss.backward()

    assert d_pred.grad is not None
    assert not torch.isnan(d_pred.grad).any()
    assert torch.norm(d_pred.grad).item() > 0.0
