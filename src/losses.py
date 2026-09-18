"""
Loss Functions for Inverse Fourier Neural Operator (Inverse FNO).

Module: src.losses
Author: Inverse FNO Project Team
Context: Phase 3 - Custom Physics-Constrained Loss Functions

Contains three separate, individually-loggable loss terms:
  1. Data Fidelity Loss: MSE / L1 between predicted and ground truth damage fields.
  2. Spatial Sparsity & Total Variation Priors: L1 norm and graph-adjacent TV penalties.
  3. Forward Cycle-Consistency Loss: Round-trip dynamic verification using pre-trained Forward FNO.
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn


class DataFidelityLoss(nn.Module):
    """
    Data fidelity loss between predicted damage field and ground truth damage.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type.lower()
        if self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif self.loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss_type == "smooth_l1":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, d_pred: torch.Tensor, d_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        d_pred : torch.Tensor
            Predicted damage field [Batch, 9].
        d_true : torch.Tensor
            Ground truth damage field [Batch, 9].
        """
        return self.criterion(d_pred, d_true)


class SparsityPriorLoss(nn.Module):
    """
    Sparsity prior on predicted damage field:
      1. L1 Norm: encourages sparse, localized damage (most elements intact d_e = 0).
      2. Total Variation (TV) on Frame Graph: penalizes high-frequency spatial noise
         across structurally connected elements.
    """

    def __init__(self):
        super().__init__()
        # Frame topology connectivity for 3-story 1-bay frame (0-indexed):
        # Columns:
        #   E0: Story 1 Col Left,  E1: Story 1 Col Right
        #   E2: Story 2 Col Left,  E3: Story 2 Col Right
        #   E4: Story 3 Col Left,  E5: Story 3 Col Right
        # Beams:
        #   E6: Story 1 Beam,      E7: Story 2 Beam,      E8: Story 3 Beam
        # Adjacent structural pairs:
        self.adjacent_pairs = [
            # Vertical column line 1 (left)
            (0, 2), (2, 4),
            # Vertical column line 2 (right)
            (1, 3), (3, 5),
            # Beam-column connections
            (0, 6), (1, 6),
            (2, 7), (3, 7),
            (4, 8), (5, 8),
        ]

    def l1_sparsity(self, d_pred: torch.Tensor) -> torch.Tensor:
        """L1 norm penalty: mean(|d_e|)."""
        return torch.mean(torch.abs(d_pred))

    def total_variation(self, d_pred: torch.Tensor) -> torch.Tensor:
        """Graph total variation across adjacent structural members."""
        tv = torch.tensor(0.0, device=d_pred.device, dtype=d_pred.dtype)
        for (i, j) in self.adjacent_pairs:
            tv = tv + torch.mean(torch.abs(d_pred[:, i] - d_pred[:, j]))
        return tv / len(self.adjacent_pairs)

    def forward(self, d_pred: torch.Tensor, use_tv: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        l1_loss : torch.Tensor
        tv_loss : torch.Tensor
        """
        l1_loss = self.l1_sparsity(d_pred)
        tv_loss = self.total_variation(d_pred) if use_tv else torch.tensor(0.0, device=d_pred.device)
        return l1_loss, tv_loss


class CycleConsistencyLoss(nn.Module):
    """
    Forward Cycle-Consistency Loss:
        y_sim = G_fwd(d_pred, a_g, t)
        L_cycle = ||y_sim - y_obs||_2 / ||y_obs||_2 + alpha * MSE(y_sim, y_obs)

    Ensures that the predicted damage field physically reproduces the observed
    sparse sensor response when simulated through the pre-trained forward FNO.
    """

    def __init__(self, forward_model: nn.Module, alpha_mse: float = 0.1):
        super().__init__()
        self.forward_model = forward_model
        self.alpha_mse = alpha_mse
        self.mse = nn.MSELoss()

        # Ensure forward model is frozen
        for param in self.forward_model.parameters():
            param.requires_grad = False
        self.forward_model.eval()

    def forward(
        self,
        d_pred: torch.Tensor,
        y_obs: torch.Tensor,
        gm_norm: torch.Tensor,
        time_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        d_pred : torch.Tensor
            Predicted damage field [Batch, 9].
        y_obs : torch.Tensor
            Actual observed sensor response [Batch, 3, Time_steps] (normalized).
        gm_norm : torch.Tensor
            Normalized ground motion channel [Batch, 1, Time_steps].
        time_norm : torch.Tensor
            Normalized temporal grid channel [Batch, 1, Time_steps].

        Returns
        -------
        loss_cycle : torch.Tensor
        """
        B, _, T = y_obs.shape

        # Broadcast predicted damage field across time dimension: [Batch, 9, Time_steps]
        d_channels = d_pred.unsqueeze(-1).repeat(1, 1, T)

        # Concatenate into Forward FNO input: [Batch, 11, Time_steps]
        fwd_input = torch.cat([gm_norm, d_channels, time_norm], dim=1)

        # Run forward simulation (gradients flow through forward_model to d_pred)
        y_sim = self.forward_model(fwd_input)

        # Relative L2 error + MSE in sensor response space
        diff_norm = torch.norm(y_sim - y_obs, p=2, dim=-1)
        obs_norm = torch.norm(y_obs, p=2, dim=-1) + 1e-6
        rel_l2 = torch.mean(diff_norm / obs_norm)
        mse_val = self.mse(y_sim, y_obs)

        return rel_l2 + self.alpha_mse * mse_val


class CompositeInverseLoss(nn.Module):
    """
    Composite Loss Function for Inverse FNO training.

    L_total = lambda_data * L_data
            + lambda_sparse * L_sparse
            + lambda_tv * L_tv
            + lambda_cycle * L_cycle
    """

    def __init__(
        self,
        forward_model: Optional[nn.Module] = None,
        lambda_data: float = 1.0,
        lambda_sparse: float = 0.05,
        lambda_tv: float = 0.02,
        lambda_cycle: float = 0.50,
    ):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_sparse = lambda_sparse
        self.lambda_tv = lambda_tv
        self.lambda_cycle = lambda_cycle

        self.data_loss_fn = DataFidelityLoss(loss_type="mse")
        self.sparsity_loss_fn = SparsityPriorLoss()
        self.cycle_loss_fn = CycleConsistencyLoss(forward_model) if forward_model is not None else None

    def forward(
        self,
        d_pred: torch.Tensor,
        d_true: torch.Tensor,
        y_obs: Optional[torch.Tensor] = None,
        gm_norm: Optional[torch.Tensor] = None,
        time_norm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes composite loss and returns individual detached terms for per-epoch logging.

        Returns
        -------
        total_loss : torch.Tensor
            Loss scalar with autograd history for backpropagation.
        loss_dict : Dict[str, float]
            Individual detached values: 'total', 'data', 'sparse', 'tv', 'cycle'.
        """
        device = d_pred.device

        # 1. Data Fidelity
        l_data = self.data_loss_fn(d_pred, d_true)

        # 2. Sparsity & Total Variation
        l_sparse, l_tv = self.sparsity_loss_fn(d_pred, use_tv=(self.lambda_tv > 0))

        # 3. Cycle-Consistency
        if self.cycle_loss_fn is not None and self.lambda_cycle > 0 and y_obs is not None:
            l_cycle = self.cycle_loss_fn(d_pred, y_obs, gm_norm, time_norm)
        else:
            l_cycle = torch.tensor(0.0, device=device)

        # Weighted Total
        total_loss = (
            self.lambda_data * l_data
            + self.lambda_sparse * l_sparse
            + self.lambda_tv * l_tv
            + self.lambda_cycle * l_cycle
        )

        loss_dict = {
            "total": float(total_loss.item()),
            "data": float(l_data.item()),
            "sparse": float(l_sparse.item()),
            "tv": float(l_tv.item()),
            "cycle": float(l_cycle.item()),
        }

        return total_loss, loss_dict
