"""
Phase 6.2: Targeted Bilateral Pair & Directional Alignment Loss.

Implements physically grounded losses for bilateral damage pairs (State A vs State B):
1. Directional Alignment Loss: L_dir = 1 - cos(Delta d_hat, v_AB)
2. Margin-Based Pair Separation Loss: L_pair = max(0, m - (Delta d_hat . v_AB))
3. Combined Bilateral Learning Objective: L_total = L_base + lambda_dir * L_dir + lambda_pair * L_pair
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class BilateralDirectionalLoss(nn.Module):
    """
    Penalizes misalignment between the predicted damage difference vector
    Delta d_hat = d_hat_A - d_hat_B and the true bilateral unit direction
    v_AB = (d_A - d_B) / ||d_A - d_B||_2.

    L_dir = 1 - cos(Delta d_hat, v_AB) in [0, 2].
    - L_dir = 0 when Delta d_hat points in the exact physical bilateral direction.
    - L_dir = 1 when Delta d_hat is orthogonal (zero bilateral sensitivity).
    - L_dir = 2 when Delta d_hat points in the reversed direction (A/B inverted).
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        d_hat_A: torch.Tensor,
        d_hat_B: torch.Tensor,
        v_AB: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        d_hat_A : torch.Tensor [B, num_elements]
            Predicted damage vector for State A.
        d_hat_B : torch.Tensor [B, num_elements]
            Predicted damage vector for State B.
        v_AB : torch.Tensor [num_elements] or [B, num_elements]
            Normalized true bilateral unit direction vector.

        Returns
        -------
        loss : torch.Tensor scalar
        """
        delta_d_hat = d_hat_A - d_hat_B  # [B, E]
        norm_delta = torch.norm(delta_d_hat, p=2, dim=-1, keepdim=True) + self.eps

        if v_AB.dim() == 1:
            v_AB = v_AB.unsqueeze(0).expand_as(delta_d_hat)
        norm_v = torch.norm(v_AB, p=2, dim=-1, keepdim=True) + self.eps
        v_AB_unit = v_AB / norm_v

        # Cosine similarity in [-1, 1]
        cos_sim = torch.sum((delta_d_hat / norm_delta) * v_AB_unit, dim=-1)  # [B]

        # Stabilize bounds
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        loss_dir = torch.mean(1.0 - cos_sim)
        return loss_dir


class BilateralMarginSeparationLoss(nn.Module):
    """
    Margin-based pair separation loss ensuring that the projection of
    Delta d_hat along v_AB reaches at least a physically reasonable fraction
    of the true separation:
    L_pair = max(0, margin - (Delta d_hat . v_AB))
    """

    def __init__(self, margin: float = 0.15):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        d_hat_A: torch.Tensor,
        d_hat_B: torch.Tensor,
        v_AB: torch.Tensor,
    ) -> torch.Tensor:
        delta_d_hat = d_hat_A - d_hat_B  # [B, E]
        if v_AB.dim() == 1:
            v_AB = v_AB.unsqueeze(0).expand_as(delta_d_hat)
        norm_v = torch.norm(v_AB, p=2, dim=-1, keepdim=True) + 1e-8
        v_unit = v_AB / norm_v

        # Scalar projection along v_AB
        proj = torch.sum(delta_d_hat * v_unit, dim=-1)  # [B]
        loss_pair = torch.mean(F.relu(self.margin - proj))
        return loss_pair


class BilateralIdentifiabilityLoss(nn.Module):
    """
    Full Phase 6.2 objective combining:
    1. Base hierarchical loss on individual frames (support BCE + masked Huber)
    2. Pairwise directional cosine alignment loss along v_AB
    3. Pairwise margin separation loss along v_AB
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        lambda_dir: float = 0.05,
        lambda_pair: float = 0.01,
        margin: float = 0.15,
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.lambda_dir = lambda_dir
        self.lambda_pair = lambda_pair
        self.dir_loss_fn = BilateralDirectionalLoss()
        self.pair_loss_fn = BilateralMarginSeparationLoss(margin=margin)

    def forward(
        self,
        out_A: Dict[str, torch.Tensor],
        out_B: Dict[str, torch.Tensor],
        d_true_A: torch.Tensor,
        d_true_B: torch.Tensor,
        v_AB: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1. Base hierarchical loss on State A and State B
        loss_A, dict_A = self.base_loss_fn(out_A, d_true_A, H_nodes=out_A.get("H_nodes"))
        loss_B, dict_B = self.base_loss_fn(out_B, d_true_B, H_nodes=out_B.get("H_nodes"))
        loss_base = 0.5 * (loss_A + loss_B)

        # 2. Directional alignment loss
        d_hat_A = out_A["damage_pred"]
        d_hat_B = out_B["damage_pred"]
        l_dir = self.dir_loss_fn(d_hat_A, d_hat_B, v_AB)

        # 3. Margin separation loss
        l_pair = self.pair_loss_fn(d_hat_A, d_hat_B, v_AB)

        # 4. Total objective
        total_loss = loss_base + self.lambda_dir * l_dir + self.lambda_pair * l_pair

        # Compute cosine similarity for logging
        delta_d = d_hat_A - d_hat_B
        norm_delta = torch.norm(delta_d, p=2, dim=-1, keepdim=True) + 1e-8
        if v_AB.dim() == 1:
            v_unit = v_AB / (torch.norm(v_AB) + 1e-8)
        else:
            v_unit = v_AB / (torch.norm(v_AB, p=2, dim=-1, keepdim=True) + 1e-8)
        cos_val = torch.mean(torch.sum((delta_d / norm_delta) * v_unit, dim=-1)).item()

        loss_dict = {
            "loss_total": float(total_loss.item()),
            "loss_base": float(loss_base.item()),
            "loss_dir": float(l_dir.item()),
            "loss_pair": float(l_pair.item()),
            "cos_sim_v_AB": float(cos_val),
            "pred_separation": float(torch.mean(torch.norm(delta_d, p=2, dim=-1)).item()),
        }
        return total_loss, loss_dict
