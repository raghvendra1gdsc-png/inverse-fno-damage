"""
Hierarchical Support-Severity and Physics Loss Functions for G-FNO.

Module: src.ml.losses
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.symmetry import StructuralSymmetry


class HierarchicalDamageLoss(nn.Module):
    """
    Composite hierarchical loss function:
      L = lambda_sup * L_sup + lambda_sev * L_sev + lambda_sym * L_sym + lambda_reg * L_reg
    """

    def __init__(
        self,
        lambda_sup: float = 1.0,
        lambda_sev: float = 2.0,
        lambda_sym: float = 0.05,
        lambda_reg: float = 1e-4,
        pos_weight_sup: float = 3.5,
        huber_delta: float = 0.05,
    ):
        super().__init__()
        self.lambda_sup = lambda_sup
        self.lambda_sev = lambda_sev
        self.lambda_sym = lambda_sym
        self.lambda_reg = lambda_reg
        self.huber_delta = huber_delta

        # Class-balanced BCE with positive weighting
        self.register_buffer("pos_weight", torch.tensor([pos_weight_sup], dtype=torch.float32))

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        d_true: torch.Tensor,
        H_nodes: Optional[torch.Tensor] = None,
        H_nodes_perm: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Parameters
        ----------
        pred : Dict containing:
          - 'logits_support': [Batch, 9]
          - 'severity_mu': [Batch, 9]
          - 'damage_pred': [Batch, 9]
        d_true : torch.Tensor [Batch, 9]
            Ground truth continuous damage vector in [0, 0.5].
        H_nodes : Optional[torch.Tensor] [Batch, 8, D]
            Latent node representations of original inputs.
        H_nodes_perm : Optional[torch.Tensor] [Batch, 8, D]
            Latent node representations of symmetric reflected inputs.

        Returns
        -------
        total_loss : torch.Tensor
        loss_dict : Dict[str, float]
        """
        B, N_ele = d_true.shape

        # 1. Ground truth support mask (damage > 1% considered damaged)
        z_true = (d_true > 0.01).float()

        # 2. Support Loss (Class-Balanced BCE on Logits)
        logits_sup = pred["logits_support"]
        loss_sup = F.binary_cross_entropy_with_logits(
            logits_sup, z_true, pos_weight=self.pos_weight
        )

        # 3. Severity Loss (Masked Huber Loss ONLY on truly damaged elements)
        mu_sev = pred["severity_mu"]
        # Element-wise Huber loss
        huber_all = F.huber_loss(mu_sev, d_true, reduction="none", delta=self.huber_delta)
        # Mask by z_true: only elements with true damage contribute
        num_damaged = torch.sum(z_true)
        if num_damaged > 0:
            loss_sev = torch.sum(huber_all * z_true) / (num_damaged + 1e-6)
        else:
            loss_sev = torch.tensor(0.0, device=d_true.device, dtype=d_true.dtype)

        # 4. Optional Latent Symmetry Consistency Loss
        loss_sym = torch.tensor(0.0, device=d_true.device, dtype=d_true.dtype)
        if H_nodes is not None and H_nodes_perm is not None:
            # P H_nodes_perm should align with H_nodes
            P_H_perm = StructuralSymmetry.apply_node_permutation(H_nodes_perm, dim=1)
            loss_sym = F.mse_loss(P_H_perm, H_nodes)

        # 5. L2 Regularization on predicted damage
        loss_reg = torch.mean(pred["damage_pred"] ** 2)

        # Total weighted loss
        total_loss = (
            self.lambda_sup * loss_sup
            + self.lambda_sev * loss_sev
            + self.lambda_sym * loss_sym
            + self.lambda_reg * loss_reg
        )

        loss_dict = {
            "total_loss": float(total_loss.item()),
            "loss_support": float(loss_sup.item()),
            "loss_severity": float(loss_sev.item()),
            "loss_symmetry": float(loss_sym.item()),
            "loss_reg": float(loss_reg.item()),
        }

        return total_loss, loss_dict
