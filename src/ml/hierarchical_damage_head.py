"""
Node-to-Edge Decoder and Hierarchical Support & Severity Heads.

Module: src.ml.hierarchical_damage_head
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.structural_graph import StructuralGraph


class NodeToEdgeDecoder(nn.Module):
    """
    Decodes pair-wise node representations into member-level representations
    for all 9 structural elements:
      h_e = MLP([h_i, h_j, |h_i - h_j|, h_i * h_j, x_e])
    """

    def __init__(
        self,
        node_dim: int,
        edge_feat_dim: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        in_dim = 4 * node_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        H_nodes: torch.Tensor,
        edge_connectivity: torch.Tensor,
        edge_static_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H_nodes : torch.Tensor [Batch, 8, node_dim]
            Latent node representations.
        edge_connectivity : torch.Tensor [2, 9]
            Start and end node indices for each member.
        edge_static_features : torch.Tensor [9, edge_feat_dim]
            Physical member features.

        Returns
        -------
        H_edges : torch.Tensor [Batch, 9, out_dim]
            Latent structural member representations.
        """
        B, N, D = H_nodes.shape
        idx_i = edge_connectivity[0]  # [9]
        idx_j = edge_connectivity[1]  # [9]

        # Gather node representations
        h_i = H_nodes[:, idx_i, :]    # [B, 9, D]
        h_j = H_nodes[:, idx_j, :]    # [B, 9, D]

        diff = torch.abs(h_i - h_j)   # [B, 9, D]
        prod = h_i * h_j              # [B, 9, D]

        # Broadcast static features
        x_e = edge_static_features.unsqueeze(0).expand(B, -1, -1)  # [B, 9, edge_feat_dim]

        pair_feats = torch.cat([h_i, h_j, diff, prod, x_e], dim=-1)  # [B, 9, 4*D + edge_feat_dim]
        return self.mlp(pair_feats)   # [B, 9, out_dim]


class HierarchicalDamageHead(nn.Module):
    """
    Hierarchical damage head decomposing damage into:
      1. Support head: p(z_e = 1 | Y) in [0, 1] (Bernoulli probability of damage)
      2. Severity head: mu_e in [0, 0.5], sigma_e > 0 (conditional severity given z_e = 1)
      3. Expected damage: d_hat_e = p_e * mu_e in [0, 0.5]
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        # Support head (Binary classification)
        self.support_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Severity head (Conditional regression with uncertainty)
        self.severity_mu_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1], multiplied by 0.5
        )
        self.severity_sigma_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensures positive variance
        )

    def forward(self, H_edges: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        H_edges : torch.Tensor [Batch, 9, in_dim]

        Returns
        -------
        Dict with:
          - 'prob_support': [Batch, 9] in [0, 1]
          - 'logits_support': [Batch, 9]
          - 'severity_mu': [Batch, 9] in [0, 0.5]
          - 'severity_sigma': [Batch, 9] > 0
          - 'damage_pred': [Batch, 9] = p_e * mu_e in [0, 0.5]
        """
        logits = self.support_net(H_edges).squeeze(-1)            # [B, 9]
        prob_sup = torch.sigmoid(logits)                          # [B, 9]

        mu = 0.5 * self.severity_mu_net(H_edges).squeeze(-1)      # [B, 9] in [0, 0.5]
        sigma = self.severity_sigma_net(H_edges).squeeze(-1) + 1e-4  # [B, 9]

        d_pred = prob_sup * mu                                    # [B, 9]

        return {
            "prob_support": prob_sup,
            "logits_support": logits,
            "severity_mu": mu,
            "severity_sigma": sigma,
            "damage_pred": d_pred,
        }


class DirectRegressionHead(nn.Module):
    """
    Standard regression head for baseline comparison without hierarchical decomposition.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU(),  # Physical constraint: non-negative damage
        )

    def forward(self, H_edges: torch.Tensor) -> Dict[str, torch.Tensor]:
        d_pred = self.net(H_edges).squeeze(-1)
        # Cap at 0.5 for fair comparison
        d_pred = torch.clamp(d_pred, 0.0, 0.5)
        prob_sup = (d_pred > 0.01).float()
        return {
            "prob_support": prob_sup,
            "logits_support": prob_sup,
            "severity_mu": d_pred,
            "severity_sigma": torch.zeros_like(d_pred),
            "damage_pred": d_pred,
        }
