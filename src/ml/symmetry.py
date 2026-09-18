"""
Structural Symmetry Operators and Group-Theoretic Latent Decomposition.

Module: src.ml.symmetry
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from src.graph.structural_graph import StructuralGraph


class StructuralSymmetry:
    """
    Formal bilateral symmetry permutation operators for nodes (8 joints)
    and edges (9 members), derived strictly from OpenSees topology.
    """

    # 0-indexed permutations
    NODE_PERMUTATION = [1, 0, 3, 2, 5, 4, 7, 6]
    EDGE_PERMUTATION = [1, 0, 3, 2, 5, 4, 6, 7, 8]

    @classmethod
    def get_node_permutation_matrix(cls) -> torch.Tensor:
        """Returns P_node in {0, 1}^(8 x 8) as float tensor."""
        P = torch.zeros((8, 8), dtype=torch.float32)
        for i, j in enumerate(cls.NODE_PERMUTATION):
            P[i, j] = 1.0
        return P

    @classmethod
    def get_edge_permutation_matrix(cls) -> torch.Tensor:
        """Returns P_edge in {0, 1}^(9 x 9) as float tensor."""
        P = torch.zeros((9, 9), dtype=torch.float32)
        for i, j in enumerate(cls.EDGE_PERMUTATION):
            P[i, j] = 1.0
        return P

    @classmethod
    def apply_node_permutation(cls, H: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Applies left/right reflection to node dimension of tensor H.
        For H of shape [Batch, 8, Features], swaps (0<->1, 2<->3, 4<->5, 6<->7).
        """
        idx = torch.tensor(cls.NODE_PERMUTATION, device=H.device, dtype=torch.long)
        return torch.index_select(H, dim, idx)

    @classmethod
    def apply_edge_permutation(cls, d: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Applies left/right reflection to edge/damage dimension of tensor d.
        For d of shape [Batch, 9], swaps (0<->1, 2<->3, 4<->5).
        """
        idx = torch.tensor(cls.EDGE_PERMUTATION, device=d.device, dtype=torch.long)
        return torch.index_select(d, dim, idx)


class LatentSymmetryProjector(nn.Module):
    """
    Decomposes latent node representation H in R^(B x 8 x D) into
    strictly symmetric (H_+) and antisymmetric (H_-) subspaces:
      H_+ = 0.5 * (H + P_node H)   -->  P H_+ =  H_+
      H_- = 0.5 * (H - P_node H)   -->  P H_- = -H_-
    """

    def __init__(self, node_dim: int = 1):
        super().__init__()
        self.node_dim = node_dim
        self.register_buffer(
            "node_perm",
            torch.tensor(StructuralSymmetry.NODE_PERMUTATION, dtype=torch.long),
        )

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H : torch.Tensor [B, 8, D] or [B, 8, D, T]

        Returns
        -------
        H_plus : torch.Tensor
            Symmetric mode (invariant under P_node reflection).
        H_minus : torch.Tensor
            Antisymmetric mode (sign-reversing under P_node reflection).
        """
        PH = torch.index_select(H, self.node_dim, self.node_perm)
        H_plus = 0.5 * (H + PH)
        H_minus = 0.5 * (H - PH)
        return H_plus, H_minus

    def verify_symmetry_identities(self, H: torch.Tensor, atol: float = 1e-6) -> Tuple[bool, float, float]:
        """
        Verifies ||P H_+ - H_+||_inf == 0 and ||P H_- + H_-||_inf == 0.
        """
        H_plus, H_minus = self.forward(H)
        PH_plus = torch.index_select(H_plus, self.node_dim, self.node_perm)
        PH_minus = torch.index_select(H_minus, self.node_dim, self.node_perm)

        err_plus = float(torch.max(torch.abs(PH_plus - H_plus)).item())
        err_minus = float(torch.max(torch.abs(PH_minus + H_minus)).item())
        is_valid = (err_plus < atol) and (err_minus < atol)
        return is_valid, err_plus, err_minus
