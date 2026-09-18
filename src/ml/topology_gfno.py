"""
Topology-Agnostic Physics-Conditioned Dual-Stream G-FNO Architecture.

Module: src.ml.topology_gfno
Author: SeismoFNO Research Team
Context: Phase 7 Cross-Structure Generalization — Unified Model Hierarchy (B1..B5, PROPOSED)
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.temporal_fno import TemporalFNO
from src.ml.symmetry import LatentSymmetryProjector


class GraphConvolution(nn.Module):
    """
    Symmetric normalized graph convolution:
        H^(l+1) = GELU( H^(l) W_self + A_norm H^(l) W_neigh )
    Operates on arbitrary node count N_v.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : [Batch, N_v, In_features]
        a_norm : [N_v, N_v] or [Batch, N_v, N_v]
        """
        h_self = self.w_self(h)
        h_neigh = self.w_neigh(h)
        if a_norm.dim() == 2:
            agg = torch.matmul(a_norm.unsqueeze(0), h_neigh)
        else:
            agg = torch.matmul(a_norm, h_neigh)
        return self.act(h_self + agg + self.bias)


class DynamicNodeDecoder(nn.Module):
    """
    Decodes edge representations from pairwise connected node states and member attributes:
        h_e = MLP([h_u || h_v || |h_u - h_v| || h_u * h_v || x_e])
    """

    def __init__(self, node_dim: int, edge_feat_dim: int, out_dim: int):
        super().__init__()
        in_dim = 4 * node_dim + edge_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_connectivity: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        node_states : [Batch, N_v, D_node]
        edge_connectivity : [N_e, 2] (node_i, node_j indices)
        edge_features : [Batch, N_e, D_edge] or [N_e, D_edge]
        """
        B = node_states.shape[0]
        u_idx = edge_connectivity[:, 0]
        v_idx = edge_connectivity[:, 1]

        h_u = node_states[:, u_idx, :]  # [B, N_e, D_node]
        h_v = node_states[:, v_idx, :]  # [B, N_e, D_node]

        diff = torch.abs(h_u - h_v)
        prod = h_u * h_v

        if edge_features.dim() == 2:
            ef = edge_features.unsqueeze(0).expand(B, -1, -1)
        else:
            ef = edge_features

        combined = torch.cat([h_u, h_v, diff, prod, ef], dim=-1)
        return self.mlp(combined)  # [B, N_e, D_out]


class TopologyDualStreamGFNO(nn.Module):
    """
    Topology-Agnostic Symmetry-Aware Dual-Stream Graph Fourier Neural Operator.
    Implements the complete Phase 7 baseline hierarchy (B1..B5, PROPOSED).
    """

    def __init__(
        self,
        model_condition: str = "PROPOSED",
        width_temporal: int = 64,
        modes_temporal: int = 16,
        n_temporal_layers: int = 4,
        width_graph: int = 64,
        n_graph_layers: int = 3,
    ):
        super().__init__()
        self.model_condition = model_condition.upper()

        # Conditioning feature dimensions based on model hierarchy:
        # B1: Flat FNO baseline
        # B2: Unconditioned (0 node feats, 0 edge feats)
        # B3: Scalar-Conditioned (2 global scalars broadcast)
        # B4: Geometry-Conditioned (2 coords for node, 2 geom for edge)
        # B5 & PROPOSED: Full Physics (4 node feats, 6 edge feats)
        if self.model_condition in ["B2", "B1"]:
            self.d_node_feat = 0
            self.d_edge_feat = 0
        elif self.model_condition == "B3":
            self.d_node_feat = 2  # mu_m, mu_k
            self.d_edge_feat = 0
        elif self.model_condition == "B4":
            self.d_node_feat = 2  # x, y
            self.d_edge_feat = 2  # L_e, theta
        else:  # B5 or PROPOSED
            self.d_node_feat = 4  # x, y, is_ground, m_v
            self.d_edge_feat = 6  # L, A, I, E, cos, sin

        self.use_graph = (self.model_condition != "B1")
        self.use_symmetry = (self.model_condition == "PROPOSED")

        # 1. BRANCH A: Global Invariant Temporal FNO (Floor Horizontal Accelerations + ag)
        # Operates on up to 5 channels (4 stories + ground motion)
        self.global_fno = TemporalFNO(
            in_channels=5,
            out_width=width_temporal,
            modes=modes_temporal,
            n_layers=n_temporal_layers,
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(width_temporal * 2, width_graph),
            nn.GELU(),
        )

        # 2. BRANCH B: Shared Node Temporal FNO (Processes each joint's 4 physical channels)
        # 4 channels: ax, ay, strain, ag (independent of N_v!)
        if self.use_graph:
            self.node_fno = TemporalFNO(
                in_channels=4,
                out_width=width_temporal,
                modes=modes_temporal,
                n_layers=n_temporal_layers,
            )
            node_in_dim = width_temporal * 2 + self.d_node_feat
            self.node_proj = nn.Sequential(
                nn.Linear(node_in_dim, width_graph),
                nn.GELU(),
            )

            # Graph Convolutions
            self.convs = nn.ModuleList([
                GraphConvolution(width_graph, width_graph)
                for _ in range(n_graph_layers)
            ])

            # Edge Decoder
            self.edge_decoder = DynamicNodeDecoder(
                node_dim=width_graph,
                edge_feat_dim=self.d_edge_feat,
                out_dim=width_graph,
            )

        # 3. Flat FNO Readout for B1
        if self.model_condition == "B1":
            self.flat_readout = nn.Sequential(
                nn.Linear(width_graph, 128),
                nn.GELU(),
                nn.Linear(128, 32),
            )

        # 4. Hierarchical Dual Heads (Edge-level damage)
        self.support_head = nn.Sequential(
            nn.Linear(width_graph, width_graph // 2),
            nn.GELU(),
            nn.Linear(width_graph // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(width_graph, width_graph // 2),
            nn.GELU(),
            nn.Linear(width_graph // 2, 1),
        )

    def forward(
        self,
        node_signals: torch.Tensor,
        global_signals: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_connectivity: torch.Tensor,
        adjacency: torch.Tensor,
        global_scalars: Optional[torch.Tensor] = None,
        node_perm: Optional[torch.Tensor] = None,
        edge_perm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        node_signals : [Batch, N_v, 4, T]
        global_signals : [Batch, C_global, T] (padded/interpolated to 5 channels)
        node_features : [Batch, N_v, 4]
        edge_features : [Batch, N_e, 6]
        edge_connectivity : [N_e, 2]
        adjacency : [N_v, N_v]
        global_scalars : Optional [Batch, 2]
        node_perm : Optional [N_v]
        edge_perm : Optional [N_e]
        """
        B, N_v, C_n, T = node_signals.shape
        N_e = edge_connectivity.shape[0]

        # 1. Branch A: Global Temporal FNO
        # Ensure 5 channels
        if global_signals.shape[1] < 5:
            pad = torch.zeros(B, 5 - global_signals.shape[1], T, device=global_signals.device)
            glob_in = torch.cat([global_signals, pad], dim=1)
        else:
            glob_in = global_signals[:, :5, :]

        h_glob_t = self.global_fno(glob_in)
        # Pool along time: mean + max
        h_glob_pool = torch.cat([torch.mean(h_glob_t, dim=-1), torch.max(h_glob_t, dim=-1)[0]], dim=-1)
        h_glob = self.global_mlp(h_glob_pool)  # [B, width_graph]

        if self.model_condition == "B1":
            # Flat baseline: project global representation to N_e damages
            flat_out = self.flat_readout(h_glob)  # [B, 32]
            d_pred = torch.sigmoid(flat_out[:, :N_e]) * 0.50
            return {
                "damage": d_pred,
                "support_prob": torch.clamp(d_pred / 0.30, 0.0, 1.0),
                "severity_cond": d_pred,
                "edge_repr": h_glob.unsqueeze(1).expand(-1, N_e, -1),
            }

        # 2. Branch B: Node Temporal FNO across all joints
        node_in = node_signals.view(B * N_v, C_n, T)
        h_node_t = self.node_fno(node_in)
        h_node_pool = torch.cat([torch.mean(h_node_t, dim=-1), torch.max(h_node_t, dim=-1)[0]], dim=-1)
        h_node = h_node_pool.view(B, N_v, -1)  # [B, N_v, 2 * width_temporal]

        # Concatenate Physical Conditioning Features
        if self.model_condition == "B3" and global_scalars is not None:
            sc_expanded = global_scalars.unsqueeze(1).expand(-1, N_v, -1)
            h_node_fused = torch.cat([h_node, sc_expanded], dim=-1)
        elif self.model_condition == "B4":
            h_node_fused = torch.cat([h_node, node_features[:, :, :2]], dim=-1)
        elif self.model_condition in ["B5", "PROPOSED"]:
            h_node_fused = torch.cat([h_node, node_features], dim=-1)
        else:  # B2
            h_node_fused = h_node

        h_v = self.node_proj(h_node_fused)  # [B, N_v, width_graph]

        # Add global context from Branch A
        h_v = h_v + h_glob.unsqueeze(1)

        # Symmetric normalized adjacency
        deg = torch.sum(adjacency, dim=-1)
        d_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        a_norm = torch.diag(d_inv_sqrt) @ adjacency @ torch.diag(d_inv_sqrt)

        # Graph Convolutions
        for conv in self.convs:
            h_v = conv(h_v, a_norm)

        # Symmetry Projection on Node States (PROPOSED)
        if self.use_symmetry and node_perm is not None:
            h_v_perm = h_v[:, node_perm, :]
            h_sym = 0.5 * (h_v + h_v_perm)
            h_anti = 0.5 * (h_v - h_v_perm)
            h_v = h_sym + h_anti

        # Decode Edge States
        if self.model_condition == "B4":
            ef = edge_features[:, :, [0, 4]]  # L_norm, cos_theta
        elif self.model_condition in ["B5", "PROPOSED"]:
            ef = edge_features
        else:
            ef = torch.zeros(B, N_e, 0, device=node_signals.device)

        h_e = self.edge_decoder(h_v, edge_connectivity, ef)  # [B, N_e, width_graph]

        # Symmetry Projection on Edges (PROPOSED)
        if self.use_symmetry and edge_perm is not None:
            h_e_perm = h_e[:, edge_perm, :]
            h_e_sym = 0.5 * (h_e + h_e_perm)
            h_e_anti = 0.5 * (h_e - h_e_perm)
            h_e = h_e_sym + h_e_anti

        # Hierarchical Readout
        s_logits = self.support_head(h_e).squeeze(-1)  # [B, N_e]
        m_logits = self.severity_head(h_e).squeeze(-1)  # [B, N_e]

        p_support = torch.sigmoid(s_logits)
        mu_severity = 0.50 * torch.sigmoid(m_logits)
        d_pred = p_support * mu_severity

        return {
            "damage": d_pred,
            "support_prob": p_support,
            "severity_cond": mu_severity,
            "edge_repr": h_e,
        }
