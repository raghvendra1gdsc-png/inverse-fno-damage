"""
Symmetry-Aware Dual-Stream Graph Fourier Neural Operator (G-FNO).

Module: src.ml.gfno
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph.structural_graph import StructuralGraph
from src.graph.sensor_mapping import SensorToGraphMapper
from src.observability import SensorConfiguration
from src.ml.symmetry import StructuralSymmetry, LatentSymmetryProjector
from src.ml.temporal_fno import TemporalFNO
from src.ml.hierarchical_damage_head import NodeToEdgeDecoder, HierarchicalDamageHead, DirectRegressionHead


class GraphConvolution(nn.Module):
    """
    Normalized Graph Convolution / Message Passing Layer:
      H_out = GELU( H W_self + A_norm H W_neigh )
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=True)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        H : [Batch, 8, in_features]
        A_norm : [8, 8]
        """
        # H W_self: [B, 8, out_features]
        h_self = self.w_self(H)
        # A_norm H: [B, 8, in_features]
        h_agg = torch.matmul(A_norm, H)
        h_neigh = self.w_neigh(h_agg)
        return F.gelu(h_self + h_neigh)


class DualStreamGFNO(nn.Module):
    """
    Symmetry-Aware Dual-Stream Graph Fourier Neural Operator for
    Structural Damage Identification.

    Configurable to instantiate:
      1. Baseline Flat Inverse FNO (use_graph=False, use_symmetry=False, use_hierarchical=False)
      2. Baseline G-FNO (use_dual_stream=False, use_symmetry=False, use_hierarchical=False)
      3. Dual-Stream G-FNO (use_dual_stream=True, use_symmetry=False, use_hierarchical=False)
      4. Symmetry-Aware G-FNO (use_dual_stream=True, use_symmetry=True, use_hierarchical=False)
      5. Proposed Full Model (use_dual_stream=True, use_symmetry=True, use_hierarchical=True)
    """

    def __init__(
        self,
        sensor_config: SensorConfiguration,
        structural_graph: Optional[StructuralGraph] = None,
        use_dual_stream: bool = True,
        use_graph: bool = True,
        use_symmetry: bool = True,
        use_hierarchical: bool = True,
        width_temporal: int = 48,
        modes_temporal: int = 24,
        n_temporal_layers: int = 3,
        width_graph: int = 64,
        n_graph_layers: int = 2,
    ):
        super().__init__()
        if structural_graph is None:
            structural_graph = StructuralGraph()
        self.graph = structural_graph
        self.sensor_config = sensor_config
        self.mapper = SensorToGraphMapper(structural_graph, sensor_config)

        self.use_dual_stream = use_dual_stream
        self.use_graph = use_graph
        self.use_symmetry = use_symmetry
        self.use_hierarchical = use_hierarchical

        # Register static graph buffers
        A_raw = torch.from_numpy(structural_graph.adjacency_matrix).float()
        deg = torch.sum(A_raw, dim=1)
        d_inv_sqrt = torch.pow(torch.clamp(deg, min=1e-6), -0.5)
        D_inv = torch.diag(d_inv_sqrt)
        A_norm = D_inv @ A_raw @ D_inv  # Normalized adjacency with symmetric normalization

        self.register_buffer("A_norm", A_norm)
        self.register_buffer("edge_conn", structural_graph.edge_connectivity_tensor)
        self.register_buffer("edge_static", structural_graph.edge_static_features)

        # ----------------------------------------------------------------------
        # BRANCH A: Global Horizontal Stream (Floor horizontal accels + ag)
        # ----------------------------------------------------------------------
        # Floor 1, 2, Roof horizontal accelerations (3 ch) + ag (1 ch) = 4 channels
        in_channels_global = 4
        self.global_fno = TemporalFNO(
            in_channels=in_channels_global,
            out_width=width_temporal,
            modes=modes_temporal,
            n_layers=n_temporal_layers,
        )
        self.global_pool = nn.Sequential(
            nn.Linear(width_temporal * 2, width_graph),
            nn.GELU(),
        )

        # ----------------------------------------------------------------------
        # BRANCH B: Asymmetric Local Node Stream
        # ----------------------------------------------------------------------
        if self.use_graph:
            # 8 nodes * 4 physical channels = 32 input channels
            in_channels_node = 8 * 4
            self.node_fno = TemporalFNO(
                in_channels=in_channels_node,
                out_width=width_temporal,
                modes=modes_temporal,
                n_layers=n_temporal_layers,
            )
            # Pool temporal features and project to 8 structural joints
            self.node_temporal_pool = nn.Sequential(
                nn.Linear(width_temporal * 2, 8 * width_graph),
                nn.GELU(),
            )

            # Graph propagation layers over structural joints
            self.graph_convs = nn.ModuleList([
                GraphConvolution(width_graph, width_graph)
                for _ in range(n_graph_layers)
            ])

        # ----------------------------------------------------------------------
        # SYMMETRY OPERATOR & LATENT FUSION
        # ----------------------------------------------------------------------
        if self.use_symmetry:
            self.sym_projector = LatentSymmetryProjector(node_dim=1)
            # Fuses H_+ (width_graph) and H_- (width_graph) and Global (width_graph)
            fuse_in_dim = width_graph * 3 if self.use_dual_stream else width_graph * 2
        else:
            fuse_in_dim = width_graph * 2 if self.use_dual_stream else width_graph

        if self.use_graph:
            self.fusion_linear = nn.Sequential(
                nn.Linear(fuse_in_dim, width_graph),
                nn.GELU(),
            )
            # Node -> Edge Decoder
            self.edge_decoder = NodeToEdgeDecoder(
                node_dim=width_graph,
                edge_feat_dim=6,
                hidden_dim=width_graph * 2,
                out_dim=width_graph,
            )
        else:
            # Flat architecture fallback (similar to Baseline Inverse FNO)
            self.flat_decoder = nn.Sequential(
                nn.Linear(width_graph, width_graph),
                nn.GELU(),
                nn.Linear(width_graph, 9 * width_graph),
            )

        # ----------------------------------------------------------------------
        # DAMAGE PREDICTION HEAD
        # ----------------------------------------------------------------------
        if self.use_hierarchical:
            self.head = HierarchicalDamageHead(in_dim=width_graph, hidden_dim=width_graph)
        else:
            self.head = DirectRegressionHead(in_dim=width_graph, hidden_dim=width_graph)

    def forward(
        self,
        Y: torch.Tensor,
        ground_accel: Optional[torch.Tensor] = None,
        Y_global: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        Y : torch.Tensor [Batch, num_channels, Time]
            Raw sensor measurements for active configuration.
        ground_accel : Optional[torch.Tensor] [Batch, 1, Time] or [Batch, Time]
            Ground motion acceleration history.
        Y_global : Optional[torch.Tensor] [Batch, 3, Time]
            Dedicated 3 horizontal floor accelerometers (H_F1, H_F2, H_RF).
            If None, extracted automatically from Y if present.

        Returns
        -------
        Dict with keys:
          - 'damage_pred': [Batch, 9] in [0, 0.5]
          - 'prob_support': [Batch, 9] in [0, 1]
          - 'severity_mu': [Batch, 9] in [0, 0.5]
          - 'severity_sigma': [Batch, 9]
          - 'H_nodes': [Batch, 8, width_graph]
        """
        B, C, T = Y.shape
        device = Y.device

        # Ensure ground_accel is [B, 1, T]
        if ground_accel is not None and ground_accel.dim() == 2:
            ground_accel = ground_accel.unsqueeze(1)

        # 1. Process Branch A (Global Horizontal Stream)
        if Y_global is None:
            # Default to first 3 channels (which in S0..S4 are always horizontal floors)
            Y_global = Y[:, :3, :]

        if ground_accel is not None:
            in_global = torch.cat([Y_global, ground_accel], dim=1)  # [B, 4, T]
        else:
            in_global = torch.cat([Y_global, torch.zeros((B, 1, T), device=device)], dim=1)

        h_global_t = self.global_fno(in_global)  # [B, width_temporal, T]
        # Pool across time (mean + max)
        h_glob_mean = torch.mean(h_global_t, dim=-1)
        h_glob_max = torch.max(h_global_t, dim=-1)[0]
        z_global = self.global_pool(torch.cat([h_glob_mean, h_glob_max], dim=-1))  # [B, width_graph]

        if not self.use_graph:
            # Flat baseline: expand global feature directly to 9 members
            h_edges = self.flat_decoder(z_global).view(B, 9, -1)
            out = self.head(h_edges)
            out["H_nodes"] = z_global.unsqueeze(1).repeat(1, 8, 1)
            return out

        # 2. Process Branch B (Asymmetric Local Node Stream)
        X_node, _ = self.mapper.map_to_node_tensors(Y, ground_accel=ground_accel)  # [B, 8, 4, T]
        # Reshape to joint node channels: [B, 32, T]
        X_node_joint = X_node.view(B, 8 * 4, T)
        h_node_t = self.node_fno(X_node_joint)  # [B, width_temporal, T]

        h_node_mean = torch.mean(h_node_t, dim=-1)
        h_node_max = torch.max(h_node_t, dim=-1)[0]
        # Project to 8 joints: [B, 8, width_graph]
        H_nodes = self.node_temporal_pool(torch.cat([h_node_mean, h_node_max], dim=-1)).view(B, 8, -1)

        # Graph message passing over structural joints
        for conv in self.graph_convs:
            H_nodes = conv(H_nodes, self.A_norm)  # [B, 8, width_graph]

        # 3. Symmetry Decomposition
        if self.use_symmetry:
            H_plus, H_minus = self.sym_projector(H_nodes)
            if self.use_dual_stream:
                # Broadcast global stream to all 8 nodes
                z_glob_expand = z_global.unsqueeze(1).expand(-1, 8, -1)  # [B, 8, width_graph]
                fused = torch.cat([H_plus, H_minus, z_glob_expand], dim=-1)
            else:
                fused = torch.cat([H_plus, H_minus], dim=-1)
        else:
            if self.use_dual_stream:
                z_glob_expand = z_global.unsqueeze(1).expand(-1, 8, -1)
                fused = torch.cat([H_nodes, z_glob_expand], dim=-1)
            else:
                fused = H_nodes

        H_fused = self.fusion_linear(fused)  # [B, 8, width_graph]

        # 4. Node -> Edge Decoding for all 9 members
        H_edges = self.edge_decoder(H_fused, self.edge_conn, self.edge_static)  # [B, 9, width_graph]

        # 5. Damage Prediction Head
        out = self.head(H_edges)
        out["H_nodes"] = H_fused
        return out
