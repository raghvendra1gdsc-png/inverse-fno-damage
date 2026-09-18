"""
Physical Sensor-to-Graph Mapping for Structural Damage Identification.

Module: src.graph.sensor_mapping
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn

from src.observability import SensorConfiguration, SensorChannelSpec
from src.graph.structural_graph import StructuralGraph


class SensorToGraphMapper:
    """
    Maps physical sensor observations Y(t) in R^(B x C x T) to the 8 structural
    graph joints and 9 structural members, retaining physical metadata and
    active measurement indicators.
    """

    def __init__(self, graph: StructuralGraph, sensor_config: SensorConfiguration):
        self.graph = graph
        self.sensor_config = sensor_config
        self.channels = sensor_config.channels
        self.num_channels = len(self.channels)
        self.num_nodes = graph.num_nodes  # 8
        self.num_edges = graph.num_edges  # 9

        self._build_mapping()

    def _build_mapping(self):
        """
        Builds explicit node-channel and edge-channel indexing tables.
        """
        # Node channels: for each node (0..7), list of (ch_idx, dof, type)
        self.node_to_channels: Dict[int, List[Tuple[int, int, str]]] = {i: [] for i in range(self.num_nodes)}
        # Edge channels: for each member (0..8), list of (ch_idx, type)
        self.edge_to_channels: Dict[int, List[Tuple[int, str]]] = {e: [] for e in range(self.num_edges)}

        for ch_idx, ch in enumerate(self.channels):
            # 1. Node acceleration channels
            if ch.node_tag is not None:
                node_idx = ch.node_tag - 1  # 0-indexed
                if 0 <= node_idx < self.num_nodes:
                    self.node_to_channels[node_idx].append((ch_idx, ch.dof, ch.channel_type))

            # 2. Member strain channels
            elif ch.ele_id is not None:
                edge_idx = ch.ele_id - 1
                if 0 <= edge_idx < self.num_edges:
                    self.edge_to_channels[edge_idx].append((ch_idx, ch.channel_type))
                    # Also associate with its boundary nodes
                    sedge = self.graph.edge_list[edge_idx]
                    self.node_to_channels[sedge.idx_i].append((ch_idx, 0, "strain_incident"))
                    self.node_to_channels[sedge.idx_j].append((ch_idx, 0, "strain_incident"))

            # 3. Story rocking observables
            elif ch.channel_type == "rocking":
                story = ch.story
                # Left node tag: 2*story + 1; Right node tag: 2*story + 2
                # In 0-index: left = 2*story, right = 2*story + 1
                n_left = 2 * story
                n_right = 2 * story + 1
                if n_left < self.num_nodes:
                    self.node_to_channels[n_left].append((ch_idx, 2, "rocking_left"))
                if n_right < self.num_nodes:
                    self.node_to_channels[n_right].append((ch_idx, 2, "rocking_right"))

    def map_to_node_tensors(
        self,
        Y: torch.Tensor,
        ground_accel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects raw multichannel sensor time histories Y [B, C, T] onto
        the 8 structural nodes.

        Returns
        -------
        X_node : torch.Tensor [B, 8, node_features, T]
            Node time-history feature tensor with 4 standardized physical channels:
              Channel 0: Horizontal acceleration a_x(t)
              Channel 1: Vertical acceleration a_y(t)
              Channel 2: Incident axial strain eps(t)
              Channel 3: Ground motion acceleration a_g(t)
        mask_node : torch.Tensor [8, 4]
            Binary indicator (1.0 = sensor physically present, 0.0 = absent/padded).
        """
        B, C, T = Y.shape
        device = Y.device
        num_node_channels = 4  # a_x, a_y, eps, a_g

        X_node = torch.zeros((B, self.num_nodes, num_node_channels, T), device=device, dtype=Y.dtype)
        mask_node = torch.zeros((self.num_nodes, num_node_channels), device=device, dtype=torch.float32)

        # Ground motion broadcast to all nodes if provided
        if ground_accel is not None:
            # ground_accel shape: [B, 1, T] or [B, T]
            if ground_accel.dim() == 2:
                gm_expanded = ground_accel.unsqueeze(1).unsqueeze(1).expand(B, self.num_nodes, 1, T)
            else:
                gm_expanded = ground_accel.unsqueeze(1).expand(B, self.num_nodes, 1, T)
            X_node[:, :, 3:4, :] = gm_expanded
            mask_node[:, 3] = 1.0

        for n_idx in range(self.num_nodes):
            for ch_idx, dof, ctype in self.node_to_channels[n_idx]:
                if ctype == "accel_horiz" or (dof == 1 and "accel" in ctype):
                    X_node[:, n_idx, 0, :] = Y[:, ch_idx, :]
                    mask_node[n_idx, 0] = 1.0
                elif ctype == "accel_vert" or (dof == 2 and "accel" in ctype):
                    X_node[:, n_idx, 1, :] = Y[:, ch_idx, :]
                    mask_node[n_idx, 1] = 1.0
                elif "strain" in ctype:
                    X_node[:, n_idx, 2, :] = Y[:, ch_idx, :]
                    mask_node[n_idx, 2] = 1.0
                elif "rocking" in ctype:
                    # Treat rocking as differential vertical signal
                    sign = -1.0 if "left" in ctype else 1.0
                    X_node[:, n_idx, 1, :] += sign * Y[:, ch_idx, :]
                    mask_node[n_idx, 1] = 1.0

        return X_node, mask_node
