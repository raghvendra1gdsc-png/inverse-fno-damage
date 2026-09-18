"""
Topology-Grounded Structural Graph Definition for OpenSees 2D Moment Frame.

Module: src.graph.structural_graph
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from src.damage_injection import MultiStoryFrame, FrameConfig


@dataclass
class StructuralNode:
    node_id: int              # OpenSees node tag (1..8)
    node_index: int           # 0-indexed tensor index (0..7)
    story: int                # 0: base, 1: floor 1, 2: floor 2, 3: roof
    bay: int                  # 0: left column line, 1: right column line
    x: float                  # x coordinate (meters)
    y: float                  # y coordinate (meters)
    is_fixed: bool            # True for base nodes (1, 2)


@dataclass
class StructuralEdge:
    edge_id: int              # OpenSees element tag (1..9)
    edge_index: int           # 0-indexed tensor index (0..8)
    edge_type: str            # 'column' or 'beam'
    story: int                # 1, 2, 3
    node_i: int               # OpenSees start node tag
    node_j: int               # OpenSees end node tag
    idx_i: int                # 0-indexed start node
    idx_j: int                # 0-indexed end node
    length: float             # member length (m)
    A: float                  # cross-sectional area (m^2)
    I: float                  # moment of inertia (m^4)
    E0: float                 # baseline Young's modulus (Pa)
    lateral_position: int     # -1: left column, +1: right column, 0: beam


class StructuralGraph:
    """
    Constructs the physical structural graph representation directly from
    the OpenSees structural model topology.
    """

    def __init__(self, frame_config: Optional[FrameConfig] = None):
        if frame_config is None:
            frame_config = FrameConfig()
        self.config = frame_config

        self.nodes: Dict[int, StructuralNode] = {}
        self.edges: Dict[int, StructuralEdge] = {}
        self.node_list: List[StructuralNode] = []
        self.edge_list: List[StructuralEdge] = []

        self._build_graph()

    def _build_graph(self):
        cfg = self.config
        num_col_lines = cfg.num_bays + 1  # 2 column lines for 1 bay

        # 1. Build Nodes (8 nodes for 3 stories, 1 bay)
        node_tag = 1
        for story in range(cfg.num_stories + 1):
            y = float(story * cfg.story_height)
            for bay in range(num_col_lines):
                x = float(bay * cfg.bay_width)
                is_fixed = (story == 0)
                node_idx = node_tag - 1
                snode = StructuralNode(
                    node_id=node_tag,
                    node_index=node_idx,
                    story=story,
                    bay=bay,
                    x=x,
                    y=y,
                    is_fixed=is_fixed,
                )
                self.nodes[node_tag] = snode
                self.node_list.append(snode)
                node_tag += 1

        # 2. Build Column Elements (Elements 1..6)
        ele_tag = 1
        for story in range(1, cfg.num_stories + 1):
            for bay in range(num_col_lines):
                # node at (story - 1, bay)
                n_bot = (story - 1) * num_col_lines + bay + 1
                # node at (story, bay)
                n_top = story * num_col_lines + bay + 1
                lat_pos = -1 if bay == 0 else 1
                sedge = StructuralEdge(
                    edge_id=ele_tag,
                    edge_index=ele_tag - 1,
                    edge_type="column",
                    story=story,
                    node_i=n_bot,
                    node_j=n_top,
                    idx_i=n_bot - 1,
                    idx_j=n_top - 1,
                    length=float(cfg.story_height),
                    A=float(cfg.A_col),
                    I=float(cfg.I_col),
                    E0=float(cfg.E),
                    lateral_position=lat_pos,
                )
                self.edges[ele_tag] = sedge
                self.edge_list.append(sedge)
                ele_tag += 1

        # 3. Build Beam Elements (Elements 7..9)
        for story in range(1, cfg.num_stories + 1):
            for bay in range(cfg.num_bays):
                n_left = story * num_col_lines + bay + 1
                n_right = story * num_col_lines + (bay + 1) + 1
                sedge = StructuralEdge(
                    edge_id=ele_tag,
                    edge_index=ele_tag - 1,
                    edge_type="beam",
                    story=story,
                    node_i=n_left,
                    node_j=n_right,
                    idx_i=n_left - 1,
                    idx_j=n_right - 1,
                    length=float(cfg.bay_width),
                    A=float(cfg.A_beam),
                    I=float(cfg.I_beam),
                    E0=float(cfg.E),
                    lateral_position=0,
                )
                self.edges[ele_tag] = sedge
                self.edge_list.append(sedge)
                ele_tag += 1

        self.num_nodes = len(self.node_list)
        self.num_edges = len(self.edge_list)
        assert self.num_nodes == 8, f"Expected 8 structural joints, got {self.num_nodes}"
        assert self.num_edges == 9, f"Expected 9 structural members, got {self.num_edges}"

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Unweighted symmetric adjacency matrix A in {0, 1}^(8 x 8)."""
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for e in self.edge_list:
            i, j = e.idx_i, e.idx_j
            A[i, j] = 1.0
            A[j, i] = 1.0
        return A

    @property
    def normalized_laplacian(self) -> np.ndarray:
        """
        Normalized graph Laplacian:
          L_norm = I - D^(-1/2) A D^(-1/2)
        """
        A = self.adjacency_matrix
        degrees = np.sum(A, axis=1)
        d_inv_sqrt = np.power(np.maximum(degrees, 1e-12), -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L_norm = np.eye(self.num_nodes, dtype=np.float32) - D_inv_sqrt @ A @ D_inv_sqrt
        return L_norm

    @property
    def edge_connectivity_tensor(self) -> torch.Tensor:
        """
        Returns edge connectivity tensor [2, num_edges] with [idx_i, idx_j]
        for each member 1..9.
        """
        conn = [[e.idx_i, e.idx_j] for e in self.edge_list]
        return torch.tensor(conn, dtype=torch.long).t().contiguous()

    @property
    def edge_static_features(self) -> torch.Tensor:
        """
        Returns normalized static structural metadata for each edge [num_edges, 6]:
          0: is_column (1.0) vs is_beam (0.0)
          1: story / 3.0
          2: lateral_position (-1, 0, +1)
          3: normalized length (L / 6.0)
          4: normalized area (A / A_beam)
          5: normalized inertia (I / I_beam)
        """
        feats = []
        max_A = self.config.A_beam
        max_I = self.config.I_beam
        for e in self.edge_list:
            is_col = 1.0 if e.edge_type == "column" else 0.0
            st = e.story / 3.0
            lat = float(e.lateral_position)
            l_norm = e.length / 6.0
            a_norm = e.A / max_A
            i_norm = e.I / max_I
            feats.append([is_col, st, lat, l_norm, a_norm, i_norm])
        return torch.tensor(feats, dtype=torch.float32)
