"""
Physical Structural Conditioning Features for Graph Neural Operators.

Module: src.structure_features
Author: SeismoFNO Research Team
Context: Phase 7 Cross-Structure Generalization — Physical Conditioning Features
"""

from typing import Dict, Any, Tuple
import numpy as np
from src.damage_injection import FrameConfig, MultiStoryFrame
from src.structure_variants import get_structure_config, REF_FLOOR_MASS, REF_I_COL, REF_A_COL, REF_STORY_HEIGHT, REF_BAY_WIDTH, REF_E


# Normalization Reference Constants (Category A: Fixed Physical Scales)
REF_L0: float = REF_BAY_WIDTH       # 6.0 m
REF_H0: float = 3.0 * REF_STORY_HEIGHT # 9.0 m (3-story height)
REF_A0: float = REF_A_COL           # 0.010 m^2
REF_I0: float = REF_I_COL           # 1.0e-4 m^4
REF_M0: float = REF_FLOOR_MASS / 2.0 # 5000.0 kg (mass per node)
REF_E0: float = REF_E               # 2.0e11 Pa


def extract_structural_graph_features(
    frame: MultiStoryFrame,
) -> Dict[str, np.ndarray]:
    """
    Extracts continuous dimensionless node, edge, and adjacency representations
    for any (N_stories, 1)-bay frame.

    Returns
    -------
    Dict containing:
      - 'node_features': (num_nodes, 4) array:
          [x / L0, y / H0, is_ground_float, m_v / M0]
      - 'edge_features': (num_elements, 6) array:
          [L_e / h0, A_e / A0, I_e / I0, E_e / E0, cos(theta), sin(theta)]
      - 'edge_connectivity': (num_elements, 2) int array [node_i_idx, node_j_idx] (0-indexed)
      - 'adjacency': (num_nodes, num_nodes) float array
      - 'global_scalars': (2,) float array [mu_m, mu_k]
    """
    cfg = frame.config
    if not getattr(frame, "_is_built", False):
        frame.build_model()  # Build pristine, unperturbed baseline model (all d_e = 0.0)

    num_nodes = (cfg.num_stories + 1) * 2
    num_elements = frame.num_elements

    # 1. Node Features (num_nodes, 4)
    # Joints are indexed 1..num_nodes in frame.nodes
    node_feats = np.zeros((num_nodes, 4), dtype=np.float32)
    for idx in range(num_nodes):
        node_id = idx + 1
        n_info = frame.nodes[node_id]
        x_norm = float(n_info.x / REF_L0)
        y_norm = float(n_info.y / REF_H0)
        is_ground = 1.0 if n_info.is_fixed else 0.0
        m_norm = float(n_info.mass_x / REF_M0) if not n_info.is_fixed else 0.0
        node_feats[idx] = [x_norm, y_norm, is_ground, m_norm]

    # 2. Edge Features (num_elements, 6) and Connectivity (num_elements, 2)
    edge_feats = np.zeros((num_elements, 6), dtype=np.float32)
    edge_conn = np.zeros((num_elements, 2), dtype=np.int64)
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for idx in range(num_elements):
        ele_id = idx + 1
        e_info = frame.elements[ele_id]
        ni_idx = e_info.node_i - 1
        nj_idx = e_info.node_j - 1
        edge_conn[idx] = [ni_idx, nj_idx]

        # Adjacency (symmetric)
        adj[ni_idx, nj_idx] = 1.0
        adj[nj_idx, ni_idx] = 1.0

        dx = frame.nodes[e_info.node_j].x - frame.nodes[e_info.node_i].x
        dy = frame.nodes[e_info.node_j].y - frame.nodes[e_info.node_i].y
        L = np.sqrt(dx**2 + dy**2)
        cos_th = float(dx / L) if L > 1e-6 else 0.0
        sin_th = float(dy / L) if L > 1e-6 else 1.0

        L_norm = float(e_info.length / REF_STORY_HEIGHT)
        A_norm = float(e_info.A / REF_A0)
        I_norm = float(e_info.I / REF_I0)
        # CRITICAL REPAIR: E_norm must be strictly nominal undegraded Young's modulus (cfg.E).
        # Must NEVER reference e_info.E (which is modified during damage injection).
        E_norm = float(cfg.E / REF_E0)

        edge_feats[idx] = [L_norm, A_norm, I_norm, E_norm, cos_th, sin_th]

    # Add self-loops to adjacency
    for i in range(num_nodes):
        adj[i, i] = 1.0

    # Global scalars
    mu_m = float(cfg.floor_mass / REF_FLOOR_MASS)
    mu_k = float(cfg.I_col / REF_I_COL)
    global_scalars = np.array([mu_m, mu_k], dtype=np.float32)

    return {
        "node_features": node_feats,
        "edge_features": edge_feats,
        "edge_connectivity": edge_conn,
        "adjacency": adj,
        "global_scalars": global_scalars,
    }
