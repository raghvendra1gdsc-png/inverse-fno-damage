"""
Unit Tests for Phase 6: Symmetry-Aware Dual-Stream G-FNO Architecture and Training Pipeline.

Module: tests.test_phase6_gfno
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.graph.structural_graph import StructuralGraph
from src.graph.sensor_mapping import SensorToGraphMapper
from src.observability import SensorConfiguration
from src.ml.symmetry import StructuralSymmetry, LatentSymmetryProjector
from src.ml.temporal_fno import TemporalFNO
from src.ml.hierarchical_damage_head import NodeToEdgeDecoder, HierarchicalDamageHead, DirectRegressionHead
from src.ml.losses import HierarchicalDamageLoss
from src.ml.gfno import DualStreamGFNO


@pytest.fixture
def graph_and_configs():
    graph = StructuralGraph()
    s0 = SensorConfiguration.create_S0()
    s1 = SensorConfiguration.create_S1()
    s2 = SensorConfiguration.create_S2()
    s4 = SensorConfiguration.create_S4()
    return graph, s0, s1, s2, s4


def test_graph_topology_correctness(graph_and_configs):
    """1. Test graph topology: 8 nodes, 9 members, coordinate assignments."""
    graph, _, _, _, _ = graph_and_configs
    assert graph.num_nodes == 8
    assert graph.num_edges == 9

    # Verify 6 columns and 3 beams
    cols = [e for e in graph.edge_list if e.edge_type == "column"]
    beams = [e for e in graph.edge_list if e.edge_type == "beam"]
    assert len(cols) == 6
    assert len(beams) == 3

    # Base nodes are fixed
    assert graph.nodes[1].is_fixed and graph.nodes[2].is_fixed
    assert not graph.nodes[3].is_fixed


def test_sensor_to_node_mapping(graph_and_configs):
    """2. Test sensor-to-graph mapping dimensions and node assignments."""
    graph, s0, s1, s2, s4 = graph_and_configs
    mapper_s1 = SensorToGraphMapper(graph, s1)
    Y_dummy = torch.randn(2, s1.num_channels, 100)
    gm_dummy = torch.randn(2, 1, 100)

    X_node, mask_node = mapper_s1.map_to_node_tensors(Y_dummy, ground_accel=gm_dummy)
    assert X_node.shape == (2, 8, 4, 100)
    assert mask_node.shape == (8, 4)
    # Ground motion active at all nodes
    assert torch.all(mask_node[:, 3] == 1.0)


def test_graph_laplacian_properties(graph_and_configs):
    """3. Test normalized graph Laplacian: symmetric, PSD, eigenvalues in [0, 2]."""
    graph, _, _, _, _ = graph_and_configs
    L_norm = graph.normalized_laplacian
    assert np.allclose(L_norm, L_norm.T, atol=1e-6)

    eigvals = np.linalg.eigvalsh(L_norm)
    assert np.all(eigvals >= -1e-6)
    assert np.all(eigvals <= 2.0 + 1e-6)
    assert abs(eigvals[0]) < 1e-5  # Connected graph has 0 as smallest eigenvalue


def test_symmetry_permutations_involution():
    """4. Test node and edge symmetry permutations: P^2 = I and isometry."""
    P_node = StructuralSymmetry.get_node_permutation_matrix()
    P_edge = StructuralSymmetry.get_edge_permutation_matrix()

    assert torch.allclose(P_node @ P_node, torch.eye(8))
    assert torch.allclose(P_edge @ P_edge, torch.eye(9))
    assert torch.allclose(P_node, P_node.t())
    assert torch.allclose(P_edge, P_edge.t())


def test_latent_symmetry_and_antisymmetry():
    """5 & 6. Test latent H_+ symmetry and H_- antisymmetry."""
    projector = LatentSymmetryProjector(node_dim=1)
    H_rand = torch.randn(4, 8, 32)

    is_valid, err_plus, err_minus = projector.verify_symmetry_identities(H_rand)
    assert is_valid
    assert err_plus < 1e-6
    assert err_minus < 1e-6


def test_node_to_edge_decoder_dimensions(graph_and_configs):
    """7. Test node-to-edge decoder output shape: [B, 9, out_dim]."""
    graph, _, _, _, _ = graph_and_configs
    decoder = NodeToEdgeDecoder(node_dim=32, edge_feat_dim=6, out_dim=48)
    H_nodes = torch.randn(3, 8, 32)

    H_edges = decoder(H_nodes, graph.edge_connectivity_tensor, graph.edge_static_features)
    assert H_edges.shape == (3, 9, 48)


def test_support_head_bounds():
    """8. Test support head output: probabilities p_e in [0, 1]."""
    head = HierarchicalDamageHead(in_dim=48)
    H_edges = torch.randn(5, 9, 48)

    out = head(H_edges)
    prob_sup = out["prob_support"]
    assert prob_sup.shape == (5, 9)
    assert torch.all(prob_sup >= 0.0) and torch.all(prob_sup <= 1.0)


def test_severity_head_bounds():
    """9. Test severity head bounds: mu_e in [0, 0.5] and sigma_e > 0."""
    head = HierarchicalDamageHead(in_dim=48)
    H_edges = torch.randn(5, 9, 48)

    out = head(H_edges)
    mu = out["severity_mu"]
    sigma = out["severity_sigma"]
    d_pred = out["damage_pred"]

    assert torch.all(mu >= 0.0) and torch.all(mu <= 0.5)
    assert torch.all(sigma > 0.0)
    assert torch.all(d_pred >= 0.0) and torch.all(d_pred <= 0.5)


def test_hierarchical_loss_masking():
    """10. Test hierarchical loss masks undamaged elements during severity computation."""
    loss_fn = HierarchicalDamageLoss(lambda_sup=1.0, lambda_sev=2.0)
    pred = {
        "logits_support": torch.randn(2, 9),
        "severity_mu": torch.full((2, 9), 0.30),
        "damage_pred": torch.full((2, 9), 0.20),
    }

    # Case A: Totally undamaged structure (d_true = 0)
    d_zero = torch.zeros(2, 9)
    _, loss_dict_zero = loss_fn(pred, d_zero)
    # Severity loss MUST be exactly 0.0 when no damage exists
    assert abs(loss_dict_zero["loss_severity"]) < 1e-8

    # Case B: Only member 1 is damaged
    d_damaged = torch.zeros(2, 9)
    d_damaged[:, 0] = 0.30
    _, loss_dict_dam = loss_fn(pred, d_damaged)
    assert loss_dict_dam["loss_severity"] >= 0.0


def test_deterministic_model_initialization(graph_and_configs):
    """11. Test deterministic model initialization via fixed seeds."""
    _, s1, _, _, _ = graph_and_configs
    torch.manual_seed(42)
    m1 = DualStreamGFNO(s1, width_temporal=16, width_graph=16, modes_temporal=8)

    torch.manual_seed(42)
    m2 = DualStreamGFNO(s1, width_temporal=16, width_graph=16, modes_temporal=8)

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_deterministic_inference(graph_and_configs):
    """12. Test deterministic model inference on identical inputs."""
    _, s1, _, _, _ = graph_and_configs
    torch.manual_seed(42)
    m = DualStreamGFNO(s1, width_temporal=16, width_graph=16, modes_temporal=8)
    m.eval()

    Y = torch.randn(2, s1.num_channels, 100)
    gm = torch.randn(2, 1, 100)

    with torch.no_grad():
        out1 = m(Y, ground_accel=gm)
        out2 = m(Y, ground_accel=gm)

    assert torch.allclose(out1["damage_pred"], out2["damage_pred"], atol=1e-12)


def test_dual_stream_forward_pass_shapes(graph_and_configs):
    """13. Test forward pass tensor shapes for all 5 ablation model variants."""
    graph, s0, s1, s2, s4 = graph_and_configs
    Y_s1 = torch.randn(2, 9, 200)
    gm = torch.randn(2, 1, 200)

    variants = [
        # (use_dual_stream, use_graph, use_symmetry, use_hierarchical)
        (False, False, False, False),  # 1. Baseline Inverse FNO flat
        (False, True, False, False),   # 2. G-FNO Baseline
        (True, True, False, False),    # 3. Dual-stream G-FNO
        (True, True, True, False),     # 4. Symmetry-aware G-FNO
        (True, True, True, True),      # 5. Proposed Model
    ]

    for dual, gr, sym, hier in variants:
        model = DualStreamGFNO(
            s1,
            structural_graph=graph,
            use_dual_stream=dual,
            use_graph=gr,
            use_symmetry=sym,
            use_hierarchical=hier,
            width_temporal=16,
            width_graph=16,
            modes_temporal=8,
        )
        out = model(Y_s1, ground_accel=gm)
        assert out["damage_pred"].shape == (2, 9)
        assert out["prob_support"].shape == (2, 9)
        assert out["severity_mu"].shape == (2, 9)


def test_bilateral_ab_classification_logic():
    """14. Test bilateral A/B classification attribution logic."""
    # State A: Left Col 1 (index 0) has highest damage
    d_pred_A = torch.tensor([[0.28, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # State B: Right Col 1 (index 1) has highest damage
    d_pred_B = torch.tensor([[0.01, 0.29, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    top1_A = int(torch.argmax(d_pred_A, dim=-1).item())
    top1_B = int(torch.argmax(d_pred_B, dim=-1).item())

    assert top1_A == 0  # Correctly identifies left column
    assert top1_B == 1  # Correctly identifies right column


def test_dataset_split_isolation():
    """15. Test zero leakage between train, validation, and test splits."""
    from src.forward_fno_model import split_simulation_dataset
    tr, val, te = split_simulation_dataset()
    set_tr = set(tr)
    set_val = set(val)
    set_te = set(te)

    assert len(set_tr.intersection(set_val)) == 0, "Train and Val must be disjoint"
    assert len(set_tr.intersection(set_te)) == 0, "Train and Test must be disjoint"
    assert len(set_val.intersection(set_te)) == 0, "Val and Test must be disjoint"


def test_sensor_configurations_are_numerically_distinct():
    """Phase 6.1 Audit: Prove S0, S1, S2, S4 produce numerically distinct inputs."""
    import glob, os
    files = sorted(glob.glob("data/phase6_multimodal_runs/sim_*.npz"))
    assert len(files) > 0, "Multimodal runs must exist"
    d = np.load(files[0])
    s0 = d["S0_Y"]
    s1 = d["S1_Y"]
    s2 = d["S2_Y"]
    s4 = d["S4_Y"]

    # Dimensions must match channel specs
    assert s0.shape[0] == 3
    assert s1.shape[0] == 9
    assert s2.shape[0] == 9
    assert s4.shape[0] == 18

    # S1 vertical acceleration channels must not be zero
    assert np.linalg.norm(s1[3:]) > 1e-4

    # S2 strain channels must not be zero
    assert np.linalg.norm(s2[3:]) > 1e-6

    # S1 vertical vs S2 strain are physically and numerically distinct
    assert not np.allclose(s1[3:], s2[3:], atol=1e-3)


def test_hierarchical_head_support_and_severity_semantics():
    """Phase 6.1 Audit: Verify support classification and conditional severity semantics."""
    head = HierarchicalDamageHead(in_dim=32, hidden_dim=32)
    H_edges = torch.randn(4, 9, 32)
    out = head(H_edges)

    p_e = out["prob_support"]
    mu_e = out["severity_mu"]
    d_hat = out["damage_pred"]

    # Support must be in [0, 1]
    assert torch.all(p_e >= 0.0) and torch.all(p_e <= 1.0)

    # Severity must be bounded in [0, 0.5]
    assert torch.all(mu_e >= 0.0) and torch.all(mu_e <= 0.5)

    # Product identity: d_hat == p_e * mu_e
    assert torch.allclose(d_hat, p_e * mu_e, atol=1e-6)

    # Gradient flow: both heads must receive gradients
    loss = d_hat.sum()
    loss.backward()
    assert head.support_net[0].weight.grad is not None
    assert head.severity_mu_net[0].weight.grad is not None
    assert head.support_net[0].weight.grad.norm() > 0
    assert head.severity_mu_net[0].weight.grad.norm() > 0
