"""
Phase 7.2 Training, Validation, and Held-Out Cross-Structure Evaluation Engine.

Script: scripts/phase7_train_and_evaluate.py
Author: SeismoFNO Research Team
Context: Phase 7.2 Complete Execution Pipeline (P1, P2, Levels 1, 2A, 2B, 3)
"""

import os
import sys
import glob
import json
import time
import math
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath("."))

from src.ml.phase7_dataset import Phase7SimulationDataset, Phase7BilateralPairDataset
from src.ml.topology_gfno import TopologyDualStreamGFNO
from src.ml.bilateral_pair_loss import BilateralDirectionalLoss, BilateralMarginSeparationLoss


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes exact Clopper-Pearson 95% binomial confidence interval."""
    if n == 0:
        return 0.0, 0.0
    from scipy.stats import beta
    lower = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lower, upper


class Phase7Loss(nn.Module):
    """Hierarchical Base Loss for Damage Inversion."""
    def __init__(self, lambda_sev: float = 2.0, lambda_exp: float = 5.0):
        super().__init__()
        self.lambda_sev = lambda_sev
        self.lambda_exp = lambda_exp

    def forward(self, out: Dict[str, torch.Tensor], d_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        d_pred = out["damage"]
        p_supp = out["support_prob"]
        mu_sev = out["severity_cond"]

        supp_target = (d_true > 1e-4).float()
        loss_supp = F.binary_cross_entropy(torch.clamp(p_supp, 1e-6, 1.0 - 1e-6), supp_target)

        # Severity MSE only on damaged elements
        mask = supp_target > 0.5
        if mask.sum() > 0:
            loss_sev = F.mse_loss(mu_sev[mask], d_true[mask])
        else:
            loss_sev = torch.tensor(0.0, device=d_true.device)

        loss_exp = F.mse_loss(d_pred, d_true)
        total_loss = loss_supp + self.lambda_sev * loss_sev + self.lambda_exp * loss_exp

        return total_loss, {
            "loss_total": float(total_loss.item()),
            "loss_supp": float(loss_supp.item()),
            "loss_sev": float(loss_sev.item()),
            "loss_exp": float(loss_exp.item()),
        }


def collate_single(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collates uniform topology batches."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        val0 = batch[0][k]
        if isinstance(val0, torch.Tensor):
            collated[k] = torch.stack([b[k] for b in batch], dim=0)
        else:
            collated[k] = [b[k] for b in batch]
    return collated


def train_single_epoch(
    model: nn.Module,
    std_loader: DataLoader,
    pair_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    base_loss_fn: Phase7Loss,
    dir_loss_fn: Optional[BilateralDirectionalLoss],
    margin_loss_fn: Optional[BilateralMarginSeparationLoss],
    device: str = "cpu",
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    # 1. Standard batches
    for b in std_loader:
        optimizer.zero_grad()
        out = model(
            node_signals=b["node_signals"].to(device),
            global_signals=b["global_signals"].to(device),
            node_features=b["node_features"].to(device),
            edge_features=b["edge_features"].to(device),
            edge_connectivity=b["edge_connectivity"][0].to(device),
            adjacency=b["adjacency"][0].to(device),
            global_scalars=b["global_scalars"].to(device),
            node_perm=b["node_perm"][0].to(device),
            edge_perm=b["edge_perm"][0].to(device),
        )
        loss, _ = base_loss_fn(out, b["damage"].to(device))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        steps += 1

    # 2. Pairwise batches (for PROPOSED model)
    if pair_loader is not None and dir_loss_fn is not None:
        for p in pair_loader:
            optimizer.zero_grad()
            bA = p["A"]
            bB = p["B"]
            v_AB = p["v_AB"].to(device)

            def prep_inputs(b: Dict[str, Any]):
                ns = b["node_signals"].to(device)
                if ns.dim() == 3: ns = ns.unsqueeze(0)
                gs = b["global_signals"].to(device)
                if gs.dim() == 2: gs = gs.unsqueeze(0)
                nf = b["node_features"].to(device)
                if nf.dim() == 2: nf = nf.unsqueeze(0)
                ef = b["edge_features"].to(device)
                if ef.dim() == 2: ef = ef.unsqueeze(0)
                ec = b["edge_connectivity"]
                if ec.dim() == 3: ec = ec[0]
                ec = ec.to(device)
                adj = b["adjacency"]
                if adj.dim() == 3: adj = adj[0]
                adj = adj.to(device)
                sc = b["global_scalars"].to(device)
                if sc.dim() == 1: sc = sc.unsqueeze(0)
                np_p = b["node_perm"]
                if np_p.dim() == 2: np_p = np_p[0]
                np_p = np_p.to(device)
                ep_p = b["edge_perm"]
                if ep_p.dim() == 2: ep_p = ep_p[0]
                ep_p = ep_p.to(device)
                d = b["damage"].to(device)
                if d.dim() == 1: d = d.unsqueeze(0)
                return ns, gs, nf, ef, ec, adj, sc, np_p, ep_p, d

            ns_A, gs_A, nf_A, ef_A, ec_A, adj_A, sc_A, np_A, ep_A, dA_true = prep_inputs(bA)
            ns_B, gs_B, nf_B, ef_B, ec_B, adj_B, sc_B, np_B, ep_B, dB_true = prep_inputs(bB)

            outA = model(
                node_signals=ns_A,
                global_signals=gs_A,
                node_features=nf_A,
                edge_features=ef_A,
                edge_connectivity=ec_A,
                adjacency=adj_A,
                global_scalars=sc_A,
                node_perm=np_A,
                edge_perm=ep_A,
            )
            outB = model(
                node_signals=ns_B,
                global_signals=gs_B,
                node_features=nf_B,
                edge_features=ef_B,
                edge_connectivity=ec_B,
                adjacency=adj_B,
                global_scalars=sc_B,
                node_perm=np_B,
                edge_perm=ep_B,
            )

            l_base_A, _ = base_loss_fn(outA, dA_true)
            l_base_B, _ = base_loss_fn(outB, dB_true)

            l_dir = dir_loss_fn(outA["damage"], outB["damage"], v_AB)
            l_margin = margin_loss_fn(outA["damage"], outB["damage"], v_AB)

            l_pair = 0.5 * (l_base_A + l_base_B) + 1.0 * l_dir + 2.0 * l_margin
            l_pair.backward()
            optimizer.step()
            total_loss += float(l_pair.item())
            steps += 1

    return total_loss / max(steps, 1)


def evaluate_model_on_pairs(
    model: nn.Module,
    pairs: List[Tuple[str, str]],
    sensor_modality: str,
    device: str = "cpu",
    tau_star: float = 0.10,
) -> Dict[str, Any]:
    """
    Evaluates bilateral pairs on a specific structure/condition.
    Returns bilateral accuracy, cosine alignment, predicted separation, and event-level metrics.
    """
    model.eval()
    pair_dataset = Phase7BilateralPairDataset([f for p in pairs for f in p], sensor_modality=sensor_modality)

    correct_pairs = 0
    cos_sims = []
    separations = []
    top1_matches = 0
    severity_errors = []
    events = []

    # Confusion matrix: rows true [A, B], cols pred [A, B]
    conf_mat = np.zeros((2, 2), dtype=int)

    with torch.no_grad():
        for idx in range(len(pair_dataset)):
            p = pair_dataset[idx]
            bA = p["A"]
            bB = p["B"]
            v_AB = p["v_AB"].numpy()

            outA = model(
                node_signals=bA["node_signals"].unsqueeze(0).to(device),
                global_signals=bA["global_signals"].unsqueeze(0).to(device),
                node_features=bA["node_features"].unsqueeze(0).to(device),
                edge_features=bA["edge_features"].unsqueeze(0).to(device),
                edge_connectivity=bA["edge_connectivity"].to(device),
                adjacency=bA["adjacency"].to(device),
                global_scalars=bA["global_scalars"].unsqueeze(0).to(device),
                node_perm=bA["node_perm"].to(device),
                edge_perm=bA["edge_perm"].to(device),
            )
            outB = model(
                node_signals=bB["node_signals"].unsqueeze(0).to(device),
                global_signals=bB["global_signals"].unsqueeze(0).to(device),
                node_features=bB["node_features"].unsqueeze(0).to(device),
                edge_features=bB["edge_features"].unsqueeze(0).to(device),
                edge_connectivity=bB["edge_connectivity"].to(device),
                adjacency=bB["adjacency"].to(device),
                global_scalars=bB["global_scalars"].unsqueeze(0).to(device),
                node_perm=bB["node_perm"].to(device),
                edge_perm=bB["edge_perm"].to(device),
            )

            d_hat_A = outA["damage"].squeeze(0).cpu().numpy()
            d_hat_B = outB["damage"].squeeze(0).cpu().numpy()

            dA_true = bA["damage"].numpy()
            dB_true = bB["damage"].numpy()

            # Top-1 Localization: Element with highest predicted damage
            pred_ele_A = int(np.argmax(d_hat_A))
            pred_ele_B = int(np.argmax(d_hat_B))

            # Bilateral Attribution logic (Phase 6.2 Standard):
            # Canonical State A has damage on Left Col 1 (index 0)
            # Canonical State B has damage on Right Col 1 (index 1)
            # If unseparated (p[0] == p[1]), attribution falls back to chance (50.0%)
            pred_class_A = 0 if d_hat_A[0] >= d_hat_A[1] else 1
            pred_class_B = 1 if d_hat_B[1] > d_hat_B[0] else 0

            conf_mat[0, pred_class_A] += 1
            conf_mat[1, pred_class_B] += 1

            pair_correct = (pred_class_A == 0 and pred_class_B == 1)
            if pair_correct:
                correct_pairs += 1

            # Directional vector analysis
            delta_d = d_hat_A - d_hat_B
            sep = float(np.linalg.norm(delta_d))
            separations.append(sep)

            if sep > 1e-6:
                cos = float(np.dot(delta_d, v_AB) / (sep * np.linalg.norm(v_AB)))
            else:
                cos = 0.0
            cos_sims.append(cos)

            # Top-1 accuracy (Col 1 is 0, Col 2 is 1)
            if pred_ele_A == 0: top1_matches += 1
            if pred_ele_B == 1: top1_matches += 1

            # Severity MAE on damaged element (0.30 true)
            sev_err_A = abs(d_hat_A[0] - 0.30)
            sev_err_B = abs(d_hat_B[1] - 0.30)
            severity_errors.extend([sev_err_A, sev_err_B])

            events.append({
                "gm_record": p["gm_record"],
                "struct_id": p["struct_id"],
                "pair_id": p["pair_id"],
                "pred_class_A": pred_class_A,
                "pred_class_B": pred_class_B,
                "pair_correct": pair_correct,
                "separation": sep,
                "cosine_v_AB": cos,
            })

    N_pairs = len(pair_dataset)
    N_evals = int(np.sum(conf_mat))
    correct_evals = int(conf_mat[0, 0] + conf_mat[1, 1])
    accuracy_pct = (correct_evals / max(N_evals, 1)) * 100.0
    top1_pct = (top1_matches / max(2 * N_pairs, 1)) * 100.0
    mean_sep = float(np.mean(separations)) if separations else 0.0
    mean_cos = float(np.mean(cos_sims)) if cos_sims else 0.0
    mean_sev_mae = float(np.mean(severity_errors)) if severity_errors else 0.0

    ci_low, ci_high = clopper_pearson_ci(correct_evals, N_evals)

    return {
        "num_pairs": N_pairs,
        "num_evaluations": N_evals,
        "correct_pairs": correct_pairs,
        "bilateral_accuracy_pct": accuracy_pct,
        "ci_95_low_pct": ci_low * 100.0,
        "ci_95_high_pct": ci_high * 100.0,
        "confusion_matrix": conf_mat.tolist(),
        "top1_accuracy_pct": top1_pct,
        "mean_predicted_separation": mean_sep,
        "mean_cosine_v_AB": mean_cos,
        "severity_mae": mean_sev_mae,
        "events": events,
    }
