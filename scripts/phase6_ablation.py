#!/usr/bin/env python3
"""
Phase 6 Controlled Ablation Matrix Runner.

Script: scripts/phase6_ablation.py
Context: Phase 6 - Symmetry-Aware Dual-Stream G-FNO
Author: Inverse FNO Project Team

Executes systematic ablation across 5 model architectures and 4 sensor configurations
(S0, S1, S2, S4) on validation data. Separates information gained from physical sensors
from information gained from architecture.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.observability import SensorConfiguration
from src.graph.structural_graph import StructuralGraph
from src.ml.gfno import DualStreamGFNO
from src.ml.dataset import MultimodalDamageDataset
from src.ml.losses import HierarchicalDamageLoss
from src.forward_fno_model import split_simulation_dataset
from scripts.phase6_train import train_phase6_model


def run_ablation_matrix(target_config: Optional[str] = None):
    print("=" * 80)
    print("🔬 [PHASE 6] CONTROLLED ABLATION MATRIX: ARCHITECTURE VS. SENSOR MODALITY")
    print("=" * 80)

    device = "cpu"  # Deterministic CPU execution
    print(f"Hardware Compute Device: {device.upper()}")

    data_dir = "data/phase6_multimodal_runs"
    model_dir = "models/phase6"
    results_dir = "results/phase6"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # 1. Dataset Splits
    src_raw_dir = "data/opensees_runs"
    train_files_raw, val_files_raw, _ = split_simulation_dataset(data_dir=src_raw_dir)
    train_files = [os.path.join(data_dir, os.path.basename(f)) for f in train_files_raw]
    val_files = [os.path.join(data_dir, os.path.basename(f)) for f in val_files_raw]

    print(f"Dataset split sizes: Train = {len(train_files)}, Val = {len(val_files)} (Frozen Test split kept untouched)")

    graph = StructuralGraph()

    # Define Sensor Configurations to evaluate
    all_cfgs = [
        ("S0", SensorConfiguration.create_S0()),
        ("S1", SensorConfiguration.create_S1()),
        ("S2", SensorConfiguration.create_S2()),
        ("S4", SensorConfiguration.create_S4()),
    ]
    if target_config is not None:
        configs = [(c, obj) for c, obj in all_cfgs if c == target_config]
    else:
        configs = all_cfgs

    # Define Model Architectures to evaluate
    # Format: (model_name, use_dual, use_graph, use_symmetry, use_hierarchical)
    model_variants = [
        ("Baseline_InverseFNO", False, False, False, False),
        ("Baseline_GFNO",       False, True,  False, False),
        ("DualStream_GFNO",     True,  True,  False, False),
        ("Symmetry_GFNO",       True,  True,  True,  False),
        ("Proposed_GFNO",       True,  True,  True,  True),
    ]

    out_json = os.path.join(results_dir, "ablation_results.json")
    if os.path.exists(out_json):
        with open(out_json, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for cfg_code, cfg_obj in configs:
        print("\n" + "#" * 80)
        print(f"📊 SENSOR CONFIGURATION: [{cfg_code}] — {cfg_obj.description} ({cfg_obj.num_channels} channels)")
        print("#" * 80)

        train_ds = MultimodalDamageDataset(train_files, cfg_obj, is_train=True)
        val_ds = MultimodalDamageDataset(val_files, cfg_obj, is_train=False)

        # Subsample time series 4x (1000 -> 250) for fast spectral operator execution
        for r in train_ds.records:
            r["Y"] = r["Y"][:, ::4]
            r["Y_global"] = r["Y_global"][:, ::4]
            r["ground_accel"] = r["ground_accel"][:, ::4]
        for r in val_ds.records:
            r["Y"] = r["Y"][:, ::4]
            r["Y_global"] = r["Y_global"][:, ::4]
            r["ground_accel"] = r["ground_accel"][:, ::4]

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        all_results[cfg_code] = {}

        for m_name, dual, gr, sym, hier in model_variants:
            exp_key = f"{cfg_code}_{m_name}"
            print(f"\n---> Training Model [{m_name}] on [{cfg_code}] ...", flush=True)

            torch.manual_seed(42)
            model = DualStreamGFNO(
                sensor_config=cfg_obj,
                structural_graph=graph,
                use_dual_stream=dual,
                use_graph=gr,
                use_symmetry=sym,
                use_hierarchical=hier,
                width_temporal=32,
                modes_temporal=12,
                n_temporal_layers=2,
                width_graph=32,
                n_graph_layers=2,
            )

            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

            # Hierarchical loss vs. direct regression loss
            if hier:
                loss_fn = HierarchicalDamageLoss(
                    lambda_sup=1.0,
                    lambda_sev=2.0,
                    lambda_sym=0.05 if sym else 0.0,
                    lambda_reg=1e-4,
                )
            else:
                # Direct MSE on predicted damage
                class DirectMSELoss(torch.nn.Module):
                    def forward(self, pred, d_true, H_nodes=None):
                        loss = torch.nn.functional.mse_loss(pred["damage_pred"], d_true)
                        return loss, {"total_loss": float(loss.item())}
                loss_fn = DirectMSELoss()

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

            train_res = train_phase6_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=15,
                device=device,
                verbose=False,
            )

            metrics = train_res["final_metrics"]
            ckpt_path = os.path.join(model_dir, f"{exp_key}_best.pt")
            torch.save(model.state_dict(), ckpt_path)

            all_results[cfg_code][m_name] = {
                "config": cfg_code,
                "model_name": m_name,
                "param_count": param_count,
                "use_dual_stream": dual,
                "use_graph": gr,
                "use_symmetry": sym,
                "use_hierarchical": hier,
                "best_epoch": train_res["best_epoch"],
                "best_val_loss": train_res["best_val_loss"],
                "checkpoint_path": ckpt_path,
                "metrics": metrics,
            }

            print(f"     Results: Top-1 Acc = {metrics['top1_localization_acc']*100:5.1f}% | "
                  f"Top-2 Acc = {metrics['top2_localization_acc']*100:5.1f}% | "
                  f"Support F1 = {metrics['support_f1']:5.3f} | "
                  f"Dam-MAE = {metrics['severity_mae_damaged']:6.4f} | "
                  f"Ghost = {metrics['ghost_damage_mag']:6.4f}", flush=True)

    # ==========================================================================
    # PRINT CONSOLIDATED ABLATION TABLE
    # ==========================================================================
    print("\n" + "=" * 105)
    print("🔬 CONSOLIDATED ABLATION MATRIX SUMMARY (EVALUATED ON VALIDATION SET)")
    print("=" * 105)
    header = f"{'Config':6s} | {'Model Architecture':22s} | {'Top-1 Acc':10s} | {'Top-2 Acc':10s} | {'Support F1':10s} | {'Damaged MAE':12s} | {'Ghost Mag':10s}"
    print(header)
    print("-" * 105)

    for cfg_code, _ in configs:
        for m_name, _, _, _, _ in model_variants:
            res = all_results[cfg_code][m_name]["metrics"]
            line = (
                f"{cfg_code:6s} | {m_name:22s} | "
                f"{res['top1_localization_acc']*100:8.1f}% | "
                f"{res['top2_localization_acc']*100:8.1f}% | "
                f"{res['support_f1']:10.3f} | "
                f"{res['severity_mae_damaged']:12.4f} | "
                f"{res['ghost_damage_mag']:10.4f}"
            )
            print(line)
        print("-" * 105)

    out_json = os.path.join(results_dir, "ablation_results.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Complete ablation dataset saved to {out_json}")
    print("🎉 Controlled ablation matrix completed successfully!")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_ablation_matrix(target)
