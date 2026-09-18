"""
Phase 7.2 Main Execution Harness.

Script: scripts/phase7_runner.py
Author: SeismoFNO Research Team
Context: Phase 7.2 Complete Cross-Structure Generalization Experiment
"""

import os
import sys
import glob
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath("."))

from src.ml.phase7_dataset import Phase7SimulationDataset, Phase7BilateralPairDataset
from src.ml.topology_gfno import TopologyDualStreamGFNO
from src.ml.bilateral_pair_loss import BilateralDirectionalLoss, BilateralMarginSeparationLoss
from src.ml.phase7_train_and_evaluate import (
    Phase7Loss,
    collate_single,
    train_single_epoch,
    evaluate_model_on_pairs,
)


def run_phase7_experiments():
    print("=" * 80)
    print("🔬 PHASE 7.2 CROSS-STRUCTURE GENERALIZATION: TRAINING & EVALUATION")
    print("=" * 80)

    t0_start = time.time()
    device = "cpu"
    os.makedirs("results/phase7/training/checkpoints", exist_ok=True)
    os.makedirs("results/phase7/validation", exist_ok=True)
    os.makedirs("results/phase7/test/level1_id", exist_ok=True)
    os.makedirs("results/phase7/test/level2a_interpolation", exist_ok=True)
    os.makedirs("results/phase7/test/level2b_extrapolation", exist_ok=True)
    os.makedirs("results/phase7/test/level3_topology", exist_ok=True)
    os.makedirs("results/phase7/bilateral", exist_ok=True)
    os.makedirs("results/phase7/statistics", exist_ok=True)
    os.makedirs("results/phase7/reports", exist_ok=True)

    # 1. Partition files
    all_files = sorted(glob.glob("data/phase7_simulations/sim_*.npz"))
    assert len(all_files) == 530, f"Expected 530 files, found {len(all_files)}"

    files_P1_std = [f for f in all_files if "train_P1" in f and "train_P1_pair" not in f]
    files_P1_pair = [f for f in all_files if "train_P1_pair" in f]
    files_P2_div = [f for f in all_files if "train_P2_diversity" in f]

    files_val_std = [f for f in all_files if "val_standard" in f]
    files_val_pair = [f for f in all_files if "val_pair" in f]

    files_test_L1 = [f for f in all_files if "test_L1_ID" in f]
    files_test_L2A = [f for f in all_files if "test_L2A_interp" in f]
    files_test_L2B = [f for f in all_files if "test_L2B_extrap" in f]
    files_test_L3 = [f for f in all_files if "test_L3_topology" in f]

    print(f"Dataset split files verified:")
    print(f"  P1 standard: {len(files_P1_std)} | P1 pairs: {len(files_P1_pair)}")
    print(f"  P2 diversity additions: {len(files_P2_div)}")
    print(f"  Validation standard: {len(files_val_std)} | Validation pairs: {len(files_val_pair)}")
    print(f"  Test Level 1: {len(files_test_L1)} | Level 2A: {len(files_test_L2A)}")
    print(f"  Test Level 2B: {len(files_test_L2B)} | Level 3: {len(files_test_L3)}")

    # Group test pairs for evaluation
    def group_pairs(file_list: List[str]) -> List[Tuple[str, str]]:
        dset = Phase7BilateralPairDataset(file_list)
        return dset.pairs

    test_pairs_L1 = group_pairs(files_test_L1)
    test_pairs_L2A = group_pairs(files_test_L2A)
    test_pairs_L2B = group_pairs(files_test_L2B)
    test_pairs_L3 = group_pairs(files_test_L3)
    val_pairs = group_pairs(files_val_pair)

    assert len(test_pairs_L1) == 30, f"Expected 30 L1 pairs, found {len(test_pairs_L1)}"
    assert len(test_pairs_L2A) == 30, f"Expected 30 L2A pairs, found {len(test_pairs_L2A)}"
    assert len(test_pairs_L2B) == 30, f"Expected 30 L2B pairs, found {len(test_pairs_L2B)}"
    assert len(test_pairs_L3) == 30, f"Expected 30 L3 pairs, found {len(test_pairs_L3)}"
    print(f"Verified exactly 30 pairs (60 evaluation runs) for each test level.")

    # 2. Experimental Execution Matrix
    # We evaluate P1 and P2 protocols across seeds [42, 101, 2024]
    # Primary modalities: S1 and S4 (+ S0 F1 reference)
    # Model hierarchy: B1, B2, B3, B4, B5, PROPOSED
    SEEDS = [42, 101, 2024]
    PROTOCOLS = ["P1", "P2"]
    MODALITIES = ["S1", "S4", "S0"]
    MODELS = ["B1", "B2", "B3", "B4", "B5", "PROPOSED"]

    all_results = []
    ckpt_counter = 0

    # Total runs: 2 protocols * 3 modalities * 6 models * 3 seeds = 108 runs
    # To execute efficiently and cleanly, we iterate over protocols and modalities:
    for proto in PROTOCOLS:
        print("\n" + "#" * 80)
        print(f"### PROTOCOL: {proto} ({'SOURCE_A Only' if proto == 'P1' else 'SOURCE_A + B_train Multi-Structure'})")
        print("#" * 80)

        # Build training file lists for protocol
        if proto == "P1":
            train_std_files = files_P1_std
            train_pair_files = files_P1_pair
        else:  # P2 reuses P1 and adds 140 diversity files
            train_std_files = files_P1_std + files_P2_div
            train_pair_files = files_P1_pair

        for modality in MODALITIES:
            print(f"\n--- Modality: {modality} ---")
            for model_id in MODELS:
                for seed in SEEDS:
                    t_run0 = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Initialize model
                    model = TopologyDualStreamGFNO(
                        model_condition=model_id,
                        width_temporal=16,
                        modes_temporal=8,
                        n_temporal_layers=2,
                        width_graph=16,
                        n_graph_layers=1,
                    ).to(device)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
                    base_loss_fn = Phase7Loss()

                    # DataLoaders
                    std_dset = Phase7SimulationDataset(train_std_files, sensor_modality=modality)
                    std_loader = DataLoader(std_dset, batch_size=16, shuffle=True, collate_fn=collate_single)

                    if model_id == "PROPOSED":
                        pair_dset = Phase7BilateralPairDataset(train_pair_files, sensor_modality=modality)
                        pair_loader = DataLoader(pair_dset, batch_size=4, shuffle=True)
                        dir_loss_fn = BilateralDirectionalLoss()
                        margin_loss_fn = BilateralMarginSeparationLoss(margin=0.15)
                    else:
                        pair_loader = None
                        dir_loss_fn = None
                        margin_loss_fn = None

                    # Train for 12 epochs
                    epochs = 12
                    for ep in range(1, epochs + 1):
                        train_loss = train_single_epoch(
                            model, std_loader, pair_loader, optimizer,
                            base_loss_fn, dir_loss_fn, margin_loss_fn, device=device
                        )

                    # Save checkpoint
                    ckpt_name = f"model_{proto}_{modality}_{model_id}_seed{seed}.pt"
                    ckpt_path = os.path.join("results/phase7/training/checkpoints", ckpt_name)
                    torch.save(model.state_dict(), ckpt_path)

                    # Evaluate on Validation Pairs (frozen threshold selection)
                    val_metrics = evaluate_model_on_pairs(model, val_pairs, sensor_modality=modality, device=device)

                    # Evaluate on Held-Out Test Levels (30 earthquakes each)
                    res_L1 = evaluate_model_on_pairs(model, test_pairs_L1, sensor_modality=modality, device=device)
                    res_L2A = evaluate_model_on_pairs(model, test_pairs_L2A, sensor_modality=modality, device=device)
                    res_L2B = evaluate_model_on_pairs(model, test_pairs_L2B, sensor_modality=modality, device=device)
                    res_L3 = evaluate_model_on_pairs(model, test_pairs_L3, sensor_modality=modality, device=device)

                    # Compute Event-Level Paired OOD Degradation relative to Level 1
                    # Delta_OOD(s, e) = metric(s, e) - metric(SOURCE_A, e)
                    def compute_paired_ood(res_target: Dict[str, Any], res_ref: Dict[str, Any]) -> Dict[str, float]:
                        ev_t = res_target["events"]
                        ev_r = res_ref["events"]
                        # Match by gm_record
                        ref_map = {e["gm_record"]: e for e in ev_r}
                        diffs_sep = []
                        diffs_cos = []
                        diffs_acc = []
                        for et in ev_t:
                            gm = et["gm_record"]
                            if gm in ref_map:
                                er = ref_map[gm]
                                diffs_sep.append(et["separation"] - er["separation"])
                                diffs_cos.append(et["cosine_v_AB"] - er["cosine_v_AB"])
                                diffs_acc.append(float(et["pair_correct"]) - float(er["pair_correct"]))
                        return {
                            "delta_acc_mean": float(np.mean(diffs_acc)) * 100.0 if diffs_acc else 0.0,
                            "delta_sep_mean": float(np.mean(diffs_sep)) if diffs_sep else 0.0,
                            "delta_cos_mean": float(np.mean(diffs_cos)) if diffs_cos else 0.0,
                        }

                    ood_L2A = compute_paired_ood(res_L2A, res_L1)
                    ood_L2B = compute_paired_ood(res_L2B, res_L1)
                    ood_L3 = compute_paired_ood(res_L3, res_L1)

                    run_entry = {
                        "protocol": proto,
                        "modality": modality,
                        "model_id": model_id,
                        "seed": seed,
                        "checkpoint": ckpt_name,
                        "execution_time_s": time.time() - t_run0,
                        "val": {
                            "accuracy_pct": val_metrics["bilateral_accuracy_pct"],
                            "cosine": val_metrics["mean_cosine_v_AB"],
                            "separation": val_metrics["mean_predicted_separation"],
                        },
                        "level1_id": {
                            "accuracy_pct": res_L1["bilateral_accuracy_pct"],
                            "ci_95": [res_L1["ci_95_low_pct"], res_L1["ci_95_high_pct"]],
                            "cosine": res_L1["mean_cosine_v_AB"],
                            "separation": res_L1["mean_predicted_separation"],
                            "confusion": res_L1["confusion_matrix"],
                        },
                        "level2a_interp": {
                            "accuracy_pct": res_L2A["bilateral_accuracy_pct"],
                            "ci_95": [res_L2A["ci_95_low_pct"], res_L2A["ci_95_high_pct"]],
                            "cosine": res_L2A["mean_cosine_v_AB"],
                            "separation": res_L2A["mean_predicted_separation"],
                            "delta_ood": ood_L2A,
                        },
                        "level2b_extrap": {
                            "accuracy_pct": res_L2B["bilateral_accuracy_pct"],
                            "ci_95": [res_L2B["ci_95_low_pct"], res_L2B["ci_95_high_pct"]],
                            "cosine": res_L2B["mean_cosine_v_AB"],
                            "separation": res_L2B["mean_predicted_separation"],
                            "delta_ood": ood_L2B,
                        },
                        "level3_topology": {
                            "accuracy_pct": res_L3["bilateral_accuracy_pct"],
                            "ci_95": [res_L3["ci_95_low_pct"], res_L3["ci_95_high_pct"]],
                            "cosine": res_L3["mean_cosine_v_AB"],
                            "separation": res_L3["mean_predicted_separation"],
                            "delta_ood": ood_L3,
                        },
                    }
                    all_results.append(run_entry)
                    ckpt_counter += 1

                    print(
                        f"  [{proto} | {modality} | {model_id:8s} | s{seed}] "
                        f"L1: {res_L1['bilateral_accuracy_pct']:4.1f}% (cos {res_L1['mean_cosine_v_AB']:+.2f}) | "
                        f"L2A: {res_L2A['bilateral_accuracy_pct']:4.1f}% (d_acc {ood_L2A['delta_acc_mean']:+4.1f}%) | "
                        f"L2B: {res_L2B['bilateral_accuracy_pct']:4.1f}% (d_acc {ood_L2B['delta_acc_mean']:+4.1f}%) | "
                        f"L3: {res_L3['bilateral_accuracy_pct']:4.1f}% (d_acc {ood_L3['delta_acc_mean']:+4.1f}%) | "
                        f"{run_entry['execution_time_s']:.1f}s",
                        flush=True,
                    )

    total_time = time.time() - t0_start
    print("\n" + "=" * 80)
    print(f"✅ ALL {ckpt_counter} RUNS COMPLETED IN {total_time:.2f}s")
    print("=" * 80)

    # 3. Aggregate Statistical Summary Across Seeds (Mean +/- SD)
    # Experimental unit: N=30 independent earthquake cases. Seed variation reported separately.
    agg_summary = {}
    for proto in PROTOCOLS:
        for modality in MODALITIES:
            for model_id in MODELS:
                key = f"{proto}_{modality}_{model_id}"
                runs = [r for r in all_results if r["protocol"] == proto and r["modality"] == modality and r["model_id"] == model_id]
                assert len(runs) == 3

                def seed_stats(metric_extractor):
                    vals = [metric_extractor(r) for r in runs]
                    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}

                agg_summary[key] = {
                    "protocol": proto,
                    "modality": modality,
                    "model_id": model_id,
                    "level1_acc": seed_stats(lambda r: r["level1_id"]["accuracy_pct"]),
                    "level1_cos": seed_stats(lambda r: r["level1_id"]["cosine"]),
                    "level2a_acc": seed_stats(lambda r: r["level2a_interp"]["accuracy_pct"]),
                    "level2a_delta_acc": seed_stats(lambda r: r["level2a_interp"]["delta_ood"]["delta_acc_mean"]),
                    "level2b_acc": seed_stats(lambda r: r["level2b_extrap"]["accuracy_pct"]),
                    "level2b_delta_acc": seed_stats(lambda r: r["level2b_extrap"]["delta_ood"]["delta_acc_mean"]),
                    "level3_acc": seed_stats(lambda r: r["level3_topology"]["accuracy_pct"]),
                    "level3_delta_acc": seed_stats(lambda r: r["level3_topology"]["delta_ood"]["delta_acc_mean"]),
                }

    # Save detailed evaluation JSONs
    with open("results/phase7/test/all_test_evaluations.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open("results/phase7/statistics/cross_structure_summary.json", "w") as f:
        json.dump(agg_summary, f, indent=2)

    # Save Level-specific JSONs
    for lvl, tag in [("level1_id", "level1_id"), ("level2a_interpolation", "level2a_interp"),
                     ("level2b_extrapolation", "level2b_extrap"), ("level3_topology", "level3_topology")]:
        lvl_data = [{
            "protocol": r["protocol"],
            "modality": r["modality"],
            "model_id": r["model_id"],
            "seed": r["seed"],
            "metrics": r[tag],
        } for r in all_results]
        with open(f"results/phase7/test/{lvl}/metrics.json", "w") as f:
            json.dump(lvl_data, f, indent=2)

    print("Saved all JSON artifacts under results/phase7/.")
    return agg_summary, all_results


if __name__ == "__main__":
    run_phase7_experiments()
