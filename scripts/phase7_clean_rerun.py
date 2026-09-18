"""
Phase 7 Clean Rerun Script.

Module: scripts.phase7_clean_rerun
Context: Re-trains and re-evaluates the 36 models affected by Phase 7 edge-feature leakage
         (B5 and PROPOSED across P1/P2, S0/S1/S4, and seeds 42/101/2024) on the pristine dataset.
         Combines with the 72 unaffected B1-B4 baselines to produce the clean 108-model benchmark.
"""

import os
import sys
import glob
import json
import time
import shutil
import numpy as np
import torch
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


def rerun_clean_phase7():
    print("=" * 80)
    print("🚀 PHASE 7 CLEAN BENCHMARK RERUN: B5 & PROPOSED MODELS")
    print("=" * 80)

    t0_start = time.time()
    device = "cpu"

    # Ensure output directories exist
    os.makedirs("results/phase7/training/checkpoints", exist_ok=True)
    os.makedirs("results/phase7/test/level1_id", exist_ok=True)
    os.makedirs("results/phase7/test/level2a_interpolation", exist_ok=True)
    os.makedirs("results/phase7/test/level2b_extrapolation", exist_ok=True)
    os.makedirs("results/phase7/test/level3_topology", exist_ok=True)
    os.makedirs("results/phase7/statistics", exist_ok=True)
    os.makedirs("results/phase7/reports", exist_ok=True)
    os.makedirs("results/experiments/phase7_clean", exist_ok=True)

    # 1. Load preserved unaffected B1-B4 evaluations (72 runs)
    contaminated_evals_path = "results/experiments/phase7_contaminated/test/all_test_evaluations.json"
    assert os.path.exists(contaminated_evals_path), f"Missing {contaminated_evals_path}!"
    with open(contaminated_evals_path, "r") as f:
        contaminated_evals = json.load(f)

    clean_b1_b4_evals = [e for e in contaminated_evals if e["model_id"] in ["B1", "B2", "B3", "B4"]]
    assert len(clean_b1_b4_evals) == 72, f"Expected 72 B1-B4 evals, found {len(clean_b1_b4_evals)}"
    print(f"✅ Loaded {len(clean_b1_b4_evals)} preserved unaffected B1-B4 evaluations.")

    # 2. Partition simulation files
    all_files = sorted(glob.glob("data/phase7_simulations/sim_*.npz"))
    assert len(all_files) == 530, f"Expected 530 files, found {len(all_files)}"

    files_P1_std = [f for f in all_files if "train_P1" in f and "train_P1_pair" not in f]
    files_P1_pair = [f for f in all_files if "train_P1_pair" in f]
    files_P2_div = [f for f in all_files if "train_P2_diversity" in f]

    files_val_pair = [f for f in all_files if "val_pair" in f]
    files_test_L1 = [f for f in all_files if "test_L1_ID" in f]
    files_test_L2A = [f for f in all_files if "test_L2A_interp" in f]
    files_test_L2B = [f for f in all_files if "test_L2B_extrap" in f]
    files_test_L3 = [f for f in all_files if "test_L3_topology" in f]

    def group_pairs(file_list: List[str]) -> List[Tuple[str, str]]:
        dset = Phase7BilateralPairDataset(file_list)
        return dset.pairs

    test_pairs_L1 = group_pairs(files_test_L1)
    test_pairs_L2A = group_pairs(files_test_L2A)
    test_pairs_L2B = group_pairs(files_test_L2B)
    test_pairs_L3 = group_pairs(files_test_L3)
    val_pairs = group_pairs(files_val_pair)

    assert len(test_pairs_L1) == 30 and len(test_pairs_L2A) == 30
    assert len(test_pairs_L2B) == 30 and len(test_pairs_L3) == 30
    assert len(val_pairs) == 10
    print("✅ Verified 30 evaluation pairs each for Levels 1, 2A, 2B, 3.")

    # 3. Models to rerun: B5 and PROPOSED
    SEEDS = [42, 101, 2024]
    PROTOCOLS = ["P1", "P2"]
    MODALITIES = ["S1", "S4", "S0"]
    MODELS_TO_RERUN = ["B5", "PROPOSED"]

    new_clean_evals = []
    ckpt_counter = 0

    for proto in PROTOCOLS:
        print("\n" + "#" * 80)
        print(f"### CLEAN RUN — PROTOCOL: {proto}")
        print("#" * 80)

        train_std_files = files_P1_std if proto == "P1" else files_P1_std + files_P2_div
        train_pair_files = files_P1_pair

        for modality in MODALITIES:
            print(f"\n--- Modality: {modality} ---")
            for model_id in MODELS_TO_RERUN:
                for seed in SEEDS:
                    t_run0 = time.time()
                    torch.manual_seed(seed)
                    np.random.seed(seed)

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

                    # Train for exactly 12 epochs
                    epochs = 12
                    for ep in range(1, epochs + 1):
                        train_single_epoch(
                            model, std_loader, pair_loader, optimizer,
                            base_loss_fn, dir_loss_fn, margin_loss_fn, device=device
                        )

                    # Save clean checkpoint
                    ckpt_name = f"model_{proto}_{modality}_{model_id}_seed{seed}.pt"
                    ckpt_path = os.path.join("results/phase7/training/checkpoints", ckpt_name)
                    torch.save(model.state_dict(), ckpt_path)

                    # Evaluate on Validation Pairs
                    val_metrics = evaluate_model_on_pairs(model, val_pairs, sensor_modality=modality, device=device)

                    # Evaluate on Held-Out Test Levels
                    res_L1 = evaluate_model_on_pairs(model, test_pairs_L1, sensor_modality=modality, device=device)
                    res_L2A = evaluate_model_on_pairs(model, test_pairs_L2A, sensor_modality=modality, device=device)
                    res_L2B = evaluate_model_on_pairs(model, test_pairs_L2B, sensor_modality=modality, device=device)
                    res_L3 = evaluate_model_on_pairs(model, test_pairs_L3, sensor_modality=modality, device=device)

                    # Paired event-level OOD degradation relative to Level 1
                    def compute_paired_ood(res_target: Dict[str, Any], res_ref: Dict[str, Any]) -> Dict[str, float]:
                        ev_t = res_target["events"]
                        ev_r = res_ref["events"]
                        ref_map = {e["gm_record"]: e for e in ev_r}
                        diffs_sep, diffs_cos, diffs_acc = [], [], []
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
                    new_clean_evals.append(run_entry)
                    ckpt_counter += 1

                    print(
                        f"  [{proto} | {modality} | {model_id:8s} | s{seed}] "
                        f"L1: {res_L1['bilateral_accuracy_pct']:4.1f}% (cos {res_L1['mean_cosine_v_AB']:+.2f}) | "
                        f"L2A: {res_L2A['bilateral_accuracy_pct']:4.1f}% (cos {res_L2A['mean_cosine_v_AB']:+.2f}) | "
                        f"L2B: {res_L2B['bilateral_accuracy_pct']:4.1f}% (cos {res_L2B['mean_cosine_v_AB']:+.2f}) | "
                        f"L3: {res_L3['bilateral_accuracy_pct']:4.1f}% (cos {res_L3['mean_cosine_v_AB']:+.2f}) | "
                        f"{run_entry['execution_time_s']:.1f}s",
                        flush=True,
                    )

    print(f"\n✅ Clean rerun finished for all {ckpt_counter} models in {time.time()-t0_start:.1f}s.")

    # 4. Merge clean B1-B4 and newly computed clean B5 & PROPOSED
    ALL_MODELS = ["B1", "B2", "B3", "B4", "B5", "PROPOSED"]
    combined_evals = clean_b1_b4_evals + new_clean_evals
    assert len(combined_evals) == 108, f"Expected 108 total evaluations, found {len(combined_evals)}"

    # Sort deterministically
    def sort_key(e):
        return (e["protocol"], e["modality"], ALL_MODELS.index(e["model_id"]), e["seed"])

    combined_evals.sort(key=sort_key)

    # 5. Aggregate summary across seeds
    agg_summary = {}
    for proto in PROTOCOLS:
        for modality in MODALITIES:
            for model_id in ALL_MODELS:
                key = f"{proto}_{modality}_{model_id}"
                runs = [r for r in combined_evals if r["protocol"] == proto and r["modality"] == modality and r["model_id"] == model_id]
                assert len(runs) == 3, f"Expected 3 runs for {key}, found {len(runs)}"

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
                    "level2a_cos": seed_stats(lambda r: r["level2a_interp"]["cosine"]),
                    "level2a_delta_acc": seed_stats(lambda r: r["level2a_interp"]["delta_ood"]["delta_acc_mean"]),
                    "level2b_acc": seed_stats(lambda r: r["level2b_extrap"]["accuracy_pct"]),
                    "level2b_cos": seed_stats(lambda r: r["level2b_extrap"]["cosine"]),
                    "level2b_delta_acc": seed_stats(lambda r: r["level2b_extrap"]["delta_ood"]["delta_acc_mean"]),
                    "level3_acc": seed_stats(lambda r: r["level3_topology"]["accuracy_pct"]),
                    "level3_cos": seed_stats(lambda r: r["level3_topology"]["cosine"]),
                    "level3_delta_acc": seed_stats(lambda r: r["level3_topology"]["delta_ood"]["delta_acc_mean"]),
                }

    # 6. Save clean artifacts
    with open("results/phase7/test/all_test_evaluations.json", "w") as f:
        json.dump(combined_evals, f, indent=2)

    with open("results/phase7/statistics/cross_structure_summary.json", "w") as f:
        json.dump(agg_summary, f, indent=2)

    for lvl, tag in [("level1_id", "level1_id"), ("level2a_interpolation", "level2a_interp"),
                     ("level2b_extrapolation", "level2b_extrap"), ("level3_topology", "level3_topology")]:
        lvl_data = [{
            "protocol": r["protocol"],
            "modality": r["modality"],
            "model_id": r["model_id"],
            "seed": r["seed"],
            "metrics": r[tag],
        } for r in combined_evals]
        with open(f"results/phase7/test/{lvl}/metrics.json", "w") as f:
            json.dump(lvl_data, f, indent=2)

    # Mirror to results/experiments/phase7_clean/
    shutil.copy("results/phase7/test/all_test_evaluations.json", "results/experiments/phase7_clean/all_test_evaluations.json")
    shutil.copy("results/phase7/statistics/cross_structure_summary.json", "results/experiments/phase7_clean/cross_structure_summary.json")

    print(f"\n🎉 Clean benchmark artifacts successfully updated and mirrored to results/experiments/phase7_clean/!")
    return agg_summary, combined_evals


if __name__ == "__main__":
    rerun_clean_phase7()
