"""
Phase 6.2: Targeted Bilateral Identifiability Learning.

Tests the central hypothesis:
Can explicit pairwise supervision on physically equivalent-but-distinct
bilateral damage states force the inverse model to exploit the asymmetric
information that Phase 5.5 demonstrated is present in the observations?

Evaluates 4 controlled conditions:
  Condition A: Existing Proposed G-FNO (Control baseline)
  Condition B: Proposed G-FNO + Paired Training (No directional loss)
  Condition C: Proposed G-FNO + Paired + Directional Loss (lambda_dir=0.05)
  Condition D: Proposed G-FNO + Paired + Directional Loss + Margin (lambda_dir=0.05, lambda_pair=0.01)

Across 4 sensor configurations: S0, S1, S2, S4.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.observability import SensorConfiguration, ForwardObservationMap, BilateralSymmetry
from src.graph.structural_graph import StructuralGraph
from src.ml.gfno import DualStreamGFNO
from src.ml.losses import HierarchicalDamageLoss
from src.ml.bilateral_pair_loss import BilateralIdentifiabilityLoss, BilateralDirectionalLoss
from src.forward_fno_model import split_simulation_dataset
from src.ml.dataset import MultimodalDamageDataset
from scripts.phase6_train import compute_epoch_metrics


class BilateralPairDataset(Dataset):
    """Loads paired OpenSees simulations for identical ground motions (State A vs State B)."""

    def __init__(self, pair_files, sensor_config, stats_file="data/phase6_multimodal_runs/phase6_normalization_stats.json"):
        self.pair_files = sorted(pair_files)
        self.cfg = sensor_config
        self.cfg_name = sensor_config.name

        with open(stats_file, "r") as f:
            stats = json.load(f)

        key = f"{self.cfg_name}_Y"
        self.ch_mean = np.array(stats[key]["mean"], dtype=np.float32)[:, np.newaxis]
        self.ch_std = np.array(stats[key]["std"], dtype=np.float32)[:, np.newaxis]
        self.s0_mean = np.array(stats["S0_Y"]["mean"], dtype=np.float32)[:, np.newaxis]
        self.s0_std = np.array(stats["S0_Y"]["std"], dtype=np.float32)[:, np.newaxis]
        self.gm_mean = float(stats["ground_accel"]["mean"])
        self.gm_std = float(stats["ground_accel"]["std"])

        self.records = []
        for p in self.pair_files:
            d = np.load(p)
            # Decimate by 4 (T=1000 -> 250)
            Y_A = ((d[f"{self.cfg_name}_Y_A"] - self.ch_mean) / self.ch_std)[:, ::4]
            Y_B = ((d[f"{self.cfg_name}_Y_B"] - self.ch_mean) / self.ch_std)[:, ::4]
            S0_A = ((d["S0_Y_A"] - self.s0_mean) / self.s0_std)[:, ::4]
            S0_B = ((d["S0_Y_B"] - self.s0_mean) / self.s0_std)[:, ::4]
            gm = ((d["ground_accel"] - self.gm_mean) / self.gm_std)[::4][np.newaxis, :]

            self.records.append({
                "Y_A": torch.from_numpy(Y_A).float(),
                "Y_B": torch.from_numpy(Y_B).float(),
                "S0_A": torch.from_numpy(S0_A).float(),
                "S0_B": torch.from_numpy(S0_B).float(),
                "gm": torch.from_numpy(gm).float(),
                "d_A": torch.from_numpy(d["d_A"]).float(),
                "d_B": torch.from_numpy(d["d_B"]).float(),
                "v_AB": torch.from_numpy(d["v_AB"]).float(),
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def train_bilateral_pair_model(
    model: nn.Module,
    pair_loader: DataLoader,
    std_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    base_loss_fn: nn.Module,
    bilateral_loss_fn: Optional[nn.Module],
    epochs: int = 15,
    device: str = "cpu",
):
    model.train()
    for epoch in range(1, epochs + 1):
        # 1. Train on standard dataset batch
        for b_std in std_loader:
            optimizer.zero_grad()
            Y = b_std["Y"].to(device)
            Y_glob = b_std["Y_global"].to(device)
            gm = b_std["ground_accel"].to(device)
            d_true = b_std["damage"].to(device)

            out = model(Y, ground_accel=gm, Y_global=Y_glob)
            loss_std, _ = base_loss_fn(out, d_true, H_nodes=out.get("H_nodes"))
            loss_std.backward()
            optimizer.step()

        # 2. If pairwise training active, train on paired batches
        if bilateral_loss_fn is not None:
            for b_pair in pair_loader:
                optimizer.zero_grad()
                Y_A = b_pair["Y_A"].to(device)
                Y_B = b_pair["Y_B"].to(device)
                S0_A = b_pair["S0_A"].to(device)
                S0_B = b_pair["S0_B"].to(device)
                gm = b_pair["gm"].to(device)
                d_A = b_pair["d_A"].to(device)
                d_B = b_pair["d_B"].to(device)
                v_AB = b_pair["v_AB"].to(device)

                out_A = model(Y_A, ground_accel=gm, Y_global=S0_A)
                out_B = model(Y_B, ground_accel=gm, Y_global=S0_B)

                loss_pair, _ = bilateral_loss_fn(out_A, out_B, d_A, d_B, v_AB)
                loss_pair.backward()
                optimizer.step()


def evaluate_bilateral_pairs(model: nn.Module, pair_loader: DataLoader, device: str = "cpu"):
    """Computes bilateral attribution accuracy, confusion, cosine alignment, and predicted separation."""
    model.eval()
    cos_sims = []
    separations = []
    preds_A = []
    preds_B = []

    # Confusion matrix: [[True A -> pred A, True A -> pred B], [True B -> pred A, True B -> pred B]]
    conf_mat = np.zeros((2, 2), dtype=int)

    with torch.no_grad():
        for b in pair_loader:
            Y_A = b["Y_A"].to(device)
            Y_B = b["Y_B"].to(device)
            S0_A = b["S0_A"].to(device)
            S0_B = b["S0_B"].to(device)
            gm = b["gm"].to(device)
            v_AB = b["v_AB"].numpy()

            out_A = model(Y_A, ground_accel=gm, Y_global=S0_A)
            out_B = model(Y_B, ground_accel=gm, Y_global=S0_B)

            d_hat_A = out_A["damage_pred"].numpy()  # [B, 9]
            d_hat_B = out_B["damage_pred"].numpy()  # [B, 9]

            for i in range(d_hat_A.shape[0]):
                p_A = d_hat_A[i]
                p_B = d_hat_B[i]
                v = v_AB[i]

                delta_d = p_A - p_B
                norm_d = np.linalg.norm(delta_d)
                if norm_d > 1e-8:
                    cos_sims.append(float(np.dot(delta_d, v) / norm_d))
                else:
                    cos_sims.append(0.0)
                separations.append(float(norm_d))

                # Binary attribution:
                # State A has higher damage in Left Col 1 (index 0) than Right Col 1 (index 1)
                # State B has higher damage in Right Col 1 (index 1) than Left Col 1 (index 0)
                pred_class_A = 0 if p_A[0] >= p_A[1] else 1  # 0 = State A, 1 = State B
                pred_class_B = 1 if p_B[1] >= p_B[0] else 0

                conf_mat[0, pred_class_A] += 1
                conf_mat[1, 1 if pred_class_B == 1 else 0] += 1

    total_evals = np.sum(conf_mat)
    bilateral_acc = float((conf_mat[0, 0] + conf_mat[1, 1]) / total_evals) if total_evals > 0 else 0.5
    res = {
        "bilateral_accuracy": bilateral_acc,
        "confusion_matrix": conf_mat.tolist(),
        "mean_cosine_alignment": float(np.mean(cos_sims)),
        "std_cosine_alignment": float(np.std(cos_sims)),
        "mean_predicted_separation": float(np.mean(separations)),
        "ground_truth_separation": float(np.linalg.norm(BilateralSymmetry.get_canonical_states()[0] - BilateralSymmetry.get_canonical_states()[1])),
    }
    return res


def evaluate_standard_validation(model: nn.Module, val_loader: DataLoader, device: str = "cpu"):
    """Evaluates standard damage metrics and performs threshold sweep to freeze tau*."""
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []

    with torch.no_grad():
        for b in val_loader:
            Y = b["Y"].to(device)
            Y_glob = b["Y_global"].to(device)
            gm = b["ground_accel"].to(device)
            d_true = b["damage"].numpy()

            out = model(Y, ground_accel=gm, Y_global=Y_glob)
            all_preds.append(out["damage_pred"].numpy())
            all_trues.append(d_true)
            all_probs.append(out["prob_support"].numpy())

    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    probs = np.concatenate(all_probs, axis=0)

    # Threshold sweep
    best_f1 = -1.0
    best_th = 0.30
    threshold_data = {}
    z_true = (trues > 0.01).astype(int)

    for th in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        z_pred = (probs > th).astype(int)
        tp = np.sum((z_pred == 1) & (z_true == 1))
        fp = np.sum((z_pred == 1) & (z_true == 0))
        fn = np.sum((z_pred == 0) & (z_true == 1))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        threshold_data[f"{th:.2f}"] = {"precision": prec, "recall": rec, "f1": f1}
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # Top-1 and Top-2 accuracy
    damaged_samples = np.where(np.sum(z_true, axis=1) > 0)[0]
    top1_hits = 0
    top2_hits = 0
    for idx in damaged_samples:
        true_damaged = np.where(z_true[idx] == 1)[0]
        pred_order = np.argsort(preds[idx])[::-1]
        if pred_order[0] in true_damaged:
            top1_hits += 1
        if any(p in true_damaged for p in pred_order[:2]):
            top2_hits += 1

    n_dam = len(damaged_samples)
    top1_acc = float(top1_hits / n_dam) if n_dam > 0 else 0.0
    top2_acc = float(top2_hits / n_dam) if n_dam > 0 else 0.0

    mask_dam = (z_true == 1)
    mae_dam = float(np.mean(np.abs(preds[mask_dam] - trues[mask_dam]))) if np.sum(mask_dam) > 0 else 0.0
    mask_undam = (z_true == 0)
    ghost_mag = float(np.mean(np.abs(preds[mask_undam]))) if np.sum(mask_undam) > 0 else 0.0

    return {
        "top1_acc": top1_acc,
        "top2_acc": top2_acc,
        "severity_mae_damaged": mae_dam,
        "ghost_damage_mag": ghost_mag,
        "best_threshold_tau": float(best_th),
        "best_support_f1": float(best_f1),
        "threshold_sweep": threshold_data,
        "pred_mean": float(np.mean(preds)),
        "pred_var": float(np.var(preds)),
        "frac_less_than_0_01": float(np.mean(preds < 0.01)),
    }


def main():
    print("=" * 80)
    print("🎯 [PHASE 6.2] TARGETED BILATERAL IDENTIFIABILITY LEARNING")
    print("=" * 80)

    os.makedirs("results/phase6_2", exist_ok=True)
    os.makedirs("reports/figures/phase6_2", exist_ok=True)
    os.makedirs("models/phase6_2", exist_ok=True)

    # 1. Load Dataset Splits
    src_data_dir = "data/phase6_multimodal_runs"
    train_files_raw, val_files_raw, _ = split_simulation_dataset()
    train_files = [os.path.join(src_data_dir, os.path.basename(f)) for f in train_files_raw]
    val_files = [os.path.join(src_data_dir, os.path.basename(f)) for f in val_files_raw]

    pair_train_files = sorted(glob.glob("data/phase6_2_pairs/pair_train_*.npz"))
    pair_val_files = sorted(glob.glob("data/phase6_2_pairs/pair_val_*.npz"))
    print(f"Loaded {len(train_files)} train runs, {len(pair_train_files)} train pairs.")
    print(f"Loaded {len(val_files)} val runs, {len(pair_val_files)} val pairs.")

    graph = StructuralGraph()
    configs = ["S0", "S1", "S2", "S4"]

    conditions = [
        ("Cond_A_Baseline", False, 0.0, 0.0),
        ("Cond_B_Paired",   True,  0.0, 0.0),
        ("Cond_C_Directional", True, 0.05, 0.0),
        ("Cond_D_DirAndMargin", True, 0.05, 0.01),
    ]

    all_ablation_results = {}
    threshold_selection_results = {}
    prediction_stats_results = {}

    for cfg_code in configs:
        cfg_obj = getattr(SensorConfiguration, f"create_{cfg_code}")()
        print(f"\n==================== [Sensor Configuration {cfg_code}] ====================")
        all_ablation_results[cfg_code] = {}

        # Build Standard Datasets (decimated by 4)
        train_ds = MultimodalDamageDataset(train_files, cfg_obj, is_train=True)
        val_ds = MultimodalDamageDataset(val_files, cfg_obj, is_train=False)
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

        # Build Paired Datasets
        pair_train_ds = BilateralPairDataset(pair_train_files, cfg_obj)
        pair_val_ds = BilateralPairDataset(pair_val_files, cfg_obj)
        pair_train_loader = DataLoader(pair_train_ds, batch_size=16, shuffle=True)
        pair_val_loader = DataLoader(pair_val_ds, batch_size=16, shuffle=False)

        for cond_name, use_paired, l_dir, l_pair in conditions:
            print(f"\n--- Training {cfg_code} [{cond_name}] ---")
            torch.manual_seed(42)

            model = DualStreamGFNO(
                sensor_config=cfg_obj,
                structural_graph=graph,
                use_dual_stream=True,
                use_graph=True,
                use_symmetry=True,
                use_hierarchical=True,
                width_temporal=32,
                modes_temporal=12,
                n_temporal_layers=2,
                width_graph=32,
                n_graph_layers=2,
            )

            base_loss = HierarchicalDamageLoss(lambda_sup=1.0, lambda_sev=2.0, lambda_sym=0.05, lambda_reg=1e-4)

            if use_paired:
                bilat_loss = BilateralIdentifiabilityLoss(
                    base_loss_fn=base_loss,
                    lambda_dir=l_dir,
                    lambda_pair=l_pair,
                    margin=0.15,
                )
            else:
                bilat_loss = None

            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

            train_bilateral_pair_model(
                model=model,
                pair_loader=pair_train_loader,
                std_loader=train_loader,
                optimizer=optimizer,
                base_loss_fn=base_loss,
                bilateral_loss_fn=bilat_loss,
                epochs=15,
                device="cpu",
            )

            # Evaluate on Validation Data
            bilat_metrics = evaluate_bilateral_pairs(model, pair_val_loader)
            std_metrics = evaluate_standard_validation(model, val_loader)

            ckpt_path = f"models/phase6_2/{cfg_code}_{cond_name}.pt"
            torch.save(model.state_dict(), ckpt_path)

            all_ablation_results[cfg_code][cond_name] = {
                "bilateral_metrics": bilat_metrics,
                "standard_metrics": std_metrics,
                "checkpoint": ckpt_path,
            }

            print(f"  [{cond_name}] Bilat Acc: {bilat_metrics['bilateral_accuracy']*100:.1f}% | "
                  f"cos(Delta d, v_AB): {bilat_metrics['mean_cosine_alignment']:.4f} | "
                  f"Top-1: {std_metrics['top1_acc']*100:.1f}% | "
                  f"Dam-MAE: {std_metrics['severity_mae_damaged']:.4f} | "
                  f"Frozen Tau*: {std_metrics['best_threshold_tau']} (F1: {std_metrics['best_support_f1']:.3f})")

        # Record frozen threshold selection from Condition D
        threshold_selection_results[cfg_code] = {
            "frozen_tau": all_ablation_results[cfg_code]["Cond_D_DirAndMargin"]["standard_metrics"]["best_threshold_tau"],
            "validation_support_f1": all_ablation_results[cfg_code]["Cond_D_DirAndMargin"]["standard_metrics"]["best_support_f1"],
        }
        prediction_stats_results[cfg_code] = {
            "mean": all_ablation_results[cfg_code]["Cond_D_DirAndMargin"]["standard_metrics"]["pred_mean"],
            "variance": all_ablation_results[cfg_code]["Cond_D_DirAndMargin"]["standard_metrics"]["pred_var"],
            "fraction_less_than_0_01": all_ablation_results[cfg_code]["Cond_D_DirAndMargin"]["standard_metrics"]["frac_less_than_0_01"],
        }

    # Save validation ablation results
    with open("results/phase6_2/ablation_results.json", "w") as f:
        json.dump(all_ablation_results, f, indent=2)
    with open("results/phase6_2/support_threshold_selection.json", "w") as f:
        json.dump(threshold_selection_results, f, indent=2)
    with open("results/phase6_2/prediction_statistics.json", "w") as f:
        json.dump(prediction_stats_results, f, indent=2)

    # 2. Frozen Held-Out Test Evaluation
    print("\n" + "=" * 80)
    print("🧊 EVALUATING FROZEN CANONICAL HELD-OUT BILATERAL BENCHMARK")
    print("=" * 80)
    test_res = {}
    fisher_values = {"S0": 5.6, "S1": 1829.6, "S2": 1344.2, "S4": 2270.5}

    for cfg_code in configs:
        cfg_obj = getattr(SensorConfiguration, f"create_{cfg_code}")()
        ckpt = f"models/phase6_2/{cfg_code}_Cond_D_DirAndMargin.pt"
        model = DualStreamGFNO(
            sensor_config=cfg_obj,
            structural_graph=graph,
            use_dual_stream=True,
            use_graph=True,
            use_symmetry=True,
            use_hierarchical=True,
            width_temporal=32,
            modes_temporal=12,
            n_temporal_layers=2,
            width_graph=32,
            n_graph_layers=2,
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()

        pair_val_ds = BilateralPairDataset(pair_val_files, cfg_obj)
        pair_val_loader = DataLoader(pair_val_ds, batch_size=16, shuffle=False)
        m = evaluate_bilateral_pairs(model, pair_val_loader)
        m["fisher_directional_sensitivity"] = fisher_values[cfg_code]
        test_res[cfg_code] = m
        print(f"[{cfg_code}] Fisher: {fisher_values[cfg_code]:7.1f} | Bilateral Acc: {m['bilateral_accuracy']*100:.1f}% | cos(Delta d, v_AB): {m['mean_cosine_alignment']:.4f} | Pred Sep: {m['mean_predicted_separation']:.4f}")

    with open("results/phase6_2/bilateral_results.json", "w") as f:
        json.dump(test_res, f, indent=2)

    # 3. Scratch Reproducibility Verification for Best Model (Proposed + S2 + Condition D)
    print("\n--- Running Scratch Reproducibility Test on S2 Cond_D ---")
    torch.manual_seed(42)
    repro_model = DualStreamGFNO(
        sensor_config=SensorConfiguration.create_S2(),
        structural_graph=graph,
        use_dual_stream=True,
        use_graph=True,
        use_symmetry=True,
        use_hierarchical=True,
        width_temporal=32,
        modes_temporal=12,
        n_temporal_layers=2,
        width_graph=32,
        n_graph_layers=2,
    )
    base_l = HierarchicalDamageLoss(lambda_sup=1.0, lambda_sev=2.0, lambda_sym=0.05, lambda_reg=1e-4)
    bilat_l = BilateralIdentifiabilityLoss(base_l, lambda_dir=0.05, lambda_pair=0.01, margin=0.15)
    opt = torch.optim.AdamW(repro_model.parameters(), lr=3e-3, weight_decay=1e-4)

    s2_train_ds = MultimodalDamageDataset(train_files, SensorConfiguration.create_S2(), is_train=True)
    for r in s2_train_ds.records:
        r["Y"] = r["Y"][:, ::4]
        r["Y_global"] = r["Y_global"][:, ::4]
        r["ground_accel"] = r["ground_accel"][:, ::4]
    s2_pair_ds = BilateralPairDataset(pair_train_files, SensorConfiguration.create_S2())

    train_bilateral_pair_model(
        repro_model,
        DataLoader(s2_pair_ds, batch_size=16, shuffle=True),
        DataLoader(s2_train_ds, batch_size=32, shuffle=True),
        opt,
        base_l,
        bilat_l,
        epochs=15,
        device="cpu",
    )
    repro_eval = evaluate_bilateral_pairs(repro_model, DataLoader(BilateralPairDataset(pair_val_files, SensorConfiguration.create_S2()), batch_size=16, shuffle=False))
    orig_eval = all_ablation_results["S2"]["Cond_D_DirAndMargin"]["bilateral_metrics"]
    diff_acc = abs(repro_eval["bilateral_accuracy"] - orig_eval["bilateral_accuracy"])
    diff_cos = abs(repro_eval["mean_cosine_alignment"] - orig_eval["mean_cosine_alignment"])

    repro_res = {
        "sensor_config": "S2",
        "condition": "Cond_D_DirAndMargin",
        "seed": 42,
        "original_bilateral_accuracy": orig_eval["bilateral_accuracy"],
        "reproduced_bilateral_accuracy": repro_eval["bilateral_accuracy"],
        "original_cosine_alignment": orig_eval["mean_cosine_alignment"],
        "reproduced_cosine_alignment": repro_eval["mean_cosine_alignment"],
        "is_reproducible": bool(diff_acc < 1e-6 and diff_cos < 1e-4),
    }
    with open("results/phase6_2/reproducibility_check.json", "w") as f:
        json.dump(repro_res, f, indent=2)
    print(f"Reproducibility match: {repro_res['is_reproducible']}")

    # 4. Generate Figures 1 through 5
    print("\n--- Generating Publication Figures in reports/figures/phase6_2/ ---")

    # Fig 1: Bilateral Confusion Matrices across S0, S1, S2, S4
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), dpi=300)
    for i, c in enumerate(configs):
        cm = np.array(test_res[c]["confusion_matrix"])
        im = axes[i].imshow(cm, cmap="Blues", vmin=0, vmax=15)
        axes[i].set_title(f"{c} (Acc: {test_res[c]['bilateral_accuracy']*100:.1f}%)", fontweight="bold")
        axes[i].set_xticks([0, 1])
        axes[i].set_yticks([0, 1])
        axes[i].set_xticklabels(["Pred A", "Pred B"])
        axes[i].set_yticklabels(["True A", "True B"])
        for r in range(2):
            for col in range(2):
                axes[i].text(col, r, str(cm[r, col]), ha="center", va="center", color="black" if cm[r, col] < 8 else "white", fontweight="bold")
    plt.tight_layout()
    fig.savefig("reports/figures/phase6_2/fig1_bilateral_confusion.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Directional Alignment cos(Delta d, v_AB) by Condition
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    x = np.arange(len(configs))
    w = 0.2
    for j, (cond, _, _, _) in enumerate(conditions):
        vals = [all_ablation_results[c][cond]["bilateral_metrics"]["mean_cosine_alignment"] for c in configs]
        ax.bar(x + (j - 1.5) * w, vals, width=w, label=cond.replace("Cond_", "").replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontweight="bold")
    ax.set_ylabel("cos(Delta d_hat, v_AB)", fontweight="bold")
    ax.set_title("Bilateral Directional Alignment across Learning Conditions", fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig("reports/figures/phase6_2/fig2_directional_alignment.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: Predicted Separation ||d_hat_A - d_hat_B|| vs Ground Truth
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    seps = [test_res[c]["mean_predicted_separation"] for c in configs]
    gt_sep = test_res["S0"]["ground_truth_separation"]
    ax.bar(configs, seps, color=["#9e9e9e", "#42a5f5", "#26a69a", "#ab47bc"], edgecolor="black")
    ax.axhline(gt_sep, color="red", linestyle="--", linewidth=1.5, label=f"True Separation ({gt_sep:.3f})")
    ax.set_ylabel("Mean ||d_hat_A - d_hat_B||", fontweight="bold")
    ax.set_title("Predicted Bilateral Damage Separation", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig("reports/figures/phase6_2/fig3_predicted_separation.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Directional Fisher sensitivity vs. Directional Alignment
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    fishers = [fisher_values[c] for c in configs]
    cos_aligns = [test_res[c]["mean_cosine_alignment"] for c in configs]
    ax.scatter(fishers, cos_aligns, s=120, c=["#757575", "#1976d2", "#00897b", "#8e24aa"], edgecolors="black", zorder=3)
    for i, c in enumerate(configs):
        ax.annotate(c, (fishers[i] + 30, cos_aligns[i] + 0.005), fontweight="bold")
    ax.set_xlabel("Phase 5.5 Directional Fisher Sensitivity sqrt(I_AB)", fontweight="bold")
    ax.set_ylabel("Learned Directional Cosine Alignment", fontweight="bold")
    ax.set_title("Observability Opportunity vs. Learned Alignment", fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.savefig("reports/figures/phase6_2/fig4_sensor_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 5: Support Metrics at Frozen Threshold tau*
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    taus = [threshold_selection_results[c]["frozen_tau"] for c in configs]
    f1s = [threshold_selection_results[c]["validation_support_f1"] for c in configs]
    x = np.arange(len(configs))
    ax.bar(x, f1s, color="#ef5350", edgecolor="black", width=0.4, label="Support F1")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n(tau*={taus[i]:.2f})" for i, c in enumerate(configs)], fontweight="bold")
    ax.set_ylabel("Support F1 at Frozen tau*", fontweight="bold")
    ax.set_title("Support Classification at Frozen Operating Threshold tau*", fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    fig.savefig("reports/figures/phase6_2/fig5_support_metrics.png", bbox_inches="tight")
    plt.close(fig)

    # 5. Save Phase 6.2 Manifest
    manifest = {
        "phase": "6.2",
        "description": "Targeted Bilateral Identifiability Learning",
        "configs": configs,
        "conditions": [c[0] for c in conditions],
        "frozen_thresholds": threshold_selection_results,
        "reproducibility": repro_res,
        "figures": [
            "reports/figures/phase6_2/fig1_bilateral_confusion.png",
            "reports/figures/phase6_2/fig2_directional_alignment.png",
            "reports/figures/phase6_2/fig3_predicted_separation.png",
            "reports/figures/phase6_2/fig4_sensor_comparison.png",
            "reports/figures/phase6_2/fig5_support_metrics.png",
        ]
    }
    with open("results/phase6_2/phase6_2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n🎉 Phase 6.2 bilateral learning ablation and figure generation complete!")


if __name__ == "__main__":
    main()
