# Phase 7 Feature Leakage Forensic Audit

**Document:** `results/experiments/phase7_contaminated/PHASE7_FEATURE_LEAKAGE_AUDIT.md`  
**Date:** September 2026  
**Auditor:** Scientific Forensic Auditor  
**Scope:** Complete trace of every tensor, array, and metadata field entering the neural operator.

---

## 1. Exact Leakage Path

The leakage path was programmatically and mathematically traced as follows:

```text
Target damage vector: d_e
       ↓
MultiStoryFrame.build_model(damage_vector)
       ↓
In src/damage_injection.py (lines 248, 270):
eff_E = cfg.E * (1.0 - d_e)
self.elements[ele_id].E = eff_E
       ↓
In scripts/phase7_dataset_generator.py (line 109):
graph_feats = extract_structural_graph_features(frame)  [CALLED POST-SIMULATION ON DAMAGED FRAME]
       ↓
In src/structure_features.py (line 83):
E_norm = float(e_info.E / REF_E0) = cfg.E * (1.0 - d_e) / REF_E0
edge_feats[idx, 3] = E_norm
       ↓
In src/ml/topology_gfno.py (line 291):
if self.model_condition in ["B5", "PROPOSED"]:
    ef = edge_features  [INGESTS ALL 6 COLUMNS, INCLUDING COL 3]
       ↓
h_e = self.edge_decoder(h_v, edge_connectivity, ef)
       ↓
Target damage d_e directly encoded into input feature representation!
```

---

## 2. Complete Feature Inventory & Target-Dependence Audit

Every input tensor and array in `data/phase7_simulations/*.npz` was evaluated across all 140 matched bilateral pairs (280 simulations). The table below documents the source, target-dependence, and model allowance for each feature:

| Feature Name | Dimension | Source | Target-Dependent? | Matched Pair Diff (A vs B) | Allowed in Pristine Model Input? | Forensic Finding |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- |
| `node_signals` | $[N_v, 4, T]$ | OpenSees dynamic response $Y(t)$ | **Yes (Physical)** | Non-zero physical response | **YES** | Legitimate measured sensor observation. |
| `global_signals` | $[5, T]$ | Floor horizontal accelerations + ground accel | **Yes (Physical)** | Non-zero physical response | **YES** | Legitimate measured sensor observation. |
| `node_features[:, 0]` ($x/L_0$) | $[N_v]$ | Geometry from `FrameConfig` | **No** | $0 / 140$ pairs differ | **YES** | Pristine geometric coordinate. |
| `node_features[:, 1]` ($y/H_0$) | $[N_v]$ | Geometry from `FrameConfig` | **No** | $0 / 140$ pairs differ | **YES** | Pristine geometric coordinate. |
| `node_features[:, 2]` ($\text{is\_ground}$) | $[N_v]$ | Boundary conditions from `FrameConfig` | **No** | $0 / 140$ pairs differ | **YES** | Pristine topological boundary flag. |
| `node_features[:, 3]` ($m_v/M_0$) | $[N_v]$ | Floor mass from `FrameConfig` | **No** | $0 / 140$ pairs differ | **YES** | Pristine nominal floor mass. |
| `edge_features[:, 0]` ($L_e/h_0$) | $[N_e]$ | Member length from `FrameConfig` | **No** | $0 / 140$ pairs differ | **YES** | Pristine geometric dimension. |
| `edge_features[:, 1]` ($A_e/A_0$) | $[N_e]$ | Nominal column/beam cross-sectional area | **No** | $0 / 140$ pairs differ | **YES** | Pristine nominal cross-sectional area. |
| `edge_features[:, 2]` ($I_e/I_0$) | $[N_e]$ | Nominal second moment of area | **No** | $0 / 140$ pairs differ | **YES** | Pristine nominal moment of inertia. |
| **`edge_features[:, 3]` ($E_e/E_0$)** | $[N_e]$ | **`e_info.E` from post-damage frame** | **YES (LEAKED)** | **$140 / 140$ pairs differ** | **STRICTLY FORBIDDEN AS DAMAGED** | **CRITICAL DEFECT:** Encoded $(1 - d_e)$. Must be replaced with nominal pristine Young's modulus $\text{cfg.E} / E_0$. |
| `edge_features[:, 4]` ($\cos\theta$) | $[N_e]$ | Member orientation angle | **No** | $0 / 140$ pairs differ | **YES** | Pristine geometric orientation. |
| `edge_features[:, 5]` ($\sin\theta$) | $[N_e]$ | Member orientation angle | **No** | $0 / 140$ pairs differ | **YES** | Pristine geometric orientation. |
| `edge_connectivity` | $[N_e, 2]$ | Frame topology | **No** | $0 / 140$ pairs differ | **YES** | Pristine graph connectivity. |
| `adjacency` | $[N_v, N_v]$ | Graph adjacency with self-loops | **No** | $0 / 140$ pairs differ | **YES** | Pristine graph connectivity. |
| `global_scalars` | $[2]$ | $[\mu_m, \mu_k]$ nominal mass and stiffness ratios | **No** | $0 / 140$ pairs differ | **YES** | Pristine structural variant descriptors. |
| `damage_vector` | $[N_e]$ | Ground truth target damage $d_e$ | **Target** | Ground truth label | **ONLY AS LOSS TARGET** | Ground truth target; must never enter encoder. |

---

## 3. Conclusions of Feature Audit

1. **Exact Defect:** Exactly one column (`edge_features[:, 3]`) contained target-dependent information.
2. **Unaffected Models:** Models `B1`, `B2`, `B3`, and `B4` were **not contaminated** because `B1..B3` do not ingest edge features and `B4` only ingests columns 0 and 4 ($L_e/h_0$ and $\cos\theta$).
3. **Contaminated Models:** Models `B5` and `PROPOSED` ingested column 3, rendering their Phase 7 evaluation numbers invalid.
4. **Correction Required:** In `src/structure_features.py`, Young's modulus must be extracted strictly from nominal structure configuration (`cfg.E`), completely decoupling structural input features from any post-damage simulation state.

---
*Forensic Feature Audit Signed — September 2026*
