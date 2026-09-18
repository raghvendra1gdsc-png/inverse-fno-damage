# Phase 7 Contaminated Benchmark Archive

**Archival Date:** September 2026  
**Status:** **INVALID FOR MODEL-PERFORMANCE CLAIMS (RETAINED FOR FORENSIC REPRODUCIBILITY ONLY)**

## Archival Statement

> **"These Phase 7 results are retained for forensic reproducibility but are INVALID for model-performance claims because target damage information entered the B5/PROPOSED edge features through damaged Young's modulus."**

---

## 1. Summary of Defect

During the scientific forensic audit of Phase 7, a critical feature leakage defect was identified in `scripts/phase7_dataset_generator.py`:
```python
# Line 104-106: Dynamic analysis runs, calling frame.build_model(damage_vector)
res_s0 = TopologyObservationMap.evaluate(frame, damage_vector, accel, target_dt, s0_cfg)

# Line 109: Structural graph features extracted from post-damage frame!
graph_feats = extract_structural_graph_features(frame)
```
In `src/structure_features.py` (line 83):
```python
E_norm = float(e_info.E / REF_E0)
```
Because `frame.build_model(damage_vector)` modified each element's Young's modulus to $E_{\text{eff}} = E_{\text{nominal}} \times (1 - d_e)$, the resulting `edge_features[:, 3]` contained the normalized post-damage stiffness rather than the nominal pristine material property.

Models `B5` and `PROPOSED` were designed to ingest all 6 columns of `edge_features` (intended to supply nominal structural geometry and stiffness parameters across structural variants). Instead, they received $(1 - d_e)$ directly, enabling the network to predict localized damage without needing to learn the inverse mapping from accelerometer signals.

---

## 2. Affected and Unaffected Components

| Model / Artifact | Ingests `edge_features[:, 3]`? | Leakage Status | Validity of Historical Results |
| :--- | :---: | :---: | :--- |
| **B1 (Global FNO Baseline)** | No (uses zero edge features) | **CLEAN** | Valid negative control ($50.0\%$ chance level) |
| **B2 (Node-Only G-FNO)** | No (uses zero edge features) | **CLEAN** | Valid negative control ($49.4\%$ chance level) |
| **B3 (Topology + Adjacency)** | No (uses zero edge features) | **CLEAN** | Valid negative control ($50.0\%$ chance level) |
| **B4 (Geometric Edge Features)** | No (uses only cols 0 & 4: length & orientation) | **CLEAN** | Valid negative control ($50.0\%$ chance level) |
| **B5 (Full Edge Features)** | Yes (uses all 6 edge features) | **CONTAMINATED** | **INVALID** ($100.0\%$ attribution artifact) |
| **PROPOSED (Dual-Stream + Bilateral Margin)** | Yes (uses all 6 edge features) | **CONTAMINATED** | **INVALID** ($100.0\%$ attribution artifact) |

---

## 3. Preserved Artifacts in this Directory

The following contaminated Phase 7 artifacts are preserved in immutable state for audit tracing:
- `checkpoints/`: All 108 original model checkpoints (`model_{proto}_{modality}_{model_id}_seed{seed}.pt`).
- `test/all_test_evaluations.json`: Complete record of all 108 test evaluations showing the $100.0\%$ attribution artifact under S0/S1/S4 for B5 and PROPOSED.
- `test/level*/metrics.json`: Level-specific evaluation JSONs.
- `statistics/cross_structure_summary.json`: Aggregated statistical metrics.
- `reports/phase7_2_inverse_evaluation_report.md`: Historical report containing the discrepancy between narrative text and raw JSON data.
- `reports/FINAL_SCIENTIFIC_FORENSIC_AUDIT.md`: Complete forensic audit report.
- `reports/FINAL_SCIENTIFIC_FORENSIC_AUDIT.json`: Machine-readable audit report.
- `design/`: Historical design matrices, structural split manifests, and protocol patches.

---
*Immutable Archive Created — September 2026*
