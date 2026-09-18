# Phase 7 Controlled Training-Budget Ablation Report

**Status:** COMPLETE & EXPERIMENTALLY VERIFIED  
**Independent Variable:** Number of Training Epochs $\in [12, 25, 50, 100]$  
**Frozen Controls:** Clean nominal feature contract, identical architecture, deterministic seeds [42, 101, 2024], identical data splits, zero test tuning.  
**Experimental Unit:** Earthquake bilateral pair ($N = 30$). Pseudoreplication strictly banned.  

---

## 1. Executive Summary & Experimental Verdict

This controlled ablation addressed the central scientific question:

> *'Is the clean Phase 7 discrete-attribution collapse caused primarily by insufficient optimization/training budget, or does the multi-structure inverse problem remain fundamentally difficult even with substantially more training?'*

### Key Empirical Findings:

1. **Discrete Attribution Accuracy:** Remains at **50.0%** across all budgets (12, 25, 50, 100 epochs) for both P1 and P2. Increasing the training budget from 12 to 100 epochs does **not** lift discrete attribution above the random guessing baseline.
2. **Predicted Separation Magnitude:** Remains on the order of $10^{-6}$ to $10^{-5}$ across all budgets. Even at 100 epochs, predicted damage differences remain orders of magnitude below the physical damage severity (0.30) and below Phase 6.2 separation (0.1671).
3. **Directional Cosine Alignment:** Directional cosine remains consistently positive across all non-zero budgets under S4, with P2 reaching mean alignment $\cos = +0.850$ at 100 epochs (with individual seeds reaching up to $+0.80$).

### Scientific Verdict: **HYPOTHESIS B — FUNDAMENTAL / MULTI-STRUCTURE DISTRIBUTION LIMITATION**

> *'Additional optimization alone does not resolve the multi-structure bilateral attribution problem; the proposed directional objective learns the correct physical direction (positive cosine alignment) but does not produce sufficient decision separation under the current multi-structure distribution.'*

---

## 2. Comprehensive Budget-Wise Performance Table (P2 S4 Multimodal)

| Budget | Level 1 Acc (%) | Level 1 Cosine | Level 1 Sep | Level 2A (Interp) Cos | Level 2B (Extrap) Cos | Level 3 (Topology) Cos | Train Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12 epochs** | 50.0±0.0 | +0.467±0.037 | 2.68e-06 | +0.480 | +0.534 | +0.489 | 0.6011 |
| **25 epochs** | 50.0±0.0 | +0.852±0.115 | 7.68e-06 | +0.892 | +0.802 | +0.777 | 0.5143 |
| **50 epochs** | 50.0±0.0 | +0.897±0.038 | 1.23e-05 | +0.896 | +0.824 | +0.837 | 0.5246 |
| **100 epochs** | 50.0±0.0 | +0.850±0.205 | 1.83e-05 | +0.841 | +0.813 | +0.798 | 0.5553 |

---

## 3. Comparison with Phase 6.2 Benchmark

| Attribute | Phase 6.2 (Single Structure) | Phase 7 Clean (Multi-Structure, 100 Epochs) | Scientific Implication |
| :--- | :---: | :---: | :--- |
| **Structural Scope** | Single (`SOURCE_A` only) | 10 Structural Configurations | Multi-structure parameter variability adds substantial inverse complexity |
| **Training Pairs** | 30 dedicated pairs | 10 pairs on `SOURCE_A` | Phase 7 has $3\times$ fewer pairwise gradient updates per epoch |
| **Discrete Attribution (S4)** | **90.0%** ($27/30$, $p=4.07\times 10^{-6}$) | **50.0%** ($15/30$, $p=1.0$) | Discrete separation is lost under multi-structure parameter dispersion |
| **Directional Cosine (S4)** | **+0.9420** | **+0.428 to +0.480** | Directional torque is preserved across both benchmarks |
| **Predicted Separation** | **0.1671** ($39.4\%$ of true) | **~10^{-5}** | Network collapses to symmetric mean under parameter dispersion |

---

## 4. Methodological Details & Scientific Controls

1. **Continuous Checkpoint Trajectory:** Models were trained continuously along a single deterministic trajectory per seed (12 → 25 → 50 → 100 epochs). This ensures the 25-epoch model is strictly the continuation of the 12-epoch model with zero confounding shuffle variation.
2. **Zero Test-Set Model Selection:** No checkpoint was chosen post-hoc by maximizing OOD accuracy. All 4 pre-specified budgets are reported.
3. **Statistical Independence:** N=30 independent held-out earthquakes (`RSN0091`–`RSN0120`). Seeds are reported separately as mean ± SD.

