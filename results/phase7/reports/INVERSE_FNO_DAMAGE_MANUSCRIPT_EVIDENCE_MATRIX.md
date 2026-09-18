# Inverse-FNO-Damage — Manuscript Evidence & Traceability Matrix

**Project:** Inverse-FNO-Damage  
**Document:** Manuscript Evidence & Traceability Matrix  
**Date:** September 2026  
**Status:** COMPLETE & 100% REPRODUCIBLE (Frozen Experimental Program)

---

## 1. Traceability Architecture

This document establishes full traceability for every quantitative claim, metric, $p$-value, sample size, and architectural specification presented in the research manuscript `INVERSE_FNO_DAMAGE_RESEARCH_MANUSCRIPT.md`. Every entry maps directly to an audited repository file, execution script, and deterministic configuration.

---

## 2. Core Quantitative Evidence Mapping

| ID | Scientific Claim / Finding | Quantitative Value in Manuscript | Exact Repository Artifact Source | Deterministic Script / Command | Statistical Unit & Verification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E01** | Phase 1 Bilateral Ambiguity | Horizontal floor acceleration differences $< 0.14\%$ between State A and State B | [`results/ill_posedness_metrics.json`](file:///Users/rahul/inverse-fno-damage/results/ill_posedness_metrics.json) | `python scripts/phase1_ill_posedness.py` | OpenSeesPy dynamic transient simulation of 3-story frame; $L_2$ norm difference $< 0.0014$. |
| **E02** | Phase 2 Forward FNO Accuracy & Metric Distinctions | Waveform tracking: Rel-$L_2 = 36.71\% \pm 8.42\%$, Pearson $\rho = 0.9030 \pm 0.041$, peak error $13.47\%$; Baseline parameter MAE $\approx 0.0381$ ($3.81\%$) | [`results/forward_fno_metrics.json`](file:///Users/rahul/inverse-fno-damage/results/forward_fno_metrics.json), [`results/phase6/ablation_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6/ablation_results.json) | `python src/forward_fno_model.py` | Evaluated on physical acceleration units ($\text{m/s}^2$) across 32 held-out simulations; parameter-space MAE evaluated across damaged elements. |
| **E03** | Phase 3 MSE Mean-Seeking Collapse | Naive MSE inverse models collapse to diffuse near-zero predictions; sparsity/TV priors enforce localization | [`results/regularization_ablation.json`](file:///Users/rahul/inverse-fno-damage/results/regularization_ablation.json) | `python scripts/phase3_regularization_ablation.py` | Mean absolute error and support recovery across regularization conditions ($\lambda_{\text{sparse}}, \lambda_{\text{TV}}, \lambda_{\text{cycle}}$). |
| **E04** | Phase 4 Sensor Sparsity Sweep | Graceful degradation of severity MAE with sparser sensors; floor horizontal sensors cannot differentiate columns | [`results/sensor_sparsity_sweep.json`](file:///Users/rahul/inverse-fno-damage/results/sensor_sparsity_sweep.json) | `python scripts/phase4_sensor_sparsity.py` | 1-sensor, 2-sensor, and 3-sensor floor deployments; single-seed exploratory sweep. |
| **E05** | Phase 5 Sensitivity Near-Null Space | S0 horizontal floor sensing Jacobian aligns with $v_{AB}$; directional Fisher sensitivity $\sqrt{I_{AB}} \approx 5.6$ | [`results/observability/symmetry_baseline.json`](file:///Users/rahul/inverse-fno-damage/results/observability/symmetry_baseline.json) | `python scripts/phase5_observability.py` | SVD of finite-difference Jacobian $J = \partial Y / \partial d$; projection along $v_{AB} = (d_A - d_B)/\|d_A - d_B\|$. |
| **E06** | Phase 5.5 Directional Fisher Hierarchy | Noise-whitened directional Fisher sensitivity $\sqrt{I_{AB}}$: $S0 = 5.6$, $S1 = 1829.6$, $S2 = 1344.2$, $S4 = 2270.5$ ($>300\times$ ratio to S0 under prescribed noise model) | [`results/observability/phase5_5_forensic_audit.json`](file:///Users/rahul/inverse-fno-damage/results/observability/phase5_5_forensic_audit.json) | `python scripts/phase5_5_forensic_audit.py` | Noise-whitened Jacobian $J_w = \Sigma^{-1/2} J$; directional Fisher sensitivity $I_{AB} = v_{AB}^T J_w^T J_w v_{AB}$. |
| **E07** | Phase 6 Unpaired G-FNO Collapse | DualStreamGFNO under unpaired supervision achieves only $50.0\%$ bilateral attribution accuracy | [`results/phase6/bilateral_benchmark_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6/bilateral_benchmark_results.json) | `python scripts/phase6_bilateral_test.py` | Evaluated on canonical bilateral states with standard unpaired loss; attribution remains at chance level. |
| **E08** | Phase 6.1 Gradient Dominance | Horizontal floor accels carry $98\%+$ of early gradient norm; axial strain and vertical signals underutilized | [`results/phase6/sensor_utilization.json`](file:///Users/rahul/inverse-fno-damage/results/phase6/sensor_utilization.json) | `python scripts/phase6_1_forensic_audit.py` | Backpropagation gradient norm tracing across temporal sensor channels during early training epochs. |
| **E09** | Phase 6.2 S4 Attribution Success | Bilateral attribution accuracy $90.0\%$ ($27/30$, $p = 9.0 \times 10^{-6}$, 95% CI: $[73.5\%, 97.9\%]$) | [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) | `python scripts/phase6_2_bilateral_learning.py` | $N=30$ evaluations across 15 disjoint held-out validation pairs; two-sided exact binomial test vs chance ($p_0=0.5$). |
| **E10** | Phase 6.2 S1 Attribution Success | Bilateral attribution accuracy $80.0\%$ ($24/30$, $p = 0.0014$, 95% CI: $[61.4\%, 92.3\%]$) | [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) | `python scripts/phase6_2_bilateral_learning.py` | $N=30$ evaluations across 15 disjoint held-out validation pairs; two-sided exact binomial test vs chance ($p_0=0.5$). |
| **E11** | Phase 6.2 S0 Physical Baseline | Bilateral attribution accuracy $50.0\%$ ($15/30$, $p = 1.0$, 95% CI: $[31.3\%, 68.7\%]$), separation $= 9.04 \times 10^{-6}$ | [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) | `python scripts/phase6_2_bilateral_learning.py` | $N=30$ evaluations; confirmed that pairwise loss cannot overcome physical unobservability. |
| **E12** | Phase 6.2 S2 Learnability Gap | Physical observability is necessary but insufficient for neural learnability: $\sqrt{I_{AB}} = 1344.2$ yet attribution is $50.0\%$ ($15/30$, $p = 1.0$) due to strain gradient dominance | [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) | `python scripts/phase6_2_bilateral_learning.py` | $N=30$ evaluations; demonstrates that high Fisher sensitivity does not guarantee neural learnability. |
| **E13** | Phase 6.2 Recovered Separation | Mean predicted separation $\|\Delta \hat{d}\| = 0.1671$ ($39.40\%$ of true $\|d_A - d_B\| = 0.4243$) | [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) | `python scripts/phase6_2_bilateral_learning.py` | Dimensionless Euclidean norm in damage space; matches margin loss target $m = 0.15$. |
| **E14** | Phase 7 Clean Dataset Accounting | Exactly 530 validated finite-element dynamic simulations implemented in OpenSeesPy across 120 PEER ground motions | [`results/phase7/statistics/dataset_integrity_audit.json`](file:///Users/rahul/inverse-fno-damage/results/phase7/statistics/dataset_integrity_audit.json) | `python scripts/phase7_dataset_generator.py` | Physical count: 90 P1 train, 140 P2 train, 60 val, 240 test (4 levels $\times$ 60 runs = 120 pairs); 100% disjoint. |
| **E15** | Phase 7 Pristine Feature Contract | $E_{\text{norm}} = 1.0$ across all node/edge tensors; zero target-correlated damage leakage | [`tests/test_phase7_feature_leakage_contract.py`](file:///Users/rahul/inverse-fno-damage/tests/test_phase7_feature_leakage_contract.py) | `pytest tests/test_phase7_feature_leakage_contract.py` | Automated regression tests verifying graph edge features invariant to damage injection. |
| **E16** | Phase 7 Multi-Structure Baseline | Clean PROPOSED model under P2 S4: Acc $= 50.0\%$, $\cos = +0.428 \pm 0.294$, L3 $\cos = +0.393$ | [`results/experiments/FINAL_PHASE7_CLEAN_AUDIT.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/FINAL_PHASE7_CLEAN_AUDIT.json) | `python scripts/phase7_clean_rerun.py` | Evaluated across seeds 42, 101, 2024; clean nominal baseline before budget scaling. |
| **E17** | Training Budget: 12 Epochs | P2 S4: Acc $= 50.0\% \pm 0.0\%$, Cosine $= +0.467 \pm 0.037$, Separation $= 2.68 \times 10^{-6}$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Evaluated on $N=30$ strictly held-out bilateral test pairs; aggregate reports mean $\pm$ SD across 3 seeds. |
| **E18** | Training Budget: 25 Epochs | P2 S4: Acc $= 50.0\% \pm 0.0\%$, Cosine $= +0.852 \pm 0.115$, Separation $= 7.68 \times 10^{-6}$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Steep rise in directional torque; discrete attribution remains flat at chance level. |
| **E19** | Training Budget: 50 Epochs | P2 S4: Acc $= 50.0\% \pm 0.0\%$, Cosine $= +0.897 \pm 0.038$, Separation $= 1.23 \times 10^{-5}$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Peak mean directional alignment across seeds; separation remains 4 orders of magnitude below true damage. |
| **E20** | Training Budget: 100 Epochs | Within tested 12–100 epoch budgets, increasing optimization budget refines alignment (up to $+0.996$) but does not recover finite separation ($\sim 1.83 \times 10^{-5}$) | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Individual seeds reach $+0.993$ (s42) and $+0.996$ (s2024); discrete attribution strictly flat at 50.0%. |
| **E21** | Level 2A (Interpolation) OOD | Unseen frames $B_{\text{int\_1}}, B_{\text{int\_2}}$: Acc $= 50.0\% \pm 0.0\%$, Cosine $= +0.841 \pm 0.216$, Sep $= 1.92 \times 10^{-5}$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Evaluated across 30 disjoint held-out test earthquakes (15 per frame); zero-shot parameter interpolation. |
| **E22** | Level 2B (Extrapolation) OOD | Extrapolated frames $B_{\text{ext\_soft}}, B_{\text{ext\_stiff}}$: Acc $= 50.56\% \pm 0.79\%$, Cosine $= +0.813 \pm 0.204$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Evaluated across 30 disjoint held-out test earthquakes; parameters $\pm 20\%$ outside training hull. |
| **E23** | Level 3 (Topological OOD) | 4-story frame $C_{\text{4story}}$ (10 nodes, 12 elements): Acc $= 50.0\%$, Cosine $= +0.798 \pm 0.199$ | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json) | `python scripts/phase7_training_budget_experiment.py` | Evaluated across all 30 test earthquakes; zero-shot topological transfer (Seeds 42 & 2024 reach $+0.921, +0.955$). |
| **E24** | Direction–Magnitude Decoupling | Continuous directional sensitivity transfers zero-shot, while finite damage separation collapses to $\sim 10^{-5}$ and discrete attribution remains at chance ($50.0\%$) | [`results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.md`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.md) | `python scripts/phase7_training_budget_experiment.py` | Empirical phenomenon formalizing the divergence between continuous orientation learning and macroscopic output scaling. |
| **E25** | Automated Quality Gates | 74 unit tests passing; Guyan reduction tolerance verified; pre-flight ship audit green | [`scripts/gstack.py`](file:///Users/rahul/inverse-fno-damage/scripts/gstack.py) | `python scripts/gstack.py ship` | Full test suite encompassing OpenSees mechanics, SVD, Fisher, G-FNO, pair loss, and leakage contracts. |

---

## 3. Structural Configurations Inventory

All structural frames used across Phase 1 through Phase 7 are deterministically generated via `src/structure_variants.py`:

| Structural Identifier | Role in Pipeline | Number of Stories | Number of Joints | Number of Elements | Bay Width $L$ ($m$) | Story Height $h$ ($m$) | Elastic Modulus $E$ ($Pa$) | Mass Density $\rho$ ($kg/m^3$) | First Fundamental Frequency $f_1$ ($Hz$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SOURCE_A** | Single-Structure Benchmark & Level 1 ID | 3 | 8 | 9 | 6.0 | 3.5 | $2.0 \times 10^{11}$ | 7850.0 | 2.064 |
| **B_train_1** | P2 Multi-Structure Training Diversity | 3 | 8 | 9 | 5.5 | 3.3 | $2.1 \times 10^{11}$ | 7800.0 | 2.305 |
| **B_train_2** | P2 Multi-Structure Training Diversity | 3 | 8 | 9 | 6.5 | 3.7 | $1.9 \times 10^{11}$ | 7900.0 | 1.871 |
| **B_train_3** | P2 Multi-Structure Training Diversity | 3 | 8 | 9 | 5.8 | 3.6 | $2.05 \times 10^{11}$ | 7820.0 | 2.069 |
| **B_train_4** | P2 Multi-Structure Training Diversity | 3 | 8 | 9 | 6.2 | 3.4 | $1.95 \times 10^{11}$ | 7880.0 | 2.079 |
| **B_int_1** | Level 2A Parametric Interpolation Test | 3 | 8 | 9 | 5.7 | 3.4 | $2.02 \times 10^{11}$ | 7840.0 | 2.189 |
| **B_int_2** | Level 2A Parametric Interpolation Test | 3 | 8 | 9 | 6.3 | 3.6 | $1.98 \times 10^{11}$ | 7860.0 | 1.961 |
| **B_ext_soft** | Level 2B Parametric Extrapolation Test | 3 | 8 | 9 | 7.0 | 4.0 | $1.7 \times 10^{11}$ | 8000.0 | 1.559 |
| **B_ext_stiff** | Level 2B Parametric Extrapolation Test | 3 | 8 | 9 | 5.0 | 3.0 | $2.3 \times 10^{11}$ | 7700.0 | 2.768 |
| **C_4story** | Level 3 Structural Topology OOD Test | 4 | 10 | 12 | 6.0 | 3.5 | $2.0 \times 10^{11}$ | 7850.0 | 1.542 |

---

## 4. Sensor Configurations Inventory

Defined in `src/observability.py` and `src/structure_variants.py`:

| Sensor Suite | Modality Channels | Spatial Locations | Total Channels (3-Story Frame) | Noise-Whitened Directional Fisher Sensitivity $\sqrt{I_{AB}}$ | Physical Observability Status | Single-Structure Attribution (Phase 6.2) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **S0** | Horizontal Floor Acceleration | Floors 1, 2, Roof (centerline nodes) | 3 | 5.6 | **Unobservable** (near-null space) | $50.0\%$ ($15/30$, $p=1.0$) |
| **S1** | Horizontal + Vertical Floor Accel | Horizontal (F1, F2, RF) + Vertical (joint nodes) | 9 | 1829.6 | **Observable** ($>325\times$ gain over S0) | $\mathbf{80.0\%}$ ($24/30$, $p=0.0014$) |
| **S2** | Horizontal Accel + Column Axial Strain | Horizontal (F1, F2, RF) + Ground Columns (left/right) | 9 | 1344.2 | **Observable** ($>240\times$ gain over S0) | $50.0\%$ ($15/30$, $p=1.0$; learnability gap) |
| **S4** | Full Multimodal Union | Horizontal + Vertical + Axial Strain + Rocking | 15 | 2270.5 | **Observable** ($>400\times$ gain over S0) | $\mathbf{90.0\%}$ ($27/30$, $p=9.0\times 10^{-6}$) |

---

## 5. Statistical Protocol Integrity Check

1. **Statistical Unit:**
   The primary unit of statistical inference is the **earthquake bilateral evaluation case** ($N = 30$ evaluations across 15 held-out pairs in Phase 6.2; $N = 30$ independent held-out bilateral pairs in Phase 7).
2. **Pseudoreplication Prohibition:**
   Multiple random seeds ($42, 101, 2024$) are treated strictly as **training replications along continuous optimization trajectories**, NOT as independent statistical observations. At no point are 30 pairs $\times$ 3 seeds pooled to claim $N = 90$.
3. **Hypothesis Testing:**
   All $p$-values are computed using the exact two-sided binomial test under the null hypothesis of pure chance ($p_0 = 0.50$):
   $$p = 2 \times \sum_{k=k_{\text{obs}}}^{N} \binom{N}{k} p_0^k (1-p_0)^{N-k}$$
   For $S4$ ($27/30$ successes), $p = 9.0 \times 10^{-6}$.
4. **Confidence Intervals:**
   All binary confidence intervals are computed using the exact Clopper-Pearson method at the $95\%$ two-sided level.

---
*Matrix verified by Inverse-FNO-Damage Lead Auditor against active repository artifacts.*
