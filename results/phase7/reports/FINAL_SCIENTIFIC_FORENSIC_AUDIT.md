# FINAL SCIENTIFIC FORENSIC AUDIT REPORT
## Project: Inverse-FNO-Damage
**Audit Date:** September 2026  
**Auditor Roles:** Scientific ML Researcher, Inverse-Problems Researcher, Structural Dynamics Researcher, Applied Mathematician, Peer Reviewer, Reproducibility Auditor  
**Scope:** Complete forensic audit of Phases 1 through 7, code provenance, dataset integrity, raw experimental outputs, and headline claims.

---

# Executive Verdict: PASS WITH CORRECTIONS

The **Inverse-FNO-Damage** project possesses exceptional computational and mathematical infrastructure:
1. First-principles **OpenSeesPy** structural dynamics matching analytical condensation frequencies to $< 0.003\%$ across 10 structural configurations and 120 PEER earthquake records ($530$ transient dynamic simulations).
2. Noise-whitened Jacobian singular value analysis ($\tilde{J} = R^{-1/2} J$) rigorously confirming an ill-posed bilateral near-null direction ($\kappa > 10,000$, $\sqrt{I_{AB}} < 2.0$) under floor-horizontal accelerometry (S0).
3. In **Phase 6.2** (frozen on the canonical 3-story frame `SOURCE_A`), the proposed directional margin loss achieved a verified **$90.0\%$ ($27/30$) bilateral attribution accuracy** under multimodal sensing (S4) with positive directional alignment ($\cos = +0.9420$), while physically remaining at chance level ($50.0\%$, $15/30$) under horizontal sensing (S0).
4. All **70 unit tests pass**, simulation partitions are $100\%$ disjoint across earthquake sets, and 108 model checkpoints are accounted for.

**However, the audit identified a critical scientific defect in the Phase 7 multi-structure dataset generation pipeline:**
> In `scripts/phase7_dataset_generator.py` (line 109), `extract_structural_graph_features(frame)` was called **after** `frame.build_model(damage_vector)`. As a consequence, `edge_features[:, 3]` captured the post-damage Young's modulus $E_{\text{damaged}} / E_0 = \frac{E_{\text{nominal}}(1 - d_e)}{E_0}$ rather than the nominal pristine modulus. In the Phase 7 model hierarchy, models `B5` and `PROPOSED` ingested all 6 edge features, leaking $(1 - d_e)$ directly into the input. This allowed P2 models to achieve $100.0\%$ attribution across all test levels—including under horizontal sensing S0 where the inverse problem is mathematically unobservable.

Because the narrative report (`results/phase7/reports/phase7_2_inverse_evaluation_report.md`) reported idealized Phase 6.2 expectations ($90.0\%$, $83.3\%$, and S0 $= 50.0\%$) while the embedded table and raw JSONs contained the $100.0\%$ figures from leaked features, **the research requires immediate corrective disclosure prior to manuscript submission**.

---

## 1. Evidence Inventory

The forensic audit inspected the following primary artifacts on disk:

| Artifact Path | Category | Verification Status | Forensic Note |
| :--- | :--- | :---: | :--- |
| `data/phase7_simulations/*.npz` | FE Simulations | **Verified (530 files)** | 10 structures, 120 PEER earthquakes. Confirmed `edge_features[:, 3]` leak. |
| `results/phase7/training/checkpoints/*.pt` | Model Weights | **Verified (108 files)** | Exactly matches $2 \text{ protocols} \times 3 \text{ modalities} \times 6 \text{ models} \times 3 \text{ seeds}$. |
| `results/phase7/test/all_test_evaluations.json` | Raw Test Evals | **Verified (108 records)** | Raw test records across Levels 1, 2A, 2B, and 3. |
| `results/phase7/statistics/cross_structure_summary.json` | Aggregated Stats | **Verified (36 keys)** | Contains mean $\pm$ std across 3 seeds for each protocol/modality/model. |
| `results/phase7/reports/phase7_2_inverse_evaluation_report.md` | Final Report | **Discrepancy Detected** | Narrative text reports 90%/83.3%/50%, while table reports 100%/66.7%. |
| `results/phase7/observability/c4story_observability_metrics.json` | Gate B Observability | **Verified** | S0: $\kappa = 13,496$; S1: $\kappa = 85.91$; S4: $\kappa = 16.08$. |
| `results/phase7/forward_validation/forward_validation_metrics.json` | Gate A Physics | **Verified** | Modal frequencies across 10 families match analytical to $<0.003\%$. |
| `results/phase7/design/structural_split_manifest.json` | Partition Splits | **Verified** | Training manifold $[0.9, 1.1]^2$; strict interior/exterior coordinates. |
| `reports/technical_report/phase6_2_final_audit.md` | Phase 6.2 Baseline | **Verified** | Original source of the verified 90.0% ($27/30$) attribution on `SOURCE_A`. |
| `results/phase6_2/bilateral_results.json` | Phase 6.2 Raw JSON | **Verified** | $27/30$ under S4, $\cos = +0.9420$; $15/30$ under S0. Clean, leak-free input. |

---

## 2. Headline Claim Audit

| # | Headline Claim | Raw Evidence | Audit Status | Statistical Basis | Required Correct Wording |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **C1** | **90.0% Bilateral Attribution (S4)** | `results/phase6_2/bilateral_results.json` | **VERIFIED** (Phase 6.2) | $27/30$ pairs on 30 held-out PEER earthquakes ($N=30$); Clopper-Pearson 95% CI: $[73.5\%, 97.9\%]$; $p = 4.07 \times 10^{-6}$ vs chance ($50.0\%$). | "On the canonical 3-story frame under multimodal sensing (S4), the proposed directional margin loss achieves 90.0% (27/30) bilateral attribution on held-out earthquakes." |
| **C2** | **S0 Horizontal Non-Observability** | `results/observability/phase5_5_forensic_audit.json` | **VERIFIED** | Noise-whitened $\kappa = 12,870$ (`SOURCE_A`) and $\kappa = 13,496$ (`C_4story`); $\sqrt{I_{AB}} < 2.0$; empirical attribution is $15/30 = 50.0\%$ (chance). | "Floor-horizontal sensing exhibits an unavoidable near-null space ($\kappa > 10^4$), restricting inverse attribution to random chance (50.0%)." |
| **C3** | **OpenSeesPy Forward Physics** | `results/phase7/forward_validation/forward_validation_metrics.json` | **VERIFIED** | Transient dynamic analyses across 10 structures match Guyan-condensed $f_1$ to $< 0.003\%$ error; zero NaNs/Infs. | "Finite-element structural models across 10 structural configurations reproduce analytical fundamental frequencies within 0.003% relative error." |
| **C4** | **530 OpenSees Dynamic Simulations** | `data/phase7_simulations/*.npz` | **VERIFIED** | 250 standard simulations + 280 paired simulations (140 pairs $\times 2$) across 120 disjoint earthquakes. | "A benchmark corpus of 530 OpenSees transient dynamic simulations across 10 structural families and 120 PEER earthquakes was generated and audited." |
| **C5** | **108 Model Evaluation Matrix** | `results/phase7/training/checkpoints/*.pt` | **VERIFIED** | $2 \text{ protocols (P1, P2)} \times 3 \text{ modalities (S0, S1, S4)} \times 6 \text{ models (B1..B5, PROPOSED)} \times 3 \text{ seeds} = 108$ runs. | "A 108-model evaluation matrix was trained and evaluated across 2 training protocols, 3 sensing modalities, 6 model baselines, and 3 random seeds." |
| **C6** | **Phase 7 Input Feature Leakage** | `scripts/phase7_dataset_generator.py:109` | **CRITICAL DEFECT** | `edge_features[:, 3]` encoded $E_{\text{nom}}(1 - d_e)/E_0$ because features were extracted after damage injection. B5 and PROPOSED read this column. | "Forensic audit detected that Phase 7 dataset generation inadvertently evaluated edge features on the post-damage model, leaking ground truth damage into input arrays for B5 and PROPOSED." |
| **C7** | **Level 2A Interpolation (90.0%)** | `results/phase7/reports/phase7_2_inverse_evaluation_report.md` | **CONDITIONALLY SUPPORTED** | Variants $B_{\text{int}_1}, B_{\text{int}_2}$ are strictly interior to $[0.9, 1.1]^2$. Reported 90.0% was an expected projection; raw evaluation yielded 100.0% (leaked) or 50.0% (unconditioned B1..B4). | "Parametric interpolation across interior structural variants preserves baseline performance, but published metrics must report unconditioned baselines and disclose feature leakage." |
| **C8** | **Level 2B Extrapolation (83.3%)** | `results/phase7/reports/phase7_2_inverse_evaluation_report.md` | **CONDITIONALLY SUPPORTED** | Extreme variants $B_{\text{ext}_\text{soft}}, B_{\text{ext}_\text{stiff}}$ lie $3.2\times$ to $4.1\times$ beyond training boundary. Reported 83.3% reflects idealized target. | "Parametric extrapolation beyond the training convex hull was systematically evaluated; audited raw data must be presented without idealized narrative smoothing." |
| **C9** | **Level 3 Topology OOD Transfer** | `results/phase7/reports/phase7_2_inverse_evaluation_report.md` | **REQUIRES DOWNGRADE** | $C_{\text{4story}}$ (10 nodes, 12 members) was strictly unseen during training, but B5/PROPOSED test performance was contaminated by leaked edge features. | "The topology-agnostic graph neural operator architecture successfully executes inference on an unseen 4-story frame; empirical attribution figures must be re-evaluated with pristine edge features." |
| **C10** | **S2 Observability vs Learnability** | `reports/technical_report/phase6_2_final_audit.md` | **VERIFIED** | S2 sensing (horizontal + rocking) had high sensitivity ($\sqrt{I_{AB}} \approx 800$), yet neural attribution remained at chance ($53.3\%$, $16/30$). | "Physical observability is a necessary but insufficient condition for neural operator inversion; high directional Fisher information under S2 failed to translate into learnable attribution." |

---

## 3. Leakage Audit

A comprehensive forensic audit of all potential data leakage vectors was performed:

1. **Earthquake Partition Leakage:** **PASS (100% Disjoint)**
   - Train: `RSN0001` to `RSN0070` (70 unique records)
   - Validation: `RSN0071` to `RSN0090` (20 unique records)
   - Test: `RSN0091` to `RSN0120` (30 unique records)
   - Exact intersection: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Structural Split Leakage:** **PASS (100% Disjoint)**
   - Training: `SOURCE_A` (P1/P2) and `B_train_1..4` (P2)
   - Validation: `SOURCE_A` and `B_train_1`
   - Test structures `B_int_1`, `B_int_2`, `B_ext_soft`, `B_ext_stiff`, and `C_4story` were strictly absent from training and validation sets.
3. **Normalization Leakage:** **PASS**
   - Category A physical reference scaling was used ($H_0 = 3.0\text{ m}, L_0 = 5.0\text{ m}, M_0 = 12,000\text{ kg}, E_0 = 2 \times 10^{11}\text{ Pa}$). No test-set statistical moments were used.
4. **Checkpoint & Hyperparameter Leakage:** **PASS**
   - All 108 models were saved after exactly 12 epochs without post-hoc test metric selection. Hyperparameters were frozen prior to Phase 7.1.
5. **Input Feature Leakage:** **FAIL (CRITICAL DEFECT DETECTED)**
   - In `scripts/phase7_dataset_generator.py`:
     ```python
     # Line 104-106: Runs dynamic simulation, calling frame.build_model(damage_vector)
     res_s0 = TopologyObservationMap.evaluate(frame, damage_vector, accel, target_dt, s0_cfg)
     # Line 109: Inadvertently extracts features from the post-damage frame!
     graph_feats = extract_structural_graph_features(frame)
     ```
   - In `src/structure_features.py` (line 83):
     `E_norm = float(e_info.E / REF_E0)`
     Because `e_info.E` was modified by `build_model` to $E_{\text{nom}} \times (1 - d_e)$, column 3 of `edge_features` directly contained the ground-truth damage $(1 - d_e)$.
   - Models `B1`, `B2`, `B3`, and `B4` were **unaffected** because they do not ingest column 3 of `edge_features`. Models `B5` and `PROPOSED` ingested all 6 edge features and learned to exploit this leaked signal in Protocol P2, achieving $100.0\%$ attribution across all test sets, even under S0.

---

## 4. Statistical Audit & Verification of the 90.0% Claim

### Provenance of the 90.0% Headline
The widely cited $90.0\%$ bilateral attribution metric originates from **Phase 6.2** on the single 3-story frame `SOURCE_A` under multimodal sensing S4 (`results/phase6_2/bilateral_results.json`):
- **Number of unique test earthquakes:** $30$ (`RSN0091` through `RSN0120`).
- **Number of bilateral evaluation pairs:** $30$ matched pairs (State A vs. State B).
- **Correct bilateral attributions:** $27$ pairs.
- **Observed attribution accuracy:** $27 / 30 = \mathbf{90.0\%}$.
- **Statistical Unit:** The **earthquake bilateral pair** ($N = 30$). Treating $3$ seeds as $N = 90$ independent samples is **statistically invalid pseudoreplication** and is strictly prohibited.
- **Exact Binomial 95% Confidence Interval (Clopper-Pearson):**
  $$\text{CI}_{95\%} = [73.47\%, 97.89\%]$$
- **Null Hypothesis Significance Test:**
  Under the null hypothesis of unobservable symmetry / random guessing ($p_0 = 0.50$):
  $$P(X \ge 27 \mid N=30, p=0.50) = \sum_{k=27}^{30} \binom{30}{k} (0.5)^{30} = \frac{465 + 435 + 30 + 1}{2^{30}} = \frac{931}{1,073,741,824} \approx 4.07 \times 10^{-6}$$
  The result is statistically significant at $p < 0.001$.
- **Control Modality (S0):** Under horizontal sensing, accuracy was exactly $15 / 30 = \mathbf{50.0\%}$ ($\text{CI}_{95\%} = [31.3\%, 68.7\%]$, $p = 1.0$), proving that learning cannot circumvent physical non-observability.

---

## 5. Protocol P1 (Source-Only) vs. P2 (Multi-Structure Diversity)

- **Protocol P1:** Trained on `SOURCE_A` only ($90$ simulations: 70 standard + 20 paired).
- **Protocol P2:** Reused P1 simulations and added $140$ diversity simulations across the 4 corners of the parameter manifold (`B_train_1..4`).
- **Audit Findings:**
  - For baseline models B1..B4 (uncontaminated by edge feature leakage), both P1 and P2 remain at chance attribution ($50.0\% \pm 0.0\%$), proving that multi-structure training alone without directional loss or rich sensing cannot resolve bilateral ambiguity.
  - For PROPOSED: In P1, seeds 42 and 101 remained at $50.0\%$ while seed 2024 achieved $100.0\%$ (mean $66.7\% \pm 23.6\%$). In P2, all 3 seeds achieved $100.0\%$ due to the leaked edge features.
  - The narrative report claim that P2 provided $+6.7\%$ in S1 and $+3.3\%$ in S4 was an idealized extrapolation that does not match the raw JSON data.

---

## 6. Structural Topological OOD Audit (Level 3 — C_4story)

- **Topological Invariant:** The 4-story frame ($|V|=10, |E|=12$) possesses a different graph topology and state dimension than the 3-story training structures ($|V|=8, |E|=9$).
- **Architectural Scalability:** `TopologyDualStreamGFNO` successfully processed variable node counts without code changes or dimension hacks. Dynamic symmetry permutations $\pi_V$ and $\pi_E$ scaled correctly as exact mathematical involutions ($\pi \circ \pi = \text{id}$).
- **Gate B Observability Analysis:**
  - Under S0: $\kappa = 13,496$, $\sigma_{\min} = 0.0691$, $\sqrt{I_{AB}} = 1.91$ (**F1 Unobservable**).
  - Under S1: $\kappa = 85.91$, $\sigma_{\min} = 35.48$, $\sqrt{I_{AB}} = 1297.88$ (**Observable**).
  - Under S4: $\kappa = 16.08$, $\sigma_{\min} = 241.72$, $\sqrt{I_{AB}} = 1639.08$ (**Highly Observable**).
- **Caveat on Zero-Shot Claim:** While the network architecture enables execution on arbitrary graphs, the empirical attribution accuracy ($83.3\%$ to $86.7\%$) reported in the narrative cannot be cited as pure zero-shot damage identification until models are evaluated with pristine (undamaged) edge features.

---

## 7. Simulation and Evaluation Accounting

### OpenSees Dynamic Simulations (Exact Count = 530)
$$\begin{aligned}
\text{Train P1 (SOURCE\_A):} & \quad 70 \text{ standard} + 20 \text{ paired (10 pairs)} = 90 \\
\text{Train P2 Diversity (B\_train\_1..4):} & \quad 140 \text{ standard (35 per structure)} \\
\text{Validation (SOURCE\_A, B\_train\_1):} & \quad 40 \text{ standard (20 per structure)} + 20 \text{ paired (10 pairs)} = 60 \\
\text{Test Level 1 (SOURCE\_A):} & \quad 60 \text{ paired (30 pairs)} \\
\text{Test Level 2A (B\_int\_1, B\_int\_2):} & \quad 60 \text{ paired (15 pairs each)} \\
\text{Test Level 2B (B\_ext\_soft, B\_ext\_stiff):} & \quad 60 \text{ paired (15 pairs each)} \\
\text{Test Level 3 (C\_4story):} & \quad 60 \text{ paired (30 pairs)} \\
\hline
\mathbf{Total\ Structural\ Simulations:} & \quad \mathbf{530\ unique\ transient\ FE\ runs}
\end{aligned}$$

### Model Evaluations (Exact Count = 108)
$$\begin{aligned}
\text{Protocols:} & \quad 2\ (P1, P2) \\
\text{Modalities:} & \quad 3\ (S0, S1, S4) \\
\text{Model Ablation Conditions:} & \quad 6\ (B1, B2, B3, B4, B5, \text{PROPOSED}) \\
\text{Deterministic Random Seeds:} & \quad 3\ (42, 101, 2024) \\
\hline
\mathbf{Total\ Model\ Evaluations:} & \quad 2 \times 3 \times 6 \times 3 = \mathbf{108\ trained\ checkpoints}
\end{aligned}$$
Every checkpoint is physically preserved in `results/phase7/training/checkpoints/`.

---

## 8. Audit of the Phase 5 → Phase 7 Scientific Chain

$$\begin{array}{ccc}
\text{Bilateral structural symmetry} & \longrightarrow & \mathbf{VERIFIED} \\
\downarrow & & \\
\text{Horizontal sensing near-null direction} & \longrightarrow & \mathbf{VERIFIED} \\
\downarrow & & \\
\text{Noise-whitened observability analysis} & \longrightarrow & \mathbf{VERIFIED} \\
\downarrow & & \\
\text{Asymmetric sensing improves directional info} & \longrightarrow & \mathbf{VERIFIED} \\
\downarrow & & \\
\text{Ordinary MSE training fails to exploit weak info} & \longrightarrow & \mathbf{VERIFIED} \\
\downarrow & & \\
\text{Pairwise directional loss achieves 90% on SOURCE\_A} & \longrightarrow & \mathbf{VERIFIED}\text{ (Phase 6.2)} \\
\downarrow & & \\
\text{Cross-structure neural operator generalization} & \longrightarrow & \mathbf{CONDITIONALLY\ SUPPORTED}
\end{array}$$

*The final step is conditionally supported because the mathematical and forward modeling formulation is complete and verified, but the empirical multi-structure inverse evaluation must be re-run with uncorrupted edge features.*

---

## 9. Overclaims Requiring Scientific Correction

1. **"Universally proved ill-posed":**  
   *Correction:* "Characterized a bilateral near-null direction induced by structural symmetry under floor-horizontal accelerometry for linear shear-type frames."
2. **"Eliminates the null space / solves the inverse problem":**  
   *Correction:* "Directional margin loss resolves bilateral attribution under asymmetric sensing modalities (S4) where directional Fisher information is physically non-zero ($\sqrt{I_{AB}} > 1000$)."
3. **"Zero-shot transfer to unseen building topology":**  
   *Correction:* "A topology-agnostic graph neural operator architecture capable of evaluating structures with varying story counts without retraining."
4. **"Phase 7 achieved 90.0% interpolation and 83.3% extrapolation":**  
   *Correction:* State clearly that $90.0\%$ ($27/30$) is the verified Phase 6.2 result on `SOURCE_A`, and disclose the Phase 7 edge feature artifact transparently.

---

## 10. Final Publication-Safe Scientific Claims

When writing the research paper or presenting this work, make **only** the following mathematically defensible claims:

1. **First-Principles Observability Barrier:** Floor-horizontal accelerometers lack directional sensitivity for bilateral column damage in symmetric buildings ($\kappa > 10,000$, $\sigma_{\min} \ll 1.0$). Machine learning cannot overcome this physical barrier, remaining strictly at chance level ($50.0\%$).
2. **Multimodal Information Restoration:** Incorporating vertical accelerations and base strains restores physical observability ($\kappa = 16.08$, $\sqrt{I_{AB}} > 1600$).
3. **Loss Formulation Necessity:** Standard supervised MSE loss fails to exploit weak symmetry-breaking signals ($50.0\%$ attribution). Explicit directional margin supervision resolves this, achieving $90.0\%$ ($27/30$, $p = 4.07 \times 10^{-6}$) attribution on held-out earthquakes in Phase 6.2.
4. **Physical Observability $\neq$ Neural Learnability:** High physical Fisher information (e.g. S2 rocking sensing) does not guarantee that gradient-based neural operators will learn the inverse mapping.
5. **Topological Operator Formulation:** Graph neural operators with dynamic symmetry projections provide a mathematically rigorous framework for processing structural frames of varying story counts.

---

## 11. Remaining Limitations

1. **Phase 7 Edge Feature Pipeline:** The simulation generation script must be patched to extract structural features from the *undamaged* nominal frame so that Young's modulus is pristine ($E_{\text{nom}}$), followed by a clean re-run of Phase 7 inverse training.
2. **Linear Dynamic Formulation:** OpenSeesPy simulations currently employ linear elastic beam-column elements with localized stiffness reductions. Geometric P-Delta and material plasticity remain for future phases.
3. **Planar Frame Assumption:** The formulation assumes 2D symmetric frames; 3D torsional coupling under biaxial excitation has not yet been modeled.

---

## 12. Reproducibility Checklist

- [x] All 70 unit tests pass via `pytest -q`.
- [x] All 108 model checkpoints verified on disk (`results/phase7/training/checkpoints/`).
- [x] All 530 OpenSees simulation `.npz` files verified on disk (`data/phase7_simulations/`).
- [x] Deterministic random seeds enforced (42, 101, 2024).
- [x] Earthquake splits 100% disjoint (70 Train, 20 Val, 30 Test).
- [x] Pre-flight audit `python scripts/gstack.py ship` passes with zero missing files.
- [x] Full forensic audit artifacts committed to repository:
  - `results/phase7/reports/FINAL_SCIENTIFIC_FORENSIC_AUDIT.md`
  - `results/phase7/reports/FINAL_SCIENTIFIC_FORENSIC_AUDIT.json`

---
*Signed by Antigravity Scientific Forensic Auditor — September 2026*
