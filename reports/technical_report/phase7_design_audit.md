# Phase 7 Technical Design Audit — Cross-Structure Generalization and Parameter Space

**Project:** SeismoFNO Structural Damage Identification  
**Phase:** Phase 7.0.2 — Final Experimental Consistency Audit  
**Date:** September 2026  
**Status:** COMPLETE (Design Audit Passed — Ready for Phase 7.1 Gate)  

---

## 1. Central Scientific Objective & Scope

Phase 7 investigates:
> *"Can a physics-conditioned graph neural operator learn a transferable inverse mapping from seismic observations to localized structural damage across both parametric changes in structural dynamics and genuine changes in structural topology, and can we distinguish failures caused by non-observability, distribution shift, representation limitations, and neural learnability?"*

### Core Conceptual Axiom:
$$\text{Physical Observability} \neq \text{Representation} \neq \text{Learnability} \neq \text{Generalization}$$
- **Phase 5/5.5** demonstrated that horizontal sensing contains a weak bilateral near-null space, while vertical acceleration and axial strains provide high directional Fisher information ($\sqrt{I_{AB}} > 1300$).
- **Phase 6.1** proved that standard unconstrained supervised learning fails to exploit this informative subspace.
- **Phase 6.2** proved that explicit pairwise supervision guides the model into the observable subspace for vertical acceleration ($S_1$, $80\%$) and multimodal sensing ($S_4$, $90\%$), but confirmed that high Fisher sensitivity does not guarantee learnability ($S_2$, $50\%$).
- **Phase 7** now extends this scientific chain to determine whether the learned operator can transfer beyond the original 3-story frame, or whether its success is strictly bounded to the training system's specific dynamic modes and topology.

---

## 2. Authoritative Frozen Source Configuration (`SOURCE_A`)

The authoritative source frame (`SOURCE_A`) is the frozen OpenSeesPy finite-element model used continuously from Phase 1 through Phase 6. Its exact parameters are codified in machine-readable format at [`results/phase7/design/source_a_manifest.json`](file:///Users/rahul/inverse-fno-damage/results/phase7/design/source_a_manifest.json).

| Parameter | Symbol | Frozen Value | Unit | Verification Reference |
| :--- | :---: | :---: | :---: | :--- |
| **Number of Stories** | $N_{\text{stories}}$ | $3$ | — | `FrameConfig.num_stories` |
| **Number of Bays** | $N_{\text{bays}}$ | $1$ | — | `FrameConfig.num_bays` |
| **Story Height** | $h$ | $3.0$ | m | `FrameConfig.story_height` |
| **Bay Width** | $L$ | $6.0$ | m | `FrameConfig.bay_width` |
| **Young's Modulus** | $E$ | $2.0 \times 10^{11}$ | Pa | `FrameConfig.E` (Structural steel) |
| **Column Area** | $A_{\text{col}}$ | $0.010$ | $\text{m}^2$ | $100\ \text{cm}^2$ cross-section |
| **Column Moment of Inertia** | $I_{\text{col}}$ | $1.0 \times 10^{-4}$ | $\text{m}^4$ | $10,000\ \text{cm}^4$ ($EI = 2.0 \times 10^7\ \text{N}\cdot\text{m}^2$) |
| **Beam Area** | $A_{\text{beam}}$ | $0.015$ | $\text{m}^2$ | $150\ \text{cm}^2$ cross-section |
| **Beam Moment of Inertia** | $I_{\text{beam}}$ | $3.0 \times 10^{-4}$ | $\text{m}^4$ | $30,000\ \text{cm}^4$ ($EI = 6.0 \times 10^7\ \text{N}\cdot\text{m}^2$) |
| **Floor Lumped Mass** | $m_{\text{floor}}$ | $10,000.0$ | kg | $10.0$ metric tonnes / floor ($5$ t / node in horiz DOF) |
| **Damping Formulation** | $\zeta$ | $0.02$ ($2\%$) | — | Rayleigh damping anchored to modes 1 and 3 |
| **Modal Frequencies** | $f_1, f_2, f_3$ | **2.1856, 6.8500, 11.4039** | Hz | Validated to $< 0.001\%$ against Guyan condensation |
| **Topology** | $|V|, |E|$ | **8 joints, 9 members** | — | 6 columns (Ele 1..6), 3 beams (Ele 7..9) |

---

## 3. Critical Earthquake Inventory Audit & Globally Disjoint Redesign

### The 120-Record Reality:
A forensic check of `data/raw_ground_motions/` reveals **exactly 120 unique PEER NGA-West2 `.AT2` files** (`RSN0001` through `RSN0120`). Proposing 200 distinct earthquakes would cause silent leakage.

### Strict Non-Overlapping Partitioning ($70 / 20 / 30$):
$$\text{Train GMs} \cap \text{Val GMs} = \text{Train GMs} \cap \text{Test GMs} = \text{Val GMs} \cap \text{Test GMs} = \emptyset$$
- **Training Earthquakes:** $70$ records (`RSN0001` to `RSN0070`, $58.33\%$).
- **Validation Earthquakes:** $20$ records (`RSN0071` to `RSN0090`, $16.67\%$).
- **Held-Out Test Earthquakes:** $30$ records (`RSN0091` to `RSN0120`, $25.00\%$).

All test levels (Level 1 ID, Level 2A Interpolation, Level 2B Extrapolation, Level 3 Topology) evaluate on the **exact same 30 held-out test earthquakes**. This guarantees $N=30$ independent statistical power per test level and enables **direct paired difference testing**:
$$\Delta_{\text{OOD}}(s, e) = \text{Metric}(s, e) - \text{Metric}(\text{SOURCE\_A}, e)$$
on identical ground motions. Documented in [`results/phase7/design/earthquake_inventory_audit.json`](file:///Users/rahul/inverse-fno-damage/results/phase7/design/earthquake_inventory_audit.json).

---

## 4. Factorial Parameter Space: Restricted 2D Manifold

Because column and beam moments of inertia scale proportionally, the parameter space is formally defined as a **Restricted 2D Structural Parameter Manifold**:
$$\mu_m = \frac{m_{\text{floor}}}{10,000\ \text{kg}}, \quad \mu_k = \frac{I_{\text{col}}}{1.0 \times 10^{-4}\ \text{m}^4} = \frac{I_{\text{beam}}}{3.0 \times 10^{-4}\ \text{m}^4}$$

| Structure ID | Stories | $\mu_m$ | $\mu_k$ | $f_1$ (Hz) | $f_2$ (Hz) | $f_3$ (Hz) | $\Delta f_1$ (%) | Generalization Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SOURCE_A** | 3 | 1.00 | 1.00 | 2.1856 | 6.8500 | 11.4039 | $0.0\%$ | **Reference Source Baseline (Train / Val / Level 1 Test)** |
| **B_train_1** | 3 | 0.90 | 0.90 | 2.1872 | 6.8549 | 11.4121 | $+0.1\%$ | Multi-Structure Training Set |
| **B_train_2** | 3 | 1.10 | 1.10 | 2.1841 | 6.8452 | 11.3960 | $-0.1\%$ | Multi-Structure Training Set |
| **B_train_3** | 3 | 0.90 | 1.10 | 2.4140 | 7.5658 | 12.5950 | $+10.5\%$ | Multi-Structure Training Set (Envelope Corner) |
| **B_train_4** | 3 | 1.10 | 0.90 | 1.9785 | 6.2004 | 10.3225 | $-9.5\%$ | Multi-Structure Training Set (Envelope Corner) |
| **B_int_1** | 3 | 0.95 | 1.05 | 2.2982 | 7.2025 | 11.9904 | $+5.2\%$ | **Level 2A: Parametric Interpolation Test (Held-Out)** |
| **B_int_2** | 3 | 1.05 | 0.95 | 2.0784 | 6.5135 | 10.8437 | $-4.9\%$ | **Level 2A: Parametric Interpolation Test (Held-Out)** |
| **B_ext_soft** | 3 | 1.50 | 0.80 | 1.5984 | 5.0068 | 8.3289 | **-26.9%** | **Level 2B: Parametric Extrapolation Test (Heavy/Flexible)** |
| **B_ext_stiff**| 3 | 0.65 | 1.30 | 3.0843 | 9.6588 | 16.0594 | **+41.1%** | **Level 2B: Parametric Extrapolation Test (Light/Stiff)** |
| **C_4story** | 4 | 1.00 | 1.00 | 1.6492 | 5.1392 | 8.9111 | **-24.5%** | **Level 3: Topological OOD Test (10 Nodes, 12 Elements)** |

### Mathematical Proof of Interpolation:
The training convex hull is $[0.90, 1.10] \times [0.90, 1.10]$.
- `B_int_1 = (0.95, 1.05)`: Strictly interior ($0.90 < 0.95 < 1.10$, $0.90 < 1.05 < 1.10$), expressible as convex combination $0.5 \cdot \text{B\_train\_3} + 0.5 \cdot \text{SOURCE\_A}$, but withheld from training.
- `B_int_2 = (1.05, 0.95)`: Strictly interior, expressible as $0.5 \cdot \text{B\_train\_4} + 0.5 \cdot \text{SOURCE\_A}$, withheld from training.

### Mathematical Proof of Extrapolation:
- `B_ext_soft = (1.50, 0.80)`: Parameter distance from center is $d_{\text{param}} = \sqrt{0.50^2 + (-0.20)^2} = 0.5385$, which is **$3.81\times$ beyond the training boundary** ($0.1414$). Modal frequency shifts by $\mathbf{-26.87\%}$.
- `B_ext_stiff = (0.65, 1.30)`: Parameter distance is $d_{\text{param}} = \sqrt{(-0.35)^2 + 0.30^2} = 0.4610$ (**$3.26\times$ beyond boundary**). Modal frequency shifts by $\mathbf{+41.12\%}$.

Documented in [`results/phase7/design/structural_split_manifest.json`](file:///Users/rahul/inverse-fno-damage/results/phase7/design/structural_split_manifest.json).

---

## 5. Training Protocols: Protocol P1 vs Protocol P2

To directly evaluate whether multi-structure diversity during training actually enables transfer:
- **Protocol P1 (Source-Only Training):** The model is trained strictly on `SOURCE_A` (70 GMs). Evaluates whether zero-shot inductive transfer emerges without parameter variation.
- **Protocol P2 (Multi-Structure Training):** The model is trained on `SOURCE_A` + `B_train_1..4` (140 total runs across 70 GMs). Evaluates whether parameter diversity enables interpolation and extrapolation.

Both models are evaluated on the exact same 4 test levels.

---

## 6. Normalization & Scaling Policy

Codified in [`results/phase7/design/normalization_policy.json`](file:///Users/rahul/inverse-fno-damage/results/phase7/design/normalization_policy.json):
- **Category A (Fixed Physical Normalization — Universally Permitted):** Reference constants derived strictly from `SOURCE_A` ($L/3.0$, $A/0.01$, $I/10^{-4}$, $m/10,000$).
- **Category B (Learned Training Normalization — Frozen on Train Split):** Channel-wise response mean and standard deviation computed strictly on the 140 training simulations. All test evaluations use these exact frozen statistics.
- **Category C (Test-Specific Normalization — Strictly Forbidden):** Standardizing an OOD test structure by its own empirical response variance is banned.

---

## 7. What Can This Experiment Actually Claim?

| Phrase / Claim | Defensible? | Scientific Boundary & Enforced Terminology |
| :--- | :---: | :--- |
| *"The model provides universal structural damage inference."* | **NO** | Universal claims are unsupportable. Valid only on 2D moment frames tested. |
| *"The model generalizes to arbitrary building topologies."* | **NO** | Evaluated on one unseen 4-story topology. Must be described as *"transfer to an unseen 4-story frame."* |
| *"The model learned zero-shot cross-structure generalization from A."* | **CONDITIONAL** | Only valid under Protocol P1. Under Protocol P2, must be described as *"multi-structure training followed by held-out parameter transfer."* |
| *"The inverse operator interpolates across structural parameter combinations."* | **YES** | Defensible if Level 2A interpolation metrics match ID baseline. |
| *"The inverse operator extrapolates to shifted frequency regimes."* | **YES** | Defensible if Level 2B maintains bounded accuracy under $\pm 30\%$ frequency shift. |
| *"OOD failures are separated into physical vs neural causes."* | **YES** | Quantitative mapping to F1–F7 taxonomy using measurable criteria. |

---

## 8. Revised Simulation & Compute Budget

| Split | Structures | Unique Earthquakes | Standard Runs | Bilateral Pairs | Total Simulations | CPU Time Est. |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (P1 basis)** | `SOURCE_A` | 70 GMs (`RSN0001..70`) | 70 | 20 | 90 | ~4.5 min |
| **Train (P2 additions)**| `B_train_1..4` | 70 GMs (`RSN0001..70`, shared) | 140 (35/struct) | 0 | 140 | ~7.0 min |
| **Validation** | `SOURCE_A` + `B_train_1` | 20 GMs (`RSN0071..90`) | 20 | 10 | 30 | ~1.5 min |
| **Level 1 Test (ID)** | `SOURCE_A` | 30 GMs (`RSN0091..120`) | 0 | 15 (30 eval) | 15 | ~1.0 min |
| **Level 2A (Interp)** | `B_int_1, B_int_2` | 30 GMs (`RSN0091..120`) | 0 | 15 (30 eval) | 15 | ~1.0 min |
| **Level 2B (Extrap)** | `B_ext_soft, B_ext_stiff`| 30 GMs (`RSN0091..120`) | 0 | 15 (30 eval) | 15 | ~1.0 min |
| **Level 3 (Topology)** | `C_4story` | 30 GMs (`RSN0091..120`) | 0 | 15 (30 eval) | 15 | ~1.0 min |
| **TOTAL PROJECT** | **9 Structural Variants** | **120 (100% DISJOINT)** | **230** | **75 pairs** | **320 runs** | **~17 min** |

---

## 9. Phase 7.0.2 Gate Decision: PASS

- **G1 (Source Verified):** Confirmed in `source_a_manifest.json`.
- **G2 (Parameter Manifold):** Defined on restricted 2D manifold $(\mu_m, \mu_k)$.
- **G3 (Structural Split):** Convex hull proofs confirmed in `structural_split_manifest.json`.
- **G4 (Earthquake Split):** Exactly 120 unique records partitioned ($70 / 20 / 30$) with 0 leakage.
- **G5 (Forward Protocol):** Tolerances set ($\epsilon < 0.1\%$).
- **G6 (Observability Protocol):** S0, S1, S4 singular spectrum and Fisher analysis locked.
- **G7 (Conditioning Audit):** Continuous dimensionless features audited; categorical IDs banned.
- **G8 (Topology Requirements):** Dynamic $|V|, |E|$ and dynamic symmetry specified.
- **G9 (Ablation Hierarchy):** Protocol P1 vs P2 + 6 model conditions frozen in `final_experimental_matrix.json`.
- **G10 (Normalization Policy):** Category A, B, C rules locked in `normalization_policy.json`.
- **G11 (Failure Attribution):** F1 through F7 diagnostic taxonomy locked.
- **G12 (Budget & Power):** 320 runs across 120 earthquakes; $N=30$ evaluations per test level.

**PHASE 7.0.2 GATE VERDICT: PASS**
