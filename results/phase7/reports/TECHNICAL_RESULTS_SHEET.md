# Inverse-FNO-Damage: One-Page Technical Results Sheet

**Project:** `Inverse-FNO-Damage`  
**Reference Document:** Comprehensive Technical Summary for Faculty Review  
**Date:** September 2026 | **Reproducibility:** 74/74 Unit Tests Passing (`scripts/gstack.py qa`)  

---

### A. Mathematical Problem Setup
- **Governing Dynamics:** $\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K}(d, \theta) \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t)$, $t \in [0, T]$, $T = 2.5\text{ s}$ ($250$ time steps at $100\text{ Hz}$).
- **Damage Parameterization:** $d \in [0, 0.5]^E$, where elemental elastic modulus is $E_e = (1 - d_e) E_0$.
- **Sparse Observation Operator:** $\mathbf{y}(t) = \mathcal{H}_S \left[ \ddot{\mathbf{u}}(t) + \mathbf{r} a_g(t), \, \mathbf{u}(t) \right] + \boldsymbol{\eta}(t) \in \mathbb{R}^{C \times T}$, with stationary Gaussian sensor noise $\boldsymbol{\eta} \sim \mathcal{N}(0, \sigma_\eta^2 \mathbf{I})$ calibrated to $2\%$ root-mean-square amplitude.
- **Canonical Bilateral Ambiguity Vector:** For symmetric single-column $30\%$ damage ($d_A = [0.30, 0, \dots]^T$, $d_B = [0, 0.30, \dots]^T$):
  $$v_{AB} = \frac{d_A - d_B}{\|d_A - d_B\|_2} = \frac{1}{\sqrt{2}} [1, -1, 0, \dots, 0]^T \in \mathbb{R}^E, \quad \|v_{AB}\|_2 = 1.0$$
  $$\text{True Physical Bilateral Separation: } \|d_A - d_B\|_2 = \sqrt{0.30^2 + (-0.30)^2} = \sqrt{0.18} \approx 0.424264$$

---

### B. Structural Finite-Element Model
- **Mechanics Engine:** Validated planar finite-element models implemented in OpenSeesPy.
- **Baseline Configuration (`SOURCE_A`):** 3-story, 1-bay frame (8 joints, 9 beam-column elements, 18 active DOFs).
- **Physical Properties:** Bay width $L = 6.0\text{ m}$, story height $h = 3.5\text{ m}$, $E_0 = 200\text{ GPa}$, density $\rho = 7850\text{ kg/m}^3$.
- **Modal Dynamics:** Fundamental frequencies $f_1 = 2.064\text{ Hz}$, $f_2 = 6.945\text{ Hz}$; Rayleigh damping $\zeta = 0.02$ at modes 1 and 2.
- **Generalization Family:** 10 structural configurations spanning $L \in [5.0, 7.0]\text{ m}$, $h \in [3.0, 4.0]\text{ m}$, $E \in [1.7, 2.3]\times 10^{11}\text{ Pa}$, and an unseen 4-story frame ($C_{\text{4story}}$, 10 joints, 12 elements, $f_1 = 1.542\text{ Hz}$).

---

### C. Dataset Accounting & D. Experimental Splits
- **Ground Motion Database:** 120 PEER strong motion earthquake acceleration records partitioned into strictly disjoint subsets:
  - Train: 70 records (`RSN0001`–`RSN0070`) | Validation: 20 records (`RSN0071`–`RSN0090`) | Test: 30 records (`RSN0091`–`RSN0120`).
- **Dynamic Simulation Volume:** Exactly **530 validated OpenSees simulations** (verified via `results/phase7/statistics/dataset_integrity_audit.json`):
  - Training: 90 runs (Protocol P1: `SOURCE_A`) + 140 runs (Protocol P2: 4 diverse frames $B_{\text{train\_1..4}}$).
  - Validation: 60 runs (`SOURCE_A` and $B_{\text{train\_1}}$).
  - Test: 240 runs (120 bilateral pairs, 30 pairs per generalization level across the 30 held-out test earthquakes).

---

### E. Sensor Configurations & F. Observability Metrics
- **Linearized Sensitivity Jacobian:** $\mathbf{J}_{ij} = \partial y_i / \partial d_j \in \mathbb{R}^{(C \cdot T) \times E}$, evaluated by central finite differences at $d = \mathbf{0}$.
- **Noise Covariance & Whitened Jacobian:** $\mathbf{\Sigma}_\eta = \text{diag}(\sigma_{\eta, 1}^2, \dots, \sigma_{\eta, C}^2) \otimes \mathbf{I}_T$; $\mathbf{J}_w = \mathbf{\Sigma}_\eta^{-1/2} \mathbf{J}$.
- **Noise-Whitened Directional Fisher Sensitivity:** $\sqrt{I_{AB}} = \|\mathbf{J}_w v_{AB}\|_2 = \sqrt{v_{AB}^T \mathbf{J}^T \mathbf{\Sigma}_\eta^{-1} \mathbf{J} v_{AB}}$.
- **Modality Observability Hierarchy (at 2% noise):**
  - **$S0$ (3 Horizontal Floor Accelerometers):** $\sqrt{I_{AB}} = 5.617$ (Right singular vector alignment $|\langle v_E, v_{AB} \rangle| = 0.9982$; near-null space).
  - **$S1$ (3 Horizontal + 6 Vertical Joint Accelerometers):** $\sqrt{I_{AB}} = 1829.63$ ($325.7\times$ gain over $S0$).
  - **$S2$ (3 Horizontal Accelerometers + 6 Column Axial Strains):** $\sqrt{I_{AB}} = 1344.18$ ($239.3\times$ gain over $S0$).
  - **$S4$ (Multimodal Union: 3 Horiz Accel + 6 Vert Accel + 6 Strains):** $\sqrt{I_{AB}} = 2270.54$ ($404.2\times$ gain over $S0$).

---

### G. Phase 6.2 Single-Structure Benchmark ($N = 30$ Independent Evaluations)
DualStreamGFNO architecture with latent symmetry projection ($\mathbf{H}^\pm = \frac{1}{2}(\mathbf{H} \pm \mathbf{P}\mathbf{H})$), hierarchical heads ($\hat{d}_e = p_e \cdot \mu_e$, where $p_e \in [0, 1]$ is damage support probability and $\mu_e \in [0, 0.5]$ is severity), and matched pair loss ($L_{\text{dir}} + L_{\text{pair}}$, margin $m = 0.15$):

$$\cos(\Delta \hat{d}, v_{AB}) = \frac{\langle \Delta \hat{d}, v_{AB} \rangle}{\|\Delta \hat{d}\|_2 \|v_{AB}\|_2}, \quad \|\Delta \hat{d}\|_2 = \|\hat{d}_A - \hat{d}_B\|_2$$

| Modality Suite | Attribution Successes | Attribution Accuracy | 95% Clopper-Pearson CI | Two-Sided $p$-value ($p_0=0.5$) | Directional Cosine | Predicted Separation $\|\Delta \hat{d}\|_2$ | True Separation Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S0$ (Horizontal Accel)** | 15 / 30 | **50.0%** | [31.3%, 68.7%] | $p = 1.000$ | $-0.1151 \pm 0.122$ | $9.04 \times 10^{-6}$ | $0.002\%$ |
| **$S1$ (Horiz + Vert Accel)** | 24 / 30 | **80.0%** | [61.4%, 92.3%] | $\mathbf{p = 0.0014}$ | $\mathbf{+0.7272 \pm 0.442}$ | $0.1576$ | $37.15\%$ |
| **$S2$ (Horiz + Column Strain)** | 15 / 30 | **50.0%** | [31.3%, 68.7%] | $p = 1.000$ | $+0.1047 \pm 0.111$ | $0.000328$ | $0.08\%$ |
| **$S4$ (Multimodal Union)** | 27 / 30 | **90.0%** | [73.5%, 97.9%] | $\mathbf{p = 9.0 \times 10^{-6}}$ | $\mathbf{+0.9420 \pm 0.098}$ | $\mathbf{0.1671}$ | $\mathbf{39.40\%}$ |

*Key Takeaways:* $S4$ resolves bilateral ambiguity on a single structure ($90.0\%$, $p = 9.0\times 10^{-6}$). $S2$ demonstrates that **physical observability is necessary but insufficient for neural learnability**: high Fisher sensitivity ($1344.2$) fails under standard optimization due to floor acceleration gradient dominance ($>98\%$).

---

### H. Phase 7 Multi-Structure Results & I. Training-Budget Ablation
Protocol P2 (Multi-Structure Training) evaluated under Multimodal Sensing $S4$ across 72 discrete model runs (12, 25, 50, 100 epochs $\times$ seeds 42, 101, 2024):
- **Discrete Attribution Accuracy:** Remains exactly at chance level (**$50.0\% \pm 0.0\%$**) across all levels (Level 1 In-Distribution, Level 2A Interpolation, Level 2B Extrapolation, Level 3 Topology $C_{\text{4story}}$).
- **Directional Cosine Scaling:**
  - 12 Epochs: $+0.467 \pm 0.037$ (L1), $+0.480 \pm 0.038$ (L2A), $+0.534 \pm 0.055$ (L2B), $+0.489 \pm 0.012$ (L3).
  - 25 Epochs: $+0.852 \pm 0.115$ (L1), $+0.892 \pm 0.072$ (L2A), $+0.802 \pm 0.204$ (L2B), $+0.777 \pm 0.158$ (L3).
  - 50 Epochs: $+0.897 \pm 0.038$ (L1), $+0.896 \pm 0.016$ (L2A), $+0.824 \pm 0.096$ (L2B), $+0.837 \pm 0.090$ (L3).
  - 100 Epochs: $+0.850 \pm 0.205$ (L1), $+0.841 \pm 0.216$ (L2A), $+0.813 \pm 0.204$ (L2B), $+0.798 \pm 0.199$ (L3).
  - Individual Seed Peaks at 100 Ep: Seed 42 reaches $+0.993$ (L1) and $+0.921$ (L3); Seed 2024 reaches $+0.996$ (L1) and $+0.955$ (L3).
- **Damage Separation $\|\Delta \hat{d}\|_2$:** Remains collapsed across all budgets: $2.68 \times 10^{-6}$ (12 ep) $\to 1.83 \times 10^{-5}$ (100 ep).
- **Empirical Direction–Magnitude Decoupling Phenomenon:** Within the tested 12–100 epoch budgets, increasing optimization budget refines continuous directional alignment but does not recover finite bilateral damage separation under cross-structure distribution shift.

---

### J. Leakage & Reproducibility Controls
- **Pristine Feature Contract:** Graph edge features strictly enforce nominal un-damaged stiffness ($E_{\text{norm}} = 1.0$), verified by 4 automated regression tests (`tests/test_phase7_feature_leakage_contract.py`).
- **Deterministic Replication:** Seeds 42, 101, 2024 govern all network initializations, synthetic noise, and dataset generation.
- **Disjoint Partitioning:** 100% disjoint earthquake splits (70 train, 20 val, 30 test) across all structural variants.
- **Statistical Integrity:** $N = 30$ independent held-out bilateral evaluations per test condition. Replications across seeds are never pooled to claim pseudoreplicated sample sizes ($N \ne 90$).

---

### K. Theoretical & Experimental Limitations
- 2D planar idealization (torsional 3D out-of-plane modes neglected).
- Linear-elastic stiffness degradation modeled; geometric $P$-$\Delta$ and hysteretic material damage omitted.
- Synthetic dynamic responses with idealized $2\%$ stationary Gaussian noise; no experimental shake-table validation.
