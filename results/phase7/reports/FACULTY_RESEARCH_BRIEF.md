# Observability, Learnability, and Transferability in Neural-Operator Inversion for Seismic Structural Damage Identification

**Authors:** Inverse-FNO-Damage Research Initiative  
**Target Audience:** Faculty in Scientific Machine Learning, Computational Mechanics, Structural Dynamics, Inverse Problems, and SHM  
**Repository:** `Inverse-FNO-Damage`  
**Status:** Complete Research Project (Experimental Program Frozen; 74/74 Unit Tests Passing)  

---

## 1. Problem Statement
The objective of this research is to infer the spatial distribution and severity of localized structural damage—quantified as fractional member stiffness reduction $d \in [0, 0.5]^E$—in civil building frames from sparse, noisy dynamic acceleration and strain recordings collected during earthquake excitation.

---

## 2. Why This Is Scientifically Hard
Civil structures exhibit nominal lateral symmetry. Under conventional instrumentation layouts restricted to horizontal floor accelerometers, localized damage in laterally symmetric structural members (e.g., ground-story exterior left column vs. exterior right column) generates global floor acceleration trajectories that are virtually identical in both amplitude and phase ($<0.14\%$ relative $L_2$ difference). 

When applied to this ill-posed inverse problem, standard neural operators (such as Fourier Neural Operators) trained with mean-squared-error objectives exhibit **mean-seeking collapse**: the network predicts the conditional expectation $\mathbb{E}[d \mid \mathbf{y}]$, outputting diffuse, symmetric predictions of negligible amplitude ($\hat{d} < 0.02$) that completely fail to identify localized damage. While spatial sparsity ($\ell_1$) and total variation priors suppress diffuse predictions, they cannot break physical symmetry, collapsing to equal damage on both bilateral members simultaneously. Furthermore, when trained across diverse structures, parameter dispersion interacts with this physical ambiguity to induce severe distribution shift.

---

## 3. Central Research Question
> *"When a seismic inverse problem is physically ambiguous, which limitations arise from the sensing physics, which from the neural representation, and which from cross-structure transfer?"*

---

## 4. Experimental Program & Research Progression
The investigation was executed through a structured, 7-phase empirical program:
1. **Phase 1 (Physical Non-Identifiability):** Validated finite-element dynamic simulations in OpenSeesPy demonstrated that $30\%$ bilateral column damage produces $<0.14\%$ horizontal acceleration discrepancy, placing the bilateral difference vector in the effective numerical near-null space of the horizontal observation operator.
2. **Phases 2–4 (Neural Baselines & Sensor Density):** Benchmarked 1D forward FNOs (waveform relative $L_2 = 36.71\% \pm 8.42\%$, Pearson $\rho = 0.9030$) and documented the failure of naive inverse FNOs and floor-sensor densification to resolve bilateral ambiguity.
3. **Phases 5 & 5.5 (Noise-Whitened Observability):** Developed a noise-whitened Jacobian SVD and Fisher Information framework quantifying directional sensitivity prior to neural training.
4. **Phase 6 (Symmetry-Aware Architecture):** Introduced the **DualStreamGFNO**, splitting temporal representations into symmetric ($H^+$) and antisymmetric ($H^-$) latent subspaces, with hierarchical support/severity heads ($\hat{d}_e = p_e \cdot \mu_e$) and matched bilateral pair supervision ($L_{\text{dir}} + L_{\text{pair}}$).
5. **Phase 6.2 (Single-Structure Benchmark):** Conducted a controlled benchmark on a 3-story frame across $N = 30$ independent held-out bilateral evaluations.
6. **Phase 7 (Cross-Structure Benchmark):** Evaluated transferability across an audited 530-simulation benchmark spanning 10 structural configurations and an unseen 4-story frame topology ($C_{\text{4story}}$).
7. **Training-Budget Ablation:** Executed a 72-model controlled sweep across 12, 25, 50, and 100 epochs over 3 deterministic seeds ($42, 101, 2024$) to isolate optimization limitations from structural distribution shift.

---

## 5. Three Core Findings

### Finding 1: Physical Observability Under Horizontal Sensing Is Near-Null
Singular value decomposition of the linearized sensitivity Jacobian reveals that under standard horizontal floor sensing ($S0$), the right-singular vector of the smallest singular value aligns almost perfectly with the canonical bilateral direction $v_{AB}$ ($|\langle v_E, v_{AB} \rangle| > 0.99$). The noise-whitened directional Fisher sensitivity is negligible ($\sqrt{I_{AB}} \approx 5.617$), confirming that bilateral ambiguity is an inherent physical sensing limitation.

### Finding 2: Physical Observability Is Necessary but Insufficient for Neural Learnability
Augmenting sensing with column axial strain ($S2$) provides high physical Fisher sensitivity ($\sqrt{I_{AB}} = 1344.18$, a $239\times$ increase over $S0$). However, under gradient-based optimization, DualStreamGFNO achieves only **$50.0\%$ bilateral attribution accuracy** (exact chance level) under $S2$. High-amplitude floor accelerations ($\sim 1.0\text{ m/s}^2$) dominate the backpropagation graph, carrying $>98\%$ of initial gradient norm and drowning out micro-strain signals ($\sim 10^{-5}\text{ m/m}$). In contrast, multimodal sensing ($S4$, combining vertical acceleration and axial strain) achieves $\sqrt{I_{AB}} = 2270.54$ and elevates attribution to **$90.0\%$** ($p = 9.0 \times 10^{-6}$). High Fisher information does not guarantee neural learnability under multiscale physical gradients.

### Finding 3: Cross-Structure Transfer Exhibits an Empirical Boundary
Across unseen structural geometries, stiffness parameters, and frame topologies, the learned inverse mapping transfers continuous directional sensitivity zero-shot ($\cos(\Delta \hat{d}, v_{AB}) \approx +0.80$ to $+0.85$, with individual seeds reaching $+0.996$ on in-distribution frames and $+0.955$ on an unseen 4-story topology). However, finite damage separation collapses ($\|\Delta \hat{d}\|_2 \sim 10^{-5}$ vs. true physical separation $0.4243$), pinning discrete attribution to chance ($50.0\%$). Scaling training budget by $8.3\times$ ($12 \to 100$ epochs) refines directional orientation but does not recover finite bilateral separation.

---

## 6. Key Quantitative Results

### Single-Structure Benchmark (Phase 6.2, $N = 30$ Held-Out Bilateral Evaluations)
| Sensor Suite | Active Channels | Noise-Whitened Fisher Sensitivity $\sqrt{I_{AB}}$ | Attribution Accuracy | Exact Two-Sided $p$-value ($p_0 = 0.5$) | 95% Clopper-Pearson CI | Mean Predicted Separation $\|\Delta \hat{d}\|_2$ | True Separation Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S0$ (Horizontal Accel)** | 3 | $5.617$ | **50.0%** (15/30) | $p = 1.000$ | [31.3%, 68.7%] | $9.04 \times 10^{-6}$ | $0.002\%$ |
| **$S1$ (Horiz + Vert Accel)** | 9 | $1829.63$ | **80.0%** (24/30) | $\mathbf{p = 0.0014}$ | [61.4%, 92.3%] | $0.1576$ | $37.15\%$ |
| **$S2$ (Horiz + Column Strain)** | 9 | $1344.18$ | **50.0%** (15/30) | $p = 1.000$ | [31.3%, 68.7%] | $0.000328$ | $0.08\%$ |
| **$S4$ (Multimodal Union)** | 15 | $2270.54$ | **90.0%** (27/30) | $\mathbf{p = 9.0 \times 10^{-6}}$ | [73.5%, 97.9%] | $\mathbf{0.1671}$ | $\mathbf{39.40\%}$ |

*True physical bilateral separation: $\|d_A - d_B\|_2 = \sqrt{0.30^2 + (-0.30)^2} \approx 0.4243$.*

### Cross-Structure Generalization & Budget Ablation (Phase 7, Multimodal Sensing $S4$)
| Generalization Regime | Evaluated Frames | Discrete Attribution Accuracy | Directional Cosine (12 Epochs) | Directional Cosine (100 Epochs) | Predicted Separation $\|\Delta \hat{d}\|_2$ (100 Ep) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Level 1 (In-Distribution)** | `SOURCE_A` | $50.0\% \pm 0.0\%$ | $+0.467 \pm 0.037$ | $\mathbf{+0.850 \pm 0.205}$ (max $+0.996$) | $1.83 \times 10^{-5}$ |
| **Level 2A (Parametric Interp)** | $B_{\text{int\_1}}, B_{\text{int\_2}}$ | $50.0\% \pm 0.0\%$ | $+0.480 \pm 0.038$ | $\mathbf{+0.841 \pm 0.216}$ | $1.92 \times 10^{-5}$ |
| **Level 2B (Parametric Extrap)** | $B_{\text{ext\_soft}}, B_{\text{ext\_stiff}}$ | $50.56\% \pm 0.79\%$ | $+0.534 \pm 0.055$ | $\mathbf{+0.813 \pm 0.204}$ | $2.14 \times 10^{-5}$ |
| **Level 3 (Topological OOD)** | $C_{\text{4story}}$ (4-story, 10-node) | $50.0\% \pm 0.0\%$ | $+0.489 \pm 0.012$ | $\mathbf{+0.798 \pm 0.199}$ (max $+0.955$) | $1.64 \times 10^{-5}$ |

---

## 7. Central Empirical Phenomenon: Direction–Magnitude Decoupling
We formalize the observed cross-structure failure as an **empirical phenomenon (not a mathematical theorem)**:

> **Direction–Magnitude Decoupling:** In neural inverse operators for symmetric dynamical systems under structural parameter dispersion, gradient-based optimization successfully aligns the orientation of predicted difference vectors with the physically observable bilateral sensitivity direction ($\cos(\Delta \hat{d}, v_{AB}) \to +0.80\text{--}+0.85$), while simultaneously failing to scale the magnitude of the predicted separation ($\|\Delta \hat{d}\|_2 \sim 10^{-5} \ll 0.30$). 

Scale-invariant losses ($L_{\text{dir}}$) rotate the output subspace correctly, but under multi-structure parameter variations, the observed behavior is consistent with predictions collapsing toward a shared symmetric solution to minimize global regression loss across diverse structural frequencies.

---

## 8. Significance for Scientific Machine Learning
In scientific ML, benchmark success is frequently claimed using continuous latent representation diagnostics (e.g., cosine similarity, alignment angles, latent embeddings). Our findings demonstrate that:
1. **A neural operator can retain high, physically meaningful directional sensitivity while completely failing the actual inverse identification task.**
2. **Physical observability, numerical learnability, and cross-structure transferability represent distinct failure modes with different mathematical origins.**
3. Continuous latent diagnostics and discrete parameter-space separation must be evaluated as independent failure modes in inverse physics problems.

---

## 9. Limitations
1. **Structural Idealizations:** Planar 2D frames are analyzed; 3D out-of-plane modes and torsional building response are not modeled.
2. **Constitutive Assumptions:** Linear-elastic flexural stiffness reduction is simulated without material hysteretic degradation or geometric $P$-$\Delta$ effects.
3. **Synthetic Numerical Traces:** Traces are generated from validated OpenSeesPy finite-element models with idealized $2\%$ stationary Gaussian noise, without temperature drift, soil-structure interaction, or non-structural component interference.
4. **Experimental Bounds:** The methodology has not yet been validated against instrumented laboratory shake-table specimens.
5. **Finite Scope:** Datasets comprise 10 planar structural configurations and 120 PEER ground motion records.

---

## 10. Next Scientific Question
> *"What changes when the same observability–learnability–transferability framework is tested on experimentally instrumented structures with richer nonlinearities, 3D coupling, and realistic nonstationary measurement noise?"*
