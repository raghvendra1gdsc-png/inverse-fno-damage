# Observability, Learnability, and Transferability in Neural-Operator Inversion for Seismic Structural Damage Identification

**Authors:** Inverse-FNO-Damage Research Initiative  
**Correspondence:** Scientific Machine Learning & Structural Dynamics Group  
**Repository Identity:** `Inverse-FNO-Damage`  
**Date:** September 2026  
**Status:** Complete Research Manuscript (Experimental Program Frozen; 74/74 Unit Tests Passing)

---

## Abstract

Identifying localized structural damage in civil building frames from sparse, noisy seismic vibration recordings constitutes a notoriously ill-posed physical inverse problem. Structural symmetry induces near-null spaces in the parameter-to-response sensitivity mapping under standard horizontal sensing, causing symmetric failure modes (e.g., left vs. right column damage) to generate virtually identical floor accelerations (differing by $<0.14\%$). Consequently, standard deep inverse models trained with mean-squared-error objectives collapse to diffuse, mean-seeking predictions that fail to localize or attribute damage. 

In this work, we investigate the conditions under which neural operators can invert structural damage under sparse sensing, structural symmetry, and cross-structure distribution shift. We establish a noise-whitened Jacobian and Fisher information framework that quantifies directional observability prior to model training, demonstrating that under the prescribed sensing and noise assumptions, adding vertical floor acceleration ($S1$) or multimodal sensing ($S4$) amplifies the noise-whitened directional Fisher sensitivity, $\sqrt{I_{AB}}$, along the bilateral ambiguity vector $v_{AB}$ by more than $300\times$ over conventional horizontal floor sensing ($S0$). To exploit this physical asymmetry, we introduce the Symmetry-Aware Dual-Stream Graph Fourier Neural Operator (**DualStreamGFNO**), which explicitly projects latent graph representations into symmetric ($H^+$) and antisymmetric ($H^-$) subspaces and trains with matched bilateral pair supervision incorporating directional alignment ($L_{\text{dir}}$) and contrastive margin ($L_{\text{pair}}$) losses. 

On a single-structure benchmark ($N = 30$ independent held-out earthquake evaluations), our proposed framework elevates bilateral damage attribution accuracy from the $50.0\%$ chance baseline to **$80.0\%$** under $S1$ ($p = 0.0014$) and **$90.0\%$** under $S4$ ($p = 9.0 \times 10^{-6}$, 95% Clopper-Pearson CI: $[73.5\%, 97.9\%]$) with macroscopic predicted separation ($\|\Delta \hat{d}\|_2 = 0.1671$, $39.4\%$ of true physical separation). However, configuration $S2$ (column axial strain) exhibits an *observability–learnability gap*, remaining at $50.0\%$ attribution despite high directional Fisher sensitivity ($\sqrt{I_{AB}} = 1344.2$) due to gradient dominance by dominant floor motions. 

When evaluated across an audited 530-simulation benchmark spanning 10 structural configurations and an unseen 4-story topological frame ($C_{\text{4story}}$), we observe an **empirical Direction–Magnitude Decoupling phenomenon**: continuous directional sensitivity transfers zero-shot across structural variations ($\cos \approx +0.80$ to $+0.85$), while finite bilateral damage separation collapses ($\|\Delta \hat{d}\|_2 \sim 10^{-5}$) and discrete attribution remains at chance ($50.0\%$). A controlled 72-evaluation training-budget sweep across 12, 25, 50, and 100 epochs demonstrates that within the tested training budgets, increasing optimization budget improves directional alignment (reaching up to $+0.996$ on individual seeds) but does not recover finite bilateral separation. Our findings demonstrate that physical observability is necessary but insufficient for neural learnability under the tested configurations, and delineate the empirical boundaries of cross-structure transfer in scientific machine learning for inverse structural dynamics.

---

## 1. Introduction

Seismic structural health monitoring (SHM) seeks to rapidly diagnose internal structural degradation—such as loss of member stiffness, yielding, or localized column cracking—following an earthquake excitation. Accurately inferring the spatial distribution and severity of damage from sparse sensor arrays deployed at a fraction of building joints is a classic inverse problem in computational mechanics and applied mathematics.

In practical field installations, economic and operational constraints restrict instrumentation to a sparse subset of degrees of freedom (DOFs), most commonly horizontal accelerometers affixed to floor slabs. Under this standard instrumentation regime, the inverse damage problem suffers from severe non-uniqueness and ill-posedness. In buildings with nominal geometric and structural symmetry, localized stiffness degradation in laterally symmetric structural members (e.g., ground-story exterior left column vs. exterior right column) produces global horizontal floor acceleration histories that are nearly indistinguishable in both amplitude and phase. 

In recent years, deep learning and neural operators—most notably Fourier Neural Operators (FNOs)—have emerged as powerful mesh-independent paradigms for solving forward and inverse problems in partial differential equations (PDEs). When applied naively to ill-posed inverse structural dynamics, however, unconstrained deep networks trained with mean squared error (MSE) loss exhibit an insidious failure mode: **mean-seeking collapse**. When two distinct, mutually exclusive physical states produce virtually identical sensor traces, gradient descent with an MSE objective forces the network to output the conditional expectation—namely, the arithmetic mean of the two states. In localized damage identification, this produces diffuse, low-amplitude predictions across all structural members, completely obscuring localized structural failures.

While physics-informed regularizations (such as sparsity $\ell_1$ penalties and total variation priors) encourage spatial localization, they do not resolve the underlying physical non-uniqueness. If the forward physical operator maps two distinct damage states into the same sensor trajectory, no mathematical prior can uniquely identify the true state without additional physical information.

To address this challenge, two central questions must be answered:
1. *Which sensor modalities physically break structural symmetry and render bilateral damage states identifiable?*
2. *Can a neural operator be architected and supervised to exploit subtle asymmetric signatures without being overwhelmed by dominant macroscopic dynamics?*

Beyond single structures, practical post-earthquake assessment demands that trained models generalize across structural dimensions, member properties, and unseen building topologies. Yet, the interaction between structural distribution shift and inverse identifiability remains largely unmapped in the scientific machine learning literature.

In this paper, we systematically investigate the conditions under which neural operators can identify structural damage from sparse seismic observations when physical observability, structural symmetry, neural learnability, and cross-structure distribution shift interact. We structure our investigation around six precise research questions:

- **RQ1:** Under what physical conditions does structural symmetry create a near-null sensitivity direction under sparse sensing?
- **RQ2:** Can the physical identifiability of bilateral damage states be quantitatively evaluated prior to neural network training using a noise-whitened sensitivity framework?
- **RQ3:** Can a symmetry-aware neural operator architecture successfully isolate and process asymmetric vibration signatures?
- **RQ4:** Does explicit bilateral pair supervision force the network to overcome gradient dominance and achieve discrete state attribution on single structures?
- **RQ5:** How does the learned inverse mapping transfer when evaluated across parametric structural variations and unseen frame topologies?
- **RQ6:** Does increasing the optimization budget recover the multi-structure discrete attribution collapse, or does it expose an inherent decoupling between directional sensitivity and finite output separation?

---

## 2. Scientific Contributions

This work provides seven defensible scientific contributions to inverse problem theory, structural health monitoring, and scientific machine learning:

1. **Empirical Characterization of Bilateral Non-Identifiability:** We provide rigorous numerical and dynamic simulation evidence demonstrating that in symmetric multi-story frames, localized 30% column stiffness damage in bilateral members produces horizontal floor acceleration discrepancies of less than $0.14\%$, placing the bilateral difference vector in the effective numerical near-null space of the forward operator.
2. **Noise-Whitened Modality-Aware Observability Framework:** We formulate a noise-whitened Jacobian singular value decomposition and directional Fisher Information framework. We demonstrate that under the prescribed sensing and noise model, while horizontal floor acceleration ($S0$) yields negligible noise-whitened directional Fisher sensitivity ($\sqrt{I_{AB}} \approx 5.6$), augmenting sensing with vertical joint acceleration ($S1$) or multimodal sensing ($S4$) amplifies $\sqrt{I_{AB}}$ by more than $300\times$ ($\sqrt{I_{AB}} = 1829.6$ and $2270.5$, respectively).
3. **Symmetry-Aware Dual-Stream Graph Fourier Neural Operator (DualStreamGFNO):** We develop an architecture that explicitly splits structural representations into symmetric ($H^+$) and antisymmetric ($H^-$) latent subspaces, decodes damage via hierarchical support/severity heads, and enforces pairwise directional alignment ($L_{\text{dir}}$) and contrastive margin ($L_{\text{pair}}$) losses.
4. **Demonstration of the Observability–Learnability Gap:** We demonstrate that physical observability is necessary but insufficient for neural learnability under the tested configurations. Configuration $S2$ (column axial strain) possesses high directional Fisher sensitivity ($\sqrt{I_{AB}} = 1344.2$), yet standard optimization yields $50.0\%$ attribution accuracy due to gradient dominance by dominant horizontal floor accelerations.
5. **Statistically Verified Single-Structure Inversion Benchmark:** On an independent, strictly disjoint held-out earthquake evaluation ($N = 30$), we demonstrate that our proposed model achieves **$90.0\%$ bilateral attribution accuracy** under multimodal sensing $S4$ ($27/30$, $p = 9.0 \times 10^{-6}$, 95% CI: $[73.5\%, 97.9\%]$) and **$80.0\%$** under vertical acceleration $S1$ ($24/30$, $p = 0.0014$), recovering $39.4\%$ of true physical damage separation.
6. **Zero-Shot Directional Transfer Across Topologies:** Across an audited 530-simulation benchmark, we demonstrate that continuous directional damage sensitivity transfers zero-shot to unseen structural parameters (interpolation $\cos = +0.841$, extrapolation $\cos = +0.813$) and to an unseen 4-story frame topology ($C_{\text{4story}}$, $\cos = +0.798$), while finite bilateral damage separation collapses and discrete attribution remains at chance level.
7. **Empirical Formalization of Direction–Magnitude Decoupling:** Through a controlled 72-model budget ablation ($12 \to 100$ epochs), we show that within the tested 12–100 epoch training budgets, increasing optimization budget refines directional alignment (reaching up to $+0.996$ on individual seeds) but does not recover finite bilateral separation, establishing an empirical decoupling between continuous orientation learning and macroscopic output scaling under structural distribution shift.

---

## 3. Mathematical Problem Formulation

### 3.1 Forward Dynamic Model
Consider an $N_s$-story, $N_b$-bay civil building frame discretized into $E$ structural beam-column elements and $V$ structural joints. The transient dynamic response under seismic ground excitation $a_g(t) \in \mathbb{R}^T$ is governed by the structural equations of motion:
$$\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K}(d, \theta) \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t), \quad t \in [0, T]$$
where $\mathbf{u}(t) \in \mathbb{R}^{N_{\text{dof}}}$ is the vector of relative joint displacements; $\mathbf{M}, \mathbf{C}, \mathbf{K} \in \mathbb{R}^{N_{\text{dof}} \times N_{\text{dof}}}$ are the structural mass, damping, and stiffness matrices; $\mathbf{r}$ is the earthquake influence vector; $\theta = \{L, h, E_0, \rho\}$ denotes the nominal structural parameter vector; and $d \in [0, 1]^E$ is the elemental damage state vector, representing the fractional reduction in elastic stiffness for each member:
$$E_e = (1 - d_e) E_0, \quad d_e \in [0, 0.5], \quad e \in \{1, \dots, E\}$$
A damage value $d_e = 0.0$ denotes a pristine, undamaged member, while $d_e = 0.30$ denotes a $30\%$ loss of flexural and axial stiffness.

### 3.2 Sensor Observation Operator
In practice, full degree-of-freedom displacement history $\mathbf{u}(t)$ is unobservable. Observations are collected via a discrete sensor layout defined by the measurement operator $\mathcal{H}_S$:
$$\mathbf{y}(t) = \mathcal{H}_S \left[ \ddot{\mathbf{u}}(t) + \mathbf{r} a_g(t), \, \mathbf{u}(t) \right] + \boldsymbol{\eta}(t), \quad \mathbf{y}(t) \in \mathbb{R}^{C \times T}$$
where $C$ is the number of active measurement channels, and $\boldsymbol{\eta}(t) \sim \mathcal{N}(0, \sigma_\eta^2 \mathbf{I})$ represents stationary sensor noise (calibrated to $2\%$ root-mean-square amplitude under the prescribed sensing model).

### 3.3 The Inverse Problem
The objective of the neural inverse operator $\mathcal{G}_\phi^\dagger$ is to reconstruct the spatial damage vector $\hat{d}$ given sparse seismic observations $\mathbf{y}(t)$, ground motion $a_g(t)$, and nominal structural features $\theta$:
$$\hat{d} = \mathcal{G}_\phi^\dagger \left( \mathbf{y}, a_g, \theta \right), \quad \hat{d} \in [0, 1]^E$$

### 3.4 Canonical Bilateral Symmetry Direction
Let a laterally symmetric building frame possess a geometric reflection involution $\mathcal{P}: V \to V$ that maps left-side nodes to right-side nodes. In damage space, this involution induces a member permutation matrix $\mathbf{P} \in \{0, 1\}^{E \times E}$ such that $\mathbf{P}^2 = \mathbf{I}$.

We define the canonical bilateral damage pair as two mutually exclusive, symmetric single-member failure modes:
- **State A:** $d_A = [0.30, 0, 0, \dots, 0]^T$ (left ground-story column damaged by 30%)
- **State B:** $d_B = [0, 0.30, 0, \dots, 0]^T = \mathbf{P} d_A$ (right ground-story column damaged by 30%)

The **canonical bilateral direction vector** $v_{AB} \in \mathbb{R}^E$ is the normalized difference vector between these states:
$$v_{AB} = \frac{d_A - d_B}{\|d_A - d_B\|_2} = \frac{1}{\sqrt{2}} [1, -1, 0, \dots, 0]^T, \quad \|v_{AB}\|_2 = 1.0$$
The true physical separation between the states is:
$$\|d_A - d_B\|_2 = \sqrt{(0.30)^2 + (-0.30)^2} = \sqrt{0.18} \approx 0.424264$$
The vector $v_{AB}$ defines the 1D subspace along which bilateral ambiguity must be resolved. Any inverse model that outputs identical predictions for State A and State B yields a predicted difference $\Delta \hat{d} = \hat{d}_A - \hat{d}_B = \mathbf{0}$, completely failing to separate the failure modes.

---

## 4. Physical Non-Identifiability under Symmetric Sensing

To empirically quantify the difficulty of the inverse problem, we simulate the baseline 3-story, 1-bay frame (`SOURCE_A`) using validated finite-element dynamic simulations implemented in OpenSeesPy. The frame has bay width $L = 6.0\text{ m}$, story height $h = 3.5\text{ m}$, elastic modulus $E_0 = 200\text{ GPa}$, and mass density $\rho = 7850\text{ kg/m}^3$. Damping is modeled via Rayleigh damping ($\zeta = 0.02$ at the first two modes, $f_1 = 2.064\text{ Hz}$ and $f_2 = 6.945\text{ Hz}$).

### 4.1 Numerical Equivalence of Symmetric States
Under standard horizontal floor acceleration sensing ($S0$, measuring centerline horizontal accelerations at Floors 1, 2, and Roof), we compare the dynamic response histories produced by State A and State B under identical earthquake excitations from the PEER strong ground motion database.

As documented in [`results/ill_posedness_metrics.json`](file:///Users/rahul/inverse-fno-damage/results/ill_posedness_metrics.json), the relative $L_2$ norm difference between the horizontal acceleration outputs is:
$$\frac{\|\mathbf{y}_A(t) - \mathbf{y}_B(t)\|_2}{\|\mathbf{y}_A(t)\|_2} < 0.0014 \quad (< 0.14\%)$$
Under realistic field instrumentation conditions where sensor noise $\sigma_\eta \ge 1\%$, the signal difference between State A and State B is completely submerged within the noise floor ($SNR < 0.14$). 

### 4.2 Sensitivity Jacobian and Near-Null Space Alignment
Linearizing the forward operator $\mathcal{F}$ around the undamaged baseline $d = \mathbf{0}$ yields the parameter-to-response sensitivity Jacobian $\mathbf{J} \in \mathbb{R}^{(C \cdot T) \times E}$:
$$\mathbf{J}_{ij} = \frac{\partial y_i}{\partial d_j}, \quad i \in \{1, \dots, C \cdot T\}, \quad j \in \{1, \dots, E\}$$
Singular Value Decomposition (SVD) of $\mathbf{J} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$ reveals that for horizontal floor sensing ($S0$), the singular values decay precipitously. Crucially, the right-singular vector $v_E$ corresponding to the smallest singular value satisfies:
$$|\langle v_E, v_{AB} \rangle| > 0.99$$
This demonstrates a central empirical finding: **the canonical bilateral damage direction $v_{AB}$ aligns almost perfectly with the numerical near-null space of the horizontal observation operator**. As a consequence, lateral floor acceleration sensors are physically incapable of distinguishing left-column damage from right-column damage in a symmetric frame under the prescribed sensing and noise assumptions.

---

## 5. Noise-Whitened Observability Framework

Because raw sensor channels involve disparate physical quantities (accelerations in $\text{m/s}^2$, displacements in $\text{m}$, strains in $\text{m/m}$), raw Jacobian singular values are unit-dependent and physically misleading. To rigorously evaluate physical observability, we formulate the **noise-whitened sensitivity framework**.

### 5.1 Noise-Whitened Fisher Information
Let $\mathbf{\Sigma}_\eta \in \mathbb{R}^{(C \cdot T) \times (C \cdot T)}$ denote the noise covariance matrix across all channels. Assuming independent stationary noise with variance $\sigma_{\eta, c}^2$ for each channel $c \in \{1, \dots, C\}$, the noise-whitened Jacobian is:
$$\mathbf{J}_w = \mathbf{\Sigma}_\eta^{-1/2} \mathbf{J} \in \mathbb{R}^{(C \cdot T) \times E}$$
The Fisher Information Matrix (FIM) in damage parameter space is given by:
$$\mathbf{F}_w = \mathbf{J}_w^T \mathbf{J}_w = \mathbf{J}^T \mathbf{\Sigma}_\eta^{-1} \mathbf{J} \in \mathbb{R}^{E \times E}$$

### 5.2 Directional Fisher Sensitivity Metric ($\sqrt{I_{AB}}$)
To quantify the physical distinguishability of the bilateral states, we project the noise-whitened FIM along the canonical bilateral unit vector $v_{AB}$:
$$I_{AB} = v_{AB}^T \mathbf{F}_w v_{AB} = \|\mathbf{J}_w v_{AB}\|_2^2$$
The metric $\sqrt{I_{AB}}$ is explicitly defined as the **noise-whitened directional Fisher sensitivity**, representing the signal-to-noise ratio of the differential response $(\mathbf{y}_A - \mathbf{y}_B)$ induced by bilateral damage.

### 5.3 Sensor Modality Hierarchy
We evaluate four distinct sensor configurations on `SOURCE_A` at $2\%$ sensor noise:
1. **$S0$ (Horizontal Accel):** 3 channels (Floors 1, 2, Roof horizontal).
2. **$S1$ (Horizontal + Vertical Accel):** 9 channels (3 horizontal + 6 vertical accelerations at beam-column joints).
3. **$S2$ (Horizontal Accel + Column Axial Strain):** 9 channels (3 horizontal + 6 axial strain gauges on exterior columns).
4. **$S4$ (Multimodal Union):** 15 channels (3 horizontal + 6 vertical accelerations + 6 column strains).

As audited in [`results/observability/phase5_5_forensic_audit.json`](file:///Users/rahul/inverse-fno-damage/results/observability/phase5_5_forensic_audit.json), the resulting noise-whitened directional Fisher sensitivities $\sqrt{I_{AB}}$ are:
$$\sqrt{I_{AB}}(S0) \approx 5.6$$
$$\sqrt{I_{AB}}(S2) \approx 1344.2 \quad (\text{ratio to } S0 \approx 240\times \text{ under the prescribed noise model})$$
$$\sqrt{I_{AB}}(S1) \approx 1829.6 \quad (\text{ratio to } S0 \approx 326\times \text{ under the prescribed noise model})$$
$$\sqrt{I_{AB}}(S4) \approx 2270.5 \quad (\text{ratio to } S0 \approx 405\times \text{ under the prescribed noise model})$$

### 5.4 The Observability Conclusion and the S2 Nuance
This mathematical framework demonstrates that under the tested assumptions, while horizontal floor acceleration ($S0$) possesses negligible directional sensitivity along $v_{AB}$, vertical acceleration ($S1$) and axial strains ($S2$) provide substantial directional Fisher sensitivity by capturing dynamic joint rocking and axial load transfer. 

Crucially, this analysis establishes physical observability under the linearized Gaussian noise model, but **does not guarantee neural learnability**. Configuration $S2$ achieves only $50.0\%$ attribution during gradient descent learning despite possessing $\sqrt{I_{AB}} = 1344.2$, demonstrating that physical observability is necessary but insufficient for neural learnability under the tested configurations.

---

## 6. Baseline Inverse Learning and Mean-Seeking Collapse

### 6.1 Forward Neural Operator Benchmark (Phase 2)
In Phase 2, we benchmarked a 1D Fourier Neural Operator to approximate the forward wave operator mapping $(d, a_g(t)) \mapsto \mathbf{y}(t)$ using 3 spectral convolution layers with 24 Fourier modes and channel width 48.

As evaluated in [`results/forward_fno_metrics.json`](file:///Users/rahul/inverse-fno-damage/results/forward_fno_metrics.json) on physical acceleration units ($\text{m/s}^2$) across 32 held-out test simulations:
- **Waveform Relative $L_2$ Error:** $36.71\% \pm 8.42\%$ (healthy frames: $34.74\% \pm 8.53\%$; damaged frames: $37.37\% \pm 8.28\%$).
- **Waveform Pearson Correlation:** $\rho = 0.9030 \pm 0.041$ (healthy: $0.9129$; damaged: $0.8996$).
- **Peak Floor Acceleration Error:** $13.47\% \pm 7.21\%$.

*Metric Definition and Context:*  
In seismic waveform modeling, the $36.71\%$ relative $L_2$ error reflects the extreme phase sensitivity of high-frequency oscillatory signals under $L_2$ norms. Concurrently, the high Pearson correlation ($\rho = 0.9030$) and low peak acceleration error ($13.47\%$) confirm that the forward operator accurately captures structural modal drift and peak response envelopes. Separately, in parameter space, earlier baseline Inverse FNO tests recorded a damage severity Mean Absolute Error of $\text{MAE} = 0.0381$ ($3.81\%$ damage error, audited in [`results/phase6/ablation_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6/ablation_results.json)). These metrics quantify distinct spaces: response-space waveform tracking ($36.71\%$ rel-$L_2$) versus parameter-space damage severity regression ($\text{MAE} = 0.0381$).

### 6.2 Inverse FNO and Mean-Seeking Collapse (Phase 3)
In Phase 3, we trained a standard Inverse FNO to map sparse acceleration histories directly to the 9-element damage vector $d \in \mathbb{R}^9$ using standard MSE loss:
$$L_{\text{MSE}} = \frac{1}{E} \sum_{e=1}^E (\hat{d}_e - d_e)^2$$
When trained on randomly distributed damage fields, the model collapses to the conditional mean:
$$\hat{d} \to \mathbb{E}[d \mid \mathbf{y}] \approx \mathbf{0}$$
The predicted damage values cluster below $0.02$, failing to identify severe $30\%$ localized damage.

### 6.3 Regularization Ablations
In [`results/regularization_ablation.json`](file:///Users/rahul/inverse-fno-damage/results/regularization_ablation.json), we ablated physics-informed regularizations:
- **Sparsity ($\ell_1$):** $L_1 = \lambda_1 \sum_e |\hat{d}_e|$ ($\lambda_1 = 0.01$)
- **Total Variation (TV):** $L_{\text{TV}} = \lambda_{\text{TV}} \sum_{(u, v) \in \mathcal{E}} |\hat{d}_u - \hat{d}_v|$ ($\lambda_{\text{TV}} = 0.005$)
- **Forward Cycle-Consistency:** $L_{\text{cycle}} = \lambda_c \|\mathcal{F}_{\text{frozen}}(\hat{d}, a_g) - \mathbf{y}\|_2^2$

While $\ell_1$ regularization successfully suppressed diffuse background predictions and restored peak damage magnitudes to $\sim 0.25$, it **completely failed on bilateral damage pairs**. When presented with State A or State B, the regularized inverse model produced an identical, symmetric prediction (predicting $15\%$ damage on both Column 1 and Column 2 simultaneously). Regularization enforces spatial sparsity, but cannot resolve physical non-uniqueness.

---

## 7. Sensor Sparsity and Placement Analysis

In Phase 4, we examined how sensor count and spatial density affect damage localization across non-symmetric damage scenarios ([`results/sensor_sparsity_sweep.json`](file:///Users/rahul/inverse-fno-damage/results/sensor_sparsity_sweep.json)):
- **3-Floor Deployment (Roof + Floor 2 + Floor 1):** Severity MAE $= 0.042$, Top-1 Member Localization $= 76.7\%$.
- **2-Floor Deployment (Roof + Floor 1):** Severity MAE $= 0.058$, Top-1 Member Localization $= 63.3\%$.
- **1-Floor Deployment (Roof Only):** Severity MAE $= 0.089$, Top-1 Member Localization $= 46.7\%$.

While general localization degrades gracefully as sensor density decreases, **no degree of horizontal floor sensor densification resolves the bilateral ambiguity**. Even with sensors at every floor slab, horizontal sensing remains blind to bilateral column damage. This provides empirical evidence that **sensor modality is substantially more critical than sensor quantity** for overcoming symmetry-induced ill-posedness under the tested configurations.

---

## 8. Symmetry-Aware DualStreamGFNO Architecture

To overcome the failure of standard inverse regression, we designed the **Symmetry-Aware Dual-Stream Graph Fourier Neural Operator (DualStreamGFNO)**, illustrated schematically in Figure 3 and implemented in `src/ml/gfno.py`.

```
                    ┌────────────────────────────────────────────────────────┐
                    │                      INPUT STREAMS                     │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
  ┌──────────────────────────────┐                              ┌──────────────────────────────┐
  │   BRANCH A: Global Stream    │                              │   BRANCH B: Asymmetric Node  │
  │ Horizontal Accel + a_g(t)    │                              │   Joint Accelerations &      │
  │ (C=4, T=250, width=48)       │                              │   Strains (C=32, width=48)   │
  └──────────────┬───────────────┘                              └──────────────┬───────────────┘
                 │ Temporal FNO                                                │ Temporal FNO
                 │ (24 Fourier Modes)                                          │ (24 Fourier Modes)
                 ▼                                                             ▼
  ┌──────────────────────────────┐                              ┌──────────────────────────────┐
  │  Global Embedding h_global   │                              │ Node Features H ∈ R^(8 x 64) │
  │  (R^64 after pooling)        │                              └──────────────┬───────────────┘
  └──────────────┬───────────────┘                                             │
                 │                                                             ▼
                 │                                              ┌──────────────────────────────┐
                 │                                              │   2x Graph Convolutions      │
                 │                                              │   H_G = A_norm H W_G         │
                 │                                              └──────────────┬───────────────┘
                 │                                                             │
                 │                                                             ▼
                 │                                              ┌──────────────────────────────┐
                 │                                              │  LATENT SYMMETRY PROJECTION  │
                 │                                              │  H^+ = 0.5 * (H + P H)       │
                 │                                              │  H^- = 0.5 * (H - P H)       │
                 │                                              └──────────────┬───────────────┘
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │ Concatenation & Fusion Linear
                                                ▼
                                 ┌──────────────────────────────┐
                                 │   Fused Joint Embeddings     │
                                 │   h_v ∈ R^64 (8 joints)      │
                                 └──────────────┬───────────────┘
                                                │
                                                ▼ NodeToEdgeDecoder
                                 ┌──────────────────────────────┐
                                 │   Element Embeddings         │
                                 │   h_e ∈ R^64 (9 elements)    │
                                 └──────────────┬───────────────┘
                                                │
                         ┌──────────────────────┴──────────────────────┐
                         ▼                                             ▼
          ┌──────────────────────────────┐              ┌──────────────────────────────┐
          │         SUPPORT HEAD         │              │        SEVERITY HEAD         │
          │ p_e = σ(W_s h_e) ∈ [0, 1]    │              │ μ_e = 0.5 σ(W_m h_e)         │
          └──────────────┬───────────────┘              └──────────────┬───────────────┘
                         │                                             │
                         └──────────────────────┬──────────────────────┘
                                                ▼
                                 ┌──────────────────────────────┐
                                 │      HIERARCHICAL DAMAGE     │
                                 │      d_hat_e = p_e * μ_e     │
                                 └──────────────────────────────┘
```

### 8.1 Dual-Stream Processing
The architecture splits sensor inputs into two distinct branches:
- **Branch A (Global Horizontal Stream):** Processes the 3 horizontal floor accelerations plus ground motion $a_g(t)$ through a 3-layer 1D Temporal FNO (width 48, 24 Fourier modes) to extract macroscopic dynamic drift features $h_{\text{global}} \in \mathbb{R}^{64}$.
- **Branch B (Local Asymmetric Stream):** Maps joint-level vertical accelerations and member strain gauges onto the structural graph nodes $V = \{1, \dots, 8\}$. A dedicated Temporal FNO extracts localized temporal embeddings, which are propagated across structural joints via 2 Graph Convolutional layers:
  $$\mathbf{H}^{(l+1)} = \text{GELU}\left( \mathbf{A}_{\text{norm}} \mathbf{H}^{(l)} \mathbf{W}_G \right)$$
  where $\mathbf{A}_{\text{norm}} = \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$ is the normalized structural adjacency matrix.

### 8.2 Latent Symmetry Decomposition
To prevent dominant symmetric dynamics from drowning out asymmetric signals, Branch B projects node embeddings $\mathbf{H} \in \mathbb{R}^{V \times 64}$ into symmetric and antisymmetric subspaces via the reflection involution permutation $\mathbf{P}$:
$$\mathbf{H}^+ = \frac{1}{2} \left( \mathbf{H} + \mathbf{P} \mathbf{H} \right) \quad (\text{Symmetric Mode})$$
$$\mathbf{H}^- = \frac{1}{2} \left( \mathbf{H} - \mathbf{P} \mathbf{H} \right) \quad (\text{Antisymmetric Mode})$$
By construction, $\mathbf{P} \mathbf{H}^+ = \mathbf{H}^+$ and $\mathbf{P} \mathbf{H}^- = -\mathbf{H}^-$. The fused representation concatenates $[h_{\text{global}} \,\|\, \mathbf{H}^+ \,\|\, \mathbf{H}^-]$ before projecting to joint representations $h_v \in \mathbb{R}^{64}$.

### 8.3 Node-to-Edge Hierarchical Decoding
Member representations $h_e \in \mathbb{R}^{64}$ are decoded by combining joint representations at member endpoints with static member features (length, cross-sectional area, moment of inertia):
$$h_e = \text{MLP}\left( [h_{u(e)} \,\|\, h_{v(e)} \,\|\, \theta_e] \right)$$
Damage is predicted via coupled **Hierarchical Heads**:
1. **Support Classification Head:** Predicts the probability that element $e$ is damaged:
   $$p_e = \sigma(\mathbf{W}_s h_e) \in [0, 1]$$
2. **Severity Regression Head:** Predicts the damage magnitude:
   $$\mu_e = 0.5 \cdot \sigma(\mathbf{W}_m h_e) \in [0, 0.5]$$
The final damage prediction is the masked product:
$$\hat{d}_e = p_e \cdot \mu_e$$
This formulation prevents the zero-gradient flat basin typical of unconstrained MSE regression.

### 8.4 Matched Bilateral Pair Supervision
As discovered in Phase 6.1, ordinary supervised training of DualStreamGFNO still fails to resolve bilateral states because horizontal floor accelerations carry $>98\%$ of initial gradient power, trapping the network in a local minimum near the symmetric mean.

To force the network to utilize asymmetric signals, we formulate **Matched Bilateral Pair Supervision**. Training batches contain paired simulations driven by the exact identical earthquake record $a_g(t)$ for State A and State B. 

Let $\Delta \hat{d} = \hat{d}_A - \hat{d}_B$ be the predicted damage difference. We introduce two complementary pairwise loss terms:

1. **Directional Alignment Loss ($L_{\text{dir}}$):** Penalizes angular misalignment between $\Delta \hat{d}$ and the true canonical direction $v_{AB}$:
   $$L_{\text{dir}} = 1 - \frac{\langle \Delta \hat{d}, v_{AB} \rangle}{\|\Delta \hat{d}\|_2 \|v_{AB}\|_2 + \epsilon}$$
   When $\Delta \hat{d}$ is perfectly aligned with $v_{AB}$, $L_{\text{dir}} = 0$; when orthogonal, $L_{\text{dir}} = 1$; when inverted, $L_{\text{dir}} = 2$.
2. **Contrastive Margin Loss ($L_{\text{pair}}$):** Penalizes predicted separation that falls below a target physical margin $m$:
   $$L_{\text{pair}} = \max\left(0, \, m - \frac{\langle \Delta \hat{d}, v_{AB} \rangle}{\|v_{AB}\|_2}\right)$$
   We set $m = 0.15$ (approximately $35\%$ of full true separation $\|d_A - d_B\|_2 = 0.4243$). Crucially, $L_{\text{pair}}$ is antisymmetric: swapping A and B triggers a heavy penalty, preventing inverted attribution.

The total optimization objective is:
$$L_{\text{total}} = L_{\text{base}} + \lambda_{\text{dir}} L_{\text{dir}} + \lambda_{\text{pair}} L_{\text{pair}}$$
where $L_{\text{base}} = L_{\text{BCE}}(p, y_{\text{supp}}) + L_{\text{Huber}}(\mu, y_{\text{sev}}) \cdot \mathbb{I}(y_{\text{supp}} > 0)$, with frozen hyperparameter weights $\lambda_{\text{dir}} = 0.05$ and $\lambda_{\text{pair}} = 0.01$.

---

## 9. Single-Structure Bilateral Attribution Benchmark (Phase 6.2)

### 9.1 Experimental Protocol
The Phase 6.2 benchmark evaluates the proposed DualStreamGFNO with bilateral pair supervision on the baseline 3-story frame (`SOURCE_A`). Training is conducted on 40 bilateral pairs, and evaluation is performed on $N = 30$ independent bilateral evaluations across 15 held-out validation pairs ($15 \times 2$ evaluations) driven by 14 unique earthquake records strictly disjoint from training.

The binary attribution metric classifies each evaluation as State A if $\hat{d}_{\text{Col1}} > \hat{d}_{\text{Col2}}$, and State B otherwise. Statistical significance is assessed via the exact two-sided binomial test under the null hypothesis of chance performance ($p_0 = 0.50$).

### 9.2 Empirical Results
The audited results from [`results/phase6_2/bilateral_results.json`](file:///Users/rahul/inverse-fno-damage/results/phase6_2/bilateral_results.json) are summarized in Table 1:

| Sensor Configuration | Attribution Successes | Attribution Accuracy | 95% Clopper-Pearson Exact CI | Two-Sided $p$-value vs. Chance ($p_0=0.5$) | Learned Cosine $\cos(\Delta \hat{d}, v_{AB})$ | Mean Predicted Separation $\|\Delta \hat{d}\|_2$ | True Separation Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$S0$ (Horizontal Accel)** | 15 / 30 | **50.0%** | [31.3%, 68.7%] | $p = 1.000$ | $-0.1151 \pm 0.122$ | $9.04 \times 10^{-6}$ | $0.002\%$ |
| **$S1$ (Horiz + Vert Accel)** | 24 / 30 | **80.0%** | [61.4%, 92.3%] | $\mathbf{p = 0.0014}$ | $\mathbf{+0.7272 \pm 0.442}$ | $0.1576$ | $37.15\%$ |
| **$S2$ (Horiz + Axial Strain)** | 15 / 30 | **50.0%** | [31.3%, 68.7%] | $p = 1.000$ | $+0.1047 \pm 0.111$ | $0.000328$ | $0.08\%$ |
| **$S4$ (Multimodal Union)** | 27 / 30 | **90.0%** | [73.5%, 97.9%] | $\mathbf{p = 9.0 \times 10^{-6}}$ | $\mathbf{+0.9420 \pm 0.098}$ | $\mathbf{0.1671}$ | $\mathbf{39.40\%}$ |

### 9.3 Findings and Analysis
1. **$S4$ Multimodal Success:** Achieving $90.0\%$ attribution accuracy ($p = 9.0 \times 10^{-6}$) with mean directional cosine $+0.9420$ and separation $0.1671$ confirms that explicit pair supervision successfully breaks bilateral ambiguity when multimodal sensing is available.
2. **$S0$ Physical Baseline:** $S0$ remains locked at $50.0\%$ accuracy with near-zero separation ($9.04 \times 10^{-6}$). This confirms that pairwise loss **cannot invent information that is physically absent** from the sensor traces, serving as a critical negative control.
3. **The $S2$ Observability–Learnability Gap:** Despite possessing a high directional Fisher sensitivity ($\sqrt{I_{AB}} = 1344.2$), configuration $S2$ fails completely ($50.0\%$ accuracy, separation $0.0003$). Because column axial strain amplitudes are small ($\sim 10^{-5}$) compared to floor accelerations ($\sim 1.0\text{ m/s}^2$), standard optimizers prioritize fitting floor accelerations, ignoring strain gradients. Physical observability is necessary but insufficient for neural learnability under the tested configurations.

---

## 10. Cross-Structure Generalization Benchmark (Phase 7)

In Phase 7, we evaluated whether the learned inverse operator transfers across varied structural geometries, stiffness properties, and frame topologies.

### 10.1 Multi-Structure Dataset Accounting
To prevent data contamination, we generated an audited benchmark of **530 validated finite-element dynamic simulations implemented in OpenSeesPy** across 120 PEER ground motions partitioned into strictly disjoint subsets (70 Train, 20 Validation, 30 Test):
- **Training:** 90 simulations on `SOURCE_A` (P1) plus 140 simulations on 4 diverse training structures $B_{\text{train\_1..4}}$ (P2).
- **Validation:** 60 simulations across `SOURCE_A` and $B_{\text{train\_1}}$.
- **Test:** 240 simulations (120 bilateral pairs, 30 per level) driven by 30 strictly held-out earthquakes (`RSN0091` to `RSN0120`).

All graph edge features adhere strictly to the pristine feature contract ($E_{\text{norm}} = 1.0$, verified by 4 automated regression tests).

### 10.2 Evaluation Hierarchy Across 4 Generalization Levels
We evaluated models trained under Protocol P2 (multi-structure training) under multimodal sensing $S4$ across four held-out test levels ($N = 30$ independent pairs per level):
- **Level 1 (In-Distribution Structure, Earthquake OOD):** `SOURCE_A`.
- **Level 2A (Parametric Interpolation OOD):** Structures $B_{\text{int\_1}}$ and $B_{\text{int\_2}}$, whose parameters lie within the convex hull of training structures.
- **Level 2B (Parametric Extrapolation OOD):** Structures $B_{\text{ext\_soft}}$ and $B_{\text{ext\_stiff}}$, with stiffness $E$ and density $\rho$ perturbed by $\pm 20\%$ outside the training convex hull.
- **Level 3 (Structural Topological OOD):** Structure $C_{\text{4story}}$, an unseen 4-story, 10-node, 12-element frame.

### 10.3 Empirical Findings (Clean Phase 7 Baseline)
As audited in [`results/experiments/FINAL_PHASE7_CLEAN_AUDIT.json`](file:///Users/rahul/inverse-fno-damage/results/experiments/FINAL_PHASE7_CLEAN_AUDIT.json), under pristine physics conditioning at 12 training epochs:
- **Continuous Directional Sensitivity:** Directional cosine alignment transfers zero-shot and remains consistently positive across all levels (Level 1: $+0.467$, Level 2A: $+0.480$, Level 2B: $+0.534$, Level 3: $+0.489$).
- **Discrete Attribution Collapse:** Finite bilateral damage separation collapses to near-zero ($\sim 10^{-6}$), and discrete attribution accuracy drops to chance level ($50.0\%$) across all four levels (Level 1: $50.0\%$, Level 2A: $50.0\%$, Level 2B: $50.0\%$, Level 3: $50.0\%$).

*Interpretation:* Continuous directional sensitivity transfers zero-shot across structural variations, while finite bilateral damage separation collapses and discrete attribution remains at chance under the tested protocol.

---

## 11. Controlled Training-Budget Ablation

To determine whether the multi-structure discrete attribution collapse was primarily caused by undertraining (12 epochs) across diverse structures, we executed a rigorous, controlled training-budget ablation on Apple Silicon GPU (`mps`).

### 11.1 Experimental Matrix
- **Budgets Evaluated:** $12, 25, 50, 100$ epochs.
- **Deterministic Replications:** 3 random seeds ($42, 101, 2024$) trained along continuous checkpoint trajectories.
- **Protocols & Modalities:** Both P1 and P2 across $S0, S1, S4$, totaling **72 discrete model evaluations**.
- **Controls:** Zero test-set model selection, identical data splits, zero leakage.

### 11.2 Empirical Results
Table 2 summarizes the budget-scaling trajectory for the proposed model under Protocol P2 and multimodal sensing $S4$:

| Training Budget | Level 1 ID Accuracy | Level 1 ID Cosine | Level 1 Separation $\|\Delta \hat{d}\|_2$ | Level 2A (Interp) Cosine | Level 2B (Extrap) Cosine | Level 3 (Topology) Cosine | Mean Training Loss |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **12 epochs** | $50.0\% \pm 0.0\%$ | $+0.467 \pm 0.037$ | $2.68 \times 10^{-6}$ | $+0.480 \pm 0.038$ | $+0.534 \pm 0.055$ | $+0.489 \pm 0.012$ | 0.6011 |
| **25 epochs** | $50.0\% \pm 0.0\%$ | $+0.852 \pm 0.115$ | $7.68 \times 10^{-6}$ | $+0.892 \pm 0.072$ | $+0.802 \pm 0.204$ | $+0.777 \pm 0.158$ | 0.5143 |
| **50 epochs** | $50.0\% \pm 0.0\%$ | $+0.897 \pm 0.038$ | $1.23 \times 10^{-5}$ | $+0.896 \pm 0.016$ | $+0.824 \pm 0.096$ | $+0.837 \pm 0.090$ | 0.5246 |
| **100 epochs** | $50.0\% \pm 0.0\%$ | $+0.850 \pm 0.205$ | $1.83 \times 10^{-5}$ | $+0.841 \pm 0.216$ | $+0.813 \pm 0.204$ | $+0.798 \pm 0.199$ | 0.5553 |

*Note: Accuracies and cosines report mean $\pm$ sample standard deviation across seeds 42, 101, and 2024. At 100 epochs, individual seeds achieve Level 1 cosines of $+0.993$ (Seed 42) and $+0.996$ (Seed 2024), and Level 3 cosines of $+0.921$ (Seed 42) and $+0.955$ (Seed 2024).*

### 11.3 Interpretative Hierarchy
We evaluated five competing scientific interpretations of this result:
- **Statement A ("Undertraining causes the attribution failure"):** **REFUTED.** Scaling training epochs by $8.3\times$ produced exactly $0.0\%$ gain in discrete attribution accuracy.
- **Statement B ("Optimization budget is not the dominant explanation"):** **STRONGLY SUPPORTED.** Within the tested 12–100 epoch training budgets, increasing optimization budget improved directional alignment (reaching $\approx +0.90$) but did not recover finite bilateral separation.
- **Statement C ("Directional sensitivity is learned without finite separation"):** **VERIFIED.** Observed directly across 72 evaluations.
- **Statement D ("Cross-structure distribution shift is a plausible mechanism"):** **STRONGLY SUPPORTED.** In single-structure training (Phase 6.2), identical losses achieved separation $0.1671$; introducing multi-structure parameter dispersion collapses separation to $10^{-5}$.
- **Statement E ("Cross-structure distribution shift is proven to be the unique fundamental cause"):** **NOT SUPPORTED.** Demonstrating correlation and physical plausibility does not constitute a mathematical uniqueness proof; confounding factors (e.g. finite pair sample density per structure) cannot be ruled out without further theoretical proofs.

---

## 12. The Empirical Direction–Magnitude Decoupling Phenomenon

The central discovery of this experimental program is formalized as an observed empirical finding:

> ### **Definition: Empirical Direction–Magnitude Decoupling Phenomenon**
> *In learning inverse operators for symmetric dynamical systems under structural parameter dispersion and sparse sensing, gradient-based optimization successfully aligns the orientation of the predicted damage difference vector with the physically observable bilateral sensitivity direction ($\cos(\Delta \hat{d}, v_{AB}) \to +0.85$ to $+0.90$), while simultaneously failing to scale the magnitude of the predicted separation ($\|\Delta \hat{d}\|_2 \sim 10^{-5} \ll 0.30$). Consequently, continuous directional sensitivity transfers robustly across structures and topologies, while discrete finite-state attribution collapses to chance level ($50.0\%$). This is characterized as an observed empirical phenomenon under the tested configurations, rather than a universal mathematical theorem.*

### 12.1 Observed Mechanism
The directional alignment loss $L_{\text{dir}} = 1 - \cos(\Delta \hat{d}, v_{AB})$ provides scale-invariant gradient signals that rotate $\Delta \hat{d}$ toward $v_{AB}$. However, the contrastive margin loss $L_{\text{pair}} = \max(0, m - \langle \Delta \hat{d}, v_{AB} \rangle)$ requires the network to scale the magnitude of $\Delta \hat{d}$ beyond margin $m = 0.15$. 

In a single structure (Phase 6.2), the network easily learns a fixed offset. Under multi-structure parameter variations, however, the observed behavior is consistent with predictions collapsing toward a shared symmetric solution under cross-structure distribution shift. To maintain low base Huber loss across diverse structures, the shared encoder predictions remain compressed near the population mean, penalizing macroscopic separation during gradient descent.

---

## 13. Forensic Failure Analysis

To maintain scientific rigor, we classify the project's failure modes across three epistemic categories:

### 13.1 Observed Facts (Empirically Verified)
1. **$S0$ Physical Blindness:** Pure horizontal floor acceleration sensing cannot separate bilateral damage states in symmetric frames (accuracy $= 50.0\%$, response diff $< 0.14\%$).
2. **$S2$ Learnability Failure:** Column axial strain provides strong physical Fisher sensitivity ($\sqrt{I_{AB}} = 1344.2$) but yields $50.0\%$ accuracy under standard optimization.
3. **Multi-Structure Attribution Collapse:** In the clean multi-structure benchmark, discrete attribution remains at $50.0\%$ despite training up to 100 epochs.
4. **Directional Transfer:** Directional cosine alignment reaches $+0.80$ to $+0.85$ on unseen frames and topologies.

### 13.2 Mechanistic Interpretations (Strongly Supported)
1. **Gradient Dominance in $S2$:** High-amplitude floor accelerations ($1.0\text{ m/s}^2$) dominate the backpropagation graph, drowning out localized micro-strain gradients ($\sim 10^{-5}\text{ m/m}$) during early optimization.
2. **Symmetric Compression Under Parameter Dispersion:** Variance in structural frequencies across multi-structure datasets acts as an implicit regularizer consistent with predictions collapsing toward a shared symmetric mean.

### 13.3 Unresolved Hypotheses (Requiring Future Theory)
1. Whether adaptive margin scaling $\lambda_{\text{pair}}(t)$ or curriculum learning across structural families can overcome the Direction–Magnitude Decoupling without collapsing forward cycle-consistency.
2. Whether functional Sobolev norm formulations can establish an analytical lower bound on sample complexity for multi-structure inverse operator learning.

---

## 14. Limitations

1. **2D Idealization:** All structural simulations employ 2D planar frames, neglecting 3D torsional coupling and out-of-plane dynamic modes.
2. **Linear-Elastic Constitutive Models:** Element damage is modeled as fractional reduction in elastic stiffness $E I$; nonlinear material hysteretic degradation and geometric $P$-$\Delta$ effects are not simulated.
3. **Synthetic Numerical Data:** While validated OpenSeesPy finite-element dynamic simulations are employed, all traces are synthetic and include idealized $2\%$ Gaussian noise, lacking environmental temperature variations, foundation compliance, or non-structural component interference.
4. **Finite Earthquake Inventory:** Evaluations are conducted on 120 PEER strong ground motion records.
5. **Absence of Experimental Shake-Table Data:** All findings represent computational and operator-learning benchmarks that have not yet been validated on instrumented physical shake-table test specimens.

---

## 15. Future Directions

1. **Adaptive Margin Scheduling:** Developing dynamic loss weight schedulers that adapt $\lambda_{\text{pair}}$ as directional cosine alignment plateaus.
2. **Modality-Specific Normalization Layers:** Designing dedicated input normalization layers that balance gradient norms between high-amplitude accelerations and localized micro-strains.
3. **Equivariant Group Convolutions:** Incorporating exact $C_2$ reflection group equivariance directly into the graph message-passing layers.
4. **Experimental Shake-Table Benchmarks:** Validating the single-structure bilateral identification framework on instrumented laboratory frame specimens subjected to table accelerations.
5. **3D Structural Inversion:** Extending the dual-stream architecture to 3D asymmetric buildings with bidirectional seismic excitation.

---

## 16. Conclusion

In this work, we investigated the interplay of physical observability, structural symmetry, neural learnability, and distribution shift in neural-operator-based seismic structural damage inversion. 

The central conclusion of this study is that:
> **"Solving ill-posed seismic inverse problems with neural operators requires disentangling physical observability, neural learnability, and cross-structure transferability. Under the tested structural and sensing regimes, multimodal sensing combined with symmetry-aware pairwise supervision resolves substantial bilateral ambiguity on a single structure, whereas cross-structure distribution shift produces an empirical direction–magnitude decoupling in which directional sensitivity transfers more readily than finite damage separation."**

Specifically, our experiments establish that:
1. Structural symmetry under horizontal floor sensing ($S0$) induces an empirical near-null space where bilateral damage states are physically indistinguishable ($\sqrt{I_{AB}} \approx 5.6$).
2. Noise-whitened Fisher information demonstrates that asymmetric sensing modalities (vertical acceleration $S1$, multimodal $S4$) amplify directional Fisher sensitivity by $>300\times$.
3. Physical observability is necessary but insufficient for neural learnability under the tested configurations: column axial strains ($S2$) fail to train despite high Fisher sensitivity due to floor-acceleration gradient dominance.
4. On a single structure, explicit bilateral pair supervision enables DualStreamGFNO to achieve **$90.0\%$ bilateral attribution accuracy** ($p = 9.0 \times 10^{-6}$) under multimodal sensing $S4$.
5. Under cross-structure distribution shift, neural inverse operators exhibit an **empirical Direction–Magnitude Decoupling phenomenon**: continuous directional sensitivity transfers zero-shot across frame parameters and unseen topologies ($\cos \approx +0.80$ to $+0.85$), but macroscopic output separation collapses ($\sim 10^{-5}$), pinning discrete attribution to chance level ($50.0\%$).
6. Within the tested 12–100 epoch training budgets, increasing optimization budget refines directional orientation (reaching up to $+0.996$) but does not recover finite bilateral separation.

These results delineate the precise boundary between what neural inverse operators can physically observe, what they can numerically learn, and what they can reliably transfer across structural systems.

---

## 17. Publication Figure Plan

The manuscript utilizes eight core figures generated from existing repository artifacts:

| Figure ID | Purpose / Title | Source Artifact File | Panel Layout | X-Axis | Y-Axis | Key Scientific Message |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Figure 1** | Physical Non-Identifiability under Horizontal Sensing | [`reports/figures/observability/fig1_damage_states_A_vs_B.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig1_damage_states_A_vs_B.png), [`fig2_response_overlay_A_vs_B.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig2_response_overlay_A_vs_B.png) | (a) Frame schematic showing State A vs B; (b) Response overlay; (c) Difference signal vs noise. | Time $t$ ($s$) | Floor Accel ($\text{m/s}^2$) | Difference between bilateral states is $<0.14\%$, submerged in $2\%$ sensor noise. |
| **Figure 2** | Observability Framework & Singular Spectra | [`reports/figures/observability/fig6_singular_spectra_comparison.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig6_singular_spectra_comparison.png), [`fig10_bilateral_direction_svd_decomposition.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig10_bilateral_direction_svd_decomposition.png) | (a) Jacobian singular value spectra S0..S4; (b) Singular vector projection along $v_{AB}$. | Singular mode index | Singular value $\sigma_k$ (log scale) | $S0$ aligns with smallest singular vector; $S1$ and $S4$ move $v_{AB}$ into dominant observable space. |
| **Figure 3** | DualStreamGFNO Architecture Diagram | [`reports/figures/phase6/fig1_phase6_architecture_diagram.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6/fig1_phase6_architecture_diagram.png) | Full dataflow: Branch A, Branch B, Latent Symmetry Projection, Node-to-Edge Decoder, Hierarchical Heads. | Processing stage | Latent dimension | Structural representation explicitly separates symmetric $H^+$ and antisymmetric $H^-$ features. |
| **Figure 4** | Single-Structure Bilateral Attribution Benchmark | [`reports/figures/phase6_2/fig1_bilateral_confusion.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig1_bilateral_confusion.png), [`fig3_predicted_separation.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig3_predicted_separation.png) | (a) Confusion matrices S0, S1, S2, S4; (b) Recovered damage separation $\|\Delta \hat{d}\|_2$. | True State (A / B) | Predicted State (A / B) | $S4$ achieves $90.0\%$ attribution ($p = 9.0\times 10^{-6}$); separation reaches $0.1671$ ($39.4\%$ of true). |
| **Figure 5** | The Observability vs. Learnability Gap | [`reports/figures/phase6_2/fig4_sensor_comparison.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig4_sensor_comparison.png) | Scatter plot: Directional Fisher sensitivity $\sqrt{I_{AB}}$ vs. Learned Cosine Alignment. | $\sqrt{I_{AB}}$ (Fisher Sensitivity) | Learned Cosine $\cos(\Delta \hat{d}, v_{AB})$ | $S2$ has high Fisher sensitivity ($1344.2$) but collapses to chance, demonstrating observability $\ne$ learnability under tested settings. |
| **Figure 6** | Cross-Structure Generalization Benchmark | [`reports/figures/phase7/fig1_cross_structure_benchmark.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase7/fig1_cross_structure_benchmark.png), [`fig3_c4story_topological_inversion.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase7/fig3_c4story_topological_inversion.png) | (a) Performance across Levels 1, 2A, 2B, 3; (b) Zero-shot damage reconstruction on 4-story frame. | Generalization Level | Attribution Acc & Cosine | Directional cosine remains $+0.80$ to $+0.85$ across all levels; discrete attribution collapses to $50.0\%$. |
| **Figure 7** | Training-Budget Direction–Magnitude Decoupling | [`results/experiments/phase7_training_budget/figures/plot1_accuracy_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot1_accuracy_vs_budget.png), [`plot2_cosine_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot2_cosine_vs_budget.png), [`plot3_separation_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot3_separation_vs_budget.png) | 3-panel progression: (a) Accuracy vs Budget; (b) Cosine vs Budget; (c) Separation vs Budget. | Training Epochs ($12, 25, 50, 100$) | Metric Value | Cosine rises to $+0.897$, separation stays $\sim 10^{-5}$, accuracy remains flat at $50.0\%$. |
| **Figure 8** | Seed Optimization Trajectories on Topology OOD | [`results/experiments/phase7_training_budget/figures/plot5_seed_trajectories.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot5_seed_trajectories.png) | Individual seed trajectories (42, 101, 2024) across budgets for $C_{\text{4story}}$. | Training Epochs | Directional Cosine on L3 | Seeds 42 and 2024 reach $+0.921$ and $+0.955$ on unseen 4-story frame; discrete separation remains unrecovered. |

---

## 18. Publication Table Plan

The manuscript incorporates six structured evidence tables:
- **Table 1:** Single-Structure Bilateral Attribution Results (Phase 6.2 Benchmark; Section 9).
- **Table 2:** Training-Budget Scaling Trajectory across 12, 25, 50, 100 Epochs (Section 11).
- **Table 3:** Structural Configurations and Dynamic Properties Inventory (Evidence Matrix Section 3).
- **Table 4:** Sensor Modality Configurations and Noise-Whitened Directional Fisher Sensitivity (Section 5 & Evidence Matrix Section 4).
- **Table 5:** Multi-Structure Generalization Across Levels 1, 2A, 2B, 3 (Section 10).
- **Table 6:** Comprehensive Manuscript Claim and Evidence Mapping Matrix (Evidence Matrix Section 2).

---

## 19. Reproducibility Statement

All dynamic simulations, neural network models, training scripts, and evaluation pipelines are deterministically reproducible. The complete software stack is implemented in Python using PyTorch and OpenSeesPy. Ground motion data are drawn from 120 PEER strong motion acceleration records with strictly enforced, globally disjoint train/validation/test splits. Random seeds ($42, 101, 2024$) govern all dataset partitioning, synthetic noise injection, and network initializations. The repository includes 74 automated unit tests verifying structural mechanics, Guyan reduction tolerances, SVD singular spectra, loss gradient flow, and pristine feature leakage prevention (`scripts/gstack.py qa` passes in 3.09s). All experimental results, checkpoints, and generated figures are permanently archived in the repository artifact directory.

---

## 20. Statistical Language & Claim Boundary Audit

To ensure unwavering scientific integrity, the manuscript enforces the following claim boundaries:

### Claims We Genuinely Make (Fully Supported)
- Symmetric horizontal floor acceleration sensing induces an empirical near-null space ($\sqrt{I_{AB}} \approx 5.6$) where bilateral damage states differ by $<0.14\%$.
- Noise-whitened Fisher information demonstrates that vertical acceleration ($S1$) and multimodal sensing ($S4$) amplify directional Fisher sensitivity by $>300\times$.
- On a single structure, explicit bilateral pair supervision enables DualStreamGFNO to achieve **$90.0\%$ attribution accuracy** ($27/30$, $p = 9.0 \times 10^{-6}$) under multimodal sensing $S4$.
- Physical observability is necessary but insufficient for neural learnability under the tested configurations, as demonstrated by configuration $S2$ (column axial strain).
- Continuous directional damage sensitivity transfers zero-shot across structural parameters ($\cos \approx +0.84$) and to an unseen 4-story topology ($\cos \approx +0.80$).
- Within the tested 12–100 epoch budgets, increasing training budget refines directional alignment (up to $+0.996$) but does not produce finite bilateral separation under multi-structure parameter dispersion.

### Claims Made Only With Strict Qualification
- *Zero-Shot Transfer:* Applies strictly to the **orientation of continuous damage difference vectors** ($\cos \approx +0.80$ to $+0.85$); it does **not** apply to discrete damage state classification ($50.0\%$).
- *Multi-Structure Limitation:* We document an **empirical finite-separation collapse** within the tested neural operator and training budget setting; we do **not** claim a universal mathematical impossibility proof across all conceivable machine learning architectures.

### Claims Prohibited from the Manuscript
- *"Mathematical proof of ill-posedness"* $\to$ Enforced: *"Numerical and empirical demonstration of bilateral non-identifiability and near-null space alignment."*
- *"Solves inverse damage identification"* $\to$ Enforced: *"Partially resolves bilateral attribution under sufficiently informative single-structure sensing."*
- *"Eliminates the null space"* $\to$ Enforced: *"Physically moves the bilateral difference vector out of the numerical near-null space."*
- *"Guarantees generalization"* $\to$ Enforced: *"Demonstrates zero-shot directional transfer while documenting the collapse of discrete attribution."*
- *"Directional loss learns the true gradient"* $\to$ Enforced: *"Directional supervision aligns the network's pairwise difference vector with the canonical bilateral orientation vector."*
- *"Real-world deployment ready / AI identifies earthquake damage reliably"* $\to$ Enforced: *"A foundational benchmark conducted on validated OpenSeesPy finite-element models, providing baseline constraints prior to physical experimental testing."*

---
*Manuscript finalized and certified by Inverse-FNO-Damage Lead Authors and Virtual Research Review Engine.*
