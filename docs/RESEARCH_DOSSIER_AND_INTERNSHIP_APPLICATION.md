# Research Dossier & Graduate Internship Application

**Project:** `Inverse-FNO-Damage`  
**Author:** Raghvendra Singh Gahlot  
**Affiliation:** 2nd Year Undergraduate, Department of Civil Engineering, MBM University, Jodhpur  
**Field:** Computational Mechanics, Structural Dynamics, Scientific Machine Learning  
**Date:** September 2026  
**Status:** Complete Research Project (74/74 Unit Tests Passing)

---

## 1. Candidate Profile & Research Statement

My name is **Raghvendra Singh Gahlot**. I am a 2nd-year undergraduate student in Civil Engineering at MBM University, Jodhpur. I am writing to apply for a research internship / visiting student position working under your mentorship. My primary research interests lie at the intersection of:
- **Scientific Machine Learning (SciML):** Neural operators (Fourier Neural Operators, Graph Neural Networks) applied to structural mechanics and dynamic wave propagation.
- **Ill-Posed Inverse Problems:** Estimating internal structural damage and boundary forces from sparse, noisy dynamic observations.
- **Multiscale Gradient Dynamics:** Understanding why optimization fails when networks are fed physical quantities operating on vastly different scales (e.g., floor accelerations vs. micro-strains).
- **Domain Generalization:** Ensuring learned neural operators can transfer across changing structural geometries, boundary conditions, and building heights without collapsing.

**Availability:** Full-time or hybrid engagement for 6 months (flexible timeline). I have developed an independent, self-contained, and deterministically tested pipeline in PyTorch and OpenSeesPy.

---

## 2. Personal Statement: Why I Chose This Problem & How I Think

### Moving Beyond Synthetic Toy Benchmarks
When I first began reading recent literature in Scientific Machine Learning, I noticed that many papers report impressive accuracy scores (such as $R^2 > 0.99$ or tiny relative $L_2$ errors) on smooth, synthetic benchmarks. However, when these methods are applied to realistic physical systems, they frequently fail. In many cases, the neural models are not actually learning the underlying physics; they are simply interpolating smooth functions across dense, over-instrumented sensor grids.

In real-world civil engineering, we do not have dense grids of thousands of sensors. We have sparse sensor arrays—perhaps one horizontal accelerometer per floor—and noise is inevitable. Furthermore, civil buildings are designed to be nominally symmetric for architectural and structural reasons.

I wanted to tackle an inverse problem that is **genuinely ill-posed from the first principles of structural mechanics**: a problem where standard deep learning is mathematically doomed to fail unless physical symmetry and observation rank are explicitly accounted for. That search led me to seismic structural damage identification in laterally symmetric building frames under sparse instrumentation.

### The Structural Symmetry Dilemma
In a multi-story building frame with lateral symmetry, suppose severe earthquake shaking causes a 30% reduction in stiffness in the ground-floor left column (State A), or symmetrically in the ground-floor right column (State B).

If we only install horizontal accelerometers on each floor slab (the conventional monitoring practice), both damage states produce floor vibrations that are virtually identical. In my OpenSees dynamic simulations under real earthquake records, the relative $L_2$ difference between the floor acceleration histories of State A and State B is **less than 0.14%**. Under an ordinary 2% field noise floor, this difference is completely submerged.

When an unconstrained deep inverse model is trained with standard Mean Squared Error (MSE) loss on this data, it exhibits *mean-seeking collapse*. Because the sensor data is virtually identical for both states, gradient descent forces the network to predict the conditional expectation $\mathbb{E}[d \mid y]$. As a result, the model predicts diffuse, low-amplitude damage across both columns simultaneously ($\hat{d} \approx 0.01\text{--}0.02$). It completely misses the localized structural failure.

### My Approach to Research: Rigor and Scientific Honesty
From the start, I adopted three core principles for this work:
1. **Analyze the mechanics before training networks:** Before writing neural network code, I computed linearized sensitivity Jacobians and performed noise-whitened singular value decomposition (SVD). This proved mathematically that the bilateral difference vector $v_{AB}$ lies directly in the near-null space of horizontal sensing ($|\langle v_E, v_{AB} \rangle| = 0.9982$).
2. **Never hide ill-posedness or negative results:** If a sensor configuration cannot physically distinguish two damage states, I did not tweak hyperparameters or search for random seeds until one run happened to guess correctly. True non-observability must be stated clearly.
3. **Traceability:** Every metric, table entry, and plot in this project is tied to deterministic scripts, frozen seeds, and audited result files in the repository.

---

## 3. Mathematical Formulation of the Dynamic Inverse Problem

The structural frame dynamic equilibrium under horizontal ground acceleration $a_g(t)$ is modeled by the transient hyperbolic equation of motion:

$$M \ddot{u}(t) + C \dot{u}(t) + K(d) u(t) = -M r a_g(t)$$

where $M$ is the lumped mass matrix, $C$ is the Rayleigh damping matrix, and $K(d)$ is the global tangent stiffness matrix parameterized by element damage vector $d \in [0, 0.5]^E$. For each structural member $e$, the effective elastic modulus is:

$$E_e(d_e) = (1 - d_e) E_0$$

Sparse observations are collected through a selection matrix $H_S$ with 2% stationary Gaussian sensor noise:

$$y(t) = H_S [\ddot{u}(t) + r a_g(t), u(t)] + \eta(t), \quad \eta(t) \sim \mathcal{N}(0, \sigma^2 I)$$

To evaluate whether a method can distinguish symmetric damage, we define canonical bilateral damage states:
- **State A:** 30% stiffness reduction in the ground-floor left column ($d_A = [0.30, 0, 0, \dots]^T$).
- **State B:** 30% stiffness reduction in the ground-floor right column ($d_B = [0, 0.30, 0, \dots]^T$).
- **Canonical Difference Unit Vector:** $v_{AB} = \frac{d_A - d_B}{\|d_A - d_B\|_2} = \frac{1}{\sqrt{2}} [1, -1, 0, \dots]^T$ with true separation $\|d_A - d_B\|_2 = 0.4243$.

---

## 4. Observability Analysis & Sensor Configurations

To determine what sensors are physically necessary to break the bilateral symmetry, I formulated the noise-whitened sensitivity Jacobian $J_w$ and directional Fisher sensitivity $\sqrt{I_{AB}}$:

$$J_w = \Sigma_\eta^{-1/2} \frac{\partial y}{\partial d}, \quad \sqrt{I_{AB}} = \|J_w v_{AB}\|_2 = \sqrt{v_{AB}^T J_w^T J_w v_{AB}}$$

I evaluated four sensor suites on a 3-story, 1-bay frame:

| Modality Suite | Sensor Types & Locations | Channels | Fisher Sensitivity $\sqrt{I_{AB}}$ | S0 Ratio | Physical Mechanism |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **S0** | Horizontal floor accelerometers (Floors 1, 2, Roof) | 3 | $5.62$ | $1.0\times$ | Near-null space baseline. Floor shear deformations are laterally symmetric. |
| **S2** | Horizontal accelerations + column axial strains | 9 | $1344.18$ | $239.3\times$ | Column axial deformation captures lateral rocking load transfer. |
| **S1** | Horizontal accelerations + vertical joint accelerations | 9 | $1829.63$ | $325.7\times$ | Vertical joint motions capture differential floor-beam rotation. |
| **S4** | Multimodal: Horizontal + Vertical + Axial Strains | 15 | $2270.54$ | $404.2\times$ | Combined translational, rotational, and internal strain observability. |

---

## 5. Inverse Model Architecture: DualStreamGFNO

Standard MLPs or CNNs assume a fixed grid and cannot easily handle frames with varying story heights or bay widths. To address this, I developed **DualStreamGFNO**, which couples temporal Fourier Neural Operators with structural Graph Neural Networks:

1. **Temporal FNO Stream:** Processes time series across global and joint sensor channels using 1D spectral convolutions with 24 Fourier modes, extracting frequency-domain dynamic features.
2. **Structural Graph Message Passing:** Passes temporal embeddings along structural frame connectivity (nodes = beam-column joints, edges = structural members).
3. **Symmetry Decomposition:** Node representations are explicitly split into symmetric $h_{\text{sym}} = \frac{1}{2}(h + \Pi h)$ and anti-symmetric $h_{\text{anti}} = \frac{1}{2}(h - \Pi h)$ components via the structural permutation matrix $\Pi$.
4. **Hierarchical Output Heads:** Edge representations are fed to two separate heads:
   - *Support Head:* Predicts probability of damage for each member $p_e \in [0, 1]$ (using Sigmoid).
   - *Severity Head:* Predicts damage magnitude $\mu_e \in [0, 0.5]$ (using scaled Sigmoid).
   - *Final Prediction:* $\hat{d}_e = p_e \cdot \mu_e$.
5. **Bilateral Pair Loss:** $L_{\text{total}} = L_{\text{base}} + \lambda_{\text{dir}} L_{\text{dir}} + \lambda_{\text{pair}} L_{\text{pair}}$, where $L_{\text{dir}} = 1 - \cos(\Delta \hat{d}, v_{AB})$ penalizes incorrect orientation and $L_{\text{pair}} = \max(0, m - \langle \Delta \hat{d}, v_{AB} \rangle)$ enforces separation beyond margin $m = 0.15$.

---

## 6. Single-Structure Benchmark Results

I evaluated the trained models on an independent, held-out test set of $N = 30$ bilateral test pairs under unseen earthquake records from the PEER strong motion database. The decision threshold was frozen prior to test evaluation:

| Modality Suite | Attributions | Attribution Accuracy | Exact Binomial $p$-value | 95% Clopper-Pearson CI | Directional Cosine | Predicted Separation $\|\Delta \hat{d}\|_2$ | True Separation Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | 15 / 30 | 50.0% | $p = 1.000$ | $[31.3\%, 68.7\%]$ | $-0.115 \pm 0.122$ | $0.000009$ | $0.002\%$ |
| **S2** | 15 / 30 | 50.0% | $p = 1.000$ | $[31.3\%, 68.7\%]$ | $+0.105 \pm 0.111$ | $0.000328$ | $0.08\%$ *(Learnability Gap)* |
| **S1** | 24 / 30 | 80.0% | $p = 0.0014$ | $[61.4\%, 92.3\%]$ | $+0.727 \pm 0.442$ | $0.1576$ | $37.15\%$ |
| **S4** | 27 / 30 | **90.0%** | **$p = 9.0 \times 10^{-6}$** | **$[73.5\%, 97.9\%]$** | **$+0.942 \pm 0.098$** | **$0.1671$** | **$39.40\%$** |

---

## 7. The Column Strain Paradox: Observability vs. Learnability

The most intriguing finding on the single building was the complete failure of the **S2 sensor suite (column axial strain)**.

Fisher sensitivity showed that adding strain gauges increased physical sensitivity by **239×** over horizontal floor sensing ($\sqrt{I_{AB}} = 1344.18$ vs. $5.62$). In structural engineering textbooks, axial strains are well-known to capture the rocking moments that break lateral symmetry.

Yet when the neural operator was trained, it achieved only **50.0% attribution accuracy (15/30, pure chance)**, exactly matching the coin-flip baseline of the unobservable S0 suite.

### Investigation of the Failure Mechanism
To understand why, I tracked backpropagation gradient norms layer by layer during early training epochs. Floor accelerations have amplitudes of order $\sim 1.0 \text{ m/s}^2$, whereas elastic column axial strains are on the order of $\sim 10^{-5} \text{ m/m}$. Even with standard input normalization, gradients originating from the acceleration channels accounted for **over 98% of the total backpropagation gradient norm**. 

The micro-strain channels were effectively drowned out in the optimizer's updates. The network settled into a symmetric minimum near the mean before strain signals could exert sufficient torque to guide the weights.

This taught me an important lesson in Scientific Machine Learning: *high physical Fisher information is necessary, but it does not guarantee neural learnability under multiscale gradient dynamics.*

---

## 8. Multi-Structure Generalization & Magnitude Collapse

To test whether the learned inversion transfers to new structures, I built an experimental dataset of **530 dynamic OpenSees simulations** across 120 PEER earthquakes, covering 10 structural configurations (stiffness variations, height changes) and an unseen 4-story frame.

When evaluating the model zero-shot on these unseen structures, I observed a consistent phenomenon: **Direction–Magnitude Decoupling**.

- **Directional Alignment Transfers:** The continuous directional cosine between predicted damage difference and the true bilateral vector remained consistently positive ($\cos \approx +0.80\text{ to }+0.85$, reaching $+0.996$ on in-distribution frames and $+0.955$ on the unseen 4-story frame). The network reliably identified *which side* had more damage.
- **Magnitude Collapses:** However, the predicted damage separation collapsed to approximately $1.8 \times 10^{-5}$ (compared to true separation $0.4243$). Because the magnitude difference was so close to zero, discrete attribution accuracy dropped back to 50.0% chance.

I ran a controlled 72-model ablation scaling training budgets from 12 to 100 epochs. Scaling compute improved directional cosine alignment from +0.47 to +0.85, but did not restore finite separation. When an inverse network is trained across buildings of differing stiffness and geometry, gradient updates across the diverse configurations average out, creating an implicit regularization that compresses output magnitudes toward the mean.

---

## 9. Proposed Research Directions in Your Laboratory

Joining your laboratory would allow me to build on these insights and address the key bottlenecks identified in this project:

1. **3D Frame Mechanics and Torsional Symmetry Breaking:**  
   Real buildings are three-dimensional. When a column is damaged in a 3D building, it shifts the center of rigidity relative to the center of mass, inducing dynamic torsion and bi-directional response coupling. I want to formulate 3D graph neural operators that leverage torsional rotational degrees of freedom to break symmetry naturally without requiring dense sensor grids.

2. **Group-Equivariant Structural Operators:**  
   In this work, I used soft loss penalties to encourage bilateral symmetry. A more mathematically rigorous approach is to build exact group-equivariant graph operators (such as $C_2$ or $D_4$ equivariant GNNs) where reflection and rotation symmetries are built directly into the message-passing layers.

3. **Multiscale Gradient Optimization:**  
   To solve the Column Strain Paradox (where accelerations drown out strain gradients), I want to investigate multi-task gradient-balancing techniques (such as PCGrad or GradNorm) that adaptively project and balance gradient components from sensors operating at vastly different physical scales.

4. **Validation on Physical Shake-Table Data:**  
   Transitioning from OpenSees numerical simulations to real-world experimental data from shake-table frame tests (e.g. from PEER or NHERI repositories) to test how neural operators handle real material non-linearities and ambient noise.

---

## 10. Reproducibility & Engineering Architecture

- **Automated Test Suite:** Running `python scripts/gstack.py qa` executes 74 automated unit tests in under 3.5 seconds, verifying finite-element model frequencies, Rayleigh damping, Jacobian dimensions, loss functions, and Clopper-Pearson confidence bounds.
- **Physical Units:** All OpenSees models enforce strict SI units ($m, N, kg, s, Pa$).
- **Core Files:**
  - `src/damage_injection.py` — OpenSees transient dynamic finite element simulation.
  - `src/forensic_observability.py` — Noise-whitened SVD and Fisher sensitivity analysis.
  - `src/phase6_models.py` — DualStreamGFNO graph-Fourier operator architecture.
  - `src/losses.py` — Hierarchical support/severity and bilateral pair losses.
  - `scripts/build_faculty_pdfs.py` — Academic PDF generator.
  - `tests/` — 74 unit tests with regression protection.
