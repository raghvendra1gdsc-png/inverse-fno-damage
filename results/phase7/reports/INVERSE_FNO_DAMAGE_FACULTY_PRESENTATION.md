# Observability, Learnability, and Transferability in Neural-Operator Inversion for Seismic Structural Damage Identification

**Subtitle:** Inverse-FNO-Damage  
**Target Audience:** Faculty Reviewers in Scientific Machine Learning, Computational Mechanics, Structural Dynamics, Inverse Problems, and SHM  
**Format:** 8-Slide Senior Research Presentation with Scripted Speaker Notes  
**Status:** Complete & Authoritative (Frozen Experimental Program; 74/74 Unit Tests Passing)  

---

## Presentation Overview & Visual Design Framework

- **Aesthetic Philosophy:** Clean computational mechanics and scientific ML visual language. No decorative gradients, glowing AI boxes, or stock marketing imagery. Technical schematic layouts, publication-grade ASCII matrices, and explicit mathematical definitions.
- **Narrative Arc:** Exactly one core scientific question per slide, following a strict 3-to-5-minute progression:
  1. *The Inverse Question* (Problem Definition)
  2. *The Physical Obstacle* (Symmetry-Induced Near-Null Space)
  3. *Quantifying Observability* (Noise-Whitened Fisher Metric $\sqrt{I_{AB}}$)
  4. *The Counterexample* (Observability $\ne$ Learnability via $S2$)
  5. *Targeted Symmetry-Aware Learning* (DualStreamGFNO & Pair Supervision on Single Structure)
  6. *Cross-Structure Transfer Failure* (Empirical Direction–Magnitude Decoupling)
  7. *Optimization Budget Ablation* ($12 \to 100$ Epochs Scaling Boundary)
  8. *Research Implication & Open Horizon* (Three Bottlenecks Framework)
- **Authoritative Figure Provenance:** Every figure reference links directly to frozen, high-resolution artifact plots generated in the repository (`reports/figures/observability/`, `reports/figures/phase6_2/`, `reports/figures/phase7/`, and `results/experiments/phase7_training_budget/figures/`).

---

---

# SLIDE 1 — THE QUESTION

## Can a Neural Operator Invert Seismic Response into Structural Damage?

```
                     TRANSIENT EXCITATION & FORWARD DYNAMICS
                     ┌─────────────────────────────────────┐
                     │   Seismic Ground Motion: a_g(t)     │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │         Structural Dynamics         │
                     │  M u''(t) + C u'(t) + K(d) u(t) =   │
                     │             -M r a_g(t)             │
                     └──────────────────┬──────────────────┘
                                        │
                                        ▼
                     ┌─────────────────────────────────────┐
                     │   Sparse, Noisy Sensors y(t)        │
                     │   Floor Accelerations / Strains     │
                     └──────────────────┬──────────────────┘
                                        │
                     ═══════════════════╪══════════════════════════════════
                                        │  INVERSE NEURAL OPERATOR G_phi
                                        ▼
                     ┌─────────────────────────────────────┐
                     │      Predicted Damage Field d       │
                     │  Element Stiffness Loss d_e ∈ [0,1] │
                     └─────────────────────────────────────┘
```

### Problem Setup & Inverse Challenge
- **The Inverse Problem:** Reconstruct the spatial distribution and severity of localized structural member damage ($d \in [0, 0.5]^E$, where $E_e = (1 - d_e)E_0$) in civil building frames from sparse, noisy surface vibration recordings.
- **Physical Observations:** Sparse boundary measurements $\mathbf{y}(t) = \mathcal{H}_S[\ddot{\mathbf{u}}(t) + \mathbf{r} a_g(t), \, \mathbf{u}(t)] + \boldsymbol{\eta}(t)$ under $2\%$ stationary Gaussian sensor noise.
- **Target Parameter:** Localized stiffness degradation vector $d \in \mathbb{R}^E$ across frame beam-column elements.
- **The Physical Ambiguity:** Civil structures possess nominal geometric and lateral symmetry. When multiple distinct structural failure modes generate virtually identical boundary sensor traces, naive mean-squared-error optimization collapses to the conditional expectation $\mathbb{E}[d \mid \mathbf{y}]$, outputting diffuse, mean-seeking predictions of negligible magnitude ($\hat{d} < 0.02$).

---

### Speaker Notes (Slide 1 — 25 Seconds)
> *"We begin with a classical question in applied mechanics: can a deep neural operator diagnose internal structural damage from sparse boundary earthquake recordings? In structural health monitoring, economic constraints restrict sensors to a handful of floor slabs. When we train standard inverse neural operators with regression losses, they exhibit an immediate, insidious failure: mean-seeking collapse. Because civil structures are laterally symmetric, different failure modes produce nearly indistinguishable floor vibrations, and standard networks simply predict the symmetric arithmetic mean—completely obscuring the true failure."*

---
---

# SLIDE 2 — THE PHYSICAL OBSTACLE

## Before Learning: Is the Damage Physically Observable?

```
         STATE A (Damaged Left Column)               STATE B (Damaged Right Column)
         ┌───────────────────────────┐               ┌───────────────────────────┐
 Floor 3 │ (Node 7) ═══════ (Node 8) │       Floor 3 │ (Node 7) ═══════ (Node 8) │
         │   ║                   ║   │               │   ║                   ║   │
 Floor 2 │ (Node 5) ═══════ (Node 6) │       Floor 2 │ (Node 5) ═══════ (Node 6) │
         │   ║                   ║   │               │   ║                   ║   │
 Floor 1 │ (Node 3) ═══════ (Node 4) │       Floor 1 │ (Node 3) ═══════ (Node 4) │
         │  [X]                 ║   │               │   ║                  [X]  │
  Ground ┴──(1)─────────────────(2)──┴        Ground ┴──(1)─────────────────(2)──┴
         Left Column Damage: d_1=30%                 Right Column Damage: d_2=30%
```

```
                     DISTINCT PHYSICAL DAMAGE STATES
         Ground Truth Damage Separation: ||d_A - d_B||_2 = 0.4243
                                    │
                                    │ Forward Simulation (OpenSeesPy)
                                    ▼
                 NEAR-IDENTICAL HORIZONTAL SENSOR RESPONSE
               Relative L_2 Difference: ||y_A - y_B||_2 / ||y_A||_2 < 0.14%
               Sensor Noise Floor: σ_η = 2.0%  ==>  SNR < 0.14
```

### Structural Mechanics Insight
- **The Experiment:** Validated finite-element dynamic simulations in OpenSeesPy on a 3-story, 1-bay frame (`SOURCE_A`). State A features $30\%$ stiffness reduction in the left ground-story column; State B features $30\%$ reduction in the symmetric right ground-story column.
- **Shear Dominance:** Under lateral seismic shaking, horizontal floor accelerations are governed by aggregate story shear ($V_{\text{story}} = \sum K_{\text{col}} \Delta u$). In a symmetric frame, swapping column stiffnesses produces an identical first-order aggregate story stiffness:
  $$K_{\text{story}} = K_{\text{left}} + K_{\text{right}} = 0.70 K_0 + 1.00 K_0 = 1.70 K_0$$
- **Near-Null Space Alignment:** Linearized sensitivity Jacobian SVD demonstrates that the bilateral difference unit vector $v_{AB} = (d_A - d_B) / \|d_A - d_B\|_2$ aligns almost perfectly with the smallest singular vector of horizontal sensing:
  $$|\langle v_E, v_{AB} \rangle| = 0.9982 > 0.99$$
- **Primary Source Figures:**  
  - Frame Schematic: [`reports/figures/observability/fig1_damage_states_A_vs_B.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig1_damage_states_A_vs_B.png)  
  - Floor Acceleration Overlay: [`reports/figures/observability/fig2_response_overlay_A_vs_B.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig2_response_overlay_A_vs_B.png)  
  - Submerged Difference Signal: [`reports/figures/observability/fig3_difference_signals_vs_noise.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig3_difference_signals_vs_noise.png)

---

### Speaker Notes (Slide 2 — 30 Seconds)
> *"Before attempting any machine learning, we must ask: is the damage physically observable? Here is our baseline experiment. We take a 3-story symmetric building and damage the left ground column by 30%. Then we take the exact same building and damage the right column by 30%. In physical parameter space, their separation is 0.4243. But when we simulate their dynamic earthquake response in OpenSeesPy, the horizontal floor accelerations differ by less than 0.14%. That difference is completely submerged under a realistic 2% noise floor. Horizontal sensing measures aggregate story shear, rendering the bilateral failure direction an empirical near-null space."*

---
---

# SLIDE 3 — QUANTIFYING OBSERVABILITY

## Fisher Information Reveals the Hidden Asymmetric Direction

```
                    NOISE-WHITENED DIRECTIONAL SENSITIVITY
                    
                    J_w = Σ_η^(-1/2) J    (Noise-Whitened Sensitivity Jacobian)
                    
                    F_w = J_w^T J_w       (Whitened Fisher Information Matrix)
                    
                    √(I_AB) = || J_w v_AB ||_2 = √( v_AB^T F_w v_AB )
```

### Modality Sensitivity Hierarchy ($2\%$ Calibrated Sensor Noise)

```
 S0: Horizontal Floor Accel (3 chan)
 █ 5.617  [Near-Null Baseline]
 
 S2: Horiz Accel + Column Axial Strain (9 chan)
 ████████████████████████ 1344.18  (239x gain over S0)
 
 S1: Horiz Accel + Vertical Joint Accel (9 chan)
 ████████████████████████████████ 1829.63  (326x gain over S0)
 
 S4: Multimodal Union: Horiz + Vert + Strain (15 chan)
 ████████████████████████████████████████ 2270.54  (405x gain over S0)
```

### Mathematical Definitions & Observations
- **Jacobian Formulation:** $\mathbf{J} = \partial \mathbf{y} / \partial d \in \mathbb{R}^{(C \cdot T) \times E}$, mapping member damage perturbations to differential time-series outputs across all active sensor channels.
- **Noise Whitening:** $\mathbf{\Sigma}_\eta = \text{diag}(\sigma_{\eta, 1}^2, \dots, \sigma_{\eta, C}^2) \otimes \mathbf{I}_T$, rendering disparate physical units ($\text{m/s}^2$ vs. $\text{m/m}$) strictly dimensionless.
- **Canonical Direction Vector:** $v_{AB} = \frac{1}{\sqrt{2}} [1, -1, 0, \dots, 0]^T \in \mathbb{R}^E$, defining the 1D bilateral ambiguity subspace.
- **Physical Asymmetry:** Vertical accelerometers capture dynamic joint rocking; axial strain gauges capture unequal dynamic column load transfer. Together in $S4$, they amplify directional sensitivity along $v_{AB}$ by **$404.2\times$ over $S0$**.
- **Crucial Qualification:**  
  $$\mathbf{\text{Physical observability under a linearized Gaussian model is not yet neural learnability.}}$$
- **Primary Source Figures:**  
  - SVD Singular Spectra: [`reports/figures/observability/fig6_singular_spectra_comparison.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig6_singular_spectra_comparison.png)  
  - Directional Fisher Comparison: [`reports/figures/observability/fig13_directional_fisher_bilateral.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/observability/fig13_directional_fisher_bilateral.png)

---

### Speaker Notes (Slide 3 — 30 Seconds)
> *"To quantify which sensors can physically break this symmetry, we formulate a noise-whitened directional Fisher information metric, square-root of I_AB. This maps the sensitivity along the unit bilateral difference vector v_AB while normalizing across different measurement units. Standard horizontal sensing, S0, yields an effective sensitivity of only 5.6. But adding vertical joint accelerations or column axial strain amplifies this directional sensitivity by over 240- to 400-fold. This quantifies that asymmetric physical signatures exist in the dynamic response. But can a neural operator actually learn them?"*

---
---

# SLIDE 4 — THE COUNTEREXAMPLE

## More Information Does Not Automatically Imply Neural Learnability

### The Single-Structure Bilateral Benchmark ($N = 30$ Independent Held-Out Evaluations)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ Sensor Suite │ Active Channels │ Fisher Sensitivity √(I_AB) │ Learned Bilateral Attribution │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│     S0       │   3 (Horiz)     │            5.617           │    50.0%  (15/30, p = 1.000)   │
│     S2       │   9 (Horiz+Str) │         1344.18            │    50.0%  (15/30, p = 1.000)   │
│     S1       │   9 (Horiz+Vrt) │         1829.63            │    80.0%  (24/30, p = 0.0014)  │
│     S4       │  15 (Multimodal)│         2270.54            │    90.0%  (27/30, p = 9.0e-6)  │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

```
                        THE S2 OBSERVABILITY-LEARNABILITY GAP
                        
               High Physical Fisher Information: √(I_AB) = 1344.18
                                        │
                                        ▼
             Backpropagation Gradient Norms: Floor Accel > 98%, Strains < 2%
                                        │
                                        ▼
                 Predicted Damage Separation: ||Δd_hat||_2 = 0.000328
                 Learned Bilateral Attribution: 50.0% (Pure Chance)
```

### Mechanistic Analysis
- **The Empirical Evidence:** Configuration $S2$ possesses $239\times$ higher Fisher sensitivity than $S0$, yet standard optimization yields exactly chance attribution ($15/30 = 50.0\%$).
- **Gradient Dominance:** Backpropagation gradient norm tracing confirms that high-amplitude horizontal floor accelerations ($\sim 1.0\text{ m/s}^2$) dominate the network's loss gradients ($>98\%$). The localized micro-strain signals ($\sim 10^{-5}\text{ m/m}$) are completely drowned out.
- **Local Minimum:** The optimizer rapidly satisfies the global dynamic loss by driving asymmetric element prediction heads to zero, collapsing into the symmetric mean basin before strain gradients can exert directional torque.
- **Scientific Takeaway:**  
  $$\mathbf{\text{Physical Observability is Necessary but Insufficient for Neural Learnability.}}$$
- **Primary Source Figures:**  
  - Observability vs. Accuracy Scatter: [`reports/figures/phase6_2/fig4_sensor_comparison.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig4_sensor_comparison.png)  
  - Bilateral Confusion Matrices: [`reports/figures/phase6_2/fig1_bilateral_confusion.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig1_bilateral_confusion.png)

---

### Speaker Notes (Slide 4 — 35 Seconds)
> *"This brings us to one of our central findings, demonstrated by the counterexample on this slide. Conventional wisdom suggests that if a physical system has high Fisher information, a neural network will simply learn it. We tested this directly. Look at configuration S2, combining horizontal acceleration with column axial strain. Its physical Fisher sensitivity is over 1300—yet its learned bilateral attribution accuracy is exactly 50.0%, pure chance. Why? Because floor accelerations have amplitudes of 1 meter per second squared, while micro-strains are on the order of 10 to the minus 5. Accelerations carry over 98% of the backprop gradient norm, completely blinding the network to the strain signals. Physical observability does not guarantee neural learnability."*

---
---

# SLIDE 5 — WHAT CHANGED THE LEARNING PROBLEM

## Symmetry-Aware Architecture & Matched Bilateral Pair Supervision

```
                  DUALSTREAMGFNO ARCHITECTURE & LOSS FORMULATION
                  
  Global Horiz Accel ──► Temporal FNO ──────► Global Vector h_global ──┐
                                                                       │ Concatenate
  Local Joint Sensor ──► Temporal FNO ──► Graph Conv ──► Subspace Split│ & MLP Decoder
                                          H = [H^+ || H^-] ────────────┘
                                          H^+ = 0.5(H + P H)  (Symmetric)
                                          H^- = 0.5(H - P H)  (Antisymmetric)
                                                                       │
                                                                       ▼
                                                          Hierarchical Heads
                                                          d_hat_e = p_e · μ_e
                                                          p_e: Support ∈ [0,1]
                                                          μ_e: Severity ∈ [0,0.5]
```

```
                     MATCHED BILATERAL PAIR SUPERVISION LOSS
                     
  L_total = L_base + λ_dir L_dir + λ_pair L_pair
  
  • Directional Alignment Loss:  L_dir  = 1 - cos(Δd_hat, v_AB)
  • Contrastive Margin Loss:     L_pair = max( 0,  m - <Δd_hat, v_AB> )   [m = 0.15]
```

### Single-Structure Success ($S4$, $N = 30$ Independent Evaluations)
- **Attribution Accuracy:** **$90.0\%$** ($27/30$, exact two-sided binomial $p = 9.0 \times 10^{-6}$, 95% Clopper-Pearson CI: $[73.5\%, 97.9\%]$).
- **Directional Orientation:** Mean learned cosine $\cos(\Delta \hat{d}, v_{AB}) = \mathbf{+0.9420 \pm 0.098}$.
- **Recovered Damage Separation:** Mean predicted separation $\|\Delta \hat{d}\|_2 = \mathbf{0.1671}$ ($39.40\%$ of true physical separation $0.4243$).
- **Critical Distinction:**  
  $$\mathbf{\text{Discrete attribution improved substantially (90%), but damage magnitude was only partially recovered (39.4%).}}$$
- **Primary Source Figures:**  
  - Architecture Schematic: [`reports/figures/phase6/fig1_phase6_architecture_diagram.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6/fig1_phase6_architecture_diagram.png)  
  - Recovered Separation Plot: [`reports/figures/phase6_2/fig3_predicted_separation.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase6_2/fig3_predicted_separation.png)

---

### Speaker Notes (Slide 5 — 30 Seconds)
> *"To force the neural operator to process subtle asymmetric dynamics, we redesigned both architecture and supervision. We introduced DualStreamGFNO, which explicitly splits latent graph node representations into symmetric H-plus and antisymmetric H-minus subspaces using structural reflection permutations. Crucially, we formulated matched bilateral pair supervision, training the network on paired simulations driven by identical earthquakes with a directional alignment loss and a contrastive margin loss. On a single structure, multimodal sensing S4 now achieves 90% bilateral attribution accuracy, 27 out of 30 held-out cases with p equals 9 times 10 to the minus 6. But notice: predicted separation is 0.1671—roughly 39% of the true physical ground truth. Attribution succeeded, but magnitude was only partially recovered."*

---
---

# SLIDE 6 — THE SURPRISING FAILURE

## Cross-Structure Transfer Exposes a Second Bottleneck

### The Multi-Structure Clean Benchmark (Audited 530 OpenSees Dynamic Simulations)
- **Structural Configurations:** 10 diverse building frames (varying bay width $L$, story height $h$, stiffness $E$, density $\rho$) and an unseen 4-story frame topology ($C_{\text{4story}}$).
- **Data Splitting:** 120 PEER earthquake records strictly partitioned into globally disjoint sets (70 train, 20 validation, 30 test). 
- **Pristine Feature Contract:** Zero target stiffness leakage ($E_{\text{norm}} = 1.0$ verified by automated tests).

```
                     CROSS-STRUCTURE BENCHMARK RESULTS (P2 S4)
                     
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ Test Level │ Target Structure Frame │ Directional Cosine │ Discrete Attribution │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │  Level 1   │ SOURCE_A (In-Dist)     │  +0.850 ± 0.205    │   50.0%  (Chance)    │
 │  Level 2A  │ B_int (Param Interp)   │  +0.841 ± 0.216    │   50.0%  (Chance)    │
 │  Level 2B  │ B_ext (Param Extrap)   │  +0.813 ± 0.204    │   50.56% (Chance)    │
 │  Level 3   │ C_4story (Topol OOD)   │  +0.798 ± 0.199    │   50.0%  (Chance)    │
 └─────────────────────────────────────────────────────────────────────────────┘
```

```
                          THE EMPIRICAL DIVERGENCE
                          
            DIRECTIONAL ORIENTATION            FINITE DAMAGE SEPARATION
            ✓ Transfers Zero-Shot              ✗ Collapses to Near-Zero
            cos(Δd_hat, v_AB) ≈ +0.80 to +0.85  ||Δd_hat||_2 ≈ 1.83 × 10^(-5)
            (Individual seeds reach +0.996)    (vs True Physical 0.4243)
```

### Formalization: Direction–Magnitude Decoupling
- **Definition (Empirical Observation, NOT a Theorem):** In neural inverse operators for symmetric dynamical systems under structural parameter dispersion, gradient-based optimization successfully aligns the orientation of predicted difference vectors with the observable bilateral direction ($\cos \to +0.85$), while simultaneously failing to scale the magnitude of predicted separation ($\|\Delta \hat{d}\|_2 \sim 10^{-5}$), pinning discrete attribution to chance ($50.0\%$).
- **Observed Mechanism:** Scale-invariant directional losses rotate the difference vector correctly, but cross-structure frequency variations act as an implicit regularizer, compressing shared encoder outputs toward the population mean.
- **Primary Source Figures:**  
  - Cross-Structure Evaluation: [`reports/figures/phase7/fig1_cross_structure_benchmark.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase7/fig1_cross_structure_benchmark.png)  
  - 4-Story Inversion: [`reports/figures/phase7/fig3_c4story_topological_inversion.png`](file:///Users/rahul/inverse-fno-damage/reports/figures/phase7/fig3_c4story_topological_inversion.png)

---

### Speaker Notes (Slide 6 — 35 Seconds)
> *"Having achieved 90% attribution on a single frame, we asked: does this inverse mapping transfer across buildings? We generated an audited benchmark of 530 OpenSees dynamic simulations across 10 structural configurations and an unseen 4-story frame topology, tested on 30 held-out earthquakes. The result was startling. Directional sensitivity transfers zero-shot remarkably well: directional cosines remain consistently positive, averaging plus 0.80 to plus 0.85, with individual seeds reaching plus 0.996. Yet discrete attribution collapsed back to 50.0% chance, and predicted separation dropped to 10 to the minus 5. We call this the empirical Direction–Magnitude Decoupling phenomenon: the network learns which way to turn in damage space, but collapses the scale of its predictions under cross-structure distribution shift."*

---
---

# SLIDE 7 — TRAINING BUDGET IS NOT THE MAIN EXPLANATION

## More Optimization Does Not Restore Finite Separation

### Controlled 72-Evaluation Training Budget Sweep (P2 S4 Trajectory across Seeds 42, 101, 2024)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ Training Budget │ Level 1 Accuracy │ Level 1 Cosine │ Level 3 Cosine │ Separation ||Δd_hat||_2 │
├────────────────────────────────────────────────────────────────────────────────────────┤
│    12 Epochs    │  50.0% ± 0.0%    │ +0.467 ± 0.037 │ +0.489 ± 0.012 │   2.68 × 10^(-6)       │
│    25 Epochs    │  50.0% ± 0.0%    │ +0.852 ± 0.115 │ +0.777 ± 0.158 │   7.68 × 10^(-6)       │
│    50 Epochs    │  50.0% ± 0.0%    │ +0.897 ± 0.038 │ +0.837 ± 0.090 │   1.23 × 10^(-5)       │
│   100 Epochs    │  50.0% ± 0.0%    │ +0.850 ± 0.205 │ +0.798 ± 0.199 │   1.83 × 10^(-5)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

```
                          BUDGET SCALING TRAJECTORY
                          
 Directional Cosine Alignment               Predicted Damage Separation ||Δd_hat||_2
 1.0 ┤          ╭───────────                0.40 ┤ [True Physical Separation = 0.4243]
 0.8 ┤     ╭────╯                           0.30 ┤
 0.6 ┤     │                                0.20 ┤
 0.4 ┼─────╯                                0.10 ┤
 0.0 ┴─────┬─────┬─────┬────                0.00 ┴─────┬─────┬─────┬──── [Flat at ~10^-5]
    12    25    50    100                      12    25    50    100
           Training Epochs                                Training Epochs
 [Cosine scales from +0.47 to +0.90]           [Separation remains 4 orders of mag low]
```

### Empirical Interpretations & Scientific Boundaries
- **Undertraining Hypothesis Refuted:** Scaling optimization budget by $8.3\times$ ($12 \to 100$ epochs) across continuous training trajectories produces **$0.0\%$ improvement in discrete attribution**.
- **Directional Torque Confirmed:** Directional alignment sharpens rapidly ($+0.467 \to +0.897$ mean, with individual seeds reaching $+0.996$ on Level 1 and $+0.955$ on Level 3). Gradient descent successfully steers the output difference vector along $v_{AB}$.
- **Rigorous Claim Boundary:**  
  $$\mathbf{\text{Under the tested budgets and protocols, optimization budget was not sufficient to restore finite separation.}}$$
  *(We do not claim that infinite optimization or unconstrained architectures can never solve it; we establish an empirical boundary for standard operator regression).*
- **Primary Source Figures:**  
  - Accuracy vs. Budget: [`results/experiments/phase7_training_budget/figures/plot1_accuracy_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot1_accuracy_vs_budget.png)  
  - Cosine vs. Budget: [`results/experiments/phase7_training_budget/figures/plot2_cosine_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot2_cosine_vs_budget.png)  
  - Separation vs. Budget: [`results/experiments/phase7_training_budget/figures/plot3_separation_vs_budget.png`](file:///Users/rahul/inverse-fno-damage/results/experiments/phase7_training_budget/figures/plot3_separation_vs_budget.png)

---

### Speaker Notes (Slide 7 — 30 Seconds)
> *"A natural question is: were the multi-structure models simply undertrained? To test this, we executed a controlled 72-evaluation training budget sweep scaling from 12 to 100 epochs across three random seeds. Look at the two trajectories. Directional cosine alignment rises dramatically from plus 0.47 to roughly plus 0.90, with individual seeds exceeding plus 0.99. The optimizer is clearly receiving and following directional torque. Yet discrete attribution remains dead flat at 50.0%, and predicted separation never exceeds 10 to the minus 5. Within the tested settings, optimization budget alone cannot overcome the separation collapse."*

---
---

# SLIDE 8 — WHAT THIS RESEARCH CHANGES

## Three Distinct Bottlenecks in Scientific Machine Learning

```
                            THE THREE-LEVEL FRAMEWORK
                            
                                 1. OBSERVABILITY
                           Can physical sensing expose
                              the parameter subspace?
                                        │
                                        │  S0 is Near-Null (√(I_AB) = 5.6)
                                        │  S4 Exposes Direction (√(I_AB) = 2270.5)
                                        ▼
                                 2. LEARNABILITY
                           Can neural gradients extract
                            subtle asymmetric signals?
                                        │
                                        │  S2 Fails (50% attribution, gradient dominance)
                                        │  S4 Succeeds (90% attribution via pair loss)
                                        ▼
                                3. TRANSFERABILITY
                           Does finite separation hold
                            across structural shift?
                                        │
                                        │  Direction Transfers Zero-Shot (cos ≈ +0.85)
                                        │  Magnitude Collapses (||Δd|| ~ 10^-5, 50% attr)
                                        ▼
```

### Established Evidence Summary
1. **Physical Observability:** Noise-whitened Fisher sensitivity demonstrates that horizontal arrays are blind to symmetric column damage, while multimodal sensing provides $>400\times$ directional information.
2. **Neural Learnability:** Physical observability is necessary but insufficient. Column strain $S2$ fails completely ($50\%$) despite high Fisher sensitivity due to floor acceleration gradient dominance. Pairwise supervision resolves this on a single structure ($S4 = 90.0\%$, $p = 9.0 \times 10^{-6}$).
3. **Cross-Structure Transferability:** Discovered the empirical Direction–Magnitude Decoupling phenomenon: continuous orientation transfers zero-shot, but finite damage separation collapses under structural distribution shift.

---

### Concluding Horizon & Final Scientific Question
> *"What representation or training principle can preserve damage magnitude under structural distribution shift while retaining the physically observable direction?"*

---

### Speaker Notes (Slide 8 — 30 Seconds)
> *"To conclude: solving ill-posed seismic inverse problems with neural operators requires disentangling three distinct bottlenecks. First, physical observability: if the sensor physics is blind, no algorithm can succeed. Second, neural learnability: even when Fisher information is high, gradient dominance can prevent learning, as demonstrated by configuration S2. And third, transferability: cross-structure distribution shift decouples directional orientation from finite output magnitude. The central open question this work leaves for computational mechanics and scientific ML is: what representation or training principle can preserve damage magnitude under structural distribution shift while retaining the physically observable direction?"*

---
---

## Slide-to-Artifact Traceability Index

| Slide # | Primary Scientific Focus | Key Quantitative Values | Governing Authoritative File | Verification Script |
| :---: | :--- | :--- | :--- | :--- |
| **1** | Inverse Problem Formulation | $d \in [0, 0.5]^E$, $2\%$ sensor noise | `results/ill_posedness_metrics.json` | `python scripts/phase1_ill_posedness.py` |
| **2** | Bilateral Symmetry Near-Null Space | diff $< 0.14\%$, $\|\Delta d\| = 0.4243$, $\|\langle v_E, v_{AB} \rangle\| = 0.9982$ | `results/observability/symmetry_baseline.json` | `python scripts/phase5_observability.py` |
| **3** | Noise-Whitened Fisher Metric | $S0=5.617$, $S2=1344.18$, $S1=1829.63$, $S4=2270.54$ | `results/observability/phase5_5_forensic_audit.json`| `python scripts/phase5_5_forensic_audit.py` |
| **4** | Observability $\ne$ Learnability ($S2$) | $S2: \sqrt{I_{AB}}=1344.2$, Acc $= 50.0\%$, Sep $= 0.000328$ | `results/phase6_2/bilateral_results.json` | `python scripts/phase6_2_bilateral_learning.py` |
| **5** | Single-Structure Inversion ($S4$) | $27/30 = 90.0\%$, $p = 9.0\text{e-}6$, $\cos = +0.942$, Sep $= 0.1671$ | `results/phase6_2/bilateral_results.json` | `python scripts/phase6_2_bilateral_learning.py` |
| **6** | Direction–Magnitude Decoupling | 530 runs, 120 GMs, $\cos \approx +0.80\text{--}+0.85$, Acc $= 50\%$, Sep $\sim 10^{-5}$ | `results/experiments/FINAL_PHASE7_CLEAN_AUDIT.json` | `python scripts/phase7_clean_rerun.py` |
| **7** | Training Budget Ablation | 72 models, $12 \to 100$ ep, $\cos: 0.47 \to 0.90$, Acc $= 50\%$ | `results/experiments/phase7_training_budget/TRAINING_BUDGET_ABLATION.json`| `python scripts/phase7_training_budget_experiment.py` |
| **8** | Three Bottlenecks Synthesis | Observability $\to$ Learnability $\to$ Transferability | `results/phase7/reports/FACULTY_RESEARCH_BRIEF.md` | Automated Quality Gate (`gstack.py qa`) |
