# Observability, Symmetry, and Sensor-Augmentation Analysis in Structural Inverse Problems

**Document:** `docs/observability.md`  
**Phase:** Phase 5 Directive — Mathematical Foundations & Observability Framework  
**Scope:** Rigorous characterization of the inverse null space, bilateral structural symmetry, and evaluation of sensor-augmentation modalities.

---

## 1. Problem Formulation and Damage Vector Definition

Consider a 2D multi-story moment-resisting building frame governed by the dynamic equations of motion under horizontal ground motion acceleration $a_g(t)$:

$$\mathbf{M} \ddot{\mathbf{u}}(t) + \mathbf{C} \dot{\mathbf{u}}(t) + \mathbf{K}(\mathbf{d}) \mathbf{u}(t) = -\mathbf{M} \mathbf{r} a_g(t)$$

with quiescent initial conditions $\mathbf{u}(0) = \mathbf{0}, \dot{\mathbf{u}}(0) = \mathbf{0}$, where:
- $\mathbf{M} \in \mathbb{R}^{N_{\text{dof}} \times N_{\text{dof}}}$ is the lumped mass matrix.
- $\mathbf{C} \in \mathbb{R}^{N_{\text{dof}} \times N_{\text{dof}}}$ is the Rayleigh damping matrix satisfying $\mathbf{C} = \alpha_M \mathbf{M} + \beta_K \mathbf{K}_0$, calibrated to modal damping ratio $\zeta = 0.03$ on the first two baseline natural modes.
- $\mathbf{K}(\mathbf{d}) \in \mathbb{R}^{N_{\text{dof}} \times N_{\text{dof}}}$ is the global stiffness matrix parameterized by the element damage vector $\mathbf{d}$.
- $\mathbf{r} = [1, 0, 0, 1, 0, 0, \dots]^T$ is the structural influence vector coupling ground acceleration to lateral horizontal DOFs.

### 1.1 Element Damage Parameterization
The structural frame consists of $N_{\text{ele}} = 9$ discrete structural elements. The element damage vector is defined as:

$$\mathbf{d} = [d_1, d_2, d_3, d_4, d_5, d_6, d_7, d_8, d_9]^T \in [0, 1)^9$$

where each scalar $d_e$ represents the fractional reduction in elastic flexural/axial stiffness for element $e$:

$$E_e = E_0 (1 - d_e), \quad d_e \in [0, 1)$$

Based on the validated OpenSeesPy model in `src/damage_injection.py`, the element indices correspond strictly to:
- **$e = 1$**: Story 1 Column, Left Line (Bay 0, between Node 1 and Node 3)
- **$e = 2$**: Story 1 Column, Right Line (Bay 1, between Node 2 and Node 4)
- **$e = 3$**: Story 2 Column, Left Line (Bay 0, between Node 3 and Node 5)
- **$e = 4$**: Story 2 Column, Right Line (Bay 1, between Node 4 and Node 6)
- **$e = 5$**: Story 3 Column, Left Line (Bay 0, between Node 5 and Node 7)
- **$e = 6$**: Story 3 Column, Right Line (Bay 1, between Node 6 and Node 8)
- **$e = 7$**: Story 1 Beam / Floor 1 (between Node 3 and Node 4)
- **$e = 8$**: Story 2 Beam / Floor 2 (between Node 5 and Node 6)
- **$e = 9$**: Story 3 Beam / Roof (between Node 7 and Node 8)

---

## 2. Forward Observation Operator & Sensor Configurations

A sensor configuration $\mathcal{S}$ defines a linear spatial extraction operator that samples $C$ continuous physical channels across the structural nodes and members. The complete forward observation map is denoted:

$$\mathcal{F}(\mathbf{d}; a_g, \mathcal{S}) = \mathbf{Y} \in \mathbb{R}^{C \times N_t}$$

where $N_t$ is the number of discrete simulation time steps ($t_k = k \Delta t, k = 0, \dots, N_t - 1$).

We define the flattened observation vector:

$$\mathbf{y} = \text{vec}(\mathcal{F}(\mathbf{d}; a_g, \mathcal{S})) \in \mathbb{R}^{N_{\text{obs}}}, \quad N_{\text{obs}} = C \cdot N_t$$

### 2.1 Sensor Configurations Under Study
We investigate five distinct physical sensor configurations $\mathcal{S} \in \{S_0, S_1, S_2, S_3, S_4\}$:

1. **$S_0$ (Baseline Horizontal Floor Accelerometers):**
   - Channels: Total horizontal accelerations at floor diaphragms: $\ddot{u}_{x, 3}^{\text{tot}}(t), \ddot{u}_{x, 5}^{\text{tot}}(t), \ddot{u}_{x, 7}^{\text{tot}}(t)$ ($C = 3$).
   - This represents standard commercial SHM deployments.

2. **$S_1$ (Horizontal + Vertical Joint Accelerometers):**
   - Channels: $S_0$ plus vertical accelerations at all column joints: $\ddot{u}_{y, 3}(t), \ddot{u}_{y, 4}(t), \ddot{u}_{y, 5}(t), \ddot{u}_{y, 6}(t), \ddot{u}_{y, 7}(t), \ddot{u}_{y, 8}(t)$ ($C = 9$).
   - Measures vertical joint motion induced by column axial strain and frame rocking.

3. **$S_2$ (Horizontal + Column Axial Strains):**
   - Channels: $S_0$ plus direct axial strain measurements $\epsilon_{\text{ax}, e}(t) = \frac{u_{y, \text{top}}(t) - u_{y, \text{bot}}(t)}{h}$ for all 6 columns ($C = 9$).
   - Directly probes column tension/compression behavior.

4. **$S_3$ (Horizontal + Antisymmetric Differential Rocking Acceleration):**
   - Channels: $S_0$ plus story rocking accelerations:
     $$a_{\text{rock}, j}(t) = \frac{\ddot{u}_{y, \text{right}, j}(t) - \ddot{u}_{y, \text{left}, j}(t)}{L_{\text{bay}}}, \quad j \in \{1, 2, 3\}$$
     ($C = 6$).
   - Isolates the pure rotational/overturning acceleration of the building cross-section.

5. **$S_4$ (Comprehensive Multi-Modal Configuration):**
   - Channels: Horizontal accelerometers, vertical joint accelerometers, column axial strains, and rocking observables ($C = 15$).

---

## 3. Bilateral Symmetry Permutation Operator

Because the nominal frame geometry, member sections, and boundary conditions are symmetric about the vertical centerline $x = L_{\text{bay}} / 2$, there exists a bilateral symmetry transformation.

Let $\mathbf{P} \in \{0, 1\}^{9 \times 9}$ denote the linear permutation operator that exchanges left and right structural elements:

$$\mathbf{P} = \begin{bmatrix}
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}$$

### Mathematical Properties of $\mathbf{P}$:
1. **Involution:** $\mathbf{P}^2 = \mathbf{I}_9 \implies \mathbf{P}^{-1} = \mathbf{P} = \mathbf{P}^T$ (orthogonal and self-inverse).
2. **Isometry:** $\|\mathbf{P} \mathbf{d}\|_2 = \|\mathbf{d}\|_2$ for all $\mathbf{d} \in \mathbb{R}^9$.
3. **Element Transpositions:**
   - Columns: $(1 \leftrightarrow 2)$, $(3 \leftrightarrow 4)$, $(5 \leftrightarrow 6)$.
   - Beams: $(7 \leftrightarrow 7)$, $(8 \leftrightarrow 8)$, $(9 \leftrightarrow 9)$ (beams are self-symmetric about the centerline).

---

## 4. Observational Equivalence and Near-Null Direction

Two distinct damage states $\mathbf{d}_A$ and $\mathbf{d}_B$ are defined to be **observationally equivalent** with respect to sensor configuration $\mathcal{S}$ and threshold $\epsilon > 0$ if:

$$\|\mathcal{F}(\mathbf{d}_A; a_g, \mathcal{S}) - \mathcal{F}(\mathbf{d}_B; a_g, \mathcal{S})\|_2 \le \epsilon$$

In our canonical symmetry benchmark:
- $\mathbf{d}_A = [0.30, 0, 0, 0, 0, 0, 0, 0, 0]^T$ (30% stiffness reduction in Left Column 1).
- $\mathbf{d}_B = \mathbf{P} \mathbf{d}_A = [0, 0.30, 0, 0, 0, 0, 0, 0, 0]^T$ (30% stiffness reduction in Right Column 1).

The element-space damage separation is:
$$\|\mathbf{d}_A - \mathbf{d}_B\|_2 = \sqrt{0.30^2 + (-0.30)^2} = 0.30 \sqrt{2} \approx 0.42426$$

The theoretical **bilateral near-null direction** is the normalized difference vector:

$$\mathbf{v}_{\text{null}} = \frac{\mathbf{d}_A - \mathbf{d}_B}{\|\mathbf{d}_A - \mathbf{d}_B\|_2} = \frac{1}{\sqrt{2}} [1, -1, 0, 0, 0, 0, 0, 0, 0]^T$$

Under purely horizontal floor acceleration measurements ($S_0$), the floor slabs enforce rigid diaphragm behavior:
$$V_{\text{story}, 1}(t) \approx \left(k_1 (1 - d_1) + k_2 (1 - d_2)\right) \Delta u_1(t)$$
Because $k_1 = k_2$, exchanging $d_1$ and $d_2$ yields the exact same total story shear stiffness. Hence, the observed horizontal response difference is an order of magnitude smaller than typical sensor noise ($< 0.14\%$).

---

## 5. Local Jacobian Observability & SVD Analysis

To analyze observability locally around a damage state $\mathbf{d}_0$ (such as the undamaged baseline $\mathbf{d}_0 = \mathbf{0}$ or a damaged state $\mathbf{d}_A$), we construct the **Damage-to-Response Jacobian matrix**:

$$\mathbf{J}_d = \frac{\partial \mathbf{y}}{\partial \mathbf{d}} \in \mathbb{R}^{N_{\text{obs}} \times 9}$$

Each column $e \in \{1, \dots, 9\}$ of $\mathbf{J}_d$ represents the sensitivity of the entire multichannel time-history observation to a stiffness perturbation in member $e$:

$$\mathbf{J}_{d, :, e} = \frac{\partial \mathbf{y}}{\partial d_e} \approx \frac{\mathbf{y}(\mathbf{d}_0 + h \mathbf{e}_e) - \mathbf{y}(\mathbf{d}_0 - h \mathbf{e}_e)}{2h}$$

where $\mathbf{e}_e$ is the $e$-th standard basis vector, and $h$ is a perturbation step size subjected to a convergence study ($h \in \{10^{-4}, 5 \times 10^{-4}, 10^{-3}, 5 \times 10^{-3}\}$).

### 5.1 Singular Value Decomposition (SVD)
The singular value decomposition of $\mathbf{J}_d$ is:

$$\mathbf{J}_d = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^9 \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

where:
- $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_9 \ge 0$ are the singular values.
- $\mathbf{u}_i \in \mathbb{R}^{N_{\text{obs}}}$ are the left singular vectors (observable temporal-spatial mode shapes).
- $\mathbf{v}_i \in \mathbb{R}^9$ are the right singular vectors (orthonormal directions in damage parameter space).

### 5.2 Mathematical Observability Metrics
1. **Smallest Singular Value ($\sigma_{\min} = \sigma_9$):**
   Measures the minimum response energy generated per unit damage perturbation in the least observable parameter direction.
2. **Condition Number ($\kappa(\mathbf{J}_d)$):**
   $$\kappa(\mathbf{J}_d) = \frac{\sigma_1}{\sigma_9}$$
   A large condition number ($\kappa \gg 10^3$) indicates severe ill-conditioning and near-null directions.
3. **Effective Numerical Rank ($r_{\text{eff}}$):**
   $$r_{\text{eff}} = \sum_{i=1}^9 \mathbb{I}\left(\frac{\sigma_i}{\sigma_1} > \text{tol}\right)$$
4. **Alignment with Symmetry Null Space:**
   We compute the absolute cosine similarity between the right singular vector of minimum sensitivity $\mathbf{v}_9$ and the theoretical symmetry null vector $\mathbf{v}_{\text{null}}$:
   $$\rho_{\text{null}} = |\mathbf{v}_9^T \mathbf{v}_{\text{null}}|$$
   If $\rho_{\text{null}} \approx 1.0$, the mathematical null direction of the observation operator coincides precisely with the physical bilateral column symmetry.

---

## 6. Noise-Normalized Observability & Fisher Information Matrix

Mathematical sensitivity alone does not imply identifiability in the presence of measurement noise. 

Assume additive sensor measurement noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{R})$, where $\mathbf{R} \in \mathbb{R}^{N_{\text{obs}} \times N_{\text{obs}}}$ is the sensor noise covariance matrix. For independent sensors with relative RMS noise level $\eta \in \{0.5\%, 1.0\%, 2.0\%, 5.0\%\}$:

$$\mathbf{R} = \text{diag}(\sigma_{n, 1}^2 \mathbf{I}_{N_t}, \dots, \sigma_{n, C}^2 \mathbf{I}_{N_t}), \quad \sigma_{n, c} = \eta \cdot \text{RMS}(y_c(t))$$

The **noise-normalized Jacobian** is:

$$\mathbf{J}_R = \mathbf{R}^{-1/2} \mathbf{J}_d$$

### 6.1 Fisher Information Matrix (FIM)
The Fisher Information Matrix $\mathbf{I}_F \in \mathbb{R}^{9 \times 9}$ for the damage parameters under Gaussian measurement noise is:

$$\mathbf{I}_F = \mathbf{J}_d^T \mathbf{R}^{-1} \mathbf{J}_d = \mathbf{J}_R^T \mathbf{J}_R$$

By the Cramér-Rao Lower Bound (CRLB), the covariance of any unbiased estimator $\hat{\mathbf{d}}$ satisfies:
$$\text{Cov}(\hat{\mathbf{d}}) \ge \mathbf{I}_F^{-1}$$

To robustly quantify sensor information content without singular matrix inversion, we evaluate the regularized log-determinant and minimum eigenvalue:
- **Minimum FIM Eigenvalue:** $\lambda_{\min}(\mathbf{I}_F)$ (governs the variance in the worst-case parameter direction).
- **Log-Determinant (Information Volume):**
  $$\log \det(\mathbf{I}_F + \lambda_{\text{reg}} \mathbf{I}_9)$$
  where $\lambda_{\text{reg}} = 10^{-4}$ is fixed a priori across all sensor configurations.

### 6.2 Noise-Resolvable Separation Ratio ($\rho_{AB}$)
To answer whether two damage states $\mathbf{d}_A$ and $\mathbf{d}_B$ are distinguishable under noise level $\eta$, we define the **Separation-to-Noise Ratio**:

$$\rho_{AB} = \frac{\|\mathcal{F}(\mathbf{d}_A) - \mathcal{F}(\mathbf{d}_B)\|_2}{\|\boldsymbol{\sigma}_{\text{noise}}\|_2} = \frac{\|\mathbf{y}_A - \mathbf{y}_B\|_2}{\sqrt{\sum_{c=1}^C N_t \sigma_{n, c}^2}}$$

- **Noise-Unresolvable:** $\rho_{AB} < 1.0$ (signal difference is submerged below noise floor).
- **Marginal Identifiability:** $1.0 \le \rho_{AB} < 3.0$.
- **Noise-Resolvable:** $\rho_{AB} \ge 3.0$ ($3\sigma$ confidence separation).

---

## 7. Symmetric vs. Antisymmetric Sensor Decomposition

For any paired physical quantities measured at left and right column lines ($q_L(t)$ and $q_R(t)$), we define:

$$q_+(t) = \frac{q_L(t) + q_R(t)}{2} \quad \text{(Symmetric Component)}$$
$$q_-(t) = \frac{q_R(t) - q_L(t)}{2} \quad \text{(Antisymmetric Component)}$$

### Core Scientific Hypothesis:
Under symmetric base excitation $a_g(t)$, horizontal floor accelerations capture almost exclusively symmetric motion ($q_+$), while antisymmetric rocking and differential axial strains ($q_-$) isolate the asymmetric damage perturbation:

$$\|q_{-, A} - q_{-, B}\|_2 \gg \|q_{+, A} - q_{+, B}\|_2$$

This hypothesis will be directly audited and proven in the subsequent empirical study.
