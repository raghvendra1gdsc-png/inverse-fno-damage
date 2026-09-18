# Workflow: Gap Analysis & Technical Roadmap

## Objective
Systematically identify mathematical, physical, and architectural gaps in the current Inverse FNO pipeline, and formulate concrete next steps to advance the project toward peer-reviewed publication and faculty collaboration.

## Gap Analysis Matrix

### Gap 1: In-Bay Bilateral Symmetry Ambiguity
* **Description:** Symmetrical columns (e.g. left vs. right column in a story) produce identical horizontal floor shear and accelerations ($>99.85\%$ identical). Horizontal floor sensors alone cannot distinguish which column has yielded.
* **Impact:** Limits single-element Top-1 localization accuracy to $\approx 15\%-20\%$.
* **Next Steps to Move Forward:**
  1. Incorporate vertical / rocking sensor channels ($\ddot{u}_y(t)$) into the forward and inverse operators. Under lateral shaking, the frame rocks, putting one column in tension ($+a_y$) and the other in compression ($-a_y$), breaking the horizontal symmetry.
  2. Test strain gauge or inclinometer channel fusion.

### Gap 2: Severity Underestimation (The Minimum-Norm Trap)
* **Description:** Standard regression combined with $L_1$ shrinkage collapses predicted damage amplitudes ($d_{\text{true}} \approx 0.35 \to d_{\text{pred}} \approx 0.08$).
* **Impact:** While damage location is identified, the severity is under-estimated by $\approx 70\%$.
* **Next Steps to Move Forward:**
  1. Implement **Iterative Reweighted $L_1$ Regularization (Candes et al.)** or non-convex SCAD/MCP penalties that do not penalize large coefficients.
  2. Decouple localization and severity via a **two-stage architecture**: Stage 1 identifies the damaged member index via classification, and Stage 2 estimates severity via conditional regression.

### Gap 3: Linear-Elastic Dynamic Scope
* **Description:** Current forward surrogate is linear-elastic with localized stiffness reduction. Real post-yield seismic response exhibits hysteretic loops and pinching (e.g., Bouc-Wen model).
* **Impact:** Model is applicable to pre-yield / micro-crack damage, but not severe post-yield structural degradation.
* **Next Steps to Move Forward:**
  1. Integrate non-linear hysteretic elements (Bouc-Wen or Steel01) into OpenSeesPy.
  2. Add hysteretic energy dissipation $E_h(t)$ and peak ductility $\mu$ as output channels in Forward FNO (drawing from SeismoFNO).

### Gap 4: Single-Seed Sensor Sweep Uncertainty
* **Description:** The current Phase 4 sweep is a single-seed result (`seed=42`) evaluated on 24 damaged test records.
* **Impact:** Discretization noise ($4.17\%$ per sample) creates slight non-monotonicity at low sensor counts.
* **Next Steps to Move Forward:**
  1. Run a 10-seed Monte Carlo sweep over sensor combinations to compute mean and confidence intervals $\mu \pm \sigma$ for each sensor count $K$.
