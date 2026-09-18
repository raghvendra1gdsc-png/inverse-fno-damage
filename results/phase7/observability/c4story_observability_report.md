# Phase 7.1 Gate B: C_4story Observability Analysis Report

**Gate Decision:** **PASS (S1, S4 Authorized; S0 Classified as F1 Non-Observable)**

### Executive Summary
Noise-whitened Jacobian singular spectrum analysis was performed for the 4-story frame (`C_4story`, 10 nodes, 12 structural elements) under a 2% channel-wise RMS noise model. The physical observability structure demonstrates an exact correspondence to the findings of Phase 5 and 5.5:
- **S0 (Horizontal Accelerometers):** Condition number $\kappa = 13,496$, $\sigma_{\min} = 0.0691 \ll 1.0$, directional Fisher sensitivity $\sqrt{I_{AB}} = 1.9141$. The bilateral damage direction lies in the noise-dominated subspace. Classified as **F1 (Physical Non-Observability)**; inverse model failure under S0 cannot be attributed to ML.
- **S1 (Horizontal + Vertical Accelerometers):** Condition number drops to $85.91$, $\sigma_{\min} = 35.48 \gg 1.0$, and directional Fisher sensitivity surges to $\sqrt{I_{AB}} = 1297.88$. Vertical sensing physically breaks symmetry.
- **S4 (Multimodal Sensing):** Condition number drops to $16.08$, $\sigma_{\min} = 241.72$, and directional Fisher sensitivity reaches $\sqrt{I_{AB}} = 1639.08$. Observability is well-conditioned and highly informative.

### C_4story Noise-Whitened Observability Metrics Table

| Modality | Channels | $\sigma_{\max}$ | $\sigma_{\min}$ | Condition Number $\kappa$ | Directional Fisher $\sqrt{I_{AB}}$ | Status | Inverse Authorized? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S0** | 4 | 932.75 | 0.0691 | 13496.28 | 1.91 | **F1_PHYSICALLY_NON_OBSERVABLE** | NO (F1 Physical Barrier) ⚠️ |
| **S1** | 12 | 3048.57 | 35.4837 | 85.91 | 1297.88 | **PASS** | YES ✅ |
| **S4** | 24 | 3887.55 | 241.7164 | 16.08 | 1639.08 | **PASS** | YES ✅ |

### Gate B Verdict & Protocol Enforcement
- **Gate Decision:** **PASS**
- **Authorized Inverse Modalities for Level 3:** **S1 and S4**
- **Enforced Scientific Rule:** S0 results on C_4story must be interpreted strictly as an F1 physical baseline.
