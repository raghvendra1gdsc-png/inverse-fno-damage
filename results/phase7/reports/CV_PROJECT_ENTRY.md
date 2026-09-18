# Inverse-FNO-Damage: Curriculum Vitae Project Entries

**Project Title:** Neural-Operator Inversion for Seismic Structural Damage Identification  
**Repository Identifier:** `Inverse-FNO-Damage`  
**Focus Areas:** Scientific Machine Learning (SciML), Computational Mechanics, Inverse Problems, Structural Health Monitoring  

---

### Option A: One-Line CV Version
- **Inverse-FNO-Damage:** Formulated an observability-aware Graph Fourier Neural Operator pipeline with bilateral pair supervision in PyTorch and OpenSeesPy, elevating symmetric seismic damage attribution from 50.0% to 90.0% ($p = 9.0 \times 10^{-6}$) on single structures while empirically characterizing cross-structure Direction–Magnitude Decoupling.

---

### Option B: Two-Bullet Technical Version
- **Physical Observability & Symmetry-Aware Neural Inversion:** Quantified symmetry-induced near-null spaces in seismic structural inversion via noise-whitened directional Fisher information, demonstrating that vertical acceleration and axial strain amplify directional sensitivity by $>300\times$ over conventional horizontal floor sensing ($S0 = 5.6$ vs. $S4 = 2270.5$).
- **Single-Structure Attribution & Cross-Structure Limits:** Built the DualStreamGFNO architecture with matched bilateral pair supervision, achieving $90.0\%$ attribution accuracy ($27/30$, $p = 9.0 \times 10^{-6}$) on held-out earthquake evaluations; demonstrated that under cross-structure distribution shift, continuous directional sensitivity transfers zero-shot ($\cos \approx +0.80$ to $+0.85$) while discrete attribution collapses to chance ($50.0\%$).

---

### Option C: Four-Bullet Research Version (Recommended for Research / Academic Applications)
- **Empirical Characterization of Bilateral Non-Identifiability:** Demonstrated through validated OpenSeesPy dynamic simulations that localized $30\%$ stiffness reduction in symmetric columns yields $<0.14\%$ discrepancy in horizontal floor accelerations, placing bilateral failure modes in the numerical near-null space of standard observation operators.
- **Noise-Whitened Fisher Observability & Learnability Gap:** Formulated a noise-whitened directional Fisher sensitivity metric ($\sqrt{I_{AB}}$) mapping a $>300\times$ sensitivity gain for multimodal sensing ($S4 = 2270.5$) over horizontal arrays ($S0 = 5.6$); demonstrated that physical observability is necessary but insufficient for neural learnability by showing that column strain ($S2 = 1344.2$) fails ($50.0\%$ attribution) due to floor-acceleration gradient dominance ($>98\%$).
- **Symmetry-Aware DualStreamGFNO with Pair Supervision:** Developed a dual-stream architecture decomposing node representations into symmetric ($H^+$) and antisymmetric ($H^-$) latent subspaces, coupled with contrastive margin ($L_{\text{pair}}$) and directional alignment ($L_{\text{dir}}$) losses to achieve **$90.0\%$ bilateral attribution accuracy** ($27/30$, $p = 9.0 \times 10^{-6}$, 95% CI: $[73.5\%, 97.9\%]$) on single-structure held-out earthquake records.
- **Cross-Structure Transfer & Direction–Magnitude Decoupling:** Evaluated transferability across an audited 530-simulation benchmark spanning 10 structural configurations and an unseen 4-story frame ($C_{\text{4story}}$); discovered an empirical Direction–Magnitude Decoupling phenomenon where continuous directional alignment transfers zero-shot ($\cos \approx +0.80$ to $+0.85$, reaching $+0.996$ on individual seeds) while finite damage separation collapses ($\sim 10^{-5}$ vs. true $0.4243$), with budget scaling ($12 \to 100$ epochs) unable to recover discrete separation.
