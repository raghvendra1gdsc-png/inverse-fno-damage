# Inverse-FNO-Damage: Three-Minute Faculty Pitch & Discussion Guide

**Project:** `Inverse-FNO-Damage`  
**Document:** Spoken Pitch Script & Defense Q&A Guide  
**Target Audience:** Faculty Reviewers in Scientific ML, Applied Math, Computational Mechanics, and Structural Dynamics  
**Date:** September 2026  

---

## Part 1: Spoken Research Pitch (3 Minutes)

*Estimated reading time: 2.5–3 minutes (~410 words)*

"I started with a simple question: if two different damage states produce almost the same sensor response, can a neural operator actually identify which structure is damaged?

In structural mechanics, when a symmetric building suffers 30% damage in an exterior left column versus the right column, their horizontal floor accelerations differ by less than 0.14%. Under standard floor accelerometer arrays and 2% noise, this bilateral failure direction lies directly inside the numerical near-null space of the observation operator. When we train standard inverse neural operators with mean-squared-error losses on this problem, they exhibit mean-seeking collapse—predicting diffuse, symmetric damage across both columns simultaneously.

To tackle this, we did not just apply a larger neural network. First, we asked what the sensor physics actually allows. Using noise-whitened Jacobian singular value decomposition and directional Fisher information, we mapped how different sensing modalities amplify sensitivity along the bilateral ambiguity vector. We found that vertical accelerations and axial strains amplify directional Fisher sensitivity by more than 240- to 400-fold over standard horizontal sensing.

However, physical observability alone does not guarantee neural learnability. When we added column axial strain sensors—configuration S2—the Fisher sensitivity was high, over 1300, yet the network remained trapped at exactly 50% chance-level attribution. High-amplitude floor accelerations generated over 98% of early backpropagation gradients, completely drowning out the localized micro-strain signals.

To overcome this gradient dominance, we developed a Symmetry-Aware Dual-Stream Graph Fourier Neural Operator with matched bilateral pair supervision. Combining directional alignment with contrastive margin loss under multimodal sensing—configuration S4—we achieved 90% bilateral attribution accuracy, 27 out of 30 independent held-out earthquake evaluations, with a p-value of 9 times 10 to the minus 6.

Finally, we tested cross-structure transfer across 530 validated OpenSees simulations spanning parametric stiffness variations and an unseen 4-story frame topology. Here, we uncovered an intriguing empirical phenomenon: continuous directional sensitivity transfers zero-shot, maintaining positive cosines between +0.80 and +0.85, but finite damage separation collapses to near-zero, and discrete attribution returns to 50% chance. A 72-model ablation scaling training budget from 12 to 100 epochs refined directional orientation up to +0.996 on individual seeds, but did not recover finite separation.

The central takeaway of this work is that physical observability, neural learnability, and cross-structure transferability are three distinct failure modes. Each requires different mathematical tools to diagnose, and solving one does not automatically resolve the others."

---

## Part 2: Likely Faculty Questions & Defensible Answers

### Q1. Why is this an inverse-problem research contribution rather than just another FNO application?
**Answer:** Because the failure modes we investigate are grounded in structural mechanics and inverse problem theory, not arbitrary deep learning benchmarking. Bilateral symmetry induces a near-null space in the forward wave operator under horizontal boundary measurements. Our primary contribution is diagnosing and disentangling the mathematical boundaries between physical observation rank, gradient optimization dynamics, and cross-structure distribution shift, rather than chasing incremental accuracy on an unconstrained dataset.

### Q2. If Fisher sensitivity is high for S2, why does the network still fail?
**Answer:** This demonstrates the observability–learnability gap. Column axial strains possess strong static and dynamic asymmetry under rocking, yielding high directional Fisher sensitivity ($\sqrt{I_{AB}} = 1344.18$). However, strains are on the order of $10^{-5}\text{ m/m}$, whereas floor accelerations are on the order of $1.0\text{ m/s}^2$. In an unweighted dual-stream loss, backpropagation gradient norms are dominated by acceleration channels ($>98\%$). The optimizer rapidly satisfies the global dynamic loss by setting asymmetric heads to zero, falling into a local minimum near the symmetric mean before strain gradients can exert torque.

### Q3. Why should I believe the 90% single-structure result?
**Answer:** Because the result is statistically verified under rigorous controls: evaluated on $N = 30$ independent held-out earthquake evaluations driven by records strictly disjoint from training, yielding an exact two-sided binomial $p$-value of $9.0 \times 10^{-6}$ against chance ($p_0 = 0.50$) with a 95% Clopper-Pearson exact confidence interval of $[73.5\%, 97.9\%]$. Furthermore, the identical architecture and loss function applied to unobservable horizontal sensing ($S0$) yields exactly $50.0\%$ ($15/30$, $p = 1.000$), confirming that the pairwise loss cannot invent information that is physically absent from sensor traces.

### Q4. Is 90% actually damage reconstruction accuracy?
**Answer:** No. 90% ($27/30$) is binary bilateral state attribution accuracy—correctly identifying whether Column 1 is more damaged than Column 2 ($\hat{d}_{\text{Col1}} > \hat{d}_{\text{Col2}}$). In continuous parameter space, the model recovers a predicted separation of $\|\Delta \hat{d}\|_2 = 0.1671$, which is $39.40\%$ of the true physical separation ($0.4243$), closely matching the contrastive margin target $m = 0.15$. We explicitly maintain the distinction between discrete bilateral attribution, directional orientation ($\cos = +0.9420$), and finite parameter separation.

### Q5. What happens when the building changes?
**Answer:** When tested across an audited 530-simulation benchmark spanning parametric variations in geometry, stiffness, density, and an unseen 4-story frame topology ($C_{\text{4story}}$), continuous directional sensitivity transfers zero-shot ($\cos \approx +0.80$ to $+0.85$, with individual seeds reaching $+0.996$ on in-distribution frames and $+0.955$ on the 4-story frame). However, finite damage separation collapses to $\sim 10^{-5}$, and discrete attribution returns to chance level ($50.0\%$).

### Q6. Why does the model preserve directional information but lose magnitude under distribution shift?
**Answer:** We formalize this as the empirical Direction–Magnitude Decoupling phenomenon. The directional alignment loss $L_{\text{dir}} = 1 - \cos(\Delta \hat{d}, v_{AB})$ provides scale-invariant gradient signals that rotate $\Delta \hat{d}$ toward $v_{AB}$. However, under multi-structure parameter variations, the shared encoder minimizes global regression loss across diverse structural frequencies by compressing predictions toward a shared symmetric mean, penalizing macroscopic separation during gradient descent. A 72-model ablation scaling training from 12 to 100 epochs refines directional orientation but does not recover finite separation.

### Q7. Are the 30 evaluations really independent?
**Answer:** Yes. The 30 evaluation cases are driven by 30 unique PEER earthquake acceleration records (`RSN0091` to `RSN0120`) that are strictly disjoint from all training and validation records. Each earthquake induces a unique dynamic excitation history and frequency content. Furthermore, multiple random seeds ($42, 101, 2024$) are strictly treated as optimization replications along continuous training trajectories and are never pooled to claim an inflated sample size ($N \ne 90$).

### Q8. What would you do next experimentally?
**Answer:** We would test the framework on physical shake-table test specimens where nominal symmetry is perturbed by construction tolerances, non-structural components, and foundation compliance. Real-world instrumentation introduces nonstationary ambient noise, thermal baseline drift, and material hysteretic degradation. Our noise-whitened Fisher framework would provide the mandatory pre-test observability audit before deploying neural operators to real-world sensory feeds.

### Q9. What is conceptually new here compared to existing literature?
**Answer:** Existing structural neural operator papers typically report in-distribution MSE on continuous damage fields or global frequency tracking, which glosses over ill-posed symmetry. We make three distinct conceptual contributions:
1. Demonstrating that physical observability does not imply neural learnability via the S2 counterexample.
2. Formulating matched bilateral pair supervision to overcome acceleration gradient dominance on single structures.
3. Characterizing the empirical Direction–Magnitude Decoupling phenomenon, showing that continuous directional representation transfers zero-shot even when discrete attribution collapses.

### Q10. What is the single biggest limitation of this work?
**Answer:** All dynamic analyses are conducted on 2D planar frames with linear-elastic stiffness reduction under idealized 2% stationary Gaussian noise. In real 3D civil structures, damage induces bi-directional torsional coupling, which provides additional asymmetric signals but also increases the parameter space dimension. Validating these phenomena on physical shake-table specimens remains the necessary next step.
