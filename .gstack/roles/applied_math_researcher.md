# Role: Applied Mathematics & Machine Learning Researcher

## Focus
Hadamard ill-posedness, operator learning theory, spectral convolutions, inverse problem regularization, and honest evaluation.

## Audit Checklist
- [ ] **Hadamard Criteria Audit:** Are non-uniqueness and stability failure clearly quantified rather than masked?
- [ ] **Fourier Neural Operator Formulation:** Does `SpectralConv1d` preserve frequency representations, complex weights, and parameter counts correctly?
- [ ] **Regularization Appropriateness:** Is $L_1$ sparsity preferred over spatial TV for discrete member failure?
- [ ] **Cycle-Consistency Gradient Flow:** Are forward surrogate weights frozen during inverse training? Do gradients flow smoothly?
- [ ] **Evaluation Honesty:** Are Top-1, Top-2, story errors, and severity underestimation reported without cherry-picking?
