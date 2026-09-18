# Workflow: Faculty Office Hours & Strategic Research Framing

## Objective
Provide an instant, compelling, and intellectually honest 3-minute pitch for CS and Applied Mathematics faculty, positioning the project as a foundational exploration of ill-posed inverse physics.

## Core Questions & Answers

### 1. What is the fundamental research problem?
Solving an ill-posed inverse problem: identifying localized structural damage (stiffness reduction in a building frame) purely from sparse, noisy acceleration data recorded during earthquakes.

### 2. Why is this problem mathematically interesting?
Floor slabs act as rigid diaphragms, meaning symmetrical columns in the same story produce $> 99.85\%$ identical horizontal floor accelerations ($0.14\%$ relative error, well below sensor noise). Standard deep learning completely fails ($87.5\%$ error rate) because MSE collapses toward an unphysical symmetric average.

### 3. What does physics-informed regularization solve?
- **$L_1$ Sparsity Prior:** Completely eliminates false alarms and ghost noise ($0.0000$).
- **Forward Cycle-Consistency Loss:** Couples the inverse operator with a pre-trained Forward FNO, enforcing dynamic physical validity and improving story-level localization error to $0.75$ stories.

### 4. What is the collaboration opportunity for faculty?
Joint research on resolving in-bay symmetry via multi-modal sensing (vertical rocking / strain), unbiased severity estimation via reweighted $L_1$ / Bayesian neural operators, and non-linear hysteretic damage identification under severe seismic shaking.
