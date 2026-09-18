# AGENTS.md — Inverse FNO Damage Identification

## Mission
Build a machine learning pipeline to solve an ill-posed inverse physics problem: identifying localized structural damage (stiffness reduction) in a building frame based purely on sparse, noisy acceleration data from a limited number of sensors, validated against OpenSeesPy ground truth.

This is a research-grade project intended for outreach to CS/Applied Math faculty. Scientific honesty, mathematical rigor, and physical constraints matter far more than superficial accuracy.

## Hard Rules (Do Not Violate)

1. **Follow the Phase Order Strictly:**
   - **Phase 1:** Data generation & empirical proof of ill-posedness (identical sensor responses for distinct damage states).
   - **Phase 2:** Forward and Inverse Neural Operators (FNO).
   - **Phase 3:** Custom loss functions (regularization + forward cycle-consistency) and training.
   - **Phase 4:** Sensor sparsity sweep (graceful degradation analysis).
   Do not jump ahead to ML architectures before ground truth FE simulation and ill-posedness are validated.

2. **Full Traceability:**
   Every number, frequency, loss value, or metric reported in `results/`, `docs/`, or conversation must be directly traceable to a runnable script, deterministic configuration, and execution log. Never report estimated or hand-waved numbers as measured facts.

3. **Honesty About Ill-Posedness:**
   Do not hide non-uniqueness. If two damage states produce sensor outputs with <1% difference, explicitly highlight and quantify this ambiguity. The entire premise of this project is demonstrating how physics constraints (sparsity prior + cycle consistency) resolve ill-posed non-uniqueness.

4. **No Quiet Tuning:**
   If the inverse model struggles to localize damage or confuses two modes, document the failure transparently. Log all ablations and failure modes rather than silently tuning hyper-parameters until an isolated cherry-picked example succeeds.

5. **Physical Unit Consistency:**
   All structural FE models in OpenSeesPy must use strict, documented SI units:
   - Length: meters ($m$)
   - Force: Newtons ($N$)
   - Mass: kilograms ($kg$)
   - Time: seconds ($s$)
   - Stress / Elastic Modulus: Pascals ($Pa = N/m^2$)
   - Density: $kg/m^3$

6. **Current Status & Gate Checks:**
   Always obtain user confirmation on structural baselines before proceeding to downstream damage injection or neural operator training.

## Current Phase
Phase 7: Cross-Structure Generalization & Out-of-Distribution Validation (Complete — 108-model experimental matrix across Protocols P1/P2 and modalities S0/S1/S4 executed across seeds 42, 101, 2024; production dataset of 530 OpenSeesPy simulations across 120 PEER GMs verified via 14-point audit; forward operator validation Gate A passed with NRMSE < 0.05; C_4story observability Gate B passed for S1/S4 and verified unobservable for S0; zero-shot topological damage inversion on 4-story frame demonstrated 83.3%-86.7% bilateral attribution under multimodal sensing S4 with cos = +0.88; parametric interpolation Level 2A achieved 90.0% accuracy; parametric extrapolation Level 2B degraded gracefully to 83.3%; S0 physical non-observability control strictly maintained 50.0% chance level; 70 unit tests passing; gstack review and ship pre-flight audits green; PHASE 7 GATE: PASS).

---

## gstack Integration & Quality Gates
This repository uses the **gstack** virtual review and QA framework:
- Always run `python scripts/gstack.py qa` after code changes to ensure all 70 unit tests pass.
- Use `python scripts/gstack.py review` to trigger multi-persona verification (Structural, Applied Math, Eng Manager, Faculty Reviewer).
- Use `python scripts/gstack.py gaps` to identify remaining mathematical and structural gaps and next milestones.
- Use `python scripts/gstack.py office-hours` for 3-minute faculty pitch framing.
- Use `python scripts/gstack.py ship` to run complete pre-flight release and publication audit.

