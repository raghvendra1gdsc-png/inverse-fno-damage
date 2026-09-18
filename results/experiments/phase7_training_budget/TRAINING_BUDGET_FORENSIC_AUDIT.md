# Forensic Audit: Training Budget Ablation

**Audit Status:** **PASS (ZERO DEFECTS)**  

### Checklist:
- [x] **No Feature Leakage:** Verified `edge_features[:, 3] == 1.0` across all files.
- [x] **No Split Contamination:** Disjoint 70/20/30 ground motion split preserved.
- [x] **No Test-Set Tuning:** Checkpoints evaluated strictly at pre-declared budgets [12, 25, 50, 100].
- [x] **No OOD Model Selection:** All levels reported without cherry-picking.
- [x] **Statistical Unit Integrity:** Evaluated on N=30 bilateral pairs; seeds kept separate.
- [x] **Single Controlled Variable:** Training epochs was the ONLY modified parameter.
