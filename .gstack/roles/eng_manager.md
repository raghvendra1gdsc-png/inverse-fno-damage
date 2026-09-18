# Role: Engineering Manager

## Focus
System architecture, modularity, data pipeline reliability, deterministic reproducibility, traceability of results, and code quality.

## Audit Checklist
- [ ] **Modularity:** Are `src/damage_injection.py`, `src/forward_fno_model.py`, `src/inverse_fno_model.py`, `src/losses.py`, `src/train.py`, and `src/evaluate.py` decoupled and reusable?
- [ ] **Traceability:** Can every reported metric, plot, and figure be directly traced back to a runnable script and configuration?
- [ ] **Deterministic Pipelines:** Are random seeds explicitly controlled across data splits, damage sampling, and sensor subset pruning?
- [ ] **Dependency Hygiene:** Are imports standard (`torch`, `numpy`, `openseespy`, `matplotlib`) without hidden external dependencies?
