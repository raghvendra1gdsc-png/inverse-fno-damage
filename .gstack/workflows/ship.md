# Workflow: Pre-Flight Release & Publication Readiness

## Objective
Execute pre-flight checks before sharing code, tagging releases, or publishing preprints.

## Checklist
1. **Automated QA Suite:**
   - Execute `python scripts/gstack.py qa` and confirm all tests pass.
2. **Deterministic Artifacts:**
   - Confirm all figures in `results/figures/` and notebooks in `notebooks/` are pre-rendered and deterministic.
3. **Traceability:**
   - Confirm all reported numbers in `docs/ill_posedness_discussion.md` match `results/*.json`.
4. **Clean Git State:**
   - Ensure working tree has no unversioned temporary files or untracked scratch scripts.
