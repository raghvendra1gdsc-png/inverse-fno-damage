# Workflow: Automated Quality Assurance & Verification

## Objective
Run the complete automated testing suite, enforce quality gates, and verify physical and numerical integrity across all modules.

## Execution
```bash
python scripts/gstack.py qa
```

## Quality Gates Enforced
1. **Physical Unit Compliance:** Strict SI units ($m, N, kg, s, Pa$).
2. **Modal Accuracy:** Baseline OpenSees frequencies match Guyan condensation within $< 0.01\%$.
3. **Operator Gradient Flow:** Differentiable forward pass and cycle-consistency backprop through frozen surrogate.
4. **Test Pass Rate:** 100% of automated unit tests pass.
