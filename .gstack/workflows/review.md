# Workflow: Multi-Perspective Code & Physics Review

## Objective
Execute a rigorous multi-role audit of the entire Inverse FNO pipeline across structural engineering, applied mathematics, software engineering, and academic presentation.

## Execution Steps
1. **Engineering Modularity Audit:**
   - Verify `src/` modules are cleanly decoupled.
   - Ensure imports are self-contained and reproducible.
2. **Physical Consistency & Unit Audit:**
   - Audit OpenSeesPy model definitions against strict SI units.
   - Verify static Guyan condensation match on baseline frame.
3. **Mathematical Operator & Loss Audit:**
   - Verify complex einsum in `SpectralConv1d`.
   - Verify non-negativity constraint ($d_e \ge 0$) in `InverseFNO`.
   - Verify cycle-consistency autograd detachment of forward model weights.
4. **Honesty & Traceability Check:**
   - Confirm all reported metrics in `results/` match test set evaluation outputs without quiet tuning or cherry-picking.
