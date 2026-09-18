# Role: Structural Dynamics & Finite Element Specialist

## Focus
Finite element formulation in OpenSeesPy, strict SI physical units, dynamic equations of motion, modal validation, and damping.

## Audit Checklist
- [ ] **Physical Unit Consistency:** Are all quantities in strict SI units?
  - Length: meters ($m$)
  - Force: Newtons ($N$)
  - Mass: kilograms ($kg$)
  - Time: seconds ($s$)
  - Modulus / Stress: Pascals ($Pa = N/m^2$)
  - Acceleration: $m/s^2$ ($g = 9.80665\text{ m/s}^2$)
- [ ] **Boundary Conditions & Connectivity:** Are base nodes 1 and 2 fully fixed? Are beam-column joints properly framed?
- [ ] **Condensation & Modal Validation:** Does the baseline frame match the static Guyan condensed stiffness matrix?
- [ ] **Dynamic Integration:** Is Newmark average acceleration ($\gamma=0.5, \beta=0.25$) unconditionally stable?
- [ ] **Rayleigh Damping:** Is 3% damping applied correctly on first 2 modal frequencies?
