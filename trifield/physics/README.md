# trifield.physics: Sentaurus' exact discrete equations, and refinement of surrogate guesses

This package evaluates the drift-diffusion equations of the production deck exactly as Sentaurus
Device (V-2024.03) discretises them: same box data, same statistics, same mobility. It does this for
any field triple (psi, Phi_n, Phi_p), for example a GNN prediction. On top of that it provides
`Solver.refine`, a small drift-diffusion solver that turns a surrogate guess into an initial guess
from which Sentaurus converges in about the same number of Newton steps as from its own solution.
The twin (`trifield.tcad.twin.Twin(refine=True)`) uses refinement by default.

## Equations

| piece | Sentaurus behaviour reproduced (verified node by node on true solutions) |
|---|---|
| box data | `MeasureCoefficientsDebug` in **internal** numbering; edge order (01,02,03,12,13,23); vertex order = TDR element order. Do **not** use `MeasureCoefficients.debug` (GrdNumbering): its writer emits slots [0,3,1,3,2,4], so edge (2,3) is lost. |
| Poisson | sum eps0 eps_T kappa (psi_j - psi_i) + alpha q M SC; eps per material (AlGaN interpolated in x); alpha = 1 + 8.2e-6 (Sentaurus constants). Interior residual about 1e-14. |
| statistics | E_C = C0 - chi - psi, E_F = -Phi. Densities come from inverting Sentaurus' Joyce-Dixon / asymptotic eta(r) (not the exact F_1/2), with kT = 0.0258519953 eV. Error about 1e-13. |
| space charge | p - n + N_D - N_A + interface-trap charge. Donor traps use steady-state SRH with Fermi gamma factors, E_t = E_i + E_mid. The volume density is sigma * A_interface / V_box. |
| electron flux | Scharfetter-Gummel with Fermi correction: w = psi + chi + V_T ln gamma_n. |
| mobility | Element-averaged (measure weights) Caughey-Thomas, beta 0.9, vsat 2.6e7. Driving force is \|grad Phi_n\| per element, or \|grad psi\| in elements touching a contact sheet. **The doping dependence is inactive in this deck**: Sentaurus sees N = 0 because Donor/AcceptorConcentration are not set, so mu_low = mumax(x) - mumin2. |
| holes | Mirror image of electrons (mu_p 20 / 116 / 150, beta 0.8). Recombination is SRH (tau = 1e-9) plus interface-trap capture: K Conc vth sigma (np - n1 p1)/(n + n1 + p + p1), with vth = 1.5626e7 cm/s. Holes carry no charge of consequence (p about 1e-50), but they set Phi_p. |
| boundary | Gate (TiN) psi = Vg + 0.846115. Gold bulk psi = V + 0.506115. Gold/semiconductor vertices: charge neutrality with Phi = V. Gold/AlGaN(x=0) vertices: psi - Phi_n = -0.140375. All of these constants are taken from the per-mesh equilibrium export. |

Residual levels on the true solutions, median relative:

| equation | level |
|---|---|
| Poisson, interior and oxide interfaces | 1e-15 |
| electron continuity, all node classes | 1e-13 to 1e-11 |
| hole continuity | 1e-12 |

## Per-mesh exports (`export_mesh_data`, cached in `<twin work>/physics/`)

Two short sdevice runs on the TDR grid the device is simulated on:

1. **box** (0 iterations, `Math{BoxMeasureFromFile BoxCoefficientsFromFile}`) writes
   `MeasureCoefficientsDebug`. The internal length unit is calibrated by the total element volume.
   The des/grid element mapping is checked through the element count and per-region volume
   conservation (`MeshData.box_check`).
2. **const** (equilibrium solve, Vg = Vd = 0, with extra Plot fields) gives:
   - per vertex: ElectronAffinity, EffectiveBandGap, N_C, N_V, DopingConcentration, xMoleFraction;
   - C0 = E_C + psi + chi;
   - the gate and Gold potentials, and the AlGaN(x=0) corner offset.

The trap capture constant, mobility, SRH and trap parameters are in `DeckPhysics` (defaults = palto
deck + parameter.par). Change them there if the deck changes.

Reading the element connectivity from the TDR (Ogawa container, zstd payloads) needs `zstandard`,
Python >= 3.14, or the `zstd` CLI.

## Refinement (`Solver(md, device).refine(guess, vg, vd)`)

1. Apply the boundary rules.
2. Damped Gummel: Poisson Newton with Phi frozen, then a linear SG solve for n. Phi_n steps are
   capped at 0.05 V and relaxed by 0.5, and Phi_n is clipped to [min(0, Vd), max(0, Vd)] +- 0.05 V
   (the maximum principle).
3. Joint (psi, Phi_n) Newton with a row-scaled merit and backtracking. If the merit falls by less
   than 30% over 6 iterations, run 10 more Gummel sweeps and restart (up to 8 restarts, 120 Newton
   iterations in total).
4. Solve for Phi_p from the linear hole continuity.

`Solver.terminal_currents` gives the electrode currents from the SG fluxes. In the deck's .plt units
they agree with Sentaurus to about 5e-4 relative on true solutions.

Backends:
- **CPU:** scipy/SuperLU.
- **GPU** (`device="cuda:k"`): cuDSS through nvmath (`pip install -e ".[gpu]"`), reusing the symbolic
  analysis per equation type. Every system is **two-sided (Ruiz) equilibrated** and gets 2 steps of
  iterative refinement: the Scharfetter-Gummel and joint Jacobians span ~1e-50..1e+10, and with row
  scaling alone cuDSS returned solutions with relative residuals up to 1e11 (joint Newton: median 1e2),
  so the refinement stalled and Sentaurus diverged from its guesses. With Ruiz scaling the GPU residuals
  match SuperLU (< 1e-13) and the refinement trajectory matches the CPU one. The linear hole solve
  (p-scaled, a few ms) stays on the CPU.
- **End to end** (25k-node mesh, capped refinement): 21 s on the GPU against 360 s on the CPU (17x);
  the surrogate ensemble pass is 1 s against 20 s. In the paired twin benchmark this took the warm twin
  from ~4.5x to 12x faster than the production Sentaurus flow.
- Worker threads of one twin can share a solver: `Backend.activate()` makes the device current for cupy,
  torch and cuda.core in the calling thread.
- cupyx `spsolve` was ~10x slower than SuperLU and inaccurate on the SG/hole systems.

## Contact-metal state (`Solver.metal_drop`)

Sentaurus keeps a Fermi-potential drop inside the conducting contact metals (Gold here, up to ~1.5 mV
between the contact sheet and the semiconductor interface) and stores it in `QuasiFermiPotential`. A guess
with the metals pinned at the applied bias starts the metal equation ~1e9 off; the drop must be right to
~5% (x0.8 of it: 53 Newton steps, x0.5 or x1.2: divergence at deep off-state). It is a linear conduction
problem: box Laplacian over the metal elements (conductance C / resistivity), the contact sheet at the
applied bias, the semiconductor current entering at the interface vertices (the SG residuals that
`terminal_currents` sums) as sources. One resistivity constant (`DeckPhysics.metal_resistance`) reproduces
Sentaurus to ~0.3% on two designs and all biases. With it, an exact solution written as a guess converges
in 2 Newton steps instead of 55. `Twin.with_metal_state` applies it (metal QuasiFermiPotential, Phi at the
ohmic interface vertices, psi = metal level + work-function offset on metal-only vertices; Sentaurus keeps
Phi_n / Phi_p there as 0 placeholders) when the guess' current is trustworthy.

## Known floors and caveats

- **Deep off-state (Vg <= -2 V at Vd = 12 V, normally-on designs).** Two differences remain. (1) At a few
  AlGaN/HfO2 trap vertices of the top AlGaN layer, Sentaurus' trapped charge is a fixed per-vertex factor
  1.6-2.8 larger than this model's (bias-independent; not the interface-area rule, interface multiplicity,
  region or box volumes); calibrated from one deep off-state solution, the true solutions' Poisson residual
  falls to the on-state floor (~1.4e-4) at every bias. (2) In fully depleted regions (n ~ 1e-20 cm^-3) the
  scaled electron-continuity rows are degenerate and Newton stalls at a merit of ~3e-2, even from
  Sentaurus' own neighbouring solution. Refined guesses there are outside Sentaurus' basin; the twin falls
  back to a ramp from its nearest converged point. A depletion-aware formulation (Slotboom variables or
  masking rows with negligible density) is the next step.

- **AlGaN/GaN heterointerface.** The deck has no double points, so each heterointerface vertex gets
  one parameter set. Sentaurus uses AlGaN(x of the vertex), and x = 0 there gives chi 3.958 / Eg 3.435
  instead of GaN's 3.437 / 3.396. The conduction band at those vertices is therefore about 0.5 eV
  deeper. This is an inconsistency in parameter.par, and its effect cannot be removed from the exported
  data. The Poisson residual on true solutions keeps a floor of about 3e-5 V-equivalent there.
  Refinement still closes it, which moves psi by up to ~3 mV from Sentaurus' solution.
- **Gold corners.** About 20 elements touching Gold/GaN vertices with n of about 1e-10 (Schottky-like)
  have a mobility we do not reproduce (driving-force rule unknown). About 22 nodes carry a continuity
  floor there.
- Phi_n at Gold/semiconductor vertices is V +- 3 mV in Sentaurus. Taking it as V leaves about 4 mV
  RMSE between refined and true fields. This does not stop Sentaurus from converging in its truth
  number of steps.
- The refinement is not globally convergent. From poor starts (equilibrium, linear Phi) it often stalls
  and Sentaurus may still fail. From the GNN ensemble guess it converged in all 6 validation cases.
- Tetrahedral meshes only; 3D only.
