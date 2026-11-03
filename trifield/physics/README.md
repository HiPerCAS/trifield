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
| holes | Mirror image of electrons (mu_p 20 / 116 / 150, beta 0.8). Recombination is SRH (tau = 1e-9) plus interface-trap capture: K Conc vth sigma (np - n1 p1)/(n + n1 + p + p1), with vth = 1.5626e7 cm/s, in both carrier equations (in deep depletion generation sets Phi_n). Holes carry no charge of consequence (p about 1e-50) except in the inversion layer under a strongly negative gate, but they set Phi_p. |
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
2. Vd = 0 (every contact grounded, the gate insulated): no current flows, so Phi_n = Phi_p = 0 exactly and
   only the nonlinear Poisson equation is solved (`equilibrium_vd0`; merit ~1e-14 in ~2 s).
3. Damped Gummel: Poisson Newton with Phi frozen, then a linear SG solve for n (with generation). Phi_n steps
   are capped at 0.05 V and relaxed by 0.5, and Phi_n is clipped to [min(0, Vd, Vg), max(0, Vd)] +- 0.05 V
   (the maximum principle; the gate term matters at deep off-state, where depleted Phi_n follows the hole level
   down to ~-2 V).
4. Joint (psi, Phi_n) Newton with backtracking on a row-scaled merit. Details that make it converge where it
   used to stall:
   - continuity rows are scaled by the one-directional (gross) SG flows, not by the net flows: next to the
     AlGaN/GaN barriers large opposite flows nearly cancel, and net-scaled rows changed by 1e3 x their scale for
     a 1e-4 V step (`Equations.gross_scale`);
   - carrier-free vertices (n < 1e-2 cm^-3; interface-trap vertices only below 1e-3 n1, where their occupancy
     saturates) are held fixed as unit rows: their Phi_n components reach 10-1e5 V and used to throttle every
     step through the clip. Afterwards their Phi_n is filled from the linear SG balance with generation;
   - the Jacobian ignores d mu / d (driving force) while the merit is above 1e-4 (Caughey-Thomas with
     beta = 0.9 has a cusp at zero driving force, which stalls the line search at near-equilibrium vertices),
     then switches to the exact Jacobian for a quadratic finish (`mobility_jacobian="hybrid"`);
   - the best iterate is kept. If the merit falls by less than 30% over 6 iterations, 10 more Gummel sweeps
     run and Newton restarts.
5. Phi_p from the linear hole continuity; on carrier-free vertices (n < 1e-10) Phi_p = Phi_n - 0.25 V
   (`hole_tie`, also used during the solve), because the lagged linear hole solve collapses a hole inversion
   layer (deep off-state) to p ~ 0 while psi was solved with one.
6. After convergence the contact-metal drop (`metal_drop`, up to ~1.5 mV at on-state) is put into the ohmic boundary
   values and Newton re-converges (`metal_boundary`), so the metal state the twin writes is consistent with the
   interior. Adding the drop only at the contact vertices of a solution computed with Phi = V there made Sentaurus
   fail or slow down at high current (101 (5,3): 61+ iterations, 15 without the metal state, 11 with this step).
7. A final nonlinear Poisson solve makes psi consistent with the quasi-Fermi levels written.

`RefineConfig(trace=True)` records per-iteration Newton diagnostics (merit parts, damping, step sizes,
frozen count, linear-solve residual) in `RefineResult.extra["trace"]`.

`Solver.refine(guess, vg, vd, config=...)` takes a RefineConfig or a dict of field overrides for one call
(thread-local, so worker threads can share a solver with different settings).

### Warm-start schedule (`trifield.tcad.twin.WARM_REFINE`, `WARM_REFINE_NEG`)

The defaults above converge the refinement to a merit of 1e-9; that is what refined-current screening needs
(at 1e-9 the refined drain current matches Sentaurus to ~2e-4, at 1e-3 only to ~10%). A Sentaurus warm start
does not: from merit 1e-3, 1e-4, 1e-5 or 1e-9 Sentaurus took the same number of Newton steps (5-7, 32 points
of 8 e-mode designs), from ~1e-2 4-15 more, from the unrefined guess it failed on 30/32. The twin therefore
refines its warm-start guesses (`Twin.solve_point` only) with
- Vg >= 0: Newton to 1e-3 (`newton_tol`, `converged_tol`) after 12 Gummel sweeps of 2 Poisson-Newton steps
  each (`gummel_iters`, `gummel_poisson_maxit`) instead of 25 fully converged ones. The sweeps cannot be
  dropped: from the raw guess a few low-density buffer vertices (n ~ 1e0..1e7) get Newton components of
  1e5-1e9 V and the global step clip throttles every step to lambda ~ 1e-10; >= 5 sweeps remove them.
- Vg < 0: the default sweeps and Newton to 1e-6. Below threshold at negative gate bias the cheap sweeps led
  to states (floating hole layer) from which Sentaurus needed 12-23 instead of 5 steps even at merit 3e-7,
  and merit 1e-3 cost 10 instead of 5 steps at another point (d-mode benchmark designs).
`Twin.refine_fields` and the escalation keep their configs.

Where the time went before (default schedule, main a86c895): CPU 20 s per point at 3-9k vertices, of which
the 25 Gummel sweeps 63% (157 Poisson factorisations per point) and the Fermi inversion `r_of_eta_t`
53% (30 Newton iterations per call, ~500 calls per point; 6 give bitwise the same result); GPU 12 s per
point at 11-13k vertices, of which the Ruiz equilibration (then 12 SpGEMMs per solve) 38%. With the
warm-start schedule: ~2 s per point on the GPU, now led by the CPU hole solve (~30%).

## Interface-trap calibration (`Twin.calibrate_traps`, `export.trap_factors`)

Our trap density per vertex (sigma * A_interface / V_box) is Sentaurus' except at convex corners of the top
AlGaN layer under the gate, where Sentaurus' AlGaN/HfO2 trap density is 1.6 to ~30 x ours (bias independent;
not explained by interface-area rules, multiplicity, region or box volumes). The twin measures it once per
design: a state with every donor trap ionised (Phi_n = +20 V, Phi_p = -20 V in the semiconductor) is loaded
and plotted after `Coupled(Iterations= 0)` with `RHSMin= 1e30` (without it Sentaurus moves the state), and
NetTrappedCharge / ours gives a factor per vertex (one ~9 s sdevice run, cached in
`work/physics/trap_factors.pkl`, applied with `Solver.set_trap_factors`). When the twin's persistent sdevice
servers are up (`trifield.tcad.server`), one of them loads the state and plots it without any solve: bitwise the
same NetTrappedCharge and factors, in ~0.6 s instead of a process of its own. The factors equal those derived from
converged deep off-state Sentaurus solutions to 1e-4, and with them those solutions satisfy our Poisson
equation at the on-state floor (rms 5e-2 -> 5e-4).

`Solver.terminal_currents` gives the electrode currents from the SG fluxes. In the deck's .plt units
they agree with Sentaurus to about 5e-4 relative on true solutions.

Backends:
- **CPU:** scipy/SuperLU.
- **GPU** (`device="cuda:k"`): cuDSS through nvmath (`pip install -e ".[gpu]"`), reusing the symbolic
  analysis per equation type (one analysis per type and design). Every system is **two-sided (Ruiz)
  equilibrated** (6 sweeps, applied to the CSR data in place: ~1 ms per solve) and gets 2 steps of
  iterative refinement: the Scharfetter-Gummel and joint Jacobians span ~1e-50..1e+10, and with row
  scaling alone cuDSS returned solutions with relative residuals up to 1e11 (joint Newton: median 1e2),
  so the refinement stalled and Sentaurus diverged from its guesses. With Ruiz scaling the GPU residuals
  match SuperLU (< 1e-13) and the refinement trajectory matches the CPU one. The linear hole solve
  (p-scaled) stays on the CPU: cuDSS + Ruiz is 15x faster there but fails on the first hole solve at
  on-state points (unknowns spanning 1e-3..1e31).
- The twin refines on the CPU below 4k vertices (`Twin.refine_cpu_below`; the warm-start refinement is
  1.6x faster on the GPU at 3.2k vertices, 3.8x at 8.6k) with torch capped at 4 intra-op threads
  (`Twin.refine_threads`): concurrent CPU refinements with one thread per core oversubscribe the host.
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

- **Deep off-state (Vg <= -2 V at Vd = 12 V, normally-on designs).** With electron generation, calibrated
  traps and frozen carrier-free vertices the refinement converges there and Sentaurus' solutions satisfy our
  equations (Poisson at the floor, continuity 5e-11). What remains is the hole inversion layer at the GaN/HfO2
  interface under the gate (Vg <~ -4 V): its level is floating (it exchanges holes only with p ~ 1e-11
  neighbours, below Sentaurus' error reference), our linear hole solve is singular there, and the 0.25 V
  tie is a heuristic (Sentaurus' states sit at Phi_p - Phi_n ~ -0.25..-0.37 V and accept a layer charge within
  ~1 decade). Some of these points still need many Sentaurus iterations or fail.

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

## Physics loss and pseudo-labels (`loss.py`, `trifield.pseudolabel`)

`ExactPhysicsLoss` evaluates these equations on a predicted triple as a training term: hard Dirichlet values from the
boundary rules above (bias-shift invariant offsets, one neutrality Newton per design), Poisson rows / (pdiag + q M
|dSC/dpsi|) [V], electron continuity rows / gross SG flow, carrier-free rows dropped, float64, differentiable. On the
float32 labels: Poisson ~1e-4 V rms, continuity ~1e-6 (median). Normalisers and the carrier-free mask must come from fixed
reference fields (`norm_from="ref"`: labels, or the first prediction of an unlabelled item): recomputed from the prediction
and detached, the dropped d(normaliser)/d(field) term is as large as the kept one on badly balanced rows, and Adam then
raises the loss at every learning rate down to 1e-6 (a weight of 0.1 doubled the field error). Even with a correct
gradient, descent on the residual lowers it 5-50x without moving psi towards the solution (a local residual barely sees
the smooth, long-range part of the surrogate error), so the loss is an option, not part of the recipe.

`trifield.pseudolabel` uses the equations through `Solver.refine` instead: refined guesses (full RefineConfig, ~10 s CPU
per bias point, 99.6% converged at Vg >= 0) are within 1.6-4 mV of Sentaurus and serve as training labels for meshed but
unsimulated designs (`train.py --unlabeled_dir <mesh graphs> --pseudo_dir <labels>`).
