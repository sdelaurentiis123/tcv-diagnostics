# Phase 2 paired potential-elliptic oracle protocol

**Status:** frozen before the first paired elliptic reconstruction of a raw
85604 state

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited from the extractor, compiled
driver, comparator, and launcher

**Purpose:** validate the exact stored-state potential inversion on
value-independent native-81 frames, then measure the continuous effect of
replacing the retained relaxed radial boundary with its instantaneous target

The machine-readable authority is
`paper0/manifests/phase2_potential_elliptic_85604.json`. This is a
selected-frame implementation and continuous-effect oracle. It does not train a
model, select a final state representation, or establish an all-frame or
held-out result.

## 1. Why this rung is required

The completed raw-guard audit establishes that the saved radial potential
midpoint is structurally valid and distinct from the instantaneous target at
every frame/y location. It does not establish how the elliptic inversion
propagates that difference into interior potential or transport.

Before comparing boundary policies, Paper 0 must first demonstrate that a
standalone driver reproduces stored `phi` from the same stored volume
state, geometry, boundary midpoint, source equation, and BOUT++ solver. A
counterfactual from an unvalidated inversion is not scientific evidence.

This rung therefore has two ordered questions:

1. **source reconstruction:** does the retained-boundary solve reproduce stored
   interior `phi`?
2. **paired boundary effect:** holding all volume fields and numerical options
   fixed, how much does using the instantaneous target change interior
   `phi` and the accepted radial ExB transport quantities?

The paired effect is accepted only if source reconstruction passes.

## 2. Exact executed equation

The 85604 input has

```text
[vorticity]
diamagnetic_polarisation = true
average_atomic_mass = i:AA = 2
split_n0 = false
phi_boundary_relax = true
```

The charged species are deuterium ions with `AA=2, charge=+1` and
electrons with `AA=1/1836, charge=-1`. Hermes therefore constructs

\[
\widehat{P}_i
=
P_i-\frac{P_e}{3672}.
\]

The electron term is retained even though it is small. Calling
\(\widehat{P}_i\) simply “ion pressure” is insufficient for this oracle.

With `split_n0=false`, Hermes uses the BOUT++ X-Z Laplacian solver on
each y slice. Let

\[
C=\frac{2}{B^2},
\qquad
u=\phi+\widehat{P}_i.
\]

At BOUT++ revision
`7d28d67c3f12c24ec281c0982e870f5369c65a6f`, the default cyclic solver
solves

\[
\nabla_\perp^2 u
+
\frac{1}{C}\nabla_\perp C\cdot\nabla_\perp u
=
\frac{B^2}{2}\,Vort.
\]

Hermes then returns

\[
\phi=u-\widehat{P}_i.
\]

The driver must use the ABI-validated BOUT++ `5.2.1` install from job
`6890766` and explicitly select the same `cyclic` default so that a
future factory-default change cannot silently alter the oracle.

## 3. Exact metric normalization

The geometry file stores SI metric values. Before creating the Laplacian,
Hermes normalizes them using

```text
Bnorm = 1 T
rho_s0 = 0.0007224847664314034 m.
```

The GPL-marked driver may adapt the exact normalization block from the locked
`hermes-3.cxx` source. It must apply every `dx`, `Bxy`,
contravariant metric, covariant metric, and Jacobian scaling before calling
`Coordinates::geometry()`. Omitting or partially reproducing this block
invalidates the solve.

## 4. Value-independent frame selection

Reuse the five frame indices frozen before the earlier native-frame oracle:

```text
[0, 156, 312, 467, 623].
```

They are the nearest-half-up indices at fractions `0, 1/4, 1/2, 3/4, 1`
of inclusive archive interval `0..623`. This selection predates the
boundary-amplitude result and does not include the post hoc maximum frames
591 or 586.

The machine-readable manifest locks both that earlier selection manifest and
the completed all-frame boundary-audit result by path and SHA-256. Those files
are provenance inputs, not parameters that the implementation may revise.

The selected frames are implementation cases within one correlated run, not
five independent experiments. A pass may authorize an unchanged all-frame
extension; it is not itself an all-frame materiality conclusion.

## 5. Canonical extraction

Verify all 256 raw rank filenames, control hashes, dimensions, time sequences,
processor coordinates, Hermes revision, limiter, `zperiod=5`, and native
`z=81`.

Assemble the guard-stripped physical fields

```text
Ne, Pe, Pi, Vort, phi
```

for the five selected frames into float64 arrays with axes

```text
[selected_frame=5, x=64, y=32, z=81].
```

From the 32 radial-boundary ranks, independently extract for each selected
frame, side, and global y:

- saved boundary midpoint;
- instantaneous target, the toroidal mean of adjacent interior `phi`;
- midpoint departure;
- exact outer-guard-copy residual;
- exact toroidal-midpoint-constancy residual.

Store midpoint, target, and departure as float64 arrays with axes

```text
[selected_frame=5, side=2, y=32].
```

The extractor must reproduce the relevant selected-frame values from the
tracked all-frame boundary audit, record deterministic array and file digests,
and refuse existing outputs. It must not read 85606.

## 6. Compiled paired solve

Use four MPI ranks with the already validated topology:

```text
NXPE=1, NYPE=4, MYSUB=8, native z=81, zperiod=5.
```

Each rank reads only its explicit canonical y slice. The GPL-marked driver
must:

1. normalize the exact geometry as Hermes did;
2. construct \(\widehat{P}_i=P_i-P_e/3672\) and apply its source Neumann
   boundary;
3. create a BOUT++ cyclic Laplacian with
   `C=2/Bxy^2` and `INVERT_SET` on both radial sides;
4. form the exact right-hand side `Vort*Bxy^2/2`;
5. solve a retained arm using the extracted saved midpoint;
6. solve an instantaneous arm using the extracted target midpoint;
7. subtract \(\widehat{P}_i\) from each solution;
8. communicate guards and write both physical solutions plus exact input and
   boundary echoes.

For either arm, set the adjacent `phi` guard from midpoint \(b\) and
adjacent interior reference \(\phi_i(k)\) as

\[
\phi_g(k)=2b-\phi_i(k).
\]

Then reproduce the Hermes half-cell adjustment for
\(\phi+\widehat{P}_i\) before inversion. The two arms must differ only in
whether \(b\) is the saved midpoint or instantaneous target.

The instantaneous arm is a frozen one-step counterfactual based on the stored
adjacent-interior target. It is not an iterated self-consistent Neumann solve
and must not be described as one.

## 7. Reconstruction gate

Before any counterfactual is accepted:

- canonical field and boundary echoes must be bitwise exact;
- all fields and solver outputs must be finite;
- the retained arm must reproduce stored interior `phi` on every selected
  frame in both the full physical domain and the previously accepted
  guard-independent transport interior;
- raw, not constant-shift-aligned, error is the primary gate because
  `INVERT_SET` fixes both radial values;
- a constant-shift-aligned diagnostic may be reported for failure attribution
  but cannot turn a raw failure into a pass.

Reuse the previously frozen compiled continuous rule:

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference.
```

Report relative L2, RMSE, bias, correlation, maximum error and location, and
per-frame values, but do not add a second post hoc gate. Any failure blocks
the paired-effect interpretation until source equation, metric normalization,
boundary half-cell handling, field timing, decomposition, and dump semantics
are resolved.

## 8. Paired boundary-effect metrics

Only after the reconstruction gate passes, compare the retained and
instantaneous arms.

### Interior potential

Report member-free deterministic paired differences in normalized units and
volts for:

- relative L2;
- RMSE;
- mean bias;
- correlation;
- maximum absolute difference and location;
- radial profile of RMS difference;
- the frozen geometry regions;
- every selected frame separately and pooled.

### Transport

Apply the already validated native-81 geometry-aware radial ExB evaluator to
each arm separately. Use direct evolved fields:

```text
particle: Ne
electron internal energy: 1.5 * Pe
ion internal energy: 1.5 * Pi
total internal energy: sum of electron and ion
```

Report:

- strict local-face relative L2, RMSE, bias, correlation, and sign error;
- integrated confined-separatrix time-point differences;
- confined edge, private-flux, SOL, outboard-midplane, X-point, and divertor
  summaries where the authoritative masks apply;
- each selected frame separately and pooled.

Do not call internal-energy ExB flow total heat flux. Do not compute transport
from an averaged potential.

### No post hoc materiality label

This selected-frame rung reports continuous effect sizes and no discovered
amplitude may be converted into a new negligible/material cutoff. Exact state
closure and pragmatic forecast sufficiency remain separate questions.

## 9. Interpretation

- If retained-boundary reconstruction fails, stop. The standalone inversion
  has not reproduced the saved source state, so the paired effect is invalid.
- If reconstruction passes, the implementation and exact boundary policy are
  validated on the five selected 85604 states.
- Any nonzero paired difference is expected because the boundary midpoint is
  already known to differ. Its interior and transport magnitude, not mere
  nonzero count, is the result of this rung.
- A selected-frame pass may authorize an unchanged all-frame extension. It
  does not establish time-block stability.
- `S6+Bphi` remains the exact source-state candidate. A pragmatic
  `S6+phi` or history-conditioned baseline may still be preferable if
  later matched predictive tests justify it.
- No result selects a stochastic architecture or authorizes training.

## 10. Execution lock

The launcher must:

1. run from a clean, committed Paper 0 checkout on Rocky 9;
2. use the ABI-validated BOUT++ install from job `6890766`;
3. verify exact Hermes, BOUT++, input, geometry, manifest, extractor, driver,
   comparator, and launcher-adjacent source hashes;
4. write a new immutable job directory and refuse overwrites;
5. record extraction, build, MPI command, environment, Slurm state, raw arrays,
   strict JSON, and SHA-256 inventories;
6. require four-rank `NXPE=1` topology;
7. report reconstruction and paired-effect status separately;
8. report `held_out_85606_read=false`.

This protocol authorizes only the selected-frame 85604 paired elliptic oracle
defined above.
