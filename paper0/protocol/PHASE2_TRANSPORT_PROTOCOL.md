# Phase 2 geometry-aware transport protocol

**Protocol status:** source audit complete and frozen before implementation

**Development data:** synthetic known-answer fields and TCV/Hermes `85604`

**Sequestered evaluation run:** `85606`

**Current gate:** shifted-field-line derivative rung passed; full transport is
blocked on the remaining Section 9 face-flow, native-grid, resampling, mask,
unit, and ensemble checks

This document replaces the exploratory image-space flux proxy with the
transport definition actually advanced by the Hermes-3 simulation. It does not
authorize model training, access to `85606`, or a transport-fidelity claim.

## 1. Why the historical proxy is not the Paper 0 metric

The predecessor function `src/tcv_eval/flux.py` used

\[
v_x^{\mathrm{proxy}}=-\frac{1}{B_{xy}}\,\frac{\partial\phi}{\partial z},
\qquad
\Gamma^{\mathrm{proxy}}=\langle N_e v_x^{\mathrm{proxy}}\rangle_{y,z},
\]

with a unit-spaced centered difference and an unweighted pixel mean. It was a
reasonable exploratory test of whether density--potential correlation survived
a rollout. It is not the conservative Hermes operator and is excluded from all
Paper 0 physical-transport gates and claims.

In particular, the proxy omits:

- the finite-volume Jacobian and face reconstruction;
- the physical `dz` of the simulated one-fifth toroidal wedge;
- the additional radial contribution enabled by `poloidal_flows = true`;
- shifted-field-line `y` derivatives and branch-cut topology;
- geometry-aware surface integration;
- declared physical-unit conversion.

Any retained use must contain `proxy` in the metric name and figure label.

## 2. Executed simulator source lock

The representative raw dump records the following embedded revisions. The
files were read from clean detached checkouts of the official repositories;
they are evidence dependencies, not vendored Paper 0 code.

| Component | Official repository | Executed revision | License |
|---|---|---|---|
| Hermes-3 | `https://github.com/boutproject/hermes-3` | `920ba829cc78cdab0dbf6101c69fecc4689bd8dd` | GPL-3.0 |
| BOUT++ | `https://github.com/boutproject/BOUT-dev` | `7d28d67c3f12c24ec281c0982e870f5369c65a6f` | LGPL-3.0 |

Critical source hashes are stored in
`paper0/manifests/phase1_85604_sources.json`. The relevant implementation is:

- Hermes `src/div_ops.cxx`, function `Div_n_bxGrad_f_B_XPPM`;
- Hermes `src/evolve_density.cxx`, which evolves
  `-Div_n_bxGrad_f_B_XPPM(N, phi, ..., poloidal_flows, true)`;
- Hermes `src/evolve_pressure.cxx`, which applies the same operator to pressure;
- BOUT++ `src/mesh/parallel/shiftedmetric.cxx` and `src/sys/derivs.cxx`, which
  define the shifted-field-line transform and `DDY` convention.

The executed configuration fixes:

```text
zperiod = 5
MZ = 81
mesh:paralleltransform:type = shifted
hermes:normalise_metric = true
e:poloidal_flows = true
i:poloidal_flows = true
vorticity:poloidal_flows = true
vorticity:exb_advection_simplified = false
Nnorm = 1e19 m^-3
Tnorm = 50 eV
Bnorm = 1 T
```

## 3. Coordinate and grid conventions

The BOUT++ logical coordinates are `x` (radial), `y` (parallel/poloidal index),
and periodic `z` (toroidal/binormal coordinate). The simulation covers one
fifth of the torus, so

\[
L_z=\frac{2\pi}{5},
\qquad
n=5k.
\]

The native run has 81 toroidal cells and therefore

\[
\Delta z_{81}=\frac{2\pi}{5\cdot81}.
\]

The model dataset strips two radial guard cells from each side and Fourier
resamples 81 cells to 88. Its metric evaluation uses

\[
\Delta z_{88}=\frac{2\pi}{5\cdot88}.
\]

Applying the same finite-volume form on 88 resampled cells is not bitwise the
same calculation as the native 81-cell simulation: the nonlinear limiter does
not commute with Fourier resampling. Paper 0 must report a native-81 versus
resampled-88 truth sensitivity before evaluating a learned model.

The model fields remain in Hermes-normalized units. For transport evaluation,
linear density is required; logarithms and training standardization must be
inverted first. Pressure is reconstructed as

\[
P_e=N_eT_e,
\qquad
P_i=N_eT_i,
\]

using quasineutral ion density for this run.

## 4. Conservative radial face flow

Let the radial face `i+1/2` separate cells `i` and `i+1`. Hermes constructs the
radial part of

\[
\nabla\!\cdot\!\left(q\,\frac{\mathbf b\times\nabla\phi}{B}\right)
\]

as the sum of an `x-z` contribution and, when `poloidal_flows = true`, an
`x-y` contribution. The face quantity below is the numerical flow whose
difference is divided by the cell volume factor `J dx` in the divergence.

### 4.1 Toroidal (`x-z`) contribution

Hermes first averages potential to the two corners of the radial face:

\[
\phi_{++}=\tfrac14(\phi_{i,j,k}+\phi_{i,j,k+1}
                         +\phi_{i+1,j,k}+\phi_{i+1,j,k+1}),
\]

\[
\phi_{+-}=\tfrac14(\phi_{i,j,k}+\phi_{i,j,k-1}
                         +\phi_{i+1,j,k}+\phi_{i+1,j,k-1}).
\]

The radial face velocity factor is

\[
V^{xz}_{i+1/2,j,k}
=\frac{J_{i,j}+J_{i+1,j}}{2}
 \frac{\phi_{++}-\phi_{+-}}{\Delta z}.
\]

The advected face value is selected by the sign of this factor from the
Monotonized-Central reconstruction used by the executed Hermes revision. The
numerical radial face flow is

\[
F^{xz}_{q,i+1/2,j,k}=V^{xz}_{i+1/2,j,k}\,q^{\mathrm{MC,up}}_{i+1/2,j,k}.
\]

### 4.2 Shifted-poloidal (`x-y`) contribution

Define

\[
A_{i,j}=\frac{g^{xx}_{i,j}g^{yz}_{i,j}}{B_{xy,i,j}^{2}}.
\]

The second radial velocity factor is

\[
V^{xy}_{i+1/2,j,k}
=\frac{J_{i,j}+J_{i+1,j}}{2}
 \frac{A_{i,j}(D_y\phi)_{i,j,k}
      +A_{i+1,j}(D_y\phi)_{i+1,j,k}}{2}.
\]

Here `D_y` is BOUT++ `DDY`, not a plain array difference. For the executed
`shifted` transform, BOUT++ Fourier-shifts each `z` line using `zShift`, takes
the configured centered derivative in field-aligned `y`, divides by `dy`, and
shifts the result back. Neighbor communication follows the single-null branch
topology and applies physical-boundary guard values.

Hermes uses a Fromm upwind face reconstruction with positivity clipping for
this term:

\[
F^{xy}_{q,i+1/2,j,k}=V^{xy}_{i+1/2,j,k}\,
q^{\mathrm{Fromm,up+}}_{i+1/2,j,k}.
\]

The total conservative radial face flow is

\[
F^x_q=F^{xz}_q+F^{xy}_q.
\]

A naive `np.roll` along `y` is forbidden because it crosses divertor targets
and X-point branch cuts incorrectly.

## 5. Reported transport quantities

The primary normalized face quantities are:

\[
F_N^x=F_q^x\big|_{q=N_e},
\]

\[
F_{U_e}^x=\frac32 F_q^x\big|_{q=P_e},
\qquad
F_{U_i}^x=\frac32 F_q^x\big|_{q=P_i}.
\]

The factor `3/2` makes the latter internal-energy advection, consistent with
the evolved pressure equation. A `5/2` enthalpy convention may be shown as a
clearly named sensitivity after simulator-owner confirmation; it must not
silently replace the primary definition. Until that terminology is confirmed,
Paper 0 calls these **advected internal-energy flows**, not unqualified heat
fluxes.

For a declared set of face cells `S`, the normalized surface flow is

\[
\mathcal F_q^x(S)=\sum_{(j,k)\in S}F^x_{q,i+1/2,j,k}\,\Delta y_{i,j}\,\Delta z.
\]

No unweighted `mean(y,z)` is substituted for this integral. The positive
coordinate direction is initially reported as `+x`; an “outward” sign label is
allowed only after the `psi`, separatrix, and physical-coordinate orientation
test passes.

The physical conversions used by the executed Hermes source are

\[
C_N=\rho_{s0}^{3}N_{\mathrm{norm}}\Omega_{ci}
\quad [\mathrm{s}^{-1}],
\]

\[
C_P=\rho_{s0}^{3}P_{\mathrm{norm}}\Omega_{ci}
\quad [\mathrm{W}],
\qquad
P_{\mathrm{norm}}=eT_{\mathrm{norm}}N_{\mathrm{norm}},
\]

with

\[
C_{s0}=\sqrt{\frac{eT_{\mathrm{norm}}}{m_p}},
\qquad
\Omega_{ci}=\frac{eB_{\mathrm{norm}}}{m_p},
\qquad
\rho_{s0}=\frac{C_{s0}}{\Omega_{ci}}.
\]

Both normalized and SI values are retained so that unit conversion cannot hide
a sign, discretization, or geometry error.

## 6. Geometry and topology masks

The mesh records `nx = 68` including radial guards and `ny = 32`. The model
crop maps local `x = 0..63` to global grid `x = 2..65`. The mesh metadata are:

```text
ixseps1 = 18
ixseps2 = 68
jyseps1_1 = 7
jyseps1_2 = 16
jyseps2_1 = 16
jyseps2_2 = 23
```

Thus the separatrix lies at model-local radial index 16 under the verified crop
mapping. These indices are sufficient to construct candidate confined-edge,
private-flux, and scrape-off-layer topology regions, but not to infer every
paper mask from image coordinates. The final masks must additionally validate
`Rxy`, `Zxy`, and `psixy`, name cell versus face membership, and identify the
X-point neighborhood without double counting branch-adjacent cells.

## 7. Data limitations discovered by the audit

The raw dump exposes `pfe_tot_ylow`, `efe_tot_ylow`, and `efi_tot_ylow`, but no
corresponding `_xlow` radial-flow outputs. The stored `ylow` quantities contain
parallel and other configured contributions and are not a like-for-like oracle
for radial ExB transport. Therefore Paper 0 cannot validate radial transport by
comparing against a convenient stored variable.

The Well conversion also removed BOUT guard cells and X-point region metadata.
Exact evaluation near radial boundaries and shifted `y` boundaries requires
either reconstruction verified against the original distributed dumps or a
BOUT++/Hermes oracle harness.

## 8. Allowed staged implementation

The implementation may proceed in explicitly labeled stages:

1. `xz` face-flow component with periodic `z`, MC reconstruction, and exact
   geometry factors;
2. shifted `DDY` plus the `xy` face-flow component;
3. total radial particle and internal-energy face flows;
4. geometry-region surface integrations and member-wise ensemble reductions.

An intermediate API must include `_xz_component` or `_partial` in its public
name and metadata. It cannot satisfy the transport gate, generate diagnostic
rankings, or appear in a figure as “particle flux” or “heat flux.”

The shifted-`DDY` primitive has passed its compiled oracle. The independently
written Fromm and shifted-`xy` face routines are present only as
`candidate_partial` APIs and remain blocked pending the separately committed
compiled-Hermes face-flow oracle. Their synthetic tests are necessary but not
sufficient evidence.

As required by the general metric protocol, nonlinear transport is computed
for every ensemble member before ensemble reduction.

## 9. Validation ladder required before release

The full metric remains blocked until all of the following pass:

1. **Manufactured `xz` modes:** constant potential gives zero flow; known
   sinusoidal potential gives the expected sign, amplitude, and periodic wrap;
   constant advected fields isolate the velocity discretization.
2. **Limiter tests:** hand-calculated MC and Fromm stencils cover positive and
   negative velocities, extrema, positivity clipping, and radial-edge refusal.
3. **Shift transform:** a manufactured Fourier mode recovers the exact
   `zShift` phase in `toFieldAligned` and `fromFieldAligned`.
4. **Shifted `DDY`:** the Paper 0 implementation matches an exact BOUT++
   operator harness, including at least one periodic-core neighbor, one branch
   connection, and one open-field boundary case.

   The compiled comparison is frozen before execution as follows. BOUT++
   `5.2.1` evaluates `DDY(..., "C2")` on the hash-locked 85604 geometry at the
   native 81-cell toroidal resolution and `zperiod=5`. Four manufactured inputs
   exercise a constant, toroidal modes, a `y` code, and a mixed `x-y-z` field.
   The candidate is compared on model radial cells `grid_x=2:66` and physical
   `y=1:31`; target cells `y=0,31` are excluded because their physical guards
   are absent from the model state. Every case must pass in the full valid
   region, ordinary sequential stencils, both private-flux connections, both
   core branch connections, and the open SOL. For each region, with reference
   scale `s=max(abs(BOUT_DDY))`, acceptance requires no non-finite values and

   `max_abs_error <= 5e-10 + 5e-10 * s`.

   This tolerance and all manufactured fields are committed before the first
   comparison job. A failure is recorded; it is not tuned away.

   The single-null topology requires the 8-cell divertor-leg segments to align
   with MPI `y` partitions. The minimal valid runtime therefore uses four ranks
   with `NXPE=1`, `NYPE=4`, and `MYSUB=8`. Each rank writes its own NetCDF
   output. Paper 0 removes exactly two `y` guards per side, orders blocks by the
   output's explicit `PE_YIND`, verifies the full decomposition metadata, and
   concatenates four 8-cell blocks into the original 32-cell domain before any
   numerical comparison. This execution correction does not alter the fields,
   masks, tolerance, or candidate implementation.

   On the file-backed geometry, the driver must construct the four tracked
   expressions explicitly through BOUT++ `FieldFactory`; `mesh->get` is not an
   accepted loader because missing grid variables become zero. Before any
   derivative score is accepted, the constant case must equal `2.5` within
   `1e-13`, and every nonconstant case must be finite with peak-to-peak range
   greater than `1e-6`.

   The authoritative `ShiftAngle` array is finite only inside `ixseps1` and
   NaN in the SOL, where BOUT++ does not apply a twist shift. Paper 0 requires
   finite values for all used model cells `x<16`; topology-unused outer entries
   are replaced by zero only after this check. A non-finite used entry is a
   hard failure.

   **Execution record:** Rocky 9 job `6891059`, from clean commit `0223035`,
   passed the unchanged rule for all four cases and every declared region. The
   largest absolute error was `3.025468764406014e-12`; all inputs passed the
   non-collapse gate and no compared value was non-finite. This releases only
   the shifted-`DDY` primitive. It does not release the shifted-`xy` face flow,
   total transport, or any model transport claim.

5. **Shifted-`xy` radial face flow:** the candidate Fromm reconstruction,
   positivity decision, velocity factor, and face flow match a compiled,
   source-derived Hermes harness.

   The comparison is frozen before its first execution. The GPL-marked C++
   driver is adapted from Hermes-3 `src/div_ops.cxx:273-326` at hash-locked
   revision `920ba829`; the launcher verifies that revision and the exact
   `div_ops.cxx` SHA-256 before compiling against the accepted BOUT++ build.
   It reads only the 85604 geometry and four FieldFactory pairs: constant,
   smooth, signed-velocity, and positivity-clipping cases. It echoes both
   inputs and writes the face velocity, selected Fromm state, binary clipping
   decision, and resulting face flow.

   Paper 0 compares the independent NumPy candidate on native-81,
   `zperiod=5`, model crop `grid_x=2:66`, safe radial faces whose model-local
   left cells are `1:62`, and physical `y=1:31`. Every continuous quantity in
   every case must pass over all valid points, ordinary/sequential cells, both
   private-flux connections, both core branch connections, the separatrix
   radial face, and open SOL. With `s=max(abs(reference))`, the unchanged rule
   is

   `max_abs_error <= 5e-10 + 5e-10 * s`.

   No compared value may be non-finite. The clipping reference must contain
   only exact binary decisions and match the candidate at every valid point.
   Every nonconstant case must contain both positive and negative face
   velocities. The clipping case must select at least one clipped and one
   unclipped state. Constant inputs must match `q=2.5` and `phi=4.0` within
   `1e-13`; every nonconstant input must be finite with peak-to-peak range
   greater than `1e-6`. The fields, masks, coverage requirements, and tolerance
   are committed before execution and are not relaxed after seeing a result.

6. **Conservation:** face-flow differences reproduce the source operator's
   volume-weighted divergence on manufactured fields.
7. **Native-grid oracle:** selected raw 81-cell 85604 frames match the
   hash-locked Hermes/BOUT++ calculation to a prospectively set tolerance.
8. **Resampling sensitivity:** truth transport on native 81 cells is compared
   with the 88-cell Fourier-resampled evaluation; the discrepancy is reported
   and cannot be tuned away.
9. **Geometry masks:** synthetic and real-grid tests verify cell/face indexing,
   the local/global radial map, separatrix membership, branch regions, and
   `+x` versus outward orientation.
10. **Units:** normalized-to-SI conversion is independently hand calculated
    and compared with the simulator metadata convention.
11. **Ensemble semantics:** a nonlinear known-answer case proves that transport
    is evaluated member-wise rather than from ensemble-mean fields.

Only after this ladder passes may O1 be rerun with a transport gate. The
current O1 field, spectral, and cross-field conclusions remain valid; the
absence of a transport result is a documented blocker, not a zero or a pass.

## 10. Immediate decision

No architecture work or training is justified by this audit. The shifted
`DDY` rung has passed; the next safe step is the prospectively frozen compiled
Hermes shifted-`xy` face-flow comparison above. If the remaining exact oracle
ladder cannot be completed, Paper 0 must retain cross-spectrum and cross-phase
as validated joint-field metrics and omit strong physical-transport claims
rather than promote the historical proxy.
