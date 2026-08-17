# Phase 2 potential-to-vorticity forward-closure protocol

**Status:** frozen before implementation or source-data execution

**Scope:** five predeclared 85604 frames, the exact executed Hermes/BOUT++
revision, and the immutable canonical inputs already used by the accepted
runtime-pressure potential replay

**85606 accessed:** no

**Training authorized:** no

## 1. Question

The accepted inverse replay established that stored generalized vorticity,
runtime species pressure, the retained radial potential midpoint, and the
executed cyclic solve reproduce stored electrostatic potential on five frozen
frames. This protocol tests the remaining direction of the deterministic
closure:

> Does applying the same discrete cyclic Laplacian matrix to stored potential
> and runtime pressure reproduce stored generalized vorticity?

This is a source-state oracle, not a forecast, codec, calibration,
assimilation, or diagnostic-ranking experiment.

## 2. Continuous relation and runtime pressure

For the executed Boussinesq configuration, define

\[
C = \frac{\bar A}{B_{xy}^{2}},
\qquad
u = \phi + \widehat P_i,
\qquad
\bar A = 2.
\]

The cyclic inverse solves

\[
\mathcal L_C u = \frac{\mathrm{Vort}}{C},
\]

where

\[
\mathcal L_C u
=
\nabla_\perp^2 u
+
\frac{1}{C}\nabla_\perp C\mathbin{\cdot}\nabla_\perp u.
\]

The forward relation tested here is therefore

\[
\mathrm{Vort}_{\mathrm{forward}}
=
C\,\mathcal L_C\!\left(\phi+\widehat P_i\right).
\]

The pressure supplied to vorticity is the runtime species pressure published
by `EvolvePressure`, not raw evolved pressure. For each species,

\[
P_s^{\mathrm{runtime}}
=
N_s\,
\frac{\max(P_s^{\mathrm{raw}},0)}
{\operatorname{softFloor}(N_s,10^{-7})},
\]

with

\[
\operatorname{softFloor}(N,f)
=
\max(N,0)+f\exp\!\left[-\frac{\max(N,0)}{f}\right].
\]

For this quasineutral deuterium-electron system,

\[
\widehat P_i
=
P_i^{\mathrm{runtime}}
-
\frac{P_e^{\mathrm{runtime}}}{3672}.
\]

No negative evolved pressure is clipped outside this exact runtime
transformation.

## 3. Primary discrete operator

The primary result must apply the matrix coefficients used by the executed
BOUT++ cyclic inversion. For every physical `y`, toroidal Fourier index
`k=0..40`, and physical radial cell `x`, the implementation calls the public
`Laplacian::tridagCoefs` method on the same configured solver with

```text
A = 0
D = 1
C1 = C2 = 2 / Bxy^2
all_terms = true
nonuniform = true
zperiod = 5
native_z_samples = 81
```

and applies

\[
\widehat{\mathcal L_Cu}_{x,k}
=
a_{x,k}\widehat u_{x-1,k}
+b_{x,k}\widehat u_{x,k}
+c_{x,k}\widehat u_{x+1,k}.
\]

The forward result is obtained with the same public BOUT++ `rfft`/`irfft`
convention used by `LaplaceCyclic`, followed by multiplication by `C` in real
space. All 41 real-FFT modes are retained; no spectral truncation, smoothing,
constant alignment, or learned correction is permitted.

The primary operator is not `FV::Div_a_Grad_perp` from the separate,
unexecuted `relax_potential` component. That operator represents the same
continuous relation but is a distinct discretization and cannot be silently
substituted for the cyclic matrix used by run 85604.

## 4. Boundary and domain contract

The input is the retained-boundary arm accepted by job `6892641`:

1. reconstruct `phi` radial guards from the saved midpoint values;
2. copy the first radial guard into the second exactly as Hermes does;
3. apply the Hermes Neumann boundary policy to runtime `Pi_hat`;
4. communicate internal MPI guard cells before applying the radial stencil;
5. form `u=phi+Pi_hat` only after those steps.

The cyclic solver replaces two radial **guard-cell** rows on each exterior
side with `INVERT_SET` equations. It does not replace any of the 64 stored
physical radial rows. The primary comparison therefore covers the complete
stored physical domain:

```text
[selected_frame, x=0..63, y=0..31, z=0..80]
```

The first and last physical radial cells use the reconstructed exterior guard
values in their source-matched stencil. No physical row may be removed after
results are inspected.

## 5. Frozen frames and immutable inputs

The value-independent frame rule and selected frames remain unchanged:

```text
[0, 156, 312, 467, 623]
```

The implementation must reuse, read-only:

- canonical inputs from job `6892446`, SHA-256
  `e090b3a23fa6eedf8c37e74421c08bafd3eb513039fa7621b5d612a7e1cbba3e`;
- extraction record from job `6892446`, SHA-256
  `e30e4f14dddfdff369387f9e8657b31a3e56bb4dc628c1e9ef5d86bd5bfd68be`;
- accepted corrected replay result from job `6892641`, SHA-256
  `ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a`;
- the same grid, BOUT++ build, Hermes revision, runtime-pressure contract,
  toroidal mapping, and retained-boundary definition.

No extraction rule, frame, numerical option, or tolerance may change in this
extension.

## 6. Implementation gates before the source comparison

The compiled forward implementation must pass all of the following before a
stored-`Vort` result is interpretable:

1. **Input echoes:** every canonical volume and retained-boundary input is
   bitwise exact after MPI assembly.
2. **Runtime pressure:** the previously frozen runtime `Pe`, `Pi`, and
   `Pi_hat` checks pass at `atol=rtol=1e-12`, including the known negative raw
   `Pi` point at frame `312` mapping to zero runtime `Pi`.
3. **Constant null:** applying the forward matrix to a spatially constant `u`
   returns zero on every physical cell to the source-comparison tolerance.
4. **Gauge invariance:** adding a fixed nonzero constant to `u` does not change
   the physical forward result to the source-comparison tolerance.
5. **Manufactured round trip:** a deterministic field containing nonconstant
   radial structure and at least the `k=0` and `k=3` toroidal Fourier modes is
   passed through forward matrix application and the configured cyclic
   inverse. The reconstructed manufactured field must pass the same
   scale-aware continuous tolerance on all physical cells.
6. **Synthetic reduction tests:** pure known-answer tests cover complex
   tridiagonal application, FFT axis ordering, rank assembly, finite-value
   rejection, and the acceptance boundary.

If any implementation gate fails, the stored-source comparison is blocked and
the job exits nonzero after writing a failure artifact.

## 7. Source-comparison metrics and acceptance

For each frame and for the pooled five-frame domain, report:

- maximum absolute difference and its physical location;
- RMSE;
- bias;
- relative L2 error;
- correlation;
- non-finite count;
- named geometry-region reductions;
- toroidal-mode residual power using the verified mapping `n=5k`.

Let

\[
s=\max\left|\mathrm{Vort}_{\mathrm{stored}}\right|.
\]

Each frame passes only if it contains no non-finite value and

\[
\max
\left|
\mathrm{Vort}_{\mathrm{forward}}
-
\mathrm{Vort}_{\mathrm{stored}}
\right|
\le
5\times10^{-10}
+
5\times10^{-10}s.
\]

All five frames must pass. Correlation, an additive alignment, a regional
subset, or an RMS value cannot override the maximum-error gate. The tolerance
is frozen before forward source values are computed.

## 8. Decision rules

- **Pass:** the selected-frame potential/vorticity closure is bidirectionally
  validated for the retained-boundary, runtime-pressure contract. This may
  support a separately frozen all-frame extension and state-candidate
  decision. It does not authorize training automatically.
- **Fail:** retain the failed artifact, localize the matrix/boundary/FFT
  discrepancy, and do not select a new state or train a codec.
- **Either outcome:** do not access 85606, alter the Phase 1 split, select an
  architecture, change channels, or reinterpret five frames as independent
  physical simulations.

No post-hoc materiality label is part of this gate. Passing five selected
frames does not establish all-frame stability.

## 9. Runtime and provenance requirements

The execution must be CPU-only on Rocky 9, use four MPI ranks, request at most
20 minutes, require a clean exact Paper 0 commit, and write to a new
job-specific directory. The launcher must verify all local protocol,
manifest, driver, comparator, external-source, ABI, canonical-input, and
predecessor-result hashes before execution. Existing result directories are
read-only and may not be overwritten.
