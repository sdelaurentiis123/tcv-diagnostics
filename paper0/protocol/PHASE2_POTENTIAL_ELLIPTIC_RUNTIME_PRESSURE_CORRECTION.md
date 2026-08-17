# Phase 2 potential-elliptic runtime-pressure correction protocol

**Status:** frozen after failed job `6892446` and before the first corrected
runtime-pressure replay

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited

**Purpose:** correct one source-contract error in the first paired
potential-elliptic replay while preserving its frames, canonical inputs,
solver, boundaries, reconstruction tolerance, comparison metrics, and
decision rules.

The machine-readable authority is
`paper0/manifests/phase2_potential_elliptic_runtime_pressure_correction.json`.
This is a deterministic source-reconstruction correction. It does not train or
select a model.

## 1. Predecessor result and failure

The original frozen protocol and manifest treated the raw evolved pressure
arrays `Pe` and `Pi` as the species pressures consumed by
`Vorticity::calculatePihat`. Job `6892446` tested that assumption without
relaxing its gate.

All volume and radial-boundary echoes were bitwise exact. Four of five frames
reconstructed saved `phi` to maximum absolute error below `2.75e-13`. Frame
312 also passed over the entire predeclared `y=1..30` transport interior. The
full-domain gate failed at exactly one point:

```text
frame = 312
(x, y, z) = (6, 31, 73)
raw evolved Pi = -5.799512988032478e-05
replay minus stored phi = +5.7995129900123565e-05
```

The unchanged comparator therefore blocked every retained-versus-instantaneous
boundary-effect metric. That result remains a failed result. This correction
does not relabel it as a pass.

## 2. Source-backed correction

Hermes `EvolvePressure::transform_impl` does not publish raw evolved pressure
directly to downstream components. For each charged species it computes

\[
P_s^{\mathrm{runtime}}
=
N_s
\frac{\max(P_s^{\mathrm{evolved}},0)}
{\operatorname{softFloor}(N_s,10^{-7})},
\]

where

\[
\operatorname{softFloor}(N,f)
=
\max(N,0)+f\exp\!\left[-\frac{\max(N,0)}{f}\right].
\]

The vorticity component then constructs

\[
\widehat P_i
=
P_i^{\mathrm{runtime}}
-
\frac{P_e^{\mathrm{runtime}}}{3672}.
\]

For this quasineutral run, `Ni = Ne`. The completed all-frame state audit found
minimum `Ne = 4.190229129105658e-05`, no value at or below the `1e-7` density
floor, and zero points changed in floating representation by the density soft
floor. The corrected driver must nevertheless evaluate the exact source
formula rather than replace it with an undocumented simplification.

The correction is only the transformation from raw evolved `Pe/Pi` to runtime
species pressure. Direct evolved `Pe/Pi` remain the archived state and remain
the transport-evaluation quantities defined by the predecessor protocol.

## 3. Immutable predecessor and source locks

The correction manifest locks by SHA-256:

- the original frozen protocol and manifest;
- the exact failed result from job `6892446`;
- the earlier selected-frame pressure finding;
- the completed all-frame pressure and density audits;
- Hermes `src/evolve_pressure.cxx`, `include/hermes_utils.hxx`,
  `src/vorticity.cxx`, and `include/vorticity.hxx`;
- the exact BOUT++ cyclic solver revision and installed shared library.

No source fact learned from the corrected replay may be inserted back into
these predecessor artifacts.

## 4. Canonical artifact reuse

The corrected job reads, without modifying, the canonical selected-frame file
and extraction record created by job `6892446`:

```text
canonical SHA-256
e090b3a23fa6eedf8c37e74421c08bafd3eb513039fa7621b5d612a7e1cbba3e

extraction-record SHA-256
e30e4f14dddfdff369387f9e8657b31a3e56bb4dc628c1e9ef5d86bd5bfd68be
```

This reuse is valid because the correction changes only the deterministic
transformation applied after `Ne`, `Pe`, and `Pi` are read. The canonical
arrays, geometry, saved boundary midpoints, instantaneous targets, frame
indices, and raw input hashes do not change. The corrected launcher must verify
both hashes and refuse to write into the predecessor job directory.

## 5. Frozen corrected-replay gates

The corrected driver must write raw input echoes plus
`Pe_runtime`, `Pi_runtime`, and `Pi_hat`. An independent Python finalizer must
recompute the runtime-pressure formula from canonical `Ne`, `Pe`, and `Pi`.
For every selected frame and full physical point, each runtime field must be
finite and satisfy the already frozen native closure rule

```text
max_abs_error <= 1e-12 + 1e-12 * max_abs_reference.
```

The expected selected-frame support, known before this correction, is zero
negative raw-`Pe` points and exactly one negative raw-`Pi` point at the
location above. The correction fails closed if that support changes or if the
compiled runtime `Pi` is nonzero at that point.

Only after the runtime-pressure gate passes does the unchanged source
reconstruction gate apply:

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference.
```

It still requires every selected frame to pass both the full physical domain
and `y=1..30` interior, with raw rather than gauge-aligned error. A
constant-shift diagnostic cannot change the decision.

## 6. Quantities held fixed

The corrected replay must retain:

- frames `[0, 156, 312, 467, 623]`;
- native `z=81` and `zperiod=5`;
- four MPI ranks with `NXPE=1`, `NYPE=4`, `MYSUB=8`;
- exact Hermes metric normalization;
- BOUT++ `5.2.1` cyclic inversion;
- `C=2/Bxy^2`, right-hand side `Vort*Bxy^2/2`, and `INVERT_SET` radial
  boundaries;
- the saved-midpoint retained arm and fixed-target instantaneous arm;
- all potential, geometry-region, direct-pressure transport, SI conversion,
  and no-post-hoc-materiality definitions from the predecessor protocol.

Changing any item above would create a different experiment.

## 7. Interpretation

- If runtime-pressure transformation fails, stop and do not interpret either
  solve.
- If transformation passes but source reconstruction fails, retain the
  failure and continue source/dump diagnosis. Do not relax the tolerance.
- If both gates pass, the retained source replay is validated on the five
  selected 85604 frames and the unchanged paired boundary effect may be
  computed.
- Any paired effect is a selected-frame deterministic effect size, not an
  all-frame stability result or a material/negligible label.
- No outcome changes the model state, selects an architecture, starts
  training, or authorizes access to 85606 automatically.

## 8. Execution

Run CPU-only on Rocky 9 from a clean committed checkout. Use a new immutable
job directory. Preserve the base-comparator output, corrected final result,
raw arrays, four rank outputs, exact commands, module list, Slurm record, and
artifact digests. A Slurm nonzero exit caused by a scientific gate must be
reported as a gate failure, not silently rewritten as an execution success.

