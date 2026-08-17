# Phase 2 combined radial-flow oracle readout

## Result

The Paper 0 NumPy sum of the radial `xz` and shifted-`xy` face terms, and its
finite-volume divergence, match the hash-locked Hermes-3 source for their
declared guard-independent scope. Rocky 9 job `6891373` completed with exit
code `0:0` from clean commit `b6926ca`.

| Frozen case | Largest face-flow absolute error | Largest divergence absolute error | Divergence reference scale | `xz` positive / negative | Total positive / negative | Passed every quantity and region? |
|---|---:|---:|---:|---:|---:|:---:|
| Constant | `0` | `0` | `0` | `0 / 0` | `0 / 0` | yes |
| Smooth | `2.562e-12` | `3.073e-8` | `3.680e6` | `74,068 / 74,162` | `74,071 / 74,159` | yes |
| Signed velocity | `1.329e-12` | `1.834e-8` | `3.117e6` | `74,220 / 74,010` | `74,213 / 74,017` | yes |
| Positivity clipping | `1.389e-12` | `2.186e-8` | `3.386e6` | `74,326 / 73,904` | `74,322 / 73,908` | yes |

The continuous acceptance rule was committed before execution:

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference
```

The largest absolute divergence discrepancy looks numerically larger than the
face-flow discrepancy because divergence divides a face difference by the
small cell-volume factor `J*dx`. Its reference magnitude was approximately
`3.680e6`, making the worst relative discrepancy `8.351e-15`. This is near
floating-point roundoff and well within the frozen tolerance of approximately
`1.840e-3` for that comparison.

Native `dz` matched `2*pi/(5*81)` exactly. Reference and NumPy total flow both
equaled `xz + xy` exactly at all 148,230 valid points per case. Reconstructing
the face difference from divergence times `J*dx` had a worst residual of
`1.137e-13`. Every nonconstant case contained both flow signs; no compared
value was non-finite.

## What this proves

On manufactured fields at the native 81-cell, one-fifth-torus resolution
(`zperiod=5`), the independent NumPy implementation reproduces the executed
Hermes revision's two radial ExB face terms, their sum, and the conservative
radial divergence on ordinary, private-flux, twisted-core, separatrix, and
open-SOL regions covered by the model state.

The accepted APIs are still named `radial_exb_face_flow_partial` and
`divergence_from_radial_face_flow_partial`. The suffix is a scientific scope
marker, not cosmetic naming.

## What this does not prove

This is a validated numerical operator component, not yet a released particle-
or heat-transport metric. The job read geometry and manufactured fields only;
it did not read a plasma-state frame, evaluate a learned model, access 85606,
or test diagnostics. Paper 0 still needs:

1. authoritative particle and advected-internal-energy field definitions;
2. a native-81 oracle on selected 85604 plasma frames;
3. native-81 versus Fourier-resampled-88 sensitivity;
4. geometry-region integration and outward orientation;
5. normalized-to-SI conversion;
6. member-wise nonlinear ensemble semantics.

The scientific status is therefore
`accepted_combined_radial_flow_stage_only`; the full transport gate remains
closed.

## Immutable evidence

The compact record is
`paper0/results/phase2_hermes_radial_flow_6891373.json`. Full arrays and the
complete comparison JSON remain on Rusty under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_radial_flow/job_6891373`,
with hashes recorded in the compact result.
