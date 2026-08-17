# Phase 2 shifted-`xy` radial-face oracle readout

## Result

The Paper 0 NumPy shifted-poloidal radial face component matches the
hash-locked Hermes-3 source for its declared guard-independent scope. Rocky 9
job `6891343` completed with exit code `0:0` from clean commit `ee2b04f`.

| Frozen case | Largest continuous absolute error | Positive / negative velocity points | Selected clipped states | Passed every quantity and region? |
|---|---:|---:|---:|:---:|
| Constant | `0` | `0 / 0` | `0` | yes |
| Smooth | `2.562e-12` | `74,138 / 74,092` | `0` | yes |
| Signed velocity | `1.329e-12` | `73,888 / 74,342` | `0` | yes |
| Positivity clipping | `1.435e-12` | `74,073 / 74,157` | `85,830` | yes |

The acceptance rule was committed before execution:

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference
```

Velocity, selected Fromm state, and face flow passed over all valid points,
ordinary stencils, both private-flux connections, both twisted core
connections, the separatrix radial face, and open SOL. All binary clipping
decisions matched exactly. No compared value was non-finite, every
nonconstant case exercised both velocity signs, and the clipping case selected
both clipped and unclipped states.

## What this proves

Within the model's available radial and target-independent stencil, the NumPy
implementation reproduces the executed Hermes revision's shifted-`DDY`
geometry factor, radial-face metric placement, sign-dependent four-cell Fromm
state, positivity clipping, and face-flow product on the native 81-cell,
one-fifth-torus grid (`zperiod=5`).

## What this does not prove

This is one component, not a released transport metric. The job read geometry
and manufactured fields only; it did not read a plasma-state frame, evaluate a
learned model, access 85606, or test diagnostics. Paper 0 still needs:

1. combined `xz + xy` face-flow conservation;
2. total particle and advected-internal-energy definitions;
3. a native-81 oracle on selected 85604 plasma frames;
4. native-81 versus resampled-88 sensitivity;
5. geometry-region integration and outward orientation;
6. unit conversion and member-wise nonlinear ensemble semantics.

The scientific status is therefore
`accepted_shifted_xy_face_stage_only`; the full transport gate remains closed.

## Immutable evidence

The compact record is
`paper0/results/phase2_hermes_xy_face_6891343.json`. Full arrays and comparison
JSON remain on Rusty under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_xy_face/job_6891343`,
with hashes recorded in the compact result.
