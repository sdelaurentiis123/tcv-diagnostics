# Phase 2 shifted-`DDY` oracle readout

## Result

The Paper 0 NumPy shifted-field-line `DDY` primitive matches the exact compiled
BOUT++ 5.2.1 operator for its declared guard-independent scope on the 85604
geometry. Rocky 9 job `6891059` completed with exit code `0:0` from clean
commit `0223035`.

| Frozen case | Largest absolute error | Passed every region? |
|---|---:|:---:|
| Constant | `0` | yes |
| Toroidal mode | `1.918e-12` | yes |
| `y`-coded topology probe | `1.776e-15` | yes |
| Mixed `x-y-z` field | `3.025e-12` | yes |

The acceptance rule, committed before execution, was

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference
```

All four manufactured inputs passed an independent non-collapse check. Every
comparison was finite. Each case passed over all valid cells, ordinary
sequential stencils, the private-flux connections, the twisted core branch,
and the open SOL.

## What this proves

For native 81-cell toroidal fields with `zperiod=5`, the implementation
reproduces BOUT++'s Fourier shifts, centered `C2` derivative, metric division,
single-null neighbor map, private-flux connection, and core branch-shift signs
away from target-boundary cells whose physical guards are not present in the
model tensor.

The result removes a concrete implementation uncertainty. It means the
shifted derivative can be used as a validated primitive in the next exact
transport step.

## What this does not prove

This is not yet a particle-flux or heat-flux result. It did not read a plasma
state frame, evaluate a learned model, use `85606`, or test a diagnostic. The
following remain open:

1. Fromm reconstruction with positivity clipping for the shifted-`xy` face
   term.
2. The combined `xz + xy` conservative radial face flow.
3. Native-grid agreement on selected 85604 plasma frames.
4. Native-81 versus resampled-88 truth sensitivity.
5. Geometry-region integrations, outward orientation, physical units, and
   member-wise nonlinear ensemble evaluation.

Therefore the scientific status is narrowly
`accepted_shifted_ddy_stage_only`; the full transport gate remains closed.

## Immutable evidence

The compact record is
`paper0/results/phase2_shifted_ddy_6891059.json`. The full comparison JSON and
arrays remain on Rusty under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_shifted_ddy/job_6891059`
with hashes recorded in the compact result.
