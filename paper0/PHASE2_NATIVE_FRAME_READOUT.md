# Phase 2 native-81 frame-oracle readout

## Bottom line

The source-matched radial-flow operator passed on every selected real 85604
state, but the prospectively frozen overall job correctly exited nonzero because
the five-channel temperature-to-pressure closure failed at exactly one point.

| Frozen subgate | Result | Decisive evidence |
|---|:---:|---|
| Canonical extraction | pass | 256/256 rank coordinates; five fixed frames; native `z=81`; no held-out access |
| Compiled `Ne` flow | pass | all five frames and every quantity/region |
| Compiled `Pe` flow | pass | all five frames and every quantity/region |
| Compiled `Pi` flow | pass | all five frames and every quantity/region |
| `Ni = Ne` | pass | worst error `1.998e-15` |
| `Pe = Ne * Te` | pass | worst error `4.441e-15` |
| `Pi = Ni * Ti = Ne * Ti` over all stored cells | **fail** | one point at frame 312; error `5.800e-5` |
| Frozen overall gate | **fail** | closure was required; no tolerance or region changed |

Rocky 9 job `6891379` therefore ended `FAILED (1:0)` by design from clean
commit `7d5522c`. The compiled four-rank operator step itself completed `0:0`.

## The positive result: the numerical operator is right on real states

There were 15 real-state cases: five frame indices
`[0, 156, 312, 467, 623]` times three directly archived advected variables
`[Ne, Pe, Pi]`. Every case passed every frozen ordinary, private-flux,
twisted-core, separatrix, and open-SOL comparison.

| Advected field | Largest face-flow absolute error | Largest divergence absolute error | Largest conservation residual |
|---|---:|---:|---:|
| `Ne` | `3.863e-13` | `3.689e-9` | `1.776e-15` |
| `Pe` | `5.226e-13` | `5.988e-9` | `3.553e-15` |
| `Pi` | `6.341e-13` | `6.941e-9` | `3.553e-15` |

The largest face reference scale was `95.016`, so its worst relative error was
`6.674e-15`. The largest divergence reference scale was `79885.997`, so its
worst relative error was `8.689e-14`. Canonical `q` and `phi` were echoed
exactly, native `dz = 2*pi/(5*81)` was exact, and `xz + xy` was exact in both
implementations.

This is much stronger than the manufactured-field result alone: it establishes
that the partial NumPy operator handles the actual 85604 value range on the
selected native frames. It still does not release a surface-integrated or SI
transport metric.

## Why the five-channel closure failed

The only failing point was:

```text
frame = 312, model (x, y, z) = (6, 31, 73)
Pi = -5.799512988032478e-05
Ni =  0.004074141720031353
Ne =  0.0040741417200313495
Ti =  1.2051641668905164e-16
Ni * Ti ~= 4.91e-19
```

This is not random numerical disagreement. The locked Hermes
`EvolvePressure::transform_impl` explicitly computes temperature from
`floor(P, 0)` while the evolved pressure variable retains negative values.
Therefore a negative pressure undershoot is intentionally mapped to an
approximately zero temperature, and no function `N * T` can reconstruct the
negative evolved value.

Across all seven fields, five frames, and 829,440 points per field, this sample
contained one negative `Pi` point and zero negative `Pe` points. All four
closures otherwise agreed near floating-point roundoff.

The point lies at `y=31`, one of the target-dependent rows excluded before this
run from the partial radial-flow comparison. As a post-hoc diagnostic only,
the maximum ion-pressure closure error over the already frozen transport scope
`y=1:31` is `5.329e-15`. That observation does not redefine or pass the frozen
full-domain closure gate.

## What this changes about the emulator state

The current five fields are physically interpretable, but they are not exactly
the evolved Hermes state:

- `Te` and `Ti` are floor-derived outputs of evolved `Pe` and `Pi`;
- `phi` is derived from evolved vorticity;
- `Vi` is derived from ion momentum and density;
- `Ne` is directly evolved.

For the perpendicular ExB transport evaluated here, `Ne`, `phi`, and pressure
are the necessary variables; `Vi` does not enter this operator. The five fields
reconstruct pressure almost everywhere in this sample, but not at retained
negative-pressure undershoots. A transport-faithful emulator therefore needs a
deliberate state policy, not an implicit assumption:

1. emulate the evolved pressures `Pe` and `Pi` directly;
2. emulate physical nonnegative temperatures and explicitly define that
   numerical negative-pressure undershoots are outside the target state; or
3. include enough evolved and derived variables to measure the discrepancy.

No choice is made silently here. The next safe analysis is to count and locate
negative pressure undershoots across all 624 frames of 85604, then freeze the
Paper 0 state variables with that evidence and simulator-owner input.

## Scope and evidence

No learned model, resampled-88 data, diagnostic, or held-out state was used.
The compact record is
`paper0/results/phase2_hermes_native_frames_6891379.json`. The immutable
canonical fields, extraction record, BOUT outputs, full comparison, and arrays
remain under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_native_frames/job_6891379`,
with every digest recorded in the compact result.
