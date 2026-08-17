# Phase 2 all-frame potential/vorticity closure readout

**Result:** pass

**Development run:** 85604 only

**Held-out 85606 read:** no

**Training performed:** no

**Rocky 9 job:** `6893033`

**Execution commit:** `d3c73231e2a2d0cf49fd3d0c87a8155a3cc20d75`

## Executive result

The executed Hermes/BOUT++ potential-vorticity relation closes in both
directions on every one of the 624 saved 85604 frames. Given the evolved
pressures, stored generalized vorticity, fixed geometry, and compact retained
radial-potential boundary state, the exact elliptic solve reconstructs stored
potential. Conversely, applying the exact executed cyclic matrix to stored
potential and runtime pressure reconstructs stored vorticity.

This is a source-state identity, not a forecast result. It resolves which
variables form the faithful saved simulator state, but it does not show that a
codec preserves that state or that a learned model can predict it.

## What was evaluated

The audit covered:

- frame indices `0..623` exactly once;
- normalized times `285000..471900` at cadence `300`;
- physical cadence `3.131905426352636 microseconds`;
- native physical shape `64 x 32 x 81`;
- `103,514,112` values per volumetric field;
- all native Fourier indices `k=0..40`, labeled by `n=5k` because
  `zperiod=5`;
- all authoritative named geometry regions;
- exact runtime-pressure publication, retained radial-boundary construction,
  compiled known answers, input echoes, and the source forward operator.

The raw 256-rank archive was read exactly once. One immutable rank file at a
time was staged into job-owned node-local storage, semantically checked, and
discarded before the next rank was staged. No source or accepted result was
overwritten.

## Ordered gate result

| Gate | Result |
|---|---:|
| Provenance and complete extraction | pass |
| Compiled constant, gauge, and manufactured known answers | pass |
| Bitwise input echo and runtime pressure | pass |
| Source forward closure on every frame | pass |
| Exact eight-shard merge and complete frame coverage | pass |

Runtime `Pe` and `Pi` match their independently calculated source formulas
with zero discrepancy at every physical point. The extraction also reproduces
the frozen raw-pressure identity: zero negative raw-`Pe` values and 3,412
negative raw-`Pi` values, distributed across the eight blocks as
`[0,116,1812,86,67,69,1262,0]`.

## Numerical closure

| Quantity | All-frame value |
|---|---:|
| Pooled relative L2 error | `6.502783244122983e-13` |
| Pooled RMSE | `2.5173334679190532e-14` |
| Pooled bias | `-4.5399881994687245e-17` |
| Pooled correlation | `0.9999999999999998` |
| Maximum absolute error | `6.105671523926048e-13` |
| Maximum-error location | frame `169`, `(x,y,z)=(25,19,69)` |
| Non-finite values | `0` |

The worst scale-aware gate fraction occurs at frame `494`:

```text
maximum absolute error:  5.593858709573851e-13
frozen frame tolerance:  7.008817082703721e-10
error / tolerance:       0.0007981173775212814
```

Thus even the worst frame uses only about `0.0798%` of its permitted error;
the result is more than 1,200 times inside the frozen gate.

## Geometry-region closure

| Region | Relative L2 | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| Confined edge | `4.0393e-13` | `2.4379e-14` | `5.5218e-13` |
| Private flux | `9.9812e-13` | `1.1131e-14` | `4.5519e-13` |
| Scrape-off layer | `7.3191e-13` | `2.8234e-14` | `6.1057e-13` |
| Separatrix cell band | `7.4295e-13` | `4.8153e-14` | `5.8537e-13` |
| Outboard midplane | `6.0073e-13` | `5.8090e-14` | `5.7342e-13` |
| X-point topology stencil | `1.6952e-12` | `8.4143e-14` | `4.9267e-13` |
| Inner divertor leg | `1.8804e-12` | `2.0970e-14` | `5.3558e-13` |
| Outer divertor leg | `1.2726e-12` | `2.1056e-14` | `5.0020e-13` |

Every region passes. The larger relative values in some low-amplitude regions
remain at numerical-roundoff scale in absolute units.

All Fourier indices also close. In the previously discussed band
`k=4..7`, corresponding to full-torus `n=20..35`, pooled relative residual
power ranges from approximately `6.11e-27` to `8.12e-26`. The largest value
over any retained mode is `4.42e-23` at `k=36`, `n=180`; this is still
numerical closure, not lost physical mode power.

## State consequence

The exact saved source-state candidate is now

```text
S6+Bphi = [Ne, Pe, Pi, NVe, NVi, Vort]
          + inner/outer retained phi midpoint over y.
```

Fixed geometry, fixed source configuration, and the executed elliptic operator
then determine interior `phi`. Therefore `phi` and `Vort` are not independent
volumetric state channels once pressure and the compact boundary state are
known.

This does not automatically make `S6+Bphi` the best neural-network interface.
Two representations remain worth a matched predictive test:

1. the exact source-state candidate above; and
2. one pragmatic history-conditioned observable state that includes `phi`
   directly and asks whether short history predicts the omitted state well
   enough.

The exact state has cleaner source semantics but requires a mixed
volume-plus-boundary representation and an elliptic reconstruction for
transport. The pragmatic state is easier to feed into existing model code but
must demonstrate one-step sufficiency rather than assume it.

## What is not established

This result does not establish:

- stationarity of the 85604 interval;
- predictive sufficiency of one saved frame;
- codec reconstruction fidelity;
- one-step or rollout accuracy;
- stochastic calibration;
- transport preservation by a learned model;
- diagnostic assimilation or ranking;
- any conclusion on 85606;
- any automatic channel change or authorization to train.

## Artifacts

The compact tracked result is
`paper0/results/phase2_potential_vorticity_all_frame_6893033.json`.

The complete 4,604,085-byte per-frame and per-mode result remains at

```text
/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_potential_vorticity_all_frame/job_6893033/potential_vorticity_all_frame_comparison.json
```

with SHA-256
`407d6a46387e22c0af8f279e2292974d2aa9f73394cec02005c8cc026ec60cfc`.
Every top-level and nested shard hash inventory was independently rechecked
after job completion.

## Next gate

Commit a separate state-candidate decision, then freeze a matched O1/O2
protocol. That protocol must compare exact source state against one pragmatic
history-conditioned state under identical data, normalization, codec budget,
deterministic backbone, and checkpoint selection. No stochastic architecture
should be selected until this representation and one-step-sufficiency ladder
is complete.
