# Phase 3 B5 joint field-residual EDM smoke readout

## Bottom line

Rocky 9 H100 job `6901469` passed the frozen B5 implementation gate at exact
commit `8cd5b3465dcc7092b9f5a50cfdccca39dcb17c07`. The complete job took
`00:01:29`, including repository tests, input staging, training, sampling,
W&B synchronization, and post-run hashing. The model and sampler portion took
`8.9542` seconds and allocated only `1.2504 GiB` at peak.

This establishes that the proposed full-field residual architecture is
mechanically viable. It is not evidence that the model generalizes, improves
forecast skill, preserves transport, or is calibrated.

## What was actually exercised

The model is an `11,604,709`-parameter conditional 3-D U-Net in decoded
standardized field space. It predicts one joint residual over
`[Ne,Pe,Pi,phi,Vi]`, conditioned on the previous complete five-field state and
the frozen deterministic H1 mean. It uses static `x/y` position channels,
circular operations in stored toroidal `z`, and no time, shot label,
diagnostic, region mask, physics-derived target, or future truth input.

The bounded smoke used eight 85604 training targets and exactly 64 optimizer
steps. No guard or validation frame was read. The H1 forecast and residual
audit inputs were hash checked before staging and rechecked after the smoke.
The full Rusty-side test suite passed with `1047 passed`, `1 skipped`, and
`29 subtests passed`.

## Mechanical results

| gate | observed | result |
|---|---:|---|
| optimizer steps | 64 | pass |
| initial fixed-probe EDM loss | 0.9942206 | reference |
| final fixed-probe EDM loss | 0.9706594 | pass |
| fixed-probe relative change | -2.3698% | pass |
| peak allocated CUDA memory | 1.2504 GiB | pass |
| checkpoint reload | bitwise exact | pass |
| periodic shift tested | 8 cells | pass |
| maximum shift-equivariance error | 9.5367e-7 | pass |
| sampled field axes | `[1,2,1,5,64,32,88]` | pass |
| sampler calls per member | 35 | pass |
| sampled members | finite and distinct | pass |
| online W&B state | finished | pass |

The fixed-probe decrease is deliberately a weak mechanics gate. With only
eight adjacent targets and 64 updates, it says that gradients and parameter
updates behave coherently. It does not quantify out-of-sample denoising or
forecast quality.

The sampled members differed by `1.5199` RMS in normalized residual space and
`0.10032` RMS after residual scaling and composition with the standardized H1
mean. This proves that the sampler has not collapsed mechanically. Because no
validation truth was read, these numbers are not a spread-skill or calibration
result.

## Why this branch remains worth testing

The job-6901393 audit found a model error that is jointly structured across
fields, strongly position dependent, and rich in non-axisymmetric toroidal
content. A full-field joint residual generator directly represents those
properties without relying on the DCAE bottleneck that constrained B2/B3/B4.
The smoke now removes compute feasibility and tensor mechanics as immediate
objections: the architecture fits comfortably on one H100, trains rapidly,
respects the periodic direction, reloads exactly, and produces a canonical
ensemble.

The central risk is statistical rather than computational. Paper 0 contains
only 430 adjacent training targets from one simulated run, not tens of
thousands of independent examples. The residual audit also measured model
error, not irreducible aleatoric noise. A flexible diffusion model can
therefore memorize conditional residuals or manufacture spread without
learning the correct conditional joint distribution.

## What must happen next

The next permitted action is to freeze a separate 85604-only full-training and
validation protocol. That protocol must prospectively define the optimizer
budget, checkpoint rule, validation noise banks, ensemble size, one-step and
autonomous forecast products, and acceptance thresholds before validation is
read. The first scientific question is whether B5 improves validation
probabilistic forecasts and joint transport-relevant behavior over the frozen
H1 mean and prior probabilistic baselines. Only after that question passes may
the work advance toward the oracle ladder, 85606, assimilation, or diagnostic
ranking.

The compact tracked record is
`paper0/results/phase3_b5_field_residual_edm_smoke_6901469.json`. The complete
Ceph result has SHA-256
`956f3490d115461ad294a6023c39a65f522fd73db6b47ce58633790ce77c9322`.
