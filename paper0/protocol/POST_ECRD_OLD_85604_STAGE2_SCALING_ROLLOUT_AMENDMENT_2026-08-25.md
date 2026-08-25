# Old-85604 Stage-2 seed-scaling and bounded-rollout amendment

**Frozen:** 2026-08-25 after seed-1701 job `6936393` completed and before any
seed-1702/1703 multi-lead optimizer update or bounded rollout evaluation

**Development simulation:** 85604 only

**Held-out 85606 and newer NERSC data:** unopened and prohibited

## Motivation and immutable evidence

The one-seed screen prospectively authorized scaling only if every mechanical
and scientific gate passed. Job `6936393` passed. Its immutable result is:

```text
paper0/results/post_ecrd_old_85604_stage2_multilead_6936393.json
SHA-256 6f4036ff6fd50a7090e60a351242f1a6ad00af6d3762322fe1075d22a9808c2a
```

The seed-1701 result is not repeated. This amendment authorizes exactly two
confirmation runs, initialized from the already selected Stage-1 C5P parents:

| Seed | Parent one-step metric | Parent checkpoint SHA-256 |
|---:|---:|---|
| 1702 | `0.005284142434913455` | `59e9927ca88878a9d31a72789c6bbaf03248c507bc87f18ce2ac77e2026ea4a6` |
| 1703 | `0.005365789313055087` | `26e369f2114e56997a11a57e8233109aa501d82bf35f4f3ac632435ce2889b18` |

Every result and checkpoint path is locked in the machine-readable scaling
manifest. A run must load the corresponding parent model bitwise and evaluate
its five-lead baseline before constructing its optimizer.

## Seed-confirmation training

For seeds 1702 and 1703, preserve the seed-1701 screen exactly:

- C5P fields `Ne, Pe, Pi, phi, Vi`;
- training `[0,432)`, unread guard `[432,496)`, validation `[496,624)`;
- one-frame history and leads `1,2,4,8,16`;
- the same 2,129 training and 609 validation pairs;
- the unchanged 2,174,021-parameter codec-free local operator;
- strict initialization from the matching Stage-1 parent weights only;
- a fresh AdamW optimizer, four epochs, gradient accumulation four, and
  exactly 2,132 optimizer updates;
- peak/minimum learning rates `5e-5`/`5e-6`, weight decay `1e-4`, five-percent
  warmup, cosine decay, clipping at 1.0, bfloat16, and TF32 disabled;
- the same training-only derivative-RMS field loss and unweighted mean
  persistence-normalized five-lead checkpoint metric;
- required online W&B and immutable artifact hashes.

The jobs may run as two independent one-GPU array tasks. Each task requests
one generic preemptible GPU, four CPUs, 12 GiB host memory, and one hour. This
is two small jobs, not one multi-GPU allocation.

## Confirmation rule

The three-seed mechanism is confirmed only if seeds 1701, 1702, and 1703 all:

1. pass exact-update, finite-metric, loss-decrease, checkpoint-reload, and
   integer toroidal-equivariance gates;
2. retain positive persistence-relative skill for every C5P field at every
   lead;
3. keep lead-1 shared MSE no more than five percent above their own frozen
   Stage-1 parent metric;
4. improve mean five-lead persistence-normalized error by at least ten
   percent relative to their own bitwise parent evaluated in the same job;
5. improve shared derivative MSE at at least three of the four longer leads.

Report each seed separately plus median, minimum, and maximum. Seed averaging
cannot rescue a failed individual gate. Adjacent temporal pairs are not
reported as independent simulations.

If confirmation fails, do not tune the schedule against the failed seed and
do not run the bounded rollout under this amendment. Proceed to a separately
frozen operator-architecture experiment.

## Conditional bounded-rollout evaluation

If all three seeds confirm, this amendment authorizes inference-only bounded
evaluation on the same chronological 85604 validation interval at terminal
horizons four and eight saved frames. For every eligible start and seed,
compare:

1. persistence;
2. one direct lead-4 or lead-8 prediction;
3. repeated autonomous lead-1 predictions;
4. repeated autonomous lead-2 predictions;
5. for horizon eight, two autonomous lead-4 predictions.

Every autonomous composition feeds the complete predicted five-field state
back as the next model input. It may use no intervening or future truth. All
methods at a terminal horizon use the same starts and targets. This is an
inference comparison; no checkpoint may be selected or tuned on it.

First report standardized per-field RMSE, persistence-relative skill, error
versus composition depth, and seed range. Then decode with the frozen
training-only normalization and run the already validated evaluation-only
spectral, cross-field, and transport diagnostics. Nonlinear quantities must
be computed from each seed forecast before aggregation. No physics quantity
enters training.

The bounded evaluation is a composition/localization test, not evidence of a
long free rollout, stochastic ensemble, calibrated covariance, assimilation,
diagnostic ranking, or steering. A separate prospective protocol is required
before any of those claims or before access to 85606.
