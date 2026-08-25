# Old-85604 Stage-2 multi-lead fine-tuning protocol

**Frozen:** 2026-08-25, after completion of the exact-state
derived-coordinate screen and before evaluating any Stage-1 parent checkpoint
at leads greater than one

**Development simulation:** 85604 only

**Held-out 85606:** unopened and prohibited

## Question

Can a literature-motivated multi-lead curriculum improve the passing
codec-free C5P transition at longer saved-frame leads without materially
degrading its validated one-step transition?

This is Stage 2 of
`POST_ECRD_STATE_DATA_SCALING_PROTOCOL.md`. It changes the temporal pair
distribution and fine-tuning schedule, not the state view or architecture. It
is not a GAOT, FNO, stochastic-emulator, transport, rollout, assimilation, or
diagnostic-ranking result.

## Motivation and literature boundary

The passing Stage-1 C5P operator already implements two pieces used by the
GAOT time-dependent formulation:

1. it predicts the standardized forward-difference derivative
   `(x[t+lead] - x[t]) / lead`;
2. lead time conditions every residual block.

The 2026 drift-wave study
[`arXiv:2603.05730`](https://arxiv.org/html/2603.05730v1) additionally trains
over all pairs up to a maximum lead and then fine-tunes on longer temporal
windows. That study had multiple independent simulations and substantially
more data. Pairing frames in this one 85604 trajectory increases supervised
constraints but does not create independent turbulence realizations.

This screen tests only the multi-lead training mechanism. The existing local
U-Net remains the processor so a result cannot be attributed to the
Geometry-Aware Operator Transformer architecture described in
[`arXiv:2505.18781`](https://arxiv.org/abs/2505.18781).

## Frozen evidence and parent

The Stage-1 chronological reduction is locked by path and SHA-256. Its
prospective decision was:

```text
retain_c5p_control_and_e6b_as_unresolved_exact_state_ablation
```

Consequently, this one-seed screen uses only the passing C5P performance
control. It does not reinterpret or rerun E6B.

Parent:

- family: C5P (`Ne, Pe, Pi, phi, Vi`);
- seed: 1701;
- one-frame history;
- selected Stage-1 epoch: 12;
- one-step shared `Ne/Pe/Pi` validation MSE:
  `0.005322341561633884`;
- checkpoint SHA-256:
  `887dfcbe37e214f7217a5d4b900381cea370ca2e2c96687d2d6cd92c9e951c33`.

The Stage-2 job must load the parent state dictionary strictly and verify
bitwise equality before evaluation or optimization. It resets AdamW rather
than restoring the Stage-1 optimizer.

## Immutable data contract

- training frames: `[0,432)`;
- guard frames: `[432,496)`, never read;
- chronological validation frames: `[496,624)`;
- normalization: the existing training-only C5P normalization;
- history: one frame;
- lead steps: `[1,2,4,8,16]` saved frames;
- random shared circular toroidal roll in training only;
- no random split or use of adjacent pairs as independent physical shots.

Pair counts are fixed:

| lead | training pairs | validation pairs |
|---:|---:|---:|
| 1 | 431 | 127 |
| 2 | 430 | 126 |
| 4 | 428 | 124 |
| 8 | 424 | 120 |
| 16 | 416 | 112 |
| **all** | **2,129** | **609** |

The training derivative RMS is refit once from all 2,129 multi-lead training
pairs. It is a numerical component scale only and is not a physics quantity.

## Parent baseline at multiple leads

Before the first optimizer update, evaluate the bitwise Stage-1 parent over
the complete frozen validation pair set separately at every lead. Record raw
standardized derivative MSE, zero-derivative persistence MSE, and
persistence-relative skill by field.

No parent multi-lead result has been inspected before this protocol was
frozen. The parent scores are baselines, not thresholds chosen after seeing
them.

## Architecture and fine-tuning

Use the unchanged `CodecFreeIncrementOperator3D` C5P architecture:

- base width 24;
- channel multipliers `[1,2,4]`;
- two residual blocks per level;
- 128-channel log-lead embedding;
- circular padding only in the toroidal axis;
- no toroidal stride or absolute toroidal coordinate;
- direct joint five-field derivative output;
- no codec and no boundary head.

Fine-tuning schedule:

- seed: 1701;
- epochs: 4;
- sample batch size: 1;
- gradient accumulation: 4 samples;
- expected optimizer updates: `ceil(2129/4) * 4 = 2,132`;
- AdamW with peak learning rate `5e-5`, minimum `5e-6`, weight decay
  `1e-4`;
- 5% linear warmup followed by cosine decay;
- gradient norm clipping at 1.0;
- bfloat16 autocast; TF32 disabled;
- required online W&B tracking;
- one right-sized Rusty GPU job.

The loss is persistence-normalized direct field derivative MSE. Flux,
spectra, cross-phase, coherence, conservation, PDE residuals, blob labels,
and all other physics-derived quantities remain outside the loss.

## Evaluation and checkpoint selection

Evaluate every epoch separately at each frozen lead. For lead `l`, define

```text
q_l = shared_model_derivative_MSE_l /
      shared_zero_derivative_persistence_MSE_l.
```

Select the checkpoint minimizing the unweighted mean of `q_l` over
`l in {1,2,4,8,16}`. This gives every lead one vote rather than allowing the
lead with the largest raw derivative variance to dominate.

For the selected checkpoint report:

- every per-lead, per-field MSE and persistence-relative skill;
- parent-to-fine-tuned change at each lead;
- mean `q_l` and parent-relative improvement;
- one-step change relative to the frozen Stage-1 metric;
- exact checkpoint reload and integer toroidal-shift equivariance;
- update count, wall time, GPU memory, hashes, command, environment, and W&B
  finished state.

## Prospective screen gates

The screen advances only if all conditions hold:

1. all mechanical, finite-metric, exact-update, checkpoint-reload, and
   toroidal-equivariance gates pass;
2. every C5P field has positive persistence-relative skill at every lead;
3. selected-checkpoint lead-1 shared MSE is no more than 5% above the frozen
   parent value, hence no greater than `0.005588458639715578`;
4. the selected checkpoint improves mean multi-lead `q_l` by at least 10%
   relative to the bitwise parent evaluated in the same job;
5. at least three of the four longer leads (`2,4,8,16`) strictly improve
   shared derivative MSE relative to the parent.

If the screen passes, a dated amendment may authorize seeds 1702 and 1703,
chronological-block evaluation, and bounded four/eight-frame autoregressive
evaluation. If it fails, do not tune the lead set, learning rate, or epoch
count against this result. Proceed to a separately frozen operator-architecture
experiment.

## Scope boundary

This screen does not authorize:

- E6B or another state view;
- an official or adapted GAOT claim;
- stochastic generation or ensemble calibration;
- free-rollout transport claims;
- assimilation, diagnostic ranking, or steering;
- inventory, metadata access, preprocessing, or evaluation of 85606;
- access to the newer NERSC dataset.
