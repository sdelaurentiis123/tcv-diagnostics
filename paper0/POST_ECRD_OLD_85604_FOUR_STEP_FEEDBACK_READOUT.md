# Old-85604 four-step feedback pilot: state gain, transport-gate failure

**Date:** 2026-08-25  
**Training job:** `6937357`  
**Frozen state-and-physics evaluation job:** `6937465`  
**Development simulation:** 85604 only  
**85606 and newer NERSC data:** unopened

## Result in one sentence

Training the codec-free five-field operator on four self-fed lead-one steps
materially improves autonomous field-state error and several local/phase
metrics, but suppresses fluctuation amplitude enough to worsen integrated
separatrix transport; the preregistered physics gate fails and confirmation
seeds are not authorized.

## What changed

The parent and pilot have the same 2,174,021-parameter architecture and use the
same C5P fields (`Ne`, `Pe`, `Pi`, `phi`, and `Vi`). The pilot begins from the
parent weights and changes only the temporal field-training curriculum:

- four consecutive lead-one predictions are made from an initial true state;
- every predicted state becomes the next input;
- gradients are detached through the state passed between steps;
- effective state-MSE step weights are `0.625, 0.125, 0.125, 0.125`;
- no flux, spectrum, phase, coherence, PDE, or conservation quantity enters
  training or checkpoint selection.

Job `6937357` selected epoch 6 after 642 optimizer updates. The selected
checkpoint SHA-256 is
`affe2589f4ce6639879ca1ed4a100af764aa48a475a653987faa18d4ce844117`.

## State gate: passed

The full-validation comparison in the training job used every eligible start
in `[496,624)` and no future truth as context.

| Horizon | Pilot / parent mean-field MSE | Error reduction |
|---:|---:|---:|
| 1 frame | 0.967870 | 3.21% |
| 4 frames | 0.896629 | 10.34% |
| 8 frames | 0.859817 | 14.02% |

The mean four/eight-frame improvement is 12.80%, exceeding the frozen 5%
threshold. Checkpoint reload, exact update count, finite metrics, and integer
toroidal-equivariance checks all passed.

The separate evaluator reproduces the direction of the result:

| Horizon | Parent state skill | Feedback-tuned state skill |
|---:|---:|---:|
| 4 frames | 0.5324 | 0.5807 |
| 8 frames | 0.3015 | 0.3994 |

All five fields improve at both evaluation horizons and retain positive skill
relative to persistence.

## Physics preservation: failed

The physics evaluation was frozen after state training and before any pilot
physics output was inspected. It compares only repeated lead-one rollouts of
the bitwise parent and selected pilot on identical starts and targets.

### Toroidal power

The gate is the median absolute log power-ratio error over five fields and the
three physical bands `n=5–15`, `n=20–25`, and `n=30–35` (`n=5k`).

| Horizon | Parent error | Pilot error | Pilot / parent | Gate |
|---:|---:|---:|---:|:---:|
| 4 frames | 0.7209 | 0.7622 | 1.057 | pass |
| 8 frames | 0.5390 | 0.9272 | 1.720 | **fail** |

At eight frames the pilot often moves the dominant `n=20–25` band closer to
unit power ratio, but further suppresses low- and high-band power. The
aggregate power-distribution error therefore worsens even though mean
truth-transfer coherence rises from 0.0242 to 0.0346.

### Cross-field structure

Across the nine density/pressure–potential pair/band summaries, mean absolute
phase error improves from 4.74° to 4.12° at four frames and from 4.42° to 3.79°
at eight frames. Mean absolute coherence change worsens from 0.118 to 0.127
and from 0.145 to 0.164, respectively. The pilot is better at phase/placement
on average but worse at preserving cross-field amplitude coherence.

### Transport

The frozen transport gate averages separatrix relative-L2 error over particle,
electron internal-energy, ion internal-energy, and total internal-energy
transport.

| Horizon | Parent separatrix error | Pilot separatrix error | Pilot / parent | Gate |
|---:|---:|---:|---:|:---:|
| 4 frames | 0.3951 | 0.4393 | 1.112 | **fail** |
| 8 frames | 0.3870 | 0.4627 | 1.195 | **fail** |

The local and integrated views move in opposite directions:

| Horizon | Mean strict-face L2, parent → pilot | Mean separatrix RMS ratio, parent → pilot | Mean separatrix correlation, parent → pilot |
|---:|---:|---:|---:|
| 4 frames | 1.0047 → 0.9623 | 0.6850 → 0.6259 | 0.6362 → 0.6517 |
| 8 frames | 1.3469 → 1.2579 | 0.7495 → 0.6554 | 0.4537 → 0.3954 |

The temporal curriculum improves local face reconstruction, but its predicted
transport is increasingly under-amplitude. At eight frames it also loses
integrated temporal correlation. Spatial integration exposes this global
coherence/amplitude deficit even when pixel and local-face errors improve.

## Decision

Do not train confirmation seeds 1701 and 1703 under this short-unroll protocol.
The frozen rule requires both state and physics preservation; state success
cannot rescue failed power and separatrix gates.

Stop further deterministic short-unroll tuning of this local operator. Preserve
the selected checkpoint as evidence that predicted-feedback exposure improves
the conditional mean. The next model experiment should add the previously
planned explicit global coherent/persistent stochastic state communicating
with a full-resolution local branch, while retaining circular toroidal
operations and joint five-field output. The new model must be trained from
fields only and judged by the same held-back physics metrics.

## Provenance

- tracked training result:
  `paper0/results/post_ecrd_old_85604_four_step_feedback_pilot_6937357.json`
  (SHA-256
  `ffcc3d4b5bdbada7c83dacc4eb85fed318a0f971ab28b329ea5ed7c15cd7938f`);
- tracked evaluation result:
  `paper0/results/post_ecrd_old_85604_four_step_feedback_evaluation_6937465.json`
  (SHA-256
  `08a41641c22e9a66dfb8116645493c5b73905b4ed4d7ae33acf2bbd7489f5497`);
- physics metrics SHA-256:
  `0c2fa7351c700ce60bc763e48b6986ed51854c42f9d632d38689bcb48716027a`;
- state metrics SHA-256:
  `1d70e2daa7eada73c5154e048afd7ba22aa4d3c5a31de2e587f402b169065c49`;
- W&B training run:
  `p0oldpush4-j6937357-s1702`;
- W&B evaluation run:
  `p0oldpush4eval-j6937465`;
- training performed in evaluation job: false;
- physics-derived training loss: false;
- 85606 read: false;
- newer NERSC data read: false.

