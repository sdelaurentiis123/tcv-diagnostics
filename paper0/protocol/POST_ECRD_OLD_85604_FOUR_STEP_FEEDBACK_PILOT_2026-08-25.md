# Old-85604 four-step feedback fine-tuning pilot

**Frozen:** 2026-08-25 after bounded state job `6937051` and physics-scoring
job `6937203`, and before any optimizer update under this protocol

**Prospective evidence commit:**
`dd768ba3970c2af46b864b3c92bb3d420e01a074`

**Development simulation:** 85604 only

**Held-out 85606, guard interval, and newer NERSC data:** unopened and
prohibited

## Question

Can exposure to four steps of the model's own predicted C5P state reduce the
small-step operator's accumulated field drift without changing its architecture
or teaching against a physics-derived loss?

The bounded comparison found that repeated one-frame predictions preserve
toroidal power and separatrix transport better than direct or coarse
transitions, while accumulating pressure-state error. This pilot changes only
the fine-tuning input distribution and temporal field objective. It is not an
architecture comparison and cannot establish stochastic calibration.

## Immutable parent

Use seed 1702 from the confirmed Stage-2 multi-lead operator:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage2_multilead_scaling/array_6936641/task_1_seed_1702_job_6936642/run/checkpoint_epoch_004.pt
```

Its SHA-256 is
`b9007f818eb35d82a1e4c21771dfc1ad870591feb777f5087f3b2a49847cd50d`.
The tracked parent-result SHA-256 is
`9ba99fc02ac8620dd1a1e917a7f5973f08925ada3389a3a65b913f941d410881`.
Load the parent model weights bitwise and construct a fresh optimizer. Do not
restore or mutate the parent checkpoint.

The unchanged model has 2,174,021 trainable parameters, one C5P history frame,
joint `Ne, Pe, Pi, phi, Vi` output, circular toroidal operations, stride one in
the toroidal direction, and no codec.

## Split and windows

Preserve the existing immutable boundaries:

```text
training   [0,432)
guard      [432,496)  unread
validation [496,624)
```

Training uses every four-step window with current frames `[0,428)`, for 428
windows. A window contains the current state and exact field targets at the
next four saved frames. It is one temporally correlated simulation interval,
not 428 independent physical realizations.

Random circular toroidal augmentation applies one common shift to the current
state and all four targets in a window. Never shift a nonperiodic axis.

## Training objective

Always query the parent's lead-one transition. Beginning from the true current
state, predict four consecutive states. Every predicted state becomes the next
model input. No intermediate truth replaces a prediction.

For numerical feasibility on a 20-GiB GPU slice, stop gradients through the
predicted input passed between steps. Each step still receives a gradient from
its own field error while being trained on the autonomous state distribution.
This is truncated pushforward training, not full backpropagation through time.
The truncation is fixed before results and must be reported.

For step `s`, define `L_s` as the equal-field MSE between the predicted and true
standardized state, divided by the fixed Stage-2 training derivative RMS for
each field. Use:

```text
L = 0.5 L_1 + 0.5 (L_1 + L_2 + L_3 + L_4) / 4
  = 0.625 L_1 + 0.125 L_2 + 0.125 L_3 + 0.125 L_4.
```

This retains a dominant one-step term. No flux, spectrum, phase, coherence,
PDE, conservation, or other physics quantity enters the loss.

## Optimization

- seed: `1702`;
- epochs: `6`;
- sample batch size: `1`;
- gradient accumulation: `4` windows;
- expected optimizer updates: `642`;
- optimizer: fresh AdamW;
- peak/minimum learning rate: `2e-5` / `2e-6`;
- weight decay: `1e-4`;
- warmup: five percent of updates, then cosine decay;
- gradient clip norm: `1.0`;
- autocast: bfloat16;
- TF32: disabled.

The detached feedback makes peak activation storage comparable to one model
call rather than four simultaneous calls. Request one generic preemptible GPU,
four CPUs, 20 GiB host memory, and two hours. Do not request multiple GPUs.

## Checkpoint selection

Physics metrics are prohibited from checkpoint selection. Before constructing
the optimizer, evaluate the immutable parent on the exact selection starts:

```text
496, 501, 506, 511, 516, 521, 526, 531,
536, 541, 546, 551, 556, 561, 566, 571,
576, 581, 586, 591, 596, 601, 606, 611
```

For each epoch and each start, evaluate autonomous lead-one rollouts at horizons
one, four, and eight. For every horizon and field, calculate state MSE divided
by persistence MSE in standardized state coordinates. Select the checkpoint
with the unweighted mean of the 15 horizon-field ratios. These 24 starts are a
chronological validation sample used only for model selection; they are not an
independent test set.

## State pilot gate

After selecting the checkpoint, evaluate all eligible validation starts at
horizons one, four, and eight. The state pilot passes only if:

1. the exact update count, loss decrease, finite-metric, checkpoint-reload, and
   integer-toroidal-equivariance checks pass;
2. full-validation one-step mean-field error is no more than five percent above
   the bitwise parent;
3. full-validation four- and eight-step mean-field errors each do not increase;
4. the mean four/eight-step error improves by at least five percent relative to
   the bitwise parent.

The run must record the gate but must not tune against a failed item.

## Physics preservation gate

Mechanical completion authorizes one inference-only comparison between the
frozen parent and frozen pilot using the same complete four/eight-frame state,
toroidal spectrum, cross-field, and transport implementations as jobs
`6937051` and `6937203`. Physics remains evaluation-only.

The pilot can advance to seeds 1701 and 1703 only if, in addition to the state
gate:

- its repeated-one-frame median absolute log power-ratio error across the three
  frozen mode bands and five fields does not increase by more than ten percent
  at either horizon;
- its separatrix relative-L2 error averaged over the four transport quantities
  does not increase by more than five percent at either horizon;
- strict-face transport error and cross-field coherence change are reported,
  even though they are not used to rescue the pilot.

No single favorable physics metric can rescue a failed state gate. These
physics thresholds select whether to scale the mechanism, not a training
checkpoint.

## Outcomes

- **Pass:** authorize two matched confirmation seeds in a separately committed
  amendment.
- **Fail:** stop further short-unroll tuning of this local deterministic
  operator. The next intervention must be the already planned nonlocal,
  state-complete, or persistent-stochastic architecture, frozen separately.

This protocol never opens 85606, the newer NERSC data, assimilation,
diagnostic ranking, steering, or a physics-derived training loss.
