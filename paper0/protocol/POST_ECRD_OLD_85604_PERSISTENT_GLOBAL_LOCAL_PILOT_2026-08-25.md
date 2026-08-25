# Old-85604 persistent global--local stochastic pilot protocol

**Frozen:** 2026-08-25, after the four-step predicted-feedback physics result
and before implementation, training, or model output under this protocol

**Development simulation:** 85604 only

**Held-out 85606:** unopened and prohibited

**New NERSC data:** unopened and prohibited in this pilot

## 1. Decision and scope amendment

The completed ECRD ladder already tested one-step joint residual diffusion,
deep raw-C5P conditioning, random toroidal rolls, exact generator
equivariance, no toroidal downsampling, a two-frame history, and
global/mesoscale/local corruption noise.  Neither ECRD arm passed the frozen
transport-covariance gate.  In particular, local transport spread was near
one while median integrated-transport spread--skill remained about `0.36`.

The subsequent deterministic four-step predicted-feedback pilot improved
five-field state error, local face error, and average cross-phase error, but
further suppressed fluctuation power and integrated separatrix transport.
Its readout prospectively identified one remaining model mechanism:

> an explicit global coherent stochastic state that persists through a
> forecast block and communicates with a full-resolution local generator.

This dated protocol amends Section 6.4 of
`POST_ECRD_STATE_DATA_SCALING_PROTOCOL.md` for one bounded old-data mechanism
pilot.  It does not reinterpret the single 85604 trajectory as evidence for
an identifiable universal stochastic law.  Passing authorizes only matched
confirmation seeds.  It does not authorize 85606, newer NERSC data,
assimilation, diagnostic ranking, control, or steering.

## 2. Scientific question

Can a joint four-frame residual diffusion model repair the observed
domain-scale transport-covariance deficit when it is forced to represent
low-toroidal-mode uncertainty with one temporally shared recurrent global
state, while retaining an equivariant full-resolution local innovation?

This differs from ECRD's multiscale corruption noise in three controlled
ways:

1. four consecutive future fields are generated jointly rather than as
   independent one-step calls;
2. one low-mode stochastic realization is shared across the four-frame block
   and propagated by a recurrent global stream;
3. the global stream conditions every local decoding stage and alone owns the
   preregistered low-toroidal-mode output band.

## 3. Immutable data contract

Only the existing processed 85604 trajectory may be read.

| region | frames | use |
|---|---:|---|
| training | `[0,432)` | fitting and training-only scales |
| guard | `[432,496)` | never read |
| validation | `[496,624)` | checkpoint selection and later evaluation |

Four-frame training windows contain one current frame `t` and targets
`t+1,...,t+4`, all inside `[0,432)`.  Validation windows follow the same rule
inside `[496,624)`.  A shared random circular roll is applied to the current
state and all four targets in training only.  No nonperiodic axis is rolled.

The fields are standardized C5P in the fixed order
`[Ne, Pe, Pi, phi, Vi]`.  Existing normalization was fit on `[0,432)` only.
The saved cadence is `3.131905426352636 microseconds`.  Only stored toroidal
`z` is periodic.  `zperiod=5`, hence full-torus mode number is `n=5k`.

Temporal windows are correlated samples from one simulation, not independent
physical shots.

## 4. Frozen deterministic mean

The mean branch is the existing 2,174,021-parameter codec-free C5P increment
operator initialized from the selected seed-1702 four-step predicted-feedback
checkpoint:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_four_step_feedback_pilot/job_6937357_seed1702/
run/checkpoint_epoch_006.pt
```

Checkpoint SHA-256:
`affe2589f4ce6639879ca1ed4a100af764aa48a475a653987faa18d4ce844117`.

The mean is rolled for four consecutive lead-one steps without future truth.
It is trainable, rather than a frozen H1 parent.  Its explicit field-only mean
loss is equal-field state MSE in frozen training-derivative RMS units.  The
state fed to the next mean step is detached, matching the already audited
bounded curriculum and preventing a change in memory from masquerading as
the stochastic intervention.  Diffusion-target construction detaches the
mean, so stochastic loss cannot move the mean branch.

## 5. Persistent global--local residual diffusion

For future step `j=1,...,4`,

```text
x_hat[t+j] = mu[t+j] + scale[j] * innovation[t+j].
```

All five fields and all four times are generated jointly.

### 5.1 Global stream

The global stream retains every one of the 88 toroidal cells and pools only
the two nonperiodic axes by a fixed factor of four.  It receives the noisy
four-step residual, current C5P state, and four-step mean trajectory.  A
mixed-boundary ConvGRU propagates one hidden state through the four future
steps.  Its decoded contribution is projected exactly onto stored toroidal
indices `|k| <= 7`, corresponding to physical modes through `n=35`.

The initial global Gaussian component is sampled once per ensemble member
and shared across all four future steps.  This is the persistent stochastic
coordinate.  It is not an absolute phase label, time label, hand-engineered
regime label, or post-hoc spread multiplier.

### 5.2 Local stream

The local stream uses circular convolutions along toroidal `z`, zero padding
along the two nonperiodic axes, random toroidal-roll augmentation, and no
toroidal downsampling.  It receives the current state, corresponding mean
state, diffusion-noise level, and decoded recurrent global features at every
resolution.  Its output is projected onto `|k| > 7`, so the local branch
cannot silently replace the persistent low-mode branch.

The complete denoiser output is the sum of the disjoint low- and high-mode
contributions.  Every operation is required to be equivariant to every
integer circular toroidal shift.

### 5.3 Training noise and loss

The EDM corruption is the unit-variance sum of:

- a low-mode, nonperiodically coarse global Gaussian draw shared across the
  four future steps; and
- independent full-resolution local Gaussian draws for each future step.

The component weights are fixed to equal RMS contribution before any 85604
model result.  Residual scales are one positive training-only RMS per future
step and field, fit from the frozen parent residuals on authorized training
windows only.

The stochastic objective is ordinary EDM denoising MSE on standardized
fields.  The total objective is the sum of the diffusion objective and the
explicit mean-state objective.  Flux, spectra, cross-phase, coherence,
conservation, PDE residuals, boundary residuals, blobs, and all other
physics-derived quantities are forbidden from training and checkpoint
selection.

## 6. Bounded execution ladder

Only seed `1702` is authorized initially.

1. **Synthetic unit checks:** shape, finite gradients, exact low/high Fourier
   partition, shared-versus-local noise identity, and every-integer-shift
   equivariance on a reduced synthetic grid.
2. **Rusty engineering smoke:** eight training windows, four validation
   windows, one epoch, one allocated GPU, online W&B, and no scientific
   interpretation.
3. **Pilot training:** 20 complete chronological-training epochs after the
   smoke passes; sample batch one; gradient accumulation over two windows;
   AdamW; bfloat16 autocast; TF32 disabled; EMA; gradient clipping at one.
   The stochastic branch uses peak learning rate `1e-4`; the initialized mean
   branch uses `1e-5`.  Checkpoints are evaluated every two epochs with a
   frozen two-probe noise bank.

The earliest checkpoint minimizing the equal-weight mean of data-only
validation EDM loss and normalized mean-state MSE is selected.  Validation
is reported separately for chronological blocks `V00`, `V01`, and `V02`.
No physics quantity may select a checkpoint.

The exact parameter count, memory estimate, optimizer update count, noise
seeds, paths, and commands are frozen in a machine-readable execution
manifest after synthetic checks and before the Rusty smoke.

## 7. Scientific forecast and gates

Scientific evaluation is separately frozen only after the engineering smoke
and pilot state gate pass.  It will generate a canonical memberwise four-step
forecast with axes

```text
[start, ensemble_member, future_time, channel, x, y, stored_toroidal_z].
```

The pilot advances to confirmation seeds only if all of the following hold:

1. checkpoint reload, update count, finite-value, shape, W&B, and exact
   toroidal-equivariance gates pass;
2. the learned mean does not worsen the frozen parent mean-field MSE by more
   than 5% at either one or four frames;
3. equal-field fair CRPS improves over the deterministic mean at one and four
   frames and no field is catastrophically overdispersed;
4. expected-member median absolute log power-ratio error over `k=1..3`,
   `k=4..5`, and `k=6..7` is no more than 1.10 times the frozen deterministic
   parent at frame four;
5. density--potential complex cross-spectrum error improves, and mean
   cross-phase error does not worsen by more than two degrees;
6. median spatial transport-covariance relative error is below `0.90`;
7. at least three of four aggregate local-transport spread--skill ratios lie
   in `[0.80,1.25]` and none exceeds `1.40`;
8. the median integrated-transport spread--skill ratio is at least `0.60`;
9. integrated separatrix transport relative-L2 error is no more than 1.05
   times the frozen deterministic parent at frame four.

Chronological moving-block bootstrap intervals are conditional on this one
85604 run.  Nonlinear transport is computed for every ensemble member before
ensemble reduction.

Failure means this old-data persistent stochastic mechanism is not scaled.
It does not establish that persistent latent models are impossible.  The
next bottleneck returns to additional independent data or a complete saved
state.  Passing authorizes matched seeds 1701 and 1703 under a dated
amendment; it does not release 85606.

## 8. Prohibited actions

This protocol does not authorize:

- reading, enumerating, preprocessing, or evaluating 85606;
- reading or inventorying the newer NERSC datasets;
- changing the frozen train/guard/validation boundaries;
- adding any physics diagnostic to a loss or checkpoint score;
- rerunning B2--B5 or the ECRD ladder unchanged;
- assimilation, sensor ranking, steering, MPC, or RL;
- describing a single trajectory's forecast members as experimentally
  validated uncertainty.
