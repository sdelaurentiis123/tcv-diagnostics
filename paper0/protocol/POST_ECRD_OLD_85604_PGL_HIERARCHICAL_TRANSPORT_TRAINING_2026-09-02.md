# Persistent global--local hierarchical transport training

**Frozen prospectively:** 2026-09-02, before implementation preflight,
gradient calibration, training, or evaluation output is inspected.

**Development data:** old processed 85604 only.

**Held-out 85606:** prohibited. This experiment may not enumerate, inspect,
hash, load, preprocess, forecast, or score any 85606 artifact.

**New NERSC 85604 segment:** prohibited for this screen.

## 1. Question and scoped protocol exception

The completed one-epoch variogram screen showed that the authoritative
transport variogram was numerically stable and slightly improved local and
integrated spread, but it did not materially change spatial transport
covariance. A spatial variogram is invariant to a common spatial offset and
therefore cannot directly identify the global fluctuation that controls the
separatrix integral.

This experiment asks:

> Can end-to-end, multiepoch training of the existing persistent
> global--local model recover local, regional, and global transport
> uncertainty when all three scales are represented by proper or
> dependence-sensitive ensemble scores?

The repository default makes physics-derived quantities evaluation-only. The
user's explicit 2026-09-02 instruction creates a narrow exception for the
transport-aware arm in this two-arm experiment. The control remains the
ordinary no-physics-loss model. Any successful transport-aware result must be
described as transport-supervised and cannot support a claim that transport
emerged entirely out of a field-only loss.

No architecture, sampler, data split, residual normalization, stochastic
noise law, or observation set changes in this experiment.

## 2. Immutable parent, data, and comparison

Both arms start bitwise from the selected epoch-20 persistent global--local
checkpoint:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_persistent_global_local_pilot/
job_6937586_seed1702/run/training/selected.pt
```

SHA-256:
`4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb`.

The checkpoint contains EMA model states but no optimizer state. Both arms
therefore use a fresh, identically initialized AdamW optimizer. This is a
matched end-to-end warm start, not an exact optimizer continuation.

The immutable split is:

| use | frames | four-frame windows |
| --- | ---: | ---: |
| training | `[0,432)` | 428 |
| guard | `[432,496)` | unread |
| validation | `[496,624)` | 124; frozen evaluation only |

Cadence is 3.131905426352636 microseconds. Only stored toroidal `z` is
periodic. `zperiod=5`, so physical toroidal mode `n=5k`. Training retains the
existing shared random circular roll of current and target states. No
nonperiodic axis is rolled.

The arms are:

| arm | objective |
| --- | --- |
| CONTROL | original mean-state plus EDM denoising objective |
| TRANSPORT | identical original objective plus normalized local, regional, and global transport scores |

Every stochastic sample is generated from the conditioning state alone with
the unchanged 18-step, 35-NFE differentiable Heun sampler. Future truth never
enters the sampler. All transport is decoded and computed separately for
every ensemble member before any score or reduction.

## 3. Transport hierarchy

The differentiable transport operator and native-81 truth are the
hash-closed authorities from variogram preflight job 6958597. All four
quantities receive equal weight:

1. particle transport;
2. electron internal-energy transport;
3. ion internal-energy transport;
4. total internal-energy transport.

### 3.1 Local score

The local term is the mean of the existing fair order-one spatial and
temporal variogram scores. It uses six equally weighted physical-distance
bins, 1,024 fixed pairs per bin, and temporal lags 1--4. Existing geometry,
pair-bank, gauge, native-resampling, and member-wise transport checks remain
unchanged.

### 3.2 Regional joint score

The already-quadrature-weighted local separatrix contribution has shape
`[16 poloidal rows, 81 toroidal cells]`. Partition it prospectively into 12
nonoverlapping sectors:

- four contiguous poloidal groups of four rows each;
- three contiguous toroidal groups of 27 cells each.

Sum within every sector. For each quantity, concatenate the 12 regional sums
over all four future frames into one 48-dimensional vector and apply the fair
multivariate energy score. Euclidean distances are divided by the square root
of vector dimension.

The regional term also includes two joint Fourier energy scores. First sum
transport over the 16 poloidal rows to obtain a toroidal transport profile,
then use an orthonormal real FFT over 81 cells:

- low band: stored `k=1,2,3`, physical `n=5,10,15`;
- transport band: stored `k=4,5,6,7`, physical `n=20,25,30,35`.

Real and imaginary coefficients from all four future frames are concatenated
within each band. The regional hierarchy term is the equal mean of the
physical-sector, low-mode, and `n=20--35` energy scores after separate initial
normalization. The `k=0` coefficient is excluded here because it is the global
term below.

For members `X_m` and truth `y`, the fair energy score is

```text
mean_m ||X_m-y|| - [1/(M(M-1))] sum_(m<m') ||X_m-X_m'||.
```

### 3.3 Global score

For each future frame and quantity, sum all 16 by 81 weighted separatrix
contributions. This is exactly the `k=0` or total-transport component. Apply
the fair finite-ensemble CRPS and average future frames and quantities
equally:

```text
mean_m |Q_m-y| - [1/(M(M-1))] sum_(m<m') |Q_m-Q_m'|.
```

The local variogram prevents a global-only solution; the regional score
constrains mesoscale joint behavior; the original field loss prevents the
model from gaming transport summaries with implausible fields.

## 4. Initial normalization and gradient-balanced weight

Use four ensemble members during training. On the 32 fixed chronological
control starts `floor(linspace(0,427,32))`, compute positive finite initial
magnitudes separately for every quantity and hierarchy component. No moving,
target-dependent, or post-result normalization is allowed.

Define each hierarchy term as an equal average of its component scores divided
by those initial magnitudes. On fixed gradient-audit starts
`[0,142,285,427]`, compute full parameter gradients for:

- original objective;
- normalized local hierarchy;
- normalized regional hierarchy;
- normalized global hierarchy;
- their unweighted sum.

Set one common auxiliary multiplier prospectively as

```text
lambda = 0.25 * ||grad(original)|| / ||grad(local+regional+global)||.
```

This makes the combined auxiliary gradient exactly 25% of the original
gradient on the frozen calibration population, within numerical tolerance.
The same scalar multiplies all three already-normalized hierarchy terms. It is
computed once, hash closed, and shared by both arms. Record raw and
learning-rate-scaled gradient norms and cosine similarities by loss and by
parameter branch. Failure to obtain finite nonzero gradients blocks training.

## 5. Trainable branches and optimization

Train all persistent-model stochastic parameters:

- recurrent global stream;
- local encoder;
- local decoder and output projection.

Also train the deterministic mean branch at one tenth the stochastic learning
rate. EDM denoising retains the original detached mean condition. Transport
scores may differentiate through the predicted mean and stochastic sampler.
Autoregressive feedback in the mean remains detached between steps.

The fixed two-epoch screen uses:

- seed 1702;
- batch one;
- gradient accumulation over two windows;
- 428 windows and 214 optimizer updates per epoch;
- exactly 428 updates total;
- stochastic learning rate `1e-6`;
- mean learning rate `1e-7`;
- fresh AdamW, betas `(0.9,0.99)`, weight decay `1e-4`;
- global gradient clipping at norm 1;
- EMA decay `0.999`, initialized from the parent;
- four full-sampler members and 18 sampling steps;
- bfloat16 network execution, with scores and transport accumulated in
  float32;
- identical window order, augmentations, sigma/noise keys, member seeds,
  and sampler work in both arms.

The CONTROL arm computes the complete hierarchy but multiplies it by exactly
zero, ensuring matched data access, sampler work, and random-number use.

Save both arms at fixed updates:

- update 107: 0.5 epoch;
- update 214: 1 epoch;
- update 428: 2 epochs.

These are fixed duration checkpoints, not candidates selected by training or
physics metrics. Each checkpoint contains EMA mean and stochastic states and
the corresponding raw state hashes. Optimizer state is retained only to make
an explicitly authorized longer continuation possible; it is not used to
select the two-epoch result.

## 6. Mandatory validation before training

Before the two-arm screen:

1. Revalidate the existing sampler and Torch/native transport equivalence
   artifacts and hashes.
2. Unit-test fair CRPS and fair energy-score known answers, permutation
   sensitivity, member-wise nonlinear evaluation, gradient finiteness, and
   exact `k` to `n=5k` mapping.
3. Verify that the 12 regional sectors are disjoint and exactly cover all
   `16*81` separatrix locations.
4. Verify that global transport equals both the sum of regional transport and
   the appropriately scaled `k=0` Fourier coefficient.
5. Verify that all hierarchy controls are positive and finite.
6. Verify a 25% combined auxiliary/original gradient-norm ratio to relative
   tolerance `1e-4` on the calibration population.
7. Run one production-size optimizer update for CONTROL and TRANSPORT. It
   must fit one allocated Rusty GPU without reducing members, sampler steps,
   resolution, quantities, or loss components.
8. Require online W&B and record immutable local artifacts as authority.

## 7. Fixed M32 evaluation and stopping rule

Truth-free generation uses the existing 36 validation starts, all four future
frames, the frozen M32 seed bank, and unchanged 18-step sampler. Score only
after each forecast is closed and hashed.

Evaluate updates 107, 214, and 428 for both arms. Report:

- field fair CRPS and spread--skill;
- local and integrated spread--skill for all four transport quantities;
- spatial transport-covariance error;
- transport variograms by physical distance and temporal lag;
- regional covariance and regional energy score;
- global CRPS and integrated transport distribution;
- transport `n=0`, `n=5--15`, and `n=20--35` summaries;
- spectra and density--potential cross-spectrum, coherence, and phase;
- mean local and integrated transport error;
- fixed-noise global/local latent interventions;
- branch-level gradients from original, local, regional, and global terms.

At update 428, continuation beyond two epochs is authorized only if the
TRANSPORT arm, relative to the matched CONTROL arm at the same update:

1. improves median integrated spread--skill by at least `0.05` absolute;
2. reduces median spatial covariance error by at least `0.01` absolute;
3. leaves mean transport, phase, spectra, and field gates passing.

The production acceptance gates remain unchanged:

- median integrated spread--skill at least `0.60`;
- median covariance error below `0.90`;
- at least three of four local spread--skill values in `[0.80,1.25]`, with
  none above `1.40`;
- all field, spectrum, cross-field, and mean-transport gates passing.

If the epoch-two continuation criterion fails, do not extend to epochs 4--8,
do not run additional seeds, and do not open 85606. If it passes, a separate
dated amendment must freeze the longer budget before training resumes.

## 8. Outputs and prohibitions

Write only new, non-overwriting artifacts under:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_pgl_hierarchical_transport/
```

Every artifact records commands, commit, dirty state, input hashes, seeds,
Slurm allocation, W&B identity, objective definitions, normalization,
gradient diagnostics, checkpoints, and output hashes.

This protocol does not authorize architecture changes, new data, 85606,
assimilation, sensor ranking, steering, or control.
