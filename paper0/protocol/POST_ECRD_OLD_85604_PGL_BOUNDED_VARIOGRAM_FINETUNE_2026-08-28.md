# Bounded persistent-model variogram fine-tuning protocol

**Frozen prospectively:** 2026-08-28, before score validation, implementation
results, or fine-tuning output.

**Development simulation:** old processed 85604 only

**Held-out 85606:** prohibited; no path enumeration, metadata inspection,
hashing, loading, preprocessing, forecast, or scoring is authorized

**New NERSC 85604 segment:** inventory only under its separate protocol; it is
not authorized for this one-seed screen

## 1. Narrow question and explicit scope amendment

The selected persistent global--local ensemble improved the conditional mean,
spectra, and density--potential phase but remained underdispersed locally and
especially in integrated separatrix transport. This bounded campaign asks:

> Does a dependence-sensitive variogram objective teach the existing
> persistent stochastic branch the missing spatial and temporal covariance,
> without changing its architecture, data, sampler, or deterministic mean?

This is a controlled training-objective ablation, not a new architecture.

The Paper 0 default forbids physics-derived training losses. The user's dated
2026-08-28 instruction explicitly amends that restriction for arms C and D of
this bounded experiment only. Those arms use authoritative particle-transport
variograms as training quantities and must be labeled as such. Flux, spectra,
cross-phase, coherence, conservation, PDE residuals, heat transport, and every
other physics quantity remain absent from the loss. A result from C or D
therefore cannot support the stronger claim that transport emerged entirely
out of loss.

No arm may launch until the variogram known-answer tests and differentiable
transport equivalence gate in Section 5 pass.

## 2. Immutable parent and data contract

All arms start from the same selected epoch-20 EMA checkpoint:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_persistent_global_local_pilot/
job_6937586_seed1702/run/training/selected.pt
```

SHA-256:
`4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb`.

The checkpoint stores EMA model weights but no optimizer, scheduler, or raw
training weights. The campaign is therefore a **matched warm-start
fine-tuning screen with a fresh optimizer**, not an exact continuation. Arm A
is required to measure the effect of ordinary additional optimization.

The architecture, residual scales, structured-noise law, field order, and
four-frame horizon are unchanged. The deterministic mean weights are loaded
bitwise from the checkpoint, set to evaluation mode, and frozen. No mean
parameter may enter the optimizer and its pre/post state hashes must match.

Only the existing old-85604 C5P artifacts are read:

| interval | frames | use |
| --- | ---: | --- |
| training | `[0,432)` | 428 four-frame windows |
| guard | `[432,496)` | never read |
| validation | `[496,624)` | final fixed evaluation only |

The cadence is 3.131905426352636 microseconds. Only stored toroidal `z` is
periodic. `zperiod=5`, hence physical toroidal mode `n=5k`. Training retains
the original deterministic random circular roll shared by the current frame
and all four targets. No nonperiodic axis is rolled.

## 3. Matched arms

| Arm | Objective terms |
| --- | --- |
| A | original EDM denoising loss only |
| B | original + field-space spatial variogram + field-space temporal variogram |
| C | original + authoritative particle-transport spatial variogram + authoritative particle-transport temporal variogram |
| D | all five terms |

Every arm uses the same parent, seed `1702`, training-window order, circular
rolls, sigma draws, two structured-noise samples per window, optimizer,
learning-rate schedule, batch/accumulation pattern, number of updates, EMA,
gradient clipping, validation population, scientific sampler seed rows, and
forecast sampler. Only the enabled objective terms differ.

## 4. Proper-score definition and pair banks

For ensemble members `m=1,...,M`, truth `y`, order `p=1`, and preregistered
pairs `(i,j)`, use the empirical variogram score

```text
sum_(i,j) w_ij [ |y_i-y_j| - mean_m |x_mi-x_mj| ]^2.
```

All training variograms use `M=2` denoised clean-field estimates produced from
two keyed structured-noise draws at the same keyed EDM sigma. The original
EDM loss is averaged across those same two draws in every arm. The auxiliary
score is a differentiable training surrogate evaluated on denoised clean
estimates; the unchanged 18-step sampler remains the authority for final
ensemble evaluation.

### 4.1 Physical-distance bins

Cell positions are constructed from the frozen geometry as cylindrical
Cartesian coordinates `(R cos(phi), R sin(phi), Z)` with
`phi=2*pi*z/(5*88)`. Only strict operator/wall cells are eligible.

Use a deterministic geometry-only candidate bank with seed `856040828` and
six distance bins whose edges are the candidate-distance quantiles at
`[0,1/6,...,1]`. Degenerate edges are an error. For every future time and
field, draw 32 spatial pairs per bin without replacement when possible. Each
bin receives total weight `1/6`, independent of the number of available
pairs; fields and future times receive equal total weight. The fixed pair bank
is hashed before training and reused by all arms.

Temporal field pairs use the same strict spatial cells, the same field at
future-frame lags 1, 2, and 3, and equal total weight per lag, field, and valid
time pair. A fixed 128-cell geometry-only sample per lag and field is used.

### 4.2 Field-space score

Field variograms operate on standardized predicted and true C5P fields. All
five fields receive equal total weight. Spatial and temporal scores are kept
separate through logging and initial normalization.

### 4.3 Authoritative particle-transport score

Transport is computed separately for every denoised member and for truth
before any ensemble reduction. Standardized `Ne` and `phi` are decoded using
the frozen train-split normalization. The differentiable operator must exactly
transcribe the audited Hermes radial `xz` plus shifted-`xy` ExB face flow,
positivity handling, single-null shifted-poloidal connection, strict confined
separatrix mask, and wedge weights. It returns signed local particle-transport
contributions on poloidal rows `y=8..23` and all 88 toroidal cells.

Transport spatial pairs use the same six-bin quantile rule on physical
positions along the confined separatrix, 32 pairs per bin and future time,
with equal total bin weight. Temporal transport pairs use the same separatrix
location at lags 1, 2, and 3 with equal total lag weight. Spatial and temporal
scores remain separate through logging and normalization.

No ensemble-mean field may be passed through the nonlinear transport operator
as a substitute for member-wise transport.

## 5. Mandatory score and operator validation

Before a training job can launch, synthetic checks must establish:

1. **Truth-like ensemble:** members identical to truth have zero field and
   transport variogram score to numerical tolerance.
2. **Spatially shuffled ensemble:** a fixed nontrivial spatial permutation
   produces a strictly positive spatial score and exceeds the truth-like
   score.
3. **Constant-biased ensemble:** adding one common constant to every coordinate
   leaves a variogram score unchanged. This expected invariance is recorded,
   not mistaken for a failure; the original EDM term is what penalizes bias.
4. **Temporal shuffle:** a fixed nontrivial future-time permutation produces a
   strictly positive temporal score on a nondegenerate trajectory.
5. **Gradient check:** every enabled score is finite, differentiable, and gives
   finite nonzero gradients on a perturbed synthetic ensemble.
6. **Torch/NumPy transport equivalence:** on at least four authorized 85604
   training frames plus synthetic edge cases, the Torch local separatrix
   particle contribution matches the authoritative NumPy result with maximum
   relative L2 at most `2e-5` and integrated relative error at most `2e-5` in
   float32. Float64 reference agreement must be at most `2e-10`. Signs, masks,
   positivity decisions away from exact zero, and toroidal periodicity must
   match. Failure blocks arms C and D rather than substituting a proxy.
7. **Pair-bank audit:** six nonempty monotonically ordered physical-distance
   bins, equal total weight per bin to `1e-12`, no nonperiodic wrapping, and an
   identical SHA-256 pair-bank identifier for all arms.

The score-validation result is a hash-closed JSON artifact and is run before
the four-arm Slurm array.

## 6. Loss normalization and fixed optimization budget

Evaluate the unmodified parent on the first 32 chronological training windows
with fixed sigma/noise keys. Record positive finite initial control magnitudes
for the original EDM term and each of the four auxiliary terms. These five
scalars are computed once by a separate read-only calibration job, hash
closed, and reused by every arm.

For the enabled term set `Q`, train with

```text
L = L_edm(0) * mean( L_edm/L_edm(0), {L_q/L_q(0) : q in Q} ).
```

Thus each term has unit initial normalized magnitude and every arm begins on
the same overall objective scale. There is no tuned lambda in the one-seed
screen.

The fixed budget is:

- 6 complete epochs;
- 428 windows per epoch;
- sample batch one;
- gradient accumulation over two windows;
- exactly 214 optimizer updates per epoch and 1,284 updates total;
- two keyed stochastic denoising samples per window in every arm;
- fresh AdamW on stochastic parameters only, betas `(0.9,0.99)`, weight decay
  `1e-4`;
- cosine learning rate with 5% warmup, peak `2e-5`, minimum `2e-6`;
- gradient clip norm 1;
- EMA decay `0.999` initialized from the parent stochastic weights;
- bfloat16 autocast, TF32 disabled;
- no early stopping and no within-arm checkpoint selection: the fixed final
  EMA state is evaluated.

All raw and normalized terms, gradient norms, latent-use probes, memory,
timing, and hashes are logged to online W&B and immutable local artifacts.

## 7. Truth-separated final evaluation

Generate each final arm with the unchanged persistent sampler:

- the same 36 preregistered starts in V00/V01/V02;
- 16 members for the one-seed screen, using the first 16 columns of the frozen
  M32 seed bank;
- all four future frames;
- 18 EDM steps / 35 denoiser evaluations per member;
- no member interaction and no post-hoc spread multiplier.

Truth remains unavailable to forecast generation. Scoring runs only after a
forecast is closed and hashed.

Report for every arm:

- field fair CRPS and spread--skill;
- low/high-mode ensemble variance and fixed-noise global/local latent-use
  ablations;
- local and integrated spread--skill for all four authoritative transport
  quantities;
- spatial transport-covariance relative Frobenius error;
- field and particle-transport variograms by physical distance and temporal
  lag;
- directional spectra and retained power;
- density--potential complex cross-spectrum, coherence, and phase;
- member-wise mean local and integrated transport error;
- mean-state equality to the frozen parent;
- inference cost and ensemble size.

## 8. Pass, selection, and escalation

An arm passes only if all seven original persistent-model physics gate
families pass at horizon four, including:

- median spatial transport-covariance relative error below `0.90`;
- at least three of four local transport spread--skill ratios in
  `[0.80,1.25]`, with none above `1.40`;
- median integrated transport spread--skill at least `0.60`;
- retained spectral and density--potential gates;
- integrated mean transport no worse than 1.05 times the frozen deterministic
  parent;
- no catastrophic field overdispersion and improved fair CRPS over the frozen
  mean.

Among passing arms, select the arm with the lowest median spatial
transport-covariance error. Ties within `0.001` use, in order: integrated
spread--skill closest to one, lower equal-field fair CRPS, then the simpler
arm order `A, B, C, D`. Metrics are not traded against a failed gate.

If no arm passes, stop: do not run confirmation seeds and do not use 85606.
If one arm passes, repeat only that objective from the same parent with
fine-tuning/order/noise seeds 1701 and 1703, while retaining the seed-1702
screen result. These are warm-start fine-tuning seeds, not three independently
trained base architectures. Use 32 members for the final three-seed readout if
compute permits; otherwise 16 is retained and disclosed.

Only after the separate NERSC inventory passes may the one selected objective
be adapted to the cadence-aware 944-frame combined training corpus. That run
requires a dated split/cadence amendment and must not connect discontinuous
source boundaries. No other screen arm is repeated on the combined corpus.

## 9. Outputs and prohibitions

Write new, non-overwriting artifacts under:

```text
paper0/results/post_ecrd_old_85604_pgl_variogram_screen_2026_08_28/
paper0/figures/post_ecrd_old_85604_pgl_variogram_screen_2026_08_28/
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_pgl_variogram_screen/
```

Every run records the command, seed, input hashes, code revision, dirty state,
Slurm resources, W&B identity, output hashes, and exact loss normalization.

This protocol does not authorize architecture changes, new data during the
screen, sampler changes, assimilation, diagnostic ranking, steering, control,
or any access to 85606.
