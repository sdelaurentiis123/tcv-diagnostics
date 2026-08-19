# Paper 0 Phase 3.5 prospective cause-localization amendment

**Protocol status:** frozen before implementation and before any new Phase 3.5
result is computed

**Freeze date:** 2026-08-19

**Development simulation:** TCV/Hermes `85604` only

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Architecture training authorized:** no

## 1. Motivation and narrow scope

The completed residual-KL oracle established the following narrow result:

> A single, fixed, condition-independent, global linear residual distribution
> learned from adjacent 85604 training frames does not describe later 85604
> residuals well.

That K4 result tested neither FGN, PDE-Refiner, diffusion, nor stochastic
emulation in general. It did not distinguish whether its failure arose from a
nonstationary interval, coherent toroidal transport represented in fixed
Eulerian coordinates, translation non-equivariance in the codec or predictor,
an incomplete observed state, memory, inadequate effective sampling, or
forecast-state-dependent residual covariance.

Phase 3.5 is a diagnostic cause-localization phase. It prospectively narrows
Paper 0 after K4 without altering K4, rerunning B2--B5 unchanged, or beginning
another architecture search. It authorizes read-only analysis of immutable
85604 artifacts, lightweight regularized diagnostic probes, truth-assisted
nondeployable oracles, deterministic equivariance inference from the frozen H1
checkpoint, and a fixed-seed B5 context-shuffle sensitivity test if the locked
B5 checkpoint reloads exactly.

No production-scale neural model, optimizer, fine-tuning, assimilation,
diagnostic ranking, steering, O3/O4/O5 rollout, or held-out-shot evaluation is
authorized.

## 2. Prospective hypotheses

The analysis keeps the following explanations distinct.

1. **H1a -- invalid or nonstationary interval.** Slow background, source,
   boundary, or transport evolution makes the earlier and later 85604
   transition distributions different.
2. **H1b -- chronological covariance drift.** Even if raw-state summaries look
   similar, H1 residual mean, variance, modes, cross-field covariance, or
   transport covariance changes chronologically.
3. **H2 -- insufficient effective sample size.** Adjacent saved frames provide
   too few effectively independent observations of important state, residual,
   mode, and transport statistics to estimate a transferable global
   covariance.
4. **H3 -- coherent toroidal translation.** A substantial fraction of the
   apparent one-step residual is phase/position error from coherent motion on
   the periodic toroidal axis rather than deformation or innovation.
5. **H4 -- codec or predictor non-equivariance.** DCAE striding, patch phase,
   absolute positional encoding, or non-periodic coordinate treatment causes
   a toroidal roll of an input not to produce the same roll of its
   reconstruction or forecast.
6. **H5 -- inappropriate global representation.** Translation-aware,
   localized, or multiscale representations transfer chronologically better
   than a fixed global PCA/KL basis at a matched coefficient budget.
7. **H6 -- forecast-state-dependent covariance.** Residual scale, mode
   allocation, cross-field dependence, phase error, or transport error is
   predictable from the available current context, while a constant or
   time-only model is not adequate.
8. **H7a -- incomplete or privileged state.** Omitted evolved electron
   momentum, generalized vorticity, or retained potential-boundary state
   explains future differences among samples that look close in C5P space.
9. **H7b -- history-dependent hidden state.** Delay information beyond the one
   extra frame tested by C5P-H2 improves lightweight chronological probes,
   even though that particular H2 neural transition did not improve O2.
10. **Unexplained failure.** None of the preregistered mechanisms provides
    material, chronologically stable evidence.

These hypotheses are not mutually exclusive. The final ranking uses the
priority order required by the Paper 0 instruction: state/protocol validity,
stationarity, coherent transport/equivariance, context/history, and only then
additional stochastic capacity.

## 3. Immutable data and artifact boundary

### 3.1 Data ranges

The existing boundaries remain unchanged:

| Object | Global frames or targets | Count | Policy |
|---|---:|---:|---|
| raw training state | frames `[0,432)` | 432 | 85604 development only |
| H1 training transitions | targets `[2,432)` | 430 | analysis/probe fitting |
| guard | frames/targets `[432,496)` | 64 | never read |
| raw validation state | frames `[496,624)` | 128 | chronological diagnosis only |
| H1 validation transitions | targets `[498,624)` | 126 | chronological diagnosis only |
| held-out simulation | 85606 | -- | never discovered or read |

The shared model volume is `[field,x,y,z]=[5,64,32,88]` in C5P field order
`[Ne,Pe,Pi,phi,Vi]`. The exact-state dataset also stores
`[Ne,Pe,Pi,NVe,NVi,Vort]` and `Bphi[side,y]`. Cadence is
`3.131905426352636 microseconds`; `zperiod=5`; stored Fourier index maps to
full-torus mode number as `n=5k`. Only `z` is periodic. No circular roll,
wraparound patch, or periodic derivative may be applied to `x` or `y`.

### 3.2 Locked inputs

Every input is hash-checked before analysis. The primary locks are:

- model-data result SHA-256
  `27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18`;
- training-only normalization SHA-256
  `f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7`;
- H1 training forecast SHA-256
  `d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`;
- H1 validation forecast SHA-256
  `a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`;
- H1 seed-1701 checkpoint SHA-256
  `5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`;
- H1 seed-1701 C5P codec SHA-256
  `9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d`;
- B5 selected seed-1701 checkpoint SHA-256
  `255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72`;
- B5 M32 forecast SHA-256
  `1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b`;
- B5 M32 sampler seed-bank SHA-256
  `013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14`;
- authoritative geometry SHA-256
  `0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8`;
- exact native-state result SHA-256
  `cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae`;
- K4 scientific result SHA-256
  `71be0e38285a06f98bd03138d3e1639a70d88665e698cbb4c96220e57dc991b7`.

The run manifest records resolved paths, file sizes, and hashes. A missing or
mismatched artifact is a blocker, not permission to substitute another seed,
checkpoint, field, split, or source.

### 3.3 Residual and gauge convention

For target `t`, the H1 residual is standardized truth minus the frozen H1
standardized mean. Potential truth and prediction are separately gauge-fixed
by subtracting each sample's full spatial mean before their residual is
formed. The training-only axisymmetric residual bias is then subtracted as in
K4. Covariance fitting additionally removes the source sample mean, but that
finite-sample nonaxisymmetric mean is never added to a forecast.

Physical transport always uses original decoded fields and the authoritative
native-grid geometry/operator. Every nonlinear quantity is evaluated on each
state or ensemble member before reduction.

## 4. Chronological blocks and uncertainty

Existing evidence gives a maximum raw-state initial-positive-sequence time of
about 40.6 frames, residual-amplitude first-`1/e` times no larger than 11
frames, and residual-amplitude stable-near-zero lags no larger than 22 frames.
The primary chronological block span is therefore at least 42 frames.

Training targets `[2,432)` are partitioned into ten adjacent 43-target blocks.
For matched basis comparisons the first 42 targets of every training block are
used; the forty-third target remains in stationarity, ESS, context, and
learning-curve summaries but not in the matched transfer matrix. Validation
targets `[498,624)` are partitioned into three adjacent 42-target blocks.
Thus every source-to-target transfer comparison uses exactly 42 samples.

The block identifiers are fixed as:

- `T00=[2,45)`, `T01=[45,88)`, ..., `T09=[389,432)` with matched subsets
  `[start,start+42)`;
- `V00=[498,540)`, `V01=[540,582)`, and `V02=[582,624)`.

The previous six 21-target validation blocks remain reportable companion
groups for comparison with B2--B5, but may not replace the primary 42-target
blocks.

Primary confidence intervals are 95% percentile intervals from 200
non-circular moving-block resamples with temporal block length 12 and master
seed `2026081935`. Sensitivity is reported at lengths 6 and 22. Resampling is
performed independently inside each contiguous data region; it never crosses
the guard or joins the end of a region to its beginning. Adjacent frames are
never counted as independent physical shots. Half-block chronological
first-to-second and second-to-first controls are reported for subspace
transfer at ranks 8 and 16.

## 5. H0 state and data audit

Before numerical analysis, `paper0/phase3_5/DATA_STATE_AUDIT.md` records:

- frame/transition counts, times, cadence, axes, sizes, periodicity, fields,
  units, transforms, normalization source, and split boundaries;
- the six exact evolved Hermes volumes `Ne,Pe,Pi,NVe,NVi,Vort`;
- source-backed transformations to derived `Te,Ti,phi,Vi`;
- whether electron momentum, vorticity, elliptic state, and radial boundary
  state are absent from C5P input;
- whether the exact saved state can be assembled from current 85604 artifacts;
- model architecture details relevant to equivariance, including DCAE
  circular `z` convolution, zero `x/y` padding, two `(2,2,2)` packing strides,
  latent grid `[16,8,22]`, predictor patch `(2,2,1)`, and absolute regular-grid
  sinusoidal/RoPE coordinates;
- every checkpoint, forecast, dataset, geometry, and sufficient-statistics
  artifact used, with hash or stable identifier.

## 6. H1 stationarity and chronological transfer

For every primary block, calculate the following for raw C5P truth and for the
gauge/bias-consistent H1 residual:

1. radial mean and radial fluctuation-RMS profiles for every field, averaging
   only over explicitly stated `y,z` axes;
2. global spatial mean (except ungauged potential), global fluctuation RMS,
   and radial RMS;
3. one-sided Parseval-consistent toroidal power for `k=0`, `k=1..3`,
   `k=4..5` (`n=20..25`), `k=6..7` (`n=30..35`), and `k>=8`;
4. density--potential cross-phase and magnitude-squared coherence in the three
   nonzero frozen bands;
5. authoritative local strict-face and exact-separatrix local contributions,
   plus integrated separatrix particle/electron-energy/ion-energy/total-energy
   transport;
6. `Bphi` inner/outer mean, RMS, and departure summaries when available, plus
   the pre-existing scalar regime indicators.

For every ordered source/target block pair, report:

- global PCA/KL target variance capture at ranks `8,16,32,41` where
  numerically available;
- equal-rank principal-angle summaries: minimum cosine, mean squared cosine,
  and maximum angle;
- residual-mean distance and per-field log variance ratio;
- spectral-band log-power/covariance distance;
- global and region-resolved cross-field covariance distance;
- local transport covariance-multiplier distance;
- integrated transport log-variance distance.

Block PCA is centered using source-block means only. Target data do not alter
the source basis. Coefficient counts and source/target sample counts must be
identical in every matrix cell.

## 7. H2 effective sample size and learning curves

Autocorrelation and integrated autocorrelation time are reported separately
for:

- raw-state means, fluctuation RMS, radial-profile principal summaries, and
  boundary-state summaries;
- coefficients of the first eight full-training residual KL directions;
- residual energy for each field;
- each field's frozen toroidal-band power;
- local transport RMS and all four integrated transport series.

The primary integrated autocorrelation time uses Geyer's initial-positive-pair
sequence. Sensitivity uses fixed truncation windows `8,16,32,64` where the
region supports them and a self-consistent window stopping at the first lag
greater than five current integrated times. Each series is reported both with
only its sample mean removed and after a linear time detrend; detrending is a
sensitivity diagnostic, not permission to relabel the interval stationary.
Effective sample size is `N/tau_int`, capped at `N`, with all definitions and
right-censoring explicit.

Representation learning curves use chronological source prefixes of
`42,84,126,168,210,252,294,336,378,420` targets. Global-PCA target capture is
reported at ranks `8,16,32` on the immediately following 42-target training
block when available and on each fixed validation block. No shuffled or random
prefix is substituted.

## 8. H3 coherent toroidal translation

### 8.1 Shift estimator

Only `z` may be circularly shifted. Before estimating displacement, each C5P
field uses training-only standardization and has its per-`(x,y)` toroidal mean
removed. Each field contribution is divided by its training-region RMS after
that removal. Shared displacement maximizes the normalized multichannel
cross-correlation over all 88 integer shifts. Per-field maxima are companion
diagnostics.

For each pair report the signed shift applied to the earlier/predicted state to
align it to the later/target state, peak correlation, second-best peak outside
the neighboring three-cell exclusion, peak margin, and entropy of the
nonnegative normalized correlation surface. A Fourier subcell displacement is
reported only when a weighted phase-slope fit over `k=1..7` has at least four
finite modes and weighted `R^2 >= 0.8`; otherwise it is explicitly unavailable.

The consecutive-truth distribution is compared with the previous 9--12 of 88
cell observation using the same sign convention. No expected shift is imposed
on the answer.

### 8.2 Nondeployable truth-assisted oracles

Two tests use future truth and are labeled nondeployable:

1. Estimate one shared multichannel toroidal shift between H1 prediction and
   target, roll the entire five-field H1 prediction by that shift, and score it.
2. Estimate one shared shift between the complete current C5P state and next
   target, roll the entire five-field current state, and score transported
   persistence.

The same shift is applied to all fields. Field-specific oracle shifts are
diagnostic only and never form a forecast. Alignment is evaluated by residual
energy, field RMSE, toroidal power, density--potential phase/coherence,
authoritative local/integrated transport, and the complete K4 rank ladder
`0,8,16,32,44,64,128,256,full-positive-rank`. Common alignment is not expected
to repair relative cross-field phase; that distinction separates position
error from deformation/covariance error.

## 9. H4 translation-equivariance audit

The frozen C5P seed-1701 DCAE and H1 transition are loaded in deterministic
float32 evaluation mode with TF32 disabled. No parameter is trainable.

One representative state/transition is fixed at the midpoint of each of the
ten primary training blocks and three primary validation blocks. For every
integer `z` shift `s=0,...,87`, calculate:

```text
codec_error(s) = ||A(T_s x) - T_s A(x)|| / ||T_s A(x)||
H1_error(s)    = ||F(T_s context) - T_s F(context)|| / ||T_s F(context)||.
```

Every history frame would be rolled by the same shift; H1 has one frame.
Report equal-field standardized and per-field errors, reconstruction/forecast
error against correspondingly rolled truth, shift modulo the total DCAE
toroidal packing stride four, and Fourier amplitudes of the 88-point error
curve. The primary stride-periodicity statistic is the between-class range of
mean error for `s mod 4`, divided by the overall median nonzero-shift error.
Low reconstruction error is not treated as evidence of equivariance.

## 10. H5 matched representation audit

The primary comparison fixes total real coefficient counts
`B in {32,64,128,256,416}`. Each method reports actual real coefficients,
learned basis float-equivalents, fixed-transform metadata, index storage,
training compute, and reconstruction compute. No method may exceed the global
PCA learned-basis storage at the same `B`; methods with lower storage are not
artificially padded. A secondary storage-frontier table reports the largest
available configuration under each global-PCA basis-storage budget.

The representations are:

1. source-centered fixed global PCA/KL;
2. toroidal Fourier-separated complex KL, fitted independently by stored `k`,
   coupling fields and `x,y`, retaining complete complex coefficient pairs,
   and allocating directions by source variance per real coefficient;
3. a three-level separable orthonormal Haar multiscale transform followed by
   source-fitted KL independently within its 22 coarse/detail subbands;
4. overlapping local PCA on `(x,y,z)=(16,8,88)` patches with strides
   `(8,4,88)`, deterministic partition-of-unity blending, no wrap in `x/y`,
   and greedy source-variance allocation;
5. the truth-assisted shared-shift-aligned global PCA;
6. shift-aligned Haar-subband KL if its required arrays pass the same closure
   tests and compute remains within the frozen job limit.

All transforms must pass known-answer round trips. Fourier and wavelet
allocation never uses target-block energy. Evaluation includes within-source
reconstruction and chronological transfer, residual variance capture,
spectral-band reconstruction, cross-field covariance, density--potential
cross-spectrum, local transport covariance, and integrated transport variance.
The primary selection quantity is transfer to later blocks, not training
reconstruction.

## 11. H6 context dependence

### 11.1 Causal context features

Features use only information available no later than context time `t-1`:

- per-field spatial mean (with potential gauge removed where appropriate) and
  fluctuation RMS;
- eight fixed radial-bin means per field and seven adjacent-bin gradients;
- per-field powers in `k=1..3`, `k=4..5`, `k=6..7`, and `k>=8`;
- sine/cosine phase summaries for material `k=4,5` when their aggregate
  amplitude is finite;
- the shared and per-field displacement from `t-2` to `t-1`, including peak
  ambiguity;
- existing causal regime indicators available in the input view;
- normalized absolute frame index only in the explicitly labeled time-only
  drift baseline.

All feature transforms and scales are fitted on training targets only.

### 11.2 Residual targets and probes

Targets include per-field residual energy; per-field low/mid/high toroidal
energy; same-cell cross-field covariance products; forecast-truth
density/pressure--potential phase and coherence errors; local transport error
energy; integrated transport squared error; oracle displacement; and alignment
gain.

Compare:

- training-mean constant;
- time-only ridge;
- context ridge with alphas `1e-4,1e-2,1,1e2`;
- one deterministic depth-two CART-style regression tree with minimum leaf 24
  and training-quantile split candidates.

Ridge alpha is selected by nested rolling-origin validation inside the training
blocks. Outer reporting uses training prefixes of at least five blocks and the
next block; final validation reporting fits the complete training region and
scores all three validation blocks separately. Metrics are out-of-block `R^2`,
MAE, normalized RMSE, and improvement over constant and time-only baselines.

Block-permutation sensitivity uses 200 permutations of complete 42-sample
target blocks with seed `2026081936`; order inside a block is preserved. It is
a descriptive null check, not an IID p-value.

### 11.3 B5 conditioning-collapse sensitivity

Because the locked B5 checkpoint exists, a fixed-seed context-shuffle test is
required unless exact checkpoint reload fails. Select the three offsets
`5,10,15` in each of the previous six 21-target validation blocks (18 targets).
Pair every selected target with the context/H1-conditioning pair 63 targets
away modulo the 126-target validation interval. Use that target's first eight
immutable scientific sampler seeds.

The correct-context baseline uses the corresponding stored B5 members. The
shuffled model receives the mismatched context and mismatched conditioning H1
mean, but its generated residual is added back to the original target's H1
mean. This isolates changes in the residual distribution from changes in the
deterministic center. Compare generated field variance, spectral-band
covariance, cross-field covariance, and member-wise local/integrated transport
covariance. No target truth enters generation. This is sensitivity analysis,
not retraining or a forecast candidate.

## 12. H7 state completeness and memory

The existing engineering dataset contains the exact saved-state candidate
`[Ne,Pe,Pi,NVe,NVi,Vort]+Bphi` and pragmatic C5P. Phase 3.5 does not rerun the
failed exact-state codec or train a new exact-state model.

For state-neighborhood probes, construct deterministic compact embeddings by
averaging `x,y` into `8 x 8` bins and retaining coupled toroidal Fourier
coefficients `k=0..7`. Distances are whitened with training-only scales.
Compare C5P against C5P plus omitted `NVe,Vort,Bphi`; `NVi` is reported but is
not counted as new information because it is algebraically reconstructible
from `Ne,Vi` on 85604. Neighbors must precede the query chronologically and be
at least 42 frames away. Report `k=5,10,20` neighbor conditional variance and
future-increment/residual prediction.

Regularized state probes use the H6 chronological machinery and compare:

1. current C5P features;
2. C5P plus summaries of `NVe,Vort,Bphi`;
3. current C5P plus causal delays `1,2,4,8,16` frames where available;
4. exact-state summaries plus the same delay policy as a companion diagnostic.

No conditional-mutual-information number is promoted to primary evidence in
this one-trajectory, high-dimensional setting. The previous C5P-H2 result is
retained as evidence against one specific extra-frame architecture, not as a
proof that all memory is irrelevant.

## 13. Evidence classification and decision logic

Thresholds below are engineering evidence tiers, not universal physical laws
or hypothesis-test significance levels.

- **Nonstationarity strong:** at least three independent raw-state/transport
  summaries have first-to-last block shifts above `0.5` pooled temporal SD and
  exclude zero under the primary block interval, or a time-only probe attains
  validation `R^2 >= 0.10` for at least two target families.
- **Coherent transport strong:** at least 75% of consecutive pairs have an
  unambiguous common peak, the median magnitude lies in 7--14 cells, alignment
  reduces H1 residual energy by at least 25%, and either validation full-span
  capture rises by 15 percentage points or transported persistence improves
  by 20% over unshifted persistence.
- **Non-equivariance strong:** median nonzero-shift equivariance error is at
  least 25% of the corresponding unshifted reconstruction/forecast error, or
  the modulo-four error-class range is at least 20% of median equivariance
  error, consistently in at least 10 of 13 states.
- **Representation strong:** a non-global method improves later-block variance
  capture by at least 10 percentage points at two coefficient budgets and
  improves at least two dependence/transport families without using target
  energy for selection.
- **Context dependence strong:** context probes improve validation `R^2` by at
  least `0.10` over constant and at least `0.05` over time-only for two target
  families with improvement in all three validation blocks; B5 shuffle changing
  a covariance family by at least 10% is corroborating, not sufficient alone.
- **State incompleteness strong:** exact-state features reduce neighbor
  conditional variance or normalized probe RMSE by at least 10% for two target
  families in all validation blocks.
- **History strong:** delay features improve normalized probe RMSE by at least
  10% or validation `R^2` by at least `0.05` over current C5P for two target
  families in all validation blocks.
- **Sampling insufficiency strong:** primary ESS is below 20 for at least half
  of the material residual/mode/transport observables and prefix learning
  curves have not plateaued, defined as more than two percentage points of
  capture gain between the last two source prefixes. ESS below 40 or only one
  of those conditions is moderate evidence.

Mechanisms are ranked first by evidence tier, then by effect size normalized
to their threshold, then by the mandated priority order. `unexplained failure`
is last unless no mechanism reaches moderate evidence.

The decision memo ends with exactly one action, selected in this priority:

1. invalid protocol/state or strong nonstationarity -> `repair interval/conditioning`;
2. unavailable required exact state -> `construct exact-state dataset`;
3. strong coherent transport or non-equivariance -> `build an equivariant transport-plus-innovation model`;
4. strong context/history or multiscale transfer -> `strengthen context-conditioned multiscale residual generation`;
5. sampling insufficiency without an earlier strong mechanism -> `request independent Hermes restarts`;
6. otherwise -> `stop model development and write the benchmark/failure result`.

The recommended action is not authorized for implementation by Phase 3.5.

## 14. Execution, outputs, and integrity

All data-bearing work runs on Rocky 9 Rusty. One clean-commit H100 job may use
at most one H100, 24 CPU cores, 128 GB memory, and eight hours. If split into
stages for scheduler reliability, every stage uses the same clean commit and
unique non-overwriting result directory, and the final manifest records the
dependency chain. Online W&B is required for compact progress/provenance only;
raw fields, forecasts, checkpoints, representations, figures, and tables stay
on Ceph.

Required tracked or hash-locked outputs are:

- `paper0/phase3_5/DATA_STATE_AUDIT.md`;
- `stationarity_summary.csv`;
- `effective_sample_size.csv`;
- `block_transfer_matrix.csv` and a labeled heatmap;
- `translation_diagnostics.csv`;
- `equivariance_audit.csv`;
- `representation_transfer.csv`;
- `context_dependence.csv`;
- publication-quality PNG and SVG figures with captions;
- raw sufficient statistics needed to regenerate compact tables/figures;
- a machine-readable run manifest with commands, seeds, inputs, outputs,
  hashes, code revision, dirty state, environment, Slurm identity, runtime,
  and W&B URL;
- `paper0/phase3_5/PHASE3_5_DECISION_MEMO.md`.

Before scientific output, tests must cover scope/path guards, exact ranges,
nonperiodic-axis shift rejection, normalized circular correlation, subcell
stability, equivariance formulas, moving-block resampling, ESS known answers,
PCA transfer, principal angles, Fourier complex pairing, Haar round trip,
patch blending, coefficient/storage accounting, chronological probe splits,
block permutations, fixed-seed context shuffling, and member-wise nonlinear
transport.

Every output is written exclusively and atomically. Existing artifacts are
never overwritten. Guard reads, a path or metadata record indicating 85606,
hash mismatch, target-order mismatch, nonfinite required values, future truth
entering a deployable feature, model training, or a circular non-`z` shift
stops the run without a scientific decision.

## 15. Deviations from the previous protocol

This amendment prospectively changes only the post-K4 diagnostic boundary:

1. K4 remains `K4_training_residual_span_does_not_transfer`; its metrics,
   thresholds, files, and interpretation are unchanged.
2. The previous K4 stop rule is narrowed to permit one diagnostic
   cause-localization phase on 85604 before choosing between new data and a
   benchmark/failure paper.
3. Frozen H1 checkpoint inference is newly permitted only for toroidal
   equivariance. Frozen B5 inference is newly permitted only for the fixed-seed
   context-shuffle sensitivity.
4. Validation remains development-only conditional-transient evidence; it is
   not relabeled stationary or independent.
5. No new model selection, training loss, architecture arm, assimilation,
   ranking, rollout, or held-out access is added.

After Phase 3.5 results are first inspected, this document may not be edited.
Any correction must be a separately dated amendment that states whether a
scientific result had already been produced.
