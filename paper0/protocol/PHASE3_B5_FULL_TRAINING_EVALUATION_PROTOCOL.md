# Phase 3 B5 full-training and one-step evaluation protocol

**Decision status:** frozen after bounded B5 smoke job `6901469` passed and
before full B5 training, validation access, checkpoint selection, scientific
forecast generation, or B5 evaluation implementation

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Newly authorized scope:** one full seed-1701 B5 training run, one bounded
four-target evaluator smoke, one M32 one-step evaluation on the existing 85604
validation interval, and one prospective one-seed gate

The machine-readable authority is
`paper0/manifests/phase3_b5_full_training_evaluation_85604.json`.

## 1. Decision boundary

Job `6901469` established that the exact joint full-field residual U-Net,
EDM loss, periodic toroidal path, optimizer, checkpoint reload, 18-step Heun
sampler, canonical ensemble composition, and online W&B tracking execute on a
Rocky 9 H100. It completed 64 updates on eight training targets, reduced a
fixed denoising probe by 2.37%, used 1.2504 GiB, and produced distinct finite
members. Those are mechanics, not scientific estimates.

This protocol is the first B5 experiment permitted to read the later 85604
validation interval. It freezes the complete training budget, checkpoint
selection rule, validation corruption bank, scientific sampling bank,
comparators, metrics, and thresholds before that access. It does not authorize
schedule tuning, architecture tuning, additional seeds, autonomous rollout,
assimilation, diagnostic ranking, steering, or 85606.

## 2. Hypotheses and failure interpretation

B5 tests a deterministic-mean plus joint stochastic-residual decomposition:

\[
x_t = \mu_{\mathrm{H1}}(x_{t-1}) + r_\theta(x_{t-1},\mu_t,\epsilon).
\]

The one-step hypotheses are separated.

### H-mean

Does the ensemble mean retain or improve the frozen H1 parent's field skill
and transport-relevant joint structure?

### H-prob

Do independent residual samples produce calibrated uncertainty for fields,
material modes, cross-spectra, and member-wise transport?

A low denoising loss is not either hypothesis. Nonzero diversity is not
calibration. Marginal field calibration with wrong density-potential
cross-phase or ExB transport is a scientific failure and evidence that the
learned joint residual distribution is inadequate. A one-step pass would
authorize an O3/O4 protocol, not establish rollout fidelity by itself.

## 3. Literature motivation and limits

The architecture remains a minimal field-coordinate analogue of CorrDiff's
regression-plus-residual-diffusion decomposition, not a new core architecture
claim. The official CorrDiff implementation supports held-out validation loss
and recommends selecting among saved checkpoints after the complete run rather
than relying on built-in early stopping. The EDM reference implementation
similarly separates loss monitoring from periodic sample-quality evaluation
and uses network snapshots plus EMA. GenCast motivates a new independent
initial noise draw at every future autoregressive step.

Primary references are:

- Mardani et al., *Residual Corrective Diffusion Modeling for Km-scale
  Atmospheric Downscaling*, <https://arxiv.org/abs/2309.15214>;
- the official PhysicsNeMo CorrDiff implementation notes,
  <https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html>;
- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative
  Models*, <https://arxiv.org/abs/2206.00364>, and its official implementation,
  <https://github.com/NVlabs/edm>;
- Price et al., *Probabilistic weather forecasting with machine learning*,
  <https://www.nature.com/articles/s41586-024-08252-9>.

These references use orders of magnitude more data. PhysicsNeMo describes at
least tens of thousands and commonly millions of processed samples, while B5
has 430 adjacent target states from one simulation. Repeated corruptions are
not new independent plasma realizations. The protocol therefore treats
overfitting and conditional-distribution non-identifiability as central risks.

## 4. Immutable data and parent means

Only the verified 85604 model dataset from job `6893525` is used:

- fields: `[Ne,Pe,Pi,phi,Vi]`;
- context: exactly standardized frame `t-1`;
- target: standardized frame `t`;
- training targets: `[2,432)`, 430 total;
- guard: `[432,496)`, never loaded;
- validation targets: `[498,624)`, 126 total;
- cadence: `3.131905426352636` microseconds;
- field shape: `[5,64,32,88]`;
- toroidal periodicity: `zperiod=5`;
- Fourier mapping: `n=5k`.

Absolute time, normalized frame index, shot label, diagnostics, region masks,
future truth, and physics-derived quantities are prohibited inputs. Time is
metadata only. Temporal windows are not counted as independent physical
shots.

The deterministic mean remains the exact frozen C5P-H1 seed-1701 checkpoint
with SHA-256
`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.
It is never trained or reselected by B5.

Training means come from the truth-separated job-6901393 artifact with
SHA-256
`d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`.
Validation means come from the already closed Phase 2 H1 artifact with
SHA-256
`a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`.
Both contain forecasts produced from context before target truth was opened.

For target `t`, define

\[
\mu_t = \mu_{\mathrm{H1}}(x_{t-1}),
\qquad
r_t = x_t-\mu_t,
\]

\[
c_t = \operatorname{concat}(x_{t-1},\mu_t).
\]

The residual target is divided without centering by the training-only global
population standard deviations:

| field | residual scale |
|---|---:|
| `Ne` | 0.05503048051260375 |
| `Pe` | 0.04825854004472835 |
| `Pi` | 0.06096460194410047 |
| `phi` | 0.04632595196855943 |
| `Vi` | 0.10251610501339582 |

No validation statistic changes these scales. No mean, regional pattern, or
axisymmetric component is removed.

## 5. Frozen architecture and EDM objective

The model is freshly initialized with seed 1701. The non-scientific smoke
checkpoint is not a warm start. The architecture is unchanged from the smoke:

- one joint five-field normalized residual;
- ten dynamic condition channels, `x_(t-1)` plus frozen H1 mean;
- static normalized `x/y` position channels and no absolute `z` coordinate;
- base width 32 and multipliers `[1,2,4,4]`;
- two residual blocks per encoder and decoder resolution;
- three all-axis downsamplings to `8x4x11`;
- zero padding in nonperiodic `x/y` and circular operations in `z`;
- periodic linear-`z` plus bilinear-`x/y` upsampling;
- GroupNorm, SiLU, FiLM noise conditioning, no dropout, no attention;
- 11,604,709 parameters;
- no DCAE, latent bottleneck, spectral operator, physics loss, or trainable
  deterministic parent.

Let `z=r/s` be the normalized joint residual and let `sigma_data=1`. EDM uses

\[
c_{\mathrm{in}}=(\sigma^2+1)^{-1/2},
\quad
c_{\mathrm{skip}}=(\sigma^2+1)^{-1},
\]

\[
c_{\mathrm{out}}=\sigma(\sigma^2+1)^{-1/2},
\quad
c_{\mathrm{noise}}=\tfrac14\log\sigma.
\]

For `z_sigma=z+sigma epsilon`,

\[
D_\theta(z_\sigma;c_t,\sigma)
=c_{\mathrm{skip}}z_\sigma
+c_{\mathrm{out}}F_\theta(c_{\mathrm{in}}z_\sigma,c_t,c_{\mathrm{noise}}).
\]

Training samples

\[
\log\sigma\sim\mathcal N(-1.2,1.2^2),
\qquad
\epsilon\sim\mathcal N(0,I),
\]

and minimizes only

\[
\mathcal L_{\mathrm{EDM}}
=\frac{\sigma^2+1}{\sigma^2}
\operatorname{mean}\left[(D_\theta-z)^2\right].
\]

The mean gives equal weight to elements and to the five residual-normalized
channels. Flux, spectrum, cross-phase, coherence, conservation, PDE residual,
region, event, diagnostic, CRPS, or other physics/scientific metric cannot
enter the training loss.

## 6. Frozen full-training budget

The exact full run is:

- seed: 1701;
- epochs: 100, always completed;
- targets per epoch: all 430 exactly once;
- target presentations: 43,000;
- no toroidal-roll augmentation;
- microbatch: one target;
- gradient accumulation: four targets;
- final partial accumulation: two targets, divided by two;
- optimizer steps per epoch: 108;
- total optimizer steps: 10,800;
- optimizer: AdamW, betas `(0.9,0.99)`, zero weight decay;
- global gradient clip: one;
- EMA decay: 0.999 after every optimizer step;
- precision: H100 bfloat16 autocast, FP32 loss, optimizer state, and EMA;
- TF32: disabled;
- warmup: none;
- early stopping: prohibited.

A single `PCG64(67501)` generator produces a `[100,430]` int64 matrix of
successive permutations of targets `2..431`. Its raw C-order SHA-256 is
`4eb79c67e03623ccb5e0b1735ff0d3a13c1202db833d0c42509af3ba7b0eafda`
and its `.npy` SHA-256 is
`0e775e59e3596e63c2324a9a9fa5ff82df9dca1ff9d4923fbbef3a4126e97806`.

Training corruption is deterministically keyed by seed `67502`, zero-based
epoch, target frame, and constant `0xB5ED0003`. Each key initializes a fresh
PCG64 stream, draws one log-noise scale, and then one complete float32
`[5,64,32,88]` standard-normal field. The bank is regenerated from keys and is
not stored as 43,000 full volumes. Resumption must reproduce the same order,
corruptions, optimizer, scaler, and EMA states.

Let `j` be the zero-based optimizer update in `0,...,10799`. The update uses

\[
\eta_j=10^{-6}+\frac{10^{-4}-10^{-6}}{2}
\left[1+\cos\left(\frac{\pi j}{10799}\right)\right].
\]

The first update therefore uses `1e-4` and the last uses `1e-6`.

## 7. Validation and checkpoint selection

Completed epochs `5,10,...,100` are the only 20 candidates. Every candidate
uses EMA weights, all 126 validation targets, and four fixed corruptions per
target. `PCG64(67503)` produces the `[126,4]` uint64 seed bank in chronological
target/probe order. Its raw SHA-256 is
`f0e736a16be18289ef64fc190fac917eda284eac13ed5117fa7be2d7c2b7d411`
and its `.npy` SHA-256 is
`fca7f1254b28fda0a1dad91aea4e1e8ce2faef5dbe7484e13c86ee885e5a5e12`.

Each seed initializes PCG64, draws `log(sigma)` under the frozen distribution,
then draws one full float32 noise field. The selection metric is the mean EDM
denoising loss over all `126*4=504` fixed probes. This is a data-only
score-matching objective in normalized residual coordinates.

After all 100 epochs finish, select the earliest candidate with the
numerically lowest validation metric. No sampled forecast, RMSE, CRPS,
spread, spectrum, cross-phase, transport, event metric, W&B value, or 85606
result may alter selection. This deliberately separates fitting/checkpoint
selection from the later scientific verdict. The validation interval remains
a development set, so its final scientific scores are not an unbiased test.

The selected EMA checkpoint and the final resumable raw training state are
separate artifacts. A fixed FP32 denoiser probe must reload bit for bit.

## 8. Truth-separated scientific ensemble

Scientific sampling uses an independent `PCG64(67532)` uint64 bank with shape
`[126,32]`, ordered by chronological target and member. Its raw SHA-256 is
`dcd4eb49682e5783508e423951108a4b47afeb103e1c3d9fcdcb1bae88b8ec19`
and its `.npy` SHA-256 is
`013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14`.
Each seed initializes one independent full normalized-residual noise field.

The frozen sampler is the deterministic EDM probability-flow ODE solver with
18 Karras steps from sigma 80 to 0.002, rho seven, Heun correction, and zero
churn. It costs 35 network evaluations per member. The generated artifact has
axes

~~~text
[target, ensemble_member, future_time, channel, x, y, z]
~~~

and exact shape `[126,32,1,5,64,32,88]`. Its uncompressed float32 payload is
14,533,263,360 bytes. Stored member order defines immutable M4, M8, M16, and
M32 prefixes. Regeneration, recentering, inflation, clipping, member rejection,
member sorting, interaction, or post-hoc calibration is prohibited.

A context-only reader obtains `x_(t-1)`; the closed Phase 2 artifact supplies
`mu_t`; the selected B5 model samples the normalized residual and composes the
five fields. The forecast closes and hashes before any target truth reader is
constructed. Nonlinear quantities are later computed for each member, never
only from ensemble-mean fields.

The four-target evaluator smoke uses targets `[498,502)` and all 32 members.
Its metrics are non-scientific. It must pass tensor, hash, truth-separation,
sampler-prefix, scorer, and memory gates before the full 126-target evaluation.

## 9. Frozen metrics and comparators

The primary comparator is the exact H1 mean underlying B5. Its validation
forecast SHA-256 is
`a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`
and its truth-separated score SHA-256 is
`ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593`.
Persistence and the training-only toroidal spectral AR(1) remain uncompressed
references. B2, B3, and B4 are descriptive context, not matched rescue
comparators. Nothing is retrained or reselected.

The byte-locked scoring components are:

- `b2_probabilistic_metrics.py`: `edef6fbb...b6650`;
- `b2_field_metrics.py`: `c2d0f5e7...0a0db`;
- `b2_spectral_metrics.py`: `382fc683...761e`;
- `b2_transport_metrics.py`: `b78ea33f...b02e`;
- `b2_scoring.py`: `2dfdf6f7...118fe`;
- `geometry.py`: `4f5eda70...424c`;
- `codec_transport.py`: `201a9628...e46d45`.

The full manifest stores every complete hash. Metrics include field RMSE, MAE,
bias, variance, anomaly correlation, fair and empirical CRPS, corrected
spread-skill, coverage, and rank; directional/toroidal spectra; material-band
power and realization coherence; member-wise cross-spectrum amplitude,
phase, and coherence; and authoritative member-wise radial ExB particle and
electron, ion, and total internal-energy transport.

For marginal `phi`, subtract the full spatial mean separately from truth and
every member at every target. Frozen toroidal bands are `k=1..3`, `k=4..5`,
and `k=6..7`, mapping to `n=5..15`, `n=20..25`, and `n=30..35`. Materiality
uses training data only.

Temporal robustness uses six 21-frame blocks:
`[498,519)`, `[519,540)`, `[540,561)`, `[561,582)`, `[582,603)`, and
`[603,624)`. Moving-block bootstrap intervals use block length 21, 2,000
replicates, six sampled blocks, 106 starts, and seed `85604032`. They quantify
temporal sampling uncertainty conditional on 85604, not cross-shot
uncertainty.

## 10. Prospective one-seed acceptance gate

The numerical thresholds are exactly the frozen B3 gate thresholds. Reusing
them prevents model-specific relaxation after B3/B4 failed.

### 10.1 Integrity

1. Exactly 100 epochs, 43,000 target presentations, 10,800 optimizer updates,
   20 candidate evaluations, earliest-minimum selection, finite values, and
   exact selected-checkpoint reload.
2. Exact frozen inputs, order and noise banks, model, protocol, data, and
   source hashes; no future truth condition, time input, physics loss, guard
   read, or 85606 access.
3. Exactly 32 finite members, canonical axes, and nonzero spread in all five
   fields and all primary regions.
4. Forecast generation finishes and hashes before target truth opens.

### 10.2 Field skill and marginal calibration

1. Equal-channel ensemble-mean RMSE and MAE are each at most 1.05 times H1.
2. Equal-channel fair CRPS is strictly below both H1 MAE and the best
   applicable uncompressed-reference MAE.
3. At least four field fair-CRPS values are strictly below H1 field MAE; the
   fifth is at most 1.05 times H1.
4. At least four fields have corrected spread-skill in `[0.80,1.25]`; the
   fifth lies in `[0.67,1.50]`.
5. At least four fields have absolute coverage error at most 0.10 for I17,
   0.08 for I27, and 0.06 for I31; the fifth may use twice those tolerances.
   Every primary region has I31 coverage in `[0.75,0.995]`.

### 10.3 Spectra and joint structure

1. Every training-material field band has member-expected power ratio in
   `[0.75,1.30]`.
2. Every material field band has ensemble-mean realization coherence at least
   0.80.
3. Every material primary cross-field band has circular cross-phase error at
   most 20 degrees and absolute coherence change at most 0.15.
4. Material band-power and cross-spectrum-projection calibration has
   spread-skill in `[0.67,1.50]` and I31 coverage in `[0.75,0.995]`.

### 10.4 Transport

For all four strict-face quantities: relative L2 at most 0.40, correlation at
least 0.70, and weighted sign disagreement at most 0.20.

For all four confined-separatrix quantities: relative L2 at most 0.30,
absolute normalized bias at most 0.15, correlation at least 0.80, and weighted
sign disagreement at most 0.15.

At least three separatrix transport fair-CRPS values must beat H1 absolute
error; the fourth may be at most 1.05 times H1. At least three separatrix
quantities must have spread-skill in `[0.67,1.50]`, I27 coverage within 0.12
of nominal, and I31 coverage within 0.10; the fourth remains finite and
noncollapsed. Every eligible upper-decile event magnitude relative error is at
most 0.50 with weighted sign disagreement at most 0.25.

### 10.5 Monte Carlo and temporal stability

For every primary aggregate field, material-band, cross-field, and
separatrix-transport fair-CRPS component,

\[
|q_{16}-q_{32}|\le 0.10|q_{32}|+10^{-8}.
\]

Every family must pass overall and in at least five of six chronological
blocks. No prefix or block may rescue an overall failure.

## 11. Stop/go rule

All integrity, field, joint-physics, transport, Monte Carlo, and temporal
conditions are required. A full pass is provisional one-seed development
evidence and authorizes writing an O3/O4 protocol. It does not itself authorize
an O3/O4 launch, more seeds, or 85606.

If marginal calibration passes but joint spectra/cross-phase/transport fails,
record that marginal residual diffusion is insufficient. Do not tune residual
scales, noise schedule, learning rate, sampler, or thresholds on the result.

If H-mean, integrity, or calibration fails, stop B5 and localize the failure
before any replication. If M32 is Monte Carlo unstable, retain the failure and
write a separately frozen larger-ensemble sensitivity; do not regenerate M32.

A genuine metric/software bug requires a documented amendment, preservation
of the original result, and a consistent rerun of every affected model. It
does not permit scientific retuning.

## 12. Execution and reproducibility

Training and forecast generation use one Rocky 9 H100 submitted from
`rusty9`. Online W&B is mandatory but is only a monitoring mirror. Local Ceph
configuration, order/noise banks, histories, checkpoints, forecasts, raw
scores, block metrics, compute records, logs, and SHA-256 inventories are the
scientific authority. Large artifacts remain out of Git; compact results and
their hashes are committed.

Required compute reporting includes parameter count, target presentations,
optimizer updates, accelerator hours, peak memory, sampler calls, ensemble
size, forecast bytes, inference time, and cost per forecasted physical
microsecond.

Before execution, the repository must contain and test:

1. exact full config, order matrix, keyed corruption, EMA, cosine schedule,
   partial accumulation, validation-bank, and checkpoint-selection mechanics;
2. a full-only entrypoint that cannot select another seed or smoke mode;
3. strict training and validation mean-artifact readers;
4. context-only M32 generation with canonical axes and immutable prefixes;
5. a separate scorer using the frozen numerical engine;
6. B5-specific acceptance reduction with known-answer fixtures;
7. a four-target M32 evaluator smoke;
8. complete local and Rocky 9 suites and exact source/input hashes.

## 13. Claims boundary

The strongest possible result from this rung is:

> On the later 85604 development interval, one seed of a full-field joint
> residual EDM produced a useful one-step conditional ensemble while retaining
> the frozen H1 mean's field skill and preserving transport-relevant joint
> diagnostics well enough to merit rollout evaluation.

It would not establish autonomous rollout, held-out 85606 performance, a
validated architecture, cross-shot generalization, experimental diagnostic
realism, assimilation value, diagnostic ranking, or steering.
