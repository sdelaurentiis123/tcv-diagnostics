# Phase 3 B2 full-training and probabilistic-evaluation protocol

**Decision status:** frozen after the bounded B2 implementation smoke and
before full B2 training, checkpoint selection, scientific ensemble generation,
or probabilistic metric implementation

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Newly authorized scope:** matched full training of three B2-LDM-H2 seeds and
one-step probabilistic evaluation on the existing 85604 validation interval

The machine-readable authority is
`paper0/manifests/phase3_b2_full_evaluation_85604.json`.

## 1. Decision boundary

Rocky 9 H100 job `6896402` established only that the frozen codec, masked
denoising objective, optimizer, checkpoint format, real Azula sampler, W&B
mirror, canonical forecast axes, member diversity, and same-device checkpoint
reload work together. The exact compact smoke result has SHA-256
`fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360`.

The smoke did not estimate forecast skill, calibration, spectra, cross-phase,
or transport. Its two-epoch loss is not a scientific baseline. This protocol
therefore freezes the scientific questions and all acceptance rules before the
three full B2 models exist.

This phase asks:

1. Does the stochastic conditional forecast retain acceptable point skill?
2. Does its 32-member ensemble behave like useful independent draws rather
   than arbitrary nonzero noise?
3. Does the ensemble preserve the joint field structure required by spectra,
   cross-phase, and radial ExB transport?

It remains an O2-style one-saved-step experiment. It does not authorize
autoregressive O3/O4 rollout, assimilation, diagnostic ranking, model
selection on 85606, or any control claim.

## 2. Literature-locked scoring convention

The probabilistic score follows the negatively oriented CRPS convention where
lower is better. Gneiting and Raftery define CRPS as a proper score and show
its expectation representation. For an ensemble of conditionally independent,
exchangeable members `x_1,...,x_M` and truth `y`, Paper 0 uses the fair
finite-ensemble estimator:

\[
\operatorname{fCRPS}_M
=
\frac{1}{M}\sum_{m=1}^{M}|x_m-y|
-
\frac{1}{M(M-1)}\sum_{m<m'}|x_m-x_{m'}|.
\]

This is the unbiased U-statistic form of the expectation representation. B2
members satisfy the required structural assumption: they share a context and
trained distribution, but use independent initial Gaussian noise and never
interact across the ensemble dimension. The 2026 ensemble-size analysis by
Roberts warns that a fair score need not remain fair when members are coupled;
Paper 0 therefore prohibits cross-member attention, shared initial noise, and
member-dependent post-processing in this arm.

The ordinary empirical-distribution CRPS is also reported:

\[
\operatorname{eCRPS}_M
=
\frac{1}{M}\sum_{m=1}^{M}|x_m-y|
-
\frac{1}{M^2}\sum_{m<m'}|x_m-x_{m'}|.
\]

Only fCRPS is used for the primary B2 probabilistic gate. eCRPS is retained so
published results can be compared with software that evaluates the empirical
ensemble CDF directly. For a deterministic forecast both scores reduce by
definition to absolute error. No score is silently switched after results are
seen.

The sorted-ensemble pairwise identity must be used for field-scale streaming;
forming a 32-by-32 pairwise tensor for every voxel is prohibited. Known-answer
tests must compare the sorted implementation to the explicit pairwise formula.

Primary method references are:

- Gneiting and Raftery, *Strictly Proper Scoring Rules, Prediction, and
  Estimation*, JASA 2007, DOI `10.1198/016214506000001437`;
- Lang et al., *AIFS-CRPS*, arXiv `2412.15832`, for finite-ensemble scoring,
  spread-skill, and spectral evaluation of stochastic forecasts;
- Roberts, *Ensemble-size-dependence of deep-learning post-processing methods
  that minimize an (un)fair score*, arXiv `2602.15830`, for the exchangeability
  limitation of fair scores;
- Rozet et al., *Lost in Latent Space*, NeurIPS 2025 / arXiv `2507.02608`, for
  the attributed LOLA control;
- Diaconu et al., *Probabilistic Retrofitting of Learned Simulators*, arXiv
  `2603.01949`, as a separate B3 motivation, not evidence that B2 must work.

## 3. Immutable full B2 training

Full training uses the implementation and model definition already frozen in
`PHASE3_B2_LDM_PROTOCOL.md`. No architecture or loss change is authorized.

For seeds 1701, 1702, and 1703 independently:

- representation: same-seed accepted `C5P-dcae_l10` codec;
- fields: `[Ne,Pe,Pi,phi,Vi]`;
- exact context: two preceding frames;
- target: one next frame;
- train targets: `[2,432)`, all 430 once per epoch;
- validation targets: `[498,624)`, all 126 chronologically every epoch;
- latent moments: fitted on `[0,432)` only for that seed's frozen codec;
- epochs: 200;
- microbatch: 1;
- gradient accumulation: 16;
- optimizer steps per epoch: 27;
- total optimizer steps: 5400;
- AdamW learning rate: `1e-4`;
- betas: `[0.9,0.99]`;
- weight decay: zero;
- schedule: cosine from `1e-4` to exactly zero over optimizer steps 1 through
  5400, without warmup;
- gradient clip: 1;
- precision: bfloat16 autocast;
- training objective: complete three-frame LOLA EDM denoising MSE;
- physics-derived loss: prohibited;
- early stopping: prohibited.

Checkpoint selection remains the earliest epoch with the numerically lowest
complete-trajectory denoising loss on all 126 fixed-noise validation targets.
Context- and target-slot losses are logged but cannot select a checkpoint.
No field, spectrum, cross-phase, transport, CRPS, coverage, or 85606 value can
select the checkpoint.

All three models are trained even if an earlier seed appears poor. Training
completion is not model acceptance.

## 4. Immutable validation cases and comparators

Scientific evaluation uses target frames `[498,624)` exactly once in
chronological order. The six non-overlapping reporting blocks remain:

~~~text
[498,519), [519,540), [540,561),
[561,582), [582,603), [603,624)
~~~

Each block contains 21 saved steps, or approximately 65.77 microseconds. This
is longer than the frozen representative full-pattern first-`1/e` crossing of
2.244 frames and slightly longer than the slow `Ne` crossing of 19.042 frames.
Intervals and bootstrap claims remain conditional on this single development
run.

Comparators are immutable:

1. uncompressed persistence;
2. uncompressed two-frame linear extrapolation;
3. training-only toroidal spectral AR(1);
4. the paired selected deterministic C5P-H2 checkpoint at the same model seed.

The deterministic forecasts and physics scores from job `6896117` remain
historical evidence. If a new shared probabilistic scorer needs additional
deterministic scalar outputs, it must consume the frozen deterministic
forecast artifacts or reproduce them byte-for-byte before extension. It may
not retrain or reselect deterministic H2.

## 5. Truth separation and immutable forecast artifacts

Forecast generation and truth scoring remain separate:

1. a context-only loader reads frames `t-2,t-1` and target identity `t` but
   does not open target field datasets;
2. the selected checkpoint generates and writes the complete ensemble;
3. the forecast file is closed, hashed, and recorded;
4. only the separate scorer opens target truth and geometry products.

For each B2 seed, the canonical forecast has axes:

~~~text
[target=126, member=32, future_time=1, channel=5, x=64, y=32, z=88]
~~~

and `float32` decoded standardized fields. Member order is immutable. For each
target and model seed, sampler initialization uses a frozen stateless CPU seed
derived from `[model_seed,target_frame,0x42324556]`; the resulting 32-member
forecast is generated once. Prefixes of that stored ensemble define the
member-count sensitivity at `M in {4,8,16,32}`. Regenerating different-sized
ensembles is prohibited because GPU random-number batching need not be prefix
stable.

The scientific sampler is Azula 0.3.1 Adams-Bashforth, 16 steps, order three,
start one, stop zero. The model is in evaluation mode. Dropout is disabled.
Every initial member noise tensor is independent. No member sorting,
calibration transform, recentering, variance inflation, clipping, or rejection
sampling is allowed before primary scoring.

## 6. Field coordinates, potential gauge, and geometry regions

Primary equal-channel field scores use the frozen standardized model
coordinates. This means transformed/log-standardized `Ne` and standardized
`Pe`, `Pi`, `phi`, and `Vi`. Physical-unit field metrics are secondary and use
the exact inverse transform without clipping.

Electrostatic potential has an additive gauge freedom. Before every marginal
`phi` error, CRPS, spread, coverage, or rank calculation, subtract one spatial
mean separately from truth and from each member at each target frame. Raw
stored-gauge `phi` metrics are descriptive only and cannot rescue or fail the
model. Toroidal `k>0` cross spectra and ExB transport are already invariant to
a spatially constant shift.

Primary field and calibration reductions use the authoritative strict
wall-interior operator cells and are reported separately for the disjoint
single-null masks:

- confined edge;
- private-flux region;
- scrape-off layer.

The following overlapping masks are secondary mandatory reports:

- separatrix cell band;
- outboard-midplane row;
- X-point topology stencil;
- inner divertor leg;
- outer divertor leg.

The masks must come from `src/tcv_diagnostics/geometry.py` and the hash-locked
geometry artifacts. Image coordinates, synthetic sensor drawings, and legacy
five-window probe labels are prohibited from defining these regions.

## 7. Point skill and marginal probabilistic metrics

For every channel and region, report member-mean RMSE, MAE, bias, population
variance ratio, and anomaly correlation. Equal-channel aggregate RMSE and MAE
use the same definitions as the deterministic O2 protocol after gauge-fixing
`phi`.

Report fCRPS and eCRPS for each channel, each primary region, the union of
eligible cells, and the equal-channel aggregate. Aggregate CRPS is the
arithmetic mean of the five channel scores; channels are not pooled in a way
that lets spatial cell counts change their weight.

For members `x_m`, ensemble mean `xbar`, unbiased member variance `s^2`, and
truth `y`, define corrected spread-skill ratio:

\[
R_{\mathrm{ss}}
=
\frac{
\sqrt{\frac{M+1}{M}\langle s^2\rangle}
}{
\sqrt{\langle(\bar{x}-y)^2\rangle}
}.
\]

The finite-member factor compares unbiased ensemble variance with the expected
error variance of an ensemble mean when members and truth are exchangeable.
Numerator and denominator are accumulated separately before division.

With sorted members `x_(1)<=...<=x_(32)`, use the following exact order-
statistic intervals and exchangeable nominal coverage:

| Name | Bounds | Nominal coverage |
|---|---|---:|
| `I17` | `[x_(8),x_(25)]` | `17/33 = 0.515151...` |
| `I27` | `[x_(3),x_(30)]` | `27/33 = 0.818181...` |
| `I31` | `[x_(1),x_(32)]` | `31/33 = 0.939393...` |

Rank histograms use 33 bins. Ties are broken by one frozen uniform draw keyed
by target, channel, cell index, and sampler-evaluation seed. No independent
sample-size claim is made from voxel counts. Report normalized rank mean,
rank-variance ratio to the discrete uniform distribution, and total variation
distance from uniform, but do not use a pixel-iid chi-square p-value.

## 8. Spectral and cross-field joint metrics

The toroidal Fourier transform is along the stored periodic 88-cell wedge,
with `zperiod=5` and full-torus mapping `n=5k`. The inherited bands remain:

~~~text
k=1..3  <=> n=5..15
k=4..5  <=> n=20..25
k=6..7  <=> n=30..35
~~~

Training-only materiality remains at least 1% of non-axisymmetric field power
or cross amplitude. Validation truth cannot select a band.

For field power, compute `|x_m,k|^2` for every member before the ensemble
average. For pairs `(Ne,phi)`, `(Pe,phi)`, and `(Pi,phi)`, compute the member-
wise cross product:

\[
S^{(m)}_{ab}(k)=a^{(m)}_k b^{(m)*}_k,
\qquad
\overline{S}_{ab}(k)=\frac{1}{M}\sum_m S^{(m)}_{ab}(k).
\]

The primary probabilistic cross-phase is `arg(overline(S_ab))`; coherence uses
member-averaged auto- and cross-spectra. The product of ensemble-mean fields is
not an admissible substitute because it removes covariance.

For every material field band, report:

- member-expected power ratio to truth;
- ensemble-mean-field realization coherence with truth;
- distribution of member-truth realization coherence;
- fCRPS, spread-skill, rank, and coverage for per-target band power.

For every material primary cross-field band, report truth and ensemble-
expected cross amplitude, circular cross-phase error, coherence change, and
per-target calibration of real and imaginary cross-spectrum projections.
Directional `x`, `y`, and toroidal spectra are mandatory reports; only the
already validated toroidal bands carry B2 acceptance thresholds at this gate.

## 9. Member-wise authoritative transport

For truth and each ensemble member independently:

1. inverse-transform `[Ne,Pe,Pi,phi]` without clipping;
2. periodically resample 88 to the authoritative native 81 toroidal cells;
3. apply the frozen geometry-aware radial ExB face operator;
4. reduce strict-face contributions and the outward confined-separatrix wedge.

The four accepted quantities are radial ExB particle, electron internal-
energy, ion internal-energy, and total internal-energy transport. Paper 0 must
not silently call these the complete experimental heat flux.

All nonlinear transport statistics are computed member by member:

\[
\mathbb{E}_m[\Gamma(x_m)]
\quad\text{is allowed, while}\quad
\Gamma(\mathbb{E}_m[x_m])
\quad\text{is not a probabilistic transport estimate.}
\]

For each quantity and reduction, report:

- ensemble-expected time series versus truth;
- relative L2, normalized bias, Pearson correlation, and truth-magnitude-
  weighted sign disagreement;
- scalar fCRPS/eCRPS at every target;
- spread-skill, rank histogram, and the three frozen interval coverages;
- pooled truth versus member distributions and 5%, 50%, 95% quantiles;
- upper-decile event-conditioned magnitude and sign performance, where the
  event threshold is fitted on 85604 training truth only.

The old synthetic-probe or ETKF rankings are not evaluated here.

## 10. Ensemble-size and Monte Carlo checks

All primary decisions use `M=32`. From the same stored members, repeat fCRPS,
eCRPS, spread-skill, and aggregate transport calibration with prefixes
`M=4,8,16`. This is a sensitivity report, not four opportunities to select a
better result.

The primary M=32 scalar metric is considered Monte Carlo stable only if the
absolute difference between M=16 and M=32 is no more than 10% of the M=32
metric magnitude plus `1e-8` in standardized units. Report failures; do not
resample a luckier ensemble. Monte Carlo instability fails the associated
acceptance component.

## 11. Conditional uncertainty intervals

The simulation run—not a voxel, window, or target—is the physical unit of
independence. Paper 0 therefore reports no sample size of `126 * pixels`.

For temporal uncertainty conditional on 85604, use a moving-block bootstrap
with block length 21 frames, 2,000 replicates, and seed 85604032. Draw starting
indices uniformly from the 106 valid length-21 blocks, concatenate six drawn
blocks, and truncate to 126 targets. Recompute nonlinear metrics inside every
replicate rather than bootstrapping already averaged ratios.

For model-seed uncertainty, report all three paired seed values plus median,
minimum, and maximum. A secondary hierarchical interval may resample model
seeds and temporal blocks, but it must be labeled as conditional on the three
trained seeds and one simulation. It cannot be described as cross-shot or
device-level uncertainty.

## 12. Per-seed acceptance gate

A seed passes only if every integrity item is true and every aggregate physics
family below passes. Components with a temporal rule must pass overall and in
at least five of six chronological blocks.

### 12.1 Integrity

1. Complete 200-epoch history, exactly 5,400 optimizer steps, frozen earliest-
   minimum checkpoint selection, finite losses/gradients, and exact checkpoint
   reload.
2. Frozen same-seed codec, training-only normalization, no clipping, no target
   read during generation, no future truth, no absolute time, no physics loss,
   and no 85606 access.
3. Exactly 32 finite, independently initialized members with canonical axes,
   nonzero spread in every field and every primary geometry region.
4. Every required scalar is finite and every required correlation is defined.

### 12.2 Field skill and CRPS

1. Equal-channel ensemble-mean RMSE and MAE are each no more than 1.05 times
   the paired deterministic H2 value.
2. Equal-channel fCRPS is strictly below both the paired deterministic H2 MAE
   and the best applicable uncompressed-reference MAE.
3. At least four of five field fCRPS values are strictly below paired
   deterministic H2 MAE; the fifth is no more than 1.05 times it.
4. At least four fields have corrected spread-skill ratio in `[0.80,1.25]` on
   the union of eligible cells; the remaining field must lie in `[0.67,1.50]`.
5. At least four fields have absolute coverage error at most 0.10 for `I17`,
   0.08 for `I27`, and 0.06 for `I31`; the remaining field may use twice those
   tolerances. Every primary region must have `I31` coverage in `[0.75,0.995]`.

### 12.3 Spectra and joint structure

1. Every training-material field band has member-expected power ratio in
   `[0.75,1.30]`.
2. Every training-material field band has ensemble-mean-field realization
   coherence with truth at least 0.80.
3. Every training-material primary cross-field band has circular cross-phase
   error at most 20 degrees and absolute coherence change at most 0.15.
4. Material band-power and cross-spectrum-projection calibration satisfies the
   field-level no-collapse bounds: corrected spread-skill in `[0.67,1.50]` and
   `I31` coverage in `[0.75,0.995]`.

### 12.4 Transport

For all four transport quantities, the ensemble-expected strict-face series
must have relative L2 at most 0.40, Pearson correlation at least 0.70, and
weighted sign disagreement at most 0.20.

For all four confined-separatrix series, relative L2 must be at most 0.30,
absolute normalized bias at most 0.15, Pearson correlation at least 0.80, and
weighted sign disagreement at most 0.15.

Additionally:

1. at least three of four separatrix transport fCRPS values are strictly below
   the paired deterministic absolute-error score; the fourth is no more than
   1.05 times it;
2. at least three of four separatrix quantities have corrected spread-skill in
   `[0.67,1.50]`, `I27` coverage within 0.12 of nominal, and `I31` coverage
   within 0.10 of nominal; the fourth must remain finite and non-collapsed;
3. each upper-decile event-conditioned magnitude relative error is at most
   0.50 and weighted sign disagreement at most 0.25.

### 12.5 Monte Carlo stability

Every primary aggregate field, material-band, cross-field, and separatrix-
transport fCRPS component satisfies the frozen M=16 versus M=32 stability
rule. No other member prefix can rescue a failure.

## 13. Architecture-level decision

B2-LDM-H2 passes this one-step gate only if:

1. at least two of three seeds pass the complete per-seed gate;
2. the median across all three seeds passes every numerical aggregate
   threshold;
3. the remaining seed has no integrity failure, ensemble collapse, non-finite
   metric, aggregate field RMSE or MAE above 1.20 times paired deterministic,
   or separatrix transport relative L2 above 0.60.

This rule was chosen before full training to avoid both seed cherry-picking
and a single numerically weak initialization vetoing an otherwise reproducible
architecture. All seed failures remain visible.

If B2 passes, it becomes eligible for a separately frozen short O3 rollout
comparison; assimilation remains unauthorized. If B2 fails calibration but
retains point/physics skill, continue to the FGN or joint-residual branch. If
it fails spectra, cross-phase, or transport, do not use its covariance for
diagnostic ranking. If it fails point skill and physics, stop B2 and continue
the baseline ladder rather than tuning on validation.

No threshold may be relaxed after results are seen. A metric implementation
bug requires one documented amendment, consistent rerun of every affected
model/reference, and retention of the original result.

## 14. Execution and artifacts

Full training is a three-task Rocky 9 H100/H200 array, one seed per task, with
required online W&B. Each directory stores config, latent moments, 200-line
history, selected checkpoint, final training state, result, environment,
Slurm record, W&B completion record, and SHA-256 inventory.

Before scientific evaluation:

1. freeze all three training artifacts into one compact matrix;
2. implement and known-answer test fCRPS/eCRPS, spread-skill correction,
   interval coverage, deterministic tie handling, member-wise nonlinear
   diagnostics, geometry masks, bootstrap, and ensemble-size prefixes;
3. run one four-target evaluator smoke without changing any frozen rule;
4. commit the exact evaluator and launcher.

Large forecasts and checkpoints remain outside Git. Compact training and
evaluation matrices, exact commands, hashes, seeds, block metrics, and compute
accounting are tracked. No evaluation may open `85606`.

