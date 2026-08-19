# Phase 3 B4 full-training and one-step evaluation protocol

**Decision status:** frozen after the completed B4 implementation smoke and
before full B4 training, checkpoint selection, scientific forecast generation,
or B4 evaluation implementation

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Newly authorized scope:** one full seed-1701 B4-PDE-Refiner-H1 training run,
one bounded evaluator smoke, and one one-saved-step scientific evaluation on
the existing 85604 validation interval

The machine-readable authority is
`paper0/manifests/phase3_b4_full_evaluation_85604.json`.

## 1. Decision boundary

Rocky 9 H100 job `6899469` established that the exact C5P-H1 parent, frozen
C5P-dcae_l10 codec, level-conditioned latent PDE-Refiner, explicit denoising
loss, fixed seed banks, EMA, checkpoint reload, canonical ensemble interface,
and online W&B tracking execute together. Its compact result has SHA-256
`fd2b5465f612eb8da4943f6284e317145eff64b25346895137981ce3e3993eef`.
It used 16 training targets, four validation targets, two epochs, and two
optimizer steps. Its four-target losses and MAEs are mechanical diagnostics,
not scientific estimates.

This protocol freezes the first experiment capable of answering the two B4
hypotheses. It does not authorize schedule tuning, architecture tuning,
additional seeds, autonomous rollout, assimilation, diagnostic ranking,
control, or access to 85606.

## 2. Two separately reported hypotheses

### H-det: deterministic refinement

Does the final refined forecast repair one-step realization spectra,
cross-field structure, and mean radial ExB transport relative to level zero,
without materially degrading field accuracy?

### H-prob: probabilistic refinement

Do independent full-latent refinement draws form calibrated conditional
ensembles for fields, material modes, cross-spectrum projections, and
member-wise transport?

H-det and H-prob are reduced independently. A useful mean or sharp sample does
not establish covariance quality. Only a joint H-det and H-prob pass can
support a later assimilation-covariance protocol.

## 3. Immutable data and representation

Only the verified 85604 model dataset from job `6893525` is used:

- state fields: `[Ne,Pe,Pi,phi,Vi]`;
- context: exactly standardized field frame `t-1`;
- target: standardized field frame `t`;
- training frames: `[0,432)`;
- training targets: `[2,432)`, all 430 once per epoch;
- guard frames: `[432,496)`, never loaded;
- validation frames: `[496,624)`;
- validation targets: `[498,624)`, all 126 in chronological order;
- cadence: `3.131905426352636` microseconds per saved step;
- physical field shape: `[5,64,32,88]`;
- simulated toroidal periodicity: `zperiod=5`;
- Fourier mapping: stored index `k` corresponds to full-torus mode `n=5k`.

Absolute time, normalized frame index, shot label, diagnostic values, future
truth, and physics-derived quantities are not inputs. Time remains metadata,
not a channel.

The exact seed-1701 C5P-dcae_l10 checkpoint has SHA-256
`9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d`.
The exact training-only per-latent-channel normalization has SHA-256
`afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738`.
Both remain frozen and in evaluation mode. The standardized latent state has
shape `[32,16,8,22]`.

## 4. Exact model and full training

The parent is the selected C5P-H1 seed-1701 transition with SHA-256
`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.
The B4 transition retains the implementation frozen in
`PHASE3_B4_PDE_REFINER_PROTOCOL.md`: 32 latent channels, 512 hidden channels,
16 transformer blocks, four heads, three explicit denoising stages, and four
network levels `0,1,2,3`. It contains 51,612,800 inherited and 9,606,144 new
refinement parameters, all trainable. The codec is not trainable.

The exact noise standard deviations are:

| Level | Standard deviation |
|---:|---:|
| 1 | `0.08583742189325572` |
| 2 | `0.007368062997280775` |
| 3 | `0.0006324555320336759` |

For level zero, the transition predicts the standardized latent increment
from the preceding latent and a zero provisional target. For levels one to
three, it receives truth corrupted by the corresponding full-latent Gaussian
noise field and predicts that noise. The loss is uniform-level standardized
latent MSE. No decoded field, spectrum, cross-phase, coherence, transport,
conservation, PDE residual, event, or calibration quantity enters training.

The full budget is immutable:

- model seed: 1701;
- epochs: 100;
- targets per epoch: 430;
- microbatch: one target;
- gradient accumulation: 16 targets;
- final partial accumulation per epoch: 14 targets, divided by 14;
- optimizer steps per epoch: 27;
- total optimizer steps: 2,700;
- optimizer: AdamW with betas `[0.9,0.999]` and weight decay `1e-5`;
- global gradient-norm clip: one;
- EMA decay: `0.995` after every optimizer update;
- precision: float32 without autocast, TF32 disabled;
- early stopping: prohibited.

Let `j` be the zero-based optimizer-update index in `0,...,2699`. The exact
learning rate used for update `j` is

\[
\eta_j=10^{-6}+\frac{10^{-4}-10^{-6}}{2}
\left[1+\cos\left(\frac{\pi j}{2699}\right)\right].
\]

Thus the first update uses exactly `1e-4` and the last uses exactly `1e-6`.
There is no warmup.

Training target order and toroidal-roll augmentation remain deterministic
functions of seed, epoch, and target under the existing O2 convention. A
NumPy `Generator(PCG64(41001))` draws the complete int64 level matrix with
shape `[100,430]` before training. Its raw C-order byte SHA-256 is
`ac370fa17291d8bd4c36ac4d451f78e63250c19ad77cf70a3f8403465e339ff6`
and its level counts are `[10831,10680,10722,10767]`. Training latent noise is
drawn from a dedicated CUDA generator initialized once with seed `41002`; its
state is saved in every resumable state artifact. Training cannot resume with
a different target order, level matrix, augmentation, or noise-generator
state.

## 5. Validation and checkpoint selection

Completed epochs `5,10,...,100` are validation candidates. Every candidate
uses EMA weights, all 126 validation targets, and the same fixed two-member
seed bank generated by `Generator(PCG64(41003))`. The uint64 bank has shape
`[126,2,3]`, raw C-order byte SHA-256
`85409dcad8eb2800bcd703a35aee502c59add718cd5956780f2ade7555f544ca`,
and `.npy` SHA-256
`127936e25054925f4b114d5b174cbe876847555ffd0963ca54ce0e6c72f29884`.
Each seed initializes a NumPy PCG64 float32 standard-normal latent field in
canonical `[32,16,8,22]` order.

After all 100 epochs, select the earliest candidate with the numerically
lowest equal-channel decoded standardized-field MAE of the two-member
ensemble-mean forecast at refinement level three. No level-zero score,
spectrum, transport, calibration metric, W&B value, or 85606 result may alter
selection. The selected EMA transition and final resumable training state are
separate immutable artifacts. A fresh same-device reconstruction must
reproduce a fixed selected-checkpoint latent and decoded probe bit for bit.

The validation interval is used for development selection and scientific
development scoring. Results are therefore model-selection results on 85604,
not unbiased final-test estimates.

## 6. Independent truth-separated scientific forecasts

Scientific generation uses a separate uint64 seed bank created by
`Generator(PCG64(41032))` with shape `[126,32,3]`. Its raw C-order byte
SHA-256 is
`f6990201934ae1d2c215458e875b9b7950965a73645a7bd28f5b034121f0a892`
and its `.npy` SHA-256 is
`a1871e069bce6244073bfe1aa835a53c1d7a59302b01f6a366b3dc88297b6205`.
The bank is independent of checkpoint-selection noise. Each seed produces one
NumPy PCG64 float32 standard-normal field in `[32,16,8,22]` order.

The generator receives context only and writes two immutable artifacts before
any target truth is opened:

1. final level-three M32 forecasts with axes
   `[target=126,member=32,future_time=1,channel=5,x=64,y=32,z=88]`;
2. the first four members at levels zero through three with axes
   `[target=126,member=4,stage=4,channel=5,x=64,y=32,z=88]`.

Stage-three M4 must be bitwise identical to the first four members of the M32
artifact. Level zero must be bitwise identical across all members for a given
target. Member order is immutable; stored prefixes define M4, M8, M16, and
M32. No regeneration, recentering, inflation, clipping, rejection, member
sorting, interaction, or post-hoc calibration is allowed. Dropout is disabled.

A naïve member costs four transition evaluations. Because level zero is
shared, an M-member target may be executed in `1+3M` transition evaluations;
both the amortized and unamortized costs must be reported. The codec decode
cost, peak memory, accelerator time, and forecast cost per physical
microsecond must also be recorded.

## 7. Frozen metrics and comparators

The final M32 artifact is scored by the byte-locked B3/B2 numerical engine:

- `b2_probabilistic_metrics.py` SHA-256
  `edef6fbbe7b40348fa450c7428d796f4b5ebc3d9b2070e135c7bb3f58a2b6650`;
- `b2_field_metrics.py` SHA-256
  `c2d0f5e764b783f7a6a240fbd3f11f6c0a4fd52a173d9f1dd1eb97ccff62a0db`;
- `b2_spectral_metrics.py` SHA-256
  `382fc683519d01185d0e5314196cd0c62f5e39e60f5e1aa06478e74acda8761e`;
- `b2_transport_metrics.py` SHA-256
  `b78ea33f641fe6409ca5a55503f3729013f2da3cc78f93671f63c6fadafcb02e`;
- `b2_scoring.py` SHA-256
  `2dfdf6f7b620302826971c9fec4ed8233f46fa1950c8461ed9d79194411178fe`;
- `geometry.py` SHA-256
  `4f5eda7001bf9b42cefb224842a1dee4a955028a1aa063a57db6c447879f424c`;
- `codec_transport.py` SHA-256
  `201a9628564b1ad5e476cbee52edf5eac458c61dadc1c7057a5b6e205de46d45`.

The exact deterministic C5P-H1 parent is the primary comparator. Its forecast
SHA-256 is
`a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`
and its score SHA-256 is
`ebdc707e2be500af7de492038ae8bfb4d126b81b271b340345b85a7fba1d5593`.
Persistence and the training-only toroidal spectral AR(1) are uncompressed
references. B2 and B3 are descriptive context only, not matched gate
comparators. No comparator is retrained or reselected.

Metrics include field RMSE, MAE, bias, variance, anomaly correlation, fair and
empirical CRPS, corrected spread-skill, coverage, and rank; directional and
toroidal spectra; material-band power and realization coherence; member-wise
cross-spectrum amplitude, phase, and coherence; and authoritative member-wise
radial ExB particle and electron, ion, and total internal-energy transport.
Transport is never computed only from ensemble-mean fields.

For primary marginal `phi` metrics, subtract the full spatial mean separately
from truth and each member at every target. The toroidal bands are
`k=1..3`, `k=4..5`, and `k=6..7`, corresponding to `n=5..15`, `n=20..25`,
and `n=30..35`. Materiality remains the training-only one-percent rule.

Overall targets are `[498,624)`. Temporal robustness uses the six fixed
21-frame blocks `[498,519)`, `[519,540)`, `[540,561)`, `[561,582)`,
`[582,603)`, and `[603,624)`. A family passes temporally only if it passes
overall and in at least five of six blocks. Moving-block bootstrap intervals
use block length 21, 2,000 replicates, six sampled blocks, 106 valid starts,
and seed `85604032`; they are conditional on this one simulation run.

## 8. Stagewise repair test for H-det

The M4 stage artifact is evaluated with the same geometry, truth, materiality,
gauge, and nonlinear member-wise conventions. For stage `s`, define:

- `E_field(s)`: equal-channel MAE of the M4 ensemble-mean fields;
- `E_power(s)`: mean over material field bands of
  `abs(log(max(P_pred/P_truth,1e-12)))`, using member-expected power;
- `E_real(s)`: mean over material field bands of one minus the realization
  coherence between the M4 ensemble-mean field and truth;
- `E_cross(s)`: mean over material cross-field bands of circular phase error
  divided by pi plus absolute coherence error, using mean member
  cross-spectra;
- `E_transport(s)`: mean separatrix relative-L2 error over the four
  member-wise radial ExB transport quantities.

All aggregates give each eligible field band, cross-field band, or transport
quantity equal weight. Undefined or non-finite components fail. To establish
an actual refinement rather than a favorable isolated sample, level three
must satisfy:

1. `E_field(3) <= 1.05 E_field(0)`;
2. each of `E_power`, `E_real`, `E_cross`, and `E_transport` is at most 1.05
   times its level-zero value;
3. at least one of `E_power(3)`, `E_real(3)`, or `E_cross(3)` is strictly below
   its level-zero value;
4. `E_transport(3)` is strictly below `E_transport(0)`.

Levels one and two are reported but do not select a stage or checkpoint.

## 9. Absolute H-det gate

H-det passes only if integrity, the stagewise repair test, and all applicable
absolute realization/mean conditions pass overall and in at least five of six
blocks:

1. final M32 ensemble-mean equal-channel MAE and RMSE are each at most 1.05
   times the deterministic H1 values;
2. every material field-band member-expected power ratio lies in
   `[0.75,1.30]`;
3. every material field-band ensemble-mean realization coherence is at least
   `0.80`;
4. every material primary cross-field band has circular phase error at most
   20 degrees and absolute coherence change at most `0.15`;
5. for all four strict-face transport quantities, relative L2 is at most
   `0.40`, correlation at least `0.70`, and weighted sign disagreement at most
   `0.20`;
6. for all four confined-separatrix quantities, relative L2 is at most `0.30`,
   absolute normalized bias at most `0.15`, correlation at least `0.80`, and
   weighted sign disagreement at most `0.15`;
7. every eligible upper-decile event has magnitude relative error at most
   `0.50` and weighted sign disagreement at most `0.25`.

Truth-empty event scopes use the frozen A016 not-applicable rule after all
integrity checks; they are neither passes nor failures.

## 10. Absolute H-prob gate

H-prob is reduced independently of H-det. It passes only if integrity and all
applicable probabilistic conditions pass overall and in at least five of six
blocks:

1. aggregate field fair CRPS is strictly below both H1 MAE and the best
   applicable uncompressed-reference MAE;
2. at least four field fair-CRPS values are strictly below H1 field MAE, with
   the fifth at most 1.05 times H1;
3. at least four fields have corrected spread-skill in `[0.80,1.25]`, with
   the fifth in `[0.67,1.50]`;
4. at least four fields have absolute coverage error at most `0.10` for I17,
   `0.08` for I27, and `0.06` for I31; the fifth may use twice those limits,
   and every primary region has I31 coverage in `[0.75,0.995]`;
5. every material band-power and cross-spectrum real/imaginary projection has
   corrected spread-skill in `[0.67,1.50]` and I31 coverage in
   `[0.75,0.995]`;
6. at least three separatrix transport fair-CRPS values beat H1 absolute
   error, with the fourth at most 1.05 times H1;
7. at least three separatrix transport quantities have corrected spread-skill
   in `[0.67,1.50]`, I27 coverage within `0.12` of nominal, and I31 coverage
   within `0.10` of nominal; the fourth remains finite and noncollapsed;
8. every primary aggregate field, material-band, cross-field, and separatrix-
   transport fair-CRPS component satisfies
   `abs(q_M16-q_M32) <= 0.10*abs(q_M32)+1e-8`.

The M32 order-statistic intervals are I17 `[x_(8),x_(25)]`, I27
`[x_(3),x_(30)]`, and I31 `[x_(1),x_(32)]`, with nominal coverages `17/33`,
`27/33`, and `31/33`. Voxel counts are not independent sample sizes.

## 11. Integrity and evaluator smoke

The full training record must prove 100 epochs, 2,700 optimizer updates,
complete fixed-target histories, exact level/noise provenance, finite parent
and refinement gradients, unchanged codec, earliest-minimum checkpoint
selection, and exact reload. Forecast generation must complete and hash both
forecast artifacts before the scorer can open target truth. All 32 members
must be finite and have nonzero spread in every field and primary region.

Before full evaluation, a bounded evaluator smoke uses targets `[498,502)`,
all 32 members from the fixed scientific bank, and the M4 all-stage prefix.
It must prove canonical axes, M4/M32 prefix identity, shared level zero,
nonzero final spread, truth separation, scoring compatibility, artifact
hashing, and online W&B completion. Its scores remain non-scientific. Passing
it releases only the already-frozen full 85604 evaluation.

The complete local suite and complete clean Rocky 9 suite must pass before
full execution. Training, evaluator smoke, and full generation require Rusty
Rocky 9 H100/H200 nodes. GPU execution outside Rusty is prohibited.

## 12. Decision rules

- If H-det fails, stop B4 before replication, O3, or assimilation.
- If H-det passes and H-prob fails, B4 may be described only as a refined
  deterministic transition. A separate O3 protocol may be written, but B4
  covariance cannot be used for assimilation.
- If H-prob passes while H-det fails, the ensemble is not transport-faithful
  and cannot be used for assimilation.
- If both pass, a separate same-seed-parent replication protocol for seeds
  1702 and 1703 may be written. Those runs are not authorized here.
- No B4 outcome opens 85606. The Paper 0 model-selection, metric,
  assimilation, and release freeze remains a later decision.
- No threshold, seed, stage, schedule, or checkpoint may be changed after
  seeing results. A metric bug requires a documented amendment and consistent
  rerun of every affected model.

## 13. Artifacts, W&B, and claims boundary

Every run stores configuration, protocol and manifest hashes, data and
normalization identity, level and seed banks, history, selected checkpoint,
final resumable state, generation manifests, both forecast artifacts, raw and
block scores, H-det/H-prob reductions, bootstrap records, compute accounting,
environment, Slurm metadata, W&B record, and a complete SHA-256 inventory.
Large artifacts remain on Ceph; compact manifests and decisions are tracked in
Git. W&B is an online monitoring mirror, not scientific authority.

The strongest possible conclusion from a joint one-seed pass is:

> On the later 85604 development interval, one seed of a parent-initialized
> latent PDE-Refiner improved one-step transport-relevant structure and
> produced a calibrated enough ensemble to merit independent-seed
> replication.

It would not establish autonomous rollout skill, held-out 85606 performance,
architecture-level robustness, transport-faithful emulation on another
simulation, experimental diagnostic realism, assimilation value, diagnostic
ranking, cross-shot generalization, or steering.
