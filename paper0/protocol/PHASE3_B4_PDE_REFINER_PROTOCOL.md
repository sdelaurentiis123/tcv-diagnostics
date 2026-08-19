# Phase 3 B4 PDE-Refiner protocol

**Decision status:** frozen after the completed failed B3 one-seed gate and
before B4 implementation, smoke testing, training, or evaluation

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implementation, CPU tests, and one bounded Rocky 9
GPU smoke for a one-seed C5P-H1 latent PDE-Refiner adaptation

The machine-readable authority is
`paper0/manifests/phase3_b4_pde_refiner_85604.json`.

## 1. Why B4 follows B3

The completed B3 functional-generative experiment improved marginal fair CRPS
but failed all field, spectral, and transport families. Eleven of 15 material
power checks passed, but only four of 15 ensemble-mean-field realization-
coherence checks passed. Member-expected cross-phase and cross-coherence were
good, while mode-resolved and nonlinear-transport uncertainty remained
underdispersed. B3 therefore stops before additional seeds, O3, assimilation,
diagnostic ranking, or 85606.

B4 tests a different mechanism. PDE-Refiner was introduced by Lippe et al. to
address low-amplitude spatial information neglected by ordinary one-step
regression. It makes a first prediction and then applies a small number of
decreasing-noise denoising refinements. The primary paper reports that gains
usually flatten after roughly three refinements, that cost grows with network
calls, and that repeated stochastic refinement can eventually inject excess
high-frequency energy into very long rollouts.

The literature does not establish that PDE-Refiner ensembles are calibrated
for fields or nonlinear transport. Its principal uncertainty result relates
sample-divergence time to truth-divergence time. A later spectral-refiner study
also reports that isotropic refinement can over-damp higher-frequency content
for more complex Navier--Stokes spectra. B4 must therefore measure, rather than
assume, both spectral repair and probabilistic calibration.

Primary sources are:

- Lippe et al., *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE
  Solvers*, NeurIPS 2023, arXiv:2308.05732v2,
  <https://arxiv.org/abs/2308.05732>;
- the official MIT-licensed PDEArena implementation at commit
  `327424a46020c2afcfd777e8339e4b61b20d0e72`,
  <https://github.com/pdearena/pdearena>;
- Luo and Liang, *PDESpectralRefiner: Achieving More Accurate Long Rollouts
  with Spectral Adjustment*, arXiv:2506.10711,
  <https://arxiv.org/abs/2506.10711>.

Paper 0 does not adopt spectral reweighting in B4. Spectrum remains an
evaluation metric, never a training loss or tuned scheduler input.

## 2. Two hypotheses, not one blended claim

B4 separates two questions.

### H-det: deterministic refinement

Does the final refined prediction improve one-step realization-level spectral
fidelity and geometry-aware transport over the initial level-0 prediction
without materially degrading field accuracy?

Passing H-det may justify writing a separate O3 free-rollout protocol. It does
not justify assimilation.

### H-prob: probabilistic refinement

Do independent refinement-noise draws form a calibrated conditional ensemble
for fields, material modes, cross-field projections, and member-wise
transport?

Only H-prob may eventually justify using B4 covariance in assimilation. A
model may pass H-det and fail H-prob. That outcome is informative and must not
be collapsed into a single score.

## 3. Immutable data task

B4 inherits the verified job-6893525 model dataset and the existing
conditional-transient development split:

- fields: `[Ne, Pe, Pi, phi, Vi]`;
- training frames: `[0,432)`;
- training targets: `[2,432)`, all 430 once per epoch;
- guard frames: `[432,496)`, never loaded;
- validation frames: `[496,624)`;
- validation targets: `[498,624)`, all 126 chronologically;
- one saved-step horizon: 3.131905426352636 microseconds;
- toroidal period: `zperiod=5`;
- Fourier mapping: `n=5k`;
- standardized field shape: `[5,64,32,88]`.

The model receives only the exact preceding C5P field frame. Absolute time,
normalized frame index, shot label, diagnostics, future truth, and physics-
derived quantities are absent. Time is metadata, not a channel.

The seed-1701 `C5P-dcae_l10` codec and its per-latent-channel training-only
normalization are reused exactly. The codec remains in evaluation mode and is
never optimized. Its standardized latent state has shape `[32,16,8,22]`.

B4 noise is applied in standardized latent coordinates, not physical field
coordinates. This is a deliberate compute-compatible adaptation, not a claim
of exact equivalence to field-space PDE-Refiner. The O1 codec passed the frozen
representation gate, but the B4 evaluation must determine whether latent
denoising transfers to physical spectra and transport.

## 4. Exact deterministic parent and initialization

The parent is the selected seed-1701 `C5P-H1` O2 transition:

~~~text
/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/
job_6894980/task_0_c5p_h1_seed_1701/selected.pt
~~~

Its SHA-256 is
`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.
It was selected at zero-based epoch 193 and optimizer step 5,238.

B4 retains the parent's 32 latent channels, `[16,8,22]` grid, 512 hidden
channels, 16 transformer blocks, four attention heads, FFN factor four,
Q/K normalization, RoPE, dropout 0.05, activation checkpointing, global
attention, and latent patch `[time=1,x=2,y=2,z=1]`.

Every shape-compatible parent parameter must load exactly. The only missing
keys may belong to the new refinement-step embedding and block adapters;
unexpected parent keys are prohibited. Each new adapter's final linear layer
is initialized with exactly zero weights and zero bias. Before optimization,
the level-0 path with a zero candidate must reproduce the parent transition
bit for bit on the same device.

The codec and latent normalization remain frozen. All parent transition
parameters and all new refinement parameters are trainable after the identity
check. This is a parent-initialized full-transition fine-tune, not a random
restart and not an adapter-only fit.

## 5. Refinement-conditioned transition

Let `z_prev` be the standardized latent representation of the exact preceding
field frame and `c` a provisional standardized target latent. The shared
transition receives:

1. the exact context slot `z_prev`;
2. the provisional target slot `c`;
3. the existing known/unknown mask, with context one and provisional target
   zero;
4. refinement level `k in {0,1,2,3}`.

The four discrete levels are mapped to the interval `[0,1]`, embedded with a
256-component sinusoidal encoding, and passed through
`Linear(256,256) -> SiLU -> Linear(256,256) -> LayerNorm(256)`.
Each transformer block receives that same embedding through its own
`Linear(256,256) -> SiLU -> Linear(256,2048)` adapter. The 2,048 outputs are
added to the block's 512-component scale, shift, residual-scale, and
skip-scale quantities. The adapter is global across tokens but distinct across
blocks.

This conditioning is deterministic. Randomness enters through the full latent
noise fields used by refinement levels one through three, not through a global
noise vector as in B3.

## 6. Explicit denoising objective

B4 uses the paper's explicit denoising formulation rather than a library DDPM
scheduler. This avoids ambiguity in the upstream variable named
`min_noise_std`, which is used as a scheduler beta in the official code, and
preserves an exactly auditable parent level-0 path.

Let `z_true` be the standardized target latent. For each training example,
sample `k` uniformly from `{0,1,2,3}`.

For `k=0`, set the provisional target slot to zero and predict the standardized
latent increment:

\[
d_\theta = f_\theta(z_{\mathrm{prev}},0,0),
\qquad
d_* = z_{\mathrm{true}}-z_{\mathrm{prev}},
\]

\[
\mathcal{L}_0 = \operatorname{mean}\left[(d_\theta-d_*)^2\right].
\]

For `k in {1,2,3}`, draw a full latent noise field
`epsilon ~ N(0,I)`, corrupt truth, and predict that noise:

\[
\widetilde z_k=z_{\mathrm{true}}+\sigma_k\epsilon,
\qquad
\widehat\epsilon_k=f_\theta(z_{\mathrm{prev}},\widetilde z_k,k),
\]

\[
\mathcal{L}_k=
\operatorname{mean}\left[(\widehat\epsilon_k-\epsilon)^2\right].
\]

The mean is over batch, standardized latent channels, and latent spatial
cells. All four levels have equal sampling probability and no additional loss
weight. No decoded field, flux, spectrum, cross-phase, coherence, conservation,
PDE residual, blob label, calibration statistic, or other physics-derived
quantity enters the training loss.

## 7. Frozen noise schedule and precision

B4 uses three refinement levels. The upstream official configuration uses the
value `4e-7` as its minimum scheduler variance despite naming the parameter a
standard deviation. B4 states the convention explicitly:

\[
v_{\min}=4\times10^{-7},
\qquad
\sigma_{\min}=\sqrt{v_{\min}}=0.0006324555320336759,
\]

\[
\sigma_k=\sigma_{\min}^{k/3},\qquad k=1,2,3.
\]

Thus the exact standard deviations are:

| Level | `sigma_k` |
|---:|---:|
| 1 | 0.08583742189325572 |
| 2 | 0.007368062997280775 |
| 3 | 0.0006324555320336759 |

No validation spectrum or B4 result may alter these values. Because the final
scale lies below bfloat16 resolution near unit magnitude, B4 training,
validation, smoke, and inference use float32 without autocast. TF32 matmul and
cuDNN TF32 are disabled. Runtime records must include PyTorch, CUDA, cuDNN,
driver, GPU, and all relevant precision flags.

## 8. Inference and ensemble semantics

One member is generated as follows. First make the deterministic parent-like
prediction:

\[
\widehat z_0=z_{\mathrm{prev}}+
f_\theta(z_{\mathrm{prev}},0,0).
\]

Then, for `k=1,2,3`, draw an independent full latent noise field and refine:

\[
\widetilde z_k=\widehat z_{k-1}+\sigma_k\epsilon_k,
\]

\[
\widehat z_k=widetilde z_k-
\sigma_k f_\theta(z_{\mathrm{prev}},\widetilde z_k,k).
\]

The frozen codec decodes `z_hat_3`. One member therefore costs four transition
network calls: one initial prediction plus three refinements. Different members
share the same exact context and level-0 result but use independent noise
fields at all refinement levels. No member interaction, recentering,
inflation, clipping, rejection, or post-hoc calibration is allowed.

The public interface remains:

~~~python
forecast = model.predict(context, horizon=1, ensemble_size=M)
~~~

with canonical axes:

~~~text
[batch, ensemble_member, future_time, channel, x, y, z]
~~~

The prediction interface cannot accept target truth. Per-step versus
trajectory-constant noise is not tested by this one-step experiment; any later
O3 protocol must choose and freeze that policy prospectively.

## 9. Prospective full-training budget

Full training is **not authorized by this protocol**. If and only if the
bounded implementation smoke passes, a separate commit may authorize this
already specified candidate budget:

- seed: 1701;
- epochs: 100;
- training targets per epoch: 430;
- microbatch: one target;
- gradient accumulation: 16 targets;
- optimizer steps per epoch: 27;
- total optimizer steps: 2,700;
- optimizer: AdamW;
- beta values: `[0.9,0.999]`;
- weight decay: `1e-5`;
- learning rate: cosine from `1e-4` to `1e-6`;
- warmup: none;
- gradient-norm clip: one;
- EMA decay: `0.995` after every optimizer step;
- optimization precision: float32 with TF32 disabled;
- early stopping: prohibited.

The training target order and toroidal-roll augmentation are deterministic
functions of seed, epoch, and target, as in O2. The refinement-level draws and
training noise are separately seeded and recorded.

Every fifth completed epoch, plus epoch 100, is evaluated using EMA weights,
all 126 validation targets, and a fixed two-member inference seed bank. Select
the earliest candidate with the numerically lowest mean-member, equal-channel,
decoded standardized-field MAE at the final refinement stage after the full
budget. Physics metrics, level-0-versus-final improvements, W&B, and 85606 may
not select the checkpoint.

The validation seed bank is produced by
`numpy.random.Generator(numpy.random.PCG64(41003))` as unsigned 64-bit seeds
with shape `[126,2,3]`, ordered by chronological target, member, and refinement
level. Each seed initializes an independent NumPy PCG64 standard-normal latent
field in canonical `[32,16,8,22]` order. The seed bank is saved and hashed;
full noise fields need not be persisted.

## 10. Implementation and bounded smoke gate

Before any full training, B4 must provide:

1. known-answer schedule and explicit-denoising algebra tests;
2. exact refinement-level embedding and per-block adapter tests;
3. strict parent-load auditing;
4. bitwise pre-optimization level-0 parent identity;
5. full-latent noise shape, independence, and deterministic-seed tests;
6. balanced access to all four training levels;
7. finite gradients for parent and new refinement parameters;
8. a frozen, unchanged codec and normalization;
9. exact EMA update and checkpoint-reload tests;
10. canonical ensemble axes and one-step-only enforcement;
11. future-truth, guard, absolute-time, and 85606 prohibitions;
12. a complete local CPU suite and complete clean Rocky 9 CPU suite.

The only authorized GPU run is a non-scientific Rocky 9 H100/H200 smoke:

- seed 1701;
- training targets `[2,18)`, 16 total;
- validation targets `[498,502)`, four total;
- two epochs and two optimizer steps;
- two validation members;
- all four refinement levels explicitly probed;
- level-0 and all three intermediate/final stages saved for the four targets;
- finite nonzero final member diversity required in every decoded field;
- bitwise selected-checkpoint reload required;
- online W&B initialization and finished remote state required.

Smoke losses and four-target metrics are mechanical diagnostics, not scientific
results. Passing the smoke authorizes only writing a separate full-training and
evaluation protocol; it does not itself authorize training.

## 11. Later scientific evaluation, fixed in outline

A separately committed full-evaluation protocol must use:

- all 126 85604 validation targets;
- M32 final-stage forecasts from a fixed seed bank independent of checkpoint
  selection;
- a fixed M4 prefix saved at levels 0, 1, 2, and 3 to identify what refinement
  changes;
- the unchanged B3 field, calibration, directional-spectrum, material-band,
  member-expected cross-spectrum, member-wise transport, event, block,
  bootstrap, and Monte Carlo definitions;
- the exact deterministic H1 parent as the primary level-0 comparator;
- B2 and B3 only as unmatched descriptive context;
- explicit parameter count, calls per member, memory, accelerator hours, and
  forecast cost per physical microsecond.

No threshold is relaxed for B4. The deterministic and probabilistic decisions
are reported separately.

### H-det gate

The final refinement must preserve the frozen point-skill bound and pass the
absolute spectral, cross-field, and mean-transport thresholds used by the
existing one-step gates. Intermediate M4 records must show whether the final
result is an actual refinement over level 0 rather than a favorable stochastic
sample. A later protocol will spell out the complete machine reduction before
scientific forecasts are generated.

### H-prob gate

The final M32 ensemble must pass the unchanged strict B3 field, mode,
cross-spectrum-projection, and transport calibration thresholds, including
M16-to-M32 stability. Good member-expected cross-phase or final-stage field
CRPS cannot substitute for member-wise transport calibration.

If H-det fails, stop B4 before replication or O3. If H-det passes but H-prob
fails, B4 may be considered only as a deterministic/refined transition and
must not supply assimilation covariance. If both pass at seed 1701, write a
separate replication protocol for seeds 1702 and 1703. Neither outcome opens
85606 without the later Paper 0 freeze.

## 12. Claims boundary

The strongest possible claim from the currently authorized smoke is only:

> The parent-initialized latent PDE-Refiner mechanics execute reproducibly on
> Rocky 9 while preserving the frozen level-0 transition before optimization.

Even a later passing one-seed 85604 evaluation would not establish autonomous
rollout skill, held-out 85606 performance, architecture-level robustness,
transport-faithful emulation on another simulation, experimental diagnostic
realism, assimilation value, diagnostic ranking, cross-shot generalization,
or steering.
