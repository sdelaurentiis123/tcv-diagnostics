# Phase 3 B3 functional-generative retrofit protocol

**Decision status:** frozen after the failed B2 one-step evaluation and before
B3 implementation, smoke testing, or training

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implementation, CPU tests, and one bounded Rocky 9
GPU smoke for a one-seed C5P-H1 functional-generative retrofit

The machine-readable authority is
`paper0/manifests/phase3_b3_fgn_85604.json`.

## 1. Why this experiment exists

The complete C5P-dcae_l10 codec passed the O1 reconstruction gate at all three
frozen seeds. The deterministic O2 transition then achieved lower ordinary
one-step field error than every uncompressed reference, but failed
realization-level spectral and nonlinear-transport gates. B2 tested a
LOLA-style conditional latent diffusion model. B2 modestly improved
ensemble-mean field error, but its 32-member one-step ensembles remained
underdispersed in every field and every transport quantity, while important
realization-coherence and transport checks still failed.

B3 asks a narrower question: can the strongest existing deterministic
transition be converted into a useful probabilistic predictor by injecting a
low-dimensional global random variable through every transformer block and
fine-tuning with a proper ensemble score?

This is a functional-generative-network (FGN) retrofit following Diaconu et
al., *Probabilistic Retrofitting of Learned Simulators*, arXiv:2603.01949, and
its official implementation at commit
`7643376c2949717ee5c2c840584689f529ba77a5`. It is not diffusion. One forward
pass and one sampled global noise vector produce one ensemble member.

## 2. Exact scientific question

For each validation target, condition on the exact preceding C5P frame and
draw multiple possible next frames. The modeled fields are

~~~text
[Ne, Pe, Pi, phi, Vi]
~~~

The primary arm is B3-FGN-H1. H1 is selected at the arm level because the
completed deterministic comparison found no benefit from the second exact
context frame: mean H2 RMSE was 0.73% higher and mean H2 MAE was 0.52% higher
than H1. Seed 1701 is declared prospectively as the first pilot seed. It is
not the lowest-error H1 seed, so the pilot does not select a favorable seed
after seeing outcomes.

B3 initially estimates only the one-saved-step conditional distribution. It
does not consume an earlier model prediction and is not an autonomous
rollout. Per-step versus trajectory-constant noise becomes meaningful only in
a later autoregressive experiment and is not inferred from this one-step
pilot.

No absolute time, normalized frame index, shot label, diagnostic value,
future truth, or physics-derived quantity is supplied to the model. The saved
cadence is metadata, not an input channel.

## 3. Immutable data, split, and representation

B3 inherits the verified job-6893525 model dataset and the Phase 2 split:

- training frames: `[0,432)`;
- matched training targets: `[2,432)`, all 430 once per epoch;
- guard frames: `[432,496)`, never loaded;
- validation frames: `[496,624)`;
- validation targets: `[498,624)`, all 126 chronologically;
- native cadence: 3.131905426352636 microseconds;
- toroidal period: `zperiod=5`;
- Fourier mapping: `n=5k`;
- standardized volume shape: `[5,64,32,88]`.

The volume normalization from job 6893525 is immutable and was fit on the
training region only. Ne uses the accepted log-offset transform; Pe, Pi, phi,
and Vi use the accepted identity transforms followed by scalar
standardization. No clipping or refitting is allowed.

The exact seed-1701 C5P-dcae_l10 codec and latent normalization embedded in the
selected H1 checkpoint are reused. The codec remains in evaluation mode and
all codec parameters have gradients disabled. The codec and latent
normalization are not retrained or refit for B3.

## 4. Exact deterministic parent

The parent is the selected C5P-H1 O2 transition at:

~~~text
/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o2_full/
job_6894980/task_0_c5p_h1_seed_1701/selected.pt
~~~

Its SHA-256 is
`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.
It was selected at epoch 193, optimizer step 5238, with full-validation
equal-channel standardized field MAE 0.04558250684515488.

The inherited deterministic transition has:

- 32 latent channels on grid `[16,8,22]`;
- one exact context frame plus one zero-filled target slot;
- one binary known/unknown mask channel;
- 512 hidden token channels;
- 16 transformer blocks;
- 4 attention heads;
- FFN factor 4;
- Q/K normalization and RoPE enabled;
- dropout 0.05;
- activation checkpointing enabled;
- latent patch `[time=1,x=2,y=2,z=1]`;
- global self-attention;
- standardized latent-increment prediction.

Every shape-compatible parent parameter is loaded exactly. A load audit must
show that the only missing keys are newly introduced noise-embedding and
noise-adapter parameters, with no unexpected parent keys.

## 5. Functional noise mechanism

For ensemble member `m`, draw one raw vector

\[
\epsilon_m \sim \mathcal{N}(0,I_{32}).
\]

Map it through

\[
h_m = \operatorname{LayerNorm}\!\left(
W_2\,\operatorname{SiLU}(W_1\epsilon_m+b_1)+b_2
\right),
\]

where the raw noise dimension is 32 and the embedded noise dimension is 256.
The same `h_m` is shared across every spatial token and supplied to every one
of the 16 transformer blocks.

In block `l`, a separate adapter produces four 512-component modulation
vectors:

\[
(a_l^{n},b_l^{n},c_l^{n},d_l^{n})
= A_l(h_m).
\]

They are added to the pretrained block's deterministic AdaLN quantities
before attention, residual, and long-skip modulation. The adapter is
`Linear(256,256) -> SiLU -> Linear(256,2048)`. Following the official code,
the last adapter weight is initialized at ordinary PyTorch scale and then
multiplied by `1e-2`; its bias is set exactly to zero.

Noise is global and functional: it is not independently sampled per pixel,
latent voxel, token, field, or layer. Different ensemble members receive
independent raw vectors. The deterministic-noise-disabled path supplies no
noise vector and must reproduce the parent transition bit for bit before any
optimization.

## 6. Fine-tuning meaning and parameter groups

"Fine-tuning" has a precise meaning here. B3 starts from the already trained
deterministic H1 weights, adds new stochastic parameters, and continues
optimizing the full transition. It does not retrain the transition from a new
random initialization, and it does not freeze the old transition while
training only the noise branch.

Parameters are divided by exact state-dict provenance:

- **common:** every shape-matched parameter loaded from the deterministic
  parent;
- **new:** the global noise embedding and all block-specific noise adapters.

The codec and normalization buffers belong to neither optimizer group. The
common group uses a smaller learning rate so that the pretrained deterministic
map moves more slowly than the new stochastic mechanism.

## 7. Fair-CRPS objective

For a scalar target `y` and `M` ensemble predictions `x_1,...,x_M`, B3 uses
the finite-ensemble fair CRPS estimator

\[
\operatorname{fCRPS}_M
= \frac{1}{M}\sum_{m=1}^{M}|x_m-y|
- \frac{1}{2M(M-1)}\sum_{m\ne j}|x_m-x_j|.
\]

Training uses exactly two independent ensemble members per example because
the decoded 3D volumes and 51-million-parameter backbone make larger training
ensembles disproportionately expensive. With `M=2`, the pointwise score is

\[
\frac{|x_1-y|+|x_2-y|}{2}-\frac{|x_1-x_2|}{2}.
\]

The score is computed after frozen decoding in the standardized five-field
space. It is averaged over spatial cells within each channel and then equally
over the five channels. This preserves the O2 equal-channel convention while
turning the marginal forecast into a proper finite-ensemble objective.

No flux, spectrum, cross-phase, coherence, gradient, conservation, PDE
residual, blob label, calibration diagnostic, or other physics-derived
quantity is used in training or checkpoint selection. Those quantities remain
evaluation metrics only. Marginal fair CRPS does not guarantee correct joint
cross-field transport; that is a hypothesis the later evaluator must test.

## 8. Frozen prospective full-run budget

Full one-seed training is not authorized by this protocol, but its candidate
budget is fixed prospectively so the smoke cannot be used to tune it:

- seed: 1701;
- epochs: 100;
- training targets per epoch: 430;
- ensemble members per training example: 2;
- microbatch: 1 target;
- gradient accumulation: 16 targets;
- optimizer steps per epoch: 27;
- total optimizer steps: 2700;
- optimizer: AdamW;
- beta values: `[0.9,0.99]`;
- weight decay: 0;
- new-parameter peak learning rate: `1e-4`;
- common-parameter peak learning rate: `3e-5`;
- linear warmup: 10 epochs, or 270 optimizer steps, for both groups;
- schedule: independent cosine decay to zero after warmup for both groups;
- gradient-norm clip: 1;
- precision: bfloat16 autocast;
- early stopping: disabled.

The raw noise dimension, 100-epoch budget, AdamW settings, staged optimizer,
and learning-rate pair follow the official FGN configuration and its
Euler-retrofit command where directly applicable. The training target count,
gradient accumulation, field representation, and architecture remain matched
to Paper 0.

## 9. Validation and checkpoint selection

Full validation uses all 126 targets every epoch. A NumPy PCG64 generator with
seed 31003 produces one immutable float32 bank of shape `[126,2,32]` in
chronological target order. The bank is saved and hashed with the run. The
same bank must be used at every epoch and, if later authorized, across model
seeds.

Checkpoint selection is the earliest epoch attaining the numerically lowest
full-validation equal-channel decoded-field fair CRPS after the complete
100-epoch budget. Validation RMSE, MAE, spread, spectra, cross-field metrics,
and transport metrics are logged or evaluated only as nonselecting evidence.

The parent deterministic forecast is evaluated by the same code path with
noise disabled. A lower validation fair CRPS is not sufficient by itself to
declare B3 transport-faithful.

## 10. Canonical prediction interface

The model exposes semantics equivalent to

~~~python
forecast = model.predict(context, horizon=1, ensemble_size=M)
~~~

with axes

~~~text
[batch, ensemble_member, future_time, channel, x, y, z]
~~~

Future time has length one in this protocol. The implementation accepts no
target tensor in `predict`. Each member uses one independent raw global noise
vector. Nonlinear diagnostics must later be computed member by member before
ensemble aggregation; flux from ensemble-mean fields is not an admissible
replacement.

## 11. Provenance boundary

The implementation may port only the minimal FGN mechanisms needed here from
the official MIT-licensed repository. Exact upstream paths and source hashes
are recorded in the machine-readable manifest. The upstream license must be
retained, and every modification must be described in `paper0/PROVENANCE.md`.

The deterministic O2, B2 diffusion, codec, data-loader, normalization, and
evaluator sources remain unmodified during the initial B3 implementation.
B3 is implemented in new modules and exercised through new entrypoints. Any
later shared-code change requires a separate documented commit and regression
evidence.

## 12. Implementation and bounded smoke gates

Before any full B3 training:

1. hash-lock this protocol, manifest, parent checkpoint, codec, latent
   normalization, dataset, and upstream source files;
2. test the exact finite-ensemble fair-CRPS formula, including `M=2` known
   answers and unbiased member permutations;
3. test global-noise shapes, sharing across tokens, and independent members;
4. test that noise-disabled inference is bitwise identical to the loaded H1
   parent before optimization;
5. test that the parent load has no unexpected keys and only new stochastic
   keys are missing;
6. test canonical ensemble axes, finite outputs, and nonzero member diversity;
7. test gradients in the noise embedding, every noise adapter, and common
   parameters;
8. test that codec parameters remain frozen and unchanged;
9. test fixed validation-noise reproducibility and checkpoint reload identity;
10. test that `predict` cannot receive future truth or use 85606;
11. run the complete local and Rocky 9 CPU suites;
12. run one Rocky 9 H100/H200 smoke at seed 1701 using exactly 16 training
    targets, 4 validation targets, 2 ensemble members, and 2 epochs;
13. require finite train/validation loss and gradients, exact artifact hashes,
    a reload-identical checkpoint, nonzero latent and decoded-field member
    diversity, and a finished online W&B run.

The smoke is an implementation gate, not scientific evidence. It must be
reported as `scientific_result=false` and may not read 85606.

## 13. Stop/go boundary

This protocol authorizes implementation and one bounded smoke only. Full
one-seed B3 training requires a later committed authorization after the smoke
passes. Scientific B3 evaluation requires a separately frozen evaluation
protocol defining ensemble sizes, deterministic and B2 comparators, field and
calibration metrics, member-wise spectra and transport, blockwise gates, and
the stop/go rule.

The required later ablations remain:

- noise disabled;
- global functional noise;
- matched input noise;
- matched output noise;
- spatially independent noise;
- per-step versus trajectory-constant noise in an autoregressive setting;
- one model versus independently trained model seeds where feasible;
- H1 versus H2 context if the primary stochastic mechanism works.

No ablation is authorized by the bounded smoke protocol.

## 14. Claims boundary

The strongest possible conclusion from the eventual one-step 85604 B3 pilot
is that a global functional-noise retrofit improves the selected H1 model's
one-step probabilistic forecast on later 85604 data while preserving or
improving member-wise physical diagnostics.

It would not establish autonomous rollout skill, held-out 85606 performance,
transport-faithful emulation, experimental diagnostic realism, cross-shot
generalization, assimilation value, diagnostic ranking, or steering.
