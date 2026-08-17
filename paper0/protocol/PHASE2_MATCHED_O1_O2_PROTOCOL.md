# Phase 2 matched O1/O2 model protocol

**Protocol status:** frozen before model implementation, smoke testing, or
training

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Training authorized after implementation gates pass:** only the deterministic
85604 O1 codec and O2 teacher-forced one-step experiments defined here

The machine-readable authority is
`paper0/manifests/phase2_matched_o1_o2_85604.json`.

## 1. Purpose

This experiment is a fault-localization ladder, not an architecture
competition. It asks, in order:

1. Can a newly trained representation reconstruct the state, coherent modes,
   cross-field phase, and authoritative transport?
2. Given a passing representation, can a deterministic model predict one
   saved step better than simple uncompressed references?
3. Does predictive skill depend on the exact Hermes state, or can two observed
   five-field frames substitute for omitted state?

The three state arms are already frozen:

| Arm | Context | Volume state | Extra state |
|---|---:|---|---|
| `E6B-H1` | 1 frame | `Ne,Pe,Pi,NVe,NVi,Vort` | retained `Bphi[2,32]` |
| `C5P-H2` | 2 frames | `Ne,Pe,Pi,phi,Vi` | none |
| `C5P-H1` | 1 frame | `Ne,Pe,Pi,phi,Vi` | none |

`C5P-H2` and `C5P-H1` share one codec for a given seed. Their only model
difference is the number of ordered context frames. `E6B-H1` uses its own
six-field codec and a declared boundary adapter.

Passing O1/O2 does not establish probabilistic calibration, autonomous rollout
skill, stationarity, cross-shot generalization, or diagnostic value. It only
permits a separately frozen O3/O4 experiment.

## 2. Immutable data and split

All arrays come from the verified output of Rocky 9 job `6893525`:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
  phase2_model_dataset/job_6893525/
~~~

The job contains eight non-overlapping HDF5 shards covering global frames
`[0,624)`. Every source, resampling, storage, boundary, and normalization
gate passed. Model code must verify the tracked compact result, normalization
result, artifact index, and every consumed shard hash before loading a tensor.

The immutable regions are:

| Region | Frames | Use |
|---|---:|---|
| training | `[0,432)` | codec optimization, normalization, O2 optimization |
| guard | `[432,496)` | never loaded by training or selection |
| validation | `[496,624)` | codec checkpoint selection and O1 gate |

All one-step arms use exactly:

| Region | Target frames | Count |
|---|---:|---:|
| training | `[2,432)` | 430 |
| validation | `[498,624)` | 126 |

No absolute time, frame index, region label, future value, or verifying phase
shift is a model input. The fixed relative cadence is
\(3.131905426352636\,\mu{\rm s}\). The toroidal wedge has
\(zperiod=5\), so \(n=5k\).

## 3. Shared preprocessing

Use the training-only statistics from job `6893525` without refitting:

- `Ne`: \(\log(N_e+10^{-6})\), then scalar standardization;
- all other volume channels: identity, then scalar standardization;
- `Bphi`: identity, then one standardization per radial side.

Training tensors are float32. Optimization parameters and optimizer states are
float32. CUDA autocast may use bfloat16, but checkpoint evaluation and saved
forecast tensors are float32; metric accumulators are float64.

No clipping, pressure floor, gauge subtraction, temperature substitution,
random crop, spatial flip, or smoothing is allowed. The only training
augmentation is one uniformly sampled integer toroidal roll in `[0,88)`,
applied identically to every volume field in a frame or window. `Bphi` has no
toroidal axis and is unchanged. Validation has no augmentation.

## 4. O1 representation ladder

The implementation is a minimal, attributed port of the MIT-licensed LOLA
deep-compression autoencoder at upstream commit
`21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`, with the predecessor TCV
per-axis padding and per-transition stride repairs ported as separately tested
code. No historical checkpoint is loaded.

Common settings for both candidates are:

- three-dimensional DCAE;
- kernel size 3;
- pixel-shuffle down/up sampling;
- LayerNorm residual blocks;
- two residual blocks at every depth;
- `softclip2` latent saturation;
- dropout 0.05;
- zero latent noise;
- circular padding only in toroidal \(z\);
- zero padding in radial \(x\) and poloidal \(y\);
- identity-initialized sampling convolutions.

### R1: `dcae_l20`

~~~text
hidden channels: [64,128,256,512]
latent channels: 64
strides:
  [(2,2,2), (2,2,2), (2,2,1)]
input grid:  [64,32,88]
latent grid: [8,4,22]
latent scalars: 45,056
nominal scalar compression:
  C5P = 20:1
  E6B = 24:1
~~~

The latent toroidal Nyquist index is `k=11`, so the predeclared useful
`k=4..7` band is representable without relying on decoder-generated
super-resolution.

### R2: `dcae_l10`

~~~text
hidden channels: [64,128,256]
latent channels: 32
strides:
  [(2,2,2), (2,2,2)]
input grid:  [64,32,88]
latent grid: [16,8,22]
latent scalars: 90,112
nominal scalar compression:
  C5P = 10:1
  E6B = 12:1
~~~

R2 is a predeclared lower-compression repair, not a causal single-axis
ablation: it changes latent scalar count, radial/poloidal resolution, depth,
and channel width.

### Escalation rule

Train R1 from scratch for both the C5P and E6B state families at all three
seeds. R1 is accepted only if every selected checkpoint in both state families
passes the complete per-seed O1 gate. If any of the six checkpoints fails, run
the complete matched R2 set. If any R2 checkpoint fails, stop before O2 and
report representation failure. If R1 passes, do not run R2 merely to obtain a
more favorable result.

## 5. Codec optimization and selection

Seeds are exactly `1701, 1702, 1703`. A seed controls initialization, frame
permutation, toroidal roll, and dropout.

Each codec run uses:

| Item | Frozen value |
|---|---|
| epochs | 200 |
| training examples per epoch | every one of 432 frames exactly once |
| microbatch | 4 |
| gradient accumulation | 4 |
| effective batch | 16 |
| optimizer | AdamW |
| learning rate | \(2\times10^{-4}\) |
| betas | \((0.9,0.95)\) |
| weight decay | \(10^{-4}\) |
| gradient norm clip | 1.0 |
| warmup | 10 epochs, linear |
| decay | cosine to \(2\times10^{-6}\) |
| validation | all 128 frames, chronological, every epoch |

The codec training loss is the arithmetic mean of the per-channel mean
absolute error in standardized model coordinates:

\[
\mathcal L_{\rm codec}
=
\frac{1}{C}\sum_{c=1}^{C}
{\rm mean}_{x,y,z}
\left|\widehat x_c-x_c\right|.
\]

It contains no spectrum, cross-phase, flux, conservation, PDE residual, blob,
gradient, increment, or other physics-derived term.

The selected checkpoint is the earliest epoch attaining the numerically
lowest full-validation codec loss. Training always completes all 200 epochs;
there is no early stopping. The final and selected checkpoints are both kept.

The E6B `Bphi` array bypasses O1 compression exactly, as previously frozen.
It is hash-checked and copied unchanged into the E6B reconstruction path.

## 6. O1 metrics and acceptance

Report train and validation metrics separately. Only validation decides the
gate. Material mode bands are determined once from training truth and then
applied unchanged to validation. The validation interval is divided into
eight contiguous 16-frame blocks.

Every state family is scored in:

1. its native predicted channels; and
2. the common transport view `[Ne,Pe,Pi,phi,Vi]`.

For E6B, `phi` is reconstructed with the hash-locked elliptic operator using
decoded `Pe,Pi,Vort` and exact O1 `Bphi`;
\(V_i=NV_i/(2N_e)\), matching the verified deuterium momentum identity.
Non-positive decoded density in the independently frozen physical scoring
support is a hard failure and is never repaired by clipping.

The field, spectrum, and cross-field gates retain the already frozen
definitions:

- every standardized field RMSE \(\le 0.10\);
- every field variance ratio in \([0.80,1.20]\);
- for every training-material band among `k=1..3`, `4..5`, and `6..7`,
  power ratio in \([0.80,1.25]\) and truth/reconstruction transfer coherence
  \(\ge0.90\);
- for `Ne-phi`, `Pe-phi`, and `Pi-phi` in every training-material band,
  cross-phase error \(\le15^\circ\) and coherence change \(\le0.10\);
- every applicable criterion must pass in at least seven of eight validation
  blocks as well as over the complete validation interval.

Authoritative native-81 transport is computed separately from every
reconstruction, never from an average over seeds. For particle, electron
internal-energy, ion internal-energy, and total internal-energy transport:

- strict-face relative L2 \(\le0.25\), RMS ratio in \([0.75,1.25]\),
  correlation \(\ge0.85\), weighted sign disagreement \(\le0.15\);
- separatrix relative L2 \(\le0.20\), absolute normalized bias \(\le0.10\),
  RMS ratio in \([0.80,1.20]\), correlation \(\ge0.90\), weighted sign
  disagreement \(\le0.10\);
- at least seven of eight separatrix blocks pass relative L2 \(\le0.30\),
  absolute normalized bias \(\le0.15\), correlation \(\ge0.80\), and weighted
  sign disagreement \(\le0.15\).

The complete O1 gate is the conjunction of preprocessing round trip, field,
spectrum, cross-field, transport, positivity, shape, finiteness, and boundary
bypass checks. Aggregate scores cannot override a failed component.

## 7. Uncompressed O2 references

All references use the same target frames as the learned models and operate in
standardized full-field space without a learned codec:

1. **Persistence:** copy the latest context state.
2. **Two-frame linear extrapolation:** for `C5P-H2` only,
   \(2x_t-x_{t-1}\).
3. **Toroidal spectral AR(1):** fit one complex least-squares multiplier per
   state channel and stored mode \(k\) using training pairs only, pooling over
   training time, \(x\), and \(y\). Use a fixed relative ridge of \(10^{-8}\).
   Fit `Bphi` real AR(1) coefficients per radial side and poloidal index.

The AR coefficients, training sufficient statistics, and hashes are saved.
No reference is tuned on validation. The best applicable reference is chosen
separately for each reported aggregate metric, and its identity is shown.

The codec target round trip is also reported as an O1 ceiling, but it is not a
forecast and cannot be counted as O2 skill.

## 8. O2 deterministic transition model

O2 uses only the first codec topology accepted by Section 4. Each O2 seed uses
the codec with the same seed; codecs are frozen.

### Volume transition

The volume transition is a deterministic LOLA-style masked ViT with:

~~~text
hidden channels: 512
transformer blocks: 16
attention heads: 4
FFN factor: 4
qk normalization: true
rotary coordinates: true
dropout: 0.05
stochastic/noise features: 0
activation checkpointing: true
~~~

An ordered trajectory consists of the context frames followed by one
all-zero, explicitly masked target slot. The context mask is supplied as a
conditioning channel. Loss is evaluated only on the target slot.

The accepted codec topology determines a fixed spatial tokenization:

| Codec | latent patch | tokens per frame |
|---|---:|---:|
| `dcae_l20` | `[1,1,1]` | \(8\times4\times22=704\) |
| `dcae_l10` | `[2,2,1]` | \(8\times4\times22=704\) |

Thus the transformer depth, width, attention, and token grid are unchanged by
the codec escalation. `C5P-H1` and `C5P-H2` have identical
parameterization and differ only in sequence length. They are trained
separately from the same initialization seed, so their fitted weights are not
shared.

Latent means and standard deviations are fit per latent channel using encoded
training frames only and saved. The network predicts a standardized latent
increment:

\[
\widehat z_{t+1}^{\,s}
=
z_t^s + f_\theta({\rm context}, {\rm mask}).
\]

The increment is unstandardized and decoded by the frozen codec. The training
loss is equal-channel standardized field MAE after decoding, not latent MSE.

### E6B boundary path

For `E6B-H1`, standardized `Bphi` is embedded by a
`64 -> 256 -> 256` SiLU MLP and modulates every transformer block through
AdaLN. The next boundary is predicted as a residual by a zero-padded
one-dimensional convolutional head. Its input at each poloidal location is:

- current inner and outer `Bphi`;
- the current inner and outer instantaneous elliptic targets obtained from the
  reconstructed current `phi`.

The head has width 128, kernel size 3, four residual blocks, SiLU activations,
and two output channels. It predicts
\(\widehat B_{\phi,t+1}=B_{\phi,t}+\Delta B_\phi\).
No future target or future boundary enters the input.

E6B loss gives equal weight to the six standardized volume-channel MAEs and
the mean of the two standardized boundary-side MAEs:

\[
\mathcal L_{\rm E6B}
=
\frac{
\sum_{c=1}^{6}{\rm MAE}_c + {\rm MAE}_{B_\phi}
}{7}.
\]

C5P loss is the mean of its five channel MAEs. No O2 loss contains a
physics-derived quantity.

## 9. O2 optimization and checkpoint rule

Use seeds `1701,1702,1703`. For every state arm:

| Item | Frozen value |
|---|---|
| epochs | 200 |
| targets per epoch | every one of the 430 training targets exactly once |
| microbatch | 1 |
| gradient accumulation | 16 |
| effective batch | 16 except the final partial step |
| optimizer | AdamW |
| learning rate | \(2\times10^{-4}\) |
| betas | \((0.9,0.95)\) |
| weight decay | \(10^{-4}\) |
| gradient norm clip | 1.0 |
| warmup | 10 epochs, linear |
| decay | cosine to \(2\times10^{-6}\) |
| validation | all 126 targets, chronological, every epoch |

The final partial accumulation step is divided by its actual microbatch count;
it is not dropped or given full-batch weight.

The selected checkpoint is the earliest epoch with the lowest full-validation
equal-channel data loss. Physics metrics are not used to choose an epoch.
Training completes the fixed budget without early stopping.

## 10. O2 metrics and gate

Validation is divided into six contiguous 21-target blocks. Report all native
state channels and the common `[Ne,Pe,Pi,phi,Vi]` view. E6B uses its predicted
boundary for elliptic reconstruction; no truth boundary reset is allowed.

Required outputs include:

- per-field RMSE, MAE, bias, variance ratio, and anomaly correlation;
- error relative to every uncompressed reference;
- mode power, realization coherence, phase error, and mode lifetime at one
  step;
- `Ne-phi`, `Pe-phi`, and `Pi-phi` coherence and cross-phase;
- authoritative member-wise particle and heat transport;
- E6B boundary error by side and derived-potential error;
- finite/non-positive counts;
- parameters, accelerator hours, peak memory, and inference time.

A seed passes O2 only if all of the following hold:

1. Aggregate equal-channel RMSE and MAE are both lower than the best applicable
   uncompressed reference.
2. At least four of five common-view fields improve over persistence in RMSE,
   and no common field exceeds \(1.05\) times persistence RMSE.
3. Every training-material `k=1..7` field band has power ratio in
   \([0.75,1.30]\) and forecast/truth realization coherence at least 0.80.
4. Every primary pair-band has cross-phase error at most \(20^\circ\) and
   coherence change at most 0.15.
5. Strict-face transport has relative L2 at most 0.40, correlation at least
   0.70, and weighted sign disagreement at most 0.20 for all four quantities.
6. Separatrix transport has relative L2 at most 0.30, absolute normalized bias
   at most 0.15, correlation at least 0.80, and weighted sign disagreement at
   most 0.15 for all four quantities.
7. Items 2--6 satisfy their applicable thresholds in at least five of six
   validation blocks.
8. E6B additionally improves aggregate standardized `Bphi` RMSE over
   boundary persistence, with neither radial side worse than \(1.05\) times
   persistence.
9. All required values are finite and no forbidden data access, clipping,
   target reset, or future-truth use occurred.

An arm is accepted only if all three seeds pass. Seed averaging cannot rescue
a failed seed.

## 11. State decision and next gate

Interpret comparisons only as follows:

- `C5P-H2 - C5P-H1` measures the value of one additional observed frame;
- `E6B-H1 - C5P-H1` compares exact-state versus five-field interfaces;
- `E6B-H1 - C5P-H2` tests whether short observed history substitutes for the
  omitted exact state.

If no arm passes, stop and report deterministic one-step failure. If only one
passes, that arm is the sole O3 candidate. If multiple arms pass, retain every
passing arm through the first short autonomous O3 comparison; do not choose
from O2 RMSE alone.

No diffusion, FGN, PDE-Refiner, stochastic residual model, assimilation,
diagnostic ranking, or 85606 evaluation is authorized by this protocol.
Those require a new committed protocol after O2.

## 12. Implementation, smoke, and execution gates

Before any full run:

1. port the minimum attributed model code into `src/tcv_diagnostics/models/`;
2. add tests for padding, stride shapes, encode/decode shape, history order,
   target masking, target-only loss, toroidal-roll alignment, latent
   normalization, residual semantics, Bphi causality, and parameter sharing;
3. add an overfit test on a synthetic known-answer batch;
4. run the complete CPU test suite on Rocky 9;
5. run a non-scientific one-GPU smoke using at most 16 frames and two epochs;
6. verify clean checkout, finite gradients, checkpoint reload identity, and
   artifact schema.

The intended full O1 command is:

~~~bash
sbatch --array=0-5 cluster/phase2_o1_train_codecs.sbatch dcae_l20
~~~

Array indices map in lexicographic order over
`state_family=[c5p,e6b]` and `seed=[1701,1702,1703]`. The predeclared R2
command differs only by the final argument `dcae_l10` and is forbidden unless
the R1 gate fails.

O1 evaluation and O2 commands are launched only by committed scripts that
verify their exact upstream artifacts. Every execution directory must contain:

- frozen config and protocol/manifest hashes;
- Git commit and dirty state;
- consumed shard and checkpoint hashes;
- seed and RNG configuration;
- training history and selected-epoch record;
- final and selected checkpoints;
- raw metric tables and block metrics;
- command, Slurm metadata, software/hardware identity;
- runtime, peak memory, and output-size accounting.

Existing job directories and checkpoints are never overwritten. Large
checkpoints, caches, and predictions remain on Ceph; compact results and hashes
are tracked in Git.

## 13. Claims

Passing O1 means the selected codec reconstructs one development-run state
well enough under the frozen field, spectral, joint, and transport criteria.

Passing O2 means a deterministic, teacher-forced one-step model has skill on a
later background of the same 85604 trajectory, conditional on the selected
state interface and codec.

Neither statement means the model can sustain an autonomous rollout or produce
a calibrated ensemble. Those are distinct downstream hypotheses.
