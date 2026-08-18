# Phase 3 B2 matched latent-diffusion protocol

**Decision status:** frozen after the completed deterministic O2 result and
before B2 implementation, smoke testing, or training

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implementation, CPU tests, and one bounded Rocky 9
GPU smoke for a C5P-H2 masked latent-diffusion baseline

The machine-readable authority is
paper0/manifests/phase3_b2_ldm_85604.json.

## 1. Why this experiment exists

The accepted C5P-dcae_l10 codec passed the complete O1 reconstruction gate at
all three frozen seeds. The deterministic O2 transition then beat the
uncompressed references on ordinary one-step field error at all six
arm-by-seed combinations, but every model failed the realization-level
spectral and nonlinear-transport gates. The failure therefore appears after
encoding but before autonomous rollout feedback.

A deterministic forecast is forced to produce one conditional estimate. At a
cadence of 3.131905426352636 microseconds, unresolved or weakly observed
dynamics may instead require a conditional distribution. B2 asks whether a
LOLA-style latent diffusion model can represent that distribution without
changing the accepted fields, codec, split, or training-only normalization.

B2 does not assume that stochasticity will repair the physics. It tests that
hypothesis. A sharp-looking member, low ensemble-mean field error, or nonzero
spread is not sufficient evidence of transport fidelity.

## 2. Exact scientific question

For each validation target, condition on the two exact preceding C5P frames
and draw multiple possible next frames. The five directly modeled fields are

~~~text
[Ne, Pe, Pi, phi, Vi]
~~~

The model estimates only the one-saved-step conditional distribution. It does
not consume a previous model prediction, and it is not O3 or O4.

The primary arm is B2-LDM-H2. H2 is retained because C5P omits independent
vorticity, electron momentum, and boundary memory, so ordered history is the
more conservative conditioning state. The completed deterministic H2 arm is
the exact architecture-matched deterministic comparator. The slightly lower
field error of deterministic H1 remains a secondary reference and is not
relabeled.

No absolute time, normalized frame number, future truth, diagnostic value, or
shot label is supplied to the model.

## 3. Immutable data and representation

B2 inherits the verified job-6893525 model dataset and the Phase 2 split:

- training frames: [0,432);
- training targets: [2,432), all 430 once per epoch;
- guard frames: [432,496), never loaded;
- validation frames: [496,624);
- validation targets: [498,624), all 126 chronologically;
- native cadence: 3.131905426352636 microseconds;
- toroidal period: zperiod=5;
- Fourier mapping: n=5k;
- standardized volume shape: [5,64,32,88].

The volume normalization from job 6893525 is immutable and was fit on the
training region only. Ne uses the already accepted log-offset transform;
Pe, Pi, phi, and Vi use the already accepted identity transforms followed by
scalar standardization. No clipping or refitting is allowed.

Each model seed uses the same-seed accepted C5P-dcae_l10 checkpoint. The codec
is always in evaluation mode and all codec parameters have gradients disabled.
Per-latent-channel population moments are fit only on frames [0,432), using the
frozen codec for that seed. The standardized latent grid is [32,16,8,22].

## 4. Attributed B2 model

The model is a conditional, masked latent diffusion model adapted from the
official LOLA implementation at upstream commit
21a4354b327e6e5ee06da5075ba3bd1dd88c61f1.

The clean training object is the ordered three-frame standardized latent
trajectory [context t-1, context t, target t+1]. The first two slots are
marked observed and the target slot is marked unknown. One nonredundant binary
mask channel is concatenated to the noisy latent input; expanding the same
mask to 32 identical channels would add no conditioning information.

The denoising backbone is the Paper 0 LOLA-style four-dimensional ViT with:

- 32 latent input and output channels;
- 256 noise-time embedding features;
- 512 hidden token channels;
- 16 transformer blocks;
- 4 attention heads;
- FFN factor 4;
- Q/K normalization and RoPE enabled;
- dropout 0.05;
- activation checkpointing enabled;
- latent patch [time=1,x=2,y=2,z=1];
- 704 tokens per frame and 2112 tokens per three-frame trajectory;
- global self-attention;
- one shared noise-time modulation supplied to every transformer block.

The time-conditioned blocks use the same AdaLN-zero mechanism as LOLA. This is
diffusion time, not physical simulation time.

## 5. Noise model and objective

The forward perturbation uses alpha(t)=1 and the LOLA log-logit noise schedule
with sigma_min=0.001, sigma_max=1000, scale=1, and shift=0. Noise time is drawn
uniformly on [0,1]. Gaussian noise is sampled independently for every training
window.

The denoiser uses LOLA's EDM-style input, output, skip, and noise
preconditioning. Observed context slots are replaced by their correctly
scaled clean values before denoising, and the binary mask is provided to the
backbone.

The training objective is the original LOLA weighted denoising mean-squared
error over the complete masked three-frame trajectory. This choice defines B2
as the current LOLA-style control rather than a target-only repair. Target-slot
and context-slot losses must also be logged separately so the known context
cannot silently dominate interpretation.

No flux, spectrum, cross-phase, coherence, gradient, conservation, PDE
residual, blob label, calibration score, or other physics-derived quantity is
used as a training loss.

## 6. Frozen optimization budget

Full-run settings are recorded prospectively even though full training is not
yet authorized:

- seeds: 1701, 1702, 1703;
- epochs: 200;
- training targets per epoch: 430;
- microbatch: 1;
- gradient accumulation: 16;
- optimizer steps per epoch: 27;
- total optimizer steps: 5400;
- optimizer: AdamW;
- learning rate: 0.0001;
- beta values: [0.9,0.99];
- weight decay: 0;
- schedule: cosine to zero over the full optimizer-step budget;
- warmup: none;
- gradient-norm clip: 1;
- precision: bfloat16 autocast;
- early stopping: disabled.

These optimizer settings are the historical TCV LOLA diffusion defaults. The
number of targets, epochs, and optimizer steps are matched to deterministic
O2 rather than to the much larger and leakage-incompatible legacy run.

## 7. Validation and checkpoint selection

Validation uses all 126 targets every epoch without augmentation. Every target
is evaluated with an immutable CPU-generated noise time and Gaussian-noise
tensor derived from a fixed validation seed and target index. The same noise
bank is used across epochs and model seeds.

Checkpoint selection is the earliest epoch attaining the numerically lowest
full-validation, complete-trajectory LOLA denoising loss after all 200 epochs.
Target-slot and context-slot losses are reported but do not change the frozen
selection rule. Validation physics metrics are not used for checkpoint
selection.

## 8. Frozen one-step sampler

Scientific forecast generation will use the Azula 0.3.1 Adams-Bashforth
sampler used by the historical LOLA stack:

- 16 reverse steps;
- order 3;
- start time 1 and stop time 0;
- exact context conditioning at every denoising call;
- independent initial Gaussian noise for every ensemble member.

The canonical prediction axes are

~~~text
[batch, ensemble_member, future_time, channel, x, y, z]
~~~

and future_time has length one in this B2 experiment. Nonlinear diagnostics
must be computed member by member before ensemble aggregation. Flux from the
ensemble-mean fields is not an admissible replacement.

The scientific ensemble size, probabilistic acceptance metrics, and complete
B2 stop/go rule will be frozen in a separate evaluation protocol before full
training is authorized. This implementation protocol does not permit changing
the trained model based on that later evaluation.

## 9. Provenance boundary

The old f8 LOLA checkpoint is historical evidence only. It used
[Ne,Te,Ti,phi,Vi], the legacy adjacent split without a guard, a different
codec and latent grid, random validation rolls, a five-frame random-context
task, and an unmatched training budget. It is not a Paper 0 B2 checkpoint.

The new implementation may port only the minimum MIT-licensed LOLA model and
diffusion mechanics required here. The exact source paths, source commit,
modifications, file hashes, and tests must be added to paper0/PROVENANCE.md.
Azula 0.3.1 is an external MIT-licensed runtime dependency; its installed
version and source hashes must be captured by every Rusty job.

## 10. Implementation and smoke gates

Before any full training:

1. add known-answer tests for the log-logit schedule and EDM coefficients;
2. test mask semantics and prohibit future truth in predict;
3. test modulated-ViT shapes, gradients, and deterministic reload;
4. test canonical ensemble axes and member diversity;
5. test that codec parameters remain frozen;
6. test fixed validation-noise reproducibility;
7. test that target and context losses sum to the complete objective with the
   documented element weighting;
8. run the complete CPU suite;
9. run one Rocky 9 H100/H200 smoke at seed 1701 using at most 16 training
   targets and two epochs;
10. require finite losses and gradients, a reload-identical checkpoint, a
    two-member finite forecast with nonzero latent diversity, and successful
    online W&B logging.

The smoke is an implementation gate, not scientific evidence. It may not read
85606 and may not authorize O3, assimilation, or diagnostic ranking.

## 11. Stop/go boundary

This protocol authorizes implementation and one bounded smoke only. Full B2
training requires all of the following in a later committed amendment or run
manifest:

- passing local and Rocky 9 CPU tests;
- passing GPU smoke with immutable artifacts;
- a frozen probabilistic B2 evaluation protocol;
- a clean exact execution commit;
- paired codec paths and hashes for all three seeds;
- successful W&B preflight.

No threshold from deterministic O2 is loosened or retroactively reinterpreted.
The stochastic evaluator must distinguish trajectory accuracy from
distributional fidelity instead of selecting a lucky ensemble member.

## 12. Claims boundary

The strongest possible result from B2 on 85604 is that a masked latent
diffusion model produces a useful one-step conditional ensemble on later
85604 data while preserving member-wise field, spectral, cross-field, and
transport statistics and exhibiting defensible calibration.

It would not establish autonomous rollout skill, held-out 85606 performance,
experimental diagnostic realism, cross-shot generalization, assimilation
value, diagnostic ranking, or steering.
