# Phase 3 B5 joint field-residual EDM implementation/smoke protocol

**Decision status:** frozen after job-6901393 residual-audit completion and
before B5 model implementation, optimization, sampling, or validation access

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implement and run one bounded full-field H100
mechanical smoke of a joint five-field conditional residual EDM

The machine-readable authority is
`paper0/manifests/phase3_b5_field_residual_edm_smoke_85604.json`.

## 1. Question answered by this smoke

Job `6901393` measured the exact frozen H1 training residual. The result is
small at one-step lead but strongly geometry dependent, cross-field coupled,
and concentrated in non-axisymmetric modes. Its fine pattern changes between
adjacent saved frames while its RMS amplitude remains context dependent over
multiple frames.

The next question is strictly mechanical:

> Can one exact, joint, full-field residual denoiser fit in one H100, execute a
> finite optimization loop, reload exactly, and produce distinct finite
> field-space samples with the required axes and boundary behavior?

This smoke does not estimate validation skill, transport fidelity,
calibration, uncertainty, or rollout quality. It does not select a scientific
checkpoint. Passing it authorizes only preparation of a separate full-training
and validation protocol.

## 2. Literature-motivated choice and its limits

The primary architecture is a conditional 3-D U-Net trained with the EDM
denoising objective on the residual of a frozen deterministic mean. This is a
minimal field-coordinate adaptation of the regression-plus-residual-diffusion
decomposition used by CorrDiff, not a new architecture claim. The standard
EDM preconditioning and sampling equations are taken from Karras et al.
GenCast motivates joint multivariable residual normalization and fresh noise
at every autoregressive step, but its two-state weather system and enormous
training corpus do not validate this plasma application.

Primary references are:

- Mardani et al., *Residual Corrective Diffusion Modeling for Km-scale
  Atmospheric Downscaling*, <https://arxiv.org/abs/2309.15214>;
- Karras et al., *Elucidating the Design Space of Diffusion-Based Generative
  Models*, <https://arxiv.org/abs/2206.00364>;
- Price et al., *Probabilistic weather forecasting with machine learning*,
  <https://www.nature.com/articles/s41586-024-08252-9>;
- the official PhysicsNeMo CorrDiff implementation notes,
  <https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html>.

The official CorrDiff guidance suggests at least 50,000 samples and often
millions of processed samples. Paper 0 has only 430 adjacent training targets,
which are not independent simulations. This data-scale mismatch is a central
risk and is why this authority stops at a bounded smoke.

## 3. Exact parent, targets, and conditioning

The deterministic mean remains the exact frozen C5P-H1 seed-1701 checkpoint:

`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.

The smoke uses the immutable job-6901393 H1 training forecast:

`d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`.

For target frame `t`, define

\[
\mu_t=\mu_{\mathrm{H1}}(x_{t-1}),
\qquad
r_t=x_t-\mu_t.
\]

The dynamic condition is exactly

\[
c_t=\operatorname{concat}(x_{t-1},\mu_t),
\]

with ten channels in field order `[Ne,Pe,Pi,phi,Vi]` for each five-channel
block. The target truth is never included in the condition. Absolute time,
normalized frame index, shot label, diagnostics, region masks, future truth,
and physics-derived quantities are prohibited model inputs.

The U-Net receives two additional deterministic internal coordinate maps for
the nonperiodic computational `x` and `y` axes, linearly scaled to `[-1,1]`.
They are fixed position metadata, not dynamic plasma channels. No absolute
toroidal coordinate is supplied, preserving circular-shift equivariance in
the stored periodic direction.

The first smoke uses exactly target frames `[2,10)`, eight complete
`[5,64,32,88]` residuals. It reads no guard or validation frame. The use of
eight adjacent training targets is an implementation stress test and carries
no sample-size or generalization interpretation.

## 4. Frozen residual normalization

The denoising target stays in decoded standardized field space but is divided
by its job-6901393 global population standard deviation:

\[
z_{t,c}=r_{t,c}/s_c.
\]

In canonical field order, the exact scales are:

| field | `s_c` |
|---|---:|
| `Ne` | 0.05503048051260375 |
| `Pe` | 0.04825854004472835 |
| `Pi` | 0.06096460194410047 |
| `phi` | 0.04632595196855943 |
| `Vi` | 0.10251610501339582 |

No mean or axisymmetric bias is subtracted. The generator must be able to
learn nonzero residual mean structure. No pointwise, regional, spectral, or
transport weighting is applied.

## 5. Frozen full-field network

The smoke architecture is `B5-H1-JOINT-FIELD-EDM-UNET3D-MINI`:

- noisy input channels: 5 joint normalized residual fields;
- dynamic condition channels: 10 (`x_{t-1}` plus frozen H1 mean);
- internal static position channels: 2 (normalized `x` and `y`);
- base channels: 32;
- encoder channel multipliers: `[1,2,4,4]`;
- two residual blocks per encoder and decoder resolution;
- three stride-2 downsampling operations over all three spatial axes;
- exact resolution path:
  `64x32x88 -> 32x16x44 -> 16x8x22 -> 8x4x11`;
- kernel size: 3 for every spatial convolution;
- nonperiodic zero padding on `x` and `y`;
- circular padding on stored toroidal `z`;
- GroupNorm with at most eight groups, SiLU activations, and no dropout;
- skip connections between matching encoder and decoder resolutions;
- trilinear upsampling to the exact saved skip shape followed by a mixed-
  boundary 3-D convolution;
- 256-feature sine/MLP noise embedding;
- one affine scale-and-shift FiLM projection from the noise embedding in every
  residual block;
- zero-initialized final five-channel convolution;
- no attention, DCAE, latent bottleneck, spectral operator, physics loss, or
  trainable deterministic parent.

The architecture processes the complete `64x32x88` volume. There is no patch
fallback in this authority. If it exceeds the H100 budget, the smoke stops and
a separately frozen x/y-patch protocol is required. Any later patch must span
at least 17 cells in `x`, 3 cells in `y`, overlap by at least 8 and 1 cells
respectively, retain all 88 `z` cells, and include global position metadata.

## 6. Frozen EDM objective

The normalized residual has `sigma_data=1`. For noise level `sigma`, define

\[
c_{\mathrm{in}}=(\sigma^2+1)^{-1/2},
\quad
c_{\mathrm{skip}}=(\sigma^2+1)^{-1},
\quad
c_{\mathrm{out}}=\sigma(\sigma^2+1)^{-1/2},
\quad
c_{\mathrm{noise}}=\tfrac14\log\sigma.
\]

With `z_sigma = z + sigma epsilon`, the denoised prediction is

\[
D_\theta(z_\sigma;c_t,\sigma)
=c_{\mathrm{skip}}z_\sigma
+c_{\mathrm{out}}F_\theta(c_{\mathrm{in}}z_\sigma,c_t,
c_{\mathrm{noise}}).
\]

Training samples

\[
\log\sigma\sim\mathcal N(-1.2,1.2^2),
\qquad \epsilon\sim\mathcal N(0,I),
\]

and minimizes the equal-element, equal-normalized-channel EDM loss

\[
\mathcal L
=\frac{\sigma^2+1}{\sigma^2}
  \operatorname{mean}\left[(D_\theta-z)^2\right].
\]

This is the only training loss. No flux, spectrum, cross-phase, conservation,
PDE residual, region, event, calibration, or diagnostic term is permitted.

## 7. Bounded optimization budget

The smoke uses seed `1701`, one H100, bfloat16 autocast with FP32 loss
accumulation, and no distributed training. It performs exactly 64 optimizer
steps, cycling deterministically over the eight smoke targets. The optimizer
is AdamW with learning rate `1e-4`, betas `(0.9,0.99)`, zero weight decay, and
global gradient clipping at `1.0`. Microbatch size is one target and there is
no gradient accumulation.

Per-step noise is reproduced from an immutable key containing smoke seed,
optimizer step, and target frame; it is not stored as a full-volume bank. A
separate fixed four-target denoising probe uses seed `67004` and targets
`[2,6)`. Training-order/noise and sampling seeds are respectively `67001`,
`67002`, and `67003`.

The job records all 64 losses, learning rates, pre-clip gradient norms,
parameter finiteness, wall time, parameter count, and peak allocated CUDA
memory. It logs online to W&B, while local hash-indexed artifacts remain the
scientific authority.

## 8. Frozen sampler probe

The sampler is the deterministic EDM probability-flow ODE with the Karras
power-law schedule:

- steps: 18;
- `sigma_max=80`;
- `sigma_min=0.002`;
- `rho=7`;
- Heun correction except on the final step to zero;
- stochastic churn disabled.

For one target condition, sample two members from two independently seeded
initial standard-normal volumes. Convert generated normalized residuals back
with the exact `s_c`, add the frozen H1 mean, and return canonical
`[batch,member,future_time,channel,x,y,z]` shape `[1,2,1,5,64,32,88]`.

This two-member probe tests plumbing and non-collapse only. It is not a CRPS,
coverage, spread-skill, calibration, or physical-fidelity result. A later
autoregressive model must draw a new initial noise volume at every rollout
step; trajectory-constant noise is not the primary B5 method.

## 9. Mechanical gates

The smoke passes only if all of the following hold:

1. the expected clean Git commit and every input hash verify;
2. the complete Rocky 9 test suite passes before optimization;
3. only 85604 targets `[2,10)` and their frame-`t-1` contexts are read;
4. the exact architecture and optimizer budget execute for 64 steps;
5. peak allocated CUDA memory is below 75 GiB;
6. every loss, gradient norm, parameter, sample, and composed field is finite;
7. the final fixed-probe denoising loss is strictly lower than the initial
   fixed-probe loss;
8. checkpoint reload reproduces one fixed denoiser output bit for bit;
9. a circular shift of noisy residual, dynamic condition, and output along
   `z` passes a frozen FP32 equivariance tolerance;
10. two sampler members have nonzero residual and field RMS difference;
11. sampler axes, field order, and composition with the deterministic mean are
    exact;
12. the frozen H1 checkpoint and forecast artifact remain unchanged;
13. W&B finishes online and every local output is hash indexed;
14. guard, validation, 85606, O3, assimilation, and ranking remain untouched.

A failed loss-decrease gate means only that this exact smoke did not establish
short-loop trainability. A memory failure means only that the frozen
full-field mini U-Net did not fit. Neither failure authorizes silent tuning or
automatic patching.

## 10. Required outputs

The immutable job directory must contain:

- the resolved model and optimizer configuration;
- the exact residual scales and source hash;
- all 64 optimization records;
- initial and final fixed-probe losses;
- selected smoke checkpoint and reload probe;
- two-member sampler probe and compact metrics;
- parameter count, runtime, CUDA peak memory, environment, command, Slurm, and
  complete-test records;
- W&B metadata;
- `result.json` and `artifact_sha256.txt`.

Large model and sample artifacts remain on Ceph and are not uploaded to W&B.

## 11. Post-smoke boundary

Passing this smoke authorizes interpretation and preparation of a separate B5
full-training/validation protocol only. It does not authorize full training,
checkpoint selection on validation, scientific scoring, replication, O3,
assimilation, diagnostic ranking, or access to 85606.
