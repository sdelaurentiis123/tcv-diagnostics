# Paper 0 ECRD model-development protocol

**Decision status:** frozen on 2026-08-20 after completion of Phase 3.5 and
before ECRD implementation, engineering smoke training, or new model results

**Development simulation:** 85604 only

**Sequestered simulation:** 85606 remains unopened

**Purpose:** train a small, controlled model ladder that tests whether deep
state conditioning and toroidal symmetry repair B5's joint-covariance failure.
This is model development, not another diagnostic phase or architecture zoo.

## 1. Authorization and scientific hypothesis

Phase 3.5 established the narrow statement:

> A single, fixed, condition-independent, global linear residual distribution
> learned from adjacent 85604 training frames does not describe later 85604
> residuals well.

It also found state-dependent residual summaries, a systematic period-four
H1/codec equivariance defect, better chronological variance transfer in
toroidal Fourier coordinates, and little H1 gain from a truth-assisted bulk
shift. These findings motivate a model intervention rather than another
cause-localization phase.

The prospective model hypothesis is:

> A plasma emulator fails at transport when stochastic innovation is modeled
> weakly conditionally in unstable coordinates. A toroidally equivariant
> transition with deep raw-state conditioning and joint multiscale innovation
> can recover transport-relevant covariance without a transport loss.

The working name is **ECRD**: equivariant, context-conditioned residual
diffusion.

This protocol authorizes implementation, bounded engineering smoke runs, and
the matched 85604 model-development ladder below. It does not authorize access
to 85606, assimilation, diagnostic ranking, control, or steering.

## 2. Immutable data discipline

All arms use the existing Paper 0 split and training-only normalization:

| purpose | frame or target interval | count | policy |
|---|---:|---:|---|
| raw training frames | `[0,432)` | 432 | model development |
| one-step training targets | `[2,432)` | 430 | optimization |
| guard | `[432,496)` | 64 | unread |
| raw validation frames | `[496,624)` | 128 | context and truth after training |
| validation targets | `[498,624)` | 126 | checkpoint and arm selection |
| held-out simulation | 85606 | — | unopened until a separate release record |

The three primary validation blocks are fixed as:

- `V00 = [498,540)`;
- `V01 = [540,582)`;
- `V02 = [582,624)`.

No boundary may be moved after model results. Temporal windows are not
independent physical simulations. The stored cadence is
`3.131905426352636 microseconds`. Only stored toroidal `z` is periodic. The
simulation has `zperiod=5`, so stored Fourier index `k` maps to full-torus mode
number `n=5k`.

Primary inputs are standardized C5P fields in the fixed order
`[Ne, Pe, Pi, phi, Vi]`. Absolute time, target truth, diagnostics, geometry
masks, and future information are not model inputs. Physics-derived quantities
are evaluation metrics only.

## 3. Transition decomposition

For the ECRD arms, the one-step forecast is

\[
\widehat{x}_{t+1}^{(m)}
=
\mu_\psi(x_{\leq t})
+
s\odot u_\theta^{(m)}(x_{\leq t}),
\]

with conditional mean

\[
\mu_\psi(x_{\leq t})
=
F_{\mathrm{sym}}(x_t)
+
s\odot b_\psi(x_{\leq t},F_{\mathrm{sym}}(x_t)),
\]

and four-phase H1 symmetrization

\[
F_{\mathrm{sym}}(x_t)
=
\frac{1}{4}\sum_{q=0}^{3}T_{-q}F_{\mathrm{H1}}(T_qx_t).
\]

The five positive scales `s` are exactly the frozen B5 training-residual
scales. They are not refit per arm. The H1 checkpoint and codec remain frozen
and are never optimized or reselected.

The small mean head predicts the normalized deterministic correction
`b_psi`. It is trained only by equal-field, equal-element squared error on the
normalized H1 residual. The diffusion target is the remaining normalized
innovation, with the current mean-head output detached from the diffusion
gradient. The denoiser is trained with the ordinary EDM field-denoising
objective. There is no flux, spectrum, cross-phase, coherence, PDE,
conservation, blob, or geometry loss.

## 4. Controlled arms

The primary ladder contains exactly four arms and three initialization seeds
`1701, 1702, 1703`.

| arm | parent mean | conditioning | toroidal downsampling | mean head | training shift | multiscale noise | history |
|---|---|---|---|---|---|---|---|
| `B5` | frozen H1 | existing input concatenation | existing stride 2 | no | no | white | 1 frame |
| `B5-Context` | frozen H1 | learned spatial FiLM at every U-Net resolution | existing stride 2 | no | random circular | white | 1 frame |
| `ECRD` | four-phase symmetrized H1 | learned spatial FiLM at every resolution | none along `z` | yes | random circular | global + mesoscale + local | 1 frame |
| `ECRD-History` | four-phase symmetrized H1 | same as ECRD | none along `z` | yes | random circular | global + mesoscale + local | 2 consecutive frames |

The existing seed-1701 B5 training and forecast artifacts may be reused after
hash verification. B5 seeds 1702 and 1703 are newly trained with the unchanged
B5 architecture. No unchanged B2, B3, or B4 run is repeated.

`B5-Context` is deliberately not called unconditional: historical B5 already
receives current C5P and the H1 mean at its input. The intervention is deep,
resolution-wise conditioning rather than the first appearance of context.

### 4.1 Deep raw-field conditioning

The conditioner receives every raw standardized C5P history frame plus the
frozen parent mean. It builds a spatial feature pyramid. At every encoder and
decoder resolution, learned spatial scale and shift fields modulate normalized
denoiser activations. No hand-engineered regime label or absolute time is
used.

### 4.2 Toroidal equivariance

ECRD uses mixed-boundary convolutions: zero padding on nonperiodic `x/y` and
circular padding on periodic `z`. All reductions and interpolation preserve
all 88 toroidal cells; only `x/y` may be downsampled. No absolute `z`
coordinate is supplied. Every history frame, target, and parent mean receives
one shared deterministic random circular shift during training.

Exact integer-shift equivariance of the ECRD generator and mean head is a
mechanical acceptance test. H1 symmetrization is an approximate repair of its
four codec stride phases and is audited separately from generator
equivariance.

### 4.3 Multiscale innovation

ECRD uses a full-rank Gaussian innovation built from independent components:

\[
\epsilon
=
\frac{
w_g\epsilon_{\mathrm{global}}
+w_m\epsilon_{\mathrm{meso}}
+w_l\epsilon_{\mathrm{local}}
}{\sqrt{w_g^2+w_m^2+w_l^2}}.
\]

The global component is broadcast over the volume, the mesoscale component is
piecewise constant on `4 x 4` nonperiodic cells while retaining all 88
toroidal cells, and the local component is full resolution. All five fields
are generated jointly. Training corruption and scientific sampling use the
same frozen component weights and seed expansion. The local component keeps
the covariance full rank. This changes covariance organization, not a
post-hoc spread multiplier.

Exact weights and parameter counts are frozen in the execution manifest after
synthetic-only shape/memory smoke and before any full training result. They may
be chosen only from parameter-count matching and numerical feasibility, never
from validation or physics performance.

## 5. Matching and training budget

All newly trained runs use:

- 100 complete epochs;
- 43,000 target presentations;
- 10,800 optimizer updates;
- microbatch one and accumulation over four targets;
- AdamW, betas `(0.9,0.99)`, zero weight decay;
- cosine learning rate from `1e-4` to `1e-6`;
- gradient clipping at 1;
- EMA decay 0.999 after each optimizer update;
- bfloat16 autocast with FP32 loss, optimizer state, and EMA;
- identical target order, diffusion-noise keys, and scientific member seeds
  across arms and model seeds wherever the architecture permits paired noise;
- no early stopping.

Trainable parameter counts must be within 10% of historical B5's
`11,604,709`; the execution manifest records exact counts and any width chosen
by the parameter-only matching rule. Accelerator hours and peak memory are
reported per arm and seed.

For models with a mean head, the frozen total objective is

\[
\mathcal L
=
\mathcal L_{\mathrm{EDM}}
+
\lambda_\mu\mathcal L_{\mathrm{mean}},
\qquad \lambda_\mu=1.
\]

The mean-head correction is detached inside the EDM target. Thus the mean head
cannot reduce EDM loss by absorbing stochastic variance; it is optimized by
the explicit field-only mean term.

## 6. Checkpoint selection

Candidate EMA checkpoints are written every five completed epochs. Validation
uses a fixed four-probe noise bank for all 126 targets and reports EDM loss
separately on V00, V01, and V02. Mean-head arms also report normalized
mean-head MSE per block.

The data-only checkpoint score is the unweighted mean of the three block mean
objectives. The objective is EDM loss for arms without a mean head and
`EDM loss + mean MSE` for arms with one. The earliest numerically lowest score
after the complete 100-epoch budget is selected. Physics metrics, sampled
forecasts, W&B values, and 85606 are forbidden in checkpoint selection.

## 7. Forecast and evaluation

Each selected checkpoint produces one truth-separated M32 one-step forecast
for `[498,624)` using the same 18-step, 35-evaluation deterministic EDM Heun
sampler and paired member seeds. Canonical axes are:

```text
[target, ensemble_member, future_time, channel, x, y, stored_toroidal_z]
```

Nonlinear transport is calculated independently for every ensemble member
before ensemble reduction. Metrics are reported for V00, V01, V02, every
model seed, and the across-seed summary. Block bootstrap uses noncircular
blocks of 12 frames, 2,000 replicates, and paired arm/B5 resampling. Intervals
are conditional on 85604 and do not turn temporal windows into independent
shots.

The required model-development measurements are:

- field RMSE, MAE, bias, variance, fair CRPS, coverage, and spread--skill;
- expected-member directional and toroidal power in `k=1..3`, `k=4..5`, and
  `k=6..7` (`n=5..15`, `20..25`, and `30..35`);
- member-wise Ne--phi cross-spectrum, cross-phase, and coherence;
- spatial covariance of all authoritative local transport contributions;
- integrated separatrix particle and electron/ion/total internal-energy
  transport distributions, CRPS, covariance, and spread--skill;
- exact generator/mean-head toroidal equivariance and approximate composed
  transition equivariance;
- inference time, network evaluations, memory, and accelerator hours.

The Phase 3.5 analog and conditional-mean probes are companion baselines only.
They run from stored forecasts or training residuals and cannot delay or veto
model training.

## 8. Frozen ECRD success rule

ECRD or ECRD-History is eligible for the held-out release only if one arm,
aggregated over all three seeds, satisfies every family below relative to the
three-seed B5 reference:

1. **Marginal forecast:** equal-field fair CRPS improves by at least 2% in
   each of V00, V01, and V02; median absolute log spread--skill error improves
   by at least 10% overall.
2. **Spectral retention:** median absolute log expected-member power-ratio
   error over the three material bands is no more than 1.05 times B5, and the
   count of B5 material power checks passed does not decrease.
3. **Density--potential dependence:** Ne--phi complex cross-spectrum error and
   coherence error each improve by at least 10% overall and improve in at
   least two of the three validation blocks; mean cross-phase error may not
   worsen by more than 2 degrees.
4. **Spatial transport covariance:** median relative covariance error across
   the four transport quantities improves by at least 15%, with improvement
   in at least three quantities.
5. **Integrated transport spread:** median absolute log spread--skill error
   improves by at least 20%, at least three of four integrated spread--skill
   ratios increase by at least 0.10 toward one, and their median is at least
   0.60.
6. **No local overdispersion:** every aggregate local-transport spread--skill
   ratio is at most 1.25 and at least three of four lie in `[0.80,1.25]`.
7. **Robustness:** the direction of the primary fair-CRPS and integrated-
   transport improvements holds for at least two of three model seeds, and
   the paired block-bootstrap 95% interval excludes no improvement for the
   across-seed aggregate.

No scalar score may substitute for these seven families. If both ECRD arms
pass, choose the simpler one-frame ECRD unless ECRD-History improves the
median integrated-transport spread-skill error by at least a further 10%.

## 9. Privileged-state ablation

Saved 85604 `NVe`, `Vort`, and `Bphi` may be added to one seed-1701 ECRD
ablation only after a manifest records their exact training-only
normalization, axes, units, and source hashes. `NVi` is not counted as new
information because `Vi=NVi/Ne` is already in C5P where defined. The
privileged ablation cannot select the primary Paper 0 model or authorize
85606; it only distinguishes an architecture bottleneck from a retained-state
bottleneck.

## 10. Decision and release logic

- If an ECRD arm passes Section 8, freeze its checkpoint, code, complete
  metric definitions, plots, and selection record. Only then may a separate
  explicit release record authorize exactly one 85606 evaluation.
- If no ECRD arm passes, do not scale generative capacity or start another
  residual-diagnostic phase. The next bottleneck is state/data: construct the
  exact evolved Hermes state and request independent restarts.
- Assimilation and diagnostic ranking remain closed until a selected model
  also passes its separately frozen 85606 forecast gate.
- Steering remains outside Paper 0 because these data contain no explicit
  action-conditioned trajectories.

