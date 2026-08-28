# Variogram fine-tuning protocol amendment: forecast samples and fair score

**Frozen prospectively:** 2026-08-28, after two read-only implementation audits
and before score preflight, timing/memory output, or any model fine-tuning

This amendment supersedes Sections 4, 6, 7, and the seed interpretation in
Section 8 of
`POST_ECRD_OLD_85604_PGL_BOUNDED_VARIOGRAM_FINETUNE_2026-08-28.md`.
All scope, data, architecture, sampler, mean-freeze, 85606 prohibition, and
non-overwrite rules remain in force.

## Why this amendment is necessary

The first protocol proposed computing auxiliary scores on multiple clean
estimates from corrupted training truth. Those estimates condition on
`clean + sigma*noise`; they are denoising probes, not samples from the
forecast distribution available at inference. Using them would not answer
the stated scientific question.

The existing PGL forecast sampler is also decorated with `no_grad`. A new
differentiable implementation must reproduce its forward output exactly while
activation-checkpointing denoiser calls. The primary sampler itself is not
changed.

## 1. Genuine forecast samples

For every training context, first compute the four-frame deterministic mean
from the frozen selected mean branch without gradient. Then draw four pure
persistent global--local initial-noise trajectories from fixed keyed seeds and
run each through the unchanged EDM schedule:

- 18 sigma steps;
- `sigma_max=80`, `sigma_min=0.002`, `rho=7`;
- Heun correction at every nonzero next sigma;
- 35 denoiser evaluations per member;
- no target truth in model input;
- no member interaction;
- no post-hoc spread multiplier.

Backpropagation uses activation checkpointing for every denoiser evaluation.
A forward-regression test must match `sample_normalized` for identical weights,
context, mean, initial noise, and schedule within `2e-6` relative L2 and
`2e-5` maximum absolute error in float32. Failure blocks all four arms.

The ordinary EDM denoising objective remains the base loss and uses one fixed
keyed sigma/noise probe per training window in every arm. The four full-sampler
members exist only for enabled auxiliary scores and their matched arm-A
compute/control probe; target truth never enters their sampler.

## 2. Fair order-one variogram

For one pair `(i,j)`, let

```text
g_m = |x_mi - x_mj|,    g_y = |y_i-y_j|,    M=4.
```

The training score is the fair finite-ensemble estimator

```text
V_fair = (g_y - mean_m g_m)^2 - sample_variance_m(g_m) / M.
```

The ordinary nonnegative score is logged alongside it. The fair correction is
required because the ordinary square contains an extra `Var(g)/M` term that
would otherwise penalize ensemble variation in this already underdispersed
model. The fair estimator can be negative for a finite batch; that is expected
and is not clipped.

Known-answer tests additionally require the exact identity

```text
V_ordinary - V_fair = sample_variance(g_m)/M
```

under the frozen weighting convention.

## 3. Spatial and temporal pair banks

Use six geometry-only physical-distance bins and **1,024 fixed pairs per bin**.
Physical distance uses the minimum angular separation in the periodic
`2*pi/5` wedge:

```text
d^2 = R_i^2 + R_j^2 - 2 R_i R_j cos(delta_phi_min) + (Z_i-Z_j)^2.
```

Field positions use cropped cell-center `Rxy,Zxy`. Exact-separatrix transport
positions use `Rxy_xlow,Zxy_xlow` at the frozen separatrix face and rows
`y=8..23`. Every bin receives equal total score weight. Store meter-valued bin
edges, pair endpoints, seed `856040828`, and a SHA-256 identifier.

Temporal scores concatenate the observed current state `t0` with the four
generated future states. Use identical field/location pairs at lags 1, 2, 3,
and 4, with equal total weight per lag. The current truth is duplicated across
members and is available at inference, so current-to-future pairs do not leak
future truth. This construction also makes a bias applied only to generated
future frames detectable, while correctly retaining invariance to a common
bias applied to the entire scored trajectory.

## 4. Authoritative transport truth and quantities

Transport-space training uses all four frozen quantities with equal total
normalized weight:

- particle transport;
- electron internal-energy transport;
- ion internal-energy transport;
- their summed total internal-energy transport.

Each generated member is decoded, Fourier-resampled from model 88 to native 81
toroidal cells, and passed independently through the differentiable
authoritative operator. The 88-to-81 map is a fixed matrix constructed by
applying the frozen SciPy resampler to every 88-dimensional basis vector; its
matrix bytes are hashed and its Torch result must match SciPy before launch.

Training truth for transport is the existing raw native-81 85604 truth, not a
resampled copy of standardized model targets. Precompute current and four
future local separatrix contributions for authorized training windows with the
existing NumPy authority, hash the artifact, and compare the Torch
implementation to it before C or D can launch.

## 5. Frozen normalization and arm weights

Use 32 unique chronological control starts
`floor(linspace(0,427,32))`, four sampler members, the frozen pair banks, and
fixed noise keys. Compute once at the unmodified checkpoint:

- `C0`: original EDM loss;
- `C_field_spatial`, `C_field_temporal`;
- one spatial and temporal transport control for each of four quantities.

Every control must be finite, positive, and hash closed. Define

```text
F = 0.5 * (V_field_spatial/C_field_spatial
           + V_field_temporal/C_field_temporal)

T = (1/8) * sum_q (V_q_spatial/C_q_spatial
                   + V_q_temporal/C_q_temporal)
```

with one preregistered auxiliary budget `lambda=0.10`:

```text
A = L0
B = L0 + lambda*C0*F
C = L0 + lambda*C0*T
D = L0 + 0.5*lambda*C0*(F+T).
```

Thus B, C, and D each begin with the same 10% auxiliary budget; D is not
silently twice as strong. No lambda sweep, adaptive normalization, or
target-dependent rescaling is allowed.

## 6. Revised bounded budget and mandatory production smoke

Full-sampler gradients are substantially more expensive than denoising. The
one-seed screen is therefore exactly:

- one complete epoch;
- 428 windows;
- gradient accumulation over two windows;
- exactly 214 optimizer updates;
- fixed final EMA, with no checkpoint selection;
- fresh AdamW on stochastic parameters only;
- constant stochastic learning rate `1e-6`, the parent training schedule's
  terminal rate;
- betas `(0.9,0.99)`, weight decay `1e-4`, clip norm 1, EMA `0.999`;
- bfloat16 for the network, with variogram accumulation and authoritative
  transport outside autocast in float32 or float64 as frozen by the equivalence
  test;
- seed 1702 and identical order/noise/pair banks in all arms.

Before the array, run exactly one production-size optimizer update for each
distinct objective path needed to establish memory and wall time: A, B, C, and
D if transport equivalence passes. The smoke uses a disposable output and may
not be interpreted scientifically. It must fail closed rather than silently
reducing members, sampler steps, pairs, resolution, or objective terms.

If one update cannot fit a generic Rusty GPU under activation checkpointing,
stop and write a dated resource amendment before changing the algorithm.

## 7. Final evaluation

The one-seed screen uses the full frozen population of 36 starts and all 32
members from the existing scientific seed bank, not an M16 approximation.
Every original seven-family horizon-one/horizon-four gate remains required.
Add fair and ordinary field/transport variograms by physical distance and
temporal lag, plus global/local latent-use interventions; none replaces an
original gate.

## 8. Confirmation seeds

Changing only fine-tuning RNG seeds from the same seed-1702 parent is not a
three-model-seed confirmation. If one arm passes, a true three-seed comparison
requires matching seed-1701 and seed-1703 base persistent checkpoints trained
under the already frozen base-model recipe, followed by the selected objective
at each matching seed. This later escalation requires a dated execution
amendment and is not part of the one-seed screen.

The later combined-944 experiment also remains blocked on a cadence-aware
amendment: the current four-frame model has no physical lead-time input, while
the old and NERSC segments have different cadences. They may not simply be
concatenated and called cadence aware.
