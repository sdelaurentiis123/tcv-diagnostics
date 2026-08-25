# Persistent global--local pilot physics evaluation

**Frozen:** 2026-08-25 after seed-1702 pilot job `6937586` completed its
mechanical and state gates, and before any stochastic forecast was sampled or
any physics diagnostic was evaluated for this model

**Development simulation:** 85604 only

**Held-out 85606 and newer NERSC data:** unopened and prohibited

## 1. Authorization boundary

The 20-epoch pilot completed exactly 4,280 optimizer updates.  Its selected
epoch-20 checkpoint has SHA-256
`4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb`.
The immutable pilot result has SHA-256
`0f3b9e71d32b16269ec93e1601af1d569827b7d67ed20884c38fe7015abe10b6`.

The mechanical gate passed.  Relative to the frozen deterministic parent, the
selected mean's equal-field MSE ratios were `0.9687217083048455` at one frame
and `0.9936877589794492` at four frames, both below the prospectively frozen
maximum of `1.05`.  The training result therefore authorizes one seed-1702
physics evaluation.  It does not yet authorize confirmation seeds, 85606,
assimilation, diagnostic ranking, control, or steering.

## 2. Forecast population

The forecast uses the existing 85604 validation region only.  One current C5P
state is read at each preregistered start; no future target is read during
forecast generation.

| block | current-frame starts |
|---|---|
| V00 | `497, 501, 504, 508, 511, 515, 518, 522, 525, 529, 532, 536` |
| V01 | `537, 541, 544, 548, 552, 555, 559, 562, 566, 570, 573, 577` |
| V02 | `578, 582, 585, 589, 593, 597, 600, 604, 608, 612, 615, 619` |

These 36 starts are approximately evenly spaced within each frozen
chronological block.  Targets are `t+1,...,t+4`; primary comparisons use
horizons one and four.  The guard interval `[432,496)` remains unread.

Each start receives 32 members.  The stored forecast axes are exactly

```text
[start, ensemble_member, future_time, channel, x, y, stored_toroidal_z].
```

All four future frames and all five fields `[Ne, Pe, Pi, phi, Vi]` are stored.
The candidate mean and frozen pre-pilot parent mean are stored alongside the
ensemble before target truth is opened.

## 3. Frozen stochastic sampling

The evaluation reuses the immutable B5/ECRD scientific uint64 seed bank with
SHA-256
`013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14`.
For current frame `t`, the 32 seeds are the bank row associated with one-frame
target `t+1`.  A seed is expanded through the model's frozen persistent
global--local noise law; it is not expanded as elementwise IID noise.

Sampling uses 18 EDM steps, 35 denoiser evaluations per member, no churn, no
classifier-free guidance, no post-hoc spread multiplier, and no member
interaction.  Member batching is an execution detail fixed at eight and may
not change the seed-to-member mapping.  The initial normalized-noise SHA-256
is stored for every start and member.

Forecast generation must close and SHA-256 hash the forecast artifact before
the scoring process may open any target truth.  Forecast generation and truth
scoring run as separate success-dependent jobs.

## 4. Frozen evaluation operators

The evaluator reuses the already validated B2/ECRD implementations for:

- gauge-aware field RMSE, MAE, fair CRPS, spread--skill, and coverage;
- expected-member toroidal and directional power;
- member-wise density--potential complex cross-spectrum, phase, and coherence;
- periodic model-88 to native-81 resampling;
- authoritative geometry-aware local and integrated radial ExB particle and
  internal-energy transport;
- the fixed 64-probe spatial transport-covariance sketch.

Potential is gauge fixed only for field and spectral comparisons.  Unmodified
physical potential is used for transport.  Nonlinear transport is computed
for every member before ensemble reduction.  Stored mode index maps to the
physical toroidal number through `n=5k` because `zperiod=5`.

Physics quantities are evaluation-only and were absent from training and
checkpoint selection.

## 5. Exact reductions and gates

All primary physics gates use future frame four unless a gate explicitly
names both horizons.  Per-field and per-transport-quantity values are retained
even where the gate uses a median.

1. **Field distribution:** equal-field fair CRPS must be strictly lower than
   the deterministic selected mean's equal-field MAE at both horizons one and
   four.  No per-field corrected spread--skill ratio may exceed `1.50` at
   either horizon.
2. **Spectral retention:** over each field and bands `k=1..3`, `k=4..5`, and
   `k=6..7`, the candidate's median absolute log expected-member power ratio
   error at horizon four must be no more than `1.10` times the frozen parent
   deterministic error on the same starts.
3. **Cross field:** over `k=1..7`, the candidate Ne--phi normalized complex
   cross-spectrum error must be strictly below the parent error.  Its
   truth-cross-amplitude-weighted mean absolute phase error may be at most two
   degrees worse.  Coherence error is reported but is not an additional gate.
4. **Spatial transport covariance:** the median relative Frobenius error of
   the fixed-probe covariance action over particle, electron internal energy,
   ion internal energy, and total internal energy transport must be below
   `0.90`.
5. **Local transport calibration:** at least three of four aggregate local
   corrected spread--skill ratios must lie in `[0.80,1.25]`, and none may
   exceed `1.40`.
6. **Integrated transport calibration:** the median of the four integrated
   corrected spread--skill ratios must be at least `0.60`.
7. **Integrated transport mean:** the median separatrix-transport relative-L2
   error across the four quantities must be no more than `1.05` times the
   parent deterministic error on the same starts.

The pilot passes only if every family passes.  A failed family cannot be
offset by a scalar aggregate score.

## 6. Conditional uncertainty

Point estimates are accompanied by a paired chronological block bootstrap.
Within each V00/V01/V02 list, noncircular blocks of three adjacent selected
starts are sampled with replacement and truncated back to 12 starts.  The
three resampled blocks are concatenated, using 2,000 replicates from NumPy
PCG64 seed `85604405`.  The same indices are used for candidate, candidate
mean, parent, and truth.  Intervals are conditional on this one 85604
simulation and are descriptive; the already frozen point gates remain the
pilot decision rule.

## 7. Decision

Passing authorizes a dated amendment for matched seeds 1701 and 1703.  Failure
stops scaling this old-data persistent stochastic mechanism and returns the
next bottleneck to additional independent data or complete evolved state.
Neither outcome establishes a universal claim about stochastic plasma
emulation.

