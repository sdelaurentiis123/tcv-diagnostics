# Old-85604 bounded-rollout state and physics readout

**Result date:** 2026-08-25

**Development simulation:** 85604 only

**Held-out 85606, guard interval, and newer NERSC data:** not read

**Forecast commit:** `039c4d4458d442693cc53aea262cfda193c08caf`

**Physics-scoring commit:** `13292b639b35f03b77d1e690ad9ed8a975d01f26`

**Forecast job:** `6937051` (`COMPLETED`, exit `0:0`)

**Physics-scoring job:** `6937203` (`COMPLETED`, exit `0:0`)

## Narrow answer

The three independently trained codec-free transition operators learn useful
finite-time and bounded-autoregressive dynamics on the chronological 85604
validation interval. Autonomous feedback is therefore not generically broken.
However, rollout path rankings reverse when the terminal state is evaluated by
transport-bearing statistics instead of pixelwise field error.

At eight saved frames, two repeated four-frame updates have the best median
five-field state skill (`0.428809`), while eight repeated one-frame updates
preserve substantially more toroidal power and give the best separatrix
transport. Direct and repeated coarse transitions are smooth and transport
poor. The repeated one-frame path still loses realization coherence after
approximately 3.6 decorrelation times and does not reconstruct the local
radial-face transport map faithfully.

The evidence supports one targeted training intervention: expose the
one-frame transition to four steps of its own predictions while retaining a
one-step field loss. It does not support opening 85606, assimilation,
diagnostic ranking, or a claim of calibrated stochastic forecasting.

## Evaluated object

The evaluated model is the same 2,174,021-parameter, codec-free transition
operator for all three seeds. It jointly predicts increments for
`Ne, Pe, Pi, phi, Vi` on the `64 × 32 × 88` model grid from one current state.
It uses circular operations along the toroidal axis, does not downsample that
axis, and receives a learned embedding of the requested lead. Stage-2 training
supervised direct leads `1, 2, 4, 8, 16` using the existing channel-normalized
field-derivative loss only.

The seeds `1701`, `1702`, and `1703` are independent training initializations.
They are not members sampled from one probabilistic forecast distribution and
must not be interpreted as a calibrated ensemble.

## Bounded-rollout design

Training remains frames `[0,432)`, the unread guard remains `[432,496)`, and
validation remains `[496,624)`. One saved frame is
`3.131905426352636 microseconds`. The representative decorrelation time is
`2.244` frames, or approximately `7.029 microseconds`.

For every eligible validation start and each seed, the evaluation compared the
same current state and terminal truth using:

- horizon four: persistence, one direct four-frame prediction, four autonomous
  one-frame updates, and two autonomous two-frame updates;
- horizon eight: persistence, one direct eight-frame prediction, eight
  autonomous one-frame updates, four autonomous two-frame updates, and two
  autonomous four-frame updates.

Every autoregressive update fed the complete predicted five-field state into
the next call. No intermediate or future truth entered the input. Four frames
are `12.5276 microseconds` or `1.783` representative decorrelation times;
eight frames are `25.0552 microseconds` or `3.565` decorrelation times.

## State result

Skill is `1 - model MSE / persistence MSE`, averaged equally across the five
fields in training-normalized coordinates. Positive values beat persistence.

| Terminal horizon | Rollout path | Seed-median skill | Seed range |
|---:|---|---:|---:|
| 4 | repeated 1-frame step | `0.510131` | `0.478756–0.532357` |
| 4 | repeated 2-frame step | `0.502095` | `0.492181–0.512792` |
| 4 | direct terminal prediction | `0.441402` | `0.439746–0.443462` |
| 8 | repeated 1-frame step | `0.253719` | `0.158112–0.301507` |
| 8 | repeated 2-frame step | `0.343117` | `0.300904–0.355224` |
| 8 | repeated 4-frame step | `0.428809` | `0.424106–0.429738` |
| 8 | direct terminal prediction | `0.405753` | `0.396957–0.418290` |

Every learned path has positive mean five-field skill for every seed. The
field-level statement is slightly weaker: for seed 1701 at horizon eight, the
repeated one-frame path has `Pe` skill `-0.021242` and `Pi` skill `-0.025168`.
All other field/seed/path skills are positive. The pressure fields are thus the
clearest accumulated-state failure of the small-step path.

## Toroidal spectra and realization placement

The saved toroidal wedge is one fifth of the full torus. With `zperiod=5`,
stored Fourier index `k` maps to physical toroidal mode `n=5k`. The evaluated
bands are `n=5–15`, `n=20–25`, and `n=30–35`.

Across all fields, bands, and seeds at horizon eight, median truth-relative
power is `0.725` for repeated one-frame updates, `0.303` for repeated two-frame
updates, `0.116` for repeated four-frame updates, and `0.151` for the direct
terminal prediction. Means are more sensitive to a seed-1701 `Vi`,
`n=30–35` outlier, so the tracked CSV retains every field/band/seed value.

Power preservation does not imply prediction of the correct future eddies.
Median truth-power-weighted transfer coherence over the same entries is only
`0.0304` for repeated one-frame updates at horizon eight, compared with
`0.0042`, `0.0015`, and `0.0019` for repeated two-frame, repeated four-frame,
and direct paths. At horizon four the repeated one-frame median is `0.2145`,
with useful mid-band entries, but high-band realization coherence is already
near zero. This is consistent with the terminal horizons spanning multiple
decorrelation times.

## Cross-field structure

The physics scorer evaluates the `Ne–phi`, `Pe–phi`, and `Pi–phi`
cross-spectra in physical decoded field space. Small-step paths retain much
better mean phase and coherence behavior than the coarse/direct alternatives.
A common stationary phase alone is not evidence of forecast skill: persistence
has a very small aggregate phase change while having little transfer coherence
with the future realization.

The result therefore separates three properties that ordinary field RMSE
conflates:

1. marginal power at each physical mode band;
2. cross-field phase/coherence needed for nonlinear transport;
3. realization placement of the future structures.

## Transport result

Transport was computed after inference with the validated geometry-aware
radial-face operator. No transport, spectrum, cross-phase, coherence, PDE, or
conservation term entered training. Nonlinear quantities were calculated for
each seed forecast before aggregation.

At horizon four, repeated one-frame forecasts have seed-median separatrix
relative-L2 errors of `0.344` for particle transport, `0.284` for electron
internal-energy transport, `0.298` for ion internal-energy transport, and
`0.280` for total internal-energy transport. Persistence lies near
`0.41–0.42` for these quantities. The corresponding repeated-one-frame
correlations are `0.529–0.685`.

At horizon eight, repeated one-frame forecasts have median separatrix
relative-L2 errors of `0.395`, `0.395`, `0.378`, and `0.380`; every seed and
quantity improves on persistence. Coarse and direct paths lie near
`0.90–0.94`, approximately twice persistence, and have near-zero or negative
median correlations. The repeated one-frame median correlation across all
seed/quantity entries is `0.371`, although the worst entry is slightly
negative.

This positive integrated result does not pass a strict local-transport gate.
The repeated one-frame strict-face relative-L2 error is approximately `1.03`
at horizon four and `1.38` at horizon eight. The model recovers useful
variation in the separatrix-integrated signal while misplacing substantial
local positive and negative contributions.

## Scientific interpretation

The bounded comparison distinguishes two failure mechanisms.

First, direct long-lead supervision and coarse updates reduce terminal field
error but remove most nonaxisymmetric power. Their behavior is consistent with
conditional-mean smoothing of a decorrelating turbulent future. This explains
why their field metric can improve while their nonlinear transport becomes
worse than persistence.

Second, the one-frame transition retains transport-bearing fluctuations but
accumulates state drift when its own outputs are repeatedly fed back. Its
eight-frame weakness is especially visible in `Pe` and `Pi`. The next model
change should therefore target small-step exposure bias, not replace the
small-step dynamics with another direct long-lead objective.

The following claims remain unsupported:

- local transport fidelity;
- high-mode realization prediction beyond decorrelation;
- probabilistic calibration or useful ensemble covariance;
- assimilation, diagnostic ranking, or steering;
- performance on 85606 or on the newer NERSC data.

## Provenance and reproduction

Authoritative state-forecast directory:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout/job_6937051/run
```

Authoritative physics-score directory:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout_physics/job_6937203/run
```

The forecast HDF5 SHA-256 is
`a610f6bae8ca2db40f661d9ab1411746e5924e22345122d73dcafca252b77196`.
The state-metric SHA-256 is
`ddca83ca524c412d14a9db96bfdd2f09085b92f9998f865c331ba0932f4b3fe3`.
The per-target state CSV SHA-256 is
`1d9d1d466e29c6e0d94236112dbdbeca887a3e97187f9e02ec1976243752bc67`.
The physics-metric SHA-256 is
`0215347da9b75ac8b8c2538844a6b8e4e78bffd1f6c6fe35937f7146129d40fc`.

The exact executed entrypoint commands, Slurm allocation, environment, tests,
and artifact ledgers are stored as `command.sh`, `slurm_job.txt`,
`environment.txt`, `test_output.txt`, and `artifact_sha256.txt` in each job
root. Both ledgers verified after completion. The W&B records are
`p0oldbounded-j6937051` and `p0oldboundphys-j6937203` in project
`tcv-diagnostics-paper0`; both finished online.

Regenerate the compact tables, figures, and HTML from the immutable scored
artifacts with:

```bash
python paper0/tools/build_old_85604_bounded_rollout_report.py \
  --repo /mnt/home/sdelaurentiis/tcv-diagnostics \
  --state-metrics /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout/job_6937051/run/state_metrics.json \
  --per-target /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout/job_6937051/run/per_target_state_rmse.csv \
  --physics-metrics /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout_physics/job_6937203/run/physics_metrics.json \
  --example-fields /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_bounded_rollout_physics/job_6937203/run/example_physical_fields_start560.npz \
  --output /mnt/home/sdelaurentiis/tcv-diagnostics/paper0/reports/paper0-old-85604-bounded-rollout-2026-08-25.html
```

## Decision

Freeze exactly one new one-seed pilot before scaling:

> Initialize from the frozen seed-1702 Stage-2 checkpoint, train the lead-one
> transition through four autonomous feedback steps using field losses only,
> retain a direct one-step term, select by chronological state validation, and
> then score the frozen pilot with the same four/eight-frame physics suite.

The pilot advances to three seeds only if it reduces four/eight-frame state
drift without materially degrading the repeated one-frame path's spectral and
separatrix-transport fidelity. Failure closes further tuning of this local
deterministic operator and returns the program to the planned nonlocal,
state-complete, or persistent-stochastic architecture.
