# Phase 1 immutable data protocol

**Protocol status:** pre-execution draft

**Run in scope:** TCV/Hermes `85604` only

**Sequestered run:** `85606`

This document freezes the Phase 1 analysis rules before the full-field temporal
profile is computed. Numerical results belong in a separate result record and
must not be copied back into the decision rules below.

## 1. Source and coordinate semantics

The canonical source trajectory contains 624 ordered frames. The existing Well
files are treated as two storage shards of that one trajectory, not as an
accepted train/validation split:

1. legacy `train`: source frames `[0, 500)`;
2. legacy `valid`: source frames `[500, 624)`.

Their time vectors must concatenate exactly to the raw BOUT `t_array`, with no
duplicates, gaps, or reordering. All axes are recorded explicitly. Raw BOUT
four-dimensional fields use `[time, x, y, z]`; the Well tensors use
`[trajectory, time, x, y, z]`. The converted `z` coordinate is an index-like
Fourier-resampling coordinate, not a physical angle.

The stored toroidal fraction is fixed at `1/5`, because `zperiod = 5` and the
raw normalized interval is `[0, 0.2]`. A stored Fourier index `k` therefore maps
to full-torus toroidal mode number `n = 5k`. Every spectrum must label the
quantity it displays; the mapping must never be implicit.

The raw normalized frame step is 300 and `Omega_ci` is
`95,788,333.03066081 s^-1`, giving a physical cadence of
`3.131905426352636 microseconds`. Models always receive ordered context and
relative lead information. Absolute frame number is not a default model input.
If a later experiment supports more than one cadence, `delta_t` or a vector of
relative time offsets becomes an explicit conditioning input through a
documented protocol amendment.

The axis convention is independently supported by the
[BOUT++ post-processing documentation](https://bout-dev.readthedocs.io/en/latest/user_docs/output_and_post.html).

## 2. Fields and units

The raw BOUT output is authoritative for field metadata. The legacy Well
conversion dropped those attributes. The currently verified primitive/output
conversions are:

| Field | Meaning | Raw unit | Multiply normalized values by |
|---|---|---:|---:|
| `Ne` | electron number density | `m^-3` | `1e19` |
| `Te` | electron temperature | `eV` | `50` |
| `Ti` | ion temperature | `eV` | `50` |
| `phi` | electrostatic potential | `V` | `50` |
| `Vi` | ion parallel velocity | `m s^-1` | `69205.61141651045` |

The inherited five-channel set `Ne, Te, Ti, phi, Vi` is retained as the
apples-to-apples **C5 legacy-observable baseline**, but it is not assumed to be
a complete Markov state. This Hermes configuration evolves vorticity and both
species' parallel momentum, while potential is recovered by an elliptic solve.
The [Hermes-3 equation documentation](https://hermes3.readthedocs.io/en/latest/equations.html)
states that the vorticity component evolves `Omega` and computes `phi` by
matrix inversion, including pressure terms when diamagnetic polarization is
enabled. The run also configures electron momentum evolution. Consequently:

- `phi` must be gauge-fixed before marginal error or calibration metrics;
- potential gradients and derived `E x B` quantities are preferred physics
  metrics;
- omission of `Vort` and `Ve` is a candidate source of one-step irreducibility;
- a C5-versus-augmented-state comparison is a data/state-sufficiency ablation,
  not an architecture victory.

No augmented field set is frozen until all raw field attributes and conversions
are recorded and the representation cost is scoped. C5 remains the shared
legacy baseline; it does not receive a presumption of physical closure.

## 3. Operational steady-state screen

The stored trajectory is described as statistically steady only in the limited,
operational sense below. One finite trajectory cannot prove strict stationarity.

Before assigning learning regions, divide all 624 frames into eight contiguous
blocks of 78 frames. For each C5 field, in model coordinates (`ln(Ne + 1e-6)`
for density and identity for the other fields), calculate two scalar series:

1. the uniform-grid spatial mean at each frame, except that the `phi` mean is
   omitted because it is gauge dependent;
2. the uniform-grid spatial fluctuation RMS after removing that frame's spatial
   mean.

For every scalar series `u_t`, report:

- each block mean;
- total fitted linear drift divided by the temporal standard deviation;
- first-half minus second-half mean divided by the pooled temporal standard
  deviation;
- block-mean range divided by the temporal standard deviation.

The full stored interval passes the operational screen only when, for every
reported series:

- absolute normalized fitted drift is at most `0.5`;
- absolute normalized half-to-half shift is at most `0.5`;
- normalized block-mean range is at most `1.0`.

These are transparent engineering tolerances, not p-values. If any criterion
fails, Phase 1 records the failing series and does **not** scan alternate start
frames for the split that looks best. A changed steady interval requires a
documented amendment based on simulator provenance or a separately justified
rule.

## 4. Chronological split

If the full-interval screen passes, the immutable 85604 split is:

| Region | Global source indices | Frames | Purpose |
|---|---:|---:|---|
| train | `[0, 432)` | 432 | optimization and normalization |
| guard | `[432, 496)` | 64 | unused leakage barrier |
| validation | `[496, 624)` | 128 | model/filter selection |

The maximum permitted input-plus-target training window under this protocol is
32 frames. The 64-frame guard is therefore strictly longer than every allowed
window. Windows may be sampled randomly only after their complete index range
has been shown to remain inside the training region. Validation windows are
deterministic and receive no random toroidal roll.

The guard is not training data, validation data, or an independent statistical
sample. The physical unit of independence remains one simulation run.

## 5. Training-only normalization

Normalization constants are fit over global frames `[0, 432)` only. C5 uses:

1. `Ne_model = ln(Ne_normalized + 1e-6)`;
2. identity transforms for `Te, Ti, phi, Vi`;
3. one scalar mean and standard deviation per channel over all training frames
   and spatial cells;
4. population variance (`ddof = 0`) accumulated in float64;
5. no validation, guard, or 85606 values in the estimator.

Every transform must provide an invertible round trip. Physical-unit
conversions are applied for reporting, not folded into the learned
standardization. Potential metrics that depend on absolute level must first
declare a gauge; the default is per-frame spatial-mean removal.

## 6. Decorrelation procedure

Decorrelation is estimated only after the split and normalization rules above
are fixed, using training frames `[0, 432)`.

For each channel:

1. apply its model transform and training-only standardization;
2. deterministically subsample cells with strides `(x=4, y=2, z=4)` starting at
   index zero;
3. remove the uniform spatial mean separately from every frame;
4. remove the temporal mean separately from every retained cell;
5. flatten the retained fluctuation field at each time;
6. calculate the normalized Eulerian pattern autocorrelation for lags
   `0..108` (one quarter of the training interval):

   ```text
   rho(lag) = sum_t <X_t, X_(t+lag)>
              / sqrt(sum_t ||X_t||^2 * sum_t ||X_(t+lag)||^2)
   ```

For each field report the linearly interpolated first `1/e` crossing, first
non-positive crossing, and the initial-positive-sequence integrated
autocorrelation time. If no crossing occurs within 108 lags, report it as
right-censored instead of extrapolating.

The representative decorrelation time is the median finite `1/e` crossing
across C5 fields. The maximum finite crossing is also reported as a conservative
reference. Forecast horizons must subsequently be labeled in frames,
microseconds, and representative decorrelation times. This uniform-grid
Eulerian estimate is a reproducible forecast-timescale diagnostic; it is not a
claim about a geometry-weighted Lagrangian turbulence time.

## 7. Blind-test and output rules

- Every Phase 1 input path is rejected if it contains `85606` or a path component
  named `test`.
- No 85606 file discovery, metadata read, hash, plot, or debug access is allowed.
- The profiler refuses to overwrite an existing result.
- The result records source paths and SHA-256 digests, exact command, Git commit,
  dirty state, package versions, split indices, transformations, all screen
  statistics, normalization values, and autocorrelation curves.
- A compact result is committed; large arrays and generated data remain outside
  Git.
- If the operational steady-state screen fails, the split is marked proposed
  rather than frozen and no learned baseline launches.

## 8. Phase 1 exit conditions

Phase 1 is complete only when:

1. source chronology and axes pass automated tests;
2. field attributes and unit conversions are recorded from raw output;
3. the steady-state screen has an immutable result;
4. the chronological split is either frozen or explicitly blocked;
5. training-only normalization is saved and round-trip tested;
6. decorrelation estimates and their full curves are recorded;
7. loaders enforce window containment and the 85606 guard;
8. the result and reproduction command are committed.

