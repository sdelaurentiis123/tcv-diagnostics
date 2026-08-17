# Phase 2 state and toroidal-resampling protocol

**Status:** frozen before implementing or evaluating any round-trip or
transport-sensitivity metric; the disclosed structural source probes below
precede the freeze

**Development data:** TCV/Hermes run 85604 only

**Sequestered data:** run 85606 is prohibited

**Purpose:** choose a source-faithful five-channel transport target and decide
how an 88-cell model grid may be used without redefining native-81 transport

The machine-readable authority is
`paper0/manifests/phase2_85604_resampling_sensitivity.json`.

## 1. State policy

Paper 0 distinguishes two five-channel states:

| Name | Ordered channels | Role |
|---|---|---|
| `C5T` | `Ne, Te, Ti, phi, Vi` | frozen legacy-observable baseline |
| `C5P` | `Ne, Pe, Pi, phi, Vi` | leading source-faithful transport candidate |

`C5P` replaces temperatures with the pressure variables directly advanced by
the executed Hermes configuration. It is not described as a complete Markov
state: `phi` and `Vi` are derived outputs, while evolved vorticity and electron
momentum remain omitted. A later one-step state-sufficiency ablation must test
that omission.

Negative direct `Pi` values are retained. No clipping, relabeling, deletion,
or post-hoc masking is permitted in the learned target. When a synthetic
temperature diagnostic requires ion temperature, derive it with the explicit
source-matched convention

```text
Ni = Ne
Ti = max(Pi, 0) / Ni
```

and label it floor-derived. Electron temperature may be derived as `Pe / Ne`
because the all-frame audit found exact closure to the frozen tolerance.

Potential is gauge-fixed by subtracting each frame's spatial mean before
marginal field loss, error, or calibration calculations. Gradient-based
transport is invariant to this constant. Density may use the already frozen
`ln(Ne + 1e-6)` training transform; `Pe`, `Pi`, gauge-fixed `phi`, and `Vi`
remain on identity transforms before training-only standardization. No
normalization is fit in this oracle.

Absolute frame number is not a model input. The data record the fixed cadence
and each forecast's relative physical lead. A `delta_t` condition becomes a
model input only if multiple cadences are trained. The existing HDF5 labels
`train` and `valid` are source-file partitions at global frame 500, not the
unaccepted Paper 0 learning split.

This protocol freezes `C5P` as a candidate for data and metric development. It
does not launch training, alter a historical checkpoint, or waive the Phase 1
stationarity decision.

## 2. Authoritative 85604 sources

Concatenate the native 12-field Well files as global frames `[0,500)` and
`[500,624)`. They have shape `[trajectory=1,time,64,32,81]` and float32
storage. They contain direct `Ne`, `Pe`, `Pi`, `phi`, and `Vi`.

The five value-independent raw-oracle frames `[0,156,312,467,623]` establish a
quantization ladder. For `Ne`, `Pe`, `Pi`, and `phi`, the native Well arrays
must equal the raw float64 oracle arrays after an explicit float32 cast. This
checks provenance; it does not claim that float32 equals raw float64.

The predecessor converter is evidence only. At exact commit `24fdb7d`,
`src/data/build_c5_fast.py` applied `scipy.signal.resample` along the periodic
axis and cast the result to the input float32 dtype. No predecessor code is
modified or imported. The new implementation independently wraps and tests
the same public transform semantics.

On the same five fixed frames, resampling each native `C5T` field must reproduce
the corresponding legacy z88 HDF5 array bit-for-bit. A mismatch is structural
failure and blocks the scientific comparison.

During protocol design, read-only probes already established both selected-frame
structural facts: raw fields equal their native Well representations after a
float32 cast, and the public SciPy call reproduces all five legacy z88 fields
bit-for-bit. They are rerun as provenance gates, not presented later as unseen
scientific findings. No 81-to-88-to-81 field metric, float32 transport metric,
or direct-88 transport metric was calculated before this protocol was frozen.

## 3. Toroidal convention

The simulated wedge has `zperiod=5` and length

```text
Lz = 2*pi/5.
```

Stored wedge Fourier index `k` maps to full-torus mode number `n=5k`. Native
81-cell real fields contain nonnegative FFT bins `k=0..40`. Upsampling to 88
retains those bins and introduces output bins `k=41..44` with zero truth
coefficient. Those four bins are numerical padding bandwidth, not newly
resolved simulator physics.

The frozen transform is exactly:

```python
scipy.signal.resample(x, 88, axis=-1, window=None, domain="time")
```

followed by a cast to float32. The inverse audit uses the same transform from
88 to 81, again cast to float32. No taper, smoothing, antialiasing window,
coordinate interpolation, or phase shift is allowed.

## 4. Three operator paths

Let `U` be the frozen periodic 81-to-88 transform, `D` its 88-to-81 companion,
and `Q_N` the validated conservative partial radial operator evaluated with
`N` toroidal cells and `dz = 2*pi/(5*N)`.

For every 85604 frame, evaluate:

```text
native:       Q_81(x_81)
round trip:   Q_81(D(U(x_81)))
direct 88:    Q_88(U(x_81))
```

For a same-grid direct-88 comparison, also form `U(Q_81(x_81))`. The
round-trip path tests whether the model grid can preserve the native metric.
The direct-88 path measures noncommutation of Fourier resampling with the
nonlinear limiter and finite-difference operator.

Use direct `Ne`, `Pe`, `Pi`, and `phi`, the hash-locked 85604 geometry,
`zperiod=5`, source positivity reconstruction, safe left-face indices
`1..61`, safe divergence cells `2..61`, and the already fixed target-independent
rows `y=1..30`. Do not infer a physical surface integral or SI flux before the
remaining geometry and units rungs pass.

Primary quantities are:

```text
particle radial face flow        = Q(Ne, phi)
electron internal-energy flow    = 1.5 * Q(Pe, phi)
ion internal-energy flow         = 1.5 * Q(Pi, phi)
total internal-energy flow       = electron + ion
```

Report their conservative radial divergences as a separate quantity. Report
`xz` and shifted-`xy` components as diagnostics, not substitutes for the
summed flow.

## 5. Metrics

For paired same-grid arrays `a` and `b`, report per frame and in a streamed
all-frame aggregate:

```text
relative L2       = sqrt(sum((a-b)^2) / sum(a^2))
normalized bias   = mean(b-a) / sqrt(mean(a^2))
RMS ratio         = sqrt(mean(b^2) / mean(a^2))
Pearson correlation after removing each array mean
weighted sign disagreement = sum(|a| where sign(a)!=sign(b)) / sum(|a|)
```

Also report relative L2 after taking the toroidal mean at each valid `(x,y)`,
and per-frame ratios of the 95th and 99th percentiles of absolute magnitude.
Every percentile uses `numpy.quantile(..., method="linear")` on finite valid
values. Per-frame acceptance percentiles use the same method across the 624
frame-level metric values.
The toroidal-mean reduction is a numerical profile diagnostic.
It is not a flux-surface integral.

All metrics remain separated by physical quantity. They may not be collapsed
into one undocumented score. Counts over temporal blocks are conditional
summaries within one simulation, never independent shots.

## 6. Prospectively frozen gates and decisions

Structural gates require all 624 frames, both source files, all five `C5P`
fields, exact axis order, finite inputs and valid outputs, exact hashes, no
85606 access, complete and unique frame coverage, and a clean Paper 0 commit.

Numerical gates are:

1. every selected raw-oracle field equals the native Well field after float32
   cast;
2. every selected legacy `C5T` z88 field is reproduced bit-for-bit;
3. for each `C5P` field, the maximum per-frame 81-to-88-to-81 relative L2 over
   all 624 frames is at most `2e-6`;
4. for each primary flow and divergence, native versus round-trip aggregate
   relative L2 is at most `1e-4` and the 99th percentile of per-frame relative
   L2 is at most `1e-3`;
5. on the five raw-oracle frames, native float64 versus native float32-cast
   operator aggregate relative L2 is at most `1e-5`.

Failure of a round-trip gate blocks 88-cell transport scoring and requires a
native-grid data/model path or a separately frozen higher-precision transform.

If all round-trip gates pass, the primary Paper 0 transport evaluator for any
88-cell truth or forecast downsamples each field member-wise to 81 and applies
`Q_81`. Direct `Q_88` remains a sensitivity result. Its aggregate relative L2
is labeled, without changing a gate after execution, as:

| Relative L2 | Label |
|---:|---|
| `< 0.01` | negligible |
| `0.01 .. <0.05` | small |
| `0.05 .. <0.10` | material |
| `>= 0.10` | severe |

The label is reported independently for every primary quantity and divergence.
A material direct-88 difference does not force an architecture change when the
native round trip passes; it forces native-81 evaluation.

For future ensembles, resampling and nonlinear transport are computed for
each member separately.
No transport quantity is computed only from ensemble-mean fields.
Nothing in this protocol becomes a training loss.

## 7. Execution and artifact rules

Parallelization may use the native HDF5 40-frame chunks, but every partial must
record an explicit global frame interval and may contain no scientific
conclusion. The reducer must prove coverage of exactly `0..623` once before
forming aggregate metrics or decision labels.

The immutable result records the Paper 0 commit and dirty-state gate; exact
commands and environment; source, geometry, converter-evidence, and result
hashes; SciPy, NumPy, and HDF5 versions; framewise metrics; aggregate sufficient
statistics; and compute accounting. Large arrays and run directories remain
outside Git. Only a compact exact result and readable interpretation are
tracked.
