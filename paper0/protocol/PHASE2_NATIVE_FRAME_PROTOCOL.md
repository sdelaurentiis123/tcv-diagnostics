# Phase 2 native-frame transport-oracle protocol

**Status:** frozen before the first state-value read performed for this oracle

**Development run:** TCV/Hermes `85604` only

**Sequestered run:** `85606`; prohibited from the extractor, launcher, and
comparison

**Purpose:** test the accepted partial radial-flow implementation on the actual
dynamic range and field relationships of raw, native-81 Hermes states

The machine-readable authority is
`paper0/manifests/phase2_native_frame_oracle.json`. This document explains why
those choices are made and what a pass would and would not establish.

## 1. Source-backed quantities

The archived `85604/BOUT.inp` selects `evolve_density` for electrons and
`evolve_pressure` for both electrons and ions. All three components set
`poloidal_flows = true`. At Hermes revision
`920ba829cc78cdab0dbf6101c69fecc4689bd8dd`, the source applies
`Div_n_bxGrad_f_B_XPPM(q, phi, ..., true)` to:

| Meaning in this rung | Direct archived `q` | Model-field reconstruction | Role of `Vi` |
|---|---|---|---|
| Electron particle advection | `Ne` | `Ne` | none |
| Electron pressure advection | `Pe` | `Ne * Te` | none |
| Ion pressure advection | `Pi` | `Ne * Ti`, after testing `Ni == Ne` | none |

`Vi` remains a physically meaningful input for joint-state evolution and
parallel dynamics. It simply does not enter this perpendicular ExB face-flow
operator. `phi` determines the ExB velocity for all three advected fields.

The archive identifies `Ne` and `Ni` as number density, `Pe` and `Pi` as
pressure, and `Te` and `Ti` as temperature. The run uses quasineutral singly
charged ions, so the five-channel reconstruction is accepted only if every
selected frame passes all four closures:

```text
Ni == Ne
Pe == Ne * Te
Pi == Ni * Ti
Pi == Ne * Ti
```

For each relation, the prospectively frozen rule is

```text
max_abs_error <= 1e-12 + 1e-12 * max_abs_reference
```

No non-finite value is allowed. A closure failure is recorded, not repaired by
redefining the target after inspection.

## 2. Internal-energy terminology

For a three-dimensional ideal species, thermal internal-energy density is
`U = 3P/2`. Positive scalar multiplication commutes with the source's limiter
and face-flow construction. Therefore this rung may derive:

```text
electron internal-energy ExB flow = 1.5 * flow(Pe, phi)
ion internal-energy ExB flow      = 1.5 * flow(Pi, phi)
total thermal internal-energy flow = sum of the two
```

This is not yet released as an experimental or total “heat flux.” It excludes
parallel advection, conductive flux, pressure work bookkeeping, sheath flux,
surface integration, and SI conversion. Those distinctions remain explicit in
every result.

## 3. Value-independent frame selection

The raw archive contains 624 frames. Selection uses no field statistic. Take
the nearest-half-up index at fractions `0`, `1/4`, `1/2`, `3/4`, and `1` of the
inclusive index interval `0:623`:

Frozen indices: `[0, 156, 312, 467, 623]`.

| Selection fraction | Frame index | Expected normalized time | Time since selected frame 0 |
|---:|---:|---:|---:|
| `0` | `0` | `285000` | `0 us` |
| `1/4` | `156` | `331800` | `488.5772465110112 us` |
| `1/2` | `312` | `378600` | `977.1544930220224 us` |
| `3/4` | `467` | `425100` | `1462.5998341066809 us` |
| `1` | `623` | `471900` | `1951.1770806176921 us` |

The physical conversion uses the frozen `Omega_ci =
95788333.03066081 s^-1` and 300 normalized-time units per frame. The extractor
must reject any time or cadence mismatch.

These five frames are implementation-oracle cases, not five independent
physical experiments and not a statistical validation sample.

## 4. Canonical extraction

The extractor must find exactly 256 rank files and verify, before assembling
values:

- Hermes revision `920ba829cc78cdab0dbf6101c69fecc4689bd8dd`;
- slope limiter `MC`;
- `NXPE=16`, `NYPE=16`, `MXSUB=4`, `MYSUB=2`;
- `MXG=2`, `MYG=2`, `MZ=81`, and `zperiod=5`;
- complete, duplicate-free `(PE_XIND, PE_YIND)` coverage;
- local dimensions `[t=624, x=8, y=6, z=81]`;
- identical selected times and variable metadata on all ranks.

For `Ne`, `Ni`, `Te`, `Ti`, `Pe`, `Pi`, and `phi`, strip the two x and y guards,
then assemble physical cells by the explicit processor indices into
`[selected_frame=5, x=64, y=32, z=81]` float64 arrays. Do not infer rank order
from filenames. Record per-variable array hashes, the canonical-file hash, and
all verified metadata. The extractor must refuse to overwrite an existing
artifact.

The locked raw controls are:

| Artifact | SHA-256 |
|---|---|
| `85604/BOUT.inp` | `c1f7f63a4210b35680f338289916f6a588dcc7881928f26066a9af2e09fb95ad` |
| `85604/BOUT.settings` | `57148a0f3d829b72192363d4d6e5da9fc1ce8aa2bff63359491bdb0b9a075d57` |
| `85604/tcv_85604_adjusted.nc` | `0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8` |

## 5. Compiled comparison

A GPL-marked driver may adapt only the already locked radial `xz` and shifted-
`xy` calculations from Hermes `src/div_ops.cxx:128-229` and `:273-326`. Run it
on the accepted BOUT++ build and the original geometry with four ranks,
`NXPE=1`, `NYPE=4`, and `MYSUB=8`. Each rank reads the canonical input slice
for its explicit `PE_YIND`, fills physical cells, and lets BOUT++ communicate
guards and shifted-field topology.

For every selected frame and each direct `q` in `Ne`, `Pe`, and `Pi`, compare:

- echoed canonical `q` and `phi` exactly;
- radial `xz` face flow;
- shifted-`xy` face flow;
- their exact pointwise sum;
- finite-volume radial divergence;
- volume-weighted divergence/face-difference conservation.

Use native 81-cell toroidal resolution, `zperiod=5`, model grid crop `2:66`,
safe model-local left-face indices `1:62`, divergence cells `2:62`, and physical
`y=1:31`. Target-dependent cells `y=0,31` remain excluded because the emulator
state does not contain their external physical guards.

Every frame, advected field, quantity, and previously frozen topology region
must pass:

```text
max_abs_error <= 5e-10 + 5e-10 * max_abs_reference
```

Exact component addition uses zero tolerance. The conservation reconstruction
must pass:

```text
max_abs_residual <= 5e-12 + 5e-12 * max_abs_face_difference
```

Every input must be finite and have peak-to-peak range greater than `1e-12`.
Every total-flow case must have maximum absolute magnitude greater than
`1e-12`. These are noncollapse checks, not accuracy tolerances.

## 6. Decision scope

A complete pass accepts the partial combined radial operator on selected real
85604 states and accepts the tested five-channel closure for these frames. It
does not establish:

- native-81 versus model-grid-88 resampling fidelity;
- outward surface integration or region masks;
- normalized-to-SI conversion;
- member-wise ensemble semantics;
- learned-model field or transport fidelity;
- diagnostic assimilation or ranking;
- any result on the sequestered run.

No tolerance, frame, variable, region, or closure may be changed after the
first execution merely to turn a failure into a pass. A genuine implementation
bug may be fixed only through a documented amendment and a consistent rerun.

## 7. Execution record

Rocky 9 job `6891379`, launched from clean commit `7d5522c`, completed the
compiled four-rank operator step successfully and then exited `1:0` because
the frozen overall acceptance rule required both the operator and closure
subgates to pass.

The operator subgate passed all 15 combinations of five selected frames and
direct archived advected variables `Ne`, `Pe`, and `Pi`, in every frozen
quantity and topology region. The largest face-flow discrepancy was
`6.341038805146582e-13`; the largest divergence discrepancy was
`6.941263563930988e-09` against a reference scale of
`79885.99666953899`; and the largest conservation residual was
`3.552713678800501e-15`.

The full-domain closure subgate failed at exactly one stored cell in frame
312: model index `(x, y, z) = (6, 31, 73)` has
`Pi = -5.799512988032478e-05`, whereas the archived floor-derived
`Ti = 1.2051641668905164e-16` makes `Ni*Ti` approximately
`4.910009611815481e-19`. The locked Hermes `EvolvePressure` implementation
derives temperature from `floor(P, 0)` while retaining the evolved pressure,
so a negative pressure undershoot is not reconstructible as `N*T`.

No frame, region, tolerance, or acceptance rule was changed. The failed point
is on target-dependent row `y=31`, outside the independently frozen transport
comparison scope; the post-hoc ion-pressure closure error over that scope is
only `5.329070518200751e-15`, but that observation does not retroactively pass
the full-domain closure gate. The operator is accepted for its declared
partial scope. The five-channel state representation is not yet accepted as
an exact representation of the evolved Hermes state.
