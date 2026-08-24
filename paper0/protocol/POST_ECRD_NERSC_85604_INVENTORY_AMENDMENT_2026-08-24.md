# Post-ECRD NERSC 85604 inventory amendment

**Frozen:** 2026-08-24

**Scope:** metadata and prospective preprocessing of the newly supplied
TCV/Hermes 85604 archive only

**Held-out 85606:** no path discovery, directory listing, metadata read,
file open, preprocessing, normalization, or model use is authorized

## 1. Motivation

Yichen supplied a new squashed 85604 archive after
`POST_ECRD_STATE_DATA_SCALING_PROTOCOL.md` was committed. This amendment
freezes how that archive will be classified, fingerprinted, split, and added
to the state-completeness/data-scaling experiment before any multidimensional
field values are inspected or any scientific model is trained.

The purpose is narrow:

1. determine how much unique 85604 development material is actually new;
2. distinguish independent trajectories from chronological continuations;
3. verify that the C5P and exact-saved-state candidate channels are present;
4. construct leakage-safe, matched data budgets without duplicating frames;
5. stage only the fields required by the frozen C5P-versus-E6B experiment.

This is not a new diagnostic phase or architecture search.

## 2. Administrative deviation recorded before further inspection

The source path became known in the user message that triggered this work. A
read-only NetCDF header query was made before this amendment could be committed
in order to establish whether the file existed and whether it was the claimed
85604 source. No multidimensional field values were loaded. The query exposed
only dimensions, variable names and attributes, selected scalar identifiers,
and the one-dimensional time and iteration coordinates.

That header query established the following facts, which are frozen rather
than silently retrofitted:

- supplied path:
  `/global/cfs/projectdirs/m4466/yfu/TCV-divertor-legs/85604/fit_profile+5`;
- canonical NERSC path:
  `/global/cfs/cdirs/m4466/yfu/TCV-divertor-legs/85604/fit_profile+5`;
- files visible at depth one: `BOUT.squash.nc`, `BOUT.settings`, and
  `tcv_85604_refine_xpoint.nc`;
- `BOUT.squash.nc` size: `50,971,029,215` bytes;
- dimensions: `t=765`, `x=68`, `y=36`, `z=81`;
- radial and poloidal guard counts: `MXG=2`, `MYG=2`;
- toroidal periodicity: `zperiod=5`;
- normalized times: `1121300` through `1312300`, uniformly spaced by `250`;
- `Omega_ci=95788333.03066081 s^-1`, giving a cadence of
  `2.6099211886271965 microseconds` and 764-interval duration of
  `1.9939797881111783 milliseconds`;
- Hermes revision: `29c1d80c6e16066444c018128335efc239de566c`;
- run UUID: `83689e6c-ba82-4fa0-ad9f-0bba054c57dd`;
- recorded restart parent:
  `f592d77a-ddfa-4f7a-959b-f39f2203b0cf`;
- the iteration coordinate resets once while normalized time remains strictly
  continuous, so this is provisionally one chronological continuation with an
  internal restart, not two independent trajectories;
- all six evolved volumes `Ne,Pe,Pi,NVe,NVi,Vort` and all reduced-state fields
  `Ne,Te,Ti,phi,Vi` are present;
- there is no separately named `Bphi` variable. The saved radial `phi` guards
  are the only authorized candidate source for the retained potential-boundary
  state, subject to the already validated extraction convention.

No scientific conclusion follows from this header inventory.

## 3. Prospective minimal inventory

Before preprocessing or training, one read-only inventory job will record:

- full SHA-256 digests, sizes, and modification times of the squash, settings,
  and grid files;
- the exact settings governing grid, output cadence, evolved equations,
  sources, boundaries, normalization, and restart behavior;
- grid/equilibrium identifiers and comparison with the existing 85604 source;
- exact frame index of every iteration reset;
- continuity across each reset using normalized time and, for each of the six
  evolved fields, the relative L2 jump at the reset compared with the 16
  adjacent transitions on either side;
- finiteness, shapes, dtypes, and min/max/RMS summaries of the six evolved
  fields, the five reduced-state fields, and the saved radial `phi` guards;
- whether the periodic z endpoint is duplicated and the exact physical-cell
  crop implied by `MXG`, `MYG`, and the existing model-data convention.

The reset-continuity calculation is a data-integrity check, not a stationarity
or physics analysis. Flux, spectra, cross-phase, coherence, and transport are
not part of this inventory.

## 4. Trajectory classification rule

The archive is one chronological continuation if:

1. normalized time is strictly increasing at constant cadence;
2. the iteration reset has no duplicated or missing saved time;
3. every evolved-field reset jump lies within the empirical range of nearby
   ordinary transitions, or any exception is physically and numerically
   documented.

If this rule fails, the two iteration segments will be treated as separate
source segments and no training pair may cross the reset. They still will not
be called independent physical trajectories without provenance demonstrating
independent initial conditions or branching.

The old and new 85604 sources are never joined by a temporal training pair.
Their normalized-time gap is not interpolated.

## 5. Frozen development splits and budgets

The existing source retains its immutable split:

| source | training | guard | validation |
|---|---:|---:|---:|
| existing 624-frame 85604 | `[0,432)` | `[432,496)` | `[496,624)` |

If the new archive passes the compatibility and continuity checks, its split
is prospectively frozen as:

| source | training | guard | validation |
|---|---:|---:|---:|
| new 765-frame 85604 | `[0,512)` | `[512,640)` | `[640,765)` |

The 128-frame guard is longer than the maximum authorized history-plus-lead
window. No pair may cross the internal iteration reset or either source
boundary. Every budget is evaluated on both frozen validation regions.

The attainable unique-frame training budgets are:

| label | included training frames | total |
|---|---|---:|
| `1x` | all 432 existing-source training frames | 432 |
| `2x` | `1x` plus the first 432 eligible new-source training frames | 864 |
| `all` | all eligible training frames from both sources | 944 |

The nominal `4x` budget is omitted because it is unattainable. Internal
restart exclusions can reduce the number of forecast pairs but do not change
the unique-frame labels. Frames are not repeated, oversampled, or counted as
independent trajectories.

If grid, state-definition, or solver-revision incompatibility prevents a
matched combined training set, these scaling labels are void. The new archive
will instead be retained as a separately reported 85604 development segment,
and a second dated amendment must freeze the revised comparison before
training.

## 6. State and preprocessing rules

The matched state views remain:

- `C5P = [Ne, Pe, Pi, phi, Vi]`;
- `E6B = [Ne, Pe, Pi, NVe, NVi, Vort] + Bphi`.

For the new archive, physical volume cells are cropped using the saved guard
metadata. The periodic z convention must be verified before removing or
resampling an endpoint. No spatial axis may be silently transposed.

`Bphi` may be extracted only from the saved radial `phi` guards using the
already validated Hermes boundary-state convention. If the required guard
state is absent or incompatible, the exact-state candidate becomes `E6`, the
deviation is documented prospectively, and E6 is not called complete.

Normalization is fitted separately for each training budget using only the
training frames included in that budget. Validation frames, guards, and the
held-out simulation never contribute normalization statistics.

## 7. Release condition

Scientific Stage-1 training is released only after a machine-readable
inventory manifest and preprocessing smoke result are committed and all of
the following are true:

1. source hashes and exact commands are recorded;
2. field and boundary shapes are unambiguous and finite;
3. source compatibility or the handling of incompatibility is documented;
4. no future or guard frame enters training or normalization;
5. no transition crosses a source gap or internal restart;
6. the 85606 access flag remains false.

The first released scientific run remains the matched one-step C5P-versus-E6B
codec-free operator comparison. No GAOT port, stochastic model, assimilation,
diagnostic ranking, or steering is released by this amendment.
