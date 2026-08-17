# Phase 2 O1 codec-transport protocol

**Protocol status:** frozen before the first codec-transport evaluation

**Simulation in scope:** TCV/Hermes `85604` only

**Sequestered simulation:** `85606` (access remains forbidden)

**Purpose:** complete the O1 representation oracle by applying the released
geometry-aware transport evaluator to the historical f8 and z44 codec round
trips.

This is deterministic evaluation, not training. Transport is an evaluation
metric only. The failed Phase 1 stationarity gate remains failed, and the
result cannot authorize a dynamics model or access to 85606.

## 1. Known evidence before this freeze

The original O1 job `6890650` already evaluated all 624 historically exposed
85604 frames. Both codecs passed the field reconstruction gate. The f8 codec
passed the cross-field gate but failed the frozen spectral gate; z44 failed
both the spectral and cross-field gates. Thus both preliminary statuses are
already `fail` before this transport extension.

No geometry-aware codec-transport value, curve, aggregate, or block statistic
was inspected before the present thresholds were frozen.

The checkpoints are unchanged:

| Codec | State | Latent | Checkpoint SHA-256 |
|---|---|---|---|
| f8 | `C5T=[Ne,Te,Ti,phi,Vi]` | `[64,8,4,11]` | `9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e` |
| z44 | `C5T=[Ne,Te,Ti,phi,Vi]` | `[64,8,4,44]` | `095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d` |

They are not C5P codecs. Nothing in this oracle renames their channels or
changes their training provenance. Their unmatched training schedules still
forbid a causal claim about toroidal latent resolution.

## 2. Locked data and alignment

Read all 624 frames exactly once from both storage representations:

```text
native 81:
  /mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/train/TCV_85604_train.hdf5
  /mnt/home/sdelaurentiis/ceph/tcv_well/TCV_85604/data/valid/TCV_85604_valid.hdf5

legacy C5T 88:
  /mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5
  /mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5
```

The machine-readable manifest locks all four hashes. The files named `train`
and `valid` are storage shards, not new Paper 0 statistical splits.

The native reader requires `Ne,Te,Ti,Pe,Pi,phi,Vi`. The legacy reader requires
`Ne,Te,Ti,phi,Vi`. Both virtual trajectories must have the same 624 times,
`x,y` coordinates, and chronological ordering. The native shape is
`[64,32,81]`; the codec shape is `[64,32,88]`.

Every legacy input field and every decoded field is independently transformed
from 88 to 81 with the already released unwindowed periodic resampler. No
window, smoothing, clipping, or phase shift is permitted. The per-frame
relative L2 of each downsampled legacy input against native C5T must remain at
most `2e-6`; this repeats the alignment gate rather than assuming it.

## 3. Four state paths

For each frame define four native-81 paths.

### P0: authoritative direct-pressure truth

\[
P0 = [N_e, P_e, P_i, \phi]_{81}^{\mathrm{direct}}.
\]

Direct evolved pressure is preserved, including negative `Pi`. No floor or
post-hoc mask is applied.

### P1: native C5T-derived truth

\[
P1 = [N_e, N_eT_e, N_eT_i, \phi]_{81}^{\mathrm{native}}.
\]

`Pe` and `Pi` are multiplied in float64. Comparing P0 with P1 isolates the
state-parameterization gap caused by representing temperature rather than
direct evolved pressure.

### P2: downsampled codec input

\[
P2 = D_{88\rightarrow81}
     [N_e,T_e,T_i,\phi]_{88}^{\mathrm{codec\ input}},
\]

followed by float64 `Ne*Te` and `Ne*Ti`. Comparing P1 with P2 isolates the
already-audited storage/resampling round trip in the exact O1 path.

### R: downsampled codec reconstruction

The historical preprocessing and inverse preprocessing are unchanged:

```text
fields = [Ne, Te, Ti, phi, Vi]
Ne transform = ln(Ne + 1e-6)
legacy mean/std from each locked config
decode noise = false
```

After decoding and inverse transformation, downsample every reconstructed
field independently to native 81 and form

\[
R = [\widehat N_e,
     \widehat N_e\widehat T_e,
     \widehat N_e\widehat T_i,
     \widehat\phi]_{81}.
\]

No field is clipped. The source-matched positive face reconstruction remains
enabled inside the Hermes operator, exactly as in the released evaluator.
Counts and minima of non-positive reconstructed `Ne`, `Pe`, and `Pi` are
reported.

## 4. Transport quantities

Apply the released combined radial ExB face operator at native 81 with
`zperiod=5` to every state path. Report:

\[
F_N = Q(N_e,\phi),
\]

\[
F_{U_e}=\frac{3}{2}Q(P_e,\phi),
\qquad
F_{U_i}=\frac{3}{2}Q(P_i,\phi),
\]

\[
F_{U_{\mathrm{total}}}=F_{U_e}+F_{U_i}.
\]

The `3/2` factor is applied once before SI conversion. Particle flow uses
`3.612423832157018e17 s^-1` per normalized unit; pressure/internal-energy
flow uses `2.893870527993356 W` per normalized unit.

## 5. Two spatial reductions

### 5.1 Strict-wall face contributions

Safe radial faces have local left cells `1..61`. Use `y=1..30` and retain a
face row only when both adjacent cells satisfy `penalty_mask == 0` and the
operator marks the row valid. Compare physical coordinate-face contributions

\[
C_{f,j,k}=F_{f,j,k}\,\Delta y_{f,j}\,\Delta z.
\]

No unweighted image-space proxy or outside-wall cell enters the primary face
score.

### 5.2 Confined-separatrix wedge flow

Integrate only the exact closed-field separatrix face:

\[
i_{\mathrm{left}}=15,\qquad y=8..23,
\]

over the simulated one-fifth wedge. Positive values are outward from the
confined region. Store all 624 per-frame values in normalized and SI units.
A five-wedge full-torus equivalent is not needed for O1 and is not computed.

## 6. Comparisons and attribution

For every quantity and both spatial reductions, calculate:

| Comparison | Meaning |
|---|---|
| `P0_vs_P1_state_gap` | direct-pressure versus temperature-derived state |
| `P1_vs_P2_input_roundtrip` | native versus codec-storage/resampling path |
| `P2_vs_R_codec_only` | compression/decompression error in the codec's own C5T state |
| `P0_vs_R_authoritative` | end-to-end historical codec versus source-faithful truth |

Never collapse these into one unexplained error.

For each comparison report overall and eight fixed 78-frame blocks:

- relative L2;
- normalized signed bias;
- RMS ratio;
- Pearson correlation and whether it is defined;
- truth-magnitude-weighted sign disagreement;
- point count.

For separatrix time series also store the raw paired values and absolute-value
`p95` and `p99` ratios using NumPy's linear quantile convention. Truth paths
must be bitwise identical between the two codec passes.

## 7. Frozen acceptance gates

These are engineering stop/go thresholds, not universal plasma tolerances.
They are applied separately to all four transport quantities; total energy
cannot hide failure of an electron or ion component.

### 7.1 Input alignment

- every legacy-input/native field per-frame relative L2: `<= 2e-6`;
- aggregate strict-face relative L2 for `P1_vs_P2`: `<= 1e-4`;
- separatrix-series relative L2 for `P1_vs_P2`: `<= 1e-4`.

### 7.2 C5T state adequacy

For `P0_vs_P1`:

- particle and electron internal-energy strict-face and surface relative L2:
  `<= 1e-10`;
- ion and total internal-energy strict-face and surface relative L2: `<= 0.05`.

Failure means C5T is not an adequate source-faithful transport state even
before codec compression.

### 7.3 Codec and authoritative transport

Apply the following both to `P2_vs_R_codec_only` and
`P0_vs_R_authoritative`.

For aggregate strict-wall face contributions:

- relative L2 `<= 0.25`;
- RMS ratio in `[0.75,1.25]`;
- Pearson correlation `>= 0.85`;
- weighted sign disagreement `<= 0.15`.

For the aggregate separatrix time series:

- relative L2 `<= 0.20`;
- absolute normalized bias `<= 0.10`;
- RMS ratio in `[0.80,1.20]`;
- Pearson correlation `>= 0.90`;
- weighted sign disagreement `<= 0.10`.

At least seven of eight temporal blocks must also satisfy, for the separatrix
series:

- relative L2 `<= 0.30`;
- absolute normalized bias `<= 0.15`;
- Pearson correlation `>= 0.80`;
- weighted sign disagreement `<= 0.15`.

Undefined required metrics fail. Thresholds are not changed after the output
is observed.

## 8. Decision rules

For each codec report three distinct statuses:

1. `codec_only_transport`: whether R preserves transport relative to P2.
2. `authoritative_transport`: whether R preserves transport relative to P0.
3. `full_codec_acceptance`: pass only if the original field/spectral/cross
   preliminary gate, the common C5T state-adequacy gate, and authoritative
   transport all pass.

Because the original preliminary status of both codecs is already `fail`,
neither can become fully accepted in this extension. Transport may still show
which information survives compression and where it is lost.

Decision consequences:

- if P0 versus P1 fails, train future representations on direct C5P rather
  than treating C5T as source-faithful;
- if P2 versus R fails, representation repair precedes dynamics;
- if codec-only passes but P0 versus R fails, the state parameterization—not
  only compression—is the blocker;
- if f8 and z44 differ, treat the observation as checkpoint-specific until a
  matched from-scratch comparison exists;
- no outcome reopens 85606 or the failed stationarity/training gate.

## 9. Execution requirements

The job must run on Rocky 9 from a clean exact commit, verify all data,
geometry, checkpoint, config, source, prior-result, and evaluator hashes,
refuse overwrite, use deterministic decode, and record exact command,
environment, GPU, elapsed time, and peak memory. It stores compact raw
transport series and sufficient-statistic tables, not reconstructed fields.

Any failed attempt receives a unique directory and ledger entry. Code fixes
are committed before resubmission; scientific thresholds remain unchanged.
