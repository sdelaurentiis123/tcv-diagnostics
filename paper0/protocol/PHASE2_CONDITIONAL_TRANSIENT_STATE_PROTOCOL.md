# Phase 2 conditional-transient and state-candidate protocol

**Protocol status:** frozen before any matched Paper 0 codec or dynamics
training

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Training authorized:** only the 85604 O1/O2 development experiments defined
here and in a subsequently committed matched-model protocol

## 1. Purpose

This protocol resolves two blockers without rewriting either one:

1. the predeclared whole-interval stationarity screen failed; and
2. the historical five-channel state is not the exact state advanced by the
   executed Hermes simulation.

The replacement is deliberately narrow. Paper 0 may train and validate
short-horizon, time-homogeneous transition models on one evolving 85604
trajectory. The validation region is described as **later-background
extrapolation**, not as an independent shot or a sample from a demonstrated
stationary climatology.

The protocol also freezes one exact source-state interface and one pragmatic
history-conditioned interface before either is trained.

## 2. Immutable temporal partition

The original chronological boundaries are retained:

| Region | Global frames | Count | Role |
|---|---:|---:|---|
| training | `[0,432)` | 432 | normalization and optimization |
| guard | `[432,496)` | 64 | unused leakage barrier |
| validation | `[496,624)` | 128 | deterministic later-background selection |

The failed Phase 1 screen remains a failed result. These regions are **not**
reclassified as stationary samples.

For every matched one-step state comparison, the longest context contains two
frames. To prevent a one-frame arm from receiving an extra target, every arm
uses the common target sets:

| Region | Context-supporting target frames | Count |
|---|---:|---:|
| training targets | `[2,432)` | 430 |
| validation targets | `[498,624)` | 126 |

No context or target may intersect the guard. No validation frame contributes
to a normalization statistic, gradient, optimizer state, early-stopping
patience counter other than the declared checkpoint score, augmentation
choice, or architecture change.

Windows may be sampled randomly inside the training target set only after
containment is asserted. Validation targets are evaluated once in
chronological order with no random toroidal roll.

## 3. Meaning of time

The saved cadence is

\[
\Delta t = 3.131905426352636\ \mathrm{\mu s}.
\]

Allowed temporal information is:

- ordered field history;
- fixed relative offsets, such as \((-\Delta t,0)\);
- requested relative forecast lead.

Forbidden temporal information is:

- absolute frame index;
- normalized simulation time used as a trajectory lookup;
- a train/validation-region label;
- any future field, future boundary value, or verifying Fourier shift.

The configured sources and geometry are fixed. They are recorded as
provenance, not repeated as learned time-dependent controls.

## 4. Frozen state candidates

### E6B-H1: exact source-state candidate

The exact arm receives one saved state:

\[
E6B =
\left[
N_e,\ P_e,\ P_i,\ NV_e,\ NV_i,\ \mathrm{Vort}
\right] + B_\phi ,
\]

where \(B_\phi\) is the retained potential midpoint at the inner and outer
radial sides for every one of the 32 poloidal indices. Its canonical shape per
frame is \([2,32]\).

Interior potential is not an independent predicted volume in this arm. It is
reconstructed from decoded \(P_e\), \(P_i\), \(\mathrm{Vort}\), fixed geometry,
and decoded \(B_\phi\) using the hash-locked Hermes/BOUT++ elliptic operator.

The all-frame closure result makes this the exact saved-state candidate. It
does not prove that one output-cadence state is easy for a neural network to
advance.

### C5P-H2: pragmatic primary baseline

The pragmatic arm receives the two most recent frames, oldest first:

\[
C5P_{t-1:t} =
\left[
N_e,\ P_e,\ P_i,\ \phi,\ V_i
\right]_{t-1:t}.
\]

This state:

- retains the direct evolved pressures needed by the validated transport
  operator;
- includes interior potential directly for a conventional volumetric model;
- omits electron parallel momentum, generalized vorticity, and explicit
  potential-boundary state;
- asks whether one finite difference of recent observed state predicts those
  omissions well enough at the saved cadence.

Two frames are the minimum history that exposes a resolved temporal change.
They also span one saved interval, after which the configured one-microsecond
boundary relaxation retains only

\[
\exp(-3.131905426)=0.0436346
\]

of an undriven perturbation. This is a prospective engineering rationale, not
a claim that two frames form a mathematically complete delay embedding.

### C5P-H1: required history ablation

\[
C5P_t =
\left[
N_e,\ P_e,\ P_i,\ \phi,\ V_i
\right]_t
\]

is trained with the same target frames, optimization budget, codec budget, and
deterministic backbone as C5P-H2. It is not an optional ablation. The
C5P-H2-minus-C5P-H1 comparison measures the value of history; without it,
success or failure cannot be attributed to temporal context.

No historical C5T model is promoted into this comparison. C5T remains a
documented legacy baseline because floor-derived \(T_i\) does not exactly
recover raw evolved \(P_i\). Historical f8 and z44 results remain evidence,
not initialization or matched baselines.

## 5. What the comparison can identify

The three arms form a controlled state-sufficiency ladder:

1. **E6B-H1 versus C5P-H1:** effect of using the exact evolved and boundary
   state rather than a partially observed five-volume state.
2. **C5P-H2 versus C5P-H1:** effect of one additional observed history frame.
3. **E6B-H1 versus C5P-H2:** whether short observed history is a viable
   practical substitute for the omitted exact state.

These are interface-level comparisons. Because the channel meanings and the
elliptic reconstruction path differ, they are not claims that a single field
caused an observed score difference. More targeted field-drop ablations may be
frozen later if the matched result warrants them.

## 6. Normalization and representation rules

All statistics are fit on frames `[0,432)` only in float64.

- \(N_e\): \(\log(N_e + 10^{-6})\), followed by one scalar mean and standard
  deviation.
- \(P_e,P_i,NV_e,NV_i,\mathrm{Vort},\phi,V_i\): identity transform followed by
  one scalar mean and standard deviation per channel.
- \(B_\phi\): identity transform followed by one mean and standard deviation
  per radial side, pooled over training frames and the 32 poloidal positions.

Direct negative \(P_i\) values are retained. No positivity clipping, pressure
floor, or temperature substitution is allowed in the learned state.

Every transform must round-trip within a separately frozen numerical
tolerance. Potential evaluation reports both stored-gauge and per-frame
constant-shift-aligned errors. Gradient, spectral \(k>0\), cross-phase, and
transport metrics remain gauge invariant.

The exact arm's 64 boundary scalars bypass the volumetric codec during the O1
compression-only test so that O1 isolates volume compression. They do **not**
bypass the O2 transition: O2 must predict the next boundary state or explicitly
document a separately frozen deterministic boundary update.

## 7. Allowed claims under nonstationarity

Allowed on 85604 validation:

- one-step field error and calibration;
- error by contiguous time block and slow-background bin;
- codec reconstruction, spectra, cross-field phase/coherence, and transport;
- comparison with persistence on the identical targets;
- a statement that validation extrapolates to a later background in the same
  simulated trajectory.

Not allowed:

- calling train and validation independent physical samples;
- calling the aggregate validation distribution stationary;
- selecting a post-decorrelation stationary transport distribution;
- interpreting 430 windows as 430 independent experiments;
- broad cross-shot, device, regime, or experimental generalization;
- any 85606 result before the complete held-out protocol is frozen;
- any steering or control claim.

Long autonomous rollouts may be computed later as stress tests, but stationary
post-decorrelation distribution claims remain closed unless Ben supplies a
physically justified steady interval or additional simulation provenance.

## 8. Required next protocol

Before a training launch, a separate committed matched O1/O2 protocol must
freeze:

- volumetric codec topology and latent-size escalation ladder;
- loss functions, which may not contain physics-derived metrics;
- parameter and optimization budgets;
- seeds;
- checkpoint-selection rule;
- deterministic backbone and history handling;
- persistence and no-compression references;
- field, spectral, cross-field, transport, and boundary metrics;
- acceptance gates and failure actions;
- exact Rusty/Rocky 9 commands and artifact schema.

No LOLA, diffusion, FGN, PDE-Refiner, residual generator, assimilation, or
diagnostic-ranking experiment is authorized until this deterministic
representation/state-sufficiency ladder passes its frozen gates.

## 9. Evidence locks

This decision depends on immutable tracked evidence:

| Evidence | SHA-256 |
|---|---|
| Phase 1 failed stationarity result | `cf6a60c7e4c24cac42efcc1ba877e80cb04c5803e0687397fb814f7652837749` |
| Phase 1 toroidal coherence result | `9ef0868a21ebbee883f154f13fe4068d50d47474017cf775ba3d5c3e51b7fc15` |
| all-frame evolved-state/momentum closure | `565a4e27e87d4f5a3e647daf77486020ac627f43ffb5cd30a8daf74b7199cf20` |
| all-frame saved boundary audit | `79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc` |
| paired exact elliptic reconstruction | `ae0aea28efc8719c7c3c91419a8f122256f9fe7e6d64c94e6aa9e1827dd2297a` |
| all-frame potential/vorticity closure | `cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae` |
| historical codec transport attribution | `140bf3faabb0922edd9108af7d3e00e76c71075caa3a43e5c29760cc043b0a23` |

This protocol changes no prior result, threshold, source data, or held-out
status.
