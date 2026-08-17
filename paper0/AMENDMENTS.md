# Paper 0 protocol amendments

The governing specification is preserved verbatim in `PAPER0_SPEC.txt`. Necessary clarifications are recorded here rather than silently rewriting it.

## A001 - Historical exposure of shot 85606

**Status:** active from repository initialization.

Shot 85606 was inspected repeatedly during exploratory work in the predecessor repository before this clean Paper 0 protocol existed. Therefore it cannot honestly be described as historically or researcher blind.

Paper 0 will nevertheless sequester 85606 from all new training, validation, architecture selection, checkpoint selection, metric development, assimilation tuning, and acceptance-threshold selection. After the complete protocol is frozen and committed, 85606 may be used for one prospectively locked confirmatory evaluation. The paper must describe it as a held-out simulation with prior exploratory exposure, not as a never-seen blind test.

A genuinely blind confirmation requires an additional unseen Hermes simulation supplied after the protocol is frozen.

## A002 - Diagnostic-only decorrelation after the steady-state gate failed

**Status:** active after Phase 1 result `phase1_85604_profile_6890563.json`.

The predeclared operational steady-state screen failed for every reported C5
series. That result is immutable and remains a failed gate. In particular:

- the proposed `[0, 432)` training, `[432, 496)` guard, and `[496, 624)`
  validation regions are not reclassified as samples from one demonstrated
  stationary distribution;
- the Phase 1 learning gate remains closed;
- the tolerances in `PHASE1_DATA_PROTOCOL.md` are not relaxed, and no alternate
  start frame is scanned for a prettier result.

The failed gate combines two distinct questions: slow evolution of spatial
profiles and fast decorrelation of mean-removed turbulent patterns. The
autocorrelation method in Section 6 of the Phase 1 protocol was frozen before
the gate result, but the first profiler skipped it whenever the gate failed.
That prevents the project from measuring the fluctuation timescale needed to
interpret the failure.

This amendment therefore authorizes exactly one additional calculation:

1. apply the already-frozen Section 6 autocorrelation procedure to candidate
   training indices `[0, 432)`;
2. keep the original transforms, training-only normalization, spatial strides
   `(4, 2, 4)`, maximum lag 108, per-frame spatial-mean removal, and per-cell
   temporal-mean removal unchanged;
3. label every resulting time as **diagnostic-only under nonstationarity**;
4. do not use the value to select a steady interval, alter the split, tune a
   model, or open the learning gate.

No high-pass window, detrending hyperparameter, alternate block length, or
field subset is added by this amendment. Any such analysis requires another
prospectively committed rule. The immediate purpose is only to distinguish
short-lived fluctuation patterns from the documented slow background evolution.

## A003 - Separate axisymmetric background from non-axisymmetric fluctuations

**Status:** proposed before calculation, following job `6890591`.

The A002 calculation removes one scalar spatial mean per frame, but it retains
the time-varying axisymmetric profile. Its density correlation therefore has a
long positive tail: the first `1/e` crossing is approximately 19 frames and no
non-positive crossing occurs within the frozen 108-frame lag. BOUT++ uses `z`
as the periodic toroidal coordinate, and the stored domain has `zperiod = 5`.
The toroidal mean is therefore the stored `k = 0`, full-torus `n = 0`
component. Removing it is a physics-defined decomposition with no fitted
timescale or cutoff.

This amendment authorizes a second diagnostic-only autocorrelation:

1. use the same candidate training frames `[0, 432)`, model transforms,
   training-only normalization, `x=4` and `y=2` strides, lag range `0..108`,
   and crossing definitions as A002;
2. retain every toroidal sample while constructing
   `delta_z X(t,x,y,z) = X(t,x,y,z) - mean_z X(t,x,y,z)`;
3. after that subtraction, retain every fourth toroidal cell for the same
   computational sampling density as A002;
4. apply the existing per-cell temporal-mean removal inside the pattern
   autocorrelation;
5. report full-pattern and toroidal-residual curves side by side for every C5
   field;
6. report the fraction of temporally varying model-coordinate energy in the
   axisymmetric and non-axisymmetric components, using a temporal-mean-removed
   orthogonal decomposition over the sampled `x,y` cells and all 88 toroidal
   cells.

For density, the toroidal residual is taken after the declared logarithmic
model transform. It therefore measures decorrelation of the representation the
model is asked to forecast, not a linear-density transport fluctuation. The
later transport metrics must still use physical linear density.

The result cannot establish stationarity, choose a split, authorize training,
or replace geometry-aware correlation measures. It exists to identify how
much of the apparent memory belongs to the slowly evolving `n=0` background
versus non-axisymmetric turbulent structure.

## A004 - Distinguish toroidal translation from loss of mode coherence

**Status:** proposed before calculation, following job `6890601`.

After removal of the toroidal mean, every C5 field crosses `1/e` in less than
one stored frame under fixed-grid Eulerian correlation. This can mean that the
saved cadence does not resolve the realization, but it can also occur when a
coherent structure translates or its Fourier phase rotates between frames.
Because the domain is periodic in `z`, these possibilities can be separated
without choosing a learned model.

This amendment authorizes two oracle diagnostics on the same model-coordinate
toroidal residual and candidate training frames `[0, 432)`:

1. **Global circular-shift alignment.** For every lag `0..32`, calculate the
   normalized cross-correlation for all 88 circular `z` shifts after per-cell
   temporal-mean removal. Report the maximum correlation and its signed shift
   in stored cells and full-torus degrees. One global shift is shared across
   all sampled times and `(x,y)` cells at each lag.
2. **Complex mode coherence.** Fourier transform the full 88-cell residual in
   `z`. For each stored mode `k=1..16` and lag `0..32`, calculate the magnitude
   and phase of the normalized complex cross-correlation across sampled times
   and `(x,y)` cells. Report `n = 5k`, lag-one magnitude and phase, and the first
   `1/e` magnitude crossing. The band `k=4..7` is labeled explicitly as
   full-torus `n=20..35`.

The calculation retains the A003 `x=4`, `y=2` subsampling and all toroidal
cells. A synthetic translating-wave test must show low or sign-changing
fixed-grid correlation together with unit shift-aligned correlation and unit
mode-coherence magnitude.

These are oracle diagnostics: the maximizing shift and future Fourier phase
use the verifying frame. They cannot be supplied to a forecast, used as a
training target or loss, select the split, or open the learning gate. Their
purpose is to decide whether future architectures need to represent coherent
phase transport or genuinely stochastic re-sampling at the saved cadence.

## A005 - Deterministic GPU numerics for the O1 codec oracle

**Status:** active before any O1 checkpoint inference.

`PHASE2_O1_CODEC_PROTOCOL.md` freezes deterministic, noise-free codec inference
but does not name the CUDA precision flags. O1 therefore fixes the following
execution details before either real-data reconstruction is run:

1. inputs, parameters, and outputs use float32;
2. TF32 is disabled for CUDA matrix multiplication and cuDNN;
3. cuDNN benchmarking is disabled and deterministic mode is enabled;
4. PyTorch deterministic algorithms are required rather than merely warned;
5. CPU and CUDA random seeds are zero, although no stochastic layer is active
   under `eval()` and `decode(noisy=False)`;
6. each codec is loaded and evaluated separately, using the same chronological
   input chunks.

The result records the exact PyTorch, CUDA, cuDNN, driver, and GPU identities.
This amendment changes no data selection, metric, band, threshold, checkpoint,
or scientific claim.

## A006 - Recover the exact Hermes revision from the raw dump

**Status:** active after the O1 codec result and before transport implementation.

The Phase 1 manifest originally recorded the BOUT++ revision from
`BOUT.settings` but not the Hermes-3 revision embedded in the representative
raw dump. A read-only source audit recovered:

- Hermes-3 revision `920ba829cc78cdab0dbf6101c69fecc4689bd8dd`;
- slope limiter `MC`;
- BOUT++ revision `7d28d67c3f12c24ec281c0982e870f5369c65a6f` and version `5.2.1`.

Clean detached checkouts of the official Hermes-3 and BOUT++ repositories were
used only to identify the executed equations and hash the critical source
files. Their revisions, repository URLs, licenses, and file hashes are added to
`phase1_85604_sources.json`. This is a provenance correction. It changes no
data, split, model, threshold, or earlier result.

## A007 - Replace the exploratory radial-flux proxy before transport claims

**Status:** frozen before implementation.

The predecessor radial-flux function used a centered `z` difference divided by
`Bxy` and an unweighted `y,z` mean. Exact-source review shows that the executed
Hermes density and pressure equations instead use the conservative
`Div_n_bxGrad_f_B_XPPM` operator. Because `poloidal_flows = true`, its radial
face flow contains both a Jacobian-weighted `x-z` contribution and an `x-y`
contribution involving BOUT++ shifted-field-line `DDY`, `g11`, and `g23`.

The old result remains historical evidence about density--potential
correlation but is not an authoritative physical particle or energy flux.
`paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md` freezes the replacement
definition and validation ladder. It permits an explicitly named partial
`xz`-component implementation for oracle development, but keeps the transport
gate closed until shifted topology, native-grid agreement, geometry masks,
resampling sensitivity, sign, and units are validated. No failed or missing
transport value may be silently recorded as zero.

## A008 - Separate the transport target from the legacy temperature baseline

**Status:** frozen after all-frame pressure closure and before resampling code.

Job `6891583` established that direct electron pressure closes through `Ne*Te`
throughout 85604, while negative evolved ion pressure cannot be reconstructed
from floor-derived `Ti` at 3,412 points, including 1,421 in the accepted
operator interior. Paper 0 therefore names two states rather than silently
changing the old one:

- `C5T = [Ne, Te, Ti, phi, Vi]` remains the legacy baseline;
- `C5P = [Ne, Pe, Pi, phi, Vi]` is the leading transport-target candidate.

Direct negative `Pi` is retained. Temperature diagnostics use the explicit
Hermes floor convention. Neither state is presumed Markov-complete, absolute
frame number remains excluded, and no new training split is accepted here.

The same amendment makes native-81 transport primary even when a model uses an
88-cell convenience grid. If the prospectively gated 81-to-88-to-81 round trip
passes, every future 88-cell ensemble member is downsampled to 81 before the
primary nonlinear transport calculation. Direct 88-cell transport is reported
only as a resampling sensitivity. Exact definitions and thresholds are frozen
in `protocol/PHASE2_STATE_RESAMPLING_PROTOCOL.md` and its machine manifest.

## A009 - Replace the failed stationary split with a conditional-transient development protocol

**Status:** active before any new Paper 0 learning run.

The original eight-block stationarity screen remains failed, and its
tolerances and result are unchanged. Paper 0 nevertheless needs to distinguish
whether field compression and one-step state insufficiency fail before any
stochastic architecture is considered. Strict stationarity is not required to
fit a local time-homogeneous transition when the changing background is
present in the input state, but it is required for the stronger stationary
post-decorrelation distribution interpretation originally proposed.

This amendment therefore retains the chronological `[0,432)` training,
`[432,496)` guard, and `[496,624)` validation boundaries and relabels the
development task as **conditional-transient later-background extrapolation**.
It authorizes matched 85604 O1 codec and O2 one-step experiments only after
their model protocol is separately committed. Absolute time is forbidden.
Metrics must be reported by contiguous block or predeclared slow-background
bin, and stationary climatology, independent-window, held-out-85606,
assimilation, and diagnostic-ranking claims remain closed.

## A010 - Freeze exact and pragmatic state candidates before matched learning

**Status:** active after all-frame potential/vorticity closure and before new
codec training.

The exact saved source-state candidate is
`E6B-H1=[Ne,Pe,Pi,NVe,NVi,Vort]+Bphi`, with one current frame and
the retained `[2,32]` radial-potential midpoint. Interior potential is
reconstructed by the exact elliptic operator.

The pragmatic primary baseline is
`C5P-H2=[Ne,Pe,Pi,phi,Vi]` over the two most recent frames. It is paired
with the mandatory one-frame `C5P-H1` control on identical target indices.
This makes the value of history measurable and does not presume that history
recovers omitted electron momentum, vorticity, or boundary state.

All arms use common one-step targets `[2,432)` for training and
`[498,624)` for validation. Direct negative `Pi` is retained, absolute
frame number is excluded, historical codecs are not used as initialization,
and no stochastic model is authorized. Exact rules and evidence hashes are in
`protocol/PHASE2_CONDITIONAL_TRANSIENT_STATE_PROTOCOL.md` and
`manifests/phase2_conditional_transient_state_85604.json`.

## A011 - Require one verified shared dataset before matched state learning

**Status:** frozen before converter implementation and before any new codec or
dynamics training.

The exact `E6B-H1` arm and pragmatic `C5P-H2/H1` arms require different
views of the state, but they must not use different source, frame, resampling,
precision, or normalization paths. Paper 0 therefore requires one immutable
85604 dataset containing the union
`[Ne,Pe,Pi,NVe,NVi,Vort,phi,Vi]` on the frozen 88-cell periodic grid plus
the exact retained `Bphi` boundary state.

The converter reads the already audited native-81 Well files and the
hash-locked boundary extraction from job `6893033`; it does not traverse raw
rank files again. It preserves all direct pressure values, uses the exact
SciPy Fourier resampling already attributed to the historical C5T path, and
fits normalization only on frames `[0,432)`. Complete frame coverage,
source and array hashes, writer/reopen equality, legacy z88 equality,
81-to-88-to-81 error, boundary casting, and independent normalization
recomputation are prospective hard gates.

Passing this conversion establishes common engineering provenance only. It
does not authorize or imply codec fidelity, predictive state sufficiency,
forecast skill, stationarity, stochastic modeling, assimilation, diagnostic
ranking, or access to 85606. Exact rules are in
`protocol/PHASE2_MODEL_DATASET_PROTOCOL.md` and
`manifests/phase2_model_dataset_85604.json`.

## A012 - Distinguish normalized simulator time from physical cadence

**Status:** corrected before converter implementation and before any converted
array was read.

The Well coordinate is normalized ion-cyclotron time, not microseconds. It
runs from `285000` to `471900` in steps of `300`. The source-locked
`Omega_ci=95788333.03066081 s^-1` converts that step to
`3.131905426352636 microseconds`. The model-dataset protocol and manifest
now require both checks and store the normalized coordinate without
mislabeling it as physical time. No source, split, field, threshold, or
scientific decision changed.

## A013 - Block degenerate normalization before model training

**Status:** frozen before converter execution and before any converted array
was read.

Exact count and moment recomputation alone would accept a channel with zero
training variance even though it cannot be standardized. The dataset gate now
also requires every fitted population standard deviation to be finite and
strictly positive. This is a usability and fail-closed integrity condition; it
does not alter a field, transform, split, or scientific acceptance threshold.
