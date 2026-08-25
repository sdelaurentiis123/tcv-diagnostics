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

## A014 - Select O2 eligibility within a representation, not across all representations

**Status:** frozen after the complete R2 O1 result and before O2
implementation, smoke testing, or training.

The original matched protocol correctly recorded the complete six-run R2
matrix as failed because its escalation rule required all three C5P and all
three E6B checkpoints to pass. That result remains unchanged. Within the
matrix, however, `C5P-dcae_l10` passed every complete O1 field, spectral,
cross-field, positivity, and authoritative transport gate at all three frozen
seeds, while `E6B-dcae_l10` failed at all three seeds.

Paper 0 now separates robustness within a candidate from success of every
candidate. A representation may enter O2 only when all three of its seeds pass
the already frozen complete O1 gate; seed averaging cannot rescue failure.
Failure of a different representation is retained as a negative ablation but
does not veto the passing representation. This outcome-informed amendment
therefore authorizes only the already specified `C5P-H1` and `C5P-H2`
teacher-forced one-step comparison on 85604. E6B O2, autonomous rollout,
stochastic models, assimilation, diagnostic ranking, and 85606 remain closed.

The rationale, immutable evidence hashes, exact C5P checkpoint identities,
unchanged O2 settings, W&B requirement, and stop/go rules are frozen in
`protocol/PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md` and
`manifests/phase2_c5p_o2_continuation_85604.json`.

## A015 - Move the unstarted full O2 job from H100 to H200

**Status:** frozen after the bounded H200 smoke and before any full O2
optimizer step.

Full O2 job `6894979` was submitted for four H100 GPUs but remained pending
with runtime `00:00:00`. Queue inspection showed that nearly every available
H100 node was fragmented by a one-GPU job, while multiple Rocky 9 H200 nodes
were reserved and idle. The pending H100 job was placed on user hold before
preparing its replacement, preventing duplicate execution.

The full six-run matrix therefore uses four H200 GPUs on one nonpreemptible
`gpuxl` node. The same implementation already passed the bounded Rocky 9 H200
smoke in job `6894971`. This is an execution-only revision: data, split,
fields, codecs, model weights at initialization, seeds, loss, optimizer,
schedule, training budget, W&B policy, checkpoint selection, evaluation
metrics, acceptance thresholds, and claims boundaries are unchanged. All six
runs remain on one accelerator type, and the exact device identities and
compute usage are recorded for fairness.

## A016 - Treat truth-empty event blocks as not applicable

**Status:** frozen after original B2 matrix job `6897564` and before amended
evaluator implementation or execution.

The original B2 gate remains immutable and failed. Post-result inspection
showed that validation block 3 contains zero truth events above the frozen
training-only threshold for every transport quantity. The scorer correctly
returned undefined event-conditioned metrics, but the gate required finite
event errors in every block. Because forecasts cannot change a truth-only
event count, that requirement is an undefined-estimand defect rather than a
model-quality threshold.

The versioned amendment treats event-conditioned metrics as explicitly not
applicable only when the stored truth event count is exactly zero and the
undefined/null record is internally consistent. It still requires every
eligible block to pass the unchanged event thresholds, at least five eligible
blocks per quantity, the existing five-of-six complete-block rule, and every
non-event check. The original evaluator and matrix are retained. All three B2
seeds and every future model must be reduced consistently under the new rule.
No forecast, score, threshold, block, or other acceptance rule changes, and
O3, assimilation, ranking, and 85606 access remain closed. Exact rules are in
`protocol/PHASE3_B2_EVENT_ELIGIBILITY_AMENDMENT.md`.

## A017 - Normalize the B4 transport L2 display key at the reducer boundary

**Status:** frozen after job `6901015` completed forecast generation and
truth-separated scoring, and before any retry of its acceptance reduction.

Job `6901015` completed the full 126-target M32 final forecast, M4 all-stage
forecast, both frozen scientific score records, and its online W&B run. The
subsequent pure gate failed before producing an H-det or H-prob decision. The
frozen B4 manifest spells the two displayed transport thresholds
`relative_L2_max`; the unchanged B2 Python reducer consumes
`relative_l2_max`. The B4 adapter copied the display key verbatim and raised
`KeyError: 'relative_l2_max'` on the first transport check.

The repair explicitly maps only `relative_L2_max` to `relative_l2_max` for
the strict-face and separatrix dictionaries, validates the complete source
key sets, and preserves every numerical value. A regression now asserts the
exact reducer-facing dictionaries and rejects schema drift. One CPU-only
Rocky 9 retry may reduce the immutable job-`6901015` result and score hashes.
It may not regenerate or mutate forecasts, rescore truth, change a threshold,
train a model, run O3, open 85606, assimilate diagnostics, or rank sensors.
The failed attempt remains tracked in
`results/phase3_b4_pde_refiner_gate_adapter_failure_6901015.json`.

## A018 - Place the B4 separatrix calibration count in the reducer schema

**Status:** frozen after CPU-only gate attempt `6901282` and before a second
CPU-only retry.

Job `6901282` verified every immutable job-`6901015` evaluation artifact,
every training and comparator artifact, the exact source hashes, and the
complete Rocky 9 test suite (`986 passed, 1 skipped, 29 subtests passed`). It
then reached the unchanged B2 transport reducer and raised
`KeyError: 'probabilistically_calibrated_required'`. The B4 adapter had put
the unchanged frozen value `3` in `separatrix_calibration`; the reused reducer
reads that count from `separatrix` and reads only spread and coverage
tolerances from `separatrix_calibration`.

The repair moves no numerical value and changes no scientific rule. It maps
`separatrix_calibrated_required` to
`separatrix.probabilistically_calibrated_required`, validates the complete
frozen H-det and H-prob source schemas, and validates every reducer-facing
field, spectral, and transport dictionary. Before resubmission, the repaired
reducer must execute end to end on the exact score SHA-256
`055d81979f46a96bc0c983e0ef2f387f3032a2505117849089047e4f00b67dd3`
and exact comparator SHA-256
`2b04c10971e6d38ee439e33aa0b5331305acf16b38a96e7952fb26046049b5d2`.
One further CPU-only Rocky 9 retry may issue the frozen H-det/H-prob decision.
It may not regenerate or mutate forecasts, rescore truth, alter thresholds,
train, launch O3, open 85606, assimilate, or rank diagnostics. The failed
attempt is retained in
`results/phase3_b4_pde_refiner_gate_adapter_failure_6901282.json`.

## A019 - Localize the cause of K4 before closing model development

**Status:** frozen after the completed K4 interpretation and before any new
Phase 3.5 computation.

K4 established that one fixed, global, condition-independent linear residual
distribution fitted to adjacent 85604 training frames does not describe later
85604 residuals well. It did not test FGN, PDE-Refiner, diffusion, or
stochastic emulation generally, and it could not distinguish interval drift,
coherent periodic transport, translation non-equivariance, omitted state,
memory, conditional covariance, or inadequate effective sampling.

This amendment narrows the K4 stop rule to permit exactly one 85604-only
diagnostic cause-localization phase under
`PHASE3_5_PROTOCOL_AMENDMENT.md`. It authorizes no production model training.
Frozen H1 inference is allowed only for the all-shift equivariance audit, and
frozen B5 inference is allowed only for the preregistered fixed-seed
context-shuffle sensitivity. The train/guard/validation boundaries are
unchanged; the guard and 85606 remain closed. Physics quantities remain
evaluation-only. The phase must end with a ranked evidence memo and exactly
one recommended, not automatically authorized, next action.

## A020 - Freeze Phase 3.5 execution definitions before computation

**Status:** frozen during implementation and before any Phase 3.5 numerical
result or checkpoint inference.

The original Phase 3.5 amendment left the numerical meaning of an
"unambiguous" translation peak and the exact matched H5 source subset
implicit. A separate prospective clarification fixes those definitions,
retains K4's zero-empirical-mean projection convention, specifies the
first-to-last stationarity contrast, and fixes the eight-member B5 sampler
path. It changes no split, hypothesis, representation family, architecture
authorization, or held-out access rule. See
`PHASE3_5_PROTOCOL_AMENDMENT_2026-08-19A.md`.

## A021 - Begin the ECRD model-development repair

**Status:** frozen after completed Phase 3.5 job `6907468` and before ECRD
implementation, engineering smoke training, or new model results.

Phase 3.5 localized K4's failure sufficiently to stop treating diagnosis as a
blocking phase. It found strong evidence of H1/codec non-equivariance,
within-85604 drift, and state-dependent residual summaries; partial evidence
for retained state and history; and little H1-residual benefit from a single
truth-assisted bulk toroidal shift. The user explicitly authorized model
development on 2026-08-20.

The next intervention is the four-arm, three-seed 85604 ladder frozen in
`protocol/ECRD_MODEL_DEVELOPMENT_PROTOCOL.md` and
`manifests/ecrd_model_development_85604.json`: historical B5, deeply
conditioned B5, ECRD, and two-frame ECRD-History. ECRD combines a four-phase
symmetrized frozen H1 mean, a small equivariant residual-mean head, deep raw
C5P spatial FiLM, no toroidal downsampling, joint multiscale Gaussian
innovation, and shared circular-shift augmentation. All training objectives
remain field-only.

Simulation 85606, the guard, assimilation, diagnostic ranking, and steering
remain closed. A successful three-seed 85604 gate may only create a separate
explicit held-out release record; it does not silently open 85606.

## A022 - Test multi-lead fine-tuning before changing the operator

**Status:** frozen after the old-85604 state-view and derived-coordinate
screens and before any Stage-1 parent checkpoint was evaluated beyond its
trained one-frame lead.

The passing C5P Stage-1 operator already predicted finite-difference state
derivatives and conditioned every block on lead time, but it had been trained
only at lead one. A one-seed prospective screen therefore retained its
architecture and weights, evaluated the bitwise parent at leads
`1,2,4,8,16`, and fine-tuned it over the corresponding 2,129 training pairs.
No physics quantity entered the loss. Exact rules and advancement gates are
in
`protocol/POST_ECRD_OLD_85604_STAGE2_MULTILEAD_PROTOCOL_2026-08-25.md`.

## A023 - Confirm the passing multi-lead mechanism before bounded rollouts

**Status:** frozen after seed-1701 job `6936393` passed every prospective
screen gate and before seed-1702/1703 multi-lead training or bounded rollout
evaluation.

The selected seed-1701 model preserved its one-frame transition, beat
persistence for every C5P field at all five direct leads, and improved all
four longer leads. The result authorizes exactly two matched confirmation
runs initialized from the frozen Stage-1 seed-1702/1703 parents. All three
seeds must pass individually; seed aggregation cannot rescue a failure.

Only after that confirmation may inference compare direct lead-4/8 forecasts
with truth-free autoregressive compositions using leads one, two, and four on
the same chronological validation starts. This does not authorize long free
rollouts, stochastic training, assimilation, diagnostic ranking, 85606, or
the newer NERSC data. Exact rules are in
`protocol/POST_ECRD_OLD_85604_STAGE2_SCALING_ROLLOUT_AMENDMENT_2026-08-25.md`.

## A024 - Advance the confirmed direct-transition repair to bounded rollout

**Status:** recorded after seed-confirmation array `6936641` completed and
before any direct-versus-autoregressive forecast was evaluated.

Seeds 1701, 1702, and 1703 each passed every frozen multi-lead gate. The
three-seed median persistence-normalized direct-transition score over leads
`1,2,4,8,16` is `0.48813885947672625`; every C5P field has positive skill at
every lead for every seed. The verified reduction records
`three_seed_mechanism_confirmed: true` and
`bounded_rollout_authorized: true`.

This result activates only the inference comparison already frozen in A023:
matched 85604 validation starts and terminal horizons four and eight,
comparing persistence, direct prediction, and truth-free compositions of
leads one, two, and four. It does not authorize checkpoint tuning, further
training, a long free rollout, 85606, newer NERSC data, stochastic claims,
assimilation, diagnostic ranking, or steering. See
`POST_ECRD_OLD_85604_STAGE2_MULTILEAD_THREE_SEED_READOUT.md` and the tracked
three-seed reduction for the complete evidence and provenance.

## A025 - Train the small-step map against four steps of its own feedback

**Status:** frozen after bounded state job `6937051` and physics-scoring job
`6937203` completed, and before any new optimizer update.

The bounded 85604 comparison found a reproducible metric reversal. At horizon
eight, two repeated four-frame steps have the best median five-field state
skill (`0.428809`), while eight repeated one-frame steps preserve the most
toroidal power and give the lowest separatrix transport error. Direct and
repeated four-frame paths retain only approximately 11–17 percent mean
truth-relative evaluated band power and have separatrix relative-L2 error near
`0.9`, roughly twice persistence. The repeated one-frame path improves every
eight-frame separatrix quantity for every seed, but accumulates pressure-field
error, loses high-mode realization coherence, and retains strict-face
relative-L2 error near `1.38`.

This amendment authorizes one seed-1702 pilot initialized from the immutable
Stage-2 checkpoint SHA-256
`b9007f818eb35d82a1e4c21771dfc1ad870591feb777f5087f3b2a49847cd50d`.
The pilot must roll the lead-one transition autonomously for four steps during
training, use predicted intermediate states without teacher-forcing, supervise
all four future states through the existing channel-normalized field loss, and
retain a direct one-step field term. No physics-derived quantity may enter the
loss or checkpoint selection.

The train/guard/validation boundaries remain `[0,432)`, `[432,496)`, and
`[496,624)`. Only old simulation 85604 may be read. The pilot must use the same
five C5P fields, normalization, one-frame context, operator, precision policy,
and toroidal periodicity as Stage 2. Checkpoint selection is state-only on the
chronological validation interval. The exact schedule, weights, comparison
gates, seed, and artifact paths must be frozen in a separate prospective
protocol and manifest before submission.

After selection, inference must repeat the frozen four/eight-frame state,
spectral, cross-field, and transport evaluation. Advancement requires improved
four/eight-frame repeated-one-step state stability without material regression
in one-step accuracy, evaluated toroidal power, or separatrix transport. Only a
passing pilot may authorize two confirmation seeds. The pilot does not open
85606, newer NERSC data, assimilation, diagnostic ranking, steering, stochastic
claims, or any physics loss. See
`POST_ECRD_OLD_85604_BOUNDED_ROLLOUT_READOUT.md` for the complete evidence.
