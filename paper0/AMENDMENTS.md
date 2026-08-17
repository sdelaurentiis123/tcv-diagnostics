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
