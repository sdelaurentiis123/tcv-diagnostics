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
