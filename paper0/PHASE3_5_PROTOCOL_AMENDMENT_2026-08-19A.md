# Paper 0 Phase 3.5 prospective execution clarification A

**Date:** 2026-08-19

**Timing:** written after implementation began but before any Phase 3.5 data
array, transfer result, translation estimate, probe score, or checkpoint
inference was computed

**Development simulation:** 85604 only

**Held-out simulation 85606:** unopened and unauthorized

This dated clarification does not change the hypotheses, split boundaries,
representation families, budgets, or decision priorities in
`PHASE3_5_PROTOCOL_AMENDMENT.md`. It closes execution-level ambiguities found
while turning that frozen document into fail-closed code.

## C1. Unambiguous shared toroidal peak

A shared integer displacement is classified as unambiguous only when all
three conditions hold:

1. peak normalized correlation is at least `0.25`;
2. peak minus the best non-neighbor peak is at least `0.05`;
3. normalized nonnegative correlation-surface entropy is at most `0.90`.

The complete continuous values remain reported. These thresholds apply only
to the evidence-tier decision; they do not censor plots or distributions.

## C2. Primary H5 representation source

The primary matched representation source is the concatenation of the first
42 targets in each training block, in chronological order: 420 targets total.
This is the largest source consistent with the already frozen equal-block
comparisons and the maximum real coefficient budget 416. The ten unused
forty-third targets remain in stationarity, ESS, context, and auxiliary
learning-curve summaries as specified previously.

H5 transfer is evaluated separately on V00, V01, and V02. Representation
allocation uses only the 420-target source. Target energy never selects a
direction, subband, patch, or coefficient.

## C3. Centering and K4-compatible aligned projection

Matched representation reconstruction uses the fitted 420-target source mean
when measuring source-centered representation transfer. This is a diagnostic
reconstruction quantity, not a forecast.

The separate alignment rerun of the K4 rank ladder follows K4 exactly:
training covariance modes are fitted after training empirical centering, but
the empirical nonaxisymmetric mean is not added to a forecast. Truth-assisted
validation projection contains only modal coefficients, and a field candidate
contains the frozen H1 center, the training axisymmetric bias, and that modal
projection.

## C4. First-to-last stationarity contrast

The primary first-to-last contrast is T00 versus V02. Its standardized effect
uses the pooled temporal standard deviation of the scalar frame series in
those two blocks. The confidence interval is formed from independent
non-circular moving-block resamples within T00 and V02 with the already frozen
block length, replicate count, and seed derivation. It never resamples across
the guard.

## C5. Fixed eight-member B5 inference

For the B5 context shuffle, the existing sampler is called through its frozen
lower-level `sample_normalized` and `denormalize_residual` methods with exactly
the first eight immutable seeds. This avoids generating and discarding 24
unauthorized extra members. The 18-step, 35-evaluation deterministic Heun
schedule, CPU PCG64 initial noise, seed order, field scaling, mismatch rule,
and original-H1 recentering are unchanged.

## C6. Execution accounting

The run manifest must hash this clarification in addition to the original
protocol and manifest. Any result produced without this hash lock is not an
authoritative Phase 3.5 result.

## C7. Moderate evidence tier

Where the original amendment specifies a strong threshold but not a moderate
threshold, moderate evidence requires either (a) at least one half of the
strong effect-size threshold with the same sign and no contradictory
validation block, or (b) the full strong effect size with fewer than the
required target families or consistency blocks. Weak evidence is a smaller
directionally consistent effect. This rule affects ranking labels only; every
underlying effect and consistency count is reported.
