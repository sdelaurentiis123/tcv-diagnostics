# Phase 3 B2 truth-event eligibility amendment

**Decision status:** frozen after the original B2 gate result and before an
amended evaluator is implemented or run

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

## 1. Why an amendment is necessary

The original three-seed B2 matrix is immutable:

```text
job: 6897564
status: completed_failed_frozen_one_step_gate
SHA-256: cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6
```

It applied the prospectively frozen B2 gate exactly as written. All three B2
seeds failed the field, spectral, and transport families, and the architecture
failed. Those failures remain the historical result.

Inspection after finalization identified one model-independent defect in the
event-conditioned chronological-block reduction. Event thresholds were fitted
on the 85604 training region only. In validation block 3, none of the four
truth transport series exceeds its frozen threshold. The scorer correctly
records, for every quantity:

```text
validation_event_count = 0
defined = false
magnitude_relative_error = null
truth_magnitude_weighted_sign_disagreement = null
```

The original gate nevertheless required `defined=true` and finite event
errors in every block. No forecast can make a truth-only event subset exist.
Consequently this condition is not a model-quality threshold; it is an
undefined-estimand bug. It also makes the catastrophic finite-metric check fail
for every possible model on these fixed blocks.

This amendment changes only the treatment of event-conditioned metrics in
chronological blocks with zero truth events. It does not change any model,
forecast, score, event threshold, validation frame, block boundary, numerical
acceptance threshold, or non-event check.

## 2. Original result is retained

The original evaluator, finalizer, launcher, and matrix are not edited or
overwritten. The amended gate must live under new versioned names and write a
new unique artifact. The amended artifact must reference and hash the original
matrix.

Both results remain reportable:

1. **Original frozen execution:** the exact pre-result gate, including its
   impossible zero-event requirement.
2. **Amended scientific interpretation:** the same stored forecasts and scores
   reduced under the truth-event eligibility rule below.

The amendment may not be used to describe B2 as prospectively passing. It is a
post-result correction to an undefined metric and must be labeled as such.

## 3. Frozen truth-event eligibility rule

For each transport quantity and each chronological validation block, event
eligibility is determined only from the stored truth count:

```text
eligible := validation_event_count > 0
```

The count must be an exact nonnegative integer. Forecast values, forecast
errors, ensemble spread, and gate outcomes may not influence eligibility.

### 3.1 Eligible block

When `validation_event_count > 0`:

- `defined` must be true;
- event-conditioned magnitude relative error must be finite and at most 0.50;
- truth-magnitude-weighted sign disagreement must be finite and at most 0.25.

Every eligible block must pass both numerical event thresholds. The general
five-of-six complete chronological-block rule remains separately required.

### 3.2 Ineligible block

When `validation_event_count == 0`:

- `defined` must be false;
- both event-conditioned numerical values must be JSON null;
- the event-conditioned accuracy checks are recorded explicitly as not
  applicable, never as zero and never as a pass inferred from a fabricated
  number;
- every field, spectral, ordinary transport, calibration, and integrity check
  in that block remains required.

The catastrophic all-numeric-finite check excludes only these explicitly
not-applicable event-conditioned values. Every other required numerical metric
must remain finite.

### 3.3 Support requirement

For each of the four transport quantities, at least five of the six frozen
validation blocks must be event eligible. If fewer than five are eligible, the
amended event gate fails for insufficient validation support. It is not
reported as model failure and cannot be rescued by changing thresholds or
blocks.

## 4. Consistent rerun

The amended evaluator must be applied to all three stored B2 seeds, not only a
selected seed. It must consume the exact forecast, truth-separated score,
deterministic comparator, training matrix, event-threshold record, manifest,
and protocol hashes referenced by original matrix job 6897564.

No inference, training, resampling, truth scoring, or metric recomputation is
authorized. This is a deterministic gate-only reduction of already immutable
records. Automated tests must cover:

- five eligible blocks plus one internally consistent zero-event block;
- fewer than five eligible blocks;
- positive truth count with `defined=false`;
- zero truth count with non-null event errors;
- continued failure for a non-event non-finite metric;
- unchanged behavior of the original frozen evaluator.

The amended output must state whether the architecture decision changes. If
B2 still fails, the failure branch remains FGN or a joint stochastic residual;
O3, assimilation, diagnostic ranking, and 85606 access remain closed.

## 5. Claims boundary

This amendment can establish only that B2 is judged without treating a
nonexistent truth subset as a forecast error. It cannot improve or reinterpret
the measured field, spectral, cross-field, transport, or calibration values.
It cannot authorize model tuning on 85604 validation and cannot be applied
differently across model families.

The same versioned truth-event eligibility rule must be used for every future
model evaluated on these six validation blocks. Any later change requires a
new prospective amendment and a consistent rerun of every affected model.
