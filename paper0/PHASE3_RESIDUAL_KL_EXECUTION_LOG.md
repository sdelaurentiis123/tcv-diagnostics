# Phase 3 residual-KL execution log

This log records operational failures and post-opening bug fixes for the
frozen 85604 residual-KL oracle.  It does not amend the scientific protocol,
selection rule, thresholds, rank ladder, seed bank, or data boundaries.

## Job 6904340 — launcher failure before scientific access

- Submitted from Rocky 9 at commit
  `5ff472faf41fc13e881f97b38bef45d421e275ab`.
- Failed before tests or scientific-data access because the worker did not
  define `SLURM_TMPDIR` and the launcher used shell nounset semantics.
- No training, validation, guard, or held-out truth was opened.
- Fixed at commit `644ee866c6edd6e9636b696f321a19b840e09782`
  by selecting writable node scratch from `SLURM_TMPDIR`, then `TMPDIR`, then
  `/tmp`, with an explicit writability check.

## Job 6904346 — evaluation metadata-order failure after validation opening

- Submitted from Rocky 9 at commit
  `644ee866c6edd6e9636b696f321a19b840e09782`.
- All frozen source and input hashes matched.
- The complete Rocky 9 suite passed: 1,215 tests passed, one skipped, and 29
  subtests passed.
- W&B run: `p0reskl-6904346-s1701`.
- The training-only pretruth stage completed and selected static rank 128 by
  the frozen rule.  Its pretruth closure SHA-256 was
  `5cbb6d15084fdb794c5c01b8dc5c6f3dc8b0b7356987bc90f06fb8dac582001e`.
- Validation truth was then opened.  Rank zero completed.  Rank eight stopped
  before its dependence-distance record was produced with
  `ValueError: KL dependence-distance cross-field region order differs`.
- Slurm state: failed with exit code 1 after 00:05:04 on `worker7401`.

### Root cause

The training covariance record was persisted with the repository's strict
JSON writer, which sorts mapping keys.  Fresh validation and projection
records retained the frozen B2 insertion order.  The evaluator compared the
iteration order of the dictionaries even though every comparison was already
performed by explicit region name.  Thus semantically identical region sets
were rejected solely because serialization changed their presentation order.

### Permitted bug fix

The evaluator helper now requires exact equality of cross-field region names
and continues to access every matrix by its explicit region name.  It emits
records in validation-record order.  It still fails closed for any missing or
additional region.  Regression tests cover both reordered-equal and
different-name cases.

No numerical estimator, data value, rank, seed, threshold, metric definition,
or acceptance rule changed.  The failed partial evaluation is not used as a
result.  Every rank and both oracle tiers will be rerun from a fresh Slurm job
directory after the fix passes the complete test suite.  Run 85606 remains
unread.
