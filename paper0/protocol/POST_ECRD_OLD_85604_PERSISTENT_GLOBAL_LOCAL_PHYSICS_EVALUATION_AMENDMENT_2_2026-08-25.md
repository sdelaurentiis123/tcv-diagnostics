# Persistent global--local physics evaluation: sparse-context execution amendment

**Dated:** 2026-08-25

**Prior amendment:** `POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PHYSICS_EVALUATION_AMENDMENT_2026-08-25.md`

## Scope

Rocky 9 job `6938336` passed its implementation tests, artifact authority,
runtime check, and checkpoint load.  It then stopped while constructing the
truth-free context dataset because the historical O2 loader deliberately
accepts only contiguous target lists whereas this evaluation preregistered 36
sparse starts.  The failure occurred before W&B initialization, forecast-file
creation, stochastic sampling, future-target access, or physics scoring.  Its
dependent scorer `6938337` therefore never ran.

## Authorized implementation correction

The immutable O2 context loader remains byte-for-byte unchanged.  It is
constructed over the enclosing contiguous validation interval, and a new
evaluation-specific adapter maps each preregistered sparse target to its exact
index.  The adapter exposes only those 36 contexts to the generator and checks
the returned target coordinate on every access.  It cannot expose target
truth.

This changes no data boundary, input frame, target, model, checkpoint, seed,
sampler, ensemble size, horizon, metric, bootstrap, threshold, or decision.
The evaluation manifest is reissued before sampling to lock the adapter,
generator, forecast module, and this amendment.  All other frozen hashes are
unchanged.
