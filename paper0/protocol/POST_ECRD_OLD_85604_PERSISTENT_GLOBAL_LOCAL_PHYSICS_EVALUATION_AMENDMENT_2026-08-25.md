# Persistent global--local physics evaluation: pre-sampling execution amendment

**Dated:** 2026-08-25

**Base protocol:** `POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PHYSICS_EVALUATION_2026-08-25.md`, SHA-256 `1079a8f6f42385f64ffd171b88a681eb8b4feb82eaa8d22a48c3237a77604896`

## Scope

This amendment records two execution-only failures that occurred before any
stochastic forecast member was sampled and before any target truth or physics
diagnostic was opened.

1. Job `6938320`, submitted from the Rocky 8 `rusty` login, stopped at the
   Rocky 9 runtime guard.
2. Job `6938324`, submitted correctly from `rusty9`, passed its tests and
   artifact authority checks, then stopped while comparing two aliases for the
   same selected checkpoint path.  The training result records the file below
   `/mnt/home/sdelaurentiis/ceph`, whose canonical Rocky path is below
   `/mnt/ceph/users/sdelaurentiis`.  Both aliases resolve to the same inode and
   the already frozen SHA-256 remained
   `4430eb1af96ee48faac80420227be42db363f5703712726b86d02836d42937eb`.

No forecast HDF5 file, generation result, W&B evaluation run, target-truth
read, physics score, checkpoint selection, or training update was produced by
either failed job.

## Authorized implementation correction

The generator now canonicalizes the checkpoint path with
`Path.resolve(strict=True)` before comparing it with the already canonicalized
manifest path.  SHA-256 verification remains mandatory and unchanged.  This
does not alter the checkpoint, model, contexts, starts, seeds, sampler,
ensemble size, horizons, physics operators, bootstrap, thresholds, or decision
logic.

The evaluation manifest is reissued before sampling to lock this amended
protocol and the corrected generator source.  Every other evidence and code
hash remains unchanged.
