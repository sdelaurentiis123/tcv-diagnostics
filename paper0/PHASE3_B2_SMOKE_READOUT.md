# Phase 3 B2 latent-diffusion smoke readout

## Decision

The bounded B2-LDM-H2 implementation smoke passed. This accepts the mechanics
needed to train and sample the matched C5P latent-diffusion baseline. It is not
a forecast-quality result, does not authorize held-out access, and does not by
itself authorize full training.

The exact compact result is
`paper0/results/phase3_b2_ldm_gpu_smoke_6896402.json`, SHA-256
`fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360`.
The immutable full artifact root is:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_gpu_smoke/job_6896402
~~~

Its top-level artifact index has SHA-256
`b3e2ed075c7fa3dc635cafce00014ac598886ef0cf88aecaa883d75588ac3dbf`.

## What actually ran

Rocky 9 H100 job `6896402` ran clean Paper 0 commit
`d58b4cc261a901b69c772b01270f38a89deb042f` and completed with Slurm exit
`0:0` in 54 seconds. Before GPU execution it:

- rechecked the protocol, manifest, source, codec, and 85604 data hashes;
- authenticated the required online W&B run;
- passed the complete suite (`652 passed, 1 skipped, 29 subtests`);
- staged and re-hashed all eight 85604 model-data shards.

The model then used seed 1701, 16 training targets, four validation targets,
two epochs, and two optimizer steps. The codec remained frozen. The complete
masked-trajectory, context-slot, and target-slot denoising losses were finite,
as were both gradient norms. These loss values are smoke diagnostics only: the
budget is intentionally too small for a scientific comparison.

## Sampler and reload gate

The job exercised the real Azula 0.3.1 Adams-Bashforth sampler at 16 steps and
order three, rather than a unit-test substitute. For one validation context it
generated two members with canonical shape:

~~~text
[batch=1, member=2, future_time=1, channel=5, x=64, y=32, z=88]
~~~

All values were finite. The two members differed in both standardized latent
space (RMS difference 1.5679228541491845) and decoded standardized field space
(RMS difference 0.40717558917264185). A fresh model and codec reload using the
selected checkpoint and the same sampler seed reproduced the sampled latents
and decoded forecasts bit-for-bit in the same process on the same H100.

This establishes functioning stochastic sampling, canonical axes, nonzero
member diversity, and checkpoint reproducibility. It does not establish that
the diversity has the right magnitude, spatial structure, modal structure,
cross-field covariance, or transport distribution.

## Explicitly still open

The following remain false after this smoke:

- scientific forecast quality evaluated;
- full B2 training authorized;
- probabilistic acceptance gate evaluated;
- O3 or autonomous rollout authorized;
- assimilation or diagnostic ranking authorized;
- 85606 access authorized.

Before full training, Paper 0 still requires a committed protocol fixing the
scientific ensemble size, fair probabilistic metrics, realization-level
physics metrics, comparator treatment, uncertainty intervals, and stop/go
criteria using 85604 validation only.
