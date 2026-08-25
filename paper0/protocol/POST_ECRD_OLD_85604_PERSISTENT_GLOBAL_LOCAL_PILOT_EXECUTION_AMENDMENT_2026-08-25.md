# Persistent global--local pilot execution amendment

**Frozen:** 2026-08-25 after engineering smoke job `6937573` completed and
before the 20-epoch pilot was submitted or produced any result

**Development simulation:** 85604 only

**Held-out 85606 and newer NERSC data:** unopened and prohibited

## Smoke gate

The engineering smoke authorized by
`POST_ECRD_OLD_85604_PERSISTENT_GLOBAL_LOCAL_PILOT_2026-08-25.md`
completed at Paper 0 commit `58cc3d098c2cad1a5ebbcf624c85a8b25ae997fa`.

- Slurm job `6937573`: `COMPLETED`, exit `0:0`;
- online W&B run: `p0oldpglsmoke-j6937573-s1702`, remotely `finished`;
- four optimizer updates and one EMA checkpoint completed exactly;
- bitwise checkpoint reload passed;
- every one of 88 integer toroidal shifts passed with maximum relative L2
  equivariance error `8.322180722331041e-10`;
- peak CUDA allocation was `4.84102725982666 GiB`;
- no physics diagnostic entered training, validation, or interpretation;
- guard frames, 85606, and newer NERSC data were not read.

The smoke fit the frozen parent-residual scale procedure on all 428 authorized
training windows.  The resulting immutable artifact is:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
post_ecrd_old_85604_persistent_global_local_smoke/
job_6937573_seed1702/run/residual_scales.json
```

SHA-256:
`497a655bc6914c30d78831b04b157ad4c07e17a7de6c5887e378a36a53d475bd`.

The scale artifact is a numerical training-only RMS, not a physics-derived
quantity.  It is reused without refitting in the pilot.

## Authorized pilot

The original protocol's 20-epoch seed-1702 pilot is now authorized exactly as
frozen:

- all 428 old-85604 training windows;
- all 124 old-85604 validation windows;
- 20 epochs and exactly 4,280 optimizer updates;
- checkpoint candidates every two epochs;
- two fixed data-only validation probes per window;
- stochastic-branch peak learning rate `1e-4`;
- initialized mean-branch peak learning rate `1e-5`;
- one generic preemptible Rusty GPU, four CPUs, and 24 GiB host memory;
- required online W&B and immutable local checkpoints.

Checkpoint selection remains the equal-block mean of EDM validation loss plus
normalized mean-state MSE.  Physics quantities remain absent from training
and selection.  The pilot may authorize a separately frozen physics
evaluation only if its mechanical and state gates pass.  It cannot authorize
confirmation seeds, 85606, assimilation, sensor ranking, or steering by
itself.
