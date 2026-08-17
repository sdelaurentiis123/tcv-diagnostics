# Cluster launchers

Cluster scripts are committed, bounded, and provenance-locked before submission. They must print the repository revision and refuse dirty worktrees or mismatched inputs.

## Phase 0 legacy reproduction

`phase0_reproduce_legacy_valid.sbatch` runs one 48-frame, 64-member forecast on the legacy 85604 validation interval. One invocation produces:

- an autonomous free-rollout arm;
- an ETKF-assimilated arm using the legacy 69-channel `iter` direct-state layout;
- a read-only 85604 data audit;
- an exact command record;
- a compact comparison with the historical result.

This is an audit reproduction, not a final scientific sensor experiment. The `iter` layout directly samples standardized `Ne` and `phi` values and is classified as an oracle/control under `paper0/protocol/OBSERVATION_OPERATORS.md`.

Submit only from Rocky 9 after pulling a clean committed revision:

```bash
mkdir -p /mnt/home/sdelaurentiis/ceph/tcv_diagnostics/logs
export PAPER0_EXPECTED_COMMIT="$(git rev-parse HEAD)"
sbatch --export=ALL,PAPER0_EXPECTED_COMMIT cluster/phase0_reproduce_legacy_valid.sbatch
```

The launcher requests one Rocky 9 H100/H200 GPU through the `gpupreempt` partition and matching QOS. The `gpuxl` QOS has a four-GPU minimum and is intentionally not used for this one-GPU audit. Automatic requeue is disabled: a preempted attempt must be resubmitted under a new job ID, preserving the partial attempt rather than overwriting it.

The output directory is unique to the Slurm job ID, and the launcher refuses to overwrite it.
