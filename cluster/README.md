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

## Phase 1 immutable 85604 profile

`phase1_profile_85604.sbatch` is a CPU-only, read-only pass over the two audited
85604 Well storage shards plus representative raw metadata. It verifies all
source hashes, reconstructs the 624-frame chronology, applies the precommitted
steady-state screen, fits candidate training-only normalization, and computes
the frozen decorrelation estimate only if the screen passes. It cannot read a
path containing `85606` or a `test` path component.

Submit from the Rocky 9 login node after non-destructively syncing an exact,
clean commit:

```bash
export PAPER0_EXPECTED_COMMIT="$(git rev-parse HEAD)"
sbatch --export=ALL,PAPER0_EXPECTED_COMMIT cluster/phase1_profile_85604.sbatch
```

The unique result directory is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase1_85604_profile/job_<id>`.
No learned model is launched by this job.

## Phase 2 exact BOUT++ transport-oracle build

`build_bout_transport_oracle.sbatch` builds the exact BOUT++ revision embedded
in the 85604 simulation together with its pinned netCDF C++ dependency. The
source and submodule revisions and all transport-critical BOUT++ files are
hash-checked before compilation. The build is CPU-only, reads no shot data,
uses no GPU, and refuses an existing output directory.

The external source caches are intentionally outside Git under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external`. Submit from the Rocky 9
login node only after the launcher itself is committed and synced:

```bash
export PAPER0_EXPECTED_COMMIT="$(git rev-parse HEAD)"
sbatch --export=ALL,PAPER0_EXPECTED_COMMIT cluster/build_bout_transport_oracle.sbatch
```

This build does not release the transport metric. It only supplies the exact
BOUT++ dependency required to validate shifted `DDY` and the later full
Hermes face-flow oracle.

The accepted dependency build is Slurm job `6890722`, produced from Paper 0
commit `e298337918582293b682cc3c0465175634f29da3`. Its compact tracked index is
`paper0/results/phase2_bout_build_6890722.json`; the immutable install is under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890722`.
