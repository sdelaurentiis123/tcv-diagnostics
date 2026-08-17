# Paper 0 provenance

## Repository lineage

This is a clean Paper 0 repository. It does not inherit the Git history of the exploratory predecessor.

The predecessor evidence source is:

- Local path: `/Users/stanislavdelaurentiis/tcv-gaot-3d`
- Branch at initialization: `probe-conditioning-diagnostics`
- Final presentation commit before Paper 0 initialization: `b367d8b`
- Prior evidence-audit commit: `a10fcc5`

For Paper 0, the predecessor is an auditable, read-only source of candidate loaders, model implementations, checkpoints, commands, and historical results. Any code ported from it must be listed below with its exact origin commit and the modifications made.

## Ported code ledger

| New path | Source path | Source commit | Modifications | Verification |
|---|---|---|---|---|
| _none yet_ | | | | |

## External method ledger

External implementations or method-specific code must record repository URL, revision, license, local modifications, and validation tests here before use.

## Phase 0 execution evidence

No predecessor code has been ported yet. The audit-only legacy reproduction executed the predecessor files in place after verifying their byte hashes. Job `6890428` used Paper 0 commit `7e2b5d2`, Rocky Linux 9.8, an NVIDIA H100, and only the legacy 85604 validation region. Its compact result record is `paper0/results/phase0_legacy_valid_6890428.json`; the full Rusty artifacts remain under `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase0_legacy_valid/job_6890428`.

The first submission, job `6890410`, stopped before model inference because the manifest contained a truncated validation-file hash. Commit `7e2b5d2` corrected the record and added a test that validates every JSON digest before launch.
