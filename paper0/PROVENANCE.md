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

### Phase 2 O1 historical codec execution

The O1 representation oracle imports the existing predecessor LOLA package in
place; it does not copy or modify that package. The embedded checkout reports
base revision `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`, but local changes are
part of the executed implementation, so the revision alone is insufficient.
The locked launcher verifies the composite SHA-256
`3fb6e6be7649e86fc0626f5d847adf13649e213c82b543c714ae258332bfdf7d`
over all package Python files and the five critical file hashes frozen in
`paper0/protocol/PHASE2_O1_CODEC_PROTOCOL.md`.

The original upstream method repository is
[PolymathicAI/lola](https://github.com/PolymathicAI/lola), under its recorded
upstream license. The executed code is the audited local TCV extension rather
than an unmodified upstream release. Paper 0 adds only a read-only adapter and
independently tested NumPy reductions:

| Paper 0 path | Role | Verification |
|---|---|---|
| `paper0/tools/evaluate_codec_oracle.py` | Loads the hash-locked historical codecs and streams 85604 frames through `decode(encode(x))` | Clean-commit launcher, shape/config assertions, deterministic CUDA settings |
| `src/tcv_diagnostics/codec_oracle.py` | New Paper 0 metric reductions; no copied predecessor functions | Synthetic identity, missing-mode, phase-shift, Parseval, merge, and no-clipping tests |
| `cluster/phase2_o1_codec_oracle.sbatch` | Rocky 9 execution and integrity gate | Literal hash/path regression test and `bash -n` |

The f8 and z44 checkpoints remain external immutable artifacts; their paths,
hashes, lineage, and non-comparability caveat are frozen in the O1 protocol.

## Phase 0 execution evidence

No predecessor code has been ported yet. The audit-only legacy reproduction executed the predecessor files in place after verifying their byte hashes. Job `6890428` used Paper 0 commit `7e2b5d2`, Rocky Linux 9.8, an NVIDIA H100, and only the legacy 85604 validation region. Its compact result record is `paper0/results/phase0_legacy_valid_6890428.json`; the full Rusty artifacts remain under `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase0_legacy_valid/job_6890428`.

The first submission, job `6890410`, stopped before model inference because the manifest contained a truncated validation-file hash. Commit `7e2b5d2` corrected the record and added a test that validates every JSON digest before launch.
