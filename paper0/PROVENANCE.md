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

### Executed Hermes-3/BOUT++ transport source

The representative 85604 dump embeds Hermes-3 revision
`920ba829cc78cdab0dbf6101c69fecc4689bd8dd`; `BOUT.settings` embeds BOUT++
revision `7d28d67c3f12c24ec281c0982e870f5369c65a6f` and version `5.2.1`.
Clean detached checkouts of the official
[Hermes-3](https://github.com/boutproject/hermes-3) (GPL-3.0) and
[BOUT++](https://github.com/boutproject/BOUT-dev) (LGPL-3.0) repositories were
inspected without modification. Critical file hashes are recorded in
`paper0/manifests/phase1_85604_sources.json`.

No simulator code has been copied into Paper 0. The audit establishes that the
predecessor `src/tcv_eval/flux.py` is a geometry-incomplete proxy rather than a
port of the executed conservative operator. The exact definition, staged
implementation restrictions, and required oracle comparison are frozen in
`paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md`.

`src/tcv_diagnostics/transport.py` is a new NumPy transcription of only the
guard-independent radial `x-z` face-flow component. It was written from the
hash-locked equations rather than copied from GPL source, uses source-matched
corner placement and Monotonized-Central reconstruction, and is covered by
synthetic known-answer and finite-volume conservation tests. Its API and
metadata say `partial`; it remains scientifically blocked from use as total
transport until the shifted `x-y` component passes the BOUT++ oracle ladder.

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

Job `6890650` completed from clean Paper 0 commit
`2bf810ff226641ac1955367a18bd492ab08c442c` on Rocky 9 with an NVIDIA H100.
The tracked compact result is
`paper0/results/phase2_o1_codec_6890650.json`; it identifies the immutable full
artifact at
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec/job_6890650/o1_codec_metrics.json`
with SHA-256
`d9440ecf7182c434976b67a33118d8c3dcb81b0fcec9a16f89745a5398aa850e`.
The compact-record generator verifies that digest, the run ID, the blind-test
exclusion flag, the commit, and the SLURM job before writing. The O1 readout and
all four figure pairs are regenerated only from that tracked compact record.

## Phase 0 execution evidence

No predecessor code has been ported yet. The audit-only legacy reproduction executed the predecessor files in place after verifying their byte hashes. Job `6890428` used Paper 0 commit `7e2b5d2`, Rocky Linux 9.8, an NVIDIA H100, and only the legacy 85604 validation region. Its compact result record is `paper0/results/phase0_legacy_valid_6890428.json`; the full Rusty artifacts remain under `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase0_legacy_valid/job_6890428`.

The first submission, job `6890410`, stopped before model inference because the manifest contained a truncated validation-file hash. Commit `7e2b5d2` corrected the record and added a test that validates every JSON digest before launch.
