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

The first external BOUT++ build attempt, Slurm job `6890717`, stopped before
configuration because the launcher mistyped the exact `fmt` gitlink. It read no
shot data and compiled no source. Its unique failed-attempt directory is kept
under the external build cache; the correction is documented in
`paper0/AUDIT.md` rather than deleting or reusing that directory.

The second attempt, job `6890720`, passed all source locks but stopped while
configuring netCDF C++ because HDF5 was not explicitly present in the module
environment. It likewise read no shot data and did not compile BOUT++. The
launcher correction adds only the `hdf5/1.14.5` Rocky 9 module; the unique
failed-attempt directory is retained.

The third attempt, job `6890722`, completed compilation from clean Paper 0
commit `e298337918582293b682cc3c0465175634f29da3` on Rocky 9 and installed
BOUT++ `5.2.1` at the exact embedded revision. Runtime job `6890751` then
proved that install unusable: `netcdf-c/4.9.2` requires HDF5 `1.12.3`, while
the build launcher had also linked HDF5 `1.14.5`. HDF5 correctly aborted on
the header/library mismatch before opening the geometry or evaluating `DDY`.
No unsafe version-check bypass was used. The immutable build and runtime
failure are indexed by `paper0/results/phase2_bout_build_6890722.json` and
`paper0/results/phase2_shifted_ddy_6890751.json`; the install is superseded.
The corrected build uses the matching HDF5 `1.12.3` ABI in a new result
directory.

The corrected fourth build, job `6890766`, completed from clean Paper 0 commit
`b70ec7ea608c89451a4e3f269b4f11ace4a87269`. Dynamic-link inspection resolves
both `libhdf5.so.200` and `libhdf5_hl.so.200` exclusively from the Rocky 9
HDF5 `1.12.3` module, with no mixed ABI and no version-check bypass. Its exact
environment, Slurm record, revisions, install, and hashes are tracked in
`paper0/results/phase2_bout_build_6890766.json`. It is an ABI-validated
dependency build; the scientific shifted-`DDY` gate still requires the
unchanged runtime comparison.

The candidate shifted-derivative transcription in
`src/tcv_diagnostics/transport.py` follows the hash-locked BOUT++ FFT phase,
default centered `C2` stencil, single-null logical connections, and
`fixZShiftGuards` branch signs. Synthetic tests cover phase shifts, inverse
shifts, core and private-flux neighbor maps, nonzero `ShiftAngle`, and an
analytic field-aligned gradient. It remains explicitly `partial` and is not an
accepted source oracle until its arrays match the compiled BOUT++ harness.

Compiled runtime job `6890782` confirmed that the ABI-clean install opens the
geometry and reads its topology, but the original one-rank harness layout is
invalid for this diverted mesh: each 8-cell leg region must be divisible by
`MYSUB`, whereas one rank gives `MYSUB=32`. BOUT++ stopped before any operator
evaluation. The preserved no-result record is
`paper0/results/phase2_shifted_ddy_6890782.json`. The corrected harness uses
the minimal valid `NXPE=1`, `NYPE=4`, `MYSUB=8` decomposition and reconstructs
the global `y` axis by verified `PE_YIND`; numerical acceptance settings and
candidate code are unchanged.

Four-rank job `6890792` then evaluated all four operators and wrote rank-local
outputs, but the executable's final `checkForUnusedOptions` rejected the four
manufactured `mesh:input_*` nodes. BOUT++'s `mesh->get` consumed the expressions
through its FieldFactory fallback without marking the corresponding `Options`
nodes. No comparison was accepted. The narrow correction calls
`setConditionallyUsed()` on exactly those four named nodes; it does not disable
global unused-option checking or modify an input, operator, mask, or tolerance.
The no-result artifact is `paper0/results/phase2_shifted_ddy_6890792.json`.

Job `6890796` exposed a deeper harness issue before comparison. On a
file-backed mesh, `mesh->get` warned and substituted zero rather than using the
GridFromOptions expression fallback assumed from BOUT++'s small synthetic
test. It also showed that the authoritative grid stores finite `ShiftAngle`
only for `grid_x < ixseps1=18` and NaN in the SOL, where BOUT++ never uses the
value. The job is an explicit no-result artifact at
`paper0/results/phase2_shifted_ddy_6890796.json`. The corrected driver reads
each exact tracked expression and calls BOUT++ `FieldFactory::create3D`; the
comparator now rejects collapsed manufactured inputs. The NumPy candidate
requires finite branch shifts only for model `x < 16` and replaces only the
topology-unused outer entries. No numerical error tolerance changed.

The compiled comparison harness under
`paper0/oracles/bout_shifted_ddy/` uses the initialization, mesh-load,
communication, derivative, and default-output pattern from BOUT++
`tests/integrated/test-yupdown/test_yupdown.cxx` at the same locked revision.
Because that reference test is GPL-licensed, the small adapted driver is
explicitly marked `GPL-3.0-or-later`. The driver adds Paper 0's four frozen
manufactured cases and calls the installed BOUT++ `DDY` with explicit `C2`.
The independently written NumPy comparator is
`paper0/tools/compare_shifted_ddy_oracle.py`; its region masks, guard removal,
and prospective numerical tolerance are tested locally and frozen in the
transport protocol before the first execution.

Corrected runtime job `6891059` completed from clean Paper 0 commit
`0223035106cb6a81c041ecd8701a99df9e39c59b` on Rocky 9. The four explicit
FieldFactory inputs passed their independent non-collapse checks. Every
constant, toroidal-mode, `y`-code, and mixed-field comparison passed in the
full valid region, ordinary sequential stencils, both private-flux
connections, both twisted core connections, and the open SOL. The largest
absolute discrepancy was `3.025468764406014e-12`, versus the unchanged frozen
rule `5e-10 + 5e-10 * max_abs_reference`; no compared value was non-finite.
The tracked compact record is
`paper0/results/phase2_shifted_ddy_6891059.json`; the immutable full JSON and
arrays have SHA-256 `44389079dd64203b7dc9706b38180afad48beb91fd86e6b93f33041d696c99cb`
and `1d366e0643d2ae3420460743a926aaf291093e7e8facc3e60cd5841f43bc85d9`.
This accepts the NumPy shifted-`DDY` primitive for its declared interior scope.
It does not accept the shifted-`xy` face term or total particle/internal-energy
transport.

`fromm_radial_face_states_partial` and the function initially committed as
`radial_exb_xy_face_flow_candidate_partial` are new independent NumPy
transcriptions of the radial `x`-face code at Hermes-3 revision `920ba829`,
`src/div_ops.cxx:273-326`; no GPL source text was copied into the NumPy
implementation. Synthetic tests lock the four-cell Fromm formulas, both
velocity signs, positivity clipping, geometry factors, target masking, and
constant-potential zero flow. After the compiled comparison below passed, the
same implementation was promoted mechanically to
`radial_exb_xy_face_flow_partial`; the `_partial` scope remains mandatory.

The prospective comparison under `paper0/oracles/hermes_xy_face/` is marked
GPL-3.0-or-later because its small C++ driver adapts the internal radial-face
calculation from the locked Hermes source. It exposes the otherwise private
velocity, chosen Fromm state, clipping decision, and flow without copying that
code into the independent NumPy candidate. The launcher refuses a dirty Paper
0 checkout, verifies the official Hermes revision and `div_ops.cxx` hash, uses
the accepted BOUT++ install and four-rank topology, and reads no plasma-state
frame. `paper0/tools/compare_hermes_xy_face_oracle.py` contains the independently
written comparison and the protocol's prospectively frozen continuous,
binary-decision, topology-region, input-noncollapse, sign-coverage, and
clipping-coverage gates. This paragraph records the pre-execution design; the
immutable execution result follows.

Rocky 9 job `6891343` completed from clean Paper 0 commit
`ee2b04ff381466ae62054616f7e59410b868ed08`. All continuous velocity, selected
Fromm-state, and face-flow comparisons passed in every case and topology
region. The largest absolute discrepancy was
`2.5619506516250112e-12`, versus the unchanged rule
`5e-10 + 5e-10 * max_abs_reference`. All 592,920 clipping decisions matched;
the dedicated case included 85,830 selected clipped states and 62,400 selected
unclipped states. Every nonconstant case contained both velocity signs and no
compared value was non-finite. The compact record is
`paper0/results/phase2_hermes_xy_face_6891343.json`; the immutable full JSON and
arrays have SHA-256 `c024ab47a82e1cdbb59a032b97593e4e5ecdecfa13dea228384b24264de2ac10`
and `bfa42628c536cd6833a3ccd8aeb235d83bd5e934caea27d5b96194777f382c13`.
This accepts only the guard-independent shifted-`xy` radial face component.

The implementation initially named `radial_exb_face_flow_candidate_partial`
forms the pointwise sum of the separately validated `xz` and shifted-`xy` face
components on their common guard-independent mask. Its companion divergence
uses consecutive radial faces and the source volume factor `J*dx`. Synthetic
tests verify exact component addition, constant-potential zero flow, and
volume-weighted telescoping for batched fields.

The prospective combined comparison under
`paper0/oracles/hermes_radial_flow/` is GPL-3.0-or-later because its C++ driver
adapts the private radial `xz` and shifted-`xy` face calculations from the
locked Hermes source. It writes each component, their sum, and the radial
finite-volume divergence. The independent comparator verifies native `dz`,
each topology region, exact component addition, volume-weighted telescoping,
both signs of nonconstant `xz` and total flow, and the precommitted tolerances.
The launcher uses the same clean-commit, official-source, ABI, geometry,
four-rank, no-plasma-frame, and no-blind-test gates as the accepted component
oracle. This paragraph records the pre-execution design; the immutable result
follows.

Rocky 9 job `6891373` completed from clean Paper 0 commit
`b6926caf6aba4cc14c947a0542246564845b8d9d`. Every `xz`, shifted-`xy`, total
face-flow, and divergence comparison passed in all four cases and every frozen
topology region. Native `dz` matched `2*pi/(5*81)` exactly; component addition
was exact in reference and candidate; and the worst volume-weighted
conservation residual was `1.1368683772161603e-13`. The worst face discrepancy
was `2.5619506516250112e-12`. The worst divergence discrepancy was
`3.073364496231079e-08` on a reference magnitude of `3680114.2065100316`, or
`8.351274780533639e-15` relative. The compact record is
`paper0/results/phase2_hermes_radial_flow_6891373.json`; the immutable full JSON
and arrays have SHA-256
`111054902cfe43f4e07826f693fb394567ff93313243c5c28a161789ceda269c` and
`9dbe98ebefe0ad2baaecca5f650cbb9101fb5bcf4160974721b7bdda27d7daa7`.

After acceptance, the same implementation was promoted mechanically to
`radial_exb_face_flow_partial` and
`divergence_from_radial_face_flow_partial`. The `_partial` suffix remains
mandatory: no plasma frame, particle/internal-energy definition, physical
surface integral, SI conversion, ensemble, learned model, diagnostic, or
85606 data was evaluated by this job.

The native-frame oracle was then frozen in
`paper0/protocol/PHASE2_NATIVE_FRAME_PROTOCOL.md` and the machine-readable
`paper0/manifests/phase2_native_frame_oracle.json` before its extractor read
the selected 85604 state values. Its extractor assembles seven fields from all
256 raw rank files by explicit processor coordinates, strips exactly two
guards on each decomposed axis, retains native `z=81` and `zperiod=5`, verifies
the archived times and source/configuration identities, and refuses existing
output paths. The GPL-marked driver under
`paper0/oracles/hermes_native_frames/` adapts the same locked private Hermes
radial-flow calculations as the manufactured combined-flow oracle and reads
only the resulting canonical 85604 file. The independent comparator checks
the compiled operator and the prospectively required `Ni`, `Pe`, and `Pi`
closures separately.

Rocky 9 job `6891379` ran from clean Paper 0 commit
`7d5522c2d060580e2ec292e8cb7354b8990305f4`. Its compiled four-rank step
completed `0:0`; all 15 real-state operator cases passed every frozen quantity
and region. The overall batch job intentionally exited `1:0` because one
full-domain closure point failed: frame 312 at model `(6, 31, 73)` retains
negative evolved `Pi`, whereas `Ti` is derived by the locked Hermes
`EvolvePressure` source from `floor(Pi, 0)`. No acceptance setting changed.
The compact record is
`paper0/results/phase2_hermes_native_frames_6891379.json`; it records the
immutable root
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_hermes_native_frames/job_6891379`
and all artifact digests. In particular, the canonical input, full oracle
JSON, and comparison arrays have SHA-256
`61dee30a92bb1a3eefcece80faa42d143271bfe200f4024b9747aeb06747bc21`,
`c31ef79e98669c54b2a005efeb4581f9cce84a75e7c277583422aa07a05d2987`,
and `9c5fe8a72a4ed6a1d67dc94e93913be978dda603db609af1f7dcdceaf8237082`.
This execution read no 85606 state.

The prospectively frozen all-frame pressure-closure audit was first attempted
serially by Rocky 9 job `6891417` from clean commit
`39bfb22ebd2eed9ee67bc193d958298857fd1e21`. All provenance gates passed, but
the native NetCDF files are chunked by full local rank and frame. After roughly
37 minutes, a read-only inspection of the running process showed the open file
was only `BOUT.dmp.42.nc` of 256. The job was cancelled at 38:46 with peak RSS
6,619,428 KiB; it had written no result JSON. This is a no-result implementation
performance failure, not a data or closure finding. Its compact record is
`paper0/results/phase2_pressure_closure_6891417.json`. The unique external
directory is retained. The next implementation may parallelize disjoint rank
reads and merge only after complete coordinate coverage, while leaving every
frozen scientific choice unchanged.

The prospective correction uses 16 modulo-partitioned rank shards in one
CPU-only Slurm allocation. Each shard re-verifies its assigned files and emits
only sufficient statistics and field-stream digests. The independent merger
under `paper0/tools/merge_85604_pressure_closure_shards.py` refuses a result
unless shard, rank, and processor-coordinate coverage are each complete and
duplicate-free. Synthetic tests establish that merging value and closure
statistics exactly reproduces a single-pass calculation, including frame
gates, target/interior counts, extrema, and most-negative locations. This is
an execution repair only; no data definition or acceptance rule changes.

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
