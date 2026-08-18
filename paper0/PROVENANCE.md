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
| `src/tcv_diagnostics/models/layers.py` | PolymathicAI/LOLA `lola/nn/layers.py`; predecessor evidence `external/lola/lola/nn/layers.py` | upstream `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`; predecessor file SHA-256 `ad6aab36d52ea7aba2a0c45006a33413304c0d9ceb9abffd52a497a24adf616f` | Minimum PyTorch-only port of channel normalization and patch ordering; removes `einops`; adds stricter argument checks; ports the predecessor's explicit per-axis padding so `x/y=zeros` and `z=circular`; retained under the upstream MIT license | Paper 0 commit `c0cd66b`; new file SHA-256 `87265976a250ef1de81f19a59607b1c1493906ca2b72bc561816bf956302d12b`; known-answer patch ordering, inverse, wall, and periodic-wrap tests |
| `src/tcv_diagnostics/models/dcae.py` | PolymathicAI/LOLA `lola/nn/dcae.py` and `lola/autoencoder.py`; predecessor evidence `external/lola/lola/nn/dcae.py` | upstream `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`; predecessor file SHA-256 `281f8541aa09822147f8769e9a11fb63497aa54783dd9a806b173e76c5fbaede` | Minimum DCAE encoder/decoder and `softclip2` port; removes configuration-framework dependencies and optional attention; ports per-transition anisotropic strides; adds frozen named codec configs, latent-shape validation, and equal-channel standardized MAE | Paper 0 commit `c0cd66b`; new file SHA-256 `0f2e9a7445ae47915f334f01993fbf49adc4ac462bf69ef988a736e22bb9c554`; shape, backward, saturation, manifest-lock, and exact checkpoint-reload tests |
| `src/tcv_diagnostics/models/modulated_vit.py` | PolymathicAI/LOLA `lola/nn/vit.py` | upstream `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`; upstream file SHA-256 `33a0ebbbb6da8254f7129f564b783d5b98de12a812e403d1c790356ca184d6a2` | Minimum PyTorch-only port of the noise-modulated path; reuses Paper 0's already attributed attention, sine encoding, coordinate, patch, and unpatch primitives; removes `einops`, xFormers, windowed attention, and unneeded options; retains global AdaLN-zero modulation with strict shape checks; kept separate from the completed deterministic `vit.py` | Paper 0 commit `899b89d0d561dbf6d5c50064805fdbbffc0fc988`; new file SHA-256 `9bfa4c86287f533e6baa488be3f0ac6727a318a2e5a83f5c0b1be11f9dfe3bbe`; shape, gradient, modulation, exact reload, and frozen-O2-file-hash tests |
| `src/tcv_diagnostics/models/latent_diffusion.py` | PolymathicAI/LOLA `lola/diffusion.py` and the denoising call contract used by `lola/emulation.py` | upstream `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`; upstream diffusion file SHA-256 `7c64b9af9851139794602e3118183ba614fba3891b6de1ca77281f39a9d7c443` | Minimum conditional C5P-H2 adaptation of the log-logit schedule, EDM preconditioning, masked context treatment, and weighted denoising loss; uses one nonredundant mask channel; adds explicit context/target loss accounting, strict one-step canonical axes, codec freezing, training-only latent normalization, future-truth exclusion, sampler-probe latent access, and a fail-closed Azula version check; no physics loss | Initially ported in Paper 0 commit `899b89d0d561dbf6d5c50064805fdbbffc0fc988`; current file SHA-256 `447427a5dd3273689b52d012ebe73fbeeaac15d143a4fda00a06ee065b1e1d4f`; schedule, EDM, mask, loss-accounting, context-clamp, codec-freeze, ensemble-axis, diversity, optimization, and dependency-failure tests |

The copied upstream MIT text is
`src/tcv_diagnostics/models/LOLA_LICENSE.txt`, SHA-256
`6a483108d787c61c7e2306216ecaa2e15f80a0a2e7fb44cb70d200bbbab63605`.
No predecessor training script, optimizer, checkpoint, or configuration
framework was ported. The Phase 3 diffusion mechanics listed above were ported
from the clean upstream LOLA commit rather than from the modified predecessor
checkout.

## External method ledger

External implementations or method-specific code must record repository URL, revision, license, local modifications, and validation tests here before use.

The prospective B2 model uses the public `azula.sample.ABSampler` API from
Azula `0.3.1` (MIT), pinned exactly in the `models` dependency group. No Azula
source is copied into Paper 0. On the existing Rusty model environment, the
installed source hashes are `2a03dc54a763bce8df6d870744bdbcae28f5d5aaf3a01cd4f5eac6eeedaea127`
for `sample.py`, `9979f1d0557a72ab3b33823d5c3ce7d6716e8c3a8a1200c65365416eda020bab`
for `denoise.py`, and
`2538db4eb869fe225a93868ca38b90c66aaf3a929bc5110340671581a731f5e5`
for `noise.py`. Every B2 launcher must verify the installed version and source
hashes before model construction. The local dependency-absent test verifies
that forecast sampling fails closed rather than substituting another sampler;
the Rocky 9 smoke must additionally exercise the real 16-step, order-three
sampler contract before full training is authorized.

### Phase 3 B2 execution code

`src/tcv_diagnostics/b2_training.py` is a new Paper 0 implementation of the
already frozen bounded-smoke optimizer, fixed validation-noise bank,
checkpoint selection, sampler probe, and reload gate. It was not copied from
the predecessor or upstream LOLA training code. Its SHA-256 is
`e6e1743332be263c0d1984b32864a8d328198de9f766c9dbc75cf12949cbfb1a`.
It reuses only the already verified Paper 0 C5P data adapter, codec loader, and
latent-normalization primitives. The module rejects full training before data
or CUDA access and contains no physics-derived loss.

`src/tcv_diagnostics/b2_wandb_tracking.py` and
`paper0/tools/train_b2_ldm.py` are new fail-closed monitoring and authorization
code with SHA-256 values
`c3f87a79ee71dd90f9c3b852c479aed611ee06f9ddf2b1bdb47fcca2477eaaba`
and `5b905350aeab1972c48810036b4cb7d114063b7eed5ddc2aa7e560aa9bd0b175`,
respectively. W&B is a required online monitoring mirror for this run; the
local hash-indexed files remain authoritative and no checkpoint is uploaded.
The entrypoint independently checks the frozen manifest, protocol hash,
codec, OS, accelerator, and installed Azula sources before training.

`src/tcv_diagnostics/resampling.py` calls the public
`scipy.signal.resample` API (SciPy, BSD-3-Clause) with the exact unwindowed
arguments frozen by the protocol, then casts to float32. It does not import or
copy predecessor converter code. The dependency floor is recorded in
`pyproject.toml`; every scientific execution must capture the exact SciPy
version. Synthetic tests verify known Fourier amplitude and phase, zero padded
bandwidth, bitwise equality to the public SciPy call, 81-to-88-to-81 error,
mergeable metric statistics, tail-quantile convention, and all frozen
materiality boundaries.

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

The first parallel submission, job `6891530` from clean commit `b672d69`,
showed that this Slurm release gives an `--exclusive` step all resources not
otherwise constrained. Only shard step `6891530.0` became active; none of the
other 15 started. The job was cancelled after 49 seconds, before any partial
JSON existed. Its no-result record is
`paper0/results/phase2_pressure_closure_6891530.json`. The launcher correction
adds `srun --exact` to each one-CPU exclusive step. No Python statistic, input,
or scientific rule changes.

Job `6891570` from clean commit `347495f` then confirmed that exact CPU
allocation was not sufficient: `scontrol` reported shard zero as
`TRES=cpu=1,mem=64G,node=1`, so its inherited full-allocation memory prevented
the other exclusive steps from launching. It was cancelled after 48 seconds,
before any partial JSON existed. The tracked no-result record is
`paper0/results/phase2_pressure_closure_6891570.json`. The next launcher adds
only `--mem=4G` to each shard step, exactly partitioning the existing 64 GB
allocation across 16 processes.

With the corrected resource request, job `6891571` started all 16 shards from
clean commit `f5d4541`; it was externally preempted after 11:39 before any
partial JSON was complete. No scientific statistic is accepted from that
attempt. Its immutable no-result record is
`paper0/results/phase2_pressure_closure_6891571.json`.

The identical clean revision was then submitted on the non-preemptible `gen`
partition as job `6891583`. All shards completed, the reducer verified every
rank and processor coordinate exactly once, and the job exited `0:0` after
13:32. The full strict result has SHA-256
`db340843ba77fe4d06da2842561ced77ac2814bfd084224baa85b4485ad840c2`.
It accounts for 624 frames and 103,514,112 native cells per field, and confirms
that 85606 was not accessed. `Ni = Ne` and `Pe = Ne*Te` pass throughout;
negative evolved `Pi` produces 3,412 ion-pressure closure discrepancies,
including 1,421 in `y=1..30`. The tracked accepted record is
`paper0/results/phase2_pressure_closure_6891583.json` and the full immutable
root is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_pressure_closure/job_6891583`.

The subsequent state/resampling protocol traces the legacy public-transform
semantics to predecessor commit
`24fdb7df11bad5dc6d7b0436afb938ecd09308e9`, file
`src/data/build_c5_fast.py`, Git blob
`ffbb23f917244e3ed847c2568f038533a6d9df76`, without importing or modifying
that file. The native 85604 source shards have SHA-256
`843f9ae99d08fbcdabce977b53e4f6b49be05641a82a387d100b237224b77777`
and `a17b536856c6b8108c0553c300200e074e41407129e47ef402a4de51882ea1ba`.
Read-only design probes on the five already fixed raw-oracle frames established
that the native arrays equal the raw arrays after float32 cast and that the
public SciPy transform reproduces all five legacy z88 fields bit-for-bit. Those
known structural observations are disclosed in the protocol manifest and will
be rerun as execution gates. No round-trip or resampling-transport metric was
calculated before its thresholds were frozen.

`paper0/tools/audit_85604_resampling.py` and
`paper0/tools/merge_85604_resampling_shards.py` are new Paper 0 code. The
reader uses 17 non-overlapping intervals aligned to the source files' native
40-frame HDF5 chunks, computes all frozen operator paths, and emits only
framewise metrics and additive sufficient statistics. The reducer rejects any
missing, duplicated, reordered, or out-of-interval frame before deriving a
gate or materiality label. Synthetic tests lock interval coverage, source
boundary mapping, held-out path rejection, energy factors, operator scopes,
nonzero component behavior, paired scaling, strict JSON, linear quantiles, and
order-sensitive digest trees. No predecessor audit implementation was copied.

Rocky 9 job `6891664` executed the frozen audit from clean commit
`67abc70763135dcd33f64c0c03f4fbf4b6396575`. It completed all 17 shard steps
and the strict reducer with exit code `0:0` in 1:41. The full 53,299,031-byte
result has SHA-256
`4b903d27d303e7b5db086d4e1ea62856f65cac7aacc3e623ac98bab1706d2781`;
the post-run artifact manifest verified every partial, command, environment,
and final-result digest. Run 85606 was not read.

Every prospective field, transport-round-trip, and selected-frame float32
quantization gate passed. Direct 88-cell total face flow and divergence were
prospectively labeled small, so the released primary evaluator downsamples
each 88-cell member to native 81 before applying `Q_81`. This is a numerical
operator policy, not an emulator, architecture, or physical transport result.
The tracked compact record is
`paper0/results/phase2_resampling_6891664.json`; its SHA-256 is
`2d1ed6e7af5a1559e213590ed6315775400ebdcb1db849cfc826d77ef7d8b4a5`.

The first compacting attempt under commit `4c4c5e2` failed before creating an
output because the helper assumed that the five selected raw-float64 frames
had the same eight temporal-block summaries as the all-frame paths. Commit
`836417c` encodes that scope difference explicitly and adds a regression test.
The immutable job result was not altered or rerun.

### Phase 2 geometry, units, and ensemble semantics

The 85604 grid embeds Hypnotoad revision
`e4a1dff39b80e30aaa05eb6903a8dc72cf4ed832`. Paper 0 reads that official
source from the clean detached evidence checkout
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/hypnotoad-e4a1dff`;
`hypnotoad/core/mesh.py` has SHA-256
`3c4a3d8f5b94ab728650726fbf010af70f63ae6452a83e024460d34ab99336e3`.
The BOUT++ constants file at the already locked executed revision has SHA-256
`4a89ceb00a66799668b1b73d3598e2995d9e171680be0d5ce0d20fe6b33e63b2`.

Rocky 9 job `6891709` ran the new Paper 0 mask, surface-integration, unit, and
member-wise reductions from clean commit
`9dd8780ca2b68b76624aaefa1d8b3638c5c6377c`. The full immutable result is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_geometry_units/job_6891709/geometry_units.json`,
SHA-256
`9a62f47aaa15edba3ca6b17159862b026dbcf03977eb535306a4ed8702dde1cc`.
The tracked copy is
`paper0/results/phase2_geometry_units_6891709.json`. All prospective gates
passed, and run 85606 was not read.

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

### Phase 2 O1 geometry-aware codec transport

The codec-transport extension was prospectively frozen in
`paper0/protocol/PHASE2_O1_TRANSPORT_PROTOCOL.md` and
`paper0/manifests/phase2_o1_transport_85604.json`. It reuses the same
hash-locked predecessor LOLA package and f8/z44 checkpoints as job `6890650`,
without copying or modifying that implementation. New Paper 0 code is limited
to the independently tested state-path assembly, validated transport
reductions, strict result compaction, and figure generation:

| Paper 0 path | Role | Verification |
|---|---|---|
| `src/tcv_diagnostics/codec_transport.py` | Builds frozen geometry masks and accumulates four-path transport comparisons | Synthetic state-path, masks, aggregation, gate-boundary, and strict-JSON tests |
| `paper0/tools/evaluate_codec_transport_oracle.py` | Streams all 624 exposed 85604 frames through each deterministic codec and the released native-81 transport evaluator | Clean-commit launcher; locked data, geometry, LOLA, checkpoint, run, and blind-test gates |
| `paper0/tools/summarize_codec_transport_oracle.py` | Verifies and compacts the immutable full result while retaining all figure-complete surface series | Exact raw digest, scope, job, commit, codec set, and shared-truth assertions |
| `paper0/tools/plot_codec_transport_oracle.py` | Regenerates the three labeled transport figures from the tracked compact record | Figure-file, provenance-text, series-length, and source-path regression tests |
| `cluster/phase2_o1_codec_transport.sbatch` | Rocky 9 H100 launcher | Literal path/hash tests, dirty-state refusal, `bash -n`, and 85606-path rejection |

Rocky 9 job `6891766` completed from clean Paper 0 commit
`47a26e3ad7e7c8c9a216930dbddd3954e1213e60` on an NVIDIA H100. It accessed
only all 624 historically exposed 85604 frames and performed no training. Its
full result is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_transport/job_6891766/o1_codec_transport.json`,
SHA-256
`c8434cfea29fb4fb9bfa3f8e7fb455985aed6885b478513b06b8d6d8214e3df1`.
The tracked compact record is
`paper0/results/phase2_o1_codec_transport_6891766.json`, SHA-256
`140bf3faabb0922edd9108af7d3e00e76c71075caa3a43e5c29760cc043b0a23`.
The full and compact records both state that 85606 was not accessed.

Native/legacy alignment and the `P1/P2` storage-resampling transport gates
pass. The shared `P0/P1` C5T state-adequacy gate formally fails because its
electron strict-face relative L2 is `5.1434788769506797e-8`, above the frozen
numerical-identity threshold `1e-10`; particle error is zero and every other
state-path transport error is below `5.1e-7`. That formal failure is retained
without threshold revision.

For authoritative `P0/R` transport, f8 has strict-face relative L2
`0.288--0.305` and therefore fails the frozen `0.25` local criterion, while
its four integrated-separatrix errors are `0.0268--0.0533` and all eight
temporal blocks pass. z44 has strict-face errors `0.2019--0.2232` and
integrated errors `0.0387--0.0857`, passing the radial-transport subgate and
every temporal block. Neither codec passes complete O1 acceptance because the
prior spectral gate fails for both and the prior cross-field gate also fails
for z44. Their unmatched training histories still forbid attributing the
difference to latent toroidal capacity.

All three O1 transport figure pairs are generated only from the tracked
compact result. Their interpretation and exact regeneration command are in
`paper0/PHASE2_O1_TRANSPORT_READOUT.md`.

### Phase 2 evolved-state inventory and momentum closure

The prospective state audit was frozen in
`paper0/protocol/PHASE2_STATE_COMPLETENESS_PROTOCOL.md` and
`paper0/manifests/phase2_85604_state_completeness.json` before reading
all raw momentum values. `src/tcv_diagnostics/state_completeness.py`,
`paper0/tools/audit_85604_state_completeness.py`, and
`paper0/tools/merge_85604_state_completeness_shards.py` are new Paper 0
code. They implement source-locked Hermes velocity/momentum relations,
mergeable exact counts, deterministic stream digests, strict rank/time/axis
coverage, and strict JSON. No predecessor audit implementation was copied.

CPU-only Rocky 9 job `6891855` executed from clean Paper 0 commit
`4913361b4f1ee5f04f8fd3e95ac9240b3941c9fc` on Rusty worker
`worker5594`. All 16 deterministic rank shards and the strict reducer
completed with exit `0:0` in 28:16. The immutable full result is

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_state_completeness/job_6891855/state_completeness_audit.json
```

with SHA-256
`9fec0426a97fab9e15b0029d80f1f6c6464d0d7e34aac4216ec4a76ceb3bda93`.
The launcher's `artifact_sha256.txt` records the command, environment,
16 partials, and merged-result hashes. The result verifies all 256 rank files,
all 624 saved times, complete `16 x 16` processor-coordinate coverage,
and 103,514,112 native physical values per stream. It explicitly records
`held_out_85606_read=false`.

The result was compacted without metric changes by the prospectively committed
helper at Paper 0 commit
`54d2bba33cf4a5458bc8e61cb794024de0849d7f`. The tracked compact record is
`paper0/results/phase2_state_completeness_6891855.json`, SHA-256
`565a4e27e87d4f5a3e647daf77486020ac627f43ffb5cd30a8daf74b7199cf20`.
Regression tests lock the raw identity, compacting commit, archive coverage,
field finiteness, inactive density floor, exact closure counts, and
non-overreaching decision flags.

Both exact velocity/momentum closures pass throughout 85604. This accepts only
their algebraic equivalence on the saved development run. It does not accept a
model state, a temporal split, potential reconstruction, forecast dynamics,
or held-out generalization. The complete interpretation is
`paper0/PHASE2_STATE_COMPLETENESS_READOUT.md`.

### Phase 2 saved potential-boundary state

The prospective boundary audit was frozen in
`paper0/protocol/PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md` and
`paper0/manifests/phase2_85604_phi_boundary_state.json` before the first
all-frame read of raw radial `phi` guards.
`src/tcv_diagnostics/phi_boundary.py` and
`paper0/tools/audit_85604_phi_boundary_state.py` are new Paper 0 code.
They implement gauge-invariant midpoint departure, exact guard-copy and
toroidal-constancy checks, mergeable continuous summaries, frozen temporal and
spatial reductions, and strict JSON. No predecessor audit code was copied.

CPU-only Rocky 9 job `6891890` executed from clean Paper 0 commit
`cee2264a88ae7a912f8a70a06086137bf16d4e76` on Rusty worker
`worker5594`. It completed with exit `0:0` in 5:17, read only
the 32 declared radial-boundary ranks of 85604, and explicitly records
`held_out_85606_read=false`. The immutable full result is

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_phi_boundary_state/job_6891890/phi_boundary_state_audit.json
```

with SHA-256
`79c67709c921caa1ddf1ea3e4d8f431ce88e220adc70247527c7a8a5e5f637cc`.
The job's artifact inventory records the same digest plus command and
environment hashes. The exact file is tracked unchanged at
`paper0/results/phase2_phi_boundary_state_6891890.json`; a regression
test locks its complete SHA-256, provenance, structural identities, amplitude
summaries, and open materiality flags.

Both source-structural checks pass, while the instantaneous-Neumann state fails
at every saved frame/y location on both sides. This accepts the presence of a
distinct saved compact boundary value only. It does not establish its effect
on interior potential or transport, close the potential/vorticity gate, select
a model state, or authorize held-out access. The complete interpretation is
`paper0/PHASE2_PHI_BOUNDARY_STATE_READOUT.md`.

## Phase 0 execution evidence

At the time of the Phase 0 reproduction, no predecessor code had been ported.
The audit-only legacy reproduction executed the predecessor files in place
after verifying their byte hashes. Job `6890428` used Paper 0 commit
`7e2b5d2`, Rocky Linux 9.8, an NVIDIA H100, and only the legacy 85604
validation region. Its compact result record is
`paper0/results/phase0_legacy_valid_6890428.json`; the full Rusty artifacts
remain under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase0_legacy_valid/job_6890428`.

The first submission, job `6890410`, stopped before model inference because the manifest contained a truncated validation-file hash. Commit `7e2b5d2` corrected the record and added a test that validates every JSON digest before launch.

### Phase 2 shared 85604 model dataset

The conversion is governed by
`paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md` and
`paper0/manifests/phase2_model_dataset_85604.json`. New clean-room code is:

- `src/tcv_diagnostics/model_data.py`;
- `paper0/tools/build_85604_model_dataset_shard.py`;
- `paper0/tools/merge_85604_model_dataset_shards.py`;
- `cluster/phase2_85604_model_dataset.sbatch`.

The code reuses only the already audited public
`scipy.signal.resample` wrapper in `src/tcv_diagnostics/resampling.py`.
It does not import or modify predecessor conversion code. Its inputs are the
two native 85604 Well files, the two historical z88 files used only as a
transform oracle, and the hash-locked `Bphi` extraction record from job
`6893033`. All source hashes are frozen in the manifest and rechecked before
output.

CPU-only Rocky 9 job `6893525` ran from clean commit
`929ed0cb2a861742bcab34101bc60fd53970d40c`. The immutable artifact root is:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525
~~~

The full model-dataset result has SHA-256
`27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18`;
the normalization record has SHA-256
`f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7`;
and the complete artifact index has SHA-256
`6e33bd22615d556714334fff4f06abb53ef49e8711f0712d7332d363ad25cd01`.
The two compact records are tracked byte-for-byte under `paper0/results/`.
All 3,599,761,472 HDF5 bytes remain on Ceph. The run explicitly records
`held_out_85606_read=false` and `training_performed=false`.

### Prospective matched O1/O2 model implementation

The training and evaluation contract is frozen in
`paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md` and
`paper0/manifests/phase2_matched_o1_o2_85604.json` before any model code is
ported or executed.

The codec implementation is a minimum attributed port from
[PolymathicAI/lola](https://github.com/PolymathicAI/lola), upstream commit
`21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`, under its MIT license. The
predecessor project's per-axis padding and per-transition stride repairs are
ported explicitly and independently tested. The implementation is commit
`c0cd66b`; exact source and license hashes appear in the ported-code ledger.
It imports nothing from the dirty predecessor checkout.

No historical codec or forecast checkpoint initializes the matched models.
A deterministic codec implementation, verified data reader, data-only
trainer, CPU portability result, and bounded GPU smoke now exist. No
scientific training result exists under this section yet. On local PyTorch
`2.8.0`, the 11 codec tests and then-current
complete Paper 0 suite passed (`393 passed, 1 skipped`). CPU-only Rocky 9 job
`6893674` ran clean commit `0d633164050281160f37b79f33d4332c536e7970`
on worker `worker5582`, passed the expanded suite (`399 passed, 1 skipped, 29
subtests`) in 23 seconds, requested no GPU, read no simulation data, and
performed no training. The immutable full result remains at
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_cpu_gate/job_6893674/result.json`;
its tracked byte-identical copy has SHA-256
`d46bca572c52b8beab1f0bda18b3b869fe63ec6627b76382c350c52a972ed6f9`.
No reconstruction-quality gate is implied by an implementation test.

Bounded Rocky 9 H100 job `6893713` ran clean commit
`c3d03289dd1e1e1a80eb03579febbb4255022a75` on `workergpu205` and
completed `0:0` in 57 seconds. It used only 16 training and four validation
frames for two epochs at seed 1701. Both the five-channel and six-channel
`dcae_l20` models completed two real optimizer steps and reproduced their
selected-checkpoint validation probe bit-for-bit after a fresh model reload.
Peak allocated CUDA memory was 5,978,285,056 and 5,981,825,024 bytes,
respectively. The tracked summary has SHA-256
`f16bb41c2ce03d9ae3ec0128d759748a594bd0fb8120a38a24825f373cb725db`;
the full checkpoints and artifact index remain under
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_gpu_smoke/job_6893713`.
The two-epoch losses are explicitly non-scientific smoke outputs: no O1
metric or acceptance gate was evaluated, and no 85606 file was accessed.

### Phase 3 B2 latent-diffusion implementation smoke

Bounded Rocky 9 H100 job `6896402` ran clean commit
`d58b4cc261a901b69c772b01270f38a89deb042f` on worker `workergpu217` and
completed with exit `0:0` in 54 seconds. It first passed the complete Rocky 9
suite (`652 passed, 1 skipped, 29 subtests`) and re-verified every frozen code,
codec, and 85604 data hash. The required W&B run completed online.

The job used seed 1701, 16 training targets, four validation targets, two
epochs, and two optimizer steps. It exercised the exact Azula 0.3.1 16-step,
order-three sampler with two members. Both members were finite and distinct in
latent and decoded field space; a fresh same-process reload reproduced both
tensors bit-for-bit. The immutable full artifact root is
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_b2_ldm_gpu_smoke/job_6896402`.
Its artifact index has SHA-256
`b3e2ed075c7fa3dc635cafce00014ac598886ef0cf88aecaa883d75588ac3dbf`.
The byte-identical tracked compact summary has SHA-256
`fa2b29665b4b39b60c9ce24c1e8b067ebc6165322d40bb8de169bf9492ae5360`.

This accepts implementation mechanics only. The result explicitly leaves
scientific quality, full B2 training, probabilistic evaluation, O3,
assimilation, diagnostic ranking, and 85606 access unauthorized. The exact
interpretation is `paper0/PHASE3_B2_SMOKE_READOUT.md`.

### Phase 3 B2 full-training authorization and launcher

After the bounded smoke passed, the complete 85604-only training and
probabilistic-evaluation contract was frozen independently in
`paper0/protocol/PHASE3_B2_FULL_EVALUATION_PROTOCOL.md` and
`paper0/manifests/phase3_b2_full_evaluation_85604.json`. The protocol file has
SHA-256 `8b0345cbff7de588e52c73a40232cdf0a97b5d3f0c18728dc35dadaca9d25490`;
the manifest has SHA-256
`a5567dccb4e536bb33c3e160c2b5bb64b586e2465b20957e2a5d1144ffd2706c`.
They were committed before the full trainer, launcher, or probabilistic
metrics were implemented.

The historical smoke entrypoint remains smoke-only. Full training uses the
separate fail-closed entrypoint `paper0/tools/train_b2_ldm_full.py`, which
requires the exact post-smoke manifest, its exact protocol and evidence
hashes, one of seeds 1701--1703, and the same-seed accepted C5P codec. The
shared optimizer core in `src/tcv_diagnostics/b2_training.py` now exposes
separate smoke and full wrappers and rejects a forged dataclass budget before
opening data or CUDA. Full latent moments use all training frames `[0,432)`
and are marked authoritative; the historical 16-frame smoke moments remain
non-scientific.

`cluster/phase3_b2_ldm_full.sbatch` is a three-task Rocky 9 `gpupreempt` array
with one H100/H200 allocation per seed. The regular `gpuxl/gen` QoS rejected
the one-GPU dry-run with `QOSMinGRES`, whereas the one-GPU preempt QoS accepted
the exact array. Tasks are deliberately non-requeued: a preempted task fails
visibly and a later submission must use a new immutable job/task directory.
Every task verifies the
clean commit, immutable 85604 data, codec, protocol, evidence, source, and
Azula hashes; passes the complete test suite; stages and rehashes all eight
85604 shards; starts a required online W&B run; and enforces exactly 200
epochs and 5,400 optimizer steps. Checkpoints and decoded artifacts remain on
Ceph and compact provenance will be tracked after completion.

Completing this array is not scientific acceptance. Training computes only
the frozen complete-trajectory denoising loss. It neither computes nor
selects on CRPS, field error, spectra, cross-phase, coherence, transport, or
diagnostic performance. It does not authorize O3, assimilation, diagnostic
ranking, control, or any access to 85606.

### Phase 3 B2 completed gate and A016 gate-only correction

Original finalizer job `6897564` ran exact commit
`361f0f27a9ece3b56f529a72c2fcfa19aa0be719` on all three immutable B2
evaluation records. Its matrix has SHA-256
`cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6`
and remains unchanged.

A016 is frozen in
`protocol/PHASE3_B2_EVENT_ELIGIBILITY_AMENDMENT.md` and
`manifests/phase3_b2_event_eligibility_amendment_85604.json`. The original
gate source remains byte-identical with SHA-256
`6bb5d825b30c9c8292cda020d3bec824d9b04198617dc89afafa264daab44ea5`.
The separate implementation is
`src/tcv_diagnostics/b2_acceptance_gate_event_eligibility.py`; it reuses the
original non-event checks and changes only truth-empty chronological event
records to explicit N/A after enforcing count and null-record integrity.

CPU-only Rocky 9 job `6898348` ran clean commit
`c9b136d832b387f4f5596885d2338c75491ab3af`, passed 19 focused tests,
and completed `0:0` in 26 seconds. It performed no training, inference, truth
scoring, forecast mutation, score mutation, or 85606 access. Its amended
matrix has SHA-256
`4f054365d32d3e1725091ba58c8fa014f104e204748217dda482045a6c0df600`.
The architecture remains failed with zero passing seeds. The tracked compact
record is `results/phase3_b2_event_eligibility_amendment_6898348.json`.

### Prospective Phase 3 B3 functional-generative retrofit

The B3 implementation contract was frozen before model code in
`paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md` and
`paper0/manifests/phase3_b3_fgn_85604.json`. B3 is adapted from the official
[cddcam/lola_crps](https://github.com/cddcam/lola_crps) implementation of
Diaconu et al., *Probabilistic Retrofitting of Learned Simulators*, at upstream
commit `7643376c2949717ee5c2c840584689f529ba77a5`. The retained upstream MIT
license is `src/tcv_diagnostics/models/FGN_LICENSE.txt`, SHA-256
`cb4951b3d04c3153a6192951a520d7042c927d1f0574978d503596d6c14e36c9`.

The exact upstream source hashes are machine-locked in the B3 manifest. The
ported concepts are limited to the two-layer global Gaussian-noise embedding,
LayerNorm of that embedding, a separate additive four-vector AdaLN noise
adapter in every ViT block, the small final-adapter initialization, staged
common/new parameter groups, and finite-ensemble fair CRPS. No upstream data
loader, dataset, autoencoder, checkpoint, training result, or evaluation
result is used.

`src/tcv_diagnostics/models/functional_noise.py` is new Paper 0 code. It adapts
the mechanism to the existing 3D C5P latent-increment transition, preserves
all deterministic O2 state-dict names, fails closed unless every deterministic
parent key and shape matches, provides a noise-disabled inherited forward
path for bitwise parent identity, decodes through the frozen C5P codec, and
returns canonical `[batch,member,future_time,channel,x,y,z]` axes. Its fair
CRPS implementation accumulates unordered member pairs without materializing
an infeasible full pairwise 3D tensor.

The initial known-answer tests verify the two-member fair-CRPS formula,
member-permutation invariance, exact deterministic parent identity, strict
load accounting, global rather than token-wise noise, official adapter
initialization, finite noncollapsed members, gradient flow into both common
and new parameters, frozen codec behavior, canonical axes, and exact full
state reload.

`src/tcv_diagnostics/fgn_training.py` is a new fail-closed bounded-smoke
trainer. It reuses the exact parent codec and latent-normalization bytes,
separates common and new optimizer groups by state-dict provenance, generates
and persists the frozen PCG64 validation-noise bank, optimizes decoded-field
fair CRPS, verifies nonzero gradients in both groups, hashes the in-memory
codec state before and after optimization, and provides no callable full-run
trainer. `paper0/tools/train_b3_fgn.py` authorizes only the exact seed-1701
smoke manifest. `src/tcv_diagnostics/fgn_wandb_tracking.py` mirrors separate
accuracy, spread, and fair-CRPS terms without uploading checkpoints.
`cluster/phase3_b3_fgn_gpu_smoke.sbatch` requests one Rocky 9 H100/H200,
refuses reused output paths, verifies all frozen source and evidence hashes,
runs the complete test suite, requires online W&B, stages only 85604, and
enforces the post-run identity, diversity, gradient, codec, reload, and scope
gates.

The complete pre-execution local suite passed with `795 passed, 1 skipped`.
Bounded Rocky 9 H100 job `6898604` then ran clean commit
`fb89828f6837ce0568fe7fc565b931810da68262` on worker `workergpu217` and
completed with exit `0:0` in 58 seconds. It first passed the complete Rocky 9
suite (`795 passed, 1 skipped, 29 subtests passed`) and then performed exactly
two optimizer steps on 16 training and four validation targets from 85604.
The required W&B run completed online.

All mechanical gates passed: the 151 inherited deterministic state tensors
loaded exactly; the 70 new stochastic tensors were the only parent-missing
keys; the noise-disabled child was bitwise identical to the parent before
optimization; common and new parameter groups both received finite nonzero
gradients; the codec remained bitwise unchanged; two fixed-noise members were
finite and distinct; the canonical forecast shape was
`[1,2,1,5,64,32,88]`; and a fresh checkpoint reload reproduced latent and
decoded outputs bit for bit. The immutable full run-result SHA-256 is
`fccb26d5ee22d7bf8e716a3ac483263d0bce6fde32d4bdaadf8a015a6700ffd1`;
the byte-identical tracked compact record is
`paper0/results/phase3_b3_fgn_gpu_smoke_6898604.json`, SHA-256
`dbac54c033917abbfec7e380d96a0c9be93667ae58240b4403400b57c76e2808`.
The exact interpretation is `paper0/PHASE3_B3_SMOKE_READOUT.md`.

This accepts implementation mechanics only. The four-frame, two-epoch loss
is explicitly non-scientific: no B3 scientific metric, 85606 access,
autonomous rollout, assimilation, or diagnostic result exists yet.
