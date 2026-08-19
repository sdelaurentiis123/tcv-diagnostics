# Phase 0 repository and result audit

**Status:** complete

**Audit date:** 2026-08-16

**Current scope:** read-only inspection and 85604-only reproduction

This document records the evidence gathered before any Paper 0 architecture redesign. It separates facts verified directly from historical claims that still require reproduction.

## Scope and rules

- Audit source: predecessor repository and existing Rusty artifacts.
- Shot 85604 may be used for reproduction.
- Shot 85606 is sequestered from new Paper 0 analysis.
- Historical 85606 exposure is documented in `AMENDMENTS.md`.
- No model architecture is modified during this phase.
- The stored toroidal domain has `zperiod = 5`; stored Fourier index `k` therefore maps to full-torus toroidal mode number `n = 5k`.

## Repository state at audit start

### Paper 0 repository

- Local path: `/Users/stanislavdelaurentiis/tcv-diagnostics`
- Branch: `main`
- Initial audit commit: `a404577`
- Remote: `git@github-personal:sdelaurentiis123/tcv-diagnostics.git`
- Worktree at inspection: clean

### Predecessor evidence tree

- Local path: `/Users/stanislavdelaurentiis/tcv-gaot-3d`
- Branch: `probe-conditioning-diagnostics`
- Root commit: `b367d8bd89ab8119f6070f9c0533eb17f5d438cd`
- Root worktree: clean except the pre-existing untracked `.codex/`, which Paper 0 does not modify.
- Rusty path: `/mnt/home/sdelaurentiis/tcv-gaot-3d`
- The Rusty root is an rsync-style snapshot without root `.git` metadata. A root commit cannot be recovered from that copy alone.
- The audited top-level experiment files on Rusty match the local predecessor files byte-for-byte.

The embedded `external/lola` checkout is not tracked by the predecessor root repository. Its Git base is `21a4354b327e6e5ee06da5075ba3bd1dd88c61f1`, but the checkout has many modified and untracked files. Therefore that revision alone does **not** identify the executed implementation. Paper 0 records hashes for every critical executed file and will port only audited code.

## Data and metadata inventory

### Raw 85604 BOUT output

- Representative file: `/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/BOUT.dmp.0.nc`
- Raw per-processor dimensions: `x=8`, `y=6`, `t=624`, `z=81`
- `zperiod = 5`
- `ZMIN = 0.0`, `ZMAX = 0.2`
- Stored domain: one fifth of the torus
- Fourier mapping: `n = 5k`
- `Omega_ci = 95,788,333.03066081 s^-1`
- Stored normalized time step: `300`
- Physical frame cadence: `300 / Omega_ci = 3.131905426352636 microseconds`
- Time range: `285000` through `471900`, 624 frames
- Other normalization scalars: `Bnorm=1.0`, `Nnorm=1e19`, `Tnorm=50.0`, `rho_s0=0.0007224847664314034`, `Cs0=69205.61141651045`

### Existing five-field Well conversion

Train file:

`/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/train/TCV_c5_train.hdf5`

Validation file:

`/mnt/home/sdelaurentiis/ceph/tcv_well/TCV_c5_z88/data/valid/TCV_c5_valid.hdf5`

Verified schema:

- Field order: `Ne`, `Te`, `Ti`, `phi`, `Vi`
- Train field shape: `[1, 500, 64, 32, 88]`
- Validation field shape: `[1, 124, 64, 32, 88]`
- Dtype: `float32`
- Dimensions: `x`, `y`, `z`; `z` is periodic
- The original 81 stored toroidal samples were Fourier-resampled to 88 samples.
- The stored `z` coordinate in the converted file is an index-like resampling coordinate, not a physical toroidal angle.
- Field units are absent from the HDF5 attributes. SI-valued field claims are blocked until an authoritative conversion is defined.

The existing split uses frames 0--499 for train and 500--623 for validation. The last train time is `434700`; the first validation time is `435000`. The files are immediately adjacent and contain **zero guard frames**. This split is acceptable only for reproducing the legacy result. Phase 1 must define new chronological regions with a guard longer than the maximum input-plus-target window.

### Existing preprocessing

- Statistics were stored in the legacy run configuration for the five fields.
- Mean: `[-1.9359, 0.9337, 1.2636, 2.8614, -0.1795]`
- Standard deviation: `[1.4488, 0.5312, 0.4681, 1.2784, 0.9219]`
- `Ne` uses `log_eps`, implemented as the natural logarithm `log(Ne + 1e-6)`, followed by standardization.
- Other fields are standardized without the log transform.
- The training loader applies a random periodic roll on the toroidal axis after temporal windows are constructed.

### Time and conditioning semantics

The legacy diffusion checkpoint does not receive physical simulation time. Its six configured label features are the constant boundary-condition code `[0, 0, 0, 0, 2, 2]`; they are not timestamps. The denoiser's separate time embedding receives diffusion noise-time only. Physical sequence order is represented by the temporal tensor axis, and the 3.131905426352636 microsecond frame interval is implicit because the legacy data use one fixed cadence.

Phase 1 must freeze a deliberate policy:

- always record physical cadence and relative forecast lead;
- condition on `delta_t` or relative time offsets if models are asked to support more than one cadence;
- do not use absolute frame number by default, because one training trajectory makes it a high-risk lookup/leakage feature;
- condition on actual time-varying sources or boundary drivers if such metadata exist, rather than using clock time as their surrogate.

### Latent-cache replication

The f8 cache used to train the legacy diffusion model stores shape `[4, time, 64, 8, 4, 11]` and four identical label rows. These four rows are repeated encodings of the same physical trajectory under random toroidal-roll augmentation. They are symmetry augmentations, not four simulations, four shots, or four independent validation units. The legacy cache builder applied the configured random roll while constructing every split, including validation. Paper 0 must keep validation deterministic and report statistical uncertainty by temporal block and training seed, never by treating cache repeats as independent data.

## Code-path inventory

| Purpose | Legacy path | Audit note |
|---|---|---|
| C5 conversion | `src/data/build_c5_fast.py` | Selects five fields and Fourier-resamples `z: 81 -> 88`. |
| Dataset/preprocessing | `external/lola/lola/data.py` | Natural-log density transform; Well loading and temporal windows. |
| Codec | `external/lola/lola/autoencoder.py` | Encodes/decodes the five jointly modeled fields. |
| Diffusion | `external/lola/lola/diffusion.py` | EDM-style denoiser and sampler; historical spread comes from the sampling path, not a learned variance head. |
| Rollout composition | `external/lola/lola/emulation.py` | Composes five-frame diffusion blocks autoregressively. |
| Assimilation driver | `lola_ext/experiments/da_anchored_rollout.py` | Produces free and assimilated arms in one invocation. Defaults are unsafe for Paper 0: omitted `split` means `test`, and omitted `update` means `var4d`. Every Paper 0 call must set both explicitly. |
| ETKF | `lola_ext/experiments/etkf.py` | Deterministic ensemble-space square-root update. |
| Diagnostic layouts | `lola_ext/experiments/diagnostic_layouts.py` | Contains synthetic and proxy observation operators; several are not instrument-complete forward models. |
| Legacy flux helper | `src/tcv_eval/flux.py` | Requires revalidation before use; see discrepancies below. |

## Checkpoint inventory

### Primary legacy f8 diffusion stack selected for reproduction

Diffusion run:

`/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/dm/idllqvqe_tcv_c5_f8c64_vit_small`

- Target: `state_best.pth`
- Checkpoint SHA-256: `5d8e14807b62e81e928628c8413c2563d00884bf339f50e9b221bd758c02e759`
- Config SHA-256: `b7ce08657d142cf9d7139798dff0aec847697dd4b198e41d79b6e03ea85a35a5`
- Trajectory length: 5 frames
- Field count: 5
- Denoiser: 16-block ViT, hidden width 512, four attention heads
- Historical training budget: 300 epochs, 2048 examples per epoch, batch size 16

Codec run:

`/mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64`

- Target: `state.pth`
- Checkpoint SHA-256: `9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e`
- Config SHA-256: `66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776`
- Five fields are compressed jointly from `[64, 32, 88]` to a latent spatial grid `[8, 4, 11]` with 64 latent channels.

### Other historical families

The exact paths, hashes, training histories, and comparison blockers are recorded in `paper0/manifests/legacy_model_families.json`.

- The deterministic Walrus/ViT3D artifact used four C4 fields (`Ne`, `Te`, `Vort`, `phi`), `log10(Ne)`, six context frames, sample-wise reversible normalization, one-step delta prediction, and MAE. Its selected epoch-236 checkpoint has validation MAE `0.0220873`, but that number is not comparable to the five-field standardized LOLA errors. The nearest saved rollout artifact is epoch 230 rather than the selected checkpoint; it contains one aggregate batch, its full-field freeze ratio falls from `0.8852` to `0.1427`, and its reported standard deviation is `NaN`.
- The z22 diffusion artifact is a continuation using a separately trained z22 codec.
- The z44 codec is itself a non-strict continuation from z22 with an added multiscale-increment term, and its diffusion model is another continuation. It is not an isolated latent-resolution ablation.
- The f8-short artifact is also a continuation, so its total exposure is not a clean matched-budget control.
- The rollout-CRPS model is a 40-epoch fine-tune of the primary f8 checkpoint, not a from-scratch retraining. Fair CRPS, absolute anchored error, and ETKF gain select epochs 7, 4, and 18 respectively. The frozen history's best-fair-CRPS checkpoint (`0.143996`) does not beat the parent (`0.143918`), and no joint-dependence loss was active. Its requested latent band edges `[1, 6, 16]` collapse to the single effective split `[1]` on the f8 latent grid, so it did not separately supervise the physical `n ≈ 20–35` range.

These artifacts are retained only as historical evidence and initialization candidates. Paper 0 will rebuild the actual baseline comparison using one shared data protocol, field set, preprocessing definition, training budget, and validation rule.

## Forecast reproduction

Job `6890428` completed on Rocky Linux 9.8 with one NVIDIA H100 in 190 seconds. It used Paper 0 commit `7e2b5d268b2d5176a5b26cba9ac129e3caf317b5`, verified every source/config/checkpoint hash, and accessed only the legacy 85604 validation region.

- Start frame: 24
- Rollout tensor: 48 frames including the lead-zero state; maximum evaluated lead: 47 cadence intervals = `147.1996 microseconds`
- Ensemble: 64
- Sampler: Adams-Bashforth, 16 steps
- Seed: 0
- Codec reconstruction RMSE floor: `0.0349768`
- Mean free-rollout RMSE: `0.2049902`
- Final free-rollout RMSE: `0.2592772`
- Mean free ensemble spread: `0.1054020`

The historical free-RMSE reference was `0.2011699`. The fresh value differs by `0.0038203`, outside the frozen `0.001` numerical tolerance. The reproduction is therefore recorded as a numerical discrepancy; its cause is not established and no tuning or repeated sampling was used to erase it.

## Assimilation reproduction

The same job produced the free and ETKF arms with paired random numbers:

- Synthetic layout: `iter`
- Observation operator: 69 direct-state point samples on toroidal plane `z=0`
- Channels: 54 `Ne` samples on an idealized midplane radial line and 15 `phi` samples on idealized divertor-leg boundary positions
- This is a synthetic direct-state layout, **not** a realistic ITER diagnostic forward model.
- ETKF cadence: every 4 frames
- Observation standard deviation: 0.05 in standardized model space
- Inflation: 1.0
- Mean anchored RMSE: `0.1800128`
- Final anchored RMSE: `0.2133195`
- Mean anchored ensemble spread: `0.0782927`
- Absolute mean-RMSE reduction: `0.0249774`
- Relative mean-RMSE reduction: `12.1847%`

The old control improves aggregate error, but its effect is not uniform across fields. Mean global RMSE improves for `Te` (`0.16878 -> 0.14847`) and especially `phi` (`0.28140 -> 0.18874`), is essentially unchanged for `Vi` (`0.14622 -> 0.14599`), and worsens for `Ne` (`0.14728 -> 0.15883`) and `Ti` (`0.22901 -> 0.23725`). ETKF also contracts the average ensemble spread. This is a useful forecast/assimilation smoke test, not a physical diagnostic-ranking result.

The historical anchored-RMSE reference was `0.1770674`; the fresh value differs by `0.0029455`, also outside the frozen tolerance. Full metrics and artifact hashes are in `paper0/results/phase0_legacy_valid_6890428.json`.

## Discrepancies and known failure modes

1. **No legacy split guard.** Existing train and validation files are adjacent. Paper 0 will not use this boundary for selection.
2. **Missing field units.** The converted HDF5 files do not declare units. Physics diagnostics cannot silently assume SI values.
3. **Toroidal labels were previously easy to misread.** `zperiod=5` means `n=5k`; plots and tests must label both quantities where ambiguity is possible.
4. **No executable root commit on Rusty.** The Rusty source snapshot lacks root Git metadata. Critical file hashes are the execution identity.
5. **Dirty embedded LOLA lineage.** Base revision `21a4354...` is insufficient because executed LOLA files include local changes.
6. **Dangerous legacy defaults.** The assimilation script defaults to shot-85606 `test` data and to `var4d`. Locked launchers must explicitly specify `split=valid` and `update=etkf`.
7. **Synthetic-layout naming overstates realism.** The legacy `iter` layout samples simulated state variables directly and should be described only as an idealized synthetic configuration.
8. **Density-transform mismatch in a legacy transport helper.** `src/tcv_eval/flux.py` defaults to reconstructing density with `10**Ne`, while the C5 data pipeline uses natural log and `exp`. Every historical flux artifact must be traced to its caller; the helper is not authoritative for Paper 0 until known-answer tests pass.
9. **Historical 85606 exposure.** The final simulation was viewed in exploratory work. New use remains prospectively sequestered under amendment A001.
10. **Marginal calibration is not transport validation.** Historical CRPS/spread improvements cannot establish cross-phase or nonlinear-flux fidelity without member-wise physics evaluation.
11. **Physical geometry does not imply a physical channel response.** Corrected target, reflectometry, and GPI supports exist, but the prior ranked channels are largely direct-state proxies. The acceptance ledger in `protocol/OBSERVATION_OPERATORS.md` prevents those proxies from entering final diagnostic claims.
12. **The six legacy labels are not physical time.** They are constant boundary-condition codes. Cadence is implicit, and absolute frame time is absent.
13. **Four cache rows are augmentation copies.** Random toroidal rolls of one run must never be counted as independent physical trajectories; validation augmentation also needs removal in the new protocol.
14. **The first locked launch failed before inference due to a transcription error.** Job `6890410` correctly stopped at the integrity gate because the expected validation hash was only 61 characters. Commit `7e2b5d2` restored the leading `eed` and added a regression test requiring complete 64-character hashes and manifest/launcher agreement.
15. **The fresh legacy result has small numerical drift.** Job `6890428` preserves the qualitative and absolute ETKF gain but does not reproduce the historical free and anchored RMSE within the predeclared `0.001` tolerance. The cause remains unknown.
16. **The deterministic artifact is not an apples-to-apples baseline.** It uses C4 rather than C5, a different density logarithm, different normalization, six context frames, and a different validation objective. It must be retrained under the common protocol.
17. **The historical representation sweep is confounded.** z22, z44, and f8-short have different parentage, codec losses, and training exposure. No latent-resolution conclusion is accepted from those artifacts.
18. **The historical CRPS fine-tune did not establish a calibration solution.** Its best-fair-CRPS score is slightly worse than the parent in the frozen history; separate objectives select separate epochs; and it does not validate cross-field transport statistics.
19. **The first exact-BOUT build attempt stopped at its provenance gate.** CPU
    job `6890717` read no shot data and compiled nothing. The launcher contained
    a mistyped `fmt` submodule revision, so it exited with code 2 before the
    dependency build. The immutable failed-attempt log remains under
    `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890717`;
    the corrected launcher uses the gitlink actually recorded by BOUT++ commit
    `7d28d67`, namely `407c905e45ad75fc29bf0f9bb7c5c2fd3475976f`.
20. **The second exact-BOUT build attempt stopped during dependency
    configuration.** CPU job `6890720` passed every repository revision and
    critical-file hash check, then netCDF C++ configuration exited because
    CMake could not discover HDF5. No BOUT++ source was compiled and no shot
    data were read. The immutable attempt remains under
    `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/external/builds/bout_7d28d67_job_6890720`;
    the launcher now loads the Rocky 9 `hdf5/1.14.5` module explicitly.
21. **The exact BOUT++ dependency compiled reproducibly but failed runtime ABI
    validation.** CPU job
    `6890722` completed with exit code `0:0` from clean Paper 0 commit
    `e298337918582293b682cc3c0465175634f29da3`. The installed BOUT++ reports
    version `5.2.1`, and `libbout++.so.5.2.0` has SHA-256
    `fa11368c90e5c392b20290910d5856b39c122154cbcf3b814a1e98ed65f9a6ce`.
    It read no shot data. Runtime job `6890751` then aborted before geometry
    read or operator evaluation: netCDF-C was built against HDF5 `1.12.3`, but
    this install also linked HDF5 `1.14.5`. The install is therefore
    superseded, not accepted, and the shifted derivative and full Hermes
    transport gates remain closed.
22. **The HDF5 mismatch is corrected without bypassing safety checks.** The
    Rocky 9 `netcdf-c/4.9.2` module resolves `libhdf5.so.200` from
    `hdf5/1.12.3`. The rebuild launcher now loads that exact ABI rather than
    setting `HDF5_DISABLE_VERSION_CHECK`; it writes a new immutable job
    directory and must pass runtime validation before becoming the oracle
    dependency.
23. **The corrected dependency has one consistent HDF5 ABI.** Build job
    `6890766` completed from clean commit `b70ec7e`. Dynamic-link inspection
    resolves `libhdf5.so.200` and `libhdf5_hl.so.200` only from HDF5 `1.12.3`,
    matching `netcdf-c/4.9.2`; no HDF5 `1.14` library appears. This clears the
    ABI blocker but not the shifted-derivative or transport gates. The exact
    numerical comparison must run unchanged against this install.
24. **The diverted mesh requires a topology-compatible MPI layout.** Runtime
    job `6890782` opened the ABI-clean geometry, then BOUT++ rejected the
    one-rank layout because an 8-cell leg region is not divisible by
    `MYSUB=32`. No derivative was evaluated. The corrected four-rank layout is
    forced to `NXPE=1`, yielding `NYPE=4` and `MYSUB=8`; rank-local outputs are
    reassembled only after verifying `NXPE`, `NYPE`, `MYSUB`, `PE_XIND`, and
    `PE_YIND`.
25. **The four-rank harness reached output but failed its final input lint.**
    Job `6890792` evaluated the manufactured fields and BOUT++ `DDY`, then
    wrote four complete rank files. Its subsequent `checkForUnusedOptions`
    treated the four FieldFactory fallback expressions as unused because
    `mesh->get` did not mark their `Options` nodes. The output is not accepted
    or compared. The fix marks exactly those four nodes conditionally used and
    retains global unused-option validation.
26. **A file-backed mesh does not inherit the synthetic test's expression
    fallback.** Job `6890796` made BOUT++'s warning visible: `mesh->get` could
    not read any `input_*` variable from the geometry file and substituted
    zero. Those rank outputs are invalid even though `DDY` executed. The
    corrected driver explicitly constructs each tracked expression with
    BOUT++ `FieldFactory`, and the comparator independently rejects a constant
    zero fallback.
27. **`ShiftAngle` is topology-scoped.** The 85604 grid contains finite values
    only through `grid_x=17` and NaN for `grid_x>=18`, exactly where no inner
    core twist connection exists. After the model crop, only `x<16` uses the
    branch shift. Paper 0 now requires those 16 values to be finite and ignores
    non-finite outer values only because the topology proves they are unused.
28. **The shifted-`DDY` transcription matches compiled BOUT++.** Corrected
    four-rank job `6891059` completed with exit code `0:0` from clean commit
    `0223035`. All four manufactured inputs and every frozen topology region
    passed without a non-finite value. The worst absolute discrepancy was
    `3.025468764406014e-12`, well inside the prospectively frozen tolerance.
    This closes only validation-ladder item 4: the Fromm-reconstructed `xy`
    face flow, total conservative transport, native-frame comparison,
    resampling sensitivity, geometry masks, units, and member-wise ensemble
    semantics remain open.
29. **The shifted-`xy` radial face term matches the locked Hermes source.**
    Four-rank Rocky 9 job `6891343` completed with exit code `0:0` from clean
    commit `ee2b04f`. Velocity, selected Fromm state, positivity clipping, and
    face flow passed in four manufactured cases and every frozen topology
    region; the worst continuous absolute discrepancy was
    `2.5619506516250112e-12`, and all clipping decisions matched. This closes
    validation-ladder item 5 only. Combined-flow conservation, native plasma
    frames, resampling, geometry integrations, SI units, and member-wise
    ensemble semantics remain open.
30. **The combined radial-flow transcription conserves exactly as the locked
    Hermes source does.** Four-rank Rocky 9 job `6891373` completed with exit
    code `0:0` from clean commit `b6926ca`. The `xz`, shifted-`xy`, and summed
    face flows and their finite-volume divergence passed every frozen case and
    topology region. Native `dz` and component addition were exact; the worst
    face discrepancy was `2.5619506516250112e-12`; and the maximum
    volume-weighted conservation residual was `1.1368683772161603e-13`. The
    largest divergence absolute discrepancy, `3.073364496231079e-08`, occurred
    at reference scale `3.6801142065100316e6` and is
    `8.351274780533639e-15` relative. This closes validation-ladder item 6
    only. The partial API is not yet a particle- or heat-transport metric;
    native 85604 plasma frames, resampling, masks/orientation, units, and
    member-wise ensemble semantics remain open, while 85606 remains untouched.
31. **The native real-state operator passed, but the exact five-channel state
    closure did not.** Four-rank Rocky 9 job `6891379` ran from clean commit
    `7d5522c`. All 15 combinations of five value-independent 85604 frames and
    direct `Ne`, `Pe`, and `Pi` inputs passed every frozen face-flow,
    divergence, conservation, and topology comparison. The worst face error
    was `6.341038805146582e-13`; the worst divergence error was
    `6.941263563930988e-09` at scale `79885.99666953899`; and the worst
    conservation residual was `3.552713678800501e-15`. The overall job
    correctly exited nonzero because its separate full-domain closure gate
    found one frame-312 cell with negative evolved `Pi` and approximately zero
    floor-derived `Ti`. The source-backed cause is Hermes' deliberate use of
    `floor(P, 0)` to derive temperature while retaining the evolved pressure.
    No threshold, frame, or region was changed. This accepts the partial
    operator on selected real states but blocks an implicit claim that
    `[Ne, Te, Ti, phi, Vi]` exactly reconstructs the evolved Hermes state.
32. **The first all-frame closure audit produced no result because serial
    native-rank I/O cannot fit the short-job budget.** Rocky 9 job `6891417`
    ran from clean commit `39bfb22` and passed every source, archive, OS, and
    dirty-state gate. A read-only live descriptor check at about 37 minutes
    showed it still processing `BOUT.dmp.42.nc` of 256; it was cancelled at
    38:46 rather than allowed to consume the remaining allocation without any
    chance of completion. Peak RSS was 6,619,428 KiB. No strict result JSON was
    written, no partial statistic is accepted, and 85606 was not read. The
    immutable logs remain under the job-specific directory. The correction is
    an execution-only parallelization over disjoint rank shards; it does not
    change a field, cell, scope, formula, tolerance, or decision rule.
33. **The first parallel launch exposed a Slurm step-allocation setting before
    producing any shard.** Job `6891530`, from clean commit `b672d69`, passed
    all provenance gates, but `srun --exclusive` assigned the first shard step
    the allocation's full CPU set because `--exact` was absent. A read-only
    step query showed only `6891530.0` active instead of 16. The job was
    cancelled after 49 seconds; no partial JSON or scientific result existed,
    and 85606 was not read. The correction adds only `srun --exact`, forcing
    each exclusive shard step to consume precisely its requested one CPU.
34. **Exact CPU allocation still required explicit step memory.** Job
    `6891570`, from clean commit `347495f`, confirmed that `--exact` limited
    shard zero to one CPU, but `scontrol` reported its step TRES as
    `cpu=1,mem=64G,node=1`. That full-memory inheritance again blocked the
    other 15 exclusive steps. The job was cancelled after 48 seconds with no
    partial JSON or scientific result and no 85606 access. The correction adds
    `--mem=4G` to each shard step, partitioning the already requested 64 GB
    allocation without changing code or scientific settings.
35. **The corrected audit survived only after moving off the preemptible
    partition.** Job `6891571`, from clean commit `f5d4541`, started all 16
    one-CPU, 4-GB shards and remained below the resource limits. Slurm
    preempted it at 11:39 before any shard finished. No partial JSON or
    scientific result exists. Resubmitting the identical commit and command on
    `gen` changes execution reliability only, not the frozen audit.
36. **Negative evolved ion pressure reaches the transport interior.** Job
    `6891583` completed all 16 shards and strict merge over every one of the
    624 native 85604 frames. All six fields were finite; `Ni = Ne` and
    `Pe = Ne*Te` had zero point discrepancies. Direct `Pi` was negative at
    3,412 of 103,514,112 cells, and every one of those points failed both
    temperature-to-ion-pressure closures. Crucially, 1,421 points across 47
    frames lie in the predeclared `y=1..30` interior; the largest interior
    miss is `0.00302343566`. Thus the legacy temperature state does not exactly
    reproduce pressure for the accepted radial operator scope. Paper 0 must
    forecast evolved ion pressure or declare and validate a floor policy
    before transport scoring. No automatic channel change was made, and 85606
    remained untouched.
37. **The next state and grid decision is frozen before transport-sensitivity
    evaluation.** `C5T=[Ne,Te,Ti,phi,Vi]` remains the historical comparison,
    while `C5P=[Ne,Pe,Pi,phi,Vi]` is the direct-pressure candidate. The primary
    proposed score for an 88-cell forecast is now explicit: downsample each
    member to native 81 and apply the validated native operator. Direct
    88-cell operator values are a separately reported noncommutation
    sensitivity. Read-only source tracing had already verified exact selected-
    frame float32 provenance and legacy SciPy-resampling reproduction; those
    facts are disclosed rather than presented as unseen results. No round-trip
    field or transport metric was inspected before the gates in
    `PHASE2_STATE_RESAMPLING_PROTOCOL.md` were committed, and 85606 remains
    prohibited.
38. **The resampling primitive and comparison reductions pass synthetic known
    answers before shot-level execution.** The new module is a narrow wrapper
    around the public unwindowed SciPy Fourier resampler, not copied legacy
    code. Tests preserve known `k=7` and `k=40` modes under `81->88`, verify
    zero padding in `k=41..44`, bound the float32 round trip, and prove that
    disjoint sufficient-statistic merges reproduce a single pass. Paired
    relative L2, bias, RMS ratio, correlation, weighted sign error, profile,
    linear-quantile tails, and materiality interval boundaries all have known
    answers. No 85604 round-trip or transport-sensitivity value was read by
    these unit tests.
39. **The shot-level resampling audit is shard-complete by construction before
    it is executable.** Seventeen half-open intervals align with the two
    source files' 40-frame HDF5 chunks and concatenate exactly to `0..623`.
    Each shard records framewise field round trips, native versus round-trip
    transport, aligned native versus direct-88 transport, and the five-frame
    raw-float64 quantization ladder. The reducer requires every interval and
    frame exactly once before it can calculate a gate or label. Synthetic
    transport tests exercise the exact `61 x 30` face and `60 x 30`
    divergence scopes with nonzero `xz` and shifted-`xy` terms. No real
    round-trip or resampling-transport metric has yet been evaluated.
40. **The native round trip passes, while direct 88-cell transport is a small
    numerical sensitivity.** Rocky 9 job `6891664` completed all 17 shards and
    strict merge from clean commit `67abc70`. All five `C5P` fields have a
    maximum per-frame `81->88->81` relative L2 below `1.61e-7`. Primary total
    face flows have aggregate round-trip error below `4.92e-6`, and their
    divergences remain below `2.45e-5`; every frozen aggregate, frame-p99, and
    five-frame float-quantization gate passes. Applying the nonlinear operator
    directly at 88 cells changes total face flow by about 1.6% and divergence
    by 3.5--3.8%, all prospectively labeled small. Paper 0 therefore evaluates
    every 88-cell ensemble member by downsampling to native 81 before applying
    `Q_81`. This closes only the resampling rung: geometry-region integrations,
    orientation, units, ensemble semantics, codec transport, dynamics, and
    85606 remain open.
41. **The geometry, sign, SI-unit, and ensemble-ordering ladder passes.** Rocky
    9 job `6891709`, from clean commit `9dd8780`, verified the exact geometry,
    Hypnotoad, BOUT++, and Hermes source hashes and passed all 19 frozen gates.
    The strict wall-interior operator cells partition exactly into 256
    confined-edge, 219 private-flux, and 1,394 SOL cells. The exact confined
    separatrix is local face `15->16`, `y=8..23`; all radial `psi` differences
    are positive, so positive `+x` is outward on that surface. Unit scales
    reproduce source metadata to `1e-14`. A real nonlinear face-operator test
    gives mean member-wise transport `10.8470` but zero transport from the
    ensemble-mean fields, proving the required evaluation order. This releases
    the evaluator, not codec, rollout, assimilation, or 85606 claims.
42. **Codec compression, not state conversion or resampling, dominates the O1
    radial-transport error.** Rocky 9 job `6891766`, from clean commit
    `47a26e3`, deterministically evaluated both historical codecs on all 624
    exposed 85604 frames and read no 85606 field. Legacy/native input
    alignment is below `1.6e-7` per frame; the strict-face `P1/P2`
    resampling errors are below `5e-6`; and every `P0/P1` state-path error is
    below `5.1e-7`. The electron state-identity gate still records a formal
    failure because its frozen `1e-10` tolerance is tighter than the observed
    `5.14e-8` numerical residue. f8 reconstructs the integrated separatrix
    particle and ExB internal-energy series at `2.7--5.3%` relative L2 in all
    eight blocks, but fails the local-face gate at `28.8--30.5%`. z44 passes
    that transport subgate with `20.2--22.3%` local error, yet has worse
    integrated errors than f8 and retains the prior spectral/cross-field
    failures from an unmatched training lineage. Thus neither historical codec
    passes complete O1 acceptance, no learning gate reopens, and 85606 remains
    sequestered.
43. **The exact source-state audit narrows the partial-observation problem.**
    The hash-locked 85604 input and Hermes revision show that the volumetric
    solver state is `[Ne, Pe, Pi, NVe, NVi, Vort]`; `Te`, `Ti`, `Ve`, `Vi`,
    and interior `phi` are derived. Potential is obtained by an elliptic
    inversion and its radial guard values retain a one-microsecond relaxation
    memory, whereas the saved cadence is 3.132 microseconds and the converted
    grid strips those guards. C5 therefore omits direct electron momentum and
    exact boundary state; adding an absolute frame index would not repair that
    omission. The design memo records three candidate state policies and a
    deterministic closure ladder without changing a split, manifest, model,
    threshold, or 85606 lock.
44. **The all-frame momentum audit confirms representation equivalence but
    not historical-state completeness.** CPU-only Rocky 9 job `6891855`
    completed from clean commit
    `4913361b4f1ee5f04f8fd3e95ac9240b3941c9fc`. All 256 rank files,
    624 saved times, 11 metadata fields, and 103,514,112 physical values per
    streamed field passed the frozen structural checks. Every value in
    `Ne,Pe,Pi,NVe,NVi,Vort,Ve,Vi` is finite, and `Ne` never reaches
    the `1e-7` density floor. Both exact source identities,
    `NVe=(1/1836)*softFloor(Ne)*Ve` and
    `NVi=2*softFloor(Ne)*Vi`, pass every frame with zero discrepant
    points; full-domain relative L2 errors are `5.185e-16` and
    `2.951e-16`. Therefore density-plus-velocity and
    density-plus-momentum are algebraically equivalent representations for
    this output. Historical C5 remains incomplete because it contains neither
    `Ve` nor `NVe`, and the independent `phi/Vort` boundary
    gate remains open. No channels were changed, no model was trained, and
    85606 was not read.
45. **The saved radial-potential guards contain a distinct compact boundary
    state.** CPU-only Rocky 9 job `6891890` completed from clean commit
    `cee2264a88ae7a912f8a70a06086137bf16d4e76`. It verified all 256
    rank filenames and read only the 32 prospectively declared 85604 radial
    boundary ranks. Both sides have zero non-finite values, zero outer-guard
    copy discrepancies, and zero toroidal-midpoint constancy discrepancies.
    The midpoint differs from the instantaneous target at all 19,968
    frame/y locations per side and in every frozen temporal block. Departure
    RMS is `1.07261 V` inner and `0.512986 V` outer; maximum
    absolute departure is `8.11711 V` and `1.99169 V`. Thus
    guard-stripped evolved volumes are not the exact saved discrete state.
    These amplitudes do not establish interior materiality: the paired exact
    elliptic solve and potential/vorticity forward closure remain required.
    No state was changed, no model was trained, and 85606 was not read.
46. **The first potential-oracle launches produced no scientific result and
    are retained as execution failures.** Rocky 9 job `6892220` was preempted
    during read-only extraction. The first nonpreempting submission requested
    an unauthorized QoS and was rejected before Slurm created a job. Job
    `6892235`, from clean commit `aa0ea3c`, completed the canonical extraction
    and every provenance check but failed compilation because the driver used
    the nonexistent public member `mesh->NYPE` rather than this BOUT++
    revision's `mesh->getNYPE()` accessor. Commit `47737c7` changes only that
    accessor, its hash lock, and a regression assertion; all 246 tests and a
    separate Rocky 9 compile/link smoke check passed before relaunch. No
    failed attempt was overwritten, no result gate was changed, and 85606 was
    not read.
47. **The first completed potential replay exposes a raw-versus-runtime
    pressure error in the frozen equation contract.** CPU-only Rocky 9 job
    `6892446`, from clean commit `47737c7`, verified every locked input and
    source hash, extracted the five frozen 85604 frames, compiled, linked, and
    ran the exact BOUT++ cyclic solver. Every volume and radial-boundary echo
    is bitwise exact. Frames `0`, `156`, `467`, and `623` reproduce stored
    `phi` to maximum absolute error below `2.75e-13`; frame `312` also does so
    over the complete guard-independent `y=1..30` transport interior. The
    unchanged full-domain gate nevertheless fails at exactly one point,
    `(x,y,z)=(6,31,73)`: replay minus stored potential is
    `+5.7995129900123565e-05`, while raw evolved `Pi` there is
    `-5.799512988032478e-05`. The pre-run protocol incorrectly treated raw
    evolved `Pi` as the pressure consumed by vorticity. Locked
    `EvolvePressure` instead publishes `floor(P,0)` as runtime species
    pressure before `Vorticity::calculatePihat` reads it. The comparator
    correctly blocks every paired-boundary effect and exits nonzero. This
    failed result is evidence for a separately frozen source-contract
    correction, not permission to relax the tolerance or reinterpret the
    counterfactual. No model was trained, no state was changed, and 85606
    remained untouched.
48. **The source-corrected potential replay passes and bounds the retained-
    boundary sensitivity on the five frozen frames.** CPU-only Rocky 9 job
    `6892641`, from clean commit `df7fa7d`, first reproduced Hermes'
    `EvolvePressure` runtime transformation and only then reran the unchanged
    potential gate. The single negative raw-`Pi` point at frame `312` maps to
    zero runtime pressure exactly; runtime `Pe` and `Pi` are bitwise exact,
    `Pi_hat` differs by at most `4.441e-16`, and all five stored potentials are
    reconstructed to maximum absolute error below `2.82e-13`. The now-
    interpretable paired comparison changes normalized potential by
    `0.004268` relative L2 overall (`0.6605 V` RMS and `3.521 V` maximum);
    private flux is the most sensitive named region at `0.03199` relative L2.
    The strict-face particle-transport change is much smaller: `2.3425e-4`
    relative L2, with `0.1471%` facewise sign disagreement. Across the five
    confined-separatrix wedges, the instantaneous-minus-retained particle
    transport ranges from `-0.1019%` to `+0.1672%` of the retained value.
    These measurements establish an exact source-matched replay and indicate
    a small sampled transport sensitivity; no post-hoc materiality label is
    assigned, and five selected frames do not establish all-frame stability.
    No model was trained, no state was changed, and 85606 remained untouched.
49. **Stored potential and vorticity close bidirectionally under the executed
    source discretization on the five frozen frames.** CPU-only Rocky 9 job
    `6892764`, from clean commit `ab1a5e8`, independently applied the
    BOUT++ cyclic matrix represented by `Laplacian::tridagCoefs` and the public
    `rfft/irfft` path to `u=phi+Pi_hat`, then formed
    `Vort=(2/Bxy^2)L_C(u)`. Input echoes, runtime-pressure reproduction, the
    constant null test, gauge invariance, and a manufactured `k=0+k=3`
    forward/inverse round trip all pass before source values are scored. Over
    all `829,440` physical values, forward versus stored vorticity has pooled
    relative L2 `6.363e-13`, RMS error `2.499e-14`, and maximum absolute error
    `4.738e-13`, more than three orders of magnitude below the frozen
    scale-aware tolerance. Every named geometry region and all native Fourier
    indices `k=0..40` (`n=5k`) are consistent. Together with job `6892641`,
    this validates the retained-boundary, runtime-pressure potential/vorticity
    relation in both directions on the selected frames. It does not establish
    all-frame stability, choose `S6+Bphi` versus a pragmatic observed state,
    authorize training, or read 85606.
50. **The first all-frame potential/vorticity launch reached no scientific
    gate because its canonical chunk layout was incompatible with Ceph.**
    CPU-only Rocky 9 job `6892955`, from clean commit `ef1ca75`, passed every
    repository, source, archive, ABI, OS, and dirty-state check. The extractor
    then wrote each local 78-frame assignment as 78 separate 5-KiB chunks.
    Read-only rate observations showed only 52,202,685 bytes written after
    4:47, projecting far beyond the frozen one-hour cap. The job was cancelled
    at 7:31 with no completed extraction, compilation, shard replay, result
    JSON, accepted statistic, or 85606 access. All partial files and logs are
    retained under its unique job directory. The correction changes only the
    canonical NetCDF chunk from `[1,4,2,81]` to the exact assignment slab
    `[78,4,2,81]`; it does not alter a canonical value, field, frame, equation,
    tolerance, source lock, sequential replay, memory rule, or decision gate.
51. **Matching the canonical write slab exposed an independent raw-HDF5 read-
    order bottleneck before any scientific gate.** CPU-only Rocky 9 job
    `6892978`, from clean commit `8bd9eb2`, passed all provenance gates and
    created eight canonical files with the corrected chunk shape. It then read
    1,128,539,772 bytes by 3:50 without completing the first canonical field
    block. The cause is the raw archive's one-frame HDF5 allocation: requesting
    all 624 scalar times and then full temporal slabs field by field makes
    nonsequential, full-file-scale reads. The job was cancelled at 4:23 with
    no completed extraction, compilation, shard replay, scientific JSON,
    accepted statistic, or 85606 access; its unique directory is retained.
    The next execution-only correction reads `Ne,Pe,Pi,Vort,phi,t` together
    per frame in raw allocation order, buffering only the current 78-frame
    local-rank slab. It preserves the one-open traversal, every timestamp and
    value check, canonical content, memory bound, equation, tolerance, replay
    order, and decision rule.
52. **Native-order semantics alone do not overcome small-chunk Ceph latency.**
    CPU-only Rocky 9 job `6893017`, from clean commit `8b062c0`, passed all
    provenance gates and read the required variables per frame in verified
    on-disk order. At 1:26 it had issued 357,392,824 bytes of reads without
    completing raw rank zero or writing one canonical volume chunk. The job
    was cancelled at 2:07 with no completed extraction, compile, replay,
    scientific JSON, accepted statistic, or 85606 access; its partial files
    remain immutable. The next execution-only correction sequentially stages
    one raw rank file at a time into a unique node-local job directory, runs
    the unchanged time-major semantic read there, and removes only that
    temporary staged copy before proceeding. This keeps one raw-archive byte
    traversal, one simultaneous 818-MB staged file, exact value/timestamp
    checks, bounded memory, and every frozen scientific rule unchanged.
53. **The all-frame source-matched potential/vorticity closure passes with a
    large numerical margin.** CPU-only Rocky 9 job `6893033`, from clean
    commit `d3c7323`, completed in 18:44 after staging each of the 256 immutable
    raw rank files exactly once through job-owned node-local storage. All eight
    78-frame replays, all ordered gates, all 624 frames, every named geometry
    region, and every native Fourier index `k=0..40` (`n=5k`) pass. Across
    103,514,112 values, forward versus stored vorticity has pooled relative L2
    `6.503e-13`, RMSE `2.517e-14`, and maximum absolute error `6.106e-13` at
    frame 169. The worst per-frame error/tolerance ratio is only
    `7.981e-4` at frame 494. Runtime pressures are exact, the frozen 3,412
    negative raw-`Pi` cells are reproduced, and all nested artifact hashes
    revalidate. Together with inverse job `6892641`, this establishes the
    complete-archive bidirectional source identity and permits a separately
    committed state-candidate decision. It does not establish predictive
    sufficiency, stationarity, codec or rollout fidelity, authorize training,
    or read 85606.
54. **The project now has a prospective conditional-transient state decision.**
    The failed Phase 1 stationarity result remains failed. Without scanning for
    a more convenient suffix, Paper 0 retains the original `[0,432)`
    training, `[432,496)` guard, and `[496,624)` validation boundaries but
    interprets validation as later-background extrapolation within 85604.
    Short-horizon O1/O2 development may proceed only under a separately frozen
    matched-model protocol; stationary post-decorrelation, assimilation,
    ranking, and 85606 claims remain closed. The exact arm is
    `E6B-H1=[Ne,Pe,Pi,NVe,NVi,Vort]+Bphi`. The pragmatic arm is
    `C5P-H2=[Ne,Pe,Pi,phi,Vi]` with two ordered frames, paired with mandatory
    `C5P-H1` on the same 430 training and 126 validation target frames. This
    commits the scientific comparison before training; it trains no model,
    changes no source data, and reads no 85606 file.
55. **One verified shared 85604 model dataset now removes the remaining input
    attribution ambiguity.** CPU-only Rocky 9 job `6893525`, from clean
    commit `929ed0c`, completed with exit `0:0` in 1:03 and wrote eight
    immutable 78-frame HDF5 shards containing the union
    `[Ne,Pe,Pi,NVe,NVi,Vort,phi,Vi]` at model z88 plus the explicit
    two-sided `Bphi` state. Source hashes, exact frame/time coverage, all
    reopened array digests, all-frame legacy z88 equality for `Ne/phi/Vi`,
    and the explicit boundary float32 cast pass. The worst native
    81-to-88-to-81 per-frame relative L2 is `1.9675e-7`, below the frozen
    `2e-6` bound for all eight fields. Float64 normalization recomputed
    independently from output frames `[0,432)` matches every partial moment
    at `1e-12`, with exact counts and finite positive scales. The complete
    artifact index revalidates. This accepts common engineering provenance
    only: no codec, one-step, rollout, stochastic, assimilation, ranking, or
    85606 claim is accepted, and training remains closed pending a separately
    committed matched O1/O2 protocol.
56. **The matched deterministic O1/O2 experiment is frozen before model code
    or training.** The first candidate is a from-scratch DCAE with latent grid
    `[64,8,4,22]`, which retains direct representational support through
    `k=11`; a lower-compression `[32,16,8,22]` repair is predeclared and
    may run only if the first candidate fails. Both C5P and E6B codecs receive
    the same 200-epoch, all-training-frame, data-only MAE budget at seeds
    `1701,1702,1703`. Spectrum, cross-phase, and authoritative transport
    remain evaluation gates and never enter a loss. Only a codec passing every
    seed/state gate may feed the matched deterministic one-step comparison of
    `E6B-H1`, `C5P-H2`, and `C5P-H1`, each scored against persistence,
    linear extrapolation where applicable, and a training-only toroidal
    spectral AR(1). No O3 rollout, stochastic model, assimilation, ranking, or
    85606 access is authorized. Implementation, Rocky 9 CPU tests, and a
    bounded non-scientific GPU smoke remain required before the first full
    launch.
57. **The deterministic codec is now isolated in an attributed, tested Paper
    0 implementation.** Commit `c0cd66b` ports only the LOLA DCAE structure,
    patch ordering, channel normalization, and `softclip2` saturation needed
    by the frozen candidates. It also ports the audited predecessor repairs
    for zero-wall `x/y` padding, periodic `z` padding, and per-transition
    anisotropic strides. It imports no predecessor source or training code and
    loads no historical checkpoint. Eleven CPU tests establish exact patch
    inversion and ordering, known wall/wrap behavior, both frozen latent
    shapes, config/manifest identity, anisotropic forward/backward shapes,
    finite saturation, equal-channel loss semantics, and exact checkpoint
    reload. The complete local suite passes (`393 passed, 1 skipped`) with
    PyTorch `2.8.0`. This is an implementation gate, not evidence that the
    codec reconstructs the simulation: no training, GPU smoke, O1 metric,
    forecast result, or 85606 access has occurred.
58. **The codec implementation passes the required clean Rocky 9 CPU gate.**
    CPU-only job `6893674` ran exact commit `0d63316` on Rocky Linux 9.8,
    exited `0:0` in 23 seconds, and reported `399 passed, 1 skipped, 29
    subtests passed`. Its launcher verified the frozen protocol, manifest,
    model files, test file, and MIT license by SHA-256 before testing. It
    requested no GPU, read no simulation data, trained no model, and did not
    access 85606. The tracked result is byte-identical to the immutable Ceph
    result (SHA-256
    `d46bca572c52b8beab1f0bda18b3b869fe63ec6627b76382c350c52a972ed6f9`).
    This closes the CPU portability portion of the implementation gate only;
    data-loading, optimizer, checkpoint, bounded GPU smoke, and O1 scientific
    gates remain open.
59. **The complete data/optimizer/checkpoint path passes its bounded Rocky 9
    H100 smoke gate for both state families.** Job `6893713` ran clean commit
    `c3d0328`, passed the then-current repository suite, hash-verified the
    official dataset records, and used only frames `[0,16)` and `[496,500)`.
    The 123,007,365-parameter C5P model and 123,010,822-parameter E6B model
    each completed two epochs and two optimizer steps without OOM or
    non-finite values. Peak allocated CUDA memory was about 5.98 GB for each,
    and both selected checkpoints reproduced a save-time validation probe
    bit-for-bit after fresh reload. Job runtime was 57 seconds with exit
    `0:0`. These deliberately undertrained losses are not model evidence: the
    summary fixes `training_result_accepted=false` and
    `O1_scientific_gate_evaluated=false`. This closes the implementation and
    portability gates only. Full three-seed C5P/E6B R1 training, complete O1
    evaluation, the predeclared R2 decision, and all later phases remain open;
    85606 was not accessed.

## Exact commands

The locked launcher is `cluster/phase0_reproduce_legacy_valid.sbatch`. It:

- runs only `split=valid` from 85604;
- sets `update=etkf` explicitly;
- verifies source/config/checkpoint hashes before allocating model state;
- uses a unique output directory;
- prints Paper 0 commit, dirty state, host OS, GPU, and exact arguments;
- produces both free and assimilated metrics in `da_summary.json`.

It was submitted from Rocky 9 with:

```bash
sbatch --export=ALL,PAPER0_EXPECTED_COMMIT=7e2b5d268b2d5176a5b26cba9ac129e3caf317b5 cluster/phase0_reproduce_legacy_valid.sbatch
```

The exact model command, including every override, is stored directly in `paper0/results/phase0_legacy_valid_6890428.json`. That record also contains the hashes of the immutable Rusty `command.sh`, raw metric summary, environment record, and data audit.

The machine-readable evidence inventory is `paper0/manifests/legacy_phase0_inventory.json`.

## Phase 0 exit decision

**Phase 0 is complete with documented discrepancies.** The predecessor forecast and ETKF paths execute on Rocky 9; the fresh result and its numerical drift are preserved; deterministic and stochastic checkpoint families are located and hashed; data, preprocessing, time, toroidal-mode, geometry, and observation-operator hazards are explicit; and 85606 remained untouched.

No historical score is accepted for Paper 0 model selection. Phase 1 begins from the raw 85604 timeline and creates a guarded chronological protocol with training-only normalization and deterministic validation behavior.

## Phase 3 B2 completed probabilistic result

Three-seed B2 training, truth-separated 32-member evaluation, and the original
prospectively frozen gate completed on 85604. Original matrix job `6897564`
ran commit `361f0f27a9ece3b56f529a72c2fcfa19aa0be719` and produced SHA-256
`cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6`.
All three seeds fail every complete field, spectral, and transport family; the
architecture has zero passing seeds and fails 106 of 249 median numerical
checks. The ensemble improves point error and marginal fair CRPS but is
underdispersed, loses realization-level spectral coherence, and does not
calibrate transport. O3, assimilation, diagnostic ranking, and 85606 remain
closed.

The original gate also treated null event-conditioned values in a truth-empty
chronological block as non-finite model outputs. A016 was committed before its
separate evaluator was implemented. CPU-only Rocky 9 job `6898348` then
reduced the same stored records without training, inference, rescoring, or
85606 access. The amended matrix SHA-256 is
`4f054365d32d3e1725091ba58c8fa014f104e204748217dda482045a6c0df600`.
It makes the catastrophic finite-metric bound pass at all three seeds, but
changes zero median numerical checks, zero family decisions, and no
architecture decision. The definitive interpretation is
`PHASE3_B2_READOUT.md`; compact evidence is
`results/phase3_b2_event_eligibility_amendment_6898348.json`.

## Phase 3 B3 seed-1701 training completion

The prospectively frozen B3 functional-generative H1 training job `6898702`
completed on Rocky 9 at training commit
`a2a17cf3fc30fd504bc3eee3274e78623bf15e2b`. It ran all 100 epochs and
2,700 optimizer steps, selected epoch 72 by the fixed two-member validation
fair-CRPS bank, reloaded its selected checkpoint bit-for-bit, preserved the
frozen codec bit-for-bit, and produced nonzero latent and decoded-field member
diversity. The finished online W&B record contains all 100 epochs. Both the
per-run and wrapper SHA-256 ledgers revalidate in full. The selected
equal-channel validation fair CRPS is `0.032789371468658964`; this is a
training-selection statistic, not a scientific calibration or physics gate.
No physics-derived loss was used and 85606 was not read. The compact immutable
record is `results/phase3_b3_fgn_full_training_6898702.json`.

B3 remains scientifically undecided. The frozen independent M32 forecast,
field/spectral/cross-field/transport scorer, matched H1 comparator, and
one-seed acceptance gate must complete before any replication, O3,
assimilation, diagnostic-ranking, or held-out-run decision is permitted.

The first bounded evaluator preflight, job `6899064` at commit `d029055a`,
passed the complete Rocky 9 suite (`856 passed, 1 skipped`) and all immutable
input hashes, then failed before forecast generation in
`audit_full_training_result`. The evaluator's synthetic contract compared the
trainer's expanded runtime-provenance configuration to a bare
`FGNRunConfig`, and also called the frozen validation count `examples` rather
than the trainer's actual `target_count`. No forecast or score was created,
and no validation target fields or 85606 data was loaded. The repair validates
the exact expanded
model/noise/parent/codec/normalization/parameter-group/selection-noise schema,
uses the real two-member validation record, and is regression-tested directly
against the immutable job `6898702` result and 100-epoch history. Compact
failure evidence is
`results/phase3_b3_fgn_evaluator_smoke_failure_6899064.json`; a fresh smoke is
required and the failed output is never reused.

Fresh bounded evaluator smoke job `6899071` ran the repaired, hash-pinned
evaluator at commit `aa96db0c` on Rocky 9/H100 and completed with exit `0:0`
in 76 seconds. The complete suite passed (`856 passed, 1 skipped`), both
artifact ledgers revalidate, the independent PCG64 seed-31032 M32 noise bank
matches its frozen hash, and the forecast was hashed before validation truth
was opened. It generated four targets with 32 direct single-pass members,
found nonzero decoded-field spread for every field, and completed the required
online W&B record. The four-target MAE, fair CRPS, and corrected spread/skill
ratio are respectively `0.04353`, `0.03134`, and `0.76737`; these are bounded
preflight diagnostics, not scientific acceptance estimates. Compact evidence
is `results/phase3_b3_fgn_evaluator_smoke_6899071.json`. This authorizes only
the already-frozen full 85604 validation evaluation; O3, additional seeds,
assimilation, diagnostic ranking, and 85606 remain closed.

The first frozen B3 reduction, CPU-only job `6899154`, completed all numerical
families but reported one false provenance failure:
`integrity.evaluation.checkpoint_matches_training`. Training, evaluation, and
a direct file re-hash all record the same selected-checkpoint SHA-256
`0e0fdca97f13e2e33934d667167294d293cfc6ceedd9dee8b0504bf724acdbe9`.
Only the path spellings differ because evaluator input verification resolves
the shared Ceph alias from `/mnt/home/sdelaurentiis/ceph/...` to
`/mnt/ceph/users/sdelaurentiis/...`. The reducer had incorrectly compared the
entire `{path, sha256}` objects. The repair defines immutable checkpoint
identity by the content hash and adds a regression in which alias paths with
the same hash pass while a forged hash fails. No forecast, score, threshold,
family reducer, or numerical value changes. A fresh CPU-only reduction is
required; the original gate remains preserved for exact before/after audit.

## Phase 3 B3 completed evaluation and one-seed decision

The frozen full B3 evaluation job `6899073` completed on Rocky 9/H100 at
commit `aa96db0c3a01a1424aa900edaeaf90f706229ee2`. It generated all 126
validation targets with 32 direct single-pass members, wrote the canonical
14,535,252,816-byte forecast, and closed and hashed it before validation truth
was opened. The forecast SHA-256 is
`0f5c97b20fbf7ef32f2bd2b9695dc173d78155dcde356ef5b1a451dc4276e3ef`;
the complete score SHA-256 is
`c32508a85a68859aa676d2fada4f76a304984fea5988c81fb106ae6f724654d0`.
Its required online W&B run finished, both nested artifact ledgers revalidate,
the guard remains unread, and 85606 was not accessed. The tracked root result
is byte-identical to the Rusty manifest with SHA-256
`87b6ea353bfe9928404f01d1b494c94bfd2491395c28c0ec0a46105f0ee5e20c`.

CPU-only amended reduction job `6899224` then ran commit
`bade06465f2bdb7b3d951ce9fa51a3362ab865f7`. It replaces only the erroneous
path-object identity check from job `6899154` with immutable checkpoint
content identity. All 85604 numerical values and family summaries are exactly
unchanged; integrity now passes with zero failed checks. The complete amended
gate SHA-256 is
`882ba10898bbf132eea7713098202d8e814e3f709e8693c5b25366c52ffbc391`.

B3 fails scientifically. The field family fails 6 of 54 checks, the spectral
family 59 of 148, and the transport family 6 of 77; each has zero of five
required chronological blocks passing. Aggregate ensemble-mean MAE is 1.0031
times the deterministic H1 parent and fair CRPS is 0.7230 times parent H1 MAE.
Only one of five fields meets strict spread--skill and coverage criteria.
Eleven of 15 material power checks pass but only four of 15 realization-
coherence checks pass. All nine member-expected cross-phase and all nine
member-expected cross-coherence-change checks pass, yet mode-resolved
uncertainty remains underdispersed. All four separatrix mean-error and
fair-CRPS checks are useful, but zero of four separatrix transport quantities
is calibrated and all four
strict-face relative-L2 errors are approximately 0.74 against a 0.40 limit.

The frozen decision is `stop_B3_and_diagnose_before_replication`. Seeds 1702
and 1703, O3, assimilation, diagnostic ranking, and 85606 remain closed. The
definitive interpretation is `PHASE3_B3_READOUT.md`; the compact machine-
readable gate is
`results/phase3_b3_fgn_one_seed_gate_6899224.json`. The next permitted work is
to freeze a separately justified B4 PDE-Refiner protocol before implementation
or training.

## Phase 3 B4 prospective PDE-Refiner protocol

After the integrity-clean B3 one-seed failure was frozen, a separate B4
protocol was written before implementation, smoke testing, or training. B4 is
an explicit latent denoising adaptation of PDE-Refiner initialized from the
exact seed-1701 C5P-H1 deterministic transition and the frozen C5P-dcae_l10
codec. It retains one prior physical frame, all five physically valid state
channels, no absolute-time input, the 85604-only chronological split, and the
verified `zperiod=5` mapping `n=5k`. The codec remains frozen and noise is
introduced in per-channel standardized codec-latent coordinates; this is not
claimed to be field-space noise.

The protocol separates two hypotheses. The deterministic hypothesis asks
whether three refinement stages repair one-step realization spectra and mean
transport without materially degrading point skill. The probabilistic
hypothesis asks whether independent refinement draws form calibrated field,
mode, cross-field, and member-wise transport ensembles. Passing only the first
cannot authorize assimilation because it does not validate forecast
covariance. Both hypotheses remain prospective.

The frozen refinement schedule has three denoising stages with variance floor
`4e-7`, four network evaluations per member including the initial transition,
uniform training over levels zero through three, and no physics-derived loss.
Level zero predicts the standardized latent increment; later levels predict
the sampled latent noise. Full FP32 with TF32 disabled is required because the
last noise standard deviation is below bfloat16 resolution near unit scale.
The level-conditioning adapters are zero initialized so the unoptimized level-
zero candidate must reproduce the exact deterministic parent bit for bit.

Only implementation in new files, local and Rocky 9 CPU tests, and a bounded
two-epoch Rocky 9 GPU smoke on 85604 are authorized. The smoke is explicitly
not a scientific result and must initialize and finish an online W&B record.
Full B4 training, scientific evaluation, O3 rollout, replication, assimilation,
diagnostic ranking, and every access to 85606 remain forbidden until a later
protocol is frozen. The human-readable authority is
`protocol/PHASE3_B4_PDE_REFINER_PROTOCOL.md`; its machine-readable companion is
`manifests/phase3_b4_pde_refiner_85604.json`.

## Phase 3 B4 bounded implementation smoke

Rocky 9 H100 job `6899469` ran the exact hash-pinned implementation at commit
`2277e1b9d402a2b1627950f772b9f77a6a054f9e` and completed with Slurm state
`COMPLETED`, exit `0:0`, and elapsed time 61 seconds. The complete in-job CPU
suite passed (`905 passed, 1 skipped, 29 subtests passed`), all ten indexed
run artifacts revalidate against their SHA-256 inventory, and the required
online W&B run finished remotely. The 85604-only result manifest is tracked
byte for byte at
`results/phase3_b4_pde_refiner_gpu_smoke_6899469.json` with SHA-256
`fd2b5465f612eb8da4943f6284e317145eff64b25346895137981ce3e3993eef`.

The smoke establishes the intended engineering invariants. Before any
optimization, the conditioned level-zero transition reproduced the selected
C5P-H1 parent bit for bit with maximum absolute difference zero. The frozen
codec remained bitwise unchanged, the selected checkpoint reproduced both
latent and decoded probes bit for bit after reload, all four refinement levels
were exercised with counts `[8,10,7,7]`, and both the 51,612,800 inherited
transition parameters and 9,606,144 new refinement parameters received finite
gradients. The saved decoded stage tensor has canonical shape
`[4,2,4,5,64,32,88]`; level zero is shared exactly across members, while the
final stage has finite nonzero member diversity in every field.

The four-target equal-channel MAE values at levels zero through three are
respectively `0.0423872`, `0.0453535`, `0.0455151`, and `0.0455289`. Thus the
untrained-near-initial refinement stages are worse than level zero after only
two optimizer steps. This observation is retained rather than hidden, but it
is not an H-det failure or any other scientific conclusion: four targets, two
members, and two optimizer steps were prospectively designated only as a
mechanical preflight. Passing the smoke authorizes only a separately frozen
full-training and evaluation protocol. Full training, scientific evaluation,
O3, replication, assimilation, diagnostic ranking, and 85606 remain closed
until that authority is committed.

## Phase 3 B4 full-training and evaluation freeze

After the bounded B4 smoke passed and its immutable result was committed, the
full seed-1701 experiment was frozen before any full-run implementation or
execution. The new authority permits exactly one 100-epoch B4 training run on
the 85604 development split, one four-target evaluator smoke, and one
truth-separated one-step 85604 evaluation. It does not permit schedule or
architecture tuning, additional seeds, O3 execution, assimilation, diagnostic
ranking, or any 85606 access.

Training retains the smoke-tested FP32 explicit-denoising model and all five
physically valid state fields with no time channel. The complete 100 by 430
refinement-level matrix, inclusive-endpoint cosine learning-rate schedule,
2,700 optimizer updates, M2 checkpoint-selection bank, twenty validation
epochs, and earliest-minimum final-stage MAE selection are specified before
the run. Physics and calibration quantities remain evaluation-only.

The later scientific generator must close and hash both a final M32 forecast
and an M4 all-stage artifact before truth opens. The final-stage M4 tensor must
be the exact prefix of M32, while level zero remains exactly shared across
members. H-det and H-prob are separate decisions: H-det requires absolute
field/spectral/cross-field/transport quality and an explicit final-versus-
level-zero repair test; H-prob retains the strict field, material-mode,
cross-spectrum, member-wise transport, coverage, spread-skill, and Monte Carlo
calibration gates. Neither a sharp mean nor nonzero diversity can substitute
for the other hypothesis.

The human-readable authority is
`protocol/PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md` with SHA-256
`ffa56b2111074253a70c7453f1e36f91ca747ec59a68d632288764d60387aad1`.
The machine-readable authority is
`manifests/phase3_b4_full_evaluation_85604.json` with SHA-256
`e69af9c0e06fa1b0b33333966866098ce9ef20d6f415407ac911504f07ac9229`.

## Phase 3 B4 full seed-1701 training

Rocky 9 H100 job `6899682` completed all 100 epochs and 2,700 optimizer
updates at training commit `0350b063a1f7e5c6a00b76bf4d6aeaec71d511ef`.
Slurm reports `COMPLETED`, exit `0:0`, and elapsed time `01:41:13`. The job
first passed the complete Rocky 9 suite, all recorded losses and gradients
remained finite, the frozen codec was bitwise unchanged, and checkpoint
reload reproduced the saved latent and decoded probes bit for bit. W&B logged
all 100 epochs and finished online. The run did not read 85606 and did not
evaluate H-det, H-prob, O3, or assimilation.

The prospectively specified earliest-minimum rule selected the final candidate,
completed epoch 100. On the fixed M2 selection bank, equal-channel decoded
standardized MAE was `0.0450318` at shared level zero, `0.0438262` after the
first refinement, `0.0438300` after the second, and `0.0438299` after the
third. Thus the selection diagnostic improved by approximately 2.67 percent,
almost entirely at the first refinement stage; later stages were numerically
neutral on this marginal diagnostic. These are checkpoint-selection
measurements, not evidence of calibrated covariance or transport fidelity.

The compact immutable record is
`results/phase3_b4_pde_refiner_full_training_6899682.json` with SHA-256
`12d9e65aebccd31612484ad25735bbcb49a4d89ed395be2eddeb717f30deebe2`.
The 257 MB selected checkpoint, 980 MB resumable state, complete history, seed
banks, environment, logs, and artifact inventories remain on Ceph under job
`6899682`.

## Phase 3 B4 evaluator smoke

After training closed, evaluator commit
`029f6d9d425fd9bbac11aebf82466588a97ac658` was transferred to Rusty by a
verified immutable Git bundle. Rocky 9 H100 job `6900991` completed in 93
seconds with exit `0:0`. Its complete in-job suite passed (`971 passed, 1
skipped, 29 subtests passed`), all training and input artifacts revalidated,
all eight 85604 shards were staged and checked, and the online W&B record
finished remotely. It generated and hashed the canonical four-target M32
final artifact and M4 four-stage artifact before opening truth. The M4 stage-
three forecasts are the exact M32 prefix, level zero is bitwise shared across
members, and final spread is nonzero in every field and primary region.

The bounded score shows why the full frozen evaluation is necessary. Over
only four smoke targets, the first refinement lowered stage field error and
slightly improved realization coherence, while spectral-power, cross-field,
and separatrix-transport aggregate errors worsened. Final aggregate corrected
spread-skill was only `0.226`, indicating marked underdispersion. Stages two
and three again changed the stage aggregates negligibly. These numbers are
retained as a mechanical warning, but the four-target smoke was prospectively
declared non-scientific and neither H-det nor H-prob was evaluated.

The compact record is
`results/phase3_b4_pde_refiner_evaluator_smoke_6900991.json` with SHA-256
`f2a5b62620063372b600e63d36da2b0c6aef30b45925700a3906b61f81d23a88`.
Its approximately 692 MB of forecast artifacts and detailed score tables
remain on Ceph under job `6900991`.

## Phase 3 B4 frozen acceptance implementation

Before opening the full 126-target score, the H-det/H-prob reducer was
implemented as a pure reduction over already scored records. It reuses the
tested B2 field, spectral, cross-field, event-eligibility, and member-wise
transport reducers without changing their numerical definitions. A name
adapter maps the already frozen B4 thresholds, and an explicit projection
separates deterministic-fidelity checks from probabilistic-calibration checks.
The reducer requires the same five of six chronological blocks to pass all
applicable field, spectral, and transport families; five different passing
blocks in separate families cannot satisfy a hypothesis. H-det additionally
requires the prospectively frozen M4 stagewise-repair test. H-prob is reduced
independently and cannot borrow an H-det pass.

The full launcher pins the completed training, the passing smoke and its
historical commit, the frozen H1 and uncompressed comparators, metric-engine
source hashes, the scientific seed bank, the geometry, and the event
thresholds. It runs the complete Rocky 9 suite before generation, performs
truth-separated M32/M4 scoring, closes W&B, and only then invokes the pure
acceptance finalizer. Regardless of outcome, it cannot authorize O3 execution,
assimilation, diagnostic ranking, additional seeds, or access to 85606.

## Phase 3 B4 full evaluation completion and gate-adapter failure

Rocky 9 H100 job `6901015` completed the expensive scientific portion before
its wrapper exited. It passed the complete in-job suite (`982 passed, 1
skipped, 29 subtests passed`), generated all 126 M32 final forecasts and M4
four-stage forecasts, closed and hashed both artifacts before truth opened,
completed both frozen score records, and finished W&B online. The nested
evaluation result has SHA-256
`0988f71aa0749044e51ded92b9ea594563232df746415dccbbc6031443ca7e92`;
the final and stage score hashes are respectively
`055d81979f46a96bc0c983e0ef2f387f3032a2505117849089047e4f00b67dd3`
and `33cb4ba9256c214bf0e625511464b51076f96eb1b979c0be32ec0f3b3180e9c2`.
The exact compact evaluation manifest is tracked at
`results/phase3_b4_pde_refiner_evaluation_full_6901015.json`.

The final pure reducer then raised `KeyError: 'relative_l2_max'`: the B4
manifest's display spelling is `relative_L2_max`, while the reused B2 reducer
expects `relative_l2_max`. Therefore Slurm records job `6901015` as `FAILED`
after 23:01 even though no forecast or score failed. No H-det/H-prob decision
was emitted. A017 maps only that schema name, preserves the two threshold
values, and authorizes a fresh CPU-only reduction over the immutable records.
The original failure remains in
`results/phase3_b4_pde_refiner_gate_adapter_failure_6901015.json`.
Until that reduction completes, replication, O3, assimilation, diagnostic
ranking, and 85606 remain closed.

## Phase 3 B4 first CPU-only gate retry

Rocky 9 general-partition job `6901282` reduced no science because it stopped
on a second adapter-schema error. Before that point it verified every
job-`6901015` evaluation hash, every B4 training hash, the matched-comparator
hash, the exact commit `aa33a402b27c0468122dca9400eceeed6a0dd34d`, and
the complete suite (`986 passed, 1 skipped, 29 subtests passed`). It performed
no training, inference, truth scoring, forecast mutation, 85606 access, O3,
assimilation, or diagnostic ranking.

The unchanged B2 transport reducer consumes the required calibrated-quantity
count from `transport.separatrix.probabilistically_calibrated_required`. The
B4 adapter instead placed the frozen count under `separatrix_calibration`,
which is reserved by that reducer for spread-skill and coverage tolerances.
It therefore raised `KeyError: 'probabilistically_calibrated_required'`
before H-det or H-prob was formed. The log SHA-256 is
`36bd9f0b6567256aa87c0166f9b5d66ef8aa4b7a28490e7e9dbbcbce3b4d2088`.

A018 corrects only this nesting and adds complete source- and destination-
schema assertions. The repaired numerical reducer was then exercised locally
on the exact immutable score and comparator records. It completed every
family without a schema exception: H-det field passed while H-det spectral
and transport failed; all three H-prob families failed. That dry run is a
pre-submission regression, not the formal gate. A clean CPU-only Rocky 9 job
must still produce and archive the authoritative decision.

## Phase 3 B4 final one-seed decision

CPU-only Rocky 9 job `6901285` completed the frozen reduction over immutable
evaluation job `6901015`. It passed the complete suite (`987 passed, 1
skipped, 29 subtests passed`) and all 138 provenance/integrity checks. The
exact final gate has SHA-256
`4c07a7f4886c14ca2e53d6e322fe309e5efde1f76ab2ed779a3acd14d110f6be`
and is tracked at
`results/phase3_b4_pde_refiner_one_seed_gate_6901285.json`.

H-det fails: the field family passes overall and all six blocks, but spectral
and transport each pass zero blocks, and the stagewise repair test fails.
H-prob fails: field, spectral, and transport each pass zero blocks. The final
aggregate field spread--skill is `0.2172696873820019`; all material-mode and
cross-projection calibration checks fail; separatrix transport spread--skill
is only approximately `0.060--0.073`. B4 replication, O3, assimilation,
diagnostic ranking, and 85606 access remain closed. The complete interpretation
and numerical tables are in `PHASE3_B4_READOUT.md`.

## Phase 3 B5 residual audit freeze

After the completed B4 failure and before any B5 implementation or training,
the deterministic-mean residual audit was frozen in
`protocol/PHASE3_B5_RESIDUAL_AUDIT_PROTOCOL.md` and
`manifests/phase3_b5_residual_audit_85604.json`. It authorizes one Rocky 9
H100 audit of the exact seed-1701 H1 residual on training targets `[2,432)`.
The context-only forecast must close and hash before target truth opens.

The audit measures standardized-field residual bias, scale, spatial and
temporal autocorrelation, geometry-dependent heteroscedasticity, cross-field
correlation, and toroidal support. It explicitly treats the residuals as
in-sample model errors rather than independent simulations or identified
aleatoric noise. The complete stored toroidal axis remains mandatory because
`zperiod=5`, `n=5k`, and material `k=1` structure spans that domain. This
protocol authorizes no B5 training, validation read, O3, assimilation,
diagnostic ranking, or 85606 access.

Before execution, the reporting-timescale provenance was made explicit without
changing an audit metric or architecture rule. The frozen reference is the
Phase 1 job-6890606 median finite C5 `1/e` crossing: `2.2443947105846638`
frames (`7.0292319729572625` microseconds). Its status remains
`diagnostic_only_under_nonstationarity`; it is not an independence or
stationarity claim.

The audit implementation is truth separated. A training-context-only adapter
generates all H1 forecasts, closes and hashes the HDF5 artifact, and only then
constructs the target-only reader. Pure known-answer tests cover residual bias,
valid-overlap and circular autocorrelation, the `n=5k` Fourier mapping,
cross-field rank, artifact order/hash locks, plot generation, and exact B4 and
Phase 1 evidence schemas. The initial full local implementation suite passed
with `1007 passed, 1 skipped`. No cluster audit had been submitted at this
checkpoint.

## Phase 3 B5 residual audit completion

Rocky 9 H100 job `6901393` completed the frozen 85604 training-residual audit
at exact commit `88fdcc8fa9ce7e2ba24958cd873cb7c4c5a771ff`. Slurm reports
`COMPLETED`, exit `0:0`, and elapsed time `00:17:29`. The complete in-job suite
passed (`1011 passed, 1 skipped, 29 subtests passed`), all eight model-data
shards revalidated, the W&B run finished online, and every archived artifact
hash passed. Guard, validation, and 85606 data remained unread.

The context-only H1 forecast was closed and hashed before target truth opened.
Its canonical `[430,5,64,32,88]` HDF5 has SHA-256
`d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`;
the full residual audit has SHA-256
`d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67`.

The in-sample one-step residual is small in global standardized RMS
(`0.046--0.103`) but strongly heteroscedastic, cross-field coupled, and
concentrated in non-axisymmetric toroidal content. `Pe` and `Pi` residuals
correlate at `0.804`; global entropy effective rank is `4.047`; pointwise
95th-to-5th-percentile scale ratios reach `33.40`; and `Vi` residual-to-truth
power is `1.307` for `k=4..5` and `1.825` for `k=6..7`. The detailed residual
pattern decorrelates in one frame, while residual RMS retains multi-frame
dependence.

These observations support a joint, context-conditioned, full-toroidal,
field-coordinate B5 residual generator with a fresh per-step innovation. They
do not identify irreducible aleatoric uncertainty and do not establish
validation skill. Full B5 training, O3, assimilation, diagnostic ranking, and
85606 remain closed. The complete numerical interpretation is in
`PHASE3_B5_RESIDUAL_AUDIT_READOUT.md`.

## Phase 3 B5 joint field-residual EDM smoke completion

Rocky 9 H100 job `6901469` completed the frozen bounded B5 implementation
smoke at exact commit `8cd5b3465dcc7092b9f5a50cfdccca39dcb17c07`. Slurm
reports `COMPLETED`, exit `0:0`, and elapsed time `00:01:29`. The complete
Rusty-side suite passed (`1047 passed, 1 skipped, 29 subtests passed`), W&B
finished online, all artifacts were hash indexed, and the frozen H1 forecast
and residual audit were reverified after execution.

The `11,604,709`-parameter full-field joint residual U-Net completed all 64
optimizer steps with `1.2504 GiB` peak allocated CUDA memory. Its fixed-probe
EDM loss decreased from `0.9942206` to `0.9706594`; checkpoint reload was
bitwise exact; the eight-cell toroidal shift probe passed with maximum absolute
difference `9.5367e-7`; and the 18-step Heun sampler produced finite, distinct
members with canonical axes `[1,2,1,5,64,32,88]`.

This is an implementation result only. It establishes feasibility, correct
boundary behavior, and a non-collapsed sampling path; it does not establish
validation skill, transport fidelity, or calibration. Full B5 training and
validation remain unauthorized until a separate prospective protocol is
frozen. The detailed interpretation is in
`PHASE3_B5_EDM_SMOKE_READOUT.md`, and the compact result is tracked at
`results/phase3_b5_field_residual_edm_smoke_6901469.json`.

## Phase 3 B5 full protocol, implementation, and execution

After the bounded job-`6901469` smoke passed, the seed-1701 full-training and
scientific-evaluation contract was frozen in
`protocol/PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md` and
`manifests/phase3_b5_full_training_evaluation_85604.json`. The contract pins
one 100-epoch, 10,800-step training run, fixed-seed data-only EDM-loss
selection every five epochs, one four-target M32 evaluator smoke, one complete
126-target M32 one-step evaluation, and one frozen acceptance reduction. It
forbids architecture or schedule sweeps, post-hoc calibration, more seeds,
O3/O4/O5, assimilation, diagnostic ranking, steering, and 85606 access.

The implementation preserves the frozen H1 mean and learns a fresh
11,604,709-parameter joint residual directly in five-field space. No DCAE is
used by the residual generator. Context-only forecast generation closes and
hashes the 14.5-GB M32 forecast before validation truth is opened. Checkpoint
selection and the scientific ensemble use separate frozen seed banks. The
complete local and Rocky 9 suite at execution commit
`512c987d49a1a572430ed6f9fca18975798fc599` passed with
`1120 passed, 1 skipped`.

Rocky 9 H100 training job `6901531` completed in `00:45:38` with all 100
epochs, 10,800 optimizer steps, and 43,000 target presentations. Validation
EDM loss decreased through the final candidate, so epoch 100 was selected at
`0.30807498889783075`. The selected checkpoint SHA-256 is
`255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72`;
fresh reload was bitwise exact. The online W&B run finished, and guard frames
and 85606 remained unread.

Bounded evaluator job `6901582` completed exactly four targets with M32, 18
Heun steps, and 35 network evaluations per member. It was marked
non-scientific and used only to establish mechanics. Full evaluation job
`6901587` then completed all 126 targets. Its closed 14,535,535,504-byte
forecast has SHA-256
`1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b`;
the independent score has SHA-256
`c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6`.
All prefixes and hashes verified, and W&B finished online.

CPU-only Rocky 9 gate job `6901661` passed all 120 integrity checks and found
every required metric finite. The field, spectral, and transport families all
failed with zero of six passing chronological blocks. Overall failed-check
counts are respectively 4 of 54, 83 of 148, and 7 of 77. The authoritative
gate SHA-256 is
`a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b`.
The exact disposition is
`B5_one_step_gate_failed_localize_without_retuning`; O3 protocol writing,
additional seeds, assimilation, diagnostic ranking, and 85606 access remain
closed.

## Phase 3 B5 reproducible failure localization

The completed B5 artifacts were copied read-only for a no-rescoring
localization. `tools/summarize_b5_residual_edm_gate.py` verifies the exact
training, history, evaluation, score, gate, B3, B4, and H1 comparator hashes
before extracting a compact record. It performs no training, inference, truth
scoring, or threshold modification. `tools/plot_b5_residual_edm_gate.py`
renders six fully labeled figure pairs from that compact record.

The localization establishes that B5 improves H1 mean MAE by `4.41%`, mean
RMSE by `5.95%`, and fair CRPS to `0.687` times H1 MAE. Aggregate pixel
spread--skill is `0.802`, and 13 of 15 expected-member power checks pass.
Nevertheless only 4 of 15 realization-coherence checks pass; zero of 15
mode-power and zero of 18 cross-projection calibration checks pass. All nine
cross-phase checks pass, while the three high-band cross-coherence checks
fail. Strict transport relative-L2 remains approximately `0.71`, and none of
the four separatrix transport distributions is calibrated.

The pooled strict-face flux spread is close to its local error
(`0.979--0.994` spread--skill), but integrated separatrix spread is only
`0.413--0.485`. This local-to-integrated collapse identifies missing coherent
spatial/cross-field covariance more specifically than a global lack of noise.
The complete interpretation is `PHASE3_B5_READOUT.md`; the tracked compact
record is
`results/phase3_b5_residual_edm_one_seed_localization_6901661.json`, SHA-256
`ae10349b98394914f6a87dc99bebdc965056a941356f32b0392e261169cbf1f6`.
