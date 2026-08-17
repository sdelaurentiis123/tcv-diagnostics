# Phase 0 repository and result audit

**Status:** in progress

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

Other historical families to inventory before Phase 0 closes:

- z44 codec/diffusion
- z22 continuation
- matched-budget f8-short continuation
- actual-sampled rollout-CRPS fine-tune
- deterministic Walrus/ViT3D baselines

## Forecast reproduction

Pending a fresh 85604-validation execution on Rocky 9. The selected legacy driver produces an autonomous free-rollout arm in the same invocation as the ETKF arm.

Historical reference only, not yet a Paper 0 reproduction:

- Split: 85604 legacy validation
- Start frame: 24
- Horizon: 48
- Ensemble: 64
- Sampler: AB, 16 steps
- Mean free-rollout RMSE: `0.2011699302399412` in standardized legacy model space

## Assimilation reproduction

Pending the same fresh 85604-validation execution.

Historical reference only:

- Synthetic layout: `iter`
- Observation operator: 69 direct-state point samples on toroidal plane `z=0`
- Channels: 54 `Ne` samples on an idealized midplane radial line and 15 `phi` samples on idealized divertor-leg boundary positions
- This is a synthetic direct-state layout, **not** a realistic ITER diagnostic forward model.
- ETKF cadence: every 4 frames
- Observation standard deviation: 0.05 in standardized model space
- Inflation: 1.0
- Historical mean anchored RMSE: `0.17706738111186535`

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

## Exact commands

The first locked reproduction launcher will be added under `cluster/`. It will:

- run only `split=valid` from 85604;
- set `update=etkf` explicitly;
- verify source/config/checkpoint hashes before allocating model state;
- use a unique output directory;
- print Paper 0 commit, dirty state, host OS, GPU, and exact arguments;
- produce both free and assimilated metrics in `da_summary.json`.

The machine-readable evidence inventory is `paper0/manifests/legacy_phase0_inventory.json`.

## Phase 0 exit decision

Not yet reached. Phase 0 closes only after the primary forecast/assimilation result is freshly reproduced, the deterministic baseline artifacts are located, remaining checkpoint families are inventoried, discrepancies are recorded, and exact commands are committed.
