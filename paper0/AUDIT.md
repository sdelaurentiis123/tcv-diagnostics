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
