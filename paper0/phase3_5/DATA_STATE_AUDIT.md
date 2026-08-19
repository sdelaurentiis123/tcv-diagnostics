# Phase 3.5 data and state audit

**Audit status:** completed from previously frozen Paper 0 evidence before new
Phase 3.5 computation

**Audit date:** 2026-08-19

**Simulation inspected:** 85604 only

**85606 inspected by Phase 3.5:** no

## Bottom line

The available 85604 engineering dataset is sufficient to run the complete
Phase 3.5 diagnosis. It contains both the pragmatic C5P model view and the
exact saved-state candidate. No new raw extraction is required.

The deterministic H1 model does **not** receive the exact saved Hermes state.
It receives one frame of `[Ne,Pe,Pi,phi,Vi]`. This omits evolved electron
momentum `NVe`, replaces evolved generalized vorticity `Vort` by derived
potential `phi`, and omits the retained two-sided potential-boundary midpoint
`Bphi`. Ion momentum `NVi` is not independently missing on this archive because
it is algebraically reconstructible from `Ne,Vi` to roundoff.

The H1 codec uses correct circular convolution on `z`, but it packs the
toroidal axis twice with stride two. The H1 transformer then uses absolute
regular-grid sinusoidal coordinates and learned RoPE directions. Consequently,
periodic padding alone does not establish translation equivariance; the
requested H4 inference audit is necessary.

## 1. Trajectory, axes, and time

The canonical engineering dataset contains 624 ordered 85604 frames:

| Quantity | Value |
|---|---:|
| global frame indices | `0..623` |
| normalized simulator time | `285000..471900` |
| normalized step | `300` |
| physical cadence | `3.131905426352636 microseconds` |
| volume axes | `[time,x,y,z]` |
| volume shape per frame | `[64,32,88]` |
| native toroidal cells before resampling | `81` |
| simulated toroidal fraction | `1/5` |
| periodic axes | `z` only |
| nonperiodic axes | `x,y` |
| full-torus mode mapping | `n=5k` |

The 88-cell grid is an unwindowed periodic Fourier resampling of the native
81-cell output. Its added `k=41..44` bins are numerical padding, not new
simulator resolution. Circular shifts are valid only along `z`; Phase 3.5
rejects attempts to wrap `x` or `y`.

The immutable learning boundaries are:

| Region | Raw frames | H1 target frames |
|---|---:|---:|
| training | `[0,432)` | `[2,432)` |
| guard | `[432,496)` | unread |
| validation | `[496,624)` | `[498,624)` |

There are 431 consecutive raw training transitions and 127 consecutive raw
validation transitions. Matched H1 analysis contains 430 training transitions
and 126 validation transitions because target indices were aligned with the
historical H2 comparison. The guard remains unused.

## 2. Stored variables and source semantics

The shared dataset stores the ordered union:

```text
Ne, Pe, Pi, NVe, NVi, Vort, phi, Vi, Bphi
```

The six volumetric variables advanced by the executed Hermes configuration are
exactly:

```text
Ne, Pe, Pi, NVe, NVi, Vort.
```

The saved derived fields include `Te,Ti,Ve,Vi,phi`. The exact saved-state
candidate is:

```text
S6+Bphi = [Ne,Pe,Pi,NVe,NVi,Vort] + Bphi[inner/outer,y].
```

Fixed geometry, the executed elliptic operator, evolved pressures, `Vort`, and
`Bphi` reconstruct stored interior `phi` in both directions on all 624 frames
under the previously validated gauge policy.

### 2.1 Derived transformations

On this archive:

- quasineutral `Ni=Ne` exactly to the frozen tolerance;
- `Te=Pe/Ne`, equivalently `Pe=Ne*Te`, closes on every stored cell;
- `Ti` is obtained from nonnegative/floored ion pressure divided by density;
  therefore the 3,412 retained negative `Pi` cells cannot be reconstructed
  from `Ne,Ti`;
- the source density soft floor is

  ```text
  softFloor(N,f) = max(N,0) + f exp(-max(N,0)/f), f=1e-7;
  ```

- `Vi=NVi/(2*softFloor(Ne,1e-7))` for deuterium; the density floor is inactive
  at every saved 85604 cell, so `(Ne,Vi)` and `(Ne,NVi)` are algebraically
  equivalent here;
- `Ve=1836*NVe/softFloor(Ne,1e-7)` under the executed normalized mass
  convention; C5P contains neither `Ve` nor `NVe`;
- `phi` is recovered by the executed geometry-dependent elliptic inversion
  from `Vort`, pressures, and retained radial boundary state, up to gauge. It
  is not an independently evolved volume.

Physical reporting conversions already frozen in Paper 0 are `Ne*1e19 m^-3`,
`Pe,Pi*80.1088317 Pa`, `phi*50 V`, and
`Vi*69205.61141651045 m s^-1`.

### 2.2 Model-coordinate transformations

Training-only normalization was fitted on raw frames `[0,432)` with float64
population moments. `Ne` first uses `log(Ne+1e-6)`; every other stored volume
uses the identity transform. The C5P statistics are:

| Field | Mean | Population standard deviation |
|---|---:|---:|
| `Ne` after log | `-1.9368449594652586` | `1.4353625058372068` |
| `Pe` | `0.41732773` | `0.73493589` |
| `Pi` | `0.47917411` | `0.69350059` |
| `phi` | `2.848521239601473` | `1.2796587059175046` |
| `Vi` | `-0.17667431880104253` | `0.9209582202997134` |

Validation and guard values contributed no normalization statistic. Potential
residual analysis additionally applies the frozen per-sample gauge correction.

## 3. State completeness for Phase 3.5

### 3.1 Information absent from H1 input

The C5P-H1 input excludes:

- evolved electron momentum `NVe`;
- evolved generalized vorticity `Vort`;
- retained radial-potential boundary midpoint `Bphi[2,32]`;
- any iterative elliptic-solver state, preconditioner state, internal substep,
  source history, or boundary relaxation history not encoded in the saved
  fields.

It contains current `phi`, so some consequences of vorticity and boundary
history are present implicitly in the current observed field. That does not
make the omitted evolved/boundary variables algebraically recoverable from a
single C5P snapshot.

### 3.2 Exact-state availability

The exact saved-state candidate can be assembled without touching raw rank
files:

- all six evolved volumes are present on the common 88-cell grid for all 624
  frames;
- `Bphi` is present as `[time,2,32]`;
- frame index and normalized time are stored explicitly;
- the original native-state/elliptic audit proves exact source closure;
- the H0/H7 probes can therefore compare C5P with omitted-state summaries
  directly.

This does not authorize retraining the earlier failed E6B codec. Phase 3.5
uses exact state only in read-only/lightweight information probes.

## 4. Frozen H1 architecture and equivariance risks

The selected deterministic model is C5P-H1 seed 1701 with the passing
`dcae_l10` codec.

### 4.1 Codec

- input/output fields: five C5P channels;
- convolutions: kernel `3 x 3 x 3`;
- padding: zeros on `x,y`, circular on `z`;
- hidden widths: `64,128,256`, two residual blocks per level;
- packing/downsampling: two `PatchifyND(2,2,2)` transitions;
- total packing stride: `(4,4,4)`;
- latent channels/grid: `[32,16,8,22]`;
- no pooling and no strided convolution: downsampling is a deterministic
  phase-sensitive space-to-channel permutation followed by convolution;
- decoder reverses packing with `UnpatchifyND`;
- latent map: LOLA `softclip2` saturation;
- codec reconstruction passed O1, but that result did not test all toroidal
  shift phases.

Circular convolution is equivariant by itself. Packing by stride four is
equivariant only to compatible shift classes unless channel permutations and
subsequent operations preserve the corresponding phase action. H4 tests this
rather than assuming it.

### 4.2 Deterministic transition

- family: masked latent ViT residual;
- history: one exact C5P frame;
- latent patch: `(2,2,1)`, yielding 704 tokens per frame;
- hidden width/blocks/heads: `512/16/4`;
- absolute regular-grid sine/cosine position projection;
- learned RoPE direction on time and all three latent spatial coordinates;
- no stochastic features;
- predicts a standardized latent increment added to the last context latent;
- decodes through the frozen codec.

The regular-grid coordinates do not encode the toroidal axis as an angle on a
circle. Absolute position projection and generic learned RoPE can therefore
break periodic translation equivariance even when the codec's convolutions use
circular padding.

## 5. Immutable artifacts used by Phase 3.5

| Role | Stable path or identifier | SHA-256 |
|---|---|---|
| shared dataset root | `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_model_dataset/job_6893525` | manifest `27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18` |
| training normalization | job 6893525 `normalization.json` authority | `f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7` |
| exact native-state closure | `paper0/results/phase2_potential_vorticity_all_frame_6893033.json` | `cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae` |
| H1 training forecast | `.../phase3_b5_h1_residual_audit/job_6901393/audit/h1_training_forecast.h5` | `d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18` |
| H1 validation forecast | `.../phase2_o2_evaluation_full/job_6896117/task_0_c5p_h1_seed_1701/forecast.h5` | `a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed` |
| H1 checkpoint | `.../phase2_o2_full/job_6894980/task_0_c5p_h1_seed_1701/selected.pt` | `5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e` |
| H1 codec | `.../phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt` | `9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d` |
| H1 residual audit | job 6901393 `residual_audit.json` | `d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67` |
| H1 residual sufficient statistics | job 6901393 `raw_accumulators.npz` | `50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b` |
| B5 selected checkpoint | `.../phase3_b5_field_residual_edm_full/job_6901531/b5_joint_field_residual_edm_seed_1701/selected.pt` | `255904ef362c4d3f0fdb873131cd0b30bc02ea384e76e244d50698bd50df0c72` |
| B5 M32 forecast | `.../phase3_b5_residual_edm_evaluation_full/job_6901587/b5_joint_field_residual_edm_seed_1701/forecast_M32.h5` | `1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b` |
| B5 seed bank | same job, `scientific_sampler_seeds_M32.npy` | `013a5cbd11525f41d9ec44a5f7bab30a6fd52aa8e4541757cf1e30ded0e63e14` |
| authoritative geometry | `/mnt/home/sdelaurentiis/ceph/tcv-fresh-proj/85604/tcv_85604_adjusted.nc` | `0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8` |
| K4 compact result | `paper0/results/phase3_residual_kl_oracle_6904897.json` | compact result `4f0166308e71d308a960c004cb6f9c247f6e0d9de038d01df5f3a85037fb2879` |
| K4 scientific result | job 6904897 `residual_kl_oracle.json` | `71be0e38285a06f98bd03138d3e1639a70d88665e698cbb4c96220e57dc991b7` |
| K4 training basis | job 6904897 `training_kl_basis.npz` | `fcc32c3baaf0deb85fa55456612d3ab8beaf859af20b5ba86f94233c15e0dbbc` |

Ellipses denote the common authoritative Ceph prefix
`/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0` or its equivalent mounted
`/mnt/ceph/users/sdelaurentiis/...` path. The Phase 3.5 run manifest records
resolved absolute paths and verifies bytes, hashes, target arrays, and axes
before use.

## 6. Existing evidence retained without reinterpretation

- The original whole-interval stationarity screen failed. Phase 3.5 measures
  the mechanism and transfer consequence; it does not erase that result.
- Consecutive nonaxisymmetric truth previously showed an approximately 9--12
  cell common toroidal displacement. Phase 3.5 uses a new preregistered
  multichannel estimator and treats the old value only as a comparison.
- `dcae_l10` passed the frozen O1 reconstruction gate. That does not imply
  shift equivariance.
- C5P-H2 did not improve the deterministic O2 transition. That is evidence
  about one extra frame under one architecture, not a proof against all memory.
- K4 tested a fixed global condition-independent linear residual model only.
- B5 exists and is therefore eligible for the preregistered fixed-seed
  context-shuffle sensitivity. Its failure is documented if the exact
  checkpoint or sampler configuration cannot be reconstructed.

No missing artifact will be silently regenerated from a different checkpoint,
seed, model arm, shot, or split.
