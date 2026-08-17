# Phase 2 O1 codec-reconstruction protocol

**Protocol status:** frozen before O1 implementation or execution

**Simulation in scope:** TCV/Hermes `85604` only

**Sequestered simulation:** `85606`

**Purpose:** isolate field-compression error before any one-step or rollout
dynamics are evaluated

This protocol applies the already-trained historical f8 and z44 codecs to the
complete 85604 trajectory. It is a deterministic representation audit, not a
new training experiment and not a cross-shot generalization test. The failed
Phase 1 stationarity gate remains failed; O1 cannot reopen the learning gate.

## 1. Codec identities and comparison scope

### f8 reference codec

```text
run: /mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/
     w24x2ybf_tcv_c5_dcae_3d_tcv_f8c64
checkpoint: state.pth
checkpoint SHA-256:
  9f65dc523b8ee32ea5dd87842b99075de15f9aae86d2e71a5da55bc37091a44e
config SHA-256:
  66509d2b0c9a1aaa03959e0e33691d443f39fa24bbad93a0dbb41e291176e776
input:  [5, 64, 32, 88]
latent: [64, 8, 4, 11]
training: 50 epochs x 2048 examples, MAE, from scratch
```

The scalar stride is two at all three downsampling transitions. The latent has
22,528 scalars per frame versus 901,120 input scalars, a nominal 40-fold scalar
compression.

### z44 feasibility codec

```text
run: /mnt/home/sdelaurentiis/ceph/lola_tcv/runs/ae/
     z44c6604191_tcv_c5_dcae_3d_tcv_f8z2c64
checkpoint: state.pth
checkpoint SHA-256:
  095d25f9b6e867103d4cfb946cc9ea8a172a5a6db5b28e5726428c4c57e4979d
config SHA-256:
  5d868c1cfc5a17ce26c2f6ce86ced50d7b55525c6967c5b599b1074058b67284
input:  [5, 64, 32, 88]
latent: [64, 8, 4, 44]
training: 12 additional epochs x 1024 examples
parent: z22 codec loaded non-strictly
loss: MAE + 0.03 multiscale-increment loss
```

Its strides are `(2,2,2)`, `(2,2,1)`, `(2,2,1)`. The latent has 90,112
scalars per frame, a nominal 10-fold scalar compression.

The two checkpoints have different parentage, training budgets, and losses.
O1 may establish that one checkpoint preserves a quantity and another does
not; it cannot attribute that difference specifically to toroidal latent
resolution. A matched retraining would be required for a causal architecture
ablation.

The f8 latent-grid Nyquist index is `k=5` (`n=25`), but this is not treated as
an automatic hard cutoff in decoded field space. A nonlinear decoder can emit
higher output modes, and encoded channels may carry aliased or coupled
information. Mode preservation is therefore measured empirically rather than
inferred from latent shape.

## 2. Executed source identity

O1 imports the predecessor LOLA package read-only after verifying the complete
package hash and the critical codec files. The expected critical hashes are:

| File under `external/lola/lola` | SHA-256 |
|---|---|
| `autoencoder.py` | `5c20d880799301a636ff8de67d34d39221d9c5a7e9e0bc2123ae84ee43fc5c83` |
| `data.py` | `def0e35e3a97a31627415186130b5b8ac6bf69611dc50caacafb73398a706bc5` |
| `emulation.py` | `a56aef6d1d04c86af91238eeb51d93d345dce5a74481dde5ae3dc244f842691f` |
| `nn/dcae.py` | `281f8541aa09822147f8769e9a11fb63497aa54783dd9a806b173e76c5fbaede` |
| `nn/layers.py` | `ad6aab36d52ea7aba2a0c45006a33413304c0d9ceb9abffd52a497a24adf616f` |

The expected composite hash of all Python files in that package is
`3fb6e6be7649e86fc0626f5d847adf13649e213c82b543c714ae258332bfdf7d`.
No predecessor source file is modified or copied silently into Paper 0.

## 3. Data and historical exposure labels

Read the two audited 85604 Well storage shards as one chronological trajectory:

```text
global frames [0, 500): legacy train storage shard
global frames [500, 624): legacy valid storage shard
```

The input hashes must match the Phase 0 manifest. Evaluate every global frame
`[0,624)` exactly once in chronological order, with no random roll, random
flip, window resampling, or duplicate cache row. The eight reporting blocks are
the same fixed 78-frame blocks used in the Phase 1 stationarity screen.

The codecs were optimized on the legacy train shard and historically selected
with access to the legacy validation shard. Therefore:

- frames are not called independent samples;
- no block is called a blind test;
- block-to-block variation is descriptive robustness within one run;
- O1 measures the behavior and capacity of historical representations, not
  out-of-run generalization.

Any input path containing `85606` or a path component named `test` is rejected.

## 4. Preprocessing and inverse transformation

The historical checkpoints must receive the coordinates in which they were
trained. Both configs declare:

```text
fields = [Ne, Te, Ti, phi, Vi]
mean   = [-1.9359, 0.9337, 1.2636, 2.8614, -0.1795]
std    = [ 1.4488, 0.5312, 0.4681, 1.2784,  0.9219]
Ne transform = ln(Ne + 1e-6)
```

These legacy statistics include a different temporal region from the candidate
Paper 0 training-only statistics. Using them here is required to evaluate the
fixed checkpoints on their intended input distribution; it does not authorize
their use for new model training. Every result is labeled `legacy-preprocessed`.

For field-space physics metrics, invert the codec transform exactly. Density is
returned to linear simulation-normalized space with `exp(u)-1e-6`; other fields
are unstandardized linearly. Positive physical-unit multipliers may be applied
for reporting, but phase, coherence, and reconstruction power ratios are
unchanged by those positive scalings. No density clipping is permitted; the
number and minimum of any non-positive reconstructed density cells are
reported.

Potential has a gauge-dependent constant. Report both the raw standardized
pixel error expected by the checkpoint and a per-frame spatial-mean-removed
field error. Toroidal modes `k>0`, gradients, coherence, and cross-phase are
invariant to this constant; `k=0` potential is not used in a physics gate.

## 5. Deterministic round trip

For every codec and input chunk:

```text
model.eval()
model.requires_grad_(False)
z = model.encode(x)
x_hat = model.decode(z, noisy=False)
```

The expected latent and reconstructed shapes are asserted. Dropout is disabled
by evaluation mode. Latent noise is zero in both configs and decode noise is
also disabled explicitly. The same float32 input tensor is supplied to both
codecs. Inference chunks are chronological and must not alter results; chunk
size is recorded. No dynamics model, sampler, sensor, filter, or future frame
is involved.

## 6. Required metrics

All metric conventions inherit `PHASE2_METRIC_PROTOCOL.md`.

### 6.1 Field reconstruction

For each field, overall and in each fixed temporal block, report in legacy
standardized model coordinates:

- RMSE;
- MAE;
- signed bias;
- truth variance;
- reconstruction variance;
- reconstruction/truth variance ratio.

For `phi`, repeat RMSE, MAE, and bias after removing one spatial mean from each
truth and reconstructed frame. The aggregate five-field RMSE is reported only
alongside the five field-specific values.

### 6.2 Toroidal spectrum and truth-to-reconstruction transfer

On linear, inverse-transformed fields, calculate the Parseval-normalized
one-sided toroidal power for every `k=0..44`, averaging over frames, `x`, and
`y`. Store truth power, reconstructed power, reconstruction/truth power ratio,
and each mode's fraction of total non-axisymmetric truth power.

For every field, calculate

```text
S_XR(k) = mean(X_k conjugate(R_k)),
gamma_XR^2(k) = |S_XR|^2 / (S_XX S_RR),
delta_XR(k) = angle(S_XR).
```

This measures whether the decoded field matches the actual input realization,
not merely whether its marginal spectrum has the right amplitude.

### 6.3 Cross-field structure

For the primary transport-relevant pairs

```text
(Ne, phi), (Te, phi), (Ti, phi)
```

calculate truth and reconstructed cross-spectrum, magnitude-squared coherence,
and phase at every toroidal mode `k>0`. The signed circular phase error is

```text
Delta_phi(k) = angle(exp(i (phi_recon(k) - phi_truth(k)))).
```

Report all curves, not only a selected mode. `Vi-phi` and all remaining field
pairs may be recorded as descriptive secondary outputs but do not enter the
preliminary transport gate.

Nonlinear flux is not computed from ensemble-mean fields. O1 has no ensemble,
but the same member-wise rule remains binding for later stochastic oracles.

### 6.4 Frozen mode bands

Summaries use the following stored-index/full-torus labels:

| Stored `k` | Full-torus `n` | Label |
|---:|---:|---|
| `1..3` | `5..15` | low non-axisymmetric |
| `4..5` | `20..25` | coherent study band |
| `6..7` | `30..35` | upper study band |
| `8..16` | `40..80` | measured high band |
| `17..44` | `85..220` | remaining resolved output |

For a field, a band is `material` when it contains at least 1% of that field's
total non-axisymmetric truth power. For a cross-field pair, it is material when
it contains at least 1% of the total truth cross-amplitude
`sum_(k>0)|S_ab(k)|`. Materiality is evaluated from truth once and is not tuned
per codec.

Within each band report:

- truth-power fraction;
- total reconstructed/true power ratio;
- truth-power-weighted truth/reconstruction coherence;
- truth-cross-amplitude-weighted absolute cross-phase error in degrees;
- truth-cross-amplitude-weighted absolute coherence change.

Weights and denominators are saved. Undefined zero-power quantities remain
undefined and fail any applicable gate.

## 7. Preliminary representation gates

These are engineering stop/go criteria, not universal physics tolerances. They
are frozen before the O1 curves are opened.

A codec receives a **field reconstruction pass** only if every C5 field has:

- overall standardized RMSE at most `0.10`;
- reconstruction/truth variance ratio between `0.80` and `1.20`.

It receives a **spectral transfer pass** only if, for every material band among
`k=1..7` in `Ne`, `Te`, `Ti`, and `phi`:

- total power ratio lies in `[0.80, 1.25]`;
- truth-power-weighted truth/reconstruction coherence is at least `0.90`.

It receives a **cross-field preliminary pass** only if, for every material band
among `k=1..7` for `(Ne,phi)`, `(Te,phi)`, and `(Ti,phi)`:

- truth-cross-amplitude-weighted absolute phase error is at most `15 degrees`;
- truth-cross-amplitude-weighted absolute coherence change is at most `0.10`.

The thresholds must also pass independently in at least seven of the eight
fixed temporal blocks; a full-record average cannot hide a single transient
catastrophe. A band that is non-material overall is reported but does not fail
the gate.

The **full codec acceptance gate remains unresolved** until the authoritative
geometry-aware particle and heat-flux calculations are identified, frozen, and
tested. O1 can produce `preliminary pass`, `preliminary fail`, or `blocked`; it
cannot certify transport fidelity by spectra alone.

Decision rules:

1. If both codecs fail before transport, representation repair precedes any
   new dynamics training.
2. If f8 passes and z44 adds no material preservation, retain f8 as the simpler
   historical representation candidate.
3. If only z44 passes, treat higher toroidal capacity as a candidate need and
   perform a matched from-scratch codec comparison before attributing cause.
4. If both pass, dynamics becomes the next measured layer; z44 is not preferred
   merely because it is larger.
5. If flux later fails despite the preliminary pass, the codec fails the full
   Paper 0 gate and spectra alone are declared insufficient.

## 8. Outputs and execution requirements

The O1 job must:

- run from a clean, exact Paper 0 commit on Rocky 9;
- verify every data, config, checkpoint, and source hash before model loading;
- refuse to overwrite an existing job directory;
- print Python, PyTorch, CUDA, GPU, and package identities;
- record exact command, SLURM job, host, chunk size, timestamps, and dirty state;
- save compact raw per-mode and per-block metric tables before summaries;
- save no reconstructed 2.2 GB trajectory unless separately authorized;
- write one machine-readable result from which O1 tables and figures can be
  reproduced without rerunning inference;
- never discover, inspect, hash, or read a path for `85606`.

Any failed attempt retains its unique log and is entered in the execution
ledger. A code or metadata fix is committed before resubmission; thresholds are
not altered after results are seen.
