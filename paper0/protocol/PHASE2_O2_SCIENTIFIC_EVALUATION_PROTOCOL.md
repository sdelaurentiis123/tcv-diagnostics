# Phase 2 C5P O2 scientific-evaluation protocol

**Decision status:** frozen after all six selected checkpoints were immutable
and before any O2 reference forecast, spectral metric, cross-field metric,
transport metric, or scientific O2 gate was evaluated

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Authorized scope:** deterministic, teacher-forced, one-saved-step evaluation
of the already-trained `C5P-H1` and `C5P-H2` models

This document formalizes the evaluation details inherited from
`PHASE2_MATCHED_O1_O2_PROTOCOL.md` and
`PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md`. It does not change a threshold,
checkpoint, seed, field, validation target, or arm.

## 1. Immutable training evidence

Rocky 9 job `6895637` completed the separate checkpoint freeze:

~~~text
paper0/results/phase2_o2_training_freeze_6895637.json
sha256 dd8951e39e60d1631866ebe7af7c4d529ad543daf211233369b8fec9936ee837
~~~

The freeze verified the complete 200-epoch histories, earliest-minimum
checkpoint rule, latent normalizations, codec identities, artifact indices,
online W&B completion, and checkpoint reload identity for all six runs.

The selected checkpoints are fixed:

| Run | Arm | Seed | Epoch | Checkpoint SHA-256 |
|---:|---|---:|---:|---|
| 0 | `C5P-H1` | 1701 | 193 | `5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e` |
| 1 | `C5P-H1` | 1702 | 198 | `d15c74717fad6a3ccf5b5af895e3eefb7271667f4bbde2164514a61a526bc0e8` |
| 2 | `C5P-H1` | 1703 | 198 | `a718b2135c7019d05541bd5ffb029ce9408df8225603cffc957c42d2ce5abae3` |
| 3 | `C5P-H2` | 1701 | 198 | `3b971b2081901469e1f98adbe27b5cdbf3281d08a99ee28e0d8d8b1577722a84` |
| 4 | `C5P-H2` | 1702 | 199 | `5edc3e002730eb78232967255cfab66ee860b8b3858eed007f7061341b5c36eb` |
| 5 | `C5P-H2` | 1703 | 191 | `a70bd271117f1b0afb21258e4c5d7d4eb4919dc4a528509ccbf6ac2464622d85` |

The training-only selected validation MAE is slightly lower on average for
H1 than H2, but it cannot select an arm. No reference or physics metric has
yet been used.

## 2. Evaluation interval and separation of prediction from truth

Full scientific evaluation uses the 126 target frames `[498,624)` exactly
once, in chronological order. The blocks are fixed as:

~~~text
[498,519), [519,540), [540,561),
[561,582), [582,603), [603,624)
~~~

Each block has 21 targets. A target at frame `t` uses context `t-1` for H1
and ordered context `[t-2,t-1]` for H2. Absolute time is absent.

Prediction and scoring are separate operations:

1. A context-only loader reads only context fields and the integer target
   identity. It does not open the target field datasets.
2. The model or reference writes an immutable standardized forecast artifact.
3. The scorer opens that completed forecast artifact and only then loads truth
   for field and physics evaluation.

The forecast artifact has axes:

~~~text
[target_frame, channel, x, y, stored_toroidal_z]
~~~

with shape `[126,5,64,32,88]`, `float32`, horizon one, ensemble size one,
channels `[Ne,Pe,Pi,phi,Vi]`, and `zperiod=5`.

No target reset, clipping, future truth, guard frame, or `85606` path is
allowed during forecast generation.

## 3. Uncompressed references

References operate in the same standardized full-field coordinates as the
learned model targets and are never passed through a learned codec.

### Persistence

For both arms:

\[
\widehat{x}_{t}=x_{t-1}.
\]

### Two-frame linear extrapolation

For H2 only:

\[
\widehat{x}_{t}=2x_{t-1}-x_{t-2}.
\]

### Training-only toroidal spectral AR(1)

For each channel and stored Fourier index `k`, fit one complex coefficient
from the 431 training pairs `0->1` through `430->431`, pooling training time,
`x`, and `y`:

\[
a_{c,k}
=
\frac{
\sum x_{t,c,k}\,x_{t-1,c,k}^{*}
}{
(1+10^{-8})\sum |x_{t-1,c,k}|^2
}.
\]

Zero-power modes receive coefficient zero. The prediction is

\[
\widehat{x}_{t,c,k}=a_{c,k}x_{t-1,c,k}.
\]

There is no C5P boundary field, so the previously specified `Bphi` AR fit is
inapplicable to this continuation. Coefficients and sufficient statistics are
saved. Validation does not tune a coefficient or choose a reference.

The best applicable aggregate-RMSE reference and best applicable
aggregate-MAE reference are chosen separately and named explicitly. H1 uses
`[persistence, spectral_ar1]`; H2 additionally uses linear extrapolation.

## 4. Field metrics

All gate field metrics use the frozen standardized coordinates. For field
`c`, pool all target frames and spatial cells in the applicable interval.
With error `e = forecast - truth`:

\[
\operatorname{RMSE}_c
=
\sqrt{\langle e_c^2\rangle},
\qquad
\operatorname{MAE}_c
=
\langle |e_c|\rangle,
\qquad
\operatorname{bias}_c
=
\langle e_c\rangle.
\]

The variance ratio is the candidate population variance about its own pooled
mean divided by the truth population variance about its own pooled mean.

The aggregate equal-channel RMSE and MAE pool the same number of cells for
all five channels:

\[
\operatorname{RMSE}_{\rm agg}
=
\sqrt{
\frac{1}{5}\sum_c \langle e_c^2\rangle
},
\qquad
\operatorname{MAE}_{\rm agg}
=
\frac{1}{5}\sum_c \langle |e_c|\rangle.
\]

### Anomaly correlation

Anomalies are defined relative to the training-only normalization mean.
Standardized arrays are therefore already anomaly fields. Validation means
are not subtracted a second time:

\[
\operatorname{ACC}_c
=
\frac{
\sum x^{s}_{c}\widehat{x}^{s}_{c}
}{
\sqrt{
\sum (x^{s}_{c})^2
\sum (\widehat{x}^{s}_{c})^2
}
}.
\]

ACC is reported but has no O2 acceptance threshold.

## 5. Toroidal spectra and cross-field metrics

The Fourier transform is along the 88-point stored periodic toroidal axis.
The simulated wedge is one fifth of the full torus:

\[
n=5k.
\]

Physical-coordinate spectra use the inherited real-FFT Parseval weights.
For each field, forecast/truth realization coherence is

\[
\gamma^2_{x\widehat{x}}(k)
=
\frac{
|\sum x_k\widehat{x}_k^{*}|^2
}{
\left(\sum |x_k|^2\right)
\left(\sum |\widehat{x}_k|^2\right)
},
\]

where sums pool target time, `x`, and `y`.

The inherited bands are `k=1..3`, `k=4..5`, and `k=6..7`, corresponding to
full-torus mode ranges `n=5..15`, `n=20..25`, and `n=30..35`. A field band is
material only if it contains at least 1% of that field's non-axisymmetric
power on training frames `[0,432)`. Validation truth cannot select a band.

Primary physical-coordinate cross spectra use `(Ne,phi)`, `(Pe,phi)`, and
`(Pi,phi)`:

\[
S_{ab}(k)=\sum a_k b_k^{*},
\qquad
\gamma^2_{ab}(k)
=
\frac{|S_{ab}(k)|^2}{S_{aa}(k)S_{bb}(k)},
\qquad
\varphi_{ab}(k)=\arg S_{ab}(k).
\]

Cross-band materiality uses the inherited 1% threshold on training-truth
non-axisymmetric cross amplitude. Phase error is circular and weighted by
training-relevant truth cross amplitude. Coherence change is absolute and
weighted by the same truth cross amplitude.

## 6. Descriptive one-step mode lifetime

The original protocol requested a one-step mode-lifetime report but did not
attach an acceptance threshold. Define the normalized complex lag
correlation from the latest context to target or forecast:

\[
\rho_c(k)
=
\frac{
\sum x_{t,c,k}x_{t-1,c,k}^{*}
}{
\sqrt{
\sum |x_{t,c,k}|^2
\sum |x_{t-1,c,k}|^2
}
}.
\]

Under a purely exponential decorrelation interpretation, report

\[
\tau_c(k)
=
-\frac{\Delta t}{\log |\rho_c(k)|}
\]

only when `0 < |rho| < 1`; otherwise the finite e-folding estimate is
undefined and recorded as null rather than clipped. The cadence is
`3.131905426352636` microseconds. This diagnostic is descriptive and cannot
change the O2 decision.

## 7. Authoritative transport evaluation

Transport is never a training loss. For each target forecast independently:

1. inverse-transform the five model fields without clipping;
2. periodically resample `[Ne,Pe,Pi,phi]` from 88 to the authoritative native
   81-point toroidal grid using the frozen unwindowed SciPy Fourier resampler
   and store float32;
3. compare with direct-pressure native Hermes truth;
4. apply the authoritative geometry-aware radial ExB face operator separately
   to truth and forecast;
5. only then reduce strict-face and confined-separatrix transport.

The four quantities are particle, electron internal energy, ion internal
energy, and total internal energy transport. No transport is computed from a
mean of fields. The deterministic ensemble has one member.

## 8. Per-seed O2 gate

Every required check applies over all 126 targets. Items 2 through 6 also
apply chronologically by block. A component satisfies the temporal rule only
if its applicable threshold passes in at least five of six blocks. Different
components cannot rescue each other by averaging.

1. Aggregate RMSE and aggregate MAE are each strictly below their best
   applicable uncompressed reference.
2. At least four of five fields have RMSE strictly below persistence, and no
   field exceeds `1.05 * persistence_RMSE`.
3. Every training-material field band has power ratio in `[0.75,1.30]` and
   truth/forecast realization coherence at least `0.80`.
4. Every training-material primary cross-field band has absolute cross-phase
   error at most 20 degrees and absolute coherence change at most `0.15`.
5. For all four transport quantities, strict-face relative L2 is at most
   `0.40`, Pearson correlation is at least `0.70`, and truth-magnitude-weighted
   sign disagreement is at most `0.20`.
6. For all four transport quantities, separatrix relative L2 is at most
   `0.30`, absolute normalized bias is at most `0.15`, Pearson correlation is
   at least `0.80`, and truth-magnitude-weighted sign disagreement is at most
   `0.15`.
7. Each component in items 2--6 passes its applicable block thresholds in at
   least five of six blocks.
8. Every required metric is finite and every required correlation and sign
   statistic is defined.
9. Provenance confirms no forbidden data access, target read during forecast
   generation, clipping, target reset, absolute time input, or physics loss.

A seed passes only when the complete conjunction passes. An arm passes only
when seeds 1701, 1702, and 1703 all pass independently. Seed averaging cannot
rescue a failed seed.

If neither arm passes, stop and report deterministic one-step failure. If one
passes, it is the sole candidate for a newly frozen O3 protocol. If both pass,
retain both through the first newly frozen short O3 comparison. This
evaluation never launches O3 itself.

## 9. Execution and artifact rules

Before the full matrix:

1. run the complete local and Rocky 9 CPU suite;
2. run a non-scientific Rocky 9 smoke on targets `[498,502)` using one frozen
   checkpoint and all three references;
3. verify context-only prediction, forecast hashes, separate truth scoring,
   native transport, and artifact schemas;
4. commit the exact implementation and launcher.

The full run uses Rocky 9 and the exact clean commit. Large forecasts remain
outside Git. Every reference and learned run stores forecast, generation
record, raw score, result, command, environment, Slurm allocation, and
artifact index. The compact final matrix and hashes are tracked in Git.

No diffusion, FGN, PDE-Refiner, stochastic residual model, O3 rollout,
assimilation, diagnostic ranking, or `85606` evaluation is authorized here.
