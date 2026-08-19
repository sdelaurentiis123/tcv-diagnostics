# Phase 3 residual KL representation-oracle protocol

**Decision status:** frozen after the completed B5 covariance localization and
before implementation or execution

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implement and run one training-fitted,
chronologically validated residual Karhunen--Loève representation oracle and
one training-rank-selected static Gaussian KL ensemble

The machine-readable authority is
`paper0/manifests/phase3_residual_kl_oracle_85604.json`.

## 1. Scientific question

B5's local exact-separatrix transport spread is calibrated, but its integrated
spread is only `0.413--0.485`. The frozen localization supports covariance
organization failure and rejects a predominantly scalar-amplitude failure.
Before training a low-rank stochastic head, a coefficient diffusion model, or
a dependence-sensitive objective, determine whether a compact residual
subspace can represent the missing dependence at all.

This protocol asks two ordered questions:

1. Can a basis fitted only to H1 training residuals reconstruct the joint
   five-field validation residual and its transport-relevant dependence?
2. If so, how much calibration is obtained by sampling a static Gaussian
   distribution in that basis without learned conditional coefficients?

The first tier is a truth-projected representation oracle. The second is a
causal one-step statistical baseline. They must never be conflated.

## 2. Immutable evidence and data

The experiment uses only already completed 85604 artifacts:

- H1 training forecast, targets `[2,432)`, job `6901393`, SHA-256
  `d1342e53ac5f9077665f7a707ca035b9069a616d7db10371e1f6c1ae504d9c18`;
- H1 validation forecast, targets `[498,624)`, job `6896117`, SHA-256
  `a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`;
- training residual audit, SHA-256
  `d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67`;
- training residual raw sufficient statistics, SHA-256
  `50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b`;
- model-data manifest SHA-256
  `27816929afde84b1666a15a06bc5dc7f8c82a9435078839c5641465275e4ec18`;
- training-only normalization SHA-256
  `f751b73601b625d4d32088d3c49b72afa106d2b680016ff4faf60ded0c71dbd7`;
- native-truth result SHA-256
  `cc6b00b72605c6457ae3a3b24db7e90dac16989c852df92036a57272f62eddae`;
- authoritative geometry SHA-256
  `0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8`;
- completed B5 covariance-localization result SHA-256
  `331e7f3ff5d221d0d3720d9112ce90436d8330647501a2268f974867bbc140d2`.

The fields are `[Ne,Pe,Pi,phi,Vi]`, standardized model shape is
`[5,64,32,88]`, cadence is `3.131905426352636` microseconds, toroidal
periodicity is `zperiod=5`, and stored Fourier index maps as `n=5k`.

The immutable chronological regions are:

| Region | Target frames | Count | Policy |
|---|---:|---:|---|
| training | `[2,432)` | 430 | fit bias, basis, eigenvalues, and rank selector |
| guard | `[432,496)` | 64 | unread |
| validation | `[498,624)` | 126 | projection and static-ensemble evaluation only |
| held-out | simulation 85606 | -- | unopened |

No checkpoint is loaded and no model inference is performed. The existing H1
forecasts are immutable inputs.

## 3. Residual and gauge convention

For target (t), define the decoded standardized H1 residual

\[
r_t=x_t-\mu_t,
\]

with immutable sign truth minus H1 mean.

For covariance fitting, subtract the full spatial mean of `phi` separately
for every truth and H1 forecast before forming `phi` residuals. Other fields
are unchanged. Then subtract the training-only axisymmetric bias

\[
b_{cxy}=\frac{1}{T Z}\sum_{t,z}r_{tcxyz}
\]

from every training fluctuation residual. The same frozen (b) is subtracted
from validation residuals for projection and added to static-ensemble means.
With the canonical `[field,x,y,z] = [5,64,32,88]` field volume, (b) has
shape `[5,64,32]`.

The axisymmetric bias does not, by itself, center every stored toroidal cell
over the finite set of 430 targets. Define

\[
q_t=r_t-b,
\qquad
\bar q=\frac{1}{T}\sum_t q_t,
\qquad
R_t=q_t-\bar q.
\]

Only (R) is used to estimate covariance. The finite-sample pattern
(\bar q), whose toroidal mean is zero by construction, is recorded but is
never added to a forecast mean. Treating it as a persistent mean would lock an
arbitrary empirical toroidal phase into every forecast. Validation
fluctuations remain (r_t-b) when they are projected onto the linear subspace;
validation data do not alter (\bar q), the basis, or the eigenvalues.

The bias and fluctuation subspace remain separate. The experiment may not
silently force the physical forecast residual mean to zero or silently treat
the finite-sample nonaxisymmetric pattern as a forecast bias.

Transport is evaluated on the original physical fields. A constant `phi`
offset is irrelevant to the authoritative radial (E\times B) operator.

## 4. Method-of-snapshots KL basis

Flatten the five equally standardized field volumes into one vector using
canonical field and spatial axis order. Let (R\in\mathbb{R}^{T\times D})
contain the 430 gauge-consistent training fluctuations after both frozen
axisymmetric-bias subtraction and the covariance-only empirical centering
defined in Section 3. Thus, the rows of (R) sum to zero and the maximum rank
is 429.

Compute the sample Gram matrix in float64 with one fixed streaming reduction:

\[
G=\frac{1}{T-1}RR^{\mathsf T}.
\]

Diagonalize (G=V\Lambda V^{\mathsf T}), sort eigenvalues descending, and
construct spatial modes by the method of snapshots:

\[
U_j=\frac{R^{\mathsf T}V_j}{\sqrt{(T-1)\lambda_j}}.
\]

Verify orthonormality in the frozen equal-field standardized inner product.
Numerically nonpositive eigenvalues are excluded using a relative threshold
of `1e-10` times the largest eigenvalue. The maximum centered rank is 429.

The predeclared reported rank ladder is:

~~~text
0, 8, 16, 32, 44, 64, 128, 256, full positive training rank
~~~

Ranks exceeding the numerical positive rank are reported as unavailable, not
silently clipped. Rank 44 is included only to make the scale visibly
comparable with the historical z44 discussion; it does not share z44's codec,
representation, or meaning.

Report training cumulative variance, per-field diagonal variance recovery,
orthogonality error, eigenvalue spectrum, participation-ratio rank, entropy
effective rank, the norm and per-field energy of the removed
covariance-centering pattern (\bar q), and basis storage/compute cost. These
training summaries may not inspect validation truth.

## 5. Training-only static-rank selection

Select exactly one static-ensemble rank before validation truth is opened.

Let (r_{90}) be the smallest available rank in
`[8,16,32,44,64,128]` whose cumulative training fluctuation variance is at
least `0.90`. If no listed rank reaches `0.90`, set `r_static=128` and label it
`training_variance_cap_bound`. Otherwise set `r_static=r90` and label it
`training_90_percent_rank`.

No validation, transport, spectrum, cross-field, or B5 metric may change this
rank. There is no post-validation rank selection.

## 6. Tier A: truth-projected representation oracle

For each validation residual fluctuation (r'_t=r_t-b), calculate

\[
\widetilde r^{(r)}_t=U_rU_r^{\mathsf T}r'_t
\]

for every available rank in the frozen ladder. The reconstructed oracle state
is

\[
\widetilde x^{(r)}_t=\mu_t+b+\widetilde r^{(r)}_t.
\]

This uses current target truth to obtain projection coefficients. It is an
upper bound on representation capacity, not a forecast, ensemble, or model.
It may not appear in a forecast-comparison table without an explicit `oracle`
label.

### Required Tier-A metrics

For every rank report:

- total and per-field validation residual variance captured;
- residual reconstruction RMSE and MAE relative to H1 residual scale;
- all spatial ACF curves and curve distances from the B5 localization;
- global and authoritative-region cross-field matrices and distances;
- toroidal residual power in `k=0`, `k=1..3`, `k=4..5`, `k=6..7`, and
  `k>=8`, always labeled with `n=5k`;
- cross-phase and cross-coherence for the nine frozen density/pressure--phi
  material-band pairs;
- authoritative strict-face and exact-separatrix particle, electron internal
  energy, ion internal energy, and total internal energy transport;
- all metrics by the same six 21-frame validation blocks used for B5.

Physics quantities are evaluation metrics only.

### Frozen representation-pass rule

A rank passes the representation oracle only if all four conditions hold:

1. at least `0.80` of total validation residual variance and at least `0.60`
   for every field are captured;
2. projection-to-validation dependence distance is no larger than the frozen
   training-to-validation drift for at least 9 of the 11 systematic L3
   identities, with that direction holding in at least five of six blocks;
3. projected-to-validation absolute power ratio is in `[0.80,1.20]` for at
   least 12 of 15 material field bands `k=1..7`;
4. reconstructed-state strict-face relative L2 is at most `0.40` for at least
   three of four transport quantities and exact-separatrix relative L2 is at
   most `0.30` for all four.

The minimum passing rank is reported. Thresholds may not be relaxed after
seeing validation.

## 7. Tier B: one static Gaussian KL ensemble

Tier B uses only `r_static` from Section 5. It is a condition-independent
covariance baseline around the frozen H1 mean, not a proposed final model.

Before validation truth is opened, write and hash:

- the training bias;
- the first `r_static` modes and eigenvalues;
- an immutable PCG64 seed bank with master seed `2026081901`, target/member
  shape `[126,32]`;
- exact H1 validation target order and hash;
- the formula and canonical axis metadata defining the compressed forecast.

These objects are an exact compressed forecast representation. A scorer may
reconstruct member (m) for target (t) as

\[
X^{\mathrm{KL}}_{t,m}
=
\mu_t+b+
\sum_{j=1}^{r_{\mathrm{static}}}
\sqrt{\lambda_j}\,z_{t,m,j}U_j,
\qquad
z_{t,m,j}\sim\mathcal N(0,1).
\]

Every (z) is generated deterministically from the stored target/member seed.
Validation truth, absolute time, diagnostics, guard frames, and 85606 cannot
enter this forecast.

### Required Tier-B metrics

Use the exact B5 M32 finite-member conventions and report:

- field RMSE, MAE, bias, fair CRPS, spread--skill, coverage, and rank/PIT
  diagnostics;
- member-expected toroidal power, realization coherence, power calibration,
  and cross-field complex projections;
- member-wise authoritative strict-face and exact-separatrix transport;
- local versus integrated transport covariance decomposition;
- field and transport variogram scores;
- spread--error association;
- all metrics in six frozen chronological blocks;
- parameter-free generation cost and compressed-basis storage.

Every nonlinear quantity is computed separately for each member before
ensemble reduction. Transport of ensemble-mean fields is a labeled companion
diagnostic only.

### Frozen static-covariance usefulness rule

Call the static covariance useful only if, for at least three of four
transport quantities:

- local corrected spread--skill lies in `[0.80,1.25]`;
- integrated corrected spread--skill is at least `0.67` and at most `1.50`;
- ensemble-to-error coherence multiplier ratio is at least `0.67`;
- the scalar factor that would match integrated spread would leave local
  spread--skill at most `1.50`;

and the aggregate field corrected spread--skill lies in `[0.80,1.25]` with
finite noncollapsed members. This is a diagnostic rule, not Paper 0 forecast
acceptance.

## 8. Frozen outcome classification

Exactly one primary outcome is emitted:

- `K1_compact_representation_static_covariance_useful`: minimum passing Tier-A
  rank is at most 64 and Tier B passes its usefulness rule;
- `K2_compact_representation_conditional_coefficients_required`: minimum
  passing Tier-A rank is at most 64 and Tier B fails;
- `K3_only_moderate_or_high_rank_adequate`: no rank at most 64 passes, but a
  rank of at least 128 passes;
- `K4_training_residual_span_does_not_transfer`: no available rank, including
  the full positive training span, passes.

If numerical failures prevent classification, report
`execution_failed_without_scientific_outcome`. Do not choose the nearest
scientific label.

Any logical edge case not covered above, such as Tier B passing while the
complete Tier-A span fails, is reported as
`inconsistent_diagnostic_requires_review`; it does not authorize training.

## 9. Integrity gates

Before scientific output, verify:

1. every immutable input and source hash;
2. exact target coverage and chronological order;
3. guard and 85606 paths are absent from opened-file logs;
4. training bias and legacy covariance anchors reproduce within their frozen
   tolerances;
5. the Gram matrix is symmetric and positive semidefinite to numerical
   tolerance;
6. basis orthonormality maximum error is at most `2e-5`;
7. rank reconstructions are nested and reconstruction error is nonincreasing;
8. rank-zero reconstruction exactly recovers the H1-plus-bias reference;
9. full-rank reconstruction of the covariance-centered training matrix (R)
   closes to `2e-5` relative RMS; this is not a claim that the removed
   finite-sample pattern (\bar q) is part of the forecast subspace;
10. the Tier-B compressed forecast is closed and hashed before validation
    truth is opened;
11. member seeds and reconstructed members are deterministic under fresh
    reload;
12. every transport sum reproduces its local contributions to `2e-12` relative
    tolerance;
13. no model checkpoint is loaded and no optimizer or trainable parameter is
    created.

Failure stops without an outcome label.

## 10. Execution and outputs

One Rocky 9 CPU job on Rusty is authorized, requesting at most 32 CPU cores,
256 GB memory, and six hours. Online W&B is required for compact monitoring.
Ceph artifacts remain authoritative. Raw fields, basis arrays, forecast
members, raw accumulators, figures, and tables may not be uploaded to W&B.

Required outputs are:

- `training_kl_basis.npz` or HDF5 with bias, modes, eigenvalues, and metadata;
- immutable Tier-B seed bank and compressed forecast manifest;
- rank-resolved projection metrics and block tables;
- Tier-B field, spectral, cross-field, calibration, and transport metrics;
- raw sufficient statistics required to reproduce every compact metric;
- labeled eigenvalue, reconstruction, covariance, toroidal, transport, and
  calibration figures;
- command, environment, Slurm, W&B, test, timing, and hash records;
- `result.json` and `artifact_sha256.txt`.

The implementation must have known-answer tests for method-of-snapshots PCA,
gauge handling, nested projections, static Gaussian covariance, fixed seeds,
member-wise nonlinear metrics, no-future-truth generation, and all scope
guards before submission.

## 11. Post-experiment boundary

Completion authorizes one interpretation memo and, only for K1 or K2, a
proposal for one conditional low-rank or global-plus-local stochastic model.
It does not authorize that model's implementation or training automatically.

K3 permits only a multiscale representation design memo with explicit compute
cost. K4 requires stopping architecture iteration until more simulator
trajectories are obtained or Paper 0 is narrowed.

All outcomes keep O3/O4/O5, additional model seeds, assimilation, diagnostic
ranking, steering, and 85606 closed.
