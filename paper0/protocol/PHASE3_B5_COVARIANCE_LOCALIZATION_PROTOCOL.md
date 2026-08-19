# Phase 3 B5 covariance and conditionality localization protocol

**Decision status:** frozen after the completed B5 seed-1701 one-step gate
failure and before localization implementation or execution

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** one read-only analysis of the immutable B5 M32
validation forecast, frozen H1 validation mean, 85604 validation truth, and
training-only H1 residual sufficient statistics

The machine-readable authority is
`paper0/manifests/phase3_b5_covariance_localization_85604.json`.

## 1. Why this analysis comes next

The B5 joint field-residual EDM improved the one-step ensemble mean and the
aggregate fair CRPS. Its global corrected spread--skill ratio is about 0.80,
and the pooled strict-face transport spread--skill values are close to one.
Nevertheless, mode-resolved realization coherence, cross-field projection
calibration, and integrated separatrix-transport calibration all failed. The
separatrix spread--skill ratios are only about 0.41--0.49.

Those observations distinguish neither of the following explanations:

1. B5 produces too little stochastic amplitude everywhere.
2. B5 produces enough local variance but arranges it with the wrong spatial,
   modal, or cross-field covariance, so coherent projections cancel.

Scalar inflation could address the first failure but cannot repair the
second. Additional training, sampler tuning, O3 rollout, assimilation, and
diagnostic ranking therefore remain closed until the failure is localized.

This distinction is also required statistically. Marginal CRPS assesses the
one-dimensional predictive marginals. It does not establish the joint copula
or the covariance of a high-dimensional field. Scheuerer and Hamill introduced
the variogram score specifically because the multivariate energy score has
limited sensitivity to misspecified correlation, and Wilks found that no
single multivariate calibration diagnostic detects every covariance defect.
Allen, Ziegel, and Ginsbourger likewise recommend complementary, interpretable
pre-rank summaries for location, scale, and dependence.

Primary statistical references are:

- Scheuerer and Hamill, *Variogram-Based Proper Scoring Rules for
  Probabilistic Forecasts of Multivariate Quantities*,
  <https://doi.org/10.1175/MWR-D-14-00269.1>;
- Wilks, *On assessing calibration of multivariate ensemble forecasts*,
  <https://doi.org/10.1002/qj.2906>;
- Allen, Ziegel, and Ginsbourger, *Assessing the calibration of multivariate
  probabilistic forecasts*, <https://arxiv.org/abs/2307.05846>.

These sources motivate dependence-sensitive evaluation. They do not validate
B5, tokamak transport, or any particular next architecture.

## 2. Immutable inputs

The localization uses only these already completed artifacts:

- B5 M32 one-step standardized forecast, targets `[498,624)`, job `6901587`,
  SHA-256
  `1a5f3ea7e0d1722363205be569d2db60905cdda798b4597a6c47e74d99fab68b`;
- frozen C5P-H1 seed-1701 standardized validation forecast, targets
  `[498,624)`, SHA-256
  `a76b389dfc090d1d3061ec4be3526c524645b4cd9cec2f309601af0dd405f1ed`;
- B5 score SHA-256
  `c81c0e06313c652816be77025c2b42bbfce10728df7ac14787e00edf7d978ba6`;
- B5 final gate SHA-256
  `a1d9cf00de0a2b0b3cc0c13d31c727420214040dcbf575afa67c6ae64015974b`;
- training-only H1 residual audit SHA-256
  `d8f05af4d1619fbe7ad8774360c29aa38cd69494d80624d55e30fdb57361cf67`;
- training-only H1 residual sufficient statistics SHA-256
  `50c54a8e9dd0f0983cb8360f598bdf00eae22854de2ab471cd7385e767f3058b`;
- model-data manifest, training-only normalization, native-truth catalog, and
  authoritative geometry already frozen for the B5 evaluation.

All files are hash-checked before analysis. The B5 and H1 forecasts are read
without modification. No checkpoint is loaded and no model inference occurs.

The fields are `[Ne,Pe,Pi,phi,Vi]`, the standardized model volume is
`[5,64,32,88]`, the cadence is `3.131905426352636` microseconds, the simulated
toroidal periodicity is `zperiod=5`, and stored Fourier index `k` maps to
full-torus mode number `n=5k`.

## 3. Frozen random objects and terminology

For validation target `t` and ensemble member `m`, let

\[
x_t
\]

be the standardized truth,

\[
\mu_t
\]

the frozen deterministic H1 forecast, and

\[
X_{t,m}
\]

the stored B5 forecast member. Define the realized H1 residual,

\[
R^{\mathrm{H1}}_t=x_t-\mu_t,
\]

the B5-generated residual,

\[
R^{\mathrm{B5}}_{t,m}=X_{t,m}-\mu_t,
\]

the B5 ensemble mean,

\[
\bar X_t=\frac{1}{M}\sum_{m=1}^{M}X_{t,m},
\]

the ensemble anomaly,

\[
A_{t,m}=X_{t,m}-\bar X_t,
\]

and the realized innovation against the B5 mean,

\[
D_t=x_t-\bar X_t.
\]

Here `residual` means truth minus H1. `Generated residual` means a B5 member
minus H1. `Anomaly` means a B5 member minus the B5 ensemble mean. `Innovation`
means truth minus the B5 ensemble mean. These terms must not be interchanged.

With one realized truth per context, neither
`R^{H1}` nor `D` is an unbiased direct observation of the conditional
covariance. Both mix stochastic variability, model error, conditional bias,
and within-run distribution shift. The analysis therefore localizes
inconsistency; it cannot identify irreducible aleatoric noise.

For field-domain covariance diagnostics, subtract the full spatial mean from
`phi` separately for every truth, H1 forecast, B5 member, and B5 ensemble mean.
Then subtract an axisymmetric bias `mean(target,z)` from the realized H1
residual and innovation before computing fluctuation correlations. B5
anomalies already have zero conditional ensemble mean; only numerical closure
is checked. Transport uses the original physical fields because the
authoritative radial `E cross B` operator is invariant to a constant potential
offset.

## 4. Required integrity anchors

Before producing a new diagnostic, the analysis must verify:

1. both forecast files are complete and have the exact target order
   `498,...,623`;
2. B5 axes are
   `[target,member,future,field,x,y,stored_toroidal_z]` with shape
   `[126,32,1,5,64,32,88]`;
3. H1 axes are `[target,field,x,y,stored_toroidal_z]` with shape
   `[126,5,64,32,88]`;
4. no artifact metadata reports use of absolute time, future truth, guard
   frames, post-hoc calibration, or 85606;
5. the B5 ensemble mean and spread recompute the already stored aggregate
   field RMSE and corrected spread--skill values within numerical tolerance;
6. `sum_m A_{t,m}` is zero to floating-point tolerance;
7. every nonlinear transport quantity is evaluated member by member before
   ensemble reduction;
8. no path, metadata field, or input hash mentions an authorized 85606 read.

Failure of an integrity anchor stops the analysis without scientific output.

## 5. Field covariance localization

### 5.1 Spatial correlation

For each field and for axes `x`, `y`, and stored toroidal `z`, compute the same
pooled normalized correlation estimator frozen by the training-residual audit:

\[
\rho(\ell)=
\frac{\sum u_i u_{i+\ell}}
{\sqrt{\sum u_i^2\sum u_{i+\ell}^2}}.
\]

Use valid overlap for `x` and `y`, circular pairing for `z`, and lags through
half the corresponding axis. Compute curves for:

- training H1 residual fluctuations, read from the frozen sufficient
  statistics;
- validation H1 residual fluctuations;
- B5 ensemble anomalies, pooled over targets and members;
- validation B5 innovations.

Report the first nonpositive lag, first `1/e` crossing, first three-lag stable
near-zero crossing at `|rho| <= 0.10`, and positive-lobe integral scale. Also
report the root-mean-square curve distance excluding lag zero,

\[
d_{\rho}(a,b)=
\sqrt{\frac{1}{L}\sum_{\ell=1}^{L}
\left(\rho_a(\ell)-\rho_b(\ell)\right)^2}.
\]

The training-to-validation H1-residual distance is the empirical within-run
drift reference. It is not a sampling distribution.

### 5.2 Cross-field correlation

For the full domain and every authoritative 2-D geometry mask, calculate the
five-by-five pooled correlation matrix of the same four objects. For two
correlation matrices define the off-diagonal RMS distance

\[
d_C(C_1,C_2)=
\sqrt{\frac{2}{p(p-1)}
\sum_{i<j}\left(C_{1,ij}-C_{2,ij}\right)^2},
\qquad p=5.
\]

Report eigenvalues, entropy effective rank, participation-ratio effective
rank, and all pairwise distances to the validation H1 residual and innovation.
No five-dimensional rank is interpreted as the effective rank of the full
field covariance.

### 5.3 Toroidal support

Using `n=5k`, report Parseval-weighted power for `k=0`, `k=1..3`, `k=4..5`,
`k=6..7`, and `k>=8` for validation H1 residuals, B5 anomalies, and B5
innovations. Report both band fractions and absolute power ratios. This is an
evaluation diagnostic, never a loss.

## 6. Dependence-sensitive scores

Marginal fair CRPS is retained as an anchor but is not reranked or retuned.
The new analysis adds a variogram score only for predeclared low-dimensional
or structured projections. For outcome vector `y`, ensemble members `X_m`,
order `p=1`, and nonnegative fixed pair weights `w_ij`,

\[
\mathrm{VS}(F,y)=
\sum_{i<j}w_{ij}
\left(
|y_i-y_j|
-\frac{1}{M}\sum_{m=1}^{M}|X_{m,i}-X_{m,j}|
\right)^2.
\]

Compute it separately for:

1. the five standardized fields at the same cell, with all ten field pairs
   equally weighted, pooled separately by authoritative geometry region;
2. exact-separatrix local transport contributions along periodic toroidal
   lags `1,2,4,8,16,32,40`, with equal weight within each lag and each of the
   four transport quantities.

Report per-target values and the six frozen 21-frame block means. Do not
collapse field and transport variogram scores into one scalar. The score is
not a training loss and has no pass threshold in this localization.

## 7. Exact-separatrix transport covariance decomposition

Use the authoritative native-81 geometry, direct pressure fields, shifted
`y` derivative, periodic resampling, outward sign, and member-wise nonlinear
operator already frozen for B2--B5. For each of particle, electron internal
energy, ion internal energy, and total internal energy transport, retain the
weighted local contribution `q_{t,m,j}` at every exact confined-separatrix
face sample `j`. Its sum must reproduce the existing integrated separatrix
transport to numerical tolerance.

Let `qbar` be the ensemble mean contribution and let `qtruth` be truth. Define

\[
V_{\mathrm{diag}}^{\mathrm{ens}}
=\frac{1}{T}\sum_t\sum_j
\operatorname{Var}_m(q_{t,m,j}),
\]

\[
V_{\mathrm{total}}^{\mathrm{ens}}
=\frac{1}{T}\sum_t
\operatorname{Var}_m\!\left(\sum_j q_{t,m,j}\right),
\]

and the ensemble coherence multiplier

\[
K_{\mathrm{ens}}=
\frac{V_{\mathrm{total}}^{\mathrm{ens}}}
{V_{\mathrm{diag}}^{\mathrm{ens}}}.
\]

For local innovation `e_{t,j}=q^{truth}_{t,j}-qbar_{t,j}`, report both the
uncentered mean-square analogue and the time-centered covariance analogue:

\[
K_{\mathrm{err}}=
\frac{\langle(\sum_j e_{t,j})^2\rangle_t}
{\langle\sum_j e_{t,j}^2\rangle_t},
\]

\[
K_{\mathrm{innov}}=
\frac{\operatorname{Var}_t(\sum_j e_{t,j})}
{\sum_j\operatorname{Var}_t(e_{t,j})}.
\]

Report diagonal and integrated corrected spread--skill ratios using the frozen
finite-member factor `(M+1)/M`, the off-diagonal contribution
`Vtotal-Vdiag`, toroidal correlation curves of local transport anomalies and
innovations, and the 16-by-16 correlation matrix of toroidally integrated
separatrix-row contributions. The row coordinate is a topology index, not an
assumed Euclidean distance.

As a diagnostic counterfactual only, calculate the scalar anomaly factor that
would force each integrated spread--skill ratio to one:

\[
\alpha_Q=1/\mathrm{SSR}_{Q,\mathrm{integrated}}.
\]

Report the exact-separatrix local spread--skill that the same factor would
produce. Do not write an inflated forecast, rescore it as a model, or use it
for downstream assimilation.

## 8. Conditionality and one-step history probe

The training residual audit found nonzero lag-one pattern correlations. To
test whether explicit residual history contains useful causal information,
construct one fixed deterministic diagnostic baseline.

For each field, calculate the scalar training-only coefficient

\[
a_c=
\frac{\sum R'_{t-1,c}R'_{t,c}}
{\sum (R'_{t-1,c})^2}
\]

from the frozen training sufficient statistics. Let `b_{cxy}` be the frozen
training axisymmetric residual bias. For validation targets `t=499,...,623`,
define

\[
\widehat R^{\mathrm{AR1}}_{t,cxyz}
=b_{cxy}
+a_c\left(R^{\mathrm{H1}}_{t-1,cxyz}-b_{cxy}\right),
\]

\[
\widehat x^{\mathrm{AR1}}_t
=\mu_t+\widehat R^{\mathrm{AR1}}_t.
\]

This is a teacher-forced causal diagnostic because the previous realized H1
residual is available only when the previous truth is available. It is not an
autonomous rollout and cannot establish controllability. Evaluate field RMSE,
MAE, bias, toroidal bands, and authoritative transport for this baseline,
H1, and the B5 ensemble mean on the identical 125 targets. Report every frozen
21-frame block after dropping target 498; the first partial comparison block
contains 20 targets and is labeled explicitly.

Also report Pearson and Spearman association across targets between predicted
per-target ensemble variance and squared ensemble-mean error for field RMS and
integrated transport. These associations measure flow dependence only; they
are not proof of calibration.

## 9. Frozen interpretation rules

These rules classify evidence; they are not a new scientific acceptance gate.

### L1: predominantly amplitude-limited

Support this label only if, for at least three of four transport quantities:

- exact-separatrix local and integrated corrected spread--skill are both
  below `0.67` or both above `1.50`; and
- `Kens/Kerr` lies in `[0.67,1.50]`.

### L2: covariance-organization limited

Support this label only if, for at least three of four transport quantities:

- exact-separatrix local corrected spread--skill lies in `[0.80,1.25]`;
- integrated corrected spread--skill is below `0.67`;
- `Kens/Kerr` is below `0.67`; and
- scalar matching of the integrated spread would make local spread--skill
  exceed `1.50`.

### L3: field-dependence mismatch beyond within-run drift

For each field/axis spatial curve and each regional cross-field matrix, compare
the B5-to-validation-residual distance with the training-to-validation
residual distance. Report the count and identities for which the B5 distance
is larger. Call the result systematic only when the direction holds in at
least five of the six frozen validation blocks. No universal threshold is
inferred from these counts.

### L4: explicit residual-history signal

Support this label only if the frozen AR1 residual baseline improves
equal-field standardized RMSE over H1 by at least two percent in the aggregate
and improves it in at least five of the six chronological comparisons. State
separately whether it beats the B5 ensemble mean. A positive L4 result
authorizes only a protocol proposal for explicit history conditioning; it does
not authorize training.

### L5: unresolved by one realized trajectory

Use this label whenever the training-to-validation residual drift is as large
as or larger than the B5 covariance discrepancy, block directions are
unstable, or the diagnostics disagree. The correct response is then more data
or a narrower claim, not retrospective threshold tuning.

## 10. Execution and outputs

One Rocky 9 CPU job on Rusty is authorized. It requests 16 CPU cores, 160 GB
memory, and at most four hours. It uses online W&B for monitoring, while
immutable Ceph outputs remain authoritative. The job may stage the existing
B5 forecast, H1 forecast, model-data shards, and compact residual statistics
to node-local storage after hash verification.

Required outputs are:

- `covariance_localization.json`;
- `raw_accumulators.npz` containing recomputable sufficient statistics;
- per-target and per-block tables;
- labeled plots for spatial ACF, cross-field matrices, toroidal power,
  separatrix covariance decomposition, variogram scores, and the history
  baseline;
- `wandb.json`, command, environment, Slurm, test, timing, and hash records;
- `result.json` and `artifact_sha256.txt`.

The job may not load a trainable checkpoint, perform model inference, alter a
forecast, apply inflation, tune a sampler, change an acceptance threshold,
train any model, run O3/O4/O5, assimilate observations, rank diagnostics,
steer, or access 85606.

## 11. Post-analysis boundary

Completion authorizes interpretation and one decision memo. Depending on the
frozen labels, that memo may propose exactly one next 85604 development
experiment, such as explicit residual history or a structured global-plus-
local covariance parameterization. The proposed experiment still requires a
separate committed protocol before implementation or execution.

Additional seeds, autonomous rollout, assimilation covariance, diagnostic
ranking, steering, and 85606 remain closed regardless of the localization
result.
