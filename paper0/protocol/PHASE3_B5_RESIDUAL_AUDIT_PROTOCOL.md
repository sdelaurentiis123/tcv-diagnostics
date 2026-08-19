# Phase 3 B5 deterministic-mean residual audit protocol

**Decision status:** frozen after the completed B4 failure and before B5
residual-audit implementation, execution, architecture selection, or training

**Development simulation:** TCV/Hermes 85604

**Sequestered simulation:** 85606

**85606 access authorized:** no

**Newly authorized scope:** implement and run one training-region-only audit of
the exact frozen C5P-H1 seed-1701 one-step residual

The machine-readable authority is
`paper0/manifests/phase3_b5_residual_audit_85604.json`.

## 1. Why an audit precedes B5

B3 and B4 fail for different reasons. B3 functional noise creates useful
marginal field spread but not calibrated mode-resolved or nonlinear-transport
covariance. B4 latent refinement improves the one-step ensemble mean but has
aggregate field spread--skill `0.217`, fails its spectral and transport repair
test, and fails both H-det and H-prob. Neither result supports replication,
O3, assimilation, or access to 85606.

Paper 0 predeclares B5 as a deterministic mean plus one joint stochastic
residual:

\[
x_t = \mu_{\mathrm{H1}}(x_{t-1})
      + r_\phi\!\left(x_{t-1},\mu_{\mathrm{H1}}(x_{t-1}),\epsilon\right).
\]

The next responsible step is not to choose a diffusion network by analogy. We
must first measure the target residual that such a network would learn. The
audit asks whether that residual is local or domain-scale, independent or
cross-field coupled, rapidly varying or temporally persistent, approximately
homoscedastic or strongly geometry dependent, and small enough for a
field-coordinate generator to be computationally credible.

This decomposition is motivated by CorrDiff, which trains a deterministic
regression model and then a diffusion model for its residual. CorrDiff reports
improved spectra and distributions but explicitly says calibration remains
challenging. Its official implementation recommends that any training patch
span at least the measured residual autocorrelation distance. These are
motivations for measurement, not evidence that B5 will work here.

Primary references are:

- Mardani et al., *Residual Corrective Diffusion Modeling for Km-scale
  Atmospheric Downscaling*, <https://arxiv.org/abs/2309.15214>;
- the official NVIDIA PhysicsNeMo CorrDiff implementation and residual-patch
  guidance,
  <https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html>;
- Price et al., *Probabilistic weather forecasting with machine learning*
  (GenCast), <https://www.nature.com/articles/s41586-024-08252-9>.

GenCast supports the usefulness of joint multi-variable residual diffusion,
but it models residuals from the preceding state using two context states. It
does not validate this H1 deterministic-mean decomposition. CorrDiff is a
downscaling system trained with orders of magnitude more samples. The current
430 adjacent training targets are not 430 independent physical simulations.

## 2. Exact data and parent

The audit uses only the verified job-6893525 85604 model dataset:

- fields: `[Ne,Pe,Pi,phi,Vi]`;
- frozen training frames: `[0,432)`;
- audited targets: every integer in `[2,432)`, exactly once and in order;
- context for target `t`: only standardized field frame `t-1`;
- target: standardized field frame `t`;
- target count: 430;
- guard frames `[432,496)`: unread;
- validation frames `[496,624)`: unread;
- saved cadence: `3.131905426352636` microseconds;
- frozen diagnostic-only training decorrelation reference: `2.2443947105846638`
  frames (`7.0292319729572625` microseconds), from the median finite `1/e`
  crossing across the five C5 training fields in Phase 1 job `6890606`;
- standardized field shape: `[5,64,32,88]`;
- simulated toroidal periodicity: `zperiod=5`;
- stored/full-torus Fourier mapping: `n=5k`.

The exact parent is seed-1701 C5P-H1 checkpoint
`5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e`.
It remains in evaluation mode and is neither retrained nor reselected. Its
frozen C5P-dcae_l10 codec has SHA-256
`9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d`.

Absolute time, normalized frame index, shot label, diagnostics, target truth,
and physics-derived quantities are not model inputs. Time remains metadata.
The five C5P state fields are the only dynamic inputs.

The H1 parent was trained on these same 430 targets. These are therefore
in-sample residuals, suitable for defining B5 training targets and sizing an
architecture but not for estimating unbiased forecast skill or irreducible
aleatoric uncertainty.

## 3. Truth-separated residual construction

First generate and close a deterministic H1 forecast artifact using context
only. Hash the complete artifact before any target truth is opened by the audit
scorer. Then define, in decoded standardized-field coordinates,

\[
r_{t,c,x,y,z}
=x_{t,c,x,y,z}
-\mu_{\mathrm{H1}}(x_{t-1})_{c,x,y,z}.
\]

This sign convention is immutable: truth minus deterministic mean. The
decoded standardized coordinates include the existing training-only field
transforms and scalar normalizations. They are field coordinates rather than
DCAE latent coordinates. Physical-unit summaries may be reported after the
frozen inverse normalization, but physical quantities do not become training
losses.

The audit must preserve a potentially nonzero residual mean. It may center a
copy for covariance calculations, but it may not redefine the future B5 target
as intrinsically zero mean.

Define the axisymmetric training residual bias

\[
b_{c,x,y}=\frac{1}{T Z}\sum_{t=1}^{T}\sum_{z=1}^{Z}
r_{t,c,x,y,z},
\]

and the fluctuation residual

\[
r'_{t,c,x,y,z}=r_{t,c,x,y,z}-b_{c,x,y}.
\]

The bias field and fluctuation statistics must be reported separately.

## 4. Required audit measurements

### 4.1 Scale, bias, and heteroscedasticity

For every field, report standardized residual bias, MAE, RMS, population
standard deviation, absolute-value quantiles, maximum, and residual-to-target
variance ratio. Report the same basic scale summaries under every authoritative
geometry mask.

At each `(c,x,y)`, calculate fluctuation standard deviation over target and
toroidal position. Report its spatial minimum, median, 90th, 95th, and maximum
and the ratio of the 95th to 5th percentile. These statistics determine
whether one global per-field residual scale is plausible; they do not by
themselves select a normalization.

### 4.2 Spatial autocorrelation

For the periodic toroidal axis, calculate the pooled circular autocorrelation

\[
\rho_{c,z}(\ell)=
\frac{\sum r'_{t,c,x,y,z}\,
r'_{t,c,x,y,(z+\ell)\bmod Z}}
{\sum (r'_{t,c,x,y,z})^2}.
\]

For `x` and `y`, use the analogous valid-overlap, nonperiodic estimator and
normalize by the corresponding zero-lag energy. Report every lag through half
the axis extent. For each curve report:

- the first nonpositive lag;
- the first lag at or below `1/e`;
- the first stable near-zero lag, defined as three consecutive lags with
  absolute correlation at most `0.10`;
- a positive-lobe integral scale;
- whether any distance is censored by the domain boundary.

Raw accumulator values and normalized curves must be stored so the result can
be recomputed without inference.

### 4.3 Temporal dependence

For lags 1 through 64 saved frames, compute the normalized pooled pattern
correlation of `r'_t` with `r'_{t+lag}` separately for every field. Report the
same crossing and integral summaries in frames, microseconds, and the frozen
training-only decorrelation-time units where applicable. Also compute the
autocorrelation of per-frame residual RMS.

This describes residual persistence. It does not identify irreducible noise:
with one realized trajectory per context, deterministic model error and
conditional stochastic variability cannot be separated uniquely.

The decorrelation-time conversion is a reporting unit only. The Phase 1
steady-state gate did not pass, so this value is not evidence that the later
training targets are independent or samples from one stationary distribution.

### 4.4 Cross-field dependence

Calculate the five-by-five zero-lag correlation matrix

\[
C_{ij}=\frac{\mathbb{E}[r'_i r'_j]}
{\sqrt{\mathbb{E}[(r'_i)^2]\,\mathbb{E}[(r'_j)^2]}}.
\]

Report its eigenvalues, effective rank, and the corresponding matrices under
the primary geometry masks. This measurement cannot authorize independent
per-field residual models as the primary method: B5 remains joint across all
five fields. Independent field generators are only a later ablation.

### 4.5 Toroidal support

Using the verified `n=5k` mapping, report residual and truth power fractions
for `k=0`, `k=1..3`, `k=4..5`, `k=6..7`, and `k>=8`, plus residual-to-truth
power in each band. Spectra are audit and evaluation quantities only. They are
not training losses.

The complete stored toroidal extent of 88 cells is mandatory in the primary
B5 field-coordinate model, regardless of a short estimated autocorrelation,
because the material `k=1` structure spans the stored periodic domain. B5 may
patch only the two nonperiodic axes.

## 5. Predeclared architecture implications

The audit does not select or train a network. It freezes the following rules
for the subsequent implementation protocol:

1. The initial deterministic mean remains the exact frozen H1 predictor.
2. The primary residual target is joint across all five standardized decoded
   fields and may have nonzero conditional mean.
3. The primary representation is field-coordinate, not DCAE latent.
4. The full toroidal axis is retained with periodic treatment.
5. For a nonperiodic axis with stable near-zero distance `d`, a candidate
   patch must span at least `min(domain_extent, 2d+1)` cells and inference
   overlap must be at least `d`. If no stable distance exists, that full axis
   is required.
6. If a full-field model fits the H100 smoke budget, it is preferred over
   patch fusion. If it does not, only `x/y` patches satisfying rule 5 may be
   tested.
7. A latent residual branch remains closed unless field-coordinate smoke is
   infeasible and a separately frozen residual-specific codec oracle passes.
8. No observed audit statistic may be converted into a physics-derived
   training loss.

The later B5 implementation protocol must freeze the diffusion or flow
parameterization, residual normalization, conditioning tensor, noise schedule,
training budget, checkpoint rule, ensemble size, and one-step acceptance gate
before any validation forecast is scored.

## 6. Execution and outputs

One Rocky 9 H100 job is authorized. It must:

1. verify the clean expected Git commit and all input hashes;
2. stage only the 85604 model dataset to node-local storage;
3. run the complete test suite;
4. generate the 430 context-only H1 forecasts in canonical order;
5. close and hash the forecast artifact;
6. only then open training target truth and compute the frozen audit;
7. write raw accumulators, compact metrics, plots, runtime, and an artifact
   hash index;
8. finish an online W&B audit run while keeping local immutable files as the
   scientific authority.

Required outputs are:

- `h1_training_forecast.h5`;
- `generation.json`;
- `residual_audit.json`;
- machine-readable ACF curves and cross-field matrices;
- labeled plots for scale, spatial ACF, temporal ACF, cross-field correlation,
  and toroidal support;
- `wandb.json`, environment, command, Slurm, test, and compute records;
- `result.json` and `artifact_sha256.txt`.

The job may not train a B5 model, alter H1, read validation or guard frames,
open 85606, launch O3, assimilate diagnostics, or rank diagnostics.

## 7. Post-audit boundary

Audit completion authorizes only interpretation and preparation of a separate
B5 implementation/smoke protocol. It does not authorize B5 training. B4
replication, O3, assimilation, diagnostic ranking, and 85606 remain closed.

If the residual is domain-scale in either nonperiodic axis and a full-field
model cannot fit a bounded H100 smoke, report that limitation before changing
representation. If the residual shows strong cross-field or temporal
dependence, report it explicitly rather than calling the residual independent
aleatoric noise.
