# Phase 2 metric and oracle protocol

**Protocol status:** frozen before implementation

**Development data:** synthetic known-answer arrays and TCV/Hermes `85604`

**Sequestered evaluation run:** `85606`

This document freezes the conventions for the Phase 2 diagnostic oracle
(`O0`) before the metric code is written or applied to a learned forecast. It
does not authorize model training, does not reopen the failed Phase 1 learning
gate, and does not authorize access to shot `85606`.

## 1. Canonical forecast semantics

All learned and reference forecasts must be adapted to the following axes:

```text
forecast: [batch, ensemble_member, future_time, channel, x, y, z]
truth:    [batch,                  future_time, channel, x, y, z]
```

The canonical names are `B, M, T, C, X, Y, Z`. The initial implementation
supports exactly three spatial axes so that accidental collapse or permutation
of the stored Hermes coordinates fails loudly. A deterministic forecast is
represented with `M = 1`; the member axis is never omitted. Forecast and truth
must have identical non-member dimensions, contain finite values, and have at
least one member and one future time.

Metrics return unreduced arrays where practical. Every reporting reduction
must state the axes and weights used. Uniform pixel averages and later
geometry-weighted averages are different estimands and must not share an
unqualified label.

## 2. Toroidal Fourier convention

The periodic stored coordinate is `z`, with `N_z` cells and `zperiod = 5`.
For a real field `a`, use NumPy's unnormalized forward transform:

```text
A_k = rfft(a)_k = sum_(j=0)^(N_z-1) a_j exp(-2 pi i j k / N_z).
```

The one-sided mean-square contribution of mode `k` is

```text
P_k = w_k |A_k|^2 / N_z^2,
```

where `w_0 = 1`, the Nyquist weight is one when `N_z` is even, and every
remaining positive-frequency weight is two. Therefore

```text
sum_k P_k = mean_j |a_j|^2
```

to numerical precision. A mean-removed spectrum subtracts `mean_z(a)` before
the transform and must say so. No radial, poloidal, or geometry weighting is
implicit in a toroidal spectrum.

The stored Fourier index maps to the full-torus toroidal mode number as

```text
n = zperiod * k = 5k.
```

Returned coordinates must include both `k` and `n`. In particular, `k = 4..7`
means `n = 20..35`; these labels are not interchangeable.

The real-transform and Nyquist conventions follow the
[NumPy FFT documentation](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html).

## 3. Cross-spectrum, coherence, and phase

For Fourier coefficients `A_k` and `B_k`, reduce only over explicitly supplied
sample axes:

```text
S_ab(k) = mean(A_k conjugate(B_k)),
S_aa(k) = mean(|A_k|^2),
S_bb(k) = mean(|B_k|^2).
```

The magnitude-squared coherence and phase are

```text
gamma_ab^2(k) = |S_ab(k)|^2 / (S_aa(k) S_bb(k)),
phi_ab(k) = angle(S_ab(k)).
```

This sign convention means that if `B_k = A_k exp(i delta)`, then
`phi_ab = -delta`. The convention is frozen here so a visually plausible but
sign-reversed phase cannot pass unnoticed. Coherence is reported as undefined
(`NaN`) where either autospectrum is zero; those bins are never silently set to
zero or one.

No smoothing, windowing, radial averaging, mode binning, or phase unwrapping is
implicit. A later physics report must declare those choices prospectively. The
coherence definition agrees with the standard cross-spectral convention
documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html).

## 4. Scalar ensemble CRPS

For scalar verifying value `y` and `M` ensemble values `x_i`, the ordinary
empirical-distribution CRPS in loss orientation is

```text
CRPS_M = (1/M) sum_i |x_i - y|
         - (1/(2 M^2)) sum_i sum_j |x_i - x_j|.
```

It reduces to absolute error when `M = 1`. This is the CRPS of the finite
empirical distribution. It is useful for evaluating the actual ensemble that
will be delivered, but its expectation depends on ensemble size.

When ensemble members are intended as independent random samples from an
underlying forecast distribution, the finite-ensemble fair form is

```text
fair-CRPS_M = (1/M) sum_i |x_i - y|
              - (1/(2 M (M - 1))) sum_(i != j) |x_i - x_j|,
```

which requires `M >= 2`. Fair CRPS estimates the score of the underlying
sampling distribution rather than treating the finite member set as the
forecast distribution. The distinction follows Ferro's
[fair-score construction](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf)
and the expectation representation of CRPS in
[Gneiting and Raftery](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

Paper 0 will report both forms and the ensemble size. It will not compare
ordinary CRPS values across unequal member counts as though they were directly
equivalent. A CRPS value alone is not a dimensionless calibration coefficient,
and no claim such as "calibration = 0.7" is permitted without a separately
defined normalized statistic.

The Phase 2 implementation is pointwise and marginal: it can reduce over any
chosen set of pixels only after each scalar CRPS is evaluated. Marginal field
CRPS does not establish correct cross-field dependence, spatial covariance,
mode covariance, or transport calibration.

## 5. Spread, skill, and empirical coverage

At every scalar verification location:

```text
ensemble_mean = mean_i x_i
spread         = standard_deviation_i(x_i, ddof=0)
skill_error    = ensemble_mean - y.
```

For a declared reduction set, report:

```text
RMS spread = sqrt(mean(spread^2))
RMSE       = sqrt(mean(skill_error^2))
SSR        = RMS spread / RMSE.
```

`SSR = 1` is only a scale comparison, not proof of calibration. If RMSE is
zero, the ratio is undefined unless both numerator and denominator are zero;
the implementation returns `NaN` rather than inventing a finite score.

For nominal central coverage `q`, use empirical ensemble quantiles at
`(1-q)/2` and `(1+q)/2`, inclusive of endpoints. Report the nominal level,
observed coverage, mean interval width, ensemble size, quantile method, and
reduction axes. With small ensembles, coverage resolution and quantile
interpolation limitations must be stated.

PIT and randomized rank-histogram conventions will be frozen in a later
amendment before they are applied; ties and finite ensembles make an implicit
implementation unsafe.

## 6. Nonlinear and joint diagnostics

Every nonlinear diagnostic is calculated separately for every ensemble member:

```text
d_i = D(x_i),
```

and only then summarized over `i`. In general,

```text
mean_i D(x_i) != D(mean_i x_i).
```

This rule is mandatory for particle flux, heat flux, products, ratios,
threshold events, cross-phase, and any other nonlinear functional. The code
will include a generic member-preserving helper and a synthetic product test
whose result differs from the product of ensemble-mean fields.

The first metric module does **not** implement particle flux, heat flux, region
masks, or physical-volume reductions. Those remain blocked until an
authoritative Hermes/TCV geometry-aware implementation, field staggering,
sign convention, and unit conversion are identified and tested. A uniform-grid
proxy may later be added only under an explicit `proxy` name; it cannot be
reported as physical transport.

## 7. Required O0 known-answer tests

The metric implementation must pass all of the following without reading a
simulation file:

1. reject a forecast without canonical seven-axis semantics;
2. reject forecast/truth dimension disagreement and non-finite values;
3. map `k = 0..` to `n = 5k` exactly;
4. identify a real cosine at known stored `k` and satisfy Parseval for even and
   odd `N_z`;
5. recover unit coherence and the frozen signed phase for two known complex
   Fourier relationships;
6. return undefined coherence for a zero-power channel;
7. reproduce hand-calculated ordinary and fair CRPS values;
8. reduce a one-member ordinary CRPS to absolute error and reject one-member
   fair CRPS;
9. recover known central-interval coverage and spread/skill values;
10. preserve the member axis through a nonlinear diagnostic and demonstrate
    that member-wise products differ from a product of ensemble means.

Random known-answer arrays use fixed seeds. Tolerances must be stated in each
test. Tests do not compare one implementation only against a second copy of
the same formula.

## 8. Application order and claims

The Phase 2 order is:

1. implement and pass O0 on synthetic arrays;
2. freeze the authoritative physical diagnostics or record their blockers;
3. run O1 codec reconstruction on `85604` only;
4. stop and repair the representation if O1 loses material cross-phase,
   coherence, spectral power, or transport;
5. proceed to one-step and rollout oracles only after the representation gate.

No model is selected by these metrics yet. No threshold for "adequate"
transport fidelity is set in this document. Acceptance thresholds must be
frozen after metric validation and before access to `85606`.
