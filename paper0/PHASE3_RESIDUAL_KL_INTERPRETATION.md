# Phase 3 residual-KL interpretation

**Decision date:** 2026-08-19

**Development simulation:** TCV/Hermes 85604 only

**Held-out simulation 85606:** unopened

**Authoritative clean execution:** Slurm job 6904897, commit
`6e3469b1a37430a2493e5889f24c653f2f5f5418`

**Primary outcome:** `K4_training_residual_span_does_not_transfer`

## Bottom line

The error left by the frozen H1 one-step mean is not well represented on the
later 85604 validation interval by a global linear residual basis fitted to
the earlier 85604 training interval.  This remains true even when the oracle
uses all 429 positive training directions and obtains each validation
coefficient from validation truth itself.  A rank-128 static Gaussian
ensemble is noncollapsed, but materially underdispersed and fails every frozen
transport-usefulness decision.

This is evidence against another compact global residual head on the present
data.  It is not evidence that every nonlinear stochastic architecture is
impossible, nor does it invalidate the deterministic H1 mean.  It says that
the current single-trajectory training sample does not support the proposed
low-rank residual representation strongly enough to justify more architecture
iteration.

## What the oracle tested

The analysis starts from the five-field, one-step H1 validation residual after
the frozen potential gauge correction and training-fitted axisymmetric bias
removal.  Each sample contains 901,120 standardized field values.  The 430
permitted training residuals therefore define at most 429 positive centered
directions.

Two separate questions were evaluated:

1. **Tier A, representation upper bound.** Validation truth was orthogonally
   projected into the training residual span at ranks 0, 8, 16, 32, 44, 64,
   128, 256, and the full positive rank 429.  Because validation truth chooses
   the coefficients, this is more favorable than any real forecast model.
2. **Tier B, static uncertainty baseline.** A 32-member, condition-independent
   Gaussian ensemble was sampled at rank 128 around the frozen H1 mean.  Rank
   128 was fixed from training before validation was opened; it was the
   protocol's cap-bound fallback because none of the permitted ranks through
   128 reached 90% training variance.

No checkpoint was loaded, no model inference or training occurred, and no
physics metric entered a loss.

## Training residual complexity

The training covariance is not compact:

- participation-ratio effective rank: 262.78;
- entropy effective rank: 330.08;
- rank 64 captures 37.81% of training variance;
- rank 128 captures 58.45%;
- rank 256 captures 84.02%;
- the 90% training-variance crossing occurs above the frozen rank-128 static
  cap.

The full 429-mode basis reconstructs its own centered training matrix to
`2.64e-8` relative RMS, and the stored basis passes its orthogonality checks.
The negative result is therefore not caused by a failed PCA implementation.

## Tier-A result: the training span does not transfer

| Rank | Validation residual variance captured | Dependence identities passing | Material field/mode bands passing | Representation gate |
|---:|---:|---:|---:|:---:|
| 0 | 0.00% | 0 / 11 | 0 / 15 | fail |
| 8 | 1.68% | 0 / 11 | 0 / 15 | fail |
| 16 | 3.09% | 0 / 11 | 0 / 15 | fail |
| 32 | 5.18% | 0 / 11 | 0 / 15 | fail |
| 44 | 6.43% | 0 / 11 | 0 / 15 | fail |
| 64 | 8.31% | 0 / 11 | 0 / 15 | fail |
| 128 | 12.52% | 1 / 11 | 0 / 15 | fail |
| 256 | 17.74% | 1 / 11 | 0 / 15 | fail |
| 429 | 22.34% | 6 / 11 | 0 / 15 | fail |

At full rank, validation residual variance capture by field is:

- Ne: 21.24%;
- Pe: 22.79%;
- Pi: 19.50%;
- phi: 11.62%;
- Vi: 27.52%.

For the 15 frozen field/mode combinations covering stored modes 1 through 7,
the full projection carries only 14.5% to 27.6% of validation residual power.
None lies in the required 80% to 120% range.  Remember that the simulated
toroidal fraction is one fifth of the torus, so stored index `k` maps to full
mode number `n=5k`.

The full-rank projection passes six of eleven systematic dependence
identities, short of the required nine.  The identities that transfer are:

- private-flux cross-field correlation;
- toroidal potential correlation;
- radial Pe and Pi correlations;
- poloidal Ne and potential correlations.

It does not transfer the frozen toroidal Pi, radial potential, or poloidal Pe,
Pi, and Vi identities strongly enough across five of six validation blocks.

Transport improves only modestly as rank grows.  At full rank, relative L2
errors remain 69.9% to 71.4% for strict-face transport and 71.2% to 71.5% for
exact-separatrix local transport.  Zero of four transport quantities passes
either frozen transport requirement.

## Tier-B result: noncollapsed but underdispersed

The rank-128 static ensemble has finite spread at every evaluated field voxel,
so this is not ensemble collapse.  Its corrected field spread divided by
ensemble-mean error is nevertheless too small:

| Field | Corrected spread / error |
|---|---:|
| Ne | 0.555 |
| Pe | 0.598 |
| Pi | 0.586 |
| phi | 0.524 |
| Vi | 0.737 |
| Equal-field aggregate | 0.639 |

The frozen useful range is 0.80 to 1.25.  Rank histograms and interval
coverage also show underdispersion; for example, the nominal 93.9% finite
ensemble interval covers only 68.5% to 78.9% across fields.

Transport spread divided by transport error is tightly clustered but low:

| Transport quantity | Local | Toroidally integrated | Covariance-coherence multiplier ratio |
|---|---:|---:|---:|
| Particle | 0.653 | 0.621 | 0.902 |
| Electron internal energy | 0.655 | 0.609 | 0.866 |
| Ion internal energy | 0.652 | 0.632 | 0.940 |
| Total internal energy | 0.653 | 0.618 | 0.896 |

The coherence multiplier ratios are reasonably close to one, and a scalar
counterfactual chosen to repair integrated transport would place local
transport spread near 1.03 to 1.07.  That is a useful secondary observation:
some cancellation structure is present.  It does not reverse K4, because the
static ensemble is underdispersed as run, its field calibration fails, and the
truth-projected training span itself misses most validation residual energy
and power.

## What this teaches us

### 1. Bigger latent size is not the immediate answer

This experiment removed the DCAE and its tokenization from the question.  It
used a direct field-space residual basis and still failed at the complete
training rank.  Increasing f8 to z44, or replacing rank 128 with a somewhat
larger compact head, cannot address a validation residual that largely lies
outside the complete observed training span.

### 2. Calibration is downstream of representation and data coverage

One can inflate a narrow ensemble and improve a marginal spread number.  That
does not create the missing validation directions, field/mode power, or
transport-bearing cross-field structure.  CRPS, diffusion noise schedules,
FGN conditioning, and inflation are meaningful only after the representation
and data support the variability that must be calibrated.

### 3. The single trajectory is the limiting evidence base

The 430 training targets are temporally adjacent samples from one simulator
trajectory, not 430 independent runs or replicated futures from the same
condition.  Their centered span is necessarily rank-limited, and the later
chronological interval contains substantial residual structure not covered by
that span.  The result may reflect high-dimensional turbulence, slow
within-run evolution, residual nonstationarity, or a combination.  This
experiment cannot uniquely distinguish those explanations.

### 4. The deterministic mean and stochastic residual are separate issues

The oracle examines only what H1 leaves unexplained after one step.  It does
not say H1 has no predictive value.  It says a condition-independent global
linear model of H1's remaining error is inadequate.  A future conditional
model would still need enough independent simulator coverage to establish
which residual variations are learnable rather than accidental features of
one trajectory.

## Decision and next admissible work

Under the prospectively frozen decision rule, K4 stops further architecture
training on the present data.  In particular, this result does not authorize
FGN, DLL, MNO, another diffusion run, PDE-Refiner retraining, O3/O4/O5,
assimilation, diagnostic ranking, steering, or access to 85606.

There are two honest paths:

1. **Obtain additional Hermes trajectories.** Ask for simulations with the
   same stored fields, geometry, cadence, and preprocessing, plus complete
   forcing, source, heating, boundary, and operating-condition metadata.
   Treat simulator runs—not temporal windows—as the independent units.  A new
   prospective protocol should first test whether residual covariance and
   transport statistics transfer across training runs before choosing FGN,
   diffusion, or a multiscale residual representation.
2. **Narrow Paper 0 to a benchmark/failure analysis.** Document the codec,
   one-step mean, residual drift, calibration, spectral/cross-field, and
   transport failures without proceeding to diagnostic-design claims.  This
   would be scientifically honest but would not support the original
   transport-faithful assimilation headline.

If additional trajectories become available and representation transfer is
then demonstrated, the present evidence favors evaluating a multiscale or
field-space conditional stochastic model rather than assuming a compact
global coefficient head.  That is a future decision, not an authorization
from this result.

## Reproducibility and scope closure

- Clean Slurm job 6904897 completed with exit code 0 on Rocky 9 worker6203 in
  33 minutes 23 seconds; peak RSS was approximately 20.2 GB.
- The full Rocky 9 suite passed: 1,216 tests, one skip, and 29 subtests.
- The training basis SHA-256 is
  `fcc32c3baaf0deb85fa55456612d3ab8beaf859af20b5ba86f94233c15e0dbbc`.
- The Tier-A and Tier-B compact CSV files are byte-identical to the prior
  complete scientific computation in job 6904413, which ran on a different
  worker before a post-result telemetry error.
- W&B run `p0reskl-6904897-s1701` is remotely verified as finished.  Only 78
  compact scalars and provenance metadata were logged; no scientific arrays,
  fields, forecasts, basis, figures, or tables were uploaded.
- Guard frames were not read.  Run 85606 was not read.
- Authoritative Ceph directory:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_residual_kl_oracle/job_6904897`
- W&B URL:
  `https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0reskl-6904897-s1701`
