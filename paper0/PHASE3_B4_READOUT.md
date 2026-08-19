# Phase 3 B4 latent PDE-Refiner one-seed readout

**Full training job:** `6899682`

**Full scientific evaluation job:** `6901015`

**Final gate job:** `6901285`

**Development simulation:** 85604 only

**Held-out simulation 85606:** unopened

**Decision:** B4 fails both frozen hypotheses; do not replicate it, launch O3,
or use its ensemble for assimilation

## Executive conclusion

B4 produced a better one-step point forecast but did not produce the physical
or probabilistic repair it was designed to test. Relative to the frozen
deterministic H1 parent, its M32 ensemble mean reduced equal-channel MAE by
`5.57%` and RMSE by `8.48%`. The deterministic field family therefore passed
overall and in all six chronological blocks.

That gain did not extend to the decisive joint quantities. Across the explicit
refinement ladder, the final stage reduced field error and realization error
slightly, but worsened the aggregate power, cross-field, and separatrix-
transport errors. The stagewise repair test failed. In the final M32 forecast,
only 9 of 15 material power checks and 4 of 15 realization-coherence checks
passed. Strict geometry-aware transport relative-L2 errors remained near
`0.68`, above the frozen `0.40` limit.

The stochastic result is more decisive. The aggregate corrected field
spread--skill ratio was only `0.217`, and the five field values were
`0.187--0.252`. The nominal `0.939` widest-interval field coverage was only
`0.474--0.534`. All 33 material field-band or cross-projection spread--skill
checks and all 33 corresponding widest-interval coverage checks failed.
Separatrix transport spread--skill was only `0.060--0.073`.

The supported conclusion is:

> In this latent adaptation, PDE-Refiner fine-tuning improves the one-step
> conditional mean but neither repairs transport-relevant structure nor
> supplies a calibrated conditional ensemble.

This is a development result on the model-selection interval of 85604. It is
not an autonomous-rollout result, a held-out-shot result, or a general claim
about PDE-Refiner.

## What B4 tested

B4 began from the selected C5P-H1 deterministic transition and the frozen
C5P-dcae_l10 codec. It retained the 32-channel latent grid `[16,8,22]`, added
a refinement-level embedding and adapters to the 16 transformer blocks, and
fine-tuned the complete transition while keeping the codec fixed.

One forecast member used four levels:

1. level 0 made a deterministic parent-like latent increment prediction;
2. levels 1--3 added independent full-latent Gaussian noise at three frozen,
   decreasing scales and predicted that noise for explicit denoising;
3. the frozen codec decoded the final latent state to all five fields
   `[Ne,Pe,Pi,phi,Vi]` jointly.

The task was one saved step, or `3.131905426352636` microseconds. The model
received exactly frame `t-1` and predicted frame `t`. Absolute time was
metadata, not an input. Training used only standardized latent MSE. No field
metric, spectrum, cross-phase, coherence, flux, transport, PDE residual, or
calibration statistic entered the loss or checkpoint selection.

This is a latent, TCV-specific adaptation of the method in Lippe et al.,
*PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*
(NeurIPS 2023, <https://arxiv.org/abs/2308.05732>). It is not an exact
field-space reproduction, and the paper does not establish calibrated
transport ensembles for this setting.

## Provenance and execution

| Stage | Job | Commit | Result |
|---|---:|---|---|
| bounded GPU implementation smoke | `6899469` | pre-full-training protocol | completed |
| full training | `6899682` | `0350b063` | 100 epochs, 2,700 steps |
| bounded evaluator smoke | `6900991` | `e43439cb` | completed; non-authoritative warning |
| full truth-separated evaluation | `6901015` | `e43439cb` | all 126 targets and scores completed |
| first gate inside evaluation wrapper | `6901015` | `e43439cb` | adapter key error after evaluation |
| CPU-only gate retry | `6901282` | `aa33a402` | second adapter nesting error |
| final CPU-only gate | `6901285` | `f062ebe6` | completed; H-det and H-prob fail |

The two gate-adapter failures occurred after immutable forecasts and scores
were complete. They changed no forecast, score, threshold, or scientific
number. Both are retained in the audit and result history. The final gate
passed all 138 provenance and integrity checks.

Training used one H100 for `6022.94` seconds (`1:40:22`), with a peak recorded
CUDA allocation of `3.84` GB. The model has `61,218,944` parameters. The
selected checkpoint is the 100th completed epoch, stored internally as
zero-based epoch `99`; it was selected from 20 candidates using only the
frozen two-member decoded-field MAE rule.

Scientific generation produced 126 chronological targets and 32 members with
canonical shape `[126,32,1,5,64,32,88]`. It took `355.18` seconds on one H100
and peaked at `9.78` GB. Because level zero is shared, one target required 97
member-equivalent transition evaluations rather than the unamortized 128.

## Did the refinements repair the parent-like level?

No. The predeclared stage test compared final level 3 directly with level 0
using the first four members.

| Error aggregate | Level 0 | Level 3 | Level 3 / level 0 | Required |
|---|---:|---:|---:|---:|
| decoded field MAE | 0.04490 | 0.04325 | 0.963 | at most 1.05 |
| material power error | 0.23842 | 0.26868 | 1.127 | at most 1.05 |
| realization error | 0.40430 | 0.39319 | 0.973 | at most 1.05 |
| cross-field error | 0.03236 | 0.03524 | 1.089 | at most 1.05 |
| separatrix transport error | 0.23120 | 0.24355 | 1.053 | at most 1.05 and strictly lower |

The field and realization aggregates improved, but power and cross-field error
worsened beyond tolerance. Transport worsened by `5.34%`, missing both its
non-degradation and strict-improvement requirements. Levels 1, 2, and 3 were
nearly indistinguishable, so the smaller second and third noise scales made
negligible additional changes after the first refinement.

## Field accuracy and marginal calibration

Values below one in the MAE-ratio and fair-CRPS-ratio columns are better than
the deterministic H1 parent's corresponding MAE.

| Field | H1 MAE | B4 mean MAE | B4/H1 MAE | B4 fair CRPS / H1 MAE | Spread--skill | Widest coverage |
|---|---:|---:|---:|---:|---:|---:|
| Ne | 0.04338 | 0.04198 | 0.968 | 0.822 | 0.229 | 0.483 |
| Pe | 0.03269 | 0.03072 | 0.940 | 0.817 | 0.187 | 0.491 |
| Pi | 0.04201 | 0.04006 | 0.953 | 0.828 | 0.189 | 0.501 |
| phi | 0.04560 | 0.04410 | 0.967 | 0.819 | 0.252 | 0.474 |
| Vi | 0.06509 | 0.05917 | 0.909 | 0.772 | 0.223 | 0.534 |

Equal-weight channel aggregates were:

- ensemble-mean MAE `0.04321`, versus H1 `0.04575`;
- ensemble-mean RMSE `0.07289`, versus H1 `0.07965`;
- fair CRPS `0.03696`, or `0.808` times H1 MAE;
- corrected spread--skill `0.217`.

The fair CRPS improves because the mean is better and the ensemble has small,
nonzero spread. It does not imply calibration. A calibrated ensemble should
have spread comparable to its error and coverage near the nominal interval.
Here the primary spread--skill target was `[0.80,1.25]`, while every field was
below `0.26`. The rank and coverage diagnostics therefore identify severe
underdispersion.

For context, the earlier B3 functional-noise arm had aggregate fair CRPS
`0.03308` and spread--skill `0.790`, but failed joint spectral and transport
calibration. B4 trades that useful marginal spread for a sharper mean and a
much tighter, less useful ensemble. This comparison is descriptive; the two
stochastic mechanisms and training objectives differ.

## Toroidal spectra and cross-field structure

The stored domain has `zperiod=5`, so stored Fourier index `k` maps to
full-torus mode number `n=5k`. The frozen material bands are `k=1..3`
(`n=5..15`), `k=4..5` (`n=20..25`), and `k=6..7` (`n=30..35`).

The deterministic spectral family passed:

- 9 of 15 expected-member power checks;
- 4 of 15 ensemble-mean realization-coherence checks;
- all 9 cross-phase checks;
- all 9 cross-coherence-change checks.

The highest material band illustrates the remaining loss:

| Field | Expected-member power ratio, `k=6..7` | Realization coherence, `k=6..7` |
|---|---:|---:|
| Ne | 0.590 | 0.457 |
| Pe | 0.550 | 0.508 |
| Pi | 0.551 | 0.487 |
| phi | 0.630 | 0.320 |
| Vi | 0.698 | 0.0087 |

The frozen power interval is `[0.75,1.30]`, and realization coherence must be
at least `0.80`. The model therefore loses high-band amplitude and generally
puts the remaining structure in the wrong next-frame realization. Good mean
cross-phase is still present: the nine absolute errors were `0.50--3.61`
degrees, well inside the 20-degree bound, and all cross-coherence changes were
below `0.067` against the `0.15` limit.

Those mean joint statistics are not a calibrated mode distribution. All 33
spread--skill checks and all 33 widest-interval coverage checks across the 15
field bands and 18 real/imaginary cross projections failed. All 33 M16-versus-
M32 stability checks passed, showing that the conclusion is not caused by an
insufficient 32-member Monte Carlo estimate.

## Nonlinear transport

Transport was computed independently for every member with the frozen,
geometry-aware radial ExB implementation. It was never computed only from the
ensemble-mean fields.

| Quantity | Strict relative L2 | Strict correlation | Separatrix relative L2 | Separatrix bias | Separatrix correlation | fCRPS / H1 error | Spread--skill |
|---|---:|---:|---:|---:|---:|---:|---:|
| particle | 0.680 | 0.746 | 0.291 | -0.231 | 0.836 | 1.123 | 0.073 |
| electron internal energy | 0.676 | 0.751 | 0.216 | -0.175 | 0.926 | 1.103 | 0.060 |
| ion internal energy | 0.677 | 0.749 | 0.242 | -0.183 | 0.878 | 1.070 | 0.069 |
| total internal energy | 0.676 | 0.750 | 0.226 | -0.179 | 0.909 | 1.100 | 0.062 |

Strict correlations and sign-disagreement checks pass, so the prediction
retains a temporal and directional signal. Local facewise amplitudes and
structure do not: all strict relative-L2 values fail the `0.40` limit.

At the separatrix, relative-L2, correlation, sign, and upper-decile event
checks mostly retain useful signal. All four normalized biases exceed the
`0.15` absolute limit, however. More importantly, every transport fair CRPS
is worse than the deterministic H1 absolute-error reference and every
transport ensemble is severely underdispersed. Widest-interval empirical
coverage is only `0.063--0.119`, versus nominal `0.939`.

## Formal gate result

| Hypothesis | Family | Overall checks failed | Passing chronological blocks | Required | Result |
|---|---|---:|---:|---:|---|
| H-det | field | 0 / 2 | 6 | 5 | pass |
| H-det | spectral | 17 / 48 | 0 | 5 | fail |
| H-det | transport | 8 / 50 | 0 | 5 | fail |
| H-prob | field | 37 / 51 | 0 | 5 | fail |
| H-prob | spectral | 66 / 99 | 0 | 5 | fail |
| H-prob | transport | 6 / 26 | 0 | 5 | fail |

All 138 integrity checks pass. H-det and H-prob each have zero chronological
blocks in which every required family passes. The final disposition is
`stop_B4_before_replication_O3_or_assimilation`.

## What we learned and what comes next

B3 and B4 now localize two different failures:

- B3 shows that global functional noise plus marginal fair CRPS can create
  useful field spread without learning calibrated mode-resolved and transport
  covariance.
- B4 shows that explicit latent denoising refinements can improve the one-step
  mean while adding far too little conditional spread and failing to repair
  power or transport.

The evidence does not support more B4 seeds or scheduler tuning. The next
predeclared branch is B5: retain the frozen H1 deterministic mean initially
and learn one joint residual distribution across all five decoded fields.
That separation asks the residual generator to model exactly what the mean
misses instead of asking one latent transition to learn the mean and
uncertainty simultaneously.

Before choosing a field-space architecture or patch size, B5 should run a
training-region-only residual audit. It must measure the H1 residual's bias,
scale, spatial autocorrelation, toroidal extent, cross-field covariance, and
temporal dependence. This follows the CorrDiff decomposition and its explicit
recommendation that any residual patch span at least the residual
autocorrelation distance (<https://arxiv.org/abs/2309.15214> and
<https://docs.nvidia.com/physicsnemo/latest/physicsnemo/examples/weather/corrdiff/README.html>).
CorrDiff itself reports that calibration remains challenging, so residual
diffusion is a hypothesis to test, not an assumed solution.

Only after that audit should a separate B5 implementation/smoke protocol be
frozen. O3, assimilation, diagnostic ranking, and 85606 remain closed.

## Immutable evidence

The exact full evaluation record is
`paper0/results/phase3_b4_pde_refiner_evaluation_full_6901015.json`, with
SHA-256
`0988f71aa0749044e51ded92b9ea594563232df746415dccbbc6031443ca7e92`.

The exact final gate is
`paper0/results/phase3_b4_pde_refiner_one_seed_gate_6901285.json`, with
SHA-256
`4c07a7f4886c14ca2e53d6e322fe309e5efde1f76ab2ed779a3acd14d110f6be`.

The final and stage score hashes are respectively
`055d81979f46a96bc0c983e0ef2f387f3032a2505117849089047e4f00b67dd3`
and `33cb4ba9256c214bf0e625511464b51076f96eb1b979c0be32ec0f3b3180e9c2`.
The M32 final forecast and M4 all-stage forecast hashes are respectively
`16fe7f594228d8290b05f7da8505cf16722364c21bcfb7a252416445356e6813`
and `4393465ba8bdf9f7d9177ea8f36dcb02ecd246c34ec551d26eb16bd318386aac`.

Training W&B:
<https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0b4full-6899682-s1701>

Evaluation W&B:
<https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0b4evalfull-6901015-s1701>
