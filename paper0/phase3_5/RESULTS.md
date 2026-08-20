# Paper 0 Phase 3.5 results: 85604 stochastic-transition cause localization

## Result identity and scope

The authoritative result is Rocky 9 Slurm job `6907468`, produced from clean
commit `5f7124075ab1510790c62789950ec00a878846d6` on one NVIDIA H100. The job
completed with exit code `0:0` in 1:03:25 and reached 107,696,764 KiB peak RSS.

- immutable output: `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase3_5_cause_localization/job_6907468/analysis`
- W&B: `https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p35-6907468`
- run-manifest SHA-256: `ede330866602c6606ee87e1b91eed31e683cb692d7e494b813148d551e9f2a23`
- complete scientific-artifact index SHA-256: `95435ee2eeed0b3e8b22e0b161d431f4d6c76dc9ddc5fb82d90d8a14f5e9b285`

All 29 scientific output hashes were independently rechecked. Only simulation
85604 was used. Frames `[432,496)` were not read. Simulation 85606 remains
unopened. No production neural model was trained, no optimizer was created,
and no assimilation, diagnostic ranking, steering, or control experiment was
performed.

## Narrow result

> A single, fixed, condition-independent, global linear residual distribution
> learned from adjacent 85604 training frames does not describe later 85604
> residuals well.

This is a statement about K4. It is not evidence that stochastic emulation is
impossible, and it is not a failed test of FGN, PDE-Refiner, or diffusion.

## What was learned

### 1. The residual subspace changes even between nearby blocks

A rank-41 basis fits all centered variance in its own 42-sample source block
by construction, but captures only about 2.6–4.5% in most other blocks. A T00
basis captures 2.81% on average across V00–V02. The largest principal angle
between the T00 rank-41 subspace and every other block is approximately 90
degrees.

This is not merely a train/validation discontinuity. Equal 21-sample
first-half-to-second-half controls capture only 2.62% on average at rank 16.
The residual field is high-dimensional and its dominant sample directions are
not stable under a fixed global basis.

### 2. The permitted interval is valid but not statistically stationary

The frozen evidence rule labels this explanation `invalid/nonstationary
interval`; the `protocol_invalid` flag is false. The evidence is drift, not a
corrupt split.

From T00 to V02, mean fluctuation RMS changes by approximately:

| field | fractional change | standardized T00-to-V02 effect |
|---|---:|---:|
| Ne | +7.4% | +5.73 pooled temporal SD |
| Pe | +13.4% | +6.10 pooled temporal SD |
| Pi | +10.9% | +5.18 pooled temporal SD |
| phi | +9.8% | +1.44 pooled temporal SD |
| Vi | +3.7% | +2.62 pooled temporal SD |

The inner and outer `Bphi` means also change substantially, although they vary
non-monotonically block to block. In contrast, the T00-to-V02 changes in all
four integrated transport means have block-bootstrap intervals containing
zero. Absolute time alone reaches `R2 >= 0.10` for none of the residual probe
targets. The useful repair is therefore conditioning on physical state or
regime, not treating time as the physical cause and not opportunistically
discarding validation blocks.

### 3. More adjacent frames do not solve the representation problem

For the primary Geyer estimate, the least effectively sampled material series
has ESS 21.36 and the median material ESS is 99.18. None is below the frozen
ESS=20 evidence threshold. Window sensitivity matters: one Pe residual series
falls to ESS 19.04 under the self-consistent-window estimator, so the data are
not information-rich in every observable.

Nevertheless, increasing the chronological source from 378 to 420 targets
improves mean validation variance capture by only 0.0004. More highly
correlated frames from the same evolving trajectory are not the main repair.
Independent restarts would still be valuable, but insufficient ESS is not the
localized cause selected by this experiment.

### 4. Bulk toroidal motion is real, but H1 already predicts most of it

Consecutive truth states have a median shared displacement of 11 of 88 stored
toroidal cells, consistent with the earlier 9–12-cell observation. None of
these peaks passes the complete ambiguity rule because the median correlation
surface entropy is about 0.95.

The best truth-assisted displacement between H1 and its target is zero for at
least 75% of samples. The median H1 residual-energy reduction after alignment
is exactly zero, and the full-rank aligned K4 capture is 22.384%, only 0.044
percentage points above the prior 22.34% reference. A transported-persistence
oracle improves persistence MSE by 67.1%, but remains materially worse than
H1.

Therefore coherent advection matters to a naive Eulerian persistence model,
but missing one shared displacement is not the cause of H1 residual-transfer
failure. A common shift also leaves cross-field phase relationships unchanged
by construction.

### 5. The codec and H1 predictor are not exactly toroidally equivariant

Every one of the 13 representative states shows a period-four sawtooth tied to
the codec's two stride-two packings. Across nonzero shifts:

| scope | median normalized equivariance error | median error relative to its unshifted truth error |
|---|---:|---:|
| codec | 0.00453 | 0.0275 |
| H1 | 0.01293 | 0.1986 |

The absolute errors are not catastrophic, but the systematic shift-class
dependence is structural and reaches the frozen strong-evidence criterion.
It plausibly contributes to unstable residual coordinates, but it does not by
itself explain the full block-to-block subspace collapse.

### 6. A toroidal Fourier representation transfers substantially better, but is not transport-faithful by itself

Mean V00–V02 residual variance captured at matched real coefficient budgets is:

| real coefficients | global PCA/KL | toroidal Fourier complex KL |
|---:|---:|---:|
| 32 | 0.0515 | 0.0849 |
| 64 | 0.0822 | 0.1471 |
| 128 | 0.1243 | 0.2240 |
| 256 | 0.1762 | 0.3172 |
| 416 | 0.2192 | 0.3940 |

Truth-assisted co-moving PCA differs from ordinary global PCA by at most 0.17
percentage points, confirming that bulk alignment is not the source of the
Fourier gain. Overlapping local PCA and Haar variants capture less total later
variance at the same budgets.

The Fourier result is not a universal win. At budget 416, mean cross-spectrum
phase error is about 50.0 degrees for Fourier versus 13.9 degrees for global
PCA. Mean integrated-transport-covariance relative error is 0.398 versus 0.243.
Overlapping local PCA is best for local and integrated transport covariance at
that budget (0.194 and 0.180). Fourier improves variance transfer,
cross-field covariance, and coherence, while other representations preserve
other nonlinear functionals better. No tested fixed linear representation is
yet transport-faithful across all required objectives.

### 7. Residual covariance is context dependent, in a specific sense

Available C5P context beats both a constant and time-only baseline by the
frozen margin in every validation block for 15 scalar targets spanning three
families: residual field energy, residual spectral energy, and residual
cross-field covariance. Thirteen of those 15 have positive ridge `R2` in all
three validation blocks. The strongest examples are Vi residual energy and Vi
spectral energy. No local or integrated transport-error target passes this
all-block rule.

The fixed B5 context shuffle is corroborating rather than decisive. With the
same eight seeds, chronologically mismatched contexts change median covariance
summaries by:

| covariance family | median absolute relative change |
|---|---:|
| cross-field covariance | 0.0026 |
| field variance | 0.0624 |
| spectral-band covariance | 0.0815 |
| integrated transport covariance | 0.2229 |
| integrated transport variance | 0.2062 |
| local transport covariance | 0.2607 |

Thus B5 is almost invariant in its global cross-field covariance but more
sensitive in spectral and nonlinear transport summaries. This does not prove
that its conditioning is adequate.

### 8. Omitted state and longer history carry clues, not a completed solution

Adding summaries of saved `NVe`, `Vort`, and `Bphi` reduces normalized probe
RMSE by at least 10% in every validation block for two targets: Ne residual
energy and low-band phi residual spectral energy. The causal-neighbor analysis
finds zero target-family successes, and later-block `R2` is often still
negative. This is evidence to preserve and test exact state, not proof that
exact state alone fixes the emulator.

The delay set `[1,2,4,8,16]` improves five Pe/Pi residual-energy,
spectral-energy, or cross-covariance targets in all three legal matched
validation subsets. All five delayed probes still have negative held-out
`R2`. This reconciles the result with the earlier failed one-extra-frame H2
configuration: one failed adjacent-history choice does not prove memory is
irrelevant, while these diagnostic probes do not show that longer history is
already sufficient.

## Frozen evidence ranking and practical interpretation

| rank | explanation | frozen tier | practical interpretation |
|---:|---|---|---|
| 1 | codec or predictor non-equivariance | strong | clear period-four structural error; important but not the whole failure |
| 2 | invalid/nonstationary interval | strong | interval is valid; physical/statistical drift requires conditioning |
| 3 | forecast-state-dependent covariance | strong | robust for residual energy/spectra/covariance, not yet transport error |
| 4 | history-dependent hidden state | strong | consistent relative gain, but absolute probes remain worse than constant |
| 5 | insufficient or incorrect retained state | strong | two probe families improve; nearest-neighbor confirmation is absent |
| 6 | coherent transport in an inappropriate Eulerian representation | moderate | truth advects, but H1 alignment gain is negligible |
| 7 | insufficient effective sample size | none | some ESS values are low, but transfer learning curves have saturated |
| 8 | unexplained failure | none | several preregistered mechanisms have evidence |

The representation companion result is strong for Fourier transfer, with the
physics caveats above.

## Exact next experiment to authorize

Do not open 85606 and do not begin a production architecture. First
preregister a 85604-only interval/conditioning repair:

1. Define causal, training-only regime features from current C5P field RMS,
   radial profiles and gradients, selected toroidal amplitudes/phases, and
   recent displacement. Keep absolute time as a drift baseline only.
2. Add one explicit ablation using the available `NVe`, `Vort`, and `Bphi`
   summaries; do not silently make them privileged default inputs.
3. Fit a small, regularized condition-dependent covariance baseline in a
   matched-budget Fourier/global representation. This remains a diagnostic
   baseline, not a production neural model.
4. Freeze the conditioning rule and compare it with condition-independent
   global KL and Fourier baselines on unchanged V00, V01, and V02 using the
   same field, phase/coherence, and member-wise transport metrics.
5. Require consistent improvement in all three blocks and reject any repair
   that gains variance capture while materially worsening cross-phase or
   transport covariance.

This experiment has not been launched and the recommendation is not
automatically authorized.

Recommended next action: repair interval/conditioning
