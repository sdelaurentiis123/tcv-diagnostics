# Phase 3 post-B5 research note

**Status:** provisional literature synthesis written after the frozen B5 gate
and before the covariance-localization result

**Development simulation:** 85604 only

**Held-out simulation 85606:** unopened

**Decision authority:** none. This note does not authorize training, tuning,
O3/O4/O5, assimilation, diagnostic ranking, or access to 85606.

## 1. What the completed evidence says

B3 and B5 both improved scalar probabilistic scores without producing the
joint forecast distribution required by Paper 0.

- B3 implemented the literature-standard functional-generative mechanism: a
  32-dimensional global Gaussian draw was shared across all spatial tokens
  and injected through conditional normalization in all 16 transition blocks.
  Its aggregate fair CRPS improved to `0.723` times the H1 MAE reference, but
  mode-resolved and separatrix-transport ensembles remained underdispersed.
- B5 removed the latent residual bottleneck and sampled a joint five-field
  residual in full field space with EDM. It further improved fair CRPS to
  `0.687` times H1 MAE and aggregate spread--skill to `0.802`. It nevertheless
  passed zero mode-power calibration checks and zero cross-spectrum projection
  calibration checks.
- B5's strict-face flux contributions have spread--skill near one, while the
  spatially integrated separatrix flux has spread--skill only `0.413--0.485`.
  That contrast is evidence against a purely scalar amplitude explanation:
  locally plausible fluctuations can cancel incorrectly when their spatial
  and cross-field covariance is wrong.

These are one-step development results on 85604. They do not yet establish an
autonomous-rollout failure, a held-out-shot failure, or a general failure of
FGN or diffusion.

## 2. What the literature adds

### Marginal CRPS is not a joint-distribution guarantee

Diaconu et al. explicitly state that CRPS evaluates univariate marginals and
does not guarantee spatial consistency. Their retrofit uses a low-dimensional
global noise vector, shared over space through conditional normalization, as
an architectural inductive bias for coherent variability. This is motivation,
not a theorem that the covariance will be correct.

Alet et al.'s FGN result shows that this inductive bias can recover useful
spatial and inter-variable structure in weather forecasting. It does not imply
that it must work for one-run TCV/Hermes training. Our B3 is already a direct
negative test of the simplest claim that global shared noise plus marginal
fair CRPS is sufficient here.

Primary sources:

- Diaconu et al., *Probabilistic Retrofitting of Learned Simulators*,
  <https://arxiv.org/abs/2603.01949>.
- Alet et al., *Skillful joint probabilistic weather forecasting from
  marginals*, <https://arxiv.org/abs/2506.10772>.

### The deterministic mean may need an explicit anchor

Kastor reports that an unconstrained FGN retrofit can perform worse than its
deterministic parent. Its Mean Prediction Regularization (MPR) makes the
null-noise path estimate the deterministic distribution mean while noisy
paths represent conditional variability. In their benchmarks, MPR materially
improves both FGN and diffusion stability.

This is relevant because B3 nearly preserves the H1 mean but does not improve
it, while B5 improves the mean through a separately trained residual model.
MPR is therefore a plausible *conditional* repair if localization shows that
the stochastic branch is corrupting or failing to organize covariance around
an otherwise useful mean. It is not yet an authorized experiment.

Kastor also uses spatial-gradient matching and temporal super-resolution.
Neither should be imported automatically. A spatial-gradient training term is
outside the frozen Paper 0 training design and would require a separate,
prospective protocol amendment. Non-causal temporal super-resolution cannot
be used as a real-time forecast transition and would be an offline
reconstruction component rather than the forecast prior required for EnKF.

Primary source:

- Couairon et al., *Kastor: An Efficient Fine-Tuning Strategy for Generative
  Emulation of PDE Simulations*, <https://arxiv.org/abs/2608.06107>.

### A dependence-sensitive score is a distinct intervention

Scheuerer and Hamill introduced the variogram score because common
multivariate scores can have limited sensitivity to misspecified correlation.
Pacchiardi et al. show how spatial proper scores, including variogram and
localized patched scores, can be used for generative forecasting and how a
proper dependence score can be added to a strictly proper score.

Paper 0 currently uses variograms only for evaluation. If B5 localization
supports a covariance-organization failure, a future protocol could compare
marginal fair CRPS with a generic multivariate proper-score objective. Such an
experiment must remain statistical: no flux, spectrum, cross-phase,
coherence, conservation, or PDE residual may enter the training loss.

Primary sources:

- Scheuerer and Hamill, *Variogram-Based Proper Scoring Rules for
  Probabilistic Forecasts of Multivariate Quantities*,
  <https://doi.org/10.1175/MWR-D-14-00269.1>.
- Pacchiardi et al., *Probabilistic Forecasting with Generative Networks via
  Scoring Rule Minimization*, <https://www.jmlr.org/papers/v25/23-0038.html>.

### Spread--skill near one is necessary evidence, not sufficient evidence

Recent ensemble-verification work shows that spread--error equality, rank
histograms, and the reliability component of CRPS can all look favorable
under incorrect joint covariance. This supports the Paper 0 choice to retain
separate climatological variance, member--truth predictability, spatial ACF,
cross-field covariance, mode projection, variogram, and transport checks.

Primary source:

- Dirkson and Buehner, *Are we misdiagnosing ensemble forecast reliability?*,
  <https://arxiv.org/abs/2512.02160>.

## 3. Frozen interpretation branches

The localization analysis must select evidence, not a preferred architecture.

### L1: predominantly amplitude-limited

Support would require broadly correct dependence shapes with uniformly small
variance, plus a scalar-inflation counterfactual that repairs integrated
transport without making local contributions overdispersed.

If supported, test the smallest amplitude intervention first. Do not redesign
the architecture merely to increase a scalar spread ratio.

### L2: covariance-organization limited

Support would require locally adequate variance but deficient off-diagonal
spatial transport covariance, incorrect spatial ACF/cross-field dependence,
or poor dependence-sensitive scores. This is the leading interpretation from
the completed B5 gate, but it remains a hypothesis until the frozen analysis
runs.

If supported, the next prospective comparison should isolate one mechanism at
a time:

1. deterministic-mean anchoring such as MPR;
2. structured global-plus-multiscale noise rather than independent output
   noise;
3. a generic multivariate proper score in addition to marginal fair CRPS.

Repeating B3 unchanged or inflating B5 after the fact would not test this
branch.

### L3: mismatch beyond within-run drift

The validation dependence mismatch must be compared with the chronological
training-to-validation H1 residual drift. If the apparent covariance failure
does not exceed that empirical drift consistently, the one-run evidence is
not strong enough to identify a model defect separately from distribution
shift.

### L4: missing temporal conditionality

The frozen teacher-forced AR(1) probe asks whether the previous realized H1
residual contains usable information about the next residual. A positive
aggregate result in at least five chronological blocks would justify a
history-conditioned stochastic model.

This would not mean that H2 must have lower deterministic RMSE than H1. Extra
history can be unhelpful for the conditional mean while still being necessary
to identify the conditional covariance. Both FGN weather models and the LOLA
retrofit literature use explicit temporal history, which makes this a
literature-supported distinction rather than a post-hoc architecture guess.

### L5: unresolved with one realized trajectory

If neither amplitude, organization, nor history evidence is stable across
chronological blocks, the correct outcome is to narrow the claim and request
additional simulator trajectories. More optimization on the same 430 training
targets would not create independent physical evidence.

## 4. Decision after the frozen localization

The next experiment may be proposed only after the read-only localization
finishes and its integrity anchors pass.

- `L1 only`: a prospectively frozen amplitude/noise-scale ablation.
- `L2 without L4`: mean-anchored structured stochasticity versus a
  dependence-sensitive proper-score arm.
- `L4`: a matched one-frame versus causal-history stochastic comparison,
  holding representation and compute as fixed as possible.
- `L3/L5 unresolved`: stop architecture iteration and obtain more simulator
  trajectories or narrow Paper 0 to a documented failure benchmark.

No branch opens 85606, autonomous rollout, assimilation, or diagnostic
ranking. Those remain behind the original forecast acceptance gate.
