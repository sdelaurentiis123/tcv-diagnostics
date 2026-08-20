A single, fixed, condition-independent, global linear residual distribution learned from adjacent 85604 training frames does not describe later 85604 residuals well.

# Paper 0 Phase 3.5 decision memo

**Authoritative job:** `6907468` at commit `5f7124075ab1510790c62789950ec00a878846d6`

**Scope:** simulation 85604 only. The guard was not read. Simulation 85606 remains unopened. This phase trained no production neural model and performed no assimilation.

## What this memo does and does not conclude

Phase 3.5 localizes why K4's fixed global linear residual model failed. It does not say that stochastic emulation of 85604 is impossible, and it does not reinterpret K4 as a test of FGN, PDE-Refiner, or diffusion.

## Ranked explanations

1. **codec or predictor non-equivariance — strong evidence.**
   13/13 representative states cross a frozen equivariance criterion; median equivariance/base-error ratio=0.098, median modulo-4 ratio=1.019.

2. **invalid/nonstationary interval — strong evidence.**
   10 material T00-to-V02 shifts exceed 0.5 pooled SD with block CI excluding zero; 0 targets have time-only R2 >= 0.10.

3. **forecast-state-dependent covariance — strong evidence.**
   15 scalar targets spanning 3 target families satisfy the all-three-block context-probe rule; median fixed-seed B5 covariance-family change=0.086 (corroborating only).

4. **history-dependent hidden state — strong evidence.**
   5 scalar targets spanning 3 target families improve at the frozen history threshold in all three validation blocks.

5. **insufficient or incorrect retained state — strong evidence.**
   2 target families pass the all-three-block >=10% exact-state improvement rule (0 by causal neighbors; 2 scalar targets by normalized probe RMSE).

6. **coherent transport in an inappropriate Eulerian representation — moderate evidence.**
   unambiguous=0.000, median |shift|=11.00/88, median H1 energy reduction=0.000, aligned full-span gain=0.000, transported-persistence gain=0.671.

7. **insufficient effective sample size — none evidence.**
   0.000 of material residual/transport observables have ESS<20; 378-to-420 prefix mean capture gain=0.0004.

8. **unexplained failure — none evidence.**
   Reserved for the case in which no preregistered mechanism reaches moderate evidence.

## Representation companion result

**strong evidence:** best method=toroidal_Fourier_separated_complex_KL; 2 budgets improve later variance transfer by >=0.10; maximum gain=0.175; consistently improved dependence/transport families=cross_field_covariance,cross_spectrum_coherence.

The representation comparison is used to decide whether localization or multiscale structure transfers chronologically. It is not an architecture competition and no target-block energy selected its coefficients.

## Interpretation

The decision follows the preregistered priority: state/protocol validity, stationarity, coherent transport/equivariance, context/history, and only then stochastic capacity. Truth-assisted shifts and truth-projected residual coefficients are explicitly nondeployable diagnostic upper bounds.

Recommended next action: repair interval/conditioning
