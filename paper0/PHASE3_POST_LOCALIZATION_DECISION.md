# Phase 3 post-localization decision memo

**Decision date:** 2026-08-19

**Development simulation:** TCV/Hermes 85604 only

**Held-out simulation 85606:** unopened

**Authority:** this memo selects one prospective read-only experiment. It does
not authorize model training, O3/O4/O5, assimilation, diagnostic ranking,
steering, or access to 85606.

## Decision

Do not train another stochastic architecture yet.

The next experiment is one training-fitted, chronologically validated
Karhunen--Loève residual-covariance oracle. It will determine whether the
missing coherent B5 covariance can be represented by a compact linear
function-space basis before we ask a neural network or coefficient-space
diffusion model to predict that basis's coefficients.

The experiment has two linked tiers:

1. a truth-projected validation oracle that measures representation capacity;
2. one condition-independent Gaussian KL ensemble around the frozen H1 mean,
   at a rank selected from training eigenvalues only, that measures how far a
   static coherent covariance prior gets without learned conditionality.

This is one representation experiment, not a model sweep.

## Evidence selecting this experiment

The completed B5 localization supports L2 covariance-organization limitation
for all four transport quantities. Exact-separatrix local spread--skill is
`0.996--1.002`, while integrated spread--skill is only `0.413--0.485`.
Scaling anomalies to calibrate integrated transport would raise local
spread--skill to `2.06--2.43`. Scalar inflation is therefore rejected.

The same analysis rejects explicit one-frame residual history as the next
priority. A training-frozen AR(1) correction improves equal-field RMSE by
`1.72%`, below the predeclared `2%` threshold, does not beat B5's field mean,
and worsens integrated transport.

Eleven spatial or regional dependence identities exceed observed
training-to-validation drift in at least five of six blocks. However, several
other identities do not. One realized trajectory therefore localizes a robust
organization failure without identifying a unique conditional covariance.

The smallest remaining question is not “which generator should we train?” It
is:

> Does a moderate-rank coherent residual subspace contain the validation
> covariance and transport structure that B5 misses?

If the answer is no, a low-rank stochastic head is the wrong next model. If
the answer is yes, the oracle gives a defensible representation and rank range
for a later conditional coefficient model.

## Literature synthesis

Park et al.'s 2026 Diffusion Last Layer represents an output field with an
input-dependent low-rank, Karhunen--Loève-like basis and learns diffusion in
coefficient space. Their rank ablation shows an important warning:
reconstruction improves monotonically with rank, but distributional scores do
not. They use rank 64 as a moderate default and report that stochastic
distributional fidelity benefits from thousands rather than hundreds of
examples. This motivates a rank oracle; it does not validate DLL on our data.

- Park et al., *Generative Neural Operators through Diffusion Last Layer*,
  ICML 2026, <https://arxiv.org/abs/2602.04139>.

Hidajat's 2026 Martingale Neural Operator directly parameterizes a positive
semidefinite low-rank covariance factor. Its Gaussian experimental
instantiation makes the value of explicit covariance rank clear, but it also
uses ensemble residual samples and notes that full cross-covariance is only
indirectly constrained by its diagonal NLL and factor rank. Our single
trajectory does not provide replicated outcomes for identical conditions, so
we should not port that loss and claim conditional covariance identification.

- Hidajat, *Martingale Neural Operators: Learning Stochastic Marginals via
  Doob--Meyer Factorization*, <https://arxiv.org/abs/2605.15806>.

Pacchiardi et al. show that proper spatial scores can train generative
forecasts on dependent temporal data and that a proper variogram term may be
combined with a strictly proper score. This supports a later
dependence-sensitive objective if representation is adequate. It does not
remove the need to test whether the chosen representation can carry the
dependence first.

- Pacchiardi et al., *Probabilistic Forecasting with Generative Networks via
  Scoring Rule Minimization*, JMLR 2024,
  <https://www.jmlr.org/papers/v25/23-0038.html>.

The 2026 spatial-copula downscaling work by Huk et al. explicitly separates
marginal modeling from spatial dependence modeling. That staged logic matches
our evidence: B5's local marginal amplitude is useful, while dependence is the
failure. The rainfall-specific copula is not proposed for plasma fields.

- Huk et al., *Probabilistic Rainfall Downscaling: Joint Generalized Neural
  Models with Censored Spatial Gaussian Copula*, JMLR 2026,
  <https://www.jmlr.org/papers/v27/23-1381.html>.

## Why this is not a return to the old DCAE question

The old f8/L10/z44 question concerned nonlinear compression of the complete
plasma state before deterministic or stochastic dynamics. That representation
had to preserve every field and every downstream physical diagnostic.

The proposed KL oracle acts on a different object: the error remaining after
the frozen H1 mean has already predicted the next state. It asks whether the
*residual covariance* is low rank. The basis is linear, inspectable, fitted
only on training residuals, and evaluated separately at each rank. No DCAE
tokenization, decoder, or latent dynamics is involved.

A positive residual-rank result would not retroactively validate z44 or any
state codec. A negative result would tell us not to build DLL- or MNO-like
residual heads for this dataset.

## Alternatives rejected for the next experiment

### Scalar noise or lambda tuning

Rejected because L1 is false and the same scalar correction would make local
transport strongly overdispersed.

### One-frame history conditioning

Rejected because L4 is false and the teacher-forced history correction
worsens transport.

### Mean-prediction regularization alone

B5 already uses a separately frozen deterministic H1 mean plus a residual
generator. Its dominant localized failure is not loss of a deterministic mean
anchor. Mean anchoring may remain useful inside a later architecture, but it
does not directly test the missing off-diagonal covariance.

### Immediate variogram-loss fine-tuning

Deferred because it would change the objective before establishing that the
current or proposed representation can express the required dependence. A
generic proper dependence term remains a later arm if the representation
oracle passes.

### Immediate DLL, MNO, or another full diffusion run

Rejected as premature. Their low-rank assumption and coefficient dimension
must be tested on our residuals, and our 430 adjacent training targets are not
independent stochastic realizations or replicated outcomes for the same
condition.

### Opening 85606

Rejected. Architecture, rank, one-step acceptance, O3, assimilation, and
diagnostic settings are not frozen at a passing forecast model.

## Selected experiment in one sentence

Fit a joint five-field residual KL basis on the 430 training residuals, measure
rank-resolved reconstruction on the chronological 126-target validation
region, and evaluate exactly one training-rank-selected static Gaussian KL
ensemble around the frozen H1 mean.

## Decision outcomes

The experiment must return one of four outcomes.

### K1: compact subspace and useful static covariance

The validation projection passes at rank at most 64, and the static KL
ensemble materially repairs integrated transport covariance without destroying
local spread. A later model should preserve a static coherent KL component and
learn only flow-dependent coefficient statistics or a local complement.

### K2: compact subspace but conditional coefficients required

The validation projection passes at rank at most 64, but the static KL
ensemble fails. A later DLL-like coefficient generator or conditional
low-rank covariance head is justified; the basis can represent the target, but
condition-independent coefficients cannot forecast it.

### K3: only moderate or high rank is adequate

No rank at most 64 passes, but rank 128 or higher does. A single compact global
head is not supported. The next architecture must be multiscale or
global-plus-local, and its cost must be compared honestly with field-space
diffusion.

### K4: training residual span does not transfer

Even the complete centered training span fails the validation representation
criteria. More architecture iteration on this single run is not justified.
Paper 0 must either obtain additional simulator trajectories or narrow itself
to a reproducible failure/benchmark analysis.

## Boundary after this memo

Only a separate committed
`protocol/PHASE3_RESIDUAL_KL_ORACLE_PROTOCOL.md` and matching manifest may
authorize implementation and one Rusty execution. No model training is
authorized by this memo.
