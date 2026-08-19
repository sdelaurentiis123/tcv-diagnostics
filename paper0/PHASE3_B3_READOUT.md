# Phase 3 B3 functional-generative one-seed readout

**Training job:** `6898702`  
**Full evaluation job:** `6899073`  
**Final amended gate job:** `6899224`  
**Development simulation:** 85604 only  
**Held-out simulation 85606:** unopened  
**Decision:** B3 fails its frozen one-seed gate; do not replicate it or use it
for O3, assimilation, or diagnostic ranking

## Executive conclusion

The B3 functional-generative retrofit produced a real, non-collapsed ensemble
and materially improved the marginal probabilistic score over its deterministic
parent. It retained almost the same ensemble-mean point accuracy, preserved
the evaluated member-expected cross-phases and cross-coherences, and recovered
useful mean and event structure at the separatrix.

It did **not** produce the joint probabilistic forecast Paper 0 needs. Only one
of five fields met the strict spread--skill and coverage criteria, the
ensemble-mean field generally failed to place material toroidal structure in
the correct next-frame realization, mode-resolved uncertainty was strongly
underdispersed, and none of the four separatrix transport quantities was
calibrated. All
three frozen metric families therefore fail in every chronological block.

The result cleanly supports the predeclared negative conclusion:

> Better marginal fair CRPS does not by itself preserve the mode-resolved,
> cross-field distribution required for calibrated nonlinear transport.

This is a one-step development result on the model-selection portion of 85604,
not an autonomous-rollout result or held-out claim.

## What B3 tested

B3 is **not diffusion**. It starts from the selected deterministic
`C5P-H1` transition and adds one low-dimensional random input to the complete
transition network. For each ensemble member, one 32-component Gaussian
vector is embedded to 256 components and shared across every spatial token.
Each of the 16 transformer blocks has its own adapter that turns the shared
embedding into block modulation. One sampled vector and one network pass
produce one member.

The task is one saved step:

~~~text
exact frame t-1  ->  distribution for frame t
~~~

The five modeled channels are `[Ne, Pe, Pi, phi, Vi]`. Absolute time is not an
input. The saved-frame interval is 3.131905426352636 microseconds. The training
targets are `[2,432)`, the guard `[432,496)` is unread, and all 126 validation
targets `[498,624)` are evaluated chronologically. The stored toroidal period
is five, so native Fourier index and full-torus mode number obey `n=5k`.

The seed-1701 `C5P-dcae_l10` codec and training-only normalization remain
frozen. Fine-tuning means that B3 loads the existing deterministic H1
transition, adds the new stochastic parameters, and optimizes both the old
transition weights and new noise path at different learning rates. It does
not restart the transition from random weights, and it does not alter the
codec.

Training uses two members per target and minimizes equal-channel fair CRPS in
decoded standardized field space. No spectrum, cross-phase, coherence, flux,
transport, PDE residual, or other physics-derived term enters the loss or
checkpoint selection. That separation is what makes the failed physics gate
scientifically informative rather than circular.

## Provenance and execution

| Stage | Job | Commit | Result |
|---|---:|---|---|
| full training | `6898702` | `a2a17cf3` | 100 epochs, 2,700 steps; selected epoch 72 |
| matched H1 comparator | `6899063` | `d029055a` | completed before B3 acceptance |
| first evaluator preflight | `6899064` | `d029055a` | schema error before any forecast; no scientific result |
| repaired evaluator smoke | `6899071` | `aa96db0c` | four targets, M32, completed |
| full independent evaluation | `6899073` | `aa96db0c` | 126 targets, M32, completed |
| original reduction | `6899154` | `aa96db0c` | numerical result complete; one false path-identity failure |
| amended reduction | `6899224` | `bade0646` | integrity clean; numerical decision unchanged |

The selected checkpoint SHA-256 is
`0e0fdca97f13e2e33934d667167294d293cfc6ceedd9dee8b0504bf724acdbe9`.
The 14,535,252,816-byte M32 forecast SHA-256 is
`0f5c97b20fbf7ef32f2bd2b9695dc173d78155dcde356ef5b1a451dc4276e3ef`.
The complete score SHA-256 is
`c32508a85a68859aa676d2fada4f76a304984fea5988c81fb106ae6f724654d0`.
Forecast generation closed and hashed this artifact before validation truth
was opened. Both training and evaluation completed their required online W&B
runs; immutable local artifacts, rather than W&B, remain authoritative.

The original reduction compared complete `{path, sha256}` checkpoint records.
Two valid Ceph aliases named the same bytes, which caused a false integrity
failure despite identical hashes. The amended gate compares content identity
by SHA-256. It changes no forecast, score, threshold, family reducer, or
scientific number. The original gate is retained; the amended gate passes all
integrity checks and reaches the same scientific failure decision.

## Field accuracy and marginal calibration

The deterministic H1 forecast is a degenerate distribution, so its MAE is the
corresponding CRPS reference. Values below one in the final column mean that
the B3 marginal distribution has a better proper score than that deterministic
reference.

| Field | Parent H1 MAE | B3 mean MAE | B3 fair CRPS | fCRPS / parent MAE | Spread--skill |
|---|---:|---:|---:|---:|---:|
| Ne | 0.04338 | 0.04375 | 0.03201 | 0.738 | 0.704 |
| Pe | 0.03269 | 0.03255 | 0.02325 | 0.711 | 0.798 |
| Pi | 0.04201 | 0.04198 | 0.02999 | 0.714 | 0.787 |
| phi | 0.04560 | 0.04524 | 0.03296 | 0.723 | 0.726 |
| Vi | 0.06509 | 0.06596 | 0.04720 | 0.725 | 0.839 |

Aggregated over equal-weight channels:

- ensemble-mean MAE is `0.04590`, or `1.0031` times parent H1;
- ensemble-mean RMSE is `0.07942`, or `0.9972` times parent H1;
- fair CRPS is `0.03308`, or `0.7230` times parent H1 MAE;
- fair CRPS is `0.5988` times the frozen best uncompressed-reference MAE;
- corrected spread--skill is `0.7897`.

All five fields improve fair CRPS, but the strict spread--skill interval is
`[0.80,1.25]` and only Vi enters it. Only one field also meets all primary
coverage tolerances. In the private-flux region, the widest M32 interval
coverage is below the frozen 0.75 lower bound for Ne (`0.696`), Pe (`0.730`),
and phi (`0.725`). The ensemble therefore has useful spread without having
enough spread in the required places and fields.

The aggregate M16 and M32 fair-CRPS values are `0.0330897` and `0.0330824`, a
relative difference of only `2.22e-4`. The failure is not explained by
stopping at 32 ensemble members.

## Toroidal spectra and cross-field structure

The evaluated material bands are `k=1..3` (`n=5..15`), `k=4..5`
(`n=20..25`), and `k=6..7` (`n=30..35`). Expected member power asks whether
the ensemble has the right fluctuation amplitude. The frozen realization-
coherence gate asks whether the ensemble-mean forecast field places that
structure in the correct next-frame realization. A model can pass the former
while failing the latter.

Across all fields and material bands, 11 of 15 power checks pass, but only 4
of 15 realization-coherence checks pass. In the highest material band:

| Field | Expected-member power ratio, `k=6..7` | Ensemble-mean-field realization coherence, `k=6..7` |
|---|---:|---:|
| Ne | 0.692 | 0.443 |
| Pe | 0.643 | 0.486 |
| Pi | 0.652 | 0.473 |
| phi | 0.726 | 0.303 |
| Vi | 0.910 | 0.0068 |

The frozen power interval is `[0.75,1.30]` and the realization-coherence
minimum is `0.80`. Vi illustrates the distinction sharply: its high-band
power is close to target, but that power is almost entirely in the wrong
next-frame realization.

All nine member-expected cross-phase checks pass, with absolute errors from
`0.48` to `1.98` degrees against a 20-degree limit. All nine member-expected
cross-coherence-change checks also pass, ranging from `0.0089` to `0.0951`
against a `0.15` limit. These are calculated from the mean of the member-wise
cross-spectra, not from cross-products of ensemble-mean fields. They are
genuinely positive joint-mean results, but do not establish a calibrated
member-wise joint distribution: mode and cross-spectrum coverage remains
strongly underdispersed, especially at `k=6..7`.

## Nonlinear transport

Transport is calculated separately for every ensemble member using the
frozen geometry-aware metric engine. It is never calculated only from the
ensemble-mean fields.

| Quantity | Strict relative L2 | Strict correlation | Strict sign disagreement | Separatrix relative L2 | Separatrix correlation | Separatrix fCRPS / H1 error | Separatrix spread--skill |
|---|---:|---:|---:|---:|---:|---:|---:|
| particle | 0.738 | 0.712 | 0.177 | 0.232 | 0.800 | 0.695 | 0.562 |
| electron internal energy | 0.744 | 0.708 | 0.175 | 0.181 | 0.912 | 0.719 | 0.604 |
| ion internal energy | 0.738 | 0.712 | 0.175 | 0.203 | 0.850 | 0.720 | 0.530 |
| total internal energy | 0.741 | 0.710 | 0.175 | 0.187 | 0.890 | 0.713 | 0.576 |

The strict geometry-aware facewise relative-L2 gate is at most `0.40`; all
four quantities fail at approximately `0.74`. Their correlations and sign
statistics pass, so the forecast retains temporal/sign signal while missing
local amplitude and structure.

At the separatrix, all relative-L2, correlation, sign, event-conditioned, and
fair-CRPS-versus-H1 checks pass. That is a meaningful positive result: B3 has
useful separatrix mean and event information. But none of the four transport
quantities is probabilistically calibrated. Separatrix spread--skill is only
`0.53--0.60`, and interval-coverage errors are large. A good mean separatrix
series is therefore not yet a trustworthy transport ensemble.

## Gate result

| Family | Numerical checks | Failed checks | Passing chronological blocks | Required blocks | Result |
|---|---:|---:|---:|---:|---|
| field | 54 | 6 | 0 | 5 | fail |
| spectral/cross-field | 148 | 59 | 0 | 5 | fail |
| transport | 77 | 6 | 0 | 5 | fail |

All required numeric values are finite and all provenance checks pass. The
failure is scientific, not mechanical. The frozen post-gate instruction is
`stop_B3_and_diagnose_before_replication`.

## What this localizes, and what it does not

The earlier O1 ladder showed that the `C5P-dcae_l10` codec passes the frozen
reconstruction gates. B3 therefore does not point first to codec capacity.
The failure appears after reconstruction, in the learned transition and its
stochastic representation. B3 shows specifically that global functional
noise plus a marginal field score can improve one-point uncertainty without
learning enough mode-resolved, cross-field covariance for nonlinear
transport.

This experiment does not uniquely prove whether the remaining limitation is
the deterministic mean transition, the low-dimensional/shared form of the
noise, or the marginal objective. It also does not test autonomous error
accumulation. That is why tuning B3 after viewing this result, repeating more
seeds, or jumping directly to assimilation would obscure rather than answer
the next question.

B2 provides useful context but is not a matched comparator: B2 conditions on
two frames and uses iterative latent diffusion, whereas B3 conditions on one
frame and uses one functional-noise pass. B3 improves B2's cross-coherence
count from 7/9 to 9/9 and obtains one strictly calibrated field rather than
zero, while both have 11/15 power and 4/15 realization-coherence checks. B3's
strict transport relative-L2 is roughly `0.74`, compared with roughly `0.61`
for B2. These differences are descriptive; they do not isolate architecture
because the histories and conditioning tasks differ.

## Decision and next rung

B3 seeds 1702 and 1703 are not authorized. O3, assimilation, diagnostic
ranking, and 85606 remain closed. The next action is to freeze a separately
justified B4 PDE-Refiner protocol before implementing or training it.

The B4 question should be narrow: can iterative error refinement repair the
one-step realization/spectral failure while retaining the already useful
mean cross-field and separatrix structure under a comparable compute budget?
If it cannot, the oracle ladder points next toward B5, a deterministic mean
plus a **joint** stochastic residual over all fields. No B4 threshold or
architecture choice should be selected after viewing B4 outputs.

## Immutable evidence

The tracked full evaluation manifest is
`paper0/results/phase3_b3_fgn_evaluation_full_6899073.json`, byte-identical to
the Rusty root manifest with SHA-256
`87b6ea353bfe9928404f01d1b494c94bfd2491395c28c0ec0a46105f0ee5e20c`.

The tracked compact gate record is
`paper0/results/phase3_b3_fgn_one_seed_gate_6899224.json`. It points to the
complete immutable final gate at:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
phase3_b3_fgn_acceptance/job_6899224/gate/final_gate.json
~~~

That complete gate has SHA-256
`882ba10898bbf132eea7713098202d8e814e3f709e8693c5b25366c52ffbc391`.
