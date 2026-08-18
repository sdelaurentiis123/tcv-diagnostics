# Phase 3 B2 probabilistic one-step readout

**Original gate job:** `6897564`  
**A016 gate-only rerun:** `6898348`  
**Development run:** 85604 only  
**Held-out run 85606:** unopened  
**Decision:** B2 fails; O3 and assimilation remain closed

## Executive conclusion

The three-seed LOLA-style latent-diffusion baseline is useful but not adequate
for Paper 0. It improves one-step ensemble-mean field error and marginal fair
CRPS, preserves band-averaged cross-phase well, and contains meaningful
integrated transport signal. Its ensemble is nevertheless too narrow, its
members do not place enough spectral structure in the correct next-frame
realization, and its local and probabilistic transport metrics fail.

The post-result A016 correction changes no scientific value and does not
change the decision. It only stops treating event-conditioned accuracy as a
numeric forecast error in the third chronological block, where the truth has
zero events above the frozen training-only threshold. After the correction,
all three seeds satisfy the catastrophic finite-metric bound, but zero seeds
pass the complete field, spectral, or transport gate. The architecture still
fails 106 of 249 median numerical checks.

The immutable original matrix has SHA-256
`cd5d3a22b1a5f665c493417c3ea47bc7fd21d731e116f35a6a84eae68b462fd6`.
The amended matrix has SHA-256
`4f054365d32d3e1725091ba58c8fa014f104e204748217dda482045a6c0df600`.
The compact machine-readable record is
`results/phase3_b2_event_eligibility_amendment_6898348.json`.

## What B2 tested

B2 conditions on the two exact preceding C5P frames and generates a
32-member distribution for the next saved frame. It uses the accepted
same-seed `C5P-dcae_l10` codec, a masked latent ViT, and 16 reverse diffusion
steps per member. It is a teacher-forced one-step conditional distribution,
not an autonomous rollout.

The five directly modeled fields are `[Ne, Pe, Pi, phi, Vi]`. The validation
targets are `[498,624)`, corresponding to 126 saved frames at
3.131905426352636 microseconds per frame. The stored toroidal period is five,
so full-torus mode number is `n=5k`.

## What genuinely improved

At the three-seed median, relative to the paired deterministic H2 transition:

| Quantity | B2 / deterministic H2 | Interpretation |
|---|---:|---|
| ensemble-mean RMSE | 0.899 | 10.1% lower |
| ensemble-mean MAE | 0.940 | 6.0% lower |
| fair CRPS / deterministic MAE | 0.684 | probabilistic score is materially lower |

The median fair CRPS is also 0.569 times the best uncompressed-reference MAE.
All nine material cross-phase checks pass, with errors between approximately
0.69 and 2.77 degrees. All primary M16-versus-M32 stability checks pass, so
the observed failure is not an artifact of stopping at 32 members.

Transport is not absent. All four separatrix relative-L2 values satisfy the
0.30 bound: 0.299 for particle transport and 0.239--0.274 for the three heat
quantities. All four separatrix sign-disagreement values are zero, and all
three heat-series correlations exceed 0.80. The model has useful integrated
transport timing and sign information.

## Why B2 still fails

### Marginal field calibration

The median field spread-skill ratios are:

| Ne | Pe | Pi | phi | Vi |
|---:|---:|---:|---:|---:|
| 0.617 | 0.692 | 0.732 | 0.671 | 0.732 |

A value near one would indicate matched ensemble spread and error scale. None
of the five fields reaches the primary lower bound of 0.80. The ensemble is
non-collapsed but systematically underdispersed.

### Realization-level spectral fidelity

Eleven of 15 material band-power checks pass, but only four of 15
realization-coherence checks pass. In the stored `k=6..7`, full-torus
`n=30..35` band, median realization coherence ranges from 0.040 for Vi to
0.503 for Pe. Vi is particularly diagnostic: its power ratio is 1.193 while
its realization coherence is only 0.040. The ensemble can carry approximately
the right amount of fluctuation power and still put it in the wrong
next-frame realization.

All nine cross-phase errors pass, but two high-band cross-coherence-change
checks fail: Ne-phi changes by 0.234 and Pi-phi by 0.171, above the 0.15 bound.
Low cross-phase error alone is therefore not enough to establish the joint
distribution needed by nonlinear transport.

### Transport

All four strict-face relative-L2 values are approximately 0.61, far above the
0.40 bound. Their correlations and weighted sign-disagreement values pass,
so the failure is primarily magnitude and local structure rather than complete
temporal or sign loss.

At the separatrix, all four normalized biases have magnitude 0.187--0.218 and
fail the 0.15 bound; the forecasts are biased low. Only particle transport
fair CRPS marginally beats its paired deterministic absolute-error score.
The other three transport fair-CRPS values are worse, and zero of four
transport quantities is probabilistically calibrated.

## What A016 changed

The frozen training-only thresholds produce no truth events in target block
`[540,561)` for any of the four transport quantities. The original scorer
correctly stored zero events, `defined=false`, and null event errors. The
original gate then incorrectly required those null values to be finite.

A016 preserves the original result and applies a separate versioned rule:

- a block is event eligible only when its truth event count is positive;
- all five eligible blocks must pass the unchanged event thresholds;
- the zero-event record must be internally consistent and is marked N/A;
- every non-event metric and the five-of-six complete-block rule remain;
- all three stored seeds are rerun consistently.

Every eligible event check passes for every seed. The all-required-numeric
catastrophic bound changes from false to true for all three seeds. No median
scientific check changes, no family changes from fail to pass, and the final
architecture decision remains fail.

## Decision

B2 must not supply forecast covariance for diagnostic ranking. More members
will not solve the observed underdispersion, because the M16-to-M32 results are
already stable. The next inexpensive hypothesis is a functional-noise retrofit
of the deterministic transition: one global noise vector shared spatially and
injected through conditional normalization, trained with marginal fair CRPS.

That branch must improve spread without sacrificing realization coherence,
cross-field structure, or transport. If it improves marginal CRPS but leaves
transport wrong, Paper 0 should report the important negative conclusion that
marginal calibration is insufficient for nonlinear scientific functionals.
