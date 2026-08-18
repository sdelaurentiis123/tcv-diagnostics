# Phase 2 O2 deterministic one-step readout

**Result:** complete  
**Scientific job:** `6896117`  
**Execution commit:** `5183023c4df3a38bd6821f7e7cf587507aacc241`  
**Development run:** `85604` only  
**Held-out run `85606`:** unopened  
**Frozen disposition:** stop and report deterministic one-step failure

## Executive conclusion

The accepted `C5P-dcae_l10` codec reconstructs the evaluated information, but
the deterministic LOLA-style transition does not evolve that information
faithfully for even one saved step. All six independently trained transitions
beat the uncompressed forecast references on ordinary field error. All six
also pass the frozen band-aggregated cross-field checks. Nevertheless, every
seed fails both the realization-level spectral gate and the nonlinear
transport gate.

This is a useful fault-localization result rather than a null result:

1. the main failure is no longer attributable to the codec alone;
2. it appears before autonomous error accumulation can occur;
3. an additional exact history frame does not repair it;
4. low field error and marginal cross-field agreement are insufficient
   evidence for transport-faithful dynamics.

No O3 rollout, stochastic model, assimilation experiment, diagnostic ranking,
or access to `85606` is authorized by this result.

## What O2 tested

`C5P` contains the five directly modeled fields
`[Ne, Pe, Pi, phi, Vi]`. The frozen codec is `dcae_l10`, the approximately
10-to-1 scalar-compression representation that passed the complete O1 codec
gate at seeds `1701`, `1702`, and `1703`.

O2 is teacher-forced one-step prediction:

- `C5P-H1` receives one exact simulation frame and predicts the next saved
  frame;
- `C5P-H2` receives two exact ordered simulation frames and predicts the same
  next frame.

The six transition models were trained separately. For each validation target,
the context comes from truth rather than from a previous model prediction.
Therefore O2 measures one-step transition error without autonomous rollout
accumulation.

The scientific evaluation covers targets `[498,624)`: 126 saved frames at
`3.131905426352636` microseconds per frame, partitioned chronologically into
six blocks of 21 frames. Forecast generation was context-only; target truth
was opened later by a separate scoring stage. Physics quantities were used
only for evaluation.

## Aggregate field result

The best applicable uncompressed reference for both aggregate RMSE and MAE was
the frozen spectral AR(1) reference:

| Forecast | Mean standardized RMSE | Improvement | Mean standardized MAE | Improvement |
|---|---:|---:|---:|---:|
| Spectral AR(1) | 0.095656 | reference | 0.056301 | reference |
| `C5P-H1`, 3 seeds | 0.080037 | 16.33% | 0.045450 | 19.27% |
| `C5P-H2`, 3 seeds | 0.080619 | 15.72% | 0.045688 | 18.85% |

Every seed beats persistence for all five fields, both over the full interval
and within all six chronological blocks. The result is highly consistent
across seeds. It is real forecast skill, but it is not sufficient physics
skill.

The extra history frame provides no benefit here. Relative to H1, the H2 mean
RMSE is 0.73% higher and its mean MAE is 0.52% higher. This does not prove that
history is never useful. It shows that one additional saved C5P frame does not
repair this transition under the frozen architecture and training protocol.

## Why the spectral gate failed

Because the simulation stores one fifth of the full torus, stored Fourier
index `k` maps to full-torus mode number `n=5k`. The evaluated bands are:

| Stored band | Full-torus band |
|---|---|
| `k=1..3` | `n=5..15` |
| `k=4..5` | `n=20..25` |
| `k=6..7` | `n=30..35` |

The gate tests both power and realization coherence with the actual next
frame. Reproducing approximately the correct amount of power is not enough if
that power is placed in the wrong phase or spatial realization.

Two middle-band results are genuinely good: all six seeds pass `Pe` and `Pi`
at `k=4..5`. The complete spectrum is not faithful, however:

- `Ne`, `Pe`, and `Pi` at `k=6..7` retain only about 62% to 70% of truth power,
  with realization coherence only about 0.40 to 0.48;
- `phi` at `k=6..7` retains about 71% to 75% of truth power, with coherence
  about 0.27 to 0.30;
- `Vi` has plausible power in `k=4..7` but almost no realization coherence:
  about 0.065 to 0.092 in `k=4..5` and about 0.004 to 0.006 in `k=6..7`;
- none of those failing high-band checks passes even one chronological block.

This distinguishes spectral amplitude from predictive phase. In particular,
the `Vi` result shows that a forecast can reproduce a power spectrum while
placing the fluctuations almost entirely in the wrong realization.

## Why the cross-field gate passed but transport failed

All six seeds pass every frozen band-aggregated `Ne-phi`, `Pe-phi`, and
`Pi-phi` cross-phase/coherence-change check in every chronological block. This
does not contradict the spectral failure. The two checks ask different
questions:

- the cross-field gate asks whether forecast fields retain their relationship
  to each other after aggregation within a toroidal band;
- the realization-coherence gate asks whether the forecast places those
  fluctuations where the next true frame places them;
- the transport operator additionally depends on geometry-aware local
  gradients, faces, signs, and nonlinear field products.

The transition can therefore generate mutually consistent forecast fields
that are jointly displaced or smoothed relative to the true next state. That
is enough to preserve the aggregate relationship while corrupting local flux.

## Transport result

Every seed fails the complete transport gate. Across all 24 combinations of
six seeds and four transport quantities, no strict-face comparison passes.
Strict-face relative L2 lies between about 0.74 and 0.77, far above the frozen
maximum of 0.40.

Separatrix-integrated heat transport is substantially closer than local
strict-face transport:

| Quantity | Relative-L2 range | Correlation range | Normalized-bias range |
|---|---:|---:|---:|
| Particle | 0.257–0.300 | 0.770–0.810 | -0.229 to -0.164 |
| Electron internal energy | 0.181–0.216 | 0.911–0.927 | -0.175 to -0.123 |
| Ion internal energy | 0.215–0.236 | 0.837–0.870 | -0.169 to -0.116 |
| Total internal energy | 0.193–0.223 | 0.888–0.907 | -0.173 to -0.120 |

These integrated quantities often satisfy the full-interval error and
correlation thresholds, but they are systematically low and temporally
unstable. Only one of the 24 seed-by-quantity separatrix checks passes the
complete overall-plus-five-of-six-block rule. Particle flux passes none.

The proper conclusion is not that the model contains no transport signal. It
contains useful integrated heat-flux signal, but not enough local or
block-stable fidelity for the stated Paper 0 acceptance claim.

## Fault localization

The current oracle ladder says:

1. **O1 codec reconstruction:** `C5P-dcae_l10` passes at all three seeds.
2. **O2 one-step field error:** all six transitions beat the references.
3. **O2 realization-level spectra:** all six transitions fail.
4. **O2 nonlinear transport:** all six transitions fail.
5. **O3/O4 rollout:** not run, because the failure is already present before
   autoregressive feedback.

This is stronger and cleaner than diagnosing a poor rollout after many model
steps. It tells us that recursively rolling these checkpoints out would only
compound a transition that already evolves the wrong fine-scale realization
and local flux.

## Frozen decision and next modeling question

Under the protocol committed before scientific scoring:

- accepted deterministic arms: none;
- passing H1 seeds: 0 of 3;
- passing H2 seeds: 0 of 3;
- new O3 protocol: prohibited;
- `85606`: remains blinded.

The next Paper 0 model must be evaluated against the same separation of field,
spectral, cross-field, and transport behavior. Thresholds must not be loosened
to rescue these checkpoints. A new model protocol may compare an existing
stochastic LOLA/diffusion baseline or a minimally repaired dynamics model, but
that is a new experiment—not a continuation of the failed O2 arm and not yet
authorized by this readout alone.

## Evidence

The compact immutable audit record is
`paper0/results/phase2_o2_evaluation_full_6896117.json`. It contains every
checkpoint, codec, forecast, score, and result hash; the cross-seed numerical
summary; external artifact hashes; compute accounting; and the W&B run.

The external result root is:

```text
/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/
  phase2_o2_evaluation_full/job_6896117/
```

The W&B record is:

```text
https://wandb.ai/sdelaurentiis123-columbia-university/
  tcv-diagnostics-paper0/runs/p0-o2-eval-full-6896117
```
