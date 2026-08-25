# Old-85604 Stage-2 multi-lead screen: final readout

**Result date:** 2026-08-25

**Development simulation:** 85604 only

**Held-out 85606:** not read

**Code commit:** `c5711e3c8b6e4d31291adb272c519c38a3763856`

**Slurm job:** `6936393` (`COMPLETED`, exit `0:0`)

**W&B:** [completed online run](https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldmultilead-j6936393-s1701)

## Outcome

The seed-1701 C5P operator passed every prospective multi-lead screen gate.
The intervention changed the temporal training-pair distribution and
fine-tuning schedule only. It did not change the architecture, add a codec,
or add a physics-derived loss.

- The selected checkpoint was epoch 4 after exactly 2,132 optimizer updates.
- The unweighted mean persistence-normalized score over leads
  `1,2,4,8,16` was `0.48813885947672625`; a score of one is persistence.
- The one-step-only parent scored `54.721883757255` when extrapolated over
  those five leads. This extreme contrast is primarily evidence that the
  parent's lead-time embedding was not trained for unseen leads, not a useful
  measure of architecture superiority.
- The selected model beat persistence for every C5P field at every lead.
- It improved all four longer leads relative to the bitwise parent.
- Its lead-1 shared `Ne`/`Pe`/`Pi` MSE was `0.005299497073090922`, slightly
  better than the parent's same-job value `0.005322341488745838` and below
  the frozen non-degradation threshold `0.005588458639715578`.
- Decision: authorize the prospectively defined seed-1702/1703 confirmation
  and, only after that confirmation, bounded validation rollouts.

`status: passed` means that training and mechanical gates passed.
`advance_to_three_seed_scaling: true` records the separate prospective
scientific decision.

## Direct finite-lead result

These are teacher-forced direct transitions: each prediction receives the
true current state and predicts one target at the stated saved-frame gap.
They are not autonomous rollouts.

| Lead (saved frames) | Shared derivative MSE | Skill versus persistence | Error / persistence |
|---:|---:|---:|---:|
| 1 | `0.00529950` | `0.8628` | `0.1372` |
| 2 | `0.00277300` | `0.5732` | `0.4268` |
| 4 | `0.00106308` | `0.4209` | `0.5791` |
| 8 | `0.00031803` | `0.4063` | `0.5937` |
| 16 | `0.00010849` | `0.2962` | `0.7038` |

Skill remains positive at every tested lead but declines at the longest
gap. The falling raw derivative MSE with lead is expected because the target
is the finite difference divided by the lead. The dimensionless ratio to
persistence is the appropriate across-lead comparison.

## Field-level persistence skill

Every entry is positive, satisfying the strict field-by-lead gate.

| Lead | `Ne` | `Pe` | `Pi` | `phi` | `Vi` |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.6744` | `0.8976` | `0.8915` | `0.8071` | `0.6286` |
| 2 | `0.5528` | `0.5769` | `0.5849` | `0.6315` | `0.5778` |
| 4 | `0.3609` | `0.4314` | `0.4433` | `0.4996` | `0.4118` |
| 8 | `0.3570` | `0.4073` | `0.4325` | `0.4019` | `0.4134` |
| 16 | `0.2812` | `0.2750` | `0.3215` | `0.3073` | `0.3294` |

## Learning curve

Checkpoint selection used validation only.

| Epoch | Mean training loss | Five-lead validation ratio |
|---:|---:|---:|
| 1 | `0.488284` | `0.651694` |
| 2 | `0.270117` | `0.529746` |
| 3 | `0.245182` | `0.496805` |
| 4 | `0.231661` | `0.488139` |

The validation improvement was monotonic over the frozen four epochs and
was already below persistence after one epoch.

## What was learned

The Stage-1 parent was genuinely useful at its trained one-frame lead but
failed when its lead embedding was queried at unseen gaps. Explicit
multi-lead supervision repaired that failure while retaining one-frame
accuracy. Therefore, part of the prior finite-horizon failure was a
training-distribution problem rather than evidence that the local C5P
operator architecture could not represent finite-time transitions.

This result does **not** establish autonomous rollout stability. It also does
not establish spectral, cross-phase, coherence, flux, transport-covariance,
probabilistic-calibration, assimilation, diagnostic-ranking, or steering
performance. The 2,129 training pairs come from the same chronological 85604
training interval and are not independent simulations.

## Why the result is valid

- The protocol was frozen before any parent result at leads 2--16 was read.
- The job loaded the selected Stage-1 parent bitwise and evaluated it at all
  five leads before optimizer construction.
- The one legacy config-serialization omission was handled explicitly:
  Stage-1 predates the `auxiliary_context_channels` record, whose only
  accepted historical default was verified as zero. No other config drift
  was accepted.
- Training used only frames `[0,432)`; guard `[432,496)` was not read;
  validation used `[496,624)`.
- All component scales were refit from the 2,129 authorized training pairs.
- No flux, spectrum, cross-phase, coherence, conservation, PDE residual, or
  other physics quantity entered the loss.
- Exact checkpoint reload and a seven-cell integer toroidal-shift
  equivariance check passed.
- All 2,132 optimizer updates ran, all metrics were finite, training loss
  decreased, and TF32 remained disabled.
- Slurm completed with exit `0:0`; W&B remotely reported `finished`.
- Both artifact manifests and all four checkpoint, result, W&B, scale, and
  parent-baseline hashes verified after completion.

## Provenance

- Frozen manifest:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage2_multilead_screen/input_freeze_commit_c5711e3c/manifest.json`
  (`SHA-256 0ac5ca2a33996a76399307f024463929246e2c61009b8364ad8c0791f7eab78f`).
- Result directory:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_stage2_multilead_screen/job_6936393_c5p_seed1701`.
- Tracked result:
  `paper0/results/post_ecrd_old_85604_stage2_multilead_6936393.json`
  (`SHA-256 6f4036ff6fd50a7090e60a351242f1a6ad00af6d3762322fe1075d22a9808c2a`).
- Best checkpoint: epoch 4,
  `SHA-256 35b82de5d16001dac99e1e97c5002b75d22fa9836dd5e83a90a68a4d6c64adaa`.
- Slurm elapsed time: 40 minutes; model wall time: 2,164.89 seconds;
  peak CUDA allocation: 0.8876 GiB on an A100 MIG 1g.20gb slice.

## Decision

The narrow multi-lead mechanism passed at seed 1701. The next experiment is
not another architecture variation: repeat the identical schedule from the
independent Stage-1 seed-1702 and seed-1703 parents. If the three-seed result
confirms the gate, compare direct finite-lead forecasts with bounded
autoregressive compositions at four and eight frames before any transport or
stochastic-emulator claim.
