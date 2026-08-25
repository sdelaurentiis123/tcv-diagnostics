# Old-85604 exact-state derived-coordinate screen: final readout

**Result date:** 2026-08-25  
**Development simulation:** 85604 only  
**Held-out 85606:** not read  
**Code commit:** `deb6e141459548ac5e784a35b2ceffad51498af2`  
**Slurm job:** `6936156` (`COMPLETED`, exit `0:0`)  
**W&B:** [completed online run](https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldcoord-local_current_phi_vi-j6936156-s1701)

## Outcome

The exact-state local operator trained successfully when supplied with both
current `phi` and current `Vi`, but it did **not** pass the prospective repair
gate.

- Best shared `Ne`/`Pe`/`Pi` one-step derivative MSE: `0.007607179289215706`.
- Original local E6B baseline: `0.007772147896373167`.
- Improvement over that baseline: `2.1226%`.
- Required improvement: `15%`, or MSE no greater than
  `0.006606325711917192`.
- Decision: do not scale this arm to three seeds and do not run its rollout.

The machine-readable result uses `status: passed` to mean that the training
execution and mechanical gates passed. The separate scientific decision is
`advance_to_three_seed_scaling: false`.

## Frozen comparison

All learned rows below used seed 1701, one-frame history, one-frame lead,
training frames `[0,432)`, no reads from guard `[432,496)`, and checkpoint
selection over the complete chronological validation interval `[496,624)`.
The metric is comparable across rows only for the shared predicted fields
`Ne`, `Pe`, and `Pi`.

| Model interface | Best shared MSE | Change versus local E6B | Screen result |
|---|---:|---:|---|
| Local C5P control (`Ne, Pe, Pi, phi, Vi`) | `0.00532234` | `-31.52%` | reference only |
| Local exact state E6B (`Ne, Pe, Pi, NVe, NVi, Vort`; retained `Bphi`) | `0.00777215` | baseline | unresolved |
| Local E6B + current `phi` | `0.00776223` | `-0.13%` | failed repair gate |
| Axial-attention E6B + current `phi` | `0.00916430` | `+17.91%` | failed repair gate |
| Local E6B + current `phi` + current `Vi` | `0.00760718` | `-2.12%` | failed repair gate |

The new arm remains `42.93%` above the C5P shared-field error. This does not
prove that the C5P state is physically superior: the two models solve
different joint prediction problems. It does show that simply presenting the
two convenient derived coordinates does not explain most of the observed
optimization/transfer gap.

## Field-level result

The values are the best-checkpoint validation derivative MSEs. Negative
change is better.

| Predicted E6B field | Local + `phi` | Local + `phi` + `Vi` | Change |
|---|---:|---:|---:|
| `Ne` | `0.00803203` | `0.00761796` | `-5.16%` |
| `Pe` | `0.00669181` | `0.00663105` | `-0.91%` |
| `Pi` | `0.00856284` | `0.00857253` | `+0.11%` |
| `NVe` | `0.74053554` | `0.74038809` | `-0.02%` |
| `NVi` | `0.01172694` | `0.01160094` | `-1.07%` |
| `Vort` | `0.92122081` | `0.92415363` | `+0.32%` |

The aggregate gain is mostly a better `Ne` transition. It does not represent
a broad repair of the six-field dynamics.

Every field still beats zero-derivative persistence. The final
persistence-relative skills are:

| Field | Skill |
|---|---:|
| `Ne` | `0.5378` |
| `Pe` | `0.8393` |
| `Pi` | `0.8525` |
| `NVe` | `0.6702` |
| `NVi` | `0.6036` |
| `Vort` | `0.5229` |

## Why the result is valid

- The frozen manifest authorized exactly one architecture and one seed.
- Current/history `phi` and `Vi` were inputs; future auxiliary values were
  prohibited. A known-answer test poisoned target-frame `phi` and `Vi` with
  non-finite values and confirmed that the returned pair remained finite.
- `Vi` is causally reconstructible from predicted E6B state as
  `NVi / (2 * softFloor(Ne, 1e-7))`.
- A free rollout would still require the already validated external elliptic
  operator to reconstruct `phi`; no rollout was performed in this screen.
- The model had 2,183,000 parameters, only `0.0297%` more than the matched
  local-plus-`phi` control.
- All 1,296 optimizer updates ran. Training loss decreased and all validation
  metrics remained finite.
- Exact checkpoint reload passed. There was no toroidal stride, and the
  integer toroidal-shift and boundary-invariance gates passed.
- All 12 checkpoint hashes, the result, the W&B record, and both nested
  artifact manifests verified after completion.
- The run used no flux, spectrum, phase, coherence, PDE, conservation, or
  other physics-derived training loss.

## Provenance

- Frozen manifest:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_exact_state_derived_coordinate_screen/freeze_deb6e14_20260825/manifest.json`
  (`SHA-256 9b757b828a3508e5fff8c8e7374a14d9977cc6992da54bbd85b965e50f4c83a0`).
- Result directory:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_exact_state_derived_coordinate_screen/job_6936156_local_current_phi_vi_seed1701`.
- Tracked result:
  `paper0/results/post_ecrd_old_85604_local_current_phi_vi_6936156.json`
  (`SHA-256 d0040b3f14aab4478b1e6b7783eee1ec6d2b15b58360d89dc80e93f1da0505bd`).
- Best checkpoint: epoch 12,
  `SHA-256 c8d8e690bdac216d0da31a436b637f6e5e2541b247f334dda9df4f51de7d526c`.
- Slurm elapsed time: 15 minutes 54 seconds; model wall time: 892.25
  seconds; peak CUDA allocation: 0.9002 GiB.

## Scientific interpretation and decision

The result rejects the narrow explanation that the local exact-state model
mainly fails because it must internally recover both `phi` and `Vi` from the
saved evolved variables. Supplying those coordinates helps modestly, mainly
for `Ne`, but does not close the chronological-transfer gap.

This screen was teacher-forced, one-step, and deterministic. It says nothing
directly about free-rollout transport, stochastic calibration, assimilation,
or diagnostic ranking. Per the frozen failure rule, no additional
derived-coordinate variants are authorized. The next model experiment must
change the temporal/operator learning problem rather than append another
algebraically redundant input channel.
