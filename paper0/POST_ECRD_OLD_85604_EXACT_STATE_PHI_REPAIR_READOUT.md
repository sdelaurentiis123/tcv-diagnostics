# Old-85604 exact-state potential-repair readout

**Result:** neither screened arm advances

**Development simulation:** 85604 only

**Held-out 85606 read:** no

**Execution commit:** `b146879b05582136e3ab5d1d7156eab2db1bb20e`

**Slurm array:** `6935835`

## Question

The matched Stage-1 comparison found that the local codec-free E6B model
learned every saved evolved field better than persistence, but its one-step
Ne/Pe/Pi validation MSE was about 46% higher than the C5P control. This screen
tested whether that gap was repaired by either:

1. supplying current, history-only `phi` to the same local E6B model; or
2. supplying the same `phi` to a matched nonlocal axial operator.

The screen used the frozen 85604 training frames `[0,432)`, read no guard
frame in `[432,496)`, and selected checkpoints only on validation frames
`[496,624)`. Both arms used seed 1701, 12 epochs, one-frame context, one-step
targets, and the same persistence-normalized component-balanced state loss.
No physics-derived quantity entered training.

## Result

| model | parameters | best epoch | shared Ne/Pe/Pi derivative MSE | change from local E6B without `phi` | ratio to C5P | advances |
|---|---:|---:|---:|---:|---:|---:|
| C5P local control | comparable | 12 | `0.00532234` | - | `1.000` | retained control |
| E6B local control | comparable | 12 | `0.00777215` | - | `1.460` | unresolved ablation |
| E6B local + current `phi` | `2,182,352` | 12 | `0.00776223` | `-0.128%` | `1.458` | no |
| E6B axial + current `phi` | `2,131,544` | 12 | `0.00916430` | `+17.912%` | `1.722` | no |

The preregistered repair gate required at least a 15% reduction relative to
the seed-1701 local E6B control, or an MSE no greater than `0.00660633`.
Neither arm passed.

Every predicted E6B field retained positive persistence-relative skill in
both new arms. All exact-reload, finite-metric, optimizer-update, no-toroidal-
stride, circular-shift-equivariance, and boundary-invariance gates passed.
The negative result is therefore not a numerical collapse or failed launch.

## Field attribution

Adding current `phi` to the local model did not have a uniform effect relative
to the E6B control. Pe MSE changed by `-1.90%` and Pi by `-4.98%`, while Ne
changed by `+7.34%`, NVi by `+4.71%`, NVe by `+0.43%`, and Vort by `+1.24%`.
The gains and losses nearly cancelled in the shared Ne/Pe/Pi metric.

The axial arm was worse than the local E6B control in every field at its best
checkpoint. Its training loss continued to fall while chronological
validation improvement slowed, consistent with under-transfer of the more
globally expressive operator on this one early-to-later 85604 split.

## Interpretation boundary

This establishes only that current `phi` and the tested matched axial
operator do not repair the old-85604 **teacher-forced one-step** E6B gap. It
does not establish a rollout, spectrum, cross-phase, transport, stochastic
calibration, assimilation, diagnostic-ranking, steering, or 85606 result.

Current truth-derived `phi` was causal for this teacher-forced screen. A free
E6B rollout would still have to reconstruct `phi` after every predicted step
with the already validated Hermes/BOUT++ elliptic operator before building
the next context.

## Decision

Neither arm is scaled to seeds 1702/1703 or chronological-block scoring. The
next bounded representation repair supplies both analytically derived C5P
coordinates, current `phi` and current `Vi`, to the local E6B transition while
continuing to predict the full evolved E6B state. This tests coordinate
learnability without introducing future truth, stochastic capacity, or a
physics loss.

## Authoritative artifacts

- local result: `paper0/results/post_ecrd_old_85604_local_current_phi_6935836.json`,
  SHA-256 `4246a6000d77aa5498cccd87da1bb2458602e03ef97f994d0d67c25bc0ec7525`;
- axial result: `paper0/results/post_ecrd_old_85604_axial_current_phi_6935835.json`,
  SHA-256 `b178803893490f3633dc220af277483732895b3b07e3dd6d0deb1678b1d077da`;
- local W&B run:
  `https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldphirepair-local_current_phi-a6935835-t0-s1701`;
- axial W&B run:
  `https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldphirepair-axial_current_phi-a6935835-t1-s1701`.

The complete checkpoints, environment records, commands, logs, Slurm records,
and nested checksum inventories remain under the immutable Ceph result root
`paper0/post_ecrd_old_85604_exact_state_phi_repair_screen/array_6935835`.
