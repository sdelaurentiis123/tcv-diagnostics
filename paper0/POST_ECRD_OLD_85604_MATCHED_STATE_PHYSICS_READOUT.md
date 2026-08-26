# Old-85604 matched-state physics readout

**Date:** 2026-08-27  
**Decision job:** `6948598`  
**Development data:** old simulation 85604 only  
**Held-out 85606 read:** no  
**New NERSC data read:** no  
**Training in this stage:** no  

## Outcome

The one-seed, equal-budget E6B transition did not improve the common-view
rollout physics relative to the C5P control.  It remained close on the shared
thermodynamic fields but failed after its predicted vorticity and boundary
state were mapped causally through the pinned Hermes/BOUT++ elliptic solve.

The prospective decision therefore retains C5P as the control and stops the
unchanged saved-state branch.  This is not evidence that a complete evolved
state is physically wrong or that stochastic emulation is impossible.  It is
evidence that the present compact local operator, objective allocation,
checkpoint rule, and 12-epoch budget did not exploit the E6B state well enough
for transport.

## Frozen comparison

Both transitions used seed 1701, the same old-85604 split, the same direct
multi-lead targets at 1, 2, 4, 8, and 16 frames, comparable 2.18-million
parameter operators, and the same 12-epoch budget.  Both selected epoch 12 at
the budget boundary.

The state views were:

- **C5P control:** `Ne, Pe, Pi, phi, Vi`.
- **E6B evolved-state candidate:** `Ne, Pe, Pi, NVe, NVi, Vort`, plus the two
  predicted `Bphi` boundary sides.

For E6B, `phi` was not copied from truth.  Seven bounded rollout candidates
were passed through the pinned exact elliptic operator using predicted
`Ne`, `Pi`, `Vort`, and predicted `Bphi` boundaries.  The truth-state replay
of this operator had already reconstructed stored `phi` to numerical
roundoff.  `Vi` was then derived as `NVi / (2 softFloor(Ne, 1e-7))`.

The comparison pooled horizon-4 and horizon-8 direct and truth-free composed
forecasts over three chronological validation blocks.  Physics quantities
were evaluation-only.

## Preregistered decision

Lower error is better.  Ratios are E6B divided by C5P.

| Quantity | E6B/C5P | Gate | Result |
|---|---:|---:|---|
| Shared `Ne, Pe, Pi` standardized RMSE | 1.058 | at most 1.10 | pass |
| Spectral-power absolute log error | 1.237 | at most 1.10 | fail |
| Complex density/pressure-potential cross-spectrum error | 1.173 | strictly below 1.00 | fail |
| Separatrix-transport relative L2 error | 5.872 | at most 0.90 | fail |

All primary scalars were finite and the causal exact-potential provenance gate
passed.  Three-seed confirmation of this unchanged E6B setup was not
authorized.

## Robustness of the transport failure

E6B lost all 84 matched separatrix-transport comparisons across horizons,
rollout compositions, time blocks, and transport quantities.

| Separatrix quantity | C5P median error | E6B median error | E6B/C5P |
|---|---:|---:|---:|
| Particle | 0.555 | 4.021 | 7.243 |
| Electron internal energy | 0.525 | 2.409 | 4.586 |
| Ion internal energy | 0.517 | 3.077 | 5.954 |
| Total internal energy | 0.515 | 2.739 | 5.317 |

The E6B/C5P transport ratio ranged from 2.94 for the horizon-8 repeated
four-frame map to 16.31 for the horizon-8 repeated one-frame map.  Thus the
aggregate failure is not caused by a single rollout composition or
chronological block.

## Where the state views diverged

The shared thermodynamic fields remained close:

| Shared field | E6B/C5P median standardized RMSE |
|---|---:|
| `Ne` | 1.087 |
| `Pe` | 1.026 |
| `Pi` | 1.012 |

The failure appeared in the derived coordinates required for joint physics.
The raw standardized `phi` RMSE was about 130 times the C5P value and derived
`Vi` RMSE about 14.5 times the C5P value.  Raw potential error is gauge
dependent, so it is not a sufficient physics conclusion by itself.  The
gauge-invariant evidence agrees, however:

- E6B lost all 21 matched `Ne` RMSE comparisons and all 21 raw-`phi` and
  derived-`Vi` RMSE comparisons.
- In the physically important stored band `k=4-5`, corresponding to
  `n=20-25` because `zperiod=5`, E6B had roughly 2.18-2.32 times the C5P
  complex cross-spectrum error for all three pressure/density-potential
  pairs.
- Across pairs in that band, median phase error was 13.86 degrees for E6B
  versus 2.63 degrees for C5P, and median coherence change was 0.490 versus
  0.111.
- No predicted E6B density was nonpositive or below the Hermes soft floor, so
  density clipping is not the explanation.

## Persistence remains a serious baseline

Both learned state views beat persistence on shared-field RMSE.  They did not
uniformly beat persistence on turbulence physics.

At horizon 8, the C5P repeated-two-frame rollout improved median separatrix
transport error by about 6.4 percent relative to persistence, while the
repeated-one-frame rollout was approximately tied.  Nevertheless, persistence
remained much better on spectral power and complex cross-spectrum.  At horizon
4, every learned rollout composition was worse than persistence on median
separatrix transport.

Therefore `retain C5P` means retain it as the stronger learned control; it does
not mean that C5P has passed the Paper 0 transport-faithfulness gate.

## Scientific interpretation

An exact evolved state can define a deterministic simulator transition while
still being hard for this neural approximation to learn.  Here the extra E6B
channels are precisely the difficult channels: the validation derivative MSE
for `Vort` and `NVe` was much larger than for `Ne`, `Pe`, and `Pi`.  The
elliptic inverse then exposes coherent low- and mid-mode vorticity/boundary
errors in `phi`; transport depends on spatial derivatives of that potential
and its realization-level relationship with density and pressure.

The current setup also selected checkpoints only by the shared `Ne, Pe, Pi`
score, not by `Vort`, momenta, or `Bphi`, and both models stopped at the
training-budget boundary.  The architecture was a compact local U-Net-like
increment operator with no explicit geometry channels or global operator
path.  Those choices make this a valid matched control, but not a definitive
test of a geometry-aware, state-complete neural operator.

## Frozen next action

Do not run more seeds or a longer unchanged E6B job.  Retain C5P as the current
control and do not proceed to assimilation with either prior.

Any later state-complete experiment must be authorized as a new model
hypothesis.  The narrow repair would be a codec-free, geometry-aware/global
operator trained to convergence, selected using every evolved volume and
boundary component, and required to pass causal `Vort/Bphi -> phi` validation
before transport scoring.  That is materially different from rerunning the
present E6B control.

## Authoritative artifacts

- Frozen physics manifest:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_physics_freeze/job_6948589/matched_state_physics_manifest.json`
  (`1e7199f5cda4855b96bf9c9943ecd00b1fde76b2d94ba4ccb595a112acb428d7`).
- Exact causal-potential result:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_exact_phi/job_6948578/result.json`
  (`047e8da68cab75cd5a0820842df82421fa4ff18e11a9af9cf3246f88bf7a2140`).
- Paired result:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_bounded_physics/job_6948598/scoring/result.json`
  (`a35fa64c27fc67396d802111e8dc13c71823b614ec752321009d317ed6de8ad6`).
- Primary metric table:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_matched_state_bounded_physics/job_6948598/scoring/primary_metrics.csv`
  (`86d2838da2c99293602fb2b96b507f1483235ba80ed08b1eaf4c53b469cba8ed`).
- W&B:
  `https://wandb.ai/sdelaurentiis123-columbia-university/tcv-diagnostics-paper0/runs/p0oldstatephys-j6948598`.
