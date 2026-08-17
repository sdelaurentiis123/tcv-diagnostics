# Phase 2 geometry, units, and ensemble-transport readout

## Outcome

The geometry-aware transport evaluator passed every prospectively frozen
geometry, sign, surface-integration, SI-unit, and ensemble-ordering gate on
development run 85604.

This releases the evaluator needed for the next codec and forecast tests. It
does **not** show that f8, z44, LOLA, or any learned rollout preserves
transport. It also does not open held-out run 85606.

## Execution record

- Rocky 9 SLURM job: `6891709`
- Paper 0 commit: `9dd8780ca2b68b76624aaefa1d8b3638c5c6377c`
- State: `COMPLETED`, exit `0:0`
- Runtime: 16 seconds
- Peak batch-step memory: 76,448 KB
- Full immutable directory:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_85604_geometry_units/job_6891709`
- Full result SHA-256:
  `9a62f47aaa15edba3ca6b17159862b026dbcf03977eb535306a4ed8702dde1cc`
- Tracked result:
  `paper0/results/phase2_geometry_units_6891709.json`

The immutable command, environment, result, and artifact-manifest hashes are:

| Artifact | SHA-256 |
|---|---|
| `commands.sh` | `f78efd72fd3c056e4f8b93986a44de6b8737230a8944d8f27fb3ffa3ed56eb42` |
| `environment.txt` | `bd633ea574134e2763e65e071410459f900d1725bbf1e8d0ff74e6a80a9231da` |
| `geometry_units.json` | `9a62f47aaa15edba3ca6b17159862b026dbcf03977eb535306a4ed8702dde1cc` |
| `artifact_sha256.txt` | `51ef4bc0a9986dd208d7db5ae5b976e1032dfef5453170e1e24294b74eaac063` |
| completed `job.log` | `253756cdda077dd7d147c954a59149dabc6a3356097834c3bb51c009859b7c2f` |

## What passed

All 19 recorded gates are `true`.

### Geometry and topology

- The 68-by-32 source grid maps to the 64-by-32 model crop by removing two
  radial guards at each end.
- The exact separatrix is grid x-low face 18, between local model cells 15 and
  16. The closed-field surface contains exactly `y=8..23`, or 16 poloidal
  rows.
- All 2,016 radial `psi` differences are positive; the minimum is
  `4.1698824270566684e-05`. Therefore positive `+x` crosses the confined
  separatrix outward.
- Separatrix-face `psi` agrees with `psi_bdry` to
  `1.6318089728854096e-16` absolute.
- The two branch representations of the X point coincide exactly at
  `(R,Z)=(0.7942120340630372,-0.4420630317234957) m`.
- The discrete outboard-midplane row is uniquely `y=18`, at separatrix
  `(R,Z)=(1.087493128717754,-0.017535153092040314) m`.

The strict wall-interior operator region has 1,869 cells and is partitioned
exactly once into:

| Region | Strict cells |
|---|---:|
| Confined edge | 256 |
| Private-flux region | 219 |
| Scrape-off layer | 1,394 |

The remaining operator cells include 35 wall-crossing and 16 exterior cells.
They are excluded from primary regional metrics. Hypnotoad source confirms
that a fractional `penalty_mask` is a poloidal-edge fraction, not a physical
cell-area weight.

### Units

Independent reconstruction from the exact BOUT++ constants reproduces the
stored run normalization to relative tolerance `1e-14`:

| Quantity | Released value |
|---|---:|
| `Cs0` | `69205.61141651045 m/s` |
| `Omega_ci` | `95788333.03066081 s^-1` |
| `rho_s0` | `0.0007224847664314034 m` |
| normalized particle-flow multiplier | `3.612423832157018e17 s^-1` |
| normalized pressure-flow multiplier | `2.893870527993356 W` |

The internal-energy metric forms `1.5 * Q(P,phi)` before applying the pressure
multiplier. The converter does not apply `1.5` again.

### Surface convention

The primary integral is over the simulated one-fifth wedge. The binary-exact
known answer returned `[12.5,-18.75]`, including the correct signs. Explicitly
requesting a five-wedge full-torus-equivalent sensitivity returned
`[62.5,-93.75]`, exactly five times the primary value.

### Ensemble ordering

The actual nonlinear face operator gives a decisive two-member counterexample:

| Evaluation | Normalized transport |
|---|---:|
| Member 1 | `43.3881636538414` |
| Member 2 | `-21.694081826920712` |
| Mean of member-wise transport | `10.847040913460345` |
| Transport of ensemble-mean fields | `0.0` |

Therefore Paper 0 must downsample, invert state transforms, reconstruct face
flow, integrate, and convert units separately for each member. A transport
calculation on ensemble-mean fields is not an acceptable substitute.

## Boundaries of the result

- Target rows `y=0,31` remain outside the model-state operator because their
  physical guard values are absent.
- “Outward” is released only for the closed-field separatrix. Other surfaces
  retain the unambiguous coordinate label `+x`.
- Full-torus equivalent means five identical periodic copies of the simulated
  wedge; it is a sensitivity, not five independent observations.
- No learned field, codec reconstruction, forecast, or assimilation result was
  scored in this job.
- Run 85606 was not read.

## Next decision

The transport validation ladder is complete. The next rung is O1: apply this
released evaluator to codec reconstructions on 85604, separately identifying
particle flow, electron internal-energy flow, and the legacy ion-pressure
representation limitation. Architecture work remains premature until that
representation test and the already-open data-stationarity decision are
resolved.
