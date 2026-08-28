# Physics-first forecast readout and S0 execution amendment

**Frozen:** 2026-08-28, before extracting the representative forecast target or
inspecting S0 validation results.

This amendment resolves execution details that were not explicit in
`PHYSICS_FIRST_SPATIAL_S0_PROTOCOL_2026-08-28.md`. It does not change the data
split, diagnostics, masks, model family, or acceptance language.

## Matched model realization

All learned-model visual comparisons use **model-training seed 1702**:

- codec-free multi-lead operator, seed 1702;
- conditioned field-residual diffusion, seed 1702;
- equivariant conditioned diffusion, seed 1702;
- persistent global--local generator, seed 1702.

Seed 1702 is fixed across architectures and is not selected by a plotted
physics metric. Aggregate claims that were originally evaluated across three
training seeds remain based on their frozen three-seed result, not on this
single visual realization.

## Shared forecast population

The shared one-frame comparison uses the 36 preregistered persistent-model
starts, whose one-frame targets are the intersection of all four forecast
artifacts. The representative target is chosen from these 36 targets using the
already frozen median conditioned-diffusion transport-error rule.

The conditioned-diffusion selection error is the absolute error of the
**ensemble expectation of member-wise confined-separatrix particle transport**.
Transport is evaluated for every member before the ensemble average. It is not
the transport of the ensemble-mean fields.

## Static field plane

The cross-model static field comparison uses stored toroidal plane `z=44`, the
midpoint of the 88-cell periodic wedge. The plane is fixed independently of
field values. The poloidal panel uses the authoritative `Rxy,Zxy` geometry,
with the wall categories, separatrix, X-point stencil, and outboard-midplane
landmark overlaid.

Field panels show standardized train-normalized fluctuations after removing
the toroidal mean independently at each `(x,y)` location. Truth-derived color
limits are reused for every model. The potential gauge is fixed by subtracting
the strict-interior spatial mean independently for truth and every member.

## Ensemble curves

All displayed probabilistic curves use the first 16 members of the immutable
32-member seed bank. Scores and selection continue to use all 32 members.

The transport-variogram uncertainty band uses a chronological moving-block
bootstrap over the ordered 36-target population with block length three
selected starts and 2,000 deterministic bootstrap replicates (seed 856040828).
This interval is conditional on one 85604 realization and is descriptive, not
a cross-shot confidence interval.

## Scope locks

- Development simulation 85604 only.
- Guard frames remain unread.
- Simulation 85606 remains unopened and must not be enumerated.
- The newer NERSC 85604 segment remains inventory-only.
- No training, assimilation, diagnostic ranking, or steering is performed by
  the forecast-figure extraction.
