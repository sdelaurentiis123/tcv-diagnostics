# ECRD parent CPU timing probe — 2026-08-20

## Status

Prospectively frozen before executing the probe.

This note does not amend the ECRD scientific model-development protocol or
authorize CPU generation of a scientific parent artifact.  It authorizes one
non-scientific execution-feasibility measurement on Rusty Rocky Linux 9.

## Motivation

The immutable four-phase symmetrized H1 parent job was submitted to one Rocky
Linux 9 H100 as job `6912245`.  Slurm reports that it is pending for resources
with an estimated start on 2026-08-24.  A data-free compatibility probe on an
RTX Pro 6000 Blackwell node, job `6912397`, failed before reading simulation
data because the frozen PyTorch build does not contain `sm_120` kernels.

Before changing the H100-only parent execution rule, measure whether exact
float32 CPU inference is a practical fallback.  This probe is not evidence
about model quality and cannot be used for model selection.

## Authorized operation

The probe may:

- run only on Rusty under Rocky Linux 9;
- load the immutable C5P H1 seed-1701 checkpoint and its immutable codec;
- load only standardized context frame 1 from simulation 85604, corresponding
  to training target frame 2;
- evaluate the exact four-phase expression
  `mean_q=0..3 T_-q H1(T_q x)` once on CPU in float32;
- record wall-clock time, thread count, resident memory, tensor shape, finite
  status, and a SHA-256 digest of the transient output bytes.

The probe must not:

- read target frame 2 or any other future truth;
- read the guard interval, validation truth, or any path mentioning 85606;
- save the predicted field tensor;
- train or update parameters;
- calculate field, spectral, cross-field, transport, calibration, or
  assimilation metrics;
- alter or cancel H100 job `6912245`;
- authorize a scientific CPU parent artifact.

## Decision rule

Extrapolate the measured one-frame wall time to the frozen 556-parent-frame
workload (430 train plus 126 validation frames).  CPU parent generation may be
considered only if the conservative extrapolation is operationally shorter
than waiting for the queued H100 job.  Any scientific CPU generation requires
a separate, dated prospective execution amendment and an explicit numerical
equivalence check against H100 output before the CPU artifact can become an
accepted parent.

The H100 requirement for ECRD training is unchanged.
