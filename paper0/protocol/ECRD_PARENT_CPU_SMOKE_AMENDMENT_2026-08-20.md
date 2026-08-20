# ECRD CPU parent amendment for bounded engineering smoke — 2026-08-20

## Status and scope

Prospectively frozen after the CPU timing probe and before full CPU parent
generation or any ECRD engineering-smoke result.

This is a narrow execution amendment to the frozen ECRD model-development
protocol.  It authorizes a CPU-generated four-phase H1 parent only as an input
to the bounded, non-scientific four-arm engineering smoke.  It does not
authorize that CPU artifact for full training, checkpoint selection,
scientific forecasting, or scientific comparison.

Simulation 85606 remains unopened and unauthorized.

## Evidence motivating the amendment

The data-free Blackwell compatibility probe, Slurm job `6912397`, failed before
simulation access because the frozen PyTorch build lacks kernels for CUDA
capability `sm_120`.  The original Rocky Linux 9 H100 parent job `6912245`
remains pending for resources.

The prospectively registered one-context CPU timing probe completed as Slurm
job `6912444` on Rocky Linux 9.8.  It used 32 PyTorch threads, loaded only
85604 context frame 1, did not read target truth, and timed one exact
four-phase float32 parent evaluation at `2.898978430996067` seconds.  The
linear inference-only extrapolation for 556 parent frames is
`1611.8320076338132` seconds, or `0.4477311132316148` hours.

Immutable timing evidence:

- result:
  `/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/ecrd_parent_cpu_probe/job_6912444/result.json`
- result SHA-256:
  `fd57ec47ae9230ae98135a8cac17d90f31664299a9f5180f9a1b35bffdd3b39d`
- artifact index SHA-256:
  `c72d55831b02675244c940e7d053ccfa5fc0f80dbcc20cb7e5e16bccdf3cdae2`

## Authorized CPU parent generation

One clean, commit-locked Rusty Rocky Linux 9 CPU job may:

1. verify the original ECRD parent manifest locks, H1 checkpoint, codec,
   model-data catalog, normalization, and shards;
2. stage only the existing 85604 model dataset to node-local storage;
3. read the frozen training contexts for targets `[2,432)` and frozen
   validation contexts for targets `[498,624)` without reading target truth;
4. evaluate the unchanged expression
   `mean_q=0..3 T_-q H1(T_q x)` in float32 on CPU;
5. write immutable training and validation parent HDF5 artifacts with their
   hashes and complete provenance;
6. report execution health to W&B online, while retaining Ceph as artifact
   authority.

The job may not read guard frames or 85606, calculate physics or forecast
metrics, update weights, select a checkpoint, or perform assimilation.

## Use restriction and comparison rule

The resulting CPU parent is provisional.  It may be used only by smoke-mode
ECRD jobs with four training targets `[2,6)`, four validation targets
`[498,502)`, one epoch, and two optimizer steps.  Those smoke outputs are
mechanical checks, not scientific results.

The original H100 job `6912245` remains queued.  Before full ECRD training,
the H100 and CPU parents must be compared frame by frame.  The full-training
manifest must either:

- use the H100 parent as originally frozen; or
- prospectively define and pass an explicit CPU/H100 numerical-equivalence
  tolerance before granting the CPU parent scientific authority.

No result from the bounded smoke may choose a scientific threshold,
architecture width, validation rule, or physics metric.  The H100-only rule
for all model training, including the bounded smoke, remains unchanged.
