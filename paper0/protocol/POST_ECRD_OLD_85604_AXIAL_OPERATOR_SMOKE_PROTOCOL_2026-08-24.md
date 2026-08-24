# Old-85604 nonlocal axial-operator engineering smoke

**Frozen:** 2026-08-24, before any axial-operator GPU execution

**Scientific authority:** none; bounded engineering smoke only

## Motivation

The matched Stage-1 comparison tests a local, codec-free U-Net on C5P and on
the saved Hermes evolved state. The next controlled architecture needs a
full-domain receptive field without removing toroidal cells. This smoke asks
only whether the proposed implementation trains, reloads, and preserves the
toroidal symmetry at production width.

The implementation uses standard axial self-attention and is inspired by the
global processor used in operator-transformer literature. It is not the
authors' public GAOT implementation and must not be reported as a GAOT result.

## Immutable scope

- simulation: old processed 85604 only;
- held-out 85606: unopened and prohibited;
- training frames: `[0,432)`;
- guard frames: `[432,496)`, unread;
- validation frames: `[496,624)`;
- smoke training currents: `[2,4)`, two one-step transitions;
- smoke validation current: `496`, target `497`;
- state target: `E6B = [Ne,Pe,Pi,NVe,NVi,Vort] + Bphi`;
- auxiliary input: current/history `phi` only, never target `phi`;
- one context frame and one saved-frame lead;
- two optimizer steps total;
- no rollout, checkpoint selection, assimilation, or diagnostic ranking.

## Architecture

`AxialIncrementOperator3D` uses:

- width 104 and four blocks;
- four attention heads;
- full-domain axial attention along x, y, and z;
- a mixed-boundary local convolution in every block;
- circular padding only along z;
- no toroidal stride, pooling, or absolute z coordinate;
- a lead-time embedding at every block;
- joint E6B derivative output and a retained Bphi boundary head;
- zero-initialized derivative projections so the initial forecast is
  persistence;
- approximately 2.13 million parameters, matched to the local Stage-1
  processor rather than enlarged for this comparison.

The loss is an equal-component standardized state-derivative MSE. It contains
no flux, spectrum, cross-phase, coherence, PDE, conservation, or other
physics-derived quantity.

## Mechanical gates

The smoke passes only if:

1. the focused test set passes on the allocated Rusty GPU node;
2. exactly two optimizer steps complete with finite loss and gradients;
3. all predicted tensors are finite;
4. the checkpoint reloads exactly;
5. every 3D convolution has toroidal stride one;
6. integer toroidal-roll equivariance remains within the frozen numerical
   tolerance;
7. W&B initializes online and reaches the finished state;
8. peak CUDA memory and wall time are recorded;
9. all artifacts receive SHA-256 records.

Passing authorizes writing a separate scientific-training protocol only. It
does not authorize training, model selection, transport claims, 85606 access,
or replacing the frozen Stage-1 comparison.
