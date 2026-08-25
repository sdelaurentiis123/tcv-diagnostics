# Old-85604 exact-state derived-coordinate screen

**Frozen:** 2026-08-25, after the exact-state current-phi repair screen was
complete and before this screen was implemented or executed

**Development simulation:** 85604 only

**Held-out 85606:** unopened and prohibited

## Motivation

The matched local E6B Stage-1 model learned every saved evolved field better
than persistence but had `0.007772147896373167` shared Ne/Pe/Pi one-step MSE.
The matched C5P control reached `0.005322341561633884`. Supplying current
`phi` alone to local E6B changed the E6B metric by only `-0.128%`; replacing
the local processor with the screened axial operator made it `17.912%` worse.

C5P presents two derived dynamical coordinates directly: `phi` and `Vi`.
E6B instead presents their saved evolved sources, `Vort` plus retained `Bphi`
and `NVi` plus `Ne`. The exact transformations are already verified:

```text
phi = HermesEllipticSolve(Ne, Pe, Pi, Vort, Bphi, fixed geometry)
Vi  = NVi / (2 * softFloor(Ne, 1e-7))
```

This screen asks whether the E6B gap is a coordinate-learning burden rather
than missing information. It adds both derived current coordinates as model
inputs while retaining the full exact-state prediction target.

## Model and causal interface

One arm is authorized:

**Local E6B + current phi + current Vi.** Use the same mixed-boundary,
codec-free local U-Net as the completed local E6B and local-plus-phi arms. It
jointly predicts standardized one-step derivatives of
`Ne, Pe, Pi, NVe, NVi, Vort` plus the retained `Bphi` boundary state. Its two
auxiliary volume channels are current/history-only `phi` and `Vi`.

No target or future auxiliary value may be loaded. A test must poison target
`phi` and target `Vi` independently and show that the returned training pair
is unchanged and finite.

The auxiliary channels add no physical state beyond E6B. In a free rollout,
`phi` must be reconstructed with the already hash-validated compiled
Hermes/BOUT++ elliptic operator and `Vi` must be reconstructed with the exact
algebraic source transformation after each predicted E6B step. The screen is
teacher-forced and does not invoke either rollout reconstruction.

## Frozen comparison

- training frames: `[0,432)`;
- guard frames: `[432,496)`, never read;
- validation frames: `[496,624)`;
- history: one frame;
- lead: one frame;
- random shared circular toroidal roll in training only;
- seed: 1701;
- epochs: 12;
- sample batch size: 1 with four-sample gradient accumulation;
- AdamW, identical learning-rate schedule, clipping, precision, and
  persistence-normalized component-balanced state loss as the completed
  local repair arm;
- checkpoint selection: shared Ne/Pe/Pi derivative MSE on the entire frozen
  chronological validation interval;
- online W&B tracking required;
- no flux, spectrum, cross-phase, coherence, PDE residual, conservation, or
  other physics-derived training term.

The model must stay within 3% of the completed local-plus-phi parameter count.
The completed local E6B-without-phi, local E6B-plus-phi, and C5P artifacts are
locked controls and are not rerun.

## Prerequisites and launch lock

Before Slurm submission, a machine-readable manifest must lock by path and
SHA-256:

1. the completed Stage-1 reduction;
2. the completed local-plus-phi result;
3. the completed axial-plus-phi result;
4. the seed-1701 local E6B baseline result;
5. this repository commit.

The launcher must refuse a dirty or mismatched checkout, an overwritten
output path, an offline W&B run, any access permission for 85606, or any
manifest that authorizes more than this single seed-1701 arm.

## Gates

The screen passes only if:

1. the exact optimizer-update count is reached and epoch-mean training loss
   decreases;
2. all validation values and gradients remain finite;
3. exact checkpoint reload, no toroidal stride, integer toroidal-shift
   equivariance, and boundary invariance pass the existing numerical gates;
4. every predicted E6B field retains positive persistence-relative skill;
5. shared Ne/Pe/Pi MSE improves by at least 15% relative to the seed-1701
   local E6B baseline, hence is at most `0.006606325711917192`.

Only a passing arm may be proposed unchanged for seeds 1702/1703 and frozen
chronological-block scoring. Failure prohibits more derived-coordinate
variants under this screen. Either result remains one-step development
evidence and does not authorize rollout, transport, stochastic calibration,
assimilation, diagnostic ranking, steering, or access to 85606.
