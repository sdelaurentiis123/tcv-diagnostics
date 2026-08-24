# Old-85604 exact-state plus current-phi repair screen

**Frozen:** 2026-08-24 while Rusty was unavailable, after the three-seed
Stage-1 aggregate metrics were inspected but before any chronological-block
scores or axial-operator smoke result were inspected

**Development run:** 85604 only

**Held-out 85606:** unopened and prohibited

## Motivation

The normalized Stage-1 comparison established two facts on three seeds. The
local E6B model learned every retained evolved field with positive
persistence-relative skill, so its earlier numerical collapse was primarily
a loss-scaling error. It nevertheless had approximately 45--46% higher
matched Ne/Pe/Pi one-step error than the C5P model, which receives derived
potential and velocity directly.

This screen tests one narrow repair hypothesis: the E6B transition is harder
for a local architecture because potential is obtained from a nonlocal
elliptic relation to vorticity. It is not a stochastic-model or transport
experiment.

## State and causality

Both new arms predict standardized one-step derivatives of
`Ne, Pe, Pi, NVe, NVi, Vort` and the retained `Bphi` boundary state. They also
receive **current/history-only** `phi` as an auxiliary input. No target or
future `phi` may be read.

Current truth-derived `phi` is causal in teacher-forced evaluation but is not
by itself a deployable free-rollout interface. A rollout would have to derive
the next `phi` from the predicted evolved state with the authoritative Hermes
elliptic transformation, or introduce and validate a separate closure. This
screen therefore cannot establish rollout or transport fidelity.

## Matched arms

1. **Local E6B + current phi:** the existing mixed-boundary codec-free U-Net,
   with x/y downsampling only and no toroidal downsampling.
2. **Nonlocal E6B + current phi:** the mixed-boundary full-resolution axial
   operator, with axial attention, circular toroidal operations, no toroidal
   downsampling, and no absolute toroidal coordinate.

The local E6B-without-phi and C5P results from the completed Stage-1 matrix are
locked external controls; they are not rerun. The two new arms use matched
parameter scale, data, optimization, loss, seed, and checkpoint rule.

## Data and optimization

- training frames `[0,432)`;
- guard frames `[432,496)`, never read;
- validation frames `[496,624)`;
- lead time: one frame;
- history: one frame;
- random circular toroidal shifts in training only;
- seed 1701 for the screen;
- 12 epochs and the same persistence-normalized component-balanced derivative
  loss, optimizer schedule, accumulation, numerical precision, and validation
  checkpoint rule as full Stage 1;
- no flux, spectrum, phase, coherence, PDE, or conservation quantity in the
  loss;
- W&B online tracking required.

The axial engineering smoke must pass before either scientific screen arm is
launched. The completed Stage-1 chronological reduction must also retain E6B
as unresolved rather than advancing it unchanged. Both prerequisite artifacts
must be locked by path and SHA-256 in the launch manifest.

## Screen gates

For each new arm:

- all mechanical, finite-gradient, exact-reload, and toroidal-equivariance
  gates pass;
- every E6B predicted field has positive aggregate persistence-relative skill;
- matched Ne/Pe/Pi validation MSE improves by at least 15% relative to the
  seed-1701 local E6B-without-phi checkpoint.

Any arm passing all gates advances unchanged to seeds 1702 and 1703 and the
three frozen chronological blocks. An arm that does not pass is not scaled.
The final exact-state competitiveness threshold remains the preregistered
median E6B/C5P matched-MSE ratio of at most 1.10, including the existing block
consistency rules.

## Authorized conclusion

This screen may show whether current potential and/or nonlocal model capacity
repairs exact-state one-step prediction on old 85604. It cannot authorize a
free-rollout, transport, stochastic calibration, assimilation, diagnostic
ranking, steering, or held-out-run claim.
