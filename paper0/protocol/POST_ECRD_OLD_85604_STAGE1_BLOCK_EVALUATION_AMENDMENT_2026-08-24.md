# Old-85604 Stage-1 chronological-block evaluation amendment

**Frozen:** 2026-08-24 while seed 1701 training was in progress, before any
full Stage-1 checkpoint had been evaluated by chronological block

**Training protocol:**
`POST_ECRD_OLD_85604_STAGE1_FULL_PROTOCOL_2026-08-24.md`

**Development run:** 85604 only

**Held-out 85606:** unopened and prohibited

## Motivation

The full Stage-1 trainer selects checkpoints by aggregate Ne/Pe/Pi validation
derivative MSE over eligible targets `[497,624)`. Existing Paper 0 protocols
also define three primary 42-target chronological validation blocks. The
aggregate alone could conceal a failure concentrated in the latest part of
the trajectory. This amendment freezes the companion block evaluation before
those block results are inspected. It does not change training, checkpoints,
the split, or the guard.

## Frozen target blocks

- `V00 = [498,540)`;
- `V01 = [540,582)`;
- `V02 = [582,624)`.

Target 497 remains part of the already frozen aggregate checkpoint metric but
is excluded from the matched block comparison so every block has 42 targets
and the definitions remain identical to the Phase 3.5/ECRD protocols.

## Metrics

For every selected seed/state-view checkpoint and each block, report:

- raw standardized derivative MSE and zero-derivative persistence MSE by
  predicted field;
- persistence-relative skill by field;
- the matched mean Ne/Pe/Pi MSE and skill;
- the E6B inner/outer Bphi metrics;
- seed median, minimum, and maximum for every quantity.

The evaluation uses inference only. No metric is added to the training loss.
Adjacent targets are not described as independent simulations.

## Decision use

The aggregate 10% state-view rule in the parent protocol remains primary. In
addition, an E6B advance requires its median matched Ne/Pe/Pi MSE to be no more
than 10% above C5P in at least two of the three blocks and no more than 25%
above C5P in the remaining block. Every E6B evolved volume must have positive
median persistence-relative skill overall and in at least two blocks.

If this consistency rule fails, C5P remains the short-horizon performance
control and E6B remains the physically preferred but unresolved exact-state
ablation. The result does not establish that exact state is harmful or that
partial state is Markovian.

No rollout, transport, stochastic, assimilation, diagnostic-ranking, or 85606
claim is authorized by this amendment.
