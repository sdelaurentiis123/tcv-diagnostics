# Repository instructions

This repository implements the Paper 0 program in `paper0/PAPER0_SPEC.txt`.

## Governing rules

- Work phase by phase. Do not begin architecture redesign until Phase 0 is complete and committed.
- Treat `/Users/stanislavdelaurentiis/tcv-gaot-3d` and its Rusty counterpart as read-only evidence sources for this project. Port only the minimum audited code needed and record its origin commit in `paper0/PROVENANCE.md`.
- Do not use shot 85606 for training, validation, checkpoint selection, metric development, filter tuning, plotting decisions, or debugging.
- Access to 85606 in this repository requires a committed frozen protocol and an explicit release record under `paper0/protocol/`.
- Record the historical fact that 85606 was inspected in the predecessor project. Never describe the new evaluation as historically blind.
- Physics-derived quantities are evaluation metrics only, never training losses.
- Never treat temporal windows as independent physical shots.
- Never claim experimental diagnostic realism, broad cross-shot generalization, or steering from the current data.
- Keep large datasets, forecasts, checkpoints, and run directories out of Git. Track immutable paths, hashes, configurations, and compact metrics instead.
- Every reported result must identify data split, checkpoint, seed, horizon, ensemble size, code commit, and exact reproduction command.
- Preserve unrelated user changes. Use scoped commits at phase boundaries.

## Canonical forecast interface

Forecast implementations must expose semantics equivalent to:

```python
forecast = model.predict(context, horizon, ensemble_size)
```

with output axes:

```text
[batch, ensemble_member, future_time, channel, spatial_axes...]
```

Adapters may wrap legacy implementations, but axis meaning must be explicit and tested.

## Development expectations

- Prefer small, testable modules under `src/tcv_diagnostics/`.
- Put automated checks under `tests/`.
- Put immutable research decisions and audit artifacts under `paper0/`.
- Put cluster launch templates under `cluster/`; scripts must print the Git commit and dirty state.
- Do not silently repair discrepant historical results. Document the discrepancy, cause, and reproduction outcome in `paper0/AUDIT.md`.
