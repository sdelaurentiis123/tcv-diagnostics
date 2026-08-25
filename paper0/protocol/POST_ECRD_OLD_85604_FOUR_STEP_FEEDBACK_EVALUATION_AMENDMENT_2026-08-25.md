# Old-85604 four-step feedback physics-evaluation amendment

**Frozen:** 2026-08-25 after state-only training job `6937357` completed and
before any spectrum, cross-field, or transport metric was evaluated for its
selected checkpoint

**Development simulation:** 85604 only

**Held-out 85606, guard interval, and newer NERSC data:** unopened and
prohibited

## Authorized evidence

Job `6937357` completed the prospectively frozen four-step detached-feedback
pilot with no physics-derived loss or checkpoint-selection metric. Its compact
result is tracked as:

```text
paper0/results/post_ecrd_old_85604_four_step_feedback_pilot_6937357.json
SHA-256 ffcc3d4b5bdbada7c83dacc4eb85fed318a0f971ab28b329ea5ed7c15cd7938f
```

The scientific authority remains the immutable Rusty artifact:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/post_ecrd_old_85604_four_step_feedback_pilot/job_6937357_seed1702/run/result.json
```

The selected epoch-6 checkpoint has SHA-256
`affe2589f4ce6639879ca1ed4a100af764aa48a475a653987faa18d4ce844117`.
Online W&B run `p0oldpush4-j6937357-s1702` finished successfully. All 642
optimizer updates, finite-metric checks, checkpoint reload, and integer
toroidal-equivariance checks passed.

## State result and authorization

Relative to the immutable Stage-2 seed-1702 parent, full-validation mean-field
state MSE ratios are:

| Horizon | Pilot / parent MSE | Reduction |
|---:|---:|---:|
| 1 frame | 0.967870 | 3.21% |
| 4 frames | 0.896629 | 10.34% |
| 8 frames | 0.859817 | 14.02% |

The mean four/eight-frame improvement is 12.80%, above the prospectively
declared 5% gate. Every predicted field retains positive persistence-relative
skill at all three horizons. The state pilot therefore passes and authorizes
one inference-only physics-preservation comparison. It does not yet authorize
training seeds 1701 or 1703.

## Frozen comparison

Compare exactly two deterministic models:

1. `pre_feedback_parent`: the bitwise Stage-2 seed-1702 epoch-4 parent;
2. `four_step_feedback_finetuned`: the selected job-6937357 epoch-6 checkpoint.

For both models, use repeated autonomous lead-one prediction only. Evaluate all
eligible starts in the unchanged validation interval `[496,624)` at terminal
horizons four and eight. The complete predicted five-field state is fed back;
no intervening or future truth becomes model input. Use inference batch size
four.

The evaluation must reproduce the already validated old-85604 implementations
for:

- standardized field RMSE and persistence-relative skill;
- directional toroidal power in stored bands `k=1–3`, `k=4–5`, and `k=6–7`,
  reported physically as `n=5k`;
- density/pressure–potential cross-spectrum and coherence;
- strict local radial-face transport;
- integrated confined-separatrix particle and internal-energy transport.

Nonlinear transport is calculated from each model realization before any
summary. No physics quantity enters training or checkpoint selection.

## Frozen decision rule

The pilot advances to matched confirmation seeds 1701 and 1703 only if all of
the following hold:

1. the already passed full-validation state gate remains authoritative;
2. at both horizons, median absolute log power-ratio error across five fields
   and three frozen toroidal bands is no more than 10% above the parent error;
3. at both horizons, separatrix relative-L2 error averaged over particle,
   electron internal-energy, ion internal-energy, and total internal-energy
   transport is no more than 5% above the parent error.

Strict-face transport and cross-field coherence are mandatory report-only
metrics. They cannot rescue a failed gate. No favorable individual metric can
rescue a failed horizon.

## Scope

This amendment authorizes evaluation only. It does not authorize more training,
checkpoint reselection, assimilation, diagnostic ranking, steering, stochastic
calibration claims, access to 85606, or access to the newer NERSC data.

