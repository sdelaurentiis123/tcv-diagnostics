# Phase 2 C5P-only O2 continuation protocol

**Decision status:** frozen after the complete R2 O1 result and before O2
implementation, smoke testing, or training

**Development simulation:** TCV/Hermes `85604`

**Sequestered simulation:** `85606`

**85606 access authorized:** no

**Newly authorized scope:** deterministic teacher-forced one-step O2 for
`C5P-H1` and `C5P-H2` only

The machine-readable authority is
`paper0/manifests/phase2_c5p_o2_continuation_85604.json`.

## 1. Why this continuation exists

The original matched O1/O2 protocol required every C5P and E6B codec at all
three seeds to pass O1 before any arm could enter O2. The complete R2 decision
from Rocky 9 job `6894863` applied that rule correctly:

- `C5P-dcae_l10` passed the complete O1 gate at seeds 1701, 1702, and 1703;
- `E6B-dcae_l10` failed the complete O1 gate at seeds 1701, 1702, and 1703;
- consequently the original matrix recorded `R2_accepted=false` and
  `O2_launch_allowed=false`.

That historical decision is immutable. This document does not relabel the
six-run R2 matrix as a pass and does not relax an O1 metric or threshold.

The failed conjunction mixed two distinct requirements: robustness to random
training seed within one representation, and success of every competing
representation. The first is needed to avoid selecting a lucky codec. The
second is not needed to ask whether an independently passing representation
supports one-step prediction. Paper 0's governing plan permits the simplest
representation that passes the codec oracle to proceed, while retaining
failed representations as negative ablations.

This is therefore an explicitly outcome-informed scope amendment made after
O1 and prospectively frozen before O2. It authorizes the three independently
passing C5P codecs and leaves E6B closed.

## 2. Immutable O1 evidence and selection rule

The authoritative compact R2 result is:

~~~text
paper0/results/phase2_matched_o1_finalize_r2_6894863.json
sha256 ad1cf868d9532869b918ba5892e3926bb06f9961653ac718857b42ac0a182f89
~~~

The continuation selection rule is representation-local:

1. A representation is O2-eligible only if its complete O1 gate passed
   separately at all three frozen seeds `1701`, `1702`, and `1703`.
2. Seed averaging cannot rescue a failed seed.
3. Failure of a competing representation does not veto an independently
   eligible representation.
4. No checkpoint, epoch, metric, threshold, or seed may be changed after this
   selection.

Applying this rule gives:

| Representation | Complete O1 seeds | O2 status |
|---|---:|---|
| `C5P-dcae_l10` | 3/3 pass | eligible |
| `E6B-dcae_l10` | 0/3 pass | closed; negative representation ablation |
| `dcae_l20` candidates | original R1 matrix failed | closed |

## 3. Scientific question

The only new comparison is:

- `C5P-H1`: predict the next five-field state from one current frame;
- `C5P-H2`: predict the same target from the two most recent frames.

Both use `[Ne,Pe,Pi,phi,Vi]`, the same target indices, the same codec topology,
and matched model and optimization settings. Their difference estimates the
incremental value of one additional observed frame when electron momentum,
vorticity, and boundary state are omitted from the model interface.

This comparison cannot prove that C5P is Markov-complete. A passing H2 result
would show only that two saved frames improve teacher-forced prediction at one
saved-step horizon on the later portion of 85604. A passing H1 result would
show only that the current five-field state is sufficient for that same
limited task.

## 4. Inherited data, model, and evaluation rules

Except for removing the ineligible E6B arm, Sections 2, 3, and 7--13 of
`paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md` remain binding.
In particular:

- dataset: verified job `6893525` only;
- training targets: global frames `[2,432)`, all 430 once per epoch;
- guard: `[432,496)`, never loaded;
- validation targets: `[498,624)`, all 126 chronologically every epoch;
- cadence: `3.131905426352636` microseconds per saved frame;
- toroidal mapping: `zperiod=5`, hence `n=5k`;
- codec: frozen `C5P-dcae_l10`, matched by seed, with no fine-tuning;
- arms: separately trained `C5P-H1` and `C5P-H2` at all three seeds;
- model: deterministic masked latent ViT with the already frozen architecture;
- prediction: standardized latent increment followed by frozen decoding;
- loss: equal-channel standardized field MAE after decoding;
- physics-derived losses: forbidden;
- checkpoint selection: earliest epoch attaining the lowest full-validation
  equal-channel data loss after the full 200-epoch budget;
- references: persistence, C5P-H2 linear extrapolation where applicable, and
  training-only toroidal spectral AR(1);
- O2 metrics and per-seed gates: unchanged;
- arm acceptance: all three seeds must independently pass O2.

The H1 and H2 models for one seed begin from the same predictor initialization
seed but train separately. They share the same already-trained codec for that
seed. The codec is always in evaluation mode and every codec parameter must
have gradients disabled.

## 5. W&B and execution provenance

Every full training run must log online to the W&B project
`tcv-diagnostics-paper0`. The run record must include the arm, seed, Slurm job
and array-task IDs, exact Git commit, dirty state, dataset hashes, codec path
and hash, protocol and manifest hashes, optimizer schedule, epoch metrics,
selected epoch, runtime, accelerator identity, and peak memory.

A missing W&B credential or failed online initialization is a pre-training
failure for a full run. A non-scientific smoke may disable external logging,
but its artifact must say so and it cannot be used as evidence.

Cluster execution remains Rocky 9 only, from a clean checkout at the exact
committed implementation. Existing job directories and checkpoints are never
overwritten.

## 6. Implementation and launch gates

Before full O2 training:

1. implement the frozen uncompressed references and deterministic transition;
2. add the CPU tests required by Section 12 of the original protocol,
   including a synthetic overfit test;
3. verify the full CPU suite on Rocky 9;
4. run a one-GPU Rocky 9 smoke with at most 16 frames and at most two epochs;
5. verify finite gradients, frozen codec parameters, history order, target
   masking, target-only loss, checkpoint reload identity, W&B policy, and the
   output schema;
6. commit the implementation before submission.

The full array contains six independent runs in lexicographic order over
`arm=[C5P-H1,C5P-H2]` and `seed=[1701,1702,1703]`. These are six runs using
three seeds, not six distinct seeds.

## 7. Stop/go interpretation

Each arm is evaluated independently under the unchanged O2 gate:

- if neither arm passes 3/3 seeds, stop and report deterministic one-step
  failure;
- if exactly one arm passes 3/3 seeds, it is the sole candidate for a newly
  frozen O3 protocol;
- if both pass 3/3 seeds, retain both through the first newly frozen short O3
  comparison rather than selecting on O2 RMSE alone.

No O3 rollout, stochastic model, diffusion, FGN, PDE-Refiner, assimilation,
diagnostic ranking, or 85606 access is authorized here.

## 8. Claims boundary

The strongest possible result from this continuation is that a deterministic
C5P model predicts one saved step on later 85604 data better than the frozen
uncompressed references while passing the frozen field, spectral,
cross-field, and transport criteria at all three seeds.

It would not establish autonomous rollout skill, probabilistic calibration,
cross-shot generalization, diagnostic value, or causal steering.
