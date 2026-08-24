# Post-ECRD old-85604 Stage-1 codec-free pilot protocol

**Frozen:** 2026-08-24

**Development source:** existing 624-frame TCV/Hermes 85604 archive only

**Held-out 85606:** unopened and prohibited

## Purpose

This pilot answers whether the new codec-free operator can learn a complete
one-step transition on the already verified old dataset before the new NERSC
archive is used. It is both an end-to-end machinery proof and a preliminary
state-completeness comparison.

It is not a final model-selection result. It does not authorize multi-lead
training, autoregressive rollout, a GAOT port, stochastic generation,
assimilation, diagnostic ranking, or steering.

## Immutable data contract

- training frames: `[0,432)`;
- guard frames: `[432,496)`, never read;
- chronological validation frames: `[496,624)`;
- training-only normalization: the existing immutable normalization fitted on
  `[0,432)`;
- one-frame context and one-frame lead only;
- all 431 eligible training pairs and all 127 eligible validation pairs;
- no shuffled split, no duplicated frame, and no pair crossing a split.

The two matched state views are:

- `C5P = [Ne,Pe,Pi,phi,Vi]`;
- `E6B = [Ne,Pe,Pi,NVe,NVi,Vort] + Bphi`.

## Architecture

Both arms use `CodecFreeIncrementOperator3D` with:

- one history frame;
- base width 24;
- channel multipliers `[1,2,4]`;
- two residual blocks per level;
- 128-channel lead embedding;
- circular toroidal padding;
- zero/wall padding on x and y;
- x/y-only stride `[2,2,1]`;
- no toroidal downsampling;
- no autoencoder or latent codec;
- one shared random toroidal roll per training pair;
- direct prediction of the normalized one-frame state derivative.

The processors are identical. Only the state-dimension-dependent stems and
the E6B boundary head differ.

## Loss correction

The previous engineering smoke added the mean volume loss and mean boundary
loss, which implicitly assigned one half of the objective to only two boundary
profiles. That smoke remains immutable at its original commit.

This pilot uses a component-balanced direct-state loss. Standardized MSE is
computed separately for every volume field and each of the two `Bphi` sides,
then averaged equally over those components. Therefore C5P averages five
field losses; E6B averages six field losses and two boundary-side losses.

This contains no flux, spectrum, cross-phase, coherence, PDE, conservation,
or other physics-derived term.

## Frozen optimization

- seed: `1701`;
- epochs: `2`;
- sample batch size: `1`;
- gradient accumulation: `4` samples;
- optimizer: AdamW;
- initial/peak learning rate: `2e-4`;
- weight decay: `1e-4`;
- linear warmup over the first 5% of optimizer updates, with a minimum of one
  update;
- cosine decay to `2e-5`;
- gradient-norm clipping at `1.0`;
- bfloat16 autocast;
- cuDNN and matmul TF32 disabled;
- full-float32 equivariance/checkpoint computations;
- seed-fixed deterministic epoch permutations;
- online W&B plus authoritative local artifacts.

## Checkpoint and reporting rule

The best checkpoint for each arm minimizes the same validation quantity:
mean standardized derivative MSE over the shared evolved fields
`[Ne,Pe,Pi]`. This avoids selecting E6B on privileged-only targets while
selecting C5P on a different observable.

Every epoch reports:

- total component-balanced training loss;
- validation derivative MSE per field;
- E6B validation boundary MSE by side;
- zero-derivative persistence MSE per field;
- shared-field mean validation MSE and persistence-relative skill;
- optimizer updates, learning rate, wall time, and peak CUDA memory.

The two numeric training losses are not compared directly because the target
component sets differ. The apples-to-apples preliminary comparison is only on
the three shared fields.

## Pilot gate and next action

The pilot passes mechanically only if both arms:

1. complete the exact frozen pair and optimizer-update counts;
2. have finite loss, gradients, validation metrics, and checkpoints;
3. reduce epoch-mean training loss from epoch 1 to epoch 2;
4. reload the selected checkpoint exactly;
5. retain the frozen toroidal numerical-equivariance gate;
6. preserve `held_out_85606_read=false`.

A pass authorizes a separately committed full Stage-1 matrix with three seeds
and a longer frozen optimization budget. It does not itself authorize later
architecture stages. A failure is diagnosed before any capacity increase.
