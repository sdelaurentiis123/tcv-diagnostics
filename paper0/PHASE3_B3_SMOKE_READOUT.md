# Phase 3 B3 functional-generative smoke readout

**Status:** bounded implementation gate passed

**Scientific result:** no

**Development simulation:** 85604 only

**Held-out 85606 read:** no

**Slurm job:** 6898604

**Execution commit:** `fb89828f6837ce0568fe7fc565b931810da68262`

## What ran

Job 6898604 executed the exact bounded scope frozen in
`paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md`:

- the selected deterministic C5P-H1 seed-1701 transition was the parent;
- the accepted seed-1701 C5P-dcae_l10 codec and its existing training-only
  latent normalization were reused without refitting;
- one 32-component Gaussian vector was embedded globally and supplied to all
  16 transformer blocks for each ensemble member;
- decoded standardized five-field fair CRPS used two members per target;
- 16 training targets and four validation targets were used for two epochs;
- exactly two optimizer steps were taken;
- no physics-derived quantity entered the loss or checkpoint-selection rule.

The run used one Rocky 9.8 NVIDIA H100. Slurm completed with exit `0:0` in 58
seconds. The measured training-core wall time was 7.84 seconds, peak allocated
CUDA memory was 2,450,822,144 bytes, and batch maximum resident memory was
7,054,808 KiB. The complete Rocky 9 suite passed first with `795 passed, 1
skipped, 29 subtests passed`.

## Mechanical gate results

Every required implementation check passed:

1. **Exact parent load.** All 151 deterministic child state keys matched the
   parent. The only missing keys were the 70 newly introduced noise-embedding
   and noise-adapter tensors. There were no unexpected parent keys.
2. **Noise-disabled identity.** Before optimization, the retrofitted
   transition and deterministic H1 parent were bitwise identical on the fixed
   validation probe; the maximum absolute difference was exactly zero.
3. **Staged optimization.** The 51,612,800 pretrained parameters and 9,548,800
   new stochastic parameters both received finite nonzero gradients. The
   complete transition contains 61,161,600 trainable parameters.
4. **Frozen representation.** The in-memory codec state digest was identical
   before and after optimization. The latent-normalization file retained its
   original SHA-256 and was not refit.
5. **Noncollapsed members.** The two fixed-noise members were finite and
   distinct. Their standardized-latent RMS difference was 0.1648830 and their
   decoded-field RMS difference was 0.0467720.
6. **Canonical interface.** The fixed probe returned
   `[1,2,1,5,64,32,88]`, meaning batch, member, future time, channel, x, y,
   and z.
7. **Exact reload.** A newly constructed model loaded from the selected B3
   checkpoint reproduced both fixed-noise latent members and decoded members
   bit for bit.
8. **Tracking.** W&B initialized online, logged both epochs, finished
   remotely, and uploaded no checkpoints. Immutable Ceph artifacts remain the
   source of truth.

## What the short losses do and do not mean

For plumbing visibility, the four-target validation fair CRPS was 0.0456659
after epoch zero and 0.0364771 after epoch one. The selected smoke checkpoint
is therefore epoch one. The corresponding member-spread term was nonzero in
all five fields.

These values are **not scientific performance estimates**. Four validation
frames, two epochs, and one seed cannot establish calibration, transport
fidelity, or superiority to deterministic O2 or B2. Their only valid use here
is to show that the objective is finite, responds to optimization, and does
not collapse immediately.

## Immutable evidence

The byte-identical tracked summary is
`paper0/results/phase3_b3_fgn_gpu_smoke_6898604.json`, SHA-256
`dbac54c033917abbfec7e380d96a0c9be93667ae58240b4403400b57c76e2808`.
The immutable full run result has SHA-256
`fccb26d5ee22d7bf8e716a3ac483263d0bce6fde32d4bdaadf8a015a6700ffd1`.
The complete artifact index has SHA-256
`8cd352368e40639d230b384d528a93c67d833098f9d9ff4ee1efa53cd3f20f62`.
The selected smoke checkpoint has SHA-256
`0390d6f7b96497688b34e0d0aedf5ffeccacae59eacacdcd09438eabf47345de`.

The full artifact root is:

~~~text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/
phase3_b3_fgn_gpu_smoke/job_6898604
~~~

The online monitoring mirror is:

~~~text
https://wandb.ai/sdelaurentiis123-columbia-university/
tcv-diagnostics-paper0/runs/p0b3smoke-6898604-s1701
~~~

## Decision

The implementation gate is accepted. It is now technically defensible to
freeze and run the prospective 100-epoch, one-seed 85604 B3 pilot and its
separate scientific evaluator.

This result does not itself authorize 85606, autonomous rollout, transport or
calibration claims, assimilation, diagnostic ranking, control, additional
seeds, or any architectural ablation.
