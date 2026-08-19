# Paper 0 Phase 3.5 prospective execution clarification B

**Date:** 2026-08-19

**Timing:** written after failed execution job `6906826`, but before any
authoritative Phase 3.5 table, figure, decision memo, or scientific result was
serialized or inspected

**Development simulation:** 85604 only

**Held-out simulation 85606:** unopened and unauthorized

This dated clarification records a provenance-loader correction. It does not
change a hypothesis, split boundary, block, uncertainty procedure,
representation family, coefficient budget, evidence threshold, or decision
priority in `PHASE3_5_PROTOCOL_AMENDMENT.md` or clarification A.

## B1. Failed execution retained

Rocky 9 job `6906826` stopped before the H4 equivariance audit and before the
atomic output-writing stage. The job directory and W&B failure record are
retained. The run read neither guard frames nor 85606. No in-memory scientific
quantity from the failed process was recovered, serialized, or used to alter
the protocol.

The exception occurred while loading the already frozen H1 checkpoint. Its
checkpoint metadata records the codec at:

```text
/mnt/ceph/users/sdelaurentiis/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt
```

The Phase 3.5 artifact lock names the same file through Rusty's home-mounted
Ceph alias:

```text
/mnt/home/sdelaurentiis/ceph/tcv_diagnostics/paper0/phase2_o1_codec_r2/job_6894463/task_0_c5p_seed_1701/selected.pt
```

Both paths resolve to the same canonical file. Both produce SHA-256
`9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d`.
The H1 checkpoint records that hash in both its codec provenance and latent
normalization provenance, and records the codec as non-trainable.

## B2. Narrow Phase 3.5 path correction

Before Phase 3.5 calls the unchanged frozen O2 loader, it canonicalizes its
H1 codec path through strict filesystem resolution. An existing development
artifact is accepted as the recorded codec only when:

1. both path spellings pass the held-out-path prohibition;
2. both existing paths resolve to the same canonical filesystem object;
3. the provided codec bytes match the frozen SHA-256;
4. the checkpoint's recorded codec SHA-256 matches that same frozen hash; and
5. the checkpoint records the codec as non-trainable.

The shared O2 loader and historical hash-pinned launchers remain unchanged.
Canonicalization only makes the Phase 3.5 argument use the checkpoint's
canonical path spelling before the loader performs its existing literal path
comparison. It does not relax byte identity or any
model/configuration/normalization check.

## B3. Rerun rule

The complete analysis must be rerun from the beginning under a new Slurm job
and a new exclusive result directory. Job `6906826` is not resumable and is
not an authoritative Phase 3.5 result. The new run manifest must hash this
clarification as well as the original protocol and clarification A.
