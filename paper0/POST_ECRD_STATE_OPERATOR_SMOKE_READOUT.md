# Post-ECRD codec-free state-operator smoke readout

**Date:** 2026-08-24

**Development simulation:** 85604 only

**Held-out simulation 85606:** unopened

**Scientific result:** no; this is an implementation gate

## Outcome

The direct field-space operator is mechanically ready for a matched one-step
C5P-versus-exact-saved-state experiment.

Rusty job `6933543` completed at clean commit
`c1946dc8e568084f0a06dc89d3f79f81d6d2b5aa`. Slurm reports `COMPLETED`, exit
`0:0`, elapsed time 29 seconds, and maximum batch RSS `1266396K`. The focused
on-node suite passed all 35 tests. The online W&B run is remotely verified as
finished.

The smoke made four optimizer updates total: two for C5P and two for E6B. It
used two early training pairs and one later validation pair solely to verify
full-volume forward, backward, checkpoint, and data-routing mechanics. These
losses do not compare model quality.

## What was built

The shared processor is a codec-free mixed-boundary 3D increment operator:

- C5P input/output: `Ne, Pe, Pi, phi, Vi`;
- E6B volume input/output: `Ne, Pe, Pi, NVe, NVi, Vort`;
- E6B retained-boundary input/output: inner and outer `Bphi` profiles;
- direct normalized state-derivative prediction;
- no autoencoder or latent codec;
- zero/wall padding in x and y;
- circular padding in toroidal z;
- no toroidal stride or downsampling;
- no absolute toroidal coordinate;
- random shared toroidal rolls for training pairs;
- x/y-only multiresolution processing;
- field and saved-boundary MSE only, with no flux, spectrum, cross-phase,
  coherence, PDE, or conservation loss.

The smoke models are intentionally tiny: 33,845 parameters for C5P and 35,640
for E6B. They establish wiring, not capacity.

## Mechanical evidence

| check | C5P | E6B |
|---|---:|---:|
| finite forward/loss/gradient | pass | pass |
| optimizer steps | 2 | 2 |
| checkpoint reload bitwise exact | pass | pass |
| every toroidal convolution stride is one | pass | pass |
| volume roll normalized maximum error | 1.58744e-6 | 1.09972e-6 |
| volume roll normalized RMS error | 2.03258e-7 | 2.51639e-7 |
| Bphi roll normalized maximum error | not applicable | 5.96046e-8 |
| Bphi roll normalized RMS error | not applicable | 1.48647e-8 |

Peak allocated CUDA memory was only `0.1715 GiB`, confirming that this bounded
smoke did not require the large resource requests used by earlier ECRD runs.

## What the two failed jobs taught us

Jobs `6933527` and `6933538` completed training mechanics but failed the
toroidal numerical gate with cuDNN TF32 enabled. Full-volume normalized roll
errors were around `6e-4` to `8e-4`. The second run added an RMS diagnostic and
confirmed that the discrepancy was distributed.

The A100 reported `torch.backends.cudnn.allow_tf32=True`. The third job kept
the committed thresholds but disabled TF32. Errors fell by roughly three
orders of magnitude to `1e-6` maximum and `2e-7` RMS. This supports a numerical
precision explanation rather than an architectural toroidal-symmetry defect.

The failures are retained in:

- `paper0/results/post_ecrd_state_operator_smoke_failure_6933527.json`;
- `paper0/results/post_ecrd_state_operator_smoke_failure_6933538.json`;
- `paper0/protocol/POST_ECRD_OPERATOR_SMOKE_NUMERICAL_AMENDMENT_2026-08-24.md`;
- `paper0/protocol/POST_ECRD_OPERATOR_SMOKE_TF32_AMENDMENT_2026-08-24.md`.

## What this does not show

The E6B smoke loss is numerically larger than the C5P smoke loss, but that is
not evidence against exact state. The two views contain different variables,
different numbers of outputs, and only two updates. Vorticity, momentum, and
the saved boundary also have different conditional difficulty. A meaningful
comparison requires the frozen matched training budget and transport
evaluation.

This smoke does not show:

- that exact state improves forecasting;
- that more data improves forecasting;
- that the dynamics are Markov-complete;
- that GAOT is better than the U-Net/operator baseline;
- calibrated uncertainty;
- transport-faithful rollout;
- held-out generalization;
- assimilation, sensor ranking, or steering.

## Exact next gate

The repository and known Rusty paths still expose only one 624-frame restart
continuation. No larger 85604 path is staged or recorded. Before scientific
training, a dated dataset amendment must inventory Ben's additional material,
identify trajectories versus continuations, hash the sources, and freeze
attainable data budgets and blind splits.

Once that exists, the first scientific experiment is the matched one-step
factorial comparison:

```text
state view:  C5P versus E6B
data amount: attainable 1x, 2x, 4x, all-data budgets
model:       same codec-free processor
loss:        state variables only
selection:   every chronological 85604 validation block
evaluation:  fields, spectra, cross-field phase/coherence, and transport
```

No production training, 85606 access, stochastic extension, assimilation, or
diagnostic ranking is authorized by this smoke.

