# TCV Diagnostics

Clean-room Paper 0 repository for transport-faithful probabilistic emulation and synthetic diagnostic assimilation of TCV/Hermes edge-turbulence simulations.

The governing research specification is [`paper0/PAPER0_SPEC.txt`](paper0/PAPER0_SPEC.txt). Phase 0 is complete with documented legacy discrepancies; [`paper0/AUDIT.md`](paper0/AUDIT.md) is the authoritative handoff. Phase 1 now defines the guarded 85604-only data protocol before any architecture work begins.

The pre-execution Phase 1 rules are frozen in
[`paper0/protocol/PHASE1_DATA_PROTOCOL.md`](paper0/protocol/PHASE1_DATA_PROTOCOL.md).
They explicitly distinguish the inherited five observable channels from a
physically complete Markov state and keep physical time as relative cadence and
forecast lead rather than an absolute trajectory lookup feature.

The executed data findings and original go/no-go decision are in
[`paper0/PHASE1_READOUT.md`](paper0/PHASE1_READOUT.md). The 85604 trajectory has
slow background evolution plus fast, coherently translating toroidal
fluctuations; the data characterization is complete, and the stationary
screen remains failed.

The prospective replacement is now frozen in
[`paper0/protocol/PHASE2_CONDITIONAL_TRANSIENT_STATE_PROTOCOL.md`](paper0/protocol/PHASE2_CONDITIONAL_TRANSIENT_STATE_PROTOCOL.md).
It permits only short-horizon 85604 development under a
conditional-transient, later-background-extrapolation interpretation. It
freezes the exact `E6B-H1` source state, the pragmatic `C5P-H2` history
baseline, and a mandatory `C5P-H1` control. It does not authorize stationary
post-decorrelation claims, stochastic architecture selection, assimilation,
diagnostic ranking, or any 85606 access. A separate matched O1/O2 model
protocol remains required before training.

The shared engineering input to that comparison is now prospectively frozen
in
[`paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md`](paper0/protocol/PHASE2_MODEL_DATASET_PROTOCOL.md).
It requires all three state views to come from the same eight-field 85604
source union, the same exact saved `Bphi` boundary record, the same
`zperiod=5` Fourier resampling, and normalization fit only on frames
`[0,432)`. Rocky 9 job `6893525` passed every source, coverage,
resampling, legacy-equality, storage, boundary, and normalization gate. The
compact evidence and its intentionally limited meaning are in
[`paper0/PHASE2_MODEL_DATASET_READOUT.md`](paper0/PHASE2_MODEL_DATASET_READOUT.md).
The matched deterministic model experiment is now prospectively frozen in
[`paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md`](paper0/protocol/PHASE2_MATCHED_O1_O2_PROTOCOL.md).
It first trains from-scratch codecs with data-only loss and requires
field/spectral/cross-field/transport reconstruction gates before any one-step
model may run. A three-arm one-step ladder then compares exact Hermes state,
two-frame pragmatic state, and one-frame pragmatic state against persistence
and uncompressed linear/spectral references. The attributed DCAE port,
verified 85604 reader, data-only optimizer, Rocky 9 CPU gate, and bounded H100
smoke now pass. The smoke trained only 16 frames for two epochs and is not a
scientific result. The next authorized step is the full three-seed C5P/E6B R1
codec training followed by the complete O1 gate; O2 remains blocked until R1
passes or the predeclared R2 repair is evaluated.

The historical codecs have now been evaluated through the frozen O1 oracle;
[`paper0/PHASE2_O1_READOUT.md`](paper0/PHASE2_O1_READOUT.md) records why neither
passes the spectral/cross-field representation gate. The exact-source
transport definition is frozen in
[`paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md`](paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md):
the former image-space flux is retained only as a proxy. The shifted-`DDY`,
shifted-`xy` radial-face, and combined conservative radial-flow primitives
match their frozen compiled oracles.
The narrow execution evidence and remaining blockers are summarized in
[`paper0/PHASE2_SHIFTED_DDY_READOUT.md`](paper0/PHASE2_SHIFTED_DDY_READOUT.md)
and
[`paper0/PHASE2_XY_FACE_READOUT.md`](paper0/PHASE2_XY_FACE_READOUT.md).
The combined-flow result is documented separately in
[`paper0/PHASE2_RADIAL_FLOW_READOUT.md`](paper0/PHASE2_RADIAL_FLOW_READOUT.md).
The native-81, 85604-only comparison was frozen before execution in
[`paper0/protocol/PHASE2_NATIVE_FRAME_PROTOCOL.md`](paper0/protocol/PHASE2_NATIVE_FRAME_PROTOCOL.md).
Its compiled transport-operator subgate passed all 15 real-state cases, while
the prospectively required full-domain five-channel closure failed at one
negative ion-pressure point. The exact split result and its implications for
the emulator state are recorded in
[`paper0/PHASE2_NATIVE_FRAME_READOUT.md`](paper0/PHASE2_NATIVE_FRAME_READOUT.md).

The follow-up all-frame 85604 audit is complete. It shows that negative evolved
ion pressure is rare by cell count but reaches the fixed transport interior in
47 of 624 frames, so the legacy ion-temperature channel cannot exactly recover
the pressure used by the conservative operator. The accepted evidence and
explicitly limited state recommendation are in
[`paper0/PHASE2_PRESSURE_CLOSURE_READOUT.md`](paper0/PHASE2_PRESSURE_CLOSURE_READOUT.md).

Before any new data conversion or training, the resulting `C5T` versus `C5P`
state distinction and the native-81 transport scoring rule are frozen in
[`paper0/protocol/PHASE2_STATE_RESAMPLING_PROTOCOL.md`](paper0/protocol/PHASE2_STATE_RESAMPLING_PROTOCOL.md).
The complete 85604 resampling audit, geometry/unit audit, and deterministic
codec-transport extension have now passed their execution-integrity gates,
while 85606 remains prohibited. The transport extension is documented in
[`paper0/PHASE2_O1_TRANSPORT_READOUT.md`](paper0/PHASE2_O1_TRANSPORT_READOUT.md).
It shows negligible state/storage-path error, good integrated-separatrix
fidelity, and a failed local-face criterion for f8. z44 passes the radial ExB
transport subgate but retains its earlier spectral/cross-field failures and is
not a matched training ablation. Neither historical codec is accepted for
dynamics training. The active blocker is again upstream: a defensible 85604
temporal protocol and physically sufficient forecast state must be frozen
before O2 or any new architecture run.

The evidence leading to the exact source-state decision is documented in
[`paper0/PHASE1_STATE_TIME_DECISION_MEMO.md`](paper0/PHASE1_STATE_TIME_DECISION_MEMO.md).
The simulator advances six volumetric fields, while potential is an elliptic
derived field with short boundary memory. The memo distinguishes physical
history/relative lead from an unsafe absolute-frame lookup and lays out the
deterministic state-closure tests. It remains a design memo; the later frozen
conditional-transient protocol is the active decision.

The first two source-state gates are now complete. The all-rank 85604 inventory
found the six evolved fields on all 256 ranks, and both exact electron and ion
velocity-to-momentum relations pass every one of 624 frames with zero
discrepant physical cells. The density floor is inactive in this saved
interval. This establishes algebraic equivalence of density-plus-velocity and
density-plus-momentum representations, but historical C5 still omits electron
parallel state and has not passed the separate potential/vorticity boundary
gate. Exact evidence and implications are in
[`paper0/PHASE2_STATE_COMPLETENESS_READOUT.md`](paper0/PHASE2_STATE_COMPLETENESS_READOUT.md).
The subsequent closure result selects the exact source-state candidate; no
model architecture has been selected.

The follow-up 85604 radial-potential guard audit is also complete. The saved
guards satisfy the exact Hermes copy and toroidal-midpoint structure, but the
compact boundary midpoint differs from the instantaneous target at every saved
frame/y location. Guard-stripped evolved volumes are therefore not the exact
saved state. The measured departures are reported without a post hoc
materiality cutoff in
[`paper0/PHASE2_PHI_BOUNDARY_STATE_READOUT.md`](paper0/PHASE2_PHI_BOUNDARY_STATE_READOUT.md).
The paired exact elliptic solve and all-frame forward closure are now complete.
They establish `S6+Bphi` as the exact saved-state candidate and motivate the
matched exact-versus-history comparison frozen above.
