# TCV Diagnostics

Clean-room Paper 0 repository for transport-faithful probabilistic emulation and synthetic diagnostic assimilation of TCV/Hermes edge-turbulence simulations.

The governing research specification is [`paper0/PAPER0_SPEC.txt`](paper0/PAPER0_SPEC.txt). Phase 0 is complete with documented legacy discrepancies; [`paper0/AUDIT.md`](paper0/AUDIT.md) is the authoritative handoff. Phase 1 now defines the guarded 85604-only data protocol before any architecture work begins.

The pre-execution Phase 1 rules are frozen in
[`paper0/protocol/PHASE1_DATA_PROTOCOL.md`](paper0/protocol/PHASE1_DATA_PROTOCOL.md).
They explicitly distinguish the inherited five observable channels from a
physically complete Markov state and keep physical time as relative cadence and
forecast lead rather than an absolute trajectory lookup feature.

The executed data findings and current go/no-go decision are in
[`paper0/PHASE1_READOUT.md`](paper0/PHASE1_READOUT.md). The 85604 trajectory has
slow background evolution plus fast, coherently translating toroidal
fluctuations; the data characterization is complete, but the stationary
learning split remains blocked pending simulator-owner guidance.

The historical codecs have now been evaluated through the frozen O1 oracle;
[`paper0/PHASE2_O1_READOUT.md`](paper0/PHASE2_O1_READOUT.md) records why neither
passes the complete preliminary representation gate. The exact-source
transport audit is frozen in
[`paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md`](paper0/protocol/PHASE2_TRANSPORT_PROTOCOL.md):
the former image-space flux is retained only as a proxy. The shifted-`DDY`,
shifted-`xy` radial-face, and combined conservative radial-flow primitives now
match their frozen compiled oracles, but no physical transport claim is allowed
until native-grid, resampling, geometry, unit, and ensemble rungs pass.
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
The next oracle tests the complete 85604 trajectory and keeps 85606 prohibited.
