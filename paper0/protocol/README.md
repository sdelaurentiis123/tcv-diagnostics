# Held-out release protocol

Shot 85606 is unavailable to ordinary Paper 0 development code.

Before any new Paper 0 evaluation reads it, this directory must contain a committed protocol lock specifying at least:

- model architecture and immutable configuration;
- checkpoint-selection rule and selected 85604 checkpoint;
- training budget and random seeds;
- forecast horizons and ensemble sizes;
- geometry masks, outward-sign convention, surface integration, SI units, and
  member-wise nonlinear transport are frozen in
  `PHASE2_GEOMETRY_UNITS_PROTOCOL.md`;
- historical f8/z44 transport attribution and acceptance are frozen in
  `PHASE2_O1_TRANSPORT_PROTOCOL.md`;
- metric implementations and acceptance thresholds;
- diagnostic observation operators and noise assumptions;
- filter, inflation, localization, and update settings;
- diagnostic-ranking rules;
- plotting and uncertainty-interval conventions;
- Git commit and clean-state assertion;
- explicit authorization and release timestamp.

The loader guard and release format will be implemented during Phase 1. No release file exists during Phase 0.

Development-only protocol locks that do not authorize held-out access include:

- `PHASE2_STATE_COMPLETENESS_PROTOCOL.md`: all-rank 85604 evolved-state
  inventory and momentum/velocity closure;
- `PHASE2_PRESSURE_CLOSURE_AUDIT_PROTOCOL.md`: all-frame 85604 pressure and
  temperature closure;
- `PHASE2_O1_CODEC_PROTOCOL.md` and `PHASE2_O1_TRANSPORT_PROTOCOL.md`: frozen
  historical codec reconstruction and transport attribution.
