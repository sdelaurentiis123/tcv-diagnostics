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

- `PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md`: the transparent post-O1,
  pre-O2 decision to retain the 3/3-passing C5P representation, keep E6B as a
  failed representation ablation, and run only the frozen C5P-H1/H2
  teacher-forced comparison;
- `PHASE2_MODEL_DATASET_PROTOCOL.md`: the shared, hash-locked 85604
  native-81 to model-88 conversion, boundary-state storage, training-only
  normalization, and hard integrity gates required before matched O1/O2
  learning;
- `PHASE2_MATCHED_O1_O2_PROTOCOL.md`: the from-scratch representation
  escalation, deterministic one-step state/history comparison, training
  budgets, uncompressed references, and stop/go gates;
- `PHASE2_CONDITIONAL_TRANSIENT_STATE_PROTOCOL.md`: the nonstationary 85604
  development interpretation, exact `E6B-H1` state, pragmatic
  `C5P-H2` state, and mandatory `C5P-H1` history control;
- `PHASE2_STATE_COMPLETENESS_PROTOCOL.md`: all-rank 85604 evolved-state
  inventory and momentum/velocity closure;
- `PHASE2_PHI_BOUNDARY_STATE_PROTOCOL.md`: all-frame 85604 saved radial
  potential-boundary state and exact instantaneous-Neumann classification;
- `PHASE2_POTENTIAL_ELLIPTIC_PROTOCOL.md`: selected-frame 85604 source
  reconstruction and paired saved-versus-instantaneous radial-boundary
  elliptic solve;
- `PHASE2_PRESSURE_CLOSURE_AUDIT_PROTOCOL.md`: all-frame 85604 pressure and
  temperature closure;
- `PHASE2_O1_CODEC_PROTOCOL.md` and `PHASE2_O1_TRANSPORT_PROTOCOL.md`: frozen
  historical codec reconstruction and transport attribution.
- `PHASE3_B4_PDE_REFINER_PROTOCOL.md`: the post-B3, preimplementation B4
  model definition and bounded smoke authority;
- `PHASE3_B4_FULL_TRAINING_EVALUATION_PROTOCOL.md`: the post-smoke,
  pre-full-run seed-1701 B4 training budget, truth-separated one-step
  generation, separate H-det/H-prob gates, and evaluator-smoke requirement.
  It explicitly keeps 85606, O3 execution, assimilation, and diagnostic
  ranking closed.
- `PHASE3_B5_RESIDUAL_AUDIT_PROTOCOL.md`: the post-B4, training-region-only
  deterministic-H1 residual audit and truth-separated residual provenance.
- `PHASE3_B5_FIELD_RESIDUAL_EDM_SMOKE_PROTOCOL.md`: the bounded joint
  field-space residual-EDM mechanics test; it contains no scientific result.
- `PHASE3_B5_FULL_TRAINING_EVALUATION_PROTOCOL.md`: the post-smoke,
  pre-full-run seed-1701 B5 budget, data-only EDM-loss selection,
  truth-separated M32 evaluation, and complete one-seed field, spectral,
  cross-field, transport, Monte Carlo, and chronological gate. Its completed
  failure leaves O3, more seeds, assimilation, diagnostic ranking, and 85606
  closed.
- `PHASE3_B5_COVARIANCE_LOCALIZATION_PROTOCOL.md`: the post-gate,
  preimplementation read-only comparison of B5 anomalies, H1 residuals, B5
  innovations, exact-separatrix transport covariance, dependence-sensitive
  scores, and one training-frozen residual-history diagnostic. It authorizes
  no training, inference, forecast modification, downstream assimilation, or
  85606 access.
