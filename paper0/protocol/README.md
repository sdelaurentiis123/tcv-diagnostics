# Held-out release protocol

Shot 85606 is unavailable to ordinary Paper 0 development code.

Before any new Paper 0 evaluation reads it, this directory must contain a committed protocol lock specifying at least:

- model architecture and immutable configuration;
- checkpoint-selection rule and selected 85604 checkpoint;
- training budget and random seeds;
- forecast horizons and ensemble sizes;
- metric implementations and acceptance thresholds;
- diagnostic observation operators and noise assumptions;
- filter, inflation, localization, and update settings;
- diagnostic-ranking rules;
- plotting and uncertainty-interval conventions;
- Git commit and clean-state assertion;
- explicit authorization and release timestamp.

The loader guard and release format will be implemented during Phase 1. No release file exists during Phase 0.
