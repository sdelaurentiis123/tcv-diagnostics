# Paper 0 protocol amendments

The governing specification is preserved verbatim in `PAPER0_SPEC.txt`. Necessary clarifications are recorded here rather than silently rewriting it.

## A001 - Historical exposure of shot 85606

**Status:** active from repository initialization.

Shot 85606 was inspected repeatedly during exploratory work in the predecessor repository before this clean Paper 0 protocol existed. Therefore it cannot honestly be described as historically or researcher blind.

Paper 0 will nevertheless sequester 85606 from all new training, validation, architecture selection, checkpoint selection, metric development, assimilation tuning, and acceptance-threshold selection. After the complete protocol is frozen and committed, 85606 may be used for one prospectively locked confirmatory evaluation. The paper must describe it as a held-out simulation with prior exploratory exposure, not as a never-seen blind test.

A genuinely blind confirmation requires an additional unseen Hermes simulation supplied after the protocol is frozen.
