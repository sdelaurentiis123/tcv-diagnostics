"""Prospectively frozen Phase 3.5 cause-localization utilities.

The package contains diagnostic statistics and immutable-artifact adapters.
It contains no production model training or held-out-shot access.
"""

from .scope import Phase35Block, Phase35Protocol, load_phase3_5_protocol

__all__ = ["Phase35Block", "Phase35Protocol", "load_phase3_5_protocol"]
