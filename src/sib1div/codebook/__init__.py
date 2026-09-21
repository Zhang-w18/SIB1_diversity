"""Offline SSB and secondary-beam codebook generation and diagnostics."""

from .array import ArrayGeometry, build_txru_mapping
from .synthesis import BeamCodebook, BeamRegion, generate_main_codebooks

__all__ = [
    "ArrayGeometry",
    "BeamCodebook",
    "BeamRegion",
    "build_txru_mapping",
    "generate_main_codebooks",
]
