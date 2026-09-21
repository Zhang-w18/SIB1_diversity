"""Paired Monte Carlo orchestration and stopping rules."""

from .randomness import CommonSamples, common_samples
from .adaptive import (
    curve_is_resolved,
    refinement_snrs,
    run_adaptive_simulation,
    run_fixed_curve_simulation,
)
from .engine import LinkResult, link_mode_scaling, run_simulation, simulate_one_drop

__all__ = [
    "CommonSamples", "LinkResult", "common_samples", "curve_is_resolved", "link_mode_scaling",
    "refinement_snrs", "run_adaptive_simulation", "run_fixed_curve_simulation", "run_simulation",
    "simulate_one_drop",
]
