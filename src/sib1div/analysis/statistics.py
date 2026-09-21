"""BLER aggregation with Wilson score confidence intervals."""

from __future__ import annotations

from statistics import NormalDist


def wilson_interval(errors: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0 or not 0 < confidence < 1:
        raise ValueError("total must be positive and confidence must lie in (0,1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return max(0.0, center - half), min(1.0, center + half)

