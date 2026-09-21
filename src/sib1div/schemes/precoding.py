"""The four paired rank-1 precoders used by Plan 001."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.config import SimulationConfig


SCHEMES = ("baseline", "pol_cycling", "beam_cycling", "beam_cdd")


@dataclass(frozen=True)
class PrecoderResult:
    weights: NDArray[np.complex128]
    raw_power: NDArray[np.float64]


def dual_polarized(spatial: NDArray[np.complexfloating], phase_deg: float = 0.0) -> NDArray[np.complex128]:
    spatial = np.asarray(spatial, dtype=np.complex128).reshape(-1)
    result = np.concatenate((spatial, np.exp(1j * np.deg2rad(phase_deg)) * spatial)) / np.sqrt(2.0)
    return result / np.linalg.norm(result)


def build_precoder(
    config: SimulationConfig,
    scheme: str,
    selected_ssb: int,
    ssb_txru_weights: NDArray[np.complexfloating],
    secondary_txru_weights: NDArray[np.complexfloating],
    *,
    prg_size_prbs: int | None = None,
) -> PrecoderResult:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme: {scheme}")
    n_sc = config.active_subcarriers
    prg_prbs = int(config.data["nr"]["prg_size_prbs"] if prg_size_prbs is None else prg_size_prbs)
    if prg_prbs < 1 or config.pdsch_prbs % prg_prbs:
        raise ValueError("PRG size must be a positive divisor of the PDSCH allocation")
    prg_sc = prg_prbs * 12
    q = np.arange(n_sc)
    prg = q // prg_sc
    settings = config.data["schemes"]
    ssb_spatial = np.asarray(ssb_txru_weights)[:, selected_ssb]
    weights = np.empty((2 * ssb_spatial.size, n_sc), dtype=np.complex128)

    if scheme == "baseline":
        weights[:] = dual_polarized(ssb_spatial, float(settings.get("baseline_pol_phase_deg", 0.0)))[:, None]
    elif scheme == "pol_cycling":
        phases = np.asarray(settings["pol_cycling_phase_deg"], dtype=float)
        hold = int(settings.get("pol_cycling_hold_prgs", 1))
        for k in range(n_sc):
            weights[:, k] = dual_polarized(ssb_spatial, phases[(prg[k] // hold) % phases.size])
    elif scheme == "beam_cycling":
        children = (2 * selected_ssb, 2 * selected_ssb + 1)
        hold = int(settings.get("beam_cycling_hold_prgs", 1))
        for k in range(n_sc):
            spatial = np.asarray(secondary_txru_weights)[:, children[(prg[k] // hold) % 2]]
            weights[:, k] = dual_polarized(spatial)
    else:
        children = (2 * selected_ssb, 2 * selected_ssb + 1)
        branches = np.column_stack([
            dual_polarized(np.asarray(secondary_txru_weights)[:, child]) for child in children
        ])
        indices = np.asarray(settings["beam_cdd_delay_grid_indices"], dtype=float)
        denominator = float(settings["beam_cdd_phase_denominator"])
        phases = np.exp(-1j * 2 * np.pi * q[None, :] * indices[:, None] / denominator) / np.sqrt(indices.size)
        weights = branches @ phases

    raw_power = np.sum(np.abs(weights) ** 2, axis=0).real
    if settings["normalization"] == "per_active_subcarrier":
        weights = weights / np.sqrt(raw_power)[None, :]
    return PrecoderResult(weights, raw_power)


def assert_unit_norm(result: PrecoderResult, tolerance: float = 1e-12) -> None:
    error = np.max(np.abs(np.sum(np.abs(result.weights) ** 2, axis=0) - 1.0))
    if error > tolerance:
        raise AssertionError(f"precoder norm error {error} exceeds {tolerance}")
