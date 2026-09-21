"""Weighted iterative least-squares synthesis for fixed flat-top codebooks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.config import SimulationConfig

from .array import ArrayGeometry


@dataclass(frozen=True)
class BeamRegion:
    index: int
    phi_min_deg: float
    phi_max_deg: float
    eta_min_deg: float
    eta_max_deg: float
    parent_ssb: int | None = None

    @property
    def phi_center_deg(self) -> float:
        return 0.5 * (self.phi_min_deg + self.phi_max_deg)

    @property
    def eta_center_deg(self) -> float:
        return 0.5 * (self.eta_min_deg + self.eta_max_deg)


@dataclass(frozen=True)
class BeamCodebook:
    name: str
    regions: tuple[BeamRegion, ...]
    txru_weights: NDArray[np.complex128]
    ae_weights: NDArray[np.complex128]


def _horizontal_edges_uniform_u(
    count: int,
    phi_min_deg: float,
    phi_max_deg: float,
    eta_reference_deg: float,
    downtilt_deg: float,
) -> NDArray[np.float64]:
    eta_local = np.deg2rad(eta_reference_deg - downtilt_deg)
    scale = np.cos(eta_local)
    u_min = scale * np.sin(np.deg2rad(phi_min_deg))
    u_max = scale * np.sin(np.deg2rad(phi_max_deg))
    u_edges = np.linspace(u_min, u_max, count + 1)
    edges = np.rad2deg(np.arcsin(np.clip(u_edges / scale, -1.0, 1.0)))
    edges[0] = phi_min_deg
    edges[-1] = phi_max_deg
    return edges


def _regions(
    count: int,
    phi_min_deg: float,
    phi_max_deg: float,
    eta_min_deg: float,
    eta_max_deg: float,
    downtilt_deg: float,
    *,
    secondary: bool,
) -> tuple[BeamRegion, ...]:
    edges = _horizontal_edges_uniform_u(
        count, phi_min_deg, phi_max_deg,
        0.5 * (eta_min_deg + eta_max_deg), downtilt_deg,
    )
    return tuple(
        BeamRegion(
            index=i,
            phi_min_deg=float(edges[i]),
            phi_max_deg=float(edges[i + 1]),
            eta_min_deg=eta_min_deg,
            eta_max_deg=eta_max_deg,
            parent_ssb=i // 2 if secondary else None,
        )
        for i in range(count)
    )


def _initial_weight(geometry: ArrayGeometry, region: BeamRegion) -> NDArray[np.complex128]:
    phi_samples = np.linspace(region.phi_min_deg, region.phi_max_deg, 3)
    eta_samples = np.linspace(region.eta_min_deg, region.eta_max_deg, 5)
    phi_grid, eta_grid = np.meshgrid(phi_samples, eta_samples, indexing="xy")
    steering = geometry.steering_txru(phi_grid, eta_grid)
    phases = np.exp(-1j * np.angle(steering[0]))
    weight = steering @ phases
    return weight / np.linalg.norm(weight)


def synthesize_flat_top(
    geometry: ArrayGeometry,
    region: BeamRegion,
    *,
    iterations: int = 10,
    ridge: float = 2e-3,
) -> NDArray[np.complex128]:
    """Synthesize one beam on a fixed global-angle design grid."""
    phi_axis = np.arange(-75.0, 75.01, 2.5)
    eta_axis = np.arange(5.0, 40.01, 2.5)
    phi, eta = np.meshgrid(phi_axis, eta_axis, indexing="xy")
    steering = geometry.steering_txru(phi, eta)
    inside = (
        (phi.ravel() >= region.phi_min_deg)
        & (phi.ravel() <= region.phi_max_deg)
        & (eta.ravel() >= region.eta_min_deg)
        & (eta.ravel() <= region.eta_max_deg)
    )
    guard = (
        (phi.ravel() >= region.phi_min_deg - 4.0)
        & (phi.ravel() <= region.phi_max_deg + 4.0)
        & (eta.ravel() >= region.eta_min_deg - 3.0)
        & (eta.ravel() <= region.eta_max_deg + 3.0)
    )
    sample_weights = np.where(inside, 1.0, np.where(guard, 0.12, 0.025))
    target_amplitude = inside.astype(float)
    weight = _initial_weight(geometry, region)
    eye = np.eye(geometry.txru_count)
    lhs = (steering * sample_weights[None, :]) @ steering.conj().T + ridge * eye
    for _ in range(iterations):
        field = steering.conj().T @ weight
        desired = target_amplitude * np.exp(1j * np.angle(field))
        rhs = steering @ (sample_weights * desired)
        weight = np.linalg.solve(lhs, rhs)
        weight /= np.linalg.norm(weight)
    return weight


def _make_codebook(
    name: str,
    geometry: ArrayGeometry,
    regions: tuple[BeamRegion, ...],
) -> BeamCodebook:
    txru = np.column_stack([synthesize_flat_top(geometry, region) for region in regions])
    ae = geometry.ae_weights(txru)
    return BeamCodebook(name, regions, txru, ae)


def generate_main_codebooks(
    config: SimulationConfig,
) -> tuple[ArrayGeometry, BeamCodebook, BeamCodebook]:
    data = config.data
    array = data["bs_array"]
    scenario = data["scenario"]
    panel = data["panel"]
    books = data["codebooks"]
    geometry = ArrayGeometry(
        vertical_aes=int(array["vertical_aes"]),
        horizontal_aes=int(array["horizontal_aes"]),
        vertical_txrus=int(array["vertical_txrus_per_pol"]),
        horizontal_txrus=int(array["horizontal_txrus_per_pol"]),
        horizontal_spacing_lambda=float(array["horizontal_spacing_lambda"]),
        vertical_spacing_lambda=float(array["vertical_spacing_lambda"]),
        mechanical_downtilt_deg=float(panel["mechanical_downtilt_deg"]),
    )
    height = float(scenario["bs_height_m"]) - float(scenario["ue_height_m"])
    eta_min = float(np.rad2deg(np.arctan2(height, float(scenario["ue_distance_max_m"]))))
    eta_max = float(np.rad2deg(np.arctan2(height, float(scenario["ue_distance_min_m"]))))
    common = dict(
        phi_min_deg=float(scenario["sector_azimuth_min_deg"]),
        phi_max_deg=float(scenario["sector_azimuth_max_deg"]),
        eta_min_deg=eta_min,
        eta_max_deg=eta_max,
        downtilt_deg=geometry.mechanical_downtilt_deg,
    )
    ssb_regions = _regions(int(books["ssb_horizontal_beams"]), secondary=False, **common)
    secondary_regions = _regions(
        int(books["secondary_horizontal_beams"]), secondary=True, **common
    )
    return (
        geometry,
        _make_codebook("ssb_8h1v", geometry, ssb_regions),
        _make_codebook("secondary_16h1v", geometry, secondary_regions),
    )
