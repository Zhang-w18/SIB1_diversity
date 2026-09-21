"""URA steering and the fixed contiguous AE-to-TXRU mapping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


def build_txru_mapping(
    vertical_aes: int,
    horizontal_aes: int,
    vertical_txrus: int,
    horizontal_txrus: int,
) -> ComplexMatrix:
    """Build the per-polarization, contiguous, equal-phase subarray mapping."""
    if vertical_aes % vertical_txrus or horizontal_aes % horizontal_txrus:
        raise ValueError("AE dimensions must be divisible by TXRU dimensions")
    block_m = vertical_aes // vertical_txrus
    block_n = horizontal_aes // horizontal_txrus
    mapping = np.zeros(
        (vertical_aes * horizontal_aes, vertical_txrus * horizontal_txrus),
        dtype=np.complex128,
    )
    amplitude = 1.0 / np.sqrt(block_m * block_n)
    for tm in range(vertical_txrus):
        for tn in range(horizontal_txrus):
            txru = tm * horizontal_txrus + tn
            for m in range(tm * block_m, (tm + 1) * block_m):
                for n in range(tn * block_n, (tn + 1) * block_n):
                    mapping[m * horizontal_aes + n, txru] = amplitude
    return mapping


@dataclass(frozen=True)
class ArrayGeometry:
    vertical_aes: int
    horizontal_aes: int
    vertical_txrus: int
    horizontal_txrus: int
    horizontal_spacing_lambda: float
    vertical_spacing_lambda: float
    mechanical_downtilt_deg: float

    @property
    def mapping(self) -> ComplexMatrix:
        return build_txru_mapping(
            self.vertical_aes,
            self.horizontal_aes,
            self.vertical_txrus,
            self.horizontal_txrus,
        )

    @property
    def ae_count(self) -> int:
        return self.vertical_aes * self.horizontal_aes

    @property
    def txru_count(self) -> int:
        return self.vertical_txrus * self.horizontal_txrus

    def steering_ae(
        self,
        phi_global_deg: NDArray[np.floating] | float,
        eta_global_deg: NDArray[np.floating] | float,
    ) -> ComplexMatrix:
        """Return AE steering columns in global azimuth/downtilt coordinates."""
        phi, eta = np.broadcast_arrays(
            np.asarray(phi_global_deg, dtype=float),
            np.asarray(eta_global_deg, dtype=float),
        )
        phi_rad = np.deg2rad(phi.ravel())
        eta_local_rad = np.deg2rad(eta.ravel() - self.mechanical_downtilt_deg)
        u_h = np.cos(eta_local_rad) * np.sin(phi_rad)
        u_v = -np.sin(eta_local_rad)
        m = np.arange(self.vertical_aes)[:, None, None]
        n = np.arange(self.horizontal_aes)[None, :, None]
        phase = 2.0 * np.pi * (
            n * self.horizontal_spacing_lambda * u_h[None, None, :]
            + m * self.vertical_spacing_lambda * u_v[None, None, :]
        )
        return np.exp(1j * phase).reshape(self.ae_count, -1)

    def steering_txru(
        self,
        phi_global_deg: NDArray[np.floating] | float,
        eta_global_deg: NDArray[np.floating] | float,
    ) -> ComplexMatrix:
        return self.mapping.conj().T @ self.steering_ae(phi_global_deg, eta_global_deg)

    def ae_weights(self, txru_weights: NDArray[np.complexfloating]) -> ComplexMatrix:
        result = self.mapping @ np.asarray(txru_weights, dtype=np.complex128)
        norms = np.linalg.norm(result, axis=0, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero-norm beam weight")
        return result / norms

    def gain(
        self,
        ae_weights: NDArray[np.complexfloating],
        phi_global_deg: NDArray[np.floating] | float,
        eta_global_deg: NDArray[np.floating] | float,
    ) -> NDArray[np.float64]:
        steering = self.steering_ae(phi_global_deg, eta_global_deg)
        weights = np.asarray(ae_weights, dtype=np.complex128)
        if weights.ndim == 1:
            weights = weights[:, None]
        return np.abs(steering.conj().T @ weights) ** 2

