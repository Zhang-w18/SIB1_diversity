"""Common random samples shared by every transmission scheme."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sib1div.config import SimulationConfig


@dataclass(frozen=True)
class CommonSamples:
    """Scheme-independent data for one `(SNR index, drop index)` pair."""

    transport_block: NDArray[np.uint8]
    unit_variance_noise: NDArray[np.complex64]


def common_samples(
    config: SimulationConfig,
    snr_index: int,
    drop_index: int,
    noise_shape: tuple[int, ...],
) -> CommonSamples:
    if snr_index < 0 or drop_index < 0:
        raise ValueError("SNR and drop indices must be non-negative")
    seed = np.random.SeedSequence([config.seed, snr_index, drop_index])
    tb_seed, noise_seed = seed.spawn(2)
    tb_rng = np.random.default_rng(tb_seed)
    noise_rng = np.random.default_rng(noise_seed)
    tb = tb_rng.integers(0, 2, size=config.tbs_bits, dtype=np.uint8)
    scale = np.float32(1.0 / np.sqrt(2.0))
    noise = scale * (
        noise_rng.standard_normal(noise_shape, dtype=np.float32)
        + 1j * noise_rng.standard_normal(noise_shape, dtype=np.float32)
    )
    return CommonSamples(tb, noise.astype(np.complex64, copy=False))

