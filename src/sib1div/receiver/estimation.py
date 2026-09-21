"""DMRS LS/LMMSE channel estimation and four-branch MRC."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def ls_at_pilots(
    received: NDArray[np.complexfloating],
    transmitted_grid: NDArray[np.complexfloating],
    dmrs_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Average repeated DMRS LS estimates; received is [symbol,sc,rx]."""
    y = np.asarray(received)
    x = np.asarray(transmitted_grid)
    pilot_k = np.flatnonzero(np.any(dmrs_mask, axis=0))
    estimates = np.empty((y.shape[2], pilot_k.size), dtype=np.complex128)
    for i, k in enumerate(pilot_k):
        symbols = np.flatnonzero(dmrs_mask[:, k])
        estimates[:, i] = np.mean(y[symbols, k, :] / x[symbols, k, None], axis=0)
    return estimates, pilot_k


def linear_ls_interpolate(
    pilot_estimates: NDArray[np.complexfloating],
    pilot_indices: NDArray[np.integer],
    n_subcarriers: int,
) -> NDArray[np.complex128]:
    target = np.arange(n_subcarriers)
    result = np.empty((pilot_estimates.shape[0], n_subcarriers), dtype=np.complex128)
    for rx, values in enumerate(pilot_estimates):
        real = np.interp(target, pilot_indices, values.real)
        imag = np.interp(target, pilot_indices, values.imag)
        result[rx] = real + 1j * imag
    return result


def frequency_covariance(
    delays_s: NDArray[np.floating],
    path_powers: NDArray[np.floating],
    frequencies_hz: NDArray[np.floating],
) -> NDArray[np.complex128]:
    delays = np.asarray(delays_s, dtype=float).reshape(-1)
    powers = np.asarray(path_powers, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if delays.size != powers.size or np.any(powers < 0):
        raise ValueError("delays and non-negative path powers must have equal length")
    response = np.exp(-1j * 2 * np.pi * frequencies[:, None] * delays[None, :])
    covariance = (response * powers[None, :]) @ response.conj().T
    return 0.5 * (covariance + covariance.conj().T)


def projected_path_powers(
    path_spatial_covariances: NDArray[np.complexfloating],
    weights: NDArray[np.complexfloating],
) -> NDArray[np.float64]:
    """Project long-term TX spatial path covariances through fixed beams.

    ``path_spatial_covariances`` has shape ``[rx,path,tx,tx]`` and follows
    the row-channel convention used by the simulator, i.e. ``h_eff=h@w``.
    ``weights`` is ``[tx,beam]``.  The result is ``[rx,path,beam]``.
    """
    cov = np.asarray(path_spatial_covariances, dtype=np.complex128)
    beams = np.asarray(weights, dtype=np.complex128)
    if cov.ndim != 4 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError("path spatial covariance must have shape [rx,path,tx,tx]")
    if beams.ndim != 2 or beams.shape[0] != cov.shape[-1]:
        raise ValueError("beam weights must have shape [tx,beam]")
    powers = np.einsum("tm,rptu,um->rpm", beams, cov, beams.conj(), optimize=True).real
    return np.maximum(powers, 0.0)


def beam_domain_path_covariances(
    path_spatial_covariances: NDArray[np.complexfloating],
    weights: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Return per-path joint covariance between fixed beam branches.

    The returned tensor has shape ``[rx,path,beam,beam]`` and retains the
    off-diagonal cross-beam terms required by ideal Beam CDD LMMSE.
    """
    cov = np.asarray(path_spatial_covariances, dtype=np.complex128)
    beams = np.asarray(weights, dtype=np.complex128)
    if cov.ndim != 4 or cov.shape[-1] != cov.shape[-2]:
        raise ValueError("path spatial covariance must have shape [rx,path,tx,tx]")
    if beams.ndim != 2 or beams.shape[0] != cov.shape[-1]:
        raise ValueError("beam weights must have shape [tx,beam]")
    result = np.einsum("tm,rptu,un->rpmn", beams, cov, beams.conj(), optimize=True)
    return 0.5 * (result + result.swapaxes(-1, -2).conj())


def equivalent_frequency_covariance(
    path_delays_s: NDArray[np.floating],
    path_spatial_covariances: NDArray[np.complexfloating],
    frequencies_hz: NDArray[np.floating],
    frequency_weights: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Exact scheme-specific frequency covariance from long-term path stats.

    This implements the general case where the unit-norm precoder may vary on
    every occupied subcarrier.  Consequently it also includes Beam CDD branch
    correlation and its known per-subcarrier power normalization.
    """
    delays = np.asarray(path_delays_s, dtype=float).reshape(-1)
    cov = np.asarray(path_spatial_covariances, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    weights = np.asarray(frequency_weights, dtype=np.complex128)
    if cov.ndim != 3 or cov.shape[0] != delays.size or cov.shape[1] != cov.shape[2]:
        raise ValueError("one-RX path covariance must have shape [path,tx,tx]")
    if weights.shape != (cov.shape[-1], frequencies.size):
        raise ValueError("frequency weights must have shape [tx,frequency]")
    phase = np.exp(-1j * 2 * np.pi * frequencies[:, None] * delays[None, :])
    result = np.zeros((frequencies.size, frequencies.size), dtype=np.complex128)
    for path in range(delays.size):
        coupling = weights.T @ cov[path] @ weights.conj()
        result += coupling * (phase[:, path, None] * phase[:, path, None].conj().T)
    return 0.5 * (result + result.conj().T)


def independent_cdd_frequency_covariance(
    path_delays_s: NDArray[np.floating],
    branch_path_powers: NDArray[np.floating],
    frequencies_hz: NDArray[np.floating],
    artificial_delays_s: NDArray[np.floating],
    raw_precoder_power: NDArray[np.floating] | None = None,
) -> NDArray[np.complex128]:
    """CDD covariance when branches are assumed mutually uncorrelated.

    ``branch_path_powers`` is ``[branch,path]``.  Supplying the raw CDD power
    includes the known per-subcarrier normalization used by the transmitter.
    """
    delays = np.asarray(path_delays_s, dtype=float).reshape(-1)
    powers = np.asarray(branch_path_powers, dtype=float)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    artificial = np.asarray(artificial_delays_s, dtype=float).reshape(-1)
    if powers.shape != (artificial.size, delays.size) or np.any(powers < 0):
        raise ValueError("branch path powers must have shape [branch,path] and be non-negative")
    shifted = delays[None, :] + artificial[:, None]
    response = np.exp(-1j * 2 * np.pi * frequencies[:, None, None] * shifted[None, :, :])
    # 1/K is the power of each CDD branch before the optional common
    # per-subcarrier normalization.
    covariance = np.einsum(
        "kmp,mp,rmp->kr", response, powers / artificial.size,
        response.conj(), optimize=True,
    )
    if raw_precoder_power is not None:
        raw = np.asarray(raw_precoder_power, dtype=float).reshape(-1)
        if raw.size != frequencies.size or np.any(raw <= 0):
            raise ValueError("raw CDD power must be positive on every frequency")
        scale = 1.0 / np.sqrt(raw)
        covariance *= scale[:, None] * scale[None, :]
    return 0.5 * (covariance + covariance.conj().T)


def joint_cdd_frequency_covariance(
    path_delays_s: NDArray[np.floating],
    beam_path_covariances: NDArray[np.complexfloating],
    frequencies_hz: NDArray[np.floating],
    branch_coefficients: NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Ideal CDD covariance including all beam-domain cross terms.

    ``beam_path_covariances`` is ``[path,beam,beam]`` and
    ``branch_coefficients`` is the exact transmitter coefficient of every
    beam on every occupied subcarrier, including CDD phase, ``1/sqrt(K)``,
    and the known per-subcarrier normalization.
    """
    delays = np.asarray(path_delays_s, dtype=float).reshape(-1)
    beam_cov = np.asarray(beam_path_covariances, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    coefficients = np.asarray(branch_coefficients, dtype=np.complex128)
    if beam_cov.ndim != 3 or beam_cov.shape[0] != delays.size or beam_cov.shape[1] != beam_cov.shape[2]:
        raise ValueError("beam path covariance must have shape [path,beam,beam]")
    if coefficients.shape != (beam_cov.shape[1], frequencies.size):
        raise ValueError("branch coefficients must have shape [beam,frequency]")
    phase = np.exp(-1j * 2 * np.pi * frequencies[:, None] * delays[None, :])
    result = np.zeros((frequencies.size, frequencies.size), dtype=np.complex128)
    for path in range(delays.size):
        coupling = coefficients.T @ beam_cov[path] @ coefficients.conj()
        result += coupling * (phase[:, path, None] * phase[:, path, None].conj().T)
    return 0.5 * (result + result.conj().T)


def lmmse_interpolate(
    pilot_estimates: NDArray[np.complexfloating],
    pilot_indices: NDArray[np.integer],
    covariance: NDArray[np.complexfloating],
    noise_variance: float,
    pilot_repetitions: int = 2,
) -> NDArray[np.complex128]:
    pilot_indices = np.asarray(pilot_indices, dtype=int)
    cov = np.asarray(covariance, dtype=np.complex128)
    r_pp = cov[np.ix_(pilot_indices, pilot_indices)]
    regularized = r_pp + (noise_variance / pilot_repetitions) * np.eye(pilot_indices.size)
    # Solve once for all receive branches.
    # A valid PDP can yield a low-rank covariance (e.g. a flat channel).
    # The Hermitian pseudoinverse gives the zero-noise LMMSE limit without
    # introducing an arbitrary diagonal floor.
    coefficients = np.linalg.pinv(regularized, hermitian=True) @ np.asarray(pilot_estimates).T
    return (cov[:, pilot_indices] @ coefficients).T


def windowed_lmmse_interpolate(
    pilot_estimates: NDArray[np.complexfloating],
    pilot_indices: NDArray[np.integer],
    covariance: NDArray[np.complexfloating],
    noise_variance: float,
    window_subcarriers: int,
    pilot_repetitions: int = 2,
) -> NDArray[np.complex128]:
    """Apply LMMSE independently without crossing precoder boundaries."""
    cov = np.asarray(covariance)
    n_sc = cov.shape[0]
    if n_sc % window_subcarriers:
        raise ValueError("LMMSE window must divide the active allocation")
    pilot_indices = np.asarray(pilot_indices, dtype=int)
    output = np.empty((pilot_estimates.shape[0], n_sc), dtype=np.complex128)
    for start in range(0, n_sc, window_subcarriers):
        stop = start + window_subcarriers
        selected = (pilot_indices >= start) & (pilot_indices < stop)
        local_pilots = pilot_indices[selected] - start
        output[:, start:stop] = lmmse_interpolate(
            np.asarray(pilot_estimates)[:, selected],
            local_pilots,
            cov[start:stop, start:stop],
            noise_variance,
            pilot_repetitions,
        )
    return output


def mrc_equalize(
    received_data: NDArray[np.complexfloating],
    channel_estimates: NDArray[np.complexfloating],
    noise_variance: float,
) -> tuple[NDArray[np.complex128], NDArray[np.float64]]:
    """Equalize [RE,rx] observations and return symbols plus effective variance."""
    y = np.asarray(received_data)
    h = np.asarray(channel_estimates)
    denominator = np.maximum(np.sum(np.abs(h) ** 2, axis=1), 1e-15)
    symbols = np.sum(h.conj() * y, axis=1) / denominator
    return symbols, np.asarray(noise_variance / denominator, dtype=float)


def nmse(estimate: NDArray[np.complexfloating], truth: NDArray[np.complexfloating]) -> float:
    estimate = np.asarray(estimate)
    truth = np.asarray(truth)
    return float(np.sum(np.abs(estimate - truth) ** 2) / np.maximum(np.sum(np.abs(truth) ** 2), 1e-30))
