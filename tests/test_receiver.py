import numpy as np

from sib1div.receiver import (
    beam_domain_path_covariances,
    equivalent_frequency_covariance,
    frequency_covariance,
    independent_cdd_frequency_covariance,
    joint_cdd_frequency_covariance,
    linear_ls_interpolate,
    lmmse_interpolate,
    ls_at_pilots,
    mrc_equalize,
    nmse,
    projected_path_powers,
    windowed_lmmse_interpolate,
)


def test_flat_noiseless_ls_lmmse_and_mrc_are_exact():
    n_sc = 24
    x = np.zeros((14, n_sc), complex)
    mask = np.zeros_like(x, dtype=bool)
    mask[[2, 11], :] = np.tile(np.arange(n_sc) % 2 == 0, (2, 1))
    x[mask] = 1.0
    h = np.array([1 + 0.2j, -0.3 + 0.7j, 0.8 - 0.1j, -0.5 - 0.4j])
    y = x[:, :, None] * h[None, None, :]
    pilots, indices = ls_at_pilots(y, x, mask)
    ls = linear_ls_interpolate(pilots, indices, n_sc)
    truth = np.tile(h[:, None], (1, n_sc))
    np.testing.assert_allclose(ls, truth, atol=1e-14)
    cov = frequency_covariance(np.array([0.0]), np.array([1.0]), np.arange(n_sc) * 30e3)
    lmmse = lmmse_interpolate(pilots, indices, cov, 0.0)
    np.testing.assert_allclose(lmmse, truth, atol=1e-12)
    symbols = np.array([1 + 1j, -1 + 1j]) / np.sqrt(2)
    observations = symbols[:, None] * h[None, :]
    equalized, variance = mrc_equalize(observations, np.tile(h, (2, 1)), 0.1)
    np.testing.assert_allclose(equalized, symbols, atol=1e-14)
    assert np.all(variance > 0)
    assert nmse(ls, truth) < 1e-28


def test_frequency_covariance_is_hermitian_psd():
    frequencies = np.arange(48) * 30e3
    cov = frequency_covariance(np.array([0.0, 0.3e-6, 1.2e-6]), np.array([0.7, 0.2, 0.1]), frequencies)
    np.testing.assert_allclose(cov, cov.conj().T, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(cov)) > -1e-11


def test_beam_projection_and_joint_cdd_match_direct_effective_covariance():
    rng = np.random.default_rng(7)
    modes = rng.normal(size=(2, 3, 4)) + 1j * rng.normal(size=(2, 3, 4))
    path_cov = np.einsum("rpm,rpn->rpmn", modes, modes.conj())
    branches = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
    frequencies = np.arange(9) * 30e3
    delays = np.array([0.0, 0.7e-6, 1.9e-6])
    coefficients = rng.normal(size=(2, 9)) + 1j * rng.normal(size=(2, 9))

    powers = projected_path_powers(path_cov, branches)
    joint = beam_domain_path_covariances(path_cov, branches)
    np.testing.assert_allclose(np.diagonal(joint, axis1=-2, axis2=-1).real, powers)
    for rx in range(2):
        direct = equivalent_frequency_covariance(
            delays, path_cov[rx], frequencies, branches @ coefficients,
        )
        factored = joint_cdd_frequency_covariance(
            delays, joint[rx], frequencies, coefficients,
        )
        np.testing.assert_allclose(factored, direct, atol=1e-10)


def test_independent_cdd_covariance_includes_known_power_normalization():
    delays = np.array([0.0, 0.8e-6])
    powers = np.array([[0.7, 0.3], [0.2, 0.8]])
    frequencies = np.arange(12) * 30e3
    artificial = np.array([0.0, 1.0 / (12 * 30e3)])
    raw = np.linspace(0.8, 1.2, frequencies.size)
    unnormalized = independent_cdd_frequency_covariance(
        delays, powers, frequencies, artificial,
    )
    normalized = independent_cdd_frequency_covariance(
        delays, powers, frequencies, artificial, raw,
    )
    scale = 1.0 / np.sqrt(raw)
    np.testing.assert_allclose(normalized, unnormalized * scale[:, None] * scale[None, :])


def test_windowed_lmmse_does_not_use_pilots_from_adjacent_prg():
    n_sc = 8
    pilot_indices = np.array([0, 2, 4, 6])
    pilots = np.array([[1.0, 1.0, 9.0, 9.0]], dtype=complex)
    covariance = np.ones((n_sc, n_sc), dtype=complex)
    estimate = windowed_lmmse_interpolate(
        pilots, pilot_indices, covariance, 0.0, window_subcarriers=4,
        pilot_repetitions=1,
    )
    np.testing.assert_allclose(estimate[0, :4], 1.0)
    np.testing.assert_allclose(estimate[0, 4:], 9.0)
