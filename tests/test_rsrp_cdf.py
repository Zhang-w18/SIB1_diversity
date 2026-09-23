import numpy as np

from sib1div.analysis.rsrp_cdf import effective_rx_powers


def test_effective_rx_powers_average_frequency_energy_per_branch():
    response = np.zeros((2, 3, 2), dtype=complex)
    response[0, :, 0] = [1, 2, 3]
    response[1, :, 1] = [2, 2, 2]
    weights = np.zeros((2, 3), dtype=complex)
    weights[0] = 1
    weights[1] = 0.5
    powers = effective_rx_powers(response, weights)
    np.testing.assert_allclose(powers, [14 / 3, 1.0])
