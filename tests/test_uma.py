from pathlib import Path

import numpy as np

from sib1div.channel import UMaChannel, sample_ue_position
from sib1div.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-001.yaml"


def test_ue_positions_are_deterministic_area_samples_in_controlled_range():
    config = load_config(CONFIG)
    samples = [sample_ue_position(config, i) for i in range(100)]
    radii = np.array([sample[1] for sample in samples])
    assert np.all((radii >= 75.0) & (radii <= 115.0))
    first = sample_ue_position(config, 7)
    repeat = sample_ue_position(config, 7)
    np.testing.assert_array_equal(first[0], repeat[0])


def test_sionna_uma_drop_is_projected_to_txru_domain():
    config = load_config(CONFIG)
    drop = UMaChannel(config).generate(0)
    assert drop.path_coefficients.shape[:2] == (4, 128)
    assert drop.path_coefficients.shape[2] == drop.path_delays_s.size
    assert drop.path_spatial_covariances.shape == (4, drop.path_delays_s.size, 128, 128)
    np.testing.assert_allclose(
        drop.path_spatial_covariances[0, 0],
        drop.path_spatial_covariances[0, 0].conj().T,
        atol=1e-15,
    )
    assert np.min(np.linalg.eigvalsh(drop.path_spatial_covariances[0, 0])) > -1e-20
    response = drop.frequency_response(24, 30e3)
    assert response.shape == (4, 24, 128)
    assert np.isfinite(response).all()
    assert 75.0 <= drop.radius_m <= 115.0
    assert 0.0 < drop.large_scale_amplitude_gain < 1.0
    assert drop.large_scale_power_gain == drop.large_scale_amplitude_gain**2
