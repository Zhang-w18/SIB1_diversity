import numpy as np

from sib1div.codebook import ArrayGeometry, build_txru_mapping


def test_contiguous_6x1_mapping_is_column_normalized():
    mapping = build_txru_mapping(24, 16, 4, 16)
    assert mapping.shape == (384, 64)
    np.testing.assert_allclose(mapping.conj().T @ mapping, np.eye(64), atol=1e-14)
    assert np.all(np.count_nonzero(mapping, axis=0) == 6)


def test_txru_projection_matches_direct_ae_response():
    geometry = ArrayGeometry(6, 4, 2, 4, 0.5, 0.8, 12.0)
    rng = np.random.default_rng(20260913)
    txru_weight = rng.normal(size=8) + 1j * rng.normal(size=8)
    ae_weight = geometry.mapping @ txru_weight
    steering_ae = geometry.steering_ae(
        np.array([-37.0, 3.0, 42.0]), np.array([12.0, 22.0, 31.0])
    )
    steering_txru = geometry.mapping.conj().T @ steering_ae
    direct = steering_ae.conj().T @ ae_weight
    projected = steering_txru.conj().T @ txru_weight
    np.testing.assert_allclose(direct, projected, rtol=1e-13, atol=1e-13)


def test_mechanical_tilt_is_applied_once():
    geometry = ArrayGeometry(6, 4, 2, 4, 0.5, 0.8, 12.0)
    at_global_tilt = geometry.steering_ae(0.0, 12.0)
    np.testing.assert_allclose(at_global_tilt, np.ones_like(at_global_tilt), atol=1e-14)

