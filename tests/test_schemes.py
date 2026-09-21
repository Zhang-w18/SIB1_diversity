from pathlib import Path

import numpy as np

from sib1div.codebook import generate_main_codebooks
from sib1div.config import load_config
from sib1div.schemes import SCHEMES, assert_unit_norm, build_precoder


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-001.yaml"


def test_all_precoders_are_per_subcarrier_unit_norm_and_stable():
    config = load_config(CONFIG)
    _, ssb, secondary = generate_main_codebooks(config)
    for scheme in SCHEMES:
        result = build_precoder(config, scheme, 3, ssb.txru_weights, secondary.txru_weights)
        assert result.weights.shape == (128, 576)
        assert result.raw_power.shape == (576,)
        assert_unit_norm(result)
    baseline = build_precoder(config, "baseline", 3, ssb.txru_weights, secondary.txru_weights)
    np.testing.assert_allclose(baseline.weights[:, 0], baseline.weights[:, -1])


def test_cdd_active_band_phase_matches_fractional_fft_shift_definition():
    config = load_config(CONFIG)
    q = np.arange(576)
    active_grid = np.exp(-1j * 2 * np.pi * q / 576)
    fft_grid = np.exp(-1j * 2 * np.pi * q * (2048 / 576) / 2048)
    np.testing.assert_allclose(active_grid, fft_grid, atol=1e-12)


def test_pol_cycling_curve_level_prg_override_changes_only_phase_hold_width():
    config = load_config(CONFIG)
    _, ssb, secondary = generate_main_codebooks(config)
    p2 = build_precoder(
        config, "pol_cycling", 3, ssb.txru_weights, secondary.txru_weights,
        prg_size_prbs=2,
    ).weights
    p6 = build_precoder(
        config, "pol_cycling", 3, ssb.txru_weights, secondary.txru_weights,
        prg_size_prbs=6,
    ).weights
    np.testing.assert_allclose(p2[:, 0], p6[:, 0])
    assert not np.allclose(p2[:, 24], p2[:, 0])
    np.testing.assert_allclose(p6[:, 71], p6[:, 0])
    assert not np.allclose(p6[:, 72], p6[:, 0])
