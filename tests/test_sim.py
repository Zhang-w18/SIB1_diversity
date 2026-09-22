from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sib1div.config import SimulationConfig, load_config
from sib1div.sim.engine import link_mode_scaling, load_codebooks, select_ssb


ROOT = Path(__file__).parents[1]
CODEBOOK = ROOT / "outputs" / "plan-001" / "codebook_7ghz_8h1v_mainlobe" / "codebook_weights.npz"
CONFIG = ROOT / "configs" / "uma_7ghz_48prb.yaml"
WEIGHTS_SHA256 = "3142396469049e843e8018bc5596152aa70d53e46d18b406879b52cf54638259"


def test_selected_ssb_instantaneous_common_normalization_is_exact():
    # One spatial port per polarization, two candidate SSBs.
    ssb = np.array([[1.0, 1.0j]])
    response = np.array([
        [[1.0, 0.5], [2.0, -0.5], [0.5, 1.0]],
        [[0.2, 1.5], [0.7, -1.0], [1.2, 0.1]],
    ], dtype=complex)
    selected, powers = select_ssb(response, ssb)
    normalized = response / np.sqrt(powers[selected])
    _, normalized_powers = select_ssb(normalized, ssb)
    assert normalized_powers[selected] == pytest.approx(1.0, abs=1e-14)
    assert selected == int(np.argmax(powers))


def test_frozen_codebook_hash_is_enforced():
    ssb, secondary = load_codebooks(CODEBOOK, WEIGHTS_SHA256)
    assert ssb.shape[1] == 8
    assert secondary.shape[1] == 16
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_codebooks(CODEBOOK, "0" * 64)


@pytest.mark.parametrize(
    ("mode", "expected_scale"),
    [
        ("fixed_radius_ls_normalized", 0.25),
        ("fixed_radius_full_channel", 1.0),
    ],
)
def test_v4_section_2_8_link_mode_scaling_uses_g0_not_selected_ssb_power(
    mode, expected_scale,
):
    data = deepcopy(load_config(CONFIG).data)
    data["run"]["link_mode"] = mode
    data["link_normalization"] = {"reference_large_scale_power_gain_linear": 1e-10}
    config = SimulationConfig(data, CONFIG)
    scale, noise_variance = link_mode_scaling(
        config,
        large_scale_power_gain=4e-10,
        selected_ssb_power=123.0,
        snr_db=10.0,
    )
    assert scale == pytest.approx(expected_scale)
    assert noise_variance == pytest.approx(1e-11)


def test_fixed_cdl_scaling_uses_frozen_parent_ssb_reference_power():
    data = deepcopy(load_config(CONFIG).data)
    data["run"]["link_mode"] = "fixed_cdl_statistics"
    config = SimulationConfig(data, CONFIG)
    scale, noise_variance = link_mode_scaling(
        config, large_scale_power_gain=1.0,
        selected_ssb_power=2.5, snr_db=10.0,
    )
    assert scale == 1.0
    assert noise_variance == pytest.approx(0.25)
