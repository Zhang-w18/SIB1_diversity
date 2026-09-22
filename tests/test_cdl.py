from copy import deepcopy
from pathlib import Path

import numpy as np

from sib1div.channel import FixedCDLChannel
from sib1div.config import SimulationConfig, load_config
from sib1div.sim.engine import load_codebooks


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "fixed_cdl_statistics_example.yaml"
CODEBOOK = ROOT / "outputs" / "plan-001" / "codebook_7ghz_8h1v_mainlobe" / "codebook_weights.npz"


def test_fixed_cdl_freezes_statistics_and_resamples_small_scale_phase():
    original = load_config(CONFIG)
    data = deepcopy(original.data)
    data["fixed_cdl_statistics"]["covariance_realizations"] = 2
    config = SimulationConfig(data, CONFIG)
    ssb, _ = load_codebooks(CODEBOOK)
    channel = FixedCDLChannel(config, ssb)
    first = channel.generate(3)
    repeated = channel.generate(3)
    other = channel.generate(4)

    assert first.path_coefficients.shape == (4, 128, 24)
    assert np.array_equal(first.path_delays_s, other.path_delays_s)
    assert np.array_equal(first.path_spatial_covariances, other.path_spatial_covariances)
    assert np.array_equal(first.path_coefficients, repeated.path_coefficients)
    assert not np.array_equal(first.path_coefficients, other.path_coefficients)
    assert first.fixed_selected_ssb == channel.selected_ssb
    assert first.reference_receive_power == channel.reference_receive_power > 0.0
    assert np.min(np.linalg.eigvalsh(channel.transmit_covariance)) > -1e-9


def test_fixed_cdl_power_weighted_angle_targets_are_exact():
    original = load_config(CONFIG)
    data = deepcopy(original.data)
    data["fixed_cdl_statistics"]["covariance_realizations"] = 1
    transform = data["fixed_cdl_statistics"]["angle_transform"]
    transform.pop("aod_scale", None)
    transform["mean_aod_deg"] = 6.25
    transform["target_asd_deg"] = 10.0
    transform["mean_zod_deg"] = 104.4733873545822
    config = SimulationConfig(data, CONFIG)
    ssb, _ = load_codebooks(CODEBOOK)
    channel = FixedCDLChannel(config, ssb)

    assert np.isclose(channel.angle_statistics["aod"]["transformed_mean_deg"], 6.25)
    assert np.isclose(channel.angle_statistics["aod"]["transformed_rms_spread_deg"], 10.0)
    assert np.isclose(channel.angle_statistics["aod"]["sionna_internal_transformed_mean_deg"], -6.25)
    assert np.isclose(channel.angle_statistics["zod"]["transformed_mean_deg"], 104.4733873545822)
