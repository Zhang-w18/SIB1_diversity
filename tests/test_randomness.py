from pathlib import Path

import numpy as np

from sib1div.config import load_config
from sib1div.sim import common_samples


CONFIG = Path(__file__).parents[1] / "configs" / "uma_7ghz_48prb.yaml"


def test_common_samples_are_repeatable_and_indexed():
    config = load_config(CONFIG)
    first = common_samples(config, snr_index=2, drop_index=7, noise_shape=(4, 8))
    repeat = common_samples(config, snr_index=2, drop_index=7, noise_shape=(4, 8))
    other = common_samples(config, snr_index=2, drop_index=8, noise_shape=(4, 8))
    np.testing.assert_array_equal(first.transport_block, repeat.transport_block)
    np.testing.assert_array_equal(first.unit_variance_noise, repeat.unit_variance_noise)
    assert not np.array_equal(first.transport_block, other.transport_block)
    assert first.transport_block.shape == (1480,)
    assert first.unit_variance_noise.dtype == np.complex64

