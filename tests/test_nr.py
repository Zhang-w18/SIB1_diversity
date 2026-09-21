from pathlib import Path

import numpy as np

from sib1div.config import load_config
from sib1div.nr import SIB1Codec, map_pdsch, qpsk_modulate


CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "plan-001.yaml"


def test_sib1_noiseless_codec_roundtrip():
    config = load_config(CONFIG)
    codec = SIB1Codec(config)
    payload = np.random.default_rng(11).integers(0, 2, config.tbs_bits, dtype=np.uint8)
    coded = codec.encode(payload)
    result = codec.decode((1 - 2 * coded.astype(float)) * 20.0)
    assert result.crc_ok
    np.testing.assert_array_equal(result.payload, payload)
    assert codec.info["CRC"] == "16"
    assert codec.info["BGN"] == 2
    assert codec.info["N"] == 8000


def test_resource_grid_counts_and_dmrs_positions():
    config = load_config(CONFIG)
    data = qpsk_modulate(np.zeros(config.coded_bits, dtype=np.uint8))
    grid = map_pdsch(config, data)
    assert grid.symbols.shape == (14, 576)
    assert np.count_nonzero(grid.dmrs_mask) == 576
    assert np.count_nonzero(grid.data_mask) == 6336
    assert np.array_equal(np.flatnonzero(np.any(grid.dmrs_mask, axis=1)), [2, 11])
    assert np.array_equal(np.flatnonzero(grid.dmrs_mask[2]), np.arange(0, 576, 2))

