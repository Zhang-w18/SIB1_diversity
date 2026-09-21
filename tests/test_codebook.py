from pathlib import Path

import numpy as np

from sib1div.codebook import generate_main_codebooks
from sib1div.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "uma_7ghz_48prb.yaml"


def test_main_codebook_shapes_norms_and_stable_parent_indices():
    _, ssb, secondary = generate_main_codebooks(load_config(CONFIG))
    assert ssb.txru_weights.shape == (64, 8)
    assert secondary.txru_weights.shape == (64, 16)
    assert ssb.ae_weights.shape == (384, 8)
    assert secondary.ae_weights.shape == (384, 16)
    np.testing.assert_allclose(np.linalg.norm(ssb.ae_weights, axis=0), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(secondary.ae_weights, axis=0), 1.0, atol=1e-12)
    assert [region.parent_ssb for region in secondary.regions] == [i // 2 for i in range(16)]
    assert ssb.regions[0].phi_min_deg == -60.0
    assert ssb.regions[-1].phi_max_deg == 60.0

