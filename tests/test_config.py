from pathlib import Path

import pytest
import yaml

from sib1div.config import ConfigError, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "uma_7ghz_48prb.yaml"


def test_main_config_derived_grid_and_array_counts():
    config = load_config(CONFIG)
    resolved = config.resolved()
    assert config.active_subcarriers == 576
    assert config.dmrs_res == 576
    assert config.data_res == 6336
    assert config.coded_bits == 12672
    assert resolved["derived"]["total_bs_aes"] == 768
    assert resolved["derived"]["total_txru_ports"] == 128


def test_spacing_fields_cannot_be_swapped(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["bs_array"]["horizontal_spacing_lambda"] = 0.8
    raw["bs_array"]["vertical_spacing_lambda"] = 0.5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="horizontal=0.5λ"):
        load_config(path)


def test_tbs_is_checked_against_reference(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["nr"]["tbs_bits"] = 1472
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="TBS"):
        load_config(path)


def test_v4_section_2_8_modes_require_fixed_radius_and_frozen_g0(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["run"]["link_mode"] = "fixed_radius_ls_normalized"
    raw["scenario"]["ue_distance_min_m"] = 100.0
    raw["scenario"]["ue_distance_max_m"] = 100.0
    raw["link_normalization"] = {
        "reference_definition": "100 m UMa LOS, no shadow fading",
        "reference_large_scale_power_gain_linear": 1e-10,
        "reference_large_scale_power_gain_db": -100.0,
        "per_scheme_renormalization": False,
    }
    path = tmp_path / "section-2-8.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = load_config(path)
    assert config.data["run"]["link_mode"] == "fixed_radius_ls_normalized"

    raw["scenario"]["ue_distance_max_m"] = 101.0
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="fixed UE radius"):
        load_config(path)


def test_v4_section_2_8_rejects_inconsistent_g0_db(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["run"]["link_mode"] = "fixed_radius_full_channel"
    raw["scenario"]["ue_distance_min_m"] = 100.0
    raw["scenario"]["ue_distance_max_m"] = 100.0
    raw["link_normalization"] = {
        "reference_definition": "100 m UMa LOS, no shadow fading",
        "reference_large_scale_power_gain_linear": 1e-10,
        "reference_large_scale_power_gain_db": -99.0,
        "per_scheme_renormalization": False,
    }
    path = tmp_path / "bad-g0.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="same G0"):
        load_config(path)
