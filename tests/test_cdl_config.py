from pathlib import Path

import pytest
import yaml

from sib1div.config import ConfigError, load_config


CONFIG = Path(__file__).parents[1] / "configs" / "fixed_cdl_statistics_example.yaml"


def test_fixed_cdl_example_is_valid_and_resolved():
    config = load_config(CONFIG)
    assert config.data["run"]["link_mode"] == "fixed_cdl_statistics"
    assert config.data["fixed_cdl_statistics"]["profile"] == "C"
    assert config.active_subcarriers == 576


def test_fixed_cdl_requires_disjoint_statistics_and_bler_seeds(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["fixed_cdl_statistics"]["realization_seed"] = raw["fixed_cdl_statistics"]["statistics_seed"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="different seeds"):
        load_config(path)


def test_fixed_cdl_time_evolution_requires_physical_sample_spacing(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["fixed_cdl_statistics"]["time_steps_per_slot"] = 14
    raw["fixed_cdl_statistics"]["time_step_s"] = 0.0
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="time_step_s"):
        load_config(path)


def test_fixed_cdl_target_spread_and_scale_are_mutually_exclusive(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    raw["fixed_cdl_statistics"]["angle_transform"]["target_asd_deg"] = 10.0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(path)
