from pathlib import Path

import json

from sib1div.config import load_config
from sib1div.runtime import collect_environment, initialize_run


CONFIG = Path(__file__).parents[1] / "configs" / "uma_7ghz_48prb.yaml"


def test_sionna_resolves_outside_legacy_tree():
    environment = collect_environment(check_tensorflow=False)
    origin = environment["packages"]["sionna"]["origin"]
    assert origin is not None
    assert "py3gpp" not in origin.lower()
    assert environment["packages"]["PyYAML"]["origin"].endswith("yaml\\__init__.py")


def test_initialize_run_writes_provenance(tmp_path):
    config = load_config(CONFIG)
    run_dir = initialize_run(config, tmp_path / "run", check_tensorflow=False)
    assert (run_dir / "config_resolved.yaml").exists()
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["python"]["version"] == "3.11.9"
    assert environment["packages"]["sionna"]["version"] == "1.0.2"
