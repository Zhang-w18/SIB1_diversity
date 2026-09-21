"""Reproducible run initialization and environment provenance."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import SimulationConfig


def seed_everything(seed: int, include_tensorflow: bool = True) -> None:
    """Seed Python, NumPy, and optionally TensorFlow from one master seed."""
    random.seed(seed)
    np.random.seed(seed)
    if include_tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)


def _distribution(distribution: str, module: str | None = None) -> dict[str, str | None]:
    spec = importlib.util.find_spec(module or distribution)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {"version": version, "origin": None if spec is None else spec.origin}


def collect_environment(check_tensorflow: bool = True) -> dict[str, Any]:
    packages = {
        "numpy": _distribution("numpy"),
        "PyYAML": _distribution("PyYAML", "yaml"),
        "tensorflow": _distribution("tensorflow"),
        "sionna": _distribution("sionna"),
    }
    sionna_origin = packages["sionna"]["origin"]
    if sionna_origin and "py3gpp" in sionna_origin.lower():
        raise RuntimeError(f"refusing vendored Sionna import: {sionna_origin}")
    gpu_devices: list[str] | None = None
    if check_tensorflow:
        import tensorflow as tf
        gpu_devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "platform": platform.platform(),
        "packages": packages,
        "tensorflow_gpu_devices": gpu_devices,
    }


def initialize_run(
    config: SimulationConfig,
    output: str | Path,
    *,
    overwrite: bool = False,
    check_tensorflow: bool = True,
) -> Path:
    run_dir = Path(output).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    targets = [run_dir / "config_resolved.yaml", run_dir / "environment.json", run_dir / "run.log"]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"run metadata already exists: {existing[0]}")
    environment = collect_environment(check_tensorflow=check_tensorflow)
    with targets[0].open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config.resolved(), stream, sort_keys=False, allow_unicode=True)
    with targets[1].open("w", encoding="utf-8") as stream:
        json.dump(environment, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    targets[2].write_text(
        f"configuration validated: {config.source}\nmaster seed: {config.seed}\n",
        encoding="utf-8",
    )
    return run_dir
