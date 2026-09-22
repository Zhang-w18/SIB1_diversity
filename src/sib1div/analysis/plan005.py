"""Freeze Plan 005's full-run SNR grid from its completed prescan."""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from sib1div.config import SimulationConfig, validate_config


def select_full_snr_grid(
    rows: list[dict[str, str]], target: float = 0.1, step_db: float = 0.5,
) -> tuple[list[float], dict[str, list[float]]]:
    """Bracket each estimated curve and return their guarded union grid."""
    by_curve: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if row.get("curve_type") != "estimated_csi":
            continue
        by_curve.setdefault(row["curve_id"], []).append(
            (float(row["snr_db"]), float(row["bler"]))
        )
    if not by_curve:
        raise ValueError("prescan contains no estimated-CSI curves")
    brackets: dict[str, list[float]] = {}
    for curve_id, values in by_curve.items():
        values.sort()
        candidates = [
            (low_snr, high_snr)
            for (low_snr, low_bler), (high_snr, high_bler)
            in zip(values[:-1], values[1:])
            if low_bler >= target and high_bler <= target
        ]
        if not candidates:
            raise ValueError(f"prescan does not bracket BLER={target:g} for {curve_id}")
        brackets[curve_id] = list(min(
            candidates, key=lambda pair: pair[1] - pair[0],
        ))
    lower = min(pair[0] for pair in brackets.values()) - step_db
    upper = max(pair[1] for pair in brackets.values()) + step_db
    count = int(round((upper - lower) / step_db))
    grid = [round(lower + index * step_db, 10) for index in range(count + 1)]
    return grid, brackets


def prepare_full_config(
    prescan_csv: str | Path,
    prescan_config: str | Path,
    output_config: str | Path,
    selection_json: str | Path,
) -> list[float]:
    with Path(prescan_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grid, brackets = select_full_snr_grid(rows)
    source = Path(prescan_config)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    full = deepcopy(data)
    full["experiment"].update({
        "id": "plan-005",
        "title": "fixed_cdl_c_statistics_100ns_asd10_full",
        "stage": "formal",
    })
    monte = full["monte_carlo"]
    monte.update({
        "snr_points_db": grid,
        "fixed_drops_per_snr": 1000,
        "target_block_errors_diagnostic": 100,
        "drop_index_start": 1000,
        "stopping_rule": "exactly_1000_common_drops_per_snr_no_early_stop",
        "purpose": "full_10pct_bler_curves_selected_from_plan005_prescan",
    })
    config = SimulationConfig(full, Path(output_config).resolve())
    validate_config(config)
    destination = Path(output_config)
    destination.write_text(
        yaml.safe_dump(full, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    selection = {
        "schema_version": 1,
        "target_bler": 0.1,
        "selection_rule": "union of adjacent 2 dB raw-BLER brackets, filled on 0.5 dB grid, with one 0.5 dB guard point on each side",
        "curve_brackets_db": brackets,
        "full_snr_points_db": grid,
        "prescan_csv": str(Path(prescan_csv).resolve()),
        "prescan_config": str(source.resolve()),
        "full_config": str(destination.resolve()),
    }
    selection_path = Path(selection_json)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return grid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prescan-csv", type=Path, required=True)
    parser.add_argument("--prescan-config", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    args = parser.parse_args(argv)
    grid = prepare_full_config(
        args.prescan_csv, args.prescan_config,
        args.output_config, args.selection_json,
    )
    print(f"Plan 005 full SNR grid: {grid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
