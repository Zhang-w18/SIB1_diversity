"""Prepare and combine the Plan 005 one-percent-BLER extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import yaml

from sib1div.analysis.merge_fixed_runs import merge_runs
from sib1div.config import SimulationConfig, validate_config


CURVES = ("B-SSB", "P2-SSB", "P6-SSB", "BC-SSB", "CDD-SSB")


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_prescan_config(base_config: str | Path, output_config: str | Path) -> Path:
    """Create the frozen 0.5--4 dB, drops 100--599 prescan config."""
    source = Path(base_config).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["experiment"].update({
        "id": "plan-005-tail-prescan",
        "title": "fixed_cdl_c_plan005_one_percent_prescan",
        "stage": "formal_prescan",
        "created_date": "2026-09-22",
    })
    data["monte_carlo"].update({
        "snr_strategy": "fixed_common_drops",
        "snr_points_db": [0.5 * index for index in range(1, 9)],
        "fixed_drops_per_snr": 500,
        "drop_index_start": 100,
        "progress_interval_drops": 10,
        "stopping_rule": "exactly_500_common_drops_per_snr_no_early_stop",
        "purpose": "prescan_bracket_all_plan005_curves_around_1pct_bler",
    })
    config = SimulationConfig(data, source)
    validate_config(config)
    destination = Path(output_config).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return destination


def write_fast_prescan_config(base_config: str | Path, output_config: str | Path) -> Path:
    """Create the reduced prescan that complements Plan 005 run-001."""
    source = Path(base_config).resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["experiment"].update({
        "id": "plan-005-tail-fast-prescan",
        "title": "fixed_cdl_c_plan005_one_percent_fast_prescan",
        "stage": "formal_prescan",
        "created_date": "2026-09-22",
    })
    data["monte_carlo"].update({
        "snr_strategy": "fixed_common_drops",
        "snr_points_db": [1.0, 1.5, 2.5, 3.0, 3.5, 4.5],
        "fixed_drops_per_snr": 100,
        "drop_index_start": 600,
        "progress_interval_drops": 10,
        "stopping_rule": "exactly_100_common_drops_per_snr_no_early_stop",
        "purpose": "complement_plan005_run001_coarse_prescan_for_1pct_bler",
    })
    validate_config(SimulationConfig(data, source))
    destination = Path(output_config).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return destination


def _curve_points(*row_sets: list[dict[str, str]]) -> dict[str, list[tuple[float, float, int, int]]]:
    indexed: dict[str, dict[float, tuple[float, float, int, int]]] = {curve: {} for curve in CURVES}
    for rows in row_sets:
        for row in rows:
            curve = row.get("curve_id")
            if curve in indexed and row.get("curve_type", "estimated_csi") == "estimated_csi":
                snr = float(row["snr_db"])
                indexed[curve][snr] = (
                    snr, float(row["bler"]),
                    int(row["block_errors"]), int(row["blocks"]),
                )
    return {curve: sorted(values.values()) for curve, values in indexed.items()}


def select_tail_plan(
    prescan_rows: list[dict[str, str]], existing_rows: list[dict[str, str]],
    *, target: float = 0.01, step_db: float = 0.5,
) -> tuple[list[float], dict[str, list[float]], dict[float, int]]:
    """Select guarded SNR grid and per-point total drops from raw prescan counts."""
    points = _curve_points(existing_rows, prescan_rows)
    brackets: dict[str, list[float]] = {}
    for curve, values in points.items():
        candidates = [
            (left[0], right[0]) for left, right in zip(values, values[1:])
            if left[1] >= target and right[1] <= target
        ]
        if not candidates:
            raise ValueError(f"prescan does not bracket BLER={target:g} for {curve}")
        # Sparse prescan counts can fluctuate after a curve has first crossed
        # the target (for example 0/100 followed by 1/100).  The physically
        # relevant threshold bracket is the first low-to-high-SNR crossing,
        # not the last statistical re-crossing in the far tail.
        brackets[curve] = list(candidates[0])
    low = min(value[0] for value in brackets.values()) - step_db
    high = max(value[1] for value in brackets.values()) + step_db
    grid = [round(low + index * step_db, 10) for index in range(int(round((high-low)/step_db))+1)]

    prescan_by_key = {
        (row["curve_id"], float(row["snr_db"])): (int(row["block_errors"]), int(row["blocks"]))
        for row in prescan_rows if row.get("curve_id") in CURVES
    }
    existing_by_key = {
        (row["curve_id"], float(row["snr_db"])): (int(row["block_errors"]), int(row["blocks"]))
        for row in existing_rows if row.get("curve_id") in CURVES
    }
    totals: dict[float, int] = {}
    for snr in grid:
        estimates = []
        for curve in CURVES:
            sample = prescan_by_key.get((curve, snr), existing_by_key.get((curve, snr)))
            if sample is None:
                raise ValueError(f"no prescan or existing count for {curve} at {snr:g} dB")
            errors, blocks = sample
            estimates.append((errors + 0.5) / (blocks + 1.0))
        requested = min(20000.0, max(2000.0, 100.0 / min(estimates)))
        totals[snr] = int(1000 * math.ceil(requested / 1000.0))
    return grid, brackets, totals


def prepare_tail_configs(
    prescan_csv: str | Path, existing_csv: str | Path, base_config: str | Path,
    output_dir: str | Path, selection_json: str | Path,
    coarse_prescan_csv: str | Path | None = None,
) -> list[Path]:
    prescan_rows, existing_rows = _read_rows(prescan_csv), _read_rows(existing_csv)
    if coarse_prescan_csv is not None:
        prescan_rows = _read_rows(coarse_prescan_csv) + prescan_rows
    grid, brackets, totals = select_tail_plan(prescan_rows, existing_rows)
    base_path = Path(base_config).resolve()
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    existing_snrs = {float(row["snr_db"]) for row in existing_rows if row.get("curve_id") in CURVES}
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    entries, paths = [], []
    for snr in grid:
        total = totals[snr]
        prior = 1000 if snr in existing_snrs else 0
        additional = total - prior
        if additional <= 0:
            continue
        data = deepcopy(base)
        token = f"{'m' if snr < 0 else 'p'}{abs(snr):.1f}".replace(".", "p")
        data["experiment"].update({
            "id": f"plan-005-tail-{token}", "title": f"plan005_tail_{snr:g}db",
            "stage": "formal", "created_date": "2026-09-22",
        })
        data["monte_carlo"].update({
            "snr_strategy": "fixed_common_drops", "snr_points_db": [snr],
            "fixed_drops_per_snr": additional, "drop_index_start": 2000,
            "progress_interval_drops": 10,
            "stopping_rule": f"exactly_{additional}_additional_common_drops_no_early_stop",
            "purpose": "plan005_extend_all_curves_through_1pct_bler",
        })
        path = destination / f"plan-005-tail-{token}.yaml"
        validate_config(SimulationConfig(data, base_path))
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({
            "snr_db": snr, "target_total_drops": total, "existing_formal_drops": prior,
            "additional_drops": additional, "drop_index_start": 2000,
            "drop_index_end": 1999 + additional, "config": str(path), "config_sha256": digest,
        })
        paths.append(path)
    selection = {
        "schema_version": 1, "target_bler": 0.01,
        "drop_rule": "ceil_1000(clamp(100/min_curve_Jeffreys_BLER, 2000, 20000))",
        "curve_brackets_db": brackets, "formal_snr_points_db": grid,
        "points": entries, "prescan_csv": str(Path(prescan_csv).resolve()),
        "coarse_prescan_csv": None if coarse_prescan_csv is None else str(Path(coarse_prescan_csv).resolve()),
        "existing_csv": str(Path(existing_csv).resolve()),
    }
    target = Path(selection_json).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return paths


def merge_tail(existing_dir: str | Path, tail_root: str | Path, output: str | Path) -> None:
    sources = [("run-001/full", Path(existing_dir).resolve())]
    root = Path(tail_root).resolve()
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "run_metadata.json").exists():
            sources.append((f"run-002/tail/{child.name}", child))
    if len(sources) == 1:
        raise ValueError("no completed tail runs found")
    merge_runs(sources, Path(output).resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prescan = sub.add_parser("prepare-prescan")
    prescan.add_argument("--base-config", type=Path, required=True)
    prescan.add_argument("--output-config", type=Path, required=True)
    fast = sub.add_parser("prepare-fast-prescan")
    fast.add_argument("--base-config", type=Path, required=True)
    fast.add_argument("--output-config", type=Path, required=True)
    prepare = sub.add_parser("prepare-tail")
    prepare.add_argument("--prescan-csv", type=Path, required=True)
    prepare.add_argument("--existing-csv", type=Path, required=True)
    prepare.add_argument("--base-config", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--selection-json", type=Path, required=True)
    prepare.add_argument("--coarse-prescan-csv", type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--existing-dir", type=Path, required=True)
    merge.add_argument("--tail-root", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-prescan":
        print(write_prescan_config(args.base_config, args.output_config))
    elif args.command == "prepare-fast-prescan":
        print(write_fast_prescan_config(args.base_config, args.output_config))
    elif args.command == "prepare-tail":
        for path in prepare_tail_configs(
            args.prescan_csv, args.existing_csv, args.base_config, args.output_dir,
            args.selection_json, args.coarse_prescan_csv,
        ):
            print(path)
    else:
        merge_tail(args.existing_dir, args.tail_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
