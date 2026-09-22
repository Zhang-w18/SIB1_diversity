"""Adaptive-SNR, common-drop Monte Carlo runner for Plan 001."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from sib1div.analysis import wilson_interval
from sib1div.channel import FixedCDLChannel, UMaChannel
from sib1div.config import SimulationConfig
from sib1div.runtime import collect_environment

from .engine import LinkResult, load_codebooks, simulate_one_drop


def curve_is_resolved(
    errors: int,
    blocks: int,
    *,
    min_blocks: int,
    target_errors: int,
    below_interest_bler: float,
    confidence: float,
) -> bool:
    """Whether one curve has enough common-drop evidence at an SNR."""
    if blocks < min_blocks:
        return False
    if errors >= target_errors:
        return True
    return wilson_interval(errors, blocks, confidence)[1] < below_interest_bler


def refinement_snrs(
    point_estimates: Mapping[float, Mapping[str, float]],
    curve_ids: tuple[str, ...],
    targets: tuple[float, ...],
    refinement_step_db: float,
) -> list[float]:
    """Return unevaluated midpoint SNRs around target-BLER crossings."""
    snrs = sorted(point_estimates)
    additions: set[float] = set()
    for curve_id in curve_ids:
        for target in targets:
            for low_snr, high_snr in zip(snrs[:-1], snrs[1:]):
                low = point_estimates[low_snr][curve_id]
                high = point_estimates[high_snr][curve_id]
                if (low - target) * (high - target) <= 0 and high_snr - low_snr > refinement_step_db:
                    additions.add(round(0.5 * (low_snr + high_snr), 10))
    return sorted(additions.difference(point_estimates))


def _estimator_for_covariance(covariance: str) -> str:
    mapping = {
        "ssb_pdp": "lmmse_ssb_pdp",
        "per_beam_pdp": "lmmse_per_beam_pdp",
        "cdd_ssb_pdp": "lmmse_cdd_ssb_pdp",
        "cdd_per_beam_pdp": "lmmse_cdd_per_beam_pdp",
        "ideal_covariance": "lmmse_ideal_covariance",
    }
    try:
        return mapping[covariance]
    except KeyError as error:
        raise ValueError(f"unsupported Plan 001 covariance label: {covariance}") from error


def _result_map(
    results: list[LinkResult],
    curve_specs: list[dict[str, Any]],
    perfect_specs: list[dict[str, Any]],
) -> dict[str, LinkResult]:
    configured = {str(result.curve_id): result for result in results if result.curve_id is not None}
    if configured:
        expected = {str(spec["id"]) for spec in curve_specs + perfect_specs}
        missing = expected.difference(configured)
        if missing:
            raise RuntimeError(f"missing configured simulation results: {sorted(missing)}")
        return {curve_id: configured[curve_id] for curve_id in expected}
    lookup = {(result.scheme, result.estimator): result for result in results}
    selected: dict[str, LinkResult] = {}
    for spec in curve_specs:
        key = (str(spec["scheme"]), _estimator_for_covariance(str(spec["covariance"])))
        if key not in lookup:
            raise RuntimeError(f"missing simulated result for curve {spec['id']}: {key}")
        selected[str(spec["id"])] = lookup[key]
    for spec in perfect_specs:
        scheme = str(spec["scheme"])
        key = (scheme, "perfect")
        if key not in lookup:
            raise RuntimeError(f"missing Perfect-CSI result for {scheme}")
        selected[str(spec["id"])] = lookup[key]
    return selected


def _perfect_specs(receiver: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "perfect_csi_curves" in receiver:
        return [dict(spec) for spec in receiver["perfect_csi_curves"]]
    return [
        {"id": f"PERF-{scheme}", "scheme": str(scheme)}
        for scheme in receiver.get("perfect_csi_diagnostic_schemes", [])
    ]


def _empty_point(all_curve_ids: tuple[str, ...], estimated_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "blocks": 0,
        "curves": {
            curve_id: {"errors": 0, "nmse_sum": 0.0}
            for curve_id in all_curve_ids
        },
        "pairs": {
            f"{first}|{second}": {"n00": 0, "n01": 0, "n10": 0, "n11": 0}
            for first, second in itertools.combinations(estimated_ids, 2)
        },
        "complete": False,
        "stop_reason": "running",
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _snr_key(snr_db: float) -> str:
    return f"{snr_db:.10g}"


def _snr_stream_index(snr_db: float, search_min_db: float, refinement_step_db: float) -> int:
    return int(round((snr_db - search_min_db) / refinement_step_db))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(
    state: Mapping[str, Any],
    estimated_ids: tuple[str, ...],
    curve_labels: Mapping[str, tuple[str, str]],
    confidence: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bler_rows: list[dict[str, Any]] = []
    nmse_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for snr_text, point in sorted(state["points"].items(), key=lambda item: float(item[0])):
        snr_db = float(snr_text)
        blocks = int(point["blocks"])
        if blocks == 0:
            continue
        for curve_id, values in point["curves"].items():
            errors = int(values["errors"])
            low, high = wilson_interval(errors, blocks, confidence)
            scheme, covariance = curve_labels[curve_id]
            bler_rows.append({
                "snr_db": snr_db,
                "curve_id": curve_id,
                "curve_type": "estimated_csi" if curve_id in estimated_ids else "perfect_csi",
                "scheme": scheme,
                "covariance": covariance,
                "block_errors": errors,
                "blocks": blocks,
                "bler": errors / blocks,
                "wilson_low": low,
                "wilson_high": high,
                "stop_reason": point["stop_reason"],
            })
            if curve_id in estimated_ids:
                nmse_rows.append({
                    "snr_db": snr_db,
                    "curve_id": curve_id,
                    "scheme": scheme,
                    "covariance": covariance,
                    "blocks": blocks,
                    "mean_nmse": float(values["nmse_sum"]) / blocks,
                })
        for pair, counts in point["pairs"].items():
            first, second = pair.split("|", 1)
            paired_rows.append({
                "snr_db": snr_db,
                "curve_a": first,
                "curve_b": second,
                "blocks": blocks,
                **counts,
            })
    return bler_rows, nmse_rows, paired_rows


def bler_plot_lower_limit(plotted_bler: list[float]) -> float:
    """Keep low non-zero Monte Carlo estimates visible without excessive range."""
    return max(1e-5, min(5e-3, min(plotted_bler) / 2.0))


def format_duration(seconds: float) -> str:
    """Format a non-negative duration for progress logs."""
    total = max(0, int(round(seconds)))
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {clock}" if days else clock


def _write_plots(
    output: Path,
    bler_rows: list[dict[str, Any]],
    nmse_rows: list[dict[str, Any]],
    *,
    title_prefix: str = "Plan 001",
    x_label: str = "Normalized Es/N0 (dB)",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    estimated = [row for row in bler_rows if row["curve_type"] == "estimated_csi"]
    perfect = [row for row in bler_rows if row["curve_type"] == "perfect_csi"]
    for filename, rows, title in (
        ("bler.png", estimated, f"{title_prefix} estimated-CSI BLER"),
        ("bler_perfect_csi.png", perfect, f"{title_prefix} Perfect-CSI diagnostic BLER"),
    ):
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for curve_id in sorted({str(row["curve_id"]) for row in rows}):
            selected = sorted((row for row in rows if row["curve_id"] == curve_id), key=lambda row: row["snr_db"])
            x = np.array([row["snr_db"] for row in selected], dtype=float)
            y = np.array([row["bler"] if row["bler"] > 0 else row["wilson_high"] for row in selected])
            ax.semilogy(x, y, marker="o", label=curve_id)
        ax.axhline(1e-1, color="0.5", ls="--", lw=1)
        ax.axhline(1e-2, color="0.5", ls=":", lw=1)
        plotted_bler = [
            float(row["bler"] if row["bler"] > 0 else row["wilson_high"])
            for row in rows
        ]
        lower_limit = bler_plot_lower_limit(plotted_bler)
        ax.set(xlabel=x_label, ylabel="BLER", title=title, ylim=(lower_limit, 1.0))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    if nmse_rows:
        fig, ax = plt.subplots(figsize=(9, 6))
        for curve_id in sorted({str(row["curve_id"]) for row in nmse_rows}):
            selected = sorted((row for row in nmse_rows if row["curve_id"] == curve_id), key=lambda row: row["snr_db"])
            ax.semilogy(
                [row["snr_db"] for row in selected],
                [row["mean_nmse"] for row in selected],
                marker="o", label=curve_id,
            )
        ax.set(xlabel=x_label, ylabel="Mean channel-estimation NMSE",
               title=f"{title_prefix} estimated-CSI NMSE")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "nmse.png", dpi=180)
        plt.close(fig)


def _write_fixed_outputs(
    output: Path,
    bler_rows: list[dict[str, Any]],
    nmse_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
) -> None:
    _write_csv(output / "bler.csv", [
        "snr_db", "curve_id", "curve_type", "scheme", "covariance",
        "block_errors", "blocks", "bler", "wilson_low", "wilson_high", "stop_reason",
    ], bler_rows)
    _write_csv(output / "nmse.csv", [
        "snr_db", "curve_id", "scheme", "covariance", "blocks", "mean_nmse",
    ], nmse_rows)
    _write_csv(output / "paired_counts.csv", [
        "snr_db", "curve_a", "curve_b", "blocks", "n00", "n01", "n10", "n11",
    ], paired_rows)


def run_fixed_curve_simulation(
    config: SimulationConfig,
    codebook_path: str | Path,
    output_dir: str | Path,
    phase: str | None = None,
) -> Path:
    """Run fixed-SNR, fixed-common-drop curve points with resumable aggregation."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ".fixed_checkpoint.json"
    final_metadata = output / "run_metadata.json"
    if final_metadata.exists() and not checkpoint.exists():
        raise FileExistsError(f"completed fixed run already exists: {final_metadata}")

    receiver = config.data["receiver"]
    curve_specs = list(receiver.get("estimated_csi_curves", []))
    perfect_specs = _perfect_specs(receiver)
    monte = config.data["monte_carlo"]
    if monte.get("snr_strategy") == "fixed_phased_common_drops":
        phases = monte.get("phases", {})
        if phase not in phases:
            raise ValueError(f"--phase must name one of {sorted(phases)}")
        phase_settings = phases[phase]
        if not bool(phase_settings.get("estimated_csi", False)):
            curve_specs = []
        if not bool(phase_settings.get("perfect_csi", False)):
            perfect_specs = []
        snr_values = [float(value) for value in phase_settings["snr_points_db"]]
    else:
        if phase is not None:
            raise ValueError("--phase is only valid for fixed_phased_common_drops")
        snr_values = [float(value) for value in monte["snr_points_db"]]
    estimated_ids = tuple(str(spec["id"]) for spec in curve_specs)
    perfect_ids = tuple(str(spec["id"]) for spec in perfect_specs)
    if not estimated_ids and not perfect_ids:
        raise ValueError("fixed curve run requires at least one configured curve")
    if len(set(estimated_ids + perfect_ids)) != len(estimated_ids + perfect_ids):
        raise ValueError("fixed curve IDs must be unique")
    all_curve_ids = estimated_ids + perfect_ids
    curve_labels = {
        str(spec["id"]): (str(spec.get("label", spec["scheme"])), str(spec["covariance"]))
        for spec in curve_specs
    }
    curve_labels.update({
        str(spec["id"]): (str(spec.get("label", spec["scheme"])), "perfect_csi")
        for spec in perfect_specs
    })

    drops_per_snr = int(monte["fixed_drops_per_snr"])
    progress_interval = int(monte.get("progress_interval_drops", 10))
    confidence = float(monte["confidence_level"])
    if not snr_values or len(set(snr_values)) != len(snr_values):
        raise ValueError("fixed curve run requires unique snr_points_db values")
    if drops_per_snr < 1 or progress_interval < 1:
        raise ValueError("fixed drops and progress interval must be positive")
    stream_origin = float(monte.get("snr_stream_grid_origin_db", min(snr_values)))
    stream_step = float(monte.get("snr_stream_grid_step_db", 1.0))
    if stream_step <= 0:
        raise ValueError("SNR stream grid step must be positive")
    snr_stream_indices: dict[float, int] = {}
    for snr_db in snr_values:
        index = int(round((snr_db - stream_origin) / stream_step))
        if index < 0 or not np.isclose(stream_origin + index * stream_step, snr_db, atol=1e-9):
            raise ValueError(f"SNR {snr_db:g} dB is not on the configured random-stream grid")
        snr_stream_indices[snr_db] = index
    if len(set(snr_stream_indices.values())) != len(snr_values):
        raise ValueError("fixed SNR points map to duplicate random-stream indices")

    config_digest = hashlib.sha256(config.source.read_bytes()).hexdigest()
    expected_codebook_digest = str(config.data["codebooks"]["weights_sha256"])
    ssb, secondary = load_codebooks(codebook_path, expected_codebook_digest)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config_sha256"] != config_digest or state["codebook_sha256"] != expected_codebook_digest:
            raise RuntimeError("fixed checkpoint does not match the frozen config/codebook")
    else:
        state = {
            "schema_version": 1,
            "config_sha256": config_digest,
            "codebook_sha256": expected_codebook_digest,
            "points": {},
        }
        _atomic_json(checkpoint, state)
        with (output / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(config.resolved(), stream, sort_keys=False, allow_unicode=True)
        environment = collect_environment(check_tensorflow=False)
        _atomic_json(output / "environment.json", environment)
        (output / "run.log").write_text(
            f"fixed curve run initialized: {datetime.now(timezone.utc).isoformat()}\n",
            encoding="utf-8",
        )

    channel = (
        FixedCDLChannel(config, ssb)
        if config.data["run"]["link_mode"] == "fixed_cdl_statistics"
        else UMaChannel(config)
    )
    total_blocks = len(snr_values) * drops_per_snr
    completed_at_start = sum(
        int(point.get("blocks", 0)) for point in state["points"].values()
    )
    progress_started = time.monotonic()
    print(
        f"Fixed curve run: SNR={snr_values}, {drops_per_snr} common drops/SNR; "
        f"{len(estimated_ids)} estimated-CSI curves + {len(perfect_ids)} Perfect-CSI diagnostics",
        flush=True,
    )
    for snr_db in snr_values:
        key = _snr_key(snr_db)
        point = state["points"].setdefault(key, _empty_point(all_curve_ids, estimated_ids))
        if point["complete"]:
            print(f"SNR {snr_db:g} dB already complete at {point['blocks']} common drops", flush=True)
            continue
        print(f"SNR {snr_db:g} dB starting/resuming at {point['blocks']} common drops", flush=True)
        snr_index = snr_stream_indices[snr_db]
        while int(point["blocks"]) < drops_per_snr:
            blocks = int(point["blocks"])
            drop_index = int(monte.get("drop_index_start", 0)) + blocks
            results = simulate_one_drop(
                config, channel.generate(drop_index), ssb, secondary,
                snr_db, snr_index, estimator="plan001",
                curve_specs=curve_specs, perfect_specs=perfect_specs,
            )
            selected = _result_map(results, curve_specs, perfect_specs)
            error_flags: dict[str, int] = {}
            for curve_id, result in selected.items():
                error = int(not result.crc_ok)
                error_flags[curve_id] = error
                point["curves"][curve_id]["errors"] += error
                point["curves"][curve_id]["nmse_sum"] += float(result.nmse)
            for pair, counts in point["pairs"].items():
                first, second = pair.split("|", 1)
                counts[f"n{error_flags[first]}{error_flags[second]}"] += 1
            point["blocks"] += 1
            if int(point["blocks"]) % progress_interval == 0:
                _atomic_json(checkpoint, state)
                completed_now = sum(
                    int(saved.get("blocks", 0)) for saved in state["points"].values()
                )
                processed = completed_now - completed_at_start
                elapsed = max(time.monotonic() - progress_started, 1e-9)
                rate = processed / elapsed
                remaining_blocks = max(0, total_blocks - completed_now)
                remaining_seconds = remaining_blocks / rate if rate > 0 else float("inf")
                finish_utc = datetime.now(timezone.utc) + timedelta(seconds=remaining_seconds)
                error_text = ", ".join(
                    f"{curve_id}:{point['curves'][curve_id]['errors']}"
                    for curve_id in estimated_ids
                )
                print(
                    f"  SNR {snr_db:g} dB, common drops={point['blocks']}, "
                    f"overall={completed_now}/{total_blocks} ({100.0 * completed_now / total_blocks:.1f}%), "
                    f"rate={rate:.3f} drops/s, elapsed={format_duration(elapsed)}, "
                    f"remaining={format_duration(remaining_seconds)}, "
                    f"finish_utc={finish_utc.isoformat(timespec='seconds')}, errors=[{error_text}]",
                    flush=True,
                )
        point["complete"] = True
        point["stop_reason"] = "fixed_common_drops"
        _atomic_json(checkpoint, state)
        bler_rows, nmse_rows, paired_rows = _summary_rows(
            state, estimated_ids, curve_labels, confidence,
        )
        _write_fixed_outputs(output, bler_rows, nmse_rows, paired_rows)
        print(f"SNR {snr_db:g} dB complete: drops={point['blocks']}", flush=True)

    bler_rows, nmse_rows, paired_rows = _summary_rows(state, estimated_ids, curve_labels, confidence)
    _write_fixed_outputs(output, bler_rows, nmse_rows, paired_rows)
    experiment_id = str(config.data.get("experiment", {}).get("id", "fixed curves"))
    mode = str(config.data["run"]["link_mode"])
    x_label = (
        "Reference Es/N0 (dB)"
        if mode.startswith("fixed_radius_") or mode == "fixed_cdl_statistics"
        else "Normalized Es/N0 (dB)"
    )
    _write_plots(output, bler_rows, nmse_rows, title_prefix=experiment_id, x_label=x_label)
    environment = collect_environment(check_tensorflow=False)
    metadata = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.source),
        "config_sha256": config_digest,
        "codebook_sha256": expected_codebook_digest,
        "master_seed": config.seed,
        "link_mode": str(config.data["run"]["link_mode"]),
        "phase": phase,
        "estimated_csi_curves": curve_specs,
        "perfect_csi_curves": perfect_specs,
        "actual_snr_db": snr_values,
        "snr_stream_indices": {str(snr): index for snr, index in snr_stream_indices.items()},
        "fixed_drops_per_snr": drops_per_snr,
        "stopping": dict(monte),
        "normalization": dict(config.data.get("link_normalization", {})),
        "versions": {
            "python": environment["python"]["version"],
            **{name: details["version"] for name, details in environment["packages"].items()},
        },
    }
    if isinstance(channel, FixedCDLChannel):
        metadata["fixed_cdl_statistics"] = {
            "selected_ssb": channel.selected_ssb,
            "ssb_long_term_powers": channel.ssb_long_term_powers.tolist(),
            "reference_receive_power": channel.reference_receive_power,
            "angle_statistics": channel.angle_statistics,
            "covariance_realizations": int(config.data["fixed_cdl_statistics"]["covariance_realizations"]),
            "statistics_seed": int(config.data["fixed_cdl_statistics"]["statistics_seed"]),
            "realization_seed": int(config.data["fixed_cdl_statistics"]["realization_seed"]),
        }
    _atomic_json(final_metadata, metadata)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"fixed curve run completed: {metadata['completed_utc']}\n")
    checkpoint.unlink(missing_ok=True)
    print(f"Fixed curve run complete: {output}", flush=True)
    return output / "bler.csv"


def run_adaptive_simulation(
    config: SimulationConfig,
    codebook_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Run the frozen Plan 001 curve set with adaptive SNR and common drops."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / ".adaptive_checkpoint.json"
    final_metadata = output / "run_metadata.json"
    if final_metadata.exists() and not checkpoint.exists():
        raise FileExistsError(f"completed adaptive run already exists: {final_metadata}")

    receiver = config.data["receiver"]
    curve_specs = list(receiver["estimated_csi_curves"])
    estimated_ids = tuple(str(spec["id"]) for spec in curve_specs)
    if len(estimated_ids) != 7 or len(set(estimated_ids)) != 7:
        raise ValueError("Plan 001 requires exactly seven unique estimated-CSI curve IDs")
    perfect_specs = _perfect_specs(receiver)
    perfect_ids = tuple(str(spec["id"]) for spec in perfect_specs)
    all_curve_ids = estimated_ids + perfect_ids
    curve_labels = {
        str(spec["id"]): (str(spec["scheme"]), str(spec["covariance"]))
        for spec in curve_specs
    }
    curve_labels.update({
        str(spec["id"]): (str(spec["scheme"]), "perfect_csi") for spec in perfect_specs
    })

    monte = config.data["monte_carlo"]
    search_min, search_max = map(float, monte["search_range_db"])
    coarse_step = float(monte["coarse_step_db"])
    refine_step = float(monte["refinement_step_db"])
    lower_interest, upper_interest = map(float, monte["bler_interest_range"])
    min_blocks = int(monte["min_drops_per_snr"])
    target_errors = int(monte["target_block_errors"])
    cap = int(monte["max_drops_per_snr_safety_cap"])
    confidence = float(monte["confidence_level"])
    start_snr = float(monte.get("search_start_db", 0.0))
    progress_interval = int(monte.get("progress_interval_drops", 10))
    if progress_interval < 1:
        raise ValueError("progress_interval_drops must be positive")
    if not search_min <= start_snr <= search_max:
        raise ValueError("adaptive SNR start must lie inside the search range")

    config_digest = hashlib.sha256(config.source.read_bytes()).hexdigest()
    expected_codebook_digest = str(config.data["codebooks"]["weights_sha256"])
    ssb, secondary = load_codebooks(codebook_path, expected_codebook_digest)
    if checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config_sha256"] != config_digest or state["codebook_sha256"] != expected_codebook_digest:
            raise RuntimeError("adaptive checkpoint does not match the frozen config/codebook")
    else:
        state = {
            "schema_version": 1,
            "config_sha256": config_digest,
            "codebook_sha256": expected_codebook_digest,
            "points": {},
        }
        _atomic_json(checkpoint, state)
    channel = (
        FixedCDLChannel(config, ssb)
        if config.data["run"]["link_mode"] == "fixed_cdl_statistics"
        else UMaChannel(config)
    )
    print(
        f"Plan 001 adaptive run: {len(estimated_ids)} estimated-CSI curves + "
        f"{len(perfect_ids)} Perfect-CSI diagnostics; target={target_errors} errors, "
        f"minimum={min_blocks} drops, safety cap={cap} drops/SNR",
        flush=True,
    )

    def evaluate(snr_db: float) -> dict[str, Any]:
        key = _snr_key(snr_db)
        point = state["points"].setdefault(key, _empty_point(all_curve_ids, estimated_ids))
        if point["complete"]:
            print(f"SNR {snr_db:g} dB already complete at {point['blocks']} common drops", flush=True)
            return point
        print(f"SNR {snr_db:g} dB starting/resuming at {point['blocks']} common drops", flush=True)
        snr_index = _snr_stream_index(snr_db, search_min, refine_step)
        while int(point["blocks"]) < cap:
            blocks = int(point["blocks"])
            if blocks >= min_blocks and all(
                curve_is_resolved(
                    int(point["curves"][curve_id]["errors"]), blocks,
                    min_blocks=min_blocks, target_errors=target_errors,
                    below_interest_bler=lower_interest, confidence=confidence,
                )
                for curve_id in estimated_ids
            ):
                point["complete"] = True
                point["stop_reason"] = "all_estimated_curves_resolved"
                break
            drop_index = int(monte["drop_index_start"]) + blocks
            results = simulate_one_drop(
                config, channel.generate(drop_index), ssb, secondary,
                snr_db, snr_index, estimator="plan001",
            )
            selected = _result_map(results, curve_specs, perfect_specs)
            error_flags: dict[str, int] = {}
            for curve_id, result in selected.items():
                error = int(not result.crc_ok)
                error_flags[curve_id] = error
                point["curves"][curve_id]["errors"] += error
                point["curves"][curve_id]["nmse_sum"] += float(result.nmse)
            for pair, counts in point["pairs"].items():
                first, second = pair.split("|", 1)
                counts[f"n{error_flags[first]}{error_flags[second]}"] += 1
            point["blocks"] += 1
            if int(point["blocks"]) % progress_interval == 0:
                _atomic_json(checkpoint, state)
                error_text = ", ".join(
                    f"{curve_id}:{point['curves'][curve_id]['errors']}"
                    for curve_id in estimated_ids
                )
                print(
                    f"  SNR {snr_db:g} dB, common drops={point['blocks']}, errors=[{error_text}]",
                    flush=True,
                )
        if not point["complete"]:
            point["complete"] = True
            point["stop_reason"] = "safety_cap"
        _atomic_json(checkpoint, state)
        bler_rows, nmse_rows, paired_rows = _summary_rows(
            state, estimated_ids, curve_labels, confidence,
        )
        _write_csv(output / "bler.csv", list(bler_rows[0]), bler_rows)
        _write_csv(output / "nmse.csv", list(nmse_rows[0]), nmse_rows)
        _write_csv(output / "paired_counts.csv", list(paired_rows[0]), paired_rows)
        print(
            f"SNR {snr_db:g} dB complete: drops={point['blocks']}, reason={point['stop_reason']}",
            flush=True,
        )
        return point

    def estimates() -> dict[float, dict[str, float]]:
        return {
            float(snr): {
                curve_id: values["errors"] / point["blocks"]
                for curve_id, values in point["curves"].items()
                if curve_id in estimated_ids
            }
            for snr, point in state["points"].items()
            if point["complete"] and point["blocks"] > 0
        }

    evaluate(start_snr)
    snr = start_snr
    while snr > search_min:
        point = state["points"][_snr_key(snr)]
        blocks = int(point["blocks"])
        if all(
            wilson_interval(int(point["curves"][curve_id]["errors"]), blocks, confidence)[0]
            > upper_interest for curve_id in estimated_ids
        ):
            break
        snr = max(search_min, snr - coarse_step)
        evaluate(snr)
    snr = start_snr
    while snr < search_max:
        point = state["points"][_snr_key(snr)]
        blocks = int(point["blocks"])
        if all(
            wilson_interval(int(point["curves"][curve_id]["errors"]), blocks, confidence)[1]
            < lower_interest for curve_id in estimated_ids
        ):
            break
        snr = min(search_max, snr + coarse_step)
        evaluate(snr)

    for snr_db in refinement_snrs(
        estimates(), estimated_ids, (upper_interest, lower_interest), refine_step,
    ):
        evaluate(snr_db)

    bler_rows, nmse_rows, paired_rows = _summary_rows(state, estimated_ids, curve_labels, confidence)
    _write_csv(output / "bler.csv", list(bler_rows[0]), bler_rows)
    _write_csv(output / "nmse.csv", list(nmse_rows[0]), nmse_rows)
    _write_csv(output / "paired_counts.csv", list(paired_rows[0]), paired_rows)
    _write_plots(output, bler_rows, nmse_rows)
    environment = collect_environment(check_tensorflow=False)
    metadata = {
        "schema_version": 1,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.source),
        "config_sha256": config_digest,
        "codebook_sha256": expected_codebook_digest,
        "master_seed": config.seed,
        "link_mode": str(config.data["run"]["link_mode"]),
        "estimated_csi_curves": curve_specs,
        "perfect_csi_schemes": [str(spec["scheme"]) for spec in perfect_specs],
        "actual_snr_db": sorted(float(value) for value in state["points"]),
        "stopping": dict(monte),
        "normalization": dict(config.data.get("link_normalization", {})),
        "versions": {
            "python": environment["python"]["version"],
            **{name: details["version"] for name, details in environment["packages"].items()},
        },
    }
    if isinstance(channel, FixedCDLChannel):
        metadata["fixed_cdl_statistics"] = {
            "selected_ssb": channel.selected_ssb,
            "ssb_long_term_powers": channel.ssb_long_term_powers.tolist(),
            "reference_receive_power": channel.reference_receive_power,
            "angle_statistics": channel.angle_statistics,
            "covariance_realizations": int(config.data["fixed_cdl_statistics"]["covariance_realizations"]),
            "statistics_seed": int(config.data["fixed_cdl_statistics"]["statistics_seed"]),
            "realization_seed": int(config.data["fixed_cdl_statistics"]["realization_seed"]),
        }
    _atomic_json(final_metadata, metadata)
    checkpoint.unlink(missing_ok=True)
    print(f"Plan 001 adaptive run complete: {output}", flush=True)
    return output / "bler.csv"
