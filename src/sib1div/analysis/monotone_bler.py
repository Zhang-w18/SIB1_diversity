"""Merge verified fixed-drop runs and fit non-increasing BLER curves."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sib1div.analysis.statistics import wilson_interval


MAIN_CURVES = ("B-SSB", "P2-SSB", "P6-SSB", "BC-SSB", "CDD-SSB")
TARGETS = (0.1, 0.01)
SNR_RANGE_DB = (-18.0, -6.0)
EXPECTED_SNRS = (
    -18.0, -17.5, -17.0, -16.5, -16.0, -15.5, -15.0,
    -14.0, -13.0, -12.0, -11.0, -10.5, -10.0, -9.5, -9.0,
    -8.5, -8.0, -7.5, -7.0, -6.5, -6.0,
)


def binomial_isotonic_decreasing(errors: list[int], blocks: list[int]) -> list[float]:
    """Binomial maximum-likelihood fit constrained to decrease with SNR."""
    if len(errors) != len(blocks) or not errors:
        raise ValueError("errors and blocks must be non-empty and equally sized")
    pools: list[dict[str, int]] = []
    for index, (count, total) in enumerate(zip(errors, blocks, strict=True)):
        if total <= 0 or count < 0 or count > total:
            raise ValueError("invalid binomial count")
        pools.append({"start": index, "end": index, "errors": count, "blocks": total})
        while len(pools) >= 2:
            left, right = pools[-2], pools[-1]
            if left["errors"] / left["blocks"] >= right["errors"] / right["blocks"]:
                break
            pools[-2:] = [{
                "start": left["start"], "end": right["end"],
                "errors": left["errors"] + right["errors"],
                "blocks": left["blocks"] + right["blocks"],
            }]
    fitted = [0.0] * len(errors)
    for pool in pools:
        value = pool["errors"] / pool["blocks"]
        for index in range(pool["start"], pool["end"] + 1):
            fitted[index] = value
    return fitted


def log_bler_threshold(snrs: list[float], fitted: list[float], target: float) -> float:
    """Interpolate a target crossing linearly in log10(BLER)."""
    if len(snrs) != len(fitted) or len(snrs) < 2 or not 0 < target < 1:
        raise ValueError("invalid threshold inputs")
    for x0, x1, y0, y1 in zip(snrs[:-1], snrs[1:], fitted[:-1], fitted[1:], strict=True):
        if y0 >= target >= y1:
            if y0 == y1:
                return 0.5 * (x0 + x1)
            if y0 <= 0 or y1 <= 0:
                return x0 + (y0 - target) * (x1 - x0) / (y0 - y1)
            fraction = (math.log10(y0) - math.log10(target)) / (math.log10(y0) - math.log10(y1))
            return x0 + fraction * (x1 - x0)
    raise ValueError(f"target BLER {target:g} is not bracketed")


def threshold_bracket(snrs: list[float], fitted: list[float], target: float) -> tuple[float, float]:
    for x0, x1, y0, y1 in zip(snrs[:-1], snrs[1:], fitted[:-1], fitted[1:], strict=True):
        if y0 >= target >= y1:
            return x0, x1
    raise ValueError(f"target BLER {target:g} is not bracketed")


def threshold_estimate(
    snrs: list[float], fitted: list[float], target: float,
) -> tuple[float, float, float] | None:
    """Return an interpolated crossing, or None when the sampled grid does not bracket it."""
    try:
        bracket_low, bracket_high = threshold_bracket(snrs, fitted, target)
    except ValueError:
        return None
    return log_bler_threshold(snrs, fitted, target), bracket_low, bracket_high


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metadata(source: Path) -> dict:
    return json.loads((source / "run_metadata.json").read_text(encoding="utf-8"))


def _validate_sources(sources: list[Path]) -> list[dict]:
    metadata = [_metadata(source) for source in sources]
    for key in ("master_seed", "link_mode", "codebook_sha256", "normalization", "versions"):
        if any(item.get(key) != metadata[0].get(key) for item in metadata[1:]):
            raise ValueError(f"source metadata mismatch: {key}")
    reference_specs: dict[str, tuple[object, ...]] = {}
    for source, item in zip(sources, metadata, strict=True):
        specs = {str(spec["id"]): spec for spec in item["estimated_csi_curves"]}
        missing = set(MAIN_CURVES).difference(specs)
        if missing:
            raise ValueError(f"missing main curves in {source}: {sorted(missing)}")
        for curve in MAIN_CURVES:
            spec = specs[curve]
            signature = tuple(spec.get(key) for key in ("scheme", "covariance", "window_prbs", "prg_size_prbs"))
            if curve in reference_specs and signature != reference_specs[curve]:
                raise ValueError(f"curve definition mismatch for {curve}: {source}")
            reference_specs[curve] = signature
    intervals: dict[float, list[tuple[int, int, Path]]] = defaultdict(list)
    for source, item in zip(sources, metadata, strict=True):
        start = int(item["stopping"]["drop_index_start"])
        count = int(item["fixed_drops_per_snr"])
        for snr in map(float, item["actual_snr_db"]):
            end = start + count
            for old_start, old_end, old_source in intervals[snr]:
                if max(start, old_start) < min(end, old_end):
                    raise ValueError(f"overlapping drops at {snr:g} dB: {old_source} and {source}")
            intervals[snr].append((start, end, source))
    return metadata


def merge_fit_and_plot(sources: list[Path], output: Path) -> None:
    """Validate sources, aggregate counts, fit monotone curves, and write thresholds."""
    if not sources:
        raise ValueError("at least one source is required")
    metadata = _validate_sources(sources)
    grouped: dict[tuple[float, str], dict[str, object]] = {}
    nmse_grouped: dict[tuple[float, str], dict[str, object]] = {}
    for source in sources:
        for row in _read_csv(source / "bler.csv"):
            curve = row["curve_id"]
            if curve not in MAIN_CURVES or row["curve_type"] != "estimated_csi":
                continue
            snr_db = float(row["snr_db"])
            if not SNR_RANGE_DB[0] <= snr_db <= SNR_RANGE_DB[1]:
                continue
            key = (snr_db, curve)
            item = grouped.setdefault(key, {"errors": 0, "blocks": 0, "sources": []})
            item["errors"] = int(item["errors"]) + int(row["block_errors"])
            item["blocks"] = int(item["blocks"]) + int(row["blocks"])
            item["sources"].append(source.as_posix())
        for row in _read_csv(source / "nmse.csv"):
            curve = row["curve_id"]
            snr_db = float(row["snr_db"])
            if curve not in MAIN_CURVES or not SNR_RANGE_DB[0] <= snr_db <= SNR_RANGE_DB[1]:
                continue
            key = (snr_db, curve)
            item = nmse_grouped.setdefault(key, {"weighted_sum": 0.0, "blocks": 0, "sources": []})
            count = int(row["blocks"])
            item["weighted_sum"] = float(item["weighted_sum"]) + count * float(row["mean_nmse"])
            item["blocks"] = int(item["blocks"]) + count
            item["sources"].append(source.as_posix())

    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    thresholds: list[dict] = []
    per_curve: dict[str, list[dict]] = {}
    for curve in MAIN_CURVES:
        points = sorted((key[0], value) for key, value in grouped.items() if key[1] == curve)
        if len(points) < 2:
            raise ValueError(f"insufficient points for {curve}")
        snrs = [point[0] for point in points]
        if tuple(snrs) != EXPECTED_SNRS:
            missing = sorted(set(EXPECTED_SNRS).difference(snrs))
            extra = sorted(set(snrs).difference(EXPECTED_SNRS))
            raise ValueError(f"incomplete SNR grid for {curve}: missing={missing}, extra={extra}")
        errors = [int(point[1]["errors"]) for point in points]
        blocks = [int(point[1]["blocks"]) for point in points]
        fitted = binomial_isotonic_decreasing(errors, blocks)
        curve_rows = []
        for snr, count, total, fit, (_, item) in zip(snrs, errors, blocks, fitted, points, strict=True):
            low, high = wilson_interval(count, total)
            row = {
                "snr_db": snr, "curve_id": curve, "block_errors": count, "blocks": total,
                "raw_bler": count / total, "wilson_low": low, "wilson_high": high,
                "monotone_bler": fit, "source_runs": "+".join(item["sources"]),
            }
            rows.append(row)
            curve_rows.append(row)
        per_curve[curve] = curve_rows
        for target in TARGETS:
            estimate = threshold_estimate(snrs, fitted, target)
            if estimate is None:
                thresholds.append({
                    "curve_id": curve, "target_bler": target,
                    "snr_db": "", "bracket_low_snr_db": "", "bracket_high_snr_db": "",
                    "method": "not_estimated_target_not_bracketed",
                })
            else:
                snr_estimate, bracket_low, bracket_high = estimate
                thresholds.append({
                    "curve_id": curve, "target_bler": target,
                    "snr_db": snr_estimate,
                    "bracket_low_snr_db": bracket_low,
                    "bracket_high_snr_db": bracket_high,
                    "method": "log10_bler_linear_interpolation_on_monotone_fit",
                })

    _write_csv(output / "bler_monotone.csv", rows)
    _write_csv(output / "thresholds.csv", thresholds)
    _write_raw_plot(output / "bler_raw.png", per_curve)
    _write_plot(output / "bler_monotone.png", per_curve, thresholds)
    nmse_rows = _nmse_rows(nmse_grouped)
    _write_csv(output / "nmse_merged.csv", nmse_rows)
    pol_rows = _pol_nmse_rows(nmse_rows)
    _write_csv(output / "nmse_pol_comparison.csv", pol_rows)
    _write_nmse_plot(output / "nmse.png", nmse_rows)
    manifest = {
        "schema_version": 1,
        "method": "blocks-weighted binomial non-increasing isotonic MLE; log-BLER linear crossing interpolation",
        "sources": [source.as_posix() for source in sources],
        "source_metadata": metadata,
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _nmse_rows(grouped: dict[tuple[float, str], dict[str, object]]) -> list[dict]:
    rows = []
    for curve in MAIN_CURVES:
        snrs = sorted(snr for snr, curve_id in grouped if curve_id == curve)
        if tuple(snrs) != EXPECTED_SNRS:
            raise ValueError(f"incomplete NMSE SNR grid for {curve}")
        for snr in snrs:
            item = grouped[(snr, curve)]
            blocks = int(item["blocks"])
            rows.append({
                "snr_db": snr, "curve_id": curve, "blocks": blocks,
                "mean_nmse": float(item["weighted_sum"]) / blocks,
                "source_runs": "+".join(item["sources"]),
            })
    return rows


def _pol_nmse_rows(rows: list[dict]) -> list[dict]:
    lookup = {(float(row["snr_db"]), row["curve_id"]): row for row in rows}
    output = []
    for snr in EXPECTED_SNRS:
        p2 = float(lookup[(snr, "P2-SSB")]["mean_nmse"])
        p6 = float(lookup[(snr, "P6-SSB")]["mean_nmse"])
        output.append({
            "snr_db": snr, "p2_mean_nmse": p2, "p6_mean_nmse": p6,
            "p6_minus_p2": p6 - p2, "p6_over_p2": p6 / p2,
        })
    return output


def _write_plot(path: Path, per_curve: dict[str, list[dict]], thresholds: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    colors = {"B-SSB": "C0", "P2-SSB": "C1", "P6-SSB": "C2", "BC-SSB": "C3", "CDD-SSB": "C4"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for curve, curve_rows in per_curve.items():
        x = [row["snr_db"] for row in curve_rows]
        raw = [row["raw_bler"] for row in curve_rows]
        fit = [row["monotone_bler"] for row in curve_rows]
        ax.scatter(x, raw, s=15, facecolors="none", edgecolors=colors[curve], alpha=0.5)
        ax.semilogy(x, fit, marker="o", ms=3, lw=1.8, color=colors[curve], label=curve)
    for target, style in ((0.1, "--"), (0.01, ":")):
        ax.axhline(target, color="0.4", ls=style, lw=1)
    ax.set(xlabel="Nominal reference Es/N0 gamma0 (dB)", ylabel="BLER",
           title="Plan 004 monotone estimated-CSI BLER", xlim=(-18, -6), ylim=(5e-3, 0.25))
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_raw_plot(path: Path, per_curve: dict[str, list[dict]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    colors = {"B-SSB": "C0", "P2-SSB": "C1", "P6-SSB": "C2", "BC-SSB": "C3", "CDD-SSB": "C4"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for curve, curve_rows in per_curve.items():
        x = [row["snr_db"] for row in curve_rows]
        raw = [row["raw_bler"] for row in curve_rows]
        lower = [value - row["wilson_low"] for value, row in zip(raw, curve_rows, strict=True)]
        upper = [row["wilson_high"] - value for value, row in zip(raw, curve_rows, strict=True)]
        ax.errorbar(x, raw, yerr=[lower, upper], marker="o", ms=3, capsize=2, lw=1.2,
                    color=colors[curve], label=curve)
    for target, style in ((0.1, "--"), (0.01, ":")):
        ax.axhline(target, color="0.4", ls=style, lw=1)
    ax.set(xlabel="Nominal reference Es/N0 gamma0 (dB)", ylabel="Raw BLER",
           title="Plan 004 raw estimated-CSI BLER (95% Wilson CI)",
           xlim=SNR_RANGE_DB, ylim=(5e-3, 0.25))
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_nmse_plot(path: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    colors = {"B-SSB": "C0", "P2-SSB": "C1", "P6-SSB": "C2", "BC-SSB": "C3", "CDD-SSB": "C4"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for curve in MAIN_CURVES:
        selected = sorted((row for row in rows if row["curve_id"] == curve), key=lambda row: row["snr_db"])
        ax.semilogy(
            [row["snr_db"] for row in selected], [row["mean_nmse"] for row in selected],
            marker="o", ms=3, lw=1.6, color=colors[curve], label=curve,
        )
    ax.set(xlabel="Nominal reference Es/N0 gamma0 (dB)", ylabel="Mean channel-estimation NMSE",
           title="Plan 004 merged channel-estimation NMSE", xlim=SNR_RANGE_DB)
    ax.xaxis.set_minor_locator(MultipleLocator(1.0))
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    merge_fit_and_plot(args.source, args.output)


if __name__ == "__main__":
    main()
