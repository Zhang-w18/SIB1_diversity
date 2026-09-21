"""Merge completed fixed-drop simulation outputs without mixing drop ranges."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from sib1div.analysis.statistics import wilson_interval


MAIN_CURVES = ("B-SSB", "P2-SSB", "P6-SSB", "BC-SSB", "CDD-SSB")


def _bler_plot_lower_limit(values: list[float]) -> float:
    return max(1e-5, min(5e-3, min(values) / 2.0))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_runs(sources: list[tuple[str, Path]], output: Path) -> None:
    """Merge BLER, NMSE, and paired counts from disjoint absolute drop ranges."""
    output.mkdir(parents=True, exist_ok=True)
    bler_groups: dict[tuple[float, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    nmse_groups: dict[tuple[float, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)
    pair_groups: dict[tuple[float, str, str], list[tuple[str, dict[str, str]]]] = defaultdict(list)

    for source_name, source_dir in sources:
        for row in _read_csv(source_dir / "bler.csv"):
            if row["curve_id"] in MAIN_CURVES and row["curve_type"] == "estimated_csi":
                bler_groups[(float(row["snr_db"]), row["curve_id"])].append((source_name, row))
        for row in _read_csv(source_dir / "nmse.csv"):
            if row["curve_id"] in MAIN_CURVES:
                nmse_groups[(float(row["snr_db"]), row["curve_id"])].append((source_name, row))
        for row in _read_csv(source_dir / "paired_counts.csv"):
            if row["curve_a"] in MAIN_CURVES and row["curve_b"] in MAIN_CURVES:
                pair_groups[(float(row["snr_db"]), row["curve_a"], row["curve_b"])].append((source_name, row))

    bler_rows = []
    for (snr, curve), items in sorted(bler_groups.items()):
        first = items[0][1]
        errors = sum(int(row["block_errors"]) for _, row in items)
        blocks = sum(int(row["blocks"]) for _, row in items)
        low, high = wilson_interval(errors, blocks)
        bler_rows.append({
            "snr_db": snr, "curve_id": curve, "curve_type": "estimated_csi",
            "scheme": first["scheme"], "covariance": first["covariance"],
            "block_errors": errors, "blocks": blocks, "bler": errors / blocks,
            "wilson_low": low, "wilson_high": high,
            "source_runs": "+".join(name for name, _ in items),
        })

    nmse_rows = []
    for (snr, curve), items in sorted(nmse_groups.items()):
        first = items[0][1]
        blocks = sum(int(row["blocks"]) for _, row in items)
        weighted = sum(int(row["blocks"]) * float(row["mean_nmse"]) for _, row in items)
        nmse_rows.append({
            "snr_db": snr, "curve_id": curve, "scheme": first["scheme"],
            "covariance": first["covariance"], "blocks": blocks,
            "mean_nmse": weighted / blocks,
            "source_runs": "+".join(name for name, _ in items),
        })

    pair_rows = []
    for (snr, curve_a, curve_b), items in sorted(pair_groups.items()):
        totals = {key: sum(int(row[key]) for _, row in items) for key in ("blocks", "n00", "n01", "n10", "n11")}
        if totals["blocks"] != sum(totals[key] for key in ("n00", "n01", "n10", "n11")):
            raise ValueError(f"paired counts do not sum at {snr:g} dB: {curve_a}/{curve_b}")
        pair_rows.append({
            "snr_db": snr, "curve_a": curve_a, "curve_b": curve_b, **totals,
            "source_runs": "+".join(name for name, _ in items),
        })

    _write_csv(output / "bler.csv", list(bler_rows[0]), bler_rows)
    _write_csv(output / "nmse.csv", list(nmse_rows[0]), nmse_rows)
    _write_csv(output / "paired_counts.csv", list(pair_rows[0]), pair_rows)
    _write_plots(output, bler_rows, nmse_rows)

    manifest = {
        "schema_version": 1,
        "method": {
            "bler": "sum block_errors and blocks; recompute 95% Wilson interval",
            "nmse": "blocks-weighted mean",
            "paired_counts": "sum blocks and n00/n01/n10/n11",
        },
        "sources": [{"name": name, "path": str(path)} for name, path in sources],
        "excluded": ["outputs/plan-003/run-003/smooth"],
        "outputs": {},
    }
    for filename in ("bler.csv", "nmse.csv", "paired_counts.csv", "bler.png", "nmse.png"):
        payload = (output / filename).read_bytes()
        manifest["outputs"][filename] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    (output / "merge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_plots(output: Path, bler_rows: list[dict], nmse_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"B-SSB": "C0", "P2-SSB": "C1", "P6-SSB": "C2", "BC-SSB": "C3", "CDD-SSB": "C4"}
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for curve in MAIN_CURVES:
        selected = sorted((row for row in bler_rows if row["curve_id"] == curve), key=lambda row: row["snr_db"])
        x = [row["snr_db"] for row in selected]
        y = [row["bler"] if row["bler"] > 0 else row["wilson_high"] for row in selected]
        lower = [max(0.0, yy - row["wilson_low"]) for yy, row in zip(y, selected)]
        upper = [max(0.0, row["wilson_high"] - yy) for yy, row in zip(y, selected)]
        ax.errorbar(x, y, yerr=[lower, upper], marker="o", ms=4, capsize=2, lw=1.4,
                    color=colors[curve], label=curve)
    ax.axhline(1e-1, color="0.45", ls="--", lw=1, label="BLER = 0.1")
    ax.axhline(1e-2, color="0.45", ls=":", lw=1, label="BLER = 0.01")
    plotted = [float(row["bler"] if row["bler"] > 0 else row["wilson_high"]) for row in bler_rows]
    ax.set(xlabel="Normalized Es/N0 (dB)", ylabel="BLER",
           title="Plan 003 combined estimated-CSI BLER (95% Wilson CI)",
           ylim=(_bler_plot_lower_limit(plotted), 0.3))
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "bler.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for curve in MAIN_CURVES:
        selected = sorted((row for row in nmse_rows if row["curve_id"] == curve), key=lambda row: row["snr_db"])
        ax.semilogy([row["snr_db"] for row in selected], [row["mean_nmse"] for row in selected],
                    marker="o", ms=4, lw=1.4, color=colors[curve], label=curve)
    ax.set(xlabel="Normalized Es/N0 (dB)", ylabel="Mean channel-estimation NMSE",
           title="Plan 003 combined estimated-CSI NMSE")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "nmse.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    merge_runs([
        ("run-001/estimated", root / "outputs/plan-003/run-001/estimated"),
        ("run-002/wide_2", root / "outputs/plan-003/run-002/wide_2"),
        ("run-004/threshold", root / "outputs/plan-003/run-004/threshold"),
        ("run-004/core", root / "outputs/plan-003/run-004/core"),
    ], root / "outputs/plan-003/result-003/estimated-primary-combined")
