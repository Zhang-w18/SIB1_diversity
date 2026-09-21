import csv
from pathlib import Path

from sib1div.analysis.merge_fixed_runs import merge_runs


def _write(path: Path, name: str, header: list[str], rows: list[list[object]]) -> None:
    with (path / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_merge_fixed_runs_aggregates_counts_and_weighted_nmse(tmp_path: Path) -> None:
    sources = []
    for index, (errors, blocks, nmse, pair) in enumerate(((1, 10, 0.2, (7, 1, 2, 0)), (2, 20, 0.1, (16, 1, 1, 2)))):
        source = tmp_path / f"run-{index}"
        source.mkdir()
        _write(source, "bler.csv", ["snr_db", "curve_id", "curve_type", "scheme", "covariance", "block_errors", "blocks", "bler"],
               [[-10, "B-SSB", "estimated_csi", "baseline", "ssb_pdp", errors, blocks, errors / blocks]])
        _write(source, "nmse.csv", ["snr_db", "curve_id", "scheme", "covariance", "blocks", "mean_nmse"],
               [[-10, "B-SSB", "baseline", "ssb_pdp", blocks, nmse]])
        _write(source, "paired_counts.csv", ["snr_db", "curve_a", "curve_b", "blocks", "n00", "n01", "n10", "n11"],
               [[-10, "B-SSB", "P2-SSB", blocks, *pair]])
        sources.append((source.name, source))

    output = tmp_path / "combined"
    merge_runs(sources, output)
    bler = list(csv.DictReader((output / "bler.csv").open(encoding="utf-8")))[0]
    nmse = list(csv.DictReader((output / "nmse.csv").open(encoding="utf-8")))[0]
    paired = list(csv.DictReader((output / "paired_counts.csv").open(encoding="utf-8")))[0]
    assert (int(bler["block_errors"]), int(bler["blocks"]), float(bler["bler"])) == (3, 30, 0.1)
    assert abs(float(nmse["mean_nmse"]) - (4 / 30)) < 1e-12
    assert [int(paired[key]) for key in ("blocks", "n00", "n01", "n10", "n11")] == [30, 23, 2, 3, 2]
