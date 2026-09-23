"""Per-receive-branch effective-port RSRP distributions for fixed CDL."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sib1div.channel import FixedCDLChannel
from sib1div.config import SimulationConfig
from sib1div.schemes import build_precoder
from sib1div.sim.engine import load_codebooks


CURVES = (
    ("B-SSB", "baseline", 2), ("P2-SSB", "pol_cycling", 2),
    ("P6-SSB", "pol_cycling", 6), ("BC-SSB", "beam_cycling", 2),
    ("CDD-SSB", "beam_cdd", 2),
)


def effective_rx_powers(response: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return frequency-mean |H w|^2 for each physical receive branch."""
    if response.ndim != 3:
        raise ValueError("RSRP CDF requires one time sample with response [rx,subcarrier,tx]")
    effective = np.einsum("rkt,tk->rk", response, weights, optimize=True)
    return np.mean(np.abs(effective) ** 2, axis=1).real


def write_rsrp_cdfs(
    config: SimulationConfig, codebook_path: str | Path, output_dir: str | Path,
    *, drop_start: int = 2000, drops: int = 10000,
) -> Path:
    if config.data["run"]["link_mode"] != "fixed_cdl_statistics":
        raise ValueError("RSRP CDF requires fixed_cdl_statistics mode")
    if drops < 1 or drop_start < 0:
        raise ValueError("drop range must be non-negative and non-empty")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_metadata = output / "run_metadata.json"
    checkpoint = output / ".rsrp_checkpoint.npz"
    if final_metadata.exists() and not checkpoint.exists():
        raise FileExistsError(f"completed RSRP CDF run already exists: {final_metadata}")
    expected = str(config.data["codebooks"]["weights_sha256"])
    ssb, secondary = load_codebooks(codebook_path, expected)
    channel = FixedCDLChannel(config, ssb)
    precoders = {
        curve: build_precoder(config, scheme, channel.selected_ssb, ssb, secondary, prg_size_prbs=prg).weights
        for curve, scheme, prg in CURVES
    }
    spacing = float(config.data["nr"]["subcarrier_spacing_hz"])
    shape = (drops, int(config.data["receiver"]["rx_branches"]))
    if checkpoint.exists():
        saved = np.load(checkpoint)
        if int(saved["drop_start"]) != drop_start or int(saved["drops"]) != drops:
            raise RuntimeError("RSRP checkpoint does not match the requested drop range")
        completed = int(saved["completed"])
        values = {curve: np.asarray(saved[curve]).copy() for curve, _, _ in CURVES}
    else:
        completed = 0
        values = {curve: np.empty(shape) for curve, _, _ in CURVES}
        np.savez(checkpoint, drop_start=drop_start, drops=drops, completed=0, **values)
    for offset in range(completed, drops):
        drop_index = drop_start + offset
        response = channel.generate(drop_index).frequency_response(config.active_subcarriers, spacing)
        for curve, _, _ in CURVES:
            powers = effective_rx_powers(response, precoders[curve])
            values[curve][offset] = powers
        if (offset + 1) % 100 == 0:
            np.savez(checkpoint, drop_start=drop_start, drops=drops, completed=offset+1, **values)
            print(f"RSRP CDF drops={offset+1}/{drops}", flush=True)
    rows = []
    for offset in range(drops):
        for curve, _, _ in CURVES:
            for rx, power in enumerate(values[curve][offset]):
                rows.append({
                    "drop_index": drop_start + offset, "rx_index": rx, "curve_id": curve,
                    "power_linear": float(power),
                    "power_dbref": float(10*np.log10(max(float(power), 1e-300)/channel.reference_receive_power)),
                })
    raw_path = output / "rsrp_per_rx.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = []
    percentiles = (1, 5, 10, 50, 90, 95, 99)
    for curve, _, _ in CURVES:
        db = 10*np.log10(np.maximum(values[curve], 1e-300)/channel.reference_receive_power)
        for rx in range(db.shape[1]):
            item = {"rx_index": rx, "curve_id": curve, "samples": drops,
                    "mean_dbref": float(np.mean(db[:, rx])), "std_db": float(np.std(db[:, rx]))}
            item.update({f"p{p:02d}_dbref": float(np.percentile(db[:, rx], p)) for p in percentiles})
            summary.append(item)
    with (output / "rsrp_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for rx in range(int(config.data["receiver"]["rx_branches"])):
        fig, ax = plt.subplots(figsize=(9, 6))
        for curve, _, _ in CURVES:
            x = np.sort(10*np.log10(np.maximum(values[curve][:, rx], 1e-300)/channel.reference_receive_power))
            y = (np.arange(drops) + 0.5) / drops
            ax.plot(x, y, lw=1.5, label=curve)
        ax.set(xlabel="Per-Rx effective-port RSRP / P_ref (dB)", ylabel="Empirical CDF",
               title=f"Plan 005 effective-port RSRP CDF — Rx {rx}", ylim=(0, 1))
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        fig.savefig(output / f"rsrp_cdf_rx{rx}.png", dpi=180); plt.close(fig)
    metadata = {
        "schema_version": 1, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.source), "config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "codebook_sha256": expected, "drop_index_start": drop_start,
        "drop_index_end": drop_start+drops-1, "drops": drops,
        "selected_ssb": channel.selected_ssb, "reference_receive_power": channel.reference_receive_power,
        "definition": "mean over 576 active subcarriers of abs(H_rx,k times w_k)^2",
    }
    final_metadata.write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    checkpoint.unlink()
    return raw_path
