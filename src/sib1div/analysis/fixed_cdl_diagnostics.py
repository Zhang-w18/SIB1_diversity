"""Long-term beam-power diagnostics for fixed-statistics CDL experiments."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from sib1div.channel import FixedCDLChannel
from sib1div.config import SimulationConfig
from sib1div.sim.engine import load_codebooks


def write_fixed_cdl_beam_diagnostics(
    config: SimulationConfig,
    codebook_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Estimate and save long-term RSRP for every SSB and secondary beam."""
    if config.data["run"]["link_mode"] != "fixed_cdl_statistics":
        raise ValueError("fixed-CDL beam diagnostics require fixed_cdl_statistics mode")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_hash = str(config.data["codebooks"]["weights_sha256"])
    ssb, secondary = load_codebooks(codebook_path, expected_hash)
    channel = FixedCDLChannel(config, ssb)
    families = {
        "ssb": channel.ssb_long_term_powers,
        "secondary": channel.long_term_beam_powers(secondary),
    }
    rows: list[dict[str, object]] = []
    for family, powers in families.items():
        peak = float(np.max(powers))
        order = np.argsort(-powers)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, powers.size + 1)
        for index, power in enumerate(powers):
            rows.append({
                "family": family,
                "beam_index": index,
                "parent_ssb": index // 2 if family == "secondary" else "",
                "power_linear": float(power),
                "power_db": float(10.0 * np.log10(max(float(power), 1e-300))),
                "relative_to_family_peak_db": float(
                    10.0 * np.log10(max(float(power), 1e-300) / peak)
                ),
                "rank_within_family": int(ranks[index]),
                "selected_parent_ssb": int(
                    family == "ssb" and index == channel.selected_ssb
                ),
            })
    csv_path = output / "beam_rsrp.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for axis, (family, powers) in zip(axes, families.items()):
        relative = 10.0 * np.log10(np.maximum(powers, 1e-300) / np.max(powers))
        colors = ["tab:orange" if (
            family == "ssb" and index == channel.selected_ssb
        ) else "tab:blue" for index in range(len(powers))]
        axis.bar(np.arange(len(powers)), relative, color=colors)
        axis.set(
            title=f"Long-term {family.upper()} RSRP",
            xlabel="Beam index", ylabel="Relative power (dB)",
            xticks=np.arange(len(powers)),
        )
        axis.grid(True, axis="y", alpha=0.3)
    fig.savefig(output / "beam_rsrp.png", dpi=180)
    plt.close(fig)

    secondary_powers = families["secondary"]
    children = [2 * channel.selected_ssb, 2 * channel.selected_ssb + 1]
    summary = {
        "schema_version": 1,
        "config": str(config.source),
        "codebook": str(Path(codebook_path).resolve()),
        "profile": str(config.data["fixed_cdl_statistics"]["profile"]),
        "delay_spread_s": float(config.data["fixed_cdl_statistics"]["delay_spread_s"]),
        "covariance_realizations": int(
            config.data["fixed_cdl_statistics"]["covariance_realizations"]
        ),
        "statistics_seed": int(config.data["fixed_cdl_statistics"]["statistics_seed"]),
        "selected_ssb": channel.selected_ssb,
        "selected_ssb_power": channel.reference_receive_power,
        "selected_ssb_children": children,
        "selected_ssb_children_power": [float(secondary_powers[i]) for i in children],
        "strongest_secondary": int(np.argmax(secondary_powers)),
        "angle_statistics": channel.angle_statistics,
    }
    (output / "beam_rsrp_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return csv_path
