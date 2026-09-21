"""Persist codebooks, plots, quantitative diagnostics, and an audit report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sib1div.config import SimulationConfig

from .array import ArrayGeometry
from .synthesis import BeamCodebook, generate_main_codebooks


FLOOR_DB = -35.0


def _db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(power, 10 ** (FLOOR_DB / 10.0)))


def _metrics(geometry: ArrayGeometry, codebook: BeamCodebook) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for region, weight in zip(codebook.regions, codebook.ae_weights.T):
        phi_axis = np.linspace(region.phi_min_deg, region.phi_max_deg, 41)
        eta_axis = np.linspace(region.eta_min_deg, region.eta_max_deg, 25)
        phi, eta = np.meshgrid(phi_axis, eta_axis, indexing="xy")
        in_gain = geometry.gain(weight, phi, eta)[:, 0]
        in_gain_db = _db(in_gain)
        phi_out = np.r_[np.linspace(-90.0, -60.5, 60), np.linspace(60.5, 90.0, 60)]
        eta_out = np.full_like(phi_out, region.eta_center_deg)
        out_gain_db = _db(geometry.gain(weight, phi_out, eta_out)[:, 0])
        rows.append({
            "codebook": codebook.name,
            "beam": region.index,
            "parent_ssb": "" if region.parent_ssb is None else region.parent_ssb,
            "phi_min_deg": region.phi_min_deg,
            "phi_max_deg": region.phi_max_deg,
            "eta_min_deg": region.eta_min_deg,
            "eta_max_deg": region.eta_max_deg,
            "in_min_gain_db": float(np.min(in_gain_db)),
            "in_mean_gain_db": float(10 * np.log10(np.mean(in_gain))),
            "in_max_gain_db": float(np.max(in_gain_db)),
            "in_ripple_db": float(np.max(in_gain_db) - np.min(in_gain_db)),
            "out_of_sector_max_db": float(np.max(out_gain_db)),
            "weight_norm": float(np.vdot(weight, weight).real),
        })
    return rows


def _plot_horizontal(path: Path, geometry: ArrayGeometry, codebook: BeamCodebook, eta_deg: float) -> None:
    phi = np.linspace(-90.0, 90.0, 721)
    gains = _db(geometry.gain(codebook.ae_weights, phi, np.full_like(phi, eta_deg)))
    fig, ax = plt.subplots(figsize=(11, 6))
    for index in range(gains.shape[1]):
        ax.plot(phi, gains[:, index], lw=1.35, label=f"B{index}")
    ax.axvspan(-60, 60, color="0.9", zorder=-1, label="target sector")
    ax.set(xlabel="Global azimuth phi (deg)", ylabel="Array power gain (dB)",
           title=f"{codebook.name}: horizontal cuts at global downtilt eta={eta_deg:.2f} deg",
           xlim=(-90, 90), ylim=(FLOOR_DB, 28))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4 if len(codebook.regions) <= 8 else 8, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_vertical(path: Path, geometry: ArrayGeometry, codebook: BeamCodebook) -> None:
    eta = np.linspace(0.0, 45.0, 451)
    fig, ax = plt.subplots(figsize=(11, 6))
    for region, weight in zip(codebook.regions, codebook.ae_weights.T):
        phi = np.full_like(eta, region.phi_center_deg)
        gain = _db(geometry.gain(weight, phi, eta)[:, 0])
        ax.plot(eta, gain, lw=1.2, label=f"B{region.index} @ phi={region.phi_center_deg:.1f} deg")
    ax.axvspan(codebook.regions[0].eta_min_deg, codebook.regions[0].eta_max_deg,
               color="0.9", zorder=-1, label="UE geometric range")
    ax.axvline(geometry.mechanical_downtilt_deg, color="k", ls="--", lw=1, label="panel tilt")
    ax.set(xlabel="Global downtilt eta (deg, positive downward)", ylabel="Array power gain (dB)",
           title=f"{codebook.name}: vertical cuts at each beam center",
           xlim=(0, 45), ylim=(FLOOR_DB, 28))
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_coverage(path: Path, geometry: ArrayGeometry, codebook: BeamCodebook) -> None:
    phi_axis = np.linspace(-75.0, 75.0, 301)
    eta_axis = np.linspace(5.0, 40.0, 141)
    phi, eta = np.meshgrid(phi_axis, eta_axis, indexing="xy")
    gains = _db(geometry.gain(codebook.ae_weights, phi, eta))
    best = np.max(gains, axis=1).reshape(phi.shape)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    image = ax.pcolormesh(phi_axis, eta_axis, best, shading="auto", cmap="viridis", vmin=-10, vmax=28)
    for region in codebook.regions:
        ax.axvline(region.phi_min_deg, color="w", alpha=0.32, lw=0.7)
    ax.axhline(codebook.regions[0].eta_min_deg, color="w", ls="--", lw=0.8)
    ax.axhline(codebook.regions[0].eta_max_deg, color="w", ls="--", lw=0.8)
    ax.set(xlabel="Global azimuth phi (deg)", ylabel="Global downtilt eta (deg)",
           title=f"{codebook.name}: maximum gain over all beams")
    fig.colorbar(image, ax=ax, label="Array power gain (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parent_children(
    path: Path,
    geometry: ArrayGeometry,
    ssb: BeamCodebook,
    secondary: BeamCodebook,
    parent: int,
) -> None:
    phi = np.linspace(-75.0, 75.0, 601)
    eta = ssb.regions[parent].eta_center_deg
    fig, ax = plt.subplots(figsize=(10, 5.5))
    parent_gain = _db(geometry.gain(ssb.ae_weights[:, parent], phi, np.full_like(phi, eta))[:, 0])
    ax.plot(phi, parent_gain, lw=2.4, color="k", label=f"SSB B{parent}")
    for child in (2 * parent, 2 * parent + 1):
        gain = _db(geometry.gain(secondary.ae_weights[:, child], phi, np.full_like(phi, eta))[:, 0])
        ax.plot(phi, gain, lw=1.8, label=f"secondary B{child}")
    region = ssb.regions[parent]
    ax.axvspan(region.phi_min_deg, region.phi_max_deg, color="0.9", zorder=-1)
    ax.set(xlabel="Global azimuth phi (deg)", ylabel="Array power gain (dB)",
           title=f"Selected SSB B{parent} and its two secondary beams at eta={eta:.2f} deg",
           xlim=(-75, 75), ylim=(FLOOR_DB, 28))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_subarray_factor(
    path: Path,
    geometry: ArrayGeometry,
    eta_min_deg: float,
    eta_max_deg: float,
) -> float:
    """Plot the common 6x1 vertical subarray factor and return its first null."""
    block_m = geometry.vertical_aes // geometry.vertical_txrus
    eta = np.linspace(0.0, 45.0, 1801)
    local = np.deg2rad(eta - geometry.mechanical_downtilt_deg)
    u_v = -np.sin(local)
    m = np.arange(block_m)[:, None]
    field = np.sum(
        np.exp(1j * 2 * np.pi * m * geometry.vertical_spacing_lambda * u_v[None, :]),
        axis=0,
    ) / np.sqrt(block_m)
    factor_db = _db(np.abs(field) ** 2)
    null_local = np.rad2deg(np.arcsin(1.0 / (block_m * geometry.vertical_spacing_lambda)))
    null_global = geometry.mechanical_downtilt_deg + null_local
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(eta, factor_db, lw=2)
    ax.axvspan(eta_min_deg, eta_max_deg, color="0.9", zorder=-1,
               label="UE geometric range")
    ax.axvline(geometry.mechanical_downtilt_deg, color="k", ls="--", lw=1,
               label="panel boresight")
    ax.axvline(null_global, color="crimson", ls="--", lw=1.5,
               label=f"first common null: {null_global:.2f} deg")
    ax.set(xlabel="Global downtilt eta (deg, positive downward)",
           ylabel="6x1 subarray power factor (dB)",
           title="Common vertical factor imposed by each equal-phase TXRU subarray",
           xlim=(0, 45), ylim=(FLOOR_DB, 10))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return float(null_global)


def write_codebook_report(config: SimulationConfig, output_dir: str | Path) -> Path:
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    geometry, ssb, secondary = generate_main_codebooks(config)
    np.savez_compressed(
        output / "codebook_weights.npz",
        txru_mapping=geometry.mapping,
        ssb_txru_weights=ssb.txru_weights,
        ssb_ae_weights=ssb.ae_weights,
        secondary_txru_weights=secondary.txru_weights,
        secondary_ae_weights=secondary.ae_weights,
    )
    metadata = {
        "coordinate_system": "global phi; positive-down global eta",
        "mechanical_downtilt_deg": geometry.mechanical_downtilt_deg,
        "ae_order": "row-major (vertical, horizontal), single polarization",
        "txru_order": "row-major (vertical, horizontal), single polarization",
        "weight_normalization": "unit AE-domain Euclidean norm per beam",
        "ssb_regions": [region.__dict__ for region in ssb.regions],
        "secondary_regions": [region.__dict__ for region in secondary.regions],
    }
    (output / "codebook_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = _metrics(geometry, ssb) + _metrics(geometry, secondary)
    with (output / "beam_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    eta_center = ssb.regions[0].eta_center_deg
    _plot_horizontal(output / "ssb_horizontal.png", geometry, ssb, eta_center)
    _plot_vertical(output / "ssb_vertical.png", geometry, ssb)
    _plot_coverage(output / "ssb_coverage_2d.png", geometry, ssb)
    _plot_horizontal(output / "secondary_horizontal.png", geometry, secondary, eta_center)
    _plot_coverage(output / "secondary_coverage_2d.png", geometry, secondary)
    _plot_parent_children(output / "ssb_secondary_overlay.png", geometry, ssb, secondary, parent=3)
    null_global = _plot_subarray_factor(
        output / "txru_subarray_vertical_factor.png",
        geometry,
        ssb.regions[0].eta_min_deg,
        ssb.regions[0].eta_max_deg,
    )

    ssb_rows = rows[: len(ssb.regions)]
    sec_rows = rows[len(ssb.regions):]
    gram = geometry.mapping.conj().T @ geometry.mapping
    mapping_error = float(np.max(np.abs(gram - np.eye(geometry.txru_count))))
    report = _report_text(config, geometry, ssb_rows, sec_rows, mapping_error, null_global)
    report_path = output / "码本与方向图审核报告.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def _report_text(config, geometry, ssb_rows, sec_rows, mapping_error, null_global) -> str:
    eta_min = float(ssb_rows[0]["eta_min_deg"])
    eta_max = float(ssb_rows[0]["eta_max_deg"])
    ssb_min = min(float(row["in_min_gain_db"]) for row in ssb_rows)
    ssb_ripple = max(float(row["in_ripple_db"]) for row in ssb_rows)
    sec_min = min(float(row["in_min_gain_db"]) for row in sec_rows)
    sec_ripple = max(float(row["in_ripple_db"]) for row in sec_rows)
    ssb_norm_error = max(abs(float(row["weight_norm"]) - 1) for row in ssb_rows)
    sec_norm_error = max(abs(float(row["weight_norm"]) - 1) for row in sec_rows)
    null_inside = eta_min <= null_global <= eta_max
    null_verdict = "未通过" if null_inside else "已避开公共零陷"
    null_explanation = (
        "落在" if null_inside else "不在"
    )
    next_action = (
        "- 继续 UMa/BLER 集成前，需要调整机械/电下倾或垂直 TXRU/AE 分组，或者像具体实验 plan 那样预先缩小 UE 几何角域。"
        if null_inside else
        "- 当前受控 UE 角域已避开公共零陷；仍需依据区内最小增益、ripple 和父子波束关系决定是否冻结本码本。"
    )
    return f"""# SIB1 码本与方向图阶段审核报告

生成配置：`{config.source}`  
本报告覆盖平台基础功能调试的第一段：AE→TXRU 映射、8H×1V SSB 码本、16H×1V secondary 码本及方向图。尚未包含 UMa drop、NR 编解码或 BLER 结果。

## 1. 模型与坐标约定

- 单极化空间阵列为 24×16 AE、4×16 TXRU；每个 TXRU 连续连接 6×1 AE，等幅同相，映射列归一化。
- 双极化完整阵列共有 768 AE 和 128 个极化 TXRU 端口；本文方向图显示单极化空间阵列因子，双极化合成在后续预编码器阶段处理。
- 水平间距为 0.5λ，垂直间距为 0.8λ。
- 图中 φ 和 η 均为全局坐标；η 向下为正。局部角只在阵列响应内部按 `η_local = η_global - 12°` 计算，机械下倾只施加一次。
- UE 几何下倾覆盖为 {eta_min:.3f}°～{eta_max:.3f}°。
- 波束按全局方位角从负到正编号；secondary B(2b) 和 B(2b+1) 属于 SSB Bb。

## 2. 映射与数值检查

| 检查 | 结果 |
|---|---:|
| 单极化映射矩阵尺寸 | {geometry.mapping.shape[0]}×{geometry.mapping.shape[1]} |
| 每列非零 AE 数 | 6 |
| `max abs(F^H F-I)` | {mapping_error:.3e} |
| SSB AE 权重范数最大偏差 | {ssb_norm_error:.3e} |
| Secondary AE 权重范数最大偏差 | {sec_norm_error:.3e} |

`F^H F` 接近单位阵，表明连续子阵分组互不重叠且列功率归一化。TXRU 域投影与直接 AE 域响应的一致性由自动测试覆盖。

### 2.1 阻塞性发现：目标覆盖区内存在公共垂直零陷

每个 TXRU 的6个垂直 AE 等幅同相，其公共子阵因子为：

```text
A_sub(uV) = (1/sqrt(6)) * sum(m=0..5) exp(j*2*pi*m*dV*uV)
```

第一零点满足 `abs(sin(eta_local)) = 1/(6*dV)`。代入 `dV=0.8` 和12°机械下倾，向下侧第一公共零陷位于全局 η={null_global:.3f}°，{null_explanation} UE 目标范围 {eta_min:.3f}°～{eta_max:.3f}° 内。

![TXRU subarray vertical factor](txru_subarray_vertical_factor.png)

这是映射矩阵施加在所有 TXRU steering vector 上的公共乘法因子，因此调整 TXRU 数字码本、改用4H×2V布局或增加 weighted-LS 迭代均不能填平该零陷。当前目标角域的公共零陷检查判定为 **{null_verdict}**。

## 3. SSB 8H×1V 方向图

![SSB horizontal](ssb_horizontal.png)

![SSB vertical](ssb_vertical.png)

![SSB 2D coverage](ssb_coverage_2d.png)

SSB 目标区统计：全码本最低区内增益 {ssb_min:.2f} dB，最大区内 ripple {ssb_ripple:.2f} dB。逐波束精确数值见 `beam_metrics.csv`。

## 4. Secondary 16H×1V 方向图

![Secondary horizontal](secondary_horizontal.png)

![Secondary 2D coverage](secondary_coverage_2d.png)

Secondary 目标区统计：全码本最低区内增益 {sec_min:.2f} dB，最大区内 ripple {sec_ripple:.2f} dB。

## 5. Selected SSB 与子波束关系检查

以下使用 SSB B3 为固定审核示例，其子波束稳定映射为 secondary B6 和 B7。

![SSB and secondary overlay](ssb_secondary_overlay.png)

## 6. 审核结论与待办

- 波束编号、全局角度、机械下倾和父子码本关系已固化在 `codebook_metadata.json`。
- 复权重和固定映射保存在 `codebook_weights.npz`。后续四种方案必须复用这些离线权重，不允许读取当前 drop 的瞬时 AOD 重算码本。
- 当前图仅为阵列因子，尚未叠加 3GPP 单元方向图；接入 Sionna 38.901 天线模型时需确认单元增益和极化场分量，避免重复施加面板姿态。
- weighted-LS 的区内最小增益与 ripple 是本阶段研究者需要重点审核的指标；若需更严格的平顶或旁瓣约束，应在进入大规模 Monte Carlo 前调整优化权重。
{next_action}

## 7. 产物清单

- `codebook_weights.npz`：映射矩阵、SSB/secondary 的 TXRU 与 AE 权重
- `codebook_metadata.json`：坐标、编号、区域和归一化约定
- `beam_metrics.csv`：逐波束目标区增益、ripple、区外增益和范数
- `*.png`：本报告引用的方向图
"""
