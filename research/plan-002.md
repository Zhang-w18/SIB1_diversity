# Plan 002：−10/−12 dB固定SNR粗扫

状态：已完成

创建日期：2026-09-14

配置文件：[plan-002.yaml](../configs/experiments/plan-002.yaml)

## 1. 本次实验目的

在不修改或覆盖Plan 001 `run-001` checkpoint的前提下，快速取得−10 dB和−12 dB两点，判断七条estimated-CSI曲线是否已经跨过BLER $10^{-1}$，并据此决定后续是否需要−14 dB以及应补充哪些1 dB采样点。

相对Plan 001只改变SNR调度和统计停止口径：由自适应联合停止改为固定 `[-10,-12] dB`、每点500个公共drop。物理系统、码本、四种方案、接收机、主种子和drop起点均不变。结果定位为粗扫预览，不能替代Plan 001在目标交越附近的最终统计。

## 2. 仿真配置

### 2.1 环境、场景与链路

| 参数 | 取值 |
|---|---|
| Python / TensorFlow / Sionna | 3.11.9 / 2.15.1 / 1.0.2（系统安装） |
| 场景与方向 | 3GPP TR 38.901 UMa，下行，单UE/drop，单slot内静止 |
| 载频与链路模式 | 7 GHz，`normalized_link` |
| UE位置 | 75～115 m按扇区面积均匀；方位−60°～60°；生成信道后禁止筛drop |
| LOS/NLOS | Sionna UMa按3GPP概率产生 |
| 路径损耗/阴影衰落 | 生成后由公共selected-SSB逐drop归一化消除 |
| 横轴 | 公共归一化后的每发射RE $E_s/N_0$ |

### 2.2 阵列与码本

| 参数 | 取值 |
|---|---|
| BS AE/TXRU | 24V×16H×2极化AE；4V×16H×2极化TXRU，共128端口 |
| AE间距与映射 | H/V为0.5/0.8 λ；每TXRU连接连续6V×1H AE，等幅同相、列单位范数 |
| 面板 | 方位0°，机械下倾12°，额外电下倾0° |
| UE阵列 | 1V×2H双极化，4Rx |
| SSB / secondary码本 | 8H×1V / 16H×1V；每selected SSB对应两个子波束 |
| 冻结权重 | `outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz` |
| Selected SSB | 8个SSB中瞬时宽带平均接收功率最大者，无选择错误 |

### 2.3 SIB1与接收机

| 参数 | 取值 |
|---|---|
| SCS/CP/FFT/采样率 | 30 kHz / normal / 2048 / 61.44 Msps |
| BWP/PDSCH | 106/48 PRB，居中分配；symbol 2～13 |
| DMRS | type 1，symbol `[2,11]`，6 RE/PRB/symbol，端口1000 |
| MCS/TBS/RV | Rank-1 QPSK，MCS 0，120/1024，1480 bit，RV 0 |
| PRG与接收 | 2 PRB；4Rx MRC |
| LMMSE窗口 | B/P/BC为2 PRB；CDD为全部48 PRB |
| 主曲线 | `B-SSB`、`P-SSB`、`BC-SSB`、`BC-BEAM`、`CDD-SSB`、`CDD-BEAM`、`CDD-IDEAL` |
| 诊断曲线 | 四种方案各一条Perfect-CSI |

### 2.4 对比方案与CDD

| 方案 | 预编码 | 归一化 |
|---|---|---|
| Baseline | selected SSB双极化Rank-1，极化相位0° | 每占用子载波单位范数 |
| Pol-cycling | selected SSB，极化相位0°/90°逐PRG交替 | 同上 |
| Beam cycling | 两个secondary beam逐PRG交替 | 同上 |
| Beam CDD | 两个secondary beam同时发送，频域线性相位循环移位 | 同上 |

Beam CDD固定使用2个波束、有效带宽DFT栅格索引 `[0,1]`，等效FFT采样点 `[0,3.5555556]`，时间 `[0,57.870370 ns]`。对局部子载波 $q=0,\ldots,575$，相位为 $e^{-j2\pi qj_m/576}=e^{-j2\pi qd_m/2048}$；复合预编码后逐子载波归一化。

### 2.5 Monte Carlo

| 参数 | 取值 |
|---|---|
| SNR点与顺序 | `[-10,-12] dB` |
| 每点公共drop数 | 固定500 |
| 主种子/随机流 | `20260913` / `plan002_fixed_coarse_v1` |
| drop绝对范围 | 每个SNR均为0～499；不同SNR使用独立噪声流 |
| 方案间公共样本 | UE drop、TB、物理信道、单位方差噪声 |
| 停止条件 | 每点恰好500个公共drop；不按误块数提前停止 |
| 置信区间 | 95% Wilson |

### 2.6 追加半 dB 扫描（2026-09-15）

原始 `run-001` 已按初始配置完成−10 dB和−12 dB、每点500个公共drop；其实际展开配置保存在 `outputs/plan-002/run-001/resolved_config.yaml`，结果不修改、不覆盖。

为细化−12～−8 dB过渡区，在同一Plan 002内增加一次独立运行：

| 参数 | 追加运行取值 |
|---|---|
| Run ID | `run-002` |
| SNR点与顺序 | `[-11.5,-11,-10.5,-9.5,-8.5] dB` |
| 每点公共drop数 | 固定500 |
| 主种子 | `20260913` |
| 随机流命名空间 | `plan002_half_db_extension_v2` |
| SNR随机流映射 | 以−12 dB为原点、0.5 dB为步长，索引为 `[1,2,3,5,7]`，各点不碰撞 |
| drop范围 | 每个SNR均为0～499 |
| 停止条件 | 每点恰好500个公共drop，不按错误数提前停止 |
| 输出目录 | `outputs/plan-002/run-002/` |

本次只增加SNR采样点，不改变物理系统、码本、方案、CDD时延、接收机、归一化或统计口径。原始 `run-001` 与追加 `run-002` 必须依据各自保存的 `resolved_config.yaml` 追溯；禁止用新配置重建或覆盖 `run-001`。

## 3. 执行方式

```powershell
py -3.11 -m sib1div.cli validate-config configs/experiments/plan-002.yaml --output outputs/plan-002/config_validation
.\run_plan002.ps1 -RunId run-001
```

追加半 dB 扫描命令：

```powershell
.\run_plan002.ps1 -RunId run-002
```

正式输出为 `outputs/plan-002/<run_id>/`。运行每10 drops原子更新 `.fixed_checkpoint.json`；同一命令和RunId可恢复。它只读取Plan 001冻结码本，不读取或修改 `outputs/plan-001/run-001/`。

## 4. 结果有效条件

- [x] 配置验证和相关测试通过。
- [x] 每点完成500个公共drop。
- [x] 同一SNR内七条曲线使用相同UE、TB、信道和噪声。
- [x] DMRS与同PRG数据使用相同预编码，所有占用子载波预编码范数为1。
- [x] 无NaN、Inf、协方差求解或统计记录异常。
- [x] CSV、图和Wilson区间均来自实际计数。

## 5. 预期输出

`bler.csv`、`nmse.csv`、`paired_counts.csv`、`bler.png`、`bler_perfect_csi.png`、`nmse.png`、`resolved_config.yaml`、`environment.json`、`run_metadata.json`和`run.log`。
