# Plan 003：固定半径CSI与Pol-PRG对比

状态：未完成　创建：2026-09-15　配置：[YAML](../configs/experiments/plan-003.yaml)

## 1. 目标

以 `fixed_radius_ls_normalized` 重跑Plan001的estimated/ideal CSI，并增加Pol-cycling PRG=6。固定半径100 m，仅消除路径损耗/阴影标量；两阶段分目录运行。

## 2. 配置

| 项目 | 取值 |
|---|---|
| 环境/场景 | Python 3.11.9 / TF 2.15.1 / Sionna 1.0.2；7 GHz UMa下行；单UE/drop、100 m、方位±60°、3GPP随机LOS/NLOS |
| 链路口径 | $G_0=1.21373630034847\times10^{-9}$（−89.1587566 dB，100 m LOS无阴影）；信道/协方差乘$G_0/G_d$，$N_0=G_010^{-\gamma_0/10}$，禁止逐drop/方案再归一化 |
| 阵列/码本 | BS 24V×16H×2 AE、4V×16H×2 TXRU，H/V间距0.5/0.8λ，6V×1H映射；下倾12°；UE 1V×2H双极化；冻结8H SSB/16H secondary码本，最大宽带功率选SSB |
| SIB1 | 30 kHz，FFT2048/61.44 Msps；106 BWP内居中48 PRB、symbol 2–13；DMRS type-1在2/11；rank-1 QPSK、MCS0、TBS1480、RV0；4Rx MRC |

Estimated为 `B-SSB/P2-SSB/P6-SSB/BC-SSB/BC-BEAM/CDD-SSB/CDD-BEAM/CDD-IDEAL`。B/P/BC用2-PRB LMMSE；P2/P6按2/6 PRB切换0°/90°；BC用SSB/per-beam PDP；CDD全48 PRB用SSB、per-beam或ideal joint covariance。输出4Rx MRC BLER和NMSE。

Ideal为 `PERF-B/P2/P6/BC/CDD`。CDD双子波束索引[0,1]，延时[0,57.870370 ns]=[0,3.5555556]采样，$v_m[q]=e^{-j2\pi qj_m/576}$。全部逐子载波单位范数，DMRS与数据同预编码。

## 3. 运行

主种子20260913；同SNR共用UE、TB、信道、噪声。Estimated为−11:0.5:−8.5 dB，ideal为−12:0.5:−9 dB；每点500 drops（0–499），固定停止，95% Wilson区间。

```powershell
.\run_plan003.ps1 -RunId run-001 -Phase estimated
.\run_plan003.ps1 -RunId run-001 -Phase ideal
```

输出为 `outputs/plan-003/run-001/{estimated,ideal}/`，可独立恢复。保存配置、环境、日志、CSV和图。

## 4. 稀疏宽扫追加（2026-09-16）

原estimated未跨过BLER 0.1/0.01；ideal暂停。追加配置：[plan-003-wide-scan-01.yaml](../configs/experiments/plan-003-wide-scan-01.yaml)，仅运行 `B-SSB/BC-SSB/P2-SSB/P6-SSB/CDD-SSB`。SNR为 `[-14,-13,-12,-6,-5,-4,-2,0,2]` dB，每点500个公共drop；使用不重叠绝对drop 500–999，主种子不变，固定停止及95% Wilson区间不变。输出 `outputs/plan-003/run-002/wide/`。

```powershell
.\run_plan003.ps1 -RunId run-002 -Phase wide
```

## 5. 有效性

测试、配置、码本、公共样本与预编码范数正确；无NaN/Inf/协方差失败；计数、区间、NMSE和图可追溯，否则result标“未完成”。

## 6. 平滑曲线追加（2026-09-17）

为降低BLER约0.05处由500 drops导致的蒙特卡洛波动，并在−14 dB以下找到BLER 0.1交点，增加两个可独立恢复的批次。物理系统、接收机、五条estimated-CSI主曲线、主种子、链路归一化及统计口径均与“稀疏宽扫”相同。

| 批次 | 配置 | SNR/dB | 新增drops/点 | 绝对drop区间 | 输出 |
|---|---|---|---:|---|---|
| threshold | [plan-003-threshold-scan-01.yaml](../configs/experiments/plan-003-threshold-scan-01.yaml) | `[-18,-17,-16,-15]` | 500 | 1000–1499 | `outputs/plan-003/run-004/threshold/` |
| core | [plan-003-smooth-core-01.yaml](../configs/experiments/plan-003-smooth-core-01.yaml) | `[-14,-13,-12,-11,-10]` | 1000 | 1000–1999 | `outputs/plan-003/run-004/core/` |

两个批次共7000个SNR×drop，按`wide_2`实测吞吐预计约12小时。core点与已有500 drops汇总后为1500 drops/点；在BLER 0.05处，95%二项抽样半宽预计由约0.019降至约0.011。各批次内部及两个批次之间使用相同的新drop索引前缀，因此在共同索引1000–1499上共用UE、LOS/NLOS状态和小尺度信道；不同SNR使用各自的TB与单位方差噪声流。

运行器每10 drops输出当前SNR计数、整体完成百分比、平均drops/s、已用时间、预计剩余时间和UTC预计完成时刻。推荐在另一台复刻环境的机器上运行：

```powershell
.\run_plan003_append.ps1 -RunId run-004 -Batch all
```

也可以分别运行或恢复：

```powershell
.\run_plan003_append.ps1 -RunId run-004 -Batch threshold
.\run_plan003_append.ps1 -RunId run-004 -Batch core
```

有效性条件为：threshold的4个SNR点均完成500 drops，core的5个SNR点均完成1000 drops；CSV与配对计数完整且有限；每个配对计数分别合计500或1000；运行时预编码单位范数断言通过。

已废弃的初版大批次配置`plan-003-smooth-scan-01.yaml`原计划11点×4000 drops，预计约77小时；本机试运行在−18 dB完成40/4000 drops后停止，checkpoint位于`outputs/plan-003/run-003/smooth/`，不得与本次run-004统计混合。
