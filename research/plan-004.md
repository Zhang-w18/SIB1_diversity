# Plan 004：复用Plan 003的单调BLER主曲线与10%/1%门限

状态：计划中

创建日期：2026-09-18

## 1. 本次实验目的

在模式A `fixed_radius_ls_normalized` 下生成可直接读取10%和1% BLER门限的主曲线。相对Plan 003，本实验不改变物理系统、接收机或CDD参数，而是改变统计设计和展示口径：

1. 复用Plan 003已验证的原始计数，不全量重跑；
2. 用绝对drop `1000–1999` 补成所有主SNR点共有的平衡样本层；
3. 在1%交越区用drop `2000–5999`追加精度；
4. 原始BLER、Wilson区间和单调约束估计同时保存，主线采用按blocks加权的二项单调回归，不改写原始误块数；
5. 只覆盖`−18～−6 dB`，不运行−5 dB及更高SNR。

P2-SSB与P6-SSB均进入本次主图和NMSE图。Plan 003中两者BLER几乎相同，本次仍保留两者，用于明确检查PRG=2与PRG=6对信道估计NMSE和最终BLER的影响。

## 2. 复用数据与新增工作量

Plan 003的以下输出作为冻结输入：

- `outputs/plan-003/run-001/estimated/`：drop 0–499；
- `outputs/plan-003/run-002/wide_2/`：drop 500–999；
- `outputs/plan-003/run-004/threshold/`：drop 1000–1499；
- `outputs/plan-003/run-004/core/`：drop 1000–1999。

废弃的`run-003/smooth`仍禁止纳入。新增批次如下：

| 批次 | SNR/dB | 新增drops/点 | 绝对drop区间 | SNR×drop |
|---|---|---:|---|---:|
| threshold-integer-fill | `[-18,-17,-16,-15]` | 500 | 1500–1999 | 2000 |
| threshold-half-grid | `[-17.5,-16.5,-15.5]` | 1000 | 1000–1999 | 3000 |
| tail-balance | `[-10.5,-9.5,-9,-8.5,-8,-7.5,-7,-6.5,-6]` | 1000 | 1000–1999 | 9000 |
| low-bler-refine | `[-8,-7.5,-7,-6.5,-6]` | 4000 | 2000–5999 | 20000 |

新增总量为34000个SNR×drop。完成后，`−18～−6 dB`全部主点至少具有共同的drop 1000–1999；`−8～−6 dB`具有共同的drop 1000–5999。Plan 003其他不重叠样本用于全量加权统计，但不改变平衡样本层的定义。

## 3. 仿真配置

### 3.1 环境与场景

| 参数 | 取值 |
|---|---|
| Python / TensorFlow / Sionna | 3.11.9 / 2.15.1 / 1.0.2 |
| 场景 | 7 GHz、3GPP TR 38.901 UMa下行、单UE/drop、单slot内信道不变 |
| 链路模式 | 模式A `fixed_radius_ls_normalized` |
| UE位置 | 水平半径100 m；方位−60°～60°；3GPP随机LOS/NLOS；禁止生成后筛drop |
| 大尺度处理 | 仅消除路径损耗/阴影衰落标量；保留LOS/NLOS、K因子、时延/角度扩展、cluster/ray和小尺度总能量起伏 |
| 参考增益 | $G_0=1.21373630034847\times10^{-9}$（−89.158756590132 dB） |
| 噪声 | $N_0=G_0 10^{-\gamma_0/10}$；禁止逐drop或逐方案重新归一化 |

### 3.2 阵列与码本

| 参数 | 取值 |
|---|---|
| BS AE/TXRU | 24V×16H×2 AE；4V×16H×2 TXRU；连续6V×1H等幅同相映射 |
| AE间距 | 水平0.5λ、垂直0.8λ |
| 面板 | 方位0°，机械下倾12°，无额外电下倾 |
| UE阵列 | 1V×2H双极化，共4Rx |
| SSB/secondary | 冻结8H×1V SSB和16H×1V secondary；每个selected SSB对应两个子波束 |
| 码本 | `outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz`，SHA-256 `3142396469049e843e8018bc5596152aa70d53e46d18b406879b52cf54638259` |
| selected SSB | 最大宽带平均接收功率，无选择错误模型 |

### 3.3 SIB1与接收机

30 kHz SCS、normal CP、FFT 2048、61.44 Msps；106 PRB BWP内居中48 PRB；PDSCH symbol 2–13；DMRS type-1位于symbol 2和11；rank-1 QPSK、MCS 0、TBS 1480、RV0。4Rx分别估计后MRC。

Baseline、Pol-cycling和Beam cycling使用2-PRB LMMSE窗口及SSB PDP；Beam CDD使用全48-PRB LMMSE窗口和含已知CDD时延的SSB先验。PDP、协方差及瞬时信道使用相同的模式A功率尺度。

### 3.4 对比方案

| 曲线 | 预编码 | PRG | 归一化 |
|---|---|---:|---|
| B-SSB | selected SSB固定双极化相位 | 2 PRB | 每活动子载波单位范数 |
| P2-SSB | 0°/90°极化相位逐PRG交替 | 2 PRB | 同上 |
| P6-SSB | 0°/90°极化相位每6 PRB交替 | 6 PRB | 同上 |
| BC-SSB | selected SSB的两个secondary beam逐PRG交替 | 2 PRB | 同上 |
| CDD-SSB | 两个secondary beam叠加CDD | 全带连续相位 | 同上 |

CDD使用$K=2$、活动带576点DFT网格索引`[0,1]`、循环移位`[0,3.5555555556]`采样，即`[0,57.87037037] ns`。相位为$v_m[q]=\exp(-j2\pi qj_m/576)$，$q=0,\ldots,575$；DMRS与数据使用相同预编码。

### 3.5 Monte Carlo与单调曲线

主种子为20260913。同一SNR内五条曲线共用UE、TB、信道和噪声；信道及UE由绝对drop索引决定，因此平衡层在不同SNR复用相同的drop集合。SNR随机流网格统一为起点−18 dB、步长0.5 dB。

每个批次固定运行规定drop数，不提前停止。每个原始点保存误块数、blocks、BLER和95% Wilson区间。最终分析先验证配置、环境、码本、SNR和绝对drop区间，再累加不重叠计数。

同时输出两张BLER图：`bler_raw.png`直接连接原始点并绘制95% Wilson区间；`bler_monotone.png`的实线为按blocks加权的二项分布最大似然非增单调回归，空心点仍显示未修改的原始BLER。10%与1%先报告相邻SNR括区，再在单调点之间按$\log_{10}(\mathrm{BLER})$线性插值给出点估计，输出`thresholds.csv`。原始BLER若出现小幅上跳必须保留，不得用单调值覆盖原始CSV。

## 4. 执行方式

配置文件：

- `configs/experiments/plan-004-threshold-integer-fill.yaml`
- `configs/experiments/plan-004-threshold-half-grid.yaml`
- `configs/experiments/plan-004-tail-balance.yaml`
- `configs/experiments/plan-004-low-bler-refine.yaml`

运行：

```powershell
.\run_plan004.ps1 -RunId run-001 -Batch all
```

也可用`-Batch`单独运行或恢复四个批次。全部完成后执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
py -3.11 -m sib1div.analysis.monotone_bler `
  --source outputs/plan-003/run-001/estimated `
  --source outputs/plan-003/run-002/wide_2 `
  --source outputs/plan-003/run-004/threshold `
  --source outputs/plan-003/run-004/core `
  --source outputs/plan-004/run-001/threshold-integer-fill `
  --source outputs/plan-004/run-001/threshold-half-grid `
  --source outputs/plan-004/run-001/tail-balance `
  --source outputs/plan-004/run-001/low-bler-refine `
  --output outputs/plan-004/result-004/monotone-main
```

## 5. 结果有效条件

- 四个新批次完成且没有NaN、协方差失败或异常终止；
- 四曲线在同一SNR共用相同drop、TB、信道和噪声；
- 每个活动子载波预编码单位范数；
- `−18～−6 dB`的21个主点均具有完整drop 1000–1999；`−8～−6 dB`的五点具有完整drop 1000–5999；
- 聚合器确认同一SNR没有重叠绝对drop区间，并保留来源清单；
- 每条曲线在10%和1%两侧均有采样点，否则不得给出该门限点估计；
- 主图单调实线、原始点、95%区间和门限CSV可互相追溯。

## 6. 预期输出

- `bler_monotone.csv`：原始聚合BLER、Wilson区间及单调估计；
- `thresholds.csv`：五条曲线的10%和1% BLER对应SNR；
- `bler_raw.png`：直接连接原始BLER点并显示95%区间；
- `bler_monotone.png`：只显示−18～−6 dB的单调门限主图；
- `nmse_merged.csv`和`nmse.png`：五条曲线的blocks加权NMSE；
- `nmse_pol_comparison.csv`：逐SNR列出P2/P6的NMSE、差值和比值；
- `analysis_manifest.json`：输入目录、元数据和方法；
- 各批次原始BLER/NMSE/配对计数、展开配置、环境、日志和完成元数据。
