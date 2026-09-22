# Plan 005：固定统计 CDL-C、100 ns、10° ASD 四方案 10% BLER 比较

状态：已完成

创建日期：2026-09-21

## 1. 本次实验目的

把 Plan 004 的物理层、码本、四种传输方案和 LMMSE 接收机迁移到模式 C `fixed_cdl_statistics`，在不引入 UE 移动、几何 drop、路径损耗或阴影衰落变化的条件下，比较 Baseline、Pol-cycling、Beam cycling 和 Beam CDD 跨越 10% BLER 所需的参考 SNR。

相对 Plan 004 的变化只有信道及相应的长期统计/SNR定义：UMa 固定半径模式 A 改为 NLOS CDL-C；RMS DS 固定为 100 ns；功率加权 mean AoD 固定为 6.25°，功率加权 RMS ASD 固定为 10°；mean ZoD 对准冻结码本的垂直主瓣中心；UE 速度为 0；SSB 按长期平均 RSRP 只选一次并在全部 BLER realization 中固定。AoA、ZoA以及 ZoD 相对均值的原始扩展保持 CDL-C 原始统计。

执行严格分三阶段：先检查全部 8 个 SSB 和 16 个 secondary beam 的长期 RSRP，再用粗 SNR 网格预扫描各曲线的 10% BLER 交越范围，最后冻结细网格配置并做全量仿真。若波束对准检查失败，不进入预扫描；若预扫描未括住全部曲线的 10% 交越，不进入全量仿真。

## 2. 仿真配置

### 2.1 环境、信道与 SNR

| 参数 | 取值 |
|---|---|
| Python / TensorFlow / Sionna | 3.11.9 / 2.15.1 / 1.0.2；禁止把仓库内旧 `py3GPP/` 放入搜索路径 |
| 场景与方向 | 3GPP CDL-C，downlink，NLOS rich multipath |
| 载频 | 7 GHz |
| 链路模式 | 模式 C `fixed_cdl_statistics` |
| RMS DS | 100 ns |
| UE速度 | 0 km/h；slot 内块衰落，`time_steps_per_slot=1` |
| AoD | 功率加权 mean AoD 6.25°；功率加权 RMS ASD 10° |
| ZoD | 功率加权 mean ZoD 104.4733873545822°，即全局向下角 14.4733873545822°；保持原始 ZSD |
| AoA/ZoA | 不做角度变换，保持 CDL-C 原始统计 |
| 路径损耗/阴影衰落 | 不使用；CDL 小尺度信道单位大尺度增益 |
| 长期统计 | 1000 个独立初始相位 realization；`statistics_seed=20261001` |
| BLER信道流 | 固定 ray、coupling、阵列和角度统计，仅重采样初始相位；`realization_seed=20262001` |
| 固定 SSB | 对长期发射协方差计算全部 SSB 的宽带平均接收功率，选最大者一次，此后固定 |
| 参考功率/SNR | $P_{ref}$ 为固定 selected SSB 的长期平均接收功率；$N_0=P_{ref}10^{-\gamma_{ref,dB}/10}$，所有方案和drop共用此标尺 |

角度统计采用 cluster power 与每 cluster 20 条 ray 等分后的功率加权定义。方位均值使用 circular mean，RMS ASD 使用相对该 circular mean 的 wrapped 角差；配置直接冻结 `target_asd_deg: 10.0`，不手填依赖实现版本的近似 scale。

### 2.2 阵列与码本

| 参数 | 取值 |
|---|---|
| BS AE/TXRU | 24V×16H×2 AE；4V×16H×2 TXRU |
| AE间距 | 水平 0.5λ、垂直 0.8λ |
| AE到TXRU映射 | 每极化连续 6V×1H AE 等幅同相合成到一个 TXRU，并按列归一化 |
| 面板 | 方位 0°，机械下倾 12°，无额外电下倾 |
| UE阵列 | 1V×2H 双极化，共 4Rx；orientation `[180°,0°,0°]` |
| SSB/secondary | 冻结 8H×1V SSB 和 16H×1V secondary；每个 SSB 对应两个 secondary beam |
| 码本文件 | `outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz` |
| SHA-256 | `3142396469049e843e8018bc5596152aa70d53e46d18b406879b52cf54638259` |
| 预期 parent | mean AoD 位于 SSB #4 的 0°～12.5039° 区域；长期 RSRP 应选择 SSB #4 |

第一阶段输出全部 SSB/secondary beam 的线性功率、dB功率、相对各自峰值、排名和父 SSB。进入预扫描的条件为：SSB #4 为唯一长期最大者；secondary #8 和 #9 的功率及排名已记录，且二者均不低于 secondary 全局峰值 3 dB。该条件用来发现角度坐标、面板方向、AE顺序或 TXRU 映射错误，不用于按结果重新调角度。

### 2.3 SIB1与接收机

30 kHz SCS、normal CP、FFT 2048、61.44 Msps；106 PRB BWP 内居中 48 PRB；PDSCH symbol 2–13；DMRS type-1 位于 symbol 2 和 11，每 PRB/每 DMRS symbol 6 RE，logical port 1000；rank-1 QPSK、MCS 0、目标码率 120/1024、TBS 1480、RV0。

4Rx 分支分别估计后做 MRC。B-SSB、P2-SSB、P6-SSB 和 BC-SSB 使用 SSB PDP 与 2-PRB LMMSE 窗口；CDD-SSB 使用含已知 CDD 人工时延及逐子载波归一化的 SSB PDP，与全 48-PRB LMMSE 窗口。PDP、路径空间协方差和 $P_{ref}$ 均由同一冻结 CDL 长期统计 Monte Carlo 得到。

### 2.4 对比方案

| 曲线 | 预编码 | PRG | 功率归一化 |
|---|---|---:|---|
| B-SSB | 固定 parent SSB，双极化相位 0° | 2 PRB | 每活动子载波单位范数 |
| P2-SSB | parent SSB 上 0°/90°极化相位逐 PRG 交替 | 2 PRB | 同上 |
| P6-SSB | parent SSB 上 0°/90°极化相位每 6 PRB 交替 | 6 PRB | 同上 |
| BC-SSB | 固定 parent 的两个 secondary beam 逐 PRG 交替 | 2 PRB | 同上 |
| CDD-SSB | 固定 parent 的两个 secondary beam 叠加 CDD | 全带连续相位 | 同上 |

Beam CDD 使用 $K=2$、活动带 576 点 DFT 网格索引 `[0,1]`，循环移位 `[0,3.5555555556]` 采样。FFT 为 2048、采样率为 61.44 Msps，对应时延 `[0,57.87037037] ns`。对本地活动子载波 $q=0,\ldots,575$，第 $m$ 个波束的相位为

$$v_m[q]=\exp(-j2\pi qj_m/576).$$

DMRS 和数据使用完全相同的预编码；叠加后逐活动子载波归一化为单位范数。

### 2.5 Monte Carlo与分阶段 SNR

主种子为 20260921。同一 SNR 的五条曲线共用 TB、固定统计下的同一信道 realization 和单位方差噪声；绝对 drop index 决定 realization，方案不进入随机种子。SNR 随机流网格起点 −8 dB、步长 0.5 dB。

| 阶段 | SNR/dB | drops/点 | 绝对drop区间 | 停止条件 |
|---|---|---:|---|---|
| 波束 RSRP | 不适用 | 1000 个长期统计 realization | 独立 statistics seed | 完成协方差及全部波束功率输出 |
| 预扫描 | `[-8,-6,-4,-2,0,2,4]` | 100 | 0–99 | 每点恰好100个公共drop，不提前停止 |
| 全量 | `[-2.5,-2,-1.5,-1,-0.5,0,0.5]` | 1000 | 1000–1999 | 每点恰好1000个公共drop，不提前停止 |

预扫描后，对每条曲线寻找原始 BLER 跨越 0.1 的相邻 2 dB 点。全量 SNR 集合取所有曲线括区的并集，并在 0.5 dB 网格上填满；在并集两端各增加一个 0.5 dB guard 点。若任一曲线未被括住，则先向未括住方向以 2 dB 步长追加预扫描，仍使用 0–99 的同一绝对 drop，并为新增 SNR 使用唯一 stream index。全量配置 `configs/experiments/plan-005.yaml` 必须在预扫描结束、全量开始前创建并冻结，记录实际 SNR 点；不得把预扫描计数并入全量主结果。

2026-09-21 预扫描完成后，五条曲线的原始 BLER 均由 −2 dB 到 0 dB 跨越 0.1，因此按上述预先规定的规则冻结全量网格为 −2.5～0.5 dB、步长 0.5 dB。选择证据保存在 `outputs/plan-005/run-001/snr-selection.json`，全量配置在运行开始前冻结为 `configs/experiments/plan-005.yaml`。

每点保存 block errors、blocks、原始 BLER 和 95% Wilson 区间。10%门限先报告原始点括区，再在括区两端按 $\log_{10}(BLER)$ 对 SNR 线性插值；不得用单调化结果覆盖原始计数。

## 3. 执行方式

阶段配置：`configs/experiments/plan-005-prescan.yaml`。全量配置在预扫描后按上节规则生成：`configs/experiments/plan-005.yaml`。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
py -3.11 -m sib1div.cli validate-config configs/experiments/plan-005-prescan.yaml `
  --output outputs/plan-005/config-validation-prescan
.\run_plan005.ps1 -Phase diagnostic -RunId run-001
.\run_plan005.ps1 -Phase prescan -RunId run-001
.\run_plan005.ps1 -Phase prepare -RunId run-001
.\run_plan005.ps1 -Phase full -RunId run-001
```

预期目录：

- `outputs/plan-005/run-001/beam-rsrp/`
- `outputs/plan-005/run-001/prescan/`
- `outputs/plan-005/run-001/full/`

## 4. 结果有效条件

- 相关单元测试、配置验证和至少一个 bounded smoke 通过；
- 角度诊断记录的功率加权 mean AoD 为 6.25°、RMS ASD 为 10°、mean ZoD 为 104.4733873545822°；
- SSB #4 为长期平均 RSRP 唯一最大者，secondary #8/#9 均在 secondary 峰值 3 dB 内；
- 全量运行中五条曲线在同一 SNR 使用相同 TB、信道和噪声，固定使用同一 parent SSB；
- 预扫描和全量绝对 drop 区间不重叠；全量各点恰好1000个公共drop；
- 每个活动子载波的预编码范数为1，CDD实际循环移位与本plan一致；
- 无 NaN、协方差求解失败、异常终止或不可解释的固定信道估计误差地板；
- 每条曲线在全量 SNR 网格上均有位于10% BLER两侧的采样点，否则不得报告该曲线的10%门限点估计。

## 5. 预期输出

- `beam_rsrp.csv`、`beam_rsrp.png` 和 `beam_rsrp_summary.json`；
- 预扫描及全量的 `bler.csv`、`bler.png`、`nmse.csv`、`nmse.png` 和 `paired_counts.csv`；
- 全量主 BLER 曲线、10%括区/插值门限表及95%置信区间；
- 展开配置、环境版本、种子、运行日志和 `run_metadata.json`；
- selected SSB、全部 SSB 长期功率、$P_{ref}$、角度变换前后统计；
- CDD循环移位及逐子载波预编码归一化检查。
