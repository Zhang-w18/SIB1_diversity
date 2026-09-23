# Plan 005：固定统计 CDL-C、100 ns、10° ASD 五曲线 BLER 与接收分支 RSRP 比较

状态：计划中（run-001 已完成；1% BLER 与 RSRP CDF 补充实验待运行）

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

## 6. 2026-09-22 补充实验：1% BLER 尾部与逐接收天线 RSRP CDF

### 6.1 目的与不变项

本次追加 trial 回答两个问题：第一，扩展 SNR 范围，使 B-SSB、P2-SSB、P6-SSB、BC-SSB 和 CDD-SSB 五条曲线都以原始仿真点跨过 1% BLER；第二，比较五种预编码在每根 UE 接收天线上的瞬时等效端口 RSRP 分布。第 2 节冻结的信道、阵列、码本、五条曲线、CDD、接收机、功率归一化、主种子和随机流映射全部不变。run-001 的原始结果和配置不改写，新增数据以追加 trial 保存，并在 `result-005.md` 中记录和合并。

这属于相同系统配置下增加 SNR 点、drop 和诊断量，不用新的 plan 编号。原预扫描 drop 0–99、原正式仿真 drop 1000–1999 保留；补充预扫描使用 100–599，补充正式 BLER 从 2000 开始，因此不与已执行区间重叠。

### 6.2 1% BLER 预扫描与 SNR 网格选择

补充预扫描首先运行 SNR `[0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]` dB，每点 500 个公共 drop，绝对 drop 区间 100–599。0.5 dB 虽已有正式结果，仍在预扫描中保留，作为新旧区间的一致性检查；预扫描计数不并入正式结果。

2026-09-22 首次补充预扫描在完成 0.5 dB 的 30 个 drop 后由用户停止，未形成可用 prescan 点，其 checkpoint 原样保留但不并入任何统计。考虑到 run-001 已在同一模式 C、相同物理配置和随机流 namespace 下完成 0/2/4 dB、每点 100 drops 的粗扫，后续改用快速补扫：新增 `[1.0,1.5,2.5,3.0,3.5,4.5]` dB，每点 100 个公共 drop，使用绝对 drop 600–699。选点时合并 run-001 的 0/2/4 dB 粗扫、run-001 的 −2.5～0.5 dB 正式结果和本次快速补扫；同一 SNR 若有重复数据，以本次快速补扫为设计依据，但预扫描计数仍不并入正式 BLER。

对每条曲线寻找原始 BLER 跨越 0.01 的相邻 0.5 dB 点。若任一曲线在 4.0 dB 仍未低于 0.01，则以 0.5 dB 步长向高 SNR 追加预扫描，沿用绝对 drop 100–599；若某曲线在 0.5 dB 已低于 0.01，则用 run-001 中 0.0/0.5 dB 的正式结果提供低 SNR 一侧。只有五条曲线都得到 1% 括区后，才生成正式追加配置。

正式 SNR 集合取五条曲线 1% 括区的并集，在 0.5 dB 网格上填满，并在并集两端各加一个 0.5 dB guard 点。为保持从 10% 到 1% 的整条曲线连续，最终合图还必须包含 run-001 的 −2.5～0.5 dB 原始点；不得把预扫描计数与正式计数混合。

### 6.3 每个 SNR 点的自适应正式 drop 数

正式 drop 数按 SNR 点分别冻结，但同一 SNR 的五条曲线必须使用相同 drop 数和相同 TB、信道、噪声样本。设补充预扫描在该点对曲线 $s$ 得到 $e_s$ 个错误、$n=500$ 个 block，采用 Jeffreys 平滑估计

$$
\widetilde p_s=\frac{e_s+0.5}{n+1}.
$$

该 SNR 的设计 BLER 取五条曲线中的最小值 $p_{design}=\min_s\widetilde p_s$，使最强方案也有足够尾部统计；正式总 drop 数冻结为

$$
N_{total}=1000\left\lceil\frac{\min\left(20000,\max\left(2000,100/p_{design}\right)\right)}{1000}\right\rceil.
$$

因此每点最少 2000、最多 20000 个正式公共 drop，目标是让预期最强曲线约有 100 个错误；上限处允许少于 100 个错误，但必须如实报告置信区间。对于 run-001 已运行的 −2.5～0.5 dB 点，只追加 `N_total-1000` 个 drop，新增绝对 drop 从 2000 起；新增 SNR 点运行 `N_total` 个 drop，同样从 2000 起。不同 SNR 由冻结的 stream index 区分；所有追加配置必须记录该点的 SNR、`N_total`、新增 drop 数和绝对 drop 区间。合并时按 `curve_id + snr_db` 求和 errors/blocks/NMSE 累积量并重算 Wilson 区间，禁止平均 BLER 百分比。

曲线平滑性的最低验收条件为：五条曲线都在正式点上有 BLER 大于和小于 0.01 的相邻点；1% 附近每个非封顶点的最强曲线至少 80 个实测错误；各曲线随 SNR 的局部非单调变化必须同时检查 Wilson 区间，不能为美化曲线而替换原始计数。主图使用原始点连线和 95% Wilson 区间，可另给出仅用于观察趋势的单调拟合，但门限仍按相邻原始点的 $\log_{10}(BLER)$ 插值。

### 6.4 每根接收天线的 RSRP 分布

RSRP 统计使用 10000 个固定 CDL realization，绝对 drop 2000–11999；不生成 TB、不加噪声、不做信道估计或 MRC。对曲线 $s$、接收天线 $r$ 和 realization $i$，先应用与 BLER 完全相同的预编码（包括 P2/P6 的 PRG 图案、BC 的 beam cycling、CDD 相位和逐子载波归一化），得到 576 个活动子载波上的等效端口信道 $h^{(s)}_{i,r}[k]$，定义

$$
P^{(s)}_{i,r}=\frac{1}{576}\sum_{k=0}^{575}\left|h^{(s)}_{i,r}[k]\right|^2,
\qquad
P^{(s)}_{i,r,\mathrm{dBref}}=10\log_{10}\frac{P^{(s)}_{i,r}}{P_{ref}}.
$$

其中 $P_{ref}$ 是第 2.1 节定义的 selected SSB 四接收分支合计长期平均功率标尺；同时保存未归一化线性功率，避免 dB 标尺歧义。这里的“每根接收天线”指 UE 的 4 个物理 Rx 分支，不做跨天线求和。P2-SSB 和 P6-SSB 分别作为两条曲线，因此每个 Rx 图包含 B-SSB、P2-SSB、P6-SSB、BC-SSB、CDD-SSB 五条经验 CDF；共输出 4 张图。CDF 使用全部有限样本按升序排列，纵坐标采用 $(j-0.5)/N$，不抽样、不平滑。

输出至少包括 `rsrp_per_rx.csv`（字段含 drop index、Rx index、curve id、线性功率和 dBref）、`rsrp_summary.csv`（均值、标准差及 1/5/10/50/90/95/99 百分位）、`rsrp_cdf_rx0.png` 至 `rsrp_cdf_rx3.png`、展开配置和运行元数据。五条方案必须复用同一批 10000 个信道 realization。

### 6.5 补充实验执行顺序与目录

1. 验证补充预扫描配置并执行 prescan；输出到 `outputs/plan-005/run-002/prescan-1pct/`。
2. 根据 prescan 和 run-001 正式 CSV 冻结 SNR 集合及逐点 `N_total`，生成不可改写的追加配置清单 `outputs/plan-005/run-002/selection-1pct.json` 和 `configs/experiments/plan-005-tail-*.yaml`。
3. 逐 SNR 执行追加正式 trial；输出到 `outputs/plan-005/run-002/tail/`，支持 checkpoint 续跑。
4. 合并 run-001 与追加 trial，生成完整 BLER/NMSE CSV、主图及 1% 门限表；输出到 `outputs/plan-005/run-002/combined/`。
5. 独立执行 RSRP 分布统计；输出到 `outputs/plan-005/run-002/rsrp-cdf/`。

任何自动生成的正式配置都必须在对应仿真启动前落盘并记录 SHA-256。若 prescan 没有括住全部 1% 交越、逐点 drop 计划缺失、追加区间与旧区间重叠、合并后的 blocks 与计划不符，或任一 Rx/曲线的 RSRP 样本数不是 10000，则补充结果无效。

快速补扫与正式尾部仿真允许由 `run_plan005_tail.ps1 -Phase auto-tail` 自动串联。该阶段必须先完整结束快速补扫，再生成 `selection-1pct.json` 和逐点冻结 YAML，随后才能启动第一个正式 SNR 点；若补扫未括住全部曲线的 1% BLER，则配置生成应报错并停止，不得以猜测网格继续正式仿真。
