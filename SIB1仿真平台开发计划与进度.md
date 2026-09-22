# SIB1 仿真平台开发计划与进度

最后更新：2026-09-18

## 1. 目标

基于 Sionna 建立一套可复现的 SIB1 下行链路级仿真平台，先确认 SSB 波束、UMa 信道、SIB1 编解码和信道估计均正确工作，然后尽快得到以下 4 个方案的 BLER 对比结果：

1. Baseline
2. Pol-cycling
3. Beam cycling
4. Beam CDD

详细物理模型和公式以 [SIB1_transmission_diversity_simulation_plan_v4.md](./SIB1_transmission_diversity_simulation_plan_v4.md) 为准。本文只记录开发安排、验收要求和实际进度。

## 2. 已确定的主配置

| 项目 | 主配置 |
|---|---|
| 仿真环境 | Python 3.11.9、TensorFlow 2.15.1、Sionna 1.0.2 |
| 主场景 | 3GPP TR 38.901 UMa，下行，单 UE/drop |
| 主载频 | 7 GHz |
| 子载波间隔 | 30 kHz，normal CP |
| 第一轮带宽 | 48 PRB PDSCH allocation |
| BS阵列 | 24×16×2 AE，4×16×2 TXRU，$d_H=0.5\lambda$、$d_V=0.8\lambda$ |
| TXRU映射 | 每个TXRU连接连续6×1个AE，等幅同相，列功率归一化 |
| 面板下倾 | 12°机械下倾，不增加额外电下倾 |
| SSB码本 | 8H×1V，共8个SSB波束 |
| Secondary码本 | 每个selected SSB沿水平方向拆成2个波束，全扇区为16H×1V |
| 可选SSB配置 | 4H×2V |
| SIB1 | Rank 1、SI-RNTI、DCI 1_0、QPSK、MCS 0、PRG=2 PRB |
| PDSCH | mapping type A，symbol 2--13，DMRS symbol 2和11 |
| 第一轮TBS | 48 PRB时1480 bit，按TS 38.214计算 |
| 接收阵列 | 1×2双极化，共4Rx；各分支估计后做MRC |
| SSB选择 | 按宽带平均接收功率选择最佳SSB，不加入SSB选择错误 |
| SSB PDP | 假设UE能由SSB DMRS无误差获得 |
| 发射功率 | 每个占用子载波上的预编码向量单位范数 |
| CDD | OFDM循环移位，不作为真实传播时延；第一轮使用2个secondary beams |
| 第一轮链路模式 | `normalized_link` |
| v4第2.8节链路模式 | `fixed_radius_ls_normalized`、`fixed_radius_full_channel`、`fixed_cdl_statistics`（已实现） |
| 规划中的可选链路模式 | `coverage_link_budget`（尚未实现） |

“占用子载波”指PDSCH分配内的全部频率位置。48 PRB对应576个占用子载波，其中一部分RE承载DMRS，其余可用RE承载数据。协方差可以在576个频率位置上构造，也可以根据实际估计窗口只构造所需的 $\mathbf R_{pp}$ 和 $\mathbf R_{hp}$。

第一轮开发只使用系统安装的 Sionna 1.0.2。旧 py3GPP 目录中自带的 Sionna 1.2.0 不进入 Python 搜索路径，避免两个版本互相覆盖。开始开发时保存环境版本清单和导入路径检查结果。

### 2.1 顶层仿真方法与平台实现对照

第一轮 BLER 对比采用 `normalized_link`。其顶层方法是：UMa 每次 drop 生成一个包含路径损耗、阴影衰落和小尺度衰落的物理信道 $\mathbf H$；平台用固定 SSB 码本计算

$$
P_b=\frac{1}{N_rK}\sum_{r,k}|\mathbf H_r[k]\mathbf w_b|^2,
\qquad
b^\star=\arg\max_bP_b,
$$

再形成四方案共用的归一化信道

$$
\widetilde{\mathbf H}=\frac{\mathbf H}{\sqrt{P_{b^\star}}}.
$$

当前平台对该方法的实现状态如下：

| 顶层要求 | 当前实现 | 状态 |
|---|---|---|
| UMa drop 的 $\mathbf H$ 包含大尺度增益 | `src/sib1div/channel/uma.py` 中先由 Sionna `_step_12` 加入路径损耗和阴影衰落，再投影至 TXRU 域 | 已实现 |
| 按瞬时宽带平均每 Rx 功率选择 SSB | `src/sib1div/sim/engine.py` 的 `select_ssb()` 对 Rx 和全部占用子载波求 $|\mathbf H\mathbf w_b|^2$ 均值并取最大值 | 已实现 |
| 每个 drop 只使用一个 selected-SSB 公共归一化因子 | `simulate_one_drop()` 在进入四方案循环前计算 $P_{b^\star}$ 并归一化完整 `response` | 已实现 |
| 归一化后 selected-SSB 功率严格为 1 | 运行时记录 `normalized_selected_ssb_power`；`tests/test_sim.py` 有精确归一化单元测试 | 已实现并测试 |
| LMMSE 先验与归一化信道采用相同功率尺度 | `_prior_covariances()` 将各路径空间协方差统一除以同一个 $P_{b^\star}$ | 已实现 |
| 四方案保留相对预编码增益 | 四种预编码器均作用于同一个归一化 `response`，未按方案再次归一化信道；仅保证每个占用子载波的发射预编码向量单位范数 | 已实现 |
| 四方案共用 TB、信道和噪声 | drop 在方案循环外生成；TB 和单位方差噪声由 `(主种子, SNR索引, drop索引)` 唯一派生并在方案间复用 | 已实现 |
| 固定 SNR 的噪声定义 | 归一化后使用 $\sigma_n^2=10^{-\mathrm{SNR}_{\rm dB}/10}$；这里的 SNR 是 selected-SSB 公共参考 $E_s/N_0$，不是把四方案各自的实际接收 SNR 强制相同 | 已实现 |
| `coverage_link_budget` 保留大尺度增益 | 配置校验目前接受该模式，但仿真引擎尚未按 `link_mode` 分支，仍会执行 selected-SSB 归一化 | 未实现，不可用于正式覆盖结论 |

因此，当前 `plan-001`、`plan-002` 的 `normalized_link` 配置已经实现顶层思路，可以用于固定公共参考 SNR 下的四方案 BLER 对比。`coverage_link_budget` 在完成独立的链路预算、噪声功率和引擎分支实现及测试前，只作为规划能力，不应标记为平台已支持。

## 3. 开发安排

### A. 平台基础功能调通

这一阶段一次性打通完整基础链路，不单独开展大规模实验。

需要完成：

- 建立统一配置、随机种子和结果目录；同一个drop、传输块和噪声能够被4个方案共同使用。
- 离线生成8H×1V SSB码本和16H×1V secondary码本，保存权重、配置和方向图。
- 通过AE几何与固定映射得到TXRU域路径信道，不在每个子载波上保存完整AE域信道。
- 使用Sionna生成同一个drop的UMa路径、角度、时延、功率和瞬时信道，并保证4个方案复用同一信道。
- 实现标准SIB1 PDSCH链路，包括TBS、TB CRC、LDPC、速率匹配、扰码、QPSK、DMRS和RE映射。
- 实现4种预编码器，并检查同一PRG内的DMRS和数据使用相同预编码。
- 实现Perfect CSI、DMRS LS、基于PDP/协方差的LMMSE和4Rx MRC。
- 参考CDD_LLS的方式，只在占用子载波或当前估计窗口内构造协方差；默认不保存逐drop完整稠密协方差。

平台基础功能通过以下检查后，即进入4方案仿真：

| 检查 | 通过要求 |
|---|---|
| SSB方向图 | 8个SSB共同覆盖水平-60°到60°，方向、编号和下倾正确，无坐标重复旋转 |
| Secondary方向图 | 每个selected SSB的两个secondary beams覆盖其原目标区域，波束编号稳定 |
| TXRU信道 | 小尺寸阵列中，TXRU域投影结果与直接AE域计算一致 |
| SIB1链路 | 无噪声、Perfect CSI时连续解码通过，无CRC错误 |
| 编码与映射 | TBS、CRC、编码长度、DMRS位置和数据RE数量与独立参考结果一致 |
| 信道估计 | 平坦无噪声信道无固定误差；增加SNR时LS/LMMSE NMSE总体下降 |
| LMMSE | 使用正确协方差时无数值异常，协方差矩阵为Hermitian半正定 |
| 功率 | 所有方案在每个占用子载波满足 $\|\mathbf w[k]\|_2^2=1$ |
| 公平性 | 4个方案使用相同drop、TB、信道和噪声，仅预编码方式不同 |

### B. 直接进行4方案对比

基础功能通过后，直接运行7 GHz、48 PRB、8H×1V SSB的4方案对比，不等待24/96 PRB及其他可选配置完成。

第一轮使用自适应SNR搜索和基于误块数/Wilson上界的公共drop联合停止，覆盖 $10^{-1}$ 和 $10^{-2}$ BLER附近，不继续追踪更低BLER。正式结果至少包含：

- Baseline/Pol-cycling各1条、Beam cycling两种协方差、Beam CDD三种协方差，共7条estimated-CSI BLER--SNR曲线；
- 4个方案在Perfect CSI下的BLER--SNR检查图；
- 信道估计NMSE--SNR图；
- SSB及secondary beam方向图；
- Beam CDD循环移位和预编码范数检查；
- 每个SNR点的误块数、总块数和95%置信区间；
- 可复现的配置、随机种子、环境版本和运行日志。

Baseline和Pol-cycling使用SSB PDP；Beam cycling比较SSB-PDP与per-beam-PDP；Beam CDD比较SSB-derived、per-beam-derived和ideal joint covariance。Perfect CSI作为4方案独立诊断，不与7条estimated-CSI曲线合并。

正式运行采用“达到目标误块数或达到最大drop数”停止，不填充伪造的零值。低BLER点若未观察到误块，记录为统计上限并保留实际样本数。

### C. 补充可选配置

第一轮4方案结果完成后，平台再按需要运行以下已有配置能力：

- 24 PRB和96 PRB；
- 4H×2V SSB码本；
- Beam CDD的4波束和8波束配置；
- 0.7 GHz和2 GHz；
- `coverage_link_budget`；
- 不同下倾角、UE距离和CDD循环移位；
- SSB PDP存在估计误差的接收机；
- 时变信道和IRC接收机。

这些配置不阻塞第一轮7 GHz、48 PRB、4方案结果。

## 4. 代码与结果组织

新平台代码放在独立目录中，不继续扩展旧的大型单文件脚本。建议目录职责如下：

```text
src/sib1div/
  config.py          配置和参数检查
  codebook/          SSB与secondary码本生成、保存和画图
  channel/           UMa drop、TXRU域路径信道、PDP和协方差
  nr/                SIB1编码、DMRS和PDSCH资源映射
  schemes/           Baseline及3个候选方案
  receiver/          LS、LMMSE和MRC
  sim/               公共随机样本、Monte Carlo和停止条件
  analysis/          BLER、NMSE和方向图
configs/             可复现YAML配置
tests/               单元测试和端到端检查
outputs/             展开配置、CSV、日志、图和必要诊断数组
```

旧 py3GPP 代码只选择性复用经过测试的CRC、TBS、LDPC和NR序列函数。旧信道估计器、手工RE拼接、comb-DMRS方案和历史CSV不作为新平台真值。

## 5. 当前进度

| 内容 | 状态 | 说明 |
|---|---|---|
| 仿真需求梳理 | 已完成 | 已明确UMa、阵列、4种方案、PDP和LMMSE假设 |
| 主参数补充 | 已完成 | 已补充标准SIB1时频配置、TBS和两种链路模式 |
| 功率与CDD定义 | 已完成 | 主实验逐子载波单位范数；CDD定义为循环移位 |
| SSB布局选择 | 已完成 | 主配置8H×1V，4H×2V为可选配置 |
| 本机环境检查 | 已完成 | Python 3.11.9、TensorFlow 2.15.1、Sionna 1.0.2，可导入；当前未检测到GPU |
| CDD_LLS协方差实现检查 | 已完成 | 采用运行时按占用子载波构造、结果不保存完整协方差的方式 |
| 新平台代码骨架 | 已完成 | `src/sib1div/` 模块骨架、统一YAML配置、公共随机样本、运行元数据和CLI已建立；6项骨架测试通过，验证输出见 `outputs/scaffold_validation/` |
| 正式实验文档流程 | 已完成 | 每次正式方案仿真使用同编号的 `research/plan-XXX.md`、`research/result-XXX.md` 和 `configs/experiments/plan-XXX.yaml` |
| 平台基础功能调通 | 已完成 | UMa长期ray/极化/空间二阶统计、三类CDD协方差、PRG内/全带LMMSE及`simulate`已形成闭环；24项测试通过。75～115 m码本通过联合覆盖/交界/父子审核并冻结；逐drop瞬时selected-SSB归一化已验收。详见 `outputs/platform_acceptance/平台基础功能调试报告.md` |
| v4第2.8节固定半径双模式 | 已完成 | 已保留Sionna实际施加的$G_d$，实现$G_0/G_d$大尺度幅度归一化、完整信道模式、共同$G_0$噪声标定及匹配的LMMSE协方差缩放；详见 `SIB1仿真平台_v4第2.8节实现说明.md` |
| v4第2.8.4节固定CDL长期统计模式 | 已完成 | 已实现固定CDL ray/angle/power/XPR统计、独立样本流协方差估计、固定parent SSB、公共$P_{\rm ref}$噪声标定、角度平移/缩放和可选Doppler时间演化；详见 `SIB1仿真平台_v4第2.8节实现说明.md` |
| 4方案初步BLER结果 | 待研究者执行 | 7条estimated-CSI与4条Perfect-CSI组合、自适应SNR、公共drop联合停止、聚合输出和断点续跑runner已完成；使用 `run_plan001.ps1` 一次性启动 |
| 4方案正式结果 | 未开始 | 初步结果检查通过后增加样本数 |
| Plan 004单调主曲线与门限 | 计划中 | 复用Plan 003计数并补齐共同drop 1000–1999；1%区间扩展到1000–5999；输出原始点、单调估计及10%/1%门限 |
| 可选配置 | 未开始 | 不阻塞第一轮结果 |

## 6. 第一轮交付内容

第一轮完成时应交付：

- 可运行的Sionna仿真平台代码；
- 7 GHz、48 PRB主配置文件；
- 8H×1V SSB和16H×1V secondary码本及方向图；
- 基础功能检查结果；
- Baseline、Pol-cycling、Beam cycling和Beam CDD的初步及正式BLER结果；
- 信道估计NMSE结果；
- 原始统计CSV、运行配置、日志和结果说明文档。

本文件在开发过程中持续更新状态和结果路径，作为当前进度入口。

## 7. 最近一次开发记录

### 2026-09-13：新平台代码骨架

- 建立 `src/sib1div/`，按 codebook、channel、nr、schemes、receiver、sim 和 analysis 划分职责，不把旧 `py3GPP/` 加入 Python 搜索路径。
- 新增主配置 `configs/uma_7ghz_48prb.yaml`，集中记录7 GHz、48 PRB、8H×1V/16H×1V码本、4Rx MRC和4种方案的共同参数。
- 配置加载时交叉校验阵列/TXRU维度、水平与垂直间距、PDSCH符号、DMRS位置、PRG、TBS、码本布局和逐占用子载波单位范数策略。
- 展开配置确认：576个占用子载波、576个DMRS RE、6336个数据RE、12672个编码比特、768个BS AE和128个TXRU端口。
- 公共随机样本按主种子、SNR索引和drop索引确定性派生，同一索引下的传输块和单位方差噪声供4个方案复用。
- `outputs/scaffold_validation/environment.json` 记录Python 3.11.9、TensorFlow 2.15.1、Sionna 1.0.2及导入路径；Sionna来自系统安装目录，未使用旧目录副本，当前TensorFlow未检测到GPU。
- `py -3.11 -m pytest -q`：6项测试全部通过。

下一项工作为“平台基础功能调通”，首先实现并验收AE到TXRU映射以及8H×1V SSB、16H×1V secondary离线码本与方向图。

### 2026-09-13：平台基础功能调试——TXRU映射与离线码本

- 已实现单极化384×64的AE→TXRU连续6×1等幅同相映射，验证每列6个非零AE、列功率归一化、列间正交，并用小阵列测试确认TXRU域投影与直接AE域响应一致。
- 已实现方向余弦域均匀分区和weighted-LS固定码本综合，生成8H×1V SSB、16H×1V secondary的TXRU/AE权重、二维覆盖图、水平/垂直切面及父子波束叠加图。
- 阻塞性发现：6×1等幅同相垂直子阵、`d_V=0.8λ`和12°机械下倾共同产生约24.0°全局下倾公共零陷，位于11.55°～33.88°目标范围内。该零陷不能由TXRU数字权重消除，因此当前方向图未达到覆盖验收要求。
- 详细公式、方向图、逐波束指标和待决配置选项见 `outputs/codebook_7ghz_8h1v/码本与方向图审核报告.md`。在研究者确认机械下倾、子阵固定电下倾或垂直分组调整方案前，不进入大规模BLER仿真。

### 2026-09-13：正式实验记录流程

- 新增项目级 `AGENTS.md`，只保留本项目需要的代码、测试和正式仿真记录要求，不建立GOALS文档。
- 新增 `research/README.md` 作为实验编号索引；正式实验从 `001` 开始。
- 新增plan和result模板。plan必须写清系统配置、阵列码本、SIB1资源、接收机、SNR、随机种子，以及Beam CDD的FFT、采样率和循环移位取值；result必须包含实际统计、输出路径和仿真曲线。
- 相同物理配置仅追加trial时沿用原编号并记录新的绝对trial范围；改变配置、方案、CDD循环移位或接收机时新建编号。

### 2026-09-13：Plan 001——4方案初步BLER

- 新增 `research/plan-001.md` 和 `configs/experiments/plan-001.yaml`，固定7 GHz UMa、48 PRB、8H×1V SSB、16H×1V secondary、4Rx MRC及4方案的第一轮初步BLER配置。
- Beam CDD使用两波束 `S0_SIDON2`：有效带宽DFT索引 `[0,1]`，对应 `[0,57.870370] ns`；在2048点FFT下为 `[0,3.555556]` 个采样点，采用频域线性相位实现分数采样循环移位。
- 初步统计使用-8～6 dB、2 dB间隔、每点30个目标误块和最多2000个公共drop；4条主曲线使用联合停止条件，保证同一SNR下样本数相同。
- 配置加载校验通过，展开量为576个占用子载波、576个DMRS RE、6336个数据RE和12672个编码bit；自动测试10项通过。
- `plan-001` 只完成预注册，尚未执行。当前仍需解决现有码本目标区垂直零陷，并实现 `simulate` runner、UMa链路和接收机后方可正式运行。

### 2026-09-13：长期协方差与频域LMMSE

- UMa drop新增长期TXRU空间路径协方差。实现直接使用Sionna 38.901的cluster/ray power、XPR、收发阵列响应、强簇三子簇拆分和Rician K因子；解析求和四个独立极化随机相位的二阶矩，不再用单次瞬时路径系数模平方作为PDP功率。
- 实现任意固定波束PDP、beam-domain逐路径联合协方差，以及随子载波变化预编码对应的真实等效频域协方差。
- Beam cycling支持每个2-PRB PRG分别采用SSB-PDP或实际secondary-beam PDP；LMMSE窗口不跨PRG。
- Beam CDD支持SSB-PDP-derived、per-beam-PDP-derived和ideal joint covariance。ideal保留不同secondary beams之间的交叉项；三类方法均包含CDD时延，且已知逐子载波归一化进入协方差。
- CLI新增 `lmmse_all`、`lmmse_ssb`、`lmmse_per_beam` 和 `lmmse_ideal` 调试选择。一次20 dB UMa drop的全先验端到端smoke共10条接收机结果，全部CRC通过，输出位于 `outputs/debug/lmmse-covariance-smoke/`。
- `python -m pytest -q`：22项测试全部通过。新增测试覆盖beam投影、joint CDD与直接等效协方差一致性、CDD归一化协方差和PRG边界隔离。

### 2026-09-13：码本冻结与逐drop瞬时归一化

- 在全扇区和11.549°～17.398°受控下倾网格审核联合覆盖：SSB/secondary最低增益分别为3.98/6.02 dB；相邻波束交界最低联合增益为5.28/5.98 dB，没有额外覆盖孔洞。
- 每个父SSB的两个secondary在其父区域全部审核点上联合增益均不低于父波束，最小裕量约0.17 dB。研究者接受单波束边缘约18 dB滚降，码本判定通过并按SHA-256冻结。
- normalized-link改为每个drop按瞬时selected-SSB宽带平均每Rx功率归一化；所有方案共用该因子。输出新增 `normalized_selected_ssb_power`，真实UMa smoke得到严格的1.0。
- 加载冻结码本时强制校验SHA-256；权重发生任何变化都会在仿真开始前报错。
- `python -m pytest -q`：24项测试全部通过；正式Plan 001 preflight解除。

### 2026-09-13：Plan 001自适应Monte Carlo runner

- runner按冻结YAML一次运行7条estimated-CSI和4条Perfect-CSI曲线；同一SNR的全部曲线严格使用相同公共drop。
- 每条estimated-CSI曲线达到100 errors或95% Wilson上界低于 $10^{-2}$ 后判定完成；7条均完成才停止该SNR，20000 drop仅为安全上限。
- 从0 dB开始以2 dB搜索 $10^{-1}$～$10^{-2}$ 过渡区，在交越区自动补1 dB点；所有曲线低于 $10^{-2}$ 后不继续提高SNR。
- 常规输出仅保留 `bler.csv`、`nmse.csv`、`paired_counts.csv`、最小运行元数据和3张图；运行中使用小型聚合checkpoint，成功后自动删除。
- 新增根目录 `run_plan001.ps1` 手动一键入口；同一RunId支持中断续跑，完成目录禁止覆盖。
- 缩小规模真实UMa集成测试生成7+4条曲线、21组配对计数及全部图；全套自动测试27项通过。

### 2026-09-15：v4第2.8节固定半径双模式

- 确认Sionna 1.0.2 UMa在`_step_12()`前后分别提供未施加和已施加路径损耗/阴影衰落公共标量的CIR，信道模型能力满足第2.8节。
- `UMaDrop`新增Sionna实际施加的大尺度幅度增益$a_d$和功率增益$G_d=a_d^2$；不使用selected-SSB瞬时功率估计$G_d$。
- 新增`fixed_radius_ls_normalized`和`fixed_radius_full_channel`。前者对信道及路径空间协方差共同乘以$G_0/G_d$，后者保持完整信道；两者均以$N_0=G_0 10^{-\gamma_{0,\mathrm{dB}}/10}$标定噪声。
- 配置验证强制固定半径、显式记录$G_0$线性/dB值及物理定义，并禁止按方案重新归一化。
- 新增平台实现说明`SIB1仿真平台_v4第2.8节实现说明.md`；全套自动测试32项通过，另通过UTF-8回读、冲突标记、尾随空格和Python编译检查。
