# Plan 001：7 GHz、48 PRB四方案七种信道估计组合BLER

状态：待研究者手动执行（物理配置、码本和自适应runner均已冻结）

创建日期：2026-09-13

配置文件：[plan-001.yaml](../configs/experiments/plan-001.yaml)

## 1. 本次实验目的

本实验是平台基础功能通过后的第一轮方案仿真，目标是尽快得到以下4种SIB1传输方案的初步BLER曲线：

1. Baseline；
2. Pol-cycling；
3. Beam cycling；
4. Beam CDD。

主结果包含7条estimated-CSI曲线：Baseline和Pol-cycling各1条，Beam cycling使用2种协方差假设，Beam CDD使用3种协方差假设。另运行4条Perfect-CSI诊断曲线，用于判断各方案在理想信道已知时的分集潜力，但不与7条主曲线混为同一接收机结论。本实验不用于证明所选CDD时延是最优时延，也不比较多个CDD时延集合。

这是首个正式plan，没有前一次实验配置可供比较。

## 2. 仿真配置

### 2.1 环境、场景与链路口径

| 参数 | 取值 |
|---|---|
| Python | 3.11.9 |
| TensorFlow | 2.15.1 |
| Sionna | 1.0.2，使用系统安装版本 |
| 场景与方向 | 3GPP TR 38.901 UMa，下行，单UE/drop |
| 载频 | 7 GHz |
| 链路模式 | `normalized_link` |
| BS/UE高度、ISD | 25 m / 1.5 m / 500 m |
| UE位置 | 水平距离75～115 m，按扇区面积均匀采样；扇区方位角-60°～60°均匀采样 |
| LOS/NLOS | 按Sionna UMa的3GPP概率随机产生 |
| 移动性 | 0 m/s，单slot内信道不变 |
| 路径损耗和阴影衰落 | 先由UMa生成，再由各方案共用的逐drop瞬时selected-SSB宽带平均每Rx功率归一化因子消除 |
| 横轴 | 公共逐dropselected-SSB参考归一化后的每个发射RE的 $E_s/N_0$ |

这里使用的是Sionna的UMa系统级随机簇/射线信道，不是固定的CDL-A、CDL-B、CDL-C、CDL-D或CDL-E profile。如果后续要改为某个固定CDL profile，应另建plan，不能把两种结果混在本实验中。

对第 $b$ 个SSB定义 $P_b=(N_rK)^{-1}\sum_{r,k}|H_r[k]\mathbf w_b|^2$，并令 $b^\star=\arg\max_bP_b$。同一drop的4个方案共同使用 $H_{\rm norm}=H/\sqrt{P_{b^\star}}$，使selected SSB的瞬时宽带平均每Rx功率严格为1。禁止按每个方案自己的接收功率分别归一化；否则会抹掉波束失配和空间增益差异。该处理仍保留多径相对功率、随机相位、子载波间起伏、4Rx分支差异和方案相对增益。

本实验只比较受控主瓣区域内、给定相同物理信道时4种分集传输方案的相对BLER，不评价全小区覆盖。UE位置必须在生成UMa信道之前按几何条件抽取。令 $U\sim\mathcal U[0,1]$，水平距离按扇区面积均匀采样：

$$
r=\sqrt{75^2+U(115^2-75^2)}\ \mathrm{m}.
$$

在BS高度25 m、UE高度1.5 m时，该距离范围对应全局几何下倾角约11.55°～17.40°；相对12°机械下倾的面板局部角约为-0.45°～5.40°，处于6×1等幅同相垂直子阵的有效主瓣内，并避开约24.0°的全局公共零陷。禁止生成信道后再根据RSRP、瞬时波束增益、LOS/NLOS、信道 realization 或解码结果筛除drop。所有方案必须复用同一个预先抽取的UE位置和随后生成的物理信道。该实验结果只能解释为上述受控UE区域内的分集性能。

### 2.2 阵列与码本

| 参数 | 取值 |
|---|---|
| BS AE/TXRU | 24V×16H×2极化AE；每极化4V×16H TXRU，共128个TXRU端口 |
| AE间距 | $(d_H,d_V)=(0.5,0.8)\lambda$ |
| AE到TXRU映射 | 每个TXRU连接连续6V×1H AE，等幅同相，映射列单位范数 |
| 面板方向和下倾 | 方位0°，机械下倾12°，额外电下倾0° |
| UE阵列 | 1V×2H双极化，共4Rx |
| SSB码本 | 8H×1V weighted-LS宽波束 |
| Secondary码本 | 16H×1V；selected SSB对应相邻的2个水平子波束 |
| 码本权重 | 已冻结：`outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz`；SHA-256见冻结清单 |
| 码本元数据 | 已冻结：`outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_metadata.json` |
| Selected SSB | 8个SSB中宽带平均接收功率最大者；不模拟选择错误 |

以上码本必须离线生成，所有drop复用固定权重，不能根据当前drop的瞬时AOD重算波束。

早期码本按35～115 m、即11.55°～33.88°目标角域综合，因12°机械下倾、6×1 AE等幅同相子阵和 $d_V=0.8\lambda$ 在约24.0°全局下倾处形成公共零陷而未通过审核。Plan 001现将UE预先限制为75～115 m，对应11.55°～17.40°受控主瓣角域。新码本已避开公共零陷；数值审核确认SSB/secondary联合覆盖最低增益分别约3.98/6.02 dB，交界处没有额外深陷；每个父SSB区域内两个secondary的联合增益在全部审核网格点均不低于父波束，最小裕量约0.17 dB。研究者已接受逐波束矩形区域边缘约18 dB的滚降，因此码本审核通过并冻结。该码本只要求覆盖本实验受控角域，不要求覆盖35～75 m区域。

### 2.3 SIB1资源与接收机

| 参数 | 取值 |
|---|---|
| SCS和CP | 30 kHz，normal CP |
| FFT和采样率 | $N_{FFT}=2048$，61.44 Msps |
| BWP/PDSCH | 106 PRB BWP；居中的48 PRB PDSCH，共576个占用子载波 |
| PDSCH symbols | mapping type A，symbol 2～13，共12 symbols |
| DMRS | type 1，symbol 2和11，1个CDM group，6 RE/PRB/symbol，逻辑端口1000 |
| MCS/TBS/RV | Rank 1、QPSK、MCS 0、目标码率120/1024、TBS 1480 bit、RV 0、$x_{Overhead}=0$ |
| PRG | 2 PRB，共24个PRG |
| 数据量 | 576 DMRS RE、6336 data RE、12672个速率匹配后编码bit |
| 主信道估计 | DMRS LS后做频域LMMSE；先验来自当前drop无误差的真实SSB PDP |
| Perfect CSI | 对照曲线，不进入主结论 |
| 合并 | 4个Rx分支分别估计，再做MRC |
| 协方差保存 | 只在运行时按估计窗口构造 $R_{pp}$ 和 $R_{hp}$，默认不保存逐drop完整稠密矩阵 |

Baseline、Pol-cycling和Beam cycling的LMMSE窗口为当前2 PRB PRG。Beam CDD在连续48 PRB内没有预编码突变，主接收机在全部576个占用子载波上使用已知CDD结构构造协方差。

Beam CDD同时报告SSB-PDP-derived、per-beam-PDP-derived和ideal joint covariance三种接收机；Beam cycling同时报告SSB-PDP和实际per-beam-PDP两种接收机。各协方差假设使用独立曲线ID，不能在统计或绘图时合并。

#### 2.3.1 分集传输方案—信道估计组合

| 曲线ID | 分集传输方案 | LMMSE协方差假设 | 估计频域窗口 | 用途 |
|---|---|---|---|---|
| `B-SSB` | Baseline | selected-SSB PDP | 每个2-PRB PRG | 主比较 |
| `P-SSB` | Pol-cycling | selected-SSB PDP | 每个2-PRB PRG | 主比较 |
| `BC-SSB` | Beam cycling | selected-SSB PDP，存在窄波束PDP mismatch | 每个2-PRB PRG | 实际低先验方案 |
| `BC-BEAM` | Beam cycling | 当前PRG所用secondary beam的真实per-beam PDP | 每个2-PRB PRG | matched-covariance对照 |
| `CDD-SSB` | Beam CDD | SSB-PDP-derived CDD covariance，忽略分支交叉相关 | 全48 PRB | 实际低先验方案 |
| `CDD-BEAM` | Beam CDD | Per-beam-PDP-derived CDD covariance，忽略分支交叉相关 | 全48 PRB | 分离PDP mismatch损失 |
| `CDD-IDEAL` | Beam CDD | Ideal joint covariance，保留beam-domain交叉相关 | 全48 PRB | 理论上界 |

以上7条曲线是本实验必须完成的estimated-CSI结果。Perfect CSI另外对Baseline、Pol-cycling、Beam cycling和Beam CDD各输出1条诊断曲线。LS只用于平台调试，不进入Plan 001正式BLER图。

### 2.4 四种方案

| 方案 | 预编码方式 | 功率归一化 | 本次参数 |
|---|---|---|---|
| Baseline | 所有PRG使用selected SSB双极化Rank-1权重 | 每个占用子载波单位范数 | 两极化相对相位0° |
| Pol-cycling | 保持selected SSB空间权重，在PRG间改变两极化相对相位 | 每个占用子载波单位范数 | 相位按0°、90°逐PRG交替 |
| Beam cycling | 每个PRG只用selected SSB的一个secondary beam | 每个占用子载波单位范数 | 两个子波束逐PRG交替，每次保持1 PRG |
| Beam CDD | 两个secondary beams同时发送同一Rank-1符号，各自施加不同循环移位 | 每个占用子载波单位范数 | `S0_SIDON2`，见下一节 |

所有方案的DMRS和同一PRG中的数据必须使用相同预编码。Beam CDD必须从两个不同secondary-beam的TXRU域物理信道组合，禁止复制同一个SSB等效信道。

### 2.5 Beam CDD Sidon时延

本次采用有效带宽DFT栅格上的两波束候选：

$$
\mathcal J=\{0,1\},\qquad K_{act}=12\times48=576.
$$

第 $m$ 个波束的人工循环移位为：

$$
\tau_m=\frac{j_m}{K_{act}\Delta f}.
$$

| 参数 | 取值 |
|---|---|
| 候选名称 | `S0_SIDON2` |
| Secondary beam数量 | 2 |
| 有效带宽索引 $j_m$ | `[0, 1]`，模576 |
| 循环移位时间 | `[0, 57.870370 ns]` |
| FFT大小与采样率 | 2048 / 61.44 Msps |
| 等效FFT采样点 $d_m=j_mN_{FFT}/576$ | `[0, 3.5555556]` samples |
| 实现 | 逐占用子载波施加线性相位，精确实现分数采样循环移位 |
| 相位索引 | PDSCH内局部占用子载波 $q=0,\ldots,575$ |
| 频域相位 | $v_m[q]=e^{-j2\pi qj_m/576}=e^{-j2\pi qd_m/2048}$ |
| 人工循环周期 | $1/\Delta f=33.333333\ \mu s$ |

使用局部占用子载波作为相位原点。若实现内部改用物理FFT bin，必须补偿每个CDD分支的固定相位，使生成的576点相位向量与上式逐元素一致，最大误差不超过 $10^{-12}$。

选择该候选的理由是：它直接沿用参考CDD项目的有效带宽DFT栅格和历史 `S0_SIDON` 的前两个索引；无序二元和为 `{0,1,2}`，模576互不相同。DMRS type-1的频域comb为2，因此每个DMRS symbol有 $N_p=576/2=288$ 个不同频率位置，折叠余数为 `[0,1]`，在纯平坦两分支模型中的pilot相位矩阵应满秩且条件数为1。

上述性质只说明平坦分支模型下没有非平凡四阶加性碰撞，并且导频几何不退化。两波束时严格Sidon约束很弱，很多非退化二元集合都能满足；UMa物理簇/射线、两个secondary beams的相关性、逐子载波功率归一化和SSB-PDP协方差失配都会改变结果。因此，本次只把它作为可复现的第一候选，最终是否有效以estimated-CSI BLER为准。

先形成原始复合预编码：

$$
\mathbf w_{raw}[q]=\mathbf W_{sec}\frac{1}{\sqrt2}
\begin{bmatrix}1\\e^{-j2\pi q/576}\end{bmatrix},
$$

再逐占用子载波归一化：

$$
\widetilde{\mathbf w}[q]=\frac{\mathbf w_{raw}[q]}{\|\mathbf w_{raw}[q]\|_2}.
$$

结果中必须保存归一化前的 $\|\mathbf w_{raw}[q]\|_2^2$ 和归一化后的范数。Sidon检查针对原始相位轨迹；逐子载波归一化后的实际等效信道仍需由BLER和协方差诊断评价。

### 2.6 Monte Carlo

| 参数 | 取值 |
|---|---|
| 目标BLER范围 | 覆盖 $10^{-1}$ 和 $10^{-2}$；所有曲线低于 $10^{-2}$ 后不再向更高SNR扩展 |
| SNR选择 | 自适应：先在-12～12 dB允许范围内以2 dB步长搜索过渡区，再在 $10^{-1}$、$10^{-2}$ 交越附近增加1 dB点；不预先固定整套SNR点 |
| 主种子 | `20260913` |
| 随机流命名空间 | `plan001_preliminary_v1` |
| UE几何抽样 | 信道生成前在75～115 m内按扇区面积均匀抽样；禁止按信道或结果事后筛除 |
| 方案间公共随机样本 | UE位置、LOS/NLOS、路径/小尺度信道、TB payload和单位方差噪声 |
| 最小公共drop数 | 每个启用SNR至少200，用于避免极少样本提前判定 |
| 目标误块数 | 每条仍处于关注范围内的estimated-CSI曲线达到100 errors |
| 安全上限 | 每个SNR最多20000个公共drop；这是防失控上限，不是固定运行量 |
| 联合停止条件 | 对每条曲线分别判定“达到100 errors”或“95% Wilson上界低于 $10^{-2}$”；只有7条曲线全部满足其中一项时才停止该SNR。所有曲线使用相同公共drop数，不按方案单独截断 |
| 置信区间 | 95% Wilson区间 |
| 低于关注范围 | 只报告实际errors/N和Wilson上界，不继续追踪 $10^{-2}$ 以下BLER |
| 追加trial | 当前无；追加时继续使用本plan并采用不重叠的绝对drop区间 |

这里的drop数由误块统计自适应决定，不允许每个SNR机械运行同一个固定drop数。20000只在某些曲线难以达到100 errors、又尚不能由Wilson上界确认低于 $10^{-2}$ 时作为安全终止条件；达到上限的点必须如实报告置信区间，不做外推。

## 3. 执行方式

- 配置文件：`configs/experiments/plan-001.yaml`
- 配置检查输出：`outputs/plan-001/config_validation/`
- 正式输出：`outputs/plan-001/<run_id>/`
- 码本输入：`outputs/codebook_7ghz_8h1v/`

配置验证命令：

```powershell
py -3.11 -m sib1div.cli validate-config configs/experiments/plan-001.yaml --output outputs/plan-001/config_validation
```

推荐使用项目根目录的一键PowerShell入口：

```powershell
.\run_plan001.ps1 -RunId run-001
```

该命令自动读取冻结YAML和码本，顺序完成粗SNR搜索、公共drop误块停止、交越区1 dB补点、7条estimated-CSI曲线、4条Perfect-CSI诊断曲线、CSV和图片。终端每10个公共drop打印一次进度。若运行被中断，使用相同 `RunId` 重复执行同一命令会从聚合checkpoint继续；完成后的目录不会被同名命令覆盖。

## 4. 运行前检查与结果有效条件

平台级数值检查不在本plan中重复展开，统一引用 `outputs/platform_acceptance/平台基础功能调试报告.md`。Plan 001只保留以下必要条件：

- [x] 配置、75～115 m UE区域和冻结码本一致；码本SHA-256校验通过。
- [x] 正式runner已实现7条estimated-CSI组合、自适应SNR和公共drop联合停止，并通过缩小规模端到端集成测试。
- [ ] 每个drop的信道、TB、噪声及selected-SSB归一化因子在所有曲线间共用，且归一化功率检查为1。
- [ ] 运行中无NaN、Inf、协方差求解失败或CRC/统计记录异常。
- [ ] 每个SNR保存真实误块数、公共样本数、BLER、Wilson区间和停止原因；曲线覆盖 $10^{-1}$、$10^{-2}$ 附近，不外推更低BLER。

任一项未满足时，`result-001.md`必须标记为“未完成”并说明失败位置，不能据此比较方案优劣。

## 5. 预期输出

- `bler.csv`：7条estimated-CSI曲线和4条Perfect-CSI诊断曲线的绘图数据；每行只含SNR、方案、协方差标签、误块数、公共样本数、BLER、Wilson上下界和停止原因。
- `nmse.csv`：各estimated-CSI组合在每个SNR的平均NMSE。
- `paired_counts.csv`：需要直接比较的曲线对仅保存聚合的 `n00/n01/n10/n11` 配对计数，不保存逐drop错误标志。
- `run_metadata.json`：保存曲线ID映射、实际SNR点、随机种子、停止规则、环境版本、配置摘要及冻结码本SHA-256。
- `bler.png`、`bler_perfect_csi.png`和`nmse.png`：由上述CSV直接生成。

正式结果默认不保存逐drop信道、路径、完整协方差矩阵、逐RE数据、逐drop CDD功率谱或完整 `link_results.csv`。这些内容只在定位异常时按指定drop临时导出，不进入常规Plan 001产物。

仿真完成后，研究者将完整的 `outputs/plan-001/<run_id>/` 目录交给Codex审核；审核通过后再创建 `research/result-001.md`，嵌入主BLER、Perfect CSI和CE NMSE曲线并解释结果。
