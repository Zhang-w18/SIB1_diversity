# SIB1 传输分集链路级仿真计划

## 1. 系统设置

### 1.1 场景

主场景采用 3GPP TR 38.901 UMa：

- 载频：0.7 GHz、2 GHz、7 GHz 可配置，默认 7 GHz
- 子载波间隔：可配置，默认 30 kHz
- BS 高度：25 m
- ISD：500 m
- 每个 drop：1 个 UE
- UE 水平角：在扇区 -60° 到 60° 内均匀随机
- UE 二维距离：在 35 m 到 115 m 的扇区面积内均匀随机
- 每个 drop 独立生成 LOS/NLOS 状态、路径损耗、阴影衰落以及小尺度信道参数

115 m 作为本仿真的有效 UE drop 半径。

对于只研究给定信道下相对分集性能、而不研究覆盖概率的具体实验，可以在生成信道之前，根据固定垂直波束的有效主瓣预先约束 UE 几何位置。距离或垂直角的具体范围取决于 BS/UE 高度、面板机械/电下倾、AE 间距和 AE 到 TXRU 映射，必须写入该次实验的 plan 和配置文件，不能在全局方法中固定为通用数值。禁止先生成瞬时信道，再根据 RSRP、波束增益、信道实现或解码结果删除低增益 drop，否则会引入事后选择偏差。

每个 drop 重新生成 38.901 UMa 信道，因此 cluster/ray 的功率、时延、AOD、AOA、ZOD、ZOA 和随机相位均随 drop 变化。

### 1.2 BS 与 UE 天线阵列

不同载频下的可用天线配置
Around 0.7 GHz:

4 TXRUs, 32AEs, (M, N, P, Mg, Ng; Mp, Np) = (8, 2, 2, 1, 1; 1, 2), (dH,
dV) = (0.5, 0.5), (“Baseline Configuration”)

Around 2 GHz:

4 TXRUs, 32 AEs, (M, N, P, Mg, Ng; Mp, Np) = (8, 2, 2, 1, 1; 1, 2), (dH,
dV) = (0.5, 0.5), (“Outdoor Combination 1”)

32 TXRUs, 128 AEs, (M, N,
P, Mg, Ng; Mp, Np) = (8, 8, 2, 1, 1; 2, 8), (dH, dV) =
(0.5, 0.5)

64 TXRUs, 192 AEs, (M, N, P, Mg, Ng; Mp, Np) = (12, 8, 2, 1, 1; 4, 8),
(dH, dV) = (0.5, 0.5), (“Outdoor Combination 2”)
Around 7 GHz:

128 TXRUs, 768 AEs, (M, N, P, Mg, Ng, Mp, Np) = (24, 16, 2, 1, 1; 4, 16), (dH, dV) = (0.5, 0.8),
(“Outdoor Combination 1”)

256 TXRUs, 1024 AEs, (M, N, P, Mg, Ng, Mp, Np) = (32, 16, 2, 1, 1; 8, 16), (dH, dV) = (0.5, 0.8),
(“Outdoor Combination 2”)

512 TXRUs, 2048AEs, (M, N, P, Mg, Ng; Mp, Np) =  (64, 16, 2, 1, 1, 16, 16), (dH, dV) = (0.5, 0.5),
(“Outdoor Combination 5”)

256
TXRUs, 1536 AEs, (M, N, P, Mg, Ng; Mp, Np) = (48, 16, 2, 1, 1, 8, 16). (dH,
dV) = (0.5, 0.8), (“Outdoor Combination 3”)

128 TXRUs, 2048AEs, (M, N, P,
Mg, Ng; Mp, Np) =  (64, 16, 2, 1, 1, 8,
8), (dH, dV)
= (0.5, 0.5)


对于7GHz，默认BS 阵列采用：

$$
(M,N,P,M_g,N_g;M_p,N_p)=(24,16,2,1,1;4,16)
$$

其中：

- M：每个面板、每种极化，垂直方向天线单元数
- N：每个面板、每种极化，水平方向天线单元数
- P：每个天线位置的极化数
- M_g：垂直方向面板数
- N_g：水平方向面板数
- M_p：每个面板、每种极化，垂直方向 TXRU 数
- N_p：每个面板、每种极化，水平方向 TXRU 数

物理单元间距：

$$
(d_H,d_V)=(0.5,0.8)\lambda
$$

本文后续统一采用 $d_H=0.5\lambda$、$d_V=0.8\lambda$。代码配置必须分别使用带方向含义的字段名，禁止仅依赖二元组位置顺序，避免水平与垂直间距互换。

物理极化天线单元数为：

$$
24\times16\times2=768
$$

每种极化有：

$$
4\times16=64
$$

个 TXRU，因此两种极化共 128 个极化 TXRU 端口。

UE 配置为 1T4R，默认采用 1×2 双极化接收阵列。以下信道估计公式均针对单个 UE 接收分支描述，4R 分别估计后做 MRC；后续可扩展 IRC。

BS 面板姿态必须可配置，包括扇区方位角和下倾角。定义：

- phi_panel：面板 boresight 的全局水平指向
- eta_tilt：面板下倾角，向下为正
- eta：某个全局传播方向相对水平面的下倾角，向下为正

对于 boresight 垂直切面，可以用：

$$
\eta_{\rm local}
=
\eta-\eta_{\rm tilt}
$$

理解面板下倾后的局部角度。正式实现应采用三维旋转矩阵，把系统级信道产生的全局 AOD/ZOD 转换到面板局部坐标系后再计算阵列响应。若所用 38.901 信道模块已经根据 panel orientation 输出局部角度，则码本生成直接使用该局部角度，不能再次施加下倾，以避免 double counting。

下倾角作为仿真输入参数保存。默认值可根据部署场景设置，例如 10° 到 15°，并允许扫参。

### 1.3 SIB1 参数

SIB1 采用：

- Rank 1
- 单逻辑 DMRS port
- DCI 1_0
- SI-RNTI = 0xFFFF
- QPSK
- MCS index 0，采用 TS 38.214 Table 5.1.3.1-1，对应 $Q_m=2$、目标码率 $R=120/1024$
- PRG size = 2 PRB
- PDSCH 连续频域分配：24、48、96 PRB 可配置；这些数值表示 PDSCH allocation，而不是必须等于载波或 BWP 的总 PRB 数

为便于快速实现且满足标准，主实验采用以下固定时频配置：

- 30 kHz SCS，normal CP，每 slot 14 个 OFDM symbols
- BWP 至少包含 96 PRB；7 GHz 主配置可采用 40 MHz、106 PRB 的载波/BWP，并在其中居中分配 24、48 或 96 PRB 给 SIB1 PDSCH
- slot 内 symbol 0--1 预留给 CORESET0/PDCCH；当前链路级仿真不生成 PDCCH 波形，只保证 PDSCH 不占用这些 RE
- PDSCH mapping type A，start symbol = 2，length = 12，即占用 symbol 2--13
- 单层传输，RV = 0，$xOverhead=0$，不配置 PTRS
- PDSCH DMRS configuration type 1、single-symbol DMRS、$dmrs$-TypeA-Position = pos2、$dmrs$-AdditionalPosition = pos1
- DMRS 位于 slot symbol 2 和 11；使用逻辑 port 1000、1 个 CDM group without data，每个 DMRS symbol 每 PRB 占 6 个 RE
- PDSCH data 与同一 PRG 内 DMRS 必须使用完全相同的有效预编码向量
- PDSCH scrambling 按 TS 38.211 生成，使用 SI-RNTI 和配置的小区/扰码 ID

在上述配置下，每 PRB 可用于 PDSCH data 的 RE 数为：

$$
N_{\rm RE,PRB}=12\times12-2\times6=132.
$$

按 TS 38.214 的 TBS 算法，主配置的 TBS 为：

| PDSCH allocation | $N_{\rm RE,PRB}$ | MCS | TBS |
|---:|---:|---:|---:|
| 24 PRB | 132 | 0 | 736 bit |
| 48 PRB | 132 | 0 | 1480 bit |
| 96 PRB | 132 | 0 | 2976 bit |

上述 TBS 均不超过 3824 bit，因此 DL-SCH TB CRC 按 TS 38.212 使用 16 bit CRC；后续若改变时域资源、DMRS 开销、MCS或层数，必须重新计算 TBS 和 CRC 类型，不能继续沿用该表。

TBS 按 TS 38.214 计算，DL-SCH LDPC 按 TS 38.212 编码。所有传输方案保持相同的时频资源、总发射功率、DMRS、TBS 和编码参数。

---

## 2. SSB beam sweep 与统一码本生成框架

每个 drop 执行：

$$
\text{随机UE位置}
\rightarrow
\text{UMa随机信道}
\rightarrow
\text{SSB beam sweep}
\rightarrow
b^\star
\rightarrow
\text{SIB1 transmission}
$$

SSB codebook 和 secondary beam codebook 均为离线固定码本，不读取当前 drop 的瞬时 cluster AOD。不同载频和阵列配置使用同一套码本生成方法，只改变阵列参数、TXRU 映射、目标覆盖角域和波束数量。

### 2.1 载频对应的 SSB coverage layout

扇区水平覆盖固定为：

$$
\phi\in[-60^\circ,60^\circ]
$$

7 GHz 主配置使用 8 个 SSB 波束，布局为 8H×1V，即水平方向 8 个、垂直方向 1 个。每个 selected SSB 默认沿水平方向拆成 2 个 secondary beams，对应全扇区 16H×1V 的 secondary-beam 码本。4H×2V 作为可选 SSB 配置保留。默认面板机械下倾角为 12°，不增加额外电下倾。

垂直覆盖由 BS/UE 高度、UE drop 半径和面板下倾共同确定。若 BS 高度为 25 m、UE 高度为 1.5 m、二维距离为 35 m 到 115 m，则 UE 相对 BS 的几何下倾角约为：

$$
\eta_{\min}
=
\arctan\frac{25-1.5}{115}
$$

$$
\eta_{\max}
=
\arctan\frac{25-1.5}{35}
$$

即大约 11.5° 到 33.9°。该范围定义在全局坐标系中。若面板下倾为 eta_tilt，则 boresight 垂直切面上的局部目标范围约为：

$$
[\eta_{\min}-\eta_{\rm tilt},
\eta_{\max}-\eta_{\rm tilt}]
$$

不同阵列配置保持相同的目标角域划分，阵列规模只影响实现该目标波束时能够达到的峰值增益、平坦度和旁瓣。

若某个具体实验只比较受控主瓣区域内的分集性能，则该实验可以缩小 UE 距离或垂直角范围，并只在缩小后的角域内验收码本；相应范围、采样方法和结果适用边界必须在该实验 plan 中明确记录，结果不得解释为全小区覆盖性能。

### 2.2 AE 到 TXRU 的统一建模

以下先针对单极化描述空间权重。每个极化的 AE 数和 TXRU 数分别为：

$$
N_{\rm AE}=MN
$$

$$
N_{\rm TXRU}=M_pN_p
$$

定义固定的 AE 到 TXRU 映射矩阵：

$$
\mathbf F_{\rm TXRU}
\in
\mathbb C^{N_{\rm AE}\times N_{\rm TXRU}}
$$

矩阵第 t 列描述第 t 个 TXRU 对其所连接 AE 的固定幅相映射。对于全数字 AE-level 仿真：

$$
\mathbf F_{\rm TXRU}
=
\mathbf I
$$

对于 subarray/TXRU 架构，则由具体 AE 分组方式生成 F_TXRU，并对每个 TXRU 的列向量做功率归一化。

设某个方向在面板局部坐标系中的水平角为 phi，向下的垂直角为 eta。定义：

$$
u_H
=
\cos\eta\sin\phi
$$

$$
u_V
=
-\sin\eta
$$

对于 M × N URA，第 m 行、第 n 列 AE 的阵列响应为：

$$
[\mathbf a_{\rm AE}(\phi,\eta)]_{m,n}
=
e^{j2\pi(nd_Hu_H+md_Vu_V)}
$$

其中：

$$
m=0,\ldots,M-1
$$

$$
n=0,\ldots,N-1
$$

TXRU 域等效 steering vector 为：

$$
\mathbf a_{\rm TXRU}(\phi,\eta)
=
\mathbf F_{\rm TXRU}^H
\mathbf a_{\rm AE}(\phi,\eta)
$$

若某个波束的 TXRU 域数字权重为 v，则实际 AE 权重为：

$$
\mathbf w
=
\mathbf F_{\rm TXRU}\mathbf v
$$

所有波束最终统一归一化为：

$$
\|\mathbf w\|^2=1
$$

从而不同 SSB 和 SIB1 方案保持相同总发射功率。

链路仿真不需要在所有 OFDM 子载波上显式构造完整 AE 域信道矩阵。38.901 的 ray/cluster 参数仍在 AE 几何和极化模型上生成，但每条路径优先通过固定映射矩阵投影到 TXRU 域：

$$
\mathbf H_{{\rm TXRU},l}
=
\mathbf H_{{\rm AE},l}\mathbf F_{\rm TXRU}.
$$

后续 SSB、Beam cycling 和 Beam CDD 都在同一组 TXRU 域路径信道上施加数字权重。只在阵列与 TXRU 映射单元测试中构造小尺寸完整 AE 域信道，用于验证 TXRU 域投影与直接 AE 域计算一致。

### 2.3 SSB 宽波束的目标定义

第 b 个 SSB 对应一个固定的二维目标覆盖区域：

$$
\Omega_b
=
\Omega_{H,b}
\times
\Omega_{V,b}
$$

例如 7 GHz 的 4H × 2V 设计中，水平方向将 120° 扇区分为 4 个区域，垂直方向将目标下倾范围分为 2 个区域。

SSB 宽波束的设计目标是在 Omega_b 内形成尽可能平坦的增益，同时抑制区外旁瓣。定义实际方向图：

$$
G_b(\phi,\eta)
=
\left|
\mathbf a_{\rm AE}^H(\phi,\eta)
\mathbf w_b
\right|^2
$$

理想的 max-min 设计可写为：

$$
\max_{\mathbf v_b}
\min_{(\phi,\eta)\in\Omega_b}
G_b(\phi,\eta)
$$

并约束：

$$
G_b(\phi,\eta)
\leq
G_{\rm SL},
\qquad
(\phi,\eta)\notin\Omega_b
$$

以及：

$$
\|
\mathbf F_{\rm TXRU}\mathbf v_b
\|^2
=
1
$$

该问题可使用迭代凸化、projected gradient 或其他阵列综合优化器求解。

### 2.4 推荐的实际宽波束权重求解方法

第一版 LLS 默认采用离散角度网格上的 weighted least-squares 迭代。

在目标区域及其旁瓣区域选取角度采样点：

$$
\mathcal G
=
\{
(\phi_q,\eta_q)
\}_{q=1}^{Q}
$$

构造：

$$
\mathbf A
=
[
\mathbf a_{\rm TXRU}(\phi_1,\eta_1),
\ldots,
\mathbf a_{\rm TXRU}(\phi_Q,\eta_Q)
]
$$

定义目标幅度：

$$
d_q
=
\begin{cases}
\sqrt{G_0}, & (\phi_q,\eta_q)\in\Omega_b\\
0, & (\phi_q,\eta_q)\notin\Omega_b
\end{cases}
$$

由于波束综合只约束目标功率而不预先知道最优相位，采用迭代 phase update。第 r 次迭代根据当前权重设置目标复场：

$$
\tilde d_q^{(r)}
=
d_q
e^{j\angle{
\mathbf a_{\rm TXRU}^H(\phi_q,\eta_q)
\mathbf v_b^{(r)}
}}
$$

然后求解：

$$
\mathbf v_b^{(r+1)}
=
\arg\min_{\mathbf v}
\left\|
\mathbf W^{1/2}
(
\mathbf A^H\mathbf v
-
\tilde{\mathbf d}^{(r)}
)
\right\|^2
+
\mu\|\mathbf v\|^2
$$

其 ridge-LS 闭式解为：

$$
\mathbf v_b^{(r+1)}
=
(
\mathbf A\mathbf W\mathbf A^H
+
\mu\mathbf I
)^{-1}
\mathbf A\mathbf W
\tilde{\mathbf d}^{(r)}
$$

其中 W 用于提高目标覆盖区边缘或旁瓣区域的优化权重，mu 为正则化系数。每次迭代后按 AE 域总功率归一化：

$$
\mathbf v_b
\leftarrow
\frac{\mathbf v_b}
{
\|
\mathbf F_{\rm TXRU}\mathbf v_b
\|
}
$$

迭代到目标区最小增益和 ripple 收敛。最终保存每个 SSB 的固定权重，不随 drop 更新。

若不希望运行优化器，可采用简化初始化方案：在 Omega_b 内放置若干细粒度 steering beams 并做加权相干叠加，再归一化：

$$
\mathbf v_b^{(0)}
=
\sum_{r\in\mathcal B_b}
c_r
\frac{
\mathbf a_{\rm TXRU}(\phi_r,\eta_r)
}{
\|
\mathbf a_{\rm TXRU}(\phi_r,\eta_r)
\|
}
$$

该结果也可作为上述 LS 优化的初值。

### 2.5 Secondary 窄波束码本

selected SSB 的目标覆盖区域记为：

$$
\Omega_{b^\star}
$$

将其进一步划分为：

$$
N_{{\rm sec},H}
\times
N_{{\rm sec},V}
$$

个子区域：

$$
\Omega_{b^\star,s},
\qquad
s=1,\ldots,N_{\rm sec}
$$

其中：

$$
N_{\rm sec}
=
N_{{\rm sec},H}
N_{{\rm sec},V}
$$

默认研究：

- 2H × 1V
- 4H × 1V
- 2H × 2V

为避免大扫描角附近 beamwidth 不均匀，优先在方向余弦 u_H 和 u_V 域均匀划分，再转换回实际角度。

Secondary beam 有两种生成模式。

第一种为 pencil beam，用于最大化某个子区域中心方向的阵列增益。设子区域中心为：

$$
(\phi_s,\eta_s)
$$

则：

$$
\mathbf v_s^{\rm pencil}
=
\frac{
\mathbf a_{\rm TXRU}(\phi_s,\eta_s)
}{
\|
\mathbf F_{\rm TXRU}
\mathbf a_{\rm TXRU}(\phi_s,\eta_s)
\|
}
$$

第二种为 sub-sector flat-top beam。将目标区域 Omega_b 换成 Omega_b,s，使用与 SSB 宽波束完全相同的 weighted-LS/max-min 方法求解。主仿真建议使用该模式，使每个 secondary beam 对应一个明确子覆盖区域，而不是单一角度的理想 pencil beam。

Beam cycling 与 Beam CDD 共用同一套 secondary beam codebook，从而保证两种传输方案比较时只有频域使用方式不同。

### 2.6 双极化权重

以上空间码本先针对单极化生成。Baseline SSB 可以在两个极化上使用相同空间权重：

$$
\mathbf w_{{\rm SSB},H}
=
\mathbf w_{{\rm SSB},V}
$$

两个极化合成单逻辑端口时采用：

$$
\mathbf w_{\rm SSB}^{\rm dual-pol}
=
\frac{1}{\sqrt 2}
\begin{bmatrix}
\mathbf w_{\rm SSB}\\
e^{j\phi_{\rm pol}}
\mathbf w_{\rm SSB}
\end{bmatrix}
$$

Baseline 默认：

$$
\phi_{\rm pol}=0
$$

Pol-cycling 再在 PRG level 改变该相对相位。

### 2.7 SSB beam sweep

对每个 drop，Sionna UMa 首先生成包含路径损耗、阴影衰落和小尺度衰落的物理频域信道。记第 $r$ 个接收分支、第 $k$ 个占用子载波上的 TXRU 域行信道为 $\mathbf H_r[k]$。使用固定 SSB codebook 计算每个候选 SSB 的瞬时宽带平均每接收分支功率：

$$
P_b
=
\frac{1}{N_rK}
\sum_{r=1}^{N_r}
\sum_{k=1}^{K}
\left|
\mathbf H_r[k]\mathbf w_b
\right|^2,
$$

UE 选择：

$$
b^\star
=
\arg\max_b
P_b
$$

SIB1 随后关联该 selected SSB，并调用其对应的 secondary beam codebook。

### 2.8 `normalized_link` 的逐 drop 公共 SNR 参考

主 BLER 对比的目的，是在相同的受控 SNR 参考下比较四种传输方案的分集、预编码和信道估计性能，而不是同时比较随机 UE 位置、路径损耗、阴影衰落或 SSB 覆盖增益。因此，完成 SSB 选择后，对整个物理信道使用 selected-SSB 功率形成的单一公共归一化因子：

$$
\widetilde{\mathbf H}_r[k]
=
\frac{\mathbf H_r[k]}{\sqrt{P_{b^\star}}}.
$$

由定义可得：

$$
\frac{1}{N_rK}
\sum_{r=1}^{N_r}
\sum_{k=1}^{K}
\left|
\widetilde{\mathbf H}_r[k]\mathbf w_{b^\star}
\right|^2
=1.
$$

随后在同一个 $\widetilde{\mathbf H}$ 上运行 Baseline、Pol-cycling、Beam cycling 和 Beam CDD，并为四种方案复用相同的传输块和单位方差噪声样本。给定横轴值 $\gamma_{\rm ref}$ 时，单位能量调制符号使用：

$$
\sigma_n^2=10^{-\gamma_{\rm ref,dB}/10}.
$$

因此，不同 drop 的 selected-SSB 瞬时宽带平均每 Rx 参考 SNR 保持一致，随机的大尺度链路增益和整体信道能量被抹平。该归一化只消除一个标量总增益，不改变多径相对功率、随机相位、频率选择性、Rx 分支间差异，以及不同预编码方案相对 selected SSB 的有效信道增益。

这里“固定 SNR”严格指公共 selected-SSB 参考 $E_s/N_0$，不表示四种方案经过各自预编码后的实际接收 SNR 被再次强制为相同。禁止按方案分别归一化 $\mathbf H[k]\mathbf w_{\rm scheme}[k]$；否则会抹掉 Beam cycling、Beam CDD 等方案相对 Baseline 的真实增益或损失，使 BLER 对比失去意义。

该方法回答的是“给定相同 selected-SSB 参考信道质量时，哪种传输方案的 BLER 更好”，不回答小区覆盖概率或真实发射功率下不同位置 UE 的绝对解码成功率。后者必须使用保留大尺度衰落的 `coverage_link_budget` 模式，并与本模式的曲线分开解释。

### 2.9 码本验证与输出

每一个载频和阵列配置在进入 Monte Carlo 仿真前，先离线验证码本。至少输出：

- 全部 SSB beams 的水平极坐标方向图
- 全部 SSB beams 的垂直极坐标方向图
- selected SSB 与其 secondary beams 的叠加方向图
- 每个目标覆盖区内的最小、最大和平均增益
- in-sector ripple
- inter-beam crossing level
- out-of-sector sidelobe level
- 不同下倾角配置下的全局坐标方向图

所有方向图统一在全局坐标系展示，因此图中已经包含面板下倾角的影响。

---

## 3. 对比传输方案

主实验包含 1 个 Baseline 和 3 个候选方案，共 4 个方案：Baseline、Pol-cycling、Beam cycling 和 Beam CDD。四个方案必须使用相同的 drop、传输块、编码、时频资源和噪声样本进行成对比较。

### 3.1 Baseline

所有 PRG 使用 selected SSB 的同一 Rank-1 预编码：

$$
\mathbf w_{\rm SSB}
$$

### 3.2 Pol-cycling

两个极化采用相同空间波束，第 q 个 PRG 使用：

$$
\mathbf p_{\rm pol}[q]
=
\frac{1}{\sqrt 2}
\begin{bmatrix}
1\\
e^{j\phi_q}
\end{bmatrix}
$$

默认第二极化的相位按以下方式循环：

$$
1,\ j,\ 1,\ j,\ldots
$$

即：

$$
\phi_q
=
0,\frac{\pi}{2},0,\frac{\pi}{2},\ldots
$$

可配置 2-state 或 4-state phase cycling。

### 3.3 Beam cycling

第 q 个 PRG 只使用一个 secondary beam：

$$
\mathbf w_{b(q)}
$$

例如两个 secondary beams 时：

$$
\mathbf w_1,
\mathbf w_2,
\mathbf w_1,
\mathbf w_2,\ldots
$$

可配置：

- secondary beam 数
- 水平或二维划分
- 每个 beam 连续占用的 PRG 数
- beam cycling pattern

### 3.4 Beam CDD

Beam CDD 同时使用 K 个 secondary beams，并给每个 beam 施加不同人工 cyclic delay。

这里的 $\delta_m$ 明确定义为 OFDM 有用符号内的循环移位，而不是真实传播时延。实现时在频域施加循环移位对应的线性相位，并用时域循环移位与频域相位旋转的一致性测试验证：

$$
x_m[(n-\Delta_m)\bmod N_{\rm FFT}]
\Longleftrightarrow
X_m[k]e^{-j2\pi k\Delta_m/N_{\rm FFT}}.
$$

$\Delta_m$ 默认取整数采样点并按 $N_{\rm FFT}$ 取模；真实物理多径仍由 38.901 的 $\tau_l$ 描述，二者不得混为同一参数。

定义：

$$
\mathbf W_{\rm sec}
=
[
\mathbf w_1,
\mathbf w_2,
\ldots,
\mathbf w_K
]
$$

第 m 个 beam 的人工时延为：

$$
\delta_m
$$

第 k 个子载波对应的 CDD 权重向量为：

$$
\mathbf p_{\rm CDD}[k]
=
\frac{1}{\sqrt K}
\begin{bmatrix}
e^{-j2\pi f_k\delta_1}\\
e^{-j2\pi f_k\delta_2}\\
\vdots\\
e^{-j2\pi f_k\delta_K}
\end{bmatrix}
$$

因此实际发射预编码为：

$$
\mathbf w_{\rm CDD}[k]
=
\mathbf W_{\rm sec}
\mathbf p_{\rm CDD}[k]
$$

包括 Baseline、Pol-cycling、Beam cycling 和 Beam CDD 在内，主实验对每个占用子载波分别实施单位范数约束。对任意方案 $s$ 和任意占用子载波 $k\in\mathcal K$：

$$
\left\|\mathbf w_s[k]\right\|_2^2
=1.
$$

对于 Beam CDD，先计算原始相位型复合预编码：

$$
\mathbf w_{\rm CDD,raw}[k]
=
\mathbf W_{\rm sec}\mathbf p_{\rm CDD}[k],
$$

再用于主 BLER 链路：

$$
\widetilde{\mathbf w}_{\rm CDD}[k]
=
\frac{\mathbf w_{\rm CDD,raw}[k]}
{\left\|\mathbf w_{\rm CDD,raw}[k]\right\|_2}.
$$

这样每个 RE 上承载的单位能量调制符号都对应相同总发射功率，最终 BLER 差异主要来自不同预编码对物理路径的加权、频率选择性和分集，而不是发射功率谱起伏。

平台同时保留“仅全带平均归一化”的 raw CDD 作为可选配置：

$$
\frac{1}{|\mathcal K|}
\sum_{k\in\mathcal K}
\left\|\mathbf w_{\rm CDD,raw}[k]\right\|_2^2
=1.
$$

该模式不作为第一轮主公平对比结果。如果 $\mathbf W_{\rm sec}$ 的列正交，则原始 CDD 预编码已经在所有子载波上保持单位范数，两种模式完全一致；如果不正交，两种模式可能不同。

仿真必须同时记录原始CDD预编码的发射功率谱：

$$
P_{\rm tx,raw}[k]
=
\left\|\mathbf w_{\rm CDD,raw}[k]\right\|_2^2
$$

用于判断 secondary beams 的非正交性是否会引入明显的发射谱功率起伏。

配置：

$$
K\in\{2,4,8\}
$$

默认：

$$
\delta_m=(m-1)\Delta\delta
$$

Beam CDD 必须从 K 个不同 secondary beams 的物理信道开始计算。禁止先得到一个 SSB 等效信道后复制 K 份，否则所有分支完全相关，不产生 K 阶空间分集。

---

## 4. PDP 与频域协方差的统一定义

### 4.1 物理路径模型

对某一个 UE 接收天线，令第 l 条物理 ray 或可分辨路径具有：

- 长期平均功率：P_l
- 时延：tau_l
- 发射方向：theta_l
- 随机复系数：alpha_l

物理信道写为：

$$
\mathbf h(f)
=
\sum_{l=1}^{L}
\alpha_l
\mathbf a_t(\theta_l)
e^{-j2\pi f\tau_l}
$$

其中：

- L 为 ray 或可分辨路径数量
- a_t(theta_l) 为 BS 阵列对第 l 条路径出射方向的阵列响应
- alpha_l 满足：

$$
E\{|\alpha_l|^2\}=P_l
$$

主仿真假设不同 ray 的随机复增益互不相关：

$$
E\{\alpha_l\alpha_r^\ast\}
=
P_l\delta_{l,r}
$$

这里使用的是 38.901 信道生成器内部的长期 path/ray power，而不使用某一次瞬时 realization 的：

$$
|\alpha_l|^2
$$

作为 PDP 功率。

因此仿真中无需另外运行 PDP 估计算法。PDP 由信道模型内部的 path delay、path power、AOD 以及已知 TX precoder 直接构造。

### 4.2 任意发射波束对应的 PDP

对于任意发射波束：

$$
\mathbf w_x
$$

定义第 l 条路径经过该波束后的复数阵列增益：

$$
\beta_{x,l}
=
\mathbf a_t^H(\theta_l)\mathbf w_x
$$

该波束对应的等效路径功率为：

$$
P_{x,l}
=
P_l|\beta_{x,l}|^2
$$

因此该波束的等效 PDP 为：

$$
S_x(\tau)
=
\sum_{l=1}^{L}
P_{x,l}
\delta(\tau-\tau_l)
$$

这里用 S_x(tau) 表示 PDP，避免与接收功率变量混淆。

对应的频域协方差矩阵为：

$$
R_x[k,n]
=
\sum_{l=1}^{L}
P_{x,l}
e^{-j2\pi(f_k-f_n)\tau_l}
$$

LMMSE 使用该协方差矩阵，而不是直接使用 PDP 曲线。

若 PDP 只保留归一化形状，则还需要单独保留该波束的总平均功率。主仿真建议直接使用未归一化 PDP，从而使协方差矩阵同时包含频率相关性和平均信道功率。

---

## 5. SSB PDP 与 secondary-beam PDP

selected SSB 的预编码向量记为：

$$
\mathbf w_{\rm SSB}
$$

其等效 PDP 为：

$$
S_{\rm SSB}(\tau)
=
\sum_{l=1}^{L}
P_l
\left|
\mathbf a_t^H(\theta_l)\mathbf w_{\rm SSB}
\right|^2
\delta(\tau-\tau_l)
$$

第 m 个 secondary beam 的 PDP 为：

$$
S_m(\tau)
=
\sum_{l=1}^{L}
P_l
\left|
\mathbf a_t^H(\theta_l)\mathbf w_m
\right|^2
\delta(\tau-\tau_l)
$$

不同波束的物理路径时延 tau_l 来自同一个传播环境，因此路径时延集合相同；但不同波束对各路径功率的加权不同，所以通常：

$$
S_m(\tau)
\neq
S_{\rm SSB}(\tau)
$$

也通常存在：

$$
S_m(\tau)
\neq
S_n(\tau)
$$

这正是 secondary beam 引入 covariance mismatch 的来源。

仿真中以上 PDP 均直接从当前 drop 的 38.901 path/ray 参数和已知预编码向量计算。接收机侧假设 UE 能够从 selected SSB 的 DMRS 准确估计 $S_{\rm SSB}(\tau)$，且该 PDP 估计没有误差。因此，SSB-PDP 接收机可直接使用当前 drop 的真实 $S_{\rm SSB}(\tau)$；本阶段不仿真有限 SSB DMRS、噪声或 PDP 估计算法造成的误差。

---

## 6. Beam cycling 的信道估计假设

Beam cycling 在某个 PRG 内只使用一个 secondary beam，因此该 PRG 的实际等效信道就是对应窄波束信道。

设第 q 个 PRG 使用 beam：

$$
m=b(q)
$$

### 6.1 Ideal-PDP LMMSE

使用该 PRG 实际窄波束的真实 PDP：

$$
S_{\rm ideal}^{(q)}(\tau)
=
S_{b(q)}(\tau)
$$

由此计算：

$$
R_{\rm ideal}^{(q)}[k,n]
=
\sum_l
P_{b(q),l}
e^{-j2\pi(f_k-f_n)\tau_l}
$$

这是 Beam cycling 的 matched covariance 接收机，可作为理想基准。

### 6.2 SSB-PDP LMMSE

UE 只使用 selected SSB 获得的 PDP：

$$
S_{\rm est}^{(q)}(\tau)
=
S_{\rm SSB}(\tau)
$$

所有 PRG 都使用：

$$
R_{\rm SSB}
$$

做 LMMSE。

实际信道对应：

$$
R_{b(q)}
$$

因此：

$$
R_{\rm SSB}
\neq
R_{b(q)}
$$

时会产生 covariance mismatch penalty。

重点统计：

$$
\Delta_{\rm mismatch}^{\rm BC}
=
\text{BLER}_{\rm SSB-PDP}
-
\text{BLER}_{\rm Ideal-PDP}
$$

也可在固定 BLER 下统计所需 SNR 差值。



---

## 7. Beam CDD 的 PDP 与协方差

### 7.1 各窄波束分支信道

第 m 个 secondary beam 的频域信道为：

$$
g_m(f)
=
\sum_{l=1}^{L}
\alpha_l
\beta_{m,l}
e^{-j2\pi f\tau_l}
$$

其中：

$$
\beta_{m,l}
=
\mathbf a_t^H(\theta_l)\mathbf w_m
$$

Beam CDD 等效信道为：

$$
h_{\rm CDD}(f)
=
\frac{1}{\sqrt K}
\sum_{m=1}^{K}
g_m(f)
e^{-j2\pi f\delta_m}
$$

代入路径模型后：

$$
h_{\rm CDD}(f)
=
\frac{1}{\sqrt K}
\sum_{m=1}^{K}
\sum_{l=1}^{L}
\alpha_l
\beta_{m,l}
e^{-j2\pi f(\tau_l+\delta_m)}
$$

因此 Beam CDD 可以看成把每个 secondary-beam 信道在时延域平移 delta_m 后相加。

### 7.2 一个重要区别：CDD 的 PDP 可能不足以描述完整统计量

由于所有 secondary beams 都来自同一个物理传播信道，同一条物理路径的随机系数 alpha_l 会同时出现在多个 beam 分支中。

因此一般存在：

$$
E\{g_m(f_1)g_n^\ast(f_2)\}
\neq 0
$$

也就是不同 beam 分支相关。

此时即使定义 CDD 后的等效 PDP：

$$
S_{\rm CDD}(\tau)
=
E\{|h_{\rm CDD}(\tau)|^2\}
$$

仅知道该 PDP 仍可能不足以完全恢复频域协方差，因为不同人工延迟后的 taps 之间可能存在交叉相关。

因此 Beam CDD 的理想 LMMSE 应优先直接构造完整频域协方差，而不是强制先压缩成一个 PDP。

---

## 8. Beam CDD 的三种信道估计假设

### 8.1 Ideal covariance

首先构造 beam-domain 联合频域协方差。

第 m、n 个 secondary beam 之间：

$$
R_{g,mn}[k,r]
=
E\{g_m(f_k)g_n^\ast(f_r)\}
$$

根据当前仿真的 path/ray 参数可直接计算：

$$
R_{g,mn}[k,r]
=
\sum_{l=1}^{L}
P_l
\beta_{m,l}
\beta_{n,l}^\ast
e^{-j2\pi(f_k-f_r)\tau_l}
$$

CDD 后的理想等效协方差为：

$$
R_{\rm CDD}^{\rm ideal}[k,r]
=
\frac{1}{K}
\sum_{m=1}^{K}
\sum_{n=1}^{K}
e^{-j2\pi f_k\delta_m}
R_{g,mn}[k,r]
e^{j2\pi f_r\delta_n}
$$

该矩阵包含：

- 每个窄波束自己的 PDP
- 不同窄波束之间的相关性
- 已知人工 CDD 时延

这是非透明 Beam CDD 接收机的理论上界。

在仿真中，该矩阵直接由 38.901 path/ray power、delay、AOD、secondary beam weights 和 CDD delays 计算，不运行任何协方差估计算法。

### 8.2 SSB-PDP-derived CDD covariance

这是最接近当前设想的非透明接收机。

UE 只知道：

- selected SSB 的 PDP
- K
- 每个 CDD 分支的人工时延 delta_m

为了从一个 SSB PDP 推导 CDD 统计量，需要额外假设：

1. 每个 secondary beam 的 PDP 均近似等于 SSB PDP
2. 不同 CDD 分支之间的交叉相关可以忽略

即假设：

$$
S_m(\tau)
\approx
S_{\rm SSB}(\tau)
$$

以及：

$$
R_{g,mn}
\approx 0,\qquad m\neq n
$$

此时可构造近似 CDD PDP：

$$
S_{\rm CDD}^{\rm SSB}(\tau)
=
\frac{1}{K}
\sum_{m=1}^{K}
S_{\rm SSB}(\tau-\delta_m)
$$

再由该 PDP 计算：

$$
R_{\rm CDD}^{\rm SSB}[k,r]
=
\int
S_{\rm CDD}^{\rm SSB}(\tau)
e^{-j2\pi(f_k-f_r)\tau}
d\tau
$$

离散 path/tap 实现中直接使用求和。

该方法只需要 SSB PDP 和已知 CDD delay，最符合低额外信令的接收机设想，但存在两种 mismatch：

- secondary-beam PDP 与 SSB PDP 不一致
- secondary beams 之间实际存在相关性

因此需要重点比较：

$$
R_{\rm CDD}^{\rm SSB}
$$

和：

$$
R_{\rm CDD}^{\rm ideal}
$$

对应的 BLER 差距。

### 8.3 Per-beam-PDP-derived CDD covariance

进一步假设 UE 已知各 secondary beam 的 PDP：

$$
S_1(\tau),S_2(\tau),\ldots,S_K(\tau)
$$

但仍忽略不同 beams 的交叉相关。

则构造：

$$
S_{\rm CDD}^{\rm beam}(\tau)
=
\frac{1}{K}
\sum_{m=1}^{K}
S_m(\tau-\delta_m)
$$

由此得到：

$$
R_{\rm CDD}^{\rm beam}
$$

该方案消除了各窄波束 PDP 与 SSB PDP 不匹配造成的误差，但仍忽略 beam-domain cross-covariance。

因此三种 CDD 接收机的关系为：

$$
\text{SSB-PDP-derived}
\rightarrow
\text{Per-beam-PDP-derived}
\rightarrow
\text{Ideal covariance}
$$

它们分别用于量化：

1. SSB PDP 替代窄波束 PDP 带来的损失
2. 忽略不同窄波束相关性带来的损失
3. 完整统计先验下的理论上界

---

## 9. LMMSE 信道估计

本文中的“占用子载波”或代码中的 `active_subcarriers`，指 PDSCH allocation 覆盖的全部频域子载波，即 $12N_{\rm PRB}$ 个频率位置。它不是仅指 DMRS，也不是仅指 data RE。同一个占用子载波在不同 OFDM symbol 上可以承载 DMRS 或数据。频域协方差先定义在这些占用频率位置上，然后按实际 RE 映射抽取：

- $\mathbf R_{pp}$：DMRS RE 对应频率位置之间的协方差；
- $\mathbf R_{hp}$：待估计 data RE 与 DMRS RE 对应频率位置之间的交叉协方差。

静态单-slot主实验中，时间相关性取1；后续引入多普勒时再扩展为以 `(OFDM symbol, subcarrier)` 为坐标的二维时频协方差。

对于任意一种选定的频域协方差矩阵，先在 DMRS RE 上做 LS：

$$
\hat{\mathbf h}_{\rm LS,p}
$$

然后做频域 LMMSE：

$$
\hat{\mathbf h}
=
\mathbf R_{hp}
\left(
\mathbf R_{pp}
+
\sigma_n^2\mathbf I
\right)^{-1}
\hat{\mathbf h}_{\rm LS,p}
$$

其中：

- R_pp：DMRS RE 之间的信道协方差
- R_hp：待估计数据 RE 与 DMRS RE 之间的交叉协方差
- sigma_n^2：噪声功率

对于 Baseline 和 Beam cycling，默认在 PRG 内完成 LMMSE。

对于 Beam CDD，非透明接收机允许利用已知 CDD 结构在更宽频域范围构造协方差；具体估计窗口作为可配置参数，同时限制不能跨越实际 precoder 不连续边界。

---

## 10. Monte Carlo 与统计量

一个 drop 仅包含 1 个随机位置 UE，并计为一个独立样本。每个 SNR 点使用完全相同的 drop 集合比较所有方案。若具体实验使用受控主瓣 UE 分布，几何约束必须在生成信道前施加，并在该实验 plan 中冻结；不得依据生成后的信道或接收结果筛除 drop。

平台同时支持两种互不混用的链路模式：

1. `normalized_link`：按第 2.8 节定义的 selected-SSB 公共参考，对每个 drop 的物理信道只做一次标量归一化，四种方案共同使用。横轴为归一化后的 selected-SSB 参考 $E_s/N_0$；若实验 plan 采用受控主瓣 UE 分布，结果表示该受控区域内的分集链路性能，不表示覆盖概率。
2. `coverage_link_budget`：保留 38.901 路径损耗与阴影衰落，使用固定 BS EIRP、接收机噪声系数、热噪声和实际带宽计算噪声功率。该模式主要输出固定部署参数下的 SIB1 解码成功率、覆盖概率以及随距离/LOS状态的性能，不把结果标成与 `normalized_link` 相同定义的 BLER--SNR 曲线。

每个运行必须在输出元数据中明确记录 `link_mode`、SNR定义、归一化参考、EIRP、噪声系数和带宽中适用的字段。

每个 drop 保存：

- UE 位置
- BS panel azimuth 和 downtilt
- LOS/NLOS
- selected SSB index
- 各 SSB RSRP
- 物理 ray/cluster 的 delay、power、AOD、AOA
- SSB PDP
- 每个 secondary beam 的 PDP
- Beam cycling 每个 PRG 使用的 PDP
- Beam CDD 的 SSB-derived PDP
- Beam CDD 的 per-beam-derived PDP
- 能够重构 Beam CDD ideal frequency covariance 的路径/PDP/beam耦合参数及协方差摘要
- secondary-beam 之间的相关系数
- Beam CDD 有效空间秩
- SNR
- TBS
- TB CRC

协方差矩阵默认只在运行内按当前占用子载波、DMRS和估计窗口构造，不为每个 drop 保存完整稠密矩阵。参考 CDD_LLS 项目的实现，保存展开配置、BLER/NMSE统计、必要的 trial error flags，以及能够重建协方差的 PDP/路径参数。只有诊断指定 drop 时才选择性导出完整协方差矩阵。

### 10.1 主性能图

所有传输方案的 BLER-SNR 曲线画在同一图中。

### 10.2 Covariance mismatch penalty

Beam cycling 至少比较：

- Ideal/Per-beam PDP
- SSB PDP

Beam CDD 至少比较：

- Ideal covariance
- Per-beam-PDP-derived covariance
- SSB-PDP-derived covariance

重点给出固定 BLER 下的 SNR penalty。

### 10.3 机理统计

额外统计：

- secondary-beam PDP 与 SSB PDP 的距离
- secondary-beam 相关系数
- Beam CDD joint covariance 的有效秩
- covariance mismatch 与 BLER loss 的关系

这样可以判断 secondary-beam diversity 的收益是否能够在仅依赖 SSB 信道先验的接收机中保留下来。

---

## 参考标准

- 3GPP TR 38.901：UMa 场景、阵列模型、随机 cluster/ray 信道
- 3GPP TS 38.214：PDSCH PRG、MCS、TBS
- 3GPP TS 38.212：DL-SCH、LDPC、码块分段
- 3GPP TS 38.211：PDSCH DMRS 与资源映射
