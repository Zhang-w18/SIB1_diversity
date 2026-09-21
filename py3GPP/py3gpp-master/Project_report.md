# Project Report: SIB1 PDSCH 细波束方案 BLER 对比仿真

## 1. 项目概述

本项目是一个 3GPP 5G NR 链路级仿真系统，位于 `py3gpp-master/` 目录下，基于开源 `py3gpp` 库实现 5G NR 物理层信号处理（PSS/SSS/PBCH/PDSCH/PRACH 等）。

核心关注脚本：

```
py3gpp-master/sib1_2v4h_coarse_2v8h_fine_comb_re_dmrs_with_prg_random_angle.py
```

该脚本在统一的链路级框架下，对比两类 SIB1 PDSCH 细波束实现方案在不同 SNR 下的 BLER（块错误率）性能：

1. **阵列细波束（array-based）**：粗波束使用缩减水平孔径（Mh_active=4），细波束使用全水平孔径（Mh_active=8），可选 FDM 细化。
2. **预编码细波束（precoding-based）**：在双极化分支上使用同一空间 DFT 波束或相邻空间波束，通过 PRG 级别的循环基带预编码相位产生波束状态变化。

---

## 2. 目的

验证不同的 SIB1 传输方案在 CDL-C 信道下的链路性能，核心对比维度包括：

- **粗波束基线** vs **细波束/预编码方案**
- **传统 DMRS** vs **RE 级 comb DMRS**（01020102... 梳齿模式）
- **不同 PRG bundle 尺寸**（24RB / 8RB）对信道估计的影响
- **不同预编码模式**（traditional [1,1] / cyclic [1,1]/[1,j] 双极化状态）
- **随机 AOD 旋转**：在每个 trial 中对 CDL 信道水平到达角随机偏转 ±15°，使一个 BLER 点代表所选 SSB 扇区内随机 UE 位置，而非固定视线方向。角度仅由 trial 种子决定，与方案无关，保证各方案在相同 SNR/trial 下经历相同角度。

---

## 3. 依赖

### 3.1 Python 环境

- **Python**: 3.14.3（实测可运行）
- **numpy**: >= 1.21.0（实测 2.4.4）
- **scipy**: >= 1.9.3（实测 1.17.1，用于 `scipy.signal.fftconvolve` 做 MIMO FIR 卷积）
- **tqdm**: 进度条显示
- **matplotlib**: BLER 曲线绘图（后端 `Agg`，无 GUI 依赖）

### 3.2 本地模块（均位于 `py3gpp-master/` 目录下）

| 模块 | 用途 |
|------|------|
| `py3gpp` | 5G NR 信号处理核心库：nrPRBS, nrCRCEncode, nrLDPCEncode, nrRateMatchLDPC, nrOFDMModulate/Demodulate, nrSymbolModulate, nrDLSCHInfo 等 |
| `channel_initialize` | CDL 信道初始化（天线配置、角度扩展、多径时延等）|
| `channel_functions` | 信道响应生成、信道时间插值 |
| `myChannelEstimation4` | 信道估计 `myChannelEstimate_std`（PDSCH 全带 CE）|
| `GenTDLChannel` | 标准 TDL 信道（`ChannelInfo`），仅 TDL 模式使用，CDL 模式下可选 |

### 3.3 安装方式

依赖已在上一轮迭代中安装完成，当前环境可直接运行。`py3gpp` 以本地包形式导入（`py3gpp-master/py3gpp/` 子目录），无需额外安装。

---

## 4. 仿真方案（8 个方案）

脚本在 `__main__` 中配置了 8 个 `SchemeConfig` 进行对比：

| # | 标签 | 家族 | scheme | PRG(RB) | 预编码模式 | DMRS 模式 |
|---|------|------|--------|---------|-----------|-----------|
| 1 | Baseline: coarse 4H, 24RB CE | array_based | array_coarse_baseline | 24 | traditional | legacy |
| 2 | Baseline: coarse 4H, 8RB CE | array_based | array_coarse_baseline | 8 | traditional | legacy |
| 3 | Comb-DMRS-array: RE-level qh_a/qh_b | array_based | array_comb_re_dmrs | 48 | traditional | comb2_re |
| 4 | Comb-DMRS-precoding: dual-pol [1,1]/[1,j] | precoding_based | precoding_comb_re_dmrs | 48 | cyclic | comb2_re |
| 5 | PRG-array: 8H fine pair, PRG=24RB | array_based | array_fdm_fine | 24 | traditional | legacy |
| 6 | PRG-array: 8H fine pair, PRG=8RB | array_based | array_fdm_fine | 8 | traditional | legacy |
| 7 | PRG-precoding: centered, dual-pol, PRG=24RB | precoding_based | precoding_cyclic | 24 | cyclic | legacy |
| 8 | PRG-precoding: centered, dual-pol, PRG=8RB | precoding_based | precoding_cyclic | 8 | cyclic | legacy |

### 方案要点

- **array_coarse_baseline**：粗波束（Mh_active=4，~28.65° HPBW），中心对齐到 qh_center，传统 [1,1] 双极化预编码。
- **array_comb_re_dmrs**：RE 级 comb DMRS，state-1 用 qh_a 细波束、state-2 用 qh_b 细波束（全孔径 Mh_active=8），两状态 DMRS 在频域梳齿交错，总 DMRS 开销不变。
- **precoding_comb_re_dmrs**：RE 级 comb DMRS，两状态共用 qh_center 空间波束，通过双极化基带预编码 [1,1] vs [1,j] 区分状态。
- **array_fdm_fine**：PRG 级 FDM 细波束，两个全孔径细波束 qh_a/qh_b 按 PRG 交替。
- **precoding_cyclic**：PRG 级双极化循环预编码，同一中心空间波束，phase_table=(0, π/2) 产生 [1,1]/[1,j] 双极化状态。

---

## 5. 核心配置参数

### 5.1 天线与波束网格

| 参数 | 值 | 说明 |
|------|----|------|
| TX_VERTICAL_SIZE | 6 | 垂直天线阵元数 |
| HORIZONTAL_ARRAY_SIZE | 8 | 水平天线阵元数 |
| NUM_POLARIZATIONS | 2 | 极化数 |
| TX_PORTS | 96 | 发射端口总数 (6×8×2) |
| RX 天线 | [1,1,1,2,2] = 4R | 接收端口 |
| N_COARSE_V × N_COARSE_H | 2 × 4 = 8 | 粗 SSB 波束数 |
| N_FINE_V × N_FINE_H | 2 × 8 = 16 | 细波束数 |
| COARSE_MH_ACTIVE | 4 | 粗波束水平有效阵元 |
| FINE_MH_ACTIVE | 8 | 细波束水平有效阵元 |
| COARSE_HPBW_DEG | 28.65° | 粗波束半功率波束宽度 |
| FINE_PAIR_MODE | symmetric_halfstep | 细波束对关联模式 |
| DEFAULT_RANDOM_ANGLE_GAP_MAX_DEG | 15° | CDL AOD 随机偏转最大角度 |

### 5.2 SIB1 PDSCH 参数

| 参数 | 值 | 说明 |
|------|----|------|
| nrb_sib | 48 | SIB1 RB 数 |
| nsym_sib | 12 | 时隙内符号数 |
| dmrs_loc | (0, 6, 9) | DMRS 符号位置 |
| k_step | 2 | DMRS 频域间隔 |
| payload | 1200 bits | 传输块大小 |
| 调制 | QPSK |  |
| 编码 | LDPC, BGN=2, CRC16 |  |
| ncellid | 208 | 小区 ID |
| rnti | 0xFFFF | SIB1 RNTI |

### 5.3 信道参数

| 参数 | 值 | 说明 |
|------|----|------|
| channel_model | CDL | 信道模型类型 |
| cdl_type | CDLC | CDL 场景 |
| delay_spread | 300 ns | 时延扩展 |
| fc | 7 GHz | 载波频率 |
| ue_speed | 3 km/h | UE 移动速度 |
| fs | 61.44 MHz | 采样率 |
| scs | 30 kHz | 子载波间隔 |
| angle_gap | ±15° (随机) | CDL AOD 水平旋转 |

### 5.4 BLER 扫描参数

| 参数 | 值 | 说明 |
|------|----|------|
| snr_points (CDL) | [-17, -15, -13, -11, -9, -7, -5, -3, -1] dB | SNR 扫描点 |
| n_trials_per_target | 200 | 每个目标映射的 trial 数 |
| target_errors | 200 | 提前停止错误阈值 |
| snr_early_stop_th | 0.03 | BLER ≤ 3% 则剩余 SNR 点填 0 |
| max_workers | 4 | 并行进程数 |
| seed | 2029 | 随机种子 |
| N_comb | 1 | 合并接收次数 |
| RUN_ALL_8_SSB_SECTORS | False | 仅扫描 CENTER_SSB_INDEX=3 |
| CENTER_SSB_INDEX | 3 | 中心 SSB 扇区索引 |

---

## 6. 输入

脚本无命令行参数（`__main__` 块硬编码配置），所有参数在脚本顶部和 `__main__` 中定义：

- **波束网格配置**：模块级常量（TX_VERTICAL_SIZE, HORIZONTAL_ARRAY_SIZE, N_COARSE_*, N_FINE_* 等）
- **方案配置**：`scheme_configs` 列表中的 8 个 `SchemeConfig`
- **目标映射**：`make_target_mappings_8ssb_16fine()` 生成，默认仅 SSB03（qh_center=6, fine qh=5.5/6.5）
- **SNR 点序列**、**trial 数**、**种子**等

---

## 7. 输出

### 7.1 CSV 文件

```
py3gpp-master/sib1_2v4h_2v8h_comb_re_dmrs_with_prg_random_angle_bler.csv
```

列定义：

| 列名 | 说明 |
|------|------|
| Config | 方案标签 |
| Family | array_based / precoding_based |
| Scheme | scheme 标识符 |
| PRG_Size_RB | PRG bundle 尺寸 |
| Precoding_Mode | traditional / cyclic |
| CE_Mode | fullband |
| Precoding_Basis | polarization / spatial_adjacent |
| DMRS_Pattern | legacy / comb2_re |
| N_Coarse_SSB | 8 |
| N_Fine_Beams | 16 |
| Channel_Model | CDL |
| CDL_Type | CDLC |
| Random_AOD_Rotation | True/False |
| Angle_Gap_Max_Deg | 15 |
| SNR(dB) | SNR 点 |
| BLER | 平均块错误率 |

### 7.2 PNG 图

```
py3gpp-master/sib1_2v4h_2v8h_comb_re_dmrs_with_prg_random_angle_bler.png
```

半对数纵坐标的 BLER vs SNR 曲线，8 条方案曲线。

### 7.3 已有历史输出

目录中还存在多个相关输出文件（不同方案变体/历史迭代）：

- `sib1_2v4h_2v8h_comb_re_dmrs_with_prg_bler.csv/png` — 无随机角度版本
- `sib1_2v4h_2v8h_comb_re_dmrs_with_prg_random_angle_bler_sum.csv/png` — 汇总版本
- `sib1_2v4h_2v8h_comb_re_dmrs_with_prg_random_angle_bler - 副本.csv/png` — 备份
- `prg_precoding_bler*.csv/png` — PRG 预编码专项
- `sib1_nr_baseline_common16h_*.csv/png` — NR 基线对比

---

## 8. 运行方式

```bash
cd py3gpp-master
python sib1_2v4h_coarse_2v8h_fine_comb_re_dmrs_with_prg_random_angle.py
```

完整运行耗时较长（8 方案 × 9 SNR 点 × 200 trials × ~2.4s/trial，含提前停止）。单 trial 约 2.3–2.6 秒。

### 验证结果

本次通过导入模块方式对 3 类代表性方案各执行 1 个 trial 的冒烟测试（SNR=-5dB, SSB03, CDLC, 随机 AOD ±15°），全部通过：

| 方案 | scheme | tb_ok | 耗时 |
|------|--------|-------|------|
| Baseline coarse 4H, 24RB CE, legacy | array_coarse_baseline | True | 2.58s |
| Comb-DMRS-array, RE-level | array_comb_re_dmrs | True | 2.32s |
| PRG-precoding, dual-pol cyclic | precoding_cyclic | True | 2.43s |

环境信息：Python 3.14.3, numpy 2.4.4, scipy 1.17.1, tqdm, matplotlib 3.10.8, sionna 1.2.0，本地模块 py3gpp/channel_initialize/channel_functions/myChannelEstimation4/GenTDLChannel 均导入正常。

---

## 9. 关键实现细节

### 9.1 CDL 信道生成

`cdl_impulse_response_96t4r_full()` 调用 `channel_initialize()` 生成 CDLC 信道：
- 发射天线 `[1, 1, 6, 8, 2]` → 96 端口
- 接收天线 `[1, 1, 1, 2, 2]` → 4 端口
- `angle_gap` 参数控制水平 AOD 旋转，由 `deterministic_angle_gap_from_trial_seed()` 从 trial 种子确定性生成，保证各方案公平对比

### 9.2 RE 级 comb DMRS

`nrSIB1DMRSIndices_comb2_re_param()` 将传统 k_step=2 的 DMRS 并集拆分为两个状态：
- state-1: k mod 4 = 0 → DMRS，k mod 4 = 1 → data
- state-2: k mod 4 = 2 → DMRS，k mod 4 = 3 → data

总 DMRS 开销不变，每个状态仅看到 1/4 密度 DMRS。接收端用 `estimate_comb2_re_dmrs_reuse_existing_ce()` 分别估计 H1/H2 并按 RE 状态合并。

### 9.3 波束/预编码生成

- `build_wbf_tx_x2_subarray()`：阵列子孔径 DFT 波束（Mh_active 控制孔径宽度）
- `build_wbf_tx_x2_upa()`：全孔径 UPA DFT 波束（支持分数 qh）
- `bb_precoder_cyclic()`：PRG 级循环预编码相位 (0, π/2, π, 3π/2)
- `build_precoded_tx_grids_unified()`：PRG 级波束/预编码应用
- `build_precoded_tx_grids_comb2_re_dmrs()`：RE 级波束/预编码应用

### 9.4 链路处理流程

```
SIB1 bits → CRC → LDPC → 速率匹配 → QPSK 调制 → 资源映射
  → 波束/预编码 (Wbf × wbb) → 多 TX OFDM 调制
  → CDLC MIMO 信道 (96T×4R, AOD 随机旋转) → AWGN (per-symbol SNR)
  → OFDM 解调 → 信道估计 (fullband/prg/comb2_re)
  → ZF 均衡 → QPSK 软解调 → 速率恢复 → LDPC 译码 → CRC 校验
  → BLER 统计
```
