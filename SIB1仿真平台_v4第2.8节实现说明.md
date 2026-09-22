# SIB1 仿真平台对 v4 第 2.8 节的实现说明

## 1. 结论

更新前，平台不能严格实现第 2.8 节的两个模式。虽然信道生成代码已经在 Sionna 内部分别取得施加大尺度标量前后的 CIR，但平台只保留了完整信道，链路仿真随后又按 selected SSB 的瞬时宽带功率逐 drop 归一化。这种归一化同时消除了部分小尺度能量起伏，不等价于第 2.8 节定义的“大尺度幅度归一化”。

更新后，平台支持以下三个模式：

- `fixed_radius_ls_normalized`：保留 LOS/NLOS 状态、Rician K 因子、时延与角度扩展、cluster/ray 功率、随机相位、频率选择性、阵列响应和 Rx 分支差异，只把当前 drop 的路径损耗与阴影衰落标量功率增益 $G_d$ 替换为预先冻结的参考增益 $G_0$。
- `fixed_radius_full_channel`：保留 Sionna 生成的完整信道，包括当前 drop 的路径损耗和阴影衰落。
- `fixed_cdl_statistics`：冻结一个 3GPP CDL profile 的 delay/power/angle/XPR、ray coupling、阵列、姿态和速度方向；用独立的预计算随机流估计长期发射协方差并固定 parent SSB，BLER realization 只重新生成小尺度初始相位。

三个模式都不按某个传输方案的瞬时有效信道功率做归一化。模式 A/B 使用冻结的 $G_0$，模式 C 使用冻结 parent SSB 的长期平均接收功率 $P_{\rm ref}$ 标定公共噪声。

## 2.1 模式 C 所使用的 Sionna CDL 能力

模式 C 的实现位置为 `src/sib1div/channel/cdl.py`。平台使用 Sionna 1.0.2 的 `CDL` 参数加载和 CIR generator，但不直接反复调用公开的 `CDL.__call__()`：公开调用会在每次调用时重新进行随机 ray coupling 和随机速度方向抽样，这不满足“长期统计固定”的定义。平台在初始化时用 `statistics_seed` 完成一次 ray coupling，并冻结 velocity vector；随后每个 realization 只用由 `realization_seed` 和绝对 drop index 派生的种子调用 CIR generator，因而只改变随机初始相位。

CDL 的 AE 域 CIR 沿用与 UMa 相同的 BS AE 顺序校正和 AE-to-TXRU 投影。BS 使用 38.901 element pattern，UE 使用配置的双极化阵列，因此协方差抽样自动包含 element pattern、双极化场响应、XPR、ray structure 和 Rx array response。

## 2. Sionna 1.0.2 是否支持

支持。本平台固定使用系统安装的 Sionna 1.0.2。其 `UMa` 系统级信道的生成顺序是：

1. LSP sampler 生成 LOS/NLOS 相关的 large-scale parameters，包括阴影衰落和 K 因子；
2. ray sampler 生成 cluster/ray 时延、功率和角度；
3. CIR sampler 生成尚未施加路径损耗和阴影衰落标量的路径系数；
4. `SystemLevelChannel._step_12()` 对同一 BS--UE 链路的全部路径、天线和采样点统一乘以

   $$
   a_d=10^{-PL_d/20}\sqrt{SF_d},
   $$

   因而 $G_d=a_d^2$。

这正好满足第 2.8 节所需的分解

$$
\mathbf H_d[k]=\sqrt{G_d}\,\mathbf H_{{\rm SS},d}[k].
$$

平台在一次信道生成过程中保留 `_step_12()` 前后的 CIR，并利用两者的总功率比提取公共幅度标量 $a_d$。这里的比值不是 selected-SSB 功率，也不是对小尺度 realization 做单位能量归一化；由于 Sionna 的 `_step_12()` 本身只施加一个公共标量，该比值就是 Sionna 实际施加的路径损耗与阴影衰落幅度。

当前实现需要调用 Sionna 1.0.2 的私有分阶段 API（`_lsp_sampler`、`_ray_sampler`、`_cir_sampler` 和 `_step_12`），原因是公开的一次性调用不会返回本实现构造条件 LMMSE 协方差所需的 K 因子及缩放前 CIR。因此平台依赖已由环境锁定的 Sionna 版本；升级 Sionna 前必须重新运行 UMa、协方差和端到端测试。

## 3. 信道生成与 drop 数据

实现位置为 `src/sib1div/channel/uma.py`。

每个 `UMaDrop` 保存：

- `path_coefficients`：已施加 Sionna 大尺度幅度的 TXRU 域路径系数；
- `path_spatial_covariances`：与上述路径系数处在同一功率尺度的逐路径空间协方差；
- `large_scale_amplitude_gain`：$a_d=\sqrt{G_d}$；
- `large_scale_power_gain`：$G_d$；
- `pathloss_db`：Sionna 的 basic pathloss 诊断值；
- UE 位置、固定半径、方位、LOS/NLOS、路径时延和 cluster 功率。

UE 位置总是在调用 Sionna 生成信道前确定。对于第 2.8 节模式，配置验证强制

```yaml
scenario:
  ue_distance_min_m: 100.0
  ue_distance_max_m: 100.0
```

即水平距离固定，方位仍按配置的扇区范围和主种子抽取。

## 4. 两个链路模式的缩放

实现位置为 `src/sib1div/sim/engine.py` 的 `link_mode_scaling()` 和 `simulate_one_drop()`。

### 4.1 模式 A：`fixed_radius_ls_normalized`

平台从 Sionna 完整信道开始，对瞬时频响施加功率缩放

$$
s_A=\frac{G_0}{G_d},\qquad
\widehat{\mathbf H}_d^{(A)}[k]
=\mathbf H_d[k]\sqrt{s_A}
=\sqrt{G_0}\mathbf H_{{\rm SS},d}[k].
$$

同一个 $s_A$ 也乘到逐路径空间协方差上。这样瞬时信道和 LMMSE 先验保持相同功率尺度。小尺度宽带总能量不被强制为 1。

### 4.2 模式 B：`fixed_radius_full_channel`

平台使用

$$
s_B=1,\qquad
\mathbf H_d^{(B)}[k]=\mathbf H_d[k],
$$

逐路径空间协方差也保持 Sionna 原始的完整信道功率尺度。

### 4.3 SSB 选择

平台先在完整信道上计算所有 SSB 的宽带平均接收功率并选择最大者。模式 A 和模式 B 后续施加的都是同一 drop 内对所有波束相同的正标量，因此不会改变 selected SSB。实现还会在缩放后重新检查 selected SSB；若索引改变则立即报错。

selected-SSB 功率只作为诊断量保存为 `selected_ssb_power_before_scaling` 和 `selected_ssb_power_after_scaling`，不参与两个新模式的归一化。

## 5. 公共 SNR 与噪声

两个模式使用完全相同的横轴定义。配置必须预先给出唯一 $G_0$，并同时记录线性值、dB 值和物理定义：

```yaml
run:
  link_mode: fixed_radius_ls_normalized  # 或 fixed_radius_full_channel

link_normalization:
  reference_definition: 100 m UMa LOS、无阴影条件下的路径功率增益
  reference_large_scale_power_gain_linear: 1.0e-10  # 示例，正式值由实验 plan 冻结
  reference_large_scale_power_gain_db: -100.0
  per_scheme_renormalization: false
```

配置加载时会检查 $G_0>0$、线性值与 dB 值一致、参考定义非空、禁止按方案重新归一化，并检查固定半径。

当前每个发射 RE 的总符号能量为 $E_s=1$，因此给定标称参考 SNR $gamma_{0,{\rm dB}}$ 时，每个复接收分支的噪声方差为

$$
N_0=G_0 10^{-\gamma_{0,{\rm dB}}/10}.
$$

模式 A 与模式 B 都使用这个 $N_0$，不会按 drop 的 $G_d$、LOS/NLOS、selected SSB 或方案调整噪声。

## 6. 两个模式和四种方案如何共用随机样本

`UMaChannel.generate(drop_index)` 的 Sionna 种子只由主种子和绝对 `drop_index` 派生，不依赖 `link_mode`。TB 和单位方差噪声也只由主种子、SNR stream index 与绝对 drop index 派生。因此，用两份除 `run.link_mode` 外保持相同的冻结配置运行相同绝对 drop 区间时，两个模式会复用相同的：

- UE 方位和位置；
- LOS/NLOS、LSP、cluster/ray 与随机相位；
- 传输块；
- 单位方差噪声 realization。

在每个 drop 内，Baseline、Pol-cycling、Beam cycling 和 Beam CDD 都从同一个 `UMaDrop`、TB 和单位方差噪声构造结果。只有预编码器不同。

## 7. 运行记录和诊断

普通链路运行的 `link_results.csv` 会为每个结果记录：

- `large_scale_power_gain`，即 $G_d$；
- `channel_power_scale`，模式 A 为 $G_0/G_d$、模式 B 为 1；
- `noise_variance`；
- 缩放前后的 selected-SSB 功率；
- selected SSB、LOS/NLOS、半径、方位和解码结果。

展开配置和运行元数据保留 `run.link_mode`、完整 `link_normalization`、主种子和环境版本。历史 `normalized_link` 行为仍保留，以便复现 Plan 001/002，但它按 selected-SSB 瞬时功率归一化，不得用于声称实现 v4 第 2.8 节。

## 7.1 模式 C 的预计算、固定 SSB 与公共 SNR

`FixedCDLChannel` 启动时生成 `covariance_realizations` 个无噪声 realization，并在全部 Rx、活动子载波和配置的时间样点上估计

$$
\widehat{\mathbf R}_t=\frac{1}{DN_{\rm sc}N_t}\sum_{d,r,k,n}\mathbf h_{d,r,n}^H[k]\mathbf h_{d,r,n}[k].
$$

随后对全部候选 SSB 计算 $\overline P_b=\mathbf w_b^H\widehat{\mathbf R}_t\mathbf w_b$，固定最大者及 $P_{\rm ref}=\overline P_{b^\star}$。`simulate_one_drop()` 在模式 C 下忽略每个 realization 的瞬时最佳 SSB，始终将该固定索引传给四种预编码器，并令

$$
N_0=P_{\rm ref}10^{-\gamma_{\rm ref,dB}/10}.
$$

预计算同时估计逐 Rx、逐路径的 TXRU 空间协方差，供现有 SSB-PDP、per-beam-PDP 和 ideal covariance LMMSE 路径使用。预计算流和 BLER 流由配置验证强制使用不同的根种子。固定曲线和自适应曲线运行的 `run_metadata.json` 额外记录 selected SSB、全部 SSB 长期功率、$P_{\rm ref}$、样本数和两个根种子。

## 7.2 模式 C 的配置与时间演化

可直接复制 `configs/fixed_cdl_statistics_example.yaml`。关键字段如下：

```yaml
run:
  link_mode: fixed_cdl_statistics
scenario:
  model: CDL
fixed_cdl_statistics:
  profile: C
  delay_spread_s: 3.0e-7
  covariance_realizations: 1000
  statistics_seed: 20261001
  realization_seed: 20262001
  ue_speed_mps: 0.0
  velocity_azimuth_deg: 0.0
  velocity_elevation_deg: 0.0
  time_steps_per_slot: 1
  time_step_s: 0.0
  angle_transform:
    mean_aod_deg: 0.0
    aod_scale: 1.0
```

`time_steps_per_slot: 1` 表示 slot 内块衰落。需要 Doppler 时间演化时，将其设为 `nr.slot_symbols`（当前为 14），把 `time_step_s` 设为相邻 OFDM symbol 参考时刻的间隔，并配置非零速度。CIR generator 按固定 ray AoA 和固定 velocity vector 产生相位演化；四种方案共享同一个时变信道。现有接收机在两个 DMRS symbol 上形成共同的频域估计，因此时变配置会自然包含该接收机的时间失配，不会为每个 data symbol 注入理想 CSI（Perfect-CSI 诊断除外）。

角度变换在 ray coupling 前确定性地应用。`mean_aod_deg` 改变长期 AoD 中心，`aod_scale` 控制相对中心的扩展并对方位角执行 $[-180^\circ,180^\circ)$ wrapping；AoA/ZoD/ZoA 可用同名 `mean_*_deg` 和 `*_scale` 字段配置，天顶角裁剪到物理范围。变换结果在整个实验中不再随机平移。每个新的 mean AoD 配置都必须独立初始化协方差、重新选一次 parent SSB，并按项目规则建立新的正式实验 plan。

配置验证及有界 smoke 命令示例：

```powershell
sib1div validate-config configs/fixed_cdl_statistics_example.yaml --output outputs/cdl-config-check --skip-tensorflow-check
sib1div simulate configs/fixed_cdl_statistics_example.yaml --codebook outputs/plan-001/codebook_7ghz_8h1v_mainlobe/codebook_weights.npz --output outputs/cdl-smoke --snr 0 --max-drops 1 --estimator perfect --smoke
```

正式曲线运行前应从该示例复制到 `configs/experiments/plan-XXX.yaml`，补齐对应固定/自适应 runner 的 stopping 与 curve 配置，并先建立同编号 plan 文档；示例本身主要用于展示模式 C 字段和执行 smoke。

## 8. 验证范围

相关自动测试和 smoke 覆盖：

- Sionna UMa drop 能生成有限的 TXRU 域路径系数和半正定路径协方差；
- 提取的 $a_d$ 和 $G_d$ 为正且满足 $G_d=a_d^2$；
- 两个新模式的配置约束、固定半径和 $G_0$ 线性/dB 一致性；
- 模式 A 的信道功率缩放为 $G_0/G_d$；
- 模式 B 的信道功率缩放为 1；
- 两个模式的噪声方差都只由同一个 $G_0$ 和标称参考 SNR 决定，与 selected-SSB 瞬时功率无关。
- 模式 C 配置的 profile、种子隔离、协方差样本数、角度 scale 和时间采样约束；
- 实际 Sionna CDL-C 初始化、AE-to-TXRU 投影、长期协方差估计、固定 selected SSB、$P_{\rm ref}$ 公共噪声以及四方案单 drop Perfect-CSI 链路；
- slot 内 14 个时间样点的 Doppler 信道维度和链路消费路径。

本次是平台功能更新和 smoke/单元验证，没有启动用于生成曲线或形成物理结论的正式方案对比，因此不创建新的 `research/plan-XXX.md` 与 `research/result-XXX.md`。后续正式比较两个模式时，必须按项目规则新建一对同编号 plan/result，并在 plan 中冻结半径、$G_0$ 定义、两个配置文件及共同的绝对 drop 区间。
