# Plan XXX：实验标题

状态：计划中

创建日期：YYYY-MM-DD

## 1. 本次实验目的

说明本次要回答的问题，以及相对上一次实验改变了什么。

## 2. 仿真配置

### 2.1 环境与场景

| 参数 | 取值 |
|---|---|
| Python | |
| TensorFlow | |
| Sionna | |
| 场景与方向 | UMa / downlink |
| 载频 | |
| 链路模式 | `normalized_link` / `coverage_link_budget` |
| UE距离和角度范围 | |
| LOS/NLOS、路径损耗、阴影衰落 | |

### 2.2 阵列与码本

| 参数 | 取值 |
|---|---|
| BS AE/TXRU | |
| AE间距 | |
| AE到TXRU映射 | |
| 面板方向和下倾 | |
| UE阵列 | |
| SSB码本 | |
| Secondary码本 | |
| 码本权重文件 | |
| Selected SSB方法 | |

### 2.3 SIB1与接收机

| 参数 | 取值 |
|---|---|
| SCS和CP | |
| BWP/PDSCH PRB | |
| PDSCH symbols | |
| DMRS | |
| MCS/TBS/RV | |
| PRG | |
| 信道估计 | |
| 接收机PDP/协方差 | |
| 估计窗口 | |
| 合并方式 | 4Rx MRC |

### 2.4 对比方案

| 方案 | 预编码方式 | 功率归一化 | 其他参数 |
|---|---|---|---|
| Baseline | | | |
| Pol-cycling | | | |
| Beam cycling | | | |
| Beam CDD | | | |

Beam CDD必须额外填写：

| 参数 | 取值 |
|---|---|
| Secondary beam数量 $K$ | |
| FFT大小 $N_{FFT}$ | |
| 采样率 | |
| 循环移位采样点 $\Delta_m$ | |
| 循环移位对应时间 | |
| 频域相位公式 | $e^{-j2\pi k\Delta_m/N_{FFT}}$ |

### 2.5 Monte Carlo

| 参数 | 取值 |
|---|---|
| SNR点 | |
| 主种子 | |
| 方案间公共随机样本 | drop、TB、信道、噪声 |
| 目标误块数 | |
| 最大drop数/SNR | |
| 置信区间 | 95% |
| 追加trial起止范围 | 无 / 填写绝对范围 |

## 3. 执行方式

- 配置文件：`configs/experiments/plan-XXX.yaml`
- 输出目录：`outputs/plan-XXX/<run_id>/`
- 配置验证命令：

```powershell
# 填写命令
```

- 正式运行命令：

```powershell
# 填写命令
```

## 4. 结果有效条件

- [ ] 相关测试和配置验证通过。
- [ ] 4个方案使用相同drop、TB、信道和噪声。
- [ ] DMRS和同PRG数据使用相同预编码。
- [ ] 每个占用子载波的预编码功率符合本plan定义。
- [ ] 无NaN、协方差求解失败或固定信道估计误差地板。
- [ ] 每个SNR点保存实际误块数和样本数，不填充伪造零值。

## 5. 预期输出

- BLER CSV和曲线；
- CE NMSE CSV和曲线；
- 展开配置、环境、随机种子和运行日志；
- 必要的码本、CDD时延和功率检查图。
