# Result XXX：实验标题

状态：已完成 / 未完成

对应计划：[plan-XXX.md](plan-XXX.md)

完成日期：YYYY-MM-DD

## 1. 实际运行信息

| 项目 | 实际值 |
|---|---|
| 代码版本/状态 | |
| Python/TensorFlow/Sionna | |
| 展开配置 | |
| 输出目录 | |
| 开始和结束时间 | |
| 完成状态 | |
| 与plan的偏差 | 无 / 具体说明 |

实际运行命令：

```powershell
# 填写命令
```

Beam CDD实际参数：

| 参数 | 实际值 |
|---|---|
| 波束数 $K$ | |
| FFT大小和采样率 | |
| 循环移位采样点 | |
| 对应循环移位时间 | |
| 功率归一化 | |

## 2. BLER结果

| 方案 | SNR/dB | 误块数 | 总块数 | BLER | 95%区间下限 | 95%区间上限 |
|---|---:|---:|---:|---:|---:|---:|
| | | | | | | |

主BLER曲线：

![BLER曲线](../outputs/plan-XXX/<run_id>/figures/bler.png)

Perfect CSI检查曲线：

![Perfect CSI BLER曲线](../outputs/plan-XXX/<run_id>/figures/bler_perfect_csi.png)

## 3. 信道估计和实现检查

![信道估计NMSE](../outputs/plan-XXX/<run_id>/figures/ce_nmse.png)

记录以下检查结果：

- 预编码范数；
- CDD循环移位和频域相位；
- 协方差Hermitian/半正定检查；
- 是否出现NaN、求解失败或固定NMSE误差地板；
- 方案间公共随机样本是否一致。

## 4. 结果是否有效

- [ ] 达到plan规定的测试和配置要求。
- [ ] 达到目标误块数，或明确记录达到最大drop数。
- [ ] 曲线和CSV使用相同原始统计。
- [ ] 所有偏离plan的情况已经记录。

有效性结论：

> 填写“有效”“部分有效”或“无效”，并说明原因。

## 5. 结论

用数据说明：

- 3个候选方案相对Baseline的变化；
- 固定BLER下的SNR差异；
- 信道估计是否影响方案收益；
- 当前结果能够支持什么结论，不能支持什么结论。

## 6. 输出和复现

- BLER数据：`outputs/plan-XXX/<run_id>/bler.csv`
- NMSE数据：`outputs/plan-XXX/<run_id>/ce_nmse.csv`
- 展开配置：`outputs/plan-XXX/<run_id>/resolved_config.yaml`
- 环境信息：`outputs/plan-XXX/<run_id>/environment.json`
- 运行日志：`outputs/plan-XXX/<run_id>/run.log`

复现命令：

```powershell
# 填写命令
```
