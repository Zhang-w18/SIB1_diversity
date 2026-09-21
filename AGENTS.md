# 项目工作说明

## 1. 适用范围

本文件适用于整个 SIB1_diversity 项目。项目目标是建立可复现的 SIB1 传输分集链路级仿真平台，并比较 Baseline、Pol-cycling、Beam cycling 和 Beam CDD。

物理模型、公式和主系统假设以 `SIB1_transmission_diversity_simulation_plan_v4.md` 为准；开发状态以 `SIB1仿真平台开发计划与进度.md` 为准。

本项目采用轻量文档流程，不创建 GOALS 文档，也不照搬其他项目的全部研究管理文件。

## 2. 代码和仿真基本要求

- 新功能放入 `src/sib1div/` 对应模块，不把完整算法复制到多个实验脚本。
- 正式配置使用 YAML，运行时保存展开后的配置、环境版本、随机种子和输出路径。
- 同一次对比中的所有方案必须使用相同的 UE drop、传输块、信道和噪声样本。
- 不得把旧 `py3GPP/` 目录中的 Sionna 1.2.0 放入 Python 搜索路径；第一版使用系统安装的 Sionna 1.0.2。
- 新行为需要测试。正式仿真前至少通过相关单元测试、配置验证和小规模运行。
- 保留用户已有文件和输出，不使用破坏性 Git 或文件清理命令。

## 3. 每次方案仿真的文档

除单元测试和仅用于检查代码能否运行的 smoke 外，每次用于比较方案、生成曲线或形成结论的正式仿真，都必须建立一对同编号文档：

```text
research/plan-XXX.md
research/result-XXX.md
```

编号使用三位十进制数，从 `001` 开始递增。编号以 `research/README.md` 和实际文件为准，不重复使用。

推荐同时建立：

```text
configs/experiments/plan-XXX.yaml
outputs/plan-XXX/<run_id>/
```

### 3.1 plan-XXX.md

plan 必须在正式仿真开始前创建，并写清楚本次实验的完整定义，不能只写“沿用默认配置”。至少包括：

- 本次要回答的问题；
- 与前一次实验相比改变了什么；
- Sionna、TensorFlow和Python版本；
- UMa场景、载频、UE位置范围、LOS/NLOS、路径损耗和链路模式；
- BS/UE阵列、AE到TXRU映射、面板方向和下倾角；
- SSB及secondary beam码本布局、码本文件和selected SSB方法；
- SIB1的PRB、SCS、符号分配、DMRS、MCS、TBS、PRG和接收天线配置；
- 本次比较的方案及每个方案的预编码方式；
- Beam CDD的波束数、循环移位采样点、FFT大小、采样率、对应时间和相位公式；
- 逐子载波或平均功率归一化方式；
- LS/LMMSE使用的PDP、协方差、估计窗口和MRC方式；
- SNR点、目标误块数、最大drop数、停止条件和置信区间；
- 主种子以及方案间共用随机样本的方法；
- 运行命令、预期输出目录和判定结果是否可用的条件。

正式运行开始后不直接改写已经执行过的关键配置。如果系统、方案、CDD时延、接收机或统计口径发生变化，应新建下一个 plan。只增加相同配置的trial时，可以继续使用原 plan，但必须使用不重叠的绝对drop/trial区间并记录追加范围。

### 3.2 result-XXX.md

result 使用与 plan 相同的编号，在仿真结束后创建。至少包括：

- 对应 plan 的链接；
- 实际使用的代码版本、环境版本和展开配置路径；
- 实际运行命令、输出目录、开始/结束时间和完成状态；
- 每个方案、每个SNR点的误块数、总块数、BLER和95%置信区间；
- 主BLER曲线，并直接在Markdown中引用图片；
- 信道估计NMSE曲线和必要的码本/功率检查图；
- CDD实际使用的循环移位及预编码归一化检查；
- 是否满足plan预先规定的有效性条件；
- 结果结论、异常现象和仍未确定的问题；
- 复现实验所需的命令和文件路径。

result 不只写主观结论。所有结论必须能追溯到CSV、JSON、日志或曲线。大规模原始数组保存在 `outputs/`，不复制进 `research/`。

## 4. plan与result模板

新实验从以下模板复制：

- `research/templates/plan-template.md`
- `research/templates/result-template.md`

创建后立即在 `research/README.md` 登记编号、标题和状态。状态只使用：`计划中`、`运行中`、`已完成`、`未完成`。

## 5. 完成检查

文档或代码修改完成后：

- 以UTF-8回读修改过的Markdown和YAML；
- 检查冲突标记、尾随空格、失效的相对路径和编号重复；
- 运行与修改范围相符的测试或配置验证；
- 汇报修改内容、测试结果、未执行内容和下一项工作。
