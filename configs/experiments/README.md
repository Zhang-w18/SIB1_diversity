# 正式实验配置

每个正式实验使用与plan相同编号的配置文件：

```text
plan-001.yaml
plan-002.yaml
...
```

配置文件必须能独立确定本次仿真的系统、码本、方案、CDD循环移位、接收机、SNR和Monte Carlo停止条件。运行时仍需在输出目录保存展开后的 `resolved_config.yaml`，后续结果以展开配置为准。
