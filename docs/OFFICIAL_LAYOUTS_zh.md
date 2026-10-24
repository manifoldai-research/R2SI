# 官方 layout

外层 RSI 使用 RoboDojo 已保存的 layout，包含 34 个非 Open 基础任务和 12 个 Generalization 随机变体，共 46 个任务项。

外层采集和验证排除 8 个 Open 任务：`align_blocks`、`general_pickup`、`solve_equation`、`stack_blocks_by_language`、`classify_objects_by_language`、`pick_from_conveyor_by_image`、`store_tools_in_toolbox`、`pour_by_language`。内层 Agent 仍可通过官方评测接口调用。

## 准备 layout

```bash
python -B scripts/prepare_official_layout_cases.py \
  --robodojo-root /path/to/RoboDojo \
  --output /shared/rsi/official-cases.json
```

默认设置：
- 采集：seed 0，layout 序号 0。
- 验证：seed 0、1、2，layout 序号 1。
- 使用全部 46 个非 Open 任务项，也可通过 `--tasks` 选择子集。
- 每个案例记录官方文件名、SeedManager 序号和文件 SHA256。

`scripts/configure_rsi.py` 会直接将此计划加入可运行的 campaign 配置。

## 采集与验证

采集与验证使用不同文件身份。部分官方文件包含相同场景几何，`layout_split` 会记录内容哈希的重合情况。即使验证文件与采集文件具有相同内容，反思也只读取采集记录。

基线和候选使用相同验证文件，以得分、成功情况、动作数及能力/seed 覆盖进行比较。公开 layout 的结果按公开 layout 结果报告。

layout 记录采用 `official_layout_iteration` 并保留原文件名，供反思与验证使用；模型训练读取配置的训练数据集。

任务类别来自[官方 RoboDojo 目录](https://robodojo-benchmark.com/doc/sim-tasks/)。

