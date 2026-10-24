# WorldScape R²SI

[English](README.md) · [官网](https://r2si.worldscape.io) · [模型权重](https://huggingface.co/manifoldai-research/WorldScape-R2SI)

WorldScape R²SI 将机器人控制 Agent 与自主改进循环结合起来。内层 Agent 根据视觉观测和通用 Skill，选择原生操作工具或 WorldScape Flash/Memory 策略。外层 Agent 回顾采集回合，更新可复用的 Skill 和 Memory，验证候选版本，并将选定版本用于后续回合。

## 包含的功能

- **内层 Agent：** 视觉推理、原生运动工具、Flash/Memory 策略调用和动作边界接管。
- **通用经验库：** 跨任务共享的 34 份 Skill、4 个工具和 2 个原语。
- **自主 RSI：** 采集、多模态反思、候选构建、配对验证和本地版本启用。
- **模型训练：** RoboDojo 数据准备、Flash/Memory 训练、数据快照和新权重独立服务。
- **评测：** 官方 XPolicyLab 接口，以及 Agent、协议和外层循环测试。

## 仓库结构

```text
XPolicyLab/policy/WorldScape_RSI/  内层 Agent、通用 Skill、WAM 源码与官方入口
XPolicyLab/                      随附的官方协议及运行依赖
rsi/                             外层迭代、经验、版本、任务与模型服务
scripts/                         配置、采集、仿真 worker、迭代及导出
prompts/                         外层 Agent 反思提示词
env_cfg/                         机器人与观测配置
docs/                            安装、架构与使用说明
tests/                           外层及集成测试
```

随附 XPolicyLab 共享文件的版本记录在 [UPSTREAM.json](UPSTREAM.json)。官方 PR 对应单个 `policy/WorldScape_RSI/` 目录，完整仓库同时包含外层 RSI 代码。

## 安装

按[运行环境安装指南](XPolicyLab/policy/WorldScape_RSI/INSTALLATION.md)准备环境，再从本仓库运行：

```bash
bash install.sh
```

下载 [Flash 和 Memory 权重](XPolicyLab/policy/WorldScape_RSI/README_zh.md#模型权重)，设置 `ROBODOJO_ROOT`，并在协调机和仿真机的 shell 中配置 `R2SI_API_BASE`、`R2SI_API_KEY`。

## 运行自主 RSI

使用已安装的官方 layout 和两份权重生成配置：

```bash
python -B scripts/configure_rsi.py \
  --workspace /shared/rsi/run-001 \
  --robodojo-root /path/to/RoboDojo \
  --sim-python /isaac-sim/python.sh \
  --flash-checkpoint /data/models/WorldScape-R2SI/wsp_2_flash_posttrain_robodojo \
  --memory-checkpoint /data/models/WorldScape-R2SI/wsp_2_memory_posttrain_robodojo \
  --rounds 2
```

在仿真机启动 worker：

```bash
python -B scripts/local_worker.py --workspace /shared/rsi/run-001 --devices cuda:0
```

在协调机启动外层 Agent：

```bash
python -B scripts/iterate.py --config /shared/rsi/run-001/campaign.json
```

启动后，采集、反思、候选生成和验证自动执行，无需手动输入提案。默认使用官方非 Open layout 改进通用 Skill 和 Memory；配置训练数据与预算后可启用模型训练。

共享存储、短临时路径、多 worker、远程 WAM 服务和训练配置见[自主 RSI](docs/LOCAL_RSI.md)。

## 单独运行内层 Agent

官方 Agent 入口可独立于外层循环运行：

```bash
cd XPolicyLab/policy/WorldScape_RSI
bash eval.sh RoboDojo stack_bowls public arx_x5 joint 0 0,1 0 r2si base
```

权重、环境名和分机评测配置见[策略使用指南](XPolicyLab/policy/WorldScape_RSI/README_zh.md)。

## 文档

- [架构](docs/ARCHITECTURE.md)
- [自主 RSI](docs/LOCAL_RSI.md)
- [官方 layout 选择](docs/OFFICIAL_LAYOUTS_zh.md)
- [通用 Skill](docs/SKILLS.md)
- [测试](docs/TESTING.md)
- [导出官方策略](docs/RELEASING.md)
