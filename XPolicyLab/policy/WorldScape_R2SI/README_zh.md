# WorldScape_R2SI

[English](README.md) · [WAM 配置](WAM.md)

原来的 `R2SI/` 和 `WorldScape_RSI/` 已自然整合为一个策略。统一 `model.py` 通过 Harness 的回合隔离和版本校验运行 v4 代码编写 Agent，标准部署与外层 RSI 共用同一条执行链，没有 legacy 入口或模式开关。Agent 通过 `robo.skills()` / `robo.read_skill()` 读取通用 Skill，提示词加载当前候选版本的 Memory，通过 `robo.policy(profile="flash" 或 "memory")` 调用保留的 WAM。外层代码更新直接修改实际使用的 Agent 与工具文件。

## 功能与目录

- `agent.py`、`robo.py`：v4 根据三路 RGB、机器人状态和指令编写 Python，执行感知、规划和操作。
- `icl.py`、`icl_worker.py`：核心 ICL-RAG 工具。配置后在首次决策前检索演示，将固定参考保留在后续请求中；`robo.reference()` 可读取摘要。配置与数据要求见 [ICL.md](ICL.md)。
- `planning.py`、`harness/vendor/runtime_support/curobo/`：共享本目录内的 cuRobo 原生规划器。
- `wam.py`、`wam_model.py`、`robodojo_server/`、`worldscape-policy/`：保留 WAM Flash/Memory 推理及模型服务。
- `harness/`：回合边界、通用 Skill、WAM 分派及完整性校验。
- `training.py`、`train.sh`：预留 WSP 训练接口，当前不执行训练。

已有 `R2SI_*` 与 `WRSI_*` 环境变量保持名称不变。导出时携带整个目录，无需再携带两个旧目录。

## 安装

使用 Linux、Python 3.11 和匹配 CUDA 的 torch / torchvision，准备 RoboDojo 公开机器人资源，然后从仓库根目录运行：

```bash
bash XPolicyLab/policy/WorldScape_R2SI/install.sh
export ROBODOJO_ROOT=/path/to/RoboDojo
export R2SI_SEGMENTER_CHECKPOINT=/data/models/sam3/sam3.pt
export R2SI_API_BASE=https://your-endpoint/v1
export R2SI_API_KEY=...
```

SAM 3 权重需按 `facebook/sam3` 的访问要求下载。可选 WAM 服务的配置见 [robodojo_server/README.md](robodojo_server/README.md)，通过 `R2SI_POLICY_SERVICE` 指定地址。ICL 引擎为独立依赖，需设置 `R2SI_ICL_CONFIG`；未配置时明确报告禁用。完整参数表见 [English configuration](README.md#configuration)。

## 数据与训练

保留的 `process_data.sh` 通过 XPolicyLab 共享图像工具准备 WAM 数据，见[WAM 数据说明](WAM.md#data-processing)。

本版本仅支持推理。WSP 训练是**预留接口（stub）**：Python 调用抛出 `WSPTrainingUnavailable`（继承 `NotImplementedError`），`train.sh` 收到训练请求后以退出码 2 结束。`bash train.sh --help` 可查看保留的参数格式。不会启动训练进程、创建权重或返回虚假成功指标。外层 RSI 即使读取到旧训练配置，也会报告训练不可用。

底层模型与训练源码保留，便于后续接入；训练后端暂未承诺发布日期。WAM 推理和权重下载仍然可用。

## 评测

```bash
bash XPolicyLab/policy/WorldScape_R2SI/eval.sh RoboDojo stack_bowls default arx_x5 joint 0 0 1 r2si base
```

支持 `RoboDojo`、`arx_x5`、`joint`，每个进程一个环境，`eval_batch=false`。不读取环境步数上限；本地只统计后续观测确认的动作，终止由评测器决定。

策略端脚本启动同一份 `server.py` / `deploy.yml`，无需额外选择模式。WAM 服务地址为空时，使用本目录内的 Flash/Memory 实现和 `WRSI_*` 权重配置，或已有远程 lease 服务配置；模型在实际调用时加载。缺少权重会明确报错。配置见 [WAM.md](WAM.md)。

## 测试

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
export R2SI_TEST_XPL_ROOT="$PWD/XPolicyLab"
python -B -m pytest -p no:cacheprovider -q tests XPolicyLab/policy/WorldScape_R2SI/harness/tests XPolicyLab/policy/WorldScape_R2SI/tests
```

CPU 测试覆盖接口、控制流程及协议，不代表仿真成功率或 GPU 模型加载已验证。正式评测前应在完整环境运行 [XPolicyLab debug 与仿真检查](../../CONTRIBUTING.md#testing)。
