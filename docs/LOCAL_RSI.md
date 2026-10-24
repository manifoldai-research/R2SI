# Autonomous RSI / 自主 RSI

The coordinator and simulator worker share one writable POSIX workspace. Start the worker and the coordinator once; the outer Agent then performs collection, reflection, candidate generation, paired validation and version activation automatically.

协调机和仿真 worker 共用一个可写 POSIX 工作目录。各启动一次后，采集、反思、候选生成、配对验证和版本启用由外层 Agent 自动执行。

## Configure / 配置

Use the commands in the root README. `configure_rsi.py` writes `campaign.json` and `checkpoints.json` into the chosen workspace. It supports `--tasks`, `--rounds`, `--policy-client-config` and `--pool-bootstrap`. The configuration is read once at startup.

根目录 README 提供完整启动命令。配置脚本生成 `campaign.json` 与 `checkpoints.json`，可指定任务子集、轮数、远程 Policy 客户端配置和号池初始化模块。启动时读取配置，迭代提案由模型自行生成。

| Field | Purpose / 用途 |
|---|---|
| `workspace` | Shared campaign output / 共享输出目录 |
| `source_project` | Initial source tree / 初始源码目录 |
| `checkpoints` | Flash/Memory locations and file hashes / 权重位置与哈希 |
| `rounds` | Number of autonomous rounds / 自主轮数 |
| `collection_cases`, `validation_cases` | Official saved layouts / 官方 layout |
| `required_capabilities`, `required_seeds` | Validation coverage / 验证覆盖 |
| `episodes` | Simulator interpreter, assets, paths and timeouts / 仿真环境与超时 |
| `policy_client_config` | Optional remote WAM service / 可选远程 WAM 服务 |
| `pool_bootstrap` | Optional provider initialization module / 可选号池初始化模块 |

## Hosts and storage / 主机与存储

Both hosts need the same repository and workspace paths. Set the provider environment on both. On the simulator host, `episodes.python` selects the Isaac Sim interpreter; the worker itself can run in the coordinator Python environment.

两台主机应使用相同仓库和工作目录路径，并配置模型 API 环境变量。`episodes.python` 指定 Isaac Sim 解释器，worker 本身可使用协调机 Python 环境。

Use a short temporary root on the same shared filesystem when campaign paths are long. Export on both hosts before launch:

```bash
export R2SI_STORAGE_ROOT=/shared
export R2SI_TEMP_ROOT=/shared/rsi-tmp
export PYTHONDONTWRITEBYTECODE=1
```

The workspace manages logs, caches and temporary files. Storage must support atomic directory rename, fsync and SQLite writer locks.

工作目录管理日志、缓存和临时文件；存储需支持目录原子重命名、fsync 和 SQLite 写锁。

## Workers and WAM placement / Worker 与 WAM 部署

`--devices cuda:0 cuda:1` gives the worker two episode slots. Each child sees its assigned GPU as `cuda:0`. Size the number of slots for the simulator and model memory requirements. For smaller simulator GPUs, use `--policy-client-config /shared/private/policy-client.json` to place WAM inference on a separate model service. The inner native planner still runs with the simulator-side Agent.

`--devices cuda:0 cuda:1` 提供两个回合槽位；每个子进程将分配到的 GPU 视为 `cuda:0`。按仿真和模型显存需求设置槽位。仿真卡显存较小时，可通过远程客户端配置将 WAM 放到模型服务，内层原生规划器仍与仿真侧 Agent 一起运行。

Remote client and server use the same lease protocol. `scripts/serve_policy.py` starts the bundled model-service launcher; `scripts/serve_generation.py` starts a candidate weight generation. Check their `--help` and deployment configuration for device and capacity settings.

## Training / 训练

The default configuration updates generic Skills and Memory. To enable model training, add:
- `training`: dataset index, provenance file, `flash`/`memory` recipes, GPUs, interpreter, step budget and timeout.
- `generation_service`: address, GPUs, TLS files and startup timeout for candidate weights.
- Optional `training_collection_cases`: additional collection cases with the same layout selection rules.

The reflection Agent can request training within the configured budget. Training reads its dataset snapshot; a separate service loads the new checkpoint for candidate validation. Active episodes retain their original version.

默认配置更新通用 Skill 和 Memory。配置训练数据、预算及新权重服务后，反思 Agent 可以在预算内请求训练；候选验证通过独立服务使用新权重，正在运行的回合继续使用原版本。

See [native training](../XPolicyLab/policy/WorldScape_RSI/worldscape-policy/docs/native_training.md) and [data hot loading](../XPolicyLab/policy/WorldScape_RSI/worldscape-policy/docs/robodojo_hotload.md).

## Outputs and control / 输出与控制

```text
experience/committed/     Completed collection and validation records / 已完成记录
episode-jobs/            Episode queue and receipts / 回合队列与回执
proposals/               Outer Agent proposals / 外层提案
releases/                Versioned inner code / 内层版本
datasets/snapshots/       Training snapshots / 训练快照
generations/             Model service generations / 权重服务版本
validation/              Paired comparisons / 配对比较
iterations/              Iteration results / 迭代结果
local-loop-status.json   Loop status / 循环状态
```

```bash
python -B -m rsi --workspace /shared/rsi/run-001 status
touch /shared/rsi/run-001/STOP
```

`STOP` stops the coordinator before the next round. Current work finishes its own execution. The worker is a separate process; stop it after queued work finishes. To resume, remove `STOP` and launch the coordinator again.

