# WorldScape WAM / 世界模型策略

The unified Agent uses `robo.policy(profile="flash")` or `robo.policy(profile="memory")`. Both profiles retain the bundled explicit commit/discard lifecycle: only an observed action advances history; an unfinished prediction is discarded at episode termination. Consecutive chunks keep rolling history; a native motion or profile switch starts a fresh session. These calls use the same v4 decision loop as native tools and ICL.

统一 Agent 可通过上述两个 profile 调用 Flash/Memory。仅后续观测确认的动作推进历史；回合终止时丢弃未确认部分。连续 chunk 保持历史，原生动作或模型切换会重开会话，不存在另一套 Agent 入口。

Install the WAM service in a separate Python 3.11 environment with `bash install_wam.sh`. The pinned SAM 3 package requires `ftfy==6.1.1` and the WSP package requires `ftfy==6.3.1`; installing both dependency sets into one environment would conflict. The Agent and WAM still use one policy distribution and one decision loop.

Use `policy_service: null` with `WRSI_POLICY_CLIENT_CONFIG` to access the existing lease-based Flash/Memory model service from `scripts/serve_policy.py`. Alternatively, set `R2SI_POLICY_SERVICE` / `policy_service` to the v4 session service's WebSocket address; see [server setup](robodojo_server/README.md). An explicitly requested profile must match that service's profile. The retained in-process WAM API is available to deployments that independently provide a compatible combined environment; the supplied installers use service isolation.

建议在独立 Python 3.11 环境执行 `bash install_wam.sh`，避免两套模型依赖中的固定版本冲突。通过现有 lease 服务或 WebSocket 服务接入同一个 Agent；这不产生第二套策略入口。

## Model assets

```bash
bash download_checkpoints.sh /data/models/WorldScape-R2SI
export WRSI_WAM_MEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_memory_posttrain_robodojo
export WRSI_WAM_NOMEMORY_CHECKPOINT=/data/models/WorldScape-R2SI/wsp_2_flash_posttrain_robodojo
export WRSI_WAM_MEMORY_DEVICE=cuda:1
export WRSI_WAM_NOMEM_DEVICE=cuda:0
python -B preflight.py --check-runtime --check-cuda --check-robot-assets --check-checkpoints
```

Reserve at least 60 GB for the weights. Device indices are relative to the server's `CUDA_VISIBLE_DEVICES`. Checkpoint file hashes are recorded with the Harness release.

## Data processing

RoboDojo training reads official HDF5 trajectories through the XPolicyLab image helpers. Preparation produces normalization statistics and a temporal index:

```bash
DATA_ROOT=/data/RoboDojo ROBODOJO_PREPARED_DIR=/data/rsi-prepared \
  ROBODOJO_MODEL=flash bash process_data.sh RoboDojo baseline arx_x5 joint
```

Use `ROBODOJO_MODEL=memory` for the Memory pipeline.

## Training

WSP training is a reserved interface (stub). `train.sh` exits with code 2; Python training hooks raise `WSPTrainingUnavailable` before launching work. Bundled model/trainer source and data utilities remain for future integration. WAM inference is available.

训练接口目前仅占位，调用时明确报错，不启动训练、不生成权重；模型源码和数据准备功能保留，WAM 推理可用。
