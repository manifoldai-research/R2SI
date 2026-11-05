# WorldScape R²SI

[中文](README_zh.md) · [Website](https://r2si.worldscape.io) · [Model weights](https://huggingface.co/manifoldai-research/WorldScape-R2SI)

WorldScape R²SI combines a robot-control Agent with an autonomous improvement loop. The inner Agent uses visual observations and generic Skills to choose native manipulation tools or WorldScape Flash/Memory policies. The outer Agent reviews collected episodes, updates reusable Skills and Memory, evaluates candidates, and activates the selected version for subsequent episodes.

The unified adapter is `XPolicyLab/policy/WorldScape_R2SI/`. Its default entry is the v4 code-writing Agent with ICL-RAG. Shared Skills, Memory, Flash/Memory WAM and the outer RSI loop are connected to this single Agent through the frozen Harness boundary.

## What is included

- **Inner Agent:** visual reasoning, native motion tools, Flash/Memory policy calls and action-level takeover.
- **Generic experience library:** 34 Skills, four tools and two primitives, shared across tasks.
- **Autonomous RSI:** collection, multimodal reflection, candidate construction, paired validation and local version activation.
- **WSP training interface:** reserved stub; data preparation and WAM inference remain available, but training is not connected.
- **Evaluation:** official XPolicyLab entrypoints and tests for the Agent, protocol and outer loop.

## Repository layout

```text
XPolicyLab/policy/WorldScape_R2SI/  Inner Agent, generic Skills, WAM source and official entrypoints
XPolicyLab/                      Bundled upstream protocol and runtime dependencies
rsi/                             Outer iteration, experience, versions, jobs and model services
scripts/                         Configuration, collection, workers, iteration and export
prompts/                         Outer Agent reflection prompt
env_cfg/                         Robot and observation configuration
docs/                            Installation, architecture and usage
tests/                           Outer-loop and integration tests
```

The bundled XPolicyLab shared files are pinned in [UPSTREAM.json](UPSTREAM.json). The official PR consists of the single `policy/WorldScape_R2SI/` directory. The full repository also includes the outer RSI code.

## Installation

Follow the [runtime installation guide](XPolicyLab/policy/WorldScape_R2SI/INSTALLATION.md), then run from this repository:

```bash
bash install.sh
```

Download the [Flash and Memory checkpoints](XPolicyLab/policy/WorldScape_R2SI/WAM.md#model-assets), set `ROBODOJO_ROOT`, and configure `R2SI_API_BASE` and `R2SI_API_KEY` in the shell on the coordinator and simulator hosts.

## Run autonomous RSI

Create the campaign configuration from installed official layouts and the two checkpoints:

```bash
python -B scripts/configure_rsi.py \
  --workspace /shared/rsi/run-001 \
  --robodojo-root /path/to/RoboDojo \
  --sim-python /isaac-sim/python.sh \
  --flash-checkpoint /data/models/WorldScape-R2SI/wsp_2_flash_posttrain_robodojo \
  --memory-checkpoint /data/models/WorldScape-R2SI/wsp_2_memory_posttrain_robodojo \
  --rounds 2
```

On the simulator host, start a worker:

```bash
python -B scripts/local_worker.py --workspace /shared/rsi/run-001 --devices cuda:0
```

On the coordinator host, start the outer Agent:

```bash
python -B scripts/iterate.py --config /shared/rsi/run-001/campaign.json
```

After startup, collection, reflection, candidate generation and validation run automatically. There is no manual proposal entry step. The default campaign improves generic Skills and Memory using official non-Open layouts. WSP training is currently a reserved interface and cannot be enabled by configuration.

For shared storage, short temporary paths, multiple workers, remote WAM services and training, see [Autonomous RSI](docs/LOCAL_RSI.md).

## Run the inner Agent alone

The official Agent entrypoint runs independently of the outer loop:

```bash
cd XPolicyLab/policy/WorldScape_R2SI
bash eval.sh RoboDojo stack_bowls public arx_x5 joint 0 0,1 0 r2si base
```

See the [policy guide](XPolicyLab/policy/WorldScape_R2SI/README.md) for checkpoints, environment names and split-machine evaluation.

Both standalone evaluation and the outer RSI loop run this unified Agent and require the SAM 3 setup from the policy guide.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Autonomous RSI](docs/LOCAL_RSI.md)
- [Official layout selection](docs/OFFICIAL_LAYOUTS.md)
- [Generic Skills](docs/SKILLS.md)
- [Testing](docs/TESTING.md)
- [Export the official policy](docs/RELEASING.md)
