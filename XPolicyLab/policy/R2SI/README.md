# R2SI

**Contributor:** jwt / Sa63R | **Source:** R2SI-research, `improve/v4` at `1229cd1d0c146c5a61337bf2f1d36e15a2a32391`

An agent that controls a two-arm robot by writing Python. It sees the three RGB cameras, the robot state and the instruction, and has one tool: run a code cell. Inside a cell it can measure the scene, plan without moving, and carry out small manipulation goals through the `robo` object. One system, one prompt, one set of parameters for every task.

## How it works

- **Decision loop** (`agent.py`): the model writes a cell; the cell runs; its output and fresh images go back to the model. Cells share a namespace for the whole episode. Time passes only in native robot steps, so model and measurement time do not move the scene.
- **Core ICL-RAG reference** (`icl.py`, `icl_worker.py`, `robo.reference`): the configured [ICL Engine](https://github.com/manifoldai-research/icl-engine) retrieves one demonstration before the first decision. Its verified, frozen prefix stays in every planning request. The Agent uses it for strategy and measures the live scene for execution. See [setup, corpus contract and tests](ICL.md).
- **`robo`** (`robo.py`, `tools.py`): measuring (`locate`, `segment`, `triangulate`, `project`), looking (`show`, `watch`, `second_look`), planning at zero steps (`reach`, `estimate`), and goals that each complete at most one small step (`pick`, `place`, `stroke`, `turn`, `move`, `look`, `home`, or one per arm together with `both`). A goal is planned before anything moves and refused at zero steps when it cannot be done. Its failure stops the cell unless the program handles it. Programs the model writes may chain goals freely.
- **Measurement** (`objects.py`, `segmenter.py`, `cameras.py`): text- or point-prompted masks from SAM 3, turned into world geometry with the public camera calibration. Two cameras agreeing on a prism fix its height; pixels covered by the robot, drawn from its meshes, do not count against it.
- **Motion** (`control.py`, `planning.py`, `kinematics.py`, `timing.py`, `monitor.py`): joint goals chosen by distance from the current posture inside the arm's real joint range, robot-only cuRobo paths, straight Cartesian lines by damped least squares, trapezoidal timing at set joint speeds below the drive's limits, and joint tracking as the only contact feedback.
- **Learned policy** (`wam.py`, `robo.policy`): one more thing the agent can call. A policy that its authors post-trained on demonstrations of this benchmark's tasks drives both arms for a few chunks of native steps, seeing the three cameras, the joints and one sentence. Whether to call it is the agent's decision in each episode, from what it sees.
- **Knowledge** (`knowledge/`): one handbook, always in the prompt. It holds general manipulation practice and the cross-task scoring rules.

`INFORMATION.md` lists everything the policy reads at run time and where each constant comes from.

## Installation

Use a Linux Python 3.11 policy environment with a compatible CUDA build of torch and torchvision. Install RoboDojo's public robot assets and camera configuration first. From this repository root:

```bash
bash XPolicyLab/policy/R2SI/install.sh
```

This installs the XPolicyLab, SAM 3 and native planning dependencies. The adapter reuses the cuRobo source in `../WorldScape_RSI/harness/vendor/runtime_support/curobo/`, which is byte-identical to the 638 cuRobo files in the source branch; keep that directory when packaging this adapter.

Download `sam3.pt` from `facebook/sam3` after accepting its access conditions, and configure the public assets and checkpoint:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export R2SI_SEGMENTER_CHECKPOINT=/data/models/sam3/sam3.pt
```

`ROBODOJO_ROOT` must contain `Assets/Robots/x5/` and `env_cfg/camera/`. Without this variable, the parent workspace of XPolicyLab is used. A relative `segmenter_checkpoint` in `deploy.yml` resolves against that workspace; `R2SI_SEGMENTER_CHECKPOINT` takes precedence.

Optionally serve the learned policy as described in [`../WorldScape_RSI/robodojo_server/README.md`](../WorldScape_RSI/robodojo_server/README.md) and set `R2SI_POLICY_SERVICE=ws://your-policy-host:18880`, or set `policy_service` in `deploy.yml`. The environment variable takes precedence. Without a service, `robo.policy` returns a failure; native actions remain available. Download its public weights with `bash XPolicyLab/policy/WorldScape_RSI/download_checkpoints.sh /data/models/WorldScape-R2SI`.

Set the model service in the environment of the policy server:

```bash
export R2SI_API_BASE=https://your-endpoint/v1
export R2SI_API_KEY=...
```

## Data Processing

This is an eval-only agent adapter. It consumes official decoded RGB observations, joint/gripper states and the instruction. No dataset conversion or training entrypoint is included. Learned-policy training belongs to the existing [WorldScape_RSI adapter](../WorldScape_RSI/README.md).

## Training

There is no training procedure for the code-writing agent in this contribution, and no separate training release date is committed. SAM 3 and the optional WorldScape policy use their published checkpoints.

## Evaluation

From the workspace that holds the benchmark and this repository:

```bash
bash XPolicyLab/policy/R2SI/eval.sh RoboDojo <task> default arx_x5 joint <seed> <policy gpu> <env gpu> <policy env> <env env>
```

For example, in a configured RoboDojo evaluation workspace:

```bash
bash XPolicyLab/policy/R2SI/eval.sh RoboDojo stack_bowls default arx_x5 joint 0 0 1 r2si base
```

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, `eval_batch=false`. Run one policy process per environment; batch mode is rejected before loading models. Shared protocol conventions are in the [XPolicyLab README](../../README.md).

`deploy.py` forwards the standard observations without adding environment counters or a task step limit. The Agent counts only its own actions acknowledged by subsequent observations, including keepalive holds; extra observation metadata is ignored. The evaluator controls episode termination. The Agent can estimate motion cost but cannot report remaining steps or reject a motion based on the evaluator's unknown allowance. This adapter does not change the shared WebSocket client or evaluation rules, and is not wired into the outer RSI improvement loop.

## Configuration

| Key | Meaning |
| --- | --- |
| `model`, `effort` | Model and reasoning effort accepted by your Responses-compatible service; configure for your account |
| `request_timeout_s` | Limit for one model request (900 seconds by default) |
| `keepalive_s` | Maximum wait for a native action before returning one hold action (90 seconds, below the standard evaluator's 120-second RPC deadline) |
| `segmenter_checkpoint` | SAM 3 checkpoint file, with the path and environment override described above |
| `policy_service` | Optional learned-policy WebSocket endpoint; null by default |
| `icl_config`, `icl_python` | External ICL Engine TOML and optional worker Python; `R2SI_ICL_CONFIG` / `R2SI_ICL_PYTHON` override them; no config reports an explicitly disabled reference |
| `icl_timeout_s` | Retrieval/compilation deadline, 180 seconds by default; configured failures abort instead of silently dropping ICL |
| `icl_max_images`, `icl_max_text_chars` | Caps for the entire frozen prefix (64 images / 32000 text characters); exceeding them is an error |

`R2SI_TRACE_DIR` enables local diagnostics. `R2SI_SESSION_RELEASE_PATH` is an optional model-service session cleanup endpoint. Neither is a policy decision input. Every hold action consumes a native step; the model's longer timeout does not relax the evaluator deadline.

## Test

The agent runs against a kinematic stand-in for the evaluator with a scripted model, without a simulator, a GPU or a model service; it reads the public robot and camera files of the RoboDojo workspace:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
python -m XPolicyLab.policy.R2SI.tests.test_agent
python -m unittest XPolicyLab.policy.R2SI.tests.test_packaging
```

These CPU tests use stand-ins for the planner, segmentation model and language model. They do not establish simulator success rates or validate GPU initialization. Run the [XPolicyLab debug and simulator checks](../../CONTRIBUTING.md#testing) in a fully configured environment before a leaderboard submission. The Python cell sandbox restricts available APIs and resources; it is not an operating-system security boundary for hostile code.
