# WorldScape_R2SI

[中文](README_zh.md) · [WAM setup](WAM.md)

**Contributor:** jwt / Sa63R | **Source:** R2SI-research, `improve/v4` at `1229cd1d0c146c5a61337bf2f1d36e15a2a32391`

An agent that controls a two-arm robot by writing Python. It sees the three RGB cameras, the robot state and the instruction, and has one tool: run a code cell. Inside a cell it can measure the scene, plan without moving, and carry out small manipulation goals through the `robo` object. One system, one prompt, one set of parameters for every task.

The former `R2SI/` and `WorldScape_RSI/` adapters are now one policy. `model.py` uses the frozen Harness boundary to run the v4 code-writing Agent; standard deployment and the outer RSI loop use that same execution path. The Agent reads the shared generic Skills with `robo.skills()` / `robo.read_skill()`, receives the candidate's Memory in its prompt, and calls retained Flash/Memory WAMs through `robo.policy(profile="flash" or "memory")`. Outer-loop code updates target the actual Agent/tool files. There is no mode selector or separate legacy entry. Existing `R2SI_*` and `WRSI_*` environment variables keep their names.

## How it works

- **Decision loop** (`agent.py`): the model writes a cell; the cell runs; its output and fresh images go back to the model. Cells share a namespace for the whole episode. Time passes only in native robot steps, so model and measurement time do not move the scene.
- **Core ICL-RAG reference** (`icl.py`, `icl_worker.py`, `robo.reference`): the configured [ICL Engine](https://github.com/manifoldai-research/icl-engine) retrieves one demonstration before the first decision. Its verified, frozen prefix stays in every planning request. The Agent uses it for strategy and measures the live scene for execution. See [setup, corpus contract and tests](ICL.md).
- **`robo`** (`robo.py`, `tools.py`): measuring (`locate`, `segment`, `triangulate`, `project`), looking (`show`, `watch`, `second_look`), planning at zero steps (`reach`, `estimate`), and goals that each complete at most one small step (`pick`, `place`, `stroke`, `turn`, `move`, `look`, `home`, or one per arm together with `both`). A goal is planned before anything moves and refused at zero steps when it cannot be done. Its failure stops the cell unless the program handles it. Programs the model writes may chain goals freely.
- **Measurement** (`objects.py`, `segmenter.py`, `cameras.py`): text- or point-prompted masks from SAM 3, turned into world geometry with the public camera calibration. Two cameras agreeing on a prism fix its height; pixels covered by the robot, drawn from its meshes, do not count against it.
- **Motion** (`control.py`, `planning.py`, `kinematics.py`, `timing.py`, `monitor.py`): joint goals chosen by distance from the current posture inside the arm's real joint range, robot-only cuRobo paths, straight Cartesian lines by damped least squares, trapezoidal timing at set joint speeds below the drive's limits, and joint tracking as the only contact feedback.
- **Learned policy** (`wam.py`, `robo.policy`): one more thing the agent can call. A policy that its authors post-trained on demonstrations of this benchmark's tasks drives both arms for a few chunks of native steps, seeing the three cameras, the joints and one sentence. Whether to call it is the agent's decision in each episode, from what it sees.
- **Memory rules, Skills and tool notes** (`harness/memory/`, the Skill library, `knowledge/tool-notes.md`): the former handbook split as in the report's H = (M, K, T). Memory rules (the situation, what you know, how work is scored) and the Skills with `load_mode: always` are in the prompt at every step; other Skills are listed by id and read with `robo.read_skill`; tool notes follow the robo reference.

`INFORMATION.md` lists everything the policy reads at run time and where each constant comes from.

## Installation

Use a Linux Python 3.11 policy environment with a compatible CUDA build of torch and torchvision. Install RoboDojo's public robot assets and camera configuration first. From this repository root:

```bash
bash XPolicyLab/policy/WorldScape_R2SI/install.sh
```

This installs the XPolicyLab, SAM 3 and native planning dependencies. The adapter includes the shared cuRobo source in `harness/vendor/runtime_support/curobo/` and the WAM implementation in `worldscape-policy/`. Export this directory as a whole; neither former policy directory is required.

Download `sam3.pt` from `facebook/sam3` after accepting its access conditions, and configure the public assets and checkpoint:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export R2SI_SEGMENTER_CHECKPOINT=/data/models/sam3/sam3.pt
```

`ROBODOJO_ROOT` must contain `Assets/Robots/x5/` and `env_cfg/camera/`. Without this variable, the parent workspace of XPolicyLab is used. A relative `segmenter_checkpoint` in `deploy.yml` resolves against that workspace; `R2SI_SEGMENTER_CHECKPOINT` takes precedence.

Optionally serve the learned policy as described in [`robodojo_server/README.md`](robodojo_server/README.md) and set `R2SI_POLICY_SERVICE=ws://your-policy-host:18880`, or set `policy_service` in `deploy.yml`. The environment variable takes precedence. With `policy_service: null`, `robo.policy` uses the bundled WAM implementation and its `WRSI_*` checkpoint bindings (or the existing `WRSI_POLICY_CLIENT_CONFIG` remote lease service). Use [WAM setup](WAM.md) to bind Flash/Memory. Models load lazily when called; missing checkpoints report a failure. Download its public weights with `bash XPolicyLab/policy/WorldScape_R2SI/download_checkpoints.sh /data/models/WorldScape-R2SI`.

Set the model service in the environment of the policy server:

```bash
export R2SI_API_BASE=https://your-endpoint/v1
export R2SI_API_KEY=...
```

## Data Processing

The v4 Agent consumes official decoded RGB observations, joint/gripper states and the instruction. The retained `process_data.sh` prepares WAM data with the shared XPolicyLab image helpers; see [data preparation](WAM.md#data-processing).

## Training

This release is **inference-only**. WSP training is a **reserved interface (stub)**: `training.train_wsp(...)`, `rsi.jobs.Jobs.train(...)` and the outer loop's training hook raise `WSPTrainingUnavailable` (`NotImplementedError`) before starting work. `train.sh` documents the existing six-argument CLI with `--help`; an actual training request exits with code 2. No job, checkpoint or successful training metrics are fabricated. The outer loop reports training as unavailable even if an older campaign contains a training configuration.

The bundled model/trainer source and data utilities remain for future backend integration. WAM Flash/Memory inference and checkpoint downloads remain available. A training-backend release date is not yet committed.

## Evaluation

From the workspace that holds the benchmark and this repository:

```bash
bash XPolicyLab/policy/WorldScape_R2SI/eval.sh RoboDojo <task> default arx_x5 joint <seed> <policy gpu> <env gpu> <policy env> <env env>
```

For example, in a configured RoboDojo evaluation workspace:

```bash
bash XPolicyLab/policy/WorldScape_R2SI/eval.sh RoboDojo stack_bowls default arx_x5 joint 0 0 1 r2si base
```

Supported: `bench_name=RoboDojo`, `env_cfg_type=arx_x5`, `action_type=joint`, `eval_batch=false`. Run one policy process per environment; batch mode is rejected before loading models. Shared protocol conventions are in the [XPolicyLab README](../../README.md).

`deploy.py` forwards the standard observations without adding environment counters or a task step limit. The Agent counts only its own actions acknowledged by subsequent observations, including keepalive holds; extra observation metadata is ignored. The evaluator controls episode termination. The Agent can estimate motion cost but cannot report remaining steps or reject a motion based on the evaluator's unknown allowance. This adapter does not change the shared WebSocket client or evaluation rules, and uses the same frozen version, trace and episode interfaces as the outer RSI improvement loop.

`setup_eval_policy_server.sh` starts `server.py` with the same `deploy.yml` and Model used by the standard XPolicyLab entry. Its single-trial boundary prevents a second trial from resetting an active episode. No additional selector is needed.

## Configuration

| Key | Meaning |
| --- | --- |
| `model`, `effort` | Model and reasoning effort accepted by your Responses-compatible service; configure for your account |
| `request_timeout_s` | Limit for one model request (900 seconds by default) |
| `keepalive_s` | Maximum wait for a native action before returning one hold action (90 seconds, below the standard evaluator's 120-second RPC deadline) |
| `segmenter_checkpoint` | SAM 3 checkpoint file, with the path and environment override described above |
| `policy_service` | Optional learned-policy WebSocket endpoint; null uses bundled Flash/Memory |
| `icl_config`, `icl_python` | External ICL Engine TOML and optional worker Python; `R2SI_ICL_CONFIG` / `R2SI_ICL_PYTHON` override them; no config reports an explicitly disabled reference |
| `icl_timeout_s` | Retrieval/compilation deadline, 180 seconds by default; configured failures abort instead of silently dropping ICL |
| `icl_max_images`, `icl_max_text_chars` | Caps for the entire frozen prefix (64 images / 32000 text characters); exceeding them is an error |

`R2SI_TRACE_DIR` enables local diagnostics. `R2SI_SESSION_RELEASE_PATH` is an optional model-service session cleanup endpoint. Neither is a policy decision input. Every hold action consumes a native step; the model's longer timeout does not relax the evaluator deadline.

## Test

The agent runs against a kinematic stand-in for the evaluator with a scripted model, without a simulator, a GPU or a model service; it reads the public robot and camera files of the RoboDojo workspace:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
python -m XPolicyLab.policy.WorldScape_R2SI.tests.test_agent
python -m pytest -q tests XPolicyLab/policy/WorldScape_R2SI/harness/tests XPolicyLab/policy/WorldScape_R2SI/tests
```

These CPU tests use stand-ins for the planner, segmentation model and language model. They do not establish simulator success rates or validate GPU initialization. Run the [XPolicyLab debug and simulator checks](../../CONTRIBUTING.md#testing) in a fully configured environment before a leaderboard submission. The Python cell sandbox restricts available APIs and resources; it is not an operating-system security boundary for hostile code.
