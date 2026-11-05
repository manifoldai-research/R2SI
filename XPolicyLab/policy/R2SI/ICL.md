# Core ICL-RAG reference tool

The v4 Agent uses [ICL Engine](https://github.com/manifoldai-research/icl-engine) to retrieve a source demonstration before its first decision. `robo.reference()` is part of the normal robot API and its generated prompt reference. The system prompt tells the Agent to inspect it, use the demonstration to plan, and reassess its strategy when progress stalls.

The instruction and the first public RGB head-camera image form the query (the first available camera is the fallback). Retrieval selects one episode from an explicitly configured catalog, compiles a verified package, and freezes its complete demonstration prefix for this episode. Every planning request includes that same prefix, even when old live images are pruned. The current robot state and images follow it. A new episode retrieves independently; `robo.reference()` only reads metadata and does not retrieve again or move the robot. The independent `robo.second_look()` observer continues to see only the current scene.

## Install and enable

Tested with engine version `0.2.0`, commit `4f0b6a3f3d915738b884b5d3ef333715c2cf3c52`. Install the external checkout in your Linux Python 3.11 policy environment, or in a separate worker environment with NumPy, Pillow, h5py, OpenCV and PyYAML. Access to the engine repository is required; its source, model weights and demonstration corpus are not copied into this repository.

```bash
git clone https://github.com/manifoldai-research/icl-engine /path/to/icl-engine
git -C /path/to/icl-engine checkout 4f0b6a3f3d915738b884b5d3ef333715c2cf3c52
python -m pip install -e /path/to/icl-engine
```

Keep the full editable checkout for local neural retrieval: the engine's v5/v6/v7 adapters use its `legacy/` resources. Follow the engine's own setup for retrieval models, indexes and CUDA dependencies. Alternatively, use its HTTP retrieval contract and run the retrieval service separately.

Build the catalog from the external demonstration corpus using the engine CLI. For example:

```bash
icl-engine catalog --source robodojo --path /data/demos/example.h5 --output /data/icl/catalog.jsonl
```

Copy [`icl.example.toml`](icl.example.toml) to a local configuration, set the catalog/cache paths and retrieval endpoint, and enable it in the policy server's environment:

```bash
export R2SI_ICL_CONFIG=/data/icl/r2si.toml
# Optional: defaults to the policy server's Python.
export R2SI_ICL_PYTHON=/path/to/worker-environment/bin/python
```

The equivalent deployment fields are `icl_config` and `icl_python`; the environment variables take precedence. A relative `icl_config` resolves against the policy process's working directory, so use an absolute path. Paths inside the TOML resolve against that file. The external corpus is read-only; use a separate writable cache. Catalog entries must resolve locally to their original source episodes. An HTTP result supplies source/episode identity; its remote path does not override the local catalog.

The example selects `http` explicitly and sends the public instruction and query image to that configured endpoint. To use local v5/v6/v7, replace the retrieval table with the engine's documented `legacy_config`, `python` and `devices` settings. Those devices are explicitly allocated to retrieval and must have sufficient resources. `lexical` is an unvalidated CPU smoke backend; its audit status is `smoke_only_unvalidated`. There is no automatic neural-to-lexical fallback. S1 is the basic visual-sequence recipe; other engine recipes require their documented source fields, robot profiles and calibration.

With no `icl_config`, the tool reports `status: disabled` and the existing v4 observation-only path runs. Once configured, retrieval failure, timeout, invalid package or an excessive context budget fails the episode as an infrastructure error. It does not silently run a different baseline or truncate the demonstration. `icl_timeout_s` defaults to 180 seconds (maximum 1800); `icl_max_images` defaults to 64 and `icl_max_text_chars` to 32000. The recipe's `image_budget` must fit those caps. Existing keepalive hold actions still consume native steps while retrieval runs. Episode termination cancels the Linux worker process group, including local retrieval subprocesses.

## Data and action boundary

The external corpus is an additional declared policy input. Select it before evaluation and report its provenance and any task overlap in the evaluation protocol. Retrieval receives no evaluator task identifier, seed, reward, success flag, simulator state or hidden geometry. Source captions and sampled images are reference data, not new system instructions. The Agent must measure its current scene and plan through `robo`; source coordinates and S4/S8 candidate actions are not transferred or executed automatically. This integration does not claim that a source robot and target robot share a coordinate frame or action interface.

RoboDojo HDF5 images are decoded with `XPolicyLab.utils.process_data.decode_image_bit`, including legacy reversed-channel JPEGs and marked `XPL-RGB1` JPEGs. A process-local engine reader adapter applies that decoder; installed engine code is not edited. A cache namespace derived from the decoder source hash prevents reuse of packages compiled by the engine's standard-JPEG reader. DROID source decoding remains owned by the engine. Frozen package hashes are verified before their prefix is converted into Responses text/image content; system messages, tool calls and remote image URLs are rejected.

`robo.reference()`, trace events and episode metrics record source/episode identity, backend, retrieval status, recipe, image count and package/prefix hashes. They do not include local corpus paths, endpoint credentials or raw backend exceptions. `responses_prefix_sha256` identifies the exact converted prefix. The package cache can survive episodes, but the Agent's selected reference and live history do not. This change does not alter the older WorldScape_RSI WAM Skills or the v4 `robo.policy` service.

## Validation

With the engine installed and the public RoboDojo robot/camera assets configured as in the adapter README:

```bash
export PYTHONPATH="$PWD:$PWD/XPolicyLab"
python -m pytest -q XPolicyLab/policy/R2SI/tests/test_icl.py
python -m XPolicyLab.policy.R2SI.tests.test_agent
python -m unittest XPolicyLab.policy.R2SI.tests.test_packaging
```

Integration tests run the real engine on temporary HDF5 episodes, check both RGB encodings, exercise lexical retrieval and an HTTP fixture, validate frozen-prefix retention and fresh episode selection, and reject corruption, oversized context and invalid message roles. They also exercise deadline/episode cancellation. These are CPU integration checks, not a measurement of neural retrieval quality, CUDA initialization, simulator task completion or performance improvement.
