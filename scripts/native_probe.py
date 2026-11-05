"""Real WSP GPU prediction on synthetic RGB; never claims robot success."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "XPolicyLab")]
from rsi.storage import Workspace

parser = argparse.ArgumentParser()
parser.add_argument("--workspace", required=True)
parser.add_argument("--variant", choices=["flash", "memory"], required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--imports-only", action="store_true")
parser.add_argument("--bf16-experiment", action="store_true")
args = parser.parse_args()
ws = Workspace(args.workspace)
os.environ.update(ws.environment())
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["OMP_NUM_THREADS"] = "1"
model_root = PROJECT / "XPolicyLab/policy/WorldScape_R2SI/worldscape-policy"
sys.path[:0] = [str(model_root), str(model_root / "src"), str(PROJECT / "dependencies")]
result = {"variant": args.variant, "actual_gpu_inference": False, "synthetic_observation": True,
          "robot_actions_executed": 0, "training_performed": False,
          "bf16_experiment": args.bf16_experiment}
started = time.monotonic()
model = None
try:
    import numpy as np
    import torch
    from XPolicyLab.policy.WorldScape_R2SI.xpolicylab_adapter.runtime_config import validate_config, resolve_checkpoint_deployment_config
    from XPolicyLab.policy.WorldScape_R2SI.wam_model import Model
    manifest = json.loads((PROJECT / "provenance/checkpoints.private.json").read_text())
    catalog = json.loads((PROJECT / "XPolicyLab/policy/WorldScape_R2SI/harness/configs/wam_skills.json").read_text())
    skill = "worldscape-wam-memory" if args.variant == "memory" else "worldscape-wam-nomemory"
    config = dict(catalog[skill]["config"])
    config.update(checkpoint_path=manifest[args.variant]["path"], device="cuda:0",
                  worldscape_root=str(model_root), short_window_for_non_memory_tasks=False)
    config = resolve_checkpoint_deployment_config(validate_config(config), config["checkpoint_path"])
    result["checkpoint_task_window_flag"] = bool(config.get("short_window_for_non_memory_tasks"))
    result["rsi_history_policy"] = "uniform_checkpoint_capacity_before_first_observation"
    result.update(torch=torch.__version__, metadata_preflight=True,
                  memory_mode=config.get("wam_history_mode"),
                  config={k: config.get(k) for k in ["action_horizon", "replan_steps",
                                                     "use_causal_memory", "use_observation_history"]})
    if not args.imports_only:
        import subprocess
        query = subprocess.check_output(
            ["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True, timeout=10).strip()
        if query:
            raise RuntimeError("Selected GPU is now occupied; existing jobs are left untouched")
        if args.bf16_experiment:
            # Isolated numeric experiment only; does not modify model source,
            # checkpoint bytes or the default release's precision.
            config["device"] = "cpu"
        model = Model(config)
        if config.get("short_window_for_non_memory_tasks"):
            from XPolicyLab.policy.WorldScape_R2SI.harness.wam import enforce_uniform_history
            enforce_uniform_history(model)
        if args.bf16_experiment:
            model.backend.policy.to(device="cuda:0", dtype=torch.bfloat16)
            model.backend.device = torch.device("cuda:0")
            model.backend.model_cfg["device"] = "cuda:0"
            model.reset()
        from XPolicyLab.utils.process_data import get_robot_action_dim_info
        dims = get_robot_action_dim_info("arx_x5")
        state = {side + suffix: np.zeros(int(width), np.float32)
                 for i, side in enumerate(["left", "right"])
                 for suffix, width in [("_arm_joint_state", dims["arm_dim"][i]),
                                       ("_ee_joint_state", dims["ee_dim"][i])]}
        for side in ["left", "right"]:
            state[side + "_ee_pose"] = np.array([0, 0, 1, 1, 0, 0, 0], np.float32)
        observation = {"instruction": "Place the visible object on the stable support.",
                       "state": state, "vision": {k: {"color": np.full((240, 320, 3), 128, np.uint8)}
                                                  for k in ["cam_head", "cam_right_wrist", "cam_left_wrist"]},
                       "data_format_version": "v1.0"}
        model.reset()
        model.update_obs(observation)
        inference_start = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16_experiment):
            actions = model.get_action()
        if not actions or not all(np.isfinite(v).all() for action in actions for v in action.values()):
            raise ValueError("No finite joint-action output")
        # No environment executed these predictions; discard, never commit.
        model.discard_action({"execution_started": False})
        result.update(actual_gpu_inference=True, proposed_actions=len(actions),
                      prediction_seconds=time.monotonic() - inference_start,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
    result["ok"] = True
except BaseException:
    result.update(ok=False, error=traceback.format_exc(limit=14))
finally:
    if model is not None:
        model.close()
    result["elapsed_s"] = time.monotonic() - started
    ws.atomic("validation/native-" + args.variant + ("-imports" if args.imports_only else "") +
              ("-bf16" if args.bf16_experiment else "") + ".json", result)
    print(json.dumps(result, ensure_ascii=False))
sys.exit(0 if result["ok"] else 1)
