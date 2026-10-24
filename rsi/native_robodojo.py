"""Replay official non-Open layouts using the pinned native controller and rewards.
Layout provenance stays official/public; no scene synthesis or model-training export.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.machinery
import json
import math
from pathlib import Path
import random
import sys
import types
from .official_layouts import EVIDENCE, validate_case, load_layout

CONTROL_SHA256 = "4a09b058f90bda1087233e4622c1236211d97e91e1dd98d6e32e8f6a5371a42d"
CONTROL_METHODS = {
    "get_action_type", "take_action", "take_action_batch", "process_control_info",
    "validate_action_dict", "step", "is_episode_end", "have_empty", "get_control_empty",
}
_app = _source = None


def configure(app, source):
    global _app, _source
    _app, _source = app, Path(source).resolve(strict=True)
    if str(_source) not in sys.path:
        sys.path.insert(0, str(_source))
    # RoboDojo's utils directory is a namespace package. Isaac/XPolicyLab also
    # supplies a top-level regular "utils", which otherwise wins resolution.
    # Bind only this isolated simulator process to the reviewed task utilities.
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    module = types.ModuleType("utils")
    module.__path__ = [str(_source / "utils")]
    module.__package__ = "utils"
    module.__spec__ = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    sys.modules["utils"] = module


def control_class(task_class, source):
    """Compile exactly the pinned evaluator's control methods, without its I/O.

    Keeping the class syntax preserves zero-argument super() in native step().
    Constructor, sockets, official layouts and video writers are not imported.
    A source change fails closed until reviewed and repinned.
    """
    import numpy as np
    path = Path(source) / "src/eval_client/eval_env.py"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != CONTROL_SHA256:
        raise ValueError("RoboDojo control source differs from the reviewed commit")
    tree = ast.parse(data, filename=str(path))
    factory = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "create_eval_env")
    original = next(x for x in factory.body if isinstance(x, ast.ClassDef) and x.name == "EvalEnv")
    methods = [x for x in original.body if isinstance(x, ast.FunctionDef) and x.name in CONTROL_METHODS]
    if {x.name for x in methods} != CONTROL_METHODS:
        raise ValueError("Pinned evaluator control methods are incomplete")
    cls = ast.ClassDef(name="NativeControl", bases=[ast.Name(id="task_class", ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    scope = {"task_class": task_class, "deepcopy": deepcopy, "np": np, "__name__": __name__}
    exec(compile(module, str(path), "exec"), scope)
    return scope["NativeControl"]


def make_environment(case):
    validate_case(case)
    if _app is None or _source is None:
        raise RuntimeError("Launch Isaac Sim and configure the native adapter first")
    import numpy as np
    from omegaconf import OmegaConf
    from env.observation_manager.obs_manager import ObsManager
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization, get_robot_action_dim_info
    from task.RoboDojo.task_registry import load_task_class
    task_name, task_class = load_task_class(case["task"])
    spec = load_yaml(str(_source / "env_cfg/arx_x5.yml"))
    spec.update(task_name=task_name, seed=int(case["layout_id"]), num_envs=1)
    cfg = OmegaConf.create({
        **{k: load_yaml(str(_source / "env_cfg" / k / (spec["config"][k] + ".yml")))
           for k in ("sim", "scene", "robot", "camera")},
        "task_env": load_yaml(str(_source / f"task/RoboDojo/config/{task_name}.yml")),
        "eval_cfg": spec,
    })
    cfg = process_randomization(cfg)
    cfg, _ = process_config(cfg, task_name)
    cfg.sim.scene.num_envs = 1
    cfg.sim.seed = [int(case["layout_id"])]
    # Preserve the released evaluator's CPU physics/default tensor contract.
    # RTX camera rendering still runs on GPU. Its NumPy rewards do not accept
    # GPU pose tensors, so forcing CUDA physics changes the native interface.
    from env.environment.base_env import DEFAULT_SIM_DEVICE, DEFAULT_USE_FABRIC
    cfg.sim.device = DEFAULT_SIM_DEVICE
    cfg.sim.use_fabric = DEFAULT_USE_FABRIC
    cfg.camera.default_frequency = spec["observation"]["collect_freq"]
    # Joint actions need no native motion planner. Agent planning is a separate
    # Skill; disabling unused eager planner construction changes no action math.
    for robot in cfg.robot.robots:
        robot.need_planner = False
    layout = load_layout(_source, case)
    base = control_class(task_class, _source)

    class NativeEnvironment(base):
        def __init__(self):
            super().__init__(cfg, _app)
            self.eval_seed = int(case['eval_seed'])
            self.eval_cfg = spec
            self.physx_monitor_enabled = False
            self.success, self.end_flag, self.take_action_cnt = [True], [False], [0]
            self.robot_action_dim_info = get_robot_action_dim_info(spec)
            self.obs_manager = ObsManager(deepcopy(spec["observation"]), 1, self.dt,
                                          task_name, {}, seeds_per_env=[int(case["layout_id"])])
            self.scene_manager.layout_manager.set_saved_layout(0, layout)
            self.unstable_envs = set()

        def _post_setup_scene(self, sim):
            super()._post_setup_scene(sim)
            self.obs_manager.initialize(self)

        def mark_env_unstable(self, env_idx):
            self.unstable_envs.add(env_idx)

        def reset(self):
            super().reset(seed=[int(case["layout_id"])])
            self.obs_manager.reset()
            self.scene_manager.apply_saved_poses([0])
            ok, unstable = self.scene_manager.layout_manager.check_layout_stability(self)
            if not ok or unstable or not self.scene_manager.layout_manager.layout_valid[0]:
                raise RuntimeError("Official saved layout failed native stability checks")
            for _ in range(10):
                self.render()
            for i in range(200):
                self.sim_step()
                if i % 5 == 0:
                    self.obs_manager.render_for_capture()
                    self.obs_manager.get_obs()
            self.robot_manager.set_origin_endpose()
            self.robot_manager.set_robot_init_state()
            self.reward_manager.init_state()
            self.run_reward()
            if hasattr(self, "get_score"):
                self.get_score()
            if getattr(self, 'interact', False) and hasattr(self, 'query_support_arm_traj'):
                self.query_support_arm_traj(env_idx=0)
            return self.get_obs_batch()[0]

        def get_obs_batch(self, env_idx_list=None, last_frame=False):
            self.obs_manager.render_for_capture()
            observation = deepcopy(self.obs_manager.get_obs(env_idx_list=[0])[0])
            for k, value in observation["state"].items():
                observation["state"][k] = np.asarray(value)
                if not np.isfinite(observation["state"][k]).all():
                    raise RuntimeError("Non-finite native robot state")
            for name, cam in observation["vision"].items():
                rgb = cam["color"]
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                    raise RuntimeError(f"Invalid native RGB camera {name}")
            observation["env_idx"] = 0
            # No simulator pose/labels/reward or layout data enters policy.
            return [observation]

        def close(self):
            self.obs_manager.reset()
            super().close()

    native = NativeEnvironment()

    class Adapter:
        evidence_kind = EVIDENCE
        step_limit = int(native.step_lim)
        layout_sha256 = case["layout_sha256"]

        def reset(self):
            return native.reset()

        def step(self, action):
            if native.end_flag[0]:
                raise RuntimeError("Cannot act after the native terminal condition")
            if any(not np.isfinite(np.asarray(value)).all() for value in action.values()):
                raise ValueError("Non-finite action")
            native.take_action(action)
            if native.unstable_envs:
                raise RuntimeError('Native evaluator marked this scene unstable; excluded from task result')
            done = bool(native.end_flag[0])
            result = {}
            if done:
                success = bool(native.success[0])
                score = 100.0 if success else (
                    float(native.reward_manager.get_score()[0]) if hasattr(native, "get_score") else 0.0)
                result = {"success": success, "score": score, "reason": "native_terminal"}
            return native.get_obs_batch(last_frame=done)[0], done, result

        def close(self):
            native.close()

    return Adapter()
