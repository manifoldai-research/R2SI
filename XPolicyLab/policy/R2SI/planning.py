"""Robot-only motion planning: choose a joint goal for a pose, then plan a self-collision-free joint path to it.

cuRobo's IK ranks converged solutions by pose accuracy alone, so its answer can be a whole turn or another arm branch away
from the current posture, and the joint path to it sweeps the arm. Here IK runs from deterministic seeds around the current
joints, each solution is moved by whole turns toward the current joints inside the joint range, the nearest by a weighted
joint distance is taken, and the planner plans to that joint goal. No scene geometry is ever given to the planner.
"""
from __future__ import annotations

import logging
import math
import pathlib
import sys
from copy import deepcopy

import numpy as np
import yaml

from .robot import CUROBO_CONFIG, WORKSPACE, wrist_folded

LOG = logging.getLogger(__name__)
# Reuse the identical cuRobo source already distributed in this repository.
THIRD_PARTY = pathlib.Path(__file__).resolve().parent.parent / "WorldScape_RSI" / "harness" / "vendor" / "runtime_support" / "curobo"
SEEDS = 32
PATH_POINT_DT_S = 0.004      # time between the points of a planned path, far finer than a native step; timing.py times the path afresh
PERTURB_RAD = 0.4
HALTON_BASES = (2, 3, 5, 7, 11, 13)
# The base yaw, shoulder and elbow move the whole arm, so they cost more than the wrist joints.
JOINT_WEIGHTS = (3.0, 3.0, 2.0, 1.0, 1.0, 1.0)
# An empty open gripper may arrive half-turned about its pointing axis when that posture is clearly nearer.
TURN_SHARE = 0.8
# A joint path longer than this many times the direct joint distance plus some steps is refused, not swept.
DETOUR_RATIO = 1.5
DETOUR_SLACK_STEPS = 10.0
# How many joint goals are tried, nearest first, before a pose counts as having no path.
GOALS_TRIED = 3


def _halton(index: int, base: int) -> float:
    value, f = 0.0, 1.0
    while index > 0:
        f /= base
        value += f * (index % base)
        index //= base
    return value


def seeds(current, initial, low, high, count: int = SEEDS, perturb: float = PERTURB_RAD) -> np.ndarray:
    """Deterministic IK seeds inside [low, high]: the current joints, the wrist turned half a turn either way, the episode's
    first joints (also wrist-turned), single-joint perturbations of the current joints, then a Halton fill around them."""
    current, initial = np.asarray(current, float), np.asarray(initial, float)
    rows = []
    for base in (current, initial):
        rows.append(base.copy())
        for turn in (math.pi, -math.pi):
            q = base.copy()
            q[-1] += turn
            rows.append(q)
    for k in range(len(current)):
        for d in (perturb, -perturb):
            q = current.copy()
            q[k] += d
            rows.append(q)
    index = 1
    while len(rows) < count:
        rows.append(current + np.array([2 * _halton(index, b) - 1 for b in HALTON_BASES[:len(current)]]) * math.pi)
        index += 1
    return np.clip(np.array(rows[:count]), np.asarray(low, float), np.asarray(high, float))


def nearest_turn(q, ref, low, high) -> np.ndarray | None:
    """q with each joint moved by whole turns to the value nearest ref inside [low, high]; None when a joint has none."""
    out = np.asarray(q, float).copy()
    for k in range(len(out)):
        turns = round((ref[k] - out[k]) / (2 * math.pi))
        inside = [v for v in (out[k] + 2 * math.pi * n for n in (turns - 1, turns, turns + 1)) if low[k] - 1e-9 <= v <= high[k] + 1e-9]
        if not inside:
            return None
        out[k] = min(inside, key=lambda v: abs(v - ref[k]))
    return out


def joint_distance(a, b) -> float:
    d = np.asarray(a, float) - np.asarray(b, float)
    return float(math.sqrt(float(np.sum(np.asarray(JOINT_WEIGHTS) * d * d))))


def rank_goals(candidates, current, low, high) -> tuple[list, int]:
    """Joint goals from IK solutions given as (joints, half_turned), nearest first, a half-turned one only ahead of the
    plain ones when clearly nearer. Returns ([(goal, half_turned)], how many solutions fell outside the joint range)."""
    rows, outside = [], 0
    for joints, half in candidates:
        q = nearest_turn(joints, current, low, high)
        if q is None or wrist_folded(q):
            outside += 1
            continue
        rows.append((joint_distance(q, current) / (TURN_SHARE if half else 1.0), half, q))
    rows.sort(key=lambda r: r[0])
    kept = []
    for _, half, q in rows:
        if not any(float(np.max(np.abs(q - other))) < 0.05 for other, _ in kept):
            kept.append((q, half))
    return kept, outside


def leaves_range(path, low, high) -> bool:
    """Whether a joint path goes past a joint's range or folds the wrist onto the forearm, further than where it starts."""
    path = np.asarray(path, float)
    if np.any(path < np.minimum(low, path[0])) or np.any(path > np.maximum(high, path[0])):
        return True
    return any(wrist_folded(q) and q[3] < path[0][3] for q in path)


def along(path, share: float) -> np.ndarray:
    """The joints that share of the way along a joint path, counted in its points."""
    index = share * (len(path) - 1)
    low = int(np.floor(index))
    high = min(low + 1, len(path) - 1)
    return path[low] + (index - low) * (path[high] - path[low])


def path_steps(path, step: float) -> float:
    path = np.asarray(path, float)
    if len(path) < 2:
        return 0.0
    return float(np.sum(np.max(np.abs(np.diff(path, axis=0)), axis=1))) / step


def direct_steps(start, goal, step: float) -> float:
    return float(np.max(np.abs(np.asarray(goal, float) - np.asarray(start, float)))) / step


class Curobo:
    """cuRobo's IK solver and joint-space motion planner for one arm, robot-only."""

    def __init__(self, root=None):
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        import curobo.runtime as runtime
        from curobo.types import DeviceCfg
        runtime.cuda_graph_reset = True
        root = pathlib.Path(root or WORKSPACE)
        path = root / CUROBO_CONFIG
        self.names = [f"joint{i}" for i in range(1, 7)]
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.frame_bias = np.asarray(data["planner"]["frame_bias"], dtype=np.float64)
        robot = self._robot_config(data, root)
        self.device_cfg = DeviceCfg()
        cfg = MotionPlannerCfg.create(robot=robot, scene_model=None, device_cfg=self.device_cfg, num_ik_seeds=SEEDS, num_trajopt_seeds=4,
                                      self_collision_check=True, use_cuda_graph=True, max_batch_size=1, multi_env=False, max_goalset=1)
        cfg.trajopt_solver_config.interpolation_dt = PATH_POINT_DT_S
        cfg.graph_planner_config = None
        if cfg.scene_collision_cfg is not None:
            raise RuntimeError("the planner must not hold scene geometry")
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=False, num_warmup_iterations=5)
        cspace = robot["robot_cfg"]["kinematics"]["cspace"]
        retract = np.asarray(cspace["default_joint_position"], dtype=np.float32).reshape(-1)
        configured = list(cspace["joint_names"])
        self.joint_names = list(self.planner.joint_names)
        self.retract = np.array([retract[configured.index(n)] if n in configured else 0.0 for n in self.joint_names], dtype=np.float32)
        self.active = [self.joint_names.index(n) for n in self.names]
        self.tool_frames = list(self.planner.tool_frames)
        self.ee_link = self.tool_frames[0]
        self.ik = InverseKinematics(InverseKinematicsCfg.create(
            robot=robot, scene_model=None, device_cfg=self.device_cfg, num_seeds=SEEDS, seed_solver_num_seeds=SEEDS,
            position_tolerance=0.001, orientation_tolerance=0.02, self_collision_check=True, use_cuda_graph=True,
            max_batch_size=1, multi_env=False, max_goalset=1))
        self._prewarm()

    def _robot_config(self, data, root) -> dict:
        kinematics = deepcopy(data.get("robot_cfg", {}).get("kinematics", {}))
        tool_frames = kinematics.get("tool_frames") or [kinematics["ee_link"]]
        kinematics["tool_frames"] = list(tool_frames)
        kinematics["format_version"] = 2.0
        cspace = deepcopy(kinematics.get("cspace", {}))
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace["retract_config"]
        cspace.pop("retract_config", None)
        allowed = {"joint_names", "default_joint_position", "cspace_distance_weight", "null_space_weight", "null_space_maximum_distance",
                   "max_acceleration", "max_jerk", "velocity_scale", "acceleration_scale", "jerk_scale", "position_limit_clip"}
        cspace = {k: v for k, v in cspace.items() if k in allowed}
        cspace.setdefault("joint_names", list(self.names))
        cspace.setdefault("default_joint_position", [0.0] * len(cspace["joint_names"]))
        kinematics["cspace"] = cspace
        allowed = {"format_version", "base_link", "tool_frames", "collision_link_names", "collision_spheres", "collision_sphere_buffer",
                   "extra_collision_spheres", "self_collision_ignore", "self_collision_buffer", "asset_root_path", "mesh_link_names",
                   "lock_joints", "extra_links", "cspace", "urdf_path", "use_global_cumul", "grasp_contact_link_names"}
        kinematics = {k: v for k, v in kinematics.items() if k in allowed}
        robot_root = root / "Assets/Robots/x5"
        kinematics["asset_root_path"] = str(robot_root.resolve())
        kinematics["urdf_path"] = str(robot_root / pathlib.Path(kinematics["urdf_path"]).name)
        return {"robot_cfg": {"kinematics": kinematics}, "load_dynamics": bool(data.get("load_dynamics", False))}

    def _prewarm(self):
        """The first call of a solve mode captures its CUDA graph; pay for it at start-up."""
        from curobo.types import GoalToolPose, JointState
        try:
            state = JointState.from_position(self.planner.default_joint_state.clone().unsqueeze(0).position,
                                             joint_names=list(self.planner.joint_names))
            goal = state.clone()
            goal.position[..., 0] += 0.05
            self.planner.plan_cspace(goal_state=goal, current_state=state, max_attempts=1)
        except Exception as error:
            LOG.warning("the planner did not warm up: %s", error)
        try:
            state = JointState.from_position(self.ik.default_joint_state.clone().unsqueeze(0).position, joint_names=list(self.ik.joint_names))
            kinematics = self.ik.compute_kinematics(state)
            goal = GoalToolPose.from_poses(kinematics.tool_poses.to_dict(), ordered_tool_frames=list(self.ik.tool_frames), num_goalset=1)
            self.ik.solve_pose(goal_tool_poses=goal, current_state=state, return_seeds=1)
        except Exception as error:
            LOG.warning("the IK solver did not warm up: %s", error)

    def _reset_seeds(self):
        # cuRobo's sample generators advance with every call; reset them so the same request plans the same path.
        self.planner.reset_seed()
        self.ik.reset_seed()

    def _state(self, joints, names=None):
        import torch
        from curobo.types import JointState
        names = list(names or self.joint_names)
        values = self.retract.copy()
        for index, value in zip(self.active, np.asarray(joints, dtype=np.float32).reshape(-1)):
            values[index] = value
        ordered = [float(values[self.joint_names.index(n)]) if n in self.joint_names else 0.0 for n in names]
        return JointState.from_position(torch.as_tensor([ordered], dtype=torch.float32, device=self.device_cfg.device), joint_names=names)

    def solve(self, start, flange_in_base, seed_rows) -> list[np.ndarray]:
        """Every converged IK solution for a flange transform in the base frame, from the given seeds."""
        import torch
        from curobo.types import GoalToolPose, Pose
        from scipy.spatial.transform import Rotation
        device = self.device_cfg.device
        position = flange_in_base[:3, 3] + self.frame_bias
        x, y, z, w = Rotation.from_matrix(flange_in_base[:3, :3]).as_quat()
        goal = GoalToolPose.from_poses(
            {self.ee_link: Pose(position=torch.as_tensor(position, dtype=torch.float32, device=device).reshape(1, 3),
                                quaternion=torch.as_tensor([w, x, y, z], dtype=torch.float32, device=device).reshape(1, 4))},
            ordered_tool_frames=self.tool_frames, num_goalset=1)
        names = list(self.ik.joint_names)
        current = self._state(start, names)
        rows = []
        for row in seed_rows:
            values = self.retract.copy()
            for index, value in zip(self.active, np.asarray(row, dtype=np.float32)):
                values[index] = value
            rows.append([float(values[self.joint_names.index(n)]) if n in self.joint_names else 0.0 for n in names])
        seed_config = torch.as_tensor([rows], dtype=torch.float32, device=device)
        # The IK's seed stage keeps its success buffer from the previous call, so the first call after another target can stop
        # early with fewer converged seeds; the second call on the same request answers for this request alone.
        for _ in range(2):
            self._reset_seeds()
            result = self.ik.solve_pose(goal_tool_poses=goal, current_state=current, seed_config=seed_config, return_seeds=len(rows))
        if result.success is None:
            return []
        positions = np.asarray(result.js_solution.position.detach().cpu())
        positions = positions.reshape(-1, positions.shape[-1])
        success = np.asarray(result.success.detach().cpu()).reshape(-1)
        solved = list(getattr(result.js_solution, "joint_names", None) or names)
        index = [solved.index(n) for n in self.names]
        return [positions[k][index].astype(float) for k in range(min(len(success), len(positions))) if success[k]]

    def path(self, start, goal) -> np.ndarray | None:
        """A self-collision-free joint path from start to goal (N x 6), or None."""
        self._reset_seeds()
        result = self.planner.plan_cspace(current_state=self._state(start), goal_state=self._state(goal), max_attempts=5, enable_graph_attempt=0)
        if result is None or result.success is None or not bool(np.asarray(result.success.detach().cpu()).any()):
            return None
        interpolated = result.get_interpolated_plan()
        rows = np.asarray(interpolated.position.detach().cpu())
        while rows.ndim > 2:
            rows = rows[0]
        names = list(getattr(interpolated, "joint_names", None) or self.joint_names)
        return rows[:, [names.index(n) for n in self.names]].astype(np.float64)
