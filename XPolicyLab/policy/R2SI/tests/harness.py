"""A kinematic stand-in for the evaluator and a scripted model, to run the whole agent without a simulator, a GPU or a model service."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from XPolicyLab.policy.R2SI import cameras, kinematics, monitor, robot
from XPolicyLab.policy.R2SI.agent import Agent
from XPolicyLab.policy.R2SI.bridge import Bridge
from XPolicyLab.policy.R2SI.control import Controller
from XPolicyLab.policy.R2SI.segmenter import Instance
from XPolicyLab.policy.R2SI.tools import Tools

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARMS = ("left", "right")


class KinematicEnv:
    """Joints follow commands exactly unless `block` stops one; the gripper reading follows the official rule."""

    def __init__(self, model, step_limit=1000, holds=True):
        self.model = model
        state = json.loads((FIXTURES / "initial_state.json").read_text())
        self.state = {key: np.asarray(value, dtype=np.float64) for key, value in state.items()}
        self.bases = {arm: model.base(self.state[arm + "_arm_joint_state"], self.state[arm + "_ee_pose"]) for arm in ARMS}
        self.fingers = {arm: 1.0 for arm in ARMS}
        self.step = 0
        self.step_limit = step_limit
        self.holds = holds         # closing fingers meet an object at this share of the range
        self.block = None          # callable(arm, joints) -> True when the arm cannot go there
        self.hold_width = 0.45
        self.grasp_track = {arm: [] for arm in ARMS}

    def observation(self):
        for arm in ARMS:
            frame = self.bases[arm] @ self.model.link_frames(self.state[arm + "_arm_joint_state"])[-1]
            from scipy.spatial.transform import Rotation
            x, y, z, w = Rotation.from_matrix(frame[:3, :3]).as_quat()
            self.state[arm + "_ee_pose"] = np.r_[frame[:3, 3], w, x, y, z]
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        return {"instruction": "Do the thing.", "state": {k: v.tolist() for k, v in self.state.items()},
                "vision": {c: {"color": image} for c in ("cam_head", "cam_left_wrist", "cam_right_wrist")},
                "additional_info": {"step": self.step, "step_lim": self.step_limit}, "task_name": "must_not_reach_the_agent", "seed": 7}

    def apply(self, action):
        self.step += 1
        for arm in ARMS:
            target = np.asarray(action[arm + "_arm_joint_state"], dtype=np.float64)
            if self.block is None or not self.block(arm, target):
                self.state[arm + "_arm_joint_state"] = target
            frame = self.bases[arm] @ self.model.link_frames(self.state[arm + "_arm_joint_state"])[-1]
            self.grasp_track[arm].append((frame[:3, 3] + frame[:3, 0] * 0.1501).tolist())
            command = float(np.asarray(action[arm + "_ee_joint_state"]).reshape(-1)[0])
            floor = self.hold_width if (self.holds and command < self.fingers[arm]) else 0.0
            self.fingers[arm] = float(np.clip(command, self.fingers[arm] - 0.2, self.fingers[arm] + 0.2))
            self.fingers[arm] = max(self.fingers[arm], floor if command < 0.5 else 0.0)
            # The reading is the previous command after the simulator limited it to a fifth of the range of the fingers.
            self.state[arm + "_ee_joint_state"] = np.array([np.clip(command, self.fingers[arm] - 0.2, self.fingers[arm] + 0.2)]).clip(0.0, 1.0)


class FakePlanner:
    """IK by the numeric solver, joint paths as straight lines: no cuRobo needed."""

    def __init__(self, model, spec, env):
        self.model, self.spec, self.env = model, spec, env

    def solve(self, start, flange_in_base, seed_rows):
        rows = []
        for seed in seed_rows:
            solved = self.model.solve(flange_in_base, seed, np.eye(4), self.spec.limit_low, self.spec.limit_high)
            if solved is not None:
                rows.append(solved)
                if len(rows) >= 6 or (np.all(solved >= self.spec.low) and np.all(solved <= self.spec.high) and not robot.wrist_folded(solved)):
                    break
        return rows

    def path(self, start, goal):
        return np.linspace(start, goal, 12)


class FakeSegmenter:
    """Returns the masks a test put there."""

    def __init__(self):
        self.answers = {}

    def segment(self, image, text, key=None, min_score=0.25, limit=12):
        return list(self.answers.get(text, []))

    def segment_point(self, image, pixel, key=None):
        return []


class ScriptedClient:
    """Stands in for the model: replies with the given code cells in turn, then blocks until the episode ends."""

    def new_cache_key(self):
        self.cache_keys += 1

    def __init__(self, cells):
        self.cells = list(cells)
        self.ended = threading.Event()
        self.results = []
        self.turn = 0
        self.session_id = "test"
        self.cache_keys = 0
        self.seen = []
        self.instructions = None
        self.waiting = False
        self.fresh = []

    def complete(self, instructions, items, tools):
        self.instructions = instructions
        self.seen = items
        for item in items:
            if item.get("type") == "function_call_output" and item not in self.results:
                self.results.append(item)
        if not self.cells:
            self.waiting = True
            self.ended.wait()
            from XPolicyLab.policy.R2SI.client import InfrastructureFailure
            raise InfrastructureFailure("episode over")
        self.turn += 1
        code = self.cells.pop(0)
        return {"status": "completed", "output": [{"type": "function_call", "call_id": f"c{self.turn}", "name": "python", "arguments": json.dumps({"code": code})}]}

    def ask_fresh(self, instructions, content):
        self.fresh.append((instructions, content))
        return "something looks off"

    def _log(self, *args, **kwargs):
        pass


def outputs(client):
    return [json.loads(item["output"]) for item in client.results]


def run_episode(cells, step_limit=1000, holds=True, keepalive_s=5.0, segmenter=None, block=None, root=None, max_steps=4000, policy=None):
    """Run scripted cells against the kinematic environment until they are used up. Returns (client, env, tools)."""
    spec = robot.load_spec(root)
    model = kinematics.ArmModel(root)
    env = KinematicEnv(model, step_limit, holds)
    env.block = block
    bridge = Bridge(monitor.Monitor(spec), keepalive_s)
    rig = cameras.Rig(root)
    ctl = Controller(spec, model, FakePlanner(model, spec, env), bridge, rig.head_stand())
    tools = Tools(ctl, rig, kinematics.RobotShape(model, root), segmenter or FakeSegmenter(), None if policy is None else "ws://fake")
    if policy is not None:
        tools.open_policy = policy
        tools.read_service = lambda url: {"config_sha256": "fake"}
    client = ScriptedClient(cells)
    agent = Agent(client, tools, bridge)
    thread = threading.Thread(target=agent.run, daemon=True)
    thread.start()
    bridge.update_obs(env.observation())
    steps = 0
    while steps < max_steps:
        if bridge._failure is not None:
            raise bridge._failure
        if not bridge._queue:
            # The episode is over for this harness when the model is out of cells and has been sent every result.
            if client.waiting:
                break
            time.sleep(0.0005)
            continue
        action = bridge.get_action()
        env.apply(action)
        bridge.update_obs(env.observation())
        steps += 1
    client.ended.set()
    bridge.close()
    thread.join(timeout=5.0)
    return client, env, tools


def square_mask(u0, v0, u1, v1, shape=(480, 640)):
    mask = np.zeros(shape, dtype=bool)
    mask[v0:v1, u0:u1] = True
    return Instance(score=0.9, mask=mask, box=(u0, v0, u1, v1), area=int(mask.sum()))
