"""Lock-step hand-over between the evaluator's requests and the agent thread.

The evaluator asks for one action at a time and then reports the new observation. The agent thread thinks, plans and asks
for a chunk of native actions to be executed; it blocks until they have been executed and acknowledged, or until the
execution monitor interrupted them. While the agent has nothing to execute, a request waits up to the keepalive time and
then gets one action that holds the robot still, so a slow model reply cannot trip the evaluator's request limit.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .frames import ARMS


class EpisodeOver(BaseException):
    """Raised in the agent thread when the evaluator has finished this episode."""


@dataclass(frozen=True)
class Observation:
    """What the policy may read: official RGB, robot state, the instruction and the official step counters."""

    images: dict
    state: dict
    instruction: str
    step: int
    step_limit: int

    @property
    def remaining(self) -> int:
        return max(0, self.step_limit - self.step)


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_SIZES = {"left_arm_joint_state": 6, "right_arm_joint_state": 6, "left_ee_joint_state": 1, "right_ee_joint_state": 1,
               "left_ee_pose": 7, "right_ee_pose": 7}


def parse_observation(raw: dict) -> Observation:
    """The whitelist of what enters the policy from an evaluator observation; everything else is dropped here."""
    if not isinstance(raw, dict) or not isinstance(raw.get("state"), dict):
        raise ValueError("expected an official observation dictionary")
    instruction = raw.get("instruction")
    if isinstance(instruction, (list, tuple)) and len(instruction) == 1:
        instruction = instruction[0]
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("the public instruction is missing")
    state = {}
    for key, size in STATE_SIZES.items():
        value = np.asarray(raw["state"].get(key), dtype=np.float64).reshape(-1)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError("invalid robot channel: " + key)
        state[key] = value.copy()
    images = {}
    for camera, view in (raw.get("vision") or {}).items():
        if camera in CAMERAS and isinstance(view, dict) and "color" in view:
            array = np.asarray(view["color"])
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("the policy server must provide decoded RGB arrays")
            images[camera] = array.copy()
    if not images:
        raise ValueError("at least one RGB view is required")
    info = raw.get("additional_info")
    if not isinstance(info, dict) or type(info.get("step")) is not int or type(info.get("step_lim")) is not int:
        raise ValueError("the action count and step limit are missing: run through this policy's deploy.py")
    return Observation(images=images, state=state, instruction=instruction.strip(), step=info["step"], step_limit=info["step_lim"])


@dataclass
class Execution:
    """What happened to a chunk of native actions."""

    planned: int
    executed: int = 0
    reasons: list = field(default_factory=list)   # why the execution monitor stopped the chunk, if it did


class Bridge:
    def __init__(self, monitor, keepalive_s: float):
        self.monitor = monitor
        self.keepalive_s = float(keepalive_s)
        self._cv = threading.Condition()
        self._queue = deque()
        self._current = None           # the Execution in progress
        self._contact = False          # the running chunk expects to be stopped by contact
        self._pending = None           # the action sent, not yet acknowledged
        self._pending_internal = False
        self.observation = None
        self.grippers = {}             # latched gripper commands: re-sending a reading would relax a grip
        self._closed = False
        self._failure = None
        self.keepalive_holds = 0
        self.keepalive_unreported = 0
        self.interruptions = 0
        self.contact_stops = 0

    # -- evaluator side ---------------------------------------------------------------------------
    def update_obs(self, raw: dict):
        with self._cv:
            executed = self._pending is not None
            previous = self.observation
            observation = parse_observation(raw)
            self.observation = observation
            if executed:
                action, internal, self._pending, self._pending_internal = self._pending, self._pending_internal, None, False
                reasons = self.monitor.after(action, previous, observation)
                if not internal and self._current is not None:
                    self._current.executed += 1
                    if reasons:
                        self._current.reasons = reasons
                        self._queue.clear()
                        if self._contact and all(r.startswith("tracking_deviation") for r in reasons):
                            self.contact_stops += 1
                        else:
                            self.interruptions += 1
                    if not self._queue:
                        self._current, self._contact = None, False
            self._cv.notify_all()

    def get_action(self) -> dict:
        """The next native action: the next one of the running chunk, or a hold when the agent is still deciding."""
        with self._cv:
            if self._failure is not None:
                raise self._failure
            if self.observation is None or self._pending is not None:
                raise RuntimeError("a fresh observation is required before an action")
            if not self._queue:
                self._cv.wait_for(lambda: self._queue or self._failure is not None or self._closed, self.keepalive_s)
                if self._failure is not None:
                    raise self._failure
            if self._queue:
                action, internal = self._queue.popleft(), False
            else:
                action, internal = self.hold_action(self.observation), True
                self.keepalive_holds += 1
                self.keepalive_unreported += 1
            action = self.monitor.before(action, self.observation, internal=internal)
            self._pending, self._pending_internal = action, internal
            return action

    def fail(self, error: BaseException):
        """The agent thread stopped with an error: the next evaluator request reports it."""
        with self._cv:
            self._failure = error
            self._cv.notify_all()

    def close(self):
        with self._cv:
            self._closed = True
            self._queue.clear()
            self._cv.notify_all()

    # -- agent side -------------------------------------------------------------------------------
    def wait_observation(self) -> Observation:
        """The first observation of the episode."""
        with self._cv:
            self._cv.wait_for(lambda: self.observation is not None or self._closed)
            if self._closed:
                raise EpisodeOver()
            return self.observation

    def current(self) -> Observation:
        with self._cv:
            if self._closed:
                raise EpisodeOver()
            return self.observation

    def execute(self, actions, contact: bool = False, continuing: bool = False) -> Execution:
        """Hand native actions to the evaluator and block until they ran or were stopped. continuing: these go on from the previous call, step by step."""
        if not actions:
            raise ValueError("no native actions to execute")
        with self._cv:
            if self._closed:
                raise EpisodeOver()
            run = Execution(planned=len(actions))
            if not continuing:
                self.monitor.begin(self.observation)
            self._current, self._contact = run, contact
            self._queue.extend(actions)
            self._cv.notify_all()
            self._cv.wait_for(lambda: self._current is not run or self._closed)
            if self._closed:
                raise EpisodeOver()
            return run

    def hold_action(self, observation: Observation) -> dict:
        """Keep the arms where they are and the jaws on what they were last told.

        The official gripper reading follows the previous command after the simulator limits it to a fifth of the range ahead
        of the fingers; sending that reading back would park the jaw at its current width, drop the grip force and let a held
        object slide. Sending the latched command keeps squeezing.
        """
        action = {}
        for arm in ARMS:
            action[arm + "_arm_joint_state"] = np.asarray(observation.state[arm + "_arm_joint_state"], dtype=np.float32).copy()
            goal = self.grippers.get(arm)
            reading = observation.state[arm + "_ee_joint_state"]
            action[arm + "_ee_joint_state"] = np.asarray(reading if goal is None else [goal], dtype=np.float32).copy()
        return action
