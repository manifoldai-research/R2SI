"""Execution feedback from the official joint state: was a command followed, and did the arm stop or jump?

Tracking deviation is how contact shows up: a blocked arm stops following its commands. The limits are multiples of the robot's
own joint command step.
"""
from __future__ import annotations

import numpy as np

from .frames import ARMS

TRACKING_LIMIT_STEPS = 3.0    # tracking limit per joint, in careful joint command steps
JUMP_LIMIT_STEPS = 6.0        # largest believable joint change per native step, in the same steps
TRACKING_FACTOR = 3.0         # a fast motion may lag this many of its own recent command steps
TRACKING_MEMORY = 4           # recent command steps the scaled limit remembers: lag persists into the deceleration
COMMAND_SANITY_RAD = 1.0      # a command this far from the last one is a programming error, not a motion


class ActionError(RuntimeError):
    """A native action that must not be sent; a programming error, never a model mistake."""


class Monitor:
    def __init__(self, spec):
        self.spec = spec
        self.step_limit = float(spec.tracked_step)
        self.previous = None
        self.recent = {arm: [] for arm in ARMS}

    def begin(self, observation):
        self.previous = {arm: np.asarray(observation.state[arm + "_arm_joint_state"], dtype=float).copy() for arm in ARMS}
        self.recent = {arm: [] for arm in ARMS}

    def before(self, action, observation, internal=False) -> dict:
        """The action as it goes out. An arm pressing against a limit reads just past it, and a command built from that reading would follow: commands stay inside the range."""
        if self.previous is None or internal:
            self.begin(observation)
        sent = dict(action)
        for arm in ARMS:
            joints = np.asarray(action[arm + "_arm_joint_state"], dtype=float)
            opening = np.asarray(action[arm + "_ee_joint_state"], dtype=float)
            if joints.shape != (6,) or opening.shape != (1,) or not np.isfinite(joints).all() or not np.isfinite(opening).all():
                raise ActionError("malformed action for the " + arm + " arm")
            if not 0.0 <= float(opening[0]) <= 1.0:
                raise ActionError("gripper command outside its range on the " + arm + " arm")
            joints = np.clip(joints, self.spec.limit_low, self.spec.limit_high)
            sent[arm + "_arm_joint_state"] = joints.astype(np.float32)
            delta = np.abs(joints - self.previous[arm])
            if np.any(delta > COMMAND_SANITY_RAD):
                raise ActionError(f"command jump of {float(delta.max()):.3f} rad on the {arm} arm")
            self.recent[arm] = self.recent[arm][-(TRACKING_MEMORY - 1):] + [delta]
        return sent

    def largest_jump(self, action, observation, continuing=False) -> float:
        """How far an action would move a joint command, as before reads it, for a caller that did not plan the action and can still stop short of it. continuing: it follows
        the previous action without a new begin."""
        worst = 0.0
        for arm in ARMS:
            start = self.previous[arm] if continuing and self.previous is not None else observation.state[arm + "_arm_joint_state"]
            joints = np.clip(np.asarray(action[arm + "_arm_joint_state"], dtype=float), self.spec.limit_low, self.spec.limit_high)
            worst = max(worst, float(np.max(np.abs(joints - np.asarray(start, dtype=float)))))
        return worst

    def after(self, action, before, after) -> list:
        """Why the running motion must stop, from the official joints after this action."""
        reasons = []
        for arm in ARMS:
            key = arm + "_arm_joint_state"
            target = np.asarray(action[key], dtype=float)
            joints = np.asarray(after.state[key], dtype=float)
            tracking = self.step_limit * TRACKING_LIMIT_STEPS
            if self.recent[arm]:
                tracking = np.maximum(tracking, TRACKING_FACTOR * np.max(self.recent[arm], axis=0))
            if np.any(np.abs(target - joints) > tracking + 1e-6):
                reasons.append("tracking_deviation:" + arm)
            if np.any(np.abs(joints - np.asarray(before.state[key], dtype=float)) > self.step_limit * JUMP_LIMIT_STEPS + 1e-6):
                reasons.append("joint_jump:" + arm)
            self.previous[arm] = target.copy()
        return reasons
