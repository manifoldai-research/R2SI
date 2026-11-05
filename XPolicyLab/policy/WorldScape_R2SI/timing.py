"""Native step timing of a planned joint path: a command step on the joint that travels furthest, uniform or speeding up and slowing down at the ends. The path is the
planner's; only its timing changes."""
from __future__ import annotations

import numpy as np


def steps_at(path, step_rad: float) -> int:
    """Native steps of a path at a joint command step."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0
    return int(np.ceil((np.abs(np.diff(path, axis=0)) / float(step_rad)).max(axis=1).sum()))


def _at_shares(path, shares) -> np.ndarray:
    """Joint targets at shares of the path's furthest-joint distance."""
    path = np.asarray(path, dtype=np.float64)
    segment = np.abs(np.diff(path, axis=0)).max(axis=1)
    source = np.r_[0.0, np.cumsum(segment)]
    keep = np.r_[True, np.diff(source) > 0]
    return np.stack([np.interp(shares * source[-1], source[keep], path[keep, joint]) for joint in range(path.shape[1])], axis=1)


def uniform(path, steps: int) -> np.ndarray:
    """Joint targets at equal shares of the path; the last one is the path's end."""
    return _at_shares(path, np.linspace(0.0, 1.0, steps + 1)[1:])


def ramped(path, steps: int, ramp: int) -> np.ndarray:
    """Joint targets that speed up over the first `ramp` steps and slow down over the last ones: a load in the fingers sees no sudden start or stop."""
    ramp = min(int(ramp), steps // 2)
    speed = np.ones(steps)
    if ramp:
        rise = np.arange(1, ramp + 1) / (ramp + 1)
        speed[:ramp], speed[-ramp:] = rise, rise[::-1]
    shares = np.cumsum(speed) / speed.sum()
    shares[-1] = 1.0
    return _at_shares(path, shares)
