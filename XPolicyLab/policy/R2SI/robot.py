"""Robot limits and control cadence. The joint limits come from the public robot description; the rest are constants of this robot and its controller."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

#: The workspace that holds the benchmark's public files: <root>/XPolicyLab/policy/<POLICY>/robot.py.
WORKSPACE = Path(os.environ.get("ROBODOJO_ROOT") or Path(__file__).resolve().parents[3]).expanduser().resolve()
URDF = "Assets/Robots/x5/X5A.urdf"
CUROBO_CONFIG = "Assets/Robots/x5/curobo.yml"
ROBOT_CONFIG = "Assets/Robots/x5/robot_config.yml"
#: Joint command per native step the drive follows without visible lag when no speed is needed.
TRACKED_STEP_RAD = 0.05
#: Joint command per native step of a free motion with empty hands.
FREE_STEP_RAD = 0.18
#: Joint command per native step of a free motion carrying something.
CARRY_STEP_RAD = 0.15
CARRY_RAMP_STEPS = 3         # a load starts and stops over this many steps, so it sees no sudden change of speed
GRIPPER_STEP = 0.25          # gripper command change per native step that the fingers follow
#: The range the joints are planned in. The robot description allows far more, but the links meet each other and the table long before that, and a plan past that looks fine
#: while the arm jams. Joint 1 stays within about half a turn, and joint 3 between folded and straight, so that the way back never leads around the wrong side. Joint 2 stays above
#: where the upper arm meets the base, and joints 4 and 5 a little inside their stops, found by driving the arm against them. Joint 6 has the limits of the description.
WORK_LOW = (-3.3, -0.1, -0.1, -2.19, -1.5, -3.14)
WORK_HIGH = (3.3, 3.3, 3.2, 1.72, 1.5, 3.14)
#: How far joint 4 folds the wrist back before it meets the forearm, as (how far joint 5 is turned either way, joint 4).
WRIST_FOLD = ((0.0, -2.19), (0.25, -2.15), (0.5, -2.115), (0.75, -2.11), (1.0, -1.95), (1.25, -2.0), (1.5, -2.18))


@dataclass(frozen=True)
class RobotSpec:
    """Limits of one arm joint set, the same for both arms."""

    low: np.ndarray                 # lowest value each of the 6 joints is planned to
    high: np.ndarray                # highest value each of the 6 joints is planned to
    limit_low: np.ndarray           # the robot description's joint limits: no command goes past them
    limit_high: np.ndarray
    tracked_step: float = TRACKED_STEP_RAD
    gripper_step: float = GRIPPER_STEP
    free_step: float = FREE_STEP_RAD
    carry_step: float = CARRY_STEP_RAD
    carry_ramp: int = CARRY_RAMP_STEPS


def _arm_joints(root: Path):
    urdf = ElementTree.parse(root / URDF).getroot()
    rows = []
    for joint in urdf.findall("joint"):
        name = joint.get("name", "")
        if name in {f"joint{i}" for i in range(1, 7)}:
            limit = joint.find("limit")
            rows.append((int(name[-1]), float(limit.get("lower")), float(limit.get("upper"))))
    rows.sort()
    if len(rows) != 6:
        raise RuntimeError("expected six arm joints in the robot description")
    return np.array([r[1] for r in rows]), np.array([r[2] for r in rows])


def wrist_folded(joints) -> bool:
    """Whether these joints fold the wrist back onto the forearm."""
    return float(joints[3]) < float(np.interp(abs(float(joints[4])), [row[0] for row in WRIST_FOLD], [row[1] for row in WRIST_FOLD]))


def load_spec(root: Path | None = None) -> RobotSpec:
    low, high = _arm_joints(Path(root or WORKSPACE))
    return RobotSpec(low=np.maximum(low, WORK_LOW), high=np.minimum(high, WORK_HIGH), limit_low=low, limit_high=high)
