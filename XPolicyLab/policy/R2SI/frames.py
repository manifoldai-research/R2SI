"""World-frame conventions of the two-arm rig: wxyz quaternions, flange poses and grasp points.

A flange pose is ``[x, y, z, qw, qx, qy, qz]`` as the official state reports it. The agent works with the
grasp point, the centre of the gripping faces, which sits on the flange's own +x axis. The distance comes from
the robot description: the gripping face of the finger link spans 0.1426 to 0.1576 m from the flange.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

ARMS = ("left", "right")

#: Distance from the flange to the centre of the gripping faces, along flange +x.
GRASP_OFFSET_M = 0.1501
#: Depth of the gripping faces along the pointing axis, centred on the grasp point.
PAD_DEPTH_M = 0.015
#: How far from its base an arm can bring the grasp point, along the table with a level hand ...
ARM_REACH_M = 0.78
#: ... and how high above the table it can still look down on something.
ARM_TOP_M = 0.44
#: How far the fingertips reach past the grasp point along the pointing axis.
FINGERTIP_REACH_M = 0.0075
#: Largest distance between the gripping faces at full opening.
MAX_JAW_WIDTH_M = 0.088
#: The table surface the arms are mounted on.
TABLE_Z = 0.765
#: Flange orientation with the gripper pointing straight down, closing along world x.
DOWN = np.array([0.5, -0.5, 0.5, 0.5])
#: Arm base origins in the world, from the rig's mounting.
BASE_ORIGIN = {"left": np.array([-0.300, -0.450, 0.765]), "right": np.array([0.300, -0.450, 0.765])}
UP = np.array([0.0, 0.0, 1.0])
#: A parallel gripper turned half a turn about its pointing axis holds the same grasp point.
HALF_TURN = np.array([0.0, 1.0, 0.0, 0.0])


def normalized(q) -> np.ndarray:
    """Unit quaternion with non-negative w."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("a quaternion needs four finite numbers")
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("a quaternion must have non-zero norm")
    q = q / norm
    return q if q[0] >= 0 else -q


def multiply(left, right) -> np.ndarray:
    """Compose two wxyz quaternions (left applied after right)."""
    w0, x0, y0, z0 = (float(v) for v in left)
    w1, x1, y1, z1 = (float(v) for v in right)
    return np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])


def matrix(q) -> np.ndarray:
    """Rotation matrix of a wxyz quaternion; its columns are the body axes."""
    w, x, y, z = normalized(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def from_axes(pointing, closing) -> np.ndarray:
    """Flange orientation whose +x is `pointing` and whose +y (the closing axis) lies as close to `closing` as possible."""
    x = np.asarray(pointing, dtype=np.float64)
    x = x / np.linalg.norm(x)
    y = np.asarray(closing, dtype=np.float64)
    y = y - float(y @ x) * x
    y = y / np.linalg.norm(y)
    m = np.c_[x, y, np.cross(x, y)]
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w > 1e-4:
        return normalized([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])
    k = int(np.argmax(np.diag(m)))
    i, j = (k + 1) % 3, (k + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + m[k, k] - m[i, i] - m[j, j])) * 2.0
    q = np.zeros(4)
    q[0] = (m[j, i] - m[i, j]) / s
    q[k + 1] = s / 4.0
    q[i + 1] = (m[i, k] + m[k, i]) / s
    q[j + 1] = (m[j, k] + m[k, j]) / s
    return normalized(q)


def about(axis, radians: float) -> np.ndarray:
    """Quaternion of a rotation about a unit axis."""
    return np.r_[math.cos(radians / 2), math.sin(radians / 2) * np.asarray(axis, dtype=np.float64)]


def turned(q, axis, degrees: float) -> np.ndarray:
    """q turned about a world axis."""
    return normalized(multiply(about(axis, math.radians(degrees)), q))


def tilted(q, degrees: float) -> np.ndarray:
    """q turned about its own closing axis, so the jaw still closes along the same line."""
    return turned(q, matrix(q)[:, 1], degrees)


def half_turned(q) -> np.ndarray:
    """The same gripper orientation turned half a turn about its pointing axis."""
    return normalized(multiply(q, HALF_TURN))


def pointing_axis(q) -> np.ndarray:
    """World direction the gripper points along."""
    return matrix(q)[:, 0]


def closing_axis(q) -> np.ndarray:
    """World direction along which the fingers close."""
    return matrix(q)[:, 1]


def angle_deg(a, b) -> float:
    """Angle between two orientations."""
    dot = abs(float(np.dot(normalized(a), normalized(b))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def slerp(a, b, t: float) -> np.ndarray:
    a, b = normalized(a), normalized(b)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return normalized(a + t * (b - a))
    theta = math.acos(min(1.0, dot))
    return normalized((math.sin((1 - t) * theta) * a + math.sin(t * theta) * b) / math.sin(theta))


def flange_pose(position, q) -> np.ndarray:
    """Flange pose that puts the grasp point at `position` with orientation q."""
    q = normalized(q)
    return np.r_[np.asarray(position, dtype=np.float64) - GRASP_OFFSET_M * pointing_axis(q), q]


def pose_of(transform_matrix) -> np.ndarray:
    """A 4 x 4 world transform as position and wxyz quaternion."""
    x, y, z, w = Rotation.from_matrix(np.asarray(transform_matrix)[:3, :3]).as_quat()
    return np.r_[np.asarray(transform_matrix)[:3, 3], w, x, y, z]


def grasp_point(flange) -> tuple[np.ndarray, np.ndarray]:
    """(grasp point, orientation) of a flange pose."""
    flange = np.asarray(flange, dtype=np.float64).reshape(-1)
    q = normalized(flange[3:7])
    return flange[:3] + GRASP_OFFSET_M * pointing_axis(q), q


def transform(flange) -> np.ndarray:
    """4x4 world transform of a flange pose."""
    flange = np.asarray(flange, dtype=np.float64).reshape(-1)
    out = np.eye(4)
    out[:3, :3] = matrix(flange[3:7])
    out[:3, 3] = flange[:3]
    return out


def jaw_across(long_side_rad: float, reference=DOWN) -> np.ndarray:
    """A straight-down orientation that closes across a horizontal long side at `long_side_rad`, the one nearest `reference`."""
    wanted = long_side_rad + math.pi / 2
    closing = closing_axis(reference)
    turn = (wanted - math.atan2(float(closing[1]), float(closing[0]))) % math.pi
    if turn > math.pi / 2:
        turn -= math.pi
    return normalized(multiply(about(UP, turn), reference))


def leaning(arm: str, point, degrees: float) -> np.ndarray:
    """The hand at `point` leaning away from the arm's base by `degrees` from straight down, closing horizontally across
    that direction: the wrist comes back toward the base, which lets an arm reach further."""
    away = np.asarray(point, dtype=np.float64)[:2] - BASE_ORIGIN[arm][:2]
    heading = math.atan2(float(away[1]), float(away[0]))
    t = math.radians(degrees)
    pointing = np.array([math.sin(t) * math.cos(heading), math.sin(t) * math.sin(heading), -math.cos(t)])
    return from_axes(pointing, [-math.sin(heading), math.cos(heading), 0.0])
