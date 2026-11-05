"""Forward kinematics, straight-line inverse kinematics and collision spheres of one arm, from the public robot files.

The chain comes from the robot description (URDF); the collision spheres come from the robot's cuRobo configuration, and
the fingers, which that sphere model leaves out, get spheres along their span. Nothing here knows about objects.
"""
from __future__ import annotations

import math
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from . import frames
from .robot import CUROBO_CONFIG, URDF, WORKSPACE

CHAIN = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
LINKS = ("link1", "link2", "link3", "link4", "link5", "link6")
FINGER_X = (0.09, 0.115, 0.14, 0.16)   # along flange +x: the fingers' span up to the fingertips
FINGER_R = 0.012
FINGER_Y = 0.0249                      # finger joint origin off the flange axis
FINGER_TRAVEL = 0.044                  # finger joint travel at full opening
MIN_SPHERE_R = 0.008                   # the many tiny spheres of the sphere model add nothing to a clearance check
# Damped least squares on one Cartesian waypoint.
IK_DAMPING = 0.02
IK_ROTATION_WEIGHT = 0.2               # metres per radian: how a rotation error counts against a position error
IK_MAX_ITERATIONS = 60
IK_MAX_STEP_RAD = 0.25
IK_TOLERANCE_M = 0.0005
IK_TOLERANCE_DEG = 0.3


def _origin(joint) -> np.ndarray:
    node = joint.find("origin")
    xyz = [float(v) for v in (node.get("xyz") or "0 0 0").split()]
    rpy = [float(v) for v in (node.get("rpy") or "0 0 0").split()]
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    out[:3, 3] = xyz
    return out


class ArmModel:
    """One arm: base frame at its first joint's parent, joints 1..6, flange at link6."""

    def __init__(self, root: Path | None = None):
        root = Path(root or WORKSPACE)
        urdf = ElementTree.parse(root / URDF).getroot()
        config = yaml.safe_load((root / CUROBO_CONFIG).read_text(encoding="utf-8"))
        spheres = config["robot_cfg"]["kinematics"]["collision_spheres"]
        joints = {j.get("name"): j for j in urdf.findall("joint")}
        self.fixed = [_origin(joints[name]) for name in CHAIN]
        self.axes = [np.array([float(v) for v in joints[name].find("axis").get("xyz").split()]) for name in CHAIN]
        mount = _origin(joints["hand_to_camera_mount"])
        camera = mount @ _origin(joints["camera_joint"])
        self.base_spheres = self._spheres(spheres, "base_link")
        self.link_spheres = [self._spheres(spheres, link) for link in LINKS]
        rows = []
        for key, transform in (("camera_base", mount), ("camera", camera)):
            local = self._spheres(spheres, key)
            if len(local):
                rows.append(np.c_[(transform[:3, :3] @ local[:, :3].T).T + transform[:3, 3], local[:, 3]])
        self.hand_spheres = np.vstack(rows) if rows else np.zeros((0, 4))

    @staticmethod
    def _spheres(spheres, link) -> np.ndarray:
        rows = [[*s["center"], s["radius"]] for s in spheres.get(link) or [] if s["radius"] > MIN_SPHERE_R]
        return np.array(rows, dtype=np.float64).reshape(-1, 4)

    def link_frames(self, joints) -> list[np.ndarray]:
        """Frames (4x4) of link1..link6 in the base frame."""
        out, current = [], np.eye(4)
        for fixed, axis, angle in zip(self.fixed, self.axes, joints):
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
            current = current @ fixed @ turn
            out.append(current)
        return out

    def base(self, joints, flange_pose) -> np.ndarray:
        """World pose of the arm base from the official joints and flange pose."""
        return frames.transform(flange_pose) @ np.linalg.inv(self.link_frames(joints)[-1])

    def flange(self, joints, base) -> np.ndarray:
        """World flange transform at the given joints."""
        return base @ self.link_frames(joints)[-1]

    def jacobian(self, joints, base) -> np.ndarray:
        """Geometric Jacobian (6 x 6) of the flange origin in the world frame."""
        rows = np.zeros((6, 6))
        current = base.copy()
        points, axes = [], []
        for fixed, axis, angle in zip(self.fixed, self.axes, joints):
            current = current @ fixed
            points.append(current[:3, 3].copy())
            axes.append(current[:3, :3] @ axis)
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * float(angle)).as_matrix()
            current = current @ turn
        tip = current[:3, 3]
        for k in range(6):
            rows[:3, k] = np.cross(axes[k], tip - points[k])
            rows[3:, k] = axes[k]
        return rows

    def solve(self, target, seed, base, low, high) -> np.ndarray | None:
        """Joints that put the flange at a world transform, by damped least squares from `seed`; None when they do not
        converge inside [low, high]."""
        q = np.asarray(seed, dtype=np.float64).copy()
        weights = np.array([1.0, 1.0, 1.0, IK_ROTATION_WEIGHT, IK_ROTATION_WEIGHT, IK_ROTATION_WEIGHT])
        for _ in range(IK_MAX_ITERATIONS):
            now = self.flange(q, base)
            error_p = target[:3, 3] - now[:3, 3]
            error_r = Rotation.from_matrix(target[:3, :3] @ now[:3, :3].T).as_rotvec()
            if float(np.linalg.norm(error_p)) < IK_TOLERANCE_M and math.degrees(float(np.linalg.norm(error_r))) < IK_TOLERANCE_DEG:
                return q if np.all(q >= low - 1e-6) and np.all(q <= high + 1e-6) else None
            jac = weights[:, None] * self.jacobian(q, base)
            step = jac.T @ np.linalg.solve(jac @ jac.T + IK_DAMPING ** 2 * np.eye(6), weights * np.r_[error_p, error_r])
            largest = float(np.max(np.abs(step)))
            if largest > IK_MAX_STEP_RAD:
                step *= IK_MAX_STEP_RAD / largest
            q = np.clip(q + step, low, high)
        return None

    def spheres(self, joints, base, opening: float = 1.0) -> np.ndarray:
        """World centres and radii (N x 4) of the arm at the given joints and gripper opening."""
        rows = [(base[:3, :3] @ self.base_spheres[:, :3].T).T + base[:3, 3]]
        radii = [self.base_spheres[:, 3]]
        link_frames = self.link_frames(joints)
        for frame, local in zip(link_frames, self.link_spheres):
            if len(local):
                world = base @ frame
                rows.append((world[:3, :3] @ local[:, :3].T).T + world[:3, 3])
                radii.append(local[:, 3])
        hand = base @ link_frames[-1]
        if len(self.hand_spheres):
            rows.append((hand[:3, :3] @ self.hand_spheres[:, :3].T).T + hand[:3, 3])
            radii.append(self.hand_spheres[:, 3])
        half = FINGER_Y + FINGER_TRAVEL * float(opening)
        fingers = np.array([[x, sign * half, 0.0] for x in FINGER_X for sign in (1.0, -1.0)])
        rows.append((hand[:3, :3] @ fingers.T).T + hand[:3, 3])
        radii.append(np.full(len(fingers), FINGER_R))
        return np.c_[np.vstack(rows), np.concatenate(radii)]

    def lowest_link(self, joints, base) -> float:
        """World height of the lowest point of the upper arm and everything after it. The shoulder stands on the table and does not count."""
        lowest = math.inf
        for frame, local in zip(self.link_frames(joints), self.link_spheres):
            if len(local):
                world = base @ frame
                lowest = min(lowest, float(np.min(((world[:3, :3] @ local[:, :3].T).T + world[:3, 3])[:, 2] - local[:, 3])))
        return lowest

    def hand_clearance(self, joints, base, opening: float = 1.0) -> np.ndarray:
        """World heights of the lowest points of the wrist, the camera and the fingertips at the given joints."""
        link_frames = self.link_frames(joints)
        hand = base @ link_frames[-1]
        heights = []
        for frame, local in ((link_frames[3], self.link_spheres[3]), (link_frames[4], self.link_spheres[4]), (link_frames[5], self.link_spheres[5])):
            if len(local):
                world = base @ frame
                heights.append(((world[:3, :3] @ local[:, :3].T).T + world[:3, 3])[:, 2] - local[:, 3])
        if len(self.hand_spheres):
            heights.append(((hand[:3, :3] @ self.hand_spheres[:, :3].T).T + hand[:3, 3])[:, 2] - self.hand_spheres[:, 3])
        half = FINGER_Y + FINGER_TRAVEL * float(opening)
        tips = np.array([[frames.GRASP_OFFSET_M + frames.FINGERTIP_REACH_M, sign * half, 0.0] for sign in (1.0, -1.0)])
        heights.append(((hand[:3, :3] @ tips.T).T + hand[:3, 3])[:, 2])
        return np.concatenate(heights)


def _stl_points(path: Path) -> np.ndarray:
    """The convex hull vertices of a binary STL mesh: all a silhouette needs."""
    data = path.read_bytes()
    count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
    records = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attribute", "<u2")]), count=count, offset=84)
    points = np.unique(np.round(records["v"].reshape(-1, 3).astype(np.float64), 4), axis=0)
    return points[ConvexHull(points).vertices]


class RobotShape:
    """The robot's outline in images, from the meshes of its description. Each link is drawn as the convex outline of its mesh."""

    def __init__(self, arm_model: ArmModel, root: Path | None = None):
        root = Path(root or WORKSPACE)
        self.model = arm_model
        folder = (root / URDF).parent
        urdf = ElementTree.parse(root / URDF).getroot()
        joints = {j.get("name"): j for j in urdf.findall("joint")}
        self.points = {link: _stl_points(folder / "meshes" / f"{link}.STL") for link in ("base_link", *LINKS, "link7", "link8")}
        self.finger_origin = {"link7": _origin(joints["joint7"]), "link8": _origin(joints["joint8"])}
        self.finger_axis = {"link7": np.array([float(v) for v in joints["joint7"].find("axis").get("xyz").split()]),
                            "link8": np.array([float(v) for v in joints["joint8"].find("axis").get("xyz").split()])}

    def finger_frames(self, flange: np.ndarray, travel: float) -> dict:
        frames_out = {}
        for link, origin in self.finger_origin.items():
            slide = np.eye(4)
            slide[:3, 3] = self.finger_axis[link] * travel
            frames_out[link] = flange @ origin @ slide
        return frames_out

    def hulls(self, joints, base, travel: float, fingers_only: bool = False) -> list:
        """World points of each visible link's outline."""
        link_frames = self.model.link_frames(joints)
        flange = base @ link_frames[-1]
        rows = []
        for link, frame in self.finger_frames(flange, travel).items():
            rows.append((frame[:3, :3] @ self.points[link].T).T + frame[:3, 3])
        if fingers_only:
            return rows
        rows.append((base[:3, :3] @ self.points["base_link"].T).T + base[:3, 3])
        for link, frame in zip(LINKS, link_frames):
            world = base @ frame
            rows.append((world[:3, :3] @ self.points[link].T).T + world[:3, 3])
        return rows


def gap(a, b) -> float:
    """Smallest surface distance between two sphere sets (N x 4), metres (negative: overlapping)."""
    delta = a[:, None, :3] - b[None, :, :3]
    return float(np.min(np.sqrt(np.sum(delta * delta, axis=-1)) - a[:, None, 3] - b[None, :, 3]))
