"""Camera geometry from the public configuration: intrinsics, the fixed head camera and the wrist cameras on their flanges.

Camera frames follow the renderer: x right, y up, looking along -z. Pixels are (u, v) = (column, row) of the full image.
Every constant comes from public files of the benchmark: env_cfg/camera/template.py (intrinsics), env_cfg/camera/camera_config.yml
(head camera pose), Assets/Robots/x5/robot_config.yml and X5A.urdf (wrist camera mount).
"""
from __future__ import annotations

import ast
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from . import frames
from .robot import ROBOT_CONFIG, URDF, WORKSPACE

STAND_BEHIND_M = 0.10        # the stand of the head camera is taken to stand within this distance behind it ...
STAND_HALF_WIDTH_M = 0.03    # ... and this far to either side
MIN_CROSSING = 0.05          # sine of the angle between two rays below which depth is unresolved


def _matrix(position, rotation) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.asarray(position, dtype=np.float64)
    return out


def _templates(text: str) -> dict:
    """Camera templates (dict literals) of the configuration file, without executing it."""
    found = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                value = ast.literal_eval(node.value)
            except ValueError:
                continue
            if isinstance(value, dict):
                found[node.targets[0].id.lower().replace("_", "")] = value
    return found


def _intrinsics(template) -> tuple[np.ndarray, tuple[int, int]]:
    width, height = template["resolution"]
    focal = width * float(template["focal_length"]) / float(template["horizontal_aperture"])
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]]), (int(width), int(height))


def _fixed_chain(urdf, parent: str, child: str) -> np.ndarray:
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    out, link = np.eye(4), child
    while link != parent:
        joint = joints[link]
        if joint.get("type") != "fixed":
            raise RuntimeError("the camera link is not fixed to the flange in the robot description")
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") or "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy") or "0 0 0").split()]
        out = _matrix(xyz, Rotation.from_euler("xyz", rpy).as_matrix()) @ out
        link = joint.find("parent").get("link")
    return out


class Rig:
    """Public camera constants of the two-arm rig."""

    def __init__(self, root: Path | None = None):
        root = Path(root or WORKSPACE)
        templates = _templates((root / "env_cfg/camera/template.py").read_text(encoding="utf-8"))
        head = yaml.safe_load((root / "env_cfg/camera/camera_config.yml").read_text(encoding="utf-8"))["cam_head"]["camera"]
        robot = yaml.safe_load((root / ROBOT_CONFIG).read_text(encoding="utf-8"))
        wrist = next(c for c in robot["camera"] if str(c.get("name", "")).endswith("_wrist"))
        urdf = ElementTree.parse(root / URDF).getroot()
        self.head_k, self.head_size = _intrinsics(templates[str(head["type"]).lower().replace("_", "")])
        self.head_pose = _matrix(head["pos"], Rotation.from_euler("XYZ", head.get("ori", [0, 0, 0]), degrees=True).as_matrix())
        self.wrist_k, self.wrist_size = _intrinsics(templates[str(wrist["type"]).lower().replace("_", "")])
        local = _matrix(wrist.get("pos", [0, 0, 0]), Rotation.from_euler("XYZ", wrist.get("ori", [0, 0, 0]), degrees=True).as_matrix())
        self.wrist_mount = _fixed_chain(urdf, "link6", str(wrist["link"])) @ local

    def head_stand(self) -> tuple:
        """Where what carries the head camera may stand: below the camera and behind it, out of its view. (centre x, y; half length, half width; direction of the length; top)."""
        looking = -self.head_pose[:3, 2][:2]
        looking = looking / np.linalg.norm(looking)
        center = self.head_pose[:2, 3] - looking * STAND_BEHIND_M / 2
        return center, np.array([STAND_BEHIND_M / 2, STAND_HALF_WIDTH_M]), float(np.arctan2(looking[1], looking[0])), float(self.head_pose[2, 3])

    def view(self, camera: str, state: dict) -> "View":
        if camera == "cam_head":
            return View(camera, self.head_k, self.head_pose, self.head_size)
        if camera in ("cam_left_wrist", "cam_right_wrist"):
            arm = "left" if camera == "cam_left_wrist" else "right"
            return View(camera, self.wrist_k, frames.transform(state[arm + "_ee_pose"]) @ self.wrist_mount, self.wrist_size)
        raise ValueError("camera must be cam_head, cam_left_wrist or cam_right_wrist")


class View:
    """One camera at one moment."""

    def __init__(self, camera, k, pose, size):
        self.camera, self.k, self.pose, self.size = camera, k, pose, size
        self.origin = pose[:3, 3]
        self.rotation = pose[:3, :3]

    def rays(self, pixels) -> np.ndarray:
        """Unit world directions through full-image pixels (N x 2)."""
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        local = np.stack([(pixels[:, 0] - self.k[0, 2]) / self.k[0, 0], -(pixels[:, 1] - self.k[1, 2]) / self.k[1, 1], -np.ones(len(pixels))], axis=1)
        rays = local @ self.rotation.T
        return rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def ray(self, pixel) -> tuple[np.ndarray, np.ndarray]:
        return self.origin.copy(), self.rays(pixel)[0]

    def project(self, points) -> tuple[np.ndarray, np.ndarray]:
        """(pixels N x 2, depths N) of world points; a point behind the camera has depth <= 0."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        local = (points - self.origin) @ self.rotation
        depth = -local[:, 2]
        safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
        return np.stack([self.k[0, 0] * local[:, 0] / safe + self.k[0, 2], -self.k[1, 1] * local[:, 1] / safe + self.k[1, 2]], axis=1), depth

    def on_plane(self, pixels, z: float) -> np.ndarray:
        """World points where rays through pixels meet the plane at height z; rays that do not meet it in front give nan."""
        rays = self.rays(pixels)
        out = np.full((len(rays), 3), np.nan)
        safe = np.abs(rays[:, 2]) > 1e-6
        t = np.full(len(rays), -1.0)
        t[safe] = (float(z) - self.origin[2]) / rays[safe, 2]
        keep = t > 0
        out[keep] = self.origin + t[keep, None] * rays[keep]
        return out


def triangulate(rays) -> tuple[np.ndarray | None, float | None, float]:
    """Least-squares point closest to every ray (origin, direction): (point, rms distance to the rays, largest crossing sine)."""
    a, b = np.zeros((3, 3)), np.zeros(3)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        a += projector
        b += projector @ origin
    crossing = max(float(np.linalg.norm(np.cross(d1, d2))) for i, (_, d1) in enumerate(rays) for _, d2 in rays[i + 1:])
    if crossing < MIN_CROSSING:
        return None, None, crossing
    point = np.linalg.solve(a, b)
    gaps = [float(np.linalg.norm(np.cross(point - o, d))) for o, d in rays]
    return point, float(np.sqrt(np.mean(np.square(gaps)))), crossing
