"""What the agent's own measurements say stands on the table: one box per object from robo.locate. Free motions pass over these boxes. It holds nothing
that did not come from the images, and it knows nothing about what has not been measured or has moved since.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import frames

MIN_HEIGHT_M = frames.PAD_DEPTH_M    # flatter than the fingers' pads are deep is no obstacle for a hand above the surface ...
MAX_SIDE_M = frames.ARM_REACH_M     # ... and a footprint longer than an arm's reach is background, not one object
MIN_SCORE = 0.3              # a weaker match is too likely not to be there
MARGIN_M = 0.005             # how close the robot's sphere model may pass a box
LIKE_SIZE = 1.6              # footprints further apart in size than this are different things, one standing in or on the other
AROUND_M = 0.01              # a grasp point this close to a box belongs to it


@dataclass
class Box:
    id: int
    text: str
    center: np.ndarray       # world x, y
    half: np.ndarray         # half length, half width
    angle: float             # direction of the length from +x, radians
    bottom: float
    top: float

    def _local(self, xy) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (np.asarray(xy, dtype=np.float64) - self.center) @ np.array([[c, -s], [s, c]])

    def holds(self, xy) -> bool:
        """Whether a world x, y lies over the footprint."""
        return bool(np.all(np.abs(self._local(xy)) <= self.half))

    def touches(self, bodies) -> bool:
        """Whether any of the spheres (N x 4: centre, radius) comes within the margin of the box."""
        side = np.maximum(np.abs(self._local(bodies[:, :2])) - self.half, 0.0)
        up = np.maximum(np.maximum(self.bottom - bodies[:, 2], bodies[:, 2] - self.top), 0.0)
        return bool(np.any(np.sqrt(np.sum(side ** 2, axis=1) + up ** 2) < bodies[:, 3] + MARGIN_M))


def _box(item: dict) -> Box | None:
    length = max(float(item.get("length_m", 0.0)), float(item.get("body_length_m", 0.0)))
    width = max(float(item.get("width_m", 0.0)), float(item.get("body_width_m", 0.0)))
    top, bottom = float(item["top_z"]), float(item["bottom_z"])
    # One camera alone does not fix where a thing stands: such a box would stop motions for nothing.
    if item.get("agreement") is None or top - bottom < MIN_HEIGHT_M or max(length, width) > MAX_SIDE_M or float(item.get("score", 1.0)) < MIN_SCORE:
        return None
    return Box(int(item["id"]), str(item.get("text", "")), np.array([float(item["x"]), float(item["y"])]), np.array([length / 2, width / 2]),
               math.radians(float(item.get("long_side_deg", 0.0))), bottom, top)


def _same(a: Box, b: Box) -> bool:
    if not (a.holds(b.center) or b.holds(a.center)):
        return False
    if max(a.half) > LIKE_SIZE * max(b.half) or max(b.half) > LIKE_SIZE * max(a.half):
        return False
    return min(a.top, b.top) - max(a.bottom, b.bottom) > 0.5 * min(a.top - a.bottom, b.top - b.bottom)


class Scene:
    """The latest box of each measured object of one episode."""

    def __init__(self):
        self.boxes = {}

    def note(self, items: list, text: str | None = None, keep_out=()):
        """Take in the results of one measurement, best first. They replace what was known under the same phrase and whatever stands in the same place; among
        themselves, a weaker result of the same thing only adds its extent. `keep_out` lists world x, y points held by a closed hand: what is measured there is being carried."""
        if text is not None:
            self.boxes = {key: box for key, box in self.boxes.items() if box.text != text}
        fresh = []
        for item in items:
            box = _box(item)
            if box is None or any(box.holds(point) for point in keep_out):
                continue
            twin = next((other for other in fresh if _same(other, box)), None)
            if twin is None:
                fresh.append(box)
            else:
                twin.top, twin.bottom = max(twin.top, box.top), min(twin.bottom, box.bottom)
                twin.half = np.maximum(twin.half, box.half)
        for box in fresh:
            self.boxes = {key: old for key, old in self.boxes.items() if not _same(old, box)}
            self.boxes[box.id] = box

    def put_down(self, key, low, high, bottom, top):
        """What a hand has just put down, between these world corners, until it is measured."""
        box = Box(key, f"placed {key}", (np.asarray(low) + high) / 2, (np.asarray(high) - low) / 2, 0.0, bottom, top)
        if top - bottom >= MIN_HEIGHT_M and 2 * float(np.max(box.half)) <= MAX_SIDE_M:
            self.boxes = {other: old for other, old in self.boxes.items() if not _same(old, box)}
            self.boxes[key] = box

    def forget(self, key=None, point=None):
        """Drop what a hand took away: the box with this id, or else the smallest box over this world x, y."""
        if key in self.boxes:
            del self.boxes[key]
        elif point is not None:
            over = [k for k, box in self.boxes.items() if box.holds(point)]
            if over:
                del self.boxes[min(over, key=lambda k: float(np.prod(self.boxes[k].half)))]

    def clear(self):
        """Everything may have moved, and nothing is known until it is measured again."""
        self.boxes = {}

    def around(self, point) -> dict | None:
        """The smallest measured object around a world grasp point, in the form robo.locate gives it, or None."""
        rows = [box for box in self.boxes.values() if np.all(np.abs(box._local(point[:2])) <= box.half + AROUND_M) and box.bottom - AROUND_M <= point[2] <= box.top + AROUND_M]
        if not rows:
            return None
        box = min(rows, key=lambda row: float(np.prod(row.half)))
        return {"id": box.id, "text": box.text, "x": float(box.center[0]), "y": float(box.center[1]), "top_z": box.top, "bottom_z": box.bottom,
                "length_m": 2 * float(box.half[0]), "width_m": 2 * float(box.half[1]), "long_side_deg": math.degrees(box.angle)}

    def obstacles(self) -> list:
        return list(self.boxes.values())
