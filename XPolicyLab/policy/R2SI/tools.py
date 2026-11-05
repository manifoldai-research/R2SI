"""The agent-thread side of `robo`: what each method does. Nothing here reads anything but the current official observation, the
public robot and camera files, the configured frozen ICL reference and the results of this episode's own actions."""
from __future__ import annotations

import inspect
import math

import numpy as np

from . import actions, cameras, control, frames, monitor, objects, observe, wam
from .bridge import CAMERAS
from .client import InfrastructureFailure
from .frames import ARMS, UP

MIN_SCORE = 0.15              # phrases for small things in a wide image score low; the score is reported, and the agent looks
RANGE_HEIGHT_M = 0.05         # pointing error assumed when a plane height is itself uncertain
PIXEL_ERROR_PX = 2.0
LOOK_CLEARANCE_M = 0.10       # a pick that will look again keeps the wrist camera this far above the grasp point
LOOK_NEAR_M = 0.05            # a fresh measurement this close horizontally is the same object
GRASP_MIN_ABOVE_M = 0.009     # the lowest grasp point above a support: the fingertips reach past the grasp point and stop just above it
GRASP_SHARE = 0.25            # a straight-down grasp sits this share of the object's height below its top ...
GRASP_DEPTH_MIN_M = 0.012     # ... at least this far ...
GRASP_DEPTH_MAX_M = 0.04      # ... and at most this far
RELEASE_MARGIN_M = 0.004      # a placement on a known support releases this far above it
ROUND_ASPECT = 1.25           # below this aspect a footprint gives no jaw direction
RAY_HEIGHTS_M = (0.005, 0.02, 0.05, 0.10, 0.18)   # heights above the surface sampled along a viewing ray
STEEP_POINTING = 0.7          # a hand pointing this much downward grasps from above
SIDE_GRASP_SHARE = 0.55       # a hand coming from the side holds this far up the object
SIDE_APPROACH_MARGIN_M = 0.02    # the open fingertips stop this far from the object before a straight approach from the side
PLACED_NEAR_M = 0.10          # what is measured this close to where a hand let go is taken to be what it let go
REACH_TURNS_DEG = (0, 90, 45, 135)
REACH_LEANS_DEG = (15, 30, 45, 60, 75, 90)
REACH_MAX_POINTS = 4
REACH_MAX_OPTIONS = 6
ESTIMATE_MAX_SEGMENTS = 12
STALE_HOLD_READING = 0.1      # the gripper reading moved this much since a held point was measured: the hold changed
VIEW_DISTANCE_M = 0.2         # how far a wrist camera looks at a point from, unless asked otherwise ...
VIEW_MIN_M = 0.1              # ... the fingertips reach about this far ahead of it ...
VIEW_MAX_M = 0.6
VIEW_ELEVATIONS_DEG = (90, 75, 60, 45, 30)      # from above, then ever lower ...
VIEW_SIDES_DEG = (0, 45, -45, 90, -90)          # ... starting on the side of the arm's base ...
VIEW_ROLLS_DEG = (0, 90, 180, 270)              # ... with the camera turned any way about what it looks at
VIEW_NEAR_DEG = ((0, 0), (15, 0), (-15, 0), (0, 25), (0, -25), (15, 25), (15, -25))   # a view asked from one side may differ this much in height and around
AT_PIXEL = "object at pixel"  # the text of what robo.locate_at finds: no phrase that could find it again
POLICY_MAX_CHUNKS = 8         # one call hands the arms to the learned policy for at most this many chunks; then the model looks again


class Failed(Exception):
    """A goal that did not succeed and was asked to raise; its result goes to the cell."""

    def __init__(self, message, result):
        super().__init__(message)
        self.result = result


def grasp_height(top: float, bottom: float) -> float:
    """Where a straight-down grasp point goes on an object between these heights: the pads on its upper part, the fingertips off its support."""
    depth = min(GRASP_DEPTH_MAX_M, max(GRASP_DEPTH_MIN_M, GRASP_SHARE * (top - bottom)))
    return max(bottom + GRASP_MIN_ABOVE_M, top - depth)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return round(value, 5)
    return value


def _rounded(values, digits=4):
    return np.round(np.asarray(values, dtype=np.float64), digits).tolist()


class Tools:
    """One episode's `robo` implementation."""

    def __init__(self, ctl, rig, shape, segmenter, policy_service=None):
        self.ctl = ctl
        self.bridge = ctl.bridge
        self.rig = rig
        self.shape = shape
        self.segmenter = segmenter
        self.runner = actions.Runner(ctl)
        self.ask_fresh = None       # (observation, question) -> an answer that knows nothing of this episode's conversation
        self.reference_info = lambda: {"status": "disabled"}
        self.attachments = []       # message parts for the next model message
        self.calls = []             # what this cell's robo calls did, for its summary
        self.instances = {}         # id -> (camera, Instance) of the latest segment and locate results
        self.next_id = 1
        self.held_points = {arm: {} for arm in ARMS}
        self.grasped = {arm: None for arm in ARMS}   # what the last pick held: the part below and above the grasp point
        self.policy_service = policy_service         # where the learned policy runs
        self.open_policy = wam.Session
        self.read_service = wam.service_info
        self.policy = None          # the policy's session while nothing else has moved the robot since its last action
        self.policy_step = None
        self.policy_text = None
        self.counters = {"locate_calls": 0, "segment_calls": 0, "picks": 0, "picks_held": 0, "places": 0, "places_ok": 0, "strokes": 0, "turns": 0,
                         "both_calls": 0, "homes": 0, "watches": 0, "goal_failures": 0, "pick_refined": 0, "reach_queries": 0, "estimates": 0,
                         "policy_calls": 0, "policy_sessions": 0, "policy_chunks": 0, "policy_steps": 0, "policy_lagging_steps": 0, "policy_wait_s": 0.0, "policy_predict_s": 0.0, "policy_observe_s": 0.0,
                         "policy_config": None}

    @property
    def observation(self):
        return self.bridge.current()

    # -- dispatch -----------------------------------------------------------------------------------
    def dispatch(self, method: str, arguments: dict):
        """Run one robo call. Returns its result; raises Failed for a failed goal that should raise, ValueError for misuse."""
        handler = getattr(self, "do_" + method, None)
        if handler is None:
            raise ValueError("unknown robo method: " + method)
        before = self.observation.step
        check = arguments.pop("check", None)
        result = handler(**arguments)
        steps = self.observation.step - before
        if steps and method != "policy":
            self.end_policy()
        if isinstance(result, dict) and "ok" in result:
            result = _plain(result)
            result["steps"] = steps
            result["steps_left"] = self.observation.remaining
            self.calls.append({"call": method, "ok": result["ok"], "steps": steps, **({"failure": result["failure"]} if result.get("failure") else {})})
            if not result["ok"]:
                self.counters["goal_failures"] += 1
                if check is None or check:
                    raise Failed(f"robo.{method} failed: {result.get('failure')}: {result.get('reason')}", result)
        elif steps:
            self.calls.append({"call": method, "steps": steps})
        return result

    # -- reading ------------------------------------------------------------------------------------
    def home_steps(self, observation) -> int:
        total_arm, total_grip = 0, 0
        for arm in ARMS:
            gap = float(np.max(np.abs(self.ctl.joints(observation, arm) - self.ctl.joints_initial(arm))))
            if gap > 1e-3:
                total_arm = max(total_arm, int(math.ceil(gap / self.ctl.spec.free_step)))
            change = abs(float(self.ctl.initial[arm + "_ee_joint_state"][0]) - self.ctl.reading(observation, arm))
            total_grip = max(total_grip, int(math.ceil(change / self.ctl.spec.gripper_step)) if change > 0.02 else 0)
        return total_arm + total_grip

    def arm_state(self, observation, arm) -> dict:
        p, q = self.ctl.grasp(observation, arm)
        reading = self.ctl.reading(observation, arm)
        return {"position": _rounded(p), "quaternion_wxyz": _rounded(q), "opening": round(reading, 4), "width_m": round(self.ctl.jaw_width(observation, arm), 4),
                "joints": _rounded(self.ctl.joints(observation, arm))}

    def do_state(self):
        observation = self.observation
        return {"arms": {arm: self.arm_state(observation, arm) for arm in ARMS}, "steps_used": observation.step, "step_limit": observation.step_limit,
                "steps_left": observation.remaining, "home_steps": self.home_steps(observation)}

    def do_reference(self):
        return self.reference_info()

    def do_image(self, camera="cam_head"):
        return self._image(camera).copy()

    def _image(self, camera):
        observation = self.observation
        if camera not in observation.images:
            raise ValueError("camera must be one of " + ", ".join(sorted(observation.images)))
        return observation.images[camera]

    def do_show(self, camera="cam_head", region=None, grid=False, points=None, objects=None, outline=None, rotate_deg=None):
        image = self._image(camera)
        outlines = []
        for item in objects or []:
            found = self._instance_mask(item, camera)
            if found is not None:
                outlines.append((item.get("id", "?") if isinstance(item, dict) else item, found))
        picture, mapping = observe.annotated(image, region=region, grid=bool(grid), points=points, outlines=outlines, outline=outline)
        if rotate_deg is not None:
            picture = observe.rotated(picture, float(rotate_deg))
            mapping = {"rotated_deg": float(rotate_deg)}
        self.attachments += observe.jpeg_part(picture, f"{camera} at step {self.observation.step}:")
        return {"camera": camera, **mapping}

    def _instance_mask(self, item, camera):
        key = item.get("id") if isinstance(item, dict) else item
        entry = self.instances.get(key)
        if entry is None:
            return None
        return entry["masks"].get(camera)

    # -- segmenting and measuring -------------------------------------------------------------------
    def _robot_parts(self, observation):
        """Per arm: its joints, its base and the finger travel, for drawing the robot in the images."""
        rows = {}
        for arm in ARMS:
            command = self.ctl.commanded_opening(observation, arm)
            reading = self.ctl.reading(observation, arm)
            # The reading trails the fingers by a fifth of the range while they are held apart by something.
            opening = reading if command >= 0.5 else min(1.0, reading + control.FINGER_LEAD)
            travel = max(0.0, (control.OPENING_ZERO_M + control.WIDTH_PER_READING_M * opening) / 2)
            rows[arm] = (self.ctl.joints(observation, arm), self.ctl.base(observation, arm), travel)
        return rows

    def _sight(self, observation, camera, instances, parts) -> objects.Sight:
        view = self.rig.view(camera, observation.state)
        own = {"cam_left_wrist": "left", "cam_right_wrist": "right"}.get(camera)
        hulls = []
        for arm in ARMS:
            joints, base, travel = parts[arm]
            # A wrist camera sits on its own arm: only that arm's fingers are in its view.
            hulls += self.shape.hulls(joints, base, travel, fingers_only=arm == own)
        return objects.Sight(view, objects.robot_silhouette(view, hulls), instances)

    def _color(self, image, mask):
        pixels = np.asarray(image)[mask]
        return [int(v) for v in np.median(pixels, axis=0)] if len(pixels) else None

    def do_segment(self, text, camera="cam_head", min_score=MIN_SCORE):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a short noun phrase")
        self.counters["segment_calls"] += 1
        observation = self.observation
        image = self._image(camera)
        rows = []
        for instance in self.segmenter.segment(image, text, key=(observation.step, camera), min_score=float(min_score)):
            key = self.next_id
            self.next_id += 1
            self.instances[key] = {"masks": {camera: instance.mask}}
            ys, xs = np.nonzero(instance.mask)
            rows.append({"id": key, "camera": camera, "score": round(instance.score, 3), "pixel": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                         "box": list(instance.box), "area_px": instance.area, "color": self._color(image, instance.mask)})
        return rows

    def do_mask(self, instance):
        key = instance.get("id") if isinstance(instance, dict) else instance
        entry = self.instances.get(key)
        if entry is None:
            raise ValueError("unknown instance id")
        camera = instance.get("camera") if isinstance(instance, dict) else None
        masks = entry["masks"]
        if camera is None:
            camera = next(iter(masks))
        if camera not in masks:
            raise ValueError(f"this object was not seen in {camera}; it was seen in " + ", ".join(masks))
        return masks[camera].copy()

    def _to_object(self, measured: objects.Measured, observation) -> dict:
        key = self.next_id
        self.next_id += 1
        self.instances[key] = {"masks": dict(measured.masks), "measured": measured}
        camera = measured.cameras[0]
        return {"id": key, "text": measured.text, "score": round(measured.score, 3), "x": round(measured.x, 4), "y": round(measured.y, 4), "top_z": round(measured.top_z, 4),
                "bottom_z": round(measured.bottom_z, 4), "height_m": round(measured.top_z - measured.bottom_z, 4), "length_m": round(measured.length, 4),
                "width_m": round(measured.width, 4), "long_side_deg": round(math.degrees(measured.long_side_rad), 1), "body_length_m": round(measured.body_length, 4),
                "body_width_m": round(measured.body_width, 4), "cameras": list(measured.cameras),
                "agreement": None if measured.agreement is None else round(measured.agreement, 3), "pixels": measured.pixels,
                "color": self._color(observation.images[camera], measured.masks[camera]), "notes": list(measured.notes)}

    def _measure(self, observation, text, cameras_used, on, min_score):
        support = frames.TABLE_Z if on is None else float(on)
        parts = self._robot_parts(observation)
        sights = {}
        for camera in cameras_used:
            if camera in observation.images:
                found = self.segmenter.segment(observation.images[camera], text, key=(observation.step, camera), min_score=float(min_score))
                sights[camera] = self._sight(observation, camera, found, parts)
        return objects.locate(sights, text, support)

    def do_locate(self, text, cameras=None, on=None, min_score=MIN_SCORE):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a short noun phrase")
        self.counters["locate_calls"] += 1
        observation = self.observation
        chosen = list(cameras) if cameras else list(CAMERAS)
        found = [self._to_object(row, observation) for row in self._measure(observation, text, chosen, on, min_score)]
        self.ctl.scene.note(found, text.strip(), self._carried(observation))
        return found

    def _carried(self, observation) -> list:
        return [self.ctl.grasp(observation, arm)[0][:2] for arm in ARMS if self.ctl.commanded_opening(observation, arm) < control.HOLD_OPENING]

    def do_locate_at(self, camera, pixel, on=None):
        self.counters["locate_calls"] += 1
        observation = self.observation
        image = self._image(camera)
        pixel = np.asarray(pixel, dtype=np.float64).reshape(-1)
        if pixel.size != 2:
            raise ValueError("pixel must be [u, v]")
        found = self.segmenter.segment_point(image, pixel, key=(observation.step, camera))
        if not found:
            return None
        support = frames.TABLE_Z if on is None else float(on)
        parts = self._robot_parts(observation)
        view = self.rig.view(camera, observation.state)
        sight = self._sight(observation, camera, found[:1], parts)
        others = {}
        # In the other cameras the object lies somewhere along this pixel's ray: ask for the object under points of the ray.
        for other in CAMERAS:
            if other == camera or other not in observation.images:
                continue
            other_view = self.rig.view(other, observation.state)
            origin, direction = view.ray(pixel)
            candidates = []
            for height in RAY_HEIGHTS_M:
                if abs(direction[2]) < 1e-6:
                    continue
                t = (support + height - origin[2]) / direction[2]
                if t <= 0:
                    continue
                projected, depth = other_view.project((origin + t * direction)[None])
                u, v = projected[0]
                if depth[0] > 0 and 0 <= u < other_view.size[0] and 0 <= v < other_view.size[1]:
                    rows = self.segmenter.segment_point(observation.images[other], projected[0], key=(observation.step, other))
                    candidates += rows[:1]
            if candidates:
                others[other] = self._sight(observation, other, candidates, parts)
        measured = objects.measure(sight, found[0], others, support, AT_PIXEL)
        if measured is None:
            return None
        for name, mask in measured.masks.items():
            ys, xs = np.nonzero(mask)
            measured.pixels[name] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
        row = self._to_object(measured, observation)
        self.ctl.scene.note([row], None, self._carried(observation))
        return row

    def _cells(self, item) -> np.ndarray:
        if isinstance(item, dict):
            entry = self.instances.get(item.get("id"))
            if entry is None or entry.get("measured") is None or entry["measured"].cells is None:
                raise ValueError("this is not an object from locate")
            return entry["measured"].cells
        cells = np.asarray(item, dtype=np.float64)
        if cells.ndim != 2 or cells.shape[1] != 2 or len(cells) < 3:
            raise ValueError("a footprint is an N x 2 array of world x, y points")
        return cells

    def do_footprint(self, obj):
        return self._cells(obj).copy()

    def do_register(self, moving, fixed, symmetry=1):
        return objects.register(self._cells(moving), self._cells(fixed), symmetry)

    def do_point(self, camera, pixel, height=None):
        observation = self.observation
        self._image(camera)
        view = self.rig.view(camera, observation.state)
        z = frames.TABLE_Z if height is None else float(height)
        point = view.on_plane(np.asarray(pixel, dtype=np.float64)[None], z)[0]
        origin, direction = view.ray(pixel)
        if not np.isfinite(point).all():
            return {"point": None, "reason": "the ray does not meet that plane in front of the camera"}
        range_m = float(np.linalg.norm(point - origin))
        along = abs(direction / max(abs(float(direction[2])), 1e-6)) * (RANGE_HEIGHT_M if height is None else 0.01)
        across = PIXEL_ERROR_PX * range_m / float(view.k[0, 0])
        return {"point": _rounded(point), "ray_origin": _rounded(origin), "ray_direction": _rounded(direction), "range_m": round(range_m, 4),
                "uncertainty_m": _rounded(np.sqrt(along ** 2 + across ** 2), 3)}

    def _triangulated(self, observation, views):
        if not isinstance(views, list) or not 2 <= len(views) <= 3:
            raise ValueError("views must list two or three {camera, pixel}")
        rays, placed = [], []
        for item in views:
            view = self.rig.view(item["camera"], observation.state)
            rays.append(view.ray(item["pixel"]))
            placed.append((view, np.asarray(item["pixel"], dtype=np.float64)))
        point, gap, crossing = cameras.triangulate(rays)
        if point is None:
            return {"ok": False, "point": None, "crossing_sine": round(crossing, 3), "reason": "the views look along nearly the same line, so depth is not resolved"}
        errors = [round(float(np.linalg.norm(view.project(point[None])[0][0] - pixel)), 2) for view, pixel in placed]
        return {"ok": gap <= 0.005, "point": _rounded(point), "ray_gap_mm": round(gap * 1000, 1), "crossing_sine": round(crossing, 3), "reprojection_px": errors}

    def do_triangulate(self, views):
        return self._triangulated(self.observation, views)

    def do_project(self, points, camera="cam_head"):
        observation = self.observation
        self._image(camera)
        view = self.rig.view(camera, observation.state)
        pixels, depth = view.project(np.asarray(points, dtype=np.float64).reshape(-1, 3))
        return [{"pixel": _rounded(p, 1), "depth_m": round(float(d), 4), "in_image": bool(d > 0 and 0 <= p[0] < view.size[0] and 0 <= p[1] < view.size[1])} for p, d in zip(pixels, depth)]

    def do_watch(self, camera, steps, every=5, regions=None, stop_after_quiet=None):
        self.counters["watches"] += 1
        watcher = observe.Watcher(camera, self._image(camera), steps, every, regions, stop_after_quiet)
        start = self.observation.step
        ended = "complete"
        while True:
            offset = self.observation.step - start
            hold = watcher.next_hold(offset)
            if not hold:
                if offset < steps:
                    ended = "quiet"
                break
            observation = self.observation
            self.bridge.execute([self.bridge.hold_action(observation)] * hold)
            watcher.capture(self._image(camera), self.observation.step - start)
        if len(watcher.samples) > 1:
            self.attachments += watcher.sheet()
        return watcher.report(ended)

    # -- planning without moving ----------------------------------------------------------------------
    def do_reach(self, points, arm=None, quaternion_wxyz=None, lean_deg=None, clearance_m=0.0, approach="vertical"):
        self.counters["reach_queries"] += 1
        observation = self.observation
        points = [np.asarray(p, dtype=np.float64).reshape(-1) for p in points]
        if not 1 <= len(points) <= REACH_MAX_POINTS or any(p.size != 3 for p in points):
            raise ValueError(f"points must list 1 to {REACH_MAX_POINTS} world grasp points [x, y, z]")
        arms = [arm] if arm else list(ARMS)
        given = None if quaternion_wxyz is None else frames.normalized(quaternion_wxyz)
        rows = []
        for point in points:
            row = {"point": _rounded(point)}
            for side in arms:
                if given is not None:
                    options = [("given", given)]
                    # The same jaw line with the hand tilted, as robo.pick would try it, for when the given hand itself does not reach.
                    options += [(f"given_tilted_{degrees:g}", frames.tilted(given, degrees)) for degrees in actions.TILTS_DEG]
                elif lean_deg is not None:
                    options = [(f"leaning_{lean_deg:g}", frames.leaning(side, point, float(lean_deg)))]
                else:
                    options = [(f"down_turned_{d}" if d else "down", frames.turned(frames.DOWN, UP, d)) for d in REACH_TURNS_DEG]
                    options.append(("current", self.ctl.grasp(observation, side)[1]))
                found = []
                for label, q in options:
                    got = self._try_reach(observation, side, point, q, float(clearance_m), approach)
                    if got is not None:
                        found.append({"hand": label, "quaternion_wxyz": _rounded(q), **got})
                        if label == "given":
                            break
                if given is None and lean_deg is None and len(found) < 2:   # straight down mostly refused: how far must the hand lean
                    leans = 0
                    for degrees in REACH_LEANS_DEG:
                        q = frames.leaning(side, point, degrees)
                        got = self._try_reach(observation, side, point, q, float(clearance_m), "pointing" if degrees >= 60 else approach)
                        if got is not None:
                            found.append({"hand": f"leaning_{degrees}", "quaternion_wxyz": _rounded(q), **got})
                            leans += 1
                            if leans >= 2:
                                break
                row[side] = found[:REACH_MAX_OPTIONS]
            rows.append(row)
        return {"ok": True, "points": rows}

    def _try_reach(self, observation, arm, point, q, clearance, approach):
        direction = frames.pointing_axis(q) if approach == "pointing" else -UP
        start = point - clearance * direction
        path, _, _ = self.ctl.free_path(observation, arm, start, q, may_turn=self.ctl.commanded_opening(observation, arm) >= control.TURN_MIN_OPENING)
        if path is None:
            return None
        if clearance:
            line, _ = self.ctl.line_path(observation, arm, self.ctl.hops(start, q, point, q), start=path[-1])
            if line is None:
                return None
        row = {"steps": self.ctl.timed({arm: path}, False, self.ctl.commanded_opening(observation, arm) >= control.HOLD_OPENING)[1]}
        row["gap_to_other_arm_m"] = round(self.ctl.arm_gap(observation, {arm: path})[0], 3)
        return row

    def do_estimate(self, plan):
        self.counters["estimates"] += 1
        if not isinstance(plan, list) or not 1 <= len(plan) <= ESTIMATE_MAX_SEGMENTS:
            raise ValueError(f"plan must be a list of 1 to {ESTIMATE_MAX_SEGMENTS} segments")
        observation = self.observation
        joints = {arm: self.ctl.joints(observation, arm).copy() for arm in ARMS}
        poses = {arm: self.ctl.grasp(observation, arm) for arm in ARMS}
        openings = {arm: self.ctl.commanded_opening(observation, arm) for arm in ARMS}
        rows, total, complete = [], 0, True
        for index, item in enumerate(plan):
            op = item.get("op") if isinstance(item, dict) else None
            row = {"segment": index, "op": op}
            if op == "wait":
                row["steps"] = int(item.get("steps", 1))
            elif op == "grip":
                change = abs(float(item["opening"]) - openings[item["arm"]])
                row["steps"] = max(control.GRIP_MIN_STEPS, min(control.GRIP_MAX_STEPS, int(math.ceil(change / control.FINGER_LEAD)) + 1))
                openings[item["arm"]] = float(item["opening"])
            elif op in ("move", "line", "home"):
                arm = item["arm"]
                if op == "home":
                    path = self.ctl.home_path(observation, arm, start=joints[arm])
                    poses[arm] = frames.grasp_point(self.ctl.initial[arm + "_ee_pose"])
                    reason = None
                else:
                    position = np.asarray(item["position"], dtype=np.float64)
                    q = poses[arm][1] if item.get("quaternion_wxyz") is None else frames.normalized(item["quaternion_wxyz"])
                    if op == "move":
                        path, arrived, reason = self.ctl.free_path(observation, arm, position, q, start=joints[arm])
                        q = arrived if path is not None else q
                    else:
                        path, reason = self.ctl.line_path(observation, arm, self.ctl.hops(*poses[arm], position, q), start=joints[arm])
                    if path is not None:
                        poses[arm] = (position, q)
                if path is None:
                    row.update(reachable=False, reason=reason)
                    rows.append(row)
                    complete = False
                    break
                joints[arm] = path[-1]
                row["steps"] = self.ctl.timed({arm: path}, op == "line", op != "line" and openings[arm] >= control.HOLD_OPENING, openings[arm] < control.HOLD_OPENING)[1]
            else:
                raise ValueError("each segment needs op move, line, grip, home or wait")
            total += row.get("steps", 0)
            rows.append(row)
        left = observation.remaining
        return {"ok": complete, "segments": rows, "total_steps": total, "steps_left": left, "fits": bool(complete and total <= left)}

    def do_hold_point(self, arm, name="tip", point=None, views=None):
        if arm not in ARMS:
            raise ValueError("arm must be left or right")
        if (point is None) == (views is None):
            raise ValueError("give point (a world point of the held feature) or views (two or three cameras with its pixel)")
        observation = self.observation
        if views is not None:
            found = self._triangulated(observation, views)
            if found.get("point") is None:
                return {"ok": False, "failure": "uncertain", "reason": found.get("reason", "the views do not fix a point"), **found}
            world = np.asarray(found["point"], dtype=np.float64)
        else:
            world = np.asarray(point, dtype=np.float64)
        flange = np.asarray(observation.state[arm + "_ee_pose"], dtype=np.float64)
        rotation = frames.matrix(flange[3:7])
        grasp = np.array([frames.GRASP_OFFSET_M, 0.0, 0.0])
        local = rotation.T @ (world - flange[:3]) - grasp
        self.held_points[arm][name] = {"offset": local, "reading": self.ctl.reading(observation, arm)}
        return {"ok": True, "arm": arm, "name": name, "point": _rounded(world), "from_grasp_point_m": round(float(np.linalg.norm(local)), 4), "offset_in_gripper_frame_m": _rounded(local)}

    def _point_of(self, arm, name):
        if name is None:
            return None
        entry = self.held_points[arm].get(name)
        if entry is None:
            raise ValueError(f"no held point {name!r} for the {arm} arm: store it with robo.hold_point first")
        if abs(self.ctl.reading(self.observation, arm) - entry["reading"]) > STALE_HOLD_READING:
            raise ValueError(f"held point {name!r} is stale: the {arm} gripper reading changed since it was measured")
        return entry["offset"]

    # -- goals ------------------------------------------------------------------------------------------
    def _grasp_for(self, target, arm, observation, quaternion, z):
        """(world grasp point, orientation, the object it came from or None)."""
        if isinstance(target, dict):
            if not all(k in target for k in ("x", "y", "top_z")):
                raise ValueError("target must be an object from locate or a world grasp point [x, y, z]")
            top, bottom = float(target["top_z"]), float(target.get("bottom_z", frames.TABLE_Z))
            reference = self.ctl.grasp(observation, arm)[1] if self._pointing_down(observation, arm) else frames.DOWN
            if quaternion is not None:
                q = frames.normalized(quaternion)
            elif float(target.get("length_m", 0.0)) >= ROUND_ASPECT * max(float(target.get("width_m", 0.0)), 1e-6):
                q = frames.jaw_across(math.radians(float(target["long_side_deg"])), reference)
            else:
                q = reference
            from_side = float(frames.pointing_axis(q)[2]) > -STEEP_POINTING
            height = bottom + SIDE_GRASP_SHARE * (top - bottom) if from_side else grasp_height(top, bottom)
            position = np.array([float(target["x"]), float(target["y"]), height if z is None else float(z)])
            return position, q, target
        position = np.asarray(target, dtype=np.float64).reshape(-1)
        if position.size != 3:
            raise ValueError("target must be an object from locate or a world grasp point [x, y, z]")
        q = self.ctl.grasp(observation, arm)[1] if quaternion is None else frames.normalized(quaternion)
        if z is not None:
            position = np.r_[position[:2], float(z)]
        return position, q, None

    def _pointing_down(self, observation, arm) -> bool:
        q = self.ctl.grasp(observation, arm)[1]
        return float(frames.pointing_axis(q)[2]) < -0.95

    def _looker(self, arm, target, fixed_z):
        """What a pick does at its pre-grasp point: measure the same object again with this arm's wrist camera."""
        text = None if target.get("text") == AT_PIXEL else target.get("text")
        camera = f"cam_{arm}_wrist"
        known_top = float(target["top_z"])
        support = float(target.get("bottom_z", frames.TABLE_Z))

        def look(observation, point, q):
            if not text or camera not in observation.images:
                return None
            fresh = self._measure(observation, text, [camera, "cam_head"], support, MIN_SCORE)
            near = [row for row in fresh if math.hypot(row.x - point[0], row.y - point[1]) <= LOOK_NEAR_M and camera in row.cameras
                    and row.agreement is not None and row.agreement >= objects.TRUST_MIN]
            if near:
                best = min(near, key=lambda row: math.hypot(row.x - point[0], row.y - point[1]))
                delta = np.array([best.x - point[0], best.y - point[1]])
                if best.length >= objects.ELONGATED_ASPECT * best.width:
                    across = np.array([-math.sin(best.long_side_rad), math.cos(best.long_side_rad)])
                    delta = float(delta @ across) * across
                self.counters["pick_refined"] += 1
                row = {"delta": delta, "length": best.length, "width": best.width, "long_side_rad": best.long_side_rad, "agreement": best.agreement}
                pointing = frames.pointing_axis(q)
                if float(pointing[2]) <= -STEEP_POINTING:
                    if fixed_z is None:
                        row["height"] = grasp_height(best.top_z, best.bottom_z)
                else:   # from the side only the position across the approach is corrected
                    along = pointing[:2] / max(float(np.linalg.norm(pointing[:2])), 1e-6)
                    row["delta"] = delta - float(delta @ along) * along
                return row
            # The hand may hide the object from the head camera: then its outline in the wrist image, on the plane of its known top.
            view = self.rig.view(camera, observation.state)
            instances = self.segmenter.segment(observation.images[camera], text, key=(observation.step, camera), min_score=MIN_SCORE)
            sight = self._sight(observation, camera, instances, self._robot_parts(observation))
            found = objects.refine_on_plane(view, instances, sight.robot, point[:2], known_top, actions.REFINE_MAX_MOVE_M)
            if found is not None:
                self.counters["pick_refined"] += 1
            return found
        return look

    def _make_pick(self, observation, arm, target, quaternion_wxyz=None, z=None, approach="pointing", clearance_m=None, opening=1.0, lift_m=actions.LIFT_M, symmetry=1, check_hold=True, look=True, velocity=None):
        position, q, source = self._grasp_for(target, arm, observation, quaternion_wxyz, z)
        looking = bool(look) and source is not None
        clearance = (LOOK_CLEARANCE_M if looking else actions.CLEARANCE_M) if clearance_m is None else clearance_m
        if source is not None and clearance_m is None:
            clearance = max(clearance, self._clear_of(source, q))
        if source is not None and quaternion_wxyz is None and float(source.get("length_m", 0.0)) < ROUND_ASPECT * max(float(source.get("width_m", 0.0)), 1e-6):
            symmetry = 0 if symmetry == 1 else symmetry
        goal = actions.Pick(self.ctl, arm, observation, position, q, approach=approach, clearance_m=clearance, opening=opening, lift_m=lift_m, symmetry=symmetry, check_hold=check_hold,
                            look=self._looker(arm, source, z) if looking else None, velocity=velocity,
                            # A grasp at a chosen point still takes the measured object around that point, with its size and its phrase.
                            source=self.ctl.scene.around(position) if source is None else source)
        return goal

    @staticmethod
    def _clear_of(source, q) -> float:
        """How far back along the hand's axis the open fingertips are clear of an object's footprint, for a hand coming from the side."""
        pointing = frames.pointing_axis(q)
        if float(pointing[2]) <= -STEEP_POINTING:
            return 0.0
        angle = math.radians(float(source.get("long_side_deg", 0.0)))
        along = abs(pointing[0] * math.cos(angle) + pointing[1] * math.sin(angle))
        across = abs(-pointing[0] * math.sin(angle) + pointing[1] * math.cos(angle))
        length = max(float(source.get("length_m", 0.0)), float(source.get("body_length_m", 0.0)))
        width = max(float(source.get("width_m", 0.0)), float(source.get("body_width_m", 0.0)))
        return (along * length + across * width) / 2 + frames.FINGERTIP_REACH_M + SIDE_APPROACH_MARGIN_M

    def _make_place(self, observation, arm, position, quaternion_wxyz=None, on=None, approach="vertical", clearance_m=actions.CLEARANCE_M, retreat_m=actions.RETREAT_M, symmetry=1, point_of=None, opening=1.0, slow=False):
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        if on is not None:
            held = self.grasped.get(arm)
            if held is None or held.get("below") is None:
                raise ValueError("on=z needs an object held by robo.pick; give the world grasp point at release instead")
            position = np.r_[position[:2], float(on) + held["below"] + RELEASE_MARGIN_M]
        if position.size != 3:
            raise ValueError("position must be the world grasp point at release [x, y, z], or [x, y] with on=z")
        goal = actions.Place(self.ctl, arm, observation, position, quaternion_wxyz, approach=approach, clearance_m=clearance_m, retreat_m=retreat_m, symmetry=symmetry, point_of=self._point_of(arm, point_of), opening=opening, slow=slow,
                             support=frames.TABLE_Z if on is None else float(on))
        return goal

    def _make_goal(self, observation, arm, spec):
        spec = dict(spec)
        op = spec.pop("op", None)
        spec.pop("arm", None)
        if op == "pick":
            return self._make_pick(observation, arm, spec.pop("target"), **spec)
        if op == "place":
            return self._make_place(observation, arm, spec.pop("position"), **spec)
        if op == "move":
            return actions.Move(self.ctl, arm, observation, spec.pop("position"), spec.get("quaternion_wxyz"), spec.get("opening"), slow=spec.get("slow", False))
        if op == "line":
            return actions.Move(self.ctl, arm, observation, spec.pop("position"), spec.get("quaternion_wxyz"), spec.get("opening"), straight=True, slow=spec.get("slow", False))
        if op == "stroke":
            spec["point_of"] = self._point_of(arm, spec.get("point_of"))
            return actions.Stroke(self.ctl, arm, observation, spec.pop("position"), spec.pop("stroke"), spec.pop("quaternion_wxyz", None), **spec)
        if op == "turn":
            spec["point_of"] = self._point_of(arm, spec.get("point_of"))
            return actions.Turn(self.ctl, arm, observation, spec.pop("axis"), spec.pop("degrees"), **spec)
        if op == "home":
            return actions.Home(self.ctl, arm, observation)
        if op == "look":
            return self._make_look(observation, arm, spec.pop("point"), **spec)
        raise ValueError("op must be pick, place, move, line, look, stroke, turn or home")

    def _after_pick(self, arm, goal, result):
        self.counters["picks"] += 1
        if result.get("ok"):
            self.counters["picks_held"] += 1
            below = None
            source = goal.source
            anchor = goal.anchor
            if source is not None:
                below = float(anchor[2]) - float(source.get("bottom_z", frames.TABLE_Z))
            self.grasped[arm] = {"below": below, "text": None if source is None else source.get("text")}
            self.ctl.scene.forget(None if source is None else source.get("id"), np.asarray(anchor)[:2])
            self.ctl.loads.pop(arm, None)
            if source is not None:
                self.ctl.loads[arm] = self._load_of(source, anchor, goal.q)
            if goal.check_hold:
                self.ctl.unseen_hold.discard(arm)
            else:
                self.ctl.unseen_hold.add(arm)

    @staticmethod
    def _load_of(source, anchor, q) -> np.ndarray:
        """Points of an object's measured box in the flange frame of the hand that grasped it at `anchor`, as homogeneous rows."""
        angle = math.radians(float(source.get("long_side_deg", 0.0)))
        length, width = float(source.get("length_m", 0.0)), float(source.get("width_m", 0.0))
        along, across = np.array([math.cos(angle), math.sin(angle)]), np.array([-math.sin(angle), math.cos(angle)])
        rows = []
        for a in (-0.5, 0.0, 0.5):
            for b in (-0.5, 0.0, 0.5):
                xy = np.array([float(source["x"]), float(source["y"])]) + a * length * along + b * width * across
                rows += [np.r_[xy, z, 1.0] for z in (float(source.get("bottom_z", frames.TABLE_Z)), float(source["top_z"]))]
        return (np.linalg.inv(frames.transform(frames.flange_pose(anchor, q))) @ np.array(rows).T).T

    def _look_at_placed(self, text, goal, result):
        """Measure what was just let go, from every camera, so a placement reports where the object is and not only where the hand went."""
        observation = self.observation
        rows = [row for row in self._measure(observation, text, list(CAMERAS), goal.support, MIN_SCORE) if math.hypot(row.x - goal.anchor[0], row.y - goal.anchor[1]) <= PLACED_NEAR_M]
        if not rows:
            result["placed"] = {"seen": False}
            return None
        best = min(rows, key=lambda row: (row.agreement is None, math.hypot(row.x - goal.anchor[0], row.y - goal.anchor[1])))
        found = self._to_object(best, observation)
        self.ctl.scene.note([found])
        result["placed"] = {"seen": True, "x": found["x"], "y": found["y"], "top_z": found["top_z"], "bottom_z": found["bottom_z"], "long_side_deg": found["long_side_deg"],
                            "cameras": found["cameras"], "agreement": found["agreement"]}
        return found

    def _after_place(self, arm, goal, result=None):
        """A placed object stands where its box was carried to until it is measured again; it is measured here when its phrase is known."""
        text = (self.grasped.get(arm) or {}).get("text")
        text = None if text == AT_PIXEL else text
        self.grasped[arm] = None
        load = self.ctl.loads.pop(arm, None)
        if text and result is not None and goal.support is not None and self._look_at_placed(text, goal, result) is not None:
            return
        if load is None:
            return
        world = (frames.transform(frames.flange_pose(goal.anchor, goal.q)) @ load.T).T
        low, high = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
        key = self.next_id
        self.next_id += 1
        self.ctl.scene.put_down(key, low, high, float(world[:, 2].min()), float(world[:, 2].max()))

    def _run_one(self, arm, goal, force=False):
        with self.ctl.forcing(force):
            refused = goal.refusal or goal.plan(self.observation)
            if refused is not None:
                self.ctl.counters["refusals"] += 1
                return refused
            return self.runner.run({arm: goal}, force=force)[arm]

    def do_move(self, arm, position, quaternion_wxyz=None, opening=None, straight=False, slow=False, force=False):
        goal = actions.Move(self.ctl, arm, self.observation, position, quaternion_wxyz, opening, straight=straight, slow=slow)
        return self._run_one(arm, goal, force)

    def _view_poses(self, arm, point, distance, direction):
        """Hand poses that put a wrist camera at `distance` from a point, looking straight at it: the asked direction and those near it, or from above and then ever lower, starting on the side of the arm's base."""
        away = frames.BASE_ORIGIN[arm][:2] - point[:2]
        heading = math.atan2(float(away[1]), float(away[0]))
        if direction is None:
            rows = [(math.radians(up), heading + math.radians(side)) for up in VIEW_ELEVATIONS_DEG for side in (VIEW_SIDES_DEG if up < 90 else (0.0,))]
        else:
            given = np.asarray(direction, dtype=np.float64)
            given /= np.linalg.norm(given)
            up, around = math.asin(float(given[2])), math.atan2(float(given[1]), float(given[0]))
            rows = [(min(math.pi / 2, up + math.radians(a)), around + math.radians(b)) for a, b in VIEW_NEAR_DEG]
        inverse_mount = np.linalg.inv(self.rig.wrist_mount)
        for up, around in rows:
            back = np.array([math.cos(up) * math.cos(around), math.cos(up) * math.sin(around), math.sin(up)])   # from the point toward the camera, the camera's +z
            side = np.cross(UP, back)
            if float(np.linalg.norm(side)) < 1e-6:
                side = np.array([-math.sin(around), math.cos(around), 0.0])
            side /= np.linalg.norm(side)
            for roll in VIEW_ROLLS_DEG:
                x = math.cos(math.radians(roll)) * side + math.sin(math.radians(roll)) * np.cross(back, side)
                camera = np.eye(4)
                camera[:3, 0], camera[:3, 1], camera[:3, 2], camera[:3, 3] = x, np.cross(back, x), back, point + distance * back
                position, q = frames.grasp_point(frames.pose_of(camera @ inverse_mount))
                yield math.degrees(up), position, q

    def _make_look(self, observation, arm, point, distance_m=VIEW_DISTANCE_M, direction=None, slow=False, force=False):
        """A motion to the first pose that gives this view and can be planned: from the highest view down, among equal views the orientation nearest the hand's first, so that it is the short way."""
        if arm not in ARMS:
            raise ValueError("arm must be left or right")
        point = np.asarray(point, dtype=np.float64).reshape(3)
        distance = actions._number(distance_m, "distance_m", VIEW_MIN_M, VIEW_MAX_M)
        here = self.ctl.grasp(observation, arm)[1]
        poses = sorted(self._view_poses(arm, point, distance, direction), key=lambda row: (-row[0] if direction is None else 0.0, frames.angle_deg(here, row[2])))
        with self.ctl.forcing(force):
            for elevation, position, q in poses:
                path, _, _ = self.ctl.free_route(observation, arm, position, q)
                if path is not None and (self.ctl.forced or self.ctl.gap_refusal(observation, {arm: path}) is None):
                    goal = actions.Move(self.ctl, arm, observation, position, q, slow=slow, exact=True)
                    goal.values.update(camera="cam_left_wrist" if arm == "left" else "cam_right_wrist", seen_from_above_deg=round(elevation))
                    return goal
        return actions.Refused(self.ctl, arm, "nothing moved: no pose of this arm lets its wrist camera look at this point from this distance; try the other arm, another distance or another direction")

    def do_look(self, arm, point, distance_m=VIEW_DISTANCE_M, direction=None, slow=False, force=False):
        return self._run_one(arm, self._make_look(self.observation, arm, point, distance_m, direction, slow, force), force)

    def do_shift(self, arm, delta, slow=False, force=False):
        observation = self.observation
        p, q = self.ctl.grasp(observation, arm)
        goal = actions.Move(self.ctl, arm, observation, p + np.asarray(delta, dtype=np.float64), q, straight=True, slow=slow)
        return self._run_one(arm, goal, force)

    def do_grip(self, arm, opening):
        if arm not in ARMS:
            raise ValueError("arm must be left or right")
        self.ctl.loads.pop(arm, None)
        row = self.ctl.grip({arm: actions._number(opening, "opening", 0.0, 1.0)})[arm]
        return {"ok": True, **row}

    def do_opening_for(self, width_m):
        return round(control.opening_for_width(float(width_m)), 4)

    def do_hand(self, pointing=(0.0, 0.0, -1.0), closing=(-1.0, 0.0, 0.0)):
        return _rounded(frames.from_axes(pointing, closing), 5)

    def do_turned(self, quaternion_wxyz, axis, degrees):
        axis = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}.get(axis, axis) if isinstance(axis, str) else axis
        return _rounded(frames.turned(quaternion_wxyz, np.asarray(axis, dtype=np.float64) / np.linalg.norm(axis), float(degrees)), 5)

    def do_pick(self, arm, target, quaternion_wxyz=None, z=None, approach="pointing", clearance_m=None, opening=1.0, lift_m=actions.LIFT_M, symmetry=1, check_hold=True, look=True, velocity=None, force=False):
        goal = self._make_pick(self.observation, arm, target, quaternion_wxyz, z, approach, clearance_m, opening, lift_m, symmetry, check_hold, look, velocity)
        result = self._run_one(arm, goal, force)
        self._after_pick(arm, goal, result)
        return result

    def do_place(self, arm, position, quaternion_wxyz=None, on=None, approach="vertical", clearance_m=actions.CLEARANCE_M, retreat_m=actions.RETREAT_M, symmetry=1, point_of=None, opening=1.0, slow=False, force=False):
        goal = self._make_place(self.observation, arm, position, quaternion_wxyz, on, approach, clearance_m, retreat_m, symmetry, point_of, opening, slow)
        result = self._run_one(arm, goal, force)
        self.counters["places"] += 1
        if result.get("ok"):
            self.counters["places_ok"] += 1
            self._after_place(arm, goal, result)
        return result

    def do_stroke(self, arm, position, stroke, quaternion_wxyz=None, approach="pointing", clearance_m=actions.STROKE_CLEARANCE_M, opening=None, increment_m=actions.STROKE_INCREMENT_M, withdraw=True, point_of=None, force=False):
        self.counters["strokes"] += 1
        goal = actions.Stroke(self.ctl, arm, self.observation, position, stroke, quaternion_wxyz, approach, clearance_m, opening, increment_m, withdraw, self._point_of(arm, point_of))
        return self._run_one(arm, goal, force)

    def do_turn(self, arm, axis, degrees, frame="world", pivot=None, wrist_pivot=False, step_deg=actions.TURN_STEP_DEG, point_of=None, slow=False):
        self.counters["turns"] += 1
        goal = actions.Turn(self.ctl, arm, self.observation, axis, degrees, frame, pivot, wrist_pivot, step_deg, self._point_of(arm, point_of), slow)
        return self._run_one(arm, goal)

    def do_both(self, left, right, force=False):
        with self.ctl.forcing(force):
            return self._both(left, right, force)

    def _both(self, left, right, force):
        self.counters["both_calls"] += 1
        observation = self.observation
        goals = {"left": self._make_goal(observation, "left", left), "right": self._make_goal(observation, "right", right)}
        refusals = {arm: goal.refusal or goal.plan(observation) for arm, goal in goals.items()}
        if any(row is not None for row in refusals.values()):
            return {"ok": False, "failure": "unreachable", "reason": "nothing moved: an arm cannot do its part",
                    "left": refusals["left"] or {"ok": True, "moved": False}, "right": refusals["right"] or {"ok": True, "moved": False}}
        if not force:
            reason = self.runner.check_pair(observation, goals)
            if reason is not None:
                return {"ok": False, "failure": "unreachable", "reason": reason, "moved": False}
        results = self.runner.run(goals, force=force)
        for arm, goal in goals.items():
            if isinstance(goal, actions.Pick):
                self._after_pick(arm, goal, results[arm])
            if isinstance(goal, actions.Place) and results[arm].get("ok"):
                self._after_place(arm, goal, results[arm])
        failed = [row.get("failure") for row in results.values() if not row.get("ok")]
        return {"ok": not failed, **({"failure": failed[0], "reason": "one arm's goal failed; see its result"} if failed else {}), "left": results["left"], "right": results["right"]}

    def do_home(self, arm="both"):
        self.counters["homes"] += 1
        observation = self.observation
        if arm == "both":
            status = {name: self._at_home(observation, name) for name in ARMS}
            away = [name for name in ARMS if not status[name]]
            points = {name: self.ctl.grasp(observation, name)[0] for name in ARMS}
            if len(away) == 2:
                gap = float(np.linalg.norm(points["left"][:2] - points["right"][:2]))
                if points["left"][0] >= points["right"][0] or gap < actions.HOME_MIN_GAP_M:
                    return {"ok": False, "failure": "unreachable", "moved": False, "reason": f"the arms are crossed or closer than {actions.HOME_MIN_GAP_M:g} m; return them one at a time"}
            goals = {name: actions.Home(self.ctl, name, observation) for name in ARMS}
            results = self.runner.run(goals)
            failed = [row.get("failure") for row in results.values() if not row.get("ok")]
            return {"ok": not failed, **({"failure": failed[0], "reason": "an arm did not get back"} if failed else {}), "left": results["left"], "right": results["right"]}
        if arm not in ARMS:
            raise ValueError("arm must be left, right or both")
        return self.runner.run({arm: actions.Home(self.ctl, arm, observation)})[arm]

    def _at_home(self, observation, arm) -> bool:
        return float(np.max(np.abs(self.ctl.joints(observation, arm) - self.ctl.joints_initial(arm)))) <= 0.01

    # -- the learned policy ---------------------------------------------------------------------------
    def do_policy(self, text=None, chunks=1):
        if self.policy_service is None:
            raise ValueError("no learned policy is configured")
        if text is not None and (type(text) is not str or not text.strip()):
            raise ValueError("text must be a sentence or None")
        if type(chunks) is not int or not 1 <= chunks <= POLICY_MAX_CHUNKS:
            raise ValueError(f"chunks must be an integer from 1 to {POLICY_MAX_CHUNKS}")
        self.counters["policy_calls"] += 1
        observation = self.observation
        if observation.remaining <= 0:
            return {"ok": False, "failure": "no_steps", "moved": False, "reason": "no native steps are left"}
        sentence = observation.instruction if text is None else text.strip()
        if self.policy is not None and (self.policy_step != observation.step or self.policy_text != sentence):
            self.end_policy()
        try:
            if self.policy is None:
                self._open_policy(observation, sentence)
            result = self._run_policy(sentence, chunks)
        except wam.ServiceFailure as error:
            self.end_policy()
            raise InfrastructureFailure(str(error)) from error
        self.policy_step = self.observation.step
        return result

    def _open_policy(self, observation, sentence):
        if self.counters["policy_config"] is None:
            self.counters["policy_config"] = self.read_service(self.policy_service).get("config_sha256")
        session = self.open_policy(self.policy_service)
        session.open(wam.frame(observation.images, observation.state, sentence))
        self.policy, self.policy_text = session, sentence
        self.counters["policy_sessions"] += 1
        self.counters["policy_wait_s"] = round(self.counters["policy_wait_s"] + session.waited_s, 3)

    def _run_policy(self, sentence, chunks) -> dict:
        """Run chunks action by action, each answered with the observation after it. Contact does not stop them: cutting a chunk short would end the session and its history."""
        session = self.policy
        lagging = {arm: 0 for arm in ARMS}
        started, stopped, done, continuing = self.observation.step, None, 0, False
        predict_s, observe_s = len(session.predictions), len(session.observations)
        try:
            for _ in range(chunks):
                for action in session.chunk():
                    observation = self.observation
                    if observation.remaining <= 0:
                        stopped = "no native steps are left"
                        break
                    native = {key: np.asarray(action[key], dtype=np.float32).reshape(-1).copy() for key in wam.STATE_KEYS}
                    for arm in ARMS:
                        native[arm + "_ee_joint_state"] = np.clip(native[arm + "_ee_joint_state"], 0.0, 1.0)
                    if not all(np.isfinite(value).all() for value in native.values()) or self.bridge.monitor.largest_jump(native, observation, continuing) > monitor.COMMAND_SANITY_RAD:
                        stopped = "the policy asked for a sudden jump, which was not sent"
                        break
                    run = self.bridge.execute([native], contact=True, continuing=continuing)
                    continuing = True
                    self.bridge.grippers.update({arm: float(native[arm + "_ee_joint_state"][0]) for arm in ARMS})
                    now = self.observation
                    session.observe(wam.frame(now.images, now.state, sentence))
                    for reason in run.reasons:
                        kind, _, arm = reason.partition(":")
                        if kind == "tracking_deviation":
                            lagging[arm] += 1
                        else:
                            stopped = "an arm jumped"
                if stopped:
                    break
                done += 1
        finally:
            self.counters["policy_chunks"] += done
            self.counters["policy_steps"] += self.bridge.observation.step - started
            self.counters["policy_lagging_steps"] += sum(lagging.values())
            self.counters["policy_predict_s"] = round(self.counters["policy_predict_s"] + sum(session.predictions[predict_s:]), 3)
            self.counters["policy_observe_s"] = round(self.counters["policy_observe_s"] + sum(session.observations[observe_s:]), 3)
        self._forget_after_policy()
        now = self.observation
        result = {"ok": True, "chunks": done, "arms": {arm: self.arm_state(now, arm) for arm in ARMS}, "lagging_steps": lagging}
        if stopped:
            result["stopped"] = stopped
            self.end_policy()
        return result

    def _forget_after_policy(self):
        """The policy moved things without measuring them: what was measured, and what the hands were known to hold, no longer stands."""
        self.held_points = {arm: {} for arm in ARMS}
        self.grasped = {arm: None for arm in ARMS}
        self.ctl.loads.clear()
        self.ctl.unseen_hold.clear()
        self.ctl.scene.clear()

    def end_policy(self):
        """Free the service's GPU. The next call starts afresh, from the current frame."""
        session, self.policy = self.policy, None
        if session is not None:
            session.close()

    def do_second_look(self, question=None):
        if question is not None and type(question) is not str:
            raise ValueError("question must be text or None")
        return {"opinion": self.ask_fresh(self.observation, question)}

    def do_wait(self, steps=1):
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        observation = self.observation
        count = min(steps, max(observation.remaining, 1))
        self.bridge.execute([self.bridge.hold_action(observation)] * count)
        return {"ok": True}


def method_reference(robo_class) -> str:
    """The robo interface as text, from the signatures and docstrings of the cell-side class."""
    rows = []
    for name, member in vars(robo_class).items():
        if not callable(member) or name.startswith("_"):
            continue
        signature = str(inspect.signature(member)).replace("(self, ", "(").replace("(self)", "()")
        doc = " ".join((member.__doc__ or "").split())
        rows.append(f"robo.{name}{signature}\n    {doc}")
    return "\n".join(rows)
