"""Planning and timing of native arm motions, from official robot state and the robot-only planner.

A motion is planned before anything moves and is refused at zero steps when it cannot be done: no joint solution inside the arm's real range,
no joint path, a joint detour, the two arms too close, the arm into the table, or not enough steps left. Free motions take the joint path the
planner finds, and go up and over an object the agent has measured when that path would run into it; straight motions follow a
Cartesian line hop by hop, so the grasp point stays on the line and a line that would make the arm swing is refused.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np

from . import frames, kinematics, planning, scene, timing
from .bridge import Bridge, Execution, Observation
from .frames import ARMS
from .robot import wrist_folded

LINE_STEP_M = 0.015          # straight motions are planned in hops of at most this length ...
LINE_STEP_DEG = 6.0          # ... and this much rotation
HOP_JOINT_LIMIT_RAD = 0.5    # a hop whose furthest joint travels more than this leaves the line
ARRIVE_M = 0.01              # a stage arrived when the grasp point is this close ...
ARRIVE_DEG = 5.0             # ... and the orientation this close to the request
SETTLE_TOL_RAD = 0.01        # after a motion the final command is held until every joint is this close ...
SETTLE_MAX_STEPS = 6         # ... for at most this many steps
HOLD_OPENING = 0.5           # a latched gripper command below this means the arm may hold something, unless ...
FREE_JAW_OPENING = 0.2       # ... the jaw stands open at least this far with nothing stopping it
TURN_MIN_OPENING = 0.9       # an empty gripper this open may take the half-turned pose
MIN_GAP_M = 0.02             # the surfaces of the two arms' sphere models stay this far apart ...
HOLDING_EXTRA_M = 0.03       # ... and this much more while one holds something of unknown size
CLOSER_M = 0.005             # a motion that gets at most this much closer than the arms already are is never refused
GAP_SAMPLE_EVERY = 4         # one check per this many native steps ...
GAP_MAX_SAMPLES = 12         # ... and at most this many along a motion
TABLE_MARGIN_M = 0.002       # how far a free path's hand may dip below the table surface before it is refused
FINGER_LEAD = 0.2            # the gripper reading stays this much of the range behind the fingers
OBJECT_SAMPLE_RAD = 0.05     # a free path is checked against measured objects every this much joint travel ...
OBJECT_MAX_SAMPLES = 40      # ... at most this many times
OVER_MARGIN_M = 0.02         # going over an object, what hangs lowest passes this far above it
OVER_TRIES = 2
OVER_RAISE_M = 0.08          # a way across that still dips into the table is tried this much higher
HALF_TURN_NEAR_DEG = 30.0    # with a load, an orientation this near half a turn from the hand's is refused: what it holds would come out turned over or end for end
SPIN_LIMIT_RAD = 3.5         # with a load, a wrist turn beyond this is refused: the orientation asked for lies past the end of the wrist's range
# Gripper commands repeat until the reading stops changing.
GRIP_STILL = 0.004
GRIP_MIN_STEPS = 2
GRIP_MAX_STEPS = 10
HOLDING_READING = 0.005      # closed on nothing the reading stays below this (about 2 mm between the fingers)
# Jaw width from the reading: the command range maps to finger travel by the robot configuration, 0.108 m per unit.
WIDTH_OFFSET_M = 0.0016
WIDTH_PER_READING_M = 0.108
# A gripper command c opens the jaw to about this width.
OPENING_ZERO_M = -0.02


def width_from_reading(reading: float) -> float:
    return min(frames.MAX_JAW_WIDTH_M, max(0.0, WIDTH_OFFSET_M + WIDTH_PER_READING_M * float(reading)))


def opening_for_width(width_m: float) -> float:
    """The gripper command that opens the jaw to a width."""
    return min(1.0, max(0.0, (float(width_m) - OPENING_ZERO_M) / WIDTH_PER_READING_M))


class Controller:
    """All native motion of one episode. Planning uses the observation at call time."""

    def __init__(self, spec, model, planner: planning.Curobo, bridge: Bridge, stand):
        self.spec = spec
        self.model = model
        self.planner = planner
        self.bridge = bridge
        self.initial = None
        self.scene = scene.Scene()
        center, half, angle, top = stand
        self.post = scene.Box(0, "the post that carries the head camera", center, half, angle, frames.TABLE_Z, top)
        self.loads = {}              # per arm: what the fingers hold, as homogeneous points (N x 4) in the flange frame
        self.unseen_hold = set()     # arms whose hold does not show in the reading: the caller said so when it grasped
        self.half_held = set()       # arms whose hold began half a turn about the pointing axis from the orientation asked for
        self.forced = False          # the caller passed force=True: measured objects and the wrist turn do not refuse this motion
        self.counters = {"free_plans": 0, "line_plans": 0, "refusals": 0, "arm_conflicts": 0, "settle_steps": 0, "object_refusals": 0}

    @contextmanager
    def forcing(self, force):
        """While this lasts, force=True: measured objects, the post and the wrist turn do not refuse a motion."""
        before, self.forced = self.forced, self.forced or bool(force)
        try:
            yield
        finally:
            self.forced = before

    def begin(self, observation: Observation):
        self.initial = {key: value.copy() for key, value in observation.state.items()}

    # -- state ----------------------------------------------------------------------------------
    @staticmethod
    def grasp(observation: Observation, arm: str):
        return frames.grasp_point(observation.state[arm + "_ee_pose"])

    @staticmethod
    def joints(observation: Observation, arm: str) -> np.ndarray:
        return np.asarray(observation.state[arm + "_arm_joint_state"], dtype=np.float64)

    @staticmethod
    def reading(observation: Observation, arm: str) -> float:
        return float(observation.state[arm + "_ee_joint_state"][0])

    def base(self, observation: Observation, arm: str) -> np.ndarray:
        return self.model.base(self.joints(observation, arm), observation.state[arm + "_ee_pose"])

    def commanded_opening(self, observation: Observation, arm: str) -> float:
        goal = self.bridge.grippers.get(arm)
        return self.reading(observation, arm) if goal is None else float(goal)

    def _sphere_opening(self, observation: Observation, arm: str) -> float:
        return min(1.0, self.reading(observation, arm) + FINGER_LEAD)

    def holding(self, observation: Observation, arm: str) -> bool:
        command = self.commanded_opening(observation, arm)
        return command < HOLD_OPENING and self.reading(observation, arm) > command + HOLDING_READING

    def may_hold(self, observation: Observation, arm: str) -> bool:
        """Whether something may be in this hand: a hold on a thin thing can read like an empty hand, but a half-open jaw that reached its command closed on nothing."""
        command = self.commanded_opening(observation, arm)
        return command < HOLD_OPENING and not (command >= FREE_JAW_OPENING and self.reading(observation, arm) <= command + HOLDING_READING)

    def jaw_width(self, observation: Observation, arm: str) -> float:
        """The distance between the fingers. Blocked while closing, the reading stays ahead of them by a fixed share; otherwise they stand where they were told."""
        command, reading = self.commanded_opening(observation, arm), self.reading(observation, arm)
        return width_from_reading(reading) if reading > command + HOLDING_READING else min(frames.MAX_JAW_WIDTH_M, max(0.0, OPENING_ZERO_M + WIDTH_PER_READING_M * command))

    # -- paths ----------------------------------------------------------------------------------
    def free_path(self, observation, arm, position, q, start=None, may_turn=False):
        """(joint path, orientation it arrives with, None) or (None, None, reason). Nothing moves."""
        self.counters["free_plans"] += 1
        spec = self.spec
        start = self.joints(observation, arm) if start is None else np.asarray(start, dtype=np.float64)
        first = self.joints_initial(arm)
        base = self.base(observation, arm)
        seed_rows = planning.seeds(start, first, spec.low, spec.high)
        poses = [(frames.flange_pose(position, q), False)]
        if may_turn:
            poses.append((frames.flange_pose(position, frames.half_turned(q)), True))
        candidates = []
        inverse_base = np.linalg.inv(base)
        for flange, half in poses:
            target = inverse_base @ frames.transform(flange)
            candidates += [(solution, half) for solution in self.planner.solve(start, target, seed_rows)]
        goals, outside = planning.rank_goals(candidates, start, spec.low, spec.high)
        floor = min(frames.TABLE_Z - TABLE_MARGIN_M, self.model.lowest_link(start, base) - 0.002)
        goals = [row for row in goals if self.model.lowest_link(row[0], base) >= floor]
        if not goals:
            return None, None, ("every joint solution for this pose needs a joint beyond its range or presses the arm on the table" if outside or candidates
                                else "no joint solution reaches this pose: it is out of reach or its orientation cannot be achieved there")
        reason = "no self-collision-free joint path leads from the current posture to this pose"
        for goal, half in goals[:planning.GOALS_TRIED]:
            path = self.planner.path(start, goal)
            if path is None:
                continue
            if planning.leaves_range(path, spec.low, spec.high):
                reason = "the only joint path found goes past a joint's range; move through an intermediate pose first"
                continue
            steps = planning.path_steps(path, spec.tracked_step)
            direct = planning.direct_steps(start, goal, spec.tracked_step)
            if steps > planning.DETOUR_RATIO * direct + planning.DETOUR_SLACK_STEPS:
                reason = (f"the only joint path found is a {steps:.0f}-step detour for a {direct:.0f}-step change of posture; "
                          "move through an intermediate pose first")
                continue
            path = path.copy()
            path[0] = start
            if self._hits_post(observation, arm, path, base):
                reason = f"on this path the arm would hit {self.post.text}, which stands between the arm bases; use another hand orientation or go through a point further out"
                continue
            return path, frames.half_turned(q) if half else frames.normalized(q), None
        return None, None, reason

    def _hits_post(self, observation, arm, path, base=None) -> bool:
        """Whether a joint path runs the arm into the fixed post, when it does not touch it already. The robot-only planner does not know the post."""
        if self.forced:
            return False
        base = self.base(observation, arm) if base is None else base
        opening = self._sphere_opening(observation, arm)
        if self.post.touches(self.model.spheres(path[0], base, opening)):
            return False
        travel = float(np.sum(np.max(np.abs(np.diff(path, axis=0)), axis=1))) if len(path) > 1 else 0.0
        count = max(2, min(OBJECT_MAX_SAMPLES, 1 + int(math.ceil(travel / OBJECT_SAMPLE_RAD))))
        return any(self.post.touches(self.model.spheres(planning.along(path, t), base, opening)) for t in np.linspace(0.0, 1.0, count)[1:])

    def _load_points(self, observation, arm):
        return self.loads.get(arm) if self.commanded_opening(observation, arm) < HOLD_OPENING else None

    def _bodies(self, joints, base, opening, load):
        """The arm's spheres and its load's points, N x 4."""
        rows = self.model.spheres(joints, base, opening)
        if load is None:
            return rows
        world = (self.model.flange(joints, base) @ load.T).T
        return np.vstack([rows, np.c_[world[:, :3], np.zeros(len(world))]])

    def object_hit(self, observation, arm, path):
        """The first measured object the arm or its load would run into along a joint path, or None. One it already touches where the path starts does not count, nor
        one it is meant to touch where the path ends (beside it), nor one the hand is coming down into: its footprint then holds both the grasp point and the end of the path."""
        boxes = self.scene.obstacles()
        if not boxes or self.forced:
            return None
        base = self.base(observation, arm)
        opening = self._sphere_opening(observation, arm)
        load = self._load_points(observation, arm)
        start = self._bodies(path[0], base, opening, load)
        finish = self._bodies(path[-1], base, opening, load)
        end = frames.grasp_point(frames.pose_of(self.model.flange(path[-1], base)))[0][:2]
        boxes = [box for box in boxes if not box.touches(start) and (box.holds(end) or not box.touches(finish))]
        if not boxes:
            return None
        travel = float(np.sum(np.max(np.abs(np.diff(path, axis=0)), axis=1))) if len(path) > 1 else 0.0
        count = max(2, min(OBJECT_MAX_SAMPLES, 1 + int(math.ceil(travel / OBJECT_SAMPLE_RAD))))
        for t in np.linspace(0.0, 1.0, count)[1:]:
            joints = planning.along(path, t)
            bodies = self._bodies(joints, base, opening, load)
            for box in boxes:
                if box.touches(bodies):
                    here = frames.grasp_point(frames.pose_of(self.model.flange(joints, base)))[0][:2]
                    if box.holds(end) and box.holds(here):
                        continue
                    return box
        return None

    def _reach_below(self, observation, arm, joints, base) -> float:
        """How far the hand or its load reaches below the grasp point at these joints."""
        opening = self._sphere_opening(observation, arm)
        flange = self.model.flange(joints, base)
        lowest = float(np.min(self.model.hand_clearance(joints, base, opening)))
        load = self._load_points(observation, arm)
        if load is not None:
            lowest = min(lowest, float(np.min((flange @ load.T)[2])))
        return float(frames.grasp_point(frames.pose_of(flange))[0][2]) - lowest

    def _over(self, observation, arm, start, here, q_here, position, q, height):
        """(joint path, None) straight up to a height, across it and straight down to a pose, or (None, reason)."""
        rows, joints = [start[None]], start
        if height - float(here[2]) > 0.005:
            path, reason = self.line_path(observation, arm, self.hops(here, q_here, np.r_[here[:2], height], q_here), start=joints)
            if path is None:
                return None, reason
            rows.append(path[1:])
            joints = path[-1]
        path, _, reason = self.free_path(observation, arm, np.r_[position[:2], height], q, start=joints)
        if path is None:
            return None, reason
        rows.append(path[1:])
        joints = path[-1]
        if height - float(position[2]) > 0.005:
            path, reason = self.line_path(observation, arm, self.hops(np.r_[position[:2], height], q, position, q), start=joints)
            if path is None:
                return None, reason
            rows.append(path[1:])
        return np.vstack(rows), None

    def asked_orientation(self, observation, arm, q):
        """The hand orientation that a given one means. A hand that took its load half a turn from the orientation asked for goes on that way, so that the load stays the way
        up it was picked, unless the given one was read off the hand as it is."""
        q = frames.normalized(q)
        if arm in self.half_held:
            here = self.grasp(observation, arm)[1]
            turned = frames.half_turned(q)
            if frames.angle_deg(here, turned) < frames.angle_deg(here, q):
                return turned
        return q

    def free_route(self, observation, arm, position, q, start=None, may_turn=False, q_from=None):
        """A free motion that does not run into what has been measured or dip into the table: (joint path, orientation it arrives with, None), or (None, None, reason). The planner's
        path when that is clear, otherwise straight up, across at a height that clears the object, and straight down. q_from is the hand's orientation at start, when that is not the present one."""
        path, arrived, reason = self.free_path(observation, arm, position, q, start=start, may_turn=may_turn)
        if path is None or self.forced:
            return path, arrived, reason
        if self.may_hold(observation, arm) and abs(float(path[-1][-1] - path[0][-1])) > SPIN_LIMIT_RAD:
            turn = abs(float(path[-1][-1] - path[0][-1]))
            return None, None, (f"carrying it this way would spin the wrist {math.degrees(turn):.0f} degrees, because this orientation lies past the end of the wrist's range. Use the "
                                "orientation half a turn away if the object allows it, or put it down and grasp it with the wrist turned the other way; force=True goes anyway")
        if self.may_hold(observation, arm) and frames.angle_deg(frames.half_turned(self.grasp(observation, arm)[1] if q_from is None else q_from), arrived) <= HALF_TURN_NEAR_DEG:
            return None, None, ("this orientation is half a turn from the hand's now, about its pointing axis, so what it holds would come out turned over or end for end. To keep it as it is, "
                                "give the hand's present orientation or none, or symmetry=2 where that is offered and the object looks the same; force=True makes the half turn")
        hit = self.object_hit(observation, arm, path)
        dip = self.table_refusal(observation, arm, path)
        if hit is None and dip is None:
            return path, arrived, None
        position = np.asarray(position, dtype=np.float64)
        base = self.base(observation, arm)
        here, q_here = frames.grasp_point(frames.pose_of(self.model.flange(path[0], base)))
        below = max(self._reach_below(observation, arm, path[0], base), self._reach_below(observation, arm, path[-1], base))
        why = None
        raised = 0.0
        for _ in range(OVER_TRIES):
            height = max((frames.TABLE_Z if hit is None else hit.top) + below + OVER_MARGIN_M, float(here[2]), float(position[2])) + raised
            over, why = self._over(observation, arm, path[0], here, q_here, position, arrived, height)
            if over is None:
                break
            again = self.object_hit(observation, arm, over)
            if again is not None:
                hit, why = again, None
            elif self.table_refusal(observation, arm, over) is None:
                return over, arrived, None
            else:
                raised += OVER_RAISE_M
        if hit is None:
            return None, None, dip
        self.counters["object_refusals"] += 1
        text = (f"the {arm} hand or its load would run into \"{hit.text}\" (id {hit.id}), measured at x={hit.center[0]:.3f}, y={hit.center[1]:.3f} with its top at z={hit.top:.3f}, and no way "
                "over it can be planned" + (f" ({why})" if why else "") + ". If it has moved, measure it again; force=True goes anyway")
        return None, None, text

    def home_path(self, observation, arm, start=None):
        """Joint path back to the episode's first joints: planned, or else the straight joint line."""
        start = self.joints(observation, arm) if start is None else np.asarray(start, dtype=np.float64)
        goal = self.joints_initial(arm)
        path = self.planner.path(start, goal)
        if path is None or planning.path_steps(path, self.spec.tracked_step) > (
                planning.DETOUR_RATIO * planning.direct_steps(start, goal, self.spec.tracked_step) + planning.DETOUR_SLACK_STEPS):
            path = np.linspace(start, goal, max(2, int(np.ceil(np.max(np.abs(goal - start)) / 0.05)) + 1))
        path = np.asarray(path, dtype=np.float64).copy()
        path[0] = start
        return path

    def home_route(self, observation, arm):
        """The way back to the first joints, going up and over what the direct path would run into: the table, the post or what has been measured. When no such way can
        be planned, the direct path: an arm can always try to go home."""
        path = self.home_path(observation, arm)
        hit = None if self.forced else self.object_hit(observation, arm, path)
        if self.forced or (hit is None and self.table_refusal(observation, arm, path) is None and not self._hits_post(observation, arm, path)):
            return path
        base = self.base(observation, arm)
        here, q_here = frames.grasp_point(frames.pose_of(self.model.flange(path[0], base)))
        position, q = frames.grasp_point(self.initial[arm + "_ee_pose"])
        below = max(self._reach_below(observation, arm, path[0], base), self._reach_below(observation, arm, path[-1], base))
        height = max((frames.TABLE_Z if hit is None else hit.top) + below + OVER_MARGIN_M, float(here[2]), float(position[2]))
        for raised in (0.0, OVER_RAISE_M):
            over, _ = self._over(observation, arm, path[0], here, q_here, position, q, height + raised)
            if over is not None and float(np.max(np.abs(over[-1] - path[-1]))) < 0.2 and self.object_hit(observation, arm, over) is None and self.table_refusal(observation, arm, over) is None:
                return np.vstack([over, path[-1][None]])
        return path

    def joints_initial(self, arm: str) -> np.ndarray:
        return np.asarray(self.initial[arm + "_arm_joint_state"], dtype=np.float64)

    def hops(self, start_p, start_q, end_p, end_q) -> list:
        """Grasp point and orientation waypoints of a straight line, the start excluded and the end included."""
        start_p, end_p = np.asarray(start_p, dtype=np.float64), np.asarray(end_p, dtype=np.float64)
        start_q, end_q = frames.normalized(start_q), frames.normalized(end_q)
        distance, angle = float(np.linalg.norm(end_p - start_p)), frames.angle_deg(start_q, end_q)
        n = max(1, int(math.ceil(max(distance / LINE_STEP_M, angle / LINE_STEP_DEG) - 1e-9)))
        return [(start_p + (end_p - start_p) * k / n, frames.slerp(start_q, end_q, k / n)) for k in range(1, n + 1)]

    def line_path(self, observation, arm, waypoints, start=None):
        """(joint path, None) through grasp-point waypoints [(position, orientation), ...], or (None, reason)."""
        self.counters["line_plans"] += 1
        start = self.joints(observation, arm) if start is None else np.asarray(start, dtype=np.float64)
        low, high = np.minimum(self.spec.low, start), np.maximum(self.spec.high, start)
        base = self.base(observation, arm)
        rows, q = [start], start
        for index, (position, orientation) in enumerate(waypoints):
            target = frames.transform(frames.flange_pose(position, orientation))
            solved = self.model.solve(target, q, base, low, high)
            where = f"waypoint {index + 1} of {len(waypoints)} of a straight motion"
            if solved is None:
                return None, f"no joint solution inside the arm's range near its posture at {where}"
            if float(np.max(np.abs(solved - q))) > HOP_JOINT_LIMIT_RAD:
                return None, f"the arm would swing at {where}"
            if wrist_folded(solved) and solved[3] < q[3]:
                return None, f"the wrist would fold onto the forearm at {where}"
            rows.append(solved)
            q = solved
        path = np.array(rows)
        if self._hits_post(observation, arm, path, base):
            return None, f"this straight motion would run the arm into {self.post.text}"
        return path, None

    def route(self, observation, arm, legs, paths=None):
        """Plan legs (name, "free" or "line", position, orientation) one after another without moving. None when all of them
        can be planned, otherwise (name of the first that cannot, reason). A free leg may arrive half-turned, as an executed
        one may; later legs then keep that hand. `paths`, when given, collects (name, joint path) of the planned legs."""
        joints = self.joints(observation, arm).copy()
        here, q_here = self.grasp(observation, arm)
        half = False
        may_turn = self.commanded_opening(observation, arm) >= TURN_MIN_OPENING
        for name, kind, position, q in legs:
            position = np.asarray(position, dtype=np.float64)
            q = frames.normalized(q)
            if half:
                q = frames.half_turned(q)
            if kind == "free":
                if float(np.linalg.norm(position - here)) > 0.003 or frames.angle_deg(q_here, q) > 1.0:
                    path, arrived, reason = self.free_route(observation, arm, position, q, start=joints, may_turn=may_turn, q_from=q_here)
                    if path is None:
                        return name, reason
                    if frames.angle_deg(arrived, q) > 1.0:
                        half, q = True, arrived
                    joints = path[-1]
                    if paths is not None:
                        paths.append((name, path))
            else:
                path, reason = self.line_path(observation, arm, self.hops(here, q_here, position, q), start=joints)
                if path is None:
                    return name, reason
                joints = path[-1]
                if paths is not None:
                    paths.append((name, path))
            here, q_here = position, q
        return None

    # -- checks along a motion ------------------------------------------------------------------
    def arm_gap(self, observation, paths, details=None) -> tuple[float, float]:
        """Smallest clearance while both arms follow `paths` together, each over its whole length in the same time; an arm
        without a path stays. Returns (smallest gap, gap now)."""
        now = {arm: self.joints(observation, arm) for arm in ARMS}
        bases = {arm: self.base(observation, arm) for arm in ARMS}
        openings = {arm: self._sphere_opening(observation, arm) for arm in ARMS}
        spheres = lambda arm, joints: self.model.spheres(joints, bases[arm], openings[arm])
        start = kinematics.gap(spheres("left", now["left"]), spheres("right", now["right"]))
        longest = max((len(p) for p in paths.values()), default=1)
        count = max(2, min(GAP_MAX_SAMPLES, 1 + (longest - 1) // GAP_SAMPLE_EVERY))
        least, reached = start, start
        for t in np.linspace(0.0, 1.0, count)[1:]:
            sets = {}
            for arm in ARMS:
                path = paths.get(arm)
                if path is None or len(path) < 2:
                    joints = now[arm] if path is None else path[-1]
                else:
                    joints = planning.along(path, t)
                sets[arm] = spheres(arm, joints)
            reached = kinematics.gap(sets["left"], sets["right"])
            least = min(least, reached)
        if details is not None:
            details["end"] = reached
        return least, start

    def gap_margin(self, observation, straight: bool) -> float:
        """Required clearance: more for an arm that may hold something of unknown size, unless the motion is straight: the two
        hands may mean to work close together, and the images follow such a motion closely."""
        holding = any(self.may_hold(observation, arm) for arm in ARMS)
        return MIN_GAP_M + (HOLDING_EXTRA_M if holding and not straight else 0.0)

    def gap_refusal(self, observation, paths, straight=False):
        """Why these paths must not run, or None. A motion that gets no closer than the arms already are is never refused."""
        details = {}
        least, start = self.arm_gap(observation, paths, details)
        need = self.gap_margin(observation, straight)
        if least >= need or least >= start - CLOSER_M:
            return None
        self.counters["arm_conflicts"] += 1
        where = "on the way" if details.get("end", least) >= need else "at the target"
        return (f"the two arms would come within {max(least, 0.0) * 100:.1f} cm of each other {where} (they are {max(start, 0.0) * 100:.1f} cm "
                "apart now). Move the other arm out of the way first, go through a point between, or pass force=True if the hands are "
                "meant to work this close; the arms may then touch.")

    def table_refusal(self, observation, arm, path):
        """Why a free path must not run: the arm would dip below the table surface on the way, deeper than where it starts."""
        base = self.base(observation, arm)
        opening = self._sphere_opening(observation, arm)
        lowest = lambda joints: min(float(np.min(self.model.hand_clearance(joints, base, opening))), self.model.lowest_link(joints, base))
        start = lowest(path[0])
        count = max(2, min(GAP_MAX_SAMPLES, 1 + (len(path) - 1) // GAP_SAMPLE_EVERY))
        worst = min(lowest(path[int(round(t * (len(path) - 1)))]) for t in np.linspace(0.0, 1.0, count)[1:])
        floor = frames.TABLE_Z - TABLE_MARGIN_M
        if worst >= floor or worst >= start - 0.002:
            return None
        return f"the {arm} arm would dip {(frames.TABLE_Z - worst) * 100:.1f} cm below the table surface on the way; lift it first or go through a point above"

    # -- native actions -------------------------------------------------------------------------
    def timed(self, paths: dict, careful: bool, empty_handed: bool = False, loaded: bool = False) -> tuple[dict, int]:
        """Joint targets per arm and the arm steps of a motion along `paths`. A careful motion moves at the tracked joint step. A free motion
        goes at the free joint step with empty hands, which have nothing to shake, and with a load at the slower carrying step, starting and stopping gently."""
        spec = self.spec
        if not paths:
            return {}, 0
        step = spec.tracked_step if careful else (spec.free_step if empty_handed else spec.carry_step)
        count = max(max(1, timing.steps_at(path, step)) for path in paths.values())
        if careful or not loaded:
            return {arm: timing.uniform(path, count) for arm, path in paths.items()}, count
        count += spec.carry_ramp
        return {arm: timing.ramped(path, count, spec.carry_ramp) for arm, path in paths.items()}, count

    def actions_for(self, observation, paths, openings, careful=False, empty_handed=False) -> tuple[list, int]:
        """Native actions: the arms travel together, then the grippers change on arrival. The other arm holds where it is."""
        loaded = any(self.may_hold(observation, arm) for arm in paths)
        targets, arm_steps = self.timed(paths, careful, empty_handed, loaded)
        grip_steps = 0
        for arm, value in openings.items():
            travel = abs(float(value) - self.reading(observation, arm))
            grip_steps = max(grip_steps, int(math.ceil(travel / self.spec.gripper_step)) if travel > 1e-9 else 0)
        total = arm_steps + grip_steps
        hold = self.bridge.hold_action(observation)
        actions = []
        for index in range(max(1, total)):
            data = {key: value.copy() for key, value in hold.items()}
            for arm, rows in targets.items():
                data[arm + "_arm_joint_state"] = np.asarray(rows[min(index, arm_steps - 1)], dtype=np.float32)
            if index >= arm_steps:
                for arm, value in openings.items():
                    data[arm + "_ee_joint_state"] = np.array([value], dtype=np.float32)
            actions.append(data)
        return actions, arm_steps

    def run(self, actions, arm_steps, openings=None, contact=False, settle=True) -> Execution:
        """Execute native actions and wait for them; hold the final command until the joints have settled."""
        run = self.bridge.execute(actions, contact=contact)
        if openings and run.executed > arm_steps and not run.reasons:
            self.bridge.grippers.update({arm: float(value) for arm, value in openings.items()})
        if settle and not run.reasons and run.executed == run.planned and arm_steps:
            last = actions[-1]
            for _ in range(SETTLE_MAX_STEPS):
                observation = self.bridge.current()
                off = max(float(np.max(np.abs(self.joints(observation, arm) - np.asarray(last[arm + "_arm_joint_state"], dtype=np.float64)))) for arm in ARMS)
                if off <= SETTLE_TOL_RAD:
                    break
                self.counters["settle_steps"] += 1
                settled = self.bridge.execute([last])
                if settled.reasons:
                    break
        return run

    def grip(self, commands: dict) -> dict:
        """Command grippers and hold until the official readings stop changing; returns per-arm readings."""
        observation = self.bridge.current()
        before = {arm: self.reading(observation, arm) for arm in commands}
        self.bridge.grippers.update({arm: float(value) for arm, value in commands.items()})
        self.half_held.difference_update(arm for arm, value in commands.items() if float(value) >= 0.5)
        history = {arm: [before[arm]] for arm in commands}
        steps = 0
        if not all(abs(before[arm] - float(value)) < 0.01 for arm, value in commands.items()):
            while steps < GRIP_MAX_STEPS:
                self.bridge.execute([self.bridge.hold_action(self.bridge.current())])
                steps += 1
                observation = self.bridge.current()
                for arm in commands:
                    history[arm].append(self.reading(observation, arm))
                if steps >= GRIP_MIN_STEPS and all(abs(rows[-1] - rows[-2]) < GRIP_STILL for rows in history.values()):
                    break
        result = {}
        for arm, value in commands.items():
            now = self.reading(observation, arm)
            row = {"command": round(float(value), 3), "reading": round(now, 4), "steps": steps}
            if float(value) < 0.5:
                row["blocked"] = bool(now > float(value) + HOLDING_READING)
                row["width_m"] = round(width_from_reading(now), 4)
            result[arm] = row
        return result
