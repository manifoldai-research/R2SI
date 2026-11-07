"""Small goals of one arm: a grasp, a placement, a contact stroke, a turn, a move, a return.

Every goal completes at most one small goal. It is planned before anything moves, so a goal that cannot be done is refused at
zero steps with the reason, and it then runs as a sequence of stages; the first stage that fails ends it and the result says where.
Goals run alone or one per arm together (`run`): the stages of the two arms combine into one native motion, and a gripper
stage waits until the other arm has no motion left before it, so both grippers act together when the sequences line up.

Only official robot state, the robot-only planner, the execution monitor and the agent's own measurements are read here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import control, frames, kinematics, planning
from .control import ARRIVE_DEG, ARRIVE_M, TURN_MIN_OPENING, Controller
from .frames import ARMS, UP

CLEARANCE_M = 0.06          # pre-grasp distance back along the approach
MAX_CLEARANCE_M = 0.20
LIFT_M = 0.08               # a grasp's total lift after closing: clear of the surface, cheap in steps
MAX_LIFT_M = 0.20
CHECK_LIFT_M = 0.02         # short first lift after which the hold is checked
RETREAT_M = 0.06            # straight retreat after a placement
SMALLER_CLEARANCES_M = (0.03, 0.015)     # a goal whose first stages cannot be planned tries these
TILTS_DEG = (15.0, -15.0, 30.0, -30.0, 45.0, -45.0)  # about the closing axis: the jaw still closes along the same line
LEVEL_JAW_SINE = 0.5        # a jaw closing this close to level can pass both sides of an object coming from above
STROKE_MAX_M = 0.20
STROKE_INCREMENT_M = 0.01
STROKE_MIN_INCREMENT_M = 0.005
STROKE_MAX_INCREMENT_M = 0.02
STROKE_CLEARANCE_M = 0.04
BLOCKED_PROGRESS = 0.5      # a stroke increment that advanced less than this share was blocked
TURN_STEP_DEG = 15.0
TURN_MIN_STEP_DEG = 3.0
TURN_MAX_STEP_DEG = 30.0
TURN_MAX_DEG = 180.0
HOME_LOW_M = 0.05           # an arm this far below its starting height first draws back along its fingers ...
HOME_LIFT_M = 0.06          # ... by at most this much
HOME_MIN_GAP_M = 0.15       # both arms return together only this far apart and uncrossed
GAP_SAMPLES = 6             # clearance checks along each pair of stages of two goals before they start
REFINE_MAX_MOVE_M = 0.03    # a look at the pre-grasp point may move the target by at most this
REFINE_MIN_MOVE_M = 0.004   # smaller corrections are noise
REFINE_SHIFT_FIRST_M = 0.008  # a larger correction first moves the pre-grasp point over the new target
REFINE_MIN_ASPECT = 1.4     # an object this elongated sets the jaw direction
REFINE_MIN_TURN_DEG = 4.0
REFINE_MAX_TURN_DEG = 35.0  # a larger difference means the caller chose this direction on purpose
REFINE_VERTICAL_DEG = 15.0  # corrections apply to a hand pointing about straight down
LEAD_TRIES = 3              # a moving target: where it will be and how long the hand takes to get there settle in this many rounds
LEAD_SETTLE_STEPS = 2       # each stage's motion ends with about this many settling steps
LEAD_CLOSE_STEPS = 3        # the fingers meet the target this many steps after they start to close


# -- stages ---------------------------------------------------------------------------------------
@dataclass
class Free:
    """A planned joint-space motion to a pose, optionally changing the gripper on arrival."""
    name: str
    position: np.ndarray
    q: np.ndarray
    opening: float | None = None
    may_turn: bool = False
    home: bool = False
    optional: bool = False
    slow: bool = False


@dataclass
class Line:
    """A straight motion of the grasp point through waypoints (the last one is the goal)."""
    name: str
    waypoints: list | None = None
    position: np.ndarray | None = None
    q: np.ndarray | None = None
    stop_short: bool = False     # stopping on the line before the goal means the fingers or the load met something; go on
    contact: bool = False        # a tracking deviation ends this stage, not the goal
    optional: bool = False
    careful: bool = True
    slow: bool = False


@dataclass
class Grip:
    name: str
    opening: float


@dataclass
class StageResult:
    ok: bool = True
    moved: bool = False
    steps: int = 0
    reason: str | None = None        # why the stage could not run or finish
    code: str | None = None          # unreachable, interrupted
    reasons: list = field(default_factory=list)
    stopped_short_mm: float | None = None
    position: np.ndarray | None = None
    q: np.ndarray | None = None
    reading: float | None = None
    blocked: bool | None = None
    width_m: float | None = None
    error_mm: float | None = None
    error_deg: float | None = None
    blocked_arm: bool = False        # after settling the arm still has not reached its command: going again would not help


class Goal:
    """One small goal of one arm. `plan` runs before anything moves and may refuse; `stages` then yields stages."""

    def __init__(self, ctl: Controller, arm: str):
        if arm not in ARMS:
            raise ValueError("arm must be left or right")
        self.ctl = ctl
        self.arm = arm
        self.values = {}
        self.refusal = None

    def plan(self, observation):
        return None

    def legs(self):
        """The motion legs in order, for checking two goals against each other before they start."""
        return []

    def stages(self):
        raise NotImplementedError
        yield

    def finished(self, ok=True, failure=None, reason=None, **values) -> dict:
        out = {"ok": bool(ok)}
        if failure:
            out["failure"] = failure
        if reason:
            out["reason"] = reason
        out.update(self.values)
        out.update(values)
        return out

    def refuse(self, code: str, reason: str, **values):
        self.refusal = self.finished(False, code, reason, moved=False, **values)
        return self.refusal


def _number(value, name, low, high) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number in [{low:g}, {high:g}]")
    return float(value)


def _flag(value, name) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _vector(value, n, name) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be {n} finite numbers") from None
    if array.size != n or not np.isfinite(array).all():
        raise ValueError(f"{name} must be {n} finite numbers")
    return array


def _symmetric(q, n, current):
    """symmetry n: the object looks the same after 1/n of a turn about the vertical (0: after any turn), so the gripper takes
    the equivalent orientation closest to its current one."""
    if type(n) is not int or not 0 <= n <= 12:
        raise ValueError("symmetry must be an integer in 0..12 (0: any turn about the vertical, 1: none)")
    if n == 1:
        return q, None
    steps = 36 if n == 0 else n
    rows = [(360.0 * k / steps, frames.turned(q, UP, 360.0 * k / steps)) for k in range(steps)]
    turn, best = min(rows, key=lambda row: frames.angle_deg(current, row[1]))
    return best, round(turn if turn <= 180 else turn - 360.0, 1)


class Refused(Goal):
    """A goal already known not to be possible: it ends at zero steps with its reason."""

    def __init__(self, ctl, arm, reason: str):
        super().__init__(ctl, arm)
        self.refuse("unreachable", reason)


class Move(Goal):
    """Go to a pose. Free motion by default; straight=True keeps the grasp point on a line. exact=True keeps an open gripper from arriving half-turned."""

    def __init__(self, ctl, arm, observation, position, quaternion=None, opening=None, straight=False, slow=False, exact=False):
        super().__init__(ctl, arm)
        self.slow = _flag(slow, "slow")
        q_now = ctl.grasp(observation, arm)[1]
        self.position = _vector(position, 3, "position")
        self.q = q_now if quaternion is None else ctl.asked_orientation(observation, arm, _vector(quaternion, 4, "quaternion"))
        self.opening = None if opening is None else _number(opening, "opening", 0.0, 1.0)
        self.straight = bool(straight)
        self.held = ctl.holding(observation, arm) and arm not in ctl.unseen_hold
        # An open gripper is the same turned half a turn about its pointing axis, and may need far less joint travel that way.
        self.may_turn = not exact and quaternion is not None and ctl.commanded_opening(observation, arm) >= TURN_MIN_OPENING

    def plan(self, observation):
        blocked = self.ctl.route(observation, self.arm, self.legs())
        if blocked is not None:
            return self.refuse("unreachable", f"nothing moved: {blocked[1]}")
        return None

    def legs(self):
        return [("move", "line" if self.straight else "free", self.position, self.q)]

    def stages(self):
        if self.straight:
            done = yield Line("move", position=self.position, q=self.q, careful=False, slow=self.slow)
        else:
            done = yield Free("move", self.position, self.q, opening=self.opening, may_turn=self.may_turn, slow=self.slow)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="move", interrupted_by=done.reasons or None)
        if self.held and self.ctl.reading(self.ctl.bridge.current(), self.arm) <= control.HOLDING_READING:
            return self.finished(False, "not_holding", "the gripper reading has fallen to what an empty hand reads: the hold was probably lost during this motion. Look: what still follows the hand is still held",
                                 stage="move", moved=True)
        if self.straight and self.opening is not None:
            grip = yield Grip("grip", self.opening)
            return self.finished(True, reading=grip.reading)
        return self.finished(True, error_mm=done.error_mm, error_deg=done.error_deg)


class Pick(Goal):
    """One grasp: pre-grasp point, straight approach, close, short checked lift, lift."""

    def __init__(self, ctl, arm, observation, position, quaternion=None, approach="pointing", clearance_m=CLEARANCE_M, opening=1.0,
                 lift_m=LIFT_M, symmetry=1, check_hold=True, look=None, velocity=None, source=None, to=None):
        super().__init__(ctl, arm)
        self.source = source             # the measured object this grasp takes, if any
        self.to = None if to is None else _vector(to, 3, "to")   # the grasp point at a later set-down, to choose a hand that reaches both
        self.velocity = None if velocity is None else np.r_[_vector(velocity, 2, "velocity"), 0.0]
        self.closing_steps = 0           # native steps from the pre-grasp point to the fingers on a moving target
        self.p = _vector(position, 3, "position")
        q_now = ctl.grasp(observation, arm)[1]
        q = q_now if quaternion is None else frames.normalized(_vector(quaternion, 4, "quaternion"))
        q, turn = _symmetric(q, symmetry, q_now)
        if turn is not None:
            self.values.update(symmetry_turn_deg=turn)
        if approach not in ("pointing", "vertical"):
            raise ValueError("approach must be pointing or vertical")
        self.q, self.approach = q, approach
        self.clearance = _number(clearance_m, "clearance_m", 0.0, MAX_CLEARANCE_M)
        self.opening = _number(opening, "opening", 0.0, 1.0)
        self.lift = _number(lift_m, "lift_m", 0.0, MAX_LIFT_M)
        self.check = min(CHECK_LIFT_M, self.lift)
        if not isinstance(check_hold, bool):
            raise ValueError("check_hold must be true or false")
        # A very thin or soft item may not show in the reading; the images then decide.
        self.check_hold = check_hold
        self.look = look                 # callable(observation, point, q) -> refinement, or None
        self.anchor = self.p

    def _direction(self, q, kind):
        return frames.pointing_axis(q) if kind == "pointing" else -UP

    def _legs_for(self, q, clearance, kind, lift):
        q = frames.normalized(q)
        start = self.p - clearance * self._direction(q, kind)
        return [("pre-grasp", "free", start, q), ("approach", "line", self.p, q), ("lift", "line", self.p + lift * UP, q)]

    def plan(self, observation):
        """Plans the grasp; a moving target is met where it will be when the fingers reach it."""
        if self.velocity is None:
            return self._plan(observation)
        origin, lead = self.p, 0
        for _ in range(LEAD_TRIES):
            self.p = origin + lead * self.velocity
            refused = self._plan(observation)
            if refused is not None:
                return refused
            steps = self._steps_to_fingers(observation)
            if steps == lead:
                break
            lead = steps
        self.p = origin + lead * self.velocity
        self.anchor = self.p
        self.values["lead_steps"] = lead
        return None

    def _steps_to_fingers(self, observation) -> int:
        """Native steps from now until the closing fingers reach the target, and in `closing_steps` the part of them after the pre-grasp point."""
        paths = []
        self.ctl.route(observation, self.arm, self._legs_for(self.q, self.clearance, self.approach, self.lift)[:2], paths=paths)
        rows = {name: self.ctl.timed({self.arm: path}, False, name == "pre-grasp")[1] + LEAD_SETTLE_STEPS for name, path in paths}
        self.closing_steps = rows.get("approach", 0) + LEAD_CLOSE_STEPS
        return sum(rows.values()) + LEAD_CLOSE_STEPS

    def _place_legs(self, q, lift):
        """The carry and set-down that a later placement at `to` would need from the lifted pose, held as now."""
        top = self.p + lift * UP
        travel = max(float(top[2]), float(self.to[2]) + CLEARANCE_M)
        legs = []
        if travel - float(top[2]) > 0.005:
            legs.append(("carry up", "line", np.r_[top[:2], travel], q))
        return legs + [("carry over", "free", np.r_[self.to[:2], travel], q), ("set down", "line", self.to, q)]

    def _plan(self, observation):
        """As asked, with a shorter approach, then with the hand tilted about its closing axis; a level jaw may also come from
        straight above, the open fingers passing on both sides of the object. With `to`, the first of these from which the
        carry and the set-down at `to` can also be planned is taken; if none, the first that grasps (to_checked False).
        Refused at 0 steps when nothing can be planned."""
        ctl, arm = self.ctl, self.arm
        fallback = None
        choices = [(self.q, self.clearance, 0.0)]
        choices += [(self.q, c, 0.0) for c in SMALLER_CLEARANCES_M if c < self.clearance]
        choices += [(frames.tilted(self.q, t), self.clearance, t) for t in TILTS_DEG]
        kinds = [self.approach]
        if self.approach == "pointing" and abs(float(frames.closing_axis(self.q)[2])) <= LEVEL_JAW_SINE:
            kinds.append("vertical")
        blocked = None
        for kind in kinds:
            for q, clearance, tilt in choices:
                blocked = ctl.route(observation, arm, self._legs_for(q, clearance, kind, self.lift))
                if blocked is not None and blocked[0] == "lift":  # the grasp itself can be planned: lift as far as the arm allows
                    for shorter in (0.6 * self.lift, 0.3 * self.lift, self.check):
                        if ctl.route(observation, arm, self._legs_for(q, clearance, kind, shorter)) is None:
                            self.lift, blocked = shorter, None
                            self.values["lift_used_m"] = round(shorter, 3)
                            break
                if blocked is None:
                    if self.to is not None:
                        both = ctl.route(observation, arm, self._legs_for(q, clearance, kind, self.lift) + self._place_legs(frames.normalized(q), self.lift),
                                         hold_after="lift")
                        if both is not None:
                            if fallback is None:
                                fallback = (q, clearance, kind, tilt, self.lift, both)
                            continue
                        self.values["to_checked"] = True
                    return self._take(q, clearance, kind, tilt)
        if fallback is not None:
            q, clearance, kind, tilt, self.lift, why = fallback
            self.values["to_checked"] = False
            self.values["to_note"] = (f"no hand orientation tried reaches both this grasp and the set-down at to ({why[0]}: {why[1]}); "
                                      "grasped as it would be without to")
            return self._take(q, clearance, kind, tilt)
        other = "right" if arm == "left" else "left"
        other_can = any(ctl.route(observation, other, self._legs_for(frames.tilted(self.q, t), self.clearance, self.approach, self.lift)) is None
                        for t in (0.0, *TILTS_DEG[:2]))
        return self.refuse("unreachable", f"nothing moved: the {blocked[0]} of this grasp cannot be planned for this arm ({blocked[1]}), also not "
                           "tilted about the closing axis or with a shorter approach", other_arm_can=other_can)

    def _take(self, q, clearance, kind, tilt):
        if kind != self.approach:
            self.values["approach_used"] = kind
        if clearance != self.clearance:
            self.values["clearance_used_m"] = clearance
        if tilt:
            self.values["tilt_used_deg"] = tilt
        self.q, self.clearance, self.approach = frames.normalized(q), clearance, kind
        return None

    def legs(self):
        return self._legs_for(self.q, self.clearance, self.approach, self.lift)

    def _refine(self, observation):
        """A look at the object from the pre-grasp point: move the target over it and square the jaw to it, when that can
        still be planned from here. Returns whether the pre-grasp point should shift first."""
        if self.look is None:
            return False
        moving = self.velocity is not None   # looks from where the target is now, and goes straight on: a shift first would lose the time it takes
        found = self.look(observation, self.p - self.closing_steps * self.velocity if moving else self.p, self.q)
        row = {"seen": found is not None}
        self.values["refined"] = row
        if found is None:
            return False
        delta = np.asarray(found["delta"], dtype=np.float64)
        moved = float(np.linalg.norm(delta))
        moved = moved if REFINE_MIN_MOVE_M <= moved <= REFINE_MAX_MOVE_M else 0.0
        rotation = frames.matrix(self.q)
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(-rotation[2, 0])))))
        turn = 0.0
        if tilt <= REFINE_VERTICAL_DEG and found.get("long_side_rad") is not None and found["length"] >= REFINE_MIN_ASPECT * found["width"]:
            wanted = found["long_side_rad"] + math.pi / 2
            difference = (wanted - math.atan2(float(rotation[1, 1]), float(rotation[0, 1]))) % math.pi
            difference = difference - math.pi if difference > math.pi / 2 else difference
            turn = math.degrees(difference)
            turn = round(turn, 1) if REFINE_MIN_TURN_DEG <= abs(turn) <= REFINE_MAX_TURN_DEG else 0.0
        if not moved and not turn and found.get("height") is None:
            row.update(moved_mm=0.0)
            return False
        p = self.p + (np.r_[delta, 0.0] if moved else 0.0)
        if found.get("height") is not None:
            p = np.r_[p[:2], float(found["height"])]
            row["height"] = round(float(found["height"]), 4)
        q = frames.turned(self.q, UP, turn) if turn else self.q
        start = p - self.clearance * self._direction(q, self.approach)
        legs = [("shift", "line", start, q), ("approach", "line", p, q), ("lift", "line", p + self.lift * UP, q)]
        if self.ctl.route(observation, self.arm, legs) is not None:
            row.update(moved_mm=0.0, note="the corrected grasp cannot be planned from here")
            return False
        row.update(moved_mm=round(moved * 1000, 1), jaw_turn_deg=turn)
        self.p, self.q = p, q
        self.anchor = p
        return not moving and (moved >= REFINE_SHIFT_FIRST_M or bool(turn))

    def stages(self):
        ctl, arm = self.ctl, self.arm
        start = self.p - self.clearance * self._direction(self.q, self.approach)
        done = yield Free("pre-grasp", start, self.q, opening=self.opening, may_turn=True)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="pre-grasp", holding=False)
        if frames.angle_deg(done.q, self.q) > 1.0:
            self.values["half_turned"] = True
            ctl.half_held.add(arm)
        self.q = done.q
        observation = ctl.bridge.current()
        if self._refine(observation):
            start = self.p - self.clearance * self._direction(self.q, self.approach)
            shifted = yield Line("shift", position=start, q=self.q, optional=True, careful=False)
            if not shifted.ok:
                self.values["shift_skipped"] = shifted.reason
        done = yield Line("approach", position=self.p, q=self.q, stop_short=True, careful=False)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="approach", holding=False)
        if done.stopped_short_mm:
            self.values["stopped_short_mm"] = done.stopped_short_mm
            self.anchor = done.position
        closed = yield Grip("close", 0.0)
        self.values.update(reading=closed.reading, width_m=closed.width_m)
        if self.check_hold and not closed.blocked:
            return (yield from self._missed("the closing fingers met nothing"))
        if self.check:
            done = yield Line("check-lift", position=self.anchor + self.check * UP, q=self.q)
            if not done.ok:
                return self.finished(False, done.code, done.reason, stage="check-lift", holding=None)
            reading = ctl.reading(ctl.bridge.current(), arm)
            if self.check_hold and reading <= control.HOLDING_READING:
                return (yield from self._missed("the hold was lost during the first lift"))
        holding = True if self.check_hold else None
        if self.lift > self.check:
            done = yield Line("lift", position=self.anchor + self.lift * UP, q=self.q, optional=True)
            if not done.ok:
                self.values["lift_partial"] = done.reason
            reading = ctl.reading(ctl.bridge.current(), arm)
            if self.check_hold and reading <= control.HOLDING_READING:
                return self.finished(False, "not_holding", "the hold was lost during the lift", holding=False, stage="lift")
        reading = ctl.reading(ctl.bridge.current(), arm)
        return self.finished(True, holding=holding, grasp_point=np.round(self.anchor, 4).tolist(), quaternion_wxyz=np.round(self.q, 4).tolist(), reading=round(reading, 4),
                             width_m=round(control.width_from_reading(reading), 4))

    def _missed(self, reason):
        yield Grip("release", self.opening)
        back = self.anchor - self.clearance * self._direction(self.q, self.approach)
        yield Line("back-off", position=back, q=self.q, optional=True, careful=False)
        return self.finished(False, "not_holding", reason, holding=False)


class Place(Goal):
    """One placement: rise to a travel height, go above the target, straight descent, open, straight retreat."""

    def __init__(self, ctl, arm, observation, position, quaternion=None, approach="vertical", clearance_m=CLEARANCE_M, retreat_m=RETREAT_M,
                 symmetry=1, point_of=None, opening=1.0, slow=False, support=frames.TABLE_Z):
        super().__init__(ctl, arm)
        self.slow = _flag(slow, "slow")
        self.support = support           # the height of the surface it goes down on
        p = _vector(position, 3, "position")
        q_now = ctl.grasp(observation, arm)[1]
        q = q_now if quaternion is None else ctl.asked_orientation(observation, arm, _vector(quaternion, 4, "quaternion"))
        q, turn = _symmetric(q, symmetry, q_now)
        if turn is not None:
            self.values.update(symmetry_turn_deg=turn)
        if approach not in ("pointing", "vertical"):
            raise ValueError("approach must be pointing or vertical")
        if point_of is not None:
            p = p - frames.matrix(q) @ np.asarray(point_of, dtype=np.float64)
        self.p, self.q, self.approach = p, q, approach
        self.clearance = _number(clearance_m, "clearance_m", 0.0, MAX_CLEARANCE_M)
        self.retreat = _number(retreat_m, "retreat_m", 0.0, MAX_CLEARANCE_M)
        self.opening = _number(opening, "opening", 0.0, 1.0)
        self.travel = None
        self.anchor = p

    def _direction(self):
        return frames.pointing_axis(self.q) if self.approach == "pointing" else -UP

    def _legs_for(self, observation, travel, clearance):
        start, q_now = self.ctl.grasp(observation, self.arm)
        legs = []
        if self.approach == "vertical":
            if travel - float(start[2]) > 0.005:
                legs.append(("rise", "line", np.r_[start[:2], travel], q_now))
            above = np.r_[self.p[:2], travel]
        else:
            above = self.p - clearance * self._direction()
        return legs + [("above", "free", above, self.q), ("descend", "line", self.p, self.q)]

    def plan(self, observation):
        """The travel height is lowered when it is what cannot be reached, unless that would take the load through something measured. Refused at 0 steps when the hold has been
        lost or nothing can be planned."""
        ctl, arm = self.ctl, self.arm
        if (ctl.commanded_opening(observation, arm) < 0.5 and ctl.reading(observation, arm) <= control.HOLDING_READING and arm in ctl.bridge.grippers
                and arm not in ctl.unseen_hold):
            return self.refuse("not_holding", "nothing moved: the gripper reading shows nothing is held; check the images")
        start, _ = ctl.grasp(observation, arm)
        heights, tried, blocked = [], set(), None
        clearing, in_the_way = self._clearing_height(observation, start)
        if clearing is not None:
            heights.append((clearing, self.clearance))
        for clearance in [self.clearance] + [c for c in SMALLER_CLEARANCES_M if c < self.clearance]:
            heights.append((max(float(start[2]), float(self.p[2]) + clearance), clearance))
            heights.append((float(self.p[2]) + clearance, clearance))
        for travel, clearance in heights:
            if round(travel, 4) in tried:
                continue
            tried.add(round(travel, 4))
            blocked = ctl.route(observation, arm, self._legs_for(observation, travel, clearance))
            if blocked is None:
                if clearance != self.clearance or travel < float(start[2]) - 1e-3:
                    self.values["clearance_used_m"] = clearance
                self.travel, self.clearance = travel, clearance
                self._observation = observation
                return None
            if travel == clearing and not ctl.forced:
                return self.refuse("unreachable", f"nothing moved: carrying the load over \"{in_the_way.text}\" (id {in_the_way.id}, its top measured at z={in_the_way.top:.3f}) takes a travel height of z={clearing:.3f}, "
                                   f"which cannot be planned for this arm ({blocked[1]}). Going lower would run into it. Come from another side or with another hand, hold the load lower, or if it has moved measure it "
                                   "again; force=True goes lower anyway")
        return self.refuse("unreachable", f"nothing moved: the {blocked[0]} of this placement cannot be planned for this arm ({blocked[1]}), also not "
                           "with a lower travel height")

    def _clearing_height(self, observation, start):
        """The travel height that carries the load over every measured object between here and the target, and the tallest of them; (None, None) when none is in the way.
        A target over an object (a container, a support) counts: the load must come down into it from above."""
        ctl, arm = self.ctl, self.arm
        reach = ctl._reach_below(observation, arm, ctl.joints(observation, arm), ctl.base(observation, arm))
        boxes = []
        for box in ctl.scene.obstacles():
            if box.holds(start[:2]):
                continue
            for share in np.linspace(0.0, 1.0, 16):
                if box.holds(start[:2] + share * (self.p[:2] - start[:2])):
                    boxes.append(box)
                    break
        if not boxes:
            return None, None
        tallest = max(boxes, key=lambda box: box.top)
        return max(tallest.top + reach + control.OVER_MARGIN_M, float(start[2]), float(self.p[2]) + self.clearance), tallest

    def legs(self):
        return self._legs_for(self._observation, self.travel, self.clearance)

    def stages(self):
        ctl, arm = self.ctl, self.arm
        start, q_now = ctl.grasp(ctl.bridge.current(), arm)
        if self.approach == "vertical":
            if self.travel - float(start[2]) > 0.005:
                done = yield Line("rise", position=np.r_[start[:2], self.travel], q=q_now, careful=False, slow=self.slow)
                if not done.ok:
                    return self.finished(False, done.code, done.reason, stage="rise")
            above = np.r_[self.p[:2], self.travel]
        else:
            above = self.p - self.clearance * self._direction()
        done = yield Free("above", above, self.q, slow=self.slow)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="above")
        reading = ctl.reading(ctl.bridge.current(), arm)
        if arm in ctl.bridge.grippers and ctl.bridge.grippers[arm] < 0.5 and reading <= control.HOLDING_READING and arm not in ctl.unseen_hold:
            return self.finished(False, "not_holding", "the gripper reading shows the hold was lost on the way; nothing was released here", stage="above")
        done = yield Line("descend", position=self.p, q=self.q, stop_short=True)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="descend")
        if done.stopped_short_mm:
            self.values["stopped_short_mm"] = done.stopped_short_mm
            self.anchor = done.position
        released = yield Grip("open", self.opening)
        if self.retreat:
            done = yield Line("retreat", position=self.anchor - self.retreat * self._direction(), q=self.q, optional=True, careful=False)
            if not done.ok:
                self.values["retreat_partial"] = done.reason
            elif self.opening < 1.0:
                yield Grip("open fully", 1.0)
        return self.finished(True, released_at=np.round(self.anchor, 4).tolist(), reading=released.reading)


class Stroke(Goal):
    """A bounded straight contact stroke from a first contact point: push, press, slide or insert. Stops where the arm is
    blocked, reports how far it got, and withdraws."""

    def __init__(self, ctl, arm, observation, position, stroke, quaternion=None, approach="pointing", clearance_m=STROKE_CLEARANCE_M, opening=None,
                 increment_m=STROKE_INCREMENT_M, withdraw=True, point_of=None):
        super().__init__(ctl, arm)
        p = _vector(position, 3, "position")
        q_now = ctl.grasp(observation, arm)[1]
        self.q = q_now if quaternion is None else ctl.asked_orientation(observation, arm, _vector(quaternion, 4, "quaternion"))
        if point_of is not None:
            p = p - frames.matrix(self.q) @ np.asarray(point_of, dtype=np.float64)
        self.p = p
        if approach not in ("pointing", "vertical"):
            raise ValueError("approach must be pointing or vertical")
        direction = frames.pointing_axis(self.q) if approach == "pointing" else -UP
        self.pre = p - _number(clearance_m, "clearance_m", 0.0, MAX_CLEARANCE_M) * direction
        self.d = _vector(stroke, 3, "stroke")
        self.length = float(np.linalg.norm(self.d))
        if not 0.001 <= self.length <= STROKE_MAX_M:
            raise ValueError(f"stroke must be 0.001 to {STROKE_MAX_M:g} m long")
        self.increment = _number(increment_m, "increment_m", STROKE_MIN_INCREMENT_M, STROKE_MAX_INCREMENT_M)
        self.opening = None if opening is None else _number(opening, "opening", 0.0, 1.0)
        if not isinstance(withdraw, bool):
            raise ValueError("withdraw must be true or false")
        self.withdraw = withdraw

    def legs(self):
        return [("approach", "free", self.pre, self.q), ("contact", "line", self.p, self.q), ("stroke", "line", self.p + self.d, self.q)]

    def plan(self, observation):
        blocked = self.ctl.route(observation, self.arm, self.legs())
        if blocked is not None:
            return self.refuse("unreachable", f"nothing moved: the {blocked[0]} of this stroke cannot be planned ({blocked[1]})")
        return None

    def stages(self):
        ctl, arm = self.ctl, self.arm
        done = yield Free("approach", self.pre, self.q, opening=self.opening)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="approach")
        done = yield Line("contact", position=self.p, q=self.q, stop_short=True)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="contact")
        achieved, blocked = 0.0, False
        count = int(math.ceil(self.length / self.increment - 1e-9))
        unit = self.d / self.length
        for k in range(1, count + 1):
            goal = self.p + self.d * min(1.0, k * self.increment / self.length)
            before = ctl.grasp(ctl.bridge.current(), arm)[0]
            step = yield Line("stroke", position=goal, q=self.q, contact=True)
            measured = ctl.grasp(ctl.bridge.current(), arm)[0]
            achieved = max(achieved, float((measured - self.p) @ unit))
            wanted = goal - before
            size = float(np.linalg.norm(wanted))
            progress = 1.0 if size < 1e-6 else float((measured - before) @ wanted) / size ** 2
            if not step.ok and step.code != "contact":
                return self.finished(False, step.code, step.reason, stage="stroke", achieved_m=round(achieved, 4))
            if step.code == "contact" or progress < BLOCKED_PROGRESS:
                blocked = True
                break
        if self.withdraw:
            back = yield Line("withdraw", position=self.pre, q=self.q, optional=True)
            if not back.ok:
                self.values["withdraw_partial"] = back.reason
        return self.finished(True, blocked=blocked, achieved_m=round(achieved, 4))


class Turn(Goal):
    """Rotate the gripper about a pivot in checked increments."""

    def __init__(self, ctl, arm, observation, axis, degrees, frame="world", pivot=None, wrist_pivot=False, step_deg=TURN_STEP_DEG, point_of=None, slow=False):
        super().__init__(ctl, arm)
        self.at_start = observation
        self.slow = _flag(slow, "slow")
        if isinstance(axis, str):
            if axis not in ("x", "y", "z"):
                raise ValueError("axis must be x, y, z or a 3-vector")
            axis = np.eye(3)["xyz".index(axis)]
        else:
            axis = _vector(axis, 3, "axis")
        if float(np.linalg.norm(axis)) < 1e-6:
            raise ValueError("the rotation axis must not be zero")
        if frame not in ("world", "tool"):
            raise ValueError("frame must be world or tool")
        degrees = _number(degrees, "degrees", -TURN_MAX_DEG, TURN_MAX_DEG)
        if abs(degrees) < 0.5:
            raise ValueError("degrees must be at least 0.5 in magnitude")
        step = _number(step_deg, "step_deg", TURN_MIN_STEP_DEG, TURN_MAX_STEP_DEG)
        p0, q0 = ctl.grasp(observation, arm)
        rotation = frames.matrix(q0)
        given = sum(x is not None and x is not False for x in (pivot, point_of, wrist_pivot or None))
        if given > 1:
            raise ValueError("give pivot, point_of or wrist_pivot, not more than one")
        if wrist_pivot:   # the flange: the hand swings the held object around the wrist and the arm barely moves
            center = frames.flange_pose(p0, q0)[:3]
        elif point_of is not None:
            center = p0 + rotation @ np.asarray(point_of, dtype=np.float64)
        elif pivot is not None:
            center = _vector(pivot, 3, "pivot")
        else:
            center = p0
        world = rotation @ axis if frame == "tool" else axis
        world = world / np.linalg.norm(world)
        n = int(math.ceil(abs(degrees) / step - 1e-9))
        self.poses = []
        for k in range(1, n + 1):
            turn = frames.about(world, math.radians(degrees * k / n))
            self.poses.append((center + frames.matrix(turn) @ (p0 - center), frames.normalized(frames.multiply(turn, q0))))
        self.start = (p0, q0)
        self.center = center
        self.offset = rotation.T @ (center - frames.flange_pose(p0, q0)[:3])
        self.values.update(axis=np.round(world, 4).tolist(), pivot=np.round(center, 4).tolist(), degrees_requested=degrees, increments=n)

    def legs(self):
        return [("turn", "line", *self.poses[-1])]

    def plan(self, observation):
        path, reason = self.ctl.line_path(observation, self.arm, self.poses)
        if path is None:
            return self.refuse("unreachable", f"nothing moved: this turn cannot be planned ({reason})")
        return None

    def stages(self):
        ctl, arm = self.ctl, self.arm
        done = yield Line("turn", waypoints=self.poses, careful=False, slow=self.slow)
        flange = np.asarray(ctl.bridge.current().state[arm + "_ee_pose"], dtype=np.float64)
        q = frames.normalized(flange[3:7])
        actual = flange[:3] + frames.matrix(q) @ self.offset
        self.values.update(degrees_turned=round(frames.angle_deg(q, self.start[1]), 1),
                           pivot_drift_mm=round(float(np.linalg.norm(actual - self.center)) * 1000, 1))
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="turn")
        if ctl.holding(self.at_start, arm) and arm not in ctl.unseen_hold and ctl.reading(ctl.bridge.current(), arm) <= control.HOLDING_READING:
            return self.finished(False, "not_holding", "the gripper reading shows the hold was lost during this turn", stage="turn")
        return self.finished(True)


class Home(Goal):
    """Back to the arm's initial pose and initial opening. An arm well below its starting height first draws back along its fingers, so an open hand leaves what is between them where it is."""

    def __init__(self, ctl, arm, observation):
        super().__init__(ctl, arm)
        self.p0, self.q0 = frames.grasp_point(ctl.initial[arm + "_ee_pose"])
        self.opening = float(ctl.initial[arm + "_ee_joint_state"][0])

    def legs(self):
        return [("home", "home", self.p0, self.q0)]

    def stages(self):
        ctl, arm = self.ctl, self.arm
        here, q = ctl.grasp(ctl.bridge.current(), arm)
        if here[2] < self.p0[2] - HOME_LOW_M:
            lifted = yield Line("lift", position=here - HOME_LIFT_M * frames.pointing_axis(q), q=q, optional=True, careful=False)
            self.values["lifted_first"] = lifted.ok
        done = yield Free("home", self.p0, self.q0, opening=self.opening, home=True, optional=True)
        if not done.ok:
            return self.finished(False, done.code, done.reason, stage="home")
        return self.finished(True, error_mm=done.error_mm, error_deg=done.error_deg)


# -- running goals --------------------------------------------------------------------------------
def _stage_error(ctl, observation, arm, stage):
    measured_p, measured_q = ctl.grasp(observation, arm)
    return float(np.linalg.norm(measured_p - stage_position(stage))), frames.angle_deg(measured_q, stage_q(stage))


def stage_position(stage):
    if isinstance(stage, Line) and stage.waypoints:
        return np.asarray(stage.waypoints[-1][0], dtype=np.float64)
    return np.asarray(stage.position, dtype=np.float64)


def stage_q(stage):
    if isinstance(stage, Line) and stage.waypoints:
        return stage.waypoints[-1][1]
    return stage.q


class Runner:
    """Runs goals: one, or one per arm with their stages combined."""

    def __init__(self, ctl: Controller):
        self.ctl = ctl

    def _plan_stage(self, observation, arm, stage):
        """(joint path, orientation it arrives with, reason)."""
        ctl = self.ctl
        if isinstance(stage, Free):
            if stage.home:
                return ctl.home_route(observation, arm), frames.normalized(stage.q), None
            here, q_here = ctl.grasp(observation, arm)
            if (stage.opening is None and not stage.may_turn and float(np.linalg.norm(stage.position - here)) <= 0.003
                    and frames.angle_deg(q_here, stage.q) <= 1.0):
                return np.array([ctl.joints(observation, arm)] * 2), frames.normalized(stage.q), None
            may_turn = stage.may_turn and ctl.commanded_opening(observation, arm) >= TURN_MIN_OPENING
            return ctl.free_route(observation, arm, stage.position, stage.q, may_turn=may_turn)
        waypoints = stage.waypoints
        if waypoints is None:
            here, q_here = ctl.grasp(observation, arm)
            if float(np.linalg.norm(stage.position - here)) < 5e-4 and frames.angle_deg(q_here, stage.q) < 0.2:
                return np.array([ctl.joints(observation, arm)] * 2), frames.normalized(stage.q), None
            waypoints = ctl.hops(here, q_here, stage.position, stage.q)
        path, reason = ctl.line_path(observation, arm, waypoints)
        return path, (None if path is None else frames.normalized(stage_q(stage) if stage.waypoints else stage.q)), reason

    def _motion(self, stages: dict, force=False) -> dict:
        """One native motion combining the next stage of each arm; returns {arm: StageResult}."""
        ctl = self.ctl
        observation = ctl.bridge.current()
        results, paths, arrival, openings = {}, {}, {}, {}
        for arm, stage in stages.items():
            path, q, reason = self._plan_stage(observation, arm, stage)
            if path is None:
                ctl.counters["refusals"] += 1
                results[arm] = StageResult(ok=False, code="unreachable", reason=reason)
                continue
            paths[arm], arrival[arm] = path, q
            if isinstance(stage, Free) and stage.opening is not None:
                openings[arm] = stage.opening
        if not paths:
            return results
        lines = {arm for arm in paths if isinstance(stages[arm], Line)}
        if not force and not any(isinstance(stages[arm], Free) and stages[arm].home for arm in paths):
            reason = ctl.gap_refusal(observation, paths, straight=len(lines) == len(paths))
            if reason is not None:
                ctl.counters["refusals"] += 1
                results.update({arm: StageResult(ok=False, code="unreachable", reason=reason) for arm in paths})
                return results
            for arm in paths:
                if isinstance(stages[arm], Free):
                    reason = ctl.table_refusal(observation, arm, paths[arm])
                    if reason is not None:
                        ctl.counters["refusals"] += 1
                        results.update({a: StageResult(ok=False, code="unreachable", reason=reason) for a in paths})
                        return results
        # A straight motion is careful while it holds something or descends; free motions may use the speed profile. A caller may ask any motion to go slowly.
        careful = any(stages[arm].slow and ctl.may_hold(observation, arm) for arm in paths)   # an empty hand has nothing to shake
        for arm in lines:
            stage = stages[arm]
            if stage.careful and ctl.may_hold(observation, arm):
                careful = True
            if stage.careful and float(stage_position(stage)[2]) < float(ctl.grasp(observation, arm)[0][2]) - 0.01:
                careful = True
        empty = all(isinstance(stages[arm], Free) and not ctl.may_hold(observation, arm) for arm in paths)
        actions, arm_steps = ctl.actions_for(observation, paths, openings, careful, empty)
        contact = any(isinstance(stages[arm], Line) and stages[arm].contact for arm in paths)
        # Short increments of a contact stroke follow each other without waiting: their goals are absolute, so lag does not add up.
        run = ctl.run(actions, arm_steps, openings, contact=contact, settle=not contact)
        after = ctl.bridge.current()
        for arm in paths:
            stage = stages[arm]
            result = StageResult(moved=run.executed > 0, steps=run.executed, reasons=list(run.reasons), q=arrival[arm])
            error_m, error_deg = _stage_error(ctl, after, arm, _with_q(stage, arrival[arm]))
            result.error_mm, result.error_deg = round(error_m * 1000, 1), round(error_deg, 1)
            result.position = ctl.grasp(after, arm)[0]
            result.blocked_arm = bool(np.max(np.abs(ctl.joints(after, arm) - np.asarray(actions[run.executed - 1][arm + "_arm_joint_state"], dtype=np.float64))) > control.SETTLE_TOL_RAD) if run.executed else False
            hit = [r for r in run.reasons if r.endswith(arm)]
            if hit and contact:
                result.ok, result.code = False, "contact"
                result.reason = "the arm stopped following: it is blocked"
            elif hit and not (isinstance(stage, Line) and stage.stop_short):
                result.ok, result.code = False, "interrupted"
                result.reason = f"the arm stopped following its commands at {np.round(result.position, 3).tolist()}; {_what_may_block(ctl, observation, arm)}"
            elif error_m > ARRIVE_M or error_deg > ARRIVE_DEG:
                short = _stopped_short(stage, stage_position(stage), result.position, error_deg, ctl.grasp(observation, arm)[0])
                if isinstance(stage, Line) and stage.stop_short and short is not None:
                    result.stopped_short_mm = short
                elif not getattr(stage, "contact", False):
                    result.ok, result.code = False, "interrupted"
                    result.reason = f"the stage stopped {error_m * 100:.1f} cm and {error_deg:.0f} degrees from its goal, at {np.round(result.position, 3).tolist()}: {_what_may_block(ctl, observation, arm)}"
            results[arm] = result
        return results

    def _correct(self, stages, results, force=False) -> None:
        """A motion stage that missed its goal gets one correction toward the same goal."""
        missed = {arm: stage for arm, stage in stages.items()
                  if not results[arm].ok and results[arm].code == "interrupted" and results[arm].moved and not results[arm].blocked_arm and not getattr(stage, "contact", False)}
        for arm, stage in missed.items():
            if isinstance(stage, Line) and stage.waypoints:   # from wherever it stopped, straight to the stage's end
                stage = Line(stage.name, position=stage_position(stage), q=stage_q(stage), careful=stage.careful, slow=stage.slow)
            again = self._motion({arm: _with_q(stage, results[arm].q)}, force=force)
            fresh = again[arm]
            fresh.steps += results[arm].steps
            results[arm] = fresh

    def run(self, goals: dict, force=False) -> dict:
        """Run prepared goals, {arm: goal}; returns {arm: result dict}."""
        ctl = self.ctl
        running, outcomes, heads = {}, {}, {}

        def advance(arm, value):
            try:
                heads[arm] = running[arm].send(value)
            except StopIteration as stop:
                outcomes[arm] = stop.value
                heads.pop(arm, None)
                running.pop(arm, None)

        for arm, goal in goals.items():
            running[arm] = goal.stages()
            advance(arm, None)
        started = ctl.bridge.current().step
        while heads:
            motions = {arm: stage for arm, stage in heads.items() if isinstance(stage, (Free, Line))}
            if motions:
                results = self._motion(motions, force=force)
                self._correct(motions, results, force)
                for arm in motions:
                    result = results[arm]
                    if not result.ok and motions[arm].optional:
                        result.reason = result.reason or "not reached"
                    advance(arm, result)
                continue
            readings = ctl.grip({arm: stage.opening for arm, stage in heads.items()})
            for arm in list(heads):
                row = readings[arm]
                advance(arm, StageResult(steps=row["steps"], reading=row["reading"], blocked=row.get("blocked"), width_m=row.get("width_m")))
        used = ctl.bridge.current().step - started
        for outcome in outcomes.values():
            outcome.setdefault("moved", used > 0)
        return outcomes

    def check_pair(self, observation, goals: dict):
        """Why two goals cannot run together, or None: too little clearance along their stages in the order they would run."""
        ctl = self.ctl
        routes = {}
        for arm, goal in goals.items():
            paths = []
            legs = [(name, "free" if kind == "home" else kind, p, q) for name, kind, p, q in goal.legs()]
            blocked = ctl.route(observation, arm, legs, paths=paths)
            if blocked is not None:
                return f"the {arm} arm's {blocked[0]} cannot be planned ({blocked[1]}); nothing moved"
            routes[arm] = paths
        here = {arm: ctl.joints(observation, arm) for arm in ARMS}
        bases = {arm: ctl.base(observation, arm) for arm in ARMS}
        need = ctl.gap_margin(observation, straight=False)
        index = {arm: 0 for arm in ARMS}
        openings = {arm: control.FINGER_LEAD + ctl.reading(observation, arm) for arm in ARMS}
        while any(index[arm] < len(routes.get(arm, [])) for arm in ARMS):
            moving = {arm: routes[arm][index[arm]] for arm in ARMS if index[arm] < len(routes.get(arm, []))}
            for t in np.linspace(0.0, 1.0, GAP_SAMPLES):
                sets = {}
                for arm in ARMS:
                    leg = moving.get(arm)
                    if leg is None or len(leg[1]) < 2:
                        joints = here[arm] if leg is None else leg[1][-1]
                    else:
                        joints = planning.along(leg[1], t)
                    sets[arm] = ctl.model.spheres(joints, bases[arm], min(1.0, openings[arm]))
                value = kinematics.gap(sets["left"], sets["right"])
                if value < need:
                    where = " and ".join(f"{arm} {leg[0]}" for arm, leg in moving.items())
                    return (f"the two arms would come within {max(value, 0.0) * 100:.1f} cm of each other during {where}; nothing moved. "
                            "Do them one at a time, or send the arm that is in the way elsewhere first.")
            for arm, leg in moving.items():
                here[arm] = leg[1][-1]
                index[arm] += 1
        return None


def _what_may_block(ctl, observation, arm) -> str:
    if ctl.may_hold(observation, arm):
        return "the hand or its load may have met something, and what it holds may have moved or dropped: look before you open"
    return "something may block the arm"


def _with_q(stage, q):
    """The stage with the orientation it really arrives with (an empty gripper may arrive half-turned)."""
    if q is None or (isinstance(stage, Line) and stage.waypoints):
        return stage
    return replace(stage, q=q)


def _stopped_short(stage, goal, measured, error_deg, start):
    """mm by which the grasp point stopped short of the goal on the stage's own line, or None when it is off that line,
    turned, or went past or back."""
    if error_deg > ARRIVE_DEG:
        return None
    travel = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(travel))
    if length < 1e-6:
        return None
    rest = np.asarray(goal, dtype=np.float64) - np.asarray(measured, dtype=np.float64)
    along = float(rest @ (travel / length))
    lateral = float(np.linalg.norm(rest - along * travel / length))
    if not ARRIVE_M < along <= length or lateral > ARRIVE_M:
        return None
    return round(along * 1000, 1)
