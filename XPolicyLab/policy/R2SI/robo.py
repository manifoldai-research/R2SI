"""The `robo` object a code cell sees. Each method forwards one request to the agent thread; its docstring is the interface
documentation the model reads (prompt.py generates the reference from these signatures and docstrings).
"""
from __future__ import annotations


class Robo:
    """The robot, its cameras and the measurements made from them."""

    __slots__ = ("_channel",)

    def __init__(self, channel):
        self._channel = channel

    def __repr__(self):
        return "<robo>"

    def _send(self, method, **arguments):
        return self._channel.call(method, arguments)

    # -- reading the robot ------------------------------------------------------------------------
    def reference(self):
        """Inspect the core ICL-RAG reference selected automatically before the first decision. Returns status (ready or
        disabled), source/episode identity, recipe, image count, retrieval method and integrity hashes. When ready, the full
        frozen demonstration is already present before the live conversation on EVERY decision. Use it to understand a
        possible strategy, then measure your own scene. Source coordinates and candidate actions are not robot commands.
        The selection stays fixed for this episode; this call does not retrieve again or move the robot."""
        return self._send("reference")

    def state(self):
        """Current robot state and step counters: {"arms": {arm: {"position", "quaternion_wxyz", "opening", "width_m", "joints"}},
        "steps_used", "step_limit", "steps_left", "home_steps"}. opening is the official gripper reading; while closed on something
        it stays above 0, and width_m is the distance between the fingers it implies. home_steps estimates returning both arms."""
        return self._send("state")

    # -- seeing -----------------------------------------------------------------------------------
    def image(self, camera="cam_head"):
        """A copy of the current RGB image of a camera, uint8 H x W x 3: cam_head, cam_left_wrist or cam_right_wrist."""
        return self._send("image", camera=camera)

    def show(self, camera="cam_head", region=None, grid=False, points=None, objects=None, outline=None, rotate_deg=None):
        """Attach a view of the current image to your next message. region [left, top, right, bottom] in 0..1 enlarges a part;
        grid=True draws a labelled grid of full-image pixel coordinates; points draws numbered full-image pixels [u, v] so you
        can check them; objects draws the outlines and numbers of what segment or locate returned; outline draws a closed line through full-image pixels, to lay a shape
        over the image (robo.footprint and robo.project give them). Returns how crop pixels map to full-image pixels. rotate_deg turns the view counterclockwise, to read what lies at an
        angle; it then returns no such map."""
        return self._send("show", camera=camera, region=region, grid=grid, points=points, objects=objects, outline=outline, rotate_deg=rotate_deg)

    def segment(self, text, camera="cam_head", min_score=0.15):
        """Instances of a short noun phrase in one camera, best first: [{"id", "camera", "score", "pixel", "box", "area_px", "color"}].
        Describe what the thing looks like ("small round object", "long thin part"). A mask is not proof of identity: check it with show."""
        return self._send("segment", text=text, camera=camera, min_score=min_score)

    def mask(self, instance):
        """The H x W boolean mask of an instance from segment or of one camera's view of an object from locate (pass camera= inside
        the object via robo.mask({"id": obj["id"], "camera": camera}))."""
        return self._send("mask", instance=instance)

    def locate(self, text, cameras=None, on=None, min_score=0.15):
        """Find objects by a short noun phrase and measure them in the world from every camera that sees them, best score first:
        [{"id", "text", "score", "x", "y", "top_z", "bottom_z", "height_m", "length_m", "width_m", "long_side_deg", "body_length_m", "body_width_m", "cameras", "agreement",
        "pixels", "notes", "color"}]. x, y is the centre of the footprint; length and width are its sides near the top, long_side_deg the direction of the long
        side from +x toward +y (for a nearly square outline, of one of its sides), and the body sides those at half its height, wider when it tapers upward. It assumes the object stands on z=on (the table when None) and says so in notes when that seems wrong.
        cameras lists the cameras that agree on it: with one camera the height is a guess and the centre can be off by centimetres; agreement
        near 1 means the cameras explain each other's views (millimetres when a wrist camera is close). A wrist camera above the object measures best."""
        return self._send("locate", text=text, cameras=cameras, on=on, min_score=min_score)

    def locate_at(self, camera, pixel, on=None):
        """As locate, for the object under one full-image pixel [u, v] of a camera, when no phrase finds it. Returns one object or None."""
        return self._send("locate_at", camera=camera, pixel=pixel, on=on)

    def footprint(self, obj):
        """The world x, y points of an object's footprint, as a numpy array: its real outline, not only the bounding rectangle in length_m and width_m."""
        return self._send("footprint", obj=obj)

    def register(self, moving, fixed, symmetry=1):
        """The planar motion that lays one measured shape on another. Give two objects from locate or two footprints. Returns {"turn_deg", "shift", "residual_mm",
        "next_best_turn_deg", "next_best_residual_mm"}: turn the moving shape by turn_deg about its centre, then move its centre by shift [dx, dy]. A next best
        with a similar residual means the shape looks alike at that other turn, and the images must decide. symmetry n searches only 1/n of a full turn."""
        return self._send("register", moving=moving, fixed=fixed, symmetry=symmetry)

    def point(self, camera, pixel, height=None):
        """The world point where a pixel's ray meets the plane z=height (the table when None): {"point", "ray_origin", "ray_direction",
        "uncertainty_m"}. Only as good as the height."""
        return self._send("point", camera=camera, pixel=pixel, height=height)

    def triangulate(self, views):
        """The world point of one feature marked in two or three cameras: views=[{"camera", "pixel"}, ...]. Returns {"point", "ray_gap_mm",
        "crossing_sine", "reprojection_px", "ok"}: a large gap means the pixels do not mark the same point."""
        return self._send("triangulate", views=views)

    def project(self, points, camera="cam_head"):
        """Full-image pixels of world points in a camera: [{"pixel", "depth_m", "in_image"}]."""
        return self._send("project", points=points, camera=camera)

    def watch(self, camera, steps, every=5, regions=None, stop_after_quiet=None):
        """Hold both arms and sample one camera every few native steps. regions=[{"name", "region": [left, top, right, bottom]}] in 0..1 name the
        image parts to follow; the result tells per region when it showed activity and whether it ends different from its start, and a sheet
        of sampled frames is attached to your next message. stop_after_quiet ends the watch after that many quiet samples following activity. Pixel
        change only: an arm passing counts as activity. Costs its steps."""
        return self._send("watch", camera=camera, steps=steps, every=every, regions=regions, stop_after_quiet=stop_after_quiet)

    # -- planning without moving ------------------------------------------------------------------
    def reach(self, points, arm=None, quaternion_wxyz=None, lean_deg=None, clearance_m=0.0, approach="vertical"):
        """Which arm and hand orientation reach world grasp points from the current posture, planned at 0 steps. For each point and arm a list
        of {"hand", "quaternion_wxyz", "steps", "gap_to_other_arm_m"}; an empty list means none was found. hand "down" points straight down, closing
        along world x; "down_turned_N" turns that N degrees about the vertical; "leaning_N" leans N degrees away from the arm's base. clearance_m
        also requires a straight approach from that far back. With quaternion_wxyz, when that hand does not reach, "given_tilted_N" lists the same jaw line tilted N degrees. Only the robot is checked, never objects."""
        return self._send("reach", points=points, arm=arm, quaternion_wxyz=quaternion_wxyz, lean_deg=lean_deg, clearance_m=clearance_m, approach=approach)

    def estimate(self, plan):
        """Native steps of a list of motions planned one after another without moving, each starting where the previous one ends:
        [{"op": "move", "arm", "position", "quaternion_wxyz"}, {"op": "line", ...}, {"op": "grip", "arm", "opening"}, {"op": "home", "arm"}, {"op": "wait", "steps"}].
        Returns {"ok", "segments", "total_steps", "steps_left", "fits"}."""
        return self._send("estimate", plan=plan)

    def hold_point(self, arm, name="tip", point=None, views=None):
        """Remember a working point of a held object from a world point or from views as in triangulate.
        place, stroke and turn then take point_of=name and move or pivot that point instead of the grasp point. Measure it again after the object
        slips or the gripper opens. Returns the point and its offset from the grasp point."""
        return self._send("hold_point", arm=arm, name=name, point=point, views=views)

    # -- moving: every call below completes at most one small goal -----------------------------------
    def move(self, arm, position, quaternion_wxyz=None, opening=None, straight=False, slow=False, force=False, check=True):
        """Move the grasp point to a world position, orientation kept unless given; opening changes the gripper on arrival. A free motion takes the joint path the planner
        finds and goes up and over what robo.locate has measured when that path would run into it; if no way over can be planned it is refused, naming the object. It knows nothing of
        what has not been measured or has moved since. force=True skips this and the check between the arms. straight=True follows a straight line, whatever is in the way.
        slow=True goes at a third of the speed, for a doubtful hold, a long load or loose contents."""
        return self._send("move", arm=arm, position=position, quaternion_wxyz=quaternion_wxyz, opening=opening, straight=straight, slow=slow, force=force, check=check)

    def look(self, arm, point, distance_m=0.2, direction=None, slow=False, force=False, check=True):
        """Bring this arm's wrist camera to look straight at a world point from about distance_m, moving as in move; the orientation of the hand follows from that. It looks from above if the
        arm can, otherwise from as high as it can. direction [dx, dy, dz] asks to see the point from that side instead. Refused at 0 steps when no such pose can be planned. Returns as move, with
        the camera and seen_from_above_deg."""
        return self._send("look", arm=arm, point=point, distance_m=distance_m, direction=direction, slow=slow, force=force, check=check)

    def shift(self, arm, delta, slow=False, force=False, check=True):
        """Move the grasp point straight by [dx, dy, dz] metres, orientation kept. slow as in move."""
        return self._send("shift", arm=arm, delta=delta, slow=slow, force=force, check=check)

    def grip(self, arm, opening, check=True):
        """Set one gripper command (0 closes, 1 opens fully) and hold until the reading stops changing, 2 to 10 steps. Returns the reading, whether
        closing fingers were blocked by something, and the width between them."""
        return self._send("grip", arm=arm, opening=opening, check=check)

    def opening_for(self, width_m):
        """The gripper command that opens the jaw to a width in metres."""
        return self._send("opening_for", width_m=width_m)

    def hand(self, pointing=(0.0, 0.0, -1.0), closing=(-1.0, 0.0, 0.0)):
        """The quaternion_wxyz of a hand pointing along one world direction, its fingers closing along another (as near to it as is square to the first)."""
        return self._send("hand", pointing=pointing, closing=closing)

    def turned(self, quaternion_wxyz, axis, degrees):
        """A hand orientation turned by degrees about a world axis."""
        return self._send("turned", quaternion_wxyz=quaternion_wxyz, axis=axis, degrees=degrees)

    def pick(self, arm, target, quaternion_wxyz=None, z=None, approach="pointing", clearance_m=None, opening=1.0, lift_m=0.08, symmetry=1, check_hold=True,
             look=True, velocity=None, force=False, check=True):
        """One grasp: free motion to the pre-grasp point, straight approach, close, a short lift that checks the hold, then the lift. target is an object from
        locate or a world grasp point [x, y, z]. For an object it chooses the height and a straight-down hand closing across the short side, and at the
        pre-grasp point it measures the object again from the wrist camera and corrects the target (look=False skips this); z or quaternion_wxyz override
        them. The hand may take its orientation half a turn about its pointing axis, the same to the fingers and easier on the wrist: half_turned says so, and orientations you give
        while it holds are then taken the same way, so that the object does not turn over. approach "pointing" comes in along the hand's axis, "vertical" from straight above. symmetry n: the object looks the same after 1/n of a turn
        about the vertical (0: after any turn), so the hand takes the equivalent orientation nearest its current one. If the route cannot be planned it
        tries a shorter approach and a hand tilted about its closing axis before refusing at 0 steps. check_hold=False skips the reading check, for items whose
        hold does not show in the reading. velocity [vx, vy] in metres per native step is for a target that moves: give where it is now, and the hand meets it where it will be as the fingers close. On not_holding the hand
        has opened and backed off. Returns holding, reading, width_m, grasp_point, quaternion_wxyz (the hand as it is now) and which fallback was used."""
        return self._send("pick", arm=arm, target=target, quaternion_wxyz=quaternion_wxyz, z=z, approach=approach, clearance_m=clearance_m, opening=opening, lift_m=lift_m,
                          symmetry=symmetry, check_hold=check_hold, look=look, velocity=velocity, force=force, check=check)

    def place(self, arm, position, quaternion_wxyz=None, on=None, approach="vertical", clearance_m=0.06, retreat_m=0.06, symmetry=1, point_of=None, opening=1.0,
              slow=False, force=False, check=True):
        """One placement: rise to a travel height that clears what has been measured on the way, free motion above the target, straight descent, open, straight retreat. position is the world grasp point at
        release. With on=z, position may be [x, y]: the height follows from z (the support's surface) and how far the held object reached below the
        grasp point when it was picked, plus a few millimetres. point_of=name puts a remembered held point at position instead of the grasp point.
        A descent that stops short on its line (the load met its support) releases there. Refused when the height that clears what has been measured on the way cannot be planned, naming the object; force=True goes lower anyway.
        slow as in move, for the way there. Returns released_at, where the hand let go, and, when robo.pick took a measured object, placed: where it is now measured from every camera (it may be hidden by
        the hand, or have moved; a wrong height or orientation shows here)."""
        return self._send("place", arm=arm, position=position, quaternion_wxyz=quaternion_wxyz, on=on, approach=approach, clearance_m=clearance_m, retreat_m=retreat_m,
                          symmetry=symmetry, point_of=point_of, opening=opening, slow=slow, force=force, check=check)

    def stroke(self, arm, position, stroke, quaternion_wxyz=None, approach="pointing", clearance_m=0.04, opening=None, increment_m=0.01, withdraw=True, point_of=None, force=False, check=True):
        """One bounded straight contact stroke: push, press, slide or insert. Goes to just before position (the first contact point), moves in along the approach to it, then
        along stroke [dx, dy, dz] in small increments, stops where the arm is blocked, and withdraws. Returns achieved_m and blocked: blocked means the arm stopped following (contact),
        not that anything worked; look at the images."""
        return self._send("stroke", arm=arm, position=position, stroke=stroke, quaternion_wxyz=quaternion_wxyz, approach=approach, clearance_m=clearance_m, opening=opening,
                          increment_m=increment_m, withdraw=withdraw, point_of=point_of, force=force, check=check)

    def turn(self, arm, axis, degrees, frame="world", pivot=None, wrist_pivot=False, step_deg=15.0, point_of=None, slow=False, check=True):
        """Rotate the gripper about a pivot in checked increments; every increment is planned before anything moves. axis is "x", "y", "z" or a 3-vector in frame world or tool. The pivot
        is the grasp point unless pivot gives a world point, point_of a remembered held point, or wrist_pivot=True the wrist, so the hand swings around the wrist and the arm barely moves.
        step_deg sets how often the turn is checked, not its speed; slow as in move. Returns degrees_turned and pivot_drift_mm."""
        return self._send("turn", arm=arm, axis=axis, degrees=degrees, frame=frame, pivot=pivot, wrist_pivot=wrist_pivot, step_deg=step_deg, point_of=point_of, slow=slow, check=check)

    def both(self, left, right, force=False, check=True):
        """One small goal per arm, their stages running together in one native motion, which saves most of the steps of doing them in turn. left and right are
        dicts with op "pick", "place", "move", "line", "look", "stroke", "turn" or "home" and that call's arguments. Both are planned first and refused at 0 steps when an arm
        cannot reach or the arms would come too close. One arm's failure ends that arm only. Returns {"left": result, "right": result}."""
        return self._send("both", left=left, right=right, force=force, check=check)

    def home(self, arm="both", check=True):
        """Return an arm, or both together, to the initial pose and restore the initial gripper opening. An arm well below its starting height first draws back along its fingers.
        Release and clear what it holds first."""
        return self._send("home", arm=arm, check=check)

    def policy(self, text=None, chunks=1, check=True):
        """Hand both arms and grippers to the learned policy for up to chunks (at most 8) chunks of 24 native steps each. It sees the three cameras, the joints and one sentence: the instruction, unless text
        gives another. Calls in a row go on with its memory of them; after any other motion or another sentence it starts again from the current frame. Returns chunks done, where the arms are, and lagging_steps, how
        many steps each arm did not follow its commands (usually contact); stopped says why if it ended early. Everything measured before no longer counts: measure again."""
        return self._send("policy", text=text, chunks=chunks, check=check)

    def second_look(self, question=None):
        """An opinion from someone who has not seen your work. It gets only the instruction, the three cameras' current images and your question if you give one, and says in words what looks wrong, missing or unsure, or answers what you asked: {"opinion"}. It costs steps only as any long decision does; bring a wrist camera close to what matters first. It can be wrong: check what it says by measuring."""
        return self._send("second_look", question=question)

    def wait(self, steps=1):
        """Hold both arms for that many native steps. Put all you mean to hold in one call: each further cell costs a model decision."""
        return self._send("wait", steps=steps)
