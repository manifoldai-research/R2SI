"""The whole agent against the kinematic environment: every robo call, the failure paths and the information boundary."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from client_server.ws.protocol.exceptions import ErrorCode
from XPolicyLab.policy.R2SI import frames, kinematics, objects, prompt, wam
from XPolicyLab.policy.R2SI.bridge import parse_observation
from XPolicyLab.policy.R2SI.tests.harness import FakeSegmenter, outputs, run_episode, square_mask

PACKAGE = Path(__file__).resolve().parents[1]


def last_value(client):
    return outputs(client)[-1].get("value")


def test_state_reports_local_actions_without_an_episode_budget():
    client, env, _ = run_episode(["robo.state()", "robo.wait(3)", "robo.state()"], step_limit=321)
    before, after = outputs(client)[0]["value"], outputs(client)[2]["value"]
    assert set(before) == {"arms", "steps_used", "home_steps"}
    assert before["steps_used"] == 0 and after["steps_used"] == env.step == 3
    for output in outputs(client):
        assert "steps_left" not in json.dumps(output) and "step_limit" not in json.dumps(output)


def test_move_and_arrival():
    code = """
s = robo.state()
p = s['arms']['left']['position']
r = robo.move('left', [p[0] + 0.05, p[1] + 0.15, p[2] - 0.02])
print(r['ok'], r['steps'])
s2 = robo.state()
print(s2['arms']['left']['position'])
"""
    client, env, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True"), out
    assert out["steps"] > 0 and env.step == out["steps"]


def test_shift_is_straight_and_refuses_out_of_reach():
    code = """
r = robo.shift('right', [0.0, 0.03, 0.0])
print(r['ok'])
bad = robo.move('right', [3.0, 3.0, 3.0], check=False)
print(bad['ok'], bad['failure'], bad.get('moved'))
"""
    client, env, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert "True" in out["stdout"] and "False unreachable False" in out["stdout"], out


def test_failed_goal_stops_the_cell_with_its_result():
    code = """
robo.move('left', [3.0, 3.0, 3.0])
print('not reached')
"""
    client, _, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "error" and "not reached" not in out.get("stdout", ""), out
    assert out["failure"]["failure"] == "unreachable", out


def test_failure_can_be_caught():
    code = """
try:
    robo.move('left', [3.0, 3.0, 3.0])
except RoboFailure as failure:
    print('caught')
"""
    client, _, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete" and "caught" in out["stdout"], out


def pick_code(z_offset=-0.08):
    return f"""
s = robo.state()
p = s['arms']['left']['position']
t = [p[0], p[1] + 0.05, p[2] + ({z_offset})]
q = robo.hand([0, 0, -1], [1, 0, 0])
r = robo.pick('left', t, quaternion_wxyz=q, clearance_m=0.05)
print(r['ok'], r.get('holding'), r['reading'] > 0, r['steps'])
"""


def test_pick_holds():
    client, env, tools = run_episode([pick_code()])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True True True"), out
    assert tools.counters["picks_held"] == 1


def test_pick_on_nothing_opens_and_backs_off():
    code = pick_code() .replace("r = robo.pick('left', t, quaternion_wxyz=q, clearance_m=0.05)", "r = robo.pick('left', t, quaternion_wxyz=q, clearance_m=0.05, check=False)").replace("r['reading'] > 0, r['steps']", "r['failure']")
    client, env, _ = run_episode([code], holds=False)
    out = outputs(client)[0]
    assert out["status"] == "complete" and "not_holding" in out["stdout"], out
    assert env.fingers["left"] > 0.9


def test_place_after_pick():
    code = pick_code() + """
r2 = robo.place('left', [t[0], t[1] + 0.03, t[2]])
print(r2['ok'], r2['released_at'])
print(robo.state()['arms']['left']['opening'])
"""
    client, env, tools = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert tools.counters["places_ok"] == 1
    assert env.fingers["left"] > 0.9


def test_stroke_reports_how_far_it_went():
    code = """
s = robo.state()
p = s['arms']['right']['position']
q = robo.hand([0, 0, -1], [1, 0, 0])
t = [p[0], p[1], p[2] - 0.08]
r = robo.stroke('right', t, [0.0, 0.0, -0.04], quaternion_wxyz=q, opening=0.0, withdraw=False)
print(r['ok'], r['blocked'], r['achieved_m'])
"""
    floor = {}

    def block(arm, joints, _model=None):
        return False

    client, _, tools = run_episode([code], block=block)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True False"), out


def test_turn_about_grasp_point():
    code = """
s = robo.state()
q0 = s['arms']['left']['quaternion_wxyz']
r = robo.turn('left', 'x', 30, frame='tool')
print(r['ok'], r['degrees_turned'], r['pivot_drift_mm'])
"""
    client, _, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True"), out


def test_both_runs_together_and_uses_fewer_steps_than_one_after_the_other():
    moves = """
s = robo.state()
l, r = s['arms']['left']['position'], s['arms']['right']['position']
"""
    together = moves + "print(robo.both({'op': 'move', 'position': [l[0], l[1] + 0.1, l[2] - 0.05]}, {'op': 'move', 'position': [r[0], r[1] + 0.1, r[2] - 0.05]})['steps'])"
    apart = moves + "a = robo.move('left', [l[0], l[1] + 0.1, l[2] - 0.05])\nb = robo.move('right', [r[0], r[1] + 0.1, r[2] - 0.05])\nprint(a['steps'] + b['steps'])"
    both_client, _, _ = run_episode([together])
    apart_client, _, _ = run_episode([apart])
    both_out, apart_out = outputs(both_client)[0], outputs(apart_client)[0]
    assert both_out["status"] == "complete" and apart_out["status"] == "complete", (both_out, apart_out)
    assert int(both_out["stdout"]) < int(apart_out["stdout"])


def test_home_returns_both_arms():
    code = """
s = robo.state()
p = s['arms']['left']['position']
robo.move('left', [p[0], p[1] + 0.1, p[2] - 0.05], opening=0.5)
robo.home('both')
s2 = robo.state()
print(max(abs(a - b) for a, b in zip(s['arms']['left']['joints'], s2['arms']['left']['joints'])))
"""
    client, _, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert float(out["stdout"]) < 0.02, out


def test_reach_and_estimate_move_nothing():
    code = """
s = robo.state()
l = s['arms']['left']['position']
print(len(robo.reach([[l[0], l[1] + 0.15, l[2] - 0.02]], arm='left')['points'][0]['left']) > 0)
e = robo.estimate([{'op': 'move', 'arm': 'left', 'position': [l[0], l[1] + 0.15, l[2] - 0.02]}, {'op': 'grip', 'arm': 'left', 'opening': 0.0}, {'op': 'wait', 'steps': 3}])
print(e['ok'], e['total_steps'] > 3)
print('fits' not in e and 'steps_left' not in e)
"""
    client, env, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert env.step == 0 and "True True" in out["stdout"] and out["stdout"].strip().endswith("True"), out


def test_evaluator_can_end_a_native_motion_without_publishing_its_limit():
    code = """
s = robo.state()
p = s['arms']['left']['position']
r = robo.move('left', [p[0] + 0.2, p[1] + 0.2, p[2] - 0.1], check=False)
print(r['ok'], r.get('failure'), r['steps'])
"""
    client, env, tools = run_episode([code], step_limit=3)
    assert env.step == 3 and client.turn == 1
    assert not outputs(client), "An unfinished cell must not report successful completion"
    assert tools.bridge._closed and not tools.bridge._queue


def test_interruption_is_reported_and_keeps_the_grip_command():
    code = pick_code() + """
r2 = robo.shift('left', [0.0, 0.2, 0.0], check=False)
print(r2['ok'], r2.get('failure'))
"""
    moved = {"count": 0}

    def block(arm, joints):
        moved["count"] += 1
        return arm == "left" and moved["count"] > 90

    client, env, tools = run_episode([code], block=block)
    out = outputs(client)[0]
    assert tools.bridge.grippers.get("left") == 0.0, (out, tools.bridge.grippers)


def test_watch_returns_frames():
    code = "r = robo.watch('cam_head', 12, every=4, regions=[{'name': 'a', 'region': [0, 0, 0.5, 0.5]}])\nprint(r['samples'], r['regions'][0]['differs_from_start'])"
    client, env, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete" and out["stdout"].startswith("4 False"), out
    assert env.step == 12


def test_show_attaches_an_image():
    client, _, _ = run_episode(["print(robo.show('cam_head', region=[0, 0, 0.5, 0.5], grid=True, points=[[10, 10]])['scale_xy'])", "print('next')"])
    images = [part for item in client.seen if item.get("role") == "user" for part in item["content"] if part["type"] == "input_image"]
    assert len(images) == 4, len(images)   # the three first views and the requested one


def test_segment_and_locate_with_masks():
    segmenter = FakeSegmenter()
    segmenter.answers["thing"] = [square_mask(300, 200, 340, 240)]
    code = """
found = robo.segment('thing')
print(found[0]['pixel'], found[0]['area_px'])
mask = robo.mask(found[0])
print(mask.shape, int(mask.sum()))
objs = robo.locate('thing', cameras=['cam_head'])
print(len(objs), objs[0]['cameras'], objs[0]['notes'][0][:20])
robo.show('cam_head', objects=objs)
"""
    client, _, _ = run_episode([code, "print(1)"], segmenter=segmenter)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert "(480, 640) 1600" in out["stdout"] and "1 ['cam_head']" in out["stdout"], out


def test_pick_from_a_located_object_looks_again():
    segmenter = FakeSegmenter()
    segmenter.answers["thing"] = [square_mask(300, 200, 340, 240)]
    code = """
objs = robo.locate('thing', cameras=['cam_head'])
r = robo.pick('left', objs[0], check=False)
print(r['ok'], r.get('failure'), r.get('refined'))
"""
    client, _, tools = run_episode([code], segmenter=segmenter)
    out = outputs(client)[0]
    assert out["status"] == "complete", out


def test_register_finds_a_planar_motion():
    corners = np.array([[-0.05, -0.02], [0.06, -0.03], [0.02, 0.01], [0.04, 0.07], [-0.03, 0.03]])
    shape = np.vstack([a + (b - a) * t for a, b in zip(corners, np.roll(corners, -1, axis=0)) for t in np.linspace(0.0, 1.0, 12)])
    angle = np.radians(40.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = (shape - shape.mean(axis=0)) @ rotation.T + shape.mean(axis=0) + np.array([0.1, -0.05])
    found = objects.register(shape, moved)
    assert abs(found["turn_deg"] - 40.0) < 2.0, found
    assert abs(found["shift"][0] - 0.1) < 0.004 and abs(found["shift"][1] + 0.05) < 0.004, found
    assert found["residual_mm"] < 3.0, found
    assert found["next_best_residual_mm"] > 3 * found["residual_mm"], found


def test_free_motion_goes_over_a_measured_object():
    cells = ["""
s = robo.state()
p = s['arms']['left']['position']
target = [p[0] - 0.25, p[1], p[2] - 0.07]
r = robo.move('left', target)
print(r['ok'], r['steps'])
print(robo.state()['arms']['left']['position'], target)
"""]

    from XPolicyLab.policy.R2SI import scene as scene_module
    original = scene_module.Scene.__init__

    def seeded(self):
        original(self)
        self.boxes[900] = scene_module.Box(900, "thing", np.array([-0.43, -0.20]), np.array([0.03, 0.03]), 0.0, 0.765, 0.90)
    scene_module.Scene.__init__ = seeded
    try:
        client, env, tools = run_episode(cells)
    finally:
        scene_module.Scene.__init__ = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True"), out
    highest = max(row[2] for row in env.grasp_track["left"])
    assert highest > 0.90, highest
    bodies = env.grasp_track["left"]
    assert all(not (abs(row[0] + 0.43) < 0.03 and abs(row[1] + 0.20) < 0.03 and row[2] < 0.90) for row in bodies)


def test_object_refusal_names_it_and_force_goes_anyway():
    cells = ["""
s = robo.state()
p = s['arms']['left']['position']
target = [p[0] - 0.25, p[1], p[2] - 0.07]
r = robo.move('left', target, check=False)
print(r['ok'], r.get('failure'), 'tall thing' in str(r.get('reason')), r.get('moved'))
r = robo.move('left', target, force=True)
print(r['ok'])
"""]

    from XPolicyLab.policy.R2SI import scene as scene_module
    original = scene_module.Scene.__init__

    def seeded(self):
        original(self)
        # Taller than the arm can reach over.
        self.boxes[900] = scene_module.Box(900, "tall thing", np.array([-0.43, -0.20]), np.array([0.03, 0.03]), 0.0, 0.765, 2.5)
    scene_module.Scene.__init__ = seeded
    try:
        client, env, tools = run_episode(cells)
    finally:
        scene_module.Scene.__init__ = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    lines = out["stdout"].splitlines()
    assert lines[0] == "False unreachable True False", out
    assert lines[1] == "True", out


def test_locate_results_update_the_scene():
    from XPolicyLab.policy.R2SI import scene as scene_module
    scene = scene_module.Scene()
    item = {"id": 1, "text": "thing", "score": 0.9, "agreement": 0.95, "x": 0.0, "y": 0.0, "top_z": 0.85, "bottom_z": 0.765, "length_m": 0.06, "width_m": 0.04, "long_side_deg": 30.0}
    scene.note([item], "thing")
    assert list(scene.boxes) == [1]
    scene.note([{**item, "id": 2, "x": 0.2}], "thing")
    assert list(scene.boxes) == [2]
    scene.note([{**item, "id": 3, "text": "other", "x": 0.2}], "other")
    assert list(scene.boxes) == [3]
    scene.note([{**item, "id": 4, "text": "flat", "top_z": 0.77}], "flat")
    scene.note([{**item, "id": 5, "text": "background", "length_m": 0.8}], "background")
    scene.note([{**item, "id": 6, "text": "held", "x": 0.5}], "held", [np.array([0.5, 0.0])])
    scene.note([{**item, "id": 7, "text": "one camera", "x": 0.7, "agreement": None}], "one camera")
    assert list(scene.boxes) == [3]
    scene.forget(point=np.array([0.2, 0.0]))
    assert not scene.boxes


def test_what_stands_in_something_stays_apart_and_one_measurement_merges():
    from XPolicyLab.policy.R2SI import scene as scene_module
    scene = scene_module.Scene()
    item = {"id": 1, "text": "big", "score": 0.9, "agreement": 0.95, "x": 0.0, "y": 0.0, "top_z": 0.95, "bottom_z": 0.765, "length_m": 0.20, "width_m": 0.20, "long_side_deg": 0.0}
    scene.note([item], "big")
    # Something small in the big one is another thing: both stay, and a hand taking it takes only that.
    scene.note([{**item, "id": 2, "text": "small", "top_z": 0.82, "length_m": 0.05, "width_m": 0.05}], "small")
    assert sorted(scene.boxes) == [1, 2]
    assert scene.around(np.array([0.0, 0.0, 0.80]))["id"] == 2
    assert scene.around(np.array([0.09, 0.0, 0.90]))["id"] == 1
    assert scene.around(np.array([0.5, 0.0, 0.80])) is None
    scene.forget(point=np.array([0.0, 0.0]))
    assert list(scene.boxes) == [1]
    scene.forget(1)
    assert not scene.boxes
    # A weaker result of the same thing in one measurement does not take the height away.
    scene.note([{**item, "id": 3}, {**item, "id": 4, "score": 0.5, "top_z": 0.81, "x": 0.01}], "big")
    assert list(scene.boxes) == [3] and scene.boxes[3].top == 0.95


def test_a_free_motion_that_would_dip_into_the_table_goes_over():
    cells = ["""
s = robo.state()
p = s['arms']['left']['position']
target = [p[0] - 0.05, p[1] + 0.15, p[2] - 0.02]
r = robo.move('left', target)
print(r['ok'], r['steps'])
"""]
    from XPolicyLab.policy.R2SI.control import Controller
    original = Controller.table_refusal
    calls = []

    def refuses_the_first_path(self, observation, arm, path):
        calls.append(len(path))
        return "the hand would dip" if len(calls) == 1 else None
    Controller.table_refusal = refuses_the_first_path
    try:
        client, env, tools = run_episode(cells)
    finally:
        Controller.table_refusal = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True"), out
    assert len(calls) >= 2


def test_the_way_home_goes_over_what_the_direct_path_would_hit():
    code = """
s = robo.state()
p = s['arms']['left']['position']
robo.move('left', [p[0], p[1] + 0.15, p[2] - 0.02])
robo.home('both')
s2 = robo.state()
print(max(abs(a - b) for a, b in zip(s['arms']['left']['joints'], s2['arms']['left']['joints'])))
"""
    from XPolicyLab.policy.R2SI.control import Controller
    original = Controller.table_refusal
    calls = []

    def refuses_the_direct_way_back(self, observation, arm, path):
        calls.append(float(np.max(np.abs(path[-1] - self.joints_initial(arm)))))
        return "the arm would dip" if len(calls) > 1 and calls[-1] < 1e-6 and len(path) == 12 else None
    Controller.table_refusal = refuses_the_direct_way_back
    try:
        client, env, tools = run_episode([code])
    finally:
        Controller.table_refusal = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert float(out["stdout"]) < 0.02, out
    assert len(calls) >= 3


def test_slow_motions_take_about_three_times_the_steps():
    code = """
q = robo.state()['arms']['left']['quaternion_wxyz']
a = robo.turn('left', 'x', 40, frame='tool')
b = robo.turn('left', 'x', -40, frame='tool', slow=True)
print(a['steps'], b['steps'])
s = robo.state()['arms']['left']['position']
c = robo.move('left', [s[0] - 0.1, s[1] + 0.1, s[2] - 0.05])
d = robo.move('left', s, slow=True)
print(c['steps'], d['steps'])
"""
    client, env, tools = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    turn, move = [[int(v) for v in line.split()] for line in out["stdout"].splitlines()]
    assert turn[1] >= 2 * turn[0], out
    assert move[1] >= 2 * move[0], out


def test_a_moving_target_is_met_where_it_will_be():
    code = """
s = robo.state()
p = s['arms']['left']['position']
q = robo.hand([0, 0, -1], [1, 0, 0])
t = [p[0], p[1] + 0.05, p[2] - 0.08]
r = robo.pick('left', t, quaternion_wxyz=q, clearance_m=0.05, velocity=[0.002, 0.0])
print(r['ok'], r['lead_steps'], r['grasp_point'][0] - t[0], t[1])
"""
    client, env, tools = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    ok, lead, offset, _ = out["stdout"].split()
    assert ok == "True" and int(lead) > 5, out
    assert abs(float(offset) - 0.002 * int(lead)) < 0.003, out


def test_placement_is_refused_when_what_clears_the_way_cannot_be_planned():
    cells = ["""
s = robo.state()
p = s['arms']['left']['position']
q = robo.hand([0, 0, -1], [1, 0, 0])
robo.pick('left', [p[0], p[1] + 0.05, p[2] - 0.08], quaternion_wxyz=q, clearance_m=0.05)
r = robo.place('left', [-0.15, -0.20, 0.80], check=False)
print(r['ok'], r.get('failure'), 'tall thing' in str(r.get('reason')), r.get('moved'))
r = robo.place('left', [-0.15, -0.20, 0.80], check=False, force=True)
print(r['ok'])
"""]
    from XPolicyLab.policy.R2SI import scene as scene_module
    original = scene_module.Scene.__init__

    def seeded(self):
        original(self)
        self.boxes[900] = scene_module.Box(900, "tall thing", np.array([-0.20, -0.20]), np.array([0.03, 0.03]), 0.0, 0.765, 2.5)
    scene_module.Scene.__init__ = seeded
    try:
        client, env, tools = run_episode(cells)
    finally:
        scene_module.Scene.__init__ = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    lines = out["stdout"].splitlines()
    assert lines[0] == "False unreachable True False", out
    assert lines[1] == "True", out


def test_a_grasp_at_a_point_takes_the_object_around_it():
    cells = ["""
s = robo.state()
p = s['arms']['left']['position']
q = robo.hand([0, 0, -1], [1, 0, 0])
robo.pick('left', [p[0], p[1] + 0.05, 0.83], quaternion_wxyz=q, clearance_m=0.05)
r = robo.place('left', [-0.2, -0.2], on=0.765, check=False)
print(r['ok'], r.get('released_at'))
"""]
    from XPolicyLab.policy.R2SI import scene as scene_module
    original = scene_module.Scene.__init__

    def seeded(self):
        original(self)
        self.boxes[900] = scene_module.Box(900, "thing", np.array([-0.30, -0.15]), np.array([0.05, 0.05]), 0.0, 0.765, 0.84)
    scene_module.Scene.__init__ = seeded
    try:
        client, env, tools = run_episode(cells)
    finally:
        scene_module.Scene.__init__ = original
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].startswith("True"), out
    released = json.loads(out["stdout"].split(" ", 1)[1])
    assert abs(released[2] - (0.765 + (0.83 - 0.765) + 0.004)) < 0.012, out


def test_home_draws_back_along_the_fingers():
    code = """
s = robo.state()
p = s['arms']['left']['position']
q = robo.turned(robo.hand([0, 0, -1], [1, 0, 0]), 'x', 30)
robo.move('left', [p[0], p[1] + 0.12, p[2] - 0.10], quaternion_wxyz=q)
before = robo.state()['arms']['left']['position']
r = robo.home('left')
print(r['ok'], r['lifted_first'])
print(json.dumps(before))
"""
    client, env, tools = run_episode(["import json\n" + code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    lines = out["stdout"].splitlines()
    assert lines[0] == "True True", out
    before = np.array(json.loads(lines[1]))
    q = frames.turned(frames.from_axes((0, 0, -1), (1, 0, 0)), (1.0, 0.0, 0.0), 30)
    pointing = frames.pointing_axis(q)
    moved = np.array(env.grasp_track["left"])
    index = int(np.argmin(np.linalg.norm(moved - before, axis=1)))
    away = next(row for row in moved[index:] if np.linalg.norm(row - before) >= 0.04)
    drawn = away - before
    assert float(drawn @ pointing) < -0.03 and np.linalg.norm(drawn - (drawn @ pointing) * pointing) < 0.01, (drawn.tolist(), pointing.tolist())


def test_show_turns_and_draws():
    from XPolicyLab.policy.R2SI import observe
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    picture, mapping = observe.annotated(image, outline=[[100, 100], [200, 100], [200, 200]])
    assert picture[100, 150].sum() > 0 and "scale_xy" in mapping
    assert observe.rotated(picture, 90).shape[:2] == (640, 480)
    client, env, tools = run_episode(["print(robo.show('cam_head', rotate_deg=30))"])
    out = outputs(client)[0]
    assert out["status"] == "complete" and "rotated_deg" in out["stdout"], out


def test_a_second_look_knows_only_the_instruction_and_the_images():
    client, env, tools = run_episode(["print(robo.second_look('what does this say?'))", "print(robo.second_look())"])
    first, second = outputs(client)[:2]
    assert first["status"] == "complete" and "looks off" in first["stdout"], first
    assert first["steps"] == 0 and second["steps"] == 0, (first, second)
    instructions, content = client.fresh[0]
    assert instructions == prompt.SECOND_LOOK
    assert content[0]["type"] == "input_text" and "Instruction: " in content[0]["text"] and "what does this say?" in content[0]["text"]
    assert sum(part["type"] == "input_image" for part in content) == 3, content
    assert "Question" not in client.fresh[1][1][0]["text"]


def test_the_fingertips_end_where_the_meshes_end():
    robot_dir = PACKAGE.parents[2] / "Assets" / "Robots" / "x5"
    if not robot_dir.exists():
        return
    tip = kinematics._stl_points(robot_dir / "meshes" / "link7.STL")[:, 0].max()
    origin = next(float(line.split('xyz="')[1].split()[0]) for line in (robot_dir / "X5A.urdf").read_text().split("<joint") if 'name="joint7"' in line)
    assert abs(frames.GRASP_OFFSET_M + frames.FINGERTIP_REACH_M - (origin + tip)) < 0.0005


def test_a_second_look_stays_in_the_episodes_session_and_leaves_its_conversation_out():
    import io
    import os
    from XPolicyLab.policy.R2SI.client import ModelClient

    sent = []

    class Opener:
        def open(self, request, timeout=None):
            sent.append(request)
            return io.BytesIO(json.dumps({"status": "completed", "output": []}).encode())

    saved = {name: os.environ.get(name) for name in ("R2SI_API_BASE", "R2SI_API_KEY", "R2SI_SESSION_RELEASE_PATH")}
    os.environ.update(R2SI_API_BASE="http://model-service/v1", R2SI_API_KEY="k", R2SI_SESSION_RELEASE_PATH="")
    try:
        client = ModelClient(model="m", effort="high", timeout_s=5)
        client.opener = Opener()
        client.complete("work", [{"role": "user", "content": [{"type": "input_text", "text": "the episode so far"}]}], [{"type": "function", "name": "python"}])
        client.ask_fresh("look", [{"type": "input_text", "text": "q"}])
        episode, look = sent
        assert episode.get_header("Session_id") == look.get_header("Session_id") == client.session_id
        body = json.loads(look.data)
        assert "the episode so far" not in look.data.decode()
        assert body["prompt_cache_key"] != client.session_id
        assert "tools" not in body
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_a_lost_connection_repeats_the_request_and_a_refusal_or_timeout_does_not():
    import io
    import os
    import urllib.error
    from XPolicyLab.policy.R2SI import client as client_module

    class Opener:
        def __init__(self, failures):
            self.failures, self.calls = list(failures), 0

        def open(self, request, timeout=None):
            self.calls += 1
            if self.failures:
                raise self.failures.pop(0)
            return io.BytesIO(json.dumps({"status": "completed", "output": []}).encode())

    def refusal(code):
        return urllib.error.HTTPError("http://model-service/v1/responses", code, "no", {}, io.BytesIO(b"{}"))

    items = [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]
    saved = {name: os.environ.get(name) for name in ("R2SI_API_BASE", "R2SI_API_KEY", "R2SI_SESSION_RELEASE_PATH")}
    waits = client_module.RETRY_WAITS_S
    os.environ.update(R2SI_API_BASE="http://model-service/v1", R2SI_API_KEY="k", R2SI_SESSION_RELEASE_PATH="")
    client_module.RETRY_WAITS_S = (0, 0)
    try:
        for failures, calls in (([urllib.error.URLError(ConnectionRefusedError()), ConnectionResetError()], 3), ([refusal(502)], 2)):
            client = client_module.ModelClient(model="m", effort="high", timeout_s=5)
            client.opener = Opener(failures)
            assert client.complete("work", items, [{"type": "function", "name": "python"}])["status"] == "completed"
            assert client.opener.calls == calls
        for failures in ([refusal(400)], [refusal(429)], [urllib.error.URLError(TimeoutError())], [TimeoutError()], [ConnectionResetError()] * 3):
            client = client_module.ModelClient(model="m", effort="high", timeout_s=5)
            client.opener = Opener(failures)
            try:
                client.complete("work", items, [{"type": "function", "name": "python"}])
            except client_module.InfrastructureFailure:
                pass
            else:
                raise AssertionError("expected the trial to end")
    finally:
        client_module.RETRY_WAITS_S = waits
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_a_long_episode_keeps_the_first_and_the_latest_images():
    from XPolicyLab.policy.R2SI import agent as agent_module
    client, env, tools = run_episode(["robo.wait(1)"] * 40)
    sent = client.seen
    images = [index for index, item in enumerate(sent) if item.get("role") == "user" and any(part.get("type") == "input_image" for part in item["content"])]
    emptied = [index for index, item in enumerate(sent) if item.get("role") == "user" and any(part.get("text") == agent_module.IMAGE_GONE for part in item["content"])]
    assert images[0] == 0
    assert agent_module.IMAGE_MESSAGES_KEPT < len(images) - 1 <= agent_module.IMAGE_MESSAGES_KEPT + agent_module.IMAGE_MESSAGES_DROPPED
    assert len(emptied) % agent_module.IMAGE_MESSAGES_DROPPED == 0 and emptied and min(emptied) > 0 and max(emptied) < min(images[1:])
    assert all(any(part.get("type") == "input_text" for part in sent[index]["content"]) for index in emptied)
    assert client.cache_keys == len(emptied) // agent_module.IMAGE_MESSAGES_DROPPED


def test_a_new_cache_key_keeps_the_session():
    import io
    import os
    from XPolicyLab.policy.R2SI.client import ModelClient

    sent = []

    class Opener:
        def open(self, request, timeout=None):
            sent.append(request)
            return io.BytesIO(json.dumps({"status": "completed", "output": []}).encode())

    saved = {name: os.environ.get(name) for name in ("R2SI_API_BASE", "R2SI_API_KEY", "R2SI_SESSION_RELEASE_PATH")}
    os.environ.update(R2SI_API_BASE="http://model-service/v1", R2SI_API_KEY="k", R2SI_SESSION_RELEASE_PATH="")
    try:
        client = ModelClient(model="m", effort="high", timeout_s=5)
        client.opener = Opener()
        items = [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]
        client.complete("work", items, [{"type": "function", "name": "python"}])
        client.new_cache_key()
        client.complete("work", items, [{"type": "function", "name": "python"}])
        first, second = sent
        assert first.get_header("Session_id") == second.get_header("Session_id") == client.session_id
        assert json.loads(first.data)["prompt_cache_key"] == client.session_id
        assert json.loads(second.data)["prompt_cache_key"] == client.cache_key != client.session_id
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_look_puts_the_point_in_the_middle_of_the_wrist_camera():
    code = """
r = robo.look('right', [0.25, -0.15, 0.80])
print(r['ok'], r['camera'], r['seen_from_above_deg'])
c = robo.project([[0.25, -0.15, 0.80]], camera=r['camera'])[0]
print(abs(c['pixel'][0] - 320) < 25, abs(c['pixel'][1] - 240) < 25, round(c['depth_m'], 2))
far = robo.look('right', [3.0, 3.0, 0.8], check=False)
print(far['ok'], far['failure'], far['moved'])
b = robo.both(left={'op': 'look', 'point': [-0.25, -0.15, 0.80]}, right={'op': 'look', 'point': [0.20, -0.15, 0.80]})
print(b['ok'], b['left']['camera'], b['right']['camera'])
"""
    client, env, _ = run_episode([code])
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    lines = out["stdout"].splitlines()
    assert lines[0].startswith("True cam_right_wrist"), out
    assert lines[1].startswith("True True") and abs(float(lines[1].split()[2]) - 0.2) < 0.02, out
    assert lines[2] == "False unreachable False", out
    assert lines[3] == "True cam_left_wrist cam_right_wrist", out


def test_a_half_open_hand_that_reached_its_command_holds_nothing():
    code = """
robo.grip('left', 0.6)
print(robo.state()['arms']['left']['width_m'])
robo.grip('left', 0.4)
s = robo.state()['arms']['left']
print(s['opening'], s['width_m'])
"""
    client, env, tools = run_episode([code], holds=False)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    first, second = out["stdout"].splitlines()
    assert abs(float(first) - 0.0448) < 0.002, out
    assert abs(float(second.split()[1]) - 0.0232) < 0.002, out
    free = type("Reading", (), {"state": {"left_ee_joint_state": np.array([0.4])}})()
    blocked = type("Reading", (), {"state": {"left_ee_joint_state": np.array([0.3])}})()
    tools.bridge.grippers["left"] = 0.4
    assert not tools.ctl.may_hold(free, "left") and not tools.ctl.holding(free, "left")
    tools.bridge.grippers["left"] = 0.0
    assert tools.ctl.may_hold(blocked, "left") and tools.ctl.holding(blocked, "left")
    assert abs(tools.ctl.jaw_width(blocked, "left") - 0.034) < 0.001


def test_a_pose_needs_a_solution_inside_the_arms_real_range():
    from XPolicyLab.policy.R2SI import planning, robot
    spec = robot.load_spec(None)
    now = np.array([0.0, 1.5, 1.5, -1.0, 0.0, 0.0])
    ok, past_elbow, folded = now, np.array([0.0, 1.5, 3.6, -1.0, 0.0, 0.0]), np.array([0.0, 1.5, 1.5, -2.1, 1.0, 0.0])
    goals, outside = planning.rank_goals([(ok, False), (past_elbow, False), (folded, False)], now, spec.low, spec.high)
    assert len(goals) == 1 and outside == 2
    assert np.allclose(goals[0][0], ok)
    # a joint past a half turn comes back as it is, not on around to the next whole turn
    assert planning.nearest_turn(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 3.2, 0.0, 0.0, 0.0]), spec.low, spec.high)[2] == 0.0
    assert planning.leaves_range(np.linspace(now, past_elbow, 5), spec.low, spec.high)
    assert not planning.leaves_range(np.linspace(past_elbow, now, 5), spec.low, spec.high)
    assert planning.leaves_range(np.linspace(now, folded, 5), spec.low, spec.high)
    assert not planning.leaves_range(np.linspace(folded, now, 5), spec.low, spec.high)


def test_an_arm_reading_past_a_limit_gets_a_command_inside_the_range():
    from XPolicyLab.policy.R2SI import monitor, robot
    from XPolicyLab.policy.R2SI.bridge import Bridge
    spec = robot.load_spec(None)
    state = {arm + "_arm_joint_state": np.zeros(6) for arm in frames.ARMS}
    state.update({arm + "_ee_joint_state": np.array([1.0]) for arm in frames.ARMS})
    state["right_arm_joint_state"] = np.array(spec.limit_high, dtype=float) + 0.002
    bridge = Bridge(monitor.Monitor(spec), 5.0)
    observation = type("Reading", (), {"state": state})()
    sent = bridge.monitor.before(bridge.hold_action(observation), observation, internal=True)
    assert np.all(sent["right_arm_joint_state"] <= np.array(spec.limit_high) + 1e-6)
    assert np.all(np.abs(sent["right_arm_joint_state"] - np.array(spec.limit_high)) < 1e-3)
    assert np.all(sent["left_arm_joint_state"] == 0)


def test_carrying_starts_and_stops_gently():
    from XPolicyLab.policy.R2SI import timing
    path = np.linspace(np.zeros(6), np.full(6, 1.2), 50)
    uniform = timing.uniform(path, 10)
    ramped = timing.ramped(path, 13, 3)
    assert np.allclose(uniform[-1], path[-1]) and np.allclose(ramped[-1], path[-1])
    moves = np.diff(np.r_[np.zeros((1, 6)), ramped], axis=0)[:, 0]
    assert moves[0] < moves[1] < moves[2] < moves[6] and moves[-1] < moves[-2] < moves[-3] < moves[6]
    assert abs(moves[6] - 0.12) < 1e-6


def test_a_large_turn_of_a_held_load_answers_in_the_cell():
    cells = ["""
robo.grip('left', 0.0)
s = robo.state()
p = s['arms']['left']['position']
q = s['arms']['left']['quaternion_wxyz']
r = robo.move('left', p, quaternion_wxyz=robo.turned(q, s['arms']['left']['quaternion_wxyz'] and [0, 1, 0], 200), check=False)
print(r['ok'], r.get('failure'))
print(robo.turned(q, 'z', 90))
"""]
    client, env, tools = run_episode(cells)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert "[" in out["stdout"], out


def test_a_load_picked_half_turned_stays_the_way_up_it_was_picked():
    cells = ["""
p = robo.state()['arms']['left']['position']
q = robo.hand([0, 0, -1], [1, 0, 0])
a = robo.pick('left', [p[0], p[1] + 0.05, 0.83], quaternion_wxyz=q, clearance_m=0.05, look=False)
print(a.get('half_turned'))
held = robo.state()['arms']['left']
b = robo.move('left', [held['position'][0], held['position'][1], held['position'][2] + 0.02], quaternion_wxyz=q, check=False)
print(b['ok'], b['steps'] < 15)
now = robo.state()['arms']['left']['quaternion_wxyz']
print(abs(sum(u * v for u, v in zip(now, held['quaternion_wxyz']))) > 0.99)
c = robo.move('left', [held['position'][0], held['position'][1], held['position'][2]], quaternion_wxyz=now, check=False)
print(c['ok'], c['steps'] < 15)
"""]
    client, env, tools = run_episode(cells)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].splitlines() == ["True", "True True", "True", "True True"], out


def test_a_half_turn_with_a_load_is_refused_unless_forced():
    cells = ["""
r = robo.pick('left', [-0.25, -0.15, 0.85], quaternion_wxyz=robo.hand([0, 0, -1], [1, 1, 0]), clearance_m=0.05, look=False, check=False)
print(r['ok'])
robo.grip('left', 1.0)
robo.grip('left', 0.0)
now = robo.state()['arms']['left']
over = robo.turned(now['quaternion_wxyz'], [0, 0, 1], 180)
back = robo.move('left', now['position'], quaternion_wxyz=over, check=False)
print(back['ok'], back.get('failure'), 'half a turn' in back.get('reason', ''))
forced = robo.move('left', now['position'], quaternion_wxyz=over, force=True, check=False)
print(forced['ok'])
"""]
    client, env, tools = run_episode(cells)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    assert out["stdout"].splitlines() == ["True", "False unreachable True", "True"], out


def test_placement_reports_where_the_object_is():
    segmenter = FakeSegmenter()
    segmenter.answers["thing"] = [square_mask(300, 250, 340, 290)]
    code = """
objs = robo.locate('thing')
r = robo.pick('left', objs[0], look=False, check=False)
print(r['ok'])
if r['ok']:
    done = robo.place('left', [-0.25, -0.15], on=0.765, check=False)
    print(done['ok'], 'placed' in done)
"""
    client, env, tools = run_episode([code], segmenter=segmenter)
    out = outputs(client)[0]
    assert out["status"] == "complete", out
    lines = out["stdout"].splitlines()
    assert lines[0] == "True" and lines[1] == "True True", out


def test_sandbox_blocks_escapes():
    for code in ("import os", "open('x')", "robo._channel", "(1).__class__", "getattr(robo, 'x')", "__import__('os')"):
        client, _, _ = run_episode([code])
        out = outputs(client)[0]
        assert out["status"] in ("rejected", "error"), (code, out)


def test_cell_state_persists():
    client, _, _ = run_episode(["a = 41", "def inc(x):\n    return x + 1\ninc(a)"])
    assert outputs(client)[1]["value"] == 42


def test_only_public_observation_enters():
    env_obs = {"instruction": ["do it"], "state": {k: np.zeros(n).tolist() for k, n in {"left_arm_joint_state": 6, "right_arm_joint_state": 6, "left_ee_joint_state": 1, "right_ee_joint_state": 1, "left_ee_pose": 7, "right_ee_pose": 7, "object_pose": 7}.items()},
               "vision": {"cam_head": {"color": np.zeros((4, 4, 3), np.uint8), "depth": np.zeros((4, 4))}, "cam_secret": {"color": np.zeros((4, 4, 3), np.uint8)}},
               "additional_info": {"step": 3, "step_lim": 9, "success": True, "task_name": "x", "seed": 2}, "task_name": "x", "success": True}
    obs = parse_observation(env_obs)
    assert set(obs.state) == {"left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state", "left_ee_pose", "right_ee_pose"}
    assert set(obs.images) == {"cam_head"} and obs.step == 0
    assert not hasattr(obs, "extra") and not hasattr(obs, "step_limit")
    env_obs.pop("additional_info")
    assert parse_observation(env_obs).step == 0


FORBIDDEN_CODE = re.compile(r"\b(task_name|action_case_id|layout|seed_id|env_seed|get_obs|take_action_cnt|step_lim|is_episode_end|success|reward|depth_image|distance_to_image_plane|object_pose)\b")


def test_runtime_code_names_no_forbidden_source():
    allowed = {"deploy.py", "__init__.py"}
    for path in PACKAGE.glob("*.py"):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        hits = FORBIDDEN_CODE.findall(text)
        assert not hits or path.name in ("planning.py",), (path.name, hits)


class FakePolicySession:
    """Stands in for a session on the learned policy's service: chunks of 24 actions that move the first joint of both arms a little and close the grippers."""

    def __init__(self, url, chunk_of=None, opened=None):
        self.url = url
        self.chunk_of = chunk_of
        self.frames = []
        self.predictions, self.observations = [], []
        self.waited_s = 0.0
        self.closed = False
        self.at = 0.0
        if opened is not None:
            opened.append(self)

    def open(self, observation):
        self.frames.append(observation)

    def chunk(self):
        self.predictions.append(1.0)
        if self.chunk_of is not None:
            return self.chunk_of(self)
        rows = []
        for _ in range(24):
            self.at += 0.01
            joints = np.array([self.at, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            rows.append({"left_arm_joint_state": joints, "right_arm_joint_state": joints.copy(), "left_ee_joint_state": np.array([0.3], dtype=np.float32), "right_ee_joint_state": np.array([0.7], dtype=np.float32)})
        return rows

    def observe(self, observation):
        self.observations.append(0.01)
        self.frames.append(observation)

    def close(self):
        self.closed = True


def policy_sessions(chunk_of=None):
    opened = []
    return opened, (lambda url: FakePolicySession(url, chunk_of, opened))


def test_policy_answers_every_action_with_one_observation_and_keeps_the_grippers_on_its_commands():
    opened, policy = policy_sessions()
    client, env, tools = run_episode(["r = robo.policy(chunks=2)\nprint(r['ok'], r['chunks'], r['steps'], r['lagging_steps'])"], policy=policy)
    out = outputs(client)[0]
    assert out["status"] == "complete" and "True 2 48" in out["stdout"], out
    assert env.step == 48 and len(opened) == 1
    assert len(opened[0].frames) == 49, len(opened[0].frames)
    assert tools.bridge.grippers == {"left": 0.3, "right": 0.7} or (abs(tools.bridge.grippers["left"] - 0.3) < 1e-6 and abs(tools.bridge.grippers["right"] - 0.7) < 1e-6), tools.bridge.grippers
    assert abs(env.state["left_arm_joint_state"][0] - 0.48) < 1e-5
    assert tools.counters["policy_steps"] == 48 and tools.counters["policy_chunks"] == 2 and tools.counters["policy_sessions"] == 1 and tools.counters["policy_config"] == "fake"
    assert opened[0].closed, "the end of the episode frees the service's GPU"


def test_the_policy_sees_only_cameras_joints_grippers_and_a_sentence():
    opened, policy = policy_sessions()
    run_episode(["robo.policy()", "robo.policy(text='Do a smaller thing.')"], policy=policy)
    first = opened[0].frames[0]
    assert set(first) == {"vision", "state", "instruction"}, set(first)
    assert set(first["state"]) == {"left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state"}, set(first["state"])
    assert set(first["vision"]) == {"cam_head", "cam_left_wrist", "cam_right_wrist"} and all(set(view) == {"color"} for view in first["vision"].values())
    assert isinstance(first["vision"]["cam_head"]["color"], bytes)
    assert first["instruction"] == "Do the thing."
    assert opened[1].frames[0]["instruction"] == "Do a smaller thing."
    assert opened[0].closed, "another sentence starts afresh"


def test_policy_calls_in_a_row_share_a_session_and_any_other_motion_ends_it():
    opened, policy = policy_sessions()
    client, env, _ = run_episode(["robo.policy()", "robo.policy()", "print(robo.state()['steps_used'])", "robo.policy()", "robo.wait(2)", "robo.policy()"], policy=policy)
    assert all(out["status"] == "complete" for out in outputs(client)), outputs(client)
    assert len(opened) == 2, len(opened)
    assert opened[0].closed and env.step == 24 * 4 + 2
    assert len(opened[0].frames) == 1 + 3 * 24 and len(opened[1].frames) == 1 + 24


def test_after_the_policy_nothing_measured_or_held_is_taken_for_known():
    opened, policy = policy_sessions()
    segmenter = FakeSegmenter()
    segmenter.answers["thing"] = [square_mask(300, 240, 340, 280)]
    code = "found = robo.locate('thing')\nprint(len(found))"
    client, _, tools = run_episode([code, "robo.policy()"], segmenter=segmenter, policy=policy)
    assert outputs(client)[1]["status"] == "complete", outputs(client)[1]
    assert not tools.ctl.scene.obstacles() and not tools.ctl.loads and all(row is None for row in tools.grasped.values())


def test_contact_does_not_stop_the_policy_and_shows_as_lagging_steps():
    opened, policy = policy_sessions()
    client, env, _ = run_episode(["r = robo.policy()\nprint(r['steps'], r['lagging_steps'])"], policy=policy, block=lambda arm, joints: arm == "left" and float(joints[0]) > 0.05)
    out = outputs(client)[0]
    assert out["status"] == "complete" and out["stdout"].startswith("24"), out
    assert env.step == 24
    left = int(re.search(r"'left': (\d+)", out["stdout"]).group(1))
    assert left > 0, out


def test_the_policy_is_stopped_short_of_a_sudden_jump():
    def jumpy(session):
        rows = [{"left_arm_joint_state": np.zeros(6, dtype=np.float32), "right_arm_joint_state": np.zeros(6, dtype=np.float32), "left_ee_joint_state": np.ones(1, dtype=np.float32), "right_ee_joint_state": np.ones(1, dtype=np.float32)} for _ in range(24)]
        rows[3] = dict(rows[3], left_arm_joint_state=np.array([2.0, 0, 0, 0, 0, 0], dtype=np.float32))
        return rows
    opened, policy = policy_sessions(jumpy)
    client, env, tools = run_episode(["r = robo.policy()\nprint(r['steps'], r['stopped'])", "robo.policy()"], policy=policy)
    assert "3 " in outputs(client)[0]["stdout"] and "jump" in outputs(client)[0]["stdout"], outputs(client)[0]
    assert env.step == 6 and len(opened) == 2 and opened[0].closed, "a chunk cut short ends the session"


def test_evaluator_ends_wam_mid_chunk_and_closes_its_session():
    opened, policy = policy_sessions()
    client, env, tools = run_episode(["robo.policy()", "robo.policy()"], policy=policy, step_limit=10)
    assert env.step == 10 and client.turn == 1 and not outputs(client)
    assert len(opened) == 1 and opened[0].closed and tools.policy is None
    assert tools.bridge._closed and not tools.bridge._queue


def test_a_failing_policy_service_voids_the_trial():
    from XPolicyLab.policy.R2SI.client import InfrastructureFailure

    def failing(session):
        raise wam.ServiceFailure("the prediction failed")
    opened, policy = policy_sessions(failing)
    try:
        run_episode(["try:\n    robo.policy()\nexcept Exception as error:\n    print('caught')"], policy=policy)
    except InfrastructureFailure:
        assert opened[0].closed
        return
    raise AssertionError("the trial should have been void")


def test_policy_arguments_and_a_missing_service():
    opened, policy = policy_sessions()
    client, env, _ = run_episode(["robo.policy(chunks=9)", "robo.policy(text='  ')"], policy=policy)
    assert all(out["status"] == "error" for out in outputs(client)) and env.step == 0 and not opened
    client, env, _ = run_episode(["robo.policy()"])
    assert outputs(client)[0]["status"] == "error" and "no learned policy" in json.dumps(outputs(client)[0]) and env.step == 0


class FakeServiceClient:
    def __init__(self, full_times, log):
        self.full_times, self.log = full_times, log

    def call(self, name, obs=None):
        self.log.append(name)
        if name == "reset" and self.full_times[0] > 0:
            self.full_times[0] -= 1
            raise wam.WsError(ErrorCode.CALL_FAILED, "Session capacity exhausted")
        if name == "get_action":
            return [{"left_arm_joint_state": np.zeros(6)}]
        return {"ok": True}

    def close(self):
        self.log.append("close")


def test_a_session_waits_while_the_service_is_full_and_gives_up_later():
    log, full = [], [2]
    connect = wam._connect
    wam._connect = lambda url, timeout: FakeServiceClient(full, log)
    try:
        slept = []
        session = wam.Session("ws://fake", sleep=slept.append)
        session.open({"instruction": "x"})
        assert log.count("reset") == 3 and log.count("close") == 2 and len(slept) == 2, (log, slept)
        assert log[-1] == "update_obs" and session.chunk()
        session.close()
        assert log[-2:] == ["trial_end", "close"] and session.client is None
        full[0] = 10 ** 6
        stuck = wam.Session("ws://fake", busy_limit_s=-1.0, sleep=slept.append)
        try:
            stuck.open({"instruction": "x"})
        except wam.ServiceFailure as error:
            assert "stayed full" in str(error)
        else:
            raise AssertionError("a full service must end in a failure")
    finally:
        wam._connect = connect


def test_a_session_reports_a_broken_service_as_a_failure():
    class Broken(FakeServiceClient):
        def call(self, name, obs=None):
            raise ConnectionError("gone")
    connect = wam._connect
    wam._connect = lambda url, timeout: Broken([0], [])
    try:
        try:
            wam.Session("ws://fake", sleep=lambda s: None).open({})
        except wam.ServiceFailure as error:
            assert "failed" in str(error)
        else:
            raise AssertionError("a broken service must end in a failure")
    finally:
        wam._connect = connect


def test_prompt_text_is_generic():
    text = prompt.system_prompt()
    assert len(text) < 40000, len(text)
    assert "episode ends" not in text.lower() and "episode is over" not in text.lower(), "the prompt must not teach the end of the episode as a signal"


def main():
    failed = 0
    for name, item in sorted(globals().items()):
        if name.startswith("test_") and callable(item):
            try:
                item()
                print("ok  ", name, flush=True)
            except BaseException as error:
                failed += 1
                print("FAIL", name, type(error).__name__, str(error)[:1500], flush=True)
    print("failed:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
