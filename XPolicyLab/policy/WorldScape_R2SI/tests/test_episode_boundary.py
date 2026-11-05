"""Policy input isolation and evaluator-owned termination without a shared budget."""
from types import SimpleNamespace

import numpy as np

from XPolicyLab.policy.WorldScape_R2SI import deploy
from XPolicyLab.policy.WorldScape_R2SI.bridge import Bridge, STATE_SIZES, parse_observation
from XPolicyLab.policy.WorldScape_R2SI.tests.harness import outputs, run_episode


def observation():
    return {"instruction": "Hold still.",
            "state": {key: np.zeros(size) for key, size in STATE_SIZES.items()},
            "vision": {"cam_head": {"color": np.zeros((4, 4, 3), np.uint8)}}}


def test_environment_metadata_cannot_supply_action_count_or_budget():
    raw = observation()
    for metadata in (None, {"step": 999999, "step_lim": 0}, {"step": -2, "step_lim": "bad"}):
        raw["additional_info"] = metadata
        parsed = parse_observation(raw)
        assert parsed.step == 0
        assert set(vars(parsed)) == {"instruction", "images", "state", "step"}


def test_duplicate_observations_and_keepalive_actions_count_locally():
    monitor = SimpleNamespace(before=lambda action, *_args, **_kwargs: action, after=lambda *_args: [])
    bridge = Bridge(monitor, keepalive_s=0)
    raw = observation()
    raw["additional_info"] = {"step": 500, "step_lim": 1}
    bridge.update_obs(raw)
    bridge.update_obs(raw)
    assert bridge.current().step == 0
    for expected in (1, 2):
        bridge.get_action()  # A keepalive hold is an action, too.
        assert bridge.current().step == expected - 1  # Still unacknowledged.
        bridge.update_obs(raw)
        bridge.update_obs(raw)
        assert bridge.current().step == expected
    assert bridge.keepalive_holds == 2
    bridge.close()
    fresh = Bridge(monitor, keepalive_s=0)
    fresh.update_obs(raw)
    assert fresh.current().step == 0
    fresh.close()


def test_default_deployment_needs_only_the_official_environment_interface():
    raw, received = observation(), []
    class Environment:
        def __init__(self):
            self.executed = 0
        @property
        def step_lim(self):
            raise AssertionError("The policy must not read the evaluator's limit")
        @property
        def take_action_cnt(self):
            raise AssertionError("The policy must count its own actions")
        def is_episode_end(self):
            return self.executed == 3
        def get_obs(self):
            return raw
        def take_action(self, action):
            self.executed += 1
    class Client:
        def call(self, func_name, **kwargs):
            if func_name == "update_obs":
                received.append(kwargs["obs"])
            if func_name == "get_action":
                return [{}, {}]
    env = Environment()
    deploy.eval_one_episode(env, Client())
    assert env.executed == 3 and len(received) == 3
    assert all(item is raw for item in received) and "additional_info" not in raw


def test_long_wait_is_cancelled_without_allocating_the_requested_length():
    client, env, tools = run_episode(["robo.wait(10 ** 12)", "robo.state()"], step_limit=7)
    assert env.step == 7 and client.turn == 1 and not outputs(client)
    assert tools.bridge._closed and not tools.bridge._queue


def test_wait_can_span_multiple_bounded_batches():
    client, env, tools = run_episode(["robo.wait(260)", "robo.state()"])
    assert env.step == 260 and outputs(client)[1]["value"]["steps_used"] == 260
    assert not tools.bridge._queue
