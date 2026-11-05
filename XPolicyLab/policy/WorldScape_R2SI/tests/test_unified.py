"""Exercise the unified public boundary and retained WAM execution acknowledgements."""
import json
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from XPolicyLab.policy.WorldScape_R2SI import kinematics, model, runtime, wam
from XPolicyLab.policy.WorldScape_R2SI.tests.harness import KinematicEnv, ScriptedClient


def test_public_entry_runs_code_cells_skills_and_trace(tmp_path):
    checkpoint = tmp_path / 'sam3.pt'
    checkpoint.touch()
    client = ScriptedClient(['robo.read_skill("grasp-support-release"); robo.wait(2)'])
    client.close = lambda: client.ended.set()
    cfg = yaml.safe_load(Path(model.__file__).with_name('deploy.yml').read_text())
    cfg.update(task_name='fixture', segmenter_checkpoint=str(checkpoint), keepalive_s=.1)
    env = KinematicEnv(kinematics.ArmModel())
    first_client = ScriptedClient([])
    first_client.close = lambda: first_client.ended.set()
    with patch.object(runtime, 'ModelClient', side_effect=[first_client, client]), \
         patch.object(runtime, 'Segmenter', return_value=SimpleNamespace(forget=lambda: None)) as segmentation, \
         patch.object(runtime.planning, 'Curobo'):
        adapter = model.Model(cfg)
        try:
            adapter.reset()
            adapter.reset()  # a new episode reuses heavy models, not conversation state
            assert first_client.ended.is_set()
            assert segmentation.call_count == 1
            adapter.update_obs(env.observation())
            deadline = time.monotonic() + 10
            while not client.waiting and time.monotonic() < deadline:
                action, = adapter.get_action()
                env.apply(action)
                adapter.update_obs(env.observation())
            assert client.waiting, 'The real code-writing Agent did not finish its scripted cell'
            events = adapter.drain_trace()
            assert any(e['kind'] == 'code_cell' and e['result']['steps'] == 2 for e in events)
            assert 'robo.skills' in client.instructions
            assert 'must_not_reach_the_agent' not in json.dumps(client.seen)
            assert adapter.release_metadata()['executor'] == 'inner_agent'
            assert adapter.drain_trace() == []
        finally:
            adapter.close()
    assert client.ended.is_set()


class FakeWam:
    def __init__(self):
        self.events = []
    def reset(self): self.events.append('reset')
    def update_obs(self, obs):
        assert obs['vision']['cam_head']['color'].dtype == np.uint8
        self.events.append('observe')
    def get_action(self):
        action = {key: np.zeros(6 if '_arm_' in key else 1) for key in wam.STATE_KEYS}
        return [action, action]
    def commit_action(self, receipt): self.events.append(('commit', receipt))
    def discard_action(self, receipt): self.events.append(('discard', receipt))
    def close(self): self.events.append('close')


@pytest.mark.parametrize('profile', ['flash', 'memory'])
@pytest.mark.parametrize('acknowledged', [0, 1, 2])
def test_bundled_wam_commits_only_fully_observed_chunks(profile, acknowledged):
    catalog = {'worldscape-wam-memory': {'profile': 'memory'},
               'worldscape-wam-nomemory': {'profile': 'flash'}}
    fake = FakeWam()
    seen = []
    def factory(spec, deployment):
        seen.append(spec['profile'])
        return fake
    session = wam.BundledSessions(catalog, {'env_cfg_type': 'arx_x5'}, factory).open('bundled:' + profile)
    observation = wam.frame({c: np.zeros((8, 8, 3), np.uint8) for c in wam.CAMERAS},
                            {k: np.zeros(6 if '_arm_' in k else 1) for k in wam.STATE_KEYS}, 'test')
    session.open(observation)
    actions = session.chunk()
    for action in actions[:acknowledged]:
        session.before(action)
        session.observe(observation)
    if acknowledged < 2:
        session.before(actions[acknowledged])  # final action has no acknowledgement
    session.close()
    assert seen == [profile]
    commits = [e for e in fake.events if isinstance(e, tuple) and e[0] == 'commit']
    assert len(commits) == int(acknowledged == 2)
    assert 'close' in fake.events
    if acknowledged < 2:
        assert ('discard', {'execution_started': True}) in fake.events
