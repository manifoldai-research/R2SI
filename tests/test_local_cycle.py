"""Control-flow fixtures, not claims of robot success or model training."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rsi.jobs import Jobs
from rsi.local_loop import LocalLoop
from rsi.experience import EpisodeRecorder
from experience_fixtures import publish_fixture
from rsi.storage import Workspace
from test_recursion import metadata,observation

PROJECT=Path(__file__).resolve().parents[1]


class ContinuousLoop(unittest.TestCase):
    def test_skill_updates_flow_to_validation_local_activation_and_next_round(self):
        with tempfile.TemporaryDirectory() as directory:
            seen=[];active=['baseline'];reflection_inputs=[]
            cases=[dict(case_id='held-'+str(i),seed=i,capability='alignment',category='standard') for i in range(3)]
            cfg=dict(workspace=directory,source_project=str(PROJECT),rounds=2,
                collection_cases=[dict(case_id='collect',seed=55)],validation_cases=cases,
                required_capabilities=['alignment'],episodes={},training=dict(configs={'flash':'fixture'},max_steps_per_round=5),
                generation_service={})
            ws=Workspace(directory)
            def evaluate(release,cases,record=False):
                seen.append((release,record))
                if record:
                    m=metadata();m['release_id']=release
                    recorder=EpisodeRecorder(ws,m);obs=observation();recorder.observation(obs)
                    recorder.executed(obs,obs['state'],obs,sequence=0,acknowledged=True)
                    return [publish_fixture(ws,m,recorder.finish(success=False,terminal_confirmed=True))]
                return [dict(c,release_id=release,mode='development',evidence_kind='development_closed_loop',
                    terminal_confirmed=True,success=True,score=100.,actions=1,trace_sha256='fixture') for c in cases]
            def complete(messages,tools):
                context=json.loads(messages[1]['content']);reflection_inputs.append(context)
                episode=context['episodes'][0]['episode_id']
                proposal=dict(hypothesis='Control-flow fixture',evidence_episode_ids=[episode],skill_updates=[],
                    training_requested=False,training_reason='',
                    training=dict(profile='flash',steps=2,learning_rate=1e-6),expected_behavior=[],risks=[])
                return {'choices':[{'message':{'tool_calls':[{'id':'fixture','function':
                    {'name':'propose_rsi_update','arguments':json.dumps(proposal)}}]}}]}
            loop=LocalLoop(cfg,client_factory=lambda:SimpleNamespace(complete=complete,close=lambda:None),evaluator=evaluate)
            original=dict(root=str(PROJECT),checkpoints={},policy_client_config='old-client')
            def activate(candidate,baseline):
                self.assertEqual(baseline,active[0]);active[0]=candidate
            registry=SimpleNamespace(ws=ws,status=lambda:dict(pointers={'development':active[0]}),
                get=lambda _:original,register=lambda manifest:'candidate-'+str(len(seen)),
                record_validation=lambda *a:None,promote=activate)
            loop.registry=registry;loop.outer.registry=registry
            with patch.object(loop,'train',side_effect=AssertionError('Training must stay disabled')) as train, \
                 patch('rsi.local_loop.build_release',return_value=dict(original)) as build, \
                 patch('rsi.local_loop.deploy',return_value='new-client') as deploy, \
                 patch('rsi.local_loop.retire_unused'):
                result=loop.run()
            train.assert_not_called();deploy.assert_not_called()
            self.assertFalse(reflection_inputs[0]['training_resources']['available'])
            self.assertEqual(result[0]['status'],'activated_locally')
            self.assertEqual(seen[3],(result[0]['candidate'],True))
            self.assertEqual(build.call_args.args[1],Path(original['root']))
            self.assertEqual(reflection_inputs[0]['previous_iterations'],[])
            previous=reflection_inputs[1]['previous_iterations'][0]
            self.assertEqual(previous['status'],'activated_locally')
            self.assertNotIn('training_metrics',previous)
            self.assertEqual(json.loads(ws.path('local-loop-status.json').read_text())['state'],'completed')

    def test_training_stub_has_no_job_or_gpu_side_effects(self):
        from XPolicyLab.policy.WorldScape_R2SI.training import WSPTrainingUnavailable
        with tempfile.TemporaryDirectory() as directory:
            jobs=Jobs(directory)
            before={p.relative_to(directory) for p in Path(directory).rglob('*')}
            with patch('subprocess.check_output') as gpu, patch.object(jobs,'run') as run:
                with self.assertRaisesRegex(WSPTrainingUnavailable, 'not connected'):
                    jobs.train(python='not-an-interpreter',xpolicy_root='missing',config='missing',
                               snapshot={},gpus=[0],max_steps=3,timeout_s=60)
            gpu.assert_not_called();run.assert_not_called()
            self.assertEqual(before,{p.relative_to(directory) for p in Path(directory).rglob('*')})

    def test_outer_training_stub_rejects_before_snapshot_or_evaluation(self):
        from XPolicyLab.policy.WorldScape_R2SI.training import WSPTrainingUnavailable
        loop=object.__new__(LocalLoop)
        with patch.object(loop,'snapshot') as snapshot:
            with self.assertRaises(WSPTrainingUnavailable):
                loop.train({'profile':'flash','steps':2,'learning_rate':1e-6},{},'baseline')
        snapshot.assert_not_called()


if __name__=='__main__':unittest.main()
