"""Control-flow fixtures, not claims of robot success or model training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace,ModuleType
from unittest.mock import patch

from rsi.jobs import Jobs
from rsi.local_loop import LocalLoop
from rsi.experience import EpisodeRecorder
from experience_fixtures import publish_fixture
from rsi.storage import Workspace
from test_recursion import metadata,observation

PROJECT=Path(__file__).resolve().parents[1]


class ContinuousLoop(unittest.TestCase):
    def test_training_proposal_flows_to_validation_local_activation_and_next_round(self):
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
                    training_requested=True,training_reason='Exercise the training branch',
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
            with patch.object(loop,'train',return_value=({},dict(job_id='fixture',metrics=[{'step':2,'loss':1.5}]))) as train, \
                 patch('rsi.local_loop.build_release',return_value=dict(original)) as build, \
                 patch('rsi.local_loop.deploy',return_value='new-client') as deploy, \
                 patch('rsi.local_loop.retire_unused'):
                result=loop.run()
            self.assertEqual(train.call_count,2);self.assertEqual(deploy.call_count,2)
            self.assertEqual(result[0]['status'],'activated_locally')
            self.assertEqual(seen[3],(result[0]['candidate'],True))
            self.assertEqual(build.call_args.args[1],Path(original['root']))
            self.assertEqual(reflection_inputs[0]['previous_iterations'],[])
            previous=reflection_inputs[1]['previous_iterations'][0]
            self.assertEqual(previous['status'],'activated_locally')
            self.assertEqual(previous['training_metrics'],[{'step':2,'loss':1.5}])
            self.assertEqual(json.loads(ws.path('local-loop-status.json').read_text())['state'],'completed')

    def test_trainer_bootstraps_original_index_after_data_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            base=dict(version=0);latest=dict(version=1)
            store=SimpleNamespace(directory=Path(directory),resolve=lambda descriptor,verify_files=True:
                dict(index_path='base.npz' if descriptor==base else 'appended.npz',
                     manifest=dict(previous=None if descriptor==base else base)))
            module=ModuleType('worldscape_policy.data.robodojo_snapshots');module.SnapshotStore=lambda _:store
            jobs=Jobs(directory)
            with patch.dict(sys.modules,{'worldscape_policy.data.robodojo_snapshots':module}), \
                 patch('subprocess.check_output',return_value=''), \
                 patch.object(jobs,'run',return_value=dict(id='fixture',status='completed')) as run:
                result=jobs.train(python='python',xpolicy_root=PROJECT/'XPolicyLab',config='fixture.yaml',
                                  snapshot=latest,gpus=[4],max_steps=3,timeout_s=60)
            self.assertIn('data_loader.dataset_kwargs.precomputed_index_path=base.npz',run.call_args.args[0])
            self.assertEqual(result['requested_snapshot'],latest)
            self.assertEqual(result['starting_snapshot'],base)

if __name__=='__main__':unittest.main()
