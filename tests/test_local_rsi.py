"""Exercise reflection evidence, artifact loading and the local orchestration seam."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rsi.storage import Workspace, sha256
from rsi.experience import EpisodeRecorder
from rsi.reflection_context import read_episode, load_library, HARNESS, VENDOR
from rsi.release_builder import build_release
from rsi.outer import OuterLoop
from test_recursion import metadata, observation

PROJECT=Path(__file__).resolve().parents[1]


class LocalRSI(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.ws=Workspace(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_failed_trace_has_actions_tool_results_and_final_rgb(self):
        recorder=EpisodeRecorder(self.ws,metadata());obs=observation();after=observation()
        after['instruction']='Observe the object after the failed grasp'
        recorder.observation(obs)
        recorder.executed(obs,obs['state'],after,sequence=0,acknowledged=True,
            feedback={'score':0,'reason':'object slipped'},events=[dict(policy_step=0,
                tool='call_tool',arguments={'id':'grasp'},tool_result='grasp planned',
                execution={'played':1,'interruption':['object slipped']})])
        row=recorder.finish(success=False,terminal_confirmed=True,score=0)
        self.assertNotIn('path',row)
        self.assertTrue(Path(row['trace_path']).exists())
        page,frames=read_episode(row,0,4,images=True)
        self.assertEqual(page['steps'][0]['feedback']['reason'],'object slipped')
        self.assertEqual(page['tool_events'][0]['arguments'],{'id':'grasp'})
        self.assertEqual(len(frames),6)
        final,_=read_episode(row,1,1,images=True)
        self.assertIn('failed grasp',final['final_observation']['instruction'])

    def test_open_task_is_rejected_before_recording(self):
        meta=metadata();meta['category']='open'
        with self.assertRaises(ValueError):
            EpisodeRecorder(self.ws,meta)

    def test_non_open_zero_action_failure_is_available_to_reflection(self):
        meta=metadata()
        recorder=EpisodeRecorder(self.ws,meta);recorder.observation(observation())
        row=recorder.abort('provider connection interrupted')
        page,images=read_episode(row,0,1,True)
        self.assertEqual(page['total_steps'],0);self.assertTrue(images)
        self.assertFalse(row['recording_valid'])

    def test_general_memory_and_tool_change_enter_candidate(self):
        # Stub only heavyweight release freezing/checkpoint manifests; the real builder,
        # copied files, SkillLibrary and Memory loader execute below.
        original=PROJECT/HARNESS/VENDOR/'tools.py'
        content=original.read_text()+'\nLOCAL_RSI_TEST_SENTINEL = True\n'
        checkpoints={k:{'path':'/fixture','files':{'fixture':'0'*64}} for k in ('memory','flash')}
        manifest=build_release(self.ws,PROJECT,checkpoints,skill_updates=[
            dict(id='grasp-local-test',scope='generic',text='Check support before release.',description='Support',dependencies=[]),
            dict(id='support-fit-local-test',scope='generic',task='',text='Verify centered support after contact.',
                 description='Support geometry',dependencies=['grasp-local-test'])],
            memory_updates=[dict(id='retain-contact',text='Remember whether support was observed.')],
            code_updates=[dict(path=VENDOR+'/tools.py',content=content)])
        root=Path(manifest['root']);lib=load_library(root)
        # A single policy owns both implementations. Its WAM manifest must not
        # recursively include the Harness release that embeds that manifest.
        wam=json.loads((root/HARNESS/'configs/wam_skills.json').read_text())
        for spec in wam.values():
            self.assertIn('wam_model.py',spec['source_files'])
            self.assertNotIn('harness/release.json',spec['source_files'])
            for name,expected in spec['source_files'].items():
                self.assertEqual(sha256(root/'XPolicyLab/policy/WorldScape_RSI'/name),expected)
        self.assertEqual(lib.entries['support-fit-local-test']['scope'],'generic')
        self.assertIn('grasp-support-release',lib.entries)
        self.assertIn('LOCAL_RSI_TEST_SENTINEL',(root/HARNESS/VENDOR/'tools.py').read_text())
        self.assertEqual((root/HARNESS/'memory/retain-contact.md').read_text(),'Remember whether support was observed.')
        second=build_release(self.ws,root,checkpoints,skill_updates=[dict(id='next-round',text='Retain the earlier contact evidence.')])
        self.assertIn('support-fit-local-test',load_library(second['root']).entries)
        self.assertTrue((Path(second['root'])/HARNESS/'memory/retain-contact.md').exists())

    def test_reflection_reads_trajectory_before_proposing(self):
        recorder=EpisodeRecorder(self.ws,metadata());obs=observation();recorder.observation(obs)
        recorder.executed(obs,obs['state'],obs,sequence=0,acknowledged=True,feedback={'reason':'missed'})
        row=recorder.finish(success=False,terminal_confirmed=True)
        proposal=dict(hypothesis='Retry from observed contact',evidence_episode_ids=[row['episode_id']],
            skill_updates=[dict(id='new-grasp',text='Observe support before release.')],
            training_requested=False,training_reason='',expected_behavior=[],risks=[])
        messages_seen=[]
        def complete(messages,tools):
            messages_seen.append(list(messages))
            name,args=('read_episode',dict(episode_id=row['episode_id'],start=0,count=1,images=False)) if len(messages_seen)==1 else ('propose_rsi_update',proposal)
            return {'choices':[{'message':{'tool_calls':[{'id':'call'+str(len(messages_seen)),
                'function':{'name':name,'arguments':json.dumps(args)}}]}}]}
        loop=OuterLoop(SimpleNamespace(ws=self.ws))
        self.assertEqual(loop.reflect(SimpleNamespace(complete=complete),[row],{}),proposal)
        self.assertIn('missed',json.dumps(messages_seen[1]))


if __name__=='__main__':unittest.main()
