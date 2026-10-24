"""CPU contract tests; no native simulator success or score is measured."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from rsi.official_layouts import *
from rsi.experience import EpisodeRecorder, admit
from rsi.episode_jobs import EpisodeJobs
from rsi.storage import Workspace
from rsi.validation import compare
from rsi.registry import Registry
from rsi.local_loop import LocalLoop
from rsi.outer import OuterLoop
from test_recursion import observation
from scripts.prepare_official_layout_cases import prepare

PROJECT=Path(__file__).resolve().parents[1]

class OfficialLayouts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for seed in range(3):
            d=self.root/'Assets/Eval_Layout/RoboDojo/arx_x5'/str(seed)
            d.mkdir(parents=True)
            for task in ('stack_blocks','push_T_random'):
                # Noncontiguous filenames exercise the official sorted ordinal mapping.
                for idx in (0,2,10):
                    (d/f'{task}_{idx}.json').write_text(json.dumps({'fixture':task,'seed':seed,'index':idx}))
    def tearDown(self):self.tmp.cleanup()
    def case(self,seed=0,idx=0,task='stack_blocks'):
        return make_case(self.root,task,seed,idx)
    def meta(self,**kwargs):
        return dict(self.case(**kwargs),episode_id='fixture',release_id='baseline',
                    collection_authorized=True,hidden_state_used=False)
    def rows(self,release):
        return [dict(self.meta(seed=s,idx=1),release_id=release,evidence_kind=EVIDENCE,
            terminal_confirmed=True,success=True,score=100.,actions=1,
            trace_sha256='fixture',infrastructure_error=False) for s in range(3)]
    def test_catalog_excludes_all_open_and_keeps_random_variants(self):
        self.assertEqual(len(TASK_GROUP),46)
        self.assertFalse(set(TASK_GROUP)&OPEN_TASKS)
        self.assertEqual(sum(t.endswith('_random') for t in TASK_GROUP),12)
    def test_open_cannot_be_disguised_with_standard_category(self):
        for task in OPEN_TASKS:
            for suffix in ('','_random'):
                with self.subTest(task=task+suffix),self.assertRaises(ValueError):
                    admit(dict(self.meta(),task=task+suffix,category='standard'),training=False)
    def test_unknown_task_or_category_is_rejected(self):
        for change in ({'task':'unreviewed_task'},{'category':'open'},{'category':'random'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_case(dict(self.case(),**change))
    def test_official_numeric_order_and_bytes_are_used(self):
        case=self.case(idx=1)
        self.assertEqual(case['layout_file'],'stack_blocks_2.json')
        self.assertEqual(load_layout(self.root,case)['index'],2)
    def test_tampered_layout_is_rejected(self):
        case=self.case()
        path=self.root/'Assets/Eval_Layout/RoboDojo/arx_x5/0'/case['layout_file']
        path.write_text('{"changed":true}')
        with self.assertRaises(ValueError):load_layout(self.root,case)
    def test_path_traversal_or_wrong_task_or_changed_ordinal_rejected(self):
        for change in ({'layout_file':'../stack_blocks_0.json'},
                       {'layout_file':'push_T_0.json'},{'layout_id':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                load_layout(self.root,dict(self.case(),**change))
    def test_symlink_cannot_substitute_external_layout(self):
        path=self.root/'Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_blocks_0.json'
        other=self.root/'outside.json';other.write_text(path.read_text())
        path.unlink();path.symlink_to(other)
        with self.assertRaises(ValueError):self.case()
    def test_same_official_file_with_different_case_id_is_rejected(self):
        a=self.case();b=dict(a,case_id='alias')
        with self.assertRaises(ValueError):validate_plan([a],[b])
    def test_repeated_official_scene_content_is_reported_not_synthesized(self):
        a=self.case();b=dict(self.case(idx=1),layout_sha256=a["layout_sha256"])
        report=validate_plan([a],[b])
        self.assertTrue(report["distinct_file_identities"])
        self.assertFalse(report["geometry_disjoint"])
        self.assertEqual(report["shared_layout_content_hashes"],[a["layout_sha256"]])

    def test_whole_batch_rejected_before_any_dispatch(self):
        ws=Workspace(self.root/'work')
        jobs=EpisodeJobs(ws,dict(robodojo_root=str(self.root)))
        with self.assertRaises(ValueError):
            jobs.evaluate('baseline',[self.case(),dict(self.case(),task='align_blocks')])
        self.assertFalse(list(ws.path('episode-jobs/pending').glob('*.json')))
    def test_public_provenance_cannot_be_renamed_development(self):
        row=dict(self.meta(),mode="development",split="development")
        for training in (True,False):
            with self.subTest(training=training),self.assertRaises(ValueError):admit(row,training=training)

    def test_public_rollout_stays_out_of_model_training(self):
        meta=self.meta()
        with self.assertRaises(ValueError):admit(meta,training=True)
        ws=Workspace(self.root/'record')
        recorder=EpisodeRecorder(ws,meta,trainable=True)
        obs=observation();recorder.executed(obs,obs['state'],obs,sequence=0,acknowledged=True)
        row=recorder.finish(success=True,terminal_confirmed=True,score=100)
        self.assertEqual(row['mode'],MODE)
        self.assertEqual(row['split'],SPLIT)
        self.assertFalse(row['recording_valid'])
        self.assertFalse(list(ws.path('experience/accepted').rglob('*.hdf5')))
    def test_public_paired_validation_preserves_source_and_rejects_changed_layout(self):
        before=self.rows('baseline');after=self.rows('candidate')
        report=compare('candidate','baseline',before,after,required_capabilities=['generalization'])
        self.assertEqual(report['evidence_kind'],EVIDENCE)
        self.assertEqual(report['layout_policy'],POLICY)
        self.assertFalse(report['training_eligible'])
        after[0]['layout_sha256']='a'*64
        with self.assertRaises(ValueError):
            compare('candidate','baseline',before,after,required_capabilities=['generalization'])
    def test_public_promotion_requires_public_layout_contract(self):
        registry=Registry(self.root/'registry',verifier=lambda _:None)
        b=registry.register({'fixture':'b'});c=registry.register({'fixture':'c'})
        registry.initialize(b)
        report=compare(c,b,self.rows(b),self.rows(c),required_capabilities=['generalization'])
        registry.record_validation(c,dict(report,layout_policy='unknown'))
        with self.assertRaises(ValueError):registry.promote(c,b)
        registry.record_validation(c,report);registry.promote(c,b)
        self.assertEqual(registry.status()['pointers']['development'],c)
    def test_native_adapter_rejects_open_before_simulator_initialization(self):
        from rsi.native_robodojo import make_environment
        with self.assertRaises(ValueError):make_environment(dict(self.meta(),task='align_blocks'))
    def test_cli_plan_has_disjoint_cases_and_no_scene_generation(self):
        plan=prepare(self.root,['stack_blocks','push_T_random'])
        self.assertEqual(len(plan['collection_cases']),2)
        self.assertEqual(len(plan['validation_cases']),6)
        self.assertEqual(plan['required_seeds'],[0,1,2])
        with self.assertRaises(ValueError):prepare(self.root,['align_blocks'])
    def test_official_loop_rejects_open_before_creating_workspace(self):
        config=dict(workspace=str(self.root/'never'),source_project=str(PROJECT),episodes={},
             collection_cases=[dict(self.case(),task='align_blocks')],validation_cases=[self.case(idx=1)])
        with self.assertRaises(ValueError):LocalLoop(config)
        self.assertFalse((self.root/'never').exists())
    def test_two_round_loop_never_reflects_validation_episodes(self):
        source=self.root/'Assets/Eval_Layout/RoboDojo/arx_x5/0/stack_blocks_0.json'
        for seed in range(3):
            (self.root/f'Assets/Eval_Layout/RoboDojo/arx_x5/{seed}/stack_blocks_2.json').write_bytes(source.read_bytes())
        plan=prepare(self.root,['stack_blocks'])
        cfg=dict(plan,workspace=str(self.root/'loop'),source_project=str(PROJECT),rounds=2,episodes={})
        active=['baseline'];reflection=[];counter=[0]
        ws=Workspace(cfg['workspace'])
        def evaluate(release,cases,record=False):
            rows=[]
            for case in cases:
                counter[0]+=1
                meta=dict(case,episode_id=('collect' if record else 'validate')+str(counter[0]),
                    release_id=release,collection_authorized=True,hidden_state_used=False)
                row=dict(meta,evidence_kind=EVIDENCE,terminal_confirmed=True,success=True,score=100.,
                    actions=1,trace_sha256='fixture',completed_at=counter[0],elapsed_s=1,reason='fixture')
                # All outcomes exist on disk, including newer validation traces.
                recorder=EpisodeRecorder(ws,meta);obs=observation()
                recorder.executed(obs,obs['state'],obs,sequence=0,acknowledged=True)
                from experience_fixtures import publish_fixture
                publish_fixture(ws,meta,recorder.finish(success=True,terminal_confirmed=True,score=100.))
                rows.append(row)
            return rows
        def complete(messages,tools):
            context=json.loads(messages[1]['content']);reflection.append(context)
            episodes=context['episodes']
            self.assertTrue(episodes)
            self.assertTrue(all(e['episode_id'].startswith('collect') and e['mode']==MODE for e in episodes))
            proposal=dict(hypothesis='Fixture',evidence_episode_ids=[e['episode_id'] for e in episodes],
                skill_updates=[],memory_updates=[],code_updates=[],training_requested=False,
                training_reason='',expected_behavior=[],risks=[])
            return {'choices':[{'message':{'tool_calls':[{'id':'fixture','function':
                {'name':'propose_rsi_update','arguments':json.dumps(proposal)}}]}}]}
        loop=LocalLoop(cfg,client_factory=lambda:SimpleNamespace(complete=complete,close=lambda:None),evaluator=evaluate)
        original=dict(root=str(PROJECT),checkpoints={},policy_client_config=None)
        def promote(candidate,baseline):
            self.assertEqual(active[0],baseline);active[0]=candidate
        registry=SimpleNamespace(ws=ws,status=lambda:{'pointers':{'development':active[0]}},
            get=lambda _:original,register=lambda _:'candidate'+str(counter[0]),
            record_validation=lambda *a:None,promote=promote)
        loop.registry=registry;loop.outer.registry=registry
        with patch('rsi.local_loop.build_release',return_value=original),patch('rsi.local_loop.retire_unused'):
            result=loop.run()
        self.assertEqual(len(reflection),2)
        self.assertTrue(all(r['report']['evidence_kind']==EVIDENCE for r in result))
        self.assertTrue(all(r['status']=='activated_locally' for r in result))

if __name__=='__main__':unittest.main()
