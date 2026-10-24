"""Continuous local RSI: collect, reflect, build/train, validate, activate, repeat."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import time
import uuid

from .episode_jobs import EpisodeJobs
from .experience_publication import accepted_roots, reflection_records
from .generations import deploy, retire_unused
from .jobs import Jobs
from .outer import OuterLoop
from .reflection_context import HARNESS, load_library
from .registry import Registry
from .release_builder import build_release
from .storage import Workspace, sha256
from .training_agent import training_metrics
from .official_layouts import MODE, validate_plan, reject_open, uses_official_layouts


class LocalLoop:
    def __init__(self, config, *, client_factory=None, evaluator=None):
        native = bool(config.get('episodes', {}).get('robodojo_root'))
        for case in [*config['collection_cases'], *config['validation_cases'], *config.get('training_collection_cases', [])]:
            reject_open(case)
        self.official_layouts = native or any(uses_official_layouts(c) for c in config['collection_cases'] + config['validation_cases'])
        if self.official_layouts:
            self.layout_split = validate_plan(config['collection_cases'] + config.get('training_collection_cases', []), config['validation_cases'])
        collection_ids = {c["case_id"] for c in config["collection_cases"] + config.get("training_collection_cases", []) if c.get("case_id")}
        validation_ids = {c["case_id"] for c in config["validation_cases"] if c.get("case_id")}
        if collection_ids & validation_ids:
            raise ValueError("Collection and validation cases must be disjoint")
        self.config=config;self.ws=Workspace(config['workspace'])
        self.registry=Registry(self.ws);self.outer=OuterLoop(self.registry)
        self.project=Path(config['source_project']).resolve()
        self.evaluator=evaluator or EpisodeJobs(self.ws,config['episodes']).evaluate
        self.client_factory=client_factory
        self.results=[]

    def initialize(self):
        if self.registry.status()['pointers'].get('development'):
            return
        checkpoints=json.loads(Path(self.config['checkpoints']).read_text())
        release=build_release(self.ws,self.project,checkpoints,
                              policy_client_config=self.config.get('policy_client_config'))
        self.registry.initialize(self.registry.register(release))

    def _client(self, harness):
        if self.client_factory:return self.client_factory()
        if self.config.get('pool_bootstrap'):
            spec=importlib.util.spec_from_file_location('local_pool_bootstrap',self.config['pool_bootstrap'])
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.configure_pool()
        from XPolicyLab.policy.WorldScape_RSI.harness.provider import ResponsesClient
        return ResponsesClient(json.loads((harness/'configs/inner_agent.json').read_text()))

    def snapshot(self):
        from worldscape_policy.data.robodojo_snapshots import SnapshotStore
        training=self.config['training']
        auth=json.loads(Path(training['authorization']).read_text())
        if not (auth.get('user_authorized') is True and auth.get('official_evaluation') is False
                and auth.get('source_kind') in {'official_training','development'}
                and auth.get('index_sha256')==sha256(training['base_index'])):
            raise ValueError('Training index needs its matching authorized provenance')
        store=SnapshotStore(self.ws.path('datasets/snapshots'))
        store.initialize(training['base_index'])
        report=store.publish(accepted_roots(self.ws))
        self.ws.atomic('datasets/publish-status.json',report)
        return store.latest()

    def train(self, request, original, baseline):
        cfg=self.config['training'];steps=request['steps'];profile=request['profile']
        if steps>cfg['max_steps_per_round']:
            raise ValueError('Requested training exceeds the campaign resource budget')
        snapshot=self.snapshot()
        overrides=[*cfg.get('overrides',[]),'optimizer.lr='+str(request['learning_rate']),
                   'model.checkpoint_dir='+original['checkpoints'][profile]['path'],
                   'training.run_steps=null','training.save_at_end=true']
        with ThreadPoolExecutor(max_workers=2) as pool:
            training=pool.submit(Jobs(self.ws).train, python=cfg.get('python',os.sys.executable),
                xpolicy_root=self.project/'XPolicyLab',config=cfg['configs'][profile],snapshot=snapshot,
                gpus=cfg['gpus'],max_steps=steps,timeout_s=cfg['timeout_s'],overrides=overrides)
            collector=None
            if self.config.get('training_collection_cases'):
                collector=pool.submit(self.evaluator,baseline,self.config['training_collection_cases'],record=True)
            while not training.done():
                # Publishing appends only fully acknowledged successes. The native trainer
                # switches snapshots collectively at its own iteration boundary.
                self.snapshot()
                time.sleep(2)
            job=training.result()
            if collector:collector.result()
        latest=self.snapshot()
        if job['status']!='completed':
            raise RuntimeError('Native training '+job['status']+'; job '+job['id'])
        candidates=[p for p in Path(job['checkpoint_dir']).glob('checkpoint-*')
                    if (p/'checkpoint_manifest.json').exists() and (p/'.complete').exists()]
        if not candidates:raise RuntimeError('Trainer returned without a complete exported checkpoint')
        checkpoint=max(candidates,key=lambda p:int(p.name.split('-')[-1]))
        with self.ws.path('jobs',job['id'],'output.log').open('rb') as f:
            f.seek(0,2);f.seek(max(0,f.tell()-2*1024*1024));metrics=training_metrics(f.read().decode(errors='replace'))
        if not metrics:raise RuntimeError('No real optimizer metrics were emitted')
        checkpoints=json.loads(json.dumps(original['checkpoints']))
        checkpoints[profile]=dict(path=str(checkpoint),files={p.relative_to(checkpoint).as_posix():sha256(p)
                                   for p in checkpoint.rglob('*') if p.is_file()})
        provenance=dict(job_id=job['id'],profile=profile,request=request,
                        starting_snapshot=snapshot,latest_snapshot=latest,metrics=metrics,
                        checkpoint=str(checkpoint))
        self.ws.atomic('training/'+job['id']+'.json',provenance)
        return checkpoints,provenance

    def round(self):
        baseline=self.registry.status()['pointers']['development'];original=self.registry.get(baseline)
        source=Path(original['root']);harness=source/HARNESS
        self.ws.atomic('local-loop-phase.json',dict(phase='collecting',baseline=baseline,time=time.time()))
        collected=self.evaluator(baseline,self.config['collection_cases'],record=True)
        if not any(row.get('terminal_confirmed') for row in collected):
            errors=[row.get('error','No terminal result') for row in collected]
            raise RuntimeError('Collection could not complete: '+ '\n'.join(errors))
        outcomes=reflection_records(self.ws, self.config['validation_cases'])
        if self.official_layouts:
            allowed = {(c['task'], c['eval_seed'], c['layout_file']) for c in self.config['collection_cases'] + self.config.get('training_collection_cases', [])}
            outcomes=[r for r in outcomes if r.get('mode') == MODE and (r.get('task'), r.get('eval_seed'), r.get('layout_file')) in allowed]
        outcomes=outcomes[:self.config.get('reflection_episodes',12)]
        client=self._client(harness)
        self.ws.atomic('local-loop-phase.json',dict(phase='reflecting',baseline=baseline,time=time.time()))
        try:
            training=self.config.get('training',{})
            history=[]
            for path in sorted(self.ws.path('iterations').glob('*.json'),key=lambda p:p.stat().st_mtime)[-3:]:
                row=json.loads(path.read_text())
                # Held-out case metrics and trajectories stay in validation reports.
                history.append({k:row.get(k) for k in ('status','proposal')})
                if row.get('training'):
                    history[-1]['training_metrics']=row['training'].get('metrics', [])
            known_tasks=sorted({c['task'] for c in self.config['collection_cases']+self.config['validation_cases']+outcomes if c.get('task')})
            proposal=self.outer.reflect(client,outcomes,load_library(source).entries,known_task_ids=known_tasks,source_root=source,
                training_config={'available':bool(training), 'max_steps_per_round':training.get('max_steps_per_round'),
                                 'profiles':list(training.get('configs',{})),
                                 'data_tasks':training.get('data_tasks',[]),
                                 'data_description':training.get('data_description','')},iteration_history=history)
        finally:client.close()
        checkpoints=original['checkpoints'];provenance=None
        if proposal['training_requested']:
            if not self.config.get('training'):
                raise ValueError('This campaign has no training configuration')
            self.ws.atomic('local-loop-phase.json',dict(phase='training',baseline=baseline,time=time.time()))
            checkpoints,provenance=self.train(proposal['training'],original,baseline)
        manifest=build_release(self.ws,source,checkpoints,skill_updates=proposal['skill_updates'],
            memory_updates=proposal.get('memory_updates',()),code_updates=proposal.get('code_updates',()),
            training_provenance=provenance,policy_client_config=original.get('policy_client_config'),known_task_ids=known_tasks)
        if provenance and original.get('policy_client_config'):
            manifest['policy_client_config']=deploy(self.ws,self.project,manifest,proposal['training']['profile'],
                original['policy_client_config'],self.config['generation_service'])
        if proposal.get('code_updates'):
            check=Jobs(self.ws).run([os.sys.executable,'-B','-m','unittest','discover','-s',
                str(Path(manifest['root'])/HARNESS/'tests'),'-p','test_*.py'],
                cwd=manifest['root'],job_kind='preflight',timeout_s=180,
                environment={'PYTHONPATH':os.pathsep.join([manifest['root'],str(Path(manifest['root'])/'XPolicyLab')])})
            if check['status']!='completed':
                raise RuntimeError('Candidate tool tests failed; baseline retained. Job '+check['id'])
        candidate=self.registry.register(manifest)
        self.ws.atomic('local-loop-phase.json',dict(phase='validating',baseline=baseline,candidate=candidate,time=time.time()))
        report=self.outer.evaluate_and_promote(candidate,baseline,
            lambda release,cases:self.evaluator(release,cases,record=False),cases=self.config['validation_cases'],
            required_capabilities=self.config['required_capabilities'],
            required_seeds=self.config.get('required_seeds',(0,1,2)),
            trained_cases=[c['case_id'] for c in self.config['collection_cases']+self.config.get('training_collection_cases',[])],promote=True)
        if self.official_layouts:
            report['collection_validation_split'] = self.layout_split
        result=dict(baseline=baseline,candidate=candidate,status='activated_locally' if report['passed'] else 'rejected',
                    proposal=proposal,report=report,training=provenance,completed_at=time.time())
        self.ws.atomic('iterations/'+uuid.uuid4().hex+'.json',result)
        retire_unused(self.ws)
        return result

    def run(self):
        self.initialize()
        self.ws.atomic('local-loop-status.json',dict(state='running',rounds_completed=0,started_at=time.time()))
        for _ in range(self.config['rounds']):
            if self.ws.path('STOP').exists():break
            try:
                result=self.round()
            except Exception as error:
                self.ws.atomic('local-loop-status.json',dict(state='failed',rounds_completed=len(self.results),
                    error=type(error).__name__+': '+str(error)))
                raise
            self.results.append(result)
            self.ws.atomic('local-loop-status.json',dict(state='between_rounds',rounds_completed=len(self.results),latest=result))
        state='stopped' if self.ws.path('STOP').exists() else 'completed'
        self.ws.atomic('local-loop-phase.json',dict(phase=state,time=time.time()))
        self.ws.atomic('local-loop-status.json',dict(state=state,rounds_completed=len(self.results),
            latest=self.results[-1] if self.results else None))
        return self.results
