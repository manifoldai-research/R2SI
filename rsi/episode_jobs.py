"""Local campaign jobs shared with simulator workers; each episode gets a process."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
import uuid
from pathlib import Path

from .jobs import Jobs
from .storage import Workspace, sha256
from .official_layouts import MODE, validate_case, reject_open, uses_official_layouts


class EpisodeJobs:
    def __init__(self, workspace, config):
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self.config = config

    def evaluate(self, release, cases, *, record=False):
        # Validate the whole batch before claiming or launching any episode.
        for case in cases:
            reject_open(case)
            if self.config.get("robodojo_root") or uses_official_layouts(case):
                validate_case(case)
        from .storage_preflight import check_storage
        check_storage(self.ws)
        pending = []
        for case in cases:
            episode = uuid.uuid4().hex
            meta = dict(case, episode_id=episode, release_id=release,
                        collection_authorized=True, hidden_state_used=False,
                        record_purpose="collection" if record else "validation")
            if case.get('mode') != MODE:
                meta.update(mode='development', split='development')
            job = dict(metadata=meta, record=record, config=self.config)
            self.ws.atomic('episode-jobs/pending/'+episode+'.json', job)
            pending.append(episode)
        deadline = time.monotonic()+self.config.get('batch_timeout_s', 7200)
        results = {}
        while len(results) < len(pending):
            for episode in pending:
                path = self.ws.path('episode-jobs/results',episode+'.json')
                if episode not in results and path.exists():
                    results[episode] = json.loads(path.read_text())
            if len(results) == len(pending):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Simulator batch is still pending; retained job IDs: '+','.join(x for x in pending if x not in results))
            time.sleep(1)
        return [results[k] for k in pending]


def finalize_job_result(ws, metadata, job):
    """Publish only a receipt bound to this job after its real OS exit."""
    identity = metadata['episode_id']
    failure = dict(metadata, infrastructure_error=True, terminal_confirmed=False,
                   job_id=job['id'], error='Episode process ' + job['status'])
    final = ws.path('episode-jobs/results', identity+'.json')
    def existing():
        if not final.exists():
            return None
        prior = json.loads(final.read_text())
        if (any(prior.get(k) != v for k,v in metadata.items())
                or prior.get('job_id') != job['id']
                or prior.get('process_status') != job['status']
                or prior.get('process_exit_code') != job.get('exit_code')):
            raise ValueError('Final episode result is immutable and belongs to different evidence')
        return prior
    with ws.db():
        prior = existing()
        if prior is not None:
            return prior
    result = failure
    path = ws.path('episode-jobs/provisional', identity+'.json')
    try:
        if job['status'] != 'completed' or job.get('exit_code') != 0:
            raise ValueError('Process did not exit successfully')
        receipt = json.loads(path.read_text())
        if (receipt.get('schema') != 'rsi.episode-exit-receipt.v1'
                or receipt.get('metadata') != metadata
                or receipt.get('cleanup_before_app') is not True):
            raise ValueError('Missing cleanup or mismatched episode receipt')
        candidate = receipt['result']
        if not isinstance(candidate, dict) or any(candidate.get(k) != v for k,v in metadata.items()):
            raise ValueError('Result identity differs from its claimed job')
        result = dict(candidate, job_id=job['id'], process_exit_confirmed=True)
        if candidate.get('terminal_confirmed') is True:
            from .experience_publication import commit_experience
            committed = commit_experience(ws, metadata, candidate, job, sha256(path))
            result['experience_record'] = committed['trace_path']
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = dict(failure, error='Episode not finalized: '+str(error))
    result.update(process_status=job['status'], process_exit_code=job.get('exit_code'))
    with ws.db():
        prior = existing()
        if prior is not None:
            return prior
        ws.atomic('episode-jobs/results/'+identity+'.json', result)
    return result


def worker(workspace, project, python, devices, *, once=False):
    """Run on the simulator host. Never adopt or replay another worker's claimed job."""
    ws = Workspace(workspace)
    if not devices:
        raise ValueError('Specify a simulator device per concurrent episode slot')
    pending = ws.path('episode-jobs/pending'); pending.mkdir(parents=True, exist_ok=True)
    active = ws.path('episode-jobs/active'); active.mkdir(exist_ok=True)
    def execute(path, device):
        value = json.loads(path.read_text())
        if not device.startswith('cuda:') or not device[5:].isdigit():
            raise ValueError('Each simulator slot needs a cuda:N device')
        value['device'] = 'cuda:0'
        value['physical_device'] = device
        ws.atomic('episode-jobs/active/'+path.name, value)
        cfg = value['config']
        job = Jobs(ws).run([str(cfg.get('python', python)), '-B', str(Path(project)/'scripts/local_episode.py'),
                           '--workspace', str(ws.root), '--input', str(path)],
            cwd=project, job_kind='development_evaluation',
            timeout_s=cfg['max_wall_seconds_per_episode']+cfg.get('startup_timeout_s', 600),
            environment={'CUDA_VISIBLE_DEVICES':device[5:],
                         'PYTHONPATH': os.pathsep.join([str(project), str(Path(project)/'XPolicyLab'),
                         *cfg.get('python_paths', [])])})
        finalize_job_result(ws, value['metadata'], job)
        path.rename(ws.path('episode-jobs/finished',path.name))
    ws.path('episode-jobs/finished').mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        running = {}
        while True:
            for slot, future in list(running.items()):
                if future.done():
                    future.result()
                    del running[slot]
            for slot, device in enumerate(devices):
                if slot in running:
                    continue
                for path in sorted(pending.glob('*.json')):
                    claimed = active/path.name
                    try:
                        path.rename(claimed)
                    except FileNotFoundError:
                        continue
                    running[slot] = pool.submit(execute, claimed, device)
                    break
            if once and not running and not any(pending.glob('*.json')):
                return
            time.sleep(1)
