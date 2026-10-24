"""Start a new weight service; existing episodes keep their original endpoint."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from .storage import Workspace


def deploy(workspace, source_project, manifest, profile, previous_client, config):
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    generation=uuid.uuid4().hex;directory=ws.path('generations',generation);directory.mkdir(parents=True)
    # Retire only generations no release pointer or active episode can still use.
    retire_unused(ws)
    gpu=None
    for index in config['gpus']:
        occupied=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-compute-apps=pid',
                                          '--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        if not occupied:
            gpu=index;break
    if gpu is None:
        raise RuntimeError('No idle generation GPU; current services and candidate are retained')
    import socket
    with socket.socket() as probe:
        probe.bind((config['host'],0));port=probe.getsockname()[1]
    launch=dict(config, project=manifest['root'],workspace=str(directory),port=port,gpu=gpu,
                profile=profile,checkpoints=manifest['checkpoints'])
    spec=ws.atomic('generations/'+generation+'/launch.json',launch)
    environment=ws.environment();environment.pop('CUDA_VISIBLE_DEVICES',None)
    with (directory/'output.log').open('xb') as log:
        process=subprocess.Popen([config.get('python',os.sys.executable),'-B',str(Path(source_project)/'scripts/serve_generation.py'),
            '--config',str(spec)],cwd=source_project,env=environment,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state=dict(pid=process.pid, profile=profile, gpu=gpu, directory=str(directory),started_at=time.time())
    ws.atomic('generations/'+generation+'/process.json',state)
    deadline=time.monotonic()+config.get('startup_timeout_s',1200)
    while time.monotonic()<deadline and process.poll() is None:
        status=directory/'policy-status.json'
        if status.exists() and json.loads(status.read_text()).get('state')=='ready':
            previous=json.loads(Path(previous_client).read_text())
            profiles={name: previous.get('profiles',{}).get(name,previous) for name in ('memory','flash')}
            profiles[profile]=dict(url=('https' if config.get('certificate') else 'http')+'://'+config['host']+':'+str(port)+'/server',
                token=(directory/'private/policy-token').read_text().strip(),timeout_s=config.get('request_timeout_s',1200))
            client=ws.atomic('generations/'+generation+'/client.json',dict(profiles=profiles))
            client.chmod(0o600)
            state.update(client_config=str(client),url=profiles[profile]['url'],state='ready')
            ws.atomic('generations/'+generation+'/process.json',state)
            return str(client)
        time.sleep(1)
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise RuntimeError('Candidate weight service did not become ready; old service remains active. '+str(directory/'output.log'))


def retire_unused(workspace):
    """Called between local rounds. Shared endpoints used by retained versions stay alive."""
    from .registry import Registry
    ws=workspace if isinstance(workspace,Workspace) else Workspace(workspace)
    registry=Registry(ws);status=registry.status()
    ids=set(status['pointers'].values())|{r['release_id'] for r in status['episodes'] if r['status']=='active'}
    used_urls=set()
    for release in ids:
        path=registry.get(release).get('policy_client_config')
        if path:
            client=json.loads(Path(path).read_text())
            used_urls.update(x['url'] for x in client.get('profiles',{'all':client}).values())
    for path in ws.path('generations').glob('*/process.json'):
        state=json.loads(path.read_text())
        if state.get('state')!='ready' or state.get('url') in used_urls:
            continue
        # PID must still belong to this generation before terminating our own service.
        cmd=Path('/proc',str(state['pid']),'cmdline')
        if cmd.exists() and str(path.parent/'launch.json').encode() in cmd.read_bytes():
            os.killpg(state['pid'],signal.SIGTERM)
        state['state']='retired';ws.atomic(str(path.relative_to(ws.root)),state)
