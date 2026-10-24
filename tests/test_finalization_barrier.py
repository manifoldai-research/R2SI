"""Subprocess and cleanup regressions; fixtures are not robot-evaluation evidence."""
import concurrent.futures
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest
from rsi.storage import Workspace
from rsi.episode_jobs import EpisodeJobs, worker
from rsi.evaluator import run_episode
from test_recursion import metadata, observation

@pytest.mark.parametrize("fault", ["none","exit","cleanup","identity","missing","corrupt"])
def test_result_waits_for_actual_process_exit(tmp_path, fault):
    ws=Workspace(tmp_path/"ws")
    project=tmp_path/"fixture";(project/"scripts").mkdir(parents=True)
    script = """import argparse,json,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace');p.add_argument('--input');a=p.parse_args()
v=json.loads(Path(a.input).read_text());m=v['metadata'];fault=v['config']['fault']
d=Path(a.workspace)/'episode-jobs/provisional';d.mkdir(parents=True,exist_ok=True)
row=dict(m,terminal_confirmed=True,success=True,score=100,actions=1)
receipt=dict(schema='rsi.episode-exit-receipt.v1',metadata=dict(m),cleanup_before_app=fault!='cleanup',result=row)
if fault=='identity':receipt['metadata']['release_id']='wrong'
if fault!='missing':(d/(m['episode_id']+'.json')).write_text('{broken' if fault=='corrupt' else json.dumps(receipt))
time.sleep(2)
os._exit(17 if fault=='exit' else 0)
"""
    (project/"scripts/local_episode.py").write_text(script)
    cfg=dict(max_wall_seconds_per_episode=10,startup_timeout_s=5,batch_timeout_s=20,python=sys.executable,fault=fault)
    def run_worker():
        pending=ws.path("episode-jobs/pending")
        deadline=time.monotonic()+10
        while not list(pending.glob("*.json")):
            assert time.monotonic()<deadline
            time.sleep(.02)
        return worker(ws.root,project,sys.executable,["cuda:0"],once=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(run_worker)
        start=time.monotonic()
        rows=EpisodeJobs(ws,cfg).evaluate("test-release",[dict(task="fixture",case_id="case",seed=1)])
        elapsed=time.monotonic()-start
        future.result(timeout=10)
    assert elapsed>=2
    result=rows[0]
    jobs=list(ws.path("jobs").glob("*/status.json"))
    assert json.loads(jobs[0].read_text())["exit_code"]==(17 if fault=="exit" else 0)
    # Even a clean OS exit without the bound staged trace cannot publish success.
    assert result["infrastructure_error"] and not result["terminal_confirmed"]
    assert "score" not in result and "success" not in result

@pytest.mark.parametrize("fault",["policy_close","env_close","trial_end"])
def test_cleanup_failure_never_publishes_training_success(tmp_path,fault):
    ws=Workspace(tmp_path/"cleanup");m=metadata();obs=observation();calls=[]
    def operation(name):
        def call(*args):
            calls.append(name)
            if name==fault:raise RuntimeError("injected "+name)
        return call
    policy=SimpleNamespace(prepare_case=lambda x:None,reset=lambda:None,
        release_metadata=lambda:dict(release_id=m["release_id"]),
        update_obs=lambda x:None,get_action=lambda:[{k:obs["state"][k] for k in (
            "left_arm_joint_state","right_arm_joint_state","left_ee_joint_state","right_ee_joint_state")}],
        drain_trace=lambda:[],on_trial_end=operation("trial_end"),close=operation("policy_close"))
    env=SimpleNamespace(evidence_kind="robodojo_native",reset=lambda:obs,
        step=lambda action:(obs,True,dict(success=True,score=100)),close=operation("env_close"))
    with pytest.raises(RuntimeError):
        run_episode(env,policy,ws,m,max_wall_seconds=10,record=True)
    assert "policy_close" in calls and "env_close" in calls
    assert not list(ws.path("experience/accepted").glob("**/*.hdf5"))
    records=[json.loads(f.read_text()) for f in ws.path("experience/staged").glob("*.json")]
    assert records and not records[0]["recording_valid"]
