"""Real child-process teardown tests with CPU observations, not robot scores."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
from rsi.storage import Workspace, sha256
from rsi.experience import EpisodeRecorder
from rsi.experience_publication import committed_records, accepted_roots, commit_experience
from rsi.episode_jobs import finalize_job_result
from test_recursion import metadata, observation
from experience_fixtures import publish_fixture

CHILD = r'''
import json,os,sys,time
from types import SimpleNamespace
from pathlib import Path
from rsi.storage import Workspace
from rsi.evaluator import run_episode
from test_recursion import observation
ws=Workspace(sys.argv[1]);m=json.loads(sys.argv[2]);fault=sys.argv[3];obs=observation()
policy=SimpleNamespace(prepare_case=lambda _:None,reset=lambda:None,
release_metadata=lambda:{"release_id":m["release_id"]},update_obs=lambda _:None,
get_action=lambda:[obs["state"]],drain_trace=lambda:[],on_trial_end=lambda _:None,close=lambda:None)
env=SimpleNamespace(evidence_kind="robodojo_native",reset=lambda:obs,
step=lambda _: (obs,True,dict(success=True,score=100)),close=lambda:None)
row=run_episode(env,policy,ws,m,max_wall_seconds=10,record=True)
receipt=dict(schema="rsi.episode-exit-receipt.v1",metadata=dict(m),cleanup_before_app=fault!="cleanup",result=row)
if fault=="identity":receipt["metadata"]["release_id"]="wrong"
if fault!="missing":ws.atomic("episode-jobs/provisional/"+m["episode_id"]+".json",receipt)
ws.atomic("ready.json",{"ready":True})
time.sleep(.3)
if fault=="raise":raise RuntimeError("teardown failed")
if fault=="kill":os.kill(os.getpid(),9)
os._exit(17 if fault=="exit17" else 0)
'''

@pytest.mark.parametrize("fault",["ok","exit17","raise","kill","cleanup","identity","missing"])
def test_real_os_exit_controls_all_consumers(tmp_path, fault):
    ws=Workspace(tmp_path/fault);m=metadata()
    project = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(project), str(project / "XPolicyLab"), str(project / "tests"),
        *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))])
    proc=subprocess.Popen([sys.executable,"-B","-c",CHILD,str(ws.root),json.dumps(m),fault],
                          stdout=subprocess.PIPE,stderr=subprocess.PIPE, env=environment)
    deadline=time.monotonic()+20
    while not ws.path("ready.json").exists():
        assert time.monotonic()<deadline
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            pytest.fail("Child exited before teardown test: " + stderr.decode(errors="replace"))
        time.sleep(.01)
    assert committed_records(ws)==[] and accepted_roots(ws)==[]
    assert not list(ws.path("experience/outcomes").glob("*.json"))
    assert not list(ws.path("experience/accepted").rglob("*.hdf5"))
    stdout,stderr=proc.communicate(timeout=20)
    job=dict(id="job-"+fault,exit_code=proc.returncode,status="completed" if proc.returncode==0 else "failed")
    result=finalize_job_result(ws,m,job)
    if fault=="ok":
        assert result["process_exit_confirmed"] and result["terminal_confirmed"]
        rows=committed_records(ws)
        assert len(rows)==1 and rows[0]["success"] and rows[0]["recording_valid"]
        assert len(accepted_roots(ws))==1
        assert finalize_job_result(ws,m,job)==result
        with pytest.raises(ValueError,match="immutable"):
            finalize_job_result(ws,m,dict(job,id="unrelated"))
        assert json.loads(ws.path("episode-jobs/results",m["episode_id"]+".json").read_text())==result
    else:
        assert result["infrastructure_error"] and not result["terminal_confirmed"]
        assert "success" not in result and "score" not in result
        assert committed_records(ws)==[] and accepted_roots(ws)==[]
        assert list(ws.path("experience/.pending").glob("*.hdf5"))

def stage(ws):
    m=metadata();rec=EpisodeRecorder(ws,m);obs=observation()
    rec.executed(obs,obs["state"],obs,sequence=0,acknowledged=True)
    row=rec.finish(success=True,terminal_confirmed=True,score=100)
    return m,row

def test_no_hardlink_requirement_and_legacy_records_ignored(tmp_path):
    ws=Workspace(tmp_path);m,row=stage(ws)
    ws.atomic("experience/outcomes/legacy.json",dict(row,success=True,recording_valid=True))
    ws.path("experience/accepted/legacy").mkdir(parents=True)
    ws.path("experience/accepted/legacy/episode.hdf5").write_bytes(b"unconfirmed")
    assert committed_records(ws)==[] and accepted_roots(ws)==[]
    with patch("os.link",side_effect=OSError(38,"unsupported")):
        record=publish_fixture(ws,m,row)
    assert record["recording_valid"] and len(accepted_roots(ws))==1

def test_copy_failure_cannot_be_seen_by_snapshot_or_reflection(tmp_path):
    ws=Workspace(tmp_path);m,row=stage(ws)
    with patch("rsi.experience_publication.shutil.copyfile",side_effect=OSError("disk full")):
        with pytest.raises(OSError):publish_fixture(ws,m,row)
    assert not committed_records(ws) and not accepted_roots(ws)
    assert publish_fixture(ws,m,row)["recording_valid"]

def test_concurrent_publish_once_and_mutation_rejected(tmp_path):
    ws=Workspace(tmp_path);m,row=stage(ws)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        records=list(pool.map(lambda _:publish_fixture(ws,m,row),range(2)))
    assert records[0]==records[1] and len(committed_records(ws))==1
    with open(records[0]["trace_path"],"ab") as f:f.write(b"tampered")
    with pytest.raises(ValueError,match="hash"):accepted_roots(ws)

def test_staged_native_result_cannot_change_before_commit(tmp_path):
    ws=Workspace(tmp_path);m,row=stage(ws)
    with pytest.raises(ValueError,match="disagree"):publish_fixture(ws,m,dict(row,score=0))
    assert not committed_records(ws)
