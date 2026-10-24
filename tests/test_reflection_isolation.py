"""Held-out traces remain available for audit, never for reflection."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from rsi.storage import Workspace
from rsi.evaluator import run_episode
from rsi.experience import EpisodeRecorder
from rsi.experience_publication import committed_records, reflection_records, accepted_roots
from rsi.reflection_context import read_episode
from rsi.outer import OuterLoop
from rsi.local_loop import LocalLoop
from test_recursion import metadata, observation
from experience_fixtures import publish_fixture

PROJECT = Path(__file__).resolve().parents[1]


def episode(ws, release, case, record):
    m = dict(metadata(), **case)
    m["release_id"] = release
    obs = observation()
    obs["instruction"] = "COLLECTION_TEXT" if record else "HELDOUT_TRAJECTORY_SENTINEL"
    policy = SimpleNamespace(prepare_case=lambda _:None, reset=lambda:None,
        release_metadata=lambda: {"release_id":release}, update_obs=lambda _:None,
        get_action=lambda:[obs["state"]], drain_trace=lambda:[], on_trial_end=lambda _:None, close=lambda:None)
    env=SimpleNamespace(evidence_kind="robodojo_native",reset=lambda:obs,
        step=lambda _: (obs,True,dict(success=True,score=100,reason=obs["instruction"])),close=lambda:None)
    row=run_episode(env,policy,ws,m,max_wall_seconds=10,record=record)
    assert row["record_purpose"] == ("collection" if record else "validation")
    return publish_fixture(ws,m,row)


def test_validation_trace_is_retained_but_unavailable_to_reflection_or_training(tmp_path):
    ws=Workspace(tmp_path)
    collection=episode(ws,"source",dict(case_id="collection"),True)
    heldout=episode(ws,"source",dict(case_id="held"),False)
    assert Path(heldout["trace_path"]).is_file()
    assert heldout["record_purpose"]=="validation" and not heldout["recording_valid"]
    assert len(committed_records(ws))==2
    assert [r["episode_id"] for r in reflection_records(ws)] == [collection["episode_id"]]
    assert len(accepted_roots(ws))==1
    with pytest.raises(ValueError,match="collection records only"):
        read_episode(heldout)
    captured=[]
    proposal=dict(hypothesis="Use visible support",evidence_episode_ids=[collection["episode_id"]],
                  skill_updates=[],training_requested=False,training_reason="",expected_behavior=[],risks=[])
    def complete(messages,tools):
        captured.append(json.dumps(messages))
        return {"choices":[{"message":{"tool_calls":[{"id":"proposal","function":{
            "name":"propose_rsi_update","arguments":json.dumps(proposal)}}]}}]}
    outer=OuterLoop(SimpleNamespace(ws=ws))
    outer.reflect(SimpleNamespace(complete=complete),[heldout,collection],{})
    assert "HELDOUT_TRAJECTORY_SENTINEL" not in captured[0]
    assert heldout["episode_id"] not in captured[0]
    with pytest.raises(ValueError,match="No completed"):
        outer.reflect(SimpleNamespace(complete=complete),[heldout],{})
    assert len(captured)==1


def test_legacy_unlabelled_and_configured_validation_cases_fail_closed(tmp_path):
    ws=Workspace(tmp_path)
    row=episode(ws,"source",dict(case_id="collect"),True)
    legacy=dict(row);legacy.pop("record_purpose")
    with patch("rsi.experience_publication.committed_records",return_value=[legacy,row]):
        assert reflection_records(ws)==[row]
        assert reflection_records(ws,[dict(case_id="collect")])==[]
    with pytest.raises(ValueError,match="No completed"):
        OuterLoop(SimpleNamespace(ws=ws)).reflect(SimpleNamespace(),[legacy],{})


def test_conflicting_dispatch_purpose_rejected_before_environment_access(tmp_path):
    m=dict(metadata(),record_purpose="collection")
    with pytest.raises(ValueError,match="purpose conflicts"):
        run_episode(SimpleNamespace(),SimpleNamespace(),tmp_path,m,max_wall_seconds=1,record=False)


def test_validation_label_never_becomes_trainable_when_recorder_requested_true(tmp_path):
    ws=Workspace(tmp_path);m=dict(metadata(),record_purpose="validation")
    rec=EpisodeRecorder(ws,m,trainable=True);obs=observation()
    rec.executed(obs,obs["state"],obs,sequence=0,acknowledged=True)
    row=publish_fixture(ws,m,rec.finish(success=True,terminal_confirmed=True,score=100))
    assert not row["recording_valid"] and not accepted_roots(ws)


def test_two_rounds_exclude_prior_heldout_traces_and_detailed_validation_reports(tmp_path):
    ws=Workspace(tmp_path);active=["baseline"];contexts=[];seen=[];registered=[]
    cases=[dict(case_id="held-"+str(i),seed=i,capability="alignment",category="standard") for i in range(3)]
    cfg=dict(workspace=str(tmp_path),source_project=str(PROJECT),rounds=2,
             collection_cases=[dict(case_id="collect",seed=55,capability="alignment")],
             validation_cases=cases,required_capabilities=["alignment"],episodes={})
    def evaluate(release,cases,record=False):
        rows=[episode(ws,release,c,record) for c in cases]
        seen.extend(rows)
        return [dict(r,evidence_kind="development_closed_loop") for r in rows]
    def complete(messages,tools):
        context=json.loads(messages[1]["content"]);contexts.append(context)
        proposal=dict(hypothesis="Use visible contact",evidence_episode_ids=[context["episodes"][0]["episode_id"]],
            skill_updates=[],training_requested=False,training_reason="",expected_behavior=[],risks=[])
        return {"choices":[{"message":{"tool_calls":[{"id":"p","function":{
            "name":"propose_rsi_update","arguments":json.dumps(proposal)}}]}}]}
    loop=LocalLoop(cfg,client_factory=lambda:SimpleNamespace(complete=complete,close=lambda:None),evaluator=evaluate)
    original=dict(root=str(PROJECT),checkpoints={})
    def register(manifest):
        registered.append(manifest)
        return "candidate-"+str(len(registered))
    registry=SimpleNamespace(ws=ws,status=lambda:dict(pointers={"development":active[0]}),
        get=lambda _:original,register=register,record_validation=lambda *a:None,
        promote=lambda candidate,baseline:active.__setitem__(0,candidate))
    loop.registry=registry;loop.outer.registry=registry
    with patch("rsi.local_loop.build_release",return_value=dict(original)),patch("rsi.local_loop.retire_unused"):
        results=loop.run()
    assert len(results)==2 and all(r["status"]=="activated_locally" for r in results)
    held_ids={r["episode_id"] for r in seen if r["record_purpose"]=="validation"}
    assert len(held_ids)==12 and len(committed_records(ws))==14
    assert len(contexts[1]["episodes"])==2
    for context in contexts:
        content=json.dumps(context)
        assert "HELDOUT_TRAJECTORY_SENTINEL" not in content
        assert not any(identity in content for identity in held_ids)
        for previous in context["previous_iterations"]:
            assert "report" not in previous


def test_overlapping_case_configuration_rejected_before_collection(tmp_path):
    cfg=dict(workspace=str(tmp_path),source_project=str(PROJECT),
             collection_cases=[dict(case_id="same")],validation_cases=[dict(case_id="same")],episodes={})
    with pytest.raises(ValueError,match="disjoint"):
        LocalLoop(cfg,evaluator=lambda *a,**kw:pytest.fail("must not dispatch"))
