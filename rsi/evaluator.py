"""Actual observation/action/feedback loop for an explicit development environment."""
from __future__ import annotations

import hashlib
import time

from .experience import EpisodeRecorder, admit
from .storage import canonical, sha256
from .official_layouts import MODE, EVIDENCE


def run_episode(env, policy, workspace, metadata, *, max_wall_seconds, record=False):
    """env: reset()->RGB observation; step(action)->(observation,terminated,result).

    The environment adapter owns native success, score and action limits.
    No time-limit termination is reported as native failure/success; a timeout
    is an infrastructure interruption and cannot count toward validation.
    """
    purpose = "collection" if record else "validation"
    if metadata.get("record_purpose", purpose) != purpose:
        raise ValueError("Episode purpose conflicts with collection/validation dispatch")
    metadata = dict(metadata, record_purpose=purpose)
    recorder = None
    started = time.monotonic()
    trace = hashlib.sha256()
    steps = 0
    result = None
    try:
        admit(metadata, training=False)
        expected_evidence = EVIDENCE if metadata.get("mode") == MODE else "robodojo_native"
        if getattr(env, "evidence_kind", None) != expected_evidence:
            raise ValueError("Promotion evaluation requires a native RoboDojo environment adapter")
        if metadata.get("mode") == MODE and getattr(env, "layout_sha256", None) != metadata["layout_sha256"]:
            raise ValueError("Environment layout differs from the selected official file")
        recorder = EpisodeRecorder(workspace, metadata, trainable=record)
        policy.prepare_case({"task_name": metadata["task"]})
        policy.reset()
        if policy.release_metadata().get("release_id") != metadata["release_id"]:
            raise ValueError("Running release differs from evaluation provenance")
        before = env.reset()
        recorder.observation(before)
        policy.update_obs(before)
        while True:
            if time.monotonic() - started > max_wall_seconds:
                raise TimeoutError("Development episode wall-clock guard reached")
            actions = policy.get_action()
            if not isinstance(actions, list) or len(actions) != 1:
                raise ValueError("RSI wrapper must expose one acknowledged native action at a time")
            action = actions[0]
            after, terminated, result = env.step(action)
            if not isinstance(terminated, bool) or not isinstance(result, dict):
                raise ValueError("Invalid native environment result")
            policy.update_obs(after)
            if recorder:
                recorder.executed(before, action, after, sequence=steps, acknowledged=True,
                                  feedback=result, events=policy.drain_trace() if hasattr(policy, "drain_trace") else [])
            trace.update(canonical({"step": steps, "action": {k: v.tolist() for k, v in action.items()},
                                    "observation_sha256": _observation_hash(after)}))
            steps += 1
            before = after
            if terminated:
                if type(result.get("success")) is not bool or not isinstance(result.get("score"), (float, int)):
                    raise ValueError("Native terminal result lacks success/score")
                break
        row = {**metadata, "success": result["success"], "score": result["score"],
               "actions": steps, "elapsed_s": time.monotonic() - started,
               "terminal_confirmed": True, "evidence_kind": EVIDENCE if metadata.get("mode") == MODE else "development_closed_loop",
               "trace_sha256": trace.hexdigest(), "infrastructure_error": False}
        policy.on_trial_end(result)
    except BaseException as error:
        if recorder and not recorder.closed:
            if hasattr(policy, "drain_trace"):
                try: recorder.events(policy.drain_trace())
                except Exception: pass
            recorder.abort(type(error).__name__ + ": " + str(error))
        raise
    finally:
        errors = []
        for operation in (policy.close, env.close):
            try:
                operation()
            except BaseException as error:
                errors.append(error)
        if errors:
            if recorder and not recorder.closed:
                recorder.abort('Cleanup incomplete: ' + str(errors[0]))
            raise RuntimeError('Episode cleanup incomplete') from errors[0]
    recorder.finish(success=result["success"], terminal_confirmed=True,
                    reason=str(result.get("reason", "")), score=result["score"])
    row["staged_experience"] = {"identity": recorder.identity,
        "sha256": sha256(recorder.ws.path("experience/staged", recorder.identity + ".json"))}
    return row


def _observation_hash(observation):
    digest = hashlib.sha256(observation["instruction"].encode())
    for key, value in sorted(observation["state"].items()):
        digest.update(key.encode())
        digest.update(value.tobytes())
    for key, value in sorted(observation["vision"].items()):
        digest.update(key.encode())
        digest.update(value["color"].tobytes())
    return digest.hexdigest()
