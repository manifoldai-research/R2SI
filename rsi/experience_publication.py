"""One atomic visibility boundary for completed experience.

Only the process supervisor calls commit_experience after observing OS exit.
Private staged files and legacy outcomes/accepted directories are never scanned
by reflection or snapshot publication. Interrupted copies remain private.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import uuid

from .storage import Workspace, canonical, sha256


def _identity(meta):
    return hashlib.sha256(meta["episode_id"].encode()).hexdigest()


def staged_record(ws, meta, binding):
    identity = _identity(meta)
    source = ws.path("experience/staged", identity + ".json")
    if not isinstance(binding, dict) or binding != {"identity": identity, "sha256": sha256(source)}:
        raise ValueError("Missing or mismatched staged experience binding")
    record = json.loads(source.read_text())
    if any(record.get(k) != v for k, v in meta.items()):
        raise ValueError("Experience identity differs from its episode")
    trace = ws.path("experience/.pending", record["staged_trace"])
    if trace.name != record["staged_trace"] or sha256(trace) != record["staged_trace_sha256"]:
        raise ValueError("Staged trace path/hash mismatch")
    return record, trace


def commit_experience(ws, metadata, candidate, job, receipt_sha256):
    """Publish a complete directory once; all files become visible together."""
    if (job.get("status") != "completed" or job.get("exit_code") != 0
            or candidate.get("infrastructure_error") is True
            or candidate.get("terminal_confirmed") is not True):
        raise ValueError("Experience requires a successful OS exit and native termination")
    record, source = staged_record(ws, metadata, candidate.get("staged_experience"))
    for name in ("success", "terminal_confirmed", "actions", "score"):
        if candidate.get(name) != record.get(name):
            raise ValueError("Native result and staged experience disagree: " + name)
    if record["terminal_confirmed"] is not True or type(record["success"]) is not bool:
        raise ValueError("Missing native terminal evidence")
    identity = _identity(metadata)
    binding = dict(job_id=job["id"], exit_code=0, receipt_sha256=receipt_sha256,
                   staged_experience=candidate["staged_experience"])
    target = ws.path("experience/committed", identity)
    # The same POSIX locking requirement as Registry/Jobs. No hard links are
    # needed. Refuse conflicting identity reuse, including after a parent crash.
    with ws.db():
        if target.exists():
            old = json.loads((target / "record.json").read_text())
            if old.get("publication") != binding:
                raise ValueError("Episode already committed with different evidence")
            _verify_record(ws, target, old)
            return old
        staging = ws.path("experience/.publishing", uuid.uuid4().hex)
        staging.mkdir(parents=True, exist_ok=False)
        valid = record.pop("candidate_training_valid")
        if valid:
            if record.get("record_purpose") != "collection":
                raise ValueError("Only collection records may enter training publication")
            from .experience import admit
            admit(metadata, training=True)
        relative = (Path("accepted") / metadata["task"] / "arx_x5/data" /
                    ("episode_" + identity + ".hdf5")) if valid else Path("trace.hdf5")
        copied = staging / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, copied)
        import h5py
        with h5py.File(copied, "r+") as handle:
            handle.attrs.update(recording_valid=valid, process_exit_confirmed=True)
            handle.flush()
        with copied.open("rb") as handle:
            os.fsync(handle.fileno())
        digest = sha256(copied)
        record.pop("staged_trace")
        record.pop("staged_trace_sha256")
        record.update(schema="rsi.committed-experience.v1", publication=binding,
                      process_exit_confirmed=True, recording_valid=valid,
                      trace_path=str(target / relative), trace_sha256=digest,
                      trace_relative=relative.as_posix())
        if valid:
            record.update(path=record["trace_path"], sha256=digest)
        ws.atomic(str((staging / "record.json").relative_to(ws.root)), record)
        # Flush nested directory entries before publishing the root entry.
        for directory in sorted([staging, *[p for p in staging.rglob("*") if p.is_dir()]],
                                key=lambda p: len(p.parts), reverse=True):
            _fsync_dir(directory)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, target)
        _fsync_dir(target.parent)
        return record


def _fsync_dir(path):
    fd = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _verify_record(ws, directory, record):
    if (record.get("schema") != "rsi.committed-experience.v1"
            or record.get("process_exit_confirmed") is not True
            or record.get("publication", {}).get("exit_code") != 0):
        raise ValueError("Unconfirmed experience")
    trace = directory / record["trace_relative"]
    if (not trace.resolve().is_relative_to(directory.resolve())
            or str(trace) != record.get("trace_path") or sha256(trace) != record.get("trace_sha256")):
        raise ValueError("Committed experience path/hash mismatch")
    if record.get("recording_valid"):
        from .experience import admit
        admit(record, training=True)
        if not record.get("success") or not record.get("terminal_confirmed") or record.get("path") != str(trace):
            raise ValueError("Invalid training publication")


def committed_records(workspace):
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    records = []
    for path in ws.path("experience/committed").glob("*/record.json"):
        record = json.loads(path.read_text())
        if path.parent.name != _identity(record):
            raise ValueError("Committed directory identity mismatch")
        _verify_record(ws, path.parent, record)
        records.append(record)
    return sorted(records, key=lambda r: r["completed_at"], reverse=True)


def accepted_roots(workspace):
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    return [ws.path("experience/committed", _identity(r), "accepted")
            for r in committed_records(ws) if r.get("recording_valid")]


def reflection_records(workspace, validation_cases=()):
    """Validation and unlabelled legacy traces are audit-only, never reflection."""
    excluded = {c["case_id"] for c in validation_cases if c.get("case_id")}
    return [r for r in committed_records(workspace)
            if r.get("record_purpose") == "collection"
            and r.get("case_id") not in excluded]
