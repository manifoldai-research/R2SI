"""Policy-only RPC service: shared weights, episode-local state, explicit commit.

The bundled RoboDojo WSP source provides the policy implementation. A worker
serializes GPU calls; each session uses the model-side fork_adapter isolation.
There is no Agent, simulator, experience collection or optimizer in this process.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time
import traceback
import uuid

from .storage import Workspace, canonical, sha256

OPERATIONS = {
    "reset": 0, "update_obs": 1, "get_action": 0,
    "commit_action": 1, "discard_action": 1, "close": 0,
}


@dataclass
class Session:
    trial: str
    profile: str
    worker: int
    seq: int = 0
    fingerprint: str | None = None
    response: dict | None = None
    failed: bool = False
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    lease_token: str = field(default_factory=lambda: secrets.token_hex(32))
    touched: float = field(default_factory=time.monotonic)
    expired: bool = False


class Controller:
    def __init__(self, workers, profiles, identities, *, max_trials=60, timeout=1200, lease_seconds=90, clock=time.monotonic):
        self.workers, self.profiles, self.identities = workers, profiles, identities
        self.max_trials, self.timeout = max_trials, timeout
        if lease_seconds < 1:
            raise ValueError("Lease must be at least one second")
        self.lease_seconds, self.clock = lease_seconds, clock
        self.expired_count = 0
        self.cleanup_errors = 0
        self.instance = uuid.uuid4().hex
        self.sessions = {}
        self.worker_locks = [asyncio.Lock() for _ in workers]
        self.faulted = set()
        self.started_at = time.time()
        self.calls = 0

    def active(self):
        return {k: s for k, s in self.sessions.items() if not s.closed}

    def open(self, key, trial, profile, identity):
        if profile not in self.profiles or identity != self.identities[profile]:
            raise ValueError("Policy identity does not match the pinned WAM Skill")
        if not all(isinstance(v, str) and 1 <= len(v) <= 128 for v in [key, trial]):
            raise ValueError("Invalid session or trial identity")
        if key in self.sessions:
            s = self.sessions[key]
            if (s.trial, s.profile) != (trial, profile) or s.closed or s.failed:
                raise ValueError("Session identity is retired, uncertain or reused")
            self._live(s)
            return self._opened(key, s, identity)
        active = self.active()
        trials = {s.trial for s in active.values()}
        if trial not in trials and len(trials) >= self.max_trials:
            raise ValueError("Capacity reached; wait for an active trial to finish")
        # One context per profile per trial prevents unbounded duplicate contexts.
        if any(s.trial == trial and s.profile == profile for s in active.values()):
            raise ValueError("This trial already owns the requested Policy context")
        available = [i for i in self.profiles[profile] if i not in self.faulted]
        if not available:
            raise RuntimeError("No healthy worker for this Policy")
        i = min(available, key=lambda i: sum(s.worker == i for s in active.values()))
        self.sessions[key] = Session(trial, profile, i, touched=self.clock())
        return self._opened(key, self.sessions[key], identity)

    def _opened(self, key, s, identity):
        return dict(instance=self.instance, session=key, identity=identity,
                    lease_token=s.lease_token, lease_seconds=self.lease_seconds)

    def _live(self, s):
        if s.expired or (not s.lock.locked() and self.clock() - s.touched >= self.lease_seconds):
            s.expired = True
            raise ValueError("Policy lease expired; this identity cannot be resumed")

    def heartbeat(self, envelope):
        if envelope.get("instance") != self.instance:
            raise ValueError("Policy instance changed")
        s = self.sessions.get(envelope.get("session"))
        if (s is None or s.closed or s.failed or not hmac.compare_digest(
                str(envelope.get("lease_token", "")), s.lease_token)):
            raise ValueError("Invalid or retired lease")
        self._live(s)
        s.touched = self.clock()
        return {"lease_seconds": self.lease_seconds}

    async def reap_expired(self):
        # Reclaim only identified expired contexts, never interrupt an in-flight
        # operation or guess a client sequence. Tombstones reject late replies.
        for key, s in list(self.sessions.items()):
            if s.closed or s.lock.locked():
                continue
            if not s.expired and self.clock() - s.touched < self.lease_seconds:
                continue
            async with s.lock:
                s.expired = True
                try:
                    await self._worker(s, key, "close", [])
                except Exception:
                    # If targeted release fails, dispose the owned worker so no
                    # unknown context is silently reused on it.
                    self.faulted.add(s.worker)
                    try:
                        await asyncio.to_thread(self.workers[s.worker].close)
                    except Exception:
                        self.cleanup_errors += 1
                    for other in self.sessions.values():
                        if other.worker == s.worker:
                            other.failed = other.closed = True
                finally:
                    s.closed = True
                    s.response = None
                    self.expired_count += 1

    async def _worker(self, s, key, op, args):
        async with self.worker_locks[s.worker]:
            if s.worker in self.faulted:
                raise RuntimeError("Worker unavailable; start a new trial after recovery")
            worker = self.workers[s.worker]
            future = asyncio.create_task(asyncio.to_thread(
                worker.call, "session_call", key, "release" if op == "close" else op, args))
            try:
                result = await asyncio.wait_for(asyncio.shield(future), self.timeout)
            except BaseException:
                self.faulted.add(s.worker)
                for item in self.sessions.values():
                    if item.worker == s.worker:
                        item.failed = True
                await asyncio.to_thread(worker.close)
                # Drain the interrupted read without leaving an orphan exception.
                try:
                    await future
                except BaseException:
                    pass
                raise
            if not result["ok"]:
                raise RuntimeError(result["error"])
            self.calls += 1
            return result.get("result")

    async def call(self, envelope, fingerprint):
        if envelope.get("instance") != self.instance:
            raise ValueError("Policy instance changed; do not replay this trial")
        key = envelope.get("session")
        if key not in self.sessions:
            raise ValueError("Unknown Policy session")
        s = self.sessions[key]
        op, args, seq = envelope.get("op"), envelope.get("args"), envelope.get("seq")
        if op not in OPERATIONS or not isinstance(args, list) or len(args) != OPERATIONS[op]:
            raise ValueError("Unsupported Policy operation")
        if type(seq) is not int:
            raise ValueError("Sequence must be an integer")
        self._live(s)
        async with s.lock:
            if s.expired:
                raise ValueError("Policy lease expired; do not replay")
            s.touched = self.clock()
            if seq == s.seq and s.response is not None:
                if fingerprint != s.fingerprint:
                    raise ValueError("A request sequence was reused with different content")
                return s.response
            if seq != s.seq + 1 or s.closed or (s.failed and op != "close"):
                raise ValueError("Out-of-order, retired or uncertain session")
            if seq == 1 and op not in {"reset", "close"}:
                raise ValueError("The first operation must reset a new session")
            try:
                if op == "update_obs":
                    from XPolicyLab.utils.process_data import decode_obs_images
                    # Shared helper supports both historical and marked RGB bytes.
                    args = [decode_obs_images(args[0])]
                result = await self._worker(s, key, op, args)
                response = {"ok": True, "result": result}
                if op == "close":
                    s.closed = True
            except Exception:
                s.failed = True
                if op == "close":
                    s.closed = True
                traceback.print_exc()
                response = {"ok": False, "error": "Policy call failed; this trial is uncertain"}
            s.touched = self.clock()
            s.seq, s.fingerprint, s.response = seq, fingerprint, response
            return response

    def status(self):
        active = self.active()
        return dict(instance=self.instance, trials=len({s.trial for s in active.values()}),
                    sessions=len(active), max_trials=self.max_trials, calls=self.calls,
                    faulted_workers=sorted(self.faulted), started_at=self.started_at,
                    lease_seconds=self.lease_seconds, expired_sessions=self.expired_count,
                    cleanup_errors=self.cleanup_errors)


def skill_identity(spec):
    return hashlib.sha256(canonical(spec)).hexdigest()


def verified_config(project, profile, gpu, checkpoints=None):
    policy_root = project/"XPolicyLab/policy/WorldScape_RSI"
    catalog = json.loads((project/"XPolicyLab/policy/WorldScape_RSI/harness/configs/wam_skills.json").read_text())
    spec = catalog["worldscape-wam-memory" if profile == "memory" else "worldscape-wam-nomemory"]
    checkpoint = Path((checkpoints or json.loads((project/"provenance/checkpoints.private.json").read_text()))[profile]["path"])
    for root, files in [(policy_root, spec["source_files"]), (checkpoint, spec["checkpoint_files"])]:
        for relative, expected in files.items():
            path = (root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or sha256(path) != expected:
                raise RuntimeError("Pinned model source/checkpoint verification failed")
    from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.runtime_config import resolve_checkpoint_deployment_config, validate_config
    cfg = {**spec["config"], "checkpoint_path": str(checkpoint),
           "worldscape_root": str(policy_root/"worldscape-policy"), "device": f"cuda:{gpu}",
           "multi_gpu": {"enabled": False}, "eval_batch": False}
    cfg = resolve_checkpoint_deployment_config(validate_config(cfg), checkpoint)
    return cfg, skill_identity(spec)


def uniform_worker(config):
    from XPolicyLab.policy.WorldScape_RSI.robodojo_server.session_model import WorkerSessions
    worker = WorkerSessions(config)
    base = worker.base
    if base.short_window_for_non_memory_tasks:
        base.short_window_for_non_memory_tasks = False
        base.task_window_chunks = (base.wam_max_chunk_size, base.wam_max_chunk_size)
        base.model_cfg.update(short_window_for_non_memory_tasks=False)
        base.model_cfg.pop("memory_window_chunks", None)
        base.model_cfg.pop("ordinary_window_chunks", None)
        base.reset()
    return worker


async def serve(project, workspace, host, ports, max_trials, *, device_profiles=None, checkpoints=None, ssl_context=None):
    from aiohttp import web
    from XPolicyLab.client_server.ws.protocol.codec import encode_frame, decode_frame
    from XPolicyLab.policy.WorldScape_RSI.xpolicylab_adapter.multi_gpu import ProcessWorker
    ws = Workspace(workspace)
    os.environ.update(ws.environment())
    token_file = ws.path("private/policy-token")
    token_file.parent.mkdir(exist_ok=True)
    if not token_file.exists():
        fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32))
    token = token_file.read_text().strip()
    workers, profiles, identities = [], {}, {}
    devices = device_profiles or {"flash": [0, 1], "memory": [2, 3]}
    tasks = set()
    try:
        for profile, ids in devices.items():
            profiles[profile] = []
            cfg, identity = verified_config(project, profile, ids[0], checkpoints)
            identities[profile] = identity
            for gpu in ids:
                ws.atomic("policy-status.json", {"state": "loading", "profile": profile, "gpu": gpu})
                profiles[profile].append(len(workers))
                workers.append(ProcessWorker(dict(cfg, device=f"cuda:{gpu}"), model_factory=uniform_worker))
        await asyncio.wait_for(asyncio.gather(*[asyncio.to_thread(w.ready) for w in workers]), 1200)
        controller = Controller(workers, profiles, identities, max_trials=max_trials)

        async def rpc(request):
            if not hmac.compare_digest(request.headers.get("Authorization", ""), "Bearer "+token):
                raise web.HTTPUnauthorized()
            body = await request.read()
            digest = hashlib.sha256(body).hexdigest()
            try:
                message = decode_frame(body)
                if message.get("op") == "open":
                    result = controller.open(message.get("session"), message.get("trial"),
                                             message.get("profile"), message.get("identity"))
                    response = {"ok": True, "result": result}
                elif message.get("op") == "heartbeat":
                    response = {"ok": True, "result": controller.heartbeat(message)}
                elif message.get("op") == "status":
                    response = {"ok": True, "result": controller.status()}
                else:
                    # Client disconnects cannot cancel an in-flight model operation.
                    task = asyncio.create_task(controller.call(message, digest))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
                    response = await asyncio.shield(task)
            except (ValueError, TypeError, KeyError):
                response = {"ok": False, "error": "Invalid identity, sequence, capacity or request"}
            return web.Response(body=encode_frame(response), content_type="application/msgpack")

        async def health(request):
            return web.json_response({"ready": len(controller.faulted) == 0})

        app = web.Application(client_max_size=16*1024*1024)
        app.router.add_post("/server", rpc)
        app.router.add_get("/", health)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        for port in ports:
            await web.TCPSite(runner, host, port, ssl_context=ssl_context).start()
        ws.atomic("policy-status.json", dict(state="ready", **controller.status(),
            identities=identities, ports=ports, profiles=profiles,
            throughput_validated=False, native_evaluation_completed=False))
        async def lease_sweeper():
            while True:
                await asyncio.sleep(min(5, controller.lease_seconds / 3))
                await controller.reap_expired()
        sweeper = asyncio.create_task(lease_sweeper())
        try:
            await asyncio.Event().wait()
        finally:
            sweeper.cancel()
            await asyncio.gather(sweeper, return_exceptions=True)
            await runner.cleanup()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for worker in workers:
            worker.close()
