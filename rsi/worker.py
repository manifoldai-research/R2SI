"""Episode-isolated policy process. No production sockets, resets or replay."""
from __future__ import annotations

import multiprocessing
import hashlib
import os
from pathlib import Path
import sys
import threading
import traceback
import uuid

from .registry import Registry, verify_code
from .storage import Workspace


def _serve(connection, manifest, config, environment):
    model = None
    try:
        os.environ.update(environment)
        root = Path(manifest["root"])
        sys.path[:0] = [str(root), str(root / "XPolicyLab")]
        # Each spawn starts with a fresh import cache: model code and the skill
        # catalog cannot leak between concurrent or old/new versions.
        # spawn re-imports the launch module, which can import the source-tree
        # protocol package. Remove those parents before importing a candidate.
        for name in list(sys.modules):
            if name == "XPolicyLab" or name.startswith("XPolicyLab.") or name == "client_server" or name.startswith("client_server."):
                del sys.modules[name]
        from XPolicyLab.policy.WorldScape_R2SI.model import Model
        import inspect
        if Path(inspect.getfile(Model)).resolve() != (
                root / "XPolicyLab/policy/WorldScape_R2SI/harness/model.py").resolve():
            # The integration model may be a subclass defined in model.py.
            if not Path(inspect.getfile(Model)).resolve().is_relative_to(root):
                raise RuntimeError("Policy imported outside the pinned release")
        model = Model(config)
        connection.send(("ok", None))
        while True:
            operation, args = connection.recv()
            if operation == "close":
                model.close()
                connection.send(("ok", None))
                break
            if operation not in {"prepare_case", "reset", "update_obs", "get_action", "release_metadata", "on_trial_end", "drain_trace"}:
                raise ValueError("Unadvertised policy operation")
            connection.send(("ok", getattr(model, operation)(*args)))
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send(("error", {"type": type(error).__name__, "message": str(error),
                                       "traceback": traceback.format_exc(limit=10),
                                       "trace": model.drain_trace() if model is not None and hasattr(model, "drain_trace") else []}))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if model is not None:
            try:
                model.close()
            except Exception:
                pass
        connection.close()


class VersionedModel:
    """Standard single-environment adapter; each reset pins the current version.

    A process failure makes the trial uncertain, never silently replays it.
    Official campaign pinning remains fixed even when development is promoted.
    """
    def __init__(self, workspace, model_cfg, *, mode="development", campaign=None,
                 memory_device="cuda:0", flash_device="cuda:0", timeout_s=300,
                 registry=None, validation_release=None):
        self.ws = Workspace(workspace)
        self.registry = registry or Registry(self.ws)
        self.config = dict(model_cfg)
        self.mode, self.campaign = mode, campaign
        self.validation_release = validation_release
        self.devices = memory_device, flash_device
        self.timeout_s = timeout_s
        self.process = self.pipe = self.episode = self.release_id = None
        self.case = {}
        self.failed = False
        self.failure_trace = []
        self.lock = threading.RLock()

    def prepare_case(self, meta):
        if self.process is not None:
            raise RuntimeError("An active episode cannot change metadata")
        self.case.update(meta)
        return {"ok": True}

    def _rpc(self, operation, *args):
        if self.failed:
            raise RuntimeError("Uncertain trial cannot be continued")
        if self.pipe is None or self.process is None:
            raise RuntimeError("Reset must successfully pin an episode first")
        try:
            self.pipe.send((operation, args))
            if not self.pipe.poll(self.timeout_s):
                raise TimeoutError("Policy worker timed out; execution is uncertain")
            kind, value = self.pipe.recv()
            if kind != "ok":
                self.failure_trace = value.pop("trace", [])
                raise RuntimeError(value)
            return value
        except BaseException:
            self.failed = True
            raise

    def reset(self):
        with self.lock:
            if self.process is not None:
                raise RuntimeError("Finish the active episode before reset")
            self.failed = False
            external = self.case.get("_trial_identity")
            self.episode = (hashlib.sha256(external.encode()).hexdigest() if external else uuid.uuid4().hex)
            self.release_id = self.registry.pin(self.episode, self.mode, self.campaign, self.validation_release)
            child = None
            try:
                manifest = self.registry.get(self.release_id)
                verify_code(manifest)
                root = Path(manifest["root"])
                env = self.ws.environment()
                env.update({
                    "WRSI_EPISODE_ID": self.episode,
                    "WRSI_WAM_MEMORY_CHECKPOINT": manifest["checkpoints"]["memory"]["path"],
                    "WRSI_WAM_NOMEMORY_CHECKPOINT": manifest["checkpoints"]["flash"]["path"],
                    "WRSI_WAM_MEMORY_DEVICE": self.devices[0], "WRSI_WAM_NOMEM_DEVICE": self.devices[1],
                    "WRSI_WAM_NOMEMORY_DEVICE": self.devices[1],
                    "WRSI_WORLDSCAPE_ROOT": str(root / "XPolicyLab/policy/WorldScape_R2SI/worldscape-policy"),
                    "WORLDSCAPE_POLICY_ROOT": str(root / "XPolicyLab/policy/WorldScape_R2SI/worldscape-policy"),
                })
                if manifest.get("policy_client_config"):
                    env["WRSI_POLICY_CLIENT_CONFIG"] = manifest["policy_client_config"]
                env["R2SI_DIAGNOSTICS_DIR"] = str(self.ws.path("diagnostics", self.episode))
                cfg = dict(self.config, release_path=str(root / "XPolicyLab/policy/WorldScape_R2SI/harness/release.json"),
                           release_sha256=manifest["harness_release_sha256"])
                context = multiprocessing.get_context("spawn")
                self.pipe, child = context.Pipe()
                self.process = context.Process(target=_serve, args=(child, manifest, cfg, env))
                self.process.start()
                child.close()
                if not self.pipe.poll(self.timeout_s):
                    raise TimeoutError("Policy worker failed to initialize")
                kind, result = self.pipe.recv()
                if kind != "ok":
                    raise RuntimeError(result)
                self._rpc("prepare_case", self.case)
                return self._rpc("reset")
            except BaseException:
                self.failed = True
                if child is not None:
                    child.close()
                if self.process is None:
                    if self.pipe is not None:
                        self.pipe.close()
                        self.pipe = None
                    self.registry.finish(self.episode, uncertain=True)
                    self.case = {}
                else:
                    self.close()
                raise

    def update_obs(self, observation):
        with self.lock:
            return self._rpc("update_obs", observation)

    def get_action(self):
        with self.lock:
            return self._rpc("get_action")

    def drain_trace(self):
        with self.lock:
            if self.failed:
                result,self.failure_trace=self.failure_trace,[]
                return result
            return self._rpc("drain_trace")

    def release_metadata(self):
        return {"release_id": self.release_id, "mode": self.mode, "campaign": self.campaign,
                "task_metadata_used_for_routing": False}

    def on_trial_end(self, result):
        with self.lock:
            try:
                return self._rpc("on_trial_end", result)
            finally:
                self.close()

    def close(self):
        with self.lock:
            if self.process is None:
                return
            process = self.process
            try:
                if process.is_alive() and not self.failed:
                    self._rpc("close")
            finally:
                if process.pid is not None:
                    process.join(5)
                if process.is_alive():
                    self.failed = True
                    process.terminate()
                    process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
                self.pipe.close()
                self.process = self.pipe = None
                self.registry.finish(self.episode, uncertain=self.failed)
                self.case = {}
