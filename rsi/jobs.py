"""Bounded actual subprocess jobs; no shell interpolation or production endpoint."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .storage import Workspace, sha256


class Jobs:
    def __init__(self, workspace):
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)

    def run(self, argv, *, cwd, job_kind, timeout_s, environment=None):
        if not argv or timeout_s <= 0 or job_kind not in {"training", "development_evaluation", "preflight"}:
            raise ValueError("Explicit bounded development job required")
        from .storage_preflight import check_storage
        check_storage(self.ws)
        job = uuid.uuid4().hex
        directory = self.ws.path("jobs", job)
        directory.mkdir(parents=True)
        env = self.ws.environment()
        env.update(environment or {})
        # Environment supplied by a job cannot redirect caches/output outside workspace.
        env = self.ws.environment(env)
        env["RSI_MODE"] = "development"
        started = time.time()
        result = {"id": job, "kind": job_kind, "argv": [str(x) for x in argv],
                  "cwd": str(Path(cwd).resolve()), "started_at": started, "status": "running"}
        self.ws.atomic(f"jobs/{job}/status.json", result)
        with (directory / "output.log").open("xb") as log:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=(os.name != "nt"))
            result["pid"] = process.pid
            self.ws.atomic(f"jobs/{job}/status.json", result)
            try:
                code = process.wait(timeout=timeout_s)
                result.update(exit_code=code, status="completed" if code == 0 else "failed")
            except subprocess.TimeoutExpired:
                # Only this job's dedicated group is signalled; never existing jobs.
                if os.name == "nt":
                    process.kill()
                else:
                    import signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                result.update(exit_code=process.returncode, status="timed_out")
        result.update(elapsed_s=time.time() - started, log_sha256=sha256(directory / "output.log"))
        self.ws.atomic(f"jobs/{job}/status.json", result)
        return result

    def train(self, *, python, xpolicy_root, config, snapshot, gpus, max_steps, timeout_s, overrides=()):
        """Reserved WSP training hook; the backend is not connected yet."""
        from XPolicyLab.policy.WorldScape_R2SI.training import train_wsp
        return train_wsp(python=python, xpolicy_root=xpolicy_root, config=config,
                         snapshot=snapshot, gpus=gpus, max_steps=max_steps,
                         timeout_s=timeout_s, overrides=overrides)
