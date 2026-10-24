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
        """Start the teammate's native trainer with its collective hotload enabled.

        Config must name real base weights, transforms/statistics and data root.
        No synthetic optimizer result or model export is invented.
        """
        from worldscape_policy.data.robodojo_snapshots import SnapshotStore
        if not gpus or len(set(gpus)) != len(gpus) or any(type(x) is not int or x < 0 for x in gpus):
            raise ValueError("Select explicitly reserved GPU indices")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("A positive training step budget is required")
        for gpu in gpus:
            occupied = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                text=True, timeout=10).strip()
            if occupied:
                raise RuntimeError(f"GPU {gpu} is occupied; reserve resources before training")
        store = SnapshotStore(self.ws.path("datasets/snapshots"))
        record = store.resolve(snapshot, verify_files=True)
        # The native loader initializes against immutable version zero, then
        # consumes later published snapshots. Passing a merged index as the
        # base breaks its statistics/index identity check on the next round.
        bootstrap = snapshot
        while record['manifest'].get('previous'):
            bootstrap = record['manifest']['previous']
            record = store.resolve(bootstrap, verify_files=True)
        output = self.ws.path("checkpoints", uuid.uuid4().hex)
        output.mkdir(parents=True)
        model_root = Path(xpolicy_root) / "policy/WorldScape_RSI/worldscape-policy"
        argv = [str(python), "-B", "-m", "torch.distributed.run",
                "--standalone", "--nproc_per_node=" + str(len(gpus)),
                "-m", "worldscape_policy.cli.train", "--config", str(Path(config).resolve()),
                *overrides,
                "data_loader.hotload.enabled=true",
                "data_loader.hotload.manifest_dir=" + str(store.directory),
                "data_loader.hotload.journal_dir=" + str(output / "data_versions"),
                "data_loader.dataset_kwargs.precomputed_index_path=" + record["index_path"],
                "training.checkpoint_dir=" + str(output),
                "training.max_steps=" + str(max_steps)]
        environment = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)),
                       "PYTHONPATH": os.pathsep.join([str(Path(xpolicy_root).parent),
                                                     str(xpolicy_root), str(model_root / "src")])}
        job = self.run(argv, cwd=model_root, job_kind="training", timeout_s=timeout_s, environment=environment)
        job.update(checkpoint_dir=str(output), starting_snapshot=bootstrap, requested_snapshot=snapshot)
        self.ws.atomic("jobs/" + job["id"] + "/status.json", job)
        return job
