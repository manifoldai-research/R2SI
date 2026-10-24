"""All writes are explicitly scoped to one workspace; no HOME/tmp fallback."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Workspace:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_absolute() or self.root == Path(self.root.anchor):
            raise ValueError("A dedicated absolute workspace is required")
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, *parts):
        p = self.root.joinpath(*parts)
        if not p.resolve().is_relative_to(self.root):
            raise ValueError("Path escapes the workspace")
        return p

    def atomic(self, relative, value):
        p = self.path(relative)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name("." + p.name + "." + uuid.uuid4().hex + ".tmp")
        with tmp.open("xb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, p)
        if os.name != "nt":
            fd = os.open(p.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return p

    def environment(self, base=None):
        env = dict(os.environ if base is None else base)
        # Explicit shared-storage temp roots survive nested policy-worker setup.
        # NVRTC rejects very long temporary paths even when the filesystem can
        # store them. Do not inherit arbitrary TMPDIR or fall back to /tmp.
        temp_base = env.get("R2SI_TEMP_ROOT")
        storage_root = env.get("R2SI_STORAGE_ROOT")
        short_temp = None
        if temp_base is not None:
            if not storage_root:
                raise ValueError("An explicit temporary root requires R2SI_STORAGE_ROOT")
            storage = Path(storage_root)
            temporary = Path(temp_base)
            if (not storage.is_absolute() or not temporary.is_absolute()
                    or storage.resolve() == Path(storage.anchor)):
                raise ValueError("Use dedicated absolute shared-storage paths")
            storage = storage.resolve(strict=True)
            temporary = temporary.resolve()
            if (not self.root.is_relative_to(storage)
                    or not temporary.is_relative_to(storage)):
                raise ValueError("Workspace and temporary root must remain in shared storage")
            temporary.mkdir(parents=True, exist_ok=True)
            if temporary.stat().st_dev != self.root.stat().st_dev:
                raise ValueError("Temporary root must be on the workspace filesystem")
            identity = hashlib.sha256(str(self.root).encode()).hexdigest()[:16]
            short_temp = temporary / identity
            if short_temp.exists() and short_temp.resolve() != short_temp:
                raise ValueError("Temporary directory cannot redirect through a symlink")
            short_temp.mkdir(exist_ok=True)
        for key, rel in {
            "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
            "XDG_CACHE_HOME": "cache", "HF_HOME": "cache/huggingface",
            "TORCH_HOME": "cache/torch", "TRITON_CACHE_DIR": "cache/triton",
            "CUDA_CACHE_PATH": "cache/cuda", "WANDB_DIR": "logs/wandb",
            "MPLCONFIGDIR": "cache/matplotlib", "NUMBA_CACHE_DIR": "cache/numba",
            "TORCH_EXTENSIONS_DIR": "cache/torch_extensions", "WARP_CACHE_PATH": "cache/warp",
            "PIP_CACHE_DIR": "cache/pip", "XDG_CONFIG_HOME": "config", "XDG_DATA_HOME": "data",
        }.items():
            p = short_temp if short_temp is not None and key in {"TMPDIR", "TMP", "TEMP"} else self.path(rel)
            p.mkdir(parents=True, exist_ok=True)
            env[key] = str(p)
        env.update(PYTHONDONTWRITEBYTECODE="1", WSP_WRITABLE_ROOTS=str(self.root),
                   WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
        return env

    @contextmanager
    def db(self):
        # DELETE journal is compatible with mounted storage where WAL/shared
        # memory may be unsupported. Storage must support SQLite POSIX locks.
        connection = sqlite3.connect(self.path("control.sqlite3"), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
