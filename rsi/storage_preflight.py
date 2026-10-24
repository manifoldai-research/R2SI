"""Fail before dispatch on storage without durable POSIX publication/SQLite locks."""
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import uuid

def check_storage(ws):
    if getattr(ws, "_storage_checked", False):
        return
    probe = ws.path(".storage-probes", uuid.uuid4().hex)
    probe.mkdir(parents=True, exist_ok=False)
    connection = None
    try:
        pending = probe / "pending"
        pending.mkdir()
        payload = pending / "probe"
        with payload.open("xb") as f:
            f.write(b"rsi-storage-check\n"); f.flush(); os.fsync(f.fileno())
        committed = probe / "committed"
        os.rename(pending, committed)
        if (committed / "probe").read_bytes() != b"rsi-storage-check\n":
            raise RuntimeError("Rename/readback mismatch")
        fd = os.open(probe, os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)
        database = probe / "probe.sqlite"
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE probe (value INTEGER)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        code = """import sqlite3,sys
c=sqlite3.connect(sys.argv[1],timeout=0.15)
try:
 c.execute('BEGIN IMMEDIATE')
except sqlite3.OperationalError as e:
 sys.exit(0 if 'locked' in str(e).lower() else 3)
else:
 c.rollback()
 sys.exit(2)
"""
        result = subprocess.run([sys.executable, "-B", "-c", code, str(database)],
                                capture_output=True, timeout=10)
        if result.returncode != 0:
            raise RuntimeError("Independent process did not observe the SQLite writer lock")
        connection.rollback(); connection.close(); connection = None
        # This verifies basic local capabilities, not multi-host mount correctness.
        ws.atomic("storage-capability.json", dict(schema="rsi.storage-check.v1",
                  atomic_directory_rename=True, directory_fsync=True,
                  sqlite_cross_process_lock=True, hardlinks_required=False))
        ws._storage_checked = True
    except (OSError, sqlite3.Error, subprocess.SubprocessError, RuntimeError) as error:
        raise RuntimeError("Workspace cannot safely publish RSI experience; use a POSIX "
                           "filesystem with durable rename/fsync and SQLite locks: " + str(error)) from error
    finally:
        if connection is not None:
            connection.rollback(); connection.close()
        # Only our unique probe directory, never user data or another process's probe.
        if probe.is_dir() and probe.resolve().is_relative_to(ws.root):
            shutil.rmtree(probe)
