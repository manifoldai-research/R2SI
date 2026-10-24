"""Atomic promotion; active episodes and official campaigns retain exact versions."""
from __future__ import annotations
from .official_layouts import POLICY, EVIDENCE

import hashlib
import json
import time
from pathlib import Path

from .storage import Workspace, canonical, sha256


def verify_code(manifest):
    root = Path(manifest["root"]).resolve(strict=True)
    if not manifest.get("files"):
        raise ValueError("An immutable release requires a complete file manifest")
    for rel, expected in manifest["files"].items():
        p = root / rel
        if (p.is_symlink() or not p.resolve().is_relative_to(root)
                or not p.is_file() or sha256(p) != expected):
            raise ValueError("Release file mismatch: " + rel)


def verify_release(manifest):
    verify_code(manifest)
    if set(manifest.get("checkpoints", {})) != {"memory", "flash"}:
        raise ValueError("Both WSP variants must be supplied")
    for spec in manifest["checkpoints"].values():
        checkpoint = Path(spec["path"]).resolve(strict=True)
        if not spec.get("files"):
            raise ValueError("Checkpoint content hashes are required")
        for rel, expected in spec["files"].items():
            p = checkpoint / rel
            if (p.is_symlink() or not p.resolve().is_relative_to(checkpoint)
                    or not p.is_file() or sha256(p) != expected):
                raise ValueError("Checkpoint mismatch: " + rel)


class Registry:
    def __init__(self, workspace, verifier=verify_release):
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self.verifier = verifier
        with self.ws.db() as db:
            for statement in (
                "CREATE TABLE IF NOT EXISTS releases(id TEXT PRIMARY KEY, manifest TEXT NOT NULL, validation TEXT)",
                "CREATE TABLE IF NOT EXISTS pointers(name TEXT PRIMARY KEY, release_id TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY, release_id TEXT NOT NULL, mode TEXT NOT NULL, status TEXT NOT NULL, started REAL NOT NULL, ended REAL)",
                "CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, time REAL NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)",
            ):
                db.execute(statement)

    def register(self, manifest):
        self.verifier(manifest)
        value = canonical(manifest).decode()
        identity = hashlib.sha256(value.encode()).hexdigest()
        with self.ws.db() as db:
            db.execute("INSERT OR IGNORE INTO releases(id,manifest) VALUES(?,?)", (identity, value))
        return identity

    def get(self, release_id):
        with self.ws.db() as db:
            row = db.execute("SELECT manifest FROM releases WHERE id=?", (release_id,)).fetchone()
        if row is None:
            raise KeyError(release_id)
        return json.loads(row["manifest"])

    def initialize(self, release_id):
        self.verifier(self.get(release_id))
        with self.ws.db() as db:
            if db.execute("SELECT 1 FROM pointers WHERE name='development'").fetchone():
                raise RuntimeError("A baseline is already selected")
            db.execute("INSERT INTO pointers VALUES('development',?)", (release_id,))
            self._event(db, "baseline", {"release_id": release_id})

    def record_validation(self, release_id, report):
        if report.get("candidate") != release_id or not report.get("baseline"):
            raise ValueError("Validation must bind candidate and baseline hashes")
        self.get(release_id)
        self.get(report["baseline"])
        with self.ws.db() as db:
            db.execute("UPDATE releases SET validation=? WHERE id=?", (canonical(report).decode(), release_id))

    def promote(self, release_id, expected_baseline):
        self.verifier(self.get(release_id))
        with self.ws.db() as db:
            active = db.execute("SELECT release_id FROM pointers WHERE name='development'").fetchone()
            row = db.execute("SELECT validation FROM releases WHERE id=?", (release_id,)).fetchone()
            report = json.loads(row["validation"]) if row and row["validation"] else {}
            if active is None or active[0] != expected_baseline:
                raise RuntimeError("Baseline changed; compare the candidate again")
            if (report.get("passed") is not True or report.get("baseline") != expected_baseline
                    or report.get("evidence_kind") not in {"development_closed_loop", EVIDENCE}
                    or (report.get("evidence_kind") == EVIDENCE and (report.get("layout_policy") != POLICY or report.get("training_eligible") is not False))
                    or report.get("candidate") != release_id):
                raise ValueError("Fresh real development validation is required")
            db.execute("UPDATE pointers SET release_id=? WHERE name='development'", (release_id,))
            self._event(db, "promote", {"from": expected_baseline, "to": release_id})

    def freeze_campaign(self, name, release_id):
        self.verifier(self.get(release_id))
        with self.ws.db() as db:
            # An existing campaign can never be overwritten, including after
            # restart. Start a separately named campaign to evaluate a new model.
            db.execute("INSERT INTO pointers VALUES(?,?)", ("official:" + name, release_id))
            self._event(db, "freeze", {"campaign": name, "release_id": release_id})

    def pin(self, episode_id, mode="development", campaign=None, validation_release=None):
        if mode not in {"development", "official"} or (mode == "official" and not campaign):
            raise ValueError("Unknown mode or missing frozen official campaign")
        pointer = "development" if mode == "development" else "official:" + campaign
        if validation_release:
            if mode != "development":
                raise ValueError("An official campaign cannot override its frozen release")
            self.verifier(self.get(validation_release))
        with self.ws.db() as db:
            previous = db.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
            if previous:
                raise RuntimeError("Episode identity is already used; no replay or implicit reset")
            row = ((validation_release,) if validation_release else
                   db.execute("SELECT release_id FROM pointers WHERE name=?", (pointer,)).fetchone())
            if row is None:
                raise RuntimeError("No release selected for this mode/campaign")
            db.execute("INSERT INTO episodes VALUES(?,?,?,?,?,NULL)",
                       (episode_id, row[0], mode, "active", time.time()))
            return row[0]

    def finish(self, episode_id, uncertain=False):
        with self.ws.db() as db:
            cursor = db.execute("UPDATE episodes SET status=?,ended=? WHERE id=? AND status='active'",
                                ("uncertain" if uncertain else "ended", time.time(), episode_id))
            if cursor.rowcount != 1:
                raise RuntimeError("Episode is not active")

    def rollback(self, release_id, expected_current):
        self.verifier(self.get(release_id))
        with self.ws.db() as db:
            row = db.execute("SELECT release_id FROM pointers WHERE name='development'").fetchone()
            if row is None or row[0] != expected_current:
                raise RuntimeError("Current release changed")
            # Rollback targets must have previously been an accepted baseline.
            accepted = [json.loads(x[0]) for x in db.execute(
                "SELECT payload FROM events WHERE kind IN ('baseline','promote')")]
            if not any(x.get("release_id", x.get("to")) == release_id for x in accepted):
                raise ValueError("Rollback target was never accepted")
            db.execute("UPDATE pointers SET release_id=? WHERE name='development'", (release_id,))
            self._event(db, "rollback", {"from": expected_current, "to": release_id})

    def status(self):
        with self.ws.db() as db:
            return {"pointers": {r[0]: r[1] for r in db.execute("SELECT * FROM pointers")},
                    "episodes": [dict(r) for r in db.execute(
                        "SELECT release_id,mode,status,count(*) AS n FROM episodes GROUP BY release_id,mode,status")]}

    @staticmethod
    def _event(db, kind, payload):
        db.execute("INSERT INTO events(time,kind,payload) VALUES(?,?,?)",
                   (time.time(), kind, canonical(payload).decode()))
