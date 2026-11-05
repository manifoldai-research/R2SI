"""Real subprocess isolation with tiny fixture policies, never model inference."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi.registry import Registry
from rsi.worker import VersionedModel


class WorkerVersions(unittest.TestCase):
    def test_spawn_loads_pinned_code_and_retired_trial_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = Registry(root / "runtime", verifier=lambda manifest: None)
            ids = []
            for version in ("before", "candidate"):
                release = root / version
                package = release / "XPolicyLab/policy/WorldScape_R2SI"
                package.mkdir(parents=True)
                for parent in (release/"XPolicyLab", release/"XPolicyLab/policy", package):
                    (parent/"__init__.py").write_text("")
                (package/"model.py").write_text(
                    "class Model:\n"
                    " def __init__(self,cfg): pass\n"
                    " def prepare_case(self,meta): return None\n"
                    " def reset(self): return None\n"
                    " def get_action(self): return " + repr(version) + "\n"
                    " def close(self): pass\n"
                    " def on_trial_end(self,result): return None\n")
                manifest = {"root": str(release), "harness_release_sha256": "fixture",
                            "checkpoints": {v: {"path": str(root/v)} for v in ("memory", "flash")},
                            "files": {p.relative_to(release).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in release.rglob("*.py")}}
                ids.append(registry.register(manifest))
            registry.initialize(ids[0])
            worker = VersionedModel(registry.ws.root, {}, registry=registry,
                                    validation_release=ids[1], timeout_s=10)
            worker.prepare_case({"_trial_identity": "fixture-evaluation:one"})
            try:
                worker.reset()
                self.assertEqual(worker.get_action(), "candidate")
                self.assertEqual(worker.release_metadata()["release_id"], ids[1])
                self.assertEqual(registry.status()["pointers"]["development"], ids[0])
                old = VersionedModel(registry.ws.root, {}, registry=registry, timeout_s=10)
                new = VersionedModel(registry.ws.root, {}, registry=registry, timeout_s=10)
                try:
                    old.reset()
                    registry.record_validation(ids[1], dict(baseline=ids[0],candidate=ids[1],
                        passed=True,evidence_kind='development_closed_loop'))
                    registry.promote(ids[1],ids[0])
                    new.reset()
                    self.assertEqual(old.get_action(),'before')
                    self.assertEqual(new.get_action(),'candidate')
                    self.assertEqual(worker.get_action(),'candidate')
                    self.assertEqual(old.release_metadata()['release_id'],ids[0])
                finally:
                    old.close();new.close()
                worker.on_trial_end({})
            finally:
                worker.close()
            again = VersionedModel(registry.ws.root, {}, registry=registry, timeout_s=10)
            again.prepare_case({"_trial_identity": "fixture-evaluation:one"})
            with self.assertRaisesRegex(RuntimeError, "already used"):
                again.reset()

    def test_official_candidate_override_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory, verifier=lambda manifest: None)
            with self.assertRaisesRegex(ValueError, "cannot override"):
                registry.pin("ep", mode="official", campaign="fixed", validation_release="other")

    def test_setup_failure_retires_trial_before_process_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(directory, verifier=lambda manifest: None)
            identity = registry.register({"root": directory})
            registry.initialize(identity)
            worker = VersionedModel(directory, {}, registry=registry)
            worker.prepare_case({"_trial_identity": "setup-failure"})
            with patch("rsi.worker.verify_code"), patch.object(
                    worker.ws, "environment", side_effect=OSError("workspace unavailable")):
                with self.assertRaisesRegex(OSError, "workspace unavailable"):
                    worker.reset()
            self.assertEqual(registry.status()["episodes"][0]["status"], "uncertain")
            self.assertIsNone(worker.process)
            self.assertEqual(worker.case, {})
            worker.prepare_case({"_trial_identity": "setup-failure"})
            with self.assertRaisesRegex(RuntimeError, "already used"):
                worker.reset()


if __name__ == "__main__":
    unittest.main()
