import copy
import json
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np

from rsi.storage import Workspace
from rsi.registry import Registry
from rsi.validation import compare
from rsi.experience import EpisodeRecorder, admit
from rsi.outer import OuterLoop


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        self.registry = Registry(self.ws, verifier=lambda value: None)
        self.old = self.registry.register({"fixture": 1})
        self.new = self.registry.register({"fixture": 2})
        self.registry.initialize(self.old)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reflection_uses_proposal_tool_and_real_completed_evidence(self):
        from types import SimpleNamespace
        row = dict(metadata(), record_purpose="collection", completed_at=1., terminal_confirmed=True, success=False, actions=2)
        proposal = dict(hypothesis='Test', evidence_episode_ids=[row['episode_id']],
                        skill_updates=[], training_requested=False, training_reason='',
                        expected_behavior=['Observe after execution'], risks=[])
        calls = []
        def complete(messages, tools):
            calls.append((messages, tools))
            return {'choices': [{'message': {'tool_calls': [{'id': 'p', 'function': {
                'name': 'propose_rsi_update', 'arguments': json.dumps(proposal)}}]}}]}
        result = OuterLoop(self.registry).reflect(SimpleNamespace(complete=complete), [row], {'grasp': 'text'}, set())
        self.assertEqual(result, proposal)
        self.assertEqual(calls[0][1][0]['function']['name'], 'propose_rsi_update')
        self.assertEqual(len(list(self.ws.path('proposals').glob('*.json'))), 1)

    def test_reflection_does_not_accept_unstructured_success_claim(self):
        from types import SimpleNamespace
        row = dict(metadata(), record_purpose="collection", completed_at=1., terminal_confirmed=True)
        client = SimpleNamespace(complete=lambda *_: {'choices': [{'message': {'content': 'success'}}]})
        with self.assertRaises(ValueError):
            OuterLoop(self.registry).reflect(client, [row], {}, set())

    def promote(self):
        self.registry.record_validation(self.new, {"baseline": self.old, "candidate": self.new,
            "passed": True, "evidence_kind": "development_closed_loop"})
        self.registry.promote(self.new, self.old)

    def test_hot_promotion_preserves_old_episode_and_frozen_campaign(self):
        self.registry.freeze_campaign("seed0", self.old)
        self.assertEqual(self.registry.pin("old"), self.old)
        self.promote()
        self.assertEqual(self.registry.pin("new"), self.new)
        self.assertEqual(self.registry.pin("official", "official", "seed0"), self.old)
        self.assertEqual(self.registry.status()["episodes"][0]["n"], 1)
        self.registry.finish("old")
        with self.assertRaises(RuntimeError):
            self.registry.pin("old")

    def test_restart_preserves_pin_and_uncertain_no_replay(self):
        self.registry.pin("interrupted")
        restarted = Registry(self.ws, verifier=lambda _: None)
        with self.assertRaises(RuntimeError):
            restarted.pin("interrupted")
        restarted.finish("interrupted", uncertain=True)
        with self.assertRaises(RuntimeError):
            restarted.pin("interrupted")

    def test_mock_or_stale_validation_cannot_promote(self):
        for kind in ["mock", "replay", "metadata_preflight"]:
            self.registry.record_validation(self.new, {"baseline": self.old, "candidate": self.new,
                "passed": True, "evidence_kind": kind})
            with self.assertRaises(ValueError):
                self.registry.promote(self.new, self.old)
        with self.assertRaises(RuntimeError):
            self.registry.promote(self.new, "stale")

    def test_campaign_cannot_mutate_and_rollback_only_accepted(self):
        import sqlite3
        self.registry.freeze_campaign("official", self.old)
        with self.assertRaises(sqlite3.IntegrityError):
            self.registry.freeze_campaign("official", self.new)
        with self.assertRaises(ValueError):
            self.registry.rollback(self.new, self.old)
        self.promote()
        self.registry.rollback(self.old, self.new)
        self.assertEqual(self.registry.pin("after_rollback"), self.old)

    def test_no_escape_and_cache_paths_are_scoped(self):
        with self.assertRaises(ValueError):
            self.ws.path("..", "escaped")
        for key in ["TMPDIR", "HF_HOME", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR"]:
            self.assertTrue(Path(self.ws.environment({key: "/root/wrong"})[key]).is_relative_to(self.ws.root))

    def test_outer_loop_preserves_explicit_development_seed_coverage(self):
        def fixture_evaluator(release, cases):
            result = rows(release)
            for row in result:
                row["seed"] += 2026102110
            return result
        report = OuterLoop(self.registry).evaluate_and_promote(
            self.new, self.old, fixture_evaluator, cases=[],
            required_capabilities=["grasp", "push"],
            required_seeds=[2026102110, 2026102111, 2026102112], promote=False)
        self.assertTrue(report["passed"])
        self.assertEqual(self.registry.status()["pointers"]["development"], self.old)


def rows(release):
    return [dict(case_id=f"heldout-{cap}-{seed}", seed=seed, capability=cap, release_id=release,
                 mode="development", category="standard", evidence_kind="development_closed_loop",
                 terminal_confirmed=True, success=True, score=100.0, actions=10, trace_sha256="f"*64)
            for cap in ["grasp", "push"] for seed in [0, 1, 2]]


class Validation(unittest.TestCase):
    def check(self, b=None, c=None, **kw):
        return compare("new", "old", b or rows("old"), c or rows("new"),
                       required_capabilities=["grasp", "push"], **kw)

    def test_paired_native_metrics_pass(self):
        self.assertTrue(self.check()["passed"])

    def test_per_capability_regression_fails(self):
        c = rows("new")
        c[0].update(success=False, score=0)
        self.assertFalse(self.check(c=c)["passed"])

    def test_mock_not_actual(self):
        c = rows("new")
        c[0]["evidence_kind"] = "replay"
        with self.assertRaises(ValueError):
            self.check(c=c)

    def test_infra_is_not_a_success_or_scored_failure(self):
        c = rows("new")
        c[0]["infrastructure_error"] = True
        with self.assertRaises(ValueError):
            self.check(c=c)

    def test_case_and_seed_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self.check(c=rows("new")[:-1])

    def test_train_validation_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.check(trained_cases=["heldout-grasp-0"])

    def test_missing_three_seed_coverage_rejected(self):
        with self.assertRaises(ValueError):
            self.check(required_seeds=[0])


def metadata():
    return dict(episode_id=uuid.uuid4().hex, task="development_grasp", release_id="a"*64,
                mode="development", split="development", category="standard",
                collection_authorized=True, hidden_state_used=False, layout_source="authorized-local-development")


def observation():
    state = {side + suffix: np.zeros(n, np.float32) for side in ["left", "right"]
             for suffix, n in [("_arm_joint_state", 6), ("_ee_joint_state", 1)]}
    # Red stays red through the shared RGB encoding/decoding helpers.
    image = np.zeros((24, 32, 3), np.uint8)
    image[..., 0] = 255
    return {"instruction": "Grasp the visible object", "state": state,
            "vision": {key: {"color": image.copy()} for key in ["cam_head", "cam_right_wrist", "cam_left_wrist"]}}


class Experience(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_official_open_unknown_provenance_rejected(self):
        for field, value in [("mode", "official"), ("split", "heldout"), ("category", "open"),
                             ("collection_authorized", False), ("hidden_state_used", True)]:
            m = metadata()
            m[field] = value
            with self.assertRaises(ValueError):
                admit(m)

    def test_acknowledgement_required_and_rgb_hdf5_roundtrip(self):
        import h5py
        from XPolicyLab.utils.process_data import decode_image_bit
        meta = metadata()
        recorder = EpisodeRecorder(self.ws, meta)
        obs = observation()
        with self.assertRaises(ValueError):
            recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=False)
        recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=True)
        result = recorder.finish(success=True, terminal_confirmed=True)
        self.assertFalse(result["recording_valid"])
        from experience_fixtures import publish_fixture
        result = publish_fixture(self.ws, meta, result)
        with h5py.File(result["path"], "r") as f:
            self.assertTrue(f.attrs["recording_valid"])
            image = decode_image_bit(f["vision/cam_head/colors"][0])
            self.assertGreater(image[..., 0].mean(), 240)
            self.assertLess(image[..., 2].mean(), 10)
            self.assertEqual(f["action/left_arm_joint_states"].shape, (1, 6))

    def test_uncertain_or_failure_not_published_to_training(self):
        for success, terminal in [(False, True), (True, False)]:
            recorder = EpisodeRecorder(self.ws, metadata())
            obs = observation()
            recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=True)
            result = recorder.finish(success=success, terminal_confirmed=terminal)
            self.assertFalse(result["recording_valid"])
            self.assertNotIn("path", result)

    def test_duplicate_ack_and_budget_rejected(self):
        recorder = EpisodeRecorder(self.ws, metadata())
        obs = observation()
        recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=True)
        with self.assertRaises(ValueError):
            recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=True)
        recorder.abort("fixture")
        recorder = EpisodeRecorder(self.ws, metadata(), max_bytes=1)
        with self.assertRaises(RuntimeError):
            recorder.executed(obs, obs["state"], obs, sequence=0, acknowledged=True)
        recorder.abort("fixture")

    def test_reflection_rejects_task_hard_routing(self):
        proposal = dict(hypothesis="test", evidence_episode_ids=["e"],
                        skill_updates=[{"id": "grasp", "text": "For cover_blocks always select memory. " * 4}],
                        training_requested=False, training_reason="", expected_behavior=[], risks=[])
        with self.assertRaises(ValueError):
            OuterLoop.validate_proposal(proposal, {"grasp": "old"}, {"e"}, {"cover_blocks"})


if __name__ == "__main__":
    unittest.main()
