"""CPU startup checks for the adapter as packaged with the official repository."""
import os
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from client_server.ws.protocol.client import PolicyEvalClientConfig
from XPolicyLab.policy.R2SI import model, planning, robot


class Packaging(unittest.TestCase):
    def setUp(self):
        self.cfg = yaml.safe_load(Path(model.__file__).with_name("deploy.yml").read_text())

    def test_batch_mode_fails_before_loading_models(self):
        with patch.object(model.robot, "load_spec") as load:
            with self.assertRaisesRegex(ValueError, "one environment"):
                model.Model({**self.cfg, "eval_batch": True})
            load.assert_not_called()

    def test_incompatible_shared_dimensions_fail_before_loading_models(self):
        with patch.object(model, "get_robot_action_dim_info", return_value={"arm_dim": [7, 7], "ee_dim": [1, 1]}), patch.object(model.robot, "load_spec") as load:
            with self.assertRaisesRegex(ValueError, "arx_x5"):
                model.Model(self.cfg)
            load.assert_not_called()

    def test_external_assets_and_service_overrides(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            checkpoint = root / "sam3.pt"
            checkpoint.touch()
            stack.enter_context(patch.dict(os.environ, {"R2SI_SEGMENTER_CHECKPOINT": "sam3.pt", "R2SI_POLICY_SERVICE": "ws://example.invalid:18880"}))
            stack.enter_context(patch.object(robot, "WORKSPACE", root))
            stack.enter_context(patch.object(model, "get_robot_action_dim_info", return_value={"arm_dim": [6, 6], "ee_dim": [1, 1]}))
            for owner, name in ((robot, "load_spec"), (model.kinematics, "ArmModel"), (model.kinematics, "RobotShape"), (model.cameras, "Rig"), (planning, "Curobo")):
                stack.enter_context(patch.object(owner, name))
            segmenter = stack.enter_context(patch.object(model, "Segmenter"))
            adapter = model.Model(self.cfg)
            segmenter.assert_called_once_with(str(checkpoint))
            self.assertEqual(adapter.cfg["policy_service"], "ws://example.invalid:18880")
            adapter.close()

    def test_default_hold_precedes_official_rpc_deadline(self):
        self.assertGreater(self.cfg["keepalive_s"], 0)
        self.assertLess(self.cfg["keepalive_s"], PolicyEvalClientConfig(url="ws://example.invalid", evaluation_id="test").request_timeout_s)


if __name__ == "__main__":
    unittest.main()
