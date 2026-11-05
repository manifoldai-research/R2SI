"""CPU native-path fixtures only; no claim of contact or simulator success."""
import copy
import json
import types
import unittest

import numpy as np

from test_harness import backends, obs


class CandidateMotionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backends.register_vendor()
        from XPolicyLab.policy.RoboRSI_RoboProbe import tools
        from XPolicyLab.policy.RoboRSI_RoboProbe.candidate_motion import move_candidates
        cls.tools = tools
        cls.select = staticmethod(move_candidates)

    def setUp(self):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import RoboDojoActionSpec
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.types import ActionSpace, Observation
        from XPolicyLab.policy.RoboRSI_RoboProbe.policy import SkillEefPolicy
        labels = tuple(ActionSpace().flat_labels())
        spec = RoboDojoActionSpec(labels, np.full(len(labels), -10.), np.full(len(labels), 10.),
                                  25., 'CPU fixture', tuple(.2 for _ in labels))
        self.calls = []
        self.paths = [self.path(1.), self.path(.2)]
        def planner(**kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            result = self.paths[len(self.calls)-1]
            if isinstance(result, Exception):
                raise result
            return result
        self.actor = SkillEefPolicy(action_spec=spec, env={}, planner=planner,
                                   client=types.SimpleNamespace(config={}))
        raw = obs()
        self.obs = Observation(state=raw['state'], images={}, instruction=raw['instruction'], step=7)
        self.args = dict(op='move_candidates', arm='left', position=[0., 0., 1.],
                         quaternions_wxyz=[[1., 0., 0., 0.], [0., 0., 0., 1.]],
                         contact_evidence='Both offered orientations match the reviewed contact geometry.')
        self.original_targets = copy.deepcopy(self.actor._last_targets)
        self.original_grippers = copy.deepcopy(self.actor._gripper_goals)

    @staticmethod
    def path(value):
        end = np.zeros(6); end[0] = value
        return {'status': 'Success', 'position': np.array([np.zeros(6), end])}

    def run_tool(self):
        return self.tools.execute(self.actor, 'call_tool',
                                  {'id': 'robot-action-timing-tool', 'arguments': self.args}, self.obs)

    def test_selects_smaller_excursion_and_reuses_exact_actions(self):
        outcomes = []
        original = self.actor.native_move
        def capture(*args):
            outcome = original(*args); outcomes.append(outcome); return outcome
        self.actor.native_move = capture
        result = self.run_tool()
        self.assertEqual(len(self.calls), 2)  # No winner replan / second IK.
        self.assertIs(result.chunk.actions, outcomes[1].chunk.actions)
        report = result.chunk.meta['candidate_selection']
        self.assertEqual(report['selected_index'], 1)
        self.assertEqual(report['observation_step'], 7)
        self.assertEqual(self.actor._gripper_goals, self.original_grippers)
        np.testing.assert_allclose(self.actor._last_targets['left'], self.calls[1]['target_pose'])
        for action in result.chunk.actions:
            np.testing.assert_array_equal(action.data['right_arm_joint_state'], np.zeros(6))

    def test_same_grasp_point_with_different_flange_origins(self):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect_EEF import pose
        self.run_tool()
        for call in self.calls:
            values = pose.pose_to_values(call['target_pose'])
            np.testing.assert_allclose([values[k] for k in pose.POSITION_AXES], self.args['position'], atol=1e-7)
        self.assertFalse(np.allclose(self.calls[0]['target_pose'][:3], self.calls[1]['target_pose'][:3]))

    def test_raw_multi_turn_travel_is_not_wrapped(self):
        self.paths = [self.path(2*np.pi+.01), self.path(.3)]
        result = self.run_tool()
        report = result.chunk.meta['candidate_selection']
        self.assertEqual(report['selected_index'], 1)
        self.assertGreater(report['candidates'][0]['total_joint_travel_rad'], 6.)

    def test_whole_path_excursion_not_only_endpoint(self):
        self.paths[0]['position'] = np.array([np.zeros(6), np.full(6, 2.), np.full(6, .01)])
        self.assertEqual(self.run_tool().chunk.meta['candidate_selection']['selected_index'], 1)

    def test_quaternion_sign_and_tie_order(self):
        self.args['quaternions_wxyz'] = [[1., 0., 0., 0.], [-1., 0., 0., 0.]]
        self.paths = [self.path(.2), self.path(.2)]
        report = self.run_tool().chunk.meta['candidate_selection']
        self.assertEqual(report['selected_index'], 0)
        self.assertEqual([r['flange_rotation_rad'] for r in report['candidates']], [0., 0.])

    def test_known_unreachable_candidate_and_total_failure(self):
        self.paths[0] = {'status': 'Fail', 'position': None}
        self.assertEqual(self.run_tool().chunk.meta['candidate_selection']['selected_index'], 1)
        self.calls.clear()
        prior = copy.deepcopy(self.actor._last_targets)
        self.paths = [{'status': 'Fail', 'position': None}] * 2
        result = self.run_tool()
        self.assertIsNone(result.chunk)
        self.assertIsNone(json.loads(result.tool_result)['selected_index'])
        np.testing.assert_array_equal(self.actor._last_targets['left'], prior['left'])

    def test_exception_discards_earlier_plan_and_restores_state(self):
        self.paths[1] = RuntimeError('planner infrastructure unavailable')
        with self.assertRaisesRegex(RuntimeError, 'infrastructure'):
            self.run_tool()
        self.assertEqual(self.actor._last_targets, self.original_targets)
        self.assertEqual(self.actor._gripper_goals, self.original_grippers)

    def test_native_start_and_joint_limits_still_enforced(self):
        from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import InfrastructureFailure
        for path in [np.ones((2, 6)), np.array([np.zeros(6), np.full(6, 11.)])]:
            with self.subTest(path=path):
                self.calls.clear(); self.paths[0] = {'status':'Success', 'position':path}
                with self.assertRaises(InfrastructureFailure) as caught:
                    self.run_tool()
                self.assertIn('planner_', str(caught.exception))
                self.assertEqual(len(self.calls), 1)
                self.assertEqual(self.actor._last_targets, self.original_targets)

    def test_all_inputs_validated_before_any_planning(self):
        cases = [dict(quaternions_wxyz=[[1,0,0,0],[0,0,0,0]]),
                 dict(quaternions_wxyz=[[1,0,0,0],[float('nan'),0,0,1]]),
                 dict(quaternions_wxyz=[[1,0,0,0]]), dict(position=[0,0,float('inf')]),
                 dict(contact_evidence=''), dict(arm='both'), dict(opening=0)]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.select(self.actor, {**self.args, **change}, self.obs)
            self.assertEqual(self.calls, [])

    def test_perturbation_rejected_before_planning(self):
        self.actor._jitter_dx = .01
        with self.assertRaisesRegex(ValueError, 'unperturbed'):
            self.run_tool()
        self.assertEqual(self.calls, [])

    def test_legacy_single_move_still_plans_once(self):
        result = self.tools.execute(self.actor, 'call_tool', {'id':'robot-action-timing-tool',
                                   'arguments':dict(op='move', arm='left', position=[0,0,1])}, self.obs)
        self.assertIsNotNone(result.chunk)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn('candidate_selection', result.chunk.meta)

    def test_candidate_op_is_advertised(self):
        specs = self.tools.specs(self.actor.library)
        call = next(s['function'] for s in specs if s['function']['name']=='call_tool')
        self.assertIn('move_candidates', call['parameters']['properties']['arguments']['properties']['op']['enum'])
