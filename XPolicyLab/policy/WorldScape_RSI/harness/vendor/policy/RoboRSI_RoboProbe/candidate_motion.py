"""Opt-in orientation selection over native plans; never executes robot actions."""
import copy
import json
from dataclasses import replace

import numpy as np

from XPolicyLab.policy.RoboDojo_Agent_L3_Inspect.policy import MotionOutcome


def _vector(value, size, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(label + " must contain finite coordinates")
    return array


def move_candidates(policy, args, observation):
    """Plan contact-admissible orientations at one world grasp point.

    The Agent supplies admissibility from current RGB; this is not a contact or
    scene-collision test. Native motion construction owns planning, limits and
    resampling. Return its selected action sequence without replanning it.
    """
    from .tools import tcp_targets

    required = {"op", "arm", "position", "quaternions_wxyz", "contact_evidence"}
    if set(args) != required or args["op"] != "move_candidates":
        raise ValueError("move_candidates requires exactly " + ", ".join(sorted(required)))
    arm = args["arm"]
    if arm not in ("left", "right"):
        raise ValueError("Choose one arm")
    position = _vector(args["position"], 3, "World grasp point")
    evidence = args["contact_evidence"]
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 1600:
        raise ValueError("Describe why every candidate is admissible in the current RGB")
    values = args["quaternions_wxyz"]
    if not isinstance(values, list) or not 2 <= len(values) <= 8:
        raise ValueError("Provide 2-8 explicitly contact-admissible orientations")
    quaternions = []
    for value in values:
        q = _vector(value, 4, "Flange WXYZ quaternion")
        if not np.isclose(np.linalg.norm(q), 1., atol=1e-5, rtol=0.):
            raise ValueError("Candidate quaternions must be unit WXYZ")
        quaternions.append(q / np.linalg.norm(q))
    initial_q = _vector(observation.state[arm + "_ee_pose"][3:], 4, "Current orientation")
    norm = np.linalg.norm(initial_q)
    if norm < 1e-8:
        raise ValueError("Current orientation is invalid")
    initial_q = initial_q / norm
    current = np.asarray(observation.state[arm + "_arm_joint_state"], dtype=float)
    if current.ndim != 1 or not current.size or not np.isfinite(current).all():
        raise ValueError("Current joints are invalid")
    # Perturbed targets would invalidate the shared-contact-point contract.
    if any(getattr(policy, name, 0.) != 0. for name in
           ("_jitter_dx", "_jitter_dy", "_jitter_dz", "_jitter_sphere_r", "_jitter_sphere_mean_r")):
        raise ValueError("Candidate comparison requires unperturbed native targets")

    original_targets = copy.deepcopy(policy._last_targets)
    original_grippers = copy.deepcopy(policy._gripper_goals)
    candidates, winner, winning_key, winning_targets = [], None, None, None
    try:
        for index, quaternion in enumerate(quaternions):
            policy._last_targets = copy.deepcopy(original_targets)
            policy._gripper_goals = copy.deepcopy(original_grippers)
            outcome = policy.native_move(tcp_targets(arm, position, quaternion), observation)
            row = {"index": index, "quaternion_wxyz": quaternion.tolist()}
            if outcome.chunk is None:
                row.update(status="rejected", feedback=outcome.tool_result)
                candidates.append(row)
                continue
            # Score the actual native command sequence, including its starting
            # state. Do NOT wrap joint angles: raw multi-turn travel is physical.
            path = np.asarray([current, *[action.data[arm + "_arm_joint_state"]
                                          for action in outcome.chunk.actions]], dtype=float)
            if path.ndim != 2 or path.shape[1] != current.size or not np.isfinite(path).all():
                raise RuntimeError("Native candidate returned invalid joint commands")
            excursion = float(np.max(np.abs(path - current)))
            travel = float(np.abs(np.diff(path, axis=0)).sum())
            angle = float(2 * np.arccos(np.clip(abs(np.dot(initial_q, quaternion)), 0., 1.)))
            key = (excursion, travel, angle, len(outcome.chunk.actions), index)
            row.update(status="planned", max_joint_excursion_rad=excursion,
                       total_joint_travel_rad=travel, flange_rotation_rad=angle,
                       planned_actions=len(outcome.chunk.actions))
            candidates.append(row)
            if winning_key is None or key < winning_key:
                winner, winning_key = outcome, key
                winning_targets = copy.deepcopy(policy._last_targets)
    finally:
        # Rejected/unselected proposals must not overwrite arrival tracking or
        # grip intent, including when a later planner raises an infrastructure error.
        policy._last_targets = original_targets
        policy._gripper_goals = original_grippers

    report = {"observation_step": observation.step, "arm": arm,
              "grasp_position_m": position.tolist(), "contact_evidence": evidence,
              "ranking": ["max_joint_excursion_rad", "total_joint_travel_rad",
                          "flange_rotation_rad", "planned_actions", "index"],
              "candidates": candidates, "selected_index": winning_key[-1] if winner else None,
              "scope": "native plan selection; contact, scene clearance and arrival unverified"}
    if winner is None:
        return MotionOutcome(chunk=None, tool_result=json.dumps(report), repairable=False)
    policy._last_targets = winning_targets
    chunk = replace(winner.chunk, meta={**winner.chunk.meta, "candidate_selection": report})
    return replace(winner, chunk=chunk,
                   tool_result=winner.tool_result + "\nCandidate selection: " + json.dumps(report))
