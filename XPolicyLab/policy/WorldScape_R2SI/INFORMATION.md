# Information sources

Everything the running policy reads, where it comes from and what it is used for. The frozen Harness filters observations through `harness.contracts.observation`, then `bridge.parse_observation` constructs the Agent's view; neither forwards task budgets. The environment variables `R2SI_API_BASE`, `R2SI_API_KEY`, `R2SI_SESSION_RELEASE_PATH` and `R2SI_TRACE_DIR` name the model service, an optional call that frees its session after an episode, and where a trace goes; they do not enter any decision. `on_trial_end` receives the trial's result and does not read it.

## Observations from the evaluator

| Input | Source | Used for |
| --- | --- | --- |
| RGB of cam_head, cam_left_wrist, cam_right_wrist | Official observation `vision.<camera>.color` | Shown to the model; segmentation and measurement |
| Arm joint state, end-effector pose | Official observation `state` | Grasp point, planning, tracking feedback |
| Gripper reading | Official observation `state.<arm>_ee_joint_state` | Hold check and width, which follow from the published gripper rule |
| Instruction | Official observation `instruction` | Shown to the model |
| Local action count | `bridge.py` counts a pending action when its following observation arrives, including keepalive holds | Tracking this episode's action history and motion cost; repeated observations without a pending action do not increment it |
| Results of the agent's own actions | Joint state after each action | Arrival, blocked, interrupted |

Not read by the decision loop: the evaluator's action counter, task step limit, remaining allowance, depth, object poses, object models, simulator state, contact sensors, success flags, reward or score values, task name, layout files or indices, scene files, seed, anything from an earlier live episode. The Harness uses `prepare_case` metadata only for trial bookkeeping and excludes it from runtime configuration and observations. Deployment inputs are restricted to an explicit allowlist and the ICL settings in `configured_retriever`.

## Shared generic experience and WAM

The versioned generic Skill library and `harness/memory/*.md` are additional policy inputs. The Agent reads Skill pages through `robo.read_skill`; shared Memory enters the system prompt. The outer RSI loop updates these assets and the listed policy-level Agent/tool source in an isolated candidate, then validates it before activation. Those revisions are pinned with the candidate release, rather than changing during an episode.

The bundled Flash/Memory path uses `harness/configs/wam_skills.json` and its `WRSI_*` source/checkpoint bindings (or `WRSI_POLICY_CLIENT_CONFIG` for the existing remote lease service). Profile selection is explicit in `robo.policy`; no task-to-profile routing is added. WSP training is a reserved interface and cannot mutate model weights in this release.

## External demonstration reference

When `R2SI_ICL_CONFIG` or `icl_config` is set, ICL Engine reads an explicitly configured external demonstration catalog and selected source episode. This is an additional policy input, separate from evaluator observations. Retrieval uses only the public instruction and first available RGB head-camera view (falling back to another camera); no case metadata or trial outcome is supplied. The configured backend may send that query to its HTTP endpoint. Source captions, sampled images and recipe-specific source annotations enter the planning model as a frozen reference, not executable actions. The corpus and its task overlap must be disclosed in evaluation. See [ICL.md](ICL.md) for setup, source formats and the boundary between demonstrations and current-scene execution.

`robo.reference()` exposes the selection identity, retrieval method/status, recipe and integrity hashes. The entire compiled prefix is retained on every planning request, independently of live-image pruning. Its selected reference is discarded with the episode; verified on-disk packages may be reused. `R2SI_ICL_PYTHON` / `icl_python` select the external worker environment. `icl_timeout_s`, `icl_max_images` and `icl_max_text_chars` bound retrieval and context size. No config means an explicitly disabled tool; a configured retrieval failure aborts the episode.

## Public robot and camera files

The root of these public files is configured by `ROBODOJO_ROOT`, falling back to the parent workspace of XPolicyLab. `R2SI_SEGMENTER_CHECKPOINT` and `R2SI_POLICY_SERVICE` may override the corresponding deployment fields. `R2SI_POLICY_SERVICES` (`{"flash": address, "memory": address}`) gives each WAM profile its own service, so `robo.policy(profile=...)` can reach both. These variables locate assets and services only.

| Constant | File | Used for |
| --- | --- | --- |
| Kinematic chain, joint limits | `Assets/Robots/x5/X5A.urdf` | Forward kinematics, straight-line IK; no command goes past the limits |
| Link meshes | `Assets/Robots/x5/meshes/*.STL` | Robot outline in images |
| Collision spheres, self-collision settings, joint-space settings, acceleration limit, frame bias | `Assets/Robots/x5/curobo.yml` | The robot-only planner; the spheres also give arm clearance |
| Head camera pose | `env_cfg/camera/camera_config.yml` | Measurement; where its stand may be |
| Intrinsics | `env_cfg/camera/template.py` | Measurement |
| Wrist camera mount | `Assets/Robots/x5/robot_config.yml`, URDF | Measurement; `robo.look` |

## Constants in the code

| Constant | Where | Basis |
| --- | --- | --- |
| Table height 0.765 | `frames.TABLE_Z` | The surface the arms are mounted on. It is the default support and the lowest descent; the tool notes say to check the surface you work on |
| Where the head camera's stand may be | `cameras.Rig.head_stand`, `cameras.STAND_BEHIND_M`, `cameras.STAND_HALF_WIDTH_M` | What carries the camera stands below it and behind it, out of its view. The box follows from the camera's public pose and a control clearance. Motions do not run into it |
| Arm base origins | `frames.BASE_ORIGIN` | Rig mounting, also follows from joints and poses |
| Joint range the planner works in | `robot.WORK_LOW`, `robot.WORK_HIGH` | The description allows +-10 rad on five joints, but the links meet long before that. Joint 1: about half a turn either way, so that the way back never goes around. Joint 2: lower end inside where the upper arm meets the base, measured at about -0.23. Joint 3: from folded to straight. Joints 4 and 5: a little inside their stops, measured at about -2.22 and 1.78, and 1.58 either way. Joint 6: the description's limits. Measured by driving the arm against its stops in the simulator |
| How far the wrist folds back | `robot.WRIST_FOLD` | The same measurement: joint 4's stop as joint 5 turns, a little inside |
| Reach figures in the tool notes; `frames.ARM_REACH_M`, `frames.ARM_TOP_M` | `knowledge/tool-notes.md`, `frames.py` | Forward kinematics of the robot description inside the joint range above; they hold in any direction around the arm's base |
| Grasp point offset, pad depth, fingertip reach, jaw span | `frames.py` | Robot description: the fingertip ends 0.1576 m ahead of the flange (`joint7` origin in the URDF plus the extent of `meshes/link7.STL`) |
| Finger joint origin and travel, spheres along the fingers | `kinematics.py` | Values of the URDF and its meshes written as constants; the spheres are a conservative model of the fingers |
| Jaw width from reading and command | `control.py` | `gripper_scale` and the gripper rule in the robot configuration |
| Joint steps per native step, gripper step | `robot.py` | Control constants, inside the drive limit of 0.2 rad per step in `env/robot_manager/robot_config/x5.py` and the gripper rule in `env/robot_manager/control_manager.py` |
| Model, reasoning effort, request limit, hold time, SAM 3 weights | `deploy.yml` | Run configuration; the same for every job |
| Stage distances, retries, thresholds | `actions.py`, `agent.py`, `cameras.py`, `client.py`, `control.py`, `kinematics.py`, `monitor.py`, `objects.py`, `observe.py`, `planning.py`, `sandbox.py`, `scene.py`, `segmenter.py`, `tools.py`, `wam.py` | Control algorithm parameters |
| Bounds of a measurement | `scene.MAX_SIDE_M`, `scene.MIN_HEIGHT_M`, `objects.MAX_HEIGHT_M`, `objects.LIFTED_MAX_M`, `objects.WORK_RADIUS_M` follow from the arm's reach and the pads; `objects.MAX_MASK_SHARE` is a share of the image; `tools.RAY_HEIGHTS_M` are sampling heights along a viewing ray | What the arms could work on at all |
| Shape heuristics | `tools.GRASP_SHARE`, `tools.GRASP_DEPTH_MIN_M`, `tools.GRASP_DEPTH_MAX_M`, `tools.SIDE_GRASP_SHARE`, `tools.ROUND_ASPECT`, `objects.ELONGATED_ASPECT`, `actions.REFINE_MIN_ASPECT`, `tools.LOOK_NEAR_M`, `tools.PLACED_NEAR_M`, `objects.SAME_OBJECT_M`, `objects.SAME_OBJECT_HEIGHT_M`, `scene.LIKE_SIZE`, `scene.AROUND_M` | Control algorithm: how a measured shape turns into a grasp or a match. Applied to whatever is measured |

## Other model inputs

| Model | Input | Weights and code | Time |
| --- | --- | --- | --- |
| SAM 3 (weights `facebook/sam3`, code `RLinf/sam3` at the commit in `requirements.txt`) | One RGB image from above, a phrase or a pixel chosen by the agent | Public | About 15 ms per image and 60 ms per phrase on the policy GPU; counts toward the evaluator's request limit |
| The planning model (`deploy.yml`) | System prompt (constant text), the observations and tool results above, and the configured external ICL prefix. Live images retain only the first views and latest messages; all compiled demonstration images remain | Hosted service | Counts toward the evaluator's request limit; the keepalive holds cost native steps |
| The same model, one request outside the conversation (`robo.second_look`) | Constant text (`prompt.SECOND_LOOK`), the instruction, the three current camera images, and a question the planning model wrote | Hosted service | As above, one short request each time the planning model asks |
| The learned policy (`robo.policy`, service address `policy_service` in `deploy.yml`) | At each native step of a call: the three RGB images, both arms' joint state and gripper reading, and one sentence, the instruction or one the planning model wrote (`wam.frame`). No action count, step limit, pose or anything else; session identifiers are random. Without an address a call fails with an error. The service's configuration hash is read once for the run's metrics and takes no part in decisions | Public: weights `manifoldai-research/WorldScape-R2SI`, post-trained on RoboDojo demonstrations by their authors, and the service code in `XPolicyLab/policy/WorldScape_R2SI/robodojo_server` | About 1.3 s per chunk of 24 steps and 30 ms per step over the network; counts toward the time limit |

## What the planning model learns from the learned policy

Only what its call returns: how many chunks ran, where the arms ended, how many steps each arm did not follow its commands (from the official joint state), why it stopped early if it did, and the usual step counts. Whether to call it is the planning model's decision in each episode, from what it sees and what it knows how to do. The handbook states the policy's general abilities and limits; no task, or kind of task, is tied to it anywhere.

## Less common capabilities

`robo.register` lays any two measured planar shapes on each other. `robo.pick(velocity=)` meets anything that moves steadily. `robo.show(outline=)` draws a shape on an image and `robo.show(rotate_deg=)` turns an image. `robo.pick(check_hold=False)` turns off the reading check where a hold does not show in the reading.

## General scoring rules in the Memory rules

The section "How work is scored" states how the benchmark usually judges work: every native step is read; everything asked must be true at one step; orientation counts; inside means inside and seated, and parts that fit together fit within millimetres, as parts do; a stated order is part of the job; what was not asked to move stays; work counts per finished item or group and is read only with both grippers fully open, so leave a few steps at the end with them open; the job ends with the arms back at their initial poses and returning counts once everything is in place. Where an instruction states an amount the handbook says to go clearly past it on the side it allows, and to raise high what it only says to pick or lift: margins against measuring error and slip, not a judge's value. The handbook gives no thresholds, and no value is read at run time. Its other sections are working practice.

## Measurements kept during an episode

`scene.py` keeps one box per object from the agent's own `robo.locate` results (RGB masks and public camera geometry) and from where its own hand put a load down. Free motions use them to pass over objects. Live scene measurements and conversation state are not kept between episodes. The external demonstration package cache described above is separate.

## How an episode ends

The evaluator ends an episode when its own check succeeds, when it finds a hard failure, or when the step limit is reached, as in the official flow. The Agent does not know that limit or the ending reason. Closing the episode interrupts pending native motions, WAM calls and waits. A terminal action without a following observation is not added to the local acknowledged-action count. The handbook tells the Agent that being still able to act tells it nothing it can use. The Agent has no call that ends the work.
