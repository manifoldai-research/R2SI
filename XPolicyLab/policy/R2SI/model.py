"""The XPolicyLab adapter: one agent per episode behind the evaluator's update_obs / get_action protocol."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from . import cameras, kinematics, monitor, planning, robot
from .agent import Agent
from .bridge import Bridge
from .client import ModelClient
from .control import Controller
from .segmenter import Segmenter
from .tools import Tools


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        if model_cfg.get("eval_batch"):
            raise ValueError("R2SI supports one environment per policy process; set eval_batch: false")
        # Only these keys of the deployment configuration enter the policy; whatever else the launcher passes does not.
        self.cfg = {key: model_cfg.get(key) for key in ("bench_name", "env_cfg_type", "action_type", "model", "effort", "request_timeout_s", "keepalive_s", "segmenter_checkpoint", "policy_service")}
        for key, expected in (("bench_name", "RoboDojo"), ("env_cfg_type", "arx_x5"), ("action_type", "joint")):
            if self.cfg.get(key) != expected:
                raise ValueError("unsupported deployment field: " + key)
        dimensions = get_robot_action_dim_info(self.cfg["env_cfg_type"])
        if list(dimensions["arm_dim"]) != [6, 6] or list(dimensions["ee_dim"]) != [1, 1]:
            raise ValueError("R2SI requires the arx_x5 two-arm, one-gripper-channel configuration")
        self.cfg["policy_service"] = os.environ.get("R2SI_POLICY_SERVICE") or self.cfg["policy_service"]
        checkpoint = Path(os.environ.get("R2SI_SEGMENTER_CHECKPOINT") or self.cfg["segmenter_checkpoint"]).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = robot.WORKSPACE / checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError("Set R2SI_SEGMENTER_CHECKPOINT to the downloaded sam3.pt file: " + str(checkpoint))
        self.cfg["segmenter_checkpoint"] = str(checkpoint)
        self.spec = robot.load_spec()
        self.arm_model = kinematics.ArmModel()
        self.shape = kinematics.RobotShape(self.arm_model)
        self.rig = cameras.Rig()
        self.planner = planning.Curobo()
        self.segmenter = Segmenter(self.cfg["segmenter_checkpoint"])
        self.trace_dir = os.environ.get("R2SI_TRACE_DIR") or None
        self._lock = threading.RLock()
        self._episode = None

    # -- episode life cycle --------------------------------------------------------------------------
    def reset(self):
        with self._lock:
            self._end()
            self.segmenter.forget()
            bridge = Bridge(monitor.Monitor(self.spec), float(self.cfg["keepalive_s"]))
            controller = Controller(self.spec, self.arm_model, self.planner, bridge, self.rig.head_stand())
            tools = Tools(controller, self.rig, self.shape, self.segmenter, self.cfg["policy_service"])
            client = ModelClient(model=self.cfg["model"], effort=self.cfg["effort"], timeout_s=float(self.cfg["request_timeout_s"]), trace_dir=self.trace_dir)
            agent = Agent(client, tools, bridge)
            thread = threading.Thread(target=agent.run, name="agent", daemon=True)
            self._episode = {"bridge": bridge, "controller": controller, "tools": tools, "client": client, "agent": agent, "thread": thread}
            thread.start()
            return {"ok": True}

    def update_obs(self, obs):
        self._episode["bridge"].update_obs(obs)
        self._write_metrics()

    def get_action(self):
        action = self._episode["bridge"].get_action()
        return [{key: value.copy() for key, value in action.items()}]

    def update_obs_batch(self, *args, **kwargs):
        raise NotImplementedError("one policy process per environment")

    def get_action_batch(self, *args, **kwargs):
        raise NotImplementedError("one policy process per environment")

    def on_trial_end(self, result):
        self._end()
        return {"ok": True}

    def close(self):
        self._end()

    def _end(self):
        episode, self._episode = self._episode, None
        if episode is None:
            return
        episode["bridge"].close()
        episode["client"].ended.set()
        episode["thread"].join(timeout=5.0)
        self._write_metrics(episode)
        episode["client"].close()

    # -- diagnostics ---------------------------------------------------------------------------------
    def _write_metrics(self, episode=None):
        episode = episode or self._episode
        if episode is None or self.trace_dir is None:
            return
        agent, bridge = episode["agent"], episode["bridge"]
        observation = bridge.observation
        if observation is None or (episode is self._episode and observation.step % 25):
            return
        row = {"steps": observation.step, "step_lim": observation.step_limit, "decisions": agent.decisions, "cells": agent.cells, "keepalive_holds": bridge.keepalive_holds,
               "interruptions": bridge.interruptions, "contact_stops": bridge.contact_stops, **episode["controller"].counters, **episode["tools"].counters}
        path = Path(self.trace_dir) / ("metrics-" + episode["client"].session_id + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row), encoding="utf8")
        temporary.replace(path)
