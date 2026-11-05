"""A learned policy behind a WebSocket service: from images, joint readings and a sentence it predicts a chunk of native actions.

The service holds one session per GPU. A reset opens a session, each executed action is answered with one observation, and trial_end frees the
GPU. An observation can only follow an action of the current chunk, so what happens while the robot moves by other means cannot be shown to it:
to use the policy again after that, open a new session, whose history starts at the current frame.
"""
from __future__ import annotations

import time
import uuid

import numpy as np
from client_server.ws.model_client import WsModelClient
from client_server.ws.protocol.exceptions import ServerRestartedError, WsError
from XPolicyLab.utils.process_data import encode_image_bit

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = ("left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state", "right_ee_joint_state")
REQUEST_TIMEOUT_S = 300.0
CONNECT_ATTEMPTS = 3
CONNECT_RETRY_S = 5.0
CONNECT_LIMIT_S = 120.0
BUSY_POLL_S = 10.0
BUSY_LIMIT_S = 480.0


class ServiceFailure(RuntimeError):
    """The service broke or stayed full: what depends on it does not count."""


def frame(images: dict, state: dict, instruction: str) -> dict:
    """One observation as the service takes it: the three cameras, the joints and grippers, and the sentence. Nothing else leaves this process."""
    return {"vision": {camera: {"color": encode_image_bit(np.asarray(images[camera], dtype=np.uint8))} for camera in CAMERAS},
            "state": {key: np.asarray(state[key], dtype=np.float32) for key in STATE_KEYS},
            "instruction": instruction}


def _connect(url: str, request_timeout_s: float) -> WsModelClient:
    tag = uuid.uuid4().hex
    return WsModelClient(url=url, evaluation_id=tag[:16], trial_id=tag[16:], request_timeout_s=request_timeout_s, max_connect_attempts=CONNECT_ATTEMPTS,
                         connect_retry_delay_s=CONNECT_RETRY_S, max_connect_seconds=CONNECT_LIMIT_S)


def service_info(url: str) -> dict:
    """What the service says about itself: read only, takes no GPU."""
    client = _connect(url, 60.0)
    try:
        online, status = client.call("online_info"), client.call("session_status")
    finally:
        client.close()
    return {"profile": online.get("profile"), "config_sha256": online.get("config_sha256"), "worker_count": online.get("worker_count"), "max_sessions": status.get("max_sessions"),
            "active_sessions": status.get("active_sessions"), "idle_timeout_s": status.get("idle_timeout_s")}


class Session:
    """One unbroken stretch of chunks on one GPU of the service."""

    def __init__(self, url: str, busy_limit_s: float = BUSY_LIMIT_S, sleep=time.sleep):
        self.url = url
        self.busy_limit_s = busy_limit_s
        self.sleep = sleep
        self.client = None
        self.waited_s = 0.0
        self.predictions = []
        self.observations = []

    def open(self, observation: dict):
        """Take a GPU, waiting while the service is full, and show the current frame."""
        started = time.monotonic()
        while True:
            try:
                self.client = _connect(self.url, REQUEST_TIMEOUT_S)
                self.client.call("reset")
                break
            except (WsError, ConnectionError, OSError, TimeoutError) as error:
                self._drop()
                full = "capacity" in str(error).lower()
                if not full or time.monotonic() - started > self.busy_limit_s:
                    raise ServiceFailure(f"{'the service stayed full' if full else 'the service failed'}: {type(error).__name__}: {error}") from error
                self.sleep(BUSY_POLL_S)
        self.waited_s = time.monotonic() - started
        self.observe(observation)

    def chunk(self) -> list:
        """The next chunk of native actions. Every action of it has to run and be answered with observe before the next chunk."""
        begun = time.monotonic()
        try:
            actions = self.client.call("get_action")
        except (WsError, ServerRestartedError, ConnectionError, OSError, TimeoutError) as error:
            raise ServiceFailure(f"the prediction failed: {type(error).__name__}: {error}") from error
        self.predictions.append(time.monotonic() - begun)
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ServiceFailure("the prediction came back empty")
        return list(actions)

    def observe(self, observation: dict):
        """The observation after one executed action."""
        begun = time.monotonic()
        try:
            self.client.call("update_obs", observation)
        except (WsError, ServerRestartedError, ConnectionError, OSError, TimeoutError) as error:
            raise ServiceFailure(f"the observation failed: {type(error).__name__}: {error}") from error
        self.observations.append(time.monotonic() - begun)

    def close(self):
        """Free the GPU. A service that does not answer frees it by itself a little later."""
        client, self.client = self.client, None
        if client is None:
            return
        try:
            client.call("trial_end", {})
        except Exception:
            pass
        self._close_quietly(client)

    def _drop(self):
        client, self.client = self.client, None
        if client is not None:
            self._close_quietly(client)

    @staticmethod
    def _close_quietly(client):
        try:
            client.close()
        except Exception:
            pass

    def timing(self) -> dict:
        return {"wait_s": round(self.waited_s, 2), "chunks": len(self.predictions), "predict_s": round(sum(self.predictions), 2), "observations": len(self.observations),
                "observe_s": round(sum(self.observations), 2)}
