"""The decision loop of one episode: the model writes code cells, and each cell's outcome and fresh images go back to it.

It runs on its own thread for the whole episode. The evaluator's requests never wait on it longer than the keepalive time (bridge.py). The
loop has no notion of being finished: only the evaluator ends an episode.
"""
from __future__ import annotations

import json
import traceback

from . import observe, prompt, sandbox
from .bridge import CAMERAS, EpisodeOver
from .client import InfrastructureFailure
from .robo import Robo
from .icl import Reference
from .tools import Failed

CELL_TIME_LIMIT_S = 60.0
CALL_SUMMARY_LIMIT = 30
IMAGE_MESSAGES_KEPT = 8    # besides the first views, this many of the latest messages keep their images, so that requests stay small in a long episode ...
IMAGE_MESSAGES_DROPPED = 8 # ... and older ones lose them this many at a time, so that most requests start as the last one did and the service can reuse its work; each time the conversation goes on under a new cache key
IMAGE_GONE = "(this image is no longer shown)"
PYTHON_TOOL = {"type": "function", "name": "python",
               "description": "Run one code cell in this episode's persistent Python namespace. Variables and functions stay for later cells. Returns what the cell "
                              "printed, the value of its last expression, an error if it stopped, and what its robo calls did.",
               "parameters": {"type": "object", "properties": {"checklist": {"type": "string", "description": "Where the job stands as you start this cell: each item of your "
                                                                             "checklist, whether a fresh measurement has confirmed it, and the steps it may still use."},
                                                               "code": {"type": "string", "description": "Python source of one code cell."}},
                              "required": ["checklist", "code"], "additionalProperties": False}}


class Agent:
    def __init__(self, client, tools, bridge, retriever=None):
        self.client = client
        self.tools = tools
        self.bridge = bridge
        self.retriever = retriever
        self.reference = Reference()
        tools.reference_info = lambda: self.reference.info()
        self.items = []
        self.images_dropped = 0    # how many image messages after the first have lost their images
        self.decisions = 0
        self.cells = 0
        self.session = sandbox.Session(time_limit_s=CELL_TIME_LIMIT_S, robo_class=Robo)
        tools.ask_fresh = self._ask_fresh

    # -- messages ---------------------------------------------------------------------------------
    @staticmethod
    def _content(parts) -> list:
        content = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "input_text", "text": part["text"]})
            else:
                content.append({"type": "input_image", "image_url": part["image_url"]["url"]})
        return content

    def _user(self, parts):
        self.items.append({"role": "user", "content": self._content(parts)})

    def _conversation(self) -> list:
        """The conversation as sent: everything, except the images of messages neither among the first nor the latest."""
        rows = [index for index, item in enumerate(self.items) if item.get("role") == "user" and any(part.get("type") == "input_image" for part in item["content"])]
        if len(rows) - 1 - self.images_dropped > IMAGE_MESSAGES_KEPT + IMAGE_MESSAGES_DROPPED:
            self.images_dropped += IMAGE_MESSAGES_DROPPED
            self.client.new_cache_key()
        gone = set(rows[1:1 + self.images_dropped])
        return self.reference.messages() + [{**item, "content": [{"type": "input_text", "text": IMAGE_GONE} if part.get("type") == "input_image" else part for part in item["content"]]} if index in gone else item
                for index, item in enumerate(self.items)]

    def _ask_fresh(self, observation, question) -> str:
        text = "Instruction: " + observation.instruction + ("" if question is None else "\n\nQuestion: " + question)
        return self.client.ask_fresh(prompt.SECOND_LOOK, self._content([{"type": "text", "text": text}, *self._camera_parts(observation)]))

    def _camera_parts(self, observation):
        parts = []
        for camera in CAMERAS:
            if camera in observation.images:
                parts += observe.jpeg_part(observation.images[camera], f"{camera} at step {observation.step}:")
        return parts

    def _status(self, observation) -> dict:
        tools = self.tools
        row = {"arms": {arm: {key: value for key, value in tools.arm_state(observation, arm).items() if key != "joints"} for arm in ("left", "right")},
               "steps_used": observation.step, "step_limit": observation.step_limit, "steps_left": observation.remaining, "home_steps": tools.home_steps(observation)}
        holds = self.bridge.keepalive_unreported
        if holds:
            row["hold_steps_while_deciding"] = holds
            self.bridge.keepalive_unreported = 0
        return row

    # -- cells ------------------------------------------------------------------------------------
    def _run_cell(self, code: str) -> dict:
        """Run one cell to its end, carrying out its robo requests on this thread."""
        self.cells += 1
        tools = self.tools
        tools.calls = []
        started_step = self.bridge.current().step
        try:
            cell = self.session.compile(code)
        except (sandbox.SandboxViolation, SyntaxError) as error:
            return {"status": "rejected", "error": {"type": type(error).__name__, "message": str(error)}, "steps": 0}
        event = cell.begin()
        while event.kind != "done":
            try:
                value = tools.dispatch(event.method, dict(event.arguments))
                reply = sandbox.Reply.ok(value)
            except Failed as failed:
                reply = sandbox.Reply.failure(str(failed), failed.result)
            except Exception as error:   # misuse, or a defect here: the cell learns of it, the episode goes on
                if not isinstance(error, (ValueError, KeyError, TypeError)):
                    self.client._log("robo_call_failed", method=event.method, trace=traceback.format_exc()[-2000:])
                reply = sandbox.Reply.error(f"{type(error).__name__}: {error}")
            event = cell.resume(reply)
        result = {key: value for key, value in event.summary.items() if value not in (None, "", False, [])}
        result.pop("namespace", None)
        calls = tools.calls
        if calls:
            result["robo_calls"] = calls[:CALL_SUMMARY_LIMIT]
            if len(calls) > CALL_SUMMARY_LIMIT:
                result["robo_calls_not_listed"] = len(calls) - CALL_SUMMARY_LIMIT
        result["steps"] = self.bridge.current().step - started_step
        return result

    # -- the loop ---------------------------------------------------------------------------------
    def run(self):
        """Decide until the episode is over."""
        try:
            self._loop()
        except EpisodeOver:
            pass
        except InfrastructureFailure as error:
            if not self.client.ended.is_set():
                self.bridge.fail(error)
        except BaseException as error:
            self.client._log("agent_failed", error_type=type(error).__name__, trace=traceback.format_exc()[-2000:])
            self.bridge.fail(error)
        finally:
            self.tools.end_policy()
            self.session.close()

    def _loop(self):
        observation = self.bridge.wait_observation()
        self.tools.ctl.begin(observation)
        if self.retriever is not None:
            self.reference = self.retriever.prepare(observation, self.client.ended)
            self.client._log("icl_reference_ready", **self.reference.info())
            observation = self.bridge.current()
        instructions = prompt.system_prompt()
        self._user([{"type": "text", "text": "Instruction: " + observation.instruction + "\n\nRobot now:\n" + json.dumps(self._status(observation))},
                    *self._camera_parts(observation)])
        while True:
            raw = self.client.complete(instructions, self._conversation(), [PYTHON_TOOL])
            self.decisions += 1
            calls = [item for item in raw.get("output") or [] if isinstance(item, dict) and item.get("type") == "function_call"]
            for item in raw.get("output") or []:
                if isinstance(item, dict) and item.get("type") == "reasoning":
                    self.items.append({key: value for key, value in item.items() if key != "status"})
                if isinstance(item, dict) and item.get("type") == "message":
                    text = " ".join(c.get("text", "") for c in item.get("content", []) if isinstance(c, dict))
                    if text:
                        self.items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
            if not calls:
                self._user([{"type": "text", "text": "Reply with one python call."}])
                continue
            for call in calls:
                self.items.append({"type": "function_call", "call_id": call.get("call_id"), "name": call.get("name"), "arguments": call.get("arguments")})
                result = self._answer(call)
                self.items.append({"type": "function_call_output", "call_id": call.get("call_id"), "output": json.dumps(result, ensure_ascii=False, default=str)})
                self._after_cell(result)

    def _answer(self, call) -> dict:
        if call.get("name") != "python":
            return {"status": "rejected", "error": {"type": "UnknownTool", "message": "the only tool is python"}, "steps": 0}
        try:
            code = json.loads(call.get("arguments") or "{}").get("code")
        except (ValueError, AttributeError):
            code = None
        if not isinstance(code, str) or not code.strip():
            return {"status": "rejected", "error": {"type": "ValueError", "message": "python takes a checklist and the code as a string"}, "steps": 0}
        return self._run_cell(code)

    def _after_cell(self, result):
        """What the model sees next: images it asked for, and fresh views when the robot has moved."""
        observation = self.bridge.current()
        parts = list(self.tools.attachments)
        self.tools.attachments = []
        if result.get("steps"):
            parts = [{"type": "text", "text": "Robot now:\n" + json.dumps(self._status(observation))}, *self._camera_parts(observation), *parts]
        if parts:
            self._user(parts)
