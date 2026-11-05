"""The model client: one sticky session per episode over the OpenAI Responses API.

A request is synchronous and carries the whole conversation. What was sent under a cache key may not change later, so a conversation that
loses images goes on under a new key, in the same session. A dropped connection or a failing gateway repeats a request: nothing was done
with the lost reply. A refusal or a request out of time ends the trial as an infrastructure failure. The trace holds text and hashes only,
no images, tokens or encrypted reasoning.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


RETRY_WAITS_S = (5, 15, 30, 60, 120)
GATEWAY_FAILURES = (502, 503, 504)


class InfrastructureFailure(BaseException):
    """The model service failed; the trial is void. Not an Exception, so a code cell can never catch it."""


def _worth_repeating(error) -> bool:
    """A lost connection or a failing gateway, not a refusal and not a request that ran out of time."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in GATEWAY_FAILURES
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    return isinstance(reason, (OSError, http.client.HTTPException)) and not isinstance(reason, TimeoutError)


class ModelClient:
    def __init__(self, *, model: str, effort: str, timeout_s: float, trace_dir: str | None = None):
        self.model, self.effort, self.timeout_s = model, effort, float(timeout_s)
        self.base = os.environ.get("R2SI_API_BASE", "").rstrip("/")
        self.token = os.environ.get("R2SI_API_KEY", "")
        if not self.base.startswith(("https://", "http://")) or not self.token:
            raise ValueError("set R2SI_API_BASE and R2SI_API_KEY")
        self.release_path = os.environ.get("R2SI_SESSION_RELEASE_PATH", "")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.session_id = uuid.uuid4().hex
        self.cache_key = self.session_id
        self.ended = threading.Event()
        self.turn = 0
        self.requested = False
        self.closed = False
        self.trace_dir = Path(trace_dir) if trace_dir else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def new_cache_key(self):
        """Go on under a new cache key: the service holds a key to what was sent under it, and the next request will not repeat that."""
        self.cache_key = uuid.uuid4().hex

    # -- one decision -------------------------------------------------------------------------------
    def complete(self, instructions: str, items: list, tools: list) -> dict:
        """Send the conversation; returns the Responses reply."""
        self.turn += 1
        body = {"model": self.model, "reasoning": {"effort": self.effort}, "input": items, "instructions": instructions, "tools": tools,
                "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "function", "name": tool["name"]} for tool in tools]},
                "parallel_tool_calls": False, "store": False, "include": ["reasoning.encrypted_content"], "prompt_cache_key": self.cache_key, "stream": False}
        request = self._request(body)
        self.requested = True
        started = time.time()
        self._log("request_started", request_bytes=len(request.data), turn=self.turn)
        self._trace("request", turn=self.turn, body=body)
        box = self._send(request)
        elapsed = round(time.time() - started, 3)
        error = box.get("error")
        if error is not None:
            details = {}
            if isinstance(error, urllib.error.HTTPError):
                data = error.read(8192)
                try:
                    item = json.loads(data)
                    item = item.get("error", item) if isinstance(item, dict) else {}
                    details = {k: str(item[k]).replace(self.token, "[redacted]")[:1000] for k in ("code", "type", "message")
                               if isinstance(item, dict) and k in item and "data:image" not in str(item[k])}
                except (ValueError, TypeError):
                    details = {}
                details["http_status"] = error.code
            self._log("request_failed", error_type=type(error).__name__, elapsed_s=elapsed, **details)
            raise InfrastructureFailure("model request failed: " + type(error).__name__) from None
        raw = box["raw"]
        self._trace("response", turn=self.turn, status=raw.get("status"), output=raw.get("output"), usage=raw.get("usage"), elapsed_s=elapsed)
        self._log("request_completed", turn=self.turn, elapsed_s=elapsed, usage=raw.get("usage"))
        if raw.get("status") != "completed":
            self._log("request_incomplete", status=raw.get("status"), elapsed_s=elapsed)
            raise InfrastructureFailure("model inference did not complete")
        return raw

    def ask_fresh(self, instructions: str, content: list) -> str:
        """One question outside this episode's conversation: the request carries nothing but what is given here. Its failure costs only the answer."""
        body = {"model": self.model, "reasoning": {"effort": self.effort}, "input": [{"role": "user", "content": content}], "instructions": instructions,
                "store": False, "prompt_cache_key": uuid.uuid4().hex, "stream": False}
        started = time.time()
        box = self._send(self._request(body))
        raw = box.get("raw") or {}
        self._trace("second_look", turn=self.turn, status=raw.get("status"), output=raw.get("output"), usage=raw.get("usage"), elapsed_s=round(time.time() - started, 3))
        parts = [part for item in raw.get("output") or [] if isinstance(item, dict) and item.get("type") == "message" for part in item.get("content") or [] if isinstance(part, dict)]
        return " ".join(part.get("text", "") for part in parts).strip() or "No answer this time: the service did not reply."

    def _request(self, body: dict):
        return urllib.request.Request(self.base + "/responses", data=json.dumps(body).encode(), headers=self._headers())

    def _headers(self) -> dict:
        return {"Authorization": "Bearer " + self.token, "Content-Type": "application/json", "session_id": self.session_id, "conversation_id": self.session_id}

    def _send(self, request) -> dict:
        """The request, repeated while the failure is worth repeating: {"raw": reply} or {"error": what went wrong}."""
        box = self._send_once(request)
        for wait_s in RETRY_WAITS_S:
            if "error" not in box or not _worth_repeating(box["error"]):
                break
            self._log("request_repeated", error_type=type(box["error"]).__name__, wait_s=wait_s)
            if self.ended.wait(wait_s):
                raise InfrastructureFailure("the trial ended while the model request was pending")
            box = self._send_once(request)
        return box

    def _send_once(self, request) -> dict:
        """One try on its own thread, so that the end of the trial can stop the wait."""
        box, done = {}, threading.Event()

        def send():
            try:
                with self.opener.open(request, timeout=self.timeout_s) as response:
                    box["raw"] = json.load(response)
            except BaseException as error:
                box["error"] = error
            finally:
                done.set()

        threading.Thread(target=send, name="model-request", daemon=True).start()
        while not done.wait(0.5):
            if self.ended.is_set():
                raise InfrastructureFailure("the trial ended while the model request was pending")
        return box

    def _release(self):
        if not self.release_path:
            return
        request = urllib.request.Request(self.base + self.release_path.format(session_id=self.session_id), data=b"{}", headers=self._headers())
        try:
            with self.opener.open(request, timeout=15) as response:
                receipt = json.load(response)
            self._log("session_released", release_confirmed=receipt.get("released") is True)
        except Exception as error:
            self._log("release_unconfirmed", error_type=type(error).__name__, http_status=getattr(error, "code", None))

    # -- diagnostics --------------------------------------------------------------------------------
    def _log(self, phase, **details):
        if self.trace_dir is None:
            return
        row = dict(phase=phase, time=time.time(), session_id=self.session_id, **details)
        with (self.trace_dir / (self.session_id + ".jsonl")).open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row) + "\n")

    def _trace(self, kind, body=None, output=None, **details):
        if self.trace_dir is None:
            return
        row = dict(kind=kind, time=time.time(), session_id=self.session_id, **details)
        if body is not None:
            items = []
            for item in body.get("input", [])[-6:]:
                entry = {k: item.get(k) for k in ("type", "role", "name", "call_id") if item.get(k)}
                if item.get("type") == "function_call":
                    entry["arguments"] = item.get("arguments")
                if item.get("type") == "function_call_output":
                    entry["output"] = str(item.get("output"))[:6000]
                content = item.get("content")
                if isinstance(content, list):
                    entry["content"] = [{"type": c.get("type"), "text": c.get("text")} if c.get("type") in ("input_text", "output_text")
                                        else {"type": c.get("type"), "sha256": hashlib.sha256(str(c.get("image_url", "")).encode()).hexdigest()[:16]}
                                        for c in content if isinstance(c, dict)]
                items.append(entry)
            row.update(input_items=len(body.get("input", [])), tail=items, instructions_sha256=hashlib.sha256(str(body.get("instructions", "")).encode()).hexdigest()[:16])
        if output is not None:
            row["output"] = [{k: o.get(k) for k in ("type", "name", "arguments", "call_id", "status") if o.get(k) is not None}
                             | ({"text": " ".join(c.get("text", "") for c in o.get("content", []) if isinstance(c, dict))[:4000]} if o.get("type") == "message" else {})
                             | ({"summary": [s.get("text", "")[:2000] for s in o.get("summary", []) if isinstance(s, dict)]} if o.get("type") == "reasoning" else {})
                             for o in output if isinstance(o, dict)]
        with (self.trace_dir / ("trace-" + self.session_id + ".jsonl")).open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, default=str).replace(self.token, "[redacted]") + "\n")

    def close(self):
        """Stop waiting and release this episode's session on the provider."""
        if self.closed:
            return
        self.closed = True
        self.ended.set()
        if self.requested:
            self._release()
        self.token = ""
