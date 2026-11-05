"""Episode-scoped, frozen demonstration retrieval for the code-writing agent."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .bridge import EpisodeOver
from .client import InfrastructureFailure

MAX_RESULT_BYTES = 48 * 1024 * 1024


def response_prefix(messages, max_images, max_text_chars):
    """Convert verified reference data to Responses content, never executable tools."""
    converted, images, characters = [], 0, 0
    if not isinstance(messages, list) or not messages:
        raise ValueError("ICL reference must contain messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise ValueError("Invalid ICL message envelope")
        role, content = message["role"], message["content"]
        if role not in ("user", "assistant"):
            raise ValueError("ICL cannot supply system messages or tool calls")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            raise ValueError("Invalid ICL content")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid ICL message part")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                characters += len(part["text"])
                parts.append({"type": "input_text" if role == "user" else "output_text", "text": part["text"]})
            elif role == "user" and part.get("type") == "image_url":
                image = part.get("image_url")
                url = image.get("url", "") if isinstance(image, dict) else ""
                if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                    raise ValueError("ICL images must be embedded verified images")
                images += 1
                parts.append({"type": "input_image", "image_url": url})
            else:
                raise ValueError("Unsupported ICL message part")
        converted.append({"role": role, "content": parts})
    if not 1 <= images <= max_images or characters > max_text_chars:
        raise ValueError("ICL reference exceeds its context budget or has no images")
    return converted


class Reference:
    """A private prefix, deliberately separate from the pruned live conversation."""
    def __init__(self, messages=(), metadata=None):
        self._messages = copy.deepcopy(list(messages))
        self._metadata = copy.deepcopy(metadata or {"status": "disabled"})

    def messages(self):
        return copy.deepcopy(self._messages)

    def info(self):
        return copy.deepcopy(self._metadata)


def _terminate(process):
    # Linux is the supported policy runtime; terminate the entire retrieval tree.
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


class ReferenceRetriever:
    def __init__(self, config, python=None, timeout_s=180, max_images=64, max_text_chars=32000):
        self.config = Path(config).expanduser().resolve(strict=True)
        self.python = shutil.which(str(python or sys.executable))
        if self.python is None:
            raise ValueError("ICL Python executable was not found")
        self.timeout_s = float(timeout_s)
        if not math.isfinite(self.timeout_s) or not 0 < self.timeout_s <= 1800:
            raise ValueError("ICL timeout must be in (0, 1800] seconds")
        if type(max_images) is not int or not 1 <= max_images <= 64:
            raise ValueError("ICL image budget must be an integer in 1..64")
        if type(max_text_chars) is not int or not 1 <= max_text_chars <= 128000:
            raise ValueError("ICL text budget must be an integer in 1..128000")
        self.max_images, self.max_text_chars = max_images, max_text_chars

    def prepare(self, observation, ended):
        if ended.is_set():
            raise EpisodeOver()
        with tempfile.TemporaryDirectory(prefix="r2si-icl-") as directory:
            root = Path(directory)
            from PIL import Image
            # This is an already-decoded public observation, not a trajectory buffer.
            camera = "cam_head" if "cam_head" in observation.images else sorted(observation.images)[0]
            Image.fromarray(observation.images[camera]).save(root / "query.png")
            request = {"instruction": observation.instruction, "image": str(root / "query.png"),
                       "max_images": self.max_images, "max_text_chars": self.max_text_chars}
            (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
            command = [self.python, "-B", str(Path(__file__).with_name("icl_worker.py")),
                       str(self.config), str(root / "request.json"), str(root / "result.json")]
            try:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL, start_new_session=os.name == "posix")
            except OSError as error:
                raise InfrastructureFailure("ICL worker could not start") from error
            started = time.monotonic()
            try:
                while process.poll() is None:
                    if ended.wait(0.05):
                        raise EpisodeOver()
                    if time.monotonic() - started > self.timeout_s:
                        raise InfrastructureFailure("ICL retrieval exceeded its deadline")
                if ended.is_set():
                    raise EpisodeOver()
                result = root / "result.json"
                if not result.is_file() or result.stat().st_size > MAX_RESULT_BYTES:
                    raise InfrastructureFailure("ICL worker produced no bounded result")
                envelope = json.loads(result.read_text(encoding="utf-8"))
                if process.returncode or envelope.get("status") != "ready":
                    # Backend errors can contain paths/credentials: expose only a category.
                    raise InfrastructureFailure("ICL retrieval/compilation failed; check engine configuration and assets")
                prefix = response_prefix(envelope["messages"], self.max_images, self.max_text_chars)
                metadata = envelope["metadata"]
                metadata["responses_prefix_sha256"] = hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest()
                return Reference(prefix, metadata)
            except (ValueError, KeyError, TypeError, OSError) as error:
                raise InfrastructureFailure("Invalid ICL reference result") from error
            finally:
                _terminate(process)


def configured_retriever(cfg):
    config = os.environ.get("R2SI_ICL_CONFIG") or cfg.get("icl_config")
    if not config:
        return None
    return ReferenceRetriever(config, os.environ.get("R2SI_ICL_PYTHON") or cfg.get("icl_python"),
                              cfg.get("icl_timeout_s", 180), cfg.get("icl_max_images", 64),
                              cfg.get("icl_max_text_chars", 32000))
