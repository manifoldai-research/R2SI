"""Reject task-addressed learning artifacts without routing by task."""
import re
import json
from pathlib import Path

# Validation-only inventory. Never supplied as an executor routing table.
OFFICIAL_TASK_IDS = frozenset(json.loads(Path(__file__).with_name("official_task_ids.json").read_text()))
SELECTOR_FIELDS = frozenset({"task", "task_name", "tasks", "task_ids", "task_selector", "benchmark_task", "routing_table"})


def check_generic(update, known_task_ids=()):
    if update.get("scope", "generic") != "generic" or any(update.get(k) for k in SELECTOR_FIELDS):
        raise ValueError("Only general Skills without task selectors are permitted")
    check_text(" ".join(str(update.get(k, "")) for k in ("id", "description", "text")), known_task_ids)

def check_text(text, known_task_ids=()):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Learning artifact text is empty")
    normalized = re.sub(r"[_\s-]+", " ", text.casefold())
    for task in OFFICIAL_TASK_IDS | frozenset(known_task_ids):
        name = re.sub(r"[_\s-]+", " ", str(task).casefold())
        if name and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", normalized):
            raise ValueError("Learning artifacts must describe geometry and control, not task IDs")

    # Reject explicit task-selector formulations even for names outside the
    # current official inventory. Geometry/control descriptions remain allowed.
    if re.search(r"(?:only|specifically)\s+(?:for|on)\s+(?:the\s+)?(?:task|benchmark)\b|"
                 r"\b(?:task[_ -]?(?:name|id)|benchmark[_ -]?task)\s*(?:==|=|:)|"
                 r"\bif\s+(?:the\s+)?task\s+(?:is|equals|matches)\b", text, re.I):
        raise ValueError("Explicit task selectors are forbidden in reusable artifacts")
