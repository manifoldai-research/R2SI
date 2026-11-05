"""The system prompt: how the agent works, the robo reference generated from its docstrings, and the handbook. The same text for every episode."""
from __future__ import annotations

from pathlib import Path

from .robo import Robo
from .tools import method_reference

KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"

CONTRACT = """You control a two-arm robot on a table by writing Python. Your only tool is python(checklist, code), which runs one code cell. The namespace holds `robo`, the robot interface described below, and persists for the whole episode. Every reply must be one python call.

"""


SECOND_LOOK = """You look at a robot's work with fresh eyes. You get the instruction it was given and what its three cameras see now: cam_head looks over the table from above and behind the arms, the other two are on its wrists. You know nothing of how it got here.

Answer a question first if there is one. Then say, most important first and in a few lines:
- what the images show that does not satisfy the instruction;
- what the instruction's words normally include, by common knowledge, that you cannot see done: every part of a familiar whole, an order, a count, how each thing stands or faces, things set out near the work and not used;
- what you cannot tell from these views, and which view would tell.

Name things by where they are in the images. Do not praise.
"""


def system_prompt() -> str:
    handbook = (KNOWLEDGE / "handbook.md").read_text(encoding="utf-8")
    return CONTRACT + "# robo\n\n" + method_reference(Robo) + "\n\n" + handbook + "\n"
