"""The system prompt: Memory rules, how the agent works, the robo reference generated from its docstrings with the tool notes, the Skills
loaded at every step (load_mode always) and the catalog of the others (load_mode on_demand). The same text for every episode."""
from __future__ import annotations

from pathlib import Path

from .robo import Robo
from .tools import method_reference

KNOWLEDGE = Path(__file__).resolve().parent / "knowledge"

CONTRACT = """You control a two-arm robot on a table by writing Python. Your only tool is python(checklist, code), which runs one code cell. The namespace holds `robo`, the robot interface described below, and persists for the whole episode. Every reply must be one python call.

The Memory rules above always apply. The Skills used at every step follow the robo reference in full; more Skills are
listed at the end by id, and robo.read_skill(id, offset=0) reads one when it fits your situation (robo.skills() lists
all). Skill text is guidance, not executable code: any older tool names in it describe capabilities; execute only the
documented robo methods below. Where a Skill read on demand disagrees with the Memory rules or the Skills given in full,
follow the latter.

"""


SECOND_LOOK = """You look at a robot's work with fresh eyes. You get the instruction it was given and what its three cameras see now: cam_head looks over the table from above and behind the arms, the other two are on its wrists. You know nothing of how it got here.

Answer a question first if there is one. Then say, most important first and in a few lines:
- what the images show that does not satisfy the instruction;
- what the instruction's words normally include, by common knowledge, that you cannot see done: every part of a familiar whole, an order, a count, how each thing stands or faces, things set out near the work and not used;
- what you cannot tell from these views, and which view would tell.

Name things by where they are in the images. Do not praise.
"""


def system_prompt() -> str:
    from .skills import library, shared_context
    lib = library()
    notes = (KNOWLEDGE / "tool-notes.md").read_text(encoding="utf-8")
    more = "\n".join(f"- {row['id']}: {row['description']}" for row in sorted(lib.catalog("skill"), key=lambda row: row["id"])
                     if lib.entries[row["id"]].get("load_mode") != "always")
    return (shared_context() + CONTRACT + "# robo\n\n" + method_reference(Robo) + "\n\n" + notes + "\n\n"
            + "# Skills used at every step\n\n" + lib.initial() + "\n\n"
            + "# More Skills (robo.read_skill(id) reads one)\n\n" + more + "\n")


def parts() -> dict:
    """What every episode's system prompt loads, for the episode metrics."""
    from .skills import library, memory_files
    lib = library()
    return {"memory_rules": memory_files(), "skills_always": sorted(k for k, m in lib.entries.items() if m.get("load_mode") == "always"),
            "skills_on_demand": sum(1 for m in lib.entries.values() if m.get("load_mode") != "always")}
