"""Experience reflection and paired development validation orchestration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .generic_contract import check_generic, check_text
from .experience import admit
from .registry import Registry
from .storage import canonical
from .validation import compare
from .reflection_context import context_tools, read_episode, read_artifact, editable_path, HARNESS, VENDOR

def reflection_tool(skills):
    string_list = {"type": "array", "items": {"type": "string"}}
    fields = {
        "hypothesis": {"type": "string"},
        "evidence_episode_ids": string_list,
        "skill_updates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"id": {"type": "string"}, "text": {"type": "string"},
                           "scope": {"type": "string", "enum": ["generic"]},
                           "task": {"type": "string"}, "description": {"type": "string"},
                           "dependencies": string_list},
            "required": ["id", "text", "scope", "task", "description", "dependencies"]}},
        "training_requested": {"type": "boolean"},
        "training_reason": {"type": "string"},
        "expected_behavior": string_list, "risks": string_list,
    }
    fields.update(memory_updates={"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"id": {"type": "string"}, "text": {"type": "string"}}, "required": ["id", "text"]}},
        code_updates={"type": "array", "items": {"type": "object", "additionalProperties": False,
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
        training={"type": "object", "additionalProperties": False,
            "properties": {"profile": {"type": "string", "enum": ["flash", "memory"]},
                           "learning_rate": {"type": "number"}, "steps": {"type": "integer"}},
            "required": ["profile", "learning_rate", "steps"]})
    return {"type": "function", "function": {
        "name": "propose_rsi_update",
        "description": "Return a reflection proposal only. The inference bridge executes nothing. "
                       "The authorized caller validates it; no running task, file or model is changed by this proposal.",
        "parameters": {"type": "object", "additionalProperties": False,
                       "properties": fields, "required": list(fields)}}}


class OuterLoop:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.ws = registry.ws

    def reflect(self, client, outcomes, skills, known_task_ids=(), *, source_root=None, training_config=None, iteration_history=()):
        evidence, rows = [], {}
        for row in outcomes:
            if row.get("record_purpose") != "collection":
                continue
            admit(row, training=False)
            if not row.get("completed_at"):
                continue
            rows[row["episode_id"]] = row
            summary = {k: row[k] for k in ("episode_id", "task", "release_id", "actions", "score",
                       "success", "terminal_confirmed", "elapsed_s", "reason", "mode", "split",
                       "layout_source", "layout_sha256", "training_eligible") if k in row}
            summary['trajectory_available'] = bool(row.get('trace_path'))
            if row.get('trace_path'):
                summary['first_steps'], _ = read_episode(row, 0, 2)
                summary['ending'], _ = read_episode(row, max(0, row['actions']-2), 2)
            evidence.append(summary)
        if not evidence:
            raise ValueError("No completed authorized development evidence")
        prompt = (Path(__file__).parents[1] / "prompts/outer_reflection.md").read_text(encoding="utf8")
        catalog = {k: ({a: b for a, b in v.items() if a != 'text'} if isinstance(v, dict)
                       else {'preview': v[:180]}) for k, v in skills.items()}
        source = Path(source_root)/HARNESS if source_root else None
        inventory = ([x.relative_to(source).as_posix() for x in (source/VENDOR).glob('*.py')]
                     + [x.relative_to(source).as_posix() for x in (source/'memory').glob('*.md')]) if source else []
        messages = [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({
            "episodes": evidence, "skills": catalog, "editable_files": inventory,
            "training_resources": training_config, "previous_iterations": list(iteration_history)}, ensure_ascii=False)}]
        tools = [reflection_tool(skills), *context_tools()]
        while True:
            message = client.complete(messages, tools)["choices"][0]["message"]
            calls = message.get('tool_calls', [])
            if len(calls) != 1:
                raise ValueError('Exactly one reflection tool call is required')
            call = calls[0]; name = call['function']['name']; args = json.loads(call['function']['arguments'])
            messages.append({'role': 'assistant', 'content': message.get('content'), 'tool_calls': calls})
            if name == 'propose_rsi_update':
                proposal = args
                self.validate_proposal(proposal, skills, set(rows), known_task_ids)
                break
            images = []
            if name == 'read_episode':
                row = rows[args.pop('episode_id')]
                data, images = read_episode(row, **args)
            elif name == 'read_artifact':
                data = read_artifact(source_root, skills, **args)
            else:
                raise ValueError('Unadvertised reflection tool')
            messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(data, ensure_ascii=False)})
            if images:
                messages.append({'role': 'user', 'content': images})
        identity = hashlib.sha256(canonical(proposal)).hexdigest()
        self.ws.atomic("proposals/" + identity + ".json", {"proposal": proposal,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "evidence_sha256": hashlib.sha256(canonical(evidence)).hexdigest()})
        return proposal

    @staticmethod
    def validate_proposal(proposal, skills, episodes, known_task_ids):
        allowed = {"hypothesis", "evidence_episode_ids", "skill_updates", "training_requested",
                   "training_reason", "expected_behavior", "risks"}
        allowed |= {"memory_updates", "code_updates", "training"}
        if not set(proposal) <= allowed or type(proposal["training_requested"]) is not bool:
            raise ValueError("Invalid reflection output schema")
        for key in ('hypothesis', 'training_reason'):
            if not isinstance(proposal[key], str) or len(proposal[key]) > 16000:
                raise ValueError('Invalid reflection text')
        for key in ('evidence_episode_ids', 'expected_behavior', 'risks'):
            if not isinstance(proposal[key], list) or any(not isinstance(x, str) for x in proposal[key]):
                raise ValueError('Invalid reflection list')
        if not isinstance(proposal['skill_updates'], list):
            raise ValueError('Invalid Skill update list')
        cited = set(proposal["evidence_episode_ids"])
        if not cited or not cited <= episodes:
            raise ValueError("Proposal cites absent or incomplete evidence")
        used = set()
        for update in proposal["skill_updates"]:
            if set(update) - {"id", "text", "scope", "task", "description", "dependencies"}:
                raise ValueError("Unknown Skill fields")
            key, text = update["id"], update["text"]
            if not key.replace('-', '').replace('_', '').isalnum() or key in used:
                raise ValueError("Invalid or duplicate Skill ID")
            existing = skills.get(key)
            if isinstance(existing, dict) and existing.get('kind') == 'code':
                raise ValueError('Use code_updates to change executable implementations')
            scope = update.get('scope', existing.get('scope', 'generic') if isinstance(existing, dict) else 'generic')
            if scope != 'generic':
                raise ValueError('Only general Skills are permitted')
            check_generic(update, known_task_ids)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Skill text is empty")
            used.add(key)
        for update in proposal.get('memory_updates', []):
            check_text(update.get('text', ''), known_task_ids)
            if set(update) != {'id', 'text'} or not update['id'].replace('-', '').replace('_', '').isalnum() or not update['text'].strip():
                raise ValueError('Invalid Memory rule')
        for update in proposal.get('code_updates', []):
            if set(update) != {'path', 'content'} or not editable_path(update['path']):
                raise ValueError('Code updates must name tool/agent files inside the candidate adapter')
            compile(update['content'], update['path'], 'exec')
        if proposal['training_requested']:
            training = proposal.get('training', {})
            import math
            if (set(training) != {'profile', 'learning_rate', 'steps'} or training['profile'] not in {'memory','flash'}
                    or type(training['steps']) is not int or training['steps'] < 1
                    or type(training['learning_rate']) not in (int,float)
                    or not math.isfinite(training['learning_rate']) or training['learning_rate'] <= 0):
                raise ValueError('Training needs a profile, positive learning rate and step count')

    def evaluate_and_promote(self, candidate, baseline, evaluator, *, cases,
                             required_capabilities, required_seeds=(0, 1, 2),
                             trained_cases=(), promote=True):
        """evaluator(release_id, cases) must run the same real dev environments.

        No hidden retry or synthetic fallback: failures leave the baseline active.
        """
        before = evaluator(baseline, cases)
        after = evaluator(candidate, cases)
        report = compare(candidate, baseline, before, after,
                         required_capabilities=required_capabilities, required_seeds=required_seeds,
                         trained_cases=trained_cases)
        self.ws.atomic("validation/" + candidate + ".json", {"report": report, "baseline": before, "candidate": after})
        self.registry.record_validation(candidate, report)
        if report["passed"] and promote:
            self.registry.promote(candidate, baseline)
        return report
