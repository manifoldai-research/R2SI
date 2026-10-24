"""Build a new immutable release instead of editing a loaded code directory."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import uuid

from .generic_contract import check_generic, check_text
from .storage import Workspace, sha256
from .reflection_context import HARNESS, VENDOR, editable_path, load_library


def build_release(workspace, source_project, checkpoints, *, skill_updates=(), memory_updates=(), code_updates=(), training_provenance=None, policy_client_config=None, known_task_ids=()):
    # Validate every proposed artifact before copying or creating a candidate.
    for update in skill_updates:
        check_generic(update, known_task_ids)
    for update in memory_updates:
        check_text(update.get('text', ''), known_task_ids)
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    source = Path(source_project).resolve(strict=True)
    if not (source / "XPolicyLab/policy/WorldScape_RSI").is_dir():
        raise ValueError("Source project lacks the complete standard adapter")
    root = ws.path("releases", uuid.uuid4().hex)
    root.mkdir(parents=True)
    shutil.copytree(source / "XPolicyLab", root / "XPolicyLab",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "release.json"))
    shutil.copytree(source / "env_cfg", root / "env_cfg")
    harness = root / "XPolicyLab/policy/WorldScape_RSI/harness"
    library = load_library(root)
    for update in skill_updates:
        key = update["id"]
        if not key.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid Skill ID')
        previous = library.entries.get(key)
        scope = update.get('scope', previous['scope'] if previous else 'generic')
        task = update.get('task', previous.get('task') if previous else '')
        if scope != 'generic' or task:
            raise ValueError('Only general Skills without task selectors are permitted')
        base = harness/VENDOR/'skills'
        folder = base/'generic'/key
        if previous:
            old = base/'generic'/key if previous['scope'] == 'generic' else base/'tasks'/previous['task']/key
            if old != folder:
                raise ValueError('Keep existing Skill scope; create another ID for reusable extraction')
            meta = json.loads((folder/'manifest.json').read_text())
            if meta['kind'] != 'text':
                raise ValueError('Use code_updates for executable tools')
        else:
            meta = dict(id=key, interface='roborsi.public.v1', kind='text', artifact_type='skill',
                        stages=[], dependencies=[], load_mode='on_demand', description=key)
        folder.mkdir(parents=True, exist_ok=True)
        for field in ('description','dependencies'):
            if field in update:
                meta[field] = update[field]
        meta['version'] = 'rsi-'+root.name
        (folder/'SKILL.md').write_text(update['text'], encoding='utf8')
        (folder/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
    for update in memory_updates:
        key = update['id']
        if not key.replace('-', '').replace('_', '').isalnum():
            raise ValueError('Invalid Memory rule ID')
        (harness/'memory').mkdir(exist_ok=True)
        (harness/'memory'/ (key+'.md')).write_text(update['text'], encoding='utf8')
    for update in code_updates:
        if not editable_path(update['path']):
            raise ValueError('Code update is outside the candidate tool implementation')
        compile(update['content'], update['path'], 'exec')
        target = harness/update['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(update['content'], encoding='utf8')
    load_library(root)  # Check dependencies and actual candidate Skill loading.
    wam_path = harness / "configs/wam_skills.json"
    wam = json.loads(wam_path.read_text())
    model_root = root / "XPolicyLab/policy/WorldScape_RSI"
    model_files = {name for spec in wam.values() for name in spec["source_files"]}
    sources = {name: sha256(model_root / name) for name in sorted(model_files)}
    for name, variant in (("worldscape-wam-memory", "memory"), ("worldscape-wam-nomemory", "flash")):
        checkpoint = checkpoints[variant]
        if not checkpoint["files"]:
            raise ValueError("Checkpoint digest manifest is empty")
        wam[name].update(profile=variant, checkpoint_files=checkpoint["files"], source_files=sources)
    wam_path.write_text(json.dumps(wam, indent=2), encoding="utf8")
    module_spec = importlib.util.spec_from_file_location("rsi_frozen_builder", harness / "release.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    release = module.freeze(harness)
    manifest = {
        "schema": "rsi.release.v1", "root": str(root),
        "harness_release_sha256": sha256(release), "checkpoints": checkpoints,
        "training_provenance": training_provenance,
        "policy_client_config": policy_client_config,
        "files": {p.relative_to(root).as_posix(): sha256(p)
                  for p in root.rglob("*") if p.is_file()},
        "official_eligibility": "unverified",
    }
    ws.atomic("release-manifests/" + root.name + ".json", manifest)
    return manifest
