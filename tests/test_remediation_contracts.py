import importlib.util
import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest
from rsi.generic_contract import check_generic,check_text,OFFICIAL_TASK_IDS
from rsi.reflection_context import load_library,HARNESS
from rsi.validation import compare
from test_recursion import rows

PROJECT=Path(__file__).resolve().parents[1]

def frozen(tmp_path):
    spec=importlib.util.spec_from_file_location("release_fixture",PROJECT/HARNESS/"release.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/"configs").mkdir()
    for name in ("inner_agent.json","wam_skills.json"):
        (tmp_path/"configs"/name).write_bytes((PROJECT/HARNESS/"configs"/name).read_bytes())
    (tmp_path/"method.md").write_text("stable method")
    return module,module.FrozenRelease(module.freeze(tmp_path))

def test_metadata_guard_avoids_content_reads_and_detects_restored_mtime(tmp_path):
    module,release=frozen(tmp_path)
    with patch.object(module,"digest",side_effect=AssertionError("hot path rehash")):
        release.check_unchanged()
    f=tmp_path/"method.md";before=f.stat()
    f.write_text("edited method")
    os.utime(f,ns=(before.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(RuntimeError):release.check_unchanged()

def test_metadata_guard_detects_symlink_retarget_and_strict_mode(tmp_path,monkeypatch):
    module,release=frozen(tmp_path)
    with patch.object(module,"digest",wraps=module.digest) as calls:
        monkeypatch.setenv("WRSI_INTEGRITY_MODE","full")
        release.check_unchanged()
        assert calls.call_count>=3
    monkeypatch.delenv("WRSI_INTEGRITY_MODE")
    f=tmp_path/"method.md";f.unlink();f.symlink_to(tmp_path/"configs/inner_agent.json")
    with pytest.raises(RuntimeError):release.check_unchanged()

def test_every_official_id_rejected_independently_of_current_round():
    assert len(OFFICIAL_TASK_IDS)==54
    for task in OFFICIAL_TASK_IDS:
        for text in (task, task.replace("_","-").upper(),task.replace("_"," ")):
            with pytest.raises(ValueError):check_text("Only for "+text+": select a model",["align_blocks"])
    for value in (dict(task_name="future",text="hold"),dict(task_ids=["future"],text="hold")):
        with pytest.raises(ValueError):check_generic(value)
    with pytest.raises(ValueError):check_text("Only for task future_case, always use memory")

def test_generic_methods_all_preloaded_without_hidden_task_selectors():
    lib=load_library(PROJECT)
    initial=lib.initial()
    assert len(lib.entries)==40
    assert sum(e["artifact_type"]=="skill" for e in lib.entries.values())==34
    assert len(initial)<=24000
    for entry in lib.entries.values():
        check_generic(entry)
        if entry["load_mode"]=="always":assert entry["text"] in initial
    for key in ("support-transfer-selection","held-orientation-and-visibility","thin-section-insertion",
                "complete-interaction-cycles","dual-load-passage","post-transfer-spill-recovery"):
        assert lib.entries[key]["text"] in initial

def test_unlicensed_unused_components_not_part_of_planner_dependency_tree():
    runtime=PROJECT/HARNESS/"vendor/runtime_support"
    native=runtime/"native_runtime"
    assert {p.relative_to(native).as_posix() for p in native.rglob("*") if p.is_file()} == {
        "env/global_configs.py","env/planner_manager/curobo_planner.py","utils/transformer.py","utils/__init__.py"}
    assert not (runtime/"curobo/curobo/examples").exists()
    text=(native/"env/planner_manager/curobo_planner.py").read_text()
    assert "from ..global_configs import BATCH_NUM" in text
    assert "from ...utils.transformer import calculate_target_pose" in text

def test_tied_acceptance_not_mislabeled_improvement():
    report=compare("new","old",rows("old"),rows("new"),required_capabilities=["grasp","push"])
    assert report["passed"] and report["outcome_classification"]=="equal"
    assert not report["demonstrated_improvement"]
