"""Distribution helpers use official layouts and export one identical policy."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.configure_rsi import configuration, write_configuration, PROJECT
from scripts.export_policy import export, POLICY


def bindings(tmp_path):
    catalog = json.loads((POLICY / "harness/configs/wam_skills.json").read_text())
    result = {}
    for profile, name in [("flash", "worldscape-wam-nomemory"), ("memory", "worldscape-wam-memory")]:
        root = tmp_path / profile
        for relative in catalog[name]["checkpoint_files"]:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture: presence only; runtime verifies the bound hashes")
        result[profile] = root
    return result


def test_configuration_preserves_plan_and_checkpoint_bindings(tmp_path):
    weights = bindings(tmp_path)
    plan = dict(collection_cases=[{"case_id": "collect"}],
                validation_cases=[{"case_id": "held-0"}, {"case_id": "held-1"}, {"case_id": "held-2"}],
                required_capabilities=["precision"], required_seeds=[0, 1, 2])
    client = tmp_path / "client.json"; client.write_text("{}")
    bootstrap = tmp_path / "pool.py"; bootstrap.write_text("def configure_pool(): pass")
    with patch("scripts.configure_rsi.prepare", return_value=plan) as prepare:
        cfg, checkpoints = configuration(tmp_path / "run", tmp_path, "/isaac/python.sh",
            weights["flash"], weights["memory"], tasks=["stack_bowls"],
            policy_client_config=client, pool_bootstrap=bootstrap)
    prepare.assert_called_once_with(tmp_path, tasks=["stack_bowls"])
    assert cfg["collection_cases"] == plan["collection_cases"]
    assert cfg["validation_cases"] == plan["validation_cases"]
    assert cfg["source_project"] == str(PROJECT)
    assert cfg["episodes"]["pool_bootstrap"] == cfg["pool_bootstrap"]
    assert cfg["policy_client_config"] == str(client)
    assert "training" not in cfg
    path = write_configuration(cfg, checkpoints)
    assert json.loads(path.read_text()) == cfg
    assert json.loads((path.parent / "checkpoints.json").read_text()) == checkpoints
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_configuration(cfg, checkpoints)
    assert path.read_bytes() == original


def test_open_layout_rejection_precedes_output(tmp_path):
    with pytest.raises(ValueError, match="non-Open"):
        configuration(tmp_path / "run", tmp_path, "/isaac/python.sh",
                      tmp_path, tmp_path, tasks=["align_blocks"])
    assert not (tmp_path / "run").exists()


def test_missing_checkpoint_does_not_create_campaign(tmp_path):
    plan = dict(collection_cases=[], validation_cases=[])
    with patch("scripts.configure_rsi.prepare", return_value=plan):
        with pytest.raises(ValueError, match="checkpoint asset"):
            configuration(tmp_path / "run", tmp_path, "/isaac/python.sh", tmp_path, tmp_path)
    assert not (tmp_path / "run").exists()


def test_export_is_single_directory_and_does_not_overwrite(tmp_path):
    output = tmp_path / "policy.tar.gz"
    report = export(output)
    assert report["files"] > 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        export(output)
    assert output.read_bytes() == original


def test_export_rejects_stale_release(tmp_path):
    import shutil
    policy = tmp_path / "source"
    shutil.copytree(POLICY, policy)
    with (policy / "harness/server.py").open("a") as stream:
        stream.write("\n# changed fixture\n")
    with pytest.raises(RuntimeError, match="Frozen code"):
        export(tmp_path / "bad.tar.gz", policy=policy)
    assert not (tmp_path / "bad.tar.gz").exists()

