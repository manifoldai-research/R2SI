"""Create an autonomous campaign from official layouts and checkpoint bindings."""
import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.prepare_official_layout_cases import prepare


def configuration(workspace, robodojo_root, sim_python, flash_checkpoint,
                  memory_checkpoint, rounds=2, tasks=None,
                  policy_client_config=None, pool_bootstrap=None):
    workspace = Path(workspace).expanduser()
    if not workspace.is_absolute() or workspace == Path(workspace.anchor):
        raise ValueError("Use a dedicated absolute workspace")
    workspace = workspace.resolve()
    if workspace.is_relative_to(PROJECT):
        raise ValueError("Place the campaign workspace outside the source repository")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not Path(sim_python).is_absolute():
        raise ValueError("sim-python must be an absolute interpreter path on the simulator host")
    source = Path(robodojo_root).resolve(strict=True)
    plan = prepare(source, tasks=tasks)
    catalog = json.loads((PROJECT / "XPolicyLab/policy/WorldScape_RSI/harness/configs/wam_skills.json").read_text())
    checkpoints = {}
    for profile, skill, directory in (
        ("flash", "worldscape-wam-nomemory", flash_checkpoint),
        ("memory", "worldscape-wam-memory", memory_checkpoint),
    ):
        root = Path(directory).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError("Checkpoint root must be a directory")
        bound = dict(catalog[skill]["checkpoint_files"])
        for relative in bound:
            path = root / relative
            if not path.resolve().is_relative_to(root) or not path.is_file():
                raise ValueError("Missing checkpoint asset: " + relative)
        checkpoints[profile] = {"path": str(root), "files": bound}
    cfg = dict(workspace=str(workspace), source_project=str(PROJECT),
               checkpoints=str(workspace / "checkpoints.json"), rounds=rounds,
               reflection_episodes=12, **plan)
    cfg["episodes"] = dict(
        python=str(sim_python), robodojo_root=str(source), python_paths=[str(source)],
        environment={"ROBODOJO_ROOT": str(source)},
        max_wall_seconds_per_episode=7200, startup_timeout_s=600,
        agent_timeout_s=600,
        batch_timeout_s=max(14400, 7800 * len(plan["validation_cases"])),
    )
    if policy_client_config:
        path = Path(policy_client_config).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("policy-client-config must be a file")
        cfg["policy_client_config"] = str(path)
    if pool_bootstrap:
        path = Path(pool_bootstrap).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("pool-bootstrap must be a Python module file")
        cfg["pool_bootstrap"] = str(path)
        cfg["episodes"]["pool_bootstrap"] = str(path)
    return cfg, checkpoints


def write_configuration(config, checkpoints):
    root = Path(config["workspace"])
    targets = [(root / "checkpoints.json", checkpoints), (root / "campaign.json", config)]
    if any(path.exists() for path, _ in targets):
        raise FileExistsError("Campaign configuration already exists")
    root.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, value in targets:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            created.append(path)
            with os.fdopen(fd, "w", encoding="utf8") as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return root / "campaign.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("workspace", "robodojo-root", "sim-python", "flash-checkpoint", "memory-checkpoint"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--policy-client-config")
    parser.add_argument("--pool-bootstrap")
    args = parser.parse_args()
    cfg, checkpoints = configuration(**vars(args))
    path = write_configuration(cfg, checkpoints)
    print(json.dumps({"config": str(path), "rounds": cfg["rounds"],
                      "collection_cases": len(cfg["collection_cases"]),
                      "validation_cases": len(cfg["validation_cases"])}))


if __name__ == "__main__":
    main()

