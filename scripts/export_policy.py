"""Export the inner Agent as one policy/WorldScape_R2SI directory."""
import argparse
import hashlib
import importlib.util
from pathlib import Path
import tarfile

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "XPolicyLab/policy/WorldScape_R2SI"


def export(output, policy=POLICY):
    policy = Path(policy).resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if output.is_relative_to(policy):
        raise ValueError("Write the archive outside the policy source tree")
    spec = importlib.util.spec_from_file_location("export_release", policy / "harness/release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    release = module.FrozenRelease(policy / "harness/release.json")
    files = []
    for path in sorted(policy.rglob("*")):
        relative = path.relative_to(policy)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in relative.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("Policy export expects regular source files")
        if path.is_file():
            files.append((path, "policy/WorldScape_R2SI/" + relative.as_posix()))
    output.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output.open("xb") as stream:
            created = True
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for path, name in files:
                    archive.add(path, arcname=name, recursive=False)
        with tarfile.open(output, "r:gz") as archive:
            members = archive.getmembers()
            if {m.name for m in members} != {name for _, name in files}:
                raise RuntimeError("Archive inventory differs")
            sources = dict((name, path) for path, name in files)
            for member in members:
                if not member.isfile() or archive.extractfile(member).read() != sources[member.name].read_bytes():
                    raise RuntimeError("Archive readback differs")
    except BaseException:
        if created:
            output.unlink()
        raise
    return {"path": str(output), "files": len(files),
            "release_sha256": release.sha256,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(export(args.output)))


if __name__ == "__main__":
    main()

