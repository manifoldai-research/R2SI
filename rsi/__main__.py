import argparse
import json
import os
from pathlib import Path

from .registry import Registry
from .release_builder import build_release
from .experience import publish_snapshot
from .jobs import Jobs

parser = argparse.ArgumentParser(description="RSI isolated development and frozen evaluation control")
parser.add_argument("--workspace", required=True)
commands = parser.add_subparsers(dest="command", required=True)
commands.add_parser("status")
build = commands.add_parser("build")
build.add_argument("--source", required=True)
build.add_argument("--checkpoints", required=True)
build.add_argument("--initialize", action="store_true")
freeze = commands.add_parser("freeze")
freeze.add_argument("--campaign", required=True)
freeze.add_argument("--release", required=True)
snapshot = commands.add_parser("publish-data")
snapshot.add_argument("--base-index", required=True)
snapshot.add_argument("--provenance", required=True)
rollback = commands.add_parser("rollback")
rollback.add_argument("--release", required=True)
rollback.add_argument("--expected-current", required=True)
train = commands.add_parser("train")
train.add_argument("--python", required=True)
train.add_argument("--xpolicy-root", required=True)
train.add_argument("--config", required=True)
train.add_argument("--snapshot", required=True)
train.add_argument("--gpus", required=True)
train.add_argument("--max-steps", type=int, required=True)
train.add_argument("--timeout", type=int, required=True)
train.add_argument("--overrides", nargs="*", default=[])
args = parser.parse_args()
if args.command == 'train':
    from XPolicyLab.policy.WorldScape_R2SI.training import require_training_backend
    require_training_backend()
registry = Registry(args.workspace)
os.environ.update(registry.ws.environment())
if args.command == "status":
    result = registry.status()
elif args.command == "build":
    manifest = build_release(registry.ws, args.source, json.loads(Path(args.checkpoints).read_text()))
    identity = registry.register(manifest)
    if args.initialize:
        registry.initialize(identity)
    result = {"release_id": identity, "root": manifest["root"]}
elif args.command == "freeze":
    registry.freeze_campaign(args.campaign, args.release)
    result = registry.status()
elif args.command == "publish-data":
    result = publish_snapshot(registry.ws, args.base_index, json.loads(Path(args.provenance).read_text()))
elif args.command == "rollback":
    registry.rollback(args.release, args.expected_current)
    result = registry.status()
else:
    result = Jobs(registry.ws).train(python=args.python, xpolicy_root=args.xpolicy_root,
        config=args.config, snapshot=json.loads(Path(args.snapshot).read_text()),
        gpus=[int(x) for x in args.gpus.split(",")], max_steps=args.max_steps,
        timeout_s=args.timeout, overrides=args.overrides)
print(json.dumps(result, ensure_ascii=False, indent=2))
