"""Publish closed, acknowledged development episodes while the native trainer runs."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "XPolicyLab"),
               str(PROJECT / "XPolicyLab/policy/WorldScape_R2SI/worldscape-policy/src")]
from rsi.storage import Workspace
from rsi.experience import publish_snapshot

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--base-index", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--duration-seconds", type=int, default=0)
    p.add_argument("--interval-seconds", type=int, default=30)
    args = p.parse_args()
    if args.duration_seconds < 0 or args.interval_seconds < 1:
        p.error("Use a bounded duration and positive interval")
    ws = Workspace(args.workspace)
    os.environ.update(ws.environment())
    provenance = json.loads(Path(args.provenance).read_text())
    deadline = time.monotonic() + args.duration_seconds
    while True:
        record = publish_snapshot(ws, args.base_index, provenance)
        print(json.dumps(record), flush=True)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(args.interval_seconds, deadline - time.monotonic()))
