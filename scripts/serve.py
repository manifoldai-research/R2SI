"""Launch a separate version-pinned single-trial service on an explicit port."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "XPolicyLab")]
from rsi.storage import Workspace
from rsi.worker import VersionedModel
from XPolicyLab.policy.WorldScape_RSI.harness.server import SingleTrialServer, PolicyServerConfig, M


class PersistentTrialServer(SingleTrialServer):
    async def _dispatch_frame(self, frame):
        if frame.message_type == M.RESET and self.owner is None and frame.trial_id:
            # Persist the public trial identity across process restarts. A dropped
            # response or worker failure must never create a second execution.
            self.model.prepare_case({"_trial_identity": json.dumps(
                [frame.evaluation_id, frame.trial_id], separators=(",", ":"))})
        return await super()._dispatch_frame(frame)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--port", required=True, type=int)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--mode", choices=["development", "official"], default="development")
    p.add_argument("--campaign")
    p.add_argument("--memory-device", default="cuda:0")
    p.add_argument("--flash-device", default="cuda:0")
    args = p.parse_args()
    ws = Workspace(args.workspace)
    os.environ.update(ws.environment())
    if not 1024 <= args.port <= 65535:
        p.error("Use an explicit unprivileged port")
    if args.mode == "official" and not args.campaign:
        p.error("Official mode requires a previously frozen campaign")
    cfg = {"policy_name": "WorldScape_RSI", "bench_name": "RoboDojo", "env_cfg_type": "arx_x5", "action_type": "joint",
           "eval_batch": False, "seed": 0}
    model = VersionedModel(ws.root, cfg, mode=args.mode, campaign=args.campaign,
                           memory_device=args.memory_device, flash_device=args.flash_device)
    server = PersistentTrialServer(model, PolicyServerConfig(host=args.host, port=args.port))
    try:
        asyncio.run(server.serve_forever())
    finally:
        model.close()


if __name__ == "__main__":
    main()
