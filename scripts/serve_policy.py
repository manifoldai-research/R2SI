"""Launch only flash/memory GPU workers in the dedicated Policy container."""
import argparse
import asyncio
import os
from pathlib import Path
import socket
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/"XPolicyLab")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--expected-host", required=True)
    p.add_argument("--max-trials", type=int, default=60)
    p.add_argument("--ports", type=int, nargs="+", default=[12658, 65532])
    args = p.parse_args()
    if socket.gethostname() != args.expected_host or args.max_trials not in range(1, 61):
        p.error("Wrong deployment identity or capacity")
    from rsi.storage import Workspace
    ws = Workspace(args.workspace)
    os.umask(0o077)
    os.environ.update(ws.environment())
    os.chdir(ws.root)
    os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2,3", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for port in args.ports:
        with socket.socket() as probe:
            probe.bind(("0.0.0.0", port))
    import subprocess
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("GPU processes exist; no existing workload was stopped")
    from rsi.policy_service import serve
    asyncio.run(serve(PROJECT, ws.root, "0.0.0.0", args.ports, args.max_trials))


if __name__ == "__main__":
    main()
