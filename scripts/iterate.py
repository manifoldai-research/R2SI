"""Run a local RSI campaign; simulator workers consume its episode jobs."""
import argparse
import json
import os
from pathlib import Path
import sys
PROJECT=Path(__file__).resolve().parents[1]
NATIVE=PROJECT/'XPolicyLab/policy/WorldScape_RSI/worldscape-policy/src'
sys.path[:0]=[str(PROJECT),str(PROJECT/'XPolicyLab'),str(NATIVE)]
from rsi.local_loop import LocalLoop
from rsi.storage import Workspace
def run(config):
    os.environ.update(Workspace(config['workspace']).environment())
    return LocalLoop(config).run()
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args()
    results=run(json.loads(Path(args.config).read_text()))
    print(json.dumps([{'status':r['status'],'candidate':r['candidate']} for r in results]))
