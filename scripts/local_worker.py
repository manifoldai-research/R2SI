"""Simulator worker for local RSI; each listed device is one independent slot."""
import argparse
from pathlib import Path
import sys
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from rsi.episode_jobs import worker
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True)
    parser.add_argument('--devices',nargs='+',default=['cuda:0'])
    parser.add_argument('--once',action='store_true')
    args = parser.parse_args()
    worker(args.workspace,PROJECT,sys.executable,args.devices,once=args.once)
