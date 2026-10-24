"""Queue authorized cases for isolated simulator workers with parent-side publication."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT / "XPolicyLab")]
from rsi.storage import Workspace
from rsi.registry import Registry
from rsi.episode_jobs import EpisodeJobs
from rsi.experience import admit

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    ws = Workspace(cfg["workspace"])
    release = Registry(ws).status()["pointers"]["development"]
    # The case file supplies provenance; never relabel official/Open sources.
    for case in cfg["cases"]:
        admit(dict(case, episode_id="preflight", release_id=release), training=False)
    rows = EpisodeJobs(ws, cfg).evaluate(release, cfg["cases"], record=True)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
