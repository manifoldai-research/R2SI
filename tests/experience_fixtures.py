"""CPU-only publication helper; never native evaluation evidence."""
import hashlib
from rsi.storage import sha256
from rsi.experience_publication import commit_experience

def publish_fixture(ws, meta, row):
    identity = hashlib.sha256(meta["episode_id"].encode()).hexdigest()
    candidate = {**row, "staged_experience": {
        "identity": identity, "sha256": sha256(ws.path("experience/staged", identity + ".json"))}}
    return commit_experience(ws, meta, candidate,
        {"id": "cpu-fixture-" + meta["episode_id"], "status": "completed", "exit_code": 0}, "f" * 64)
