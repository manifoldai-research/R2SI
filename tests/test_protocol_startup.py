import os
from pathlib import Path
import subprocess
import sys

def test_official_protocol_initialization_and_isolation():
    project=Path(__file__).resolve().parents[1]
    script=project/'XPolicyLab/policy/WorldScape_RSI/harness/tests/protocol_smoke.py'
    result=subprocess.run([sys.executable,'-B',str(script)],cwd=project,
                          capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    assert '"initial_duplicate_reset": "idempotent"' in result.stdout
    assert '"executing_trial_reset": "rejected"' in result.stdout
