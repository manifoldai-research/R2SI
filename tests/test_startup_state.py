"""Startup races and bytecode-cache regressions; no model predictions."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from uuid import uuid4

import pytest

PROJECT = Path(__file__).resolve().parents[1]
POLICY = PROJECT / "XPolicyLab/policy/WorldScape_R2SI"


def fresh_process(tmp_path, source):
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.pop("PYTHONPYCACHEPREFIX", None)
    env.pop("WRSI_INTEGRITY_MODE", None)
    env["TMPDIR"] = str(tmp_path)
    result = subprocess.run([sys.executable, "-c", source], env=env, cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_real_python_cache_creation_after_freeze_is_not_tampering(tmp_path):
    helper = tmp_path / "helper"; helper.mkdir()
    shutil.copyfile(POLICY / "harness/release.py", helper / "release.py")
    root = tmp_path / "frozen"; (root / "configs").mkdir(parents=True)
    for name in ("inner_agent.json", "wam_skills.json"):
        shutil.copyfile(POLICY / "harness/configs" / name, root / "configs" / name)
    (root / "late_import.py").write_text("VALUE = 42\n")
    out = fresh_process(tmp_path, f"""
import sys, importlib
from pathlib import Path
sys.path.insert(0, {str(helper)!r})
import release
root = Path({str(root)!r})
frozen = release.FrozenRelease(release.freeze(root))
assert not sys.dont_write_bytecode
assert not (root/'__pycache__').exists()
sys.path.insert(0,str(root))
assert importlib.import_module('late_import').VALUE == 42
assert list((root/'__pycache__').glob('*.pyc'))
frozen.check_unchanged()
(root/'__pycache__'/'additional.pyc').write_bytes(b'cache only')
frozen.check_unchanged()
print('real_cache_accepted')
""")
    assert "real_cache_accepted" in out


@pytest.mark.parametrize("change", ["edit_source", "add_source", "replace_directory", "cache_symlink"])
def test_cache_exception_does_not_relax_source_protection(tmp_path, change):
    from test_remediation_contracts import frozen
    module, release = frozen(tmp_path)
    if change == "edit_source":
        file = tmp_path / "method.md"; st = file.stat()
        file.write_text("edited method")
        os.utime(file, ns=(st.st_atime_ns, st.st_mtime_ns))
    elif change == "add_source":
        (tmp_path / "injected.py").write_text("VALUE = 2")
    elif change == "replace_directory":
        (tmp_path / "configs").rename(tmp_path / "original")
        shutil.copytree(tmp_path / "original", tmp_path / "configs")
    else:
        (tmp_path / "external-cache").mkdir()
        (tmp_path / "__pycache__").symlink_to(tmp_path / "external-cache", target_is_directory=True)
    with pytest.raises(RuntimeError):
        release.check_unchanged()


def test_package_entry_disables_bytecode_without_shell_environment(tmp_path):
    package = tmp_path / "policy_entry"; package.mkdir()
    shutil.copyfile(POLICY / "__init__.py", package / "__init__.py")
    (package / "late_module.py").write_text("VALUE = 1\n")
    out = fresh_process(tmp_path, """
import sys,os
assert not sys.dont_write_bytecode
import policy_entry
assert sys.dont_write_bytecode
assert os.environ['PYTHONDONTWRITEBYTECODE'] == '1'
import policy_entry.late_module
from pathlib import Path
assert not list(Path('policy_entry').glob('__pycache__/late_module.*.pyc'))
print('startup_enforced')
""")
    assert "startup_enforced" in out


def server_types():
    from XPolicyLab.policy.WorldScape_R2SI.harness.server import SingleTrialServer
    from client_server.ws.model_server import PolicyServerConfig
    from client_server.ws.protocol.schemas import Frame
    from client_server.ws.protocol.messages import MessageType as M
    return SingleTrialServer, PolicyServerConfig, Frame, M


@pytest.mark.parametrize("phase", ["prepare", "reset"])
@pytest.mark.parametrize("failure", [False, True])
def test_concurrent_resets_wait_for_one_initialization_and_share_outcome(phase, failure):
    async def run():
        Server, Config, Frame, M = server_types()
        entered, proceed = asyncio.Event(), asyncio.Event()
        class Model:
            prepares = resets = closed = 0
            async def gate(self, step):
                if step == phase:
                    entered.set()
                    await proceed.wait()
                    if failure:
                        raise RuntimeError("initialization sentinel failure")
            async def prepare_case(self, value):
                self.prepares += 1
                await self.gate("prepare")
            async def reset(self):
                self.resets += 1
                await self.gate("reset")
                return {"ready": True}
            async def on_trial_end(self, value):
                self.closed += 1
                return {"session_released": True}
        model = Model(); server = Server(model, Config())
        def frame(kind, trial="same"):
            return Frame(message_type=kind, request_id=uuid4().hex, evaluation_id="evaluation",
                         action_case_id="example_case", trial_id=trial, payload={})
        one = asyncio.create_task(server.process_frame(frame(M.RESET)))
        await asyncio.wait_for(entered.wait(), 3)
        two = asyncio.create_task(server.process_frame(frame(M.RESET)))
        await asyncio.sleep(.03)
        assert not one.done() and not two.done()
        heartbeat = await asyncio.wait_for(server.process_frame(frame(M.HEARTBEAT)), 1)
        assert heartbeat.message_type == M.HEARTBEAT_ACK
        proceed.set()
        a,b = await asyncio.wait_for(asyncio.gather(one,two), 3)
        assert model.prepares == 1
        assert model.resets == (0 if failure and phase == "prepare" else 1)
        assert a.request_id != b.request_id
        if failure:
            assert a.message_type == b.message_type == M.ERROR
            assert a.payload == b.payload
            assert a.payload["code"] == "reset_failed"
            assert ("evaluation","same") in server.retired_trials
        else:
            assert a.message_type == b.message_type == M.RESET_RESULT
            assert a.payload["result"] == b.payload["result"] == {"ready":True}
            assert b.payload["initial_reset_already_complete"] is True
        before=(model.prepares,model.resets)
        refused=await server.process_frame(frame(M.RESET,"other"))
        assert refused.message_type == M.ERROR
        assert before == (model.prepares,model.resets)
        ended=await server.process_frame(frame(M.TRIAL_END))
        assert ended.message_type == M.TRIAL_END_ACK
        assert server.owner is None and model.closed == 1
        retired=await server.process_frame(frame(M.RESET))
        assert retired.message_type == M.ERROR
    asyncio.run(run())


def test_cancelled_duplicate_waiter_does_not_cancel_initialization():
    async def run():
        Server, Config, Frame, M = server_types()
        entered, proceed = asyncio.Event(), asyncio.Event()
        class Model:
            resets=0
            async def prepare_case(self, value): pass
            async def reset(self):
                self.resets+=1;entered.set();await proceed.wait()
                return {"ready":True}
        model=Model();server=Server(model,Config())
        def reset():
            return Frame(message_type=M.RESET,request_id=uuid4().hex,evaluation_id="e",
                         trial_id="t",action_case_id="case",payload={})
        first=asyncio.create_task(server.process_frame(reset()))
        await entered.wait()
        retry=reset();waiter=asyncio.create_task(server.process_frame(retry))
        await asyncio.sleep(.02);waiter.cancel()
        with pytest.raises(asyncio.CancelledError):await waiter
        assert not first.done()
        proceed.set()
        assert (await first).message_type == M.RESET_RESULT
        assert (await server.process_frame(retry)).message_type == M.RESET_RESULT
        assert model.resets==1
    asyncio.run(run())
