"""Trial lifecycle regressions through official frame dispatch and WS transport."""
import asyncio
from uuid import uuid4

import pytest

from test_startup_state import server_types


class TrialModel:
    def __init__(self, fail=None):
        self.fail = fail
        self.prepares = self.resets = self.updates = self.actions = self.closes = 0

    async def prepare_case(self, value):
        self.prepares += 1

    async def reset(self):
        self.resets += 1
        return {"ready": True}

    async def update_obs(self, value):
        self.updates += 1
        if self.fail == "update":
            raise RuntimeError("update sentinel failure")

    async def get_action(self):
        self.actions += 1
        if self.fail == "action":
            raise RuntimeError("action sentinel failure")
        return [{"joint": [0.0]}]

    async def release_metadata(self):
        return {"release": "fixture"}

    async def on_trial_end(self, value):
        self.closes += 1
        if self.fail == "close":
            raise RuntimeError("cleanup sentinel failure")
        return {"session_released": True}


def make_server(model):
    Server, Config, Frame, M = server_types()
    server = Server(model, Config())
    def frame(kind, trial="trial", payload=None):
        return Frame(message_type=kind, request_id=uuid4().hex, evaluation_id="evaluation",
                     action_case_id="case", trial_id=trial, payload=payload or {})
    return server, frame, M


async def initialized(server, frame, M):
    assert (await server.process_frame(frame(M.RESET))).message_type == M.RESET_RESULT
    duplicate = await server.process_frame(frame(M.RESET))
    assert duplicate.message_type == M.RESET_RESULT
    assert duplicate.payload["initial_reset_already_complete"] is True


@pytest.mark.parametrize("cleanup_recovers", [False, True])
def test_cleanup_failure_invalidates_initial_reset_and_blocks_model_reuse(cleanup_recovers):
    async def run():
        model = TrialModel("close")
        server, frame, M = make_server(model)
        await initialized(server, frame, M)
        assert (await server.process_frame(frame(M.TRIAL_END))).message_type == M.ERROR
        reset = await server.process_frame(frame(M.RESET))
        assert reset.message_type == M.ERROR
        assert server._initial_reset_result is None
        assert ("evaluation", "trial") in server.retired_trials
        assert (await server.process_frame(frame(M.INFER, payload={"observation": {}}))).message_type == M.ERROR
        assert (await server.process_frame(frame(M.CALL, payload={"func_name": "get_action"}))).message_type == M.ERROR
        assert model.resets == 1 and model.actions == model.updates == 0
        assert (await server.process_frame(frame(M.RESET, "other"))).message_type == M.ERROR
        if cleanup_recovers:
            model.fail = None
        closed = await server.process_frame(frame(M.TRIAL_END))
        assert closed.message_type == (M.TRIAL_END_ACK if cleanup_recovers else M.ERROR)
        assert (await server.process_frame(frame(M.RESET))).message_type == M.ERROR
        if cleanup_recovers:
            assert server.owner is None
            assert (await server.process_frame(frame(M.RESET, "other"))).message_type == M.RESET_RESULT
        else:
            assert server.owner == ("evaluation", "trial")
            assert (await server.process_frame(frame(M.RESET, "other"))).message_type == M.ERROR
    asyncio.run(run())


@pytest.mark.parametrize("failure", [None, "update", "action", "missing_observation"])
def test_infer_retires_on_error_and_never_replays_initial_reset(failure):
    async def run():
        model = TrialModel(failure)
        server, frame, M = make_server(model)
        await initialized(server, frame, M)
        payload = {} if failure == "missing_observation" else {"observation": {"state": [0]}}
        inferred = await server.process_frame(frame(M.INFER, payload=payload))
        assert inferred.message_type == (M.INFER_RESULT if failure is None else M.ERROR)
        assert server.owner_started
        assert server._initial_reset_result is None
        reset = await server.process_frame(frame(M.RESET))
        assert reset.message_type == M.ERROR
        assert model.resets == 1
        assert (("evaluation", "trial") in server.retired_trials) == (failure is not None)
        before = (model.updates, model.actions)
        next_infer = await server.process_frame(frame(M.INFER, payload={"observation": {}}))
        if failure is None:
            assert next_infer.message_type == M.INFER_RESULT
            assert (model.updates, model.actions) == (before[0]+1, before[1]+1)
        else:
            assert next_infer.message_type == M.ERROR
            assert (model.updates, model.actions) == before
        assert (await server.process_frame(frame(M.TRIAL_END))).message_type == M.TRIAL_END_ACK
        assert (await server.process_frame(frame(M.RESET))).message_type == M.ERROR
        assert (await server.process_frame(frame(M.RESET, "new"))).message_type == M.RESET_RESULT
    asyncio.run(run())


def test_infer_requires_initialized_owner_without_touching_model():
    async def run():
        model = TrialModel()
        server, frame, M = make_server(model)
        reply = await server.process_frame(frame(M.INFER, payload={"observation": {}}))
        assert reply.message_type == M.ERROR
        assert model.updates == model.actions == 0 and server.owner is None
        await initialized(server, frame, M)
    asyncio.run(run())


def test_other_trial_infer_does_not_change_current_initialization():
    async def run():
        model = TrialModel()
        server, frame, M = make_server(model)
        await initialized(server, frame, M)
        reply = await server.process_frame(frame(M.INFER, "other", {"observation": {}}))
        assert reply.message_type == M.ERROR
        assert not server.owner_started and model.updates == model.actions == 0
        assert (await server.process_frame(frame(M.RESET))).message_type == M.RESET_RESULT
    asyncio.run(run())


def test_readonly_metadata_does_not_consume_initial_reset():
    async def run():
        model = TrialModel()
        server, frame, M = make_server(model)
        await initialized(server, frame, M)
        assert (await server.process_frame(frame(M.CALL, payload={"func_name": "release_metadata"}))).message_type == M.CALL_RESULT
        assert (await server.process_frame(frame(M.RESET))).message_type == M.RESET_RESULT
        assert not server.owner_started and model.resets == 1
    asyncio.run(run())


@pytest.mark.parametrize("phase", ["infer", "cleanup"])
@pytest.mark.parametrize("failure", [False, True])
def test_reset_waits_for_infer_or_cleanup_then_refuses_stale_success(phase, failure):
    async def run():
        entered, proceed = asyncio.Event(), asyncio.Event()
        class SlowModel(TrialModel):
            async def gate(self, current):
                if phase == current:
                    entered.set()
                    await proceed.wait()
                    if failure:
                        raise RuntimeError("state transition sentinel failure")
            async def get_action(self):
                await self.gate("infer")
                return await super().get_action()
            async def on_trial_end(self, value):
                await self.gate("cleanup")
                return await super().on_trial_end(value)
        model = SlowModel()
        server, frame, M = make_server(model)
        await initialized(server, frame, M)
        kind = M.INFER if phase == "infer" else M.TRIAL_END
        first = asyncio.create_task(server.process_frame(frame(kind, payload={"observation": {}})))
        await asyncio.wait_for(entered.wait(), 2)
        duplicate = asyncio.create_task(server.process_frame(frame(M.RESET)))
        await asyncio.sleep(.02)
        assert not first.done() and not duplicate.done()
        heartbeat = await asyncio.wait_for(server.process_frame(frame(M.HEARTBEAT)), 1)
        assert heartbeat.message_type == M.HEARTBEAT_ACK
        proceed.set()
        outcome, reset = await asyncio.wait_for(asyncio.gather(first, duplicate), 2)
        expected = M.INFER_RESULT if phase == "infer" else M.TRIAL_END_ACK
        assert outcome.message_type == (M.ERROR if failure else expected)
        assert reset.message_type == M.ERROR and model.resets == 1
    asyncio.run(run())


@pytest.mark.parametrize("operation", ["infer_ok", "infer_failure", "cleanup_failure"])
def test_official_ws_client_observes_retired_or_started_trial(operation):
    async def run():
        from client_server.ws.protocol.client import PolicyEvalClient, PolicyEvalClientConfig
        from client_server.ws.protocol.exceptions import WsError
        from websockets.asyncio.server import serve
        model = TrialModel({"infer_failure": "action", "cleanup_failure": "close"}.get(operation))
        server, _, _ = make_server(model)
        async with serve(server._handle_connection, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            client = PolicyEvalClient(PolicyEvalClientConfig(
                url=f"ws://127.0.0.1:{port}", evaluation_id="network",
                max_connect_attempts=1, connect_timeout_s=3, handshake_timeout_s=3,
                request_timeout_s=3, close_timeout_s=2))
            await client.connect(handshake=True)
            try:
                await client.reset(trial_id="trial", action_case_id="case")
                await client.reset(trial_id="trial", action_case_id="case")
                if operation == "infer_ok":
                    await client.infer({"state": [0]}, trial_id="trial", action_case_id="case")
                elif operation == "infer_failure":
                    with pytest.raises(WsError):
                        await client.infer({"state": [0]}, trial_id="trial", action_case_id="case")
                else:
                    with pytest.raises(WsError):
                        await client.trial_end(trial_id="trial", action_case_id="case")
                with pytest.raises(WsError):
                    await client.reset(trial_id="trial", action_case_id="case")
                assert model.resets == 1
                model.fail = None
                await client.trial_end(trial_id="trial", action_case_id="case")
                with pytest.raises(WsError):
                    await client.reset(trial_id="trial", action_case_id="case")
                await client.reset(trial_id="next", action_case_id="case")
                assert model.resets == 2
                await client.trial_end(trial_id="next", action_case_id="case")
            finally:
                await client.close()
    asyncio.run(run())
