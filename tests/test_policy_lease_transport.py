"""Actual loopback HTTP/codec and lease renewal, with a CPU worker."""
import asyncio
import hashlib
import json
from unittest.mock import patch
from aiohttp import web
from rsi.policy_service import Controller
from test_policy_service import Worker

def test_remote_client_renews_while_idle_and_closes_context(tmp_path,monkeypatch):
    from XPolicyLab.client_server.ws.protocol.codec import encode_frame,decode_frame
    from XPolicyLab.policy.WorldScape_R2SI.harness.remote_wam import RemoteWam,identity
    async def run():
        spec={"profile":"flash"}
        worker=Worker()
        c=Controller([worker],{"flash":[0]},{"flash":identity(spec)},lease_seconds=1)
        async def rpc(request):
            assert request.headers["Authorization"]=="Bearer fixture"
            data=await request.read();m=decode_frame(data)
            try:
                if m["op"]=="open":
                    result=c.open(m["session"],m["trial"],m["profile"],m["identity"])
                elif m["op"]=="heartbeat":result=c.heartbeat(m)
                else:return web.Response(body=encode_frame(await c.call(m,hashlib.sha256(data).hexdigest())))
                return web.Response(body=encode_frame({"ok":True,"result":result}))
            except ValueError as error:
                return web.Response(body=encode_frame({"ok":False,"error":str(error)}))
        app=web.Application();app.router.add_post("/server",rpc)
        runner=web.AppRunner(app,access_log=None);await runner.setup()
        site=web.TCPSite(runner,"127.0.0.1",0);await site.start()
        port=site._server.sockets[0].getsockname()[1]
        config=tmp_path/"client.json"
        config.write_text(json.dumps({"url":f"http://127.0.0.1:{port}/server","token":"fixture","timeout_s":3}))
        monkeypatch.setenv("WRSI_POLICY_CLIENT_CONFIG",str(config))
        client=None
        try:
            client=await asyncio.to_thread(RemoteWam,spec,{"_wam_trial_id":"fixture"})
            await asyncio.to_thread(client.reset)
            for _ in range(8):
                await asyncio.sleep(.25);await c.reap_expired()
            assert c.status()["sessions"]==1 and not client.failed
            await asyncio.to_thread(client.close)
            assert c.status()["sessions"]==0 and worker.contexts=={}
            assert not client._heartbeat_thread.is_alive()
        finally:
            if client and not client.closed:await asyncio.to_thread(client.close)
            await runner.cleanup()
    asyncio.run(run())
