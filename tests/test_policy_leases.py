"""Lease recovery does not execute robot requests or restart services."""
import asyncio
import threading
import time
import pytest
from rsi.policy_service import Controller
from test_policy_service import Worker,call

def test_abandoned_contexts_reclaimed_and_identity_never_replayed():
    async def run():
        now=[0.];w=Worker()
        c=Controller([w],{"flash":[0]},{"flash":"f"},max_trials=2,lease_seconds=10,clock=lambda:now[0])
        for key in ("a","b"):
            c.open(key,key,"flash","f");assert (await call(c,key,1,"reset"))["ok"]
        now[0]=11
        await c.reap_expired()
        assert c.status()["sessions"]==0 and w.contexts=={}
        c.open("new","new","flash","f")
        with pytest.raises(ValueError):await call(c,"a",1,"reset")
        with pytest.raises(ValueError):c.open("a","a","flash","f")
    asyncio.run(run())

def test_heartbeat_requires_owner_and_prevents_expiry():
    async def run():
        now=[0.];w=Worker();c=Controller([w],{"flash":[0]},{"flash":"f"},lease_seconds=10,clock=lambda:now[0])
        reply=c.open("a","trial","flash","f")
        await call(c,"a",1,"reset")
        now[0]=8
        with pytest.raises(ValueError):c.heartbeat(dict(instance=c.instance,session="a",lease_token="wrong"))
        c.heartbeat(dict(instance=c.instance,session="a",lease_token=reply["lease_token"]))
        now[0]=15;await c.reap_expired()
        assert c.status()["sessions"]==1 and "a" in w.contexts
        now[0]=19
        with pytest.raises(ValueError):c.heartbeat(dict(instance=c.instance,session="a",lease_token=reply["lease_token"]))
        await c.reap_expired()
        assert not w.contexts
    asyncio.run(run())

def test_inflight_model_operation_protected_until_completion():
    class Slow(Worker):
        def call(self,method,key,op,args):
            if op=="get_action":time.sleep(.15)
            return super().call(method,key,op,args)
    async def run():
        now=[0.];w=Slow();c=Controller([w],{"flash":[0]},{"flash":"f"},lease_seconds=10,clock=lambda:now[0])
        c.open("a","t","flash","f");await call(c,"a",1,"reset")
        task=asyncio.create_task(call(c,"a",2,"get_action"))
        await asyncio.sleep(.03);now[0]=100
        await c.reap_expired()
        assert c.status()["sessions"]==1
        assert (await task)["ok"]
        await c.reap_expired();assert c.status()["sessions"]==1
        now[0]=111;await c.reap_expired();assert c.status()["sessions"]==0
    asyncio.run(run())
