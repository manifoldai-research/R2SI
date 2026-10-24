import asyncio
import time
import unittest

from rsi.policy_service import Controller


class Worker:
    def __init__(self):
        self.contexts = {}
        self.calls = 0
        self.closed = False
        self.busy = False

    def call(self, method, key, op, args):
        assert method == "session_call"
        assert not self.busy, "GPU pipe calls overlapped"
        self.busy = True
        try:
            time.sleep(.001)
            if op == "reset":
                self.contexts[key] = 0
            elif op == "get_action":
                self.contexts[key] += 1
                self.calls += 1
            elif op == "release":
                self.contexts.pop(key, None)
            return {"ok": True, "result": self.contexts.get(key)}
        finally:
            self.busy = False

    def close(self):
        self.closed = True


def call(c, key, seq, op, fingerprint=None):
    envelope = dict(instance=c.instance, session=key, seq=seq, op=op, args=[])
    return c.call(envelope, fingerprint or f"{key}:{seq}:{op}")


def test_sixty_interleaved_contexts_and_duplicate_inference():
    async def run():
        workers = [Worker() for _ in range(4)]
        c = Controller(workers, {"flash": [0, 1], "memory": [2, 3]},
                       {"flash": "f", "memory": "m"})
        for n in range(60):
            profile = "memory" if n % 2 else "flash"
            c.open(str(n), str(n), profile, "m" if n % 2 else "f")
        await asyncio.gather(*(call(c, str(n), 1, "reset") for n in range(60)))
        result = await asyncio.gather(*(call(c, str(n), 2, "get_action")
                                        for n in list(range(60))*2))
        assert all(r == {"ok": True, "result": 1} for r in result)
        assert sum(w.calls for w in workers) == 60
        assert c.status()["trials"] == 60
        with unittest.TestCase().assertRaisesRegex(ValueError, "Capacity"):
            c.open("extra", "extra", "flash", "f")
        # A different model context of the same trial does not consume a 61st
        # evaluation slot, and remains a distinct adapter state.
        c.open("alternate", "0", "memory", "m")
        assert c.status()["trials"] == 60
        await call(c, "0", 3, "close")
        with unittest.TestCase().assertRaisesRegex(ValueError, "retired"):
            c.open("0", "0", "flash", "f")
        with unittest.TestCase().assertRaisesRegex(ValueError, "different content"):
            await call(c, "1", 2, "get_action", "changed")
        with unittest.TestCase().assertRaisesRegex(ValueError, "Out-of-order"):
            await call(c, "1", 4, "get_action")
    asyncio.run(run())


def test_wrong_model_and_changed_instance_never_execute():
    async def run():
        w = Worker()
        c = Controller([w], {"flash": [0]}, {"flash": "f"})
        with unittest.TestCase().assertRaisesRegex(ValueError, "identity"):
            c.open("s", "t", "flash", "wrong")
        c.open("s", "t", "flash", "f")
        with unittest.TestCase().assertRaisesRegex(ValueError, "instance"):
            await c.call(dict(instance="old", session="s", seq=1, op="reset", args=[]), "x")
        with unittest.TestCase().assertRaisesRegex(ValueError, "first"):
            await call(c, "s", 1, "get_action")
        assert not w.contexts
    asyncio.run(run())


def test_one_failed_worker_does_not_destroy_the_other_policy():
    class Failed(Worker):
        def call(self, *args):
            raise RuntimeError("device lost")

    async def run():
        bad, good = Failed(), Worker()
        c = Controller([bad, good], {"flash": [0], "memory": [1]},
                       {"flash": "f", "memory": "m"})
        c.open("a", "a", "flash", "f")
        c.open("b", "b", "memory", "m")
        a, b = await asyncio.gather(call(c, "a", 1, "reset"), call(c, "b", 1, "reset"))
        assert not a["ok"] and b["ok"] and bad.closed and not good.closed
        assert (await call(c, "b", 2, "get_action"))["result"] == 1
        with unittest.TestCase().assertRaisesRegex(ValueError, "uncertain"):
            await call(c, "a", 2, "get_action")
    asyncio.run(run())


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in [
        test_sixty_interleaved_contexts_and_duplicate_inference,
        test_wrong_model_and_changed_instance_never_execute,
        test_one_failed_worker_does_not_destroy_the_other_policy,
    ])
