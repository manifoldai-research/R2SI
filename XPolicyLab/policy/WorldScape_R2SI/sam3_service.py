"""One SAM 3 per host for all of its policy servers: a deployment choice that leaves the segmentation unchanged.

A policy server normally loads its own SAM 3 (segmenter.py, about 5 GB of GPU memory). With R2SI_SEGMENTER_SERVICE=127.0.0.1:<port>
it sends its segment / segment_point calls to this service instead. The service runs the same Segmenter code on the same checkpoint,
so weights, bfloat16 autocast, pre-processing and thresholds are those of a local model. Each episode is a session that holds its own
cached image encoding, exactly what a local Segmenter would hold; forget() at the next episode closes it, and a client that disconnects
loses its sessions, so no two episodes or servers ever share a cache. A dropped encoding is only recomputed from the same image. Requests
run one at a time on one GPU thread in arrival order (see GpuThread); the request log records each one's wait and compute time. Before
loading SAM 3 the service imports cuRobo as a policy server does, since that import sets process-wide torch flags (cudnn.benchmark).

A client refuses to start unless the service runs the same segmenter.py and SAM 3 sources on the same checkpoint file with the same
process-wide torch settings (segmenter.numerics) as the client's own process, and every request is checked against those settings again.

    python -m XPolicyLab.policy.WorldScape_R2SI.sam3_service --checkpoint .../sam3.pt --port 18300 --log .../requests.jsonl

The service listens on the loopback interface only. A message is a JSON header followed by raw array bytes; nothing is unpickled.
"""
from __future__ import annotations

import argparse
import builtins
import collections
import hashlib
import json
import logging
import os
import queue
import signal
import socket
import socketserver
import struct
import sys
import threading
import time
import uuid

import numpy as np

from . import segmenter as _segmenter
from .segmenter import MAX_INSTANCES, MIN_SCORE, Instance, Slot

PROTOCOL = 1
MAGIC = b"R2S3"
MAX_HEADER = 1 << 20
MAX_BLOB = 1 << 28
RAISED_AS_IS = (ValueError, TypeError, IndexError, KeyError, AttributeError, AssertionError, ArithmeticError, NotImplementedError)
log = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    """The service did not answer, failed, or computes differently from a local Segmenter in this process."""


# -- wire format ----------------------------------------------------------------------------------------
def _exactly(sock, size: int) -> bytearray:
    buffer = bytearray(size)
    view, got = memoryview(buffer), 0
    while got < size:
        count = sock.recv_into(view[got:], size - got)
        if not count:
            raise ConnectionError("connection closed")
        got += count
    return buffer


def _send(sock, header: dict, blobs=()):
    blobs = [memoryview(blob).cast("B") for blob in blobs]
    raw = json.dumps({**header, "blobs": [blob.nbytes for blob in blobs]}, separators=(",", ":")).encode()
    sock.sendall(MAGIC + struct.pack(">I", len(raw)) + raw)
    for blob in blobs:
        sock.sendall(blob)


def _receive(sock):
    head = _exactly(sock, 8)
    if bytes(head[:4]) != MAGIC:
        raise ValueError("not a SAM 3 service message")
    (size,) = struct.unpack(">I", head[4:])
    if size > MAX_HEADER:
        raise ValueError("header too large")
    header = json.loads(bytes(_exactly(sock, size)))
    sizes = header.get("blobs", [])
    if any(not isinstance(n, int) or n < 0 or n > MAX_BLOB for n in sizes):
        raise ValueError("bad payload size")
    return header, [_exactly(sock, n) for n in sizes]


def _plain(value):
    """A key or argument as JSON keeps it: tuples become lists, numpy scalars Python numbers. Equality is unchanged."""
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _image(header: dict, blobs) -> np.ndarray:
    spec = header["image"]
    shape = [int(n) for n in spec["shape"]]
    if spec.get("dtype") != "|u1" or len(blobs) != 1 or len(blobs[0]) != int(np.prod(shape)):
        raise ValueError("an image is one uint8 array")
    return np.frombuffer(blobs[0], dtype=np.uint8).reshape(shape)


def _digest(path) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def code_digest() -> str:
    """The Segmenter's code: the service must run the segmenter.py its clients would run."""
    return _digest(_segmenter.__file__)


def file_identity(path) -> dict:
    real = os.path.realpath(path)
    info = os.stat(real)
    return {"path": real, "size": info.st_size, "mtime_ns": info.st_mtime_ns, "device": info.st_dev, "inode": info.st_ino}


def settings_digest(rows: dict) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def policy_server_settings():
    """The process-wide torch settings a policy server has when its SAM 3 runs: AgentRuntime builds cuRobo first, and importing cuRobo
    (planning.Curobo) turns on cudnn.benchmark and TF32. The service imports the same modules before loading SAM 3."""
    from . import planning
    if str(planning.THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(planning.THIRD_PARTY))
    import curobo.inverse_kinematics  # noqa: F401
    import curobo.motion_planner  # noqa: F401
    import curobo.runtime  # noqa: F401
    import curobo.types  # noqa: F401


def parse_address(address: str):
    host, _, port = str(address).strip().rpartition(":")
    if host not in ("127.0.0.1", "localhost") or not port.isdigit():
        raise ValueError("the SAM 3 service address is 127.0.0.1:<port> (loopback only): " + str(address))
    return "127.0.0.1", int(port)


# -- policy-server side ---------------------------------------------------------------------------------
class ServiceSegmenter:
    """The Segmenter interface backed by the host's shared SAM 3 service. Nothing is loaded here; each episode is its own session."""

    def __init__(self, address: str, checkpoint: str, wait_s: float | None = None, timeout_s: float | None = None):
        _segmenter.prepare_process()
        self.address = parse_address(address)
        self._timeout_s = float(timeout_s or os.environ.get("R2SI_SEGMENTER_SERVICE_TIMEOUT_S") or 600)
        self._retry_s = 60.0
        self._label = os.environ.get("ROBODOJO_RUN_ID") or f"{socket.gethostname()}-{os.getpid()}"
        self._lock = threading.Lock()
        self._socket = None
        self._code = code_digest()
        self._settings = _segmenter.numerics()
        self._session = uuid.uuid4().hex
        wait_s = float(wait_s or os.environ.get("R2SI_SEGMENTER_SERVICE_WAIT_S") or 300)
        with self._lock:
            reply, _ = self._call({"op": "hello", "protocol": PROTOCOL, "label": self._label}, retry_s=wait_s)
        self.info = reply
        self._check(reply, checkpoint)

    @property
    def session(self) -> str:
        return self._session

    def _check(self, info: dict, checkpoint):
        where = "%s:%d" % self.address
        problems = []
        if info.get("protocol") != PROTOCOL:
            problems.append(f"protocol {info.get('protocol')} != {PROTOCOL}")
        if info.get("code") != self._code:
            problems.append("its segmenter.py differs from this one")
        mine, theirs = file_identity(checkpoint), info.get("checkpoint") or {}
        if (mine["device"], mine["inode"]) != (theirs.get("device"), theirs.get("inode")) and mine["path"] != theirs.get("path"):
            if mine["size"] != theirs.get("size"):
                problems.append(f"checkpoint {theirs.get('path')} is not {mine['path']}")
            else:
                with self._lock:
                    reply, _ = self._call({"op": "checkpoint_sha256"})
                if reply.get("sha256") != _digest(mine["path"]):
                    problems.append(f"checkpoint {theirs.get('path')} differs from {mine['path']}")
        theirs_settings = info.get("numerics") or {}
        differ = {key: [self._settings.get(key), theirs_settings.get(key)] for key in sorted(set(self._settings) | set(theirs_settings))
                  if self._settings.get(key) != theirs_settings.get(key)}
        if differ:
            problems.append("torch settings differ (here, service): " + json.dumps(differ))
        if problems:
            self._disconnect()
            raise ServiceError(f"SAM 3 service at {where} would not compute like a local Segmenter: " + "; ".join(problems))

    def forget(self):
        """A new episode: a new session; the service drops the old one's cached image."""
        with self._lock:
            old, self._session = self._session, uuid.uuid4().hex
            try:
                self._call({"op": "close", "session": old}, retry_s=5.0)
            except ServiceError as error:
                log.warning("SAM 3 service: session %s not closed: %s", old, error)

    def close(self):
        with self._lock:
            try:
                if self._socket is not None:
                    self._call({"op": "close", "session": self._session}, retry_s=0.0)
            except ServiceError:
                pass
            self._disconnect()

    def segment(self, image: np.ndarray, text: str, key=None, min_score: float = MIN_SCORE, limit: int = MAX_INSTANCES) -> list[Instance]:
        """Instances of a noun phrase in an image, best first. `key` identifies the image so its encoding is reused."""
        image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        return self._request({"op": "segment", "key": _plain(key), "text": text, "min_score": _plain(min_score), "limit": _plain(limit)}, image)

    def segment_point(self, image: np.ndarray, pixel, key=None) -> list[Instance]:
        """The object under one pixel: candidate masks, best first."""
        image = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        return self._request({"op": "segment_point", "key": _plain(key), "pixel": [float(pixel[0]), float(pixel[1])]}, image)

    def _request(self, header: dict, image: np.ndarray) -> list[Instance]:
        header["image"] = {"shape": list(image.shape), "dtype": image.dtype.str}
        with self._lock:
            # This process's settings as they are now: a setting changed since start-up would change a local SAM 3's arithmetic too.
            header.update(session=self._session, label=self._label, numerics=settings_digest(_segmenter.numerics()))
            reply, blobs = self._call(header, [image])
        if not reply.get("ok"):
            error = reply.get("error") or {}
            kind = getattr(builtins, str(error.get("type")), None) if error.get("builtin") else None
            if isinstance(kind, type) and issubclass(kind, RAISED_AS_IS):
                raise kind(error.get("message"))
            raise ServiceError(f"SAM 3 service: {error.get('type')}: {error.get('message')}")
        rows = reply["instances"]
        if not rows:
            return []
        count, height, width = reply["masks"]["shape"]
        bits = np.unpackbits(np.frombuffer(blobs[0], dtype=np.uint8), count=count * height * width).reshape(count, height, width).astype(bool)
        return [Instance(score=row["score"], mask=bits[index].copy(), box=tuple(row["box"]), area=row["area"]) for index, row in enumerate(rows)]

    def _disconnect(self):
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None

    def _call(self, header: dict, blobs=(), retry_s: float | None = None):
        """One request and its reply, reconnecting while the service is unreachable for up to retry_s. Requests are idempotent."""
        where = "%s:%d" % self.address
        deadline = time.monotonic() + (self._retry_s if retry_s is None else retry_s)
        pause = 0.5
        while True:
            try:
                if self._socket is None:
                    self._socket = socket.create_connection(self.address, timeout=self._timeout_s)
                    self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                _send(self._socket, header, blobs)
                return _receive(self._socket)
            except socket.timeout as error:
                self._disconnect()
                raise ServiceError(f"SAM 3 service at {where} did not answer within {self._timeout_s:.0f} s") from error
            except ValueError as error:
                self._disconnect()
                raise ServiceError(f"SAM 3 service at {where}: {error}") from error
            except OSError as error:
                self._disconnect()
                if time.monotonic() + pause > deadline:
                    raise ServiceError(f"SAM 3 service at {where} unreachable: {error}") from error
                time.sleep(pause)
                pause = min(2 * pause, 5.0)


# -- service side ---------------------------------------------------------------------------------------
class GpuThread:
    """The one thread that runs SAM 3, taking requests in arrival order. With cudnn.benchmark on (cuRobo's import turns it on in every
    policy server) cuDNN chooses convolution algorithms per thread, and two choices can differ in the last bits. A policy server runs
    SAM 3 in its one agent thread; one thread here likewise keeps one set of choices for every session."""

    def __init__(self):
        self._jobs = queue.Queue()
        threading.Thread(target=self._loop, name="sam3-gpu", daemon=True).start()

    def _loop(self):
        while True:
            job = self._jobs.get()
            job["began"] = time.monotonic()
            try:
                job["value"] = job["function"]()
            except BaseException as error:   # handed back to the request's thread
                job["error"] = error
            job["ended"] = time.monotonic()
            job["done"].set()

    def run(self, function) -> dict:
        job = {"function": function, "done": threading.Event()}
        self._jobs.put(job)
        job["done"].wait()
        return job


class Service:
    """Sessions (one cached encoding each) around one Segmenter."""

    def __init__(self, model, checkpoint, log_path, max_sessions: int = 24, idle_s: float = 6 * 3600):
        self.model = model
        self.checkpoint = file_identity(checkpoint)
        self._checkpoint_sha256 = None
        self.code = code_digest()
        self.settings = _segmenter.numerics()
        self.settings_digest = settings_digest(self.settings)
        self.gpu = GpuThread()
        self.sessions = collections.OrderedDict()   # id -> [slot, last use, connection]
        self._table = threading.Lock()
        self.max_sessions, self.idle_s = int(max_sessions), float(idle_s)
        self._log = open(log_path, "a", encoding="utf-8", buffering=1) if log_path else None
        self._log_lock = threading.Lock()
        self.counts = collections.Counter()
        self.started = time.time()

    def note(self, row: dict):
        with self._log_lock:
            if "op" in row:
                self.counts[row["op"] + ("_failed" if row.get("error") else "")] += 1
            if self._log is not None:
                self._log.write(json.dumps({"t": round(time.time(), 3), **row}, default=str) + "\n")

    def memory(self) -> dict:
        try:
            import torch
            if torch.cuda.is_available():
                return {"allocated_mib": round(torch.cuda.memory_allocated() / 2 ** 20), "reserved_mib": round(torch.cuda.memory_reserved() / 2 ** 20),
                        "max_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20)}
        except Exception:
            pass
        return {}

    def _slot(self, session: str, connection) -> Slot:
        now = time.monotonic()
        with self._table:
            for stale in [sid for sid, row in self.sessions.items() if sid != session and now - row[1] > self.idle_s]:
                del self.sessions[stale]
                self.note({"event": "expired", "session": stale})
            row = self.sessions.get(session)
            if row is None:
                row = self.sessions[session] = [Slot(), now, connection]
                while len(self.sessions) > self.max_sessions:
                    evicted, _ = self.sessions.popitem(last=False)
                    self.note({"event": "evicted", "session": evicted})
            row[1], row[2] = now, connection
            self.sessions.move_to_end(session)
            return row[0]

    def disconnected(self, connection):
        with self._table:
            gone = [sid for sid, row in self.sessions.items() if row[2] is connection]
            for sid in gone:
                del self.sessions[sid]
        if gone:
            self.note({"event": "disconnected", "sessions": gone})

    def handle(self, header: dict, blobs, connection, received: float):
        op = header.get("op")
        if op == "hello":
            self.note({"event": "hello", "label": header.get("label")})
            return {"ok": True, "protocol": PROTOCOL, "code": self.code, "checkpoint": self.checkpoint, "numerics": self.settings,
                    "pid": os.getpid(), "started": self.started}, []
        if op == "checkpoint_sha256":
            if self._checkpoint_sha256 is None:
                self._checkpoint_sha256 = _digest(self.checkpoint["path"])
            return {"ok": True, "sha256": self._checkpoint_sha256}, []
        if op == "stats":
            with self._table:
                sessions = len(self.sessions)
            return {"ok": True, "sessions": sessions, "counts": dict(self.counts), "memory": self.memory(), "pid": os.getpid(),
                    "uptime_s": round(time.time() - self.started, 1)}, []
        if op == "close":
            with self._table:
                found = self.sessions.pop(header.get("session"), None) is not None
            self.note({"event": "closed", "session": header.get("session"), "found": found})
            return {"ok": True}, []
        if op not in ("segment", "segment_point"):
            raise ValueError("unknown request: " + str(op))
        session = str(header["session"])
        row = {"op": op, "session": session, "label": header.get("label"), "key": header.get("key")}
        if header.get("numerics") != self.settings_digest:
            row["error"] = "torch settings differ"
            self.note(row)
            raise ServiceError("this process's torch settings differ from the service's; restart the client or the service")
        image = _image(header, blobs)
        slot = self._slot(session, connection)

        def call():
            before = slot.state
            try:
                if op == "segment":
                    return self.model.segment(image, header["text"], key=header.get("key"), min_score=header["min_score"], limit=header["limit"], slot=slot)
                return self.model.segment_point(image, header["pixel"], key=header.get("key"), slot=slot)
            finally:
                row["encoded"] = slot.state is not before

        job = self.gpu.run(call)
        row.update(queue_s=round(job["began"] - received, 4), compute_s=round(job["ended"] - job["began"], 4))
        if "error" in job:
            error = job["error"]
            row["error"] = f"{type(error).__name__}: {error}"
            self.note(row)
            raise error
        rows = job["value"]
        row["instances"] = len(rows)
        self.note(row)
        reply = {"ok": True, "instances": [{"score": item.score, "box": list(item.box), "area": item.area} for item in rows],
                 "timing": {"queue_s": row["queue_s"], "compute_s": row["compute_s"]}}
        if not rows:
            return reply, []
        masks = np.stack([item.mask for item in rows]).astype(bool, copy=False)
        reply["masks"] = {"shape": list(masks.shape)}
        return reply, [np.packbits(masks, axis=None)]


def _failure(error: Exception) -> dict:
    return {"ok": False, "error": {"type": type(error).__name__, "builtin": type(error).__module__ == "builtins", "message": str(error)}}


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        service, connection = self.server.service, object()
        try:
            self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            while True:
                try:
                    header, blobs = _receive(self.request)
                except (OSError, ValueError):
                    return
                received = time.monotonic()
                try:
                    reply, out = service.handle(header, blobs, connection, received)
                except Exception as error:
                    reply, out = _failure(error), []
                _send(self.request, reply, out)
        except OSError:
            return
        finally:
            service.disconnected(connection)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64


def serve(service: Service, port: int, host: str = "127.0.0.1") -> _Server:
    """A started server for an existing Service (tests); main() is the deployment entry."""
    server = _Server((parse_address(f"{host}:{port}")[0], port), _Handler)
    server.service = service
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared SAM 3 for the policy servers on this host (loopback only).")
    parser.add_argument("--checkpoint", required=True, help="the sam3.pt the policy servers would load")
    parser.add_argument("--port", type=int, default=18300)
    parser.add_argument("--log", required=True, help="JSON lines: one per request, with queue_s and compute_s")
    parser.add_argument("--max-sessions", type=int, default=24, help="cached encodings kept; the least recently used goes first")
    parser.add_argument("--idle-s", type=float, default=6 * 3600, help="a session unused this long is dropped")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = _Server(parse_address(f"127.0.0.1:{args.port}"), _Handler)   # claim the port before the model loads
    began = time.time()
    policy_server_settings()
    from .segmenter import Segmenter
    model = Segmenter(args.checkpoint)
    service = Service(model, args.checkpoint, args.log, args.max_sessions, args.idle_s)
    server.service = service
    ready = {"event": "ready", "pid": os.getpid(), "port": args.port, "load_s": round(time.time() - began, 1), "checkpoint": service.checkpoint["path"],
             "memory": service.memory(), "numerics": service.settings}
    service.note(ready)
    print(json.dumps(ready), flush=True)

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        service.note({"event": "stopped", "counts": dict(service.counts)})


if __name__ == "__main__":
    main()
