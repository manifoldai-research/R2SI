"""The shared SAM 3 service's protocol, sessions and checks, with a stand-in model on the CPU (no torch, no GPU)."""
import json
import threading
from unittest.mock import patch

import numpy as np
import pytest

from XPolicyLab.policy.WorldScape_R2SI import sam3_service, segmenter
from XPolicyLab.policy.WorldScape_R2SI.segmenter import Instance, Slot

SETTINGS = {"torch": "test", "matmul_tf32": True}


class StandIn:
    """Segmenter's slot semantics: an image is encoded when its key changes; results come from the encoded image."""

    def __init__(self):
        self.encodes = 0
        self._slot = Slot()

    def _encode(self, image, key, slot):
        if key is None or key != slot.key:
            slot.state = {"image": np.asarray(image, dtype=np.uint8).copy()}
            slot.key = key
            self.encodes += 1
        return slot.state

    def segment(self, image, text, key=None, min_score=0.25, limit=12, slot=None):
        if text.strip() == "boom":
            raise ValueError("cannot read that phrase")
        picture = self._encode(image, key, self._slot if slot is None else slot)["image"]
        rows = []
        for level in range(min(int(limit), 3)):
            mask = picture[..., level % picture.shape[-1]] > 40 * (level + 1) + len(text)
            if mask.any():
                ys, xs = np.nonzero(mask)
                rows.append(Instance(score=float(np.float32(0.9 - 0.1 * level + min_score / 7)), mask=mask,
                                     box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), area=int(mask.sum())))
        return rows

    def segment_point(self, image, pixel, key=None, slot=None):
        picture = self._encode(image, key, self._slot if slot is None else slot)["image"]
        mask = picture[..., 1] >= picture[int(round(pixel[1])), int(round(pixel[0])), 1]
        ys, xs = np.nonzero(mask)
        return [Instance(score=0.75, mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), area=int(mask.sum()))]


@pytest.fixture
def running(tmp_path):
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"weights")
    with patch.object(segmenter, "prepare_process"), patch.object(segmenter, "numerics", return_value=dict(SETTINGS)):
        model = StandIn()
        service = sam3_service.Service(model, checkpoint, tmp_path / "requests.jsonl")
        server = sam3_service.serve(service, 0)
        address = "127.0.0.1:%d" % server.server_address[1]
        yield model, service, address, checkpoint, tmp_path / "requests.jsonl"
        server.shutdown()
        server.server_close()


def image(seed, height=31, width=45):
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def same(rows, expected):
    assert len(rows) == len(expected)
    for got, want in zip(rows, expected):
        assert type(got.score) is float and got.score == want.score
        assert got.box == want.box and type(got.box) is tuple and got.area == want.area
        assert got.mask.dtype == bool and got.mask.shape == want.mask.shape and np.array_equal(got.mask, want.mask)


def test_results_equal_a_local_model_bit_for_bit(running):
    model, _, address, checkpoint, _ = running
    client = sam3_service.ServiceSegmenter(address, checkpoint)
    local = StandIn()
    for step, picture in enumerate([image(1), image(2, 17, 9)]):
        for text, min_score in (("block", 0.25), ("red cup ", 0.05), ("block", 0.25)):
            same(client.segment(picture, text, key=(np.int64(step), "cam_head"), min_score=min_score),
                 local.segment(picture, text, key=(step, "cam_head"), min_score=min_score))
        same(client.segment_point(picture, np.array([3.4, 5.6]), key=(step, "cam_head")), local.segment_point(picture, [3.4, 5.6], key=(step, "cam_head")))
    assert model.encodes == local.encodes == 2
    client.close()


def test_sessions_never_share_an_encoding(running):
    model, service, address, checkpoint, log = running
    first, second = (sam3_service.ServiceSegmenter(address, checkpoint) for _ in range(2))
    a, b = image(3), image(4)
    key = (0, "cam_head")
    expected_a, expected_b = StandIn().segment(a, "block", key=key), StandIn().segment(b, "block", key=key)
    same(first.segment(a, "block", key=key), expected_a)
    same(second.segment(b, "block", key=key), expected_b)        # same key, other session: its own image
    same(first.segment(b, "block", key=key), expected_a)         # same key, same session: the cached image, as locally
    first.forget()                                               # a new episode encodes again
    same(first.segment(b, "block", key=key), expected_b)
    assert model.encodes == 3
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    calls = [row for row in rows if row.get("op") == "segment"]
    assert len(calls) == 4 and all(row["queue_s"] >= 0 and row["compute_s"] >= 0 for row in calls)
    assert [row["encoded"] for row in calls] == [True, True, False, True]
    assert any(row.get("event") == "closed" and row["found"] for row in rows)
    second.close()
    first.close()


def test_concurrent_clients_get_their_own_results(running):
    _, _, address, checkpoint, _ = running
    failures = []

    def work(seed):
        try:
            client = sam3_service.ServiceSegmenter(address, checkpoint)
            local = StandIn()
            for step in range(6):
                picture = image(seed * 100 + step)
                same(client.segment(picture, "block", key=(step % 2, "cam_head")), local.segment(picture, "block", key=(step % 2, "cam_head")))
            client.close()
        except Exception as error:   # reported below
            failures.append(repr(error))

    threads = [threading.Thread(target=work, args=(seed,)) for seed in range(6)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]
    assert failures == []


def test_errors_keep_their_type_and_disconnect_drops_sessions(running):
    _, service, address, checkpoint, _ = running
    client = sam3_service.ServiceSegmenter(address, checkpoint)
    with pytest.raises(ValueError, match="cannot read"):
        client.segment(image(5), "boom", key=(0, "cam_head"))
    client.segment(image(5), "block", key=(1, "cam_head"))
    assert client.session in service.sessions
    client._disconnect()                     # the process went away without closing its session
    for _ in range(100):
        if not service.sessions:
            break
        threading.Event().wait(0.02)
    assert not service.sessions


def test_refuses_a_service_that_would_compute_differently(running, tmp_path):
    _, service, address, checkpoint, _ = running
    with patch.object(segmenter, "prepare_process"), patch.object(segmenter, "numerics", return_value={**SETTINGS, "matmul_tf32": False}):
        with pytest.raises(sam3_service.ServiceError, match="torch settings differ"):
            sam3_service.ServiceSegmenter(address, checkpoint)
    other = tmp_path / "other.pt"
    other.write_bytes(b"other weights")
    with patch.object(segmenter, "prepare_process"), patch.object(segmenter, "numerics", return_value=dict(SETTINGS)):
        with pytest.raises(sam3_service.ServiceError, match="checkpoint"):
            sam3_service.ServiceSegmenter(address, other)
        copy = tmp_path / "copy.pt"
        copy.write_bytes(checkpoint.read_bytes())   # another file with the same bytes is the same checkpoint
        sam3_service.ServiceSegmenter(address, copy).close()
        with patch.object(sam3_service, "code_digest", return_value="0" * 64):
            with pytest.raises(sam3_service.ServiceError, match="segmenter.py"):
                sam3_service.ServiceSegmenter(address, checkpoint)


def test_a_setting_changed_after_start_up_fails_the_request(running):
    _, _, address, checkpoint, _ = running
    client = sam3_service.ServiceSegmenter(address, checkpoint)
    with patch.object(segmenter, "numerics", return_value={**SETTINGS, "matmul_tf32": False}):
        with pytest.raises(sam3_service.ServiceError, match="torch settings differ"):
            client.segment(image(6), "block", key=(0, "cam_head"))
    assert client.segment(image(6), "block", key=(0, "cam_head"))
    client.close()


def test_evicted_sessions_encode_again_with_the_same_results(running):
    model, service, address, checkpoint, _ = running
    service.max_sessions = 1
    first, second = (sam3_service.ServiceSegmenter(address, checkpoint) for _ in range(2))
    a, b = image(7), image(8)
    key = (0, "cam_head")
    for _ in range(2):
        same(first.segment(a, "block", key=key), StandIn().segment(a, "block", key=key))
        same(second.segment(b, "block", key=key), StandIn().segment(b, "block", key=key))
    assert model.encodes == 4 and len(service.sessions) == 1
    first.close()
    second.close()


def test_only_loopback_addresses():
    with pytest.raises(ValueError, match="loopback"):
        sam3_service.parse_address("10.0.0.5:18300")
    assert sam3_service.parse_address("localhost:18300") == ("127.0.0.1", 18300)
