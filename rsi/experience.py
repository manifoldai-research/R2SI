"""Executed development experience, not proposals or official evaluation data."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid

from .storage import Workspace, canonical, sha256
from .official_layouts import MODE, validate_case, reject_open, uses_official_layouts

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
CHANNELS = ("left_arm_joint_state", "left_ee_joint_state",
            "right_arm_joint_state", "right_ee_joint_state")


def admit(meta, *, training=True):
    reject_open(meta)
    official = uses_official_layouts(meta)
    if official:
        validate_case(meta)
        if training:
            raise ValueError("Official public-layout rollouts are not model-training data")
        if meta.get("collection_authorized") is not True or meta.get("hidden_state_used") is not False:
            raise ValueError("Official-layout iteration needs explicit collection provenance")
    if not official and (meta.get("mode") != "development" or meta.get("split") != "development"
            or meta.get("category") not in ({"standard"} if training else {"standard", "open"})
            or meta.get("collection_authorized") is not True
            or meta.get("hidden_state_used") is not False):
        raise ValueError("Training accepts authorized standard development layouts only; "
                         "Open, hidden, official evaluation and unknown provenance are excluded")
    if not all(isinstance(meta.get(k), str) and meta[k] for k in
               ("episode_id", "task", "release_id", "layout_source")):
        raise ValueError("Incomplete experience provenance")
    if not meta["task"].replace("_", "").isalnum():
        raise ValueError("Invalid task data identifier")


class EpisodeRecorder:
    """Bounded incremental recording. A pre-action frame is added only after ack.

    The terminal post-action frame is separate from the T aligned state/actions.
    Failed/uncertain episodes keep complete reflection traces, separate from training data.
    """
    def __init__(self, workspace, metadata, max_bytes=2 * 1024**3, *, trainable=True):
        admit(metadata, training=False)
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        from .storage_preflight import check_storage
        check_storage(self.ws)
        purpose = metadata.get("record_purpose", "collection" if trainable else "validation")
        if purpose not in {"collection", "validation"}:
            raise ValueError("Unknown episode record purpose")
        self.meta = dict(metadata, record_purpose=purpose)
        self.trainable = (purpose == "collection" and trainable
                          and metadata["mode"] == "development" and metadata["category"] == "standard")
        self.identity = hashlib.sha256(metadata["episode_id"].encode()).hexdigest()
        # Exclusive identity reservation survives interruption; never replay it.
        claim = self.ws.path("experience/claims", self.identity)
        claim.parent.mkdir(parents=True, exist_ok=True)
        claim.mkdir(exist_ok=False)
        self.max_bytes = max_bytes
        self.count = self.bytes = 0
        self.closed = False
        self.start = time.time()
        self.path = self.ws.path("experience", ".pending", self.identity + "." + uuid.uuid4().hex + ".hdf5")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import h5py
        self.file = h5py.File(self.path, "x")
        self.file.attrs.update(provenance=canonical(self.meta).decode(),
                               success=False, recording_valid=False)

    def _append(self, name, value, *, encoded=False, text=False):
        import h5py
        import numpy as np
        array = np.asarray(value)
        if name not in self.file:
            if encoded:
                self.file.create_dataset(name, (0,), maxshape=(None,), dtype=h5py.vlen_dtype("uint8"))
            elif text:
                self.file.create_dataset(name, (0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
            else:
                self.file.create_dataset(name, (0, *array.shape), maxshape=(None, *array.shape),
                                         dtype=array.dtype, chunks=True)
        dataset = self.file[name]
        dataset.resize(len(dataset) + 1, axis=0)
        dataset[-1] = value

    def observation(self, observation):
        """Keep the latest post-action frame, also covering a zero-action failure."""
        import numpy as np
        from XPolicyLab.utils.process_data import encode_image_bit
        if "last_observation" in self.file:
            del self.file["last_observation"]
        group = self.file.create_group("last_observation")
        group.attrs["public"] = json.dumps(public_observation(observation), ensure_ascii=False)
        for camera in CAMERAS:
            group.create_dataset(camera, data=np.frombuffer(encode_image_bit(observation["vision"][camera]["color"]), dtype="uint8"))

    def events(self, events):
        for event in events:
            self._append("execution/tool_events", json.dumps(jsonable(event), ensure_ascii=False), text=True)

    def executed(self, before, action, after, *, sequence, acknowledged, feedback=None, events=()):
        import numpy as np
        from XPolicyLab.utils.process_data import encode_image_bit
        if self.closed or sequence != self.count or acknowledged is not True:
            raise ValueError("Only monotonically acknowledged actions may enter experience")
        if not after or not after.get("vision"):
            raise ValueError("Execution acknowledgement needs a fresh observation")
        images = {}
        for camera in CAMERAS:
            rgb = before["vision"][camera]["color"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError("Recorder accepts decoded RGB")
            images[camera] = np.frombuffer(encode_image_bit(rgb), dtype="uint8")
        self.bytes += sum(v.nbytes for v in images.values())
        if self.bytes > self.max_bytes:
            raise RuntimeError("Experience byte budget exceeded; stop this development trial")
        for channel in CHANNELS:
            for prefix, values in (("state", before["state"]), ("action", action)):
                array = np.asarray(values[channel], dtype=np.float32)
                if not np.isfinite(array).all():
                    raise ValueError("Non-finite executed data")
                self._append(prefix + "/" + channel + "s", array)
        for camera, encoded in images.items():
            self._append("vision/" + camera + "/colors", encoded, encoded=True)
        self._append("instruction", before["instruction"], text=True)
        self._append("execution/sequence", np.int64(sequence))
        self._append("execution/observation", json.dumps(public_observation(before), ensure_ascii=False), text=True)
        self._append("execution/feedback", json.dumps(jsonable(feedback or {}), ensure_ascii=False), text=True)
        self.events(events)
        self.observation(after)
        self.count += 1

    def finish(self, *, success, terminal_confirmed, reason="", score=None):
        if self.closed:
            raise RuntimeError("Recorder is already closed")
        valid = (self.trainable and success is True and terminal_confirmed is True and self.count > 0)
        self.closed = True
        self.file.attrs.update(success=bool(success), recording_valid=False, frames=self.count,
                               source_key=self.identity, terminal_confirmed=bool(terminal_confirmed))
        self.file.flush()
        self.file.close()
        record = {**self.meta, "actions": self.count, "success": success is True,
                  "recording_valid": valid, "terminal_confirmed": terminal_confirmed is True,
                  "elapsed_s": time.time() - self.start, "reason": reason, "completed_at": time.time()}
        # Private staging only. A native terminal result is not yet a completed
        # process: SimulationApp.close may still raise or exit nonzero.
        record.update(trace_path=str(self.path), score=score, recording_valid=False,
                      candidate_training_valid=valid, staged_trace=self.path.name,
                      staged_trace_sha256=sha256(self.path))
        self.ws.atomic("experience/staged/" + self.identity + ".json", record)
        return record

    def abort(self, reason):
        return self.finish(success=False, terminal_confirmed=False, reason=reason)


def jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def public_observation(obs):
    return {"instruction": obs["instruction"], "state": jsonable(obs["state"]),
            "vision": {k: {n: jsonable(v) for n, v in cam.items() if n != "color"}
                       for k, cam in obs["vision"].items()}}


def publish_snapshot(workspace, base_index, provenance):
    """Reuse the model team's immutable snapshots, preserving statistics/index."""
    from worldscape_policy.data.robodojo_snapshots import SnapshotStore
    ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
    if (provenance.get("split") != "development" or provenance.get("category") != "standard"
            or provenance.get("collection_authorized") is not True
            or provenance.get("open_layouts_excluded") is not True
            or provenance.get("official_evaluation_excluded") is not True
            or provenance.get("index_sha256") != sha256(base_index)):
        raise ValueError("Base index requires hash-bound development-only provenance")
    store = SnapshotStore(ws.path("datasets/snapshots"))
    store.initialize(base_index)
    from .experience_publication import accepted_roots
    record = store.publish(accepted_roots(ws))
    ws.atomic("datasets/base-provenance.json", provenance)
    return record
