"""Text-prompted instance segmentation of a single RGB image with SAM 3. The only input is the image and a noun phrase."""
from __future__ import annotations

import os
import threading
import warnings
from dataclasses import dataclass

import numpy as np

MIN_SCORE = 0.25
MAX_INSTANCES = 12


@dataclass
class Instance:
    score: float
    mask: np.ndarray    # H x W bool
    box: tuple          # left, top, right, bottom in pixels
    area: int


class Slot:
    """The image an episode encoded last: its key and SAM 3 state, reused while the key repeats."""
    __slots__ = ("key", "state")

    def __init__(self):
        self.key = None
        self.state = None


def prepare_process():
    """What loading SAM 3 does to the whole process besides holding the model: importing sam3 turns on TF32 for matmuls and
    convolutions (sam3.model_builder) and sets TOKENIZERS_PARALLELISM, and the Segmenter turns on TF32 matmuls. A process that sends
    its calls to the shared service (sam3_service.py) runs this too, so its other GPU work (cuRobo, WAM) computes as next to a local model."""
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor  # noqa: F401
    from sam3.model_builder import build_sam3_image_model  # noqa: F401
    torch.backends.cuda.matmul.allow_tf32 = True


def numerics() -> dict:
    """Libraries, device and process-wide torch settings that decide SAM 3's arithmetic; a service and its clients must agree on them."""
    import torch
    import sam3

    def read(get):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return get()
        except Exception as error:
            return "unavailable: " + type(error).__name__

    cuda, cudnn = torch.backends.cuda, torch.backends.cudnn
    available = torch.cuda.is_available()
    return {
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": read(cudnn.version),
        "sam3": os.path.realpath(os.path.dirname(sam3.__file__)),
        "device": read(torch.cuda.get_device_name) if available else None,
        "capability": read(lambda: list(torch.cuda.get_device_capability())) if available else None,
        "matmul_tf32": read(lambda: cuda.matmul.allow_tf32), "cudnn_tf32": read(lambda: cudnn.allow_tf32),
        "matmul_precision": read(torch.get_float32_matmul_precision),
        "fp32_precision": {name: read(lambda owner=owner: owner.fp32_precision) for name, owner in
                           (("all", torch.backends), ("matmul", cuda.matmul), ("cudnn", cudnn),
                            ("conv", getattr(cudnn, "conv", None)), ("rnn", getattr(cudnn, "rnn", None)))},
        "bf16_reduction": read(lambda: cuda.matmul.allow_bf16_reduced_precision_reduction),
        "fp16_reduction": read(lambda: cuda.matmul.allow_fp16_reduced_precision_reduction),
        "cudnn_enabled": read(lambda: cudnn.enabled), "cudnn_benchmark": read(lambda: cudnn.benchmark),
        "cudnn_deterministic": read(lambda: cudnn.deterministic), "deterministic": read(torch.are_deterministic_algorithms_enabled),
        "sdp": [read(cuda.flash_sdp_enabled), read(cuda.mem_efficient_sdp_enabled), read(cuda.math_sdp_enabled),
                read(lambda: cuda.cudnn_sdp_enabled())],
        "blas": read(lambda: str(cuda.preferred_blas_library())),
        "default_dtype": str(torch.get_default_dtype()),
        "env": {name: os.environ.get(name) for name in ("CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")},
    }


class Segmenter:
    """One model per process, shared by all episodes; loaded once at start-up."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        import torch
        prepare_process()
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self._torch = torch
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, device=device, eval_mode=True,
                                       enable_inst_interactivity=True)
        self._processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
        self._model = model
        self._lock = threading.Lock()
        self._slot = Slot()

    def forget(self):
        """Drop the cached image: a new episode starts from nothing."""
        self._slot.key = None

    def _encode(self, image: np.ndarray, key, slot: Slot):
        from PIL import Image
        if key is None or key != slot.key:
            slot.state = self._processor.set_image(Image.fromarray(image))
            slot.key = key
        return slot.state

    def _instances(self, masks, scores, min_score: float, limit: int) -> list[Instance]:
        scores = scores.float().cpu().numpy().reshape(-1)
        masks = masks.float().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        rows = []
        for index in np.argsort(-scores)[:limit]:
            if scores[index] < min_score:
                break
            mask = masks[index] > 0.5
            area = int(mask.sum())
            if not area:
                continue
            ys, xs = np.nonzero(mask)
            rows.append(Instance(score=float(scores[index]), mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), area=area))
        return rows

    def segment(self, image: np.ndarray, text: str, key=None, min_score: float = MIN_SCORE, limit: int = MAX_INSTANCES, slot: Slot | None = None) -> list[Instance]:
        """Instances of a noun phrase in an image, best first. `key` identifies the image so its encoding is reused; the shared
        service passes each episode's own `slot`, a local model uses its one."""
        with self._lock, self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self._torch.bfloat16):
            state = self._encode(np.asarray(image, dtype=np.uint8), key, self._slot if slot is None else slot)
            out = self._processor.set_text_prompt(prompt=text.strip(), state=state)
            return self._instances(out["masks"], out["scores"], min_score, limit)

    def segment_point(self, image: np.ndarray, pixel, key=None, slot: Slot | None = None) -> list[Instance]:
        """The object under one pixel: candidate masks, best first."""
        with self._lock, self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self._torch.bfloat16):
            state = self._encode(np.asarray(image, dtype=np.uint8), key, self._slot if slot is None else slot)
            masks, scores, _ = self._model.predict_inst(state, point_coords=np.array([[float(pixel[0]), float(pixel[1])]]),
                                                        point_labels=np.array([1]), multimask_output=True)
        masks, scores = np.asarray(masks), np.asarray(scores).reshape(-1)
        rows = []
        for index in np.argsort(-scores):
            mask = masks[index] > 0
            if mask.any() and mask[int(round(float(pixel[1]))), int(round(float(pixel[0])))]:
                ys, xs = np.nonzero(mask)
                rows.append(Instance(score=float(scores[index]), mask=mask, box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), area=int(mask.sum())))
        return rows
