"""Text-prompted instance segmentation of a single RGB image with SAM 3. The only input is the image and a noun phrase."""
from __future__ import annotations

import threading
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


class Segmenter:
    """One model per process, shared by all episodes; loaded once at start-up."""

    def __init__(self, checkpoint: str, device: str = "cuda"):
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self._torch = torch
        torch.backends.cuda.matmul.allow_tf32 = True
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, device=device, eval_mode=True,
                                       enable_inst_interactivity=True)
        self._processor = Sam3Processor(model, device=device, confidence_threshold=0.0)
        self._model = model
        self._lock = threading.Lock()
        self._key = None
        self._state = None

    def forget(self):
        """Drop the cached image: a new episode starts from nothing."""
        self._key = None

    def _encode(self, image: np.ndarray, key):
        from PIL import Image
        if key is None or key != self._key:
            self._state = self._processor.set_image(Image.fromarray(image))
            self._key = key
        return self._state

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

    def segment(self, image: np.ndarray, text: str, key=None, min_score: float = MIN_SCORE, limit: int = MAX_INSTANCES) -> list[Instance]:
        """Instances of a noun phrase in an image, best first. `key` identifies the image so its encoding is reused."""
        with self._lock, self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self._torch.bfloat16):
            state = self._encode(np.asarray(image, dtype=np.uint8), key)
            out = self._processor.set_text_prompt(prompt=text.strip(), state=state)
            return self._instances(out["masks"], out["scores"], min_score, limit)

    def segment_point(self, image: np.ndarray, pixel, key=None) -> list[Instance]:
        """The object under one pixel: candidate masks, best first."""
        with self._lock, self._torch.inference_mode(), self._torch.autocast("cuda", dtype=self._torch.bfloat16):
            state = self._encode(np.asarray(image, dtype=np.uint8), key)
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
