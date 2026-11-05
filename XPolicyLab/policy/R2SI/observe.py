"""Images for the agent to look at: enlarged crops with grids and marks, object outlines, and sheets of sampled frames over time."""
from __future__ import annotations

import base64
import io
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

GRID_PX = 20
CROP_SIZE = (640, 480)
MAX_REGIONS = 8
MAX_FRAMES = 8
WATCH_MAX_STEPS = 250
PIXEL_DELTA = 30            # a pixel changed when some RGB channel moved by more than this (0-255)
ACTIVITY = 0.003            # fraction of changed pixels between consecutive samples that counts as activity
CHANGED = 0.005             # fraction of changed pixels between the final and the starting crop
SHEET_TILE = (320, 240)
OUTLINE_COLORS = ((0, 255, 255), (255, 255, 0), (255, 0, 255), (0, 255, 0), (255, 128, 0), (128, 128, 255))


def jpeg_part(image, label: str | None = None) -> list:
    """Message parts of one RGB array as a JPEG data URI."""
    array = np.asarray(image, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=95)
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    parts = [{"type": "text", "text": label}] if label else []
    return parts + [{"type": "image_url", "image_url": {"url": url}}]


def _box(region, width, height):
    if not isinstance(region, (list, tuple)) or len(region) != 4:
        raise ValueError("region is normalized [left, top, right, bottom]")
    left, top, right, bottom = (float(v) for v in region)
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError("region is normalized [left, top, right, bottom] in 0..1")
    return math.floor(left * width), math.floor(top * height), math.ceil(right * width), math.ceil(bottom * height)


def annotated(image, region=None, grid=False, points=None, outlines=None, outline=None) -> tuple[np.ndarray, dict]:
    """An enlarged, marked copy of an image and how its pixels map to full-image pixels. outlines is [(label, mask)] on the full image, outline a closed line through full-image pixels."""
    array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    box = _box(region, width, height) if region is not None else (0, 0, width, height)
    if outlines:
        array = array.copy()
        for index, (_, mask) in enumerate(outlines):
            rim = mask & ~ndimage.binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
            array[rim] = OUTLINE_COLORS[index % len(OUTLINE_COLORS)]
    crop = Image.fromarray(array).crop(box)
    scale = min(CROP_SIZE[0] / crop.width, CROP_SIZE[1] / crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))).convert("RGB")
    sx, sy = (box[2] - box[0]) / crop.width, (box[3] - box[1]) / crop.height
    draw = ImageDraw.Draw(crop)
    if grid:
        spacing = GRID_PX
        while spacing / sx < 24:
            spacing *= 2   # keep lines at least 24 crop pixels apart
        for u in range(int(math.ceil(box[0] / spacing)) * spacing, box[2] + 1, spacing):
            x = (u - box[0]) / sx
            draw.line((x, 0, x, crop.height), fill=(0, 255, 255), width=1)
            draw.text((x + 2, 2), str(u), fill=(0, 255, 255))
        for v in range(int(math.ceil(box[1] / spacing)) * spacing, box[3] + 1, spacing):
            y = (v - box[1]) / sy
            draw.line((0, y, crop.width, y), fill=(0, 255, 255), width=1)
            draw.text((2, y + 2), str(v), fill=(0, 255, 255))

    def label(x, y, text):
        left, top, right, bottom = draw.textbbox((x + 9, y - 17), text)
        draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=(0, 0, 0))
        draw.text((x + 9, y - 17), text, fill=(255, 255, 255))

    for index, (u, v) in enumerate(points or [], start=1):
        x, y = (float(u) - box[0]) / sx, (float(v) - box[1]) / sy
        if 0 <= x <= crop.width and 0 <= y <= crop.height:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(255, 0, 255), width=2)
            label(x, y, str(index))
    if outline is not None:
        corners = [((float(u) - box[0]) / sx, (float(v) - box[1]) / sy) for u, v in outline]
        draw.line(corners + corners[:1], fill=(255, 128, 0), width=2)
    for name, mask in outlines or []:
        ys, xs = np.nonzero(mask)
        if len(xs):
            x, y = (float(xs.mean()) - box[0]) / sx, (float(ys.min()) - box[1]) / sy
            if 0 <= x <= crop.width and 0 <= y <= crop.height:
                label(x, y, str(name))
    return np.asarray(crop), {"source_size": [width, height], "crop_size": list(crop.size), "offset_xy": list(box[:2]), "scale_xy": [round(sx, 4), round(sy, 4)]}


def rotated(picture, degrees: float) -> np.ndarray:
    """A picture turned counterclockwise, for reading what lies at an angle; its pixels no longer map to the camera's."""
    return np.asarray(Image.fromarray(np.asarray(picture, dtype=np.uint8)).rotate(degrees, resample=Image.BICUBIC, expand=True))


def _difference(a, b) -> float:
    """Fraction of pixels where some channel changed by more than PIXEL_DELTA: local motion such as one arm crossing a large region
    is not averaged away."""
    return float((np.abs(a - b).max(axis=-1) > PIXEL_DELTA).mean())


class Watcher:
    """Samples one camera over a window of hold steps and reports where and when the image changed."""

    def __init__(self, camera, image, steps, every, regions, quiet):
        if type(steps) is not int or not 1 <= steps <= WATCH_MAX_STEPS:
            raise ValueError(f"steps must be an integer in 1..{WATCH_MAX_STEPS}")
        if type(every) is not int or not 1 <= every <= min(50, steps):
            raise ValueError("every must be an integer in 1..min(50, steps)")
        if quiet is not None and (type(quiet) is not int or quiet < 1):
            raise ValueError("stop_after_quiet must be a positive integer")
        regions = regions or []
        if not isinstance(regions, list) or len(regions) > MAX_REGIONS:
            raise ValueError(f"regions is a list of at most {MAX_REGIONS} named boxes")
        names = [r.get("name") if isinstance(r, dict) else None for r in regions]
        if any(not isinstance(n, str) or not n or len(n) > 40 for n in names) or len(set(names)) != len(names):
            raise ValueError("every region needs a distinct name of 1-40 characters")
        self.camera, self.steps, self.every, self.quiet = camera, steps, every, quiet
        height, width = np.asarray(image).shape[:2]
        self.regions = [{"name": r["name"], "box": _box(r["region"], width, height), "region": list(r["region"])} for r in regions]
        self.samples = []
        self.start_crops = None
        self.previous = None
        self.activity = {r["name"]: [] for r in self.regions}
        self.peak = {r["name"]: 0.0 for r in self.regions}
        self.frame_activity = []
        self.quiet_run = 0
        self.seen_activity = False
        self.capture(image, 0)

    def _crops(self, image):
        return {r["name"]: np.asarray(image[r["box"][1]:r["box"][3], r["box"][0]:r["box"][2]], dtype=np.int16) for r in self.regions}

    def capture(self, image, offset):
        image = np.asarray(image)
        thumb = Image.fromarray(image.astype(np.uint8)).resize(SHEET_TILE)
        crops, whole = self._crops(image), np.asarray(thumb, dtype=np.int16)
        if self.start_crops is None:
            self.start_crops = crops
        else:
            moving = False
            for r in self.regions:
                value = _difference(crops[r["name"]], self.previous[0][r["name"]])
                self.peak[r["name"]] = max(self.peak[r["name"]], value)
                if value >= ACTIVITY:
                    self.activity[r["name"]].append(offset)
                    moving = True
            if _difference(whole, self.previous[1]) >= ACTIVITY:
                self.frame_activity.append(offset)
                moving = True
            self.seen_activity = self.seen_activity or moving
            self.quiet_run = 0 if moving else self.quiet_run + 1
        self.previous = (crops, whole, image)
        self.samples.append((offset, thumb))

    def next_hold(self, offset) -> int:
        """How many steps to hold next; 0 when the watch is over."""
        if offset >= self.steps or (self.quiet is not None and self.seen_activity and self.quiet_run >= self.quiet):
            return 0
        return min(self.every, self.steps - offset)

    def _intervals(self, offsets):
        spans = []
        for offset in offsets:
            if spans and offset - spans[-1][1] <= self.every:
                spans[-1][1] = offset
            else:
                spans.append([offset, offset])
        return spans

    def sheet(self) -> list:
        offsets = [s[0] for s in self.samples]
        keep = {offsets[0], offsets[-1]}
        ends = sorted({s[1] for spans in (self._intervals(v) for v in [*self.activity.values(), self.frame_activity]) for s in spans})
        for end in ends:   # the first sample after each burst of activity
            later = [o for o in offsets if o > end]
            keep.add(later[0] if later else offsets[-1])
        keep = sorted(keep)
        if len(keep) > MAX_FRAMES:
            keep = [keep[i] for i in sorted(set(np.linspace(0, len(keep) - 1, MAX_FRAMES).round().astype(int)))]
        frames, parts = dict(self.samples), [{"type": "text", "text": f"Sampled {self.camera} frames from the watch (+N = native steps since it started; boxes are the named regions):"}]
        sx, sy = SHEET_TILE[0] / self.previous[2].shape[1], SHEET_TILE[1] / self.previous[2].shape[0]
        for offset in keep:
            tile = frames[offset].copy()
            draw = ImageDraw.Draw(tile)
            for r in self.regions:
                left, top, right, bottom = r["box"]
                draw.rectangle([left * sx, top * sy, right * sx, bottom * sy], outline=(255, 255, 0))
                draw.text((left * sx + 2, top * sy + 1), r["name"], fill=(255, 255, 0))
            draw.text((4, 4), f"+{offset} steps", fill=(255, 255, 255))
            parts += jpeg_part(np.asarray(tile))
        return parts

    def report(self, stage) -> dict:
        final = self.previous[0]
        regions = []
        for r in self.regions:
            spans = self._intervals(self.activity[r["name"]])
            regions.append({"name": r["name"], "first_activity": spans[0][0] if spans else None, "last_activity": spans[-1][1] if spans else None, "activity": spans,
                            "peak_change": round(self.peak[r["name"]], 3), "differs_from_start": _difference(final[r["name"]], self.start_crops[r["name"]]) >= CHANGED})
        return {"ok": True, "ended": stage, "camera": self.camera, "samples": len(self.samples), "regions": regions, "frame_activity": self._intervals(self.frame_activity)}
