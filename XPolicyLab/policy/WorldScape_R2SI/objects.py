"""Objects in the world from instance masks and the public camera geometry.

One camera's mask fixes rays, not depth. A solid standing on a known height is found by asking which columns stay inside the mask
all the way up to a candidate top height: as that height grows the footprint shrinks from the side away from the camera. A second
camera decides the height: the prism of the right height projects onto that camera's mask, a wrong one spills over or falls short.
With a single camera the height is taken as the tallest prism that still explains the whole mask, which is weaker and is reported so.

The only inputs are the current RGB masks, the camera geometry, the robot's own pose (pixels on the robot say nothing about an object)
and the height of the surface the object stands on. No stored shape, size or colour.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, cKDTree

from . import frames

LEVEL_M = 0.002              # height resolution of the column scan
SEARCH_STEP_M = 0.006        # coarse step of the height search
MAX_HEIGHT_M = frames.ARM_TOP_M    # nothing the arms can work on is higher above a surface
ELONGATED_ASPECT = 1.4       # a footprint at least this long for its width has a direction
GRID_MIN_M = 0.002
GRID_CELLS = 150             # the footprint grid has at most this many cells per side
SAME_OBJECT_HEIGHT_M = 0.05  # tops this close over the same place are one object
MAX_MASK_SHARE = 0.45        # a larger region is background, not an object
ROBOT_SHARE = 0.6            # a mask mostly on the robot is the robot
MIN_AREA_PX = 12
AGREE_MIN = 0.6              # below this a second camera's instance is not the same object
TRUST_MIN = 0.8              # below this two cameras only roughly agree
LIFTED_TRY_BELOW = 0.75      # when the agreement stays below this the object may not stand on the given surface
LIFTED_STEP_M = 0.015
LIFTED_MAX_M = frames.ARM_TOP_M
LIFTED_GAIN = 0.1            # a raised bottom must explain this much better to be believed
CARVE_KEEP = 0.8             # a second camera may cut the footprint down to this share of it
SINGLE_COVERAGE = 0.93       # one camera: the tallest prism whose projection still covers this share of the mask
COVER_SCALE = 4              # masks are compared at this reduction: grid points do not cover every full-size pixel
SAME_OBJECT_M = 0.02         # two objects closer than this (or half their size) are the same object seen twice
ROBOT_MARGIN_PX = 2.0
REGISTER_POINTS = 300        # footprints are thinned to about this many points before they are compared
REGISTER_STEP_DEG = 6.0
REGISTER_ICP = 4
WORK_RADIUS_M = 1.2 * frames.ARM_REACH_M    # nothing further than this from both arm bases can matter: a mask out there is background


@dataclass
class Sight:
    """One camera's image of the scene at this moment, with what of it the robot covers."""
    view: object
    robot: np.ndarray
    instances: list


@dataclass
class Measured:
    text: str
    score: float
    x: float
    y: float
    top_z: float
    bottom_z: float
    length: float
    width: float
    long_side_rad: float
    body_length: float           # the sides of its outline at half its height: wider than length and width when it tapers upward
    body_width: float
    agreement: float | None
    cameras: list
    pixels: dict
    masks: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    cells: np.ndarray | None = None   # the footprint's grid cells in world x, y


def robot_silhouette(view, hulls) -> np.ndarray:
    """Pixels the robot covers in this view: the convex outline of each link's mesh."""
    width, height = view.size
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for points in hulls:
        pixels, depth = view.project(points)
        pixels = pixels[depth > 0.01]
        if len(pixels) < 3:
            continue
        try:
            outline = pixels[ConvexHull(pixels).vertices]
        except QhullError:
            continue
        draw.polygon([tuple(p) for p in outline], fill=255, outline=255)
    return ndimage.binary_dilation(np.asarray(image, dtype=bool), iterations=int(ROBOT_MARGIN_PX))


def _lookup(mask, pixels, depth):
    """(inside the mask, inside the image and in front of the camera) of projected points."""
    h, w = mask.shape
    u, v = np.rint(pixels[:, 0]).astype(np.int64), np.rint(pixels[:, 1]).astype(np.int64)
    seen = (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    inside = np.zeros(len(u), dtype=bool)
    inside[seen] = mask[v[seen], u[seen]]
    return inside, seen


def _grid(view, mask, bottom, top):
    """Footprint cells where a solid seen in this mask can stand: the region between the mask's rays at the two heights."""
    ys, xs = np.nonzero(mask)
    stride = max(1, len(xs) // 4000)
    pixels = np.stack([xs[::stride], ys[::stride]], axis=1)
    points = np.vstack([view.on_plane(pixels, bottom), view.on_plane(pixels, top)])[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    near = np.min([np.linalg.norm(points - origin[:2], axis=1) for origin in frames.BASE_ORIGIN.values()], axis=0) <= WORK_RADIUS_M
    if near.mean() < 0.5:
        return None
    points = points[near]
    low, high = points.min(axis=0) - 0.01, points.max(axis=0) + 0.01
    step = max(GRID_MIN_M, float(max(high - low)) / GRID_CELLS)
    gx, gy = np.arange(low[0], high[0] + step, step), np.arange(low[1], high[1] + step, step)
    xx, yy = np.meshgrid(gx, gy)
    return np.stack([xx.ravel(), yy.ravel()], axis=1), step, xx.shape


def _rise(view, mask, robot, xy, bottom, top):
    """For each column the highest level up to which it stays inside the mask or behind the robot (nan: never). A column that only ever
    falls on robot pixels says nothing about the object and is left out."""
    allowed = mask | robot
    levels = np.arange(bottom, top + LEVEL_M / 2, LEVEL_M)
    alive = np.arange(len(xy))
    rise = np.full(len(xy), np.nan)
    touched = np.zeros(len(xy), dtype=bool)
    for z in levels:
        pixels, depth = view.project(np.c_[xy[alive], np.full(len(alive), z)])
        inside, _ = _lookup(allowed, pixels, depth)
        on_mask, _ = _lookup(mask, pixels, depth)
        touched[alive[on_mask]] = True
        alive = alive[inside]
        if not len(alive):
            break
        rise[alive] = z
    rise[~touched] = np.nan
    return rise


def _solid_points(cells, bottom, top, limit=20000):
    layers = max(2, int(math.ceil((top - bottom) / 0.008)) + 1)
    stride = max(1, int(math.ceil(len(cells) * layers / limit)))
    cells = cells[::stride]
    zs = np.linspace(bottom, top, layers)
    return np.c_[np.repeat(cells, len(zs), axis=0), np.tile(zs, len(cells))]


def _reduced(mask, scale=COVER_SCALE):
    h, w = mask.shape
    return mask[:h - h % scale, :w - w % scale].reshape(h // scale, scale, w // scale, scale).any(axis=(1, 3))


def _agreement(view, mask, robot, cells, bottom, top) -> float | None:
    """F1 of a prism against another camera's mask: how much of the prism lands on the mask and how much of the mask it explains."""
    pixels, depth = view.project(_solid_points(cells, bottom, top))
    inside, seen = _lookup(mask, pixels, depth)
    on_robot, _ = _lookup(robot, pixels, depth)
    counted = seen & ~on_robot
    if counted.sum() < 8:
        return None
    precision = float(inside[counted].mean())
    cover = np.zeros((mask.shape[0] // COVER_SCALE, mask.shape[1] // COVER_SCALE), dtype=bool)
    u, v = (pixels[seen, 0] / COVER_SCALE).astype(np.int64), (pixels[seen, 1] / COVER_SCALE).astype(np.int64)
    keep = (u >= 0) & (u < cover.shape[1]) & (v >= 0) & (v < cover.shape[0])
    cover[v[keep], u[keep]] = True
    cover = ndimage.binary_dilation(cover, iterations=1)
    wanted = _reduced(mask & ~robot)
    if not wanted.any():
        return None
    recall = float(cover[wanted].mean())
    return 2 * precision * recall / max(precision + recall, 1e-9)


def _coverage(view, mask, cells, bottom, top) -> float:
    """Share of a mask explained by a prism, in the camera that made the mask."""
    pixels, depth = view.project(_solid_points(cells, bottom, top))
    cover = np.zeros((mask.shape[0] // COVER_SCALE, mask.shape[1] // COVER_SCALE), dtype=bool)
    keep = depth > 0
    u, v = (pixels[keep, 0] / COVER_SCALE).astype(np.int64), (pixels[keep, 1] / COVER_SCALE).astype(np.int64)
    inside = (u >= 0) & (u < cover.shape[1]) & (v >= 0) & (v < cover.shape[0])
    cover[v[inside], u[inside]] = True
    cover = ndimage.binary_dilation(cover, iterations=1)
    wanted = _reduced(mask)
    return float(cover[wanted].mean()) if wanted.any() else 0.0


def min_rectangle(points):
    """Smallest enclosing rectangle of N x 2 points: centre, length >= width, direction of the long side in radians
    (in (-pi/2, pi/2], from +x toward +y) and the four corners. None for fewer than three points."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return None
    try:
        hull = points[ConvexHull(points).vertices]
    except QhullError:
        hull = points
    edges = np.roll(hull, -1, axis=0) - hull
    best = None
    for angle in np.unique(np.round(np.arctan2(edges[:, 1], edges[:, 0]) % (math.pi / 2), 6)):
        c, s = math.cos(angle), math.sin(angle)
        a, b = hull @ np.array([c, s]), hull @ np.array([-s, c])
        lengths = a.max() - a.min(), b.max() - b.min()
        area = lengths[0] * lengths[1]
        if best is None or area < best[0] - 1e-12:
            best = (area, angle, lengths, np.array([(a.max() + a.min()) / 2, (b.max() + b.min()) / 2]))
    _, angle, lengths, middle = best
    c, s = math.cos(angle), math.sin(angle)
    axes = np.array([[c, s], [-s, c]])
    center = middle @ axes
    long_side = angle if lengths[0] >= lengths[1] else angle + math.pi / 2
    long_side = (long_side + math.pi / 2) % math.pi - math.pi / 2
    if long_side <= -math.pi / 2 + 1e-9:
        long_side += math.pi
    half = np.array(lengths) / 2
    corners = np.array([center + sx * half[0] * axes[0] + sy * half[1] * axes[1] for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])
    return {"center": center, "length": float(max(lengths)), "width": float(min(lengths)), "angle": float(long_side), "corners": corners}


def _largest_blob(chosen, shape):
    labels, count = ndimage.label(chosen.reshape(shape))
    if count <= 1:
        return chosen
    sizes = ndimage.sum(np.ones(shape), labels, index=np.arange(1, count + 1))
    return (labels == 1 + int(np.argmax(sizes))).ravel()


def _usable(instance, robot, image_area):
    if instance.area < MIN_AREA_PX or instance.area > MAX_MASK_SHARE * image_area:
        return False
    return float((instance.mask & robot).sum()) <= ROBOT_SHARE * instance.area


def _best_for_height(xy, rise, others, bottom, top):
    """For a prism family on one bottom: (best top, mean agreement, {camera: instance index}), searching coarse then fine."""
    def score(height):
        chosen = rise >= height - LEVEL_M / 2
        if chosen.sum() < 4:
            return None
        cells = xy[chosen]
        total, rows = 0.0, {}
        for camera, sight in others.items():
            best = (None, None)
            for index, instance in enumerate(sight.instances):
                value = _agreement(sight.view, instance.mask, sight.robot, cells, bottom, height)
                if value is not None and (best[0] is None or value > best[0]):
                    best = (value, index)
            if best[0] is not None and best[0] >= AGREE_MIN:
                total += best[0]
                rows[camera] = (best[1], best[0])
        return (total / len(rows), rows) if rows else None

    finite = rise[np.isfinite(rise)]
    if not len(finite):
        return None
    highest = float(finite.max())
    found = None
    for height in np.arange(bottom + SEARCH_STEP_M, highest + LEVEL_M, SEARCH_STEP_M):
        row = score(height)
        if row is not None and (found is None or row[0] > found[1]):
            found = (height, row[0], row[1])
    if found is None:
        return None
    for height in np.arange(max(bottom + LEVEL_M, found[0] - SEARCH_STEP_M), found[0] + SEARCH_STEP_M + LEVEL_M / 2, LEVEL_M):
        row = score(height)
        if row is not None and row[0] > found[1]:
            found = (height, row[0], row[1])
    return found


def measure(primary: Sight, instance, others: dict, support_z: float, text: str) -> Measured | None:
    """Where one instance of the primary camera stands, using the other cameras' instances."""
    view = primary.view
    top_limit = min(support_z + MAX_HEIGHT_M, float(view.origin[2]) - 0.03)
    grid = _grid(view, instance.mask & ~primary.robot if (instance.mask & ~primary.robot).any() else instance.mask, support_z, top_limit)
    if grid is None:
        return None
    xy, _, shape = grid
    rise = _rise(view, instance.mask, primary.robot, xy, support_z, top_limit)
    bottom = support_z
    notes = []
    found = _best_for_height(xy, rise, others, bottom, top_limit) if others else None
    if others and (found is None or found[1] < LIFTED_TRY_BELOW):
        best = found
        for raised in np.arange(support_z + LIFTED_STEP_M, support_z + LIFTED_MAX_M, LIFTED_STEP_M):
            lifted = _rise(view, instance.mask, primary.robot, xy, float(raised), top_limit)
            row = _best_for_height(xy, lifted, others, float(raised), top_limit)
            if row is not None and (best is None or row[1] > best[1] + (LIFTED_GAIN if best is found else 0.0)):
                best = row
                bottom, rise = float(raised), lifted
        if best is not found:
            notes.append(f"it seems not to stand on z={support_z:.3f}: its bottom was found near z={bottom:.3f}")
        found = best
    if found is not None:
        height, agreement, matched = found
        chosen = _largest_blob(rise >= height - LEVEL_M / 2, shape)
        cameras = [view.camera] + list(matched)
        masks = {view.camera: instance.mask}
        carved = chosen.copy()
        for camera, (index, _) in matched.items():
            sight = others[camera]
            masks[camera] = sight.instances[index].mask
            pixels, depth = sight.view.project(np.c_[xy, np.full(len(xy), height)])
            inside, seen = _lookup(sight.instances[index].mask | sight.robot, pixels, depth)
            carved &= inside | ~seen
        if carved.sum() >= CARVE_KEEP * chosen.sum():
            chosen = carved
        if agreement < TRUST_MIN:
            notes.append(f"the cameras only roughly agree ({agreement:.2f}): the height and centre may be off by a few centimetres")
        middle = bottom + 0.5 * (height - bottom)
        body = _largest_blob(rise >= middle - LEVEL_M / 2, shape)
        for camera, (index, _) in matched.items():
            sight = others[camera]
            pixels, depth = sight.view.project(np.c_[xy, np.full(len(xy), middle)])
            inside, seen = _lookup(sight.instances[index].mask | sight.robot, pixels, depth)
            body &= inside | ~seen
    else:
        body = None
        agreement, cameras, masks = None, [view.camera], {view.camera: instance.mask}
        finite = rise[np.isfinite(rise)]
        if not len(finite):
            return None
        height = bottom
        radial = xy[np.isfinite(rise)].mean(axis=0) - view.origin[:2]
        radial = radial / max(float(np.linalg.norm(radial)), 1e-6)
        across = np.array([-radial[1], radial[0]])
        for candidate in np.arange(bottom + SEARCH_STEP_M, float(finite.max()) + LEVEL_M, SEARCH_STEP_M):
            cells = rise >= candidate - LEVEL_M / 2
            if cells.sum() < 4 or _coverage(view, instance.mask & ~primary.robot, xy[cells], bottom, candidate) < SINGLE_COVERAGE:
                break
            height = float(candidate)
            # As the height grows the footprint shortens away from the camera. Without a second view, take it to be about as deep as it is wide.
            along_view, sideways = xy[cells] @ radial, xy[cells] @ across
            if along_view.max() - along_view.min() <= sideways.max() - sideways.min():
                break
        chosen = _largest_blob(rise >= height - LEVEL_M / 2, shape)
        notes.append("seen by one camera only: its height is a guess and its centre can be off away from the camera by half the height error; "
                     "look from a second camera, a wrist camera above it is best")
    rectangle = min_rectangle(xy[chosen])
    if rectangle is None:
        return None
    outline = None if body is None else min_rectangle(xy[body])
    body_length, body_width = (rectangle["length"], rectangle["width"]) if outline is None else (max(rectangle["length"], outline["length"]), max(rectangle["width"], outline["width"]))
    touches = bool(instance.box[0] <= 1 or instance.box[1] <= 1 or instance.box[2] >= view.size[0] - 1 or instance.box[3] >= view.size[1] - 1)
    if touches:
        notes.append(f"it reaches the edge of the {view.camera} image, so it may extend beyond what is seen")
    if float((instance.mask & primary.robot).sum()) > 0.15 * instance.area:
        notes.append(f"the robot covers part of it in {view.camera}")
    return Measured(text=text, score=instance.score, x=float(rectangle["center"][0]), y=float(rectangle["center"][1]), top_z=float(height), bottom_z=float(bottom),
                    length=rectangle["length"], width=rectangle["width"], long_side_rad=rectangle["angle"], body_length=body_length, body_width=body_width, agreement=agreement,
                    cameras=cameras, pixels={}, masks=masks, notes=notes, cells=xy[chosen])


def locate(sights: dict, text: str, support_z: float = frames.TABLE_Z) -> list[Measured]:
    """Every instance of a phrase seen in the given sights, as measured objects, best score first."""
    usable = {}
    for camera, sight in sights.items():
        image_area = sight.view.size[0] * sight.view.size[1]
        usable[camera] = Sight(sight.view, sight.robot, [i for i in sight.instances if _usable(i, sight.robot, image_area)])
    results, used = [], {camera: set() for camera in usable}
    # The wider view first: it holds most of the scene; then whatever only a closer camera sees.
    order = sorted(usable, key=lambda c: 0 if c == "cam_head" else 1)
    for camera in order:
        sight = usable[camera]
        for index, instance in sorted(enumerate(sight.instances), key=lambda row: -row[1].score):
            if index in used[camera]:
                continue
            others = {}
            for other, other_sight in usable.items():
                if other == camera:
                    continue
                free = [i for k, i in enumerate(other_sight.instances) if k not in used[other]]
                if free:
                    others[other] = Sight(other_sight.view, other_sight.robot, free)
            found = measure(sight, instance, others, support_z, text)
            if found is None:
                continue
            for other, mask in found.masks.items():
                if other != camera:
                    for k, candidate in enumerate(usable[other].instances):
                        if candidate.mask is mask:
                            used[other].add(k)
            if any(math.hypot(found.x - r.x, found.y - r.y) < max(SAME_OBJECT_M, 0.5 * min(r.width, found.width)) and abs(found.top_z - r.top_z) < SAME_OBJECT_HEIGHT_M for r in results):
                continue
            used[camera].add(index)
            for name, mask in found.masks.items():
                ys, xs = np.nonzero(mask)
                found.pixels[name] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
            results.append(found)
    return sorted(results, key=lambda r: -r.score)


def register(moving, fixed, symmetry=1) -> dict:
    """The planar motion that lays one footprint on another: turn about the moving footprint's centre, then shift. Footprints are N x 2 world
    points. Searches all turns, so it does not depend on a starting guess; the best other turn tells whether the shape leaves it ambiguous."""
    a, b = np.asarray(moving, dtype=np.float64), np.asarray(fixed, dtype=np.float64)
    a = a[:: max(1, len(a) // REGISTER_POINTS)]
    b = b[:: max(1, len(b) // REGISTER_POINTS)]
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    tree_b = cKDTree(b)

    def fit(degrees, shift):
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        turned = (a - ca) @ np.array([[c, s], [-s, c]])
        for _ in range(REGISTER_ICP):
            near = b[tree_b.query(turned + cb + shift)[1]]
            shift = shift + (near - (turned + cb + shift)).mean(axis=0)
        forward = tree_b.query(turned + cb + shift)[0]
        backward = cKDTree(turned + cb + shift).query(b)[0]
        return float(np.sqrt((np.mean(forward ** 2) + np.mean(backward ** 2)) / 2)), shift

    rows = []
    for degrees in np.arange(0.0, 360.0 / max(int(symmetry), 1), REGISTER_STEP_DEG):
        residual, shift = fit(float(degrees), np.zeros(2))
        rows.append((residual, float(degrees), shift))
    rows.sort(key=lambda row: row[0])
    best = rows[0]
    for degrees in np.arange(best[1] - REGISTER_STEP_DEG, best[1] + REGISTER_STEP_DEG + 0.25, 0.5):
        residual, shift = fit(float(degrees), best[2])
        if residual < best[0]:
            best = (residual, float(degrees), shift)
    other = next((row for row in rows if abs((row[1] - best[1] + 180.0) % 360.0 - 180.0) > 30.0), None)
    turn = (best[1] + 180.0) % 360.0 - 180.0
    return {"turn_deg": round(turn, 1), "shift": [round(float(v), 4) for v in (cb + best[2] - ca)], "residual_mm": round(best[0] * 1000, 1),
            "next_best_turn_deg": None if other is None else round((other[1] + 180.0) % 360.0 - 180.0, 1),
            "next_best_residual_mm": None if other is None else round(other[0] * 1000, 1)}


def refine_on_plane(view, instances, robot, expected_xy, z, max_move: float):
    """Where the object near an expected point lies on the plane z in this view: centre offset and rectangle, or None."""
    expected_pixel, depth = view.project(np.r_[expected_xy, z][None])
    if depth[0] <= 0:
        return None
    u, v = expected_pixel[0]
    best, best_gap = None, None
    reach = max_move * float(view.k[0, 0]) / float(depth[0])
    for instance in instances:
        if instance.area < MIN_AREA_PX or float((instance.mask & robot).sum()) > ROBOT_SHARE * instance.area:
            continue
        ys, xs = np.nonzero(instance.mask)
        gap = float(np.min(np.hypot(xs - u, ys - v)))
        if gap <= reach and (best is None or gap < best_gap):
            best, best_gap = instance, gap
    if best is None:
        return None
    w, h = view.size
    if best.box[0] <= 1 or best.box[1] <= 1 or best.box[2] >= w - 1 or best.box[3] >= h - 1:
        return None
    rim = best.mask & ~ndimage.binary_erosion(best.mask, structure=np.ones((3, 3), dtype=bool))
    ys, xs = np.nonzero(rim)
    points = view.on_plane(np.stack([xs, ys], axis=1), z)[:, :2]
    points = points[np.isfinite(points).all(axis=1)]
    rectangle = min_rectangle(points)
    if rectangle is None:
        return None
    delta = rectangle["center"] - np.asarray(expected_xy, dtype=np.float64)
    if rectangle["length"] >= ELONGATED_ASPECT * rectangle["width"]:
        # Along an elongated object the grasp may be anywhere the caller chose; only its position across the width is corrected.
        across = np.array([-math.sin(rectangle["angle"]), math.cos(rectangle["angle"])])
        delta = float(delta @ across) * across
    if float(np.linalg.norm(delta)) > max_move:
        return None
    return {"delta": delta, "length": rectangle["length"], "width": rectangle["width"], "long_side_rad": rectangle["angle"]}
