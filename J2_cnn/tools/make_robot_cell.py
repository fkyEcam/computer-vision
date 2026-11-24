"""Generate the robot-cell scenes used in Lab 8, Part 4 (Day 2): sorting flanges with the robot arm.

Same simulated wide-angle USB camera, work table, ArUco markers and robot (Galatae Gal-e) as in
Lab 4 of Day 1 (J1_vision_classique/tools/make_data.py). On the table lie printed "tiles" of
60 x 60 mm: each tile is an image of the flange dataset (tools/make_parts_dataset.py) scaled up,
i.e. a flange on its dark conveyor background. In the real cell, the sheet parts_sheet_A4.png is
printed at 100 %, glued on cardboard and cut into tiles.

Outputs (../notebooks/data/robot/):
    cell_parts_{1,2,3}.png  camera images, 5 tiles each (any rotation, classes mixed)
    cell_camera.json        intrinsics + distortion of the cell camera (result of Lab 4)
    taught_markers.json     marker centres taught to the robot (same file as in Lab 4)
    ground_truth.json       class and centre (table mm, robot mm) of every tile
    parts_sheet_A4.png      12 tiles (3 per class) to print at 100 % for the live session
    aruco_markers_A4.png    the 4 markers (same sheet as in Lab 4)

Usage:  python make_robot_cell.py
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "J1_vision_classique" / "tools"))
from make_data import (DIST, H, K, MARKERS, OUT as J1_DATA, TABLE_POSE, W, photometric,  # noqa: E402
                       render_plane, rot, table_texture, table_to_robot)
from make_parts_dataset import CLASSES, render  # noqa: E402

OUT = HERE.parent / "notebooks" / "data" / "robot"
TILE_MM = 60.0
rng = np.random.default_rng(88)


def tile_image(cls, size_px):
    """A flange of the given class on its dark background (test conditions), size_px x size_px."""
    img = render(cls, rng, (0, 360), (0.85, 1.15), 6)
    return cv2.resize(img, (size_px, size_px), interpolation=cv2.INTER_CUBIC)


def paste_tile(tex, ppm, origin, centre_mm, angle_deg, tile):
    """Paste a square grey tile into the BGR table texture, rotated by angle_deg."""
    s = tile.shape[0]
    cx, cy = (centre_mm[0] - origin[0]) * ppm, (centre_mm[1] - origin[1]) * ppm
    M = cv2.getRotationMatrix2D((s / 2, s / 2), angle_deg, 1.0)
    M[:, 2] += (cx - s / 2, cy - s / 2)
    size = (tex.shape[1], tex.shape[0])
    warped = cv2.warpAffine(tile.astype(np.float32), M, size, flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.ones((s, s), np.float32), M, size, flags=cv2.INTER_LINEAR)
    tex[:] = tex * (1 - mask[..., None]) + warped[..., None] * mask[..., None]


def random_layout(n=5):
    """n tile centres inside the markers, at least 90 mm apart, away from the markers."""
    pts = []
    while len(pts) < n:
        p = rng.uniform([60, 55], [340, 225])
        if all(np.hypot(*(p - q)) > 90 for q in pts) and \
                all(np.hypot(*(p - np.array(m))) > 75 for m in MARKERS.values()):
            pts.append(p)
    return [tuple(np.round(p, 1)) for p in pts]


def make_scenes():
    R = rot(TABLE_POSE["rx"], TABLE_POSE["ry"], TABLE_POSE["rz"])
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    C = np.array(TABLE_POSE["look_at_mm"]) - TABLE_POSE["distance_mm"] * axis
    t = -R @ C
    bg = np.full((H, W, 3), 60, np.float32)
    class_sets = [["ok", "scratch", "ok", "missing_hole", "stain"],
                  ["stain", "ok", "missing_hole", "scratch", "ok"],
                  ["missing_hole", "scratch", "ok", "stain", "ok"]]
    scenes = []
    for k, classes in enumerate(class_sets, start=1):
        tex, ppm, origin = table_texture(parts=[])
        tiles = []
        for cls, c in zip(classes, random_layout(len(classes))):
            angle = float(rng.uniform(-30, 30))
            paste_tile(tex, ppm, origin, c, angle, tile_image(cls, int(TILE_MM * ppm)))
            tiles.append({"class": cls, "centre_table_mm": [float(c[0]), float(c[1])], "angle_deg": round(angle, 1),
                          "centre_robot_mm": table_to_robot(c).round(2).tolist()})
        img = photometric(render_plane(tex, ppm, origin, R, t, bg), gain=rng.uniform(0.9, 1.1))
        name = f"cell_parts_{k}.png"
        cv2.imwrite(str(OUT / name), img)
        scenes.append({"file": name, "tiles": tiles})
    return scenes


def make_parts_sheet(path, px_per_mm=10):
    """A4, 3 x 4 tiles of 60 mm (3 per class), with cutting marks and a small class label."""
    from PIL import Image

    sheet = np.full((297 * px_per_mm, 210 * px_per_mm), 255, np.uint8)
    side = int(TILE_MM * px_per_mm)
    order = [c for _ in range(3) for c in CLASSES]
    for k, cls in enumerate(order):
        x0 = (12 + (k % 3) * 64) * px_per_mm
        y0 = (20 + (k // 3) * 66) * px_per_mm
        sheet[y0:y0 + side, x0:x0 + side] = tile_image(cls, side)
        cv2.rectangle(sheet, (x0 - 2, y0 - 2), (x0 + side + 1, y0 + side + 1), 160, 1)
        cv2.putText(sheet, f"{k + 1:02d} {cls}", (x0, y0 + side + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 140, 1,
                    cv2.LINE_AA)
    cv2.putText(sheet, "Print at 100 %, glue on cardboard, cut along the grey lines (tiles = 60 mm)",
                (60, 287 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    Image.fromarray(sheet).save(path, dpi=(25.4 * px_per_mm, 25.4 * px_per_mm))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = make_scenes()
    (OUT / "cell_camera.json").write_text(json.dumps(
        {"comment": "Calibration of the cell camera (Lab 4): K and distortion (k1, k2, p1, p2, k3).",
         "image_size": [W, H], "K": K.tolist(), "dist": DIST.tolist()}, indent=2))
    for f in ["taught_markers.json", "aruco_markers_A4.png"]:
        shutil.copy(J1_DATA / "robot" / f, OUT / f)
    (OUT / "ground_truth.json").write_text(json.dumps(
        {"comment": "Synthetic data generated by tools/make_robot_cell.py. Units: mm.",
         "tile_mm": TILE_MM, "marker_centres_mm": {str(k): v for k, v in MARKERS.items()},
         "scenes": scenes}, indent=2))
    make_parts_sheet(OUT / "parts_sheet_A4.png")
    print("written to", OUT)
