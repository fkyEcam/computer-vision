"""Generate the YOLO dataset "table_yolo" used in Lab 10 (Day 3).

Scenes of the work table of Day 1, rendered with the SAME simulated wide-angle USB camera
(J1_vision_classique/tools/make_data.py: intrinsics, distortion, ArUco markers), with 3-7 parts
per image at random positions. Three classes, defined by SHAPE (colour varies):
    0 disc    : coloured disc (red, blue, green, yellow, orange)
    1 plate   : coloured rectangular plate, any orientation
    2 washer  : grey metallic ring -- unsaturated, invisible to the HSV detector of Day 1!
Parts may touch each other (another failure case of the contour-based detector).

Labels are written in the Ultralytics *segmentation* format (class + normalised polygon), which
also works for detection (boxes are derived from the polygons). For every validation image the
true positions in mm are stored in ground_truth.json, to measure the full pixel -> mm pipeline.

Usage:  python make_yolo_dataset.py      -> ../notebooks/data/table_yolo/ (+ table_yolo.zip for Colab)
"""
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

J1_TOOLS = Path(__file__).resolve().parents[2] / "J1_vision_classique" / "tools"
sys.path.insert(0, str(J1_TOOLS))
from make_data import DIST, H, K, MARKER_SIZE, MARKERS, W, photometric, render_plane, rot  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "data" / "table_yolo"
CLASSES = ["disc", "plate", "washer"]
COLOURS = {"red": (40, 40, 200), "blue": (190, 90, 30), "green": (60, 160, 50),
           "yellow": (40, 200, 230), "orange": (30, 120, 235)}
rng = np.random.default_rng(14)
PPM = 4
X0, Y0, X1, Y1 = -350.0, -300.0, 750.0, 600.0


def base_texture():
    w, h = int((X1 - X0) * PPM), int((Y1 - Y0) * PPM)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    tex = np.dstack([175 + 10 * noise, 182 + 10 * noise, 188 + 10 * noise])
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (cx, cy) in MARKERS.items():
        m = cv2.aruco.generateImageMarker(dictionary, mid, int(MARKER_SIZE * PPM))
        b = int(8 * PPM)
        m = cv2.copyMakeBorder(m, b, b, b, b, cv2.BORDER_CONSTANT, value=255)
        u, v = to_px(cx, cy)
        hs = m.shape[0] // 2
        tex[v - hs:v - hs + m.shape[0], u - hs:u - hs + m.shape[1]] = m[..., None] * 0.92 + 8
    return tex


def to_px(x, y):
    return int(round((x - X0) * PPM)), int(round((y - Y0) * PPM))


def random_parts():
    parts, n = [], rng.integers(3, 8)
    for _ in range(200):
        if len(parts) == n:
            break
        cls = rng.choice(3, p=[0.35, 0.35, 0.30])
        if cls == 0:
            size = (rng.uniform(14, 24),)
            radius = size[0]
        elif cls == 1:
            size = (rng.uniform(40, 70), rng.uniform(20, 32))
            radius = 0.5 * np.hypot(*size)
        else:
            size = (rng.uniform(15, 22),)
            radius = size[0]
        c = np.array([rng.uniform(25, 375), rng.uniform(25, 255)])
        # no overlap, but touching is allowed (gap 0-2 mm)
        if all(np.linalg.norm(c - p["c"]) > radius + p["r"] + rng.uniform(-0.5, 2) for p in parts) \
                and all(np.linalg.norm(c - np.array(m)) > radius + 35 for m in MARKERS.values()):
            parts.append({"cls": int(cls), "c": c, "r": radius, "size": size,
                          "angle": float(rng.uniform(0, 180)),
                          "colour": list(COLOURS)[rng.integers(len(COLOURS))]})
    return parts


def outline_mm(p, n=48):
    """Polygon of the part outline in table coordinates (mm)."""
    if CLASSES[p["cls"]] == "plate":
        box = cv2.boxPoints(((p["c"][0], p["c"][1]), p["size"], p["angle"]))
        # densify the edges so that the distortion bends them correctly
        return np.concatenate([np.linspace(box[k], box[(k + 1) % 4], 12, endpoint=False) for k in range(4)])
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return p["c"] + p["size"][0] * np.c_[np.cos(a), np.sin(a)]


def draw_parts(tex, parts):
    for p in parts:
        name = CLASSES[p["cls"]]
        if name == "disc":
            cv2.circle(tex, to_px(*p["c"]), int(p["size"][0] * PPM), COLOURS[p["colour"]], -1, cv2.LINE_AA)
        elif name == "plate":
            box = cv2.boxPoints((((p["c"][0] - X0) * PPM, (p["c"][1] - Y0) * PPM),
                                 (p["size"][0] * PPM, p["size"][1] * PPM), p["angle"]))
            cv2.fillPoly(tex, [np.round(box).astype(np.int32)], COLOURS[p["colour"]], cv2.LINE_AA)
        else:  # washer: grey metal ring with a radial shading
            u, v = to_px(*p["c"])
            R = int(p["size"][0] * PPM)
            yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
            r = np.hypot(xx, yy)
            ring = (r <= R) & (r >= 0.45 * R)
            grey = 120 + 50 * np.cos(r / R * 4 + p["angle"])
            sub = tex[v - R:v + R + 1, u - R:u + R + 1]
            sub[ring] = np.stack([grey[ring] * 0.97, grey[ring], grey[ring] * 1.02], -1)
    return tex


def render_scene(parts):
    tex = draw_parts(base_texture(), parts)
    R = rot(-10 + rng.uniform(-4, 4), 7 + rng.uniform(-4, 4), 2 + rng.uniform(-5, 5))
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    C = np.array([200.0 + rng.uniform(-20, 20), 140.0 + rng.uniform(-15, 15), 0.0]) - rng.uniform(430, 480) * axis
    t = -R @ C
    img = render_plane(tex.astype(np.float32), PPM, (X0, Y0), R, t, np.full((H, W, 3), 60, np.float32))
    # random lighting gradient + exposure
    yy, xx = np.mgrid[0:H, 0:W]
    g = rng.uniform(0, 0.4)
    ang = rng.uniform(0, 2 * np.pi)
    grad = 1 + g * ((xx / W - 0.5) * np.cos(ang) + (yy / H - 0.5) * np.sin(ang))
    img = photometric(img * grad[..., None], gain=rng.uniform(0.7, 1.2))
    rvec, _ = cv2.Rodrigues(R)
    labels = []
    for p in parts:
        poly = outline_mm(p)
        uv, _ = cv2.projectPoints(np.c_[poly, np.zeros(len(poly))], rvec, t, K, DIST)
        uv = uv.reshape(-1, 2)
        uv[:, 0] = np.clip(uv[:, 0], 0, W - 1) / W
        uv[:, 1] = np.clip(uv[:, 1], 0, H - 1) / H
        labels.append(f"{p['cls']} " + " ".join(f"{x:.5f}" for x in uv.ravel()))
    return img, labels


if __name__ == "__main__":
    if OUT.exists():
        shutil.rmtree(OUT)
    truth = {}
    for split, n in [("train", 160), ("val", 40)]:
        (OUT / "images" / split).mkdir(parents=True)
        (OUT / "labels" / split).mkdir(parents=True)
        for i in range(n):
            parts = random_parts()
            img, labels = render_scene(parts)
            name = f"{split}_{i:03d}"
            cv2.imwrite(str(OUT / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(labels) + "\n")
            if split == "val":
                truth[f"{name}.jpg"] = [{"class": CLASSES[p["cls"]], "x_mm": float(p["c"][0]), "y_mm": float(p["c"][1]),
                                         "angle_deg": p["angle"] if p["cls"] == 1 else None} for p in parts]
        print(split, n, "images")
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=1))
    (OUT / "classes.txt").write_text("\n".join(CLASSES) + "\n")
    # camera calibration of the simulated USB camera (= result of Lab 4), reused to convert to mm
    fs = cv2.FileStorage(str(OUT / "calibration.yaml"), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", W); fs.write("image_height", H); fs.write("K", K); fs.write("dist", DIST.reshape(1, -1))
    fs.release()
    shutil.make_archive(str(OUT), "zip", OUT.parent, OUT.name)
    print("written", OUT, "and", OUT.with_suffix(".zip"))
