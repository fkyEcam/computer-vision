"""Generate the synthetic industrial dataset "parts" used in Labs 7 and 8 (Day 2).

Top views (64x64, grayscale) of a machined flange with a central bore and 4 bolt holes,
on a dark conveyor belt. Four classes:
    ok            : good part
    scratch       : a thin bright scratch across the part
    missing_hole  : one of the 4 bolt holes was not drilled
    stain         : a dark oil stain on the part

The split mimics a classic industrial pitfall: the TRAINING images were all acquired in
one session (part almost always in the same orientation, constant lighting), whereas the
VALIDATION and TEST images come from the production line (any orientation, varying
lighting, small offsets). Data augmentation is therefore essential (Lab 7).

Usage:  python make_parts_dataset.py     -> ../notebooks/data/parts/{train,val,test}/<class>/*.png
"""
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["ok", "scratch", "missing_hole", "stain"]
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "data" / "parts"
S = 128  # rendering size (downsampled to 64 for anti-aliasing)
SPLITS = {  # n per class, rotation range (deg), brightness range, max shift (px at 128)
    "train": (200, (-20, 20), (0.95, 1.05), 4),
    "val": (80, (0, 360), (0.65, 1.35), 10),
    "test": (80, (0, 360), (0.65, 1.35), 10),
}


def render(cls, rng, rot_range, gain_range, max_shift):
    img = np.full((S, S), 35, np.float32)
    img += cv2.GaussianBlur(rng.normal(0, 8, (S, S)).astype(np.float32), (0, 0), 2)  # belt texture
    c = np.array([S / 2, S / 2]) + rng.uniform(-max_shift, max_shift, 2)
    R = rng.uniform(40, 44)
    angle = np.radians(rng.uniform(*rot_range))
    # part body with a radial shading (machined metal)
    yy, xx = np.mgrid[0:S, 0:S]
    r = np.hypot(xx - c[0], yy - c[1])
    body = r < R
    img[body] = 165 + 25 * np.cos(r[body] / R * 3) + rng.normal(0, 3)
    # flat on one side (makes the orientation visible)
    n = np.array([np.cos(angle), np.sin(angle)])
    flat = ((xx - c[0]) * n[0] + (yy - c[1]) * n[1]) > R * 0.82
    img[body & flat] = 35
    # central bore
    cv2.circle(img, tuple(np.round(c).astype(int)), int(R * 0.33), 20, -1, cv2.LINE_AA)
    # 4 bolt holes
    missing = rng.integers(4) if cls == "missing_hole" else -1
    for k in range(4):
        if k == missing:
            continue
        a = angle + np.pi / 4 + k * np.pi / 2
        p = c + 0.66 * R * np.array([np.cos(a), np.sin(a)])
        cv2.circle(img, tuple(np.round(p).astype(int)), int(R * 0.14), 20, -1, cv2.LINE_AA)
    if cls == "scratch":
        a = rng.uniform(0, np.pi)
        d = np.array([np.cos(a), np.sin(a)]) * R * rng.uniform(0.5, 0.8)
        o = c + rng.uniform(-R * 0.3, R * 0.3, 2)
        p1, p2 = np.round(o - d).astype(int), np.round(o + d).astype(int)
        cv2.line(img, tuple(p1), tuple(p2), 250, int(rng.integers(2, 4)), cv2.LINE_AA)
    if cls == "stain":
        p = c + rng.uniform(0.3, 0.7) * R * np.array([np.cos(rng.uniform(0, 6.3)), np.sin(rng.uniform(0, 6.3))])
        blob = np.zeros((S, S), np.float32)
        for _ in range(4):
            q = p + rng.normal(0, 3, 2)
            cv2.circle(blob, tuple(np.round(q).astype(int)), int(rng.uniform(5, 9)), 1.0, -1, cv2.LINE_AA)
        blob = cv2.GaussianBlur(blob, (0, 0), 2.5) * body
        img = img * (1 - 0.75 * blob)
    gain = rng.uniform(*gain_range)
    # lighting gradient (production line only: larger gain range means stronger gradient)
    g = (gain_range[1] - gain_range[0]) * 0.5
    grad = 1 + g * ((xx - S / 2) / S * np.cos(angle * 3) + (yy - S / 2) / S * np.sin(angle * 2))
    img = img * gain * grad
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 4, img.shape)
    img = cv2.resize(np.clip(img, 0, 255), (64, 64), interpolation=cv2.INTER_AREA)
    return img.astype(np.uint8)


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    for split, (n, rot, gain, shift) in SPLITS.items():
        for cls in CLASSES:
            d = OUT / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                cv2.imwrite(str(d / f"{cls}_{i:03d}.png"), render(cls, rng, rot, gain, shift))
        print(split, n * len(CLASSES), "images")
    print("written to", OUT)
