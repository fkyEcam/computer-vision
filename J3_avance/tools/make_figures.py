"""Figures of the Day 3 slides (slides/figures/*.png).

Result figures are EXTRACTED from the executed solution notebooks; the dataset figure is drawn here.
Usage:  python build_notebooks.py --execute   then   python make_figures.py
"""
import base64
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "notebooks" / "solutions"
DS = ROOT / "notebooks" / "data" / "table_yolo"
FIG = ROOT / "slides" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EXTRACT = [  # notebook, substring identifying the code cell, output image index, file name
    ("J3_09_transfer_learning", "plt.bar(x + (k - 1) * 0.27", 0, "tl_results.png"),
    ("J3_09_transfer_learning", "nn_idx = sims.argsort", 0, "tl_neighbours.png"),
    ("J3_10_yolo_detection_segmentation", "coco_model.predict", 0, "coco_prediction.png"),
    ("J3_10_yolo_detection_segmentation", "for a, f in zip(ax, val_images[:2])", 0, "yolo_predictions.png"),
    ("J3_10_yolo_detection_segmentation", "errors, missed = [], 0", 0, "mm_errors.png"),
    ("J3_10_yolo_detection_segmentation", "r = seg_model.predict(str(val_images[1])", 0, "seg_prediction.png"),
    ("J3_11_diffusion", "steps = [0, 50, 100", 0, "forward_noise.png"),
    ("J3_11_diffusion", "samples, snaps = sample(model)", 0, "ddpm_samples.png"),
]


def extract():
    for nb_name, key, k, out in EXTRACT:
        nb = json.loads((SOL / f"{nb_name}_solution.ipynb").read_text(encoding="utf-8"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code" and key in "".join(c["source"])]
        assert cells, f"{nb_name}: no cell contains {key!r}"
        pngs = [o["data"]["image/png"] for o in cells[0].get("outputs", []) if "image/png" in o.get("data", {})]
        assert len(pngs) > k, f"{nb_name}: cell {key!r} has no image output (execute the notebooks first)"
        (FIG / out).write_bytes(base64.b64decode(pngs[k]))
        print("extracted", out)


def dataset_figure():
    colours = [(0, 0, 255), (255, 0, 0), (0, 160, 0)]
    classes = (DS / "classes.txt").read_text().split()
    tiles = []
    for name in ["train_000", "train_001", "train_002", "train_003"]:
        img = cv2.imread(str(DS / "images" / "train" / f"{name}.jpg"))
        h, w = img.shape[:2]
        for line in (DS / "labels" / "train" / f"{name}.txt").read_text().splitlines():
            v = line.split()
            c = int(v[0])
            poly = (np.array(v[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
            cv2.polylines(img, [poly], True, colours[c], 3)
            x, y = poly.min(0)
            cv2.putText(img, classes[c], (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colours[c], 2)
        tiles.append(cv2.resize(img, (640, 360)))
    cv2.imwrite(str(FIG / "yolo_dataset.png"), np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    print("figure yolo_dataset.png")


if __name__ == "__main__":
    dataset_figure()
    extract()
