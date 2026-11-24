"""Figures of the Day 2 slides (slides/figures/*.png).

1. Conceptual figures drawn with matplotlib (activation functions, ...).
2. Result figures EXTRACTED from the executed solution notebooks (training curves, filters,
   Grad-CAM...), so that the slides show exactly what the labs produce.

Usage:  python build_notebooks.py --execute   then   python make_figures.py
"""
import base64
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SOL = ROOT / "notebooks" / "solutions"
DATA = ROOT / "notebooks" / "data"
FIG = ROOT / "slides" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NAVY, RED = "#003366", "#D9534F"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})

# --- 1. extracted from the notebooks -------------------------------------------------------
EXTRACT = [  # notebook, substring identifying the code cell, output image index, file name
    ("J2_05_pytorch_mlp", "h_lin = fit(", 0, "lin_vs_mlp_curves.png"),
    ("J2_05_pytorch_mlp", "h_small = fit(", 0, "overfitting_curves.png"),
    ("J2_05_pytorch_mlp", "xb, yb = next(iter(train_loader))", 0, "fashion_samples.png"),
    ("J2_06_first_cnn", "acc_m = [accuracy_shifted", 0, "shift_robustness.png"),
    ("J2_06_first_cnn", "first-layer filters", 0, "learned_filters.png"),
    ("J2_06_first_cnn", "f1 = cnn.features", 0, "feature_maps.png"),
    ("J2_06_first_cnn", "conv.weight.data = torch.stack", 0, "conv_sobel.png"),
    ("J2_07_improve_cnn", "for row, (ds, name) in enumerate", 0, "parts_train_val.png"),
    ("J2_07_improve_cnn", "8 random augmentations", 0, "augmentations.png"),
    ("J2_07_improve_cnn", 'hist = fit(BaselineCNN(), eval_tfm)', 0, "curves_A.png"),
    ("J2_07_improve_cnn", 'hist = fit(BaselineCNN(), aug_tfm)', 0, "curves_B.png"),
    ("J2_07_improve_cnn", "plt.barh(list(results)", 0, "results_bar.png"),
    ("J2_07_improve_cnn", "np.add.at(cm", 0, "confusion.png"),
    ("J2_08_bug_hunt_gradcam", "cam, c = grad_cam(", 0, "gradcam.png"),
    ("J2_08_bug_hunt_gradcam", "for sigma in [0, 0.5", 0, "camera_vs_dataset.png"),
    ("J2_08_bug_hunt_gradcam", "top, tiles, plan = sort_cell(robot", 0, "robot_sorting.png"),
]


def extract():
    for nb_name, key, k, out in EXTRACT:
        nb = json.loads((SOL / f"{nb_name}_solution.ipynb").read_text(encoding="utf-8"))
        cells = [c for c in nb["cells"] if c["cell_type"] == "code" and key in "".join(c["source"])]
        assert cells, f"{nb_name}: no cell contains {key!r}"
        pngs = [o["data"]["image/png"] for o in cells[0].get("outputs", []) if "image/png" in o.get("data", {})]
        assert len(pngs) > k, f"{nb_name}: cell {key!r} has no image output {k} (execute the notebooks first)"
        (FIG / out).write_bytes(base64.b64decode(pngs[k]))
        print("extracted", out)


# --- 2. conceptual figures -------------------------------------------------------------------
def activations():
    x = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(1, 4, figsize=(14, 2.8))
    for a, (name, y) in zip(ax, [("sigmoid", 1 / (1 + np.exp(-x))), ("tanh", np.tanh(x)),
                                 ("ReLU", np.maximum(0, x)),
                                 ("GELU", 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))))]):
        a.plot(x, y, color=NAVY, lw=2.5); a.axhline(0, color="gray", lw=.5); a.axvline(0, color="gray", lw=.5)
        a.set_title(name)
    plt.tight_layout(); fig.savefig(FIG / "activations.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("figure activations.png")


def lr_schedules():
    steps = np.arange(100)
    one_cycle = np.where(steps < 30, 0.1 + 0.9 * steps / 30, 0.5 * (1 + np.cos(np.pi * (steps - 30) / 70)))
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(steps, np.ones(100), label="constant", color="gray")
    ax.plot(steps, 0.1 ** (steps // 35), label="step decay", color=NAVY)
    ax.plot(steps, one_cycle, label="one-cycle", color=RED, lw=2.5)
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate / max"); ax.legend()
    fig.savefig(FIG / "lr_schedules.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("figure lr_schedules.png")


def lr_effect():
    rng = np.random.default_rng(0)
    e = np.arange(40)
    curves = {"too high": 2.3 - 0.3 * (1 - np.exp(-e / 3)) + 0.25 * rng.random(40),
              "too low": 2.3 * np.exp(-e / 60),
              "good": 0.25 + 2.05 * np.exp(-e / 6)}
    fig, ax = plt.subplots(figsize=(5, 2.8))
    for (k, v), c in zip(curves.items(), [RED, "gray", NAVY]):
        ax.plot(e, v, label=k, color=c, lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax.legend()
    fig.savefig(FIG / "lr_effect.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("figure lr_effect.png")


def parts_classes():
    classes = ["ok", "scratch", "missing_hole", "stain"]
    fig, ax = plt.subplots(1, 4, figsize=(8, 2.4))
    for a, c in zip(ax, classes):
        f = sorted((DATA / "parts" / "val" / c).glob("*.png"))[3]
        a.imshow(cv2.imread(str(f), 0), cmap="gray"); a.set_title(c); a.axis("off")
    fig.savefig(FIG / "parts_classes.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    print("figure parts_classes.png")


if __name__ == "__main__":
    activations()
    lr_schedules()
    lr_effect()
    parts_classes()
    extract()
