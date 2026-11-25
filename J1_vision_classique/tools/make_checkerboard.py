"""Printable calibration checkerboard for Lab 4 (vector PDF, A4 landscape, true scale).

10 x 7 squares of 25 mm = 9 x 6 inner corners, as expected by capture_calibration.py and Lab 4.
Print at 100 % on matt paper, check the 100 mm ruler, and glue the sheet flat on a rigid board.

Usage:  python make_checkerboard.py      -> ../notebooks/data/checkerboard_9x6_25mm_A4.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

PAGE = (297.0, 210.0)  # A4 landscape, mm
COLS, ROWS, SQUARE = 10, 7, 25.0
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "data" / "checkerboard_9x6_25mm_A4.pdf"

fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4))
ax = fig.add_axes([0, 0, 1, 1])  # 1 data unit = 1 mm
ax.set(xlim=(0, PAGE[0]), ylim=(0, PAGE[1]))
ax.axis("off")

x0, y0 = (PAGE[0] - COLS * SQUARE) / 2, 21.0
for i in range(COLS):
    for j in range(ROWS):
        if (i + j) % 2 == 0:
            ax.add_patch(Rectangle((x0 + i * SQUARE, y0 + j * SQUARE), SQUARE, SQUARE, fc="black", ec="none"))

# 100 mm ruler and text, below the white margin of the board
ax.plot([x0, x0 + 100], [8, 8], color="0.4", lw=0.6, solid_capstyle="butt")
for k in range(0, 101, 10):
    ax.plot([x0 + k] * 2, [8, 10 if k % 50 else 11], color="0.4", lw=0.6, solid_capstyle="butt")
ax.text(x0 + 104, 8, "100 mm: check after printing (print at 100 %, do not fit to page)   |   "
        "9 x 6 inner corners, squares of 25 mm", fontsize=7, color="0.4", va="bottom")

fig.savefig(OUT)
print("written to", OUT)
