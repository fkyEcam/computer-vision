"""Cut-out sheets for the live robot cell of Lab 4, section 10 (PDF, 2 x A4 portrait, true scale).

Page 1: the 4 ArUco markers (ids 0-3, 40 mm) as 60 mm tiles, and where to put them on the table.
Page 2: circles (30 mm diameter) and squares (30 mm) in 4 colours, as 44 mm tiles.
Print at 100 % on matt paper, check the 100 mm ruler, cut along the DASHED lines (keep the white
border: the marker detection needs it, and the circles stay perfectly round).

Usage:  python make_robot_sheets.py      -> ../notebooks/data/robot/robot_cell_cutouts_A4.pdf
"""
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

PAGE = (210.0, 297.0)  # A4 portrait, mm
MARKER, MARKER_TILE = 40.0, 60.0
SHAPE, SHAPE_TILE = 30.0, 44.0
COLOURS = {"red": (200, 40, 40), "blue": (30, 90, 190), "green": (50, 160, 60), "yellow": (230, 200, 40)}  # RGB
LENGTH, DEPTH = 340, 260  # mm, between the marker centres
LAYOUT = {0: (0, 0), 1: (LENGTH, 0), 2: (LENGTH, DEPTH), 3: (0, DEPTH)}  # mm (MARKERS_LIVE of Labs 4 and 8)
ROBOT_TO_EDGE = 230  # mm, from the axis of the robot base to the line through markers 0 and 1
GREY = "0.45"
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "data" / "robot" / "robot_cell_cutouts_A4.pdf"


def new_page():
    fig = plt.figure(figsize=(PAGE[0] / 25.4, PAGE[1] / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])  # 1 data unit = 1 mm, origin at the bottom left
    ax.set(xlim=(0, PAGE[0]), ylim=(0, PAGE[1]))
    ax.axis("off")
    ax.plot([15, 115], [8, 8], color=GREY, lw=0.6, solid_capstyle="butt")
    for k in range(0, 101, 10):
        ax.plot([15 + k] * 2, [8, 10 if k % 50 else 11], color=GREY, lw=0.6, solid_capstyle="butt")
    ax.text(118, 8, "100 mm: print at 100 %, do not fit to page", fontsize=7, color=GREY, va="bottom")
    return fig, ax


def cut_line(ax, cx, cy, size):
    ax.add_patch(Rectangle((cx - size / 2, cy - size / 2), size, size, fc="none", ec=GREY, lw=0.5, ls=(0, (6, 4))))


def marker_tile(ax, mid, cx, cy):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    h = MARKER / 2
    ax.imshow(cv2.aruco.generateImageMarker(dictionary, mid, 600), cmap="gray", vmin=0, vmax=255,
              extent=(cx - h, cx + h, cy - h, cy + h), interpolation="none", aspect="auto", zorder=2)
    ax.set(xlim=(0, PAGE[0]), ylim=(0, PAGE[1]))
    cut_line(ax, cx, cy, MARKER_TILE)
    t = MARKER_TILE / 2
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:  # axes of the marker, to place it and to teach its centre
        ax.plot([cx + dx * (t - 2.5), cx + dx * t], [cy + dy * (t - 2.5), cy + dy * t], color=GREY, lw=0.6)
    ax.text(cx - t + 1.5, cy + t - 1.2, f"{mid}", fontsize=8, color=GREY, va="top")


def layout_diagram(ax, x0, y0, scale):
    """Top view of the table; (x0, y0) = page position of marker 0, Y of the table points down."""
    def p(x, y):
        return x0 + x * scale, y0 - y * scale

    ax.add_patch(Polygon([p(*LAYOUT[i]) for i in LAYOUT], fc="0.96", ec=GREY, lw=0.6, ls="--"))
    s = MARKER_TILE * scale
    for i, (x, y) in LAYOUT.items():
        u, v = p(x, y)
        ax.add_patch(Rectangle((u - s / 2, v - s / 2), s, s, fc="white", ec="black", lw=0.8, zorder=2))
        ax.text(u, v, str(i), ha="center", va="center", fontsize=10, weight="bold", zorder=3)
    mid = LENGTH / 2
    u, v = p(mid, -ROBOT_TO_EDGE)
    ax.add_patch(Circle((u, v), 55 * scale, fc="0.8", ec="black", lw=0.8))
    ax.text(u + 130 * scale, v, "robot base\nz up, origin 120 mm\nabove the table", ha="left", va="center", fontsize=7.5)
    for (dx, dy), name in (((0, 105), "x"), ((105, 0), "y")):  # robot frame (z towards the reader)
        ax.annotate("", p(mid + dx, -ROBOT_TO_EDGE + dy), p(mid, -ROBOT_TO_EDGE), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="#D9534F", lw=1.6, shrinkA=0, shrinkB=0))
        ax.text(*p(mid + dx * 1.02 + (10 if name == "x" else 4), -ROBOT_TO_EDGE + dy - (12 if name == "y" else 0)), name,
                fontsize=9, weight="bold", color="#D9534F", va="center", zorder=4)
    arrow = dict(arrowstyle="<->", color="#003366", lw=0.9)

    def dim(a, b, text, **kw):
        ax.annotate("", p(*a), p(*b), arrowprops=arrow)
        ax.text(*p((a[0] + b[0]) / 2 + kw.pop("dx", 0), (a[1] + b[1]) / 2 + kw.pop("dy", 0)), text,
                fontsize=8, color="#003366", **kw)

    dim((0, DEPTH + 55), (LENGTH, DEPTH + 55), f"{LENGTH} mm", dy=25, ha="center", va="center")
    dim((LENGTH + 55, 0), (LENGTH + 55, DEPTH), f"{DEPTH} mm", dx=12, ha="left", va="center")
    dim((mid - 80, -ROBOT_TO_EDGE), (mid - 80, 0), f"{ROBOT_TO_EDGE} mm", dx=-10, ha="right", va="center")
    ax.plot(*zip(p(mid - 90, -ROBOT_TO_EDGE), p(mid, -ROBOT_TO_EDGE)), color="#003366", lw=0.5, ls=":")
    ax.annotate("", p(*LAYOUT[0]), p(*LAYOUT[2]), zorder=1.5,
                arrowprops=dict(arrowstyle="-", color="#D9534F", lw=0.7, ls=":", shrinkA=0, shrinkB=0))
    ax.text(*p(mid, DEPTH * 0.68), f"both diagonals: {np.hypot(LENGTH, DEPTH):.0f} mm", fontsize=8, color="#D9534F", ha="center", va="center", zorder=3,
            bbox=dict(fc="0.96", ec="none", pad=1))


def make():
    with PdfPages(OUT) as pdf:
        fig, ax = new_page()
        for mid in range(4):
            marker_tile(ax, mid, 72 + 66 * (mid % 2), 254 - 66 * (mid // 2))
        ax.text(105, 152, "Where to put the markers (top view, distances between marker CENTRES)",
                ha="center", fontsize=9, weight="bold")
        layout_diagram(ax, 105 - 0.18 * LENGTH / 2, 96, 0.18)
        ax.text(15, 20, "Cut along the dashed lines (keep the white border). Marker 0 and 1 on the robot side, then "
                        "clockwise seen from above.\nThe distances are entered in MARKERS_LIVE (Labs 4 and 8). "
                        "Tape the markers flat.", fontsize=7.5, va="center")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = new_page()
        x0, y0 = (PAGE[0] - 4 * SHAPE_TILE) / 2, 18.0
        for r in range(6):
            for c, rgb in enumerate(COLOURS.values()):
                cx, cy = x0 + (c + 0.5) * SHAPE_TILE, y0 + (r + 0.5) * SHAPE_TILE
                fc = tuple(np.array(rgb) / 255)
                if r % 2:
                    ax.add_patch(Circle((cx, cy), SHAPE / 2, fc=fc, ec="none"))
                else:
                    ax.add_patch(Rectangle((cx - SHAPE / 2, cy - SHAPE / 2), SHAPE, SHAPE, fc=fc, ec="none"))
                cut_line(ax, cx, cy, SHAPE_TILE)
        ax.text(105, 289, "Cut along the dashed lines, not around the shapes. Circles: 30 mm diameter, squares: 30 mm. "
                          "Only ONE red circle on the table.", ha="center", va="center", fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)
    print("written to", OUT)


if __name__ == "__main__":
    make()
