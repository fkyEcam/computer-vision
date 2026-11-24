"""Generate the figures of the Day 1 slides (slides/figures/*.png) from the lab data.

Usage:  python make_figures.py      (run make_data.py first)
"""
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "notebooks" / "data"
FIG = ROOT / "slides" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
NAVY, RED = "#003366", "#D9534F"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def save(name, fig=None):
    (fig or plt.gcf()).savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close("all")
    print("figure", name)


def imread_rgb(p):
    return cv2.imread(str(p))[:, :, ::-1]


table = cv2.imread(str(DATA / "table" / "table_normal.png"))
gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)

# --- T1 ----------------------------------------------------------------------
cv2.imwrite(str(FIG / "table_normal.jpg"), table, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Pixel grid zoom: the image is an array of numbers.
# A 12x12 window across the edge of the red disc: left = the pixels, right = their values.
x0, y0, n = 533, 238, 12
crop = gray[y0:y0 + n, x0:x0 + n]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]})
ax[0].imshow(table[150:340, 400:640, ::-1], extent=(400, 640, 340, 150))
ax[0].add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), n, n, fill=False, ec="k", lw=2))
ax[0].set_title("detail of the image (black square)"); ax[0].axis("off")
ax[1].imshow(crop, cmap="gray", vmin=0, vmax=255)
ax[1].set_title(f"the {n}x{n} pixels in grey levels, enlarged")
ax[2].imshow(np.full_like(crop, 255), cmap="gray", vmin=0, vmax=255)
for (i, j), v in np.ndenumerate(crop):
    ax[2].text(j, i, str(v), ha="center", va="center", fontsize=8)
ax[2].set_title("what the computer stores")
for a in ax[1:]:
    a.set_xticks(np.arange(-.5, n, 1), minor=True); a.set_yticks(np.arange(-.5, n, 1), minor=True)
    a.grid(which="minor", color="#999999", lw=0.5); a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); save("pixels_zoom.png", fig)

# RGB channels and HSV channels
fig, ax = plt.subplots(2, 3, figsize=(12, 4.6))
for k, n in enumerate("RGB"):
    ax[0, k].imshow(table[:, :, 2 - k], cmap="gray"); ax[0, k].set_title(n); ax[0, k].axis("off")
hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
for k, n in enumerate(["H (hue)", "S (saturation)", "V (value)"]):
    ax[1, k].imshow(hsv[:, :, k], cmap="hsv" if k == 0 else "gray"); ax[1, k].set_title(n); ax[1, k].axis("off")
plt.tight_layout(); save("channels_rgb_hsv.png", fig)

# Exposure and histograms
fig, ax = plt.subplots(2, 3, figsize=(12, 5))
for k, name in enumerate(["dark", "normal", "bright"]):
    g = cv2.imread(str(DATA / "table" / f"table_{name}.png"), cv2.IMREAD_GRAYSCALE)
    ax[0, k].imshow(g, cmap="gray", vmin=0, vmax=255); ax[0, k].set_title(name); ax[0, k].axis("off")
    ax[1, k].bar(np.arange(256), np.bincount(g.ravel(), minlength=256), width=1, color=NAVY)
    ax[1, k].set_xlim(0, 255); ax[1, k].set_yticks([])
plt.tight_layout(); save("histograms_exposure.png", fig)

# Histogram equalisation: image, histogram and cumulative histogram, before and after
dark = cv2.imread(str(DATA / "table" / "table_dark.png"), cv2.IMREAD_GRAYSCALE)
eq = cv2.equalizeHist(dark)
fig, ax = plt.subplots(2, 3, figsize=(14, 5.2))
for r, (im, name) in enumerate([(dark, "dark image"), (eq, "equalised image")]):
    h = np.bincount(im.ravel(), minlength=256)
    cdf = np.cumsum(h) / im.size
    ax[r, 0].imshow(im, cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(name); ax[r, 0].axis("off")
    ax[r, 1].bar(np.arange(256), h, width=1, color=NAVY); ax[r, 1].set_xlim(0, 255); ax[r, 1].set_yticks([])
    ax[r, 1].set_title("histogram")
    ax[r, 2].plot(np.arange(256), cdf, color=RED, lw=2.5); ax[r, 2].set_xlim(0, 255); ax[r, 2].set_ylim(0, 1.02)
    ax[r, 2].set_title("cumulative histogram (CDF)")
ax[1, 1].set_xlabel("grey level"); ax[1, 2].set_xlabel("grey level")
plt.tight_layout(); save("equalisation.png", fig)

# Equalisation as a look-up table: the normalised CDF of the dark image IS the mapping
h = np.bincount(dark.ravel(), minlength=256)
cdf = np.cumsum(h)
lut = np.round(255 * (cdf - cdf[cdf > 0][0]) / (dark.size - cdf[cdf > 0][0])).clip(0, 255)
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.plot(np.arange(256), lut, color=RED, lw=2.5, label="equalisation LUT")
ax.plot([0, 255], [0, 255], "--", color="gray", label="identity")
ax.set_xlabel("input grey level"); ax.set_ylabel("output grey level"); ax.set_xlim(0, 255); ax.set_ylim(0, 255)
ax.legend(loc="lower right"); save("equalisation_lut.png", fig)

# Compression: lossless PNG vs JPEG high quality vs JPEG very low quality
import tempfile
crop_c = table[200:300, 460:580]  # red disc edge + table texture
tiles, labels = [], []
with tempfile.TemporaryDirectory() as tmp:
    for name, ext, params in [("PNG (lossless)", ".png", [cv2.IMWRITE_PNG_COMPRESSION, 9]),
                              ("JPEG quality 90", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 90]),
                              ("JPEG quality 5", ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 5])]:
        path = Path(tmp) / f"im{len(tiles)}{ext}"
        cv2.imwrite(str(path), table, params)
        size_kb = path.stat().st_size / 1024
        back = cv2.imread(str(path))
        diff = np.abs(back.astype(int) - table.astype(int)).max()
        tiles.append(back[200:300, 460:580])
        labels.append(f"{name}\n{size_kb:.0f} kB, max error {diff}")
raw_kb = table.nbytes / 1024
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for a, t, l in zip(ax, tiles, labels):
    a.imshow(t[:, :, ::-1], interpolation="nearest"); a.set_title(l); a.axis("off")
fig.suptitle(f"Same 1280x720 image (raw, uncompressed: {raw_kb:.0f} kB) -- enlarged detail", y=1.02)
plt.tight_layout(); save("compression.png", fig)

# HSV vs RGB thresholds on the dark image
dark_c = cv2.imread(str(DATA / "table" / "table_dark.png"))


def red_rgb(im):
    b, g, r = [im[:, :, k].astype(int) for k in range(3)]
    return (r > 150) & (g < 90) & (b < 90)


def red_hsv(im):
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    return cv2.inRange(h, (0, 120, 40), (10, 255, 255)) | cv2.inRange(h, (170, 120, 40), (179, 255, 255))


fig, ax = plt.subplots(2, 3, figsize=(12, 4.6))
for r, im, nm in [(0, table, "normal"), (1, dark_c, "dark")]:
    ax[r, 0].imshow(im[:, :, ::-1]); ax[r, 0].set_title(nm)
    ax[r, 1].imshow(red_rgb(im), cmap="gray"); ax[r, 1].set_title("rule on R, G, B")
    ax[r, 2].imshow(red_hsv(im), cmap="gray"); ax[r, 2].set_title("rule on H, S")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); save("rgb_vs_hsv.png", fig)

# Bayer mosaic illustration
bayer = np.zeros((8, 8, 3))
for i in range(8):
    for j in range(8):
        if (i + j) % 2 == 0:  # RGGB pattern: red/blue on the diagonal, green elsewhere
            bayer[i, j] = [0.85, 0, 0] if i % 2 == 0 else [0, 0, 0.85]
        else:
            bayer[i, j] = [0, 0.7, 0]
fig, ax = plt.subplots(figsize=(3.2, 3.2))
ax.imshow(bayer)
ax.set_xticks(np.arange(-.5, 8, 1), minor=True); ax.set_yticks(np.arange(-.5, 8, 1), minor=True)
ax.grid(which="minor", color="w", lw=2); ax.set_xticks([]); ax.set_yticks([])
save("bayer.png", fig)

# --- T2 ----------------------------------------------------------------------
# T2 figures are shown small on the slides: larger fonts
plt.rcParams["font.size"] = 19
rng = np.random.default_rng(0)
crop = gray[180:380, 430:730]  # red disc, blue bar and table: edges in every direction


def show_signed(a, im):
    v = np.abs(im).max()
    a.imshow(im, cmap="gray", vmin=-v, vmax=v)


# Kernel zoo: each kernel (top) and its effect on the same crop (bottom)
g9 = cv2.getGaussianKernel(9, 2.0); g9 = g9 @ g9.T
zoo = [("box 5x5", np.ones((5, 5)), r"$\frac{1}{25}\times$", "mean (blur)"),
       ("Gaussian 9x9, $\\sigma=2$", g9, "", "smooth blur"),
       ("Sobel x", np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), "", "vertical edges"),
       ("Sobel y", np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), "", "horizontal edges"),
       ("Laplacian", np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]), "", "all edges")]
fig, ax = plt.subplots(2, 6, figsize=(19, 6.2), gridspec_kw={"height_ratios": [1, 1.25]})
ax[0, 0].text(0.5, 0.5, "kernel $h$", ha="center", va="center", fontsize=22, color=NAVY); ax[0, 0].axis("off")
ax[1, 0].imshow(crop, cmap="gray"); ax[1, 0].set_title("input image $f$")
cf = crop.astype(np.float32)
for k, (name, h, pre, effect) in enumerate(zoo, start=1):
    v = np.abs(h).max()
    ax[0, k].imshow(h, cmap="RdBu_r", vmin=-v, vmax=v)
    if h.shape[0] <= 5:
        for (i, j), x in np.ndenumerate(h):
            ax[0, k].text(j, i, f"{x:g}", ha="center", va="center", fontsize=17,
                          color="w" if abs(x) > 0.6 * v else "k")
    ax[0, k].set_title(f"{pre} {name}".strip())
    out = cv2.filter2D(cf, -1, (h / h.sum() if name.startswith(("box", "Gauss")) else h).astype(np.float32))
    if name.startswith(("box", "Gauss")):
        ax[1, k].imshow(out, cmap="gray")
    else:
        show_signed(ax[1, k], out)
    ax[1, k].set_title(effect)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
plt.tight_layout(); save("kernels.png", fig)

# Frequencies: low-pass keeps the slow variations, high-pass the fast ones
big = gray[120:600, 250:1030].astype(np.float32)
low = cv2.GaussianBlur(big, (0, 0), 4)
high = big - low


row = 150  # this row crosses the large disc and the large bar
fig, ax = plt.subplots(2, 3, figsize=(14, 6.4), gridspec_kw={"height_ratios": [1.3, 1]})
for k, (im, t) in enumerate([(big, "original"), (low, r"low-pass (blur $\sigma=4$)"),
                             (high, "high-pass (orig. $-$ blur)")]):
    if k == 2:
        show_signed(ax[0, k], im)
    else:
        ax[0, k].imshow(im, cmap="gray")
    ax[0, k].axhline(row, color=RED, lw=1.5)
    ax[0, k].set_title(t); ax[0, k].axis("off")
    ax[1, k].plot(im[row], color=NAVY, lw=1)
    ax[1, k].set_xlim(0, im.shape[1]); ax[1, k].set_xticks([])
    ax[1, k].set_ylim(*((-60, 60) if k == 2 else (40, 230)))
    ax[1, k].set_title("profile along the red line", fontsize=15)
ax[1, 0].set_ylabel("grey level")
plt.tight_layout(); save("frequencies.png", fig)

# Denoising: the noise added is written above every image, as well as the filter applied
den = gray[170:390, 420:740]
gn = np.clip(den + rng.normal(0, 20, den.shape), 0, 255).astype(np.uint8)
sp_ = den.copy(); m = rng.random(den.shape); sp_[m < .03] = 0; sp_[m > .97] = 255
filters = [("no filter", lambda im: im),
           (r"+ Gaussian filter $\sigma=2$", lambda im: cv2.GaussianBlur(im, (0, 0), 2)),
           ("+ median filter 5x5", lambda im: cv2.medianBlur(im, 5)),
           ("+ bilateral filter", lambda im: cv2.bilateralFilter(im, 9, 60, 5))]
fig, ax = plt.subplots(2, 4, figsize=(17, 7), gridspec_kw={"hspace": 0.35})
for r, (im, nm) in enumerate([(gn, "Gaussian noise"), (sp_, "salt & pepper noise")]):
    for c, (fn, f) in enumerate(filters):
        ax[r, c].imshow(f(im), cmap="gray", vmin=0, vmax=255)
        ax[r, c].set_title(f"{nm}\n{fn}", color=NAVY if c else RED)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); save("denoising.png", fig)

# Gradients (Sobel) and Canny on the same view
view = gray[100:560, 200:1000]
gx = cv2.Sobel(view, cv2.CV_32F, 1, 0); gy = cv2.Sobel(view, cv2.CV_32F, 0, 1)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
for a, im, t in zip(ax, [gx, gy, np.hypot(gx, gy)],
                    [r"$g_x$ (Sobel x): vertical edges", r"$g_y$ (Sobel y): horizontal edges", r"magnitude $\sqrt{g_x^2 + g_y^2}$"]):
    if t.startswith("magnitude"):
        a.imshow(im, cmap="gray")
    else:
        show_signed(a, im)
    a.set_title(t); a.axis("off")
plt.tight_layout(); save("gradients.png", fig)

# Canny (1/2): threshold on the magnitude vs Canny, full view + zoom on the disc
from matplotlib.patches import Rectangle

noisy_view = np.clip(view + rng.normal(0, 8, view.shape), 0, 255).astype(np.uint8)
blur = cv2.GaussianBlur(noisy_view, (0, 0), 2)
mag = np.hypot(cv2.Sobel(noisy_view, cv2.CV_32F, 1, 0), cv2.Sobel(noisy_view, cv2.CV_32F, 0, 1))
zy, zx, zs = 100, 250, 100
panels = [(noisy_view, "input (noisy)"), (255 - 255 * (mag > 120).astype(np.uint8), "magnitude > T"),
          (255 - cv2.Canny(blur, 30, 90), "blur + Canny")]
fig, ax = plt.subplots(2, 3, figsize=(16, 7.6), gridspec_kw={"height_ratios": [1, 1.25]})
for c, (im, t) in enumerate(panels):
    ax[0, c].imshow(im, cmap="gray", vmin=0, vmax=255)
    ax[0, c].add_patch(Rectangle((zx, zy), zs, zs, fill=False, ec=RED, lw=2))
    ax[0, c].set_title(t)
    ax[1, c].imshow(im[zy:zy + zs, zx:zx + zs], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[1, c].set_title("zoom", color=RED, fontsize=16)
for r in range(2):
    for a in ax[r]:
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(r == 1); s.set_color(RED); s.set_linewidth(2)
plt.tight_layout(); save("canny.png", fig)

# Canny (2/2): non-maximum suppression on a 12x12 window across the disc edge (45 degrees)
gx_b = cv2.Sobel(blur, cv2.CV_32F, 1, 0); gy_b = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
mag_b = np.hypot(gx_b, gy_b)
cy, cx, h = 115, 328, 6                          # centre of the window, on the crest of the edge
M, GX, GY = [a[cy - h - 1:cy + h + 1, cx - h - 1:cx + h + 1] for a in (mag_b, gx_b, gy_b)]
steps = {0: (0, 1), 45: (1, 1), 90: (1, 0), 135: (1, -1)}
ang = (np.rad2deg(np.arctan2(GY, GX)) + 180) % 180
keep = np.zeros(M.shape, bool)
for y in range(1, M.shape[0] - 1):
    for x in range(1, M.shape[1] - 1):
        # keep the pixel if it is >= its two neighbours along the (quantised) gradient direction
        q = min(steps, key=lambda a: min(abs(ang[y, x] - a), 180 - abs(ang[y, x] - a)))
        dy, dx = steps[q]
        keep[y, x] = M[y, x] >= max(M[y + dy, x + dx], M[y - dy, x - dx]) and M[y, x] > 0.2 * M.max()
M, GX, GY, keep = M[1:-1, 1:-1], GX[1:-1, 1:-1], GY[1:-1, 1:-1], keep[1:-1, 1:-1]
Y, X = np.mgrid[0:2 * h, 0:2 * h]
step = np.round(np.array([GY[h, h], GX[h, h]]) / np.abs([GY[h, h], GX[h, h]]).max()).astype(int)
ts = np.arange(-4, 5)
py, px = h + ts * step[0], h + ts * step[1]
prof = M[py, px]
fig, ax = plt.subplots(1, 4, figsize=(21, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
ax[0].imshow(blur[cy - h:cy + h, cx - h:cx + h], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
ax[0].quiver(X[::2, ::2], Y[::2, ::2], (GX / M.max())[::2, ::2], (GY / M.max())[::2, ::2],  # one arrow in 2
             color=RED, angles="xy", scale_units="xy", scale=0.5, width=0.016)
ax[0].set_title("1. zoom + gradient")
ax[1].imshow(M, cmap="viridis", interpolation="nearest")
ax[1].plot(px, py, "o-", color="white", ms=5, lw=1.5)
ax[1].set_title("2. magnitude: a wide ridge")
ax[2].imshow(M, cmap="viridis", interpolation="nearest", alpha=0.35)
ax[2].plot(px, py, "-", color="#9aa5b1", lw=1.5)
yy, xx = np.nonzero(keep)
ax[2].scatter(xx, yy, s=110, marker="s", c=RED)
ax[2].set_title("3. after NMS: the crest")
for a in ax[:3]:
    a.set_xticks([]); a.set_yticks([])
ax[3].bar(ts, prof, color=[RED if v == prof.max() else "#9aa5b1" for v in prof])
ax[3].set_xlabel("pixels along the gradient")
ax[3].set_title("profile (white line)")
ax[3].set_yticks([])
plt.tight_layout(); save("canny_nms.png", fig)

# Thresholding coins.jpg with Otsu: on the grey level, then on the saturation
coins = cv2.imread(str(DATA / "coins.jpg"))
channels = {"gray": cv2.GaussianBlur(cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY), (7, 7), 0),
            "sat": cv2.GaussianBlur(cv2.cvtColor(coins, cv2.COLOR_BGR2HSV)[:, :, 1], (7, 7), 0)}
for key, ch in channels.items():
    flag = cv2.THRESH_BINARY_INV if key == "gray" else cv2.THRESH_BINARY
    T, mk = cv2.threshold(ch, 0, 255, flag + cv2.THRESH_OTSU)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1, 1.6, 1]})
    ax[0].imshow(coins[:, :, ::-1] if key == "gray" else ch, cmap="gray")
    ax[0].set_title("coins.jpg" if key == "gray" else "saturation channel S"); ax[0].axis("off")
    h = np.bincount(ch.ravel(), minlength=256)
    lo, hi = (NAVY, RED) if key == "sat" else (RED, NAVY)
    ax[1].bar(np.arange(T + 1), h[:int(T) + 1], width=1, color=lo)
    ax[1].bar(np.arange(T + 1, 256), h[int(T) + 1:], width=1, color=hi)
    ax[1].axvline(T, color="k", ls="--"); ax[1].text(T + 4, h.max() * 0.5, f" Otsu: T = {T:.0f}", backgroundcolor="w")
    ax[1].set_xlabel("grey level" if key == "gray" else "saturation"); ax[1].set_ylabel("number of pixels")
    ax[1].set_xlim(0, 255); ax[1].set_yticks([])
    if key == "sat":
        ax[1].set_yscale("log"); ax[1].set_ylabel("number of pixels (log)"); ax[1].set_yticks([], minor=True)
    ax[1].set_title("histogram (red = classified as coin)")
    ax[2].imshow(mk, cmap="gray"); ax[2].set_title("mask"); ax[2].axis("off")
    plt.tight_layout(); save(f"otsu_{key}.png", fig)

# Morphology on a toy binary image (3x3 square structuring element)
toy = np.zeros((13, 20), np.uint8)
toy[1:8, 1:8] = 1; toy[4, 4] = 0           # square with a one-pixel hole
toy[1:8, 11:18] = 1; toy[1:4, 14] = 0      # rectangle with a thin crack
toy[11, 17] = 1                             # isolated speck (noise)
toy[11, 1:8] = 1                            # thin line
se3 = np.ones((3, 3), np.uint8)
kw = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)
ero = cv2.erode(toy, se3, **kw); dil = cv2.dilate(toy, se3, **kw)
opn = cv2.dilate(ero, se3, **kw); cls = cv2.erode(dil, se3, **kw)


def grid(a, b, ref, title):
    rgb = np.ones(b.shape + (3,))
    rgb[b == 1] = matplotlib.colors.to_rgb(NAVY)
    rgb[(ref == 1) & (b == 0)] = (0.98, 0.78, 0.77)   # removed
    rgb[(ref == 0) & (b == 1)] = matplotlib.colors.to_rgb(RED)  # added
    a.imshow(rgb)
    a.set_xticks(np.arange(-.5, b.shape[1], 1), minor=True); a.set_yticks(np.arange(-.5, b.shape[0], 1), minor=True)
    a.grid(which="minor", color="0.75", lw=0.8); a.tick_params(which="both", length=0)
    a.set_xticks([]); a.set_yticks([]); a.set_title(title)
    for sp in a.spines.values():
        sp.set_visible(False)


fig, ax = plt.subplots(1, 4, figsize=(17, 3.6), gridspec_kw={"width_ratios": [0.35, 1, 1, 1]})
grid(ax[0], se3, se3, "element $B$")
grid(ax[1], toy, toy, "input")
grid(ax[2], ero, toy, "erosion (pink: removed)")
grid(ax[3], dil, toy, "dilation (red: added)")
plt.tight_layout(); save("morpho_erode_dilate.png", fig)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
grid(ax[0], toy, toy, "input")
grid(ax[1], opn, toy, "opening")
grid(ax[2], cls, toy, "closing")
plt.tight_layout(); save("morpho_open_close.png", fig)

# Morphology on a real (noisy) mask
hsv_mask = cv2.inRange(hsv, (0, 90, 40), (179, 255, 255))
mask = hsv_mask[150:450, 300:800]
noisy_mask = mask.copy(); m = rng.random(mask.shape)
noisy_mask[m < .02] = 255; noisy_mask[(m > .98) & (mask > 0)] = 0
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(noisy_mask, cv2.MORPH_OPEN, se)
ops = [("noisy mask: specks and holes", noisy_mask), ("opening: specks removed", opened),
       ("then closing: holes filled", cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se))]
fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
for a, (t, im) in zip(ax, ops):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); save("morphology.png", fig)

# Regions: mask -> connected components -> centroid and orientation
clean = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, se)
n, labels, stats, cents = cv2.connectedComponentsWithStats(clean)
lab_rgb = np.zeros(labels.shape + (3,))
cmap = plt.get_cmap("tab10")
for i in range(1, n):
    lab_rgb[labels == i] = cmap((i - 1) % 10)[:3]
cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
vis = table.copy()
for c in cnts:
    mo = cv2.moments(c)
    if mo["m00"] < 300:
        continue
    u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    th = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    cv2.drawContours(vis, [c], -1, (0, 0, 0), 2)
    cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 0), -1)
    d = 45 * np.array([np.cos(th), np.sin(th)])
    cv2.line(vis, (int(u - d[0]), int(v - d[1])), (int(u + d[0]), int(v + d[1])), (0, 0, 0), 2)
fig, ax = plt.subplots(1, 3, figsize=(16, 3.4))
for a, im, t in zip(ax, [clean, lab_rgb, vis[:, :, ::-1]],
                    ["1. binary mask", f"2. components ({n - 1} labels)",
                     "3. centroid, orientation"]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); save("regions.png", fig)

# Detection pipeline result (same drawing, with coordinates)
vis = table.copy()
for c in cnts:
    mo = cv2.moments(c)
    if mo["m00"] < 300:
        continue
    u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
    th = 0.5 * np.arctan2(2 * mo["mu11"], mo["mu20"] - mo["mu02"])
    cv2.drawContours(vis, [c], -1, (0, 0, 0), 2)
    cv2.circle(vis, (int(u), int(v)), 6, (0, 0, 0), -1)
    d = 45 * np.array([np.cos(th), np.sin(th)])
    cv2.line(vis, (int(u - d[0]), int(v - d[1])), (int(u + d[0]), int(v + d[1])), (0, 0, 0), 2)
    cv2.putText(vis, f"({u:.0f},{v:.0f})", (int(u) + 12, int(v) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
cv2.imwrite(str(FIG / "detections.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Harris: response map and detected corners
board = cv2.imread(str(DATA / "chessboard.jpg"))
bg = np.float32(cv2.cvtColor(board, cv2.COLOR_BGR2GRAY))
R = cv2.cornerHarris(bg, 2, 3, 0.04)
hb = board.copy(); hb[cv2.dilate(R, np.ones((7, 7))) > 0.01 * R.max()] = (0, 0, 255)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].imshow(board[:, :, ::-1]); ax[0].set_title("chessboard.jpg (Q. Lurkin)")
ax[1].imshow(hb[:, :, ::-1]); ax[1].set_title("Harris corners (score > 1 % of max)")
for a in ax:
    a.axis("off")
plt.tight_layout(); save("harris.png", fig)

# ORB: find the same points in a rotated, scaled and darker view of the table
h_, w_ = table.shape[:2]
Mw = cv2.getRotationMatrix2D((w_ / 2, h_ / 2), 25, 0.75)
other = cv2.convertScaleAbs(cv2.warpAffine(table, Mw, (w_, h_), borderValue=(40, 40, 40)), alpha=0.7)
orb = cv2.ORB_create(800)
k1, d1 = orb.detectAndCompute(table, None)
k2, d2 = orb.detectAndCompute(other, None)
knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
good = [m for m, n2 in knn if m.distance < 0.8 * n2.distance]  # ratio test
Hm, inl = cv2.findHomography(np.float32([k1[m.queryIdx].pt for m in good]),
                             np.float32([k2[m.trainIdx].pt for m in good]), cv2.RANSAC, 5)
matches = [m for m, ok in zip(good, inl.ravel()) if ok]
matches = sorted(matches, key=lambda x: x.distance)[:60]
mv = cv2.drawMatches(table, k1, other, k2, matches, None, matchColor=(0, 0, 255), singlePointColor=(255, 200, 0),
                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
fig = plt.figure(figsize=(14, 4.2))
plt.imshow(mv[:, :, ::-1]); plt.axis("off")
plt.title(f"ORB: {len(k1)} keypoints, {len(good)} matches after ratio test, {int(inl.sum())} consistent (shown: {len(matches)})")
save("orb_matches.png", fig)

plt.rcParams["font.size"] = 13

# --- T3 ----------------------------------------------------------------------
def distort(x, y, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
    """OpenCV distortion model on normalised coordinates."""
    r2 = x ** 2 + y ** 2
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    return x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x ** 2), y * rad + p1 * (r2 + 2 * y ** 2) + 2 * p2 * x * y


gt = json.loads((DATA / "synthetic_ground_truth.json").read_text())
dist_gt = np.array(gt["dist_k1_k2_p1_p2_k3"])
K_gt = np.array(gt["K"])
# Catalogue of distortions: a grid of straight lines (grey) and what the lens makes of it (navy)
cases = [("no distortion", {}, 1.0),
         (r"barrel: $k_1 < 0$", {"k1": -0.25}, 1.0),
         (r"pincushion: $k_1 > 0$", {"k1": 0.25}, 1.0),
         (r"moustache: $k_1 < 0,\ k_2 > 0$", {"k1": -0.5, "k2": 0.45}, 1.0),
         (r"tangential: $p_1 \neq 0$", {"p1": 0.08}, 1.0),
         ("our USB camera (calibrated)", dict(zip(["k1", "k2", "p1", "p2", "k3"], dist_gt)), 640 / 910)]
fig, ax = plt.subplots(2, 3, figsize=(13, 5.6))
for a, (t, kw, half) in zip(ax.ravel(), cases):
    s = np.linspace(-1, 1, 120)
    for c in np.linspace(-1, 1, 11):
        for gx_, gy_ in [(np.full_like(s, c), s * 0.6), (s, np.full_like(s, c * 0.6))]:
            gx_, gy_ = gx_ * half, gy_ * half
            a.plot(gx_, gy_, color="lightgray", lw=0.8)
            a.plot(*distort(gx_, gy_, **kw), color=NAVY, lw=1.0)
    a.set_aspect("equal"); a.axis("off"); a.invert_yaxis()
    a.set_title(t, color=RED if "USB" in t else "black", fontsize=19)
plt.tight_layout(); save("distortion_grids.png", fig)

# Undistortion: a grid poster filling the field of view of our USB camera (simulated with its calibrated K, dist)
W_, H_ = gt["image_size"]
pad = 300                                       # the poster is larger than the field of view
ideal = np.full((H_ + 2 * pad, W_ + 2 * pad), 235, np.uint8)
for x in range(-pad + 12, W_ + pad, 80):
    cv2.line(ideal, (x + pad, 0), (x + pad, H_ + 2 * pad), 40, 3)
for y in range(-pad + 31, H_ + pad, 80):
    cv2.line(ideal, (0, y + pad), (W_ + 2 * pad, y + pad), 40, 3)
# Distorted image by inverse mapping: for each pixel of the camera image, where is it in the ideal (pinhole) image?
uu, vv = np.meshgrid(np.arange(W_, dtype=np.float32), np.arange(H_, dtype=np.float32))
src = cv2.undistortPoints(np.stack([uu.ravel(), vv.ravel()], 1).reshape(-1, 1, 2), K_gt, dist_gt, P=K_gt).reshape(H_, W_, 2)
raw = cv2.remap(ideal, src[..., 0] + pad, src[..., 1] + pad, cv2.INTER_LINEAR)
und = cv2.undistort(raw, K_gt, dist_gt)


def to_raw(u, v):
    """Ideal pixel -> distorted pixel (pinhole, then the distortion model)."""
    x, y = (u - K_gt[0, 2]) / K_gt[0, 0], (v - K_gt[1, 2]) / K_gt[1, 1]
    xd, yd = distort(x, y, *dist_gt)
    return xd * K_gt[0, 0] + K_gt[0, 2], yd * K_gt[1, 1] + K_gt[1, 2]


s_ = np.linspace(-pad, W_ + pad, 2000)
row_y, col_x = 51, 112                          # one grid row and one grid column near the top-left corner
lines = [to_raw(s_, np.full_like(s_, row_y)), to_raw(np.full_like(s_, col_x), np.linspace(-pad, H_ + pad, 2000))]
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for a, im, t in zip(ax, [raw, und], ["what the camera sees (distorted)", "rectified: cv2.undistort(img, K, dist)"]):
    a.imshow(im, cmap="gray", vmin=0, vmax=255)
    a.set_title(t, fontsize=20); a.axis("off")
for lu, lv in lines:                            # straight chord between the ends of a curved grid line
    ok = (lu >= 0) & (lu < W_) & (lv >= 0) & (lv < H_)
    ax[0].plot(lu[ok][[0, -1]], lv[ok][[0, -1]], "--", color=RED, lw=2)
ax[1].plot([0, W_ - 1], [row_y, row_y], "--", color=RED, lw=2); ax[1].plot([col_x, col_x], [0, H_ - 1], "--", color=RED, lw=2)
plt.tight_layout(); save("undistort_example.png", fig)

# Perspective: the same face seen from Z = 10 cm and Z = 1 m (pinhole, size in the image = f * size / Z)
from matplotlib.patches import Ellipse

# (name, x, y, depth behind the nose tip, half-width, half-height, colour), in cm; y downwards
face_parts = [("ear", -8.5, 0.5, 10, 1.8, 3.2, "#e0a98c"), ("ear", 8.5, 0.5, 10, 1.8, 3.2, "#e0a98c"),
              ("face", 0, 0, 5, 7.0, 9.5, "#f2c9a8"),
              ("eye", -3.2, -1.8, 3, 0.9, 0.9, NAVY), ("eye", 3.2, -1.8, 3, 0.9, 0.9, NAVY),
              ("mouth", 0, 5.2, 3, 2.4, 0.7, RED), ("nose", 0, 1.6, 0, 1.6, 1.6, "#c9846a")]
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
for a, z0 in zip(ax, [10, 100]):
    s = (z0 + 5) / 7                                # focal length chosen so that the face has the same size
    for name, x, y, dz, hw, hh, col in sorted(face_parts, key=lambda p: -p[3]):   # far parts first
        Z = z0 + dz
        a.add_patch(Ellipse((s * x / Z, s * y / Z), 2 * s * hw / Z, 2 * s * hh / Z, color=col, ec="k", lw=0.8))
    ratio = (z0 + 10) / z0
    a.set_title(f"nose: Z = {z0} cm\nears: Z = {z0 + 10} cm\nnose ×{ratio:.1f} vs ears", fontsize=24)
    a.set_xlim(-1.9, 1.9); a.set_ylim(1.6, -1.6); a.set_aspect("equal"); a.axis("off")
plt.tight_layout(); save("face_perspective.png", fig)

# Rectified cover (homography)
cover = cv2.imread(str(DATA / "cover.jpg"))
corners = np.float32([[357, 338], [690, 240], [1054, 572], [612, 816]])
w, h = 155 * 3, 230 * 3
Hc = cv2.getPerspectiveTransform(corners, np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]))
flat = cv2.warpPerspective(cover, Hc, (w, h))
plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.6, 0.8]})
ax[0].imshow(cover[:, :, ::-1]); ax[0].plot(*np.vstack([corners, corners[:1]]).T, "-o", color=RED)
ax[0].set_title("photo: pixels $(u, v)$"); ax[0].set_xlabel("$u$ [px]"); ax[0].set_ylabel("$v$ [px]")
ax[1].imshow(flat[:, :, ::-1], extent=[0, 155, 230, 0])      # 3 px per mm: show the axes in mm
ax[1].set_title("cover plane $(X, Y)$"); ax[1].set_xlabel("$X$ [mm]"); ax[1].set_ylabel("$Y$ [mm]")
plt.tight_layout(); save("homography_cover.png", fig)
plt.rcParams["font.size"] = 13

# The inverse operation: put a new (flat) cover on the book in the photo with H^-1
new = np.full((h, w, 3), (90, 51, 0), np.uint8)                 # navy background (BGR)
for i in range(0, h, 69):                                       # checkerboard: shows the perspective
    for j in range(0, w, 93):
        if (i // 69 + j // 93) % 2:
            new[i:i + 69, j:j + 93] = (79, 83, 217)             # accent red (BGR)
for k, line in enumerate(["DEEP", "LEARNING", "FOR", "COMPUTER", "VISION"]):
    cv2.putText(new, line, (30, 150 + 95 * k), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
H_fp = np.linalg.inv(Hc)                                        # flat -> photo (H of the previous slide)
H_pf = np.linalg.inv(H_fp)                                      # photo -> flat: H^-1
uu, vv = np.meshgrid(np.arange(cover.shape[1]), np.arange(cover.shape[0]))
q = H_pf @ np.vstack([uu.ravel(), vv.ravel(), np.ones(uu.size)])
xq, yq = (q[0] / q[2]).reshape(uu.shape), (q[1] / q[2]).reshape(uu.shape)
inside = (xq >= 0) & (xq < w) & (yq >= 0) & (yq < h)            # photo pixels that fall on the cover
swapped = cover.copy()
swapped[inside] = new[yq[inside].astype(int), xq[inside].astype(int)]
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.6, 0.8, 1.6]})
ax[0].imshow(cover[:, :, ::-1]); ax[0].plot(*np.vstack([corners, corners[:1]]).T, "-o", color=RED); ax[0].set_title("photo $(u, v)$")
ax[1].imshow(new[:, :, ::-1]); ax[1].set_title("new cover $(X, Y)$")
ax[2].imshow(swapped[:, :, ::-1]); ax[2].set_title(r"new cover placed with $H^{-1}$")
for a in ax:
    a.axis("off")
plt.tight_layout(); save("homography_swap.png", fig)

# --- T4 ----------------------------------------------------------------------
files = sorted((DATA / "calib").glob("*.png"))
objp = np.zeros((54, 3), np.float32); objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 25
pts, used = [], []
for f in files:
    g = cv2.imread(str(f), 0)
    ok_, c = cv2.findChessboardCorners(g, (9, 6))
    if ok_:
        pts.append(cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (3, 30, 1e-3))); used.append(f)
rms, K, dist, rv, tv = cv2.calibrateCamera([objp] * len(pts), pts, (1280, 720), None, None)
err = []
for k in range(len(pts)):
    p, _ = cv2.projectPoints(objp, rv[k], tv[k], K, dist)
    err.append(np.sqrt(((p - pts[k]) ** 2).sum(-1).mean()))
fig, ax = plt.subplots(figsize=(10, 3))
cols = [RED if e > 1 else NAVY for e in err]
ax.bar([f.stem[-2:] for f in used], err, color=cols); ax.axhline(rms, ls="--", color="k", label=f"global RMS = {rms:.2f} px")
ax.set_ylabel("RMS error [px]"); ax.set_xlabel("image"); ax.legend()
save("reprojection_errors.png", fig)

# Calibration views montage
sel = [0, 3, 5, 9, 15, 21]
fig, ax = plt.subplots(2, 3, figsize=(12, 4.5))
for a, k in zip(ax.ravel(), sel):
    im = cv2.imread(str(files[k]))
    okk, c = cv2.findChessboardCorners(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (9, 6))
    cv2.drawChessboardCorners(im, (9, 6), c, okk)
    a.imshow(im[:, :, ::-1]); a.set_title(files[k].name, fontsize=10); a.axis("off")
plt.tight_layout(); save("calib_views.png", fig)

# Recalibrate without outlier and undistort
keep = [p for p, e in zip(pts, err) if e < 1]
rms2, K2, dist2, _, _ = cv2.calibrateCamera([objp] * len(keep), keep, (1280, 720), None, None)
b = cv2.imread(str(files[3]))
Kn, _ = cv2.getOptimalNewCameraMatrix(K2, dist2, (1280, 720), 0)
Kn1, _ = cv2.getOptimalNewCameraMatrix(K2, dist2, (1280, 720), 1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2))
for a, im, t in zip(ax, [b, cv2.undistort(b, K2, dist2, None, Kn), cv2.undistort(b, K2, dist2, None, Kn1)],
                    ["original", r"undistorted, $\alpha=0$", r"undistorted, $\alpha=1$"]):
    a.imshow(im[:, :, ::-1]); a.set_title(t); a.axis("off")
plt.tight_layout(); save("undistort.png", fig)

# ArUco detection on the table
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
mc, ids, _ = det.detectMarkers(table)
va = table.copy(); cv2.aruco.drawDetectedMarkers(va, mc, ids)
cv2.imwrite(str(FIG / "aruco_table.jpg"), va, [cv2.IMWRITE_JPEG_QUALITY, 90])

# Error with / without calibration: summary numbers stored for the slides
gt = json.loads((DATA / "synthetic_ground_truth.json").read_text())
cen = {int(i): c.reshape(4, 2).mean(0) for c, i in zip(mc, ids.ravel())}
src = np.float32([cen[i] for i in range(4)]); dst = np.float32([[0, 0], [400, 0], [400, 280], [0, 280]])
true = np.array([[p["x_mm"], p["y_mm"]] for p in gt["table"]["parts"]])
uv = np.float32([[p["u_px"], p["v_px"]] for p in gt["table"]["parts"]]).reshape(-1, 1, 2)
und = lambda x: cv2.undistortPoints(x.reshape(-1, 1, 2), K2, dist2, P=K2)
eA = np.linalg.norm(cv2.perspectiveTransform(uv, cv2.getPerspectiveTransform(src, dst)).reshape(-1, 2) - true, axis=1)
eB = np.linalg.norm(cv2.perspectiveTransform(und(uv), cv2.getPerspectiveTransform(und(src).reshape(-1, 2), dst)).reshape(-1, 2) - true, axis=1)
fig, ax = plt.subplots(figsize=(8, 3.2))
x = np.arange(len(true))
ax.bar(x - 0.2, eA, 0.4, label="homography on raw pixels", color=RED)
ax.bar(x + 0.2, eB, 0.4, label="undistort, then homography", color=NAVY)
ax.set_xticks(x, [p["name"].replace("_", " ") for p in gt["table"]["parts"]], rotation=20)
ax.set_ylabel("position error [mm]"); ax.legend()
save("calibration_gain.png", fig)
print(f"RMS with outlier {rms:.2f}, without {rms2:.2f}; mean error raw {eA.mean():.1f} mm, calibrated {eB.mean():.2f} mm")


# --- T1: challenges of vision, images from Stanford CS231n Lecture 2 (2025) ---------------
# The slides credit each photograph (CC0 / CC-BY, small captions kept in the crops).
def stanford_challenges():
    import pypdfium2
    from PIL import Image, ImageDraw, ImageFont
    pdf = ROOT.parent / "Inspirations Stanford" / "lecture_2 - Image Classification with Linear Classifiers.pdf"
    if not pdf.exists():
        print("Stanford PDF not found, stanford_challenges.png not regenerated")
        return
    doc = pypdfium2.PdfDocument(str(pdf))
    pages = [(9, "Viewpoint"), (10, "Illumination"), (11, "Background clutter"),
             (12, "Occlusion"), (13, "Deformation"), (14, "Intra-class variation")]
    try:
        font = ImageFont.truetype("arialbd.ttf", 38)
    except OSError:
        font = ImageFont.load_default()
    sheet = Image.new("RGB", (2700, 960), "white")
    for k, (pg, name) in enumerate(pages):
        im = doc[pg - 1].render(scale=3).to_pil().convert("RGB")
        crop = im.crop((0, int(im.height * 0.10), im.width, int(im.height * 0.92)))
        ys, xs = np.nonzero(np.asarray(crop.convert("L")) < 245)
        crop = crop.crop((max(xs.min() - 10, 0), max(ys.min() - 10, 0),
                          min(xs.max() + 10, crop.width), min(ys.max() + 10, crop.height)))
        crop.thumbnail((900, 420))
        tile = Image.new("RGB", (900, 480), "white")
        tile.paste(crop, ((900 - crop.width) // 2, 50 + (420 - crop.height) // 2))
        ImageDraw.Draw(tile).text((450, 5), name, fill=(0, 51, 102), font=font, anchor="ma")
        sheet.paste(tile, ((k % 3) * 900, (k // 3) * 480))
    sheet.save(FIG / "stanford_challenges.png")
    print("figure stanford_challenges.png")


stanford_challenges()


# --- T4: guiding the robot (red circle among the 4 markers) -------------------------------
def robot_pipeline():
    img = cv2.imread(str(DATA / "robot" / "robot_cell_1.png"))
    taught = json.loads((DATA / "robot" / "taught_markers.json").read_text())["markers"]
    xyz = np.array([taught[str(i)] for i in range(4)])
    xy_t = np.array([[0, 0], [400, 0], [400, 280], [0, 280]], float)
    sol, *_ = np.linalg.lstsq(np.hstack([xy_t, np.ones((4, 1))]), xyz[:, :2], rcond=None)
    A = sol.T
    mc, ids, _ = det.detectMarkers(img)
    cen = {int(i): c.reshape(4, 2).mean(0) for c, i in zip(mc, ids.ravel())}
    und = lambda x: cv2.undistortPoints(np.float32(x).reshape(-1, 1, 2), K2, dist2, P=K2).reshape(-1, 2)
    Hm = cv2.getPerspectiveTransform(und([cen[i] for i in range(4)]), np.float32(xy_t))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 60), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 60), (179, 255, 255))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blobs = []
    for c in cs:
        if cv2.contourArea(c) > 200:
            m = cv2.moments(c)
            blobs.append((4 * np.pi * cv2.contourArea(c) / cv2.arcLength(c, True) ** 2,
                          np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])))
    blobs.sort(key=lambda b: -b[0])
    uv = blobs[0][1]
    xy = cv2.perspectiveTransform(und(uv).reshape(-1, 1, 2), Hm).reshape(2)
    tgt = A[:, :2] @ xy + A[:, 2]

    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, mc, ids)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.45, 1]})
    ax[0].imshow(vis[:, :, ::-1]); ax[0].axis("off")
    for circ, p in blobs:
        ok = p is uv
        ax[0].plot(*p, "+", color="c" if ok else "orange", ms=22, mew=3)
        ax[0].annotate(f"circularity {circ:.2f}" + (" -> target" if ok else ""), p, (p[0] + 40, p[1] + 70),
                       color="c" if ok else "orange", fontsize=13, fontweight="bold",
                       bbox=dict(fc="k", alpha=0.55, lw=0))
    ax[0].set_title(f"camera: red circle at (u, v) = ({uv[0]:.0f}, {uv[1]:.0f}) px")
    # top view in the robot frame: robot x up, robot y to the left
    P = lambda x, y: (-y, x)
    a = ax[1]
    poly = np.vstack([xyz[:, :2], xyz[:1, :2]])
    a.plot(*P(poly[:, 0], poly[:, 1]), "k--", lw=1)
    for i, (x, y, _) in enumerate(xyz):
        a.plot(*P(x, y), "ks", ms=9); a.annotate(f"marker {i}", P(x, y), (P(x, y)[0] + 12, P(x, y)[1] + 14), fontsize=11)
    th = np.linspace(-np.pi / 2, np.pi / 2, 100)
    a.plot(*P(550 * np.cos(th), 550 * np.sin(th)), ":", color=NAVY, label="reach")
    a.plot(0, 0, "s", color=NAVY, ms=14)
    a.annotate("", P(110, 0), P(0, 0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
    a.annotate("", P(0, 110), P(0, 0), arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2))
    a.text(*P(125, -10), "$x_r$", color=NAVY, fontsize=15); a.text(*P(10, 150), "$y_r$", color=NAVY, fontsize=15)
    o = xyz[0, :2]
    for v, lab in [(A[:, 0] * 90, "$X$"), (A[:, 1] * 90, "$Y$")]:
        a.annotate("", P(*(o + v)), P(*o), arrowprops=dict(arrowstyle="-|>", color="0.35", lw=2))
        a.text(*P(*(o + 1.25 * v)), lab, color="0.35", fontsize=15)
    a.plot(*P(*tgt), "o", color=RED, ms=13)
    a.annotate(f"({tgt[0]:.0f}, {tgt[1]:.0f}) mm", P(*tgt), (P(*tgt)[0] - 40, P(*tgt)[1] + 35), color=RED,
               fontsize=13, fontweight="bold")
    a.text(*P(-35, 0), "robot base", ha="center", va="top", color=NAVY, fontsize=12)
    a.set_aspect("equal"); a.set_xlim(-330, 330); a.set_ylim(-80, 570)
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
    a.set_title("robot frame, seen from above")
    plt.tight_layout(); save("robot_pipeline.png", fig)


robot_pipeline()
