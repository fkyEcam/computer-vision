"""Generate the synthetic datasets used in the Day 1 labs.

A virtual "USB camera" (wide-angle lens, strong radial distortion) is used to render:
  * data/calib/calib_XX.png  : views of a 9x6 checkerboard (25 mm squares) for L4
  * data/table/table_*.png   : top views of a work table with 4 ArUco markers and
                               flat coloured parts at known positions (mm) for L1, L2, L4
  * data/robot/robot_cell_*.png : same table with a red circle (+ distractors) to reach with the robot
  * data/robot/taught_markers.json : marker centres "taught" to the simulated robot (robot frame)
  * data/synthetic_ground_truth.json : true intrinsics, distortion, poses, part positions, robot frame

Because the camera is simulated, students can compare their calibration with the truth.
These images are meant to be replaced (or complemented) by real images taken with the
project USB cameras; the notebooks only rely on the file names and on the ground-truth
json for the optional "check against the truth" cells.

Usage:  python make_data.py            (writes into ../notebooks/data)
"""
import json
from pathlib import Path

import cv2
import numpy as np

RNG = np.random.default_rng(7)
OUT = Path(__file__).resolve().parent.parent / "notebooks" / "data"

# ----------------------------------------------------------------------------
# Virtual camera
# ----------------------------------------------------------------------------
W, H = 1280, 720
K = np.array([[910.0, 0.0, 652.0],
              [0.0, 905.0, 351.0],
              [0.0, 0.0, 1.0]])
DIST = np.array([-0.32, 0.12, 0.0012, -0.0008, -0.02])  # k1 k2 p1 p2 k3 (OpenCV order)


def rot(rx, ry, rz):
    """Rotation matrix from Euler angles in degrees (applied x, then y, then z)."""
    rx, ry, rz = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# Normalised ray direction of every pixel (undistorted), computed once.
_uv = np.stack(np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64)), -1)
_norm = cv2.undistortPointsIter(_uv.reshape(-1, 1, 2), K, DIST, None, None,
                                (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-9))
RAYS = np.concatenate([_norm.reshape(H, W, 2), np.ones((H, W, 1))], -1)  # (H, W, 3)


def render_plane(texture, px_per_mm, origin_mm, R, t, background):
    """Render a textured plane Z=0 (world frame) seen by the virtual camera.

    texture    : image of the plane, texture[0,0] is at world (origin_mm)
    R, t       : world -> camera transform (x_cam = R x_world + t)
    background : image (H, W, 3) used where the ray misses the texture
    """
    d = RAYS @ R  # ray directions in world frame (R^T d for each pixel)
    C = -R.T @ t  # camera centre in world frame
    s = -C[2] / d[..., 2]
    X = C[0] + s * d[..., 0]
    Y = C[1] + s * d[..., 1]
    mx = ((X - origin_mm[0]) * px_per_mm).astype(np.float32)
    my = ((Y - origin_mm[1]) * px_per_mm).astype(np.float32)
    # Supersampling: blur the texture slightly to avoid aliasing
    img = cv2.remap(texture, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0))
    mask = cv2.remap(np.full(texture.shape[:2], 255, np.uint8), mx, my, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (mask.astype(np.float32) / 255.0)[..., None]
    return (img * mask + background * (1 - mask)).astype(np.float32)


def photometric(img, gain=1.0, noise=3.0, vignetting=0.35, blur=0.8):
    """Add the 'real camera' effects: vignetting, exposure gain, blur, sensor noise."""
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
    img = img * (1 - vignetting * r2)[..., None] * gain
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    img = img + RNG.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def background_texture():
    base = RNG.normal(70, 18, (H // 8, W // 8, 3)).astype(np.float32)
    base = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(base, (0, 0), 6) + RNG.normal(0, 6, (H, W, 3))


# ----------------------------------------------------------------------------
# 1. Checkerboard calibration images
# ----------------------------------------------------------------------------
PATTERN = (9, 6)  # inner corners (columns, rows)
SQUARE = 25.0  # mm


def checkerboard_texture(ppm=8):
    nx, ny = PATTERN[0] + 1, PATTERN[1] + 1
    margin = 15.0  # mm white border
    w_mm, h_mm = nx * SQUARE + 2 * margin, ny * SQUARE + 2 * margin
    tex = np.full((int(h_mm * ppm), int(w_mm * ppm), 3), 235, np.uint8)
    for j in range(ny):
        for i in range(nx):
            if (i + j) % 2 == 0:
                x0, y0 = int((margin + i * SQUARE) * ppm), int((margin + j * SQUARE) * ppm)
                tex[y0:y0 + int(SQUARE * ppm), x0:x0 + int(SQUARE * ppm)] = 25
    # world origin = first inner corner  ->  texture origin in mm:
    origin = (-(margin + SQUARE), -(margin + SQUARE))
    return tex.astype(np.float32), ppm, origin


def make_calibration_set():
    tex, ppm, origin = checkerboard_texture()
    centre = np.array([(PATTERN[0] - 1) * SQUARE / 2, (PATTERN[1] - 1) * SQUARE / 2, 0])
    # (distance mm, rx, ry, rz, shift_x px, shift_y px) -- chosen to cover the image
    poses = [
        (520, 0, 0, 0, 0, 0), (480, 25, 0, 3, -330, -170), (480, -25, 5, -4, 330, 170),
        (500, 5, 30, 2, 360, -180), (500, -5, -30, -2, -360, 180), (430, 20, 20, 8, -380, 190),
        (430, -20, -20, -6, 390, -190), (600, 10, -10, 15, 0, 220), (600, -10, 10, -12, 0, -220),
        (450, 35, 0, 0, 0, 120), (450, 0, 35, 0, 120, 0), (450, -35, 0, 0, 0, -120),
        (450, 0, -35, 0, -120, 0), (380, 15, -15, 30, 300, 120), (420, -15, 15, -30, -260, -100),
        (700, 0, 0, 45, -420, -200), (700, 0, 0, -45, 420, 200), (650, 20, 25, 10, 430, -210),
        (650, -25, -20, -10, -430, 210), (550, 30, -25, 5, 200, 50),
        # Two deliberately bad views: board partly outside the image, and motion blur
        (420, 10, 10, 0, 640, 0), (500, 5, 5, 0, 0, 0),
    ]
    out = OUT / "calib"
    out.mkdir(parents=True, exist_ok=True)
    truth = []
    for k, (dist, rx, ry, rz, sx, sy) in enumerate(poses):
        R = rot(180 + rx, ry, rz)  # board faces the camera (camera z points into the board)
        # put the board centre on the optical axis at 'dist', then shift it in the image
        target = np.array([sx / K[0, 0] * dist, sy / K[1, 1] * dist, dist])
        t = target - R @ centre
        img = render_plane(tex, ppm, origin, R, t, background_texture())
        img = photometric(img, gain=RNG.uniform(0.85, 1.1))
        if k == len(poses) - 1:  # motion blur
            kernel = np.zeros((1, 41), np.float32)
            kernel[0, :] = 1 / 41
            img = cv2.filter2D(img, -1, kernel)
        name = f"calib_{k:02d}.png"
        cv2.imwrite(str(out / name), img)
        rvec, _ = cv2.Rodrigues(R)
        truth.append({"file": name, "rvec": rvec.ravel().tolist(), "tvec": t.tolist()})
    return truth


# ----------------------------------------------------------------------------
# 2. Work-table scenes (ArUco markers + flat coloured parts)
# ----------------------------------------------------------------------------
MARKER_SIZE = 40.0  # mm (black square, without white border)
MARKERS = {0: (0.0, 0.0), 1: (400.0, 0.0), 2: (400.0, 280.0), 3: (0.0, 280.0)}  # centres, mm
PARTS = [  # name, colour (BGR), shape, centre (mm), size (mm), angle (deg)
    ("red_disc", (40, 40, 200), "disc", (115.0, 85.0), (22.0,), 0.0),
    ("blue_plate", (190, 90, 30), "rect", (265.0, 150.0), (70.0, 32.0), 30.0),
    ("green_disc", (60, 160, 50), "disc", (320.0, 65.0), (16.0,), 0.0),
    ("red_plate", (40, 40, 200), "rect", (150.0, 210.0), (55.0, 25.0), -20.0),
    ("yellow_square", (40, 200, 230), "rect", (70.0, 165.0), (32.0, 32.0), 10.0),
    ("blue_disc", (190, 90, 30), "disc", (345.0, 235.0), (18.0,), 0.0),
]


def table_texture(ppm=4, parts=PARTS):
    x0, y0, x1, y1 = -350.0, -300.0, 750.0, 600.0
    w, h = int((x1 - x0) * ppm), int((y1 - y0) * ppm)
    noise = cv2.GaussianBlur(RNG.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 3)
    tex = np.dstack([175 + 10 * noise, 182 + 10 * noise, 188 + 10 * noise])
    to_px = lambda x, y: (int(round((x - x0) * ppm)), int(round((y - y0) * ppm)))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (cx, cy) in MARKERS.items():
        side = int(MARKER_SIZE * ppm)
        m = cv2.aruco.generateImageMarker(dictionary, mid, side)
        border = int(8 * ppm)
        m = cv2.copyMakeBorder(m, border, border, border, border, cv2.BORDER_CONSTANT, value=255)
        u, v = to_px(cx, cy)
        hs = m.shape[0] // 2
        tex[v - hs:v - hs + m.shape[0], u - hs:u - hs + m.shape[1]] = m[..., None] * 0.92 + 8
    for name, colour, shape, (cx, cy), size, angle in parts:
        if shape == "disc":
            cv2.circle(tex, to_px(cx, cy), int(size[0] * ppm), colour, -1, cv2.LINE_AA)
        else:
            box = cv2.boxPoints(((float((cx - x0) * ppm), float((cy - y0) * ppm)),
                                 (size[0] * ppm, size[1] * ppm), angle))
            cv2.fillPoly(tex, [np.round(box).astype(np.int32)], colour, cv2.LINE_AA)
    return np.clip(tex, 0, 255).astype(np.float32), ppm, (x0, y0)


TABLE_POSE = {"rx": -10.0, "ry": 7.0, "rz": 2.0, "look_at_mm": [200.0, 140.0, 0.0], "distance_mm": 455.0}


def make_table_set():
    R = rot(TABLE_POSE["rx"], TABLE_POSE["ry"], TABLE_POSE["rz"])
    axis = R.T @ np.array([0.0, 0.0, 1.0])  # optical axis in world frame
    C = np.array(TABLE_POSE["look_at_mm"]) - TABLE_POSE["distance_mm"] * axis
    t = -R @ C
    tex, ppm, origin = table_texture()
    bg = np.full((H, W, 3), 60, np.float32)
    clean = render_plane(tex, ppm, origin, R, t, bg)
    out = OUT / "table"
    out.mkdir(parents=True, exist_ok=True)
    for name, gain in [("table_normal", 1.0), ("table_dark", 0.38), ("table_bright", 1.55)]:
        cv2.imwrite(str(out / f"{name}.png"), photometric(clean, gain=gain))
    # Same scene with a strong illumination gradient (for adaptive thresholding / HSV)
    yy, xx = np.mgrid[0:H, 0:W]
    grad = (0.45 + 0.8 * xx / W)[..., None]
    cv2.imwrite(str(out / "table_gradient.png"), photometric(clean * grad))
    rvec, _ = cv2.Rodrigues(R)
    parts_truth = []
    for name, colour, shape, (cx, cy), size, angle in PARTS:
        P = np.array([[cx, cy, 0.0]])
        uv_true, _ = cv2.projectPoints(P, rvec, t, K, DIST)
        parts_truth.append({"name": name, "shape": shape, "x_mm": cx, "y_mm": cy,
                            "size_mm": list(size), "angle_deg": angle,
                            "u_px": float(uv_true[0, 0, 0]), "v_px": float(uv_true[0, 0, 1])})
    return {"rvec": rvec.ravel().tolist(), "tvec": t.tolist()}, parts_truth


# ----------------------------------------------------------------------------
# 3. Robot cell: a red circle among the 4 ArUco markers, and a simulated robot (L4, section 8)
# ----------------------------------------------------------------------------
# Same camera and table as section 2. The robot (Galatae Gal-e) stands at the top edge of the
# table. Table frame: X right, Y "down" in the image, Z into the table. Robot frame: z up.
# Table -> robot:  [x_r, y_r] = M [X, Y] + o,  z_r = Z_TABLE_ROBOT - Z   (M = rotation x swap, det -1)
ROBOT_THETA = 2.0  # deg: the robot is not perfectly aligned with the table
_c, _s = np.cos(np.deg2rad(ROBOT_THETA)), np.sin(np.deg2rad(ROBOT_THETA))
ROBOT_M = np.array([[_c, -_s], [_s, _c]]) @ np.array([[0.0, 1.0], [1.0, 0.0]])
ROBOT_O = np.array([160.0, -200.0])  # robot coordinates of the table origin (marker 0), mm
Z_TABLE_ROBOT = -6.5  # height of the table surface in the robot frame, mm
CIRCLE_RADIUS = 15.0  # mm (red sticker, diameter 30 mm)
ROBOT_SCENES = [  # red circle centre (mm, table frame) + distractors
    {"circle": (255.0, 175.0),
     "others": [("red_square", (40, 40, 200), "rect", (90.0, 120.0), (30.0, 30.0), 15.0),
                ("blue_disc", (190, 90, 30), "disc", (320.0, 70.0), (15.0,), 0.0)]},
    {"circle": (85.0, 215.0),
     "others": [("red_square", (40, 40, 200), "rect", (300.0, 210.0), (30.0, 30.0), -25.0),
                ("green_disc", (60, 160, 50), "disc", (180.0, 80.0), (15.0,), 0.0)]},
    {"circle": (350.0, 55.0),
     "others": [("red_square", (40, 40, 200), "rect", (200.0, 140.0), (30.0, 30.0), 40.0),
                ("yellow_square", (40, 200, 230), "rect", (110.0, 230.0), (30.0, 30.0), 0.0)]},
]


def table_to_robot(xy):
    return (ROBOT_M @ np.asarray(xy, float).T).T + ROBOT_O


def make_marker_sheet(path, px_per_mm=10):
    """A4 sheet (254 dpi = 10 px/mm) with the markers 0-3, 40 mm black squares, to print at 100 %."""
    from PIL import Image

    sheet = np.full((297 * px_per_mm, 210 * px_per_mm), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = int(MARKER_SIZE * px_per_mm)
    for k in range(4):
        x0 = (30 + (k % 2) * 90) * px_per_mm
        y0 = (40 + (k // 2) * 110) * px_per_mm
        sheet[y0:y0 + side, x0:x0 + side] = cv2.aruco.generateImageMarker(dictionary, k, side)
        cv2.putText(sheet, f"id {k}  (DICT_4X4_50, 40 mm)", (x0 - 40, y0 + side + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2, cv2.LINE_AA)
    cv2.putText(sheet, "Print at 100 % (no scaling) and check: black square = 40 mm",
                (150, 280 * px_per_mm), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    Image.fromarray(sheet).save(path, dpi=(25.4 * px_per_mm, 25.4 * px_per_mm))


def make_robot_set():
    rng = np.random.default_rng(21)
    R = rot(TABLE_POSE["rx"], TABLE_POSE["ry"], TABLE_POSE["rz"])
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    C = np.array(TABLE_POSE["look_at_mm"]) - TABLE_POSE["distance_mm"] * axis
    t = -R @ C
    out = OUT / "robot"
    out.mkdir(parents=True, exist_ok=True)
    bg = np.full((H, W, 3), 60, np.float32)
    scenes = []
    for k, sc in enumerate(ROBOT_SCENES, start=1):
        parts = [("red_circle", (40, 40, 200), "disc", sc["circle"], (CIRCLE_RADIUS,), 0.0)] + sc["others"]
        tex, ppm, origin = table_texture(parts=parts)
        img = photometric(render_plane(tex, ppm, origin, R, t, bg), gain=rng.uniform(0.9, 1.1))
        name = f"robot_cell_{k}.png"
        cv2.imwrite(str(out / name), img)
        scenes.append({"file": name, "circle_table_mm": list(sc["circle"]),
                       "circle_robot_mm": table_to_robot(sc["circle"]).round(2).tolist()})
    # What the students would measure by jogging the robot tool onto each marker centre (+ noise)
    taught = {str(i): (table_to_robot(xy) + rng.normal(0, 0.3, 2)).round(1).tolist()
                      + [round(Z_TABLE_ROBOT + rng.normal(0, 0.3), 1)]
              for i, xy in MARKERS.items()}
    (out / "taught_markers.json").write_text(json.dumps(
        {"comment": "Tool pose [x, y, z] (mm, robot frame) read with robot.get_tool_pose() "
                    "with the tool tip on the centre of each ArUco marker.", "markers": taught}, indent=2))
    make_marker_sheet(out / "aruco_markers_A4.png")
    return {"M": ROBOT_M.tolist(), "o_mm": ROBOT_O.tolist(), "z_table_mm": Z_TABLE_ROBOT,
            "circle_radius_mm": CIRCLE_RADIUS, "scenes": scenes}


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    calib_truth = make_calibration_set()
    table_pose, parts_truth = make_table_set()
    robot_truth = make_robot_set()
    truth = {
        "comment": "Synthetic data generated by tools/make_data.py. Units: mm and pixels.",
        "image_size": [W, H],
        "K": K.tolist(),
        "dist_k1_k2_p1_p2_k3": DIST.tolist(),
        "checkerboard": {"inner_corners": list(PATTERN), "square_mm": SQUARE,
                         "bad_views": ["calib_20.png (board partly outside)",
                                       "calib_21.png (motion blur)"],
                         "poses": calib_truth},
        "table": {"pose": table_pose, "marker_size_mm": MARKER_SIZE,
                  "aruco_dict": "DICT_4X4_50",
                  "marker_centres_mm": {str(k): v for k, v in MARKERS.items()},
                  "parts": parts_truth},
        "robot": robot_truth,
    }
    (OUT / "synthetic_ground_truth.json").write_text(json.dumps(truth, indent=2))
    print("written to", OUT)
