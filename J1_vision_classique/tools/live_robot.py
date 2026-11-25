"""Lab 4, live: the camera finds a coloured shape on the board and the Galatae arm goes above it.

    python live_robot.py

Nothing else to prepare: the program does not need the notebook of Lab 4.

Everything happens in the video window, in 4 steps. Each step is a CHECKPOINT: the result is shown on the
video and in the terminal, and nothing goes on before you press ENTER (BACKSPACE: previous step, q/ESC: quit).

  1 INTRINSICS  K and distortion of the camera.   c: save a view of the checkerboard   k: calibrate
                                                  u: show the undistorted video
  2 CAMERA      position of the camera in the robot frame (the 4 markers must be visible)
  3 OBJECTS     the coloured circles and squares found on the board; ENTER freezes the picture
  4 ROBOT       on the frozen picture, every object has a number.
                1-9: choose an object: the robot axes (x red, y green, z blue) and its coordinates are drawn
                r: connect the arm and home it (arm in its REST position!)   SPACE: go above the chosen object
                then any key folds the arm and goes back to step 3
                (the unshifted digit keys of an AZERTY keyboard work too)

Ctrl+C, an error or a closed window brings the arm back to its rest position (safe_robot.py).
The Python API of the arm (robot.py, https://github.com/galatae-engineering/galatae-api) must be installed.
"""
import argparse
import contextlib
import threading
from pathlib import Path

import cv2
import numpy as np
from robot import Robot

from camera import DEFAULT_CAMERA, open_camera
from safe_robot import SafeRobot

HERE = Path(__file__).resolve().parent
NOTEBOOKS = HERE.parent / "notebooks"
OUT = NOTEBOOKS if NOTEBOOKS.is_dir() else HERE

# ---- the real cell (placement plan of robot_cell_cutouts_A4.pdf), in mm ---------------------------------
MARKERS = {0: (0.0, 0.0), 1: (340.0, 0.0), 2: (340.0, 260.0), 3: (0.0, 260.0)}  # centres on the board
BASE_TO_BOARD = 250.0  # from the axis of the robot base to the line through markers 0 and 1
TABLE_Z = -120.0       # height of the table in the robot frame
HOVER = 60.0           # height above the table when pointing at the object: start high
SAFE = 150.0           # height above the table for the approach and the retreat
SPEED = 20             # deg/s
REACH = (120.0, 550.0)  # allowed distance from the base axis
SHAPE_MM = 30.0        # diameter of the circles, side of the squares

# HSV ranges of the printed colours (OpenCV: H in 0..179): adapt them to your lighting
COLOURS = {"red": [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))],
           "blue": [((100, 100, 50), (130, 255, 255))],
           "green": [((40, 80, 40), (85, 255, 255))],
           "yellow": [((18, 100, 80), (36, 255, 255))]}
BGR = {"red": (0, 0, 255), "blue": (255, 80, 0), "green": (0, 180, 0), "yellow": (0, 220, 255)}
ENTER, BACKSPACE, SPACE, ESC, NO_KEY = (10, 13), (8, 127), ord(" "), 27, 255
DIGITS = {**{ord(str(n)): n for n in range(1, 10)},   # 1-9, and the same keys of an AZERTY keyboard without Shift
          **{ord(c): n for n, c in enumerate("&\u00e9\"'(\u00a7\u00e8!\u00e7", 1)}, ord("-"): 6, ord("_"): 8}

# Robot frame, facing the robot: x towards you, y to your right, z up. The board is centred in front of the
# base and not rotated: board Y -> robot x, board X -> robot y.
HALF = MARKERS[1][0] / 2
MARKERS_ROBOT = {i: (BASE_TO_BOARD + Y, X - HALF) for i, (X, Y) in MARKERS.items()}


def robot_to_board(xyz):
    """N x 3 robot coordinates -> board coordinates (X right, Y towards you, Z into the table)."""
    x, y, z = np.asarray(xyz, float).reshape(-1, 3).T
    return np.c_[y + HALF, x - BASE_TO_BOARD, TABLE_Z - z]


def board_to_robot(XYZ):
    """N x 3 board coordinates -> robot coordinates."""
    X, Y, Z = np.asarray(XYZ, float).reshape(-1, 3).T
    return np.c_[BASE_TO_BOARD + Y, X - HALF, TABLE_Z - Z]


def log(text):
    print(text, flush=True)


# ---- step 1: calibration of the camera ------------------------------------------------------------------
class Calibration:
    def __init__(self, K, dist, size, rms):
        self.K, self.dist, self.size, self.rms = K, dist, size, rms

    def report(self):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        return [f"RMS = {self.rms:.3f} px  ({'good' if self.rms < 0.5 else 'high: remove the bad views'})",
                f"fx = {fx:.1f}  fy = {fy:.1f}  cx = {cx:.1f}  cy = {cy:.1f} px",
                "dist = " + " ".join(f"{v:.3f}" for v in self.dist.ravel())]


def load_calibration(path):
    if not Path(path).exists():
        return None
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    K, dist, rms = fs.getNode("K").mat(), fs.getNode("dist").mat(), fs.getNode("rms").real()
    size = int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())
    fs.release()
    return Calibration(K, dist, size, rms)


def calibrate(folder, pattern, square, out, size):
    """Images of the checkerboard -> Calibration, saved in the file out. ValueError if it is not possible.

    Only the images of the given size (width, height) are used: K is only valid for one size."""
    files = sorted(Path(folder).glob("*.png"))
    if not files:
        raise ValueError(f"no .png image in {Path(folder).resolve()}")
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    points, used, others = [], [], []
    for f in files:
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if gray.shape[::-1] != size:
            others.append(f"{f.name} {gray.shape[1]}x{gray.shape[0]}")
            continue
        found, c = cv2.findChessboardCorners(gray, pattern)
        if found:
            points.append(cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), (3, 30, 1e-3)))
            used.append(f.name)
    if others:
        log(f"  ignored, not {size[0]}x{size[1]} like the video: {', '.join(others)}")
    if len(points) < 5:
        raise ValueError(f"{len(points)} usable views of {size[0]}x{size[1]} out of {len(files)} (5 needed, 15-25 advised)")
    rms, K, dist, *_, per_view = cv2.calibrateCameraExtended([objp] * len(points), points, size, None, None)
    for name, e in zip(used, per_view.ravel()):
        log(f"  {name}: {e:.2f} px" + ("   <- bad view: delete it and calibrate again" if e > 1 else ""))
    fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0]), fs.write("image_height", size[1])
    fs.write("K", K), fs.write("dist", dist), fs.write("rms", rms)
    fs.release()
    log(f"{len(points)} views, saved in {out}")
    return Calibration(K, dist, size, rms)


# ---- steps 2 and 3: what the camera sees ----------------------------------------------------------------
class Board:
    """The board in one frame: markers, homography and pose (None attributes if a marker is hidden)."""

    def __init__(self, frame, calib, detector):
        self.K, self.dist = calib.K, calib.dist
        self.corners, self.ids, _ = detector.detectMarkers(frame)
        ids = [] if self.ids is None else [int(i) for i in self.ids.ravel()]
        seen = {i: c.reshape(4, 2).mean(0) for c, i in zip(self.corners, ids)}
        self.missing = [i for i in MARKERS if i not in seen]
        self.H = None
        if self.missing:
            return
        obj, img = np.array([(*MARKERS[i], 0.0) for i in MARKERS]), np.array([seen[i] for i in MARKERS], float)
        self.H = cv2.getPerspectiveTransform(self.undistort(img), obj[:, :2].astype(np.float32))  # -> board mm
        _, self.rvec, self.tvec = cv2.solvePnP(obj, img, self.K, self.dist)
        px, _ = cv2.projectPoints(obj, self.rvec, self.tvec, self.K, self.dist)
        self.error_px = np.linalg.norm(px.reshape(-1, 2) - img, axis=1).mean()

    def camera(self):
        """Position of the camera in the robot frame (mm) and tilt of the optical axis from the vertical (deg)."""
        R, _ = cv2.Rodrigues(self.rvec)
        axis = board_to_robot(R.T @ [0, 0, 1])[0] - board_to_robot([0, 0, 0])[0]
        return board_to_robot((-R.T @ self.tvec).ravel())[0], np.degrees(np.arccos(np.clip(-axis[2], -1, 1)))

    def project(self, xyz):
        """Robot coordinates -> pixels."""
        px, _ = cv2.projectPoints(robot_to_board(xyz), self.rvec, self.tvec, self.K, self.dist)
        return [tuple(int(round(v)) for v in p) for p in px.reshape(-1, 2)]

    def to_mm(self, uv):
        """Pixels -> board mm."""
        return cv2.perspectiveTransform(self.undistort(uv)[:, None], self.H).reshape(-1, 2)

    def undistort(self, uv):
        points = np.asarray(uv, np.float32).reshape(-1, 1, 2)
        return cv2.undistortPoints(points, self.K, self.dist, P=self.K).reshape(-1, 2)

    def draw_markers(self, frame):
        if self.ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, self.corners, self.ids)

    def draw_axes(self, frame, origin, length=50.0):
        """Robot axes (x red, y green, z blue) drawn at a point given in the robot frame."""
        o = np.array(origin, float)
        px = self.project([o, o + [length, 0, 0], o + [0, length, 0], o + [0, 0, length]])
        for end, name, colour in zip(px[1:], "xyz", [(0, 0, 255), (0, 200, 0), (255, 80, 0)]):
            cv2.arrowedLine(frame, px[0], end, colour, 3, cv2.LINE_AA, tipLength=0.2)
            cv2.putText(frame, name, (end[0] + 6, end[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2, cv2.LINE_AA)


def find_shapes(frame, board):
    """The coloured shapes between the markers: list of dicts (colour, shape, uv in px, XY on the board in mm)."""
    hsv, shapes = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), []
    for colour, ranges in COLOURS.items():
        mask = sum(cv2.inRange(hsv, np.array(lo), np.array(hi)) for lo, hi in ranges)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            if cv2.contourArea(c) < 60:
                continue
            mm = board.to_mm(c.reshape(-1, 2)).astype(np.float32).reshape(-1, 1, 2)
            (_, sides, _), area = cv2.minAreaRect(mm), cv2.contourArea(mm)
            if not all(0.75 * SHAPE_MM < s < 1.25 * SHAPE_MM for s in sides):
                continue  # not the size of our shapes
            m = cv2.moments(c)
            uv = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])
            XY = board.to_mm(uv)[0]
            if not (0 <= XY[0] <= MARKERS[2][0] and 0 <= XY[1] <= MARKERS[2][1]):
                continue  # not between the markers
            extent = area / (sides[0] * sides[1])  # part of the bounding square that is filled: pi/4 for a disc
            shapes.append({"colour": colour, "shape": "circle" if extent < 0.89 else "square", "uv": uv, "XY": XY,
                           "error": abs(np.mean(sides) - SHAPE_MM)})
    return shapes


def stable(shapes, history, seen=4):
    """The shapes also found (same colour and shape, less than 8 mm away) in at least `seen` previous pictures."""
    def same(a, b):
        return a["colour"] == b["colour"] and a["shape"] == b["shape"] and np.linalg.norm(a["XY"] - b["XY"]) < 8
    return [a for a in shapes if sum(any(same(a, b) for b in old) for old in history) >= seen]


def label(frame, item, number=None, chosen=False):
    u, v = (int(round(c)) for c in item["uv"])
    text = f"{item['colour']} {item['shape']}" + (f"  [{number}]" if number else "")
    cv2.circle(frame, (u, v), 4, BGR[item["colour"]], -1)
    for colour, thick in (((0, 0, 0), 5), ((255, 255, 255) if chosen else BGR[item["colour"]], 2)):
        cv2.putText(frame, text, (u + 18, v - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, thick, cv2.LINE_AA)


def write(frame, lines, bottom=()):
    h = frame.shape[0]
    for i, text in enumerate([*lines, *bottom]):
        y = 32 + 30 * i if i < len(lines) else h - 18 - 30 * (len(lines) + len(bottom) - 1 - i)
        for colour, thick in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, thick, cv2.LINE_AA)


# ---- step 4: the arm ------------------------------------------------------------------------------------
class Motion(threading.Thread):
    """Movements run beside the video. An error or Ctrl+C is kept and raised by the main program."""

    def __init__(self, function):
        super().__init__(daemon=True)
        self.function, self.error = function, None
        self.start()

    def run(self):
        try:
            self.function()
        except BaseException as e:
            self.error = e


def go_above(arm, xy, height):
    """Approach from above, point at the object, retreat (tool pointing down)."""
    r = float(np.hypot(*xy))
    if not REACH[0] <= r <= REACH[1]:
        raise ValueError(f"out of reach: {r:.0f} mm from the base (allowed: {REACH[0]:.0f} to {REACH[1]:.0f})")
    for h in (SAFE, height, SAFE):
        arm.go_to_pose([float(xy[0]), float(xy[1]), TABLE_Z + h, 180.0, 0.0])


def leave_view(arm, clear=SAFE):
    """Lift the tool vertically, then fold the arm: the camera sees the whole board again."""
    pose = list(arm.get_tool_pose())[:5]
    if pose[2] < TABLE_Z + clear:
        arm.go_to_pose(pose[:2] + [TABLE_Z + clear] + pose[3:])
    arm.go_to_sleep_pos()


def prepare(arm):
    arm.reset()
    arm.home()
    arm.set_joint_speed(SPEED)
    arm.go_to_sleep_pos()  # clear view of the board


def wait_for_motion(arm):
    """Before the return to the rest position: the current movement must be finished (not the next ones)."""
    for th in threading.enumerate():
        if isinstance(th, Motion):
            arm._stops = max(arm._stops, 1)
            try:
                th.join()
            except KeyboardInterrupt:  # repeated Ctrl+C: do not read the answers of the arm from two threads
                arm._desync = True


# ---- the program ----------------------------------------------------------------------------------------
class App:
    TITLE = "live robot"

    def __init__(self, args, cap, exits):
        self.args, self.cap, self.exits = args, cap, exits
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        self.calib = load_calibration(args.calibration)
        self.views = Path(args.views)
        self.undistort, self.message = False, ""
        self.camera = None            # validated position of the camera (robot frame)
        self.history = []             # shapes found in the previous pictures
        self.frozen = None            # frozen picture, its board and its numbered objects
        self.choice = None            # index of the chosen object
        self.arm, self.motion = None, None
        self.check_size()
        self.enter("intrinsics")

    def check_size(self):
        """Show the size of the video; if the calibration has another size, ask the camera for that size."""
        size = self.cap.read()[1].shape[1::-1]
        if self.calib is not None and self.calib.size != size and min(self.calib.size) > 0:
            log(f"the calibration is for {self.calib.size}, the camera gives {size}: asking for {self.calib.size}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib.size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib.size[1])
            for _ in range(5):
                ok, frame = self.cap.read()
            size = frame.shape[1::-1] if ok else size
        log(f"camera {self.args.camera}: {size[0]} x {size[1]}, opened with {self.cap.getBackendName()}, OpenCV {cv2.__version__}")

    def enter(self, step):
        self.step, self.message = step, ""
        title = {"intrinsics": "STEP 1 - INTRINSICS", "camera": "STEP 2 - CAMERA IN THE ROBOT FRAME",
                 "object": "STEP 3 - OBJECTS", "robot": "STEP 4 - ROBOT"}.get(step)
        if title:
            log(f"\n=== {title} " + "=" * (60 - len(title)))
        if step == "intrinsics":
            log("\n".join(self.calib.report()) if self.calib else "no calibration yet: c = save a view, k = calibrate")

    def say(self, text):
        self.message = text
        log(text)

    # ---- step 1
    def intrinsics(self, frame, key):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, self.args.pattern,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        n = len(list(self.views.glob("*.png"))) if self.views.is_dir() else 0
        size = frame.shape[1::-1]
        valid = self.calib is not None and self.calib.size == size
        mismatch = "" if valid or self.calib is None else \
            f"calibrated in {self.calib.size[0]}x{self.calib.size[1]}, but the video is {size[0]}x{size[1]}: c = new views, k = calibrate"
        if key == ord("c"):
            if found:
                self.views.mkdir(parents=True, exist_ok=True)
                k = next(i for i in range(10000) if not (self.views / f"calib_{i:02d}.png").exists())
                cv2.imwrite(str(self.views / f"calib_{k:02d}.png"), frame)
                self.say(f"view {n + 1} saved in {self.views.name}")
            else:
                self.say("checkerboard not found: view not saved")
        elif key == ord("k"):
            try:
                self.calib = calibrate(self.views, self.args.pattern, self.args.square, self.args.calibration, size)
                valid, mismatch = True, ""
                self.say("calibrated: check the RMS and the undistorted video (u)")
                log("\n".join(self.calib.report()))
            except (ValueError, cv2.error) as e:
                self.say(f"calibration not possible: {e}")
        elif key == ord("u"):
            self.undistort = valid and not self.undistort
            if not valid:
                self.say(mismatch or "not calibrated: c = save views, k = calibrate")
        elif key in ENTER:
            if valid:
                self.undistort = False
                return self.enter("camera")
            self.say(mismatch or "not calibrated: c = save views, k = calibrate")
        if self.undistort and valid:
            frame = cv2.undistort(frame, self.calib.K, self.calib.dist)
        elif found:
            cv2.drawChessboardCorners(frame, self.args.pattern, corners, found)
        lines = ["1 INTRINSICS" + ("  (undistorted video)" if self.undistort else "")]
        lines += self.calib.report() if self.calib else ["not calibrated"]
        lines.append(f"video {size[0]}x{size[1]}" if valid or self.calib is None else mismatch)
        write(frame, lines, [self.message, f"{n} views   c: save view   k: calibrate   u: undistort   ENTER: validate   q: quit"])
        return frame

    # ---- step 2
    def pose(self, frame, key):
        board = Board(frame, self.calib, self.detector)
        board.draw_markers(frame)
        lines = ["2 CAMERA IN THE ROBOT FRAME"]
        if board.missing:
            lines.append(f"markers not visible: {board.missing}")
        else:
            (x, y, z), tilt = board.camera()
            board.draw_axes(frame, (*MARKERS_ROBOT[0], TABLE_Z))
            board.draw_axes(frame, (BASE_TO_BOARD + MARKERS[2][1] / 2, 0, TABLE_Z), 80)
            lines += [f"camera: x = {x:.0f}  y = {y:.0f}  z = {z:.0f} mm  ({z - TABLE_Z:.0f} mm above the table)",
                      f"tilt from the vertical: {tilt:.0f} deg   marker error: {board.error_px:.1f} px"]
        if key in BACKSPACE:
            return self.enter("intrinsics")
        if key in ENTER:
            if board.missing:
                self.say(f"markers not visible: {board.missing}")
            else:
                self.camera = np.array([x, y, z])
                self.log_camera(board)
                return self.enter("object")
        write(frame, lines, [self.message, "ENTER: validate   BACKSPACE: step 1   q: quit"])
        return frame

    def log_camera(self, board):
        (x, y, z), tilt = board.camera()
        log(f"camera in the robot frame: x = {x:.0f}  y = {y:.0f}  z = {z:.0f} mm ({z - TABLE_Z:.0f} mm above the "
            f"table), tilt {tilt:.0f} deg, marker error {board.error_px:.1f} px")

    # ---- step 3
    def objects(self, frame, key):
        raw = frame.copy()
        if key in BACKSPACE:
            return self.enter("camera")
        board = Board(raw, self.calib, self.detector)
        board.draw_markers(frame)
        lines, items = ["3 OBJECTS"], []
        if board.missing:
            lines.append(f"markers not visible: {board.missing}")
        else:
            moved = np.linalg.norm(board.camera()[0] - self.camera)
            if moved > 15:
                lines.append(f"the camera has moved by {moved:.0f} mm: BACKSPACE and check step 2")
            shapes = find_shapes(raw, board)
            items = stable(shapes, self.history)  # the noise does not stay several pictures at the same place
            self.history = [*self.history, shapes][-6:]
            lines.append(f"{len(items)} objects" + ("" if items else " (lighting? thresholds COLOURS?)"))
        for item in items:
            label(frame, item)
        if key in ENTER:
            if not items:
                self.say("nothing to freeze: the 4 markers and at least one object must be visible")
            else:
                items = sorted(sorted(items, key=lambda s: s["error"])[:9],  # 9 keys: keep the best sizes
                               key=lambda s: (list(COLOURS).index(s["colour"]), s["shape"], s["XY"][0]))
                board.draw_markers(raw)
                self.frozen, self.choice = (raw, board, items), None
                self.enter("robot")
                self.log_camera(board)
                for n, item in enumerate(items, 1):
                    x, y, _ = self.robot_xy(item)
                    log(f"  [{n}] {item['colour']} {item['shape']}: robot x = {x:.0f}  y = {y:.0f} mm")
                return None
        write(frame, lines, [self.message, "ENTER: freeze, then choose a number   BACKSPACE: step 2   q: quit"])
        return frame

    @staticmethod
    def robot_xy(item):
        """(x, y, distance to the base) of an object, robot frame."""
        x, y, _ = board_to_robot([[*item["XY"], 0]])[0]
        return x, y, np.hypot(x, y)

    # ---- step 4
    def robot(self, frame, key):
        picture, board, items = self.frozen
        frame = picture.copy()
        if key in BACKSPACE:
            return self.enter("object")
        if DIGITS.get(key, 99) <= len(items):
            self.choice, self.message = DIGITS[key] - 1, ""
            item = items[self.choice]
            x, y, r = self.robot_xy(item)
            log(f"chosen: [{self.choice + 1}] {item['colour']} {item['shape']}, pixel ({item['uv'][0]:.0f}, "
                f"{item['uv'][1]:.0f}), board ({item['XY'][0]:.0f}, {item['XY'][1]:.0f}) mm, robot ({x:.1f}, {y:.1f}) mm")
        for n, item in enumerate(items, 1):
            label(frame, item, n, chosen=n - 1 == self.choice)
        lines, goal = ["4 ROBOT (frozen picture): choose a number"], None
        if self.choice is not None:
            item = items[self.choice]
            x, y, r = self.robot_xy(item)
            board.draw_axes(frame, (x, y, TABLE_Z))
            lines = [f"4 ROBOT: [{self.choice + 1}] {item['colour']} {item['shape']}",
                     f"pixel ({item['uv'][0]:.0f}, {item['uv'][1]:.0f})   board X = {item['XY'][0]:.0f}  Y = {item['XY'][1]:.0f} mm",
                     f"robot x = {x:.0f}  y = {y:.0f}  z = {TABLE_Z:.0f} mm   (tool: {self.args.hover:.0f} mm above the table)"]
            if REACH[0] <= r <= REACH[1]:
                goal = (x, y)
            else:
                lines.append(f"out of reach: {r:.0f} mm from the base")
        if key == ord("r") and self.arm is None:
            self.connect()
            return None
        if key == SPACE:
            if self.arm is None:
                self.say("connect the arm first: r")
            elif goal is None:
                self.say("choose an object (within reach) first")
            else:
                log(f"going above ({goal[0]:.1f}, {goal[1]:.1f})")
                self.step, self.history = "moving", []
                self.motion = Motion(lambda: go_above(self.arm, goal, self.args.hover))
                return None
        help_ = "SPACE: go above" if self.arm is not None else "arm in REST position, cell clear?   r: connect"
        write(frame, lines, [self.message, f"1-{len(items)}: choose   {help_}   BACKSPACE: step 3   q: quit"])
        return frame

    def connect(self):
        log("connecting the arm...")
        self.arm = self.exits.enter_context(SafeRobot(Robot()))  # always ends in the rest position
        self.exits.callback(wait_for_motion, self.arm)           # (runs before the return to rest)
        self.step, self.motion = "homing", Motion(lambda: prepare(self.arm))

    def busy(self, frame, key):
        """A movement is running (the video goes on), or the tool is above the target."""
        if self.motion is not None and not self.motion.is_alive():
            error, self.motion = self.motion.error, None
            if error is not None:
                raise error
            if self.step == "homing":
                log("arm ready")
                return self.enter("robot")
            if self.step == "folding":
                return self.enter("object")
            self.step = "above"
            log("done: press any key to fold the arm")
        elif self.step == "above" and key != NO_KEY:
            self.step, self.motion = "folding", Motion(lambda: leave_view(self.arm))
        write(frame, [{"homing": "homing...", "moving": "moving above the target...", "folding": "folding...",
                       "above": "done: check with a ruler, then press any key to fold the arm"}[self.step]])
        return frame

    def run(self):
        handlers = {"intrinsics": self.intrinsics, "camera": self.pose, "object": self.objects, "robot": self.robot}
        key = NO_KEY
        while True:
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("no image received from the camera")
            shown = handlers.get(self.step, self.busy)(frame, key)
            if shown is not None:
                cv2.imshow(self.TITLE, shown)
            key = cv2.waitKey(1) & 0xFF
            if shown is not None and cv2.getWindowProperty(self.TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ESC):
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help=f"camera index (default {DEFAULT_CAMERA})")
    ap.add_argument("--hover", type=float, default=HOVER, help=f"height above the table, mm (default {HOVER:.0f})")
    ap.add_argument("--views", default=str(OUT / "cell_calib"), help="folder of the checkerboard views")
    ap.add_argument("--calibration", default=str(OUT / "cell_calibration.yaml"), help="calibration file")
    ap.add_argument("--pattern", type=int, nargs=2, default=(9, 6), help="inner corners of the checkerboard")
    ap.add_argument("--square", type=float, default=25.0, help="side of the squares (mm)")
    args = ap.parse_args()
    args.pattern = tuple(args.pattern)
    if not 20 <= args.hover <= SAFE - 20:
        raise SystemExit(f"--hover: between 20 and {SAFE - 20:.0f} mm")

    cap = open_camera(args.camera)
    if not cap.read()[0]:
        raise SystemExit(f"no image from camera {args.camera}: run test_cameras.py to find the right index")
    try:
        with contextlib.ExitStack() as exits:
            App(args, cap, exits).run()
    except KeyboardInterrupt:
        log("Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
