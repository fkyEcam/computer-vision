"""Which camera works, and which one fits the labs and the Galatae robot cell?

    python test_cameras.py                      # test every camera (nothing is saved)
    python test_cameras.py --show 1             # live preview of camera 1: markers and checkerboard
    python test_cameras.py --show 1 --distance 455   # + field of view (lens at 455 mm from the sheet)

In the preview, show the sheet notebooks/data/robot/aruco_markers_A4.png (printed at 100 %: 40 mm
markers) flat in front of the camera, at the distance it will have above the table.

The cameras are opened exactly as in the labs (DirectShow, 1280 x 720), so the index printed here
is the one to use: CAMERA_INDEX (Lab 1), --camera / CAMERA_ID (Lab 4), CAMERA_ID (Lab 8).
"""
import argparse
import ctypes
import os
import sys
import time
import uuid

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

W, H = 1280, 720                 # image size used in Labs 1, 4 and 8
MIN_FPS = 10                     # comfortable preview for capture_calibration.py
TABLE_MM = (380, 300)            # 340 x 260 mm between marker centres + the 40 mm markers
OTHER_SIZES = [(640, 480), (1920, 1080), (2592, 1944), (3840, 2160)]
PATTERN = (9, 6)                 # inner corners of the checkerboard of Lab 4


def directshow_names():
    """Camera names in DirectShow order (= OpenCV CAP_DSHOW order). Empty list if unavailable."""
    if sys.platform != "win32":
        return []
    ole, oleaut = ctypes.OleDLL("ole32"), ctypes.WinDLL("oleaut32")
    ptr = ctypes.c_void_p

    def guid(s):
        return (ctypes.c_byte * 16).from_buffer_copy(uuid.UUID(s).bytes_le)

    def call(obj, index, *args, types=()):
        """Call the method number `index` of a COM object."""
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ptr)))[0]
        return ctypes.WINFUNCTYPE(ctypes.c_long, ptr, *types)(table[index])(obj, *args)

    class Variant(ctypes.Structure):
        _fields_ = [("vt", ctypes.c_ushort), ("pad", ctypes.c_ushort * 3), ("ptr", ptr), ("pad2", ptr)]

    oleaut.SysFreeString.argtypes = [ptr]
    names = []
    try:
        ole.CoInitialize(None)
    except OSError:
        pass  # already initialised in another mode
    try:
        dev, enum = ptr(), ptr()
        ole.CoCreateInstance(guid("62BE5D10-60EB-11d0-BD3B-00A0C911CE86"), None, 1,
                             guid("29840822-5B84-11D0-BD3B-00A0C911CE86"), ctypes.byref(dev))
        call(dev, 3, guid("860BB310-5D01-11d0-BD3B-00A0C911CE86"), ctypes.byref(enum), 0,
             types=(ptr, ptr, ctypes.c_ulong))                       # CreateClassEnumerator (video input)
        while enum:
            moniker, bag, v = ptr(), ptr(), Variant()
            if call(enum, 3, 1, ctypes.byref(moniker), None, types=(ctypes.c_ulong, ptr, ptr)) != 0:
                break                                                # Next
            name = "?"
            if call(moniker, 9, None, None, guid("55272A00-42CB-11CE-8135-00AA004BB851"),
                    ctypes.byref(bag), types=(ptr, ptr, ptr, ptr)) == 0:   # BindToStorage
                if call(bag, 3, "FriendlyName", ctypes.byref(v), None,
                        types=(ctypes.c_wchar_p, ptr, ptr)) == 0:    # Read
                    name = ctypes.wstring_at(v.ptr)
                    oleaut.SysFreeString(v.ptr)
                call(bag, 2)
            call(moniker, 2)
            names.append(name)
        for obj in (enum, dev):
            if obj:
                call(obj, 2)
    except OSError:
        pass
    return names


def open_camera(index, size=(W, H)):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    return cap


def grab(cap, n, warmup=15):
    """n frames after a warm-up (auto-exposure), and the measured frame rate."""
    for _ in range(warmup):
        cap.read()
    frames, t0 = [], time.perf_counter()
    for _ in range(n):
        ok, f = cap.read()
        if ok and f is not None:
            frames.append(f)
    return frames, len(frames) / (time.perf_counter() - t0)


def test_camera(index, name):
    """Print the report of one camera; return True if it passes every automatic check."""
    print(f"\n=== camera {index}: {name} " + "=" * max(0, 50 - len(name)))
    cap = open_camera(index)
    frames, fps = grab(cap, 30) if cap.isOpened() else ([], 0)
    if not frames:
        cap.release()
        print("  [FAIL] no image (unplugged, used by another program, or not a normal video camera)")
        return False
    f = frames[-1]
    h, w = f.shape[:2]
    img = f.astype(float)
    colour = np.abs(img[..., 0] - img[..., 2]).mean()           # 0 for an infrared / grey camera
    noise = np.mean([np.abs(a.astype(float) - b).mean() for a, b in zip(frames, frames[1:])])
    focus, autofocus = cap.get(cv2.CAP_PROP_FOCUS), cap.get(cv2.CAP_PROP_AUTOFOCUS)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    cap.release()

    lit = 15 < img.mean() < 240
    checks = [
        (True, "opens and delivers images"),
        (noise > 0, f"live image, not frozen (change between frames {noise:.2f})"),
        (lit, f"not black / saturated (mean grey {img.mean():.0f}): open the shutter, switch the light on"
         if not lit else f"brightness OK (mean grey {img.mean():.0f})"),
        ((w, h) == (W, H), f"{W} x {H} requested, {w} x {h} received"),
        (colour > 1.0, f"colour image (|B - R| = {colour:.1f}): needed for the red circle and HSV"),
        (fps >= MIN_FPS, f"{fps:.0f} images/s (at least {MIN_FPS})"),
    ]
    for ok, text in checks:
        print(f"  [{' OK ' if ok else 'FAIL'}] {text}")

    if focus < 0:
        print("  [info] no focus control seen by the driver: probably fixed focus (good). "
              "If the lens has a ring, lock it before calibrating")
    else:
        print(f"  [info] focus can be controlled (value {focus:.0f}, autofocus flag {autofocus:.0f}): "
              "autofocus must stay OFF after calibration")
    print("  [info] exposure " + ("cannot be read" if exposure == -1 else f"value {exposure:.0f}")
          + f", sharpness {cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var():.0f} "
            "(depends on the scene)")

    sizes = []
    for s in OTHER_SIZES:
        cap = open_camera(index, s)
        ok, g = cap.read()
        cap.release()
        if ok and g.shape[1::-1] == s:
            sizes.append(f"{s[0]}x{s[1]}")
    print("  [info] other sizes:", ", ".join(sizes) or "none")
    return all(ok for ok, _ in checks)


def show(index, marker_mm, distance):
    """Live preview with the ArUco markers and the checkerboard of the labs."""
    cap = open_camera(index)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {index}")
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    print("q / ESC : quit. Markers must face the camera (not tilted) for the field of view.")
    last = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(frame)
        found, board = cv2.findChessboardCorners(
            gray, PATTERN, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        lines = [f"camera {index}  {w}x{h}  markers: {0 if ids is None else len(ids)}/4  "
                 f"board: {'yes' if found else 'no'}"]
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            side = np.mean([np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), -1, axis=0), axis=1).mean()
                            for c in corners])
            mm_px = marker_mm / side
            vw, vh = w * mm_px, h * mm_px
            fits = vw >= TABLE_MM[0] and vh >= TABLE_MM[1]
            lines.append(f"{mm_px:.2f} mm/px, sees {vw:.0f} x {vh:.0f} mm "
                         f"({'OK' if fits else 'too close'} for {TABLE_MM[0]} x {TABLE_MM[1]} mm)")
            if distance:
                need = distance * max(TABLE_MM[0] / vw, TABLE_MM[1] / vh)
                lines.append(f"horizontal FOV {2 * np.degrees(np.arctan(vw / 2 / distance)):.0f} deg, "
                             f"minimum height {need:.0f} mm")
        if found:
            cv2.drawChessboardCorners(frame, PATTERN, board, found)
        for i, text in enumerate(lines):
            for colour, thick in (((0, 0, 0), 5), ((0, 255, 255), 2)):
                cv2.putText(frame, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, thick)
        last = lines
        cv2.imshow(f"camera {index}", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\n".join(last or ["no image"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, metavar="INDEX", help="live preview of one camera")
    ap.add_argument("--marker-mm", type=float, default=40.0, help="side of the printed marker (mm)")
    ap.add_argument("--distance", type=float, help="lens to sheet distance (mm), for the field of view")
    args = ap.parse_args()

    print(f"OpenCV {cv2.__version__}" + ("   [WARNING] the labs need opencv-contrib-python<5"
                                         if int(cv2.__version__.split(".")[0]) >= 5 else ""))
    if args.show is not None:
        return show(args.show, args.marker_mm, args.distance)

    names = directshow_names() or ["(name unknown)"] * 5
    result = {i: test_camera(i, n) for i, n in enumerate(names)}
    print("\n=== summary " + "=" * 52)
    for i, n in enumerate(names):
        print(f"  camera {i}  {n:<28} {'passes the automatic checks' if result[i] else 'NOT suitable as is'}")
    print("Not tested here: field of view and marker detection -> python test_cameras.py --show INDEX")


if __name__ == "__main__":
    main()
