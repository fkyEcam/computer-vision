"""Capture checkerboard views for camera calibration (Lab 4, see live_robot.py).

    python capture_calibration.py --out cell_calib --pattern 9 6

Live preview; the detected corners are drawn in green when the full pattern is found.
  SPACE : save the current frame (only if the pattern is detected)
  q/ESC : quit
Tips: 15-25 views, tilt the board up to ~45 degrees, vary the distance, cover the image corners.
Disable autofocus if your camera allows it (the focus must not change during and after calibration).
"""
import argparse
from pathlib import Path

import cv2

from camera import DEFAULT_CAMERA, open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=DEFAULT_CAMERA, help=f"camera index (default {DEFAULT_CAMERA})")
    ap.add_argument("--out", default="my_calib", help="output folder")
    ap.add_argument("--pattern", type=int, nargs=2, default=(9, 6), help="inner corners: columns rows")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # ignored by cameras without this control
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {args.camera}")

    pattern = tuple(args.pattern)
    count = len(list(out.glob("*.png")))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK)
        vis = frame.copy()
        cv2.drawChessboardCorners(vis, pattern, corners, found)
        cv2.putText(vis, f"saved: {count}  [SPACE] save  [q] quit", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calibration capture", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" ") and found:
            name = out / f"calib_{count:02d}.png"
            cv2.imwrite(str(name), frame)  # save the raw frame, not the annotated one
            print("saved", name)
            count += 1
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
