"""Open a camera in the same way on Windows and on Linux (used by the other scripts of this folder)."""
import sys

import cv2

WINDOWS = sys.platform == "win32"
DEFAULT_CAMERA = 1 if WINDOWS else 0  # Windows laptops: 0 = built-in webcam, 1 = USB camera


def open_camera(index, width=1280, height=720):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if WINDOWS else cv2.CAP_ANY)
    if not WINDOWS:  # Linux: 1280 x 720 is often available in compressed format only
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap
