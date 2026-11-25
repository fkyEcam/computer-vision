"""Show the live image of the USB camera.

    python view_camera.py        # USB camera (index 1 on Windows, 0 on Linux)
    python view_camera.py 0      # camera 0 (laptop webcam on Windows)

q, ESC or the cross of the window: quit.
"""
import sys

import cv2

from camera import DEFAULT_CAMERA, open_camera

camera = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CAMERA
cap = open_camera(camera)
if not cap.isOpened():
    raise SystemExit(f"Cannot open camera {camera}: is it plugged in, or used by another program?")

title = f"camera {camera} (q = quit)"
while True:
    ok, frame = cap.read()
    if not ok:
        print("No image received from the camera.")
        break
    cv2.imshow(title, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
