"""Move the Galatae arm above the 4 ArUco markers, one after the other.

    python visit_markers.py          # dry run: checks and prints the poses, nothing moves
    python visit_markers.py --run    # moves the real arm (put it in its REST position first!)

robot.py (galatae-api) must be in the same folder; it is already installed on the control box.
"""
import math
import sys

# Marker centres (x, y) in the ROBOT frame, in mm. Default = placement plan of robot_cell_cutouts_A4.pdf
# (base axis centred, 160 mm from the line 0-1). Replace them by the values read with get_tool_pose().
MARKERS = {0: (160, -200), 1: (160, 200), 2: (440, 200), 3: (440, -200)}
Z_TRAVEL = 150         # mm, height of the moves between the markers
Z_POINT = 60           # mm, height above a marker: start high, lower it once the table height is known
PITCH, ROLL = 180, 0   # tool pointing down
SPEED = 20             # deg/s
REACH = (120, 550)     # mm, allowed distance from the base axis
Z_MIN = 30             # mm, the script refuses to go lower


def check(pose):
    r = math.hypot(pose[0], pose[1])
    assert REACH[0] <= r <= REACH[1], f"out of reach: {pose} is {r:.0f} mm from the base"
    assert pose[2] >= Z_MIN, f"too low: {pose}"


steps = []  # (text, pose)
for i, (x, y) in MARKERS.items():
    steps.append((f"above marker {i}", [x, y, Z_TRAVEL, PITCH, ROLL]))
    steps.append((f"point at marker {i}", [x, y, Z_POINT, PITCH, ROLL]))
    steps.append((f"up from marker {i}", [x, y, Z_TRAVEL, PITCH, ROLL]))
for text, pose in steps:  # every pose is checked before the first move
    check(pose)

if "--run" not in sys.argv:
    for text, pose in steps:
        print(f"{text:20s} {pose}")
    print("Dry run: nothing moved. Use --run to move the arm.")
    sys.exit()

from robot import Robot  # noqa: E402

input("Arm in REST position, cell clear, hand near the emergency stop? Enter to start (Ctrl+C to cancel)")
r = Robot()
r.reset()
r.home()
r.set_joint_speed(SPEED)
for text, pose in steps:
    print(f"{text:20s} {pose}")
    r.go_to_pose(pose)
    if text.startswith("point"):
        print("   tool pose:", r.get_tool_pose())
        input("   Enter to continue")
r.go_to_sleep_pos()
r.disable_motors()
