"""Move the Galatae arm above the 4 ArUco markers, one after the other.

    python visit_markers.py          # dry run: checks and prints the poses, nothing moves
    python visit_markers.py --run    # moves the real arm (put it in its REST position first!)

robot.py (galatae-api) and safe_robot.py must be in the same folder; robot.py is already on the control box.
Robot frame, facing the robot: x towards you (depth), y to your right, z up; z = 0 is 120 mm above the table.
Ctrl+C or closing the window brings the arm back to its rest position (see safe_robot.py).
"""
import math
import sys

# Marker centres (x, y) in the ROBOT frame, in mm. Default = placement plan of robot_cell_cutouts_A4.pdf
# (340 x 260 mm, base axis centred, 230 mm from the line 0-1). Replace them by the values read with get_tool_pose().
MARKERS = {0: (230, -170), 1: (230, 170), 2: (490, 170), 3: (490, -170)}
TABLE_Z = -120         # mm, the table in the robot frame
Z_TRAVEL = TABLE_Z + 200  # moves between the markers: 200 mm above the table
Z_POINT = TABLE_Z + 100   # above a marker: 100 mm above the table, lower it step by step
PITCH, ROLL = 180, 0   # tool pointing down
SPEED = 20             # deg/s
REACH = (120, 550)     # mm, allowed distance from the base axis
Z_MIN = TABLE_Z + 30   # the script refuses to go lower than 30 mm above the table


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
from safe_robot import SafeRobot  # noqa: E402

input("Arm in REST position, cell clear, hand near the emergency stop? Enter to start (Ctrl+C to cancel)")
try:
    with SafeRobot(Robot()) as r:  # always ends in the rest position
        r.reset()
        r.home()
        r.set_joint_speed(SPEED)
        for text, pose in steps:
            print(f"{text:20s} {pose}")
            r.go_to_pose(pose)
            if text.startswith("point"):
                print("   tool pose:", r.get_tool_pose())
                input("   Enter to continue")
except KeyboardInterrupt:
    print("Interrupted.")
