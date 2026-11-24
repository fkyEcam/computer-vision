"""Galatae arm that goes back to its REST position when the program is interrupted.

Script:
    from robot import Robot
    from safe_robot import SafeRobot

    with SafeRobot(Robot()) as r:      # end, Ctrl+C, error or closed window -> rest position
        r.reset(); r.home(); r.set_joint_speed(20)
        r.go_to_pose([300, 0, 200, 180, 0])

Notebook:
    arm = SafeRobot(Robot())
    with arm.guard():                  # interrupted cell or error -> rest position, motors stay on
        arm.reset(); arm.home()

What happens on an interruption:
  Ctrl+C / interrupted cell : the CURRENT movement finishes, the tool is lifted, the arm folds (G28).
                              (A 2nd Ctrl+C no longer waits for the arm; it still receives the return commands.)
  closed window (Windows)   : Windows only leaves a few seconds, so the return commands are sent without
                              waiting; the motors stay on and the arm finishes by itself.
  closed terminal (Linux)   : same as Ctrl+C.
NOT covered: killed process (task manager), unplugged USB cable, power cut. In an emergency, use the
EMERGENCY STOP: Ctrl+C does not stop the arm at once, it makes it go back.
"""
import atexit
import contextlib
import math
import os
import signal
import threading
import time

LIFT_Z = 150.0     # mm (robot frame): the tool is lifted to this height before folding
MIN_REACH = 120.0  # mm: closer to the base than this, the arm is already (almost) folded: no lift
MOVES = {"go_to_pose", "jog", "linear_move_to_pose", "probe", "home", "calibrate"}


def _say(text):
    with contextlib.suppress(OSError, ValueError):  # the window may already be closed
        print(text, flush=True)


class SafeRobot:
    """Same methods as galatae-api's Robot."""

    def __init__(self, robot, lift_z=LIFT_Z):
        self._robot, self._lift_z = robot, lift_z
        self._homed, self._parked, self._desync = False, True, False
        self._busy, self._parking, self._closing, self._stops = False, False, False, 0
        self._last_pose = None
        atexit.register(self._send_return)
        self._install_window_handler()

    # ---- the commands of the robot, protected -----------------------------------------------------
    def __getattr__(self, name):
        attr = getattr(self._robot, name)
        if not callable(attr):
            return attr
        return lambda *args, **kwargs: self._call(name, *args, **kwargs)

    def _call(self, name, *args, **kwargs):
        while self._closing:  # the window is being closed: no new command
            time.sleep(0.1)
        if self._desync:
            raise RuntimeError("The dialogue with the arm was cut: wait until it is folded, then restart the program.")
        if name in MOVES:
            self._parked = False
            self._last_pose = list(args[0]) if name == "go_to_pose" else None
        self._busy = True
        try:
            with self._signals():
                result = getattr(self._robot, name)(*args, **kwargs)
        except KeyboardInterrupt:  # 2nd Ctrl+C: the answer of the arm was not read
            self._desync = True
            raise
        finally:
            self._busy = False
        if name in ("home", "calibrate"):
            self._homed = True
        if name == "go_to_sleep_pos":
            self._parked = True
        if self._stops and not self._parking:
            self._stops = 0
            raise KeyboardInterrupt
        return result

    # ---- Ctrl+C: never cut the dialogue with the arm in the middle of a movement --------------------
    def _on_signal(self, signum, frame):
        if not self._busy and not self._parking:
            raise KeyboardInterrupt
        self._stops += 1
        if self._stops >= (3 if self._parking else 2):
            raise KeyboardInterrupt
        _say("The arm is going back to its rest position, please wait..." if self._parking else
             "Interrupted: the current movement finishes, then the arm goes back to its rest position...")

    @contextlib.contextmanager
    def _signals(self):
        """Route Ctrl+C (and closed terminal / terminate) to _on_signal."""
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        old = {}
        for n in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
            if hasattr(signal, n):
                old[n] = signal.signal(getattr(signal, n), self._on_signal)
        try:
            yield
        finally:
            for n, handler in old.items():
                signal.signal(getattr(signal, n), handler)

    # ---- back to the rest position ------------------------------------------------------------------
    def _needs_lift(self, pose):
        return pose is not None and len(pose) >= 5 and math.hypot(pose[0], pose[1]) >= MIN_REACH \
            and pose[2] < self._lift_z

    def park(self):
        """Lift the tool, then fold the arm. Never raises for a communication problem."""
        if self._parked:
            return True
        if self._desync:
            self._send_return_and_tell()
            return False
        if not self._homed:
            _say("The homing was not finished: the arm is NOT moved. Put it in its rest position before restarting.")
            return False
        self._parking, self._stops = True, 0
        try:
            with self._signals():
                try:
                    pose = list(self._call("get_tool_pose"))[:5]
                    if self._needs_lift(pose):
                        self._call("go_to_pose", pose[:2] + [self._lift_z] + pose[3:])
                except Exception as e:  # e.g. pose refused by the arm: fold it anyway
                    _say(f"Lift not possible ({e}): the arm folds directly.")
                self._call("go_to_sleep_pos")
            _say("The arm is in its rest position: you can restart.")
            return True
        except KeyboardInterrupt:
            self._desync = True
            self._send_return_and_tell()
        except Exception as e:
            _say(f"Return to the rest position FAILED ({e}): put the arm in its rest position by hand.")
        finally:
            self._parking = False
        return False

    def _send_return_and_tell(self):
        if self._homed:
            self._send_return()
            _say("Return commands sent. WAIT until the arm is folded before restarting.")
        else:
            _say("The homing was not finished: the arm is NOT moved. Put it in its rest position before restarting.")

    def _send_return(self):
        """Send the return commands without waiting for the answers (closed window, end of the kernel)."""
        if self._parked or not self._homed:
            return
        self._parked = True
        lines = ["G90"]
        if self._needs_lift(self._last_pose):
            x, y, _, a, b = self._last_pose[:5]
            lines.append("G1" + "".join(f"{w}{round(v, 2)}" for w, v in zip("XYZAB", (x, y, self._lift_z, a, b))))
        lines.append("G28")
        with contextlib.suppress(Exception):
            self._robot.arduino.write(("\n".join(lines) + "\n").encode("utf-8"))
            self._robot.arduino.flush()

    def _install_window_handler(self):
        """Windows: closing the console window is not a signal, it needs a console handler."""
        if os.name != "nt":
            return
        import ctypes

        def handler(event):
            if event not in (2, 5, 6):  # close, log off, shutdown (Ctrl+C and Ctrl+Break: see _on_signal)
                return False
            self._closing = True
            self._send_return()
            time.sleep(2)  # leave the time to the serial port (Windows kills the program after 5 s)
            return True

        self._handler = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)(handler)  # keep a reference
        ctypes.windll.kernel32.SetConsoleCtrlHandler(self._handler, True)

    # ---- context managers ---------------------------------------------------------------------------
    @contextlib.contextmanager
    def guard(self):
        """Notebook: rest position if the block is interrupted or fails (motors stay on)."""
        try:
            with self._signals():
                yield self
        except BaseException:
            self.park()
            raise

    def __enter__(self):
        self._cm = self._signals()
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.park() and self._homed:
                with contextlib.suppress(Exception):
                    self._parking = True  # a late Ctrl+C must not interrupt this last command
                    self._call("disable_motors")
        finally:
            self._parking = False
            self._cm.__exit__(None, None, None)
        return False
