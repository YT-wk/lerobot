"""Human-in-the-loop adapter for an SO101 leader arm."""

from __future__ import annotations

import logging
import os
import select
import sys
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.teleoperators.utils import TeleopEvents

logger = logging.getLogger(__name__)


class _TerminalKeyboardListener:
    """Read HIL controls from the caller's terminal when no desktop is available."""

    def __init__(self, on_key: Callable[[str], None]) -> None:
        self._on_key = on_key
        self._fd: int | None = None
        self._original_settings: list[Any] | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "SO101 HIL leader needs an interactive terminal for Space/S/Esc controls. "
                "Run it from a local terminal or provide a graphical DISPLAY."
            )
        import termios
        import tty

        self._fd = sys.stdin.fileno()
        self._original_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        print("[HIL] Terminal keyboard controls are active: Space, S, Esc.", flush=True)

    def stop(self) -> None:
        import termios

        if self._fd is not None and self._original_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_settings)
        self._fd = None
        self._original_settings = None

    def poll(self) -> None:
        """Consume terminal controls from the control-loop thread without blocking."""
        if self._fd is None:
            return

        while True:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
            if not readable:
                return
            key = os.read(self._fd, 1).decode(errors="ignore")
            if key == " ":
                self._on_key("space")
            elif key == "\x1b":
                self._on_key("esc")
            elif key.lower() == "s":
                self._on_key("s")


class HILSO101Leader:
    """Wrap an SO101 leader with HIL events and follower-tracking controls."""

    def __init__(
        self,
        leader: Any,
        *,
        fps: int,
        max_joint_speed_deg_s: float,
        max_gripper_speed: float,
        use_gripper: bool = True,
        keyboard_listener_factory: Callable[[Callable[[Any], None]], Any] | None = None,
    ) -> None:
        self.leader = leader
        self.fps = fps
        self.max_joint_speed_deg_s = max_joint_speed_deg_s
        self.max_gripper_speed = max_gripper_speed
        self.use_gripper = use_gripper
        self._keyboard_listener_factory = keyboard_listener_factory
        self._keyboard_listener: Any | None = None
        self._intervention_active = False
        self._success_pending = False
        self._failure_pending = False
        self._read_failed = False
        self._torque_enabled = False
        self._feedback_goal: dict[str, float] | None = None
        self._pressed_keys: set[str] = set()

    @property
    def action_features(self) -> dict:
        # HIL-SERL stores the action after it has been mapped to Cartesian deltas,
        # not the leader's raw six joint positions.
        features = {
            "dtype": "float32",
            "shape": (4,) if self.use_gripper else (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }
        if self.use_gripper:
            features["names"]["gripper"] = 3
        return features

    @property
    def feedback_features(self) -> dict:
        return self.leader.feedback_features

    @property
    def is_connected(self) -> bool:
        return self.leader.is_connected

    @property
    def is_intervening(self) -> bool:
        return self._intervention_active

    def connect(self, calibrate: bool = True) -> None:
        self.leader.connect(calibrate=calibrate)
        self._start_keyboard_listener()

    def calibrate(self) -> None:
        self.leader.calibrate()

    def configure(self) -> None:
        self.leader.configure()

    def get_action(self) -> dict[str, float]:
        try:
            return self.leader.get_action()
        except Exception as exc:
            logger.exception("SO101 leader read failed; terminating the current HIL episode: %s", exc)
            self._read_failed = True
            return {}

    def get_teleop_events(self) -> dict[TeleopEvents, bool]:
        poll = getattr(self._keyboard_listener, "poll", None)
        if callable(poll):
            poll()
        success = self._success_pending
        failure = self._failure_pending or self._read_failed
        self._success_pending = False
        self._failure_pending = False
        self._read_failed = False
        return {
            TeleopEvents.IS_INTERVENTION: self._intervention_active or failure,
            TeleopEvents.TERMINATE_EPISODE: failure,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: False,
        }

    def send_feedback(self, feedback: dict[str, float]) -> None:
        self.leader.send_feedback(feedback)

    def update_policy_tracking(self, follower_positions: dict[str, float]) -> None:
        """Safely move the leader toward the follower's measured joints."""
        if self._intervention_active:
            self.stop_policy_tracking()
            return

        target = {key: float(value) for key, value in follower_positions.items() if key.endswith(".pos")}
        if not target:
            return

        if not self._torque_enabled:
            current = self.get_action()
            if not current:
                return
            self.leader.send_feedback(current)
            self.leader.enable_torque()
            self._feedback_goal = {key: float(value) for key, value in current.items() if key in target}
            self._torque_enabled = True

        if self._feedback_goal is None:
            self._feedback_goal = dict(target)

        joint_step = self.max_joint_speed_deg_s / self.fps
        gripper_step = self.max_gripper_speed / self.fps
        next_goal: dict[str, float] = {}
        for key, desired in target.items():
            current_goal = self._feedback_goal.get(key, desired)
            max_step = gripper_step if key == "gripper.pos" else joint_step
            next_goal[key] = float(current_goal + np.clip(desired - current_goal, -max_step, max_step))

        self.leader.send_feedback(next_goal)
        self._feedback_goal = next_goal

    def stop_policy_tracking(self) -> None:
        if self._torque_enabled:
            self.leader.disable_torque()
            self._torque_enabled = False
        self._feedback_goal = None

    def move_to_home_position(
        self,
        home_positions: dict[str, float],
        *,
        tolerance_deg: float = 1.0,
        timeout_s: float = 15.0,
    ) -> None:
        """Return the leader to its captured home pose at the configured speed."""
        if tolerance_deg < 0.0 or timeout_s <= 0.0:
            raise ValueError("Leader home tolerance must be non-negative and timeout must be positive.")

        target = {key: float(value) for key, value in home_positions.items() if key.endswith(".pos")}
        if not target:
            return

        self.reset_episode()
        deadline = time.monotonic() + timeout_s
        try:
            while True:
                current = self.get_action()
                if not current:
                    raise RuntimeError("Unable to read SO101 leader while returning to the captured home pose.")
                error = [abs(float(current[key]) - desired) for key, desired in target.items() if key in current]
                if len(error) != len(target):
                    raise RuntimeError("SO101 leader home pose is missing one or more joint positions.")
                if max(error, default=0.0) <= tolerance_deg:
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError("Timed out returning the SO101 leader to the captured home pose.")
                self.update_policy_tracking(target)
                time.sleep(1.0 / self.fps)
        finally:
            self.stop_policy_tracking()

    def reset_episode(self) -> None:
        self._intervention_active = False
        self._success_pending = False
        self._failure_pending = False
        self._read_failed = False

    def disconnect(self) -> None:
        self.stop_policy_tracking()
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        self.leader.disconnect()

    def handle_key_press(self, key: str) -> None:
        """Handle a normalized key name. Kept public for deterministic tests."""
        key = key.lower()
        if key in self._pressed_keys:
            return
        self._pressed_keys.add(key)
        if key == "space":
            self._intervention_active = not self._intervention_active
            state = "enabled" if self._intervention_active else "disabled"
            logger.info("SO101 leader intervention %s.", state)
            print(f"[HIL] Space received: leader intervention {state}.", flush=True)
            if self._intervention_active:
                self.stop_policy_tracking()
        elif key == "s":
            self._success_pending = True
        elif key == "esc":
            self._failure_pending = True

    def handle_key_release(self, key: str) -> None:
        self._pressed_keys.discard(key.lower())

    def _start_keyboard_listener(self) -> None:
        if self._keyboard_listener is not None:
            return
        if self._keyboard_listener_factory is not None:
            self._keyboard_listener = self._keyboard_listener_factory(self._on_key_press)
            self._keyboard_listener.start()
            return

        # An SSH terminal can inherit DISPLAY from the edge device. Prefer its
        # stdin in that case: pynput would listen to the edge display instead
        # of the terminal where the operator is pressing Space.
        if sys.stdin.isatty():
            self._keyboard_listener = _TerminalKeyboardListener(self._on_terminal_key)
            self._keyboard_listener.start()
            return

        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "SO101 HIL leader needs an interactive terminal for Space/S/Esc controls. "
                "Run it from a local terminal or provide a graphical DISPLAY."
            )

        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "SO101 HIL leader mode needs pynput for Space/S/Esc controls. Install lerobot[pynput-dep]."
            ) from exc

        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self._keyboard_listener.start()

    def _on_terminal_key(self, key: str) -> None:
        self.handle_key_press(key)
        self.handle_key_release(key)

    def _on_key_press(self, key: Any) -> None:
        try:
            if key.char:
                self.handle_key_press(key.char)
                return
        except AttributeError:
            pass

        key_name = getattr(key, "name", "")
        if key_name in {"space", "esc"}:
            self.handle_key_press(key_name)

    def _on_key_release(self, key: Any) -> None:
        try:
            if key.char:
                self.handle_key_release(key.char)
                return
        except AttributeError:
            pass

        key_name = getattr(key, "name", "")
        if key_name in {"space", "esc"}:
            self.handle_key_release(key_name)
