from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.utils import TeleopEvents


class FakeGamepad:
    def __init__(self, *, ready=True, intervention=False, end_status=None, gripper="stay"):
        self.is_ready = ready
        self.intervention = intervention
        self.end_status = end_status
        self.gripper = gripper

    def update(self):
        return None

    def get_deltas(self):
        return 0.1, -0.2, 0.3

    def gripper_command(self):
        return self.gripper

    def should_intervene(self):
        return self.intervention

    def get_episode_end_status(self):
        status = self.end_status
        self.end_status = None
        return status

    def stop(self):
        return None


def test_gamepad_hil_action_and_success_event():
    teleop = GamepadTeleop(GamepadTeleopConfig())
    teleop.gamepad = FakeGamepad(intervention=True, end_status=TeleopEvents.SUCCESS, gripper="open")

    assert teleop.get_action() == {"delta_x": 0.1, "delta_y": -0.2, "delta_z": 0.3, "gripper": 2}
    events = teleop.get_teleop_events()
    assert events[TeleopEvents.IS_INTERVENTION]
    assert events[TeleopEvents.SUCCESS]
    assert not events[TeleopEvents.TERMINATE_EPISODE]


def test_gamepad_connection_reflects_controller_readiness():
    teleop = GamepadTeleop(GamepadTeleopConfig())
    teleop.gamepad = FakeGamepad(ready=False)

    assert not teleop.is_connected
