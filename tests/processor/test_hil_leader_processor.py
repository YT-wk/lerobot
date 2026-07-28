from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lerobot.rl.gym_manipulator as gym_manipulator
from lerobot.configs import FeatureType
from lerobot.envs.configs import (
    GripperConfig,
    HILSerlProcessorConfig,
    HILSerlRobotEnvConfig,
    InverseKinematicsConfig,
    LeaderKinematicsConfig,
)
from lerobot.processor import (
    CanonicalHILActionProcessorStep,
    DirectJointActionOverrideProcessorStep,
    InterventionActionProcessorStep,
    LeaderFollowerPoseErrorActionProcessorStep,
    LeaderJointDeltaActionProcessorStep,
    LeaderPolicyTrackingProcessorStep,
    MapDeltaActionToRobotActionStep,
)
from lerobot.processor.converters import create_transition
from lerobot.rl.gym_manipulator import (
    RobotEnv,
    _capture_initial_reset_positions,
    _validate_real_robot_hil_config,
    reset_and_build_transition,
)
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety, EEReferenceAndDelta
from lerobot.teleoperators.gamepad import GamepadTeleopConfig
from lerobot.teleoperators.so_leader import HILSO101Leader, SO101LeaderConfig
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE


class FakeKinematics:
    def forward_kinematics(self, joints):
        transform = np.eye(4)
        transform[:3, 3] = joints[:3]
        return transform


class FakeLeader:
    def __init__(self):
        self.action_features = {f"joint_{index}.pos": float for index in range(3)} | {"gripper.pos": float}
        self.feedback_features = self.action_features
        self.is_connected = True
        self.action = {"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
        self.feedback = []
        self.torque_enabled = False

    def get_action(self):
        return dict(self.action)

    def send_feedback(self, feedback):
        self.feedback.append(dict(feedback))

    def enable_torque(self):
        self.torque_enabled = True

    def disable_torque(self):
        self.torque_enabled = False

    def connect(self, calibrate=True):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False


class FakeTrackingTeleop:
    def __init__(self):
        self.followed = []
        self.stopped = 0
        self.resets = 0

    def update_policy_tracking(self, observation):
        self.followed.append(observation)

    def stop_policy_tracking(self):
        self.stopped += 1

    def reset_episode(self):
        self.resets += 1


class FakeCaptureRobot:
    def __init__(self, positions):
        self.bus = SimpleNamespace(motors={"joint_0": object(), "joint_1": object()})
        self.positions = iter(positions)

    def get_observation(self):
        return next(self.positions)


class FakeCaptureLeader:
    def __init__(self, positions):
        self.positions = iter(positions)

    def get_action(self):
        return next(self.positions)


class FakeResetRobot:
    def __init__(self):
        self.is_connected = True
        self.bus = SimpleNamespace(motors={"joint_0": object(), "joint_1": object()})
        self.cameras = {}

    def get_observation(self):
        return {"joint_0.pos": 1.0, "joint_1.pos": 2.0}


def _transition(*, action, intervention, observation=None):
    return create_transition(
        observation=observation or {},
        action=torch.zeros(4),
        info={TeleopEvents.IS_INTERVENTION: intervention},
        complementary_data={"teleop_action": action},
    )


def test_hil_leader_tracks_follower_at_a_bounded_speed():
    backend = FakeLeader()
    leader = HILSO101Leader(backend, fps=10, max_joint_speed_deg_s=60.0, max_gripper_speed=100.0)

    leader.update_policy_tracking({"joint_0.pos": 20.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 50.0})

    assert backend.torque_enabled
    assert backend.feedback[0] == backend.action
    assert backend.feedback[-1]["joint_0.pos"] == 6.0
    assert backend.feedback[-1]["gripper.pos"] == 10.0
    assert leader.action_features["shape"] == (4,)

    leader.handle_key_press("space")
    assert not backend.torque_enabled
    assert leader.get_teleop_events()[TeleopEvents.IS_INTERVENTION]


def test_hil_leader_does_not_enable_torque_when_already_at_home():
    backend = FakeLeader()
    leader = HILSO101Leader(backend, fps=10, max_joint_speed_deg_s=60.0, max_gripper_speed=100.0)

    leader.move_to_home_position(backend.action)

    assert not backend.torque_enabled
    assert backend.feedback == []


def test_capture_initial_reset_positions_requires_both_arms_to_be_still():
    follower_positions = [
        {"joint_0.pos": 1.0, "joint_1.pos": 2.0},
        {"joint_0.pos": 1.0, "joint_1.pos": 2.0},
    ]
    leader_positions = [
        {"joint_0.pos": 3.0, "joint_1.pos": 4.0},
        {"joint_0.pos": 3.0, "joint_1.pos": 4.0},
    ]
    follower_home, leader_home = _capture_initial_reset_positions(
        FakeCaptureRobot(follower_positions),
        FakeCaptureLeader(leader_positions),
        settle_time_s=0.0,
        max_joint_delta_deg=0.5,
    )

    assert follower_home == [1.0, 2.0]
    assert leader_home == {"joint_0.pos": 3.0, "joint_1.pos": 4.0}

    with pytest.raises(RuntimeError, match="follower:joint_0.pos"):
        _capture_initial_reset_positions(
            FakeCaptureRobot(
                [
                    {"joint_0.pos": 1.0, "joint_1.pos": 2.0},
                    {"joint_0.pos": 2.0, "joint_1.pos": 2.0},
                ]
            ),
            FakeCaptureLeader(
                [
                    {"joint_0.pos": 3.0, "joint_1.pos": 4.0},
                    {"joint_0.pos": 3.0, "joint_1.pos": 4.0},
                ]
            ),
            settle_time_s=0.0,
            max_joint_delta_deg=0.5,
        )


def test_robot_env_reset_returns_follower_then_leader_to_captured_home(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gym_manipulator,
        "reset_follower_position",
        lambda robot, pose: calls.append(("follower", pose.tolist())),
    )
    env = RobotEnv(
        FakeResetRobot(),
        reset_pose=[1.0, 2.0],
        reset_time_s=0.0,
        reset_callback=lambda: calls.append(("leader", None)),
    )

    env.reset()

    assert calls == [("follower", [1.0, 2.0]), ("leader", None)]


def test_robot_env_skips_the_reset_immediately_after_capturing_home(monkeypatch):
    calls = []
    monkeypatch.setattr(
        gym_manipulator,
        "reset_follower_position",
        lambda robot, pose: calls.append(("follower", pose.tolist())),
    )
    env = RobotEnv(
        FakeResetRobot(),
        reset_pose=[1.0, 2.0],
        reset_time_s=0.0,
        reset_callback=lambda: calls.append(("leader", None)),
        skip_initial_reset=True,
    )

    env.reset()
    assert calls == []
    env.reset()
    assert calls == [("follower", [1.0, 2.0]), ("leader", None)]


def test_ctrl_c_returns_follower_home_then_releases_both_arms(monkeypatch):
    calls = []
    robot = SimpleNamespace(
        is_connected=True,
        bus=SimpleNamespace(disable_torque=lambda: calls.append("follower_torque_off")),
    )
    env = SimpleNamespace(
        robot=robot,
        reset_pose=[1.0, 2.0],
        close=lambda: calls.append("follower_disconnected"),
    )
    leader = SimpleNamespace(
        is_connected=True,
        disable_torque=lambda: calls.append("leader_torque_off"),
    )
    teleop = SimpleNamespace(leader=leader, disconnect=lambda: calls.append("leader_disconnected"))
    monkeypatch.setattr(
        gym_manipulator,
        "reset_follower_position",
        lambda robot_arm, pose: calls.append(("follower_home", pose.tolist())),
    )

    gym_manipulator._return_home_and_shutdown_after_interrupt(env, teleop)

    assert calls == [
        ("follower_home", [1.0, 2.0]),
        "follower_torque_off",
        "leader_torque_off",
        "follower_disconnected",
        "leader_disconnected",
    ]


def test_hil_leader_events_are_consumed_once():
    leader = HILSO101Leader(FakeLeader(), fps=10, max_joint_speed_deg_s=60.0, max_gripper_speed=100.0)

    leader.handle_key_press("s")
    assert leader.get_teleop_events()[TeleopEvents.SUCCESS]
    assert not leader.get_teleop_events()[TeleopEvents.SUCCESS]

    leader.handle_key_press("esc")
    events = leader.get_teleop_events()
    assert events[TeleopEvents.TERMINATE_EPISODE]
    assert events[TeleopEvents.IS_INTERVENTION]


def test_hil_leader_terminal_key_can_repeat_after_release():
    leader = HILSO101Leader(FakeLeader(), fps=10, max_joint_speed_deg_s=60.0, max_gripper_speed=100.0)

    leader._on_terminal_key("s")
    assert leader.get_teleop_events()[TeleopEvents.SUCCESS]
    leader._on_terminal_key("s")
    assert leader.get_teleop_events()[TeleopEvents.SUCCESS]


def test_leader_tracking_processor_switches_between_policy_and_intervention():
    teleop = FakeTrackingTeleop()
    processor = LeaderPolicyTrackingProcessorStep(teleop_device=teleop)

    processor(_transition(action={}, intervention=False, observation={"joint_0.pos": 1.0}))
    processor(_transition(action={}, intervention=True))
    processor.reset()

    assert teleop.followed == [{"joint_0.pos": 1.0}]
    assert teleop.stopped == 2
    assert teleop.resets == 1


def test_leader_joint_delta_has_zero_first_frame_and_maps_following_frames():
    processor = LeaderJointDeltaActionProcessorStep(
        kinematics=FakeKinematics(),
        motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
    )
    first = processor(
        _transition(
            action={"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0},
            intervention=True,
        )
    )
    second = processor(
        _transition(
            action={"joint_0.pos": 0.05, "joint_1.pos": -0.2, "joint_2.pos": 0.0, "gripper.pos": 1.0},
            intervention=True,
        )
    )

    assert first[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "gripper": 1.0,
    }
    assert second[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == {
        "delta_x": 0.5,
        "delta_y": -1.0,
        "delta_z": 0.0,
        "gripper": 2.0,
    }


@pytest.mark.parametrize("strategy,direct", [("wenruo", False), ("ggand0", True)])
def test_pose_error_strategies_share_canonical_action_and_optional_joint_override(strategy, direct):
    processor = LeaderFollowerPoseErrorActionProcessorStep(
        leader_kinematics=FakeKinematics(),
        follower_kinematics=FakeKinematics(),
        follower_motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        strategy=strategy,
        direct_joint_mirror=direct,
        max_takeover_joint_error_deg=2.0,
        max_takeover_position_error_m=1.0 if strategy == "wenruo" else None,
    )
    leader = {"joint_0.pos": 0.05, "joint_1.pos": -0.1, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    follower = {"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    processor(_transition(action=leader, intervention=True, observation=follower))
    result = processor(_transition(action=leader, intervention=True, observation=follower))

    assert result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == pytest.approx(
        {"delta_x": 0.5, "delta_y": -1.0, "delta_z": 0.0, "gripper": 1.0}
    )
    assert ("hil_direct_joint_action" in result[TransitionKey.COMPLEMENTARY_DATA]) is direct


def test_wenruo_rejects_takeover_when_end_effectors_are_not_aligned():
    processor = LeaderFollowerPoseErrorActionProcessorStep(
        leader_kinematics=FakeKinematics(),
        follower_kinematics=FakeKinematics(),
        follower_motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        strategy="wenruo",
        max_takeover_position_error_m=0.02,
    )
    leader = {"joint_0.pos": 0.05, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    follower = {"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}

    processor(_transition(action=leader, intervention=True, observation=follower))
    result = processor(_transition(action=leader, intervention=True, observation=follower))

    assert result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "gripper": 1.0,
    }


def test_wenruo_rebases_large_initial_arm_offset_before_tracking_leader_motion():
    processor = LeaderFollowerPoseErrorActionProcessorStep(
        leader_kinematics=FakeKinematics(),
        follower_kinematics=FakeKinematics(),
        follower_motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        strategy="wenruo",
        max_takeover_position_error_m=0.1,
        rebase_on_intervention=True,
    )
    follower = {"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    initial_leader = {"joint_0.pos": 0.5, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    moved_leader = {"joint_0.pos": 0.55, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}

    first = processor(_transition(action=initial_leader, intervention=True, observation=follower))
    still = processor(_transition(action=initial_leader, intervention=True, observation=follower))
    moved = processor(_transition(action=moved_leader, intervention=True, observation=follower))

    assert first[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]["delta_x"] == 0.0
    assert still[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]["delta_x"] == 0.0
    assert moved[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == pytest.approx(
        {"delta_x": 0.5, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}
    )


def test_disabled_ee_command_holds_pose_without_workspace_clipping():
    reference = EEReferenceAndDelta(
        kinematics=FakeKinematics(),
        motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
    )
    safety = EEBoundsAndSafety(
        end_effector_bounds={"min": [-1.0, -1.0, -0.16], "max": [1.0, 1.0, 0.10]}
    )
    transition = create_transition(
        observation={"joint_0.pos": 0.2, "joint_1.pos": 0.0, "joint_2.pos": 0.26, "gripper.pos": 0.0},
        action={
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper_vel": 1.0,
        },
    )

    result = safety(reference(transition))

    assert result[TransitionKey.ACTION]["ee.z"] == pytest.approx(0.26)
    assert "ee.enabled" not in result[TransitionKey.ACTION]


def test_canonical_action_and_direct_override_keep_storage_action_separate():
    transition = create_transition(
        action=torch.tensor([2.0, -2.0, 0.25, 1.6]),
        complementary_data={
            "hil_direct_joint_action": {"joint_0.pos": 10.0, "joint_1.pos": 20.0}
        },
    )
    canonical = CanonicalHILActionProcessorStep()(transition)
    overridden = DirectJointActionOverrideProcessorStep(["joint_0", "joint_1"])(canonical)

    assert canonical[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"].tolist() == [1.0, -1.0, 0.25, 2.0]
    assert overridden[TransitionKey.ACTION].tolist() == [10.0, 20.0]
    assert overridden[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"].tolist() == [1.0, -1.0, 0.25, 2.0]


def test_ggand0_direct_mirror_uses_mapped_joints_and_holds_disabled_gripper():
    processor = LeaderFollowerPoseErrorActionProcessorStep(
        leader_kinematics=FakeKinematics(),
        follower_kinematics=FakeKinematics(),
        follower_motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        leader_action_joint_names=["raw_0", "raw_1", "raw_2", "gripper"],
        joint_scales={"raw_0": 2.0, "raw_1": -1.0},
        joint_offsets_deg={"raw_0": 1.0},
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        strategy="ggand0",
        use_gripper=False,
        direct_joint_mirror=True,
        max_takeover_joint_error_deg=0.1,
    )
    leader = {"raw_0.pos": 1.0, "raw_1.pos": -2.0, "raw_2.pos": 3.0, "gripper.pos": 80.0}
    follower = {"joint_0.pos": 3.0, "joint_1.pos": 2.0, "joint_2.pos": 3.0, "gripper.pos": 41.0}

    processor(_transition(action=leader, intervention=True, observation=follower))
    result = processor(_transition(action=leader, intervention=True, observation=follower))
    direct = result[TransitionKey.COMPLEMENTARY_DATA]["hil_direct_joint_action"]

    assert direct == {
        "joint_0.pos": 3.0,
        "joint_1.pos": 2.0,
        "joint_2.pos": 3.0,
        "gripper.pos": 41.0,
    }
    intervened = InterventionActionProcessorStep(use_gripper=False)(result)
    canonical = CanonicalHILActionProcessorStep(use_gripper=False)(intervened)
    overridden = DirectJointActionOverrideProcessorStep(
        ["joint_0", "joint_1", "joint_2", "gripper"]
    )(canonical)
    assert canonical[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"].shape == (3,)
    assert overridden[TransitionKey.ACTION].tolist() == [3.0, 2.0, 3.0, 41.0]


def test_ggand0_rejects_joint_mirror_until_arms_are_aligned():
    processor = LeaderFollowerPoseErrorActionProcessorStep(
        leader_kinematics=FakeKinematics(),
        follower_kinematics=FakeKinematics(),
        follower_motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        strategy="ggand0",
        direct_joint_mirror=True,
        max_takeover_joint_error_deg=1.0,
    )
    leader = {"joint_0.pos": 10.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}
    follower = {"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 0.0}

    processor(_transition(action=leader, intervention=True, observation=follower))
    result = processor(_transition(action=leader, intervention=True, observation=follower))

    assert result[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == {
        "delta_x": 0.0,
        "delta_y": 0.0,
        "delta_z": 0.0,
        "gripper": 1.0,
    }
    assert "hil_direct_joint_action" not in result[TransitionKey.COMPLEMENTARY_DATA]


@pytest.mark.parametrize(
    "rotation",
    [
        [[1.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 1.0]],
        [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ],
)
def test_pose_error_strategy_requires_a_proper_rotation(rotation):
    with pytest.raises(ValueError, match="rotation matrix"):
        LeaderFollowerPoseErrorActionProcessorStep(
            leader_kinematics=FakeKinematics(),
            follower_kinematics=FakeKinematics(),
            follower_motor_names=["joint_0", "joint_1", "joint_2"],
            end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
            strategy="wenruo",
            leader_to_follower_rotation=rotation,
        )


def test_reset_stops_action_processors_before_moving_the_environment():
    calls = []

    class Pipeline:
        def __init__(self, name):
            self.name = name

        def reset(self):
            calls.append(f"{self.name}_reset")

        def __call__(self, data):
            calls.append(f"{self.name}_call")
            return data

    env = SimpleNamespace(reset=lambda: (calls.append("environment_reset") or ({}, {})))
    reset_and_build_transition(env, Pipeline("env"), Pipeline("action"))

    assert calls == ["action_reset", "env_reset", "environment_reset", "env_call"]


def test_leader_joint_delta_can_invert_leader_gripper_encoder_direction():
    processor = LeaderJointDeltaActionProcessorStep(
        kinematics=FakeKinematics(),
        motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        gripper_open_on_positive_delta=False,
    )
    processor(
        _transition(
            action={"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 10.0},
            intervention=True,
        )
    )
    opened = processor(
        _transition(
            action={"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 8.0},
            intervention=True,
        )
    )
    closed = processor(
        _transition(
            action={"joint_0.pos": 0.0, "joint_1.pos": 0.0, "joint_2.pos": 0.0, "gripper.pos": 10.0},
            intervention=True,
        )
    )

    assert opened[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]["gripper"] == 2.0
    assert closed[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"]["gripper"] == 0.0


def test_leader_joint_delta_maps_raw_joints_into_a_separate_fk_and_frame():
    processor = LeaderJointDeltaActionProcessorStep(
        kinematics=FakeKinematics(),
        motor_names=["joint1", "joint2", "joint3"],
        leader_action_joint_names=["raw_a", "raw_b", "raw_c"],
        joint_scales={"raw_a": 2.0, "raw_b": -1.0, "raw_c": 1.0},
        joint_offsets_deg={"raw_a": 1.0, "raw_b": 0.0, "raw_c": 0.0},
        leader_to_follower_rotation=[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        use_gripper=False,
    )
    processor(
        _transition(
            action={"raw_a.pos": 0.0, "raw_b.pos": 0.0, "raw_c.pos": 0.0},
            intervention=True,
        )
    )
    transition = processor(
        _transition(
            action={"raw_a.pos": 0.05, "raw_b.pos": -0.02, "raw_c.pos": 0.03},
            intervention=True,
        )
    )

    assert transition[TransitionKey.COMPLEMENTARY_DATA]["teleop_action"] == pytest.approx(
        {"delta_x": 0.2, "delta_y": -1.0, "delta_z": 0.3}
    )


def test_leader_joint_delta_requires_matching_raw_and_fk_joint_counts():
    with pytest.raises(ValueError, match="same length"):
        LeaderJointDeltaActionProcessorStep(
            kinematics=FakeKinematics(),
            motor_names=["joint1", "joint2"],
            leader_action_joint_names=["raw_a"],
            end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        )


def test_leader_tracking_can_be_disabled_for_passive_intervention():
    teleop = FakeTrackingTeleop()
    processor = LeaderPolicyTrackingProcessorStep(teleop_device=teleop, enabled=False)

    processor(_transition(action={}, intervention=False, observation={"joint_0.pos": 1.0}))

    assert teleop.followed == []
    assert teleop.stopped == 1


def test_delta_processor_uses_a_stay_gripper_when_no_gripper_is_configured():
    action = MapDeltaActionToRobotActionStep().action({"delta_x": 0.1, "delta_y": 0.0, "delta_z": 0.0})

    assert action["gripper_vel"] == 1.0


def test_intervention_success_label_can_be_held_for_reward_recording():
    processor = InterventionActionProcessorStep(
        terminate_on_success=False,
        success_label_hold_steps=3,
    )

    def transition(success: bool):
        return create_transition(
            action=torch.zeros(4),
            info={TeleopEvents.SUCCESS: success},
            complementary_data={"teleop_action": {}},
        )

    rewards = [processor(transition(success=index == 0))[TransitionKey.REWARD] for index in range(4)]
    assert rewards == [1.0, 1.0, 1.0, 0.0]

    processor(transition(success=True))
    processor.reset()
    assert processor(transition(success=False))[TransitionKey.REWARD] == 0.0


def test_leader_hil_config_requires_complete_ik_configuration():
    cfg = HILSerlRobotEnvConfig(
        robot=SO101FollowerConfig(port="/dev/null", use_degrees=True),
        teleop=SO101LeaderConfig(port="/dev/null", use_degrees=True),
        processor=HILSerlProcessorConfig(control_mode="leader"),
    )

    with pytest.raises(ValueError, match="inverse_kinematics"):
        _validate_real_robot_hil_config(cfg)

    cfg.processor.inverse_kinematics = InverseKinematicsConfig(
        urdf_path="so101.urdf",
        target_frame_name="gripper_frame",
        end_effector_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
        end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
    )
    _validate_real_robot_hil_config(cfg)

    cfg.processor.inverse_kinematics.end_effector_step_sizes["x"] = 0.0
    with pytest.raises(ValueError, match="positive x/y/z"):
        _validate_real_robot_hil_config(cfg)


def test_so101_hil_config_builds_the_canonical_policy_feature_contract():
    cfg = HILSerlRobotEnvConfig(
        robot=SO101FollowerConfig(port="/dev/null", use_degrees=True),
        teleop=SO101LeaderConfig(port="/dev/null", use_degrees=True),
        processor=HILSerlProcessorConfig(control_mode="leader", gripper=GripperConfig(use_gripper=True)),
    )

    assert cfg.features[ACTION].type is FeatureType.ACTION
    assert cfg.features[ACTION].shape == (4,)
    assert cfg.features["agent_pos"].shape == (6,)
    assert cfg.features_map == {ACTION: ACTION, "agent_pos": OBS_STATE}


def test_invalid_leader_strategy_and_unsafe_ggand0_config_fail_early():
    cfg = HILSerlRobotEnvConfig(
        robot=SO101FollowerConfig(port="/dev/null", use_degrees=True),
        teleop=SO101LeaderConfig(port="/dev/null", use_degrees=True),
        processor=HILSerlProcessorConfig(
            control_mode="leader",
            inverse_kinematics=InverseKinematicsConfig(
                end_effector_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
            ),
        ),
    )
    cfg.processor.leader_control_strategy = "invalid"
    with pytest.raises(ValueError, match="Unsupported leader_control_strategy"):
        _validate_real_robot_hil_config(cfg)

    cfg.processor.leader_control_strategy = "ggand0"
    with pytest.raises(ValueError, match="max_relative_target"):
        _validate_real_robot_hil_config(cfg)


def test_non_so_leader_blocks_unverified_b601_kinematics():
    cfg = HILSerlRobotEnvConfig(
        robot=SimpleNamespace(type="koch_rebot_b601_dm_follower"),
        teleop=SimpleNamespace(type="rebot102_hilserl_leader"),
        processor=HILSerlProcessorConfig(
            control_mode="leader",
            inverse_kinematics=InverseKinematicsConfig(
                urdf_path="b601.urdf",
                target_frame_name="gripper_tcp",
                end_effector_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
            ),
            leader_kinematics=LeaderKinematicsConfig(
                urdf_path="rebot102.urdf",
                target_frame_name="end_link",
                action_joint_names=["raw1", "raw2", "raw3"],
                kinematic_joint_names=["joint1", "joint2", "joint3"],
            ),
        ),
    )

    with pytest.raises(ValueError, match="kinematics_verified"):
        _validate_real_robot_hil_config(cfg)

    cfg.processor.leader_kinematics.kinematics_verified = True
    _validate_real_robot_hil_config(cfg)


def test_gamepad_hil_config_requires_matching_gripper_settings():
    cfg = HILSerlRobotEnvConfig(
        robot=SO101FollowerConfig(port="/dev/null"),
        teleop=GamepadTeleopConfig(use_gripper=False),
        processor=HILSerlProcessorConfig(control_mode="gamepad", gripper=GripperConfig(use_gripper=True)),
    )

    with pytest.raises(ValueError, match="use_gripper"):
        _validate_real_robot_hil_config(cfg)
