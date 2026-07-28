# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.envs import HILSerlRobotEnvConfig
from lerobot.model import RobotKinematics
from lerobot.model.hil_kinematics import GGand0MujocoKinematics, WenruoSO101Kinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    CanonicalHILActionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    DirectJointActionOverrideProcessorStep,
    EnvTransition,
    GripperPenaltyProcessorStep,
    GymHILAdapterProcessorStep,
    HILLeader,
    ImageCropResizeProcessorStep,
    InterventionActionProcessorStep,
    LeaderFollowerPoseErrorActionProcessorStep,
    LeaderJointDeltaActionProcessorStep,
    LeaderPolicyTrackingProcessorStep,
    MapDeltaActionToRobotActionStep,
    MapTensorToDeltaActionDictStep,
    Numpy2TorchActionProcessorStep,
    RewardClassifierProcessorStep,
    RobotActionToPolicyActionProcessorStep,
    RobotObservation,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    VanillaObservationProcessorStep,
    create_transition,
    identity_transition,
)
from lerobot.processor.hil_diagnostics import HIL_DIAGNOSTICS_KEY, HILDiagnosticsLogger
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so_follower,
)
from lerobot.robots.robot import Robot
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
)
from lerobot.teleoperators.so_leader import HILSO101Leader
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.utils.import_utils import require_package
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say

from .joint_observations_processor import JointVelocityProcessorStep, MotorCurrentProcessorStep

logging.basicConfig(level=logging.INFO)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    """Main configuration for gym manipulator environment."""

    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None  # Either "record", "replay", None
    device: str = "cpu"


def _validate_real_robot_hil_config(cfg: HILSerlRobotEnvConfig) -> None:
    """Fail early for HIL teleoperator combinations with incompatible action semantics."""
    if cfg.name == "gym_hil":
        return
    if cfg.robot is None or cfg.teleop is None:
        return

    control_mode = cfg.processor.control_mode
    if control_mode not in {"gamepad", "keyboard", "keyboard_ee", "leader"}:
        raise ValueError(f"Unsupported HIL control_mode '{control_mode}'.")

    if control_mode == "gamepad":
        if cfg.teleop.type != "gamepad":
            raise ValueError("HIL control_mode='gamepad' requires teleop.type='gamepad'.")
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        if getattr(cfg.teleop, "use_gripper", use_gripper) != use_gripper:
            raise ValueError("The gamepad and HIL gripper configurations must use the same use_gripper value.")
        return

    if control_mode == "leader":
        strategy = cfg.processor.leader_control_strategy
        if strategy not in {"current", "wenruo", "ggand0"}:
            raise ValueError(f"Unsupported leader_control_strategy '{strategy}'.")
        ik = cfg.processor.inverse_kinematics
        if (
            ik is None
            or ik.end_effector_bounds is None
            or ik.end_effector_step_sizes is None
        ):
            raise ValueError("HIL leader mode requires a complete processor.inverse_kinematics configuration.")
        if strategy == "current" and (ik.urdf_path is None or ik.target_frame_name is None):
            raise ValueError("The current leader strategy requires an IK URDF and target frame.")
        if strategy == "wenruo":
            wenruo = cfg.processor.wenruo_kinematics
            bounds = wenruo.end_effector_bounds
            if (
                set(bounds) != {"min", "max"}
                or any(len(bounds[edge]) != 3 for edge in ("min", "max"))
                or not np.isfinite(np.asarray([bounds["min"], bounds["max"]], dtype=float)).all()
                or np.any(np.asarray(bounds["min"], dtype=float) >= np.asarray(bounds["max"], dtype=float))
            ):
                raise ValueError("wenruo_kinematics.end_effector_bounds must contain finite min/max XYZ bounds.")
            if wenruo.max_joint_update_deg <= 0 or wenruo.max_takeover_position_error_m <= 0:
                raise ValueError("Wenruo joint update and takeover position-error limits must be positive.")
        if cfg.fps <= 0:
            raise ValueError("HIL leader mode requires env.fps to be positive.")
        if (
            cfg.processor.leader_follow_max_joint_speed_deg_s <= 0
            or cfg.processor.leader_follow_max_gripper_speed <= 0
        ):
            raise ValueError("HIL leader follow speeds must be positive.")
        if any(ik.end_effector_step_sizes.get(axis, 0.0) <= 0.0 for axis in ("x", "y", "z")):
            raise ValueError("HIL leader mode requires positive x/y/z end_effector_step_sizes.")

        if cfg.teleop.type == "so101_leader":
            if cfg.robot.type != "so101_follower":
                raise ValueError("SO101 HIL leader mode requires robot.type='so101_follower'.")
            if not getattr(cfg.robot, "use_degrees", False) or not getattr(cfg.teleop, "use_degrees", False):
                raise ValueError("SO101 HIL leader mode requires use_degrees=true for both leader and follower.")
            if strategy == "ggand0":
                max_relative_target = getattr(cfg.robot, "max_relative_target", None)
                if max_relative_target is None:
                    raise ValueError("The ggand0 strategy requires robot.max_relative_target for joint safety.")
                ggand0 = cfg.processor.ggand0_kinematics
                if ggand0.ik_damping <= 0 or ggand0.ik_max_dq_rad <= 0:
                    raise ValueError("ggand0 IK damping and max dq must be positive.")
                if ggand0.max_takeover_joint_error_deg <= 0:
                    raise ValueError("ggand0 takeover alignment threshold must be positive.")
            return

        if strategy != "current":
            raise ValueError(f"The {strategy} leader strategy currently supports only an SO101 leader/follower pair.")

        leader_kinematics = cfg.processor.leader_kinematics
        if (
            leader_kinematics is None
            or leader_kinematics.urdf_path is None
            or leader_kinematics.target_frame_name is None
            or not leader_kinematics.action_joint_names
            or not leader_kinematics.kinematic_joint_names
            or len(leader_kinematics.action_joint_names) != len(leader_kinematics.kinematic_joint_names)
        ):
            raise ValueError("Non-SO101 HIL leader mode requires complete leader_kinematics configuration.")
        if leader_kinematics.gripper_deadband < 0.0:
            raise ValueError("HIL leader gripper_deadband must be non-negative.")
        if any(
            leader_kinematics.joint_scales.get(joint, 1.0) == 0.0
            for joint in leader_kinematics.action_joint_names
        ):
            raise ValueError("HIL leader joint_scales must not contain zero for an action joint.")
        rotation = np.asarray(leader_kinematics.leader_to_follower_rotation, dtype=float)
        if (
            rotation.shape != (3, 3)
            or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)
        ):
            raise ValueError("leader_to_follower_rotation must be a proper 3D rotation matrix.")
        if cfg.robot.type == "koch_rebot_b601_dm_follower" and not leader_kinematics.kinematics_verified:
            raise ValueError(
                "B601 leader execution is blocked until processor.leader_kinematics.kinematics_verified=true."
            )
        return

    if control_mode == "keyboard_ee" and cfg.teleop.type != "keyboard_ee":
        raise ValueError("HIL control_mode='keyboard_ee' requires teleop.type='keyboard_ee'.")


def _preflight_real_robot_kinematics(cfg: HILSerlRobotEnvConfig) -> None:
    """Load the selected model before either arm is connected."""
    if cfg.processor.control_mode != "leader":
        return

    strategy = cfg.processor.leader_control_strategy
    ik = cfg.processor.inverse_kinematics
    assert ik is not None
    if strategy == "current":
        if cfg.robot is not None and cfg.robot.type == "so101_follower":
            RobotKinematics(
                urdf_path=ik.urdf_path,
                target_frame_name=ik.target_frame_name,
                joint_names=[
                    "shoulder_pan",
                    "shoulder_lift",
                    "elbow_flex",
                    "wrist_flex",
                    "wrist_roll",
                    "gripper",
                ],
            )
        leader = cfg.processor.leader_kinematics
        if leader is not None:
            RobotKinematics(
                urdf_path=leader.urdf_path,
                target_frame_name=leader.target_frame_name,
                joint_names=leader.kinematic_joint_names,
            )
    elif strategy == "wenruo":
        wenruo = cfg.processor.wenruo_kinematics
        WenruoSO101Kinematics(
            robot_model=wenruo.robot_model,
            frame=wenruo.frame,
            max_iterations=wenruo.max_iterations,
            learning_rate=wenruo.learning_rate,
            position_tolerance_m=wenruo.position_tolerance_m,
            max_joint_update_deg=wenruo.max_joint_update_deg,
        )
    else:
        ggand0 = cfg.processor.ggand0_kinematics
        GGand0MujocoKinematics(
            model_path=ggand0.model_path,
            end_effector_site=ggand0.end_effector_site,
            ik_damping=ggand0.ik_damping,
            ik_max_dq_rad=ggand0.ik_max_dq_rad,
            locked_joints=ggand0.locked_joints,
            locked_joint_positions_deg=ggand0.locked_joint_positions_deg,
        )


def reset_follower_position(robot_arm: Robot, target_position: np.ndarray) -> None:
    """Reset robot arm to target position using smooth trajectory."""
    current_position_dict = robot_arm.bus.sync_read("Present_Position")
    current_position = np.array(
        [current_position_dict[name] for name in current_position_dict], dtype=np.float32
    )
    trajectory = torch.from_numpy(
        np.linspace(current_position, target_position, 50)
    )  # NOTE: 30 is just an arbitrary number
    for pose in trajectory:
        action_dict = dict(zip(current_position_dict, pose, strict=False))
        robot_arm.bus.sync_write("Goal_Position", action_dict)
        precise_sleep(0.015)


def _return_home_and_shutdown_after_interrupt(env: gym.Env, teleop_device: Teleoperator | None) -> None:
    """Return the follower home, then release both SO101 arms after Ctrl+C."""
    robot = getattr(env, "robot", None)
    reset_pose = getattr(env, "reset_pose", None)

    if robot is not None and getattr(robot, "is_connected", False) and reset_pose is not None:
        try:
            print("[HIL] Ctrl+C: returning follower to the captured home position.", flush=True)
            reset_follower_position(robot, np.asarray(reset_pose, dtype=np.float32))
        except Exception:
            logging.exception("Failed to return the follower to its captured home position during shutdown.")
    else:
        logging.warning("Ctrl+C shutdown has no captured follower home position; skipping follower motion.")

    if robot is not None and getattr(robot, "is_connected", False):
        try:
            robot.bus.disable_torque()
        except Exception:
            logging.exception("Failed to disable follower torque during shutdown.")

    if teleop_device is not None:
        try:
            leader = getattr(teleop_device, "leader", teleop_device)
            if getattr(leader, "is_connected", False):
                leader.disable_torque()
        except Exception:
            logging.exception("Failed to disable leader torque during shutdown.")

    if robot is not None:
        try:
            env.close()
        except Exception:
            logging.exception("Failed to disconnect follower during shutdown.")

    if teleop_device is not None:
        try:
            teleop_device.disconnect()
        except Exception:
            logging.exception("Failed to disconnect leader during shutdown.")


def _read_follower_joint_positions(robot: Robot) -> dict[str, float]:
    observation = robot.get_observation()
    return {
        f"{motor_name}.pos": float(observation[f"{motor_name}.pos"])
        for motor_name in robot.bus.motors
    }


def _capture_initial_reset_positions(
    robot: Robot,
    teleop_device: HILSO101Leader,
    *,
    settle_time_s: float,
    max_joint_delta_deg: float,
) -> tuple[list[float], dict[str, float]]:
    """Capture a stable aligned SO101 leader/follower pose for episode resets."""
    if settle_time_s < 0.0 or max_joint_delta_deg < 0.0:
        raise ValueError("Initial reset pose settle time and maximum joint delta must be non-negative.")

    log_say("Keeping the aligned SO101 arms still while capturing the reset pose.", play_sounds=True)
    follower_first = _read_follower_joint_positions(robot)
    leader_first = teleop_device.get_action()
    precise_sleep(settle_time_s)
    follower_home = _read_follower_joint_positions(robot)
    leader_home = teleop_device.get_action()
    if not leader_first or not leader_home:
        raise RuntimeError("Unable to read SO101 leader positions while capturing the reset pose.")

    moving_joints = {
        f"follower:{key}": abs(float(value) - float(follower_first[key]))
        for key, value in follower_home.items()
        if key in follower_first and abs(float(value) - float(follower_first[key])) > max_joint_delta_deg
    }
    moving_joints.update(
        {
            f"leader:{key}": abs(float(value) - float(leader_first[key]))
            for key, value in leader_home.items()
            if key in leader_first and abs(float(value) - float(leader_first[key])) > max_joint_delta_deg
        }
    )
    if moving_joints:
        raise RuntimeError(
            "SO101 arms moved while capturing the reset pose; align both arms, keep them still, and restart. "
            f"Joint deltas: {moving_joints}"
        )

    follower_pose = [follower_home[f"{motor_name}.pos"] for motor_name in robot.bus.motors]
    leader_pose = {key: float(value) for key, value in leader_home.items() if key.endswith(".pos")}
    logging.info("Captured SO101 reset pose: follower=%s leader=%s", follower_pose, leader_pose)
    return follower_pose, leader_pose


class RobotEnv(gym.Env):
    """Gym environment for robotic control with human intervention support."""

    def __init__(
        self,
        robot,
        use_gripper: bool = False,
        display_cameras: bool = False,
        reset_pose: list[float] | None = None,
        reset_time_s: float = 5.0,
        reset_callback: Callable[[], None] | None = None,
        skip_initial_reset: bool = False,
        diagnostic_logger: HILDiagnosticsLogger | None = None,
    ) -> None:
        """Initialize robot environment with configuration options.

        Args:
            robot: Robot interface for hardware communication.
            use_gripper: Whether to include gripper in action space.
            display_cameras: Whether to show camera feeds during execution.
            reset_pose: Joint positions for environment reset.
            reset_time_s: Time to wait during reset.
        """
        super().__init__()

        self.robot = robot
        self.display_cameras = display_cameras

        # Connect to the robot if not already connected.
        if not self.robot.is_connected:
            self.robot.connect()

        # Episode tracking.
        self.current_step = 0
        self.episode_data = None

        self._joint_names = [f"{key}.pos" for key in self.robot.bus.motors]
        self._image_keys = self.robot.cameras.keys()

        self.reset_pose = reset_pose
        self.reset_time_s = reset_time_s
        self.reset_callback = reset_callback
        self._skip_initial_reset = skip_initial_reset
        self.diagnostic_logger = diagnostic_logger

        self.use_gripper = use_gripper

        self._joint_names = list(self.robot.bus.motors.keys())
        self._raw_joint_positions = None

        self._setup_spaces()

    def _get_observation(self) -> RobotObservation:
        """Get current robot observation including joint positions and camera images."""
        obs_dict = self.robot.get_observation()
        raw_joint_joint_position = {f"{name}.pos": obs_dict[f"{name}.pos"] for name in self._joint_names}
        joint_positions = np.array([raw_joint_joint_position[f"{name}.pos"] for name in self._joint_names])

        images = {key: obs_dict[key] for key in self._image_keys}

        return {"agent_pos": joint_positions, "pixels": images, **raw_joint_joint_position}

    def _setup_spaces(self) -> None:
        """Configure observation and action spaces based on robot capabilities."""
        current_observation = self._get_observation()

        observation_spaces = {}

        # Define observation spaces for images and other states.
        if current_observation is not None and "pixels" in current_observation:
            prefix = OBS_IMAGES
            observation_spaces = {
                f"{prefix}.{key}": gym.spaces.Box(
                    low=0, high=255, shape=current_observation["pixels"][key].shape, dtype=np.uint8
                )
                for key in current_observation["pixels"]
            }

        if current_observation is not None:
            agent_pos = current_observation["agent_pos"]
            observation_spaces[OBS_STATE] = gym.spaces.Box(
                low=0,
                high=10,
                shape=agent_pos.shape,
                dtype=np.float32,
            )

        self.observation_space = gym.spaces.Dict(observation_spaces)

        # Define the action space for joint positions along with setting an intervention flag.
        action_dim = 3
        bounds = {}
        bounds["min"] = -np.ones(action_dim)
        bounds["max"] = np.ones(action_dim)

        if self.use_gripper:
            action_dim += 1
            bounds["min"] = np.concatenate([bounds["min"], [0]])
            bounds["max"] = np.concatenate([bounds["max"], [2]])

        self.action_space = gym.spaces.Box(
            low=bounds["min"],
            high=bounds["max"],
            shape=(action_dim,),
            dtype=np.float32,
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[RobotObservation, dict[str, Any]]:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            Tuple of (observation, info) dictionaries.
        """
        start_time = time.perf_counter()
        if self._skip_initial_reset:
            # The startup pose was just captured from both arms, so moving back
            # to it and waiting for a manual reset would only block the first episode.
            self._skip_initial_reset = False
            print("[HIL] Startup home was captured. Skipping the first automatic reset; controls are active.")
        else:
            if self.reset_pose is not None:
                log_say("Reset the environment.", play_sounds=True)
                reset_follower_position(self.robot, np.array(self.reset_pose))
                log_say("Reset the environment done.", play_sounds=True)
            if self.reset_callback is not None:
                self.reset_callback()
            precise_sleep(max(self.reset_time_s - (time.perf_counter() - start_time), 0.0))

        super().reset(seed=seed, options=options)

        # Reset episode tracking variables.
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        if self.diagnostic_logger is not None:
            self.diagnostic_logger.record(
                "episode_reset",
                {
                    "episode_step": 0,
                    "reset_pose_deg": self.reset_pose,
                    "measured_joint_deg": self._raw_joint_positions,
                },
            )
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    def step(self, action) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        """Execute one environment step with given action."""
        joint_targets_dict = {f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}

        sent_joint_targets = self.robot.send_action(joint_targets_dict)

        obs = self._get_observation()

        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}

        if self.display_cameras:
            self.render()

        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated = False

        return (
            obs,
            reward,
            terminated,
            truncated,
            {
                TeleopEvents.IS_INTERVENTION: False,
                "requested_joint_targets": joint_targets_dict,
                "sent_joint_targets": sent_joint_targets,
            },
        )

    def render(self) -> None:
        """Display robot camera feeds."""
        import cv2

        current_observation = self._get_observation()
        if current_observation is not None:
            image_keys = [key for key in current_observation if "image" in key]

            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(current_observation[key].numpy(), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

    def close(self) -> None:
        """Close environment and disconnect robot."""
        if self.robot.is_connected:
            self.robot.disconnect()

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions."""
        return self._raw_joint_positions


def make_robot_env(
    cfg: HILSerlRobotEnvConfig,
    diagnostic_logger: HILDiagnosticsLogger | None = None,
) -> tuple[gym.Env, Any]:
    """Create robot environment from configuration.

    Args:
        cfg: Environment configuration.

    Returns:
        Tuple of (gym environment, teleoperator device).
    """
    # Check if this is a GymHIL simulation environment
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        require_package("gym-hil", extra="hilserl", import_name="gym_hil")
        import gym_hil  # noqa: F401

        # Extract gripper settings with defaults
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # Real robot environment
    assert cfg.robot is not None, "Robot config must be provided for real robot environment"
    assert cfg.teleop is not None, "Teleop config must be provided for real robot environment"
    _validate_real_robot_hil_config(cfg)
    _preflight_real_robot_kinematics(cfg)

    robot = make_robot_from_config(cfg.robot)
    teleop_device = make_teleoperator_from_config(cfg.teleop)
    if cfg.processor.control_mode == "leader" and cfg.teleop.type == "so101_leader":
        teleop_device = HILSO101Leader(
            teleop_device,
            fps=cfg.fps,
            max_joint_speed_deg_s=cfg.processor.leader_follow_max_joint_speed_deg_s,
            max_gripper_speed=cfg.processor.leader_follow_max_gripper_speed,
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True,
        )
    elif cfg.processor.control_mode == "leader":
        configure_policy_tracking = getattr(teleop_device, "configure_policy_tracking", None)
        if configure_policy_tracking is not None:
            configure_policy_tracking(
                fps=cfg.fps,
                max_joint_speed_deg_s=cfg.processor.leader_follow_max_joint_speed_deg_s,
                max_gripper_speed=cfg.processor.leader_follow_max_gripper_speed,
                enabled=cfg.processor.leader_policy_tracking_enabled,
            )
    teleop_device.connect()

    # Create base environment with safe defaults
    use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
    display_cameras = (
        cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
    )
    reset_config = cfg.processor.reset
    reset_pose = reset_config.fixed_reset_joint_positions if reset_config is not None else None
    reset_callback = None
    if reset_config is not None and reset_config.capture_initial_joint_positions:
        if reset_pose is not None:
            raise ValueError(
                "Set either fixed_reset_joint_positions or capture_initial_joint_positions, not both."
            )
        if not isinstance(teleop_device, HILSO101Leader):
            raise ValueError("Captured reset poses currently require an SO101 HIL leader teleoperator.")
        if not robot.is_connected:
            robot.connect()
        reset_pose, leader_home = _capture_initial_reset_positions(
            robot,
            teleop_device,
            settle_time_s=reset_config.initial_position_settle_time_s,
            max_joint_delta_deg=reset_config.initial_position_max_delta_deg,
        )
        print("[HIL] Captured a stable leader/follower home pose.")
        if diagnostic_logger is not None:
            diagnostic_logger.record(
                "startup_home_captured",
                {
                    "motor_order": list(robot.bus.motors),
                    "follower_home_deg": reset_pose,
                    "leader_home_deg": leader_home,
                },
            )

        def reset_leader_to_home() -> None:
            teleop_device.move_to_home_position(
                leader_home,
                timeout_s=reset_config.leader_home_timeout_s,
            )

        reset_callback = reset_leader_to_home

    env = RobotEnv(
        robot=robot,
        use_gripper=use_gripper,
        display_cameras=display_cameras,
        reset_pose=reset_pose,
        reset_time_s=reset_config.reset_time_s if reset_config is not None else 5.0,
        reset_callback=reset_callback,
        skip_initial_reset=reset_config.capture_initial_joint_positions if reset_config is not None else False,
        diagnostic_logger=diagnostic_logger,
    )

    return env, teleop_device


def make_processors(
    env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str = "cpu"
) -> tuple[
    DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]
]:
    """Create environment and action processors.

    Args:
        env: Robot environment instance.
        teleop_device: Teleoperator device for intervention.
        cfg: Processor configuration.
        device: Target device for computations.

    Returns:
        Tuple of (environment processor, action processor).
    """
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )
    success_label_hold_steps = (
        cfg.processor.reset.success_label_hold_steps if cfg.processor.reset is not None else 1
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(
                terminate_on_success=terminate_on_success,
                success_label_hold_steps=success_label_hold_steps,
            ),
            Torch2NumpyActionProcessorStep(),
        ]

        env_pipeline_steps = [
            GymHILAdapterProcessorStep(),
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
        ]

        # Add time limit processor if reset config exists
        if cfg.processor.reset is not None:
            env_pipeline_steps.append(
                TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
            )

        env_pipeline_steps.extend(
            [
                AddBatchDimensionProcessorStep(),
                DeviceProcessorStep(device=device),
            ]
        )

        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        )

    # Full processor pipeline for real robot environment
    # Get robot and motor information for kinematics
    motor_names = list(env.robot.bus.motors.keys())
    diagnostics_enabled = getattr(env, "diagnostic_logger", None) is not None

    # Set up kinematics solver if inverse kinematics is configured
    strategy = cfg.processor.leader_control_strategy
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None and strategy == "current":
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=motor_names,
        )
    elif cfg.processor.inverse_kinematics is not None and strategy == "wenruo":
        wenruo = cfg.processor.wenruo_kinematics
        kinematics_solver = WenruoSO101Kinematics(
            robot_model=wenruo.robot_model,
            frame=wenruo.frame,
            max_iterations=wenruo.max_iterations,
            learning_rate=wenruo.learning_rate,
            position_tolerance_m=wenruo.position_tolerance_m,
            max_joint_update_deg=wenruo.max_joint_update_deg,
        )
    elif cfg.processor.inverse_kinematics is not None and strategy == "ggand0":
        ggand0 = cfg.processor.ggand0_kinematics
        kinematics_solver = GGand0MujocoKinematics(
            model_path=ggand0.model_path,
            end_effector_site=ggand0.end_effector_site,
            ik_damping=ggand0.ik_damping,
            ik_max_dq_rad=ggand0.ik_max_dq_rad,
            locked_joints=ggand0.locked_joints,
            locked_joint_positions_deg=ggand0.locked_joint_positions_deg,
        )

    env_pipeline_steps = [VanillaObservationProcessorStep()]

    if cfg.processor.observation is not None:
        if cfg.processor.observation.add_joint_velocity_to_observation:
            env_pipeline_steps.append(JointVelocityProcessorStep(dt=1.0 / cfg.fps))
        if cfg.processor.observation.add_current_to_observation:
            env_pipeline_steps.append(MotorCurrentProcessorStep(robot=env.robot))

    add_ee_pose = (
        cfg.processor.observation is not None and cfg.processor.observation.add_ee_pose_to_observation
    )
    if kinematics_solver is not None and add_ee_pose:
        env_pipeline_steps.append(
            ForwardKinematicsJointsToEEObservation(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            )
        )

    if cfg.processor.image_preprocessing is not None:
        env_pipeline_steps.append(
            ImageCropResizeProcessorStep(
                crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                resize_size=cfg.processor.image_preprocessing.resize_size,
            )
        )

    # Add time limit processor if reset config exists
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )

    # Add gripper penalty processor if gripper config exists and enabled
    # Only add if max_gripper_pos is explicitly configured (required for normalization)
    if (
        cfg.processor.gripper is not None
        and cfg.processor.gripper.use_gripper
        and cfg.processor.max_gripper_pos is not None
    ):
        env_pipeline_steps.append(
            GripperPenaltyProcessorStep(
                penalty=cfg.processor.gripper.gripper_penalty,
                max_gripper_pos=cfg.processor.max_gripper_pos,
            )
        )

    if (
        cfg.processor.reward_classifier is not None
        and cfg.processor.reward_classifier.pretrained_path is not None
    ):
        env_pipeline_steps.append(
            RewardClassifierProcessorStep(
                pretrained_path=cfg.processor.reward_classifier.pretrained_path,
                device=device,
                success_threshold=cfg.processor.reward_classifier.success_threshold,
                success_reward=cfg.processor.reward_classifier.success_reward,
                terminate_on_success=terminate_on_success,
            )
        )

    env_pipeline_steps.append(AddBatchDimensionProcessorStep())
    env_pipeline_steps.append(DeviceProcessorStep(device=device))

    action_pipeline_steps = [
        AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
        AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
    ]

    if cfg.processor.control_mode == "leader":
        if not isinstance(teleop_device, HILLeader):
            raise ValueError("HIL leader mode requires a teleoperator implementing the leader capability contract.")
        leader_kinematics_config = cfg.processor.leader_kinematics
        leader_kinematics_solver = kinematics_solver
        leader_motor_names = motor_names
        leader_action_joint_names = None
        leader_joint_scales = None
        leader_joint_offsets = None
        leader_rotation = None
        leader_gripper_key = "gripper.pos"
        leader_gripper_deadband = 0.5
        if leader_kinematics_config is not None and strategy == "current":
            leader_kinematics_solver = RobotKinematics(
                urdf_path=leader_kinematics_config.urdf_path,
                target_frame_name=leader_kinematics_config.target_frame_name,
                joint_names=leader_kinematics_config.kinematic_joint_names,
            )
            leader_motor_names = leader_kinematics_config.kinematic_joint_names
            leader_action_joint_names = leader_kinematics_config.action_joint_names
            leader_joint_scales = leader_kinematics_config.joint_scales
            leader_joint_offsets = leader_kinematics_config.joint_offsets_deg
            leader_rotation = leader_kinematics_config.leader_to_follower_rotation
            leader_gripper_key = leader_kinematics_config.gripper_action_key
            leader_gripper_deadband = leader_kinematics_config.gripper_deadband
        if leader_kinematics_solver is None:
            raise ValueError("HIL leader mode requires an FK solver.")
        leader_steps = [
            LeaderPolicyTrackingProcessorStep(
                teleop_device=teleop_device,
                enabled=cfg.processor.leader_policy_tracking_enabled,
            )
        ]
        if strategy == "current":
            leader_steps.append(
                LeaderJointDeltaActionProcessorStep(
                    kinematics=leader_kinematics_solver,
                    motor_names=leader_motor_names,
                    end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True,
                    gripper_deadband=leader_gripper_deadband,
                    gripper_open_on_positive_delta=cfg.processor.leader_gripper_open_on_positive_delta,
                    leader_action_joint_names=leader_action_joint_names,
                    joint_scales=leader_joint_scales,
                    joint_offsets_deg=leader_joint_offsets,
                    leader_to_follower_rotation=leader_rotation,
                    gripper_action_key=leader_gripper_key,
                    diagnostics_enabled=diagnostics_enabled,
                )
            )
        else:
            if leader_kinematics_config is not None:
                leader_action_joint_names = leader_kinematics_config.action_joint_names or None
                leader_joint_scales = leader_kinematics_config.joint_scales
                leader_joint_offsets = leader_kinematics_config.joint_offsets_deg
                leader_rotation = leader_kinematics_config.leader_to_follower_rotation
                leader_gripper_key = leader_kinematics_config.gripper_action_key
                leader_gripper_deadband = leader_kinematics_config.gripper_deadband
            leader_steps.append(
                LeaderFollowerPoseErrorActionProcessorStep(
                    leader_kinematics=kinematics_solver,
                    follower_kinematics=kinematics_solver,
                    follower_motor_names=motor_names,
                    end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                    strategy=strategy,
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True,
                    gripper_deadband=leader_gripper_deadband,
                    gripper_open_on_positive_delta=cfg.processor.leader_gripper_open_on_positive_delta,
                    leader_action_joint_names=leader_action_joint_names,
                    joint_scales=leader_joint_scales,
                    joint_offsets_deg=leader_joint_offsets,
                    leader_to_follower_rotation=leader_rotation,
                    gripper_action_key=leader_gripper_key,
                    direct_joint_mirror=strategy == "ggand0",
                    max_takeover_joint_error_deg=cfg.processor.ggand0_kinematics.max_takeover_joint_error_deg,
                    max_takeover_position_error_m=(
                        cfg.processor.wenruo_kinematics.max_takeover_position_error_m
                        if strategy == "wenruo"
                        else None
                    ),
                    rebase_on_intervention=(
                        cfg.processor.wenruo_kinematics.rebase_on_intervention
                        if strategy == "wenruo"
                        else False
                    ),
                    diagnostics_enabled=diagnostics_enabled,
                )
            )
        action_pipeline_steps.extend(leader_steps)

    action_pipeline_steps.append(
        InterventionActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
            terminate_on_success=terminate_on_success,
            success_label_hold_steps=success_label_hold_steps,
        )
    )
    action_pipeline_steps.append(
        CanonicalHILActionProcessorStep(
            use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
        )
    )

    # Replace InverseKinematicsProcessor with new kinematic processors
    if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
        # Add EE bounds and safety processor
        inverse_kinematics_steps = [
            MapTensorToDeltaActionDictStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
            ),
            MapDeltaActionToRobotActionStep(),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                # Human Cartesian intervention must be relative to the follower's
                # measured pose. Chaining unexecuted IK targets makes the reference
                # run ahead whenever the safety cap slows a joint.
                use_ik_solution=False,
                diagnostics_enabled=diagnostics_enabled,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=(
                    cfg.processor.wenruo_kinematics.end_effector_bounds
                    if strategy == "wenruo"
                    else cfg.processor.inverse_kinematics.end_effector_bounds
                ),
                diagnostics_enabled=diagnostics_enabled,
            ),
            GripperVelocityToJoint(
                clip_max=cfg.processor.max_gripper_pos,
                speed_factor=1.0,
                discrete_gripper=True,
            ),
            InverseKinematicsRLStep(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=strategy == "wenruo",
                diagnostics_enabled=diagnostics_enabled,
            ),
        ]
        action_pipeline_steps.extend(inverse_kinematics_steps)
        action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=motor_names))
        if strategy == "ggand0":
            action_pipeline_steps.append(DirectJointActionOverrideProcessorStep(motor_names=motor_names))

    return DataProcessorPipeline(
        steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    ), DataProcessorPipeline(
        steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
    )


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline.

    Args:
        env: The robot environment
        transition: Current transition state
        action: Action to execute
        env_processor: Environment processor
        action_processor: Action processor

    Returns:
        Processed transition with updated state.
    """

    step_started_at = time.perf_counter()

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    action_processed_at = time.perf_counter()
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)
    robot_step_finished_at = time.perf_counter()

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    diagnostics = complementary_data.pop(HIL_DIAGNOSTICS_KEY, {})

    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions

    # Merge env and action-processor info: env wins for str keys, action-processor
    # wins for `TeleopEvents` enum keys
    action_info = processed_action_transition[TransitionKey.INFO]
    new_info = info.copy()
    for key, value in action_info.items():
        if isinstance(key, TeleopEvents):
            new_info[key] = value

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data,
    )
    new_transition = env_processor(new_transition)

    diagnostic_logger = getattr(env, "diagnostic_logger", None)
    if diagnostic_logger is not None:
        requested_targets = info.get("requested_joint_targets", {})
        sent_targets = info.get("sent_joint_targets", {})
        measured_after = {
            key: value
            for key, value in obs.items()
            if isinstance(key, str) and key.endswith(".pos")
        }
        command_clip_delta = {
            key: float(sent_targets[key]) - float(requested_targets[key])
            for key in requested_targets.keys() & sent_targets.keys()
        }
        tracking_error = {
            key: float(measured_after[key]) - float(sent_targets[key])
            for key in sent_targets.keys() & measured_after.keys()
        }
        leader_diagnostics = diagnostics.get("leader", {})
        ik_diagnostics = diagnostics.get("inverse_kinematics", {})
        leader_delta_mm = np.asarray(leader_diagnostics.get("delta_xyz_m", [0.0, 0.0, 0.0])) * 1000
        ik_delta = ik_diagnostics.get("solution_minus_measured_deg", {})
        max_ik_delta = max((abs(float(value)) for value in ik_delta.values()), default=0.0)
        max_command_clip = max((abs(value) for value in command_clip_delta.values()), default=0.0)
        residual_mm = float(ik_diagnostics.get("position_residual_m", 0.0)) * 1000
        diagnostic_logger.record(
            "control_step",
            {
                "episode_step": max(getattr(env, "current_step", 1) - 1, 0),
                "is_intervention": bool(
                    processed_action_transition.get(TransitionKey.INFO, {}).get(
                        TeleopEvents.IS_INTERVENTION, False
                    )
                ),
                "timing_ms": {
                    "action_processing": (action_processed_at - step_started_at) * 1000,
                    "robot_send_and_observe": (robot_step_finished_at - action_processed_at) * 1000,
                    "before_observation_processing": (robot_step_finished_at - step_started_at) * 1000,
                },
                "stages": diagnostics,
                "requested_joint_targets_deg": requested_targets,
                "sent_joint_targets_deg": sent_targets,
                "command_clip_delta_deg": command_clip_delta,
                "measured_joint_after_deg": measured_after,
                "measured_minus_sent_deg": tracking_error,
            },
            console_summary=(
                f"step={max(getattr(env, 'current_step', 1) - 1, 0)} "
                f"intervention={bool(leader_diagnostics.get('is_intervention', False))} "
                f"leader_dp_mm={np.round(leader_delta_mm, 2).tolist()} "
                f"ik_dq_max_deg={max_ik_delta:.2f} "
                f"send_clip_max_deg={max_command_clip:.2f} "
                f"ik_residual_mm={residual_mm:.3f}"
            ),
        )

    return new_transition


def reset_and_build_transition(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """Reset env + processors and return the first env-processed transition."""
    # Stop any active leader feedback before the follower or leader starts its
    # reset trajectory, then clear all episode-local processor state.
    action_processor.reset()
    env_processor.reset()
    obs, info = env.reset()
    complementary_data: dict[str, Any] = {}
    if hasattr(env, "get_raw_joint_positions"):
        raw_joint_positions = env.get_raw_joint_positions()
        if raw_joint_positions is not None:
            complementary_data["raw_joint_positions"] = raw_joint_positions
    transition = create_transition(observation=obs, info=info, complementary_data=complementary_data)
    return env_processor(data=transition)


def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    teleop_device: Teleoperator,
    cfg: GymManipulatorConfig,
) -> None:
    """Main control loop for robot environment interaction.
    if cfg.mode == "record": then a dataset will be created and recorded

    Args:
     env: The robot environment
     env_processor: Environment processor
     action_processor: Action processor
     teleop_device: Teleoperator device
     cfg: gym_manipulator configuration
    """
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Press Space once to enable SO101 leader intervention")
    print("- Press Space again to disable leader intervention")
    print("- Press S to label success and Esc to end the episode")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    transition = reset_and_build_transition(env, env_processor, action_processor)

    # Determine if gripper is used
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        if teleop_device:
            action_features = teleop_device.action_features
        else:
            action_features = {
                "dtype": "float32",
                "shape": (4,),
                "names": ["delta_x", "delta_y", "delta_z", "gripper"],
            }
        features = {
            ACTION: action_features,
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None,
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"],
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features,
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    try:
        while episode_idx < cfg.dataset.num_episodes_to_record:
            step_start_time = time.perf_counter()

            # Create a neutral action (no movement)
            neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
            if use_gripper:
                neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

            observation = {
                k: v.squeeze(0).cpu()
                for k, v in transition[TransitionKey.OBSERVATION].items()
                if isinstance(v, torch.Tensor)
            }

            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=neutral_action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            terminated = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)

            if cfg.mode == "record":
                action_to_record = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                    "teleop_action", transition[TransitionKey.ACTION]
                )
                frame = {
                    **observation,
                    ACTION: action_to_record.cpu(),
                    REWARD: np.array([transition[TransitionKey.REWARD]], dtype=np.float32),
                    DONE: np.array([terminated or truncated], dtype=bool),
                }
                if use_gripper:
                    discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA].get(
                        "discrete_penalty", 0.0
                    )
                    frame["complementary_info.discrete_penalty"] = np.array(
                        [discrete_penalty], dtype=np.float32
                    )

                if dataset is not None:
                    frame["task"] = cfg.dataset.task
                    dataset.add_frame(frame)

            episode_step += 1

            # Handle episode termination
            if terminated or truncated:
                episode_time = time.perf_counter() - episode_start_time
                logging.info(
                    f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"
                )
                episode_step = 0
                episode_idx += 1

                if dataset is not None:
                    if transition[TransitionKey.INFO].get(TeleopEvents.RERECORD_EPISODE, False):
                        logging.info(f"Re-recording episode {episode_idx}")
                        dataset.clear_episode_buffer()
                        episode_idx -= 1
                    else:
                        logging.info(f"Saving episode {episode_idx}")
                        dataset.save_episode()

                # Reset for new episode
                transition = reset_and_build_transition(env, env_processor, action_processor)

            # Maintain fps timing
            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
    finally:
        if dataset is not None and dataset.writer is not None and dataset.writer.image_writer is not None:
            logging.info("Waiting for image writer to finish...")
            dataset.writer.image_writer.stop()

    if dataset is not None and cfg.dataset.push_to_hub:
        logging.info("Finalizing dataset before pushing to hub")
        dataset.finalize()
        logging.info("Pushing dataset to hub")
        dataset.push_to_hub()


def replay_trajectory(
    env: gym.Env, action_processor: DataProcessorPipeline, cfg: GymManipulatorConfig
) -> None:
    """Replay recorded trajectory on robot environment."""
    assert cfg.dataset.replay_episode is not None, "Replay episode must be provided for replay"

    dataset = LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.replay_episode],
        download_videos=False,
    )
    actions = dataset.select_columns(ACTION)

    _, info = env.reset()

    for action_data in actions:
        start_time = time.perf_counter()
        transition = create_transition(
            observation=env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {},
            action=action_data[ACTION],
        )
        transition = action_processor(transition)
        env.step(transition[TransitionKey.ACTION])
        precise_sleep(max(1 / cfg.env.fps - (time.perf_counter() - start_time), 0.0))


@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """Main entry point for gym manipulator script."""
    env: gym.Env | None = None
    teleop_device: Teleoperator | None = None
    diagnostic_logger: HILDiagnosticsLogger | None = None
    interrupted = False
    try:
        diagnostics_config = cfg.env.processor.diagnostics
        if diagnostics_config.enabled:
            diagnostic_logger = HILDiagnosticsLogger(
                diagnostics_config.log_dir,
                console_summary=diagnostics_config.console_summary,
            )
            diagnostic_logger.record(
                "run_started",
                {
                    "mode": cfg.mode,
                    "device": cfg.device,
                    "fps": cfg.env.fps,
                    "control_mode": cfg.env.processor.control_mode,
                    "leader_control_strategy": cfg.env.processor.leader_control_strategy,
                    "robot_type": cfg.env.robot.type if cfg.env.robot is not None else None,
                    "robot_port": getattr(cfg.env.robot, "port", None),
                    "teleop_type": cfg.env.teleop.type if cfg.env.teleop is not None else None,
                    "teleop_port": getattr(cfg.env.teleop, "port", None),
                    "max_relative_target_deg": getattr(cfg.env.robot, "max_relative_target", None),
                    "urdf_path": (
                        cfg.env.processor.inverse_kinematics.urdf_path
                        if cfg.env.processor.inverse_kinematics is not None
                        else None
                    ),
                    "target_frame_name": (
                        cfg.env.processor.inverse_kinematics.target_frame_name
                        if cfg.env.processor.inverse_kinematics is not None
                        else None
                    ),
                },
            )

        env, teleop_device = make_robot_env(cfg.env, diagnostic_logger=diagnostic_logger)
        env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

        print("Environment observation space:", env.observation_space)
        print("Environment action space:", env.action_space)
        print("Environment processor:", env_processor)
        print("Action processor:", action_processor)

        if cfg.mode == "replay":
            replay_trajectory(env, action_processor, cfg)
        else:
            control_loop(env, env_processor, action_processor, teleop_device, cfg)
    except KeyboardInterrupt:
        interrupted = True
        if diagnostic_logger is not None:
            diagnostic_logger.record("run_interrupted")
        print("[HIL] Ctrl+C received. Ending the run safely.", flush=True)
        if env is not None:
            _return_home_and_shutdown_after_interrupt(env, teleop_device)
    except Exception as exc:
        if diagnostic_logger is not None:
            diagnostic_logger.record(
                "run_failed",
                {"exception_type": type(exc).__name__, "message": str(exc)},
            )
        raise
    finally:
        if env is not None and not interrupted:
            try:
                env.close()
            except Exception:
                logging.exception("Failed to disconnect follower during shutdown.")
        if teleop_device is not None and not interrupted:
            try:
                teleop_device.disconnect()
            except Exception:
                logging.exception("Failed to disconnect leader during shutdown.")
        if diagnostic_logger is not None:
            diagnostic_logger.close()


if __name__ == "__main__":
    main()
