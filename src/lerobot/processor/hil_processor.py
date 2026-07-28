#!/usr/bin/env python

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
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

import numpy as np
import torch
import torchvision.transforms.functional as F  # noqa: N812

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.teleoperators.utils import TeleopEvents

if TYPE_CHECKING:
    from lerobot.teleoperators.teleoperator import Teleoperator

from lerobot.types import EnvTransition, PolicyAction, TransitionKey

from .hil_diagnostics import update_hil_diagnostics
from .pipeline import (
    ComplementaryDataProcessorStep,
    InfoProcessorStep,
    ObservationProcessorStep,
    ProcessorStep,
    ProcessorStepRegistry,
    TruncatedProcessorStep,
)

GRIPPER_KEY = "gripper"
DISCRETE_PENALTY_KEY = "discrete_penalty"
TELEOP_ACTION_KEY = "teleop_action"
DIRECT_JOINT_ACTION_KEY = "hil_direct_joint_action"


logger = logging.getLogger(__name__)


@runtime_checkable
class HasTeleopEvents(Protocol):
    """
    Minimal protocol for objects that provide teleoperation events.

    This protocol defines the `get_teleop_events()` method, allowing processor
    steps to interact with teleoperators that support event-based controls
    (like episode termination or success flagging) without needing to know the
    teleoperator's specific class.
    """

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the teleoperator.

        Returns:
            A dictionary containing control events such as:
            - `is_intervention`: bool - Whether the human is currently intervening.
            - `terminate_episode`: bool - Whether to terminate the current episode.
            - `success`: bool - Whether the episode was successful.
            - `rerecord_episode`: bool - Whether to rerecord the episode.
        """
        ...


@runtime_checkable
class HILLeader(HasTeleopEvents, Protocol):
    """Capability contract for leader devices used by the HIL pipeline.

    The raw leader hardware is intentionally not specified here. A leader only
    needs to expose its raw joint action, one-shot HIL events, and the methods
    needed to enter and leave policy-following mode.
    """

    def get_action(self) -> dict[str, float]: ...

    def update_policy_tracking(self, follower_positions: dict[str, float]) -> None: ...

    def stop_policy_tracking(self) -> None: ...

    def reset_episode(self) -> None: ...


# Type variable constrained to Teleoperator subclasses that also implement events
TeleopWithEvents = TypeVar("TeleopWithEvents", bound="Teleoperator")


def _check_teleop_with_events(teleop: "Teleoperator") -> None:
    """
    Runtime check that a teleoperator implements the `HasTeleopEvents` protocol.

    Args:
        teleop: The teleoperator instance to check.

    Raises:
        TypeError: If the teleoperator does not have a `get_teleop_events` method.
    """
    if not isinstance(teleop, HasTeleopEvents):
        raise TypeError(
            f"Teleoperator {type(teleop).__name__} must implement get_teleop_events() method. "
            f"Compatible teleoperators: GamepadTeleop, KeyboardEndEffectorTeleop"
        )


@ProcessorStepRegistry.register("add_teleop_action_as_complementary_data")
@dataclass
class AddTeleopActionAsComplimentaryDataStep(ComplementaryDataProcessorStep):
    """
    Adds the raw action from a teleoperator to the transition's complementary data.

    This is useful for human-in-the-loop scenarios where the human's input needs to
    be available to downstream processors, for example, to override a policy's action
    during an intervention.

    Attributes:
        teleop_device: The teleoperator instance to get the action from.
    """

    teleop_device: "Teleoperator"

    def complementary_data(self, complementary_data: dict) -> dict:
        """
        Retrieves the teleoperator's action and adds it to the complementary data.

        Args:
            complementary_data: The incoming complementary data dictionary.

        Returns:
            A new dictionary with the teleoperator action added under the
            `teleop_action` key.
        """
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = self.teleop_device.get_action()
        return new_complementary_data

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("add_teleop_action_as_info")
@dataclass
class AddTeleopEventsAsInfoStep(InfoProcessorStep):
    """
    Adds teleoperator control events (e.g., terminate, success) to the transition's info.

    This step extracts control events from teleoperators that support event-based
    interaction, making these signals available to other parts of the system.

    Attributes:
        teleop_device: An instance of a teleoperator that implements the
                       `HasTeleopEvents` protocol.
    """

    teleop_device: TeleopWithEvents

    def __post_init__(self):
        """Validates that the provided teleoperator supports events after initialization."""
        _check_teleop_with_events(self.teleop_device)

    def info(self, info: dict) -> dict:
        """
        Retrieves teleoperator events and updates the info dictionary.

        Args:
            info: The incoming info dictionary.

        Returns:
            A new dictionary including the teleoperator events.
        """
        new_info = dict(info)

        teleop_events = self.teleop_device.get_teleop_events()
        new_info.update(teleop_events)
        return new_info

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("leader_policy_tracking_processor")
class LeaderPolicyTrackingProcessorStep(ProcessorStep):
    """Coordinate any capable leader's feedback with HIL intervention state."""

    teleop_device: Any
    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
        observation = transition.get(TransitionKey.OBSERVATION) or {}

        try:
            if is_intervention or not self.enabled:
                self.teleop_device.stop_policy_tracking()
            else:
                self.teleop_device.update_policy_tracking(observation)
        except Exception as exc:
            # The next processor frame will carry the failure event; do not
            # continue forcing a leader that has lost its feedback channel.
            logger.exception("Leader policy tracking failed: %s", exc)
            report_failure = getattr(self.teleop_device, "report_failure", None)
            if report_failure is not None:
                report_failure()
            try:
                self.teleop_device.stop_policy_tracking()
            except Exception:
                logger.exception("Unable to stop leader policy tracking after failure.")
        return transition

    def reset(self) -> None:
        self.teleop_device.stop_policy_tracking()
        self.teleop_device.reset_episode()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("leader_joint_delta_action_processor")
class LeaderJointDeltaActionProcessorStep(ProcessorStep):
    """Convert calibrated leader joint motion into HIL-SERL Cartesian deltas."""

    kinematics: Any
    motor_names: list[str]
    end_effector_step_sizes: dict[str, float]
    use_gripper: bool = True
    gripper_deadband: float = 0.5
    gripper_open_on_positive_delta: bool = True
    leader_action_joint_names: list[str] | None = None
    joint_scales: dict[str, float] | None = None
    joint_offsets_deg: dict[str, float] | None = None
    leader_to_follower_rotation: list[list[float]] | np.ndarray | None = None
    gripper_action_key: str = "gripper.pos"
    diagnostics_enabled: bool = False

    _previous_position: np.ndarray | None = None
    _previous_gripper: float | None = None
    _was_intervening: bool = False

    def __post_init__(self) -> None:
        action_joint_names = self.leader_action_joint_names or self.motor_names
        if len(action_joint_names) != len(self.motor_names):
            raise ValueError(
                "leader_action_joint_names must have the same length as the leader kinematic joint list."
            )
        rotation = self.leader_to_follower_rotation
        if rotation is None:
            self._rotation = np.eye(3, dtype=float)
            return
        self._rotation = np.asarray(rotation, dtype=float)
        if self._rotation.shape != (3, 3) or not np.isfinite(self._rotation).all():
            raise ValueError("leader_to_follower_rotation must be a finite 3x3 rotation matrix.")
        if not np.allclose(self._rotation.T @ self._rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(self._rotation), 1.0, atol=1e-4
        ):
            raise ValueError("leader_to_follower_rotation must be a proper 3D rotation matrix.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        leader_action = complementary_data.get(TELEOP_ACTION_KEY)

        if not isinstance(leader_action, dict):
            self._was_intervening = False
            return transition

        action_joint_names = self.leader_action_joint_names or self.motor_names
        try:
            joints = np.array(
                [
                    float(leader_action[f"{name}.pos"])
                    * float((self.joint_scales or {}).get(name, 1.0))
                    + float((self.joint_offsets_deg or {}).get(name, 0.0))
                    for name in action_joint_names
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError):
            if is_intervention:
                return self._with_neutral_action(transition)
            self._was_intervening = False
            return transition

        position = np.asarray(self.kinematics.forward_kinematics(joints), dtype=float)[:3, 3]
        gripper = float(leader_action.get(self.gripper_action_key, 0.0))
        previous_position = None if self._previous_position is None else self._previous_position.copy()

        if not is_intervention or not self._was_intervening or self._previous_position is None:
            self._previous_position = position
            self._previous_gripper = gripper
            self._was_intervening = is_intervention
            if is_intervention:
                result = self._with_neutral_action(transition)
                state = "intervention_started"
            else:
                result = transition
                state = "intervention_inactive"
            if self.diagnostics_enabled:
                update_hil_diagnostics(
                    result,
                    "leader",
                    {
                        "state": state,
                        "is_intervention": is_intervention,
                        "raw_joint_deg": {
                            name: float(leader_action[f"{name}.pos"]) for name in action_joint_names
                        },
                        "kinematic_joint_deg": dict(zip(self.motor_names, joints, strict=True)),
                        "fk_xyz_m": position,
                        "previous_fk_xyz_m": previous_position,
                        "delta_xyz_m": [0.0, 0.0, 0.0],
                        "normalized_delta_unclipped": [0.0, 0.0, 0.0],
                        "normalized_delta_clipped": [0.0, 0.0, 0.0],
                        "gripper_position": gripper,
                        "gripper_delta": None,
                        "gripper_action": 1.0 if self.use_gripper else None,
                    },
                )
            return result

        delta = self._rotation @ (position - self._previous_position)
        self._previous_position = position
        previous_gripper = self._previous_gripper if self._previous_gripper is not None else gripper
        self._previous_gripper = gripper

        normalized_delta_unclipped = np.array(
            [
                delta[0] / self.end_effector_step_sizes["x"],
                delta[1] / self.end_effector_step_sizes["y"],
                delta[2] / self.end_effector_step_sizes["z"],
            ]
        )
        normalized_delta = np.clip(normalized_delta_unclipped, -1.0, 1.0)
        action: dict[str, float] = {
            "delta_x": float(normalized_delta[0]),
            "delta_y": float(normalized_delta[1]),
            "delta_z": float(normalized_delta[2]),
        }
        if self.use_gripper:
            gripper_delta = gripper - previous_gripper
            is_opening = (
                gripper_delta > self.gripper_deadband
                if self.gripper_open_on_positive_delta
                else gripper_delta < -self.gripper_deadband
            )
            if is_opening:
                action[GRIPPER_KEY] = 2.0
            elif abs(gripper_delta) > self.gripper_deadband:
                action[GRIPPER_KEY] = 0.0
            else:
                action[GRIPPER_KEY] = 1.0

        new_transition = transition.copy()
        new_complementary_data = dict(complementary_data)
        new_complementary_data[TELEOP_ACTION_KEY] = action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data
        if self.diagnostics_enabled:
            update_hil_diagnostics(
                new_transition,
                "leader",
                {
                    "state": "intervening",
                    "is_intervention": True,
                    "raw_joint_deg": {
                        name: float(leader_action[f"{name}.pos"]) for name in action_joint_names
                    },
                    "kinematic_joint_deg": dict(zip(self.motor_names, joints, strict=True)),
                    "fk_xyz_m": position,
                    "previous_fk_xyz_m": previous_position,
                    "delta_xyz_m": delta,
                    "normalized_delta_unclipped": normalized_delta_unclipped,
                    "normalized_delta_clipped": normalized_delta,
                    "clipped_axes": [
                        axis
                        for axis, raw, clipped in zip(
                            ("x", "y", "z"), normalized_delta_unclipped, normalized_delta, strict=True
                        )
                        if not np.isclose(raw, clipped)
                    ],
                    "gripper_position": gripper,
                    "gripper_delta": gripper - previous_gripper,
                    "gripper_action": action.get(GRIPPER_KEY),
                },
            )
        return new_transition

    def _with_neutral_action(self, transition: EnvTransition) -> EnvTransition:
        action: dict[str, float] = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        if self.use_gripper:
            action[GRIPPER_KEY] = 1.0
        new_transition = transition.copy()
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary_data[TELEOP_ACTION_KEY] = action
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    def reset(self) -> None:
        self._previous_position = None
        self._previous_gripper = None
        self._was_intervening = False

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("leader_follower_pose_error_action_processor")
class LeaderFollowerPoseErrorActionProcessorStep(ProcessorStep):
    """Map leader/follower EE error to the canonical HIL action."""

    leader_kinematics: Any
    follower_kinematics: Any
    follower_motor_names: list[str]
    end_effector_step_sizes: dict[str, float]
    strategy: str
    use_gripper: bool = True
    gripper_deadband: float = 0.5
    gripper_open_on_positive_delta: bool = True
    leader_action_joint_names: list[str] | None = None
    joint_scales: dict[str, float] | None = None
    joint_offsets_deg: dict[str, float] | None = None
    leader_to_follower_rotation: list[list[float]] | np.ndarray | None = None
    gripper_action_key: str = "gripper.pos"
    direct_joint_mirror: bool = False
    max_takeover_joint_error_deg: float = 8.0
    max_takeover_position_error_m: float | None = None
    rebase_on_intervention: bool = False
    diagnostics_enabled: bool = False

    _was_intervening: bool = False
    _previous_gripper: float | None = None
    _leader_reference_xyz: np.ndarray | None = None
    _follower_reference_xyz: np.ndarray | None = None

    def __post_init__(self) -> None:
        rotation = self.leader_to_follower_rotation
        self._rotation = np.asarray(np.eye(3) if rotation is None else rotation, dtype=float)
        if self._rotation.shape != (3, 3) or not np.isfinite(self._rotation).all():
            raise ValueError("leader_to_follower_rotation must be a finite 3x3 rotation matrix.")
        if not np.allclose(self._rotation.T @ self._rotation, np.eye(3), atol=1e-4) or not np.isclose(
            np.linalg.det(self._rotation), 1.0, atol=1e-4
        ):
            raise ValueError("leader_to_follower_rotation must be a proper 3D rotation matrix.")
        if self.strategy not in {"wenruo", "ggand0"}:
            raise ValueError(f"Unsupported pose-error strategy: {self.strategy}")
        if self.max_takeover_joint_error_deg <= 0:
            raise ValueError("max_takeover_joint_error_deg must be positive.")
        if self.max_takeover_position_error_m is not None and self.max_takeover_position_error_m <= 0:
            raise ValueError("max_takeover_position_error_m must be positive when configured.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = bool(info.get(TeleopEvents.IS_INTERVENTION, False))
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary_data.pop(DIRECT_JOINT_ACTION_KEY, None)
        transition = transition.copy()
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        raw_leader = complementary_data.get(TELEOP_ACTION_KEY)
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        leader_names = self.leader_action_joint_names or self.follower_motor_names

        if not isinstance(raw_leader, dict):
            self._was_intervening = False
            return transition

        try:
            leader_joints = np.array(
                [
                    float(raw_leader[f"{name}.pos"]) * float((self.joint_scales or {}).get(name, 1.0))
                    + float((self.joint_offsets_deg or {}).get(name, 0.0))
                    for name in leader_names
                ],
                dtype=float,
            )
            follower_joints = np.array(
                [float(observation[f"{name}.pos"]) for name in self.follower_motor_names], dtype=float
            )
        except (KeyError, TypeError, ValueError):
            return self._neutral(transition) if is_intervention else transition

        leader_xyz = np.asarray(self.leader_kinematics.forward_kinematics(leader_joints))[:3, 3]
        follower_xyz = np.asarray(self.follower_kinematics.forward_kinematics(follower_joints))[:3, 3]
        raw_delta = self._rotation @ (leader_xyz - follower_xyz)
        normalized = np.clip(
            raw_delta
            / np.array(
                [
                    self.end_effector_step_sizes["x"],
                    self.end_effector_step_sizes["y"],
                    self.end_effector_step_sizes["z"],
                ]
            ),
            -1.0,
            1.0,
        )
        gripper = float(raw_leader.get(self.gripper_action_key, 0.0))

        if is_intervention and not self._was_intervening:
            self._was_intervening = True
            self._previous_gripper = gripper
            if self.rebase_on_intervention:
                self._leader_reference_xyz = leader_xyz.copy()
                self._follower_reference_xyz = follower_xyz.copy()
            return self._neutral(transition)
        if not is_intervention:
            self._was_intervening = False
            self._previous_gripper = gripper
            self._leader_reference_xyz = None
            self._follower_reference_xyz = None
            return transition

        delta = raw_delta
        if self.rebase_on_intervention:
            if self._leader_reference_xyz is None or self._follower_reference_xyz is None:
                self._leader_reference_xyz = leader_xyz.copy()
                self._follower_reference_xyz = follower_xyz.copy()
                return self._neutral(transition)
            delta = self._rotation @ (
                (leader_xyz - self._leader_reference_xyz) - (follower_xyz - self._follower_reference_xyz)
            )
            normalized = np.clip(
                delta
                / np.array(
                    [
                        self.end_effector_step_sizes["x"],
                        self.end_effector_step_sizes["y"],
                        self.end_effector_step_sizes["z"],
                    ]
                ),
                -1.0,
                1.0,
            )

        position_error_m = float(np.linalg.norm(delta))
        if (
            self.max_takeover_position_error_m is not None
            and position_error_m > self.max_takeover_position_error_m
        ):
            result = self._neutral(transition)
            if self.diagnostics_enabled:
                update_hil_diagnostics(
                    result,
                    "leader",
                    {
                        "strategy": self.strategy,
                        "is_intervention": True,
                        "leader_fk_xyz_m": leader_xyz,
                        "follower_fk_xyz_m": follower_xyz,
                        "raw_leader_follower_delta_xyz_m": raw_delta,
                        "delta_xyz_m": delta,
                        "canonical_action": {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0},
                        "takeover_position_error_m": position_error_m,
                        "takeover_position_error_limit_m": self.max_takeover_position_error_m,
                        "state": "takeover_position_alignment_rejected",
                    },
                )
            return result

        alignment_error = None
        direct = None
        if self.direct_joint_mirror:
            follower_arm_names = [name for name in self.follower_motor_names if name != "gripper"]
            leader_arm_indices = [
                index for index, name in enumerate(leader_names) if f"{name}.pos" != self.gripper_action_key
            ]
            if len(leader_arm_indices) != len(follower_arm_names):
                raise ValueError("ggand0 joint mirroring requires matching leader/follower arm joint counts.")
            mapped_leader_arm = leader_joints[leader_arm_indices]
            follower_arm = np.array(
                [float(observation[f"{name}.pos"]) for name in follower_arm_names], dtype=float
            )
            alignment_error = float(np.max(np.abs(mapped_leader_arm - follower_arm), initial=0.0))
            if alignment_error > self.max_takeover_joint_error_deg:
                result = self._neutral(transition)
                if self.diagnostics_enabled:
                    update_hil_diagnostics(
                        result,
                        "leader",
                        {
                            "strategy": self.strategy,
                            "is_intervention": True,
                            "leader_fk_xyz_m": leader_xyz,
                            "follower_fk_xyz_m": follower_xyz,
                            "delta_xyz_m": delta,
                            "canonical_action": {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0},
                            "direct_joint_execution": False,
                            "takeover_alignment_error_deg": alignment_error,
                            "state": "takeover_alignment_rejected",
                        },
                    )
                return result
            direct = {
                f"{follower}.pos": float(value)
                for follower, value in zip(follower_arm_names, mapped_leader_arm, strict=True)
            }
            if "gripper" in self.follower_motor_names:
                direct["gripper.pos"] = (
                    gripper if self.use_gripper else float(observation["gripper.pos"])
                )

        canonical = {
            "delta_x": float(normalized[0]),
            "delta_y": float(normalized[1]),
            "delta_z": float(normalized[2]),
        }
        if self.use_gripper:
            previous = self._previous_gripper if self._previous_gripper is not None else gripper
            gripper_delta = gripper - previous
            opening = (
                gripper_delta > self.gripper_deadband
                if self.gripper_open_on_positive_delta
                else gripper_delta < -self.gripper_deadband
            )
            canonical[GRIPPER_KEY] = 2.0 if opening else 0.0 if abs(gripper_delta) > self.gripper_deadband else 1.0
        self._previous_gripper = gripper
        complementary_data[TELEOP_ACTION_KEY] = canonical
        if direct is not None:
            complementary_data[DIRECT_JOINT_ACTION_KEY] = direct

        result = transition.copy()
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        if self.diagnostics_enabled:
            update_hil_diagnostics(
                result,
                "leader",
                {
                    "strategy": self.strategy,
                    "is_intervention": True,
                    "leader_fk_xyz_m": leader_xyz,
                    "follower_fk_xyz_m": follower_xyz,
                    "raw_leader_follower_delta_xyz_m": raw_delta,
                    "delta_xyz_m": delta,
                    "takeover_position_error_m": position_error_m,
                    "canonical_action": canonical,
                    "direct_joint_execution": DIRECT_JOINT_ACTION_KEY in complementary_data,
                    "takeover_alignment_error_deg": alignment_error,
                },
            )
        return result

    def _neutral(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        complementary_data = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}
        if self.use_gripper:
            action[GRIPPER_KEY] = 1.0
        complementary_data[TELEOP_ACTION_KEY] = action
        complementary_data.pop(DIRECT_JOINT_ACTION_KEY, None)
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return result

    def reset(self) -> None:
        self._was_intervening = False
        self._previous_gripper = None
        self._leader_reference_xyz = None
        self._follower_reference_xyz = None

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register("canonical_hil_action_processor")
class CanonicalHILActionProcessorStep(ProcessorStep):
    """Validate and normalize the shared XYZ plus discrete-gripper action."""

    use_gripper: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError("Canonical HIL action must be a torch tensor.")
        expected = 4 if self.use_gripper else 3
        if action.shape != (expected,) or not torch.isfinite(action).all():
            raise ValueError(f"Canonical HIL action must have shape ({expected},) with finite values.")
        result = transition.copy()
        canonical = action.to(dtype=torch.float32).clone()
        canonical[..., :3] = canonical[..., :3].clamp(-1.0, 1.0)
        if self.use_gripper:
            canonical[..., 3] = canonical[..., 3].round().clamp(0.0, 2.0)
        result[TransitionKey.ACTION] = canonical
        complementary_data = dict(result.get(TransitionKey.COMPLEMENTARY_DATA, {}))
        complementary_data[TELEOP_ACTION_KEY] = canonical
        result[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return result

    def transform_features(self, features):
        return features


@dataclass
@ProcessorStepRegistry.register("direct_joint_action_override_processor")
class DirectJointActionOverrideProcessorStep(ProcessorStep):
    """Replace the physical command while preserving the canonical stored action."""

    motor_names: list[str]

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        direct = transition.get(TransitionKey.COMPLEMENTARY_DATA, {}).get(DIRECT_JOINT_ACTION_KEY)
        if not isinstance(direct, dict):
            return transition
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError("Direct joint override expects a tensor robot command.")
        result = transition.copy()
        values = []
        for index, name in enumerate(self.motor_names):
            key = f"{name}.pos"
            if key in direct:
                values.append(direct[key])
            elif index < action.numel():
                values.append(action[index])
            else:
                raise ValueError(f"Direct joint override is missing a target for '{key}'.")
        result[TransitionKey.ACTION] = torch.as_tensor(values, dtype=action.dtype, device=action.device)
        return result

    def transform_features(self, features):
        return features


@ProcessorStepRegistry.register("image_crop_resize_processor")
@dataclass
class ImageCropResizeProcessorStep(ObservationProcessorStep):
    """
    Crops and/or resizes image observations.

    This step iterates through all image keys in an observation dictionary and applies
    the specified transformations. It handles device placement, moving tensors to the
    CPU if necessary for operations not supported on certain accelerators like MPS.

    Attributes:
        crop_params_dict: A dictionary mapping image keys to cropping parameters
                          (top, left, height, width).
        resize_size: A tuple (height, width) to resize all images to.
    """

    crop_params_dict: dict[str, tuple[int, int, int, int]] | None = None
    resize_size: tuple[int, int] | None = None

    def observation(self, observation: dict) -> dict:
        """
        Applies cropping and resizing to all images in the observation dictionary.

        Args:
            observation: The observation dictionary, potentially containing image tensors.

        Returns:
            A new observation dictionary with transformed images.
        """
        if self.resize_size is None and not self.crop_params_dict:
            return observation

        new_observation = dict(observation)

        # Process all image keys in the observation
        for key in observation:
            if "image" not in key:
                continue

            image = observation[key]
            device = image.device
            # NOTE (maractingi): No mps kernel for crop and resize, so we need to move to cpu
            if device.type == "mps":
                image = image.cpu()
            # Crop if crop params are provided for this key
            if self.crop_params_dict is not None and key in self.crop_params_dict:
                crop_params = self.crop_params_dict[key]
                image = F.crop(image, *crop_params)
            if self.resize_size is not None:
                image = F.resize(image, self.resize_size)
                image = image.clamp(0.0, 1.0)
            new_observation[key] = image.to(device)

        return new_observation

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary with the crop parameters and resize dimensions.
        """
        return {
            "crop_params_dict": self.crop_params_dict,
            "resize_size": self.resize_size,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Updates the image feature shapes in the policy features dictionary if resizing is applied.

        Args:
            features: The policy features dictionary.

        Returns:
            The updated policy features dictionary with new image shapes.
        """
        if self.resize_size is None:
            return features
        for key in features[PipelineFeatureType.OBSERVATION]:
            if "image" in key:
                nb_channel = features[PipelineFeatureType.OBSERVATION][key].shape[0]
                features[PipelineFeatureType.OBSERVATION][key] = PolicyFeature(
                    type=features[PipelineFeatureType.OBSERVATION][key].type,
                    shape=(nb_channel, *self.resize_size),
                )
        return features


@dataclass
@ProcessorStepRegistry.register("time_limit_processor")
class TimeLimitProcessorStep(TruncatedProcessorStep):
    """
    Tracks episode steps and enforces a time limit by truncating the episode.

    Attributes:
        max_episode_steps: The maximum number of steps allowed per episode.
        current_step: The current step count for the active episode.
    """

    max_episode_steps: int
    current_step: int = 0

    def truncated(self, truncated: bool) -> bool:
        """
        Increments the step counter and sets the truncated flag if the time limit is reached.

        Args:
            truncated: The incoming truncated flag.

        Returns:
            True if the episode step limit is reached, otherwise the incoming value.
        """
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            truncated = True
        # TODO (steven): missing an else truncated = False?
        return truncated

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the `max_episode_steps`.
        """
        return {
            "max_episode_steps": self.max_episode_steps,
        }

    def reset(self) -> None:
        """Resets the step counter, typically called at the start of a new episode."""
        self.current_step = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("gym_hil_adapter_processor")
class GymHILAdapterProcessorStep(ProcessorStep):
    """
    Adapts the output of the `gym-hil` environment to the format expected by `lerobot` processors.

    This step normalizes the `transition` object by:
    1. Copying `teleop_action` from `info` to `complementary_data`.
    2. Copying `is_intervention` from `info` (using the string key) to `info` (using the enum key).
    3. Copying `discrete_penalty` from `info` to `complementary_data`.
    """

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        if TELEOP_ACTION_KEY in info:
            complementary_data[TELEOP_ACTION_KEY] = info[TELEOP_ACTION_KEY]

        if DISCRETE_PENALTY_KEY in info:
            complementary_data[DISCRETE_PENALTY_KEY] = info[DISCRETE_PENALTY_KEY]

        if "is_intervention" in info:
            info[TeleopEvents.IS_INTERVENTION] = info["is_intervention"]

        transition[TransitionKey.INFO] = info
        transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("gripper_penalty_processor")
class GripperPenaltyProcessorStep(ProcessorStep):
    """
    Applies a small per-transition cost on the discrete gripper action.

    Fires only when the commanded action would actually transition the gripper
    from one extreme to the other (close-while-open or open-while-closed).
    This discourages gripper oscillation while leaving "stay" and saturating-further
    commands unpenalized.

    Attributes:
        penalty: The negative reward value to apply.
        max_gripper_pos: The maximum position value for the gripper, used for normalization.
        open_threshold: Normalized state below which the gripper is considered "open".
        closed_threshold: Normalized state above which the gripper is considered "closed".
    """

    penalty: float = -0.02
    max_gripper_pos: float = 30.0
    open_threshold: float = 0.1
    closed_threshold: float = 0.9

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Calculates the gripper penalty and adds it to the complementary data.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with the penalty added to complementary data.
        """
        new_transition = transition.copy()
        action = new_transition.get(TransitionKey.ACTION)
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})

        raw_joint_positions = complementary_data.get("raw_joint_positions")
        if raw_joint_positions is None:
            return new_transition

        current_gripper_pos = raw_joint_positions.get(f"{GRIPPER_KEY}.pos", None)
        if current_gripper_pos is None:
            return new_transition

        # During reset, the transition may not carry any action yet.
        if action is None:
            return new_transition

        # Gripper action is expected as the last action dimension.
        gripper_action = action[-1].item()
        gripper_action_normalized = gripper_action / self.max_gripper_pos

        # Normalize gripper state and action
        gripper_state_normalized = current_gripper_pos / self.max_gripper_pos

        # Calculate penalty boolean as in original
        #   - currently open  AND target is closed  -> close transition
        #   - currently closed AND target is open   -> open transition
        is_open = gripper_state_normalized < self.open_threshold
        is_closed = gripper_state_normalized > self.closed_threshold
        cmd_close = gripper_action_normalized > self.closed_threshold
        cmd_open = gripper_action_normalized < self.open_threshold
        gripper_penalty_bool = (is_open and cmd_close) or (is_closed and cmd_open)

        gripper_penalty = self.penalty * int(gripper_penalty_bool)

        # Update complementary data with penalty info
        new_complementary_data = dict(complementary_data)
        new_complementary_data[DISCRETE_PENALTY_KEY] = gripper_penalty
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = new_complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the penalty value, max gripper position,
            and the open/closed thresholds.
        """
        return {
            "penalty": self.penalty,
            "max_gripper_pos": self.max_gripper_pos,
            "open_threshold": self.open_threshold,
            "closed_threshold": self.closed_threshold,
        }

    def reset(self) -> None:
        """Resets the processor's internal state."""
        pass

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("intervention_action_processor")
class InterventionActionProcessorStep(ProcessorStep):
    """
    Handles human intervention, overriding policy actions and managing episode termination.

    When an intervention is detected (via teleoperator events in the `info` dict),
    this step replaces the policy's action with the human's teleoperated action.
    It also processes signals to terminate the episode or flag success.

    Attributes:
        use_gripper: Whether to include the gripper in the teleoperated action.
        terminate_on_success: If True, automatically sets the `done` flag when a
                              `success` event is received.
    """

    use_gripper: bool = False
    terminate_on_success: bool = True
    success_label_hold_steps: int = 1
    _success_steps_remaining: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.success_label_hold_steps < 1:
            raise ValueError("success_label_hold_steps must be at least 1.")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes the transition to handle interventions.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition, potentially with an overridden action, updated
            reward, and termination status.
        """
        action = transition.get(TransitionKey.ACTION)
        if not isinstance(action, PolicyAction):
            raise ValueError(f"Action should be a PolicyAction type got {type(action)}")

        # Get intervention signals from complementary data
        info = transition.get(TransitionKey.INFO, {})
        complementary_data = transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        teleop_action = complementary_data.get(TELEOP_ACTION_KEY, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        terminate_episode = info.get(TeleopEvents.TERMINATE_EPISODE, False)
        success_event = bool(info.get(TeleopEvents.SUCCESS, False))
        if success_event:
            self._success_steps_remaining = self.success_label_hold_steps
        success = self._success_steps_remaining > 0
        if self._success_steps_remaining > 0:
            self._success_steps_remaining -= 1
        rerecord_episode = info.get(TeleopEvents.RERECORD_EPISODE, False)

        new_transition = transition.copy()

        # Override action if intervention is active
        if is_intervention and teleop_action is not None:
            if isinstance(teleop_action, dict):
                # Convert teleop_action dict to tensor format
                action_list = [
                    teleop_action.get("delta_x", 0.0),
                    teleop_action.get("delta_y", 0.0),
                    teleop_action.get("delta_z", 0.0),
                ]
                if self.use_gripper:
                    action_list.append(teleop_action.get(GRIPPER_KEY, 1.0))
            elif isinstance(teleop_action, np.ndarray):
                action_list = teleop_action.tolist()
            else:
                action_list = teleop_action

            teleop_action_tensor = torch.tensor(action_list, dtype=action.dtype, device=action.device)
            new_transition[TransitionKey.ACTION] = teleop_action_tensor

        # Handle episode termination
        new_transition[TransitionKey.DONE] = bool(terminate_episode) or (
            self.terminate_on_success and success
        )
        new_transition[TransitionKey.REWARD] = float(success)

        # Update info with intervention metadata
        info = new_transition.get(TransitionKey.INFO, {})
        info[TeleopEvents.IS_INTERVENTION] = is_intervention
        info[TeleopEvents.RERECORD_EPISODE] = rerecord_episode
        info[TeleopEvents.SUCCESS] = success
        new_transition[TransitionKey.INFO] = info

        # Update complementary data with teleop action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data[TELEOP_ACTION_KEY] = new_transition.get(TransitionKey.ACTION)
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "use_gripper": self.use_gripper,
            "terminate_on_success": self.terminate_on_success,
            "success_label_hold_steps": self.success_label_hold_steps,
        }

    def reset(self) -> None:
        self._success_steps_remaining = 0

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@dataclass
@ProcessorStepRegistry.register("reward_classifier_processor")
class RewardClassifierProcessorStep(ProcessorStep):
    """
    Applies a pretrained reward classifier to image observations to predict success.

    This step uses a model to determine if the current state is successful, updating
    the reward and potentially terminating the episode.

    Attributes:
        pretrained_path: Path to the pretrained reward classifier model.
        device: The device to run the classifier on.
        success_threshold: The probability threshold to consider a prediction as successful.
        success_reward: The reward value to assign on success.
        terminate_on_success: If True, terminates the episode upon successful classification.
        reward_classifier: The loaded classifier model instance.
    """

    pretrained_path: str | None = None
    device: str = "cpu"
    success_threshold: float = 0.5
    success_reward: float = 1.0
    terminate_on_success: bool = True

    reward_classifier: Any = None

    def __post_init__(self):
        """Initializes the reward classifier model after the dataclass is created."""
        if self.pretrained_path is not None:
            from lerobot.rewards.classifier.modeling_classifier import Classifier

            self.reward_classifier = Classifier.from_pretrained(self.pretrained_path)
            self.reward_classifier.to(self.device)
            self.reward_classifier.eval()

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Processes a transition, applying the reward classifier to its image observations.

        Args:
            transition: The incoming environment transition.

        Returns:
            The modified transition with an updated reward and done flag based on the
            classifier's prediction.
        """
        new_transition = transition.copy()
        observation = new_transition.get(TransitionKey.OBSERVATION)
        if observation is None or self.reward_classifier is None:
            return new_transition

        # Extract images from observation
        images = {key: value for key, value in observation.items() if "image" in key}

        if not images:
            return new_transition

        # Run reward classifier
        start_time = time.perf_counter()
        with torch.inference_mode():
            success = self.reward_classifier.predict_reward(images, threshold=self.success_threshold)

        classifier_frequency = 1 / (time.perf_counter() - start_time)

        # Calculate reward and termination
        reward = new_transition.get(TransitionKey.REWARD, 0.0)
        terminated = new_transition.get(TransitionKey.DONE, False)

        if math.isclose(success, 1, abs_tol=1e-2):
            reward = self.success_reward
            if self.terminate_on_success:
                terminated = True

        # Update transition
        new_transition[TransitionKey.REWARD] = reward
        new_transition[TransitionKey.DONE] = terminated

        # Update info with classifier frequency
        info = new_transition.get(TransitionKey.INFO, {})
        info["reward_classifier_frequency"] = classifier_frequency
        new_transition[TransitionKey.INFO] = info

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """
        Returns the configuration of the step for serialization.

        Returns:
            A dictionary containing the step's configuration attributes.
        """
        return {
            "device": self.device,
            "success_threshold": self.success_threshold,
            "success_reward": self.success_reward,
            "terminate_on_success": self.terminate_on_success,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
