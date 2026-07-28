# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Kinematics backends retained for SO101 HIL-SERL strategy comparisons."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import numpy as np


def _skew(vector: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]]
    )


def _screw_transform(axis: np.ndarray, angle: float) -> np.ndarray:
    rotation_axis = axis[:3]
    translation_axis = axis[3:]
    axis_hat = _skew(rotation_axis)
    rotation = np.eye(3) + np.sin(angle) * axis_hat + (1 - np.cos(angle)) * axis_hat @ axis_hat
    translation = (
        np.eye(3) * angle
        + (1 - np.cos(angle)) * axis_hat
        + (angle - np.sin(angle)) * axis_hat @ axis_hat
    ) @ translation_axis
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _translation(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = [x, y, z]
    return transform


class WenruoSO101Kinematics:
    """Position-priority SO101 model ported from 25wenruo_lerobot."""

    MEASUREMENTS = {
        "so_new_calibration": {
            "gripper": [0.33, 0.0, 0.285],
            "wrist": [0.30, 0.0, 0.267],
            "forearm": [0.25, 0.0, 0.266],
            "humerus": [0.06, 0.0, 0.264],
            "shoulder": [0.0, 0.0, 0.238],
            "base": [0.0, 0.0, 0.12],
        }
    }

    def __init__(
        self,
        robot_model: str = "so_new_calibration",
        frame: str = "gripper_tip",
        max_iterations: int = 5,
        learning_rate: float = 1.0,
        position_tolerance_m: float = 0.005,
        max_joint_update_deg: float = 1.0,
    ) -> None:
        if robot_model not in self.MEASUREMENTS:
            raise ValueError(f"Unsupported Wenruo robot model: {robot_model}")
        if frame not in {"gripper", "gripper_tip"}:
            raise ValueError("Wenruo frame must be 'gripper' or 'gripper_tip'.")
        if min(max_iterations, learning_rate, position_tolerance_m, max_joint_update_deg) <= 0:
            raise ValueError("Wenruo IK iteration, learning rate, tolerance, and joint update must be positive.")
        self.measurements = self.MEASUREMENTS[robot_model]
        self.frame = frame
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.position_tolerance_m = position_tolerance_m
        self.max_joint_update_deg = max_joint_update_deg
        self._setup_transforms()

    def _setup_transforms(self) -> None:
        measurements = self.measurements
        self.gripper_x0 = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        self.s_bg = np.array([1, 0, 0, 0, measurements["gripper"][2], 0], dtype=float)
        self.s_br = np.array(
            [0, 1, 0, -measurements["wrist"][2], 0, measurements["wrist"][0]], dtype=float
        )
        self.s_bf = np.array(
            [0, 1, 0, -measurements["forearm"][2], 0, measurements["forearm"][0]], dtype=float
        )
        self.s_bh = np.array(
            [0, -1, 0, measurements["humerus"][2], 0, -measurements["humerus"][0]], dtype=float
        )
        self.s_bs = np.array([0, 0, -1, 0, 0, 0], dtype=float)
        self.x_wb = _translation(*measurements["base"])
        self.x_bg = _translation(*measurements["gripper"])
        self.x_tip = _translation(x=0.12)
        self.x_center = _translation(x=0.07)

    def forward_kinematics(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        joints = np.asarray(joint_positions_deg, dtype=float)
        if joints.size < 5:
            raise ValueError("Wenruo FK requires five arm joint angles.")
        q = np.deg2rad(joints[:5])
        transform = self.x_wb @ _screw_transform(self.s_bs, q[0])
        transform = transform @ _screw_transform(self.s_bh, -q[1])
        transform = transform @ _screw_transform(self.s_bf, q[2])
        transform = transform @ _screw_transform(self.s_br, q[3])
        transform = transform @ _screw_transform(self.s_bg, q[4])
        endpoint = self.x_tip if self.frame == "gripper_tip" else self.x_center
        return transform @ endpoint @ self.x_bg @ self.gripper_x0

    def _positional_jacobian(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        joints = np.asarray(joint_positions_deg, dtype=float)
        epsilon_deg = 1e-8
        jacobian = np.zeros((3, 5), dtype=float)
        for index in range(5):
            delta = np.zeros_like(joints)
            delta[index] = epsilon_deg / 2
            jacobian[:, index] = (
                self.forward_kinematics(joints + delta)[:3, 3]
                - self.forward_kinematics(joints - delta)[:3, 3]
            ) / epsilon_deg
        return jacobian

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        del position_weight, orientation_weight
        result = np.asarray(current_joint_pos, dtype=float).copy()
        for _ in range(self.max_iterations):
            error = desired_ee_pose[:3, 3] - self.forward_kinematics(result)[:3, 3]
            if np.linalg.norm(error) < self.position_tolerance_m:
                break
            update = self.learning_rate * (np.linalg.pinv(self._positional_jacobian(result)) @ error)
            result[:5] += np.clip(update, -self.max_joint_update_deg, self.max_joint_update_deg)
        return result


def default_ggand0_model_path() -> Path:
    return Path(str(files("lerobot.model.assets.ggand0_so101").joinpath("so101_new_calib.xml")))


class GGand0MujocoKinematics:
    """MuJoCo FK and damped-least-squares IK ported from ggand0/lerobot."""

    def __init__(
        self,
        model_path: str | None = None,
        end_effector_site: str = "gripperframe",
        ik_damping: float = 0.1,
        ik_max_dq_rad: float = 0.5,
        locked_joints: list[int] | None = None,
        locked_joint_positions_deg: dict[str, float] | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("The ggand0 HIL strategy requires `mujoco`; install lerobot[hilserl].") from exc
        self._mujoco = mujoco
        path = Path(model_path) if model_path else default_ggand0_model_path()
        if not path.is_file():
            raise FileNotFoundError(f"ggand0 MuJoCo model not found: {path}")
        if ik_damping <= 0 or ik_max_dq_rad <= 0:
            raise ValueError("ggand0 IK damping and max dq must be positive.")
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site)
        if self.site_id < 0:
            raise ValueError(f"MuJoCo site '{end_effector_site}' was not found in {path}.")
        self.ik_damping = ik_damping
        self.ik_max_dq_rad = ik_max_dq_rad
        self.locked_joints = set(locked_joints or [])
        if any(index < 0 or index >= 5 for index in self.locked_joints):
            raise ValueError("ggand0 locked_joints indices must be in the five-arm-joint range [0, 4].")
        self.locked_joint_positions_deg = locked_joint_positions_deg or {}
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))

    def _sync(self, joint_positions_deg: np.ndarray) -> None:
        joints = np.deg2rad(np.asarray(joint_positions_deg, dtype=float)[:5])
        self.data.qpos[:5] = joints
        self._mujoco.mj_forward(self.model, self.data)

    def forward_kinematics(self, joint_positions_deg: np.ndarray) -> np.ndarray:
        self._sync(joint_positions_deg)
        transform = np.eye(4)
        transform[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        transform[:3, 3] = self.data.site_xpos[self.site_id]
        return transform

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        del position_weight, orientation_weight
        current = np.asarray(current_joint_pos, dtype=float)
        self._sync(current)
        error = desired_ee_pose[:3, 3] - self.data.site_xpos[self.site_id]
        self._mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site_id)
        active = [index for index in range(5) if index not in self.locked_joints]
        jacobian = self.jacp[:, active]
        lhs = jacobian.T @ jacobian + self.ik_damping**2 * np.eye(len(active))
        try:
            delta = np.linalg.solve(lhs, jacobian.T @ error)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(jacobian) @ error
        delta = np.clip(delta, -self.ik_max_dq_rad, self.ik_max_dq_rad)
        result = current.copy()
        result[active] += np.rad2deg(delta)
        for index in self.locked_joints:
            configured = self.locked_joint_positions_deg.get(str(index))
            if configured is not None:
                result[index] = configured
        joint_limits_deg = np.rad2deg(self.model.jnt_range[:5])
        result[:5] = np.clip(result[:5], joint_limits_deg[:, 0], joint_limits_deg[:, 1])
        return result
