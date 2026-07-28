import json

import numpy as np
import torch

from lerobot.processor.hil_diagnostics import (
    HIL_DIAGNOSTICS_KEY,
    HILDiagnosticsLogger,
    update_hil_diagnostics,
)
from lerobot.processor.hil_processor import LeaderJointDeltaActionProcessorStep
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsRLStep
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey


class _FakeKinematics:
    def forward_kinematics(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints[:3], dtype=float)
        return pose

    def inverse_kinematics(self, current_joints, desired_pose):
        result = np.asarray(current_joints, dtype=float).copy()
        result[:3] = desired_pose[:3, 3]
        return result


def test_hil_diagnostics_logger_writes_timestamped_jsonl_and_flushes(tmp_path):
    diagnostics = HILDiagnosticsLogger(tmp_path, console_summary=False)
    diagnostics.record(
        "control_step",
        {
            "array": np.array([1.0, 2.0]),
            "tensor": torch.tensor([3.0]),
        },
    )

    lines_before_close = diagnostics.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines_before_close] == [
        "logger_started",
        "control_step",
    ]
    control_step = json.loads(lines_before_close[1])
    assert control_step["timestamp"]
    assert control_step["array"] == [1.0, 2.0]
    assert control_step["tensor"] == [3.0]

    diagnostics.close()
    lines_after_close = diagnostics.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines_after_close[-1])["event"] == "logger_stopped"


def test_update_hil_diagnostics_preserves_other_complementary_data():
    transition = {
        TransitionKey.COMPLEMENTARY_DATA: {"teleop_action": {"delta_x": 0.0}},
    }

    update_hil_diagnostics(transition, "leader", {"delta_xyz_m": [0.1, 0.0, 0.0]})
    update_hil_diagnostics(transition, "inverse_kinematics", {"position_residual_m": 0.001})

    complementary_data = transition[TransitionKey.COMPLEMENTARY_DATA]
    assert complementary_data["teleop_action"] == {"delta_x": 0.0}
    assert complementary_data[HIL_DIAGNOSTICS_KEY] == {
        "leader": {"delta_xyz_m": [0.1, 0.0, 0.0]},
        "inverse_kinematics": {"position_residual_m": 0.001},
    }


def test_leader_diagnostics_capture_raw_fk_and_clipped_delta():
    processor = LeaderJointDeltaActionProcessorStep(
        kinematics=_FakeKinematics(),
        motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        end_effector_step_sizes={"x": 0.1, "y": 0.1, "z": 0.1},
        diagnostics_enabled=True,
    )

    def transition(joint_0):
        return {
            TransitionKey.ACTION: torch.zeros(4),
            TransitionKey.INFO: {TeleopEvents.IS_INTERVENTION: True},
            TransitionKey.COMPLEMENTARY_DATA: {
                "teleop_action": {
                    "joint_0.pos": joint_0,
                    "joint_1.pos": 0.0,
                    "joint_2.pos": 0.0,
                    "gripper.pos": 0.0,
                }
            },
        }

    processor(transition(0.0))
    result = processor(transition(0.2))
    leader = result[TransitionKey.COMPLEMENTARY_DATA][HIL_DIAGNOSTICS_KEY]["leader"]

    assert leader["raw_joint_deg"]["joint_0"] == 0.2
    assert leader["delta_xyz_m"].tolist() == [0.2, 0.0, 0.0]
    assert leader["normalized_delta_unclipped"].tolist() == [2.0, 0.0, 0.0]
    assert leader["normalized_delta_clipped"].tolist() == [1.0, 0.0, 0.0]
    assert leader["clipped_axes"] == ["x"]


def test_inverse_kinematics_diagnostics_capture_seed_solution_and_residual():
    processor = InverseKinematicsRLStep(
        kinematics=_FakeKinematics(),
        motor_names=["joint_0", "joint_1", "joint_2", "gripper"],
        diagnostics_enabled=True,
    )
    transition = {
        TransitionKey.ACTION: {
            "ee.x": 1.0,
            "ee.y": 2.0,
            "ee.z": 3.0,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": 0.0,
            "ee.gripper_pos": 25.0,
        },
        TransitionKey.OBSERVATION: {
            "joint_0.pos": 0.0,
            "joint_1.pos": 0.0,
            "joint_2.pos": 0.0,
            "gripper.pos": 25.0,
        },
        TransitionKey.COMPLEMENTARY_DATA: {},
    }

    result = processor(transition)
    ik = result[TransitionKey.COMPLEMENTARY_DATA][HIL_DIAGNOSTICS_KEY]["inverse_kinematics"]

    assert ik["observed_joint_order"] == ["joint_0", "joint_1", "joint_2", "gripper"]
    assert ik["initial_guess_joint_deg"] == {
        "joint_0": 0.0,
        "joint_1": 0.0,
        "joint_2": 0.0,
        "gripper": 25.0,
    }
    assert ik["solution_joint_deg"] == {
        "joint_0": 1.0,
        "joint_1": 2.0,
        "joint_2": 3.0,
        "gripper": 25.0,
    }
    assert ik["position_residual_m"] == 0.0
