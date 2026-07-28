import numpy as np
import pytest

from lerobot.model.hil_kinematics import (
    GGand0MujocoKinematics,
    WenruoSO101Kinematics,
    default_ggand0_model_path,
)


def test_wenruo_fk_is_deterministic_and_ik_is_bounded():
    kinematics = WenruoSO101Kinematics(max_iterations=5)
    joints = np.zeros(6)
    pose = kinematics.forward_kinematics(joints)
    target = pose.copy()
    target[2, 3] += 0.01

    solution = kinematics.inverse_kinematics(joints, target)

    assert pose.shape == (4, 4)
    assert pose[:3, 3] == pytest.approx([0.45, 0.0, 0.405])
    assert solution.shape == joints.shape
    assert np.isfinite(solution).all()
    assert np.linalg.norm(kinematics.forward_kinematics(solution)[:3, 3] - target[:3, 3]) < 0.005


def test_wenruo_ik_exits_with_finite_joints_for_an_unreachable_target():
    kinematics = WenruoSO101Kinematics(max_iterations=2)
    target = np.eye(4)
    target[:3, 3] = [10.0, 10.0, 10.0]

    solution = kinematics.inverse_kinematics(np.zeros(6), target)

    assert solution.shape == (6,)
    assert np.isfinite(solution).all()
    assert np.abs(solution[:5]).max() <= 2.0


def test_packaged_ggand0_model_loads_and_runs_fk_and_ik():
    pytest.importorskip("mujoco")
    model_path = default_ggand0_model_path()
    kinematics = GGand0MujocoKinematics()
    joints = np.zeros(6)
    pose = kinematics.forward_kinematics(joints)
    target = pose.copy()
    target[2, 3] += 0.001
    solution = kinematics.inverse_kinematics(joints, target)

    assert model_path.name == "so101_new_calib.xml"
    assert model_path.is_file()
    assert (model_path.parent / "assets" / "base_so101_v2.stl").is_file()
    assert pose.shape == (4, 4)
    assert np.isfinite(pose).all()
    assert solution.shape == joints.shape
    assert np.isfinite(solution).all()
    assert np.linalg.norm(kinematics.forward_kinematics(solution)[:3, 3] - target[:3, 3]) < 0.001
