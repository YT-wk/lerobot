import pytest
import torch

from lerobot.rl.hil_action import validate_canonical_hil_action


@pytest.mark.parametrize("strategy", ["current", "wenruo", "ggand0"])
def test_all_hil_strategies_accept_the_same_canonical_contract(strategy):
    del strategy
    action = torch.tensor([-1.0, 0.25, 1.0, 2.0], dtype=torch.float32)

    assert validate_canonical_hil_action(action, use_gripper=True) is action


@pytest.mark.parametrize(
    "action,error",
    [
        (torch.zeros(6, dtype=torch.float32), "shape"),
        (torch.zeros(4, dtype=torch.float64), "float32"),
        (torch.tensor([float("nan"), 0.0, 0.0, 1.0]), "finite"),
        (torch.tensor([1.1, 0.0, 0.0, 1.0]), "XYZ"),
        (torch.tensor([0.0, 0.0, 0.0, 1.5]), "gripper"),
    ],
)
def test_malformed_canonical_hil_actions_fail_with_a_clear_error(action, error):
    with pytest.raises(ValueError, match=error):
        validate_canonical_hil_action(action, use_gripper=True)
