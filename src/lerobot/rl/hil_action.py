# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch


def validate_canonical_hil_action(action: torch.Tensor, *, use_gripper: bool) -> torch.Tensor:
    """Validate the Cartesian action stored and transported by HIL-SERL."""
    expected = 4 if use_gripper else 3
    if not isinstance(action, torch.Tensor):
        raise ValueError(f"Canonical HIL action must be a torch tensor, got {type(action)}.")
    if action.dtype != torch.float32:
        raise ValueError(f"Canonical HIL action must use float32, got {action.dtype}.")
    if action.shape != (expected,):
        raise ValueError(f"Canonical HIL action must have shape ({expected},), got {tuple(action.shape)}.")
    if not torch.isfinite(action).all():
        raise ValueError("Canonical HIL action must contain only finite values.")
    if torch.any(action[:3] < -1.0) or torch.any(action[:3] > 1.0):
        raise ValueError("Canonical HIL XYZ values must be in [-1, 1].")
    if use_gripper:
        gripper = action[3]
        if gripper < 0.0 or gripper > 2.0 or not torch.equal(gripper, gripper.round()):
            raise ValueError("Canonical HIL gripper must be one of 0, 1, or 2.")
    return action
