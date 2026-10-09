#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Cosine learning rate schedule with linear warmup."""

from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_cycles: float = 7.0 / 16.0,
    num_warmup_steps: int = 0,
    last_epoch: int = -1,
) -> LambdaLR:
    """Get learning rate schedule with linear warmup and cosine decay.

    Args:
        optimizer: optimizer.
        num_training_steps: total number of training steps.
        num_cycles: number of cycles in the cosine decay.
        num_warmup_steps: number of warmup steps.
        last_epoch: last epoch number.

    Returns:
        learning rate scheduler.
    """

    def _lr_lambda(current_step):
        """Return a multiplicative factor given an integer parameter epochs.

        Decaying criteria: last_epoch.
        """
        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr

    return LambdaLR(optimizer, _lr_lambda, last_epoch)
