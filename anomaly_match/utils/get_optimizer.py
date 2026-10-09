#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Optimizer factory for model training."""

from __future__ import annotations

import torch


def get_optimizer(
    net: torch.nn.Module,
    name: str = "SGD",
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    nesterov: bool = True,
    bn_wd_skip: bool = True,
) -> torch.optim.Optimizer:
    """Creates a optimizer for the given network.

    Args:
        net: network to optimize.
        name: optimizer name.
        lr: learning rate.
        momentum: momentum.
        weight_decay: weight decay.
        nesterov: if True, use Nesterov momentum.
        bn_wd_skip: If bn_wd_skip, the optimizer does not apply weight decay
            regularization on parameters in batch normalization.

    Returns:
        torch.optim.Optimizer: optimizer.

    Raises:
        ValueError: If learning rate is too high for the Adam optimizer.
    """
    decay = []
    no_decay = []
    if name == "SGD":
        for param_name, param in net.named_parameters():
            if ("bn" in param_name) and bn_wd_skip:
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = torch.optim.SGD(
            per_param_args,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif name == "Adam" or name == "ADAM":
        if lr > 0.005:
            raise ValueError("Learning rate is " + str(lr) + ". That is too high for ADAM.")

        for param_name, param in net.named_parameters():
            if ("bn" in param_name) and bn_wd_skip:
                no_decay.append(param)
            else:
                decay.append(param)

        per_param_args = [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]

        optimizer = torch.optim.Adam(
            per_param_args,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    return optimizer
