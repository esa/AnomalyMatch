#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import torch
import torch.optim as optim
from anomaly_match.utils.get_cosine_schedule_with_warmup import get_cosine_schedule_with_warmup
from anomaly_match.utils.get_optimizer import get_optimizer
from anomaly_match.utils.get_net_builder import get_net_builder
from anomaly_match.utils.set_seeds import set_seeds


def test_cosine_schedule_with_warmup():
    model = torch.nn.Linear(10, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_warmup_steps = 5
    num_training_steps = 20

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Test warmup phase
    for _ in range(num_warmup_steps):
        scheduler.step()

    # Test cosine decay phase
    for _ in range(num_training_steps - num_warmup_steps):
        scheduler.step()

    assert scheduler.get_last_lr()[0] >= 0


def test_get_optimizer():
    model = torch.nn.Linear(10, 2)
    from anomaly_match.utils.get_default_cfg import get_default_cfg

    cfg = get_default_cfg()

    # Test SGD
    cfg.opt = "SGD"
    cfg.lr = 0.01
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0005
    optimizer_sgd = get_optimizer(
        model, name=cfg.opt, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    assert isinstance(optimizer_sgd, torch.optim.SGD)
    cfg_adam = get_default_cfg()
    cfg_adam.opt = "Adam"
    cfg_adam.lr = 0.0001
    cfg_adam.weight_decay = 0.0005
    optimizer_adam = get_optimizer(
        model, name=cfg_adam.opt, lr=cfg_adam.lr, weight_decay=cfg_adam.weight_decay
    )
    assert isinstance(optimizer_adam, torch.optim.Adam)


def test_get_net_builder():
    # Test valid network
    net_builder = get_net_builder("efficientnet-lite0")
    assert callable(net_builder)


def test_set_seeds():
    # Test that setting seeds doesn't raise errors
    set_seeds(42)
    set_seeds(0)
