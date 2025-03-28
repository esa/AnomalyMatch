#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch


def set_seeds(seed: int, deterministic: bool = False) -> None:
    """Sets the seeds for the random number generators in torch, numpy and random.

    Args:
        seed (int): seed for the random number generators
        deterministic (bool): if True, sets the cudnn to deterministic mode. Default: False
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True
