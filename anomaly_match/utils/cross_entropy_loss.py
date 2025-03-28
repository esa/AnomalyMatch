#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch.nn.functional as F
import torch


def cross_entropy_loss(logits, targets, use_hard_labels=True, reduction="none"):
    """Cross entropy loss that supports both hard and soft labels.

    Args:
        logits: Model output logits with shape [batch_size, num_classes]
        targets: Target values, which can be either:
            - Class indices with shape [batch_size] if use_hard_labels=True
            - Class probabilities with shape [batch_size, num_classes] if use_hard_labels=False
        use_hard_labels: If True, targets are treated as class indices
                        If False, targets are treated as probability distributions
        reduction: Specifies the reduction to apply to the output:
                  'none': no reduction will be applied
                  'mean': the sum of the output will be divided by the number of elements
                  'sum': the output will be summed

    Returns:
        Loss tensor with shape determined by the reduction mode
    """
    if use_hard_labels:
        # Standard cross-entropy with class indices as targets
        return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        # KL divergence style loss with probability distributions as targets
        assert (
            logits.shape == targets.shape
        ), "Logits and targets must have the same shape when using soft labels"
        log_pred = F.log_softmax(logits, dim=-1)
        # Negative KL divergence (equivalent to cross-entropy for soft targets)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss
