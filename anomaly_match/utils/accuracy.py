#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output: Model output logits or probabilities with shape [batch_size, num_classes]
        target: Ground truth labels with shape [batch_size] or [batch_size, 1]
        topk: Tuple specifying which top-k accuracies to compute (e.g., (1,) or (1, 5))

    Returns:
        List of tensors, each containing the top-k accuracy value (in percentage)
        for the specified k values

    Note:
        This implementation is based on the PyTorch ImageNet example
    """
    with torch.no_grad():
        maxk = max(topk)  # Get the largest k value
        batch_size = target.size(0)  # Get batch size

        # Get the top-k predictions (values and indices)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)  # shape: [batch_size, k]
        pred = pred.t()  # Transpose to shape [k, batch_size]

        # Expand target to match dimension for comparison
        # The result is a boolean tensor of shape [k, batch_size]
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # Calculate accuracy for each k value
        res = []
        for k in topk:
            # Count correct predictions for top-k
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)

            # Convert to percentage
            res.append(correct_k.mul_(100.0 / batch_size))

        return res
