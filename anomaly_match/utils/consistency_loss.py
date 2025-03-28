#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import torch
import torch.nn.functional as F

from ..utils.cross_entropy_loss import cross_entropy_loss


def consistency_loss(logits_w, logits_s, name="ce", T=1.0, p_cutoff=0.0, use_hard_labels=True):
    """Calculate consistency loss between weak and strong augmentation predictions.

    This function implements different consistency regularization losses for semi-supervised learning.

    Args:
        logits_w: Logits from weakly augmented images
        logits_s: Logits from strongly augmented images
        name: Type of consistency loss ('ce' for cross entropy or 'L2' for mean squared error)
        T: Temperature scaling parameter for sharpening predictions (only used with soft labels)
        p_cutoff: Confidence threshold to filter out low-confidence pseudo-labels
        use_hard_labels: Whether to use hard (argmax) or soft (distribution) pseudo-labels

    Returns:
        For 'L2': scalar consistency loss value
        For 'ce': tuple of (masked loss, mask ratio)

    Raises:
        AssertionError: If an unsupported loss name is provided
    """
    assert name in ["ce", "L2"], f"Unsupported consistency loss: {name}"

    # Detach weak augmentation logits to prevent gradients from flowing back
    logits_w = logits_w.detach()

    # L2 (mean squared error) loss
    if name == "L2":
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction="mean")

    # Cross-entropy consistency loss with optional confidence thresholding
    elif name == "ce":
        # Get pseudo-labels and confidence scores from weak augmentation prediction
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)

        # Create a mask to filter low-confidence predictions
        # mask=1 means the sample is used, mask=0 means it's ignored
        mask = max_probs.ge(p_cutoff).float()

        if use_hard_labels:
            # Use hard pseudo-labels (class indices)
            masked_loss = (
                cross_entropy_loss(logits_s, max_idx, use_hard_labels=True, reduction="none") * mask
            )
        else:
            # Use soft pseudo-labels (probability distribution)
            # Temperature T controls the sharpness of the target distribution
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = cross_entropy_loss(logits_s, pseudo_label, use_hard_labels=False) * mask

        # Return both the mean loss and the ratio of samples that were kept after masking
        return masked_loss.mean(), mask.mean()
