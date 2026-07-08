#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import torch.nn as nn

from anomaly_match.utils.get_net_builder import get_net_builder


def test_efficientnet_classifier_head_init_is_pytorch_scale():
    """The 2-class classifier head must use PyTorch's default init, not timm's.

    timm scales the fresh EfficientNet head for the 1000-class ImageNet head, which is
    far too large for AnomalyMatch's 2-class head and poisons FixMatch fine-tuning.
    After the head reset the weight std must sit near PyTorch's Linear bound.
    """
    model = get_net_builder("efficientnet-lite0", pretrained=False)(num_classes=2, in_channels=3)
    classifier = model.get_classifier()
    assert isinstance(classifier, nn.Linear)
    weight = classifier.weight
    pytorch_bound = 1.0 / (weight.shape[1] ** 0.5)
    assert weight.std().item() < 2 * pytorch_bound
