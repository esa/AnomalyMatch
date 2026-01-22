#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
from torchvision import transforms

from anomaly_match.datasets.augmentation.randaugment import RandAugment


def get_weak_transforms():
    """Get weak augmentation transforms.

    Args:
        train (bool, optional): Whether training, in test only normalization is applied.

    Returns:
        torchvision.transforms.Compose: transforms.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0, 0.125)),
        ]
    )


def get_prediction_transforms():
    """Get the standard image transform.

    Args:
        None

    Returns:
        torchvision.transforms.Compose: transforms.
        with an empty transform
    """
    return transforms.Compose([transforms.ToTensor()])


def get_strong_transforms():
    """Get strong augmentations for FixMatch.

    Includes RandAugment followed by the same transforms as weak (ToTensor,
    RandomHorizontalFlip, RandomAffine).

    Returns:
        torchvision.transforms.Compose: Strong augmentation pipeline.
    """
    return transforms.Compose(
        [
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, translate=(0, 0.125)),
        ]
    )
