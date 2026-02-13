#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import numpy as np
import torch
from torchvision import transforms

from anomaly_match.datasets.augmentation.randaugment import RandAugment
from anomaly_match.datasets.augmentation.randaugment_multispectral import (
    MultispectralRandAugment,
)


class NumpyToTensor:
    """Convert numpy array to torch tensor for N-channel images.

    Handles HWC to CHW conversion for arbitrary channel counts.
    """

    def __call__(self, img):
        """Convert numpy array (H, W, C) to tensor (C, H, W)."""
        if isinstance(img, np.ndarray):
            # HWC to CHW conversion
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img.copy()).float() / 255.0
        return img


class NumpyRandomHorizontalFlip:
    """Random horizontal flip for numpy arrays."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            if np.random.random() < self.p:
                return np.ascontiguousarray(img[:, ::-1, :])
            return img
        elif isinstance(img, torch.Tensor):
            if np.random.random() < self.p:
                return torch.flip(img, dims=[2])  # Flip width dimension (CHW format)
            return img
        return img


class NumpyRandomTranslate:
    """Random translation for numpy arrays."""

    def __init__(self, translate=(0, 0.125)):
        self.translate = translate

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]
            max_dx = self.translate[1] * w
            max_dy = self.translate[0] * h
            dx = np.random.uniform(-max_dx, max_dx)
            dy = np.random.uniform(-max_dy, max_dy)

            # Use numpy roll for translation
            img = np.roll(img, int(dx), axis=1)
            img = np.roll(img, int(dy), axis=0)
            return img
        elif isinstance(img, torch.Tensor):
            # For tensor (CHW format)
            _, h, w = img.shape
            max_dx = int(self.translate[1] * w)
            max_dy = int(self.translate[0] * h)
            dx = np.random.randint(-max_dx, max_dx + 1) if max_dx > 0 else 0
            dy = np.random.randint(-max_dy, max_dy + 1) if max_dy > 0 else 0
            return torch.roll(torch.roll(img, dx, dims=2), dy, dims=1)
        return img


def get_weak_transforms(num_channels=3):
    """Get weak augmentation transforms.

    Args:
        num_channels: Number of image channels (3 for RGB, other for multispectral)

    Returns:
        torchvision.transforms.Compose: transforms.
    """
    if num_channels == 3:
        # Use PIL-based transforms for RGB
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0, 0.125)),
            ]
        )
    else:
        # Use numpy-based transforms for multispectral
        return transforms.Compose(
            [
                NumpyRandomHorizontalFlip(),
                NumpyRandomTranslate(translate=(0, 0.125)),
                NumpyToTensor(),
            ]
        )


def get_prediction_transforms(num_channels=3):
    """Get the standard image transform.

    Args:
        num_channels: Number of image channels (3 for RGB, other for multispectral)

    Returns:
        torchvision.transforms.Compose: transforms.
    """
    if num_channels == 3:
        return transforms.Compose([transforms.ToTensor()])
    else:
        return transforms.Compose([NumpyToTensor()])


def get_strong_transforms(num_channels=3):
    """Get strong augmentations for FixMatch.

    Includes RandAugment followed by the same transforms as weak (ToTensor,
    RandomHorizontalFlip, RandomAffine).

    Args:
        num_channels: Number of image channels (3 for RGB, other for multispectral)

    Returns:
        torchvision.transforms.Compose: Strong augmentation pipeline.
    """
    if num_channels == 3:
        # Use PIL-based transforms for RGB
        return transforms.Compose(
            [
                RandAugment(3, 5),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(0, translate=(0, 0.125)),
            ]
        )
    else:
        # Use numpy-based transforms for multispectral
        return transforms.Compose(
            [
                MultispectralRandAugment(3, 5, num_channels=num_channels),
                NumpyRandomHorizontalFlip(),
                NumpyRandomTranslate(translate=(0, 0.125)),
                NumpyToTensor(),
            ]
        )
