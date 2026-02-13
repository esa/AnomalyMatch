#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Multispectral-compatible augmentations for arbitrary channel counts.

This module provides augmentation operations that work with images having
any number of channels (not just RGB). It excludes PIL-dependent operations
like AutoContrast, Brightness, Color, and Contrast that assume RGB images.

Based on the approach from DistMSMatch/MSMatch.
"""

import random

import albumentations as A
import numpy as np
from scipy.ndimage import uniform_filter


def Identity(img, v):
    """Return the image unchanged.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Unused parameter

    Returns:
        The original image
    """
    return img


def Posterize(img, v):
    """Reduce the number of bits for each channel.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Number of bits to keep for each channel (range [4, 8])

    Returns:
        numpy array with reduced color depth
    """
    v = int(v)
    v = max(1, min(8, v))
    shift = 8 - v
    return ((img >> shift) << shift).astype(np.uint8)


def Rotate(img, v):
    """Rotate the image by v degrees.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Rotation angle in degrees (range [-30, 30])

    Returns:
        Rotated numpy array
    """
    transform = A.Rotate(limit=(v, v), border_mode=0, p=1.0)
    return transform(image=img)["image"]


def Sharpness(img, v):
    """Adjust the sharpness of the image using unsharp masking.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Sharpness factor (range [0.05, 0.95])

    Returns:
        numpy array with adjusted sharpness
    """
    # Apply per-channel sharpening using unsharp mask
    blurred = uniform_filter(img.astype(np.float32), size=(3, 3, 1))
    sharpened = img.astype(np.float32) + v * (img.astype(np.float32) - blurred)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def ShearX(img, v):
    """Apply horizontal shear to the image.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Shear factor (range [-0.3, 0.3])

    Returns:
        numpy array with horizontal shear applied
    """
    transform = A.Affine(shear={"x": (v * 45, v * 45), "y": (0, 0)}, border_mode=0, p=1.0)
    return transform(image=img)["image"]


def ShearY(img, v):
    """Apply vertical shear to the image.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Shear factor (range [-0.3, 0.3])

    Returns:
        numpy array with vertical shear applied
    """
    transform = A.Affine(shear={"x": (0, 0), "y": (v * 45, v * 45)}, border_mode=0, p=1.0)
    return transform(image=img)["image"]


def TranslateX(img, v):
    """Translate the image horizontally by a percentage of its width.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Translation factor as a percentage of image width (range [-0.3, 0.3])

    Returns:
        numpy array with horizontal translation applied
    """
    transform = A.Affine(translate_percent={"x": (v, v), "y": (0, 0)}, border_mode=0, p=1.0)
    return transform(image=img)["image"]


def TranslateY(img, v):
    """Translate the image vertically by a percentage of its height.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Translation factor as a percentage of image height (range [-0.3, 0.3])

    Returns:
        numpy array with vertical translation applied
    """
    transform = A.Affine(translate_percent={"x": (0, 0), "y": (v, v)}, border_mode=0, p=1.0)
    return transform(image=img)["image"]


def Solarize(img, v):
    """Invert all pixel values above a threshold.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Threshold for solarization (range [0, 256])

    Returns:
        Solarized numpy array
    """
    v = int(v)
    return np.where(img >= v, 255 - img, img).astype(np.uint8)


def Cutout(img, v, num_channels=None):
    """Apply cutout augmentation to the image.

    Creates a square mask at a random location in the image.

    Args:
        img: numpy array (H, W, C) to be processed
        v: Size of the cutout as a percentage of image size (range [0.0, 0.5])
        num_channels: Number of channels (inferred from img if not provided)

    Returns:
        numpy array with cutout applied
    """
    if v <= 0.0:
        return img

    h, w = img.shape[:2]
    if num_channels is None:
        num_channels = img.shape[2] if len(img.shape) > 2 else 1

    cutout_size = int(v * min(h, w))
    if cutout_size <= 0:
        return img

    # Random position for cutout
    x0 = np.random.randint(0, max(1, w - cutout_size + 1))
    y0 = np.random.randint(0, max(1, h - cutout_size + 1))

    # Apply cutout with gray fill (128 for all channels)
    img_out = img.copy()
    img_out[y0 : y0 + cutout_size, x0 : x0 + cutout_size, :] = 128

    return img_out


def multispectral_augment_list():
    """Return a list of available augmentation operations with their value ranges.

    These operations work with arbitrary channel counts, excluding PIL-dependent
    operations (AutoContrast, Brightness, Color, Contrast, Equalize) that assume RGB.

    Returns:
        List of tuples (operation, min_value, max_value) for each augmentation
    """
    return [
        (Identity, 0, 1),
        (Posterize, 4, 8),
        (Rotate, -30, 30),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.3, 0.3),
        (ShearY, -0.3, 0.3),
        (Solarize, 0, 256),
        (TranslateX, -0.3, 0.3),
        (TranslateY, -0.3, 0.3),
    ]


class MultispectralRandAugment:
    """Random augmentation pipeline for multispectral images.

    Randomly applies a series of channel-agnostic image transformations
    that work with arbitrary channel counts.

    Attributes:
        n: Number of augmentation operations to apply
        m: Magnitude parameter (unused, kept for API compatibility)
        num_channels: Number of image channels
        augment_list: List of available augmentation operations
    """

    def __init__(self, n, m, num_channels=4):
        """Initialize the MultispectralRandAugment pipeline.

        Args:
            n: Number of augmentation operations to apply
            m: Magnitude parameter (unused, kept for API compatibility)
            num_channels: Number of channels in the images
        """
        self.n = n
        self.m = m
        self.num_channels = num_channels
        self.augment_list = multispectral_augment_list()

    def __call__(self, img):
        """Apply random augmentations to the input image.

        Args:
            img: numpy array (H, W, C) to be augmented. Will be converted to
                uint8 if necessary, as operations like Posterize and Solarize
                use bit-shift arithmetic that assumes 0-255 integer values.

        Returns:
            Augmented numpy array (uint8)
        """
        # Ensure input is numpy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        # Convert to uint8 — operations like Posterize and Solarize use
        # bit-shift arithmetic that requires integer 0-255 values.
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Apply n random augmentations
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)

        # Apply cutout (for FixMatch)
        cutout_val = random.random() * 0.5
        img = Cutout(img, cutout_val, num_channels=self.num_channels)

        return img


if __name__ == "__main__":
    # Test with 4-channel image
    randaug = MultispectralRandAugment(3, 5, num_channels=4)
    test_img = np.random.randint(0, 255, (64, 64, 4), dtype=np.uint8)
    print(f"Input shape: {test_img.shape}")

    for op, min_val, max_val in randaug.augment_list:
        val = min_val + float(max_val - min_val) * random.random()
        result = op(test_img.copy(), val)
        print(f"{op.__name__}: input {test_img.shape} -> output {result.shape}")

    # Test full pipeline
    result = randaug(test_img.copy())
    print(f"Full pipeline: input {test_img.shape} -> output {result.shape}")
