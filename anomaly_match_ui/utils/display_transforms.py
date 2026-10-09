#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Display transform functions for image visualization."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.util import img_as_ubyte


def prepare_for_display(img: np.ndarray, rgb_mapping: list[int] | None = None) -> np.ndarray:
    """Prepare N-channel image for RGB display.

    Converts images with arbitrary channel counts to 3-channel RGB for display.

    Args:
        img: Input image array with shape (H, W, C).
        rgb_mapping: List of 3 channel indices to use as RGB.
            Defaults to [0, 1, 2] (first 3 channels).

    Returns:
        np.ndarray: 3-channel RGB image with shape (H, W, 3) and dtype uint8.

    Raises:
        ValueError: If input is not a numpy array or PIL Image, or if
            ``rgb_mapping`` does not have exactly 3 elements or contains
            indices that exceed the channel count.
    """
    # Handle different input formats
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Ensure we have a valid array
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected numpy array or PIL Image, got {type(img)}")

    # Handle 2D images (grayscale without channel dimension)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    channels = img.shape[-1]

    # Convert based on channel count
    if channels == 1:
        # Grayscale: repeat to RGB
        result = np.repeat(img, 3, axis=-1)
    elif channels == 2:
        # 2 channels: average to grayscale, then repeat
        gray = img.mean(axis=-1, keepdims=True)
        result = np.repeat(gray, 3, axis=-1)
    elif channels == 3:
        # RGB: use as-is
        result = img
    else:
        # N-channels (4+): extract specified channels for RGB mapping
        if rgb_mapping is None:
            rgb_mapping = [0, 1, 2]

        # Validate rgb_mapping
        if len(rgb_mapping) != 3:
            raise ValueError(f"rgb_mapping must have 3 elements, got {len(rgb_mapping)}")
        if any(i >= channels for i in rgb_mapping):
            raise ValueError(f"rgb_mapping indices {rgb_mapping} exceed channel count {channels}")

        result = img[:, :, rgb_mapping]

    # Ensure uint8 output
    if result.dtype != np.uint8:
        if result.max() <= 1.0:
            result = (result * 255).clip(0, 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def display_image_normalisation(
    img: np.ndarray, rgb_mapping: list[int] | None = None
) -> Image.Image:
    """Normalises the image for display.

    Args:
        img: The input image array.
        rgb_mapping: For N-channel images, which channels to display as RGB.

    Returns:
        PIL.Image.Image: The normalised image.
    """
    # Handle NaN/inf values
    if not np.isfinite(img).all():
        img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

    img = img - np.min(img)
    img_max = np.max(img)

    # Avoid division by zero
    if img_max > 0:
        img = img / img_max
    else:
        img = np.zeros_like(img)  # Handle constant images

    img = img_as_ubyte(img)

    # Convert to displayable RGB
    img = prepare_for_display(img, rgb_mapping=rgb_mapping)

    return Image.fromarray(img)


# from utility_functions
def apply_transforms_ui(
    img: Image.Image,
    invert: bool,
    brightness: float,
    contrast: float,
    unsharp_mask_applied: bool,
    show_r: bool = True,
    show_g: bool = True,
    show_b: bool = True,
) -> Image.Image:
    """Applies the requested transformations to the given PIL Image.

    The image is always 3-channel RGB by the time it reaches here
    (:func:`display_image_normalisation` collapses any band count to RGB), so
    channel toggling is expressed purely as the three ``show_*`` flags.

    Args:
        img: The original image.
        invert: Whether to invert colors.
        brightness: Brightness factor.
        contrast: Contrast factor.
        unsharp_mask_applied: Whether to apply an unsharp mask.
        show_r: Whether to show the red channel.
        show_g: Whether to show the green channel.
        show_b: Whether to show the blue channel.

    Returns:
        PIL.Image.Image: The transformed image.
    """
    # Apply inversion
    if invert:
        img = ImageOps.invert(img)

    # Apply brightness
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

    # Apply contrast
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

    # Apply unsharp mask if enabled
    if unsharp_mask_applied:
        img = img.filter(ImageFilter.UnsharpMask())

    # Apply channel toggling on the displayed RGB channels.
    channels_mask = [show_r, show_g, show_b]
    if not all(channels_mask):
        # Convert PIL image to numpy array
        img_array = np.array(img)

        # Apply masking to the image array (zero out disabled channels)
        for i, show_channel in enumerate(channels_mask):
            if not show_channel and i < img_array.shape[-1]:
                img_array[:, :, i] = 0

        # Convert back to PIL image
        img = Image.fromarray(img_array)

    return img
