#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.util import img_as_ubyte


def prepare_for_display(img, rgb_mapping=None):
    """Prepare N-channel image for RGB display.

    Converts images with arbitrary channel counts to 3-channel RGB for display.

    Args:
        img (np.ndarray): Input image array with shape (H, W, C)
        rgb_mapping (list, optional): List of 3 channel indices to use as RGB.
            Defaults to [0, 1, 2] (first 3 channels).

    Returns:
        np.ndarray: 3-channel RGB image with shape (H, W, 3) and dtype uint8
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
            result = (result * 255).astype(np.uint8)
        else:
            result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def display_image_normalisation(img, rgb_mapping=None):
    """Normalises the image for display.

    Args:
        img (np.ndarray): The input image array.
        rgb_mapping (list, optional): For N-channel images, which channels to display as RGB.

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
    img,
    invert,
    brightness,
    contrast,
    unsharp_mask_applied,
    show_r=True,
    show_g=True,
    show_b=True,
    channel_visibility=None,
):
    """
    Applies the requested transformations to the given PIL Image.

    Args:
        img (PIL.Image.Image): The original image.
        invert (bool): Whether to invert colors.
        brightness (float): Brightness factor.
        contrast (float): Contrast factor.
        unsharp_mask_applied (bool): Whether to apply an unsharp mask.
        show_r (bool): Whether to show the red channel (for RGB mode).
        show_g (bool): Whether to show the green channel (for RGB mode).
        show_b (bool): Whether to show the blue channel (for RGB mode).
        channel_visibility (list, optional): For N-channel mode, list of booleans
            indicating which channels to show. If provided, overrides show_r/g/b.

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

    # Apply channel toggling
    # Determine which channels to show
    if channel_visibility is not None:
        # N-channel mode: use first 3 values for RGB display
        channels_mask = (
            channel_visibility[:3] if len(channel_visibility) >= 3 else channel_visibility
        )
        # Pad with True if less than 3 values
        while len(channels_mask) < 3:
            channels_mask.append(True)
    else:
        # RGB mode: use individual channel flags
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
