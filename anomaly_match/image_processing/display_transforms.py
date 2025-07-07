#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
import numpy as np
from PIL import ImageOps, ImageEnhance, ImageFilter, Image
from skimage.util import img_as_ubyte


def display_image_normalisation(img):
    """Normalises the image for display.

    Args:
        img (np.ndarray): The input image array.

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
    if img.shape[-1] == 1:  # Convert grayscale to RGB if necessary
        img = np.repeat(img, 3, axis=-1)
    return Image.fromarray(img)


# from utility_functions
def apply_transforms_ui(
    img,
    invert,
    brightness,
    contrast,
    unsharp_mask_applied,
    show_r,
    show_g,
    show_b,
):
    """
    Applies the requested transformations to the given PIL Image.

    Args:
        img (PIL.Image.Image): The original image.
        invert (bool): Whether to invert colors.
        brightness (float): Brightness factor.
        contrast (float): Contrast factor.
        unsharp_mask_applied (bool): Whether to apply an unsharp mask.
        show_r (bool): Whether to show the red channel.
        show_g (bool): Whether to show the green channel.
        show_b (bool): Whether to show the blue channel.

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
    if not (show_r and show_g and show_b):
        # Convert PIL image to numpy array
        img_array = np.array(img)

        # Create a mask for RGB channels
        channels_mask = [show_r, show_g, show_b]

        # Apply masking to the image array (zero out disabled channels)
        for i, show_channel in enumerate(channels_mask):
            if not show_channel:
                img_array[:, :, i] = 0

        # Convert back to PIL image
        img = Image.fromarray(img_array)

    return img
