#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Shared image conversion utilities."""

from __future__ import annotations

import numpy as np
from loguru import logger
from skimage.util import img_as_ubyte


def ensure_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """Convert an image array to uint8 HWC format.

    Handles CHW-to-HWC transposition and float-to-uint8 conversion
    using skimage for consistency across the codebase.

    Args:
        image: Input image array (CHW or HWC, any numeric dtype).

    Returns:
        uint8 HWC numpy array.

    Raises:
        ValueError: If *image* is empty or has fewer than 2 dimensions.
    """
    if image.size == 0:
        raise ValueError("Cannot convert empty image array")
    if image.ndim < 2:
        raise ValueError(f"Expected at least 2D image, got {image.ndim}D")

    # CHW → HWC
    if image.ndim == 3 and image.shape[0] <= 4 and image.shape[2] > 4:
        image = image.transpose(1, 2, 0)

    if image.ndim == 3 and image.shape[-1] > 4:
        logger.warning(
            "Image has {} channels — only first 4 can be displayed correctly",
            image.shape[-1],
        )

    # Float → uint8 via skimage
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = img_as_ubyte(image.clip(0, 1))
            else:
                image = img_as_ubyte((image / 255.0).clip(0, 1))
        else:
            image = image.clip(0, 255).astype(np.uint8)

    return image
