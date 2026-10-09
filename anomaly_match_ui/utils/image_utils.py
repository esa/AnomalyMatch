#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Image conversion utilities for widget display."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def numpy_array_to_byte_stream(numpy_array: np.ndarray, normalize: bool = True) -> bytes:
    """Convert a numpy array to a byte stream.

    Args:
        numpy_array: The input image, ``(H, W)`` or ``(H, W, C)``. Channels must
            be last: a CHW array would have its *width* treated as the channel
            axis and be cropped to three columns.
        normalize: Flag to normalize the input array.

    Returns:
        bytes: The byte stream of the image.
    """
    # Handle NaN and inf values by replacing them with finite values
    if np.any(~np.isfinite(numpy_array)):
        numpy_array = numpy_array.copy()  # Don't modify the original array
        numpy_array[~np.isfinite(numpy_array)] = 0

    # Reduce to the three channels that get displayed, *before* normalising so
    # the stretch is computed over what the viewer actually sees.  PIL reads a
    # 4-channel array as RGBA, which would make the fourth output channel an
    # alpha band: an output channel the user blanked (#588) then renders the
    # whole thumbnail fully transparent, and an unblanked one applies the band's
    # pixel values as per-pixel transparency.  [0, 1, 2] matches the default RGB
    # mapping in ``display_transforms.prepare_for_display``.
    if numpy_array.ndim == 3 and numpy_array.shape[2] > 3:
        numpy_array = numpy_array[:, :, :3]

    if normalize:
        # Check if the array has any variation after cleaning NaN/inf
        if numpy_array.max() == numpy_array.min():
            # If all values are the same, create a uniform array
            numpy_array = np.full_like(numpy_array, 0.5)
        else:
            numpy_array = (numpy_array - numpy_array.min()) / (
                numpy_array.max() - numpy_array.min()
            )

    # Squeeze single-channel (H, W, 1) → (H, W) for PIL compatibility
    if numpy_array.ndim == 3 and numpy_array.shape[2] == 1:
        numpy_array = numpy_array[:, :, 0]

    # Convert a numpy array to a PIL image
    pil_img = Image.fromarray((numpy_array * 255).astype(np.uint8))
    # Create a bytes buffer for the image
    buffer = io.BytesIO()
    # Save the image to the buffer in PNG format
    pil_img.save(buffer, format="PNG")
    # Return the buffer's contents
    return buffer.getvalue()
