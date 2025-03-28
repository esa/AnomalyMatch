#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from PIL import Image
import numpy as np
import io


def numpy_array_to_byte_stream(numpy_array, normalize=True):
    """Convert a numpy array to a byte stream.

    Args:
        numpy_array (np.ndarray): The input numpy array
        normalize (bool): Flag to normalize the input array

    Returns:
        bytes: The byte stream of the
    """
    if normalize:
        numpy_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min())
    # Convert a numpy array to a PIL image
    pil_img = Image.fromarray((numpy_array * 255).astype(np.uint8))
    # Create a bytes buffer for the image
    buffer = io.BytesIO()
    # Save the image to the buffer in PNG format
    pil_img.save(buffer, format="PNG")
    # Return the buffer's contents
    return buffer.getvalue()
