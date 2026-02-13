#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Functions to retrieve image filenames from folders.
"""

import os
from pathlib import Path

from fitsbolt import SUPPORTED_IMAGE_EXTENSIONS
from loguru import logger


def get_image_names_from_folder(folder_path, recursive=False, extensions=None):
    """
    Get all image filenames from a folder.

    Args:
        folder_path (str): Path to the folder containing images
        recursive (bool, optional): Whether to search recursively in subfolders. Defaults to False.
        extensions (list, optional): List of file extensions to include.
                                     Defaults to SUPPORTED_IMAGE_EXTENSIONS if None.

    Returns:
        list: List of image filenames (relative to folder_path)
    """
    logger.debug(f"Getting image names from folder: {folder_path}")

    # Use the default extensions if None is provided
    if extensions is None:
        extensions = SUPPORTED_IMAGE_EXTENSIONS

    image_filenames = []

    if recursive:
        # Use pathlib for recursive search
        for ext in extensions:
            # Search for both lowercase and uppercase extensions
            image_filenames.extend(
                [
                    os.path.relpath(str(path), folder_path)
                    for path in Path(folder_path).glob(f"**/*{ext}")
                ]
            )
            image_filenames.extend(
                [
                    os.path.relpath(str(path), folder_path)
                    for path in Path(folder_path).glob(f"**/*{ext.upper()}")
                ]
            )
    else:
        # Use os.listdir for non-recursive search
        for f in os.listdir(folder_path):
            if any(f.lower().endswith(ext.lower()) for ext in extensions):
                image_filenames.append(f)

    logger.debug(f"Found {len(image_filenames)} images in {folder_path}")
    return image_filenames


def get_image_paths_from_folder(folder_path, recursive=False, extensions=None):
    """
    Get all image paths from a folder.

    Args:
        folder_path (str): Path to the folder containing images
        recursive (bool, optional): Whether to search recursively in subfolders. Defaults to False.
        extensions (list, optional): List of file extensions to include.
                                     Defaults to SUPPORTED_IMAGE_EXTENSIONS if None.

    Returns:
        list: List of absolute image paths
    """
    filenames = get_image_names_from_folder(folder_path, recursive, extensions)
    return [os.path.join(folder_path, filename) for filename in filenames]
