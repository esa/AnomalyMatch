#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Functions for loading and processing images as a wrapper around fitsbolt
"""

import os

import numpy as np
from astropy.io import fits as astropy_fits
from cutana.flux_conversion import convert_mosaic_to_flux
from dotmap import DotMap
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.image_loader import _process_image, load_and_process_images
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger
from PIL import Image as PILImage

# Exact PIL equivalents for skimage interpolation orders.
# Only orders with a true PIL match are included — others would silently
# produce different results than skimage, causing train/predict divergence.
_PIL_INTERPOLATION = {
    0: PILImage.NEAREST,
    1: PILImage.BILINEAR,
    3: PILImage.BICUBIC,
}


def apply_fits_flux_conversion(filepath, zeropoint_keyword="MAGZERO"):
    """Read a FITS file and convert pixel values to flux density in Jansky.

    Uses the AB zeropoint from the FITS header and cutana's
    ``convert_mosaic_to_flux`` to ensure the training path applies the
    exact same conversion as the cutana prediction path.

    Args:
        filepath: Path to the FITS file.
        zeropoint_keyword: FITS header keyword for the AB zeropoint.

    Returns:
        np.ndarray: Flux-converted 2-D image in Jansky (float32).

    Raises:
        KeyError: If the zeropoint keyword is missing from the header.
        ValueError: If the primary HDU contains no image data.
    """
    with astropy_fits.open(filepath) as hdul:
        if hdul[0].data is None:
            raise ValueError(
                f"Primary HDU of '{filepath}' contains no image data. "
                f"Some observatories store image data in a different extension "
                f"(e.g. hdul[1]). This is not yet supported."
            )
        data = hdul[0].data.astype(np.float32)
        zeropoint = float(hdul[0].header[zeropoint_keyword])
    return convert_mosaic_to_flux(data, zeropoint)


def _pil_resize(image, target_size, interpolation_order=1):
    """Resize an HWC or HW numpy array using PIL (much faster than skimage).

    Args:
        image: numpy array of shape (H, W) or (H, W, C)
        target_size: [height, width]
        interpolation_order: 0=nearest, 1=bilinear, 3=bicubic

    Returns:
        Resized numpy array with same dtype.

    Raises:
        ValueError: If interpolation_order has no exact PIL equivalent.
    """
    h, w = target_size
    if image.shape[0] == h and image.shape[1] == w:
        return image
    if interpolation_order not in _PIL_INTERPOLATION:
        raise ValueError(
            f"interpolation_order={interpolation_order} has no exact PIL equivalent. "
            f"Supported orders for PIL resize: {sorted(_PIL_INTERPOLATION.keys())}. "
            f"Use one of these or switch to a normalisation method that uses skimage resize."
        )
    original_dtype = image.dtype
    resample = _PIL_INTERPOLATION[interpolation_order]
    pil_img = PILImage.fromarray(image)
    # PIL.resize takes (width, height)
    pil_img = pil_img.resize((w, h), resample)
    result = np.array(pil_img)
    if result.dtype != original_dtype:
        result = result.astype(original_dtype)
    return result


def get_fitsbolt_config(cfg, size_override="default"):
    """Get the fitsbolt configuration from the provided configuration.

    Args:
        cfg (DotMap): The configuration object.
        size_override: If "default", use cfg.normalisation.image_size.
                       If None, disable resizing (full resolution).
                       Otherwise use the provided value.

    Returns:
        DotMap: The cfg with a cfg.fitsbolt_cfg subdotmap.
    """
    if not hasattr(cfg, "normalisation"):
        raise ValueError("Configuration must include a normalisation config for fitsbolt.")
    size = cfg.normalisation.image_size if size_override == "default" else size_override
    cfg.fitsbolt_cfg = fb_create_cfg(
        output_dtype=cfg.normalisation.output_dtype,
        size=size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        n_output_channels=cfg.normalisation.n_output_channels,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=cfg.normalisation.channel_combination,
        num_workers=max(cfg.num_workers, 1),
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
        log_level="WARNING",
        force_dtype=True,
    )
    return cfg


def load_and_process_wrapper(filepaths, cfg, desc="Loading images", show_progress=True):
    target_size = cfg.normalisation.image_size
    norm_method = cfg.normalisation.normalisation_method

    # When flux conversion is enabled, load FITS files manually, apply the
    # conversion, and normalise via process_single_wrapper per image.
    if cfg.normalisation.apply_flux_conversion:
        keyword = cfg.normalisation.flux_conversion_zeropoint_keyword
        cfg_with_fb = get_fitsbolt_config(cfg)
        images_list = []
        for fp in filepaths:
            ext = os.path.splitext(fp)[1].lower()
            if ext in (".fits", ".fit", ".fts"):
                converted = apply_fits_flux_conversion(fp, zeropoint_keyword=keyword)
                img = process_single_wrapper(converted, cfg_with_fb, desc=desc)
            else:
                # Non-FITS files: fall through to standard fitsbolt loading
                img = load_and_process_images(
                    [fp],
                    cfg=cfg_with_fb.fitsbolt_cfg,
                    desc=desc,
                    show_progress=False,
                )[0]
            images_list.append(img)
        return [(fp, img) for fp, img in zip(filepaths, images_list)]

    # PIL resize is only safe for CONVERSION_ONLY: other normalisation methods
    # depend on the float64 dtype that fitsbolt's skimage resize produces.
    use_pil_resize = norm_method == NormalisationMethod.CONVERSION_ONLY
    images_list = load_and_process_images(
        filepaths,
        cfg=None,
        output_dtype=cfg.normalisation.output_dtype,
        size=None if use_pil_resize else target_size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=cfg.normalisation.channel_combination,
        n_output_channels=cfg.normalisation.n_output_channels,
        num_workers=max(cfg.num_workers, 1),
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
        desc=desc,
        show_progress=show_progress,
    )
    # Resize with PIL (much faster than fitsbolt's skimage resize)
    if use_pil_resize and target_size is not None:
        interp = cfg.normalisation.interpolation_order
        images_list = [_pil_resize(img, target_size, interp) for img in images_list]
    # return a list of tuples (filename, image)
    # check that images_list has same length as filepaths
    if len(images_list) != len(filepaths):
        raise ValueError(
            f"Mismatch between filepaths ({len(filepaths)}) and images_list ({len(images_list)})"
        )
    return [(fp, img) for fp, img in zip(filepaths, images_list)]


def load_and_process_single_wrapper(
    filepath,
    cfg,
    desc="image load and process",
    show_progress=False,
    prediction=False,
    size_override="default",
):
    """Load and process a single image file. Creates a fitsbolt config if not part of current cfg

    Args:
        filepath (str): The path to the image file.
        cfg (DotMap): The configuration object of AnomalyMatch.
        desc (str, optional): Description for the loading process. Defaults to "image load and process".
        show_progress (bool, optional): Whether to show progress. Defaults to False.
        prediction (bool, optional): Whether this is a prediction step. Defaults to False.
        size_override: If "default", use cfg.normalisation.image_size.
                       If None, disable resizing (full resolution).
    Returns:
        np.ndarray: The processed image in H,W,C format.
    """
    if not prediction:
        # cfg might have changed, update fitsbolt config
        cfg = get_fitsbolt_config(cfg, size_override=size_override)
    cfg.fitsbolt_cfg.num_workers = 1  # for single image processing, force to 1
    # this breaks down for prediction filepath = os.path.join(cfg.data_dir, os.path.basename(filename))
    return load_and_process_images(
        filepath,
        cfg=cfg.fitsbolt_cfg,
        desc=desc,
        show_progress=show_progress,
    )


# in the future it might make sense to have a process wrapper, right now using legacy functionality
def process_single_wrapper(image, cfg, desc="source"):
    """Process a single image using fitsbolt.

    Args:
        image: Input image as numpy array
        cfg: Configuration object with fitsbolt_cfg set
        desc: Description for logging

    Returns:
        Processed image

    Raises:
        ValueError: If fitsbolt_cfg is not properly set in cfg
    """
    # Validate fitsbolt_cfg exists and is valid
    # DotMap auto-creates empty DotMaps when accessing missing keys, so check for 'size' key
    fitsbolt_cfg = cfg.fitsbolt_cfg
    if fitsbolt_cfg is None or (isinstance(fitsbolt_cfg, DotMap) and "size" not in fitsbolt_cfg):
        raise ValueError(
            "fitsbolt_cfg is not set in configuration. "
            "Models must be saved with fitsbolt config for prediction. "
            "Please retrain and save the model to include fitsbolt config."
        )

    # processing requires n_expected channels from the input image
    if image.ndim == 2:
        fitsbolt_cfg.n_expected_channels = 1
    else:
        fitsbolt_cfg.n_expected_channels = image.shape[-1]
    fitsbolt_cfg.num_workers = 1  # for single image processing, force to 1
    return _process_image(image, fitsbolt_cfg, image_source=desc)


def detect_num_channels(root_dir, filenames):
    """Detect the number of image channels from a sample image file.

    For FITS files, returns None since channel count depends on
    fits_extension and channel_combination settings.

    Args:
        root_dir: Directory containing the images.
        filenames: List of image filenames.

    Returns:
        Detected number of channels, or None if detection is not applicable.
    """
    if not filenames:
        return None

    sample_file = filenames[0]
    sample_path = os.path.join(root_dir, sample_file)
    ext = os.path.splitext(sample_file)[1].lower()

    # FITS channel count depends on extension/combination config
    if ext in (".fits", ".fit", ".fts"):
        return None

    try:
        img = PILImage.open(sample_path)
        num_channels = len(img.getbands())
        logger.debug(f"Detected {num_channels} channels from {sample_file}")
        return num_channels
    except Exception as e:
        logger.warning(f"Could not detect channels from {sample_file}: {e}")
        return None
