#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Functions for loading and processing images as a wrapper around fitsbolt
"""

from dotmap import DotMap
from fitsbolt.image_loader import load_and_process_images, _process_image
from fitsbolt.cfg.create_config import create_config as fb_create_cfg


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
        num_workers=cfg.num_workers,
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
    images_list = load_and_process_images(
        filepaths,
        cfg=None,
        output_dtype=cfg.normalisation.output_dtype,
        size=cfg.normalisation.image_size,
        fits_extension=cfg.normalisation.fits_extension,
        interpolation_order=cfg.normalisation.interpolation_order,
        normalisation_method=cfg.normalisation.normalisation_method,
        channel_combination=cfg.normalisation.channel_combination,
        n_output_channels=cfg.normalisation.n_output_channels,
        num_workers=cfg.num_workers,
        norm_maximum_value=cfg.normalisation.norm_maximum_value,
        norm_minimum_value=cfg.normalisation.norm_minimum_value,
        norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
        norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
        norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
        norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
        desc=desc,
        show_progress=show_progress,
    )
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
