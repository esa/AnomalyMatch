#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

from dotmap import DotMap
from loguru import logger

import os
from fitsbolt.cfg.create_config import create_config as fb_create_cfg


def _return_required_and_optional_keys():
    """
    Returns the configuration parameters in a unified format.

    Returns:
        dict: Dictionary with parameter_name as key and [dtype, min, max, optional, allowed_values] as value
              - dtype: expected data type (str, int, float, bool, list, tuple, 'directory', 'file', 'special')
              - min: minimum value (None if not applicable)
              - max: maximum value (None if not applicable)
              - optional: True if parameter is optional, False if required
              - allowed_values: list of allowed values (None if not applicable)
    """
    config_spec = {
        # Required string parameters
        "name": [str, None, None, False, None],
        "save_file": [str, None, None, False, None],
        "save_dir": [str, None, None, False, None],
        "save_path": [str, None, None, False, None],
        "model_path": [str, None, None, True, None],  # Optional, set by SessionIOHandler
        "output_dir": [str, None, None, False, None],
        # Required directory parameters
        "data_dir": ["directory", None, None, False, None],
        # Required file parameters
        "label_file": ["file", None, None, False, None],
        "metadata_file": ["file", None, None, True, None],  # Optional, can be None
        # Required numeric parameter"
        "seed": [float, None, None, False, None],  # accepts int or float
        # Required positive integers
        "num_workers": [int, 1, None, False, None],
        "uratio": [int, 1, None, False, None],
        "batch_size": [int, 1, None, False, None],
        "num_train_iter": [int, 1, None, False, None],
        "eval_batch_size": [int, 1, None, False, None],
        # Required integers >= 10
        "N_to_load": [int, 10, None, False, None],
        "top_N": [int, 10, None, False, None],
        "subprocess_buffer_size": [int, 100, None, False, None],
        # Required floats in range [0, 1]
        "test_ratio": [float, 0.0, 1.0, False, None],
        "ema_m": [float, 0.0, 1.0, False, None],
        "temperature": [float, 0.0, 1.0, False, None],
        "ulb_loss_ratio": [float, 0.0, 1.0, False, None],
        "p_cutoff": [float, 0.0, 1.0, False, None],
        "lr": [float, 0.0, 1.0, False, None],
        "weight_decay": [float, 0.0, 1.0, False, None],
        "momentum": [float, 0.0, 1.0, False, None],
        "bn_momentum": [float, 0.0, 1.0, False, None],
        # Required boolean parameters
        "pin_memory": [bool, None, None, False, None],
        "oversample": [bool, None, None, False, None],
        "hard_label": [bool, None, None, False, None],
        "pretrained": [bool, None, None, False, None],
        # Required parameters with allowed values
        "opt": [str, None, None, False, ["SGD", "Adam"]],
        "net": [str, None, None, False, ["efficientnet-lite0"]],
        "log_level": [
            str,
            None,
            None,
            False,
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "TRACE"],
        ],
        # Required special parameters
        "num_eval_iter": ["special_eval_iter", None, None, False, None],
        # Optional directory parameters
        "prediction_search_dir": ["directory", None, None, True, None],
        "N_batch_prediction": [int, 1, None, True, None],
        # fitsbolt config parameters - only validate that it's a DotMap and check size
        "normalisation": ["special_fitsbolt", None, None, False, None],
        "normalisation.image_size": ["special_size", None, None, False, None],
    }

    return config_spec


def _get_nested_value(cfg: DotMap, key: str):
    """Get a nested value from the config using dot notation.

    Args:
        cfg (DotMap): Configuration object
        key (str): Key in dot notation (e.g., 'normalisation.maximum_value')

    Returns:
        Any: Value from the config
    """
    current = cfg
    for part in key.split("."):
        try:
            current = current[part]
        except (KeyError, TypeError):
            raise ValueError(f"Missing key in config: {key}")
    return current


def _get_all_keys(cfg: DotMap, parent_key: str = ""):
    """Get all keys in the config using dot notation.

    Args:
        cfg (DotMap): Configuration object
        parent_key (str): Parent key for nested values

    Returns:
        Set[str]: Set of all keys in dot notation
    """
    keys = set()
    for key, value in cfg.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.add(current_key)
        if isinstance(value, DotMap):
            keys.update(_get_all_keys(value, current_key))
    return keys


def validate_config(cfg: DotMap, check_paths: bool = True) -> None:
    """Validate configuration against required and optional keys specification.

    Args:
        cfg (DotMap): Configuration to validate
        check_paths (bool): Whether to check if file and directory paths exist

    Raises:
        ValueError: If configuration is invalid
    """
    # Get configuration specification
    config_spec = _return_required_and_optional_keys()

    # Keep track of checked keys
    expected_keys = set()

    # For relative directory paths, select base
    current_file = os.path.abspath(__file__)
    script_dir = os.path.abspath(os.path.join(current_file, "..", "..", "..", ".."))

    # Validate each parameter
    for param_name, (dtype, min_val, max_val, optional, allowed_values) in config_spec.items():
        expected_keys.add(param_name)

        # Try to get the value, handle missing optional parameters
        try:
            value = _get_nested_value(cfg, param_name)
        except ValueError:
            if optional:
                continue  # Skip missing optional parameters
            else:
                raise ValueError(
                    f"Missing required parameter: {param_name}"
                    + f"(type: {dtype.__name__ if hasattr(dtype, '__name__') else dtype})"
                )

        # Skip validation for None values on optional parameters
        if value is None and optional:
            continue

        # Helper function to format constraint info
        def _format_constraints():
            constraints = []
            if min_val is not None:
                constraints.append(f"min: {min_val}")
            if max_val is not None:
                constraints.append(f"max: {max_val}")
            if allowed_values is not None:
                constraints.append(f"allowed: {allowed_values}")
            return f" ({', '.join(constraints)})" if constraints else ""

        # Validate based on data type
        if dtype == str:
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string, got {type(value).__name__}{_format_constraints()}"
                )
            # Check allowed values for string types
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got '{value}'")

        elif dtype == "directory":
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string/directory, got {type(value).__name__}"
                )
            if (
                check_paths
                and not os.path.isdir(value)
                and not os.path.isdir(os.path.join(script_dir, value))
            ):
                raise ValueError(
                    f"{param_name} directory does not exist: {value} or {os.path.join(script_dir, value)}"
                )

        elif dtype == "file":
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string/file path, got {type(value).__name__}"
                )
            if check_paths and not os.path.isfile(value):
                raise ValueError(f"{param_name} file does not exist: {value}")

        elif dtype == int:
            if not isinstance(value, int):
                raise ValueError(
                    f"{param_name} must be an integer, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")

        elif dtype == float:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{param_name} must be a number, got {type(value).__name__}{_format_constraints()}"
                )
            if min_val is not None and value < min_val:
                raise ValueError(
                    f"{param_name} must be >= {min_val}, got {value}{_format_constraints()}"
                )
            if max_val is not None and value > max_val:
                raise ValueError(
                    f"{param_name} must be <= {max_val}, got {value}{_format_constraints()}"
                )
            if allowed_values is not None and value not in allowed_values:
                raise ValueError(f"{param_name} must be one of {allowed_values}, got {value}")

        elif dtype == bool:
            if not isinstance(value, bool):
                raise ValueError(f"{param_name} must be a boolean, got {type(value).__name__}")

        # Handle special validation cases
        elif dtype == "special_size":
            if value is not None:
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    raise ValueError(
                        f"{param_name} must be a list or tuple of length 2, got {type(value).__name__}"
                        + f"with length {len(value) if hasattr(value, '__len__') else 'unknown'}"
                    )

        elif dtype == "special_eval_iter":
            if not isinstance(value, int) or (value != -1 and value <= 0):
                raise ValueError(
                    f"{param_name} must be an integer > 0 or -1, got {value} (type: {type(value).__name__})"
                )

        elif dtype == "special_fitsbolt":
            # The fitsbolt DotMap will be validated by fb_create_cfg
            if not isinstance(value, DotMap):
                raise ValueError(f"{param_name} must be a DotMap, got {type(value).__name__}")
        else:
            raise ValueError(f"Unknown data type for {param_name}: {dtype}")

    # Also validate normalisation configuration with its own validation function if possible
    if hasattr(cfg, "normalisation"):
        try:
            # Use fitsbolt's own validation by calling its create_config function
            _ = fb_create_cfg(
                output_dtype=cfg.normalisation.output_dtype,
                size=cfg.normalisation.image_size,
                fits_extension=cfg.normalisation.fits_extension,
                interpolation_order=cfg.normalisation.interpolation_order,
                normalisation_method=cfg.normalisation.normalisation_method,
                channel_combination=cfg.normalisation.channel_combination,
                num_workers=cfg.num_workers,
                norm_maximum_value=cfg.normalisation.norm_maximum_value,
                norm_minimum_value=cfg.normalisation.norm_minimum_value,
                norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
                norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
                norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
                norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
            )
            logger.debug("fitsbolt configuration validated successfully")
            # add the fitsbolt keys to expected keys used in above function call to expected_keys
            expected_keys.update(
                [
                    "normalisation.output_dtype",
                    "normalisation.image_size",
                    "normalisation.n_output_channels",
                    "normalisation.fits_extension",
                    "normalisation.interpolation_order",
                    "normalisation.normalisation_method",
                    "normalisation.channel_combination",
                    "normalisation.norm_maximum_value",
                    "normalisation.norm_minimum_value",
                    "normalisation.norm_log_calculate_minimum_value",
                    "normalisation.norm_crop_for_maximum_value",
                    "normalisation.norm_asinh_scale",
                    "normalisation.norm_asinh_clip",
                ]
            )
        except Exception as e:
            logger.error(f"normalisation configuration validation failed: {e}")
            raise ValueError(f"normalisation configuration validation failed: {e}")

    # Check for unexpected keys
    actual_keys = _get_all_keys(cfg)
    unexpected_keys = actual_keys - expected_keys

    if unexpected_keys:
        logger.warning(f"Found unexpected keys in config: {sorted(unexpected_keys)}")
        logger.info("Config: validation partially successful")
    else:
        logger.info("Config: validation successful")
