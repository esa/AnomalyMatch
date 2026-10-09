#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Configuration validation and constraint checking."""

import os

import numpy as np
from dotmap import DotMap
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from loguru import logger

from anomaly_match.data_io.load_images import (
    fitsbolt_channel_combination,
    normalise_channel_combination,
)
from anomaly_match.utils.normalisation_parameters import (
    CUTOUT_PADDING_FACTOR_MAX,
    CUTOUT_PADDING_FACTOR_MIN,
    validate_normalisation,
)


def configs_differ(a: dict, b: dict) -> bool:
    """Compare two config dicts, handling numpy arrays.

    Args:
        a: First config dict.
        b: Second config dict.

    Returns:
        ``True`` if any values differ between *a* and *b*.
    """
    return bool(diff_configs(a, b))


def diff_configs(a: dict | None, b: dict) -> dict:
    """Return the keys that differ between two config dicts.

    Handles numpy arrays (element-wise compare) and treats missing keys
    on either side as a difference.  Returns a ``{key: (a_value,
    b_value)}`` map suitable for logging *what* changed.

    Args:
        a: Previous config dict, or ``None`` to mean "no prior snapshot"
            (all keys of *b* are treated as new).
        b: Current config dict.

    Returns:
        Dict of differing keys → ``(previous, current)`` pairs.  Empty
        when the two configs are equivalent.
    """
    if a is None:
        return {"(no previous snapshot)": (None, None)}
    diff: dict = {}
    for key in set(a) | set(b):
        va = a.get(key, "<missing>")
        vb = b.get(key, "<missing>")
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not np.array_equal(np.asarray(va), np.asarray(vb)):
                diff[key] = (va, vb)
        elif va != vb:
            diff[key] = (va, vb)
    return diff


def serialisable_config(cfg: dict) -> dict:
    """Convert a config dict into a comparison-safe form.

    Numpy arrays are flattened to nested lists so ``==`` against a
    stashed snapshot never raises ``ValueError: The truth value of an
    array is ambiguous``.  Other values pass through unchanged.

    Args:
        cfg: Config dict (typically a normalisation-config snapshot from
            a widget).

    Returns:
        Dict with the same keys and numpy-free values.
    """
    out: dict = {}
    for key, val in cfg.items():
        if isinstance(val, np.ndarray):
            out[key] = val.tolist()
        else:
            out[key] = val
    return out


def _return_required_and_optional_keys():
    """Returns the configuration parameters in a unified format.

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
        # data_dir: directory path for all source types (image folders, .zarr dirs, catalogue dirs)
        "data_dir": ["directory", None, None, False, None],
        "training_data_source": [
            str,
            None,
            None,
            True,
            ["image_folder", "zarr", "cutana"],
        ],
        "labeled_cache_path": [str, None, None, True, None],
        # Required file parameters
        "label_file": ["file", None, None, False, None],
        "metadata_file": ["file", None, None, True, None],  # Optional, can be None
        # Required numeric parameter"
        "seed": [float, None, None, False, None],  # accepts int or float
        # Required positive integers
        "num_workers": [int, 0, None, False, None],
        "uratio": [int, 1, None, False, None],
        "batch_size": [int, 1, None, False, None],
        "num_train_iter": [int, 1, None, False, None],
        "eval_batch_size": [int, 1, None, False, None],
        # Required integers >= 10
        "N_to_load": [int, 10, None, False, None],
        "top_N": [int, 10, None, False, None],
        "subprocess_buffer_size": [int, 100, None, False, None],
        "unlabeled_pool_cap": [int, 1, None, False, None],
        "unlabeled_pool_cap_hires": [int, 1, None, False, None],
        "unlabeled_pool_hires_threshold": [int, 1, None, False, None],
        "cutana_streaming_batch_size": [int, 1, None, False, None],
        "cutana_max_unlabeled_tiles": [int, 1, None, False, None],
        "cutana_size_stratify_bins": [int, 2, None, False, None],
        "cutana_size_stratify_max_px": [int, 2, None, False, None],
        "cutana_min_workers": [int, 1, None, False, None],
        "cutana_max_workers": [int, 1, None, False, None],
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
        "compile_model": [bool, None, None, False, None],
        "cutana_stratify_source_size": [bool, None, None, False, None],
        # Required parameters with allowed values
        "opt": [str, None, None, False, ["SGD", "Adam"]],
        "net": [
            str,
            None,
            None,
            False,
            [
                "efficientnet-lite0",
                "efficientnet-lite1",
                "efficientnet-lite2",
                "efficientnet-lite3",
                "efficientnet-lite4",
                "efficientnet-b0",
                "efficientnet-b1",
                "efficientnet-b2",
                "efficientnet-b3",
                "efficientnet-b4",
                "efficientnet-b5",
                "efficientnet-b6",
                "efficientnet-b7",
                "test-cnn",
            ],
        ],
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
        # Optional: local-disk dir for the live predictions.db (created on use, so
        # validated as a plain string rather than an existing directory).
        "prediction_db_dir": [str, None, None, True, None],
        "N_batch_prediction": [int, 1, None, True, None],
        "gpu": [int, 0, None, False, None],
        "num_channels": [int, 1, None, False, None],
        # fitsbolt config parameters - only validate that it's a DotMap and check size
        "fitsbolt_cfg": ["special_fitsbolt", None, None, True, None],
        "normalisation": ["special_fitsbolt", None, None, False, None],
        "normalisation.image_size": ["special_size", None, None, False, None],
        # AnomalyMatch-specific normalisation keys (not consumed by fb_create_cfg)
        "normalisation.apply_flux_conversion": [bool, None, None, True, None],
        "normalisation.flux_conversion_zeropoint_keyword": [str, None, None, True, None],
        "normalisation.cutout_padding_factor": [
            float,
            CUTOUT_PADDING_FACTOR_MIN,
            CUTOUT_PADDING_FACTOR_MAX,
            True,
            None,
        ],
    }

    return config_spec


def _get_nested_value(cfg: DotMap, key: str):
    """Get a nested value from the config using dot notation.

    Args:
        cfg: Configuration object
        key: Key in dot notation (e.g., 'normalisation.maximum_value')

    Returns:
        Value from the config

    Raises:
        ValueError: If the key is missing from the config.
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
        cfg: Configuration object
        parent_key: Parent key for nested values

    Returns:
        Set of all keys in dot notation
    """
    keys = set()
    for key, value in cfg.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.add(current_key)
        if isinstance(value, DotMap):
            keys.update(_get_all_keys(value, current_key))
    return keys


def _validate_channel_combination(
    channel_combination: np.ndarray | list | tuple | DotMap | None,
) -> None:
    """Validate channel_combination — AnomalyMatch owns this check, not fitsbolt.

    Runs on every branch, because the branch that withholds the matrix from fitsbolt
    would otherwise get no type or shape check at all.  Column count is left to
    fitsbolt, which sees the real matrix whenever it is the one applying it.

    Runs before :func:`normalise_channel_combination`, whose numpy arithmetic would
    otherwise fail on a malformed matrix with a message that names neither the config
    key nor the problem.

    Args:
        channel_combination: The configured matrix, or ``None``.

    Raises:
        ValueError: If the matrix is not a non-empty, rectangular 2-D array of finite numbers.
    """
    # DotMap.copy() turns None into an empty DotMap() on some Python versions,
    # which means "no matrix" just as None does.
    if channel_combination is None or (
        isinstance(channel_combination, DotMap) and not channel_combination
    ):
        return

    if not isinstance(channel_combination, (np.ndarray, list, tuple)):
        raise ValueError(
            "normalisation.channel_combination must be a numpy array (or nested "
            f"sequence), got {type(channel_combination).__name__}"
        )

    try:
        matrix = np.asarray(channel_combination)
    except ValueError as exc:
        # Ragged nested lists: numpy's "inhomogeneous shape" names neither the key
        # nor the problem.
        raise ValueError(
            "normalisation.channel_combination must be rectangular — every row needs "
            f"the same number of weights ({exc})"
        ) from exc
    if not np.issubdtype(matrix.dtype, np.number):
        raise ValueError(
            f"normalisation.channel_combination must contain numbers, got dtype {matrix.dtype}"
        )
    if matrix.ndim != 2:
        raise ValueError(
            "normalisation.channel_combination must be 2-D with shape "
            f"(n_output_channels, n_input_channels), got shape {matrix.shape}"
        )
    if matrix.size == 0:
        raise ValueError(
            "normalisation.channel_combination must have at least one row and one "
            f"column, got shape {matrix.shape}"
        )
    # fitsbolt never checked this either (NaN fails neither ``< 0`` nor the zero-sum
    # test), and a non-finite weight silently yields garbage images.
    if not np.all(np.isfinite(matrix)):
        raise ValueError("normalisation.channel_combination must contain only finite weights")


def validate_config(cfg: DotMap, check_paths: bool = True) -> None:
    """Validate configuration against required and optional keys specification.

    Args:
        cfg: Configuration to validate
        check_paths: Whether to check if file and directory paths exist

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
        if dtype is str:
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

        elif dtype == "path_or_directory":
            if not isinstance(value, str):
                raise ValueError(f"{param_name} must be a string path, got {type(value).__name__}")
            if check_paths and not (
                os.path.exists(value) or os.path.exists(os.path.join(script_dir, value))
            ):
                raise ValueError(
                    f"{param_name} path does not exist: {value} or {os.path.join(script_dir, value)}"
                )

        elif dtype == "file":
            if not isinstance(value, str):
                raise ValueError(
                    f"{param_name} must be a string/file path, got {type(value).__name__}"
                )
            if check_paths and not os.path.isfile(value):
                raise ValueError(f"{param_name} file does not exist: {value}")

        elif dtype is int:
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

        elif dtype is float:
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

        elif dtype is bool:
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

    # Validate normalisation parameter bounds
    if hasattr(cfg, "normalisation"):
        norm_errors = validate_normalisation(cfg.normalisation)
        if norm_errors:
            raise ValueError(
                "Normalisation parameter validation failed:\n" + "\n".join(norm_errors)
            )

    # Also validate normalisation configuration with its own validation function if possible
    if hasattr(cfg, "normalisation"):
        cc = cfg.normalisation.channel_combination
        _validate_channel_combination(cc)
        # Coerce nested sequences up front so everything below — the rescaling, the
        # n_output_channels inference — behaves the same whether the user wrote a list
        # or an ndarray.
        if isinstance(cc, (list, tuple)):
            cc = np.asarray(cc)
            cfg.normalisation.channel_combination = cc

        # Keep channel_combination loss-free: a row of non-negative weights
        # summing to >1 would push bright pixels past the uint8 range and be
        # silently clipped by fitsbolt (Lasloruhberg/fitsbolt#40).  Rescale such
        # rows to sum 1 once here so every downstream combine site (training,
        # prediction, Cutana/Zarr/image) shares the corrected matrix.
        if isinstance(cc, np.ndarray):
            normalised_cc, rescaled_rows, has_negative = normalise_channel_combination(cc)
            if rescaled_rows:
                logger.warning(
                    "channel_combination row(s) {} have non-negative weights summing to >1; "
                    "rescaling each to sum 1 so no signal is clipped on uint8 output. Use "
                    "convex weights (non-negative, row sum <= 1) to silence this.",
                    rescaled_rows,
                )
                cfg.normalisation.channel_combination = normalised_cc
                cc = normalised_cc
            if has_negative:
                logger.warning(
                    "channel_combination contains negative weights; combined values may fall "
                    "below 0 and be clipped to 0 on uint8 output."
                )

        fits_ext = cfg.normalisation.fits_extension

        # Infer n_output_channels from channel_combination matrix if provided
        # isinstance rather than hasattr(cc, "shape"): probing an attribute on a
        # dynamic DotMap creates it, so the empty-DotMap form of "no matrix"
        # (what DotMap.copy() makes of None) would gain a bogus
        # channel_combination.shape key.
        if isinstance(cc, np.ndarray) and cc.ndim == 2:
            inferred = cc.shape[0]
            if inferred != cfg.normalisation.n_output_channels:
                logger.info(
                    f"Setting n_output_channels to {inferred} "
                    f"from channel_combination shape {cc.shape}"
                )
                cfg.normalisation.n_output_channels = inferred

        # Auto-create identity channel_combination for multiple FITS extensions
        # when no explicit matrix is provided.
        elif fits_ext is not None and isinstance(fits_ext, (list, tuple)) and len(fits_ext) > 1:
            n_ext = len(fits_ext)
            cfg.normalisation.channel_combination = np.eye(n_ext)
            cfg.normalisation.n_output_channels = n_ext
            logger.info(
                f"Auto-created {n_ext}x{n_ext} identity channel_combination "
                f"for {n_ext} FITS extensions (n_output_channels set to {n_ext})"
            )

        # Guard against n_output_channels being None (e.g. user unset it
        # expecting auto-inference but only has a single FITS extension).
        if cfg.normalisation.n_output_channels is None:
            raise ValueError(
                "n_output_channels is None and could not be inferred. "
                "Set normalisation.n_output_channels explicitly or provide "
                "multiple fits_extension entries or a channel_combination matrix."
            )

        # Keep cfg.num_channels in sync with n_output_channels
        cfg.num_channels = cfg.normalisation.n_output_channels

        # Ensure per-channel normalisation lists match n_output_channels
        n_out = cfg.normalisation.n_output_channels
        for attr in ("norm_asinh_scale", "norm_asinh_clip"):
            val = cfg.normalisation[attr]
            if isinstance(val, list) and len(val) != n_out and len(val) != 1:
                if n_out < len(val):
                    cfg.normalisation[attr] = val[:n_out]
                else:
                    cfg.normalisation[attr] = val + [val[-1]] * (n_out - len(val))

        try:
            # Validate exactly the matrix get_fitsbolt_config will hand fitsbolt, so a
            # config passes here iff the run accepts it: blank rows arrive substituted
            # (#588), and non-FITS sources hand over None because AnomalyMatch applies
            # the matrix itself — fitsbolt would otherwise reject the negative weights
            # that are legal on that path.  _validate_channel_combination above covers
            # the structure fitsbolt then never sees.
            _ = fb_create_cfg(
                output_dtype=cfg.normalisation.output_dtype,
                size=cfg.normalisation.image_size,
                fits_extension=cfg.normalisation.fits_extension,
                interpolation_order=cfg.normalisation.interpolation_order,
                n_output_channels=cfg.normalisation.n_output_channels,
                normalisation_method=cfg.normalisation.normalisation_method,
                channel_combination=fitsbolt_channel_combination(cfg),
                num_workers=max(cfg.num_workers, 1),
                norm_maximum_value=cfg.normalisation.norm_maximum_value,
                norm_minimum_value=cfg.normalisation.norm_minimum_value,
                norm_log_calculate_minimum_value=cfg.normalisation.norm_log_calculate_minimum_value,
                norm_crop_for_maximum_value=cfg.normalisation.norm_crop_for_maximum_value,
                norm_asinh_scale=cfg.normalisation.norm_asinh_scale,
                norm_asinh_clip=cfg.normalisation.norm_asinh_clip,
                norm_asinh_n_samples=cfg.normalisation.norm_asinh_n_samples,
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
                    "normalisation.norm_asinh_n_samples",
                ]
            )
        except Exception as e:
            logger.error(f"normalisation configuration validation failed: {e}")
            raise ValueError(f"normalisation configuration validation failed: {e}")

    # fitsbolt_cfg is a nested DotMap created by fitsbolt's create_config()
    actual_keys = _get_all_keys(cfg)
    # Exclude the entire fitsbolt_cfg subtree — it's managed by fitsbolt,
    # not by our config schema.
    unexpected_keys = {k for k in (actual_keys - expected_keys) if not k.startswith("fitsbolt_cfg")}

    if unexpected_keys:
        logger.warning(f"Found unexpected keys in config: {sorted(unexpected_keys)}")
        logger.info("Config: validation partially successful")
    else:
        logger.info("Config: validation successful")
