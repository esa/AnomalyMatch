#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.

import toml
from pathlib import Path
from typing import Union, Dict, Any
from dotmap import DotMap
from loguru import logger

from anomaly_match.image_processing.NormalisationMethod import NormalisationMethod


def _critical_optional_fields() -> list:
    """Get a list of critical optional fields that should be preserved in the config.
    These fields are necessary for prediction processes and other components,
    and should not be removed even if they are not strictly required for validation.
    Returns:
        List of critical optional fields
    """
    return [
        "fits_extension",
        "metadata_file",
        "prediction_search_dir",
        "model_path",
        "normalisation_method",
        "normalisation.maximum_value",
        "normalisation.minimum_value",
        "normalisation.crop_for_maximum_value",
        "normalisation.log_calculate_minimum_value",
        "interpolation_order",
    ]


def _convert_enum_to_string(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert NormalisationMethod enum to string for TOML serialization.

    Args:
        config_dict: Configuration dictionary that may contain normalisation_method enum

    Returns:
        Dict with normalisation_method converted to string if present
    """
    if "normalisation_method" in config_dict:
        method = config_dict["normalisation_method"]
        if isinstance(method, NormalisationMethod):
            config_dict["normalisation_method"] = method.name
            logger.debug(f"Converted normalisation_method enum to string: {method.name}")

    return config_dict


def save_config_toml(config: Union[DotMap, Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Save configuration to TOML file.

    Args:
        config: Configuration object (DotMap) or dictionary to save
        file_path: Path where to save the TOML file

    Raises:
        ValueError: If config cannot be serialized to TOML
        IOError: If file cannot be written
    """
    file_path = Path(file_path)

    # Convert DotMap to regular dict if needed
    if isinstance(config, DotMap):
        config_dict = config.toDict()
    else:
        config_dict = dict(config)

    # Ensure critical optional fields are present even when None
    # Use string "null" to represent None in TOML since TOML doesn't support None
    for field in _critical_optional_fields():
        if field not in config_dict:
            config_dict[field] = "null"  # Use string representation for None
        elif config_dict[field] is None:
            config_dict[field] = "null"

    # Convert enums to strings for TOML serialization
    config_dict = _convert_enum_to_string(config_dict)

    try:
        with open(file_path, "w") as f:
            toml.dump(config_dict, f)
        logger.debug(f"Saved configuration to TOML file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save config to TOML: {e}")
        raise IOError(f"Could not save config to {file_path}: {e}")
