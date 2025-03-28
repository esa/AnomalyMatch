#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
import toml
import os
from pathlib import Path
from loguru import logger


def save_cfg(cfg):
    """Save configuration to the run directory as a TOML file.

    This function saves the provided configuration object to a TOML file in the
    specified save path. It creates the directory if it doesn't exist.

    Args:
        cfg: Configuration object with save_path attribute and toDict() method

    Raises:
        ValueError: If saving the configuration fails
    """
    try:
        # Create the directory if it doesn't exist
        Path(cfg.save_path).mkdir(parents=True, exist_ok=True)

        cfg_filename = os.path.join(cfg.save_path, "cfg.toml")
        logger.debug(f"Saving configuration to {cfg_filename}")

        with open(cfg_filename, "w") as handle:
            toml.dump(cfg.toDict(), handle)

        logger.debug("Configuration saved successfully")
    except Exception as e:
        error_msg = f"Failed to save configuration: {str(e)}"
        logger.error(error_msg)
        raise ValueError(error_msg)
