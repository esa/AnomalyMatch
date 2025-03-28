#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file ‘LICENCE.txt’.
from loguru import logger
import sys
import os
from dotmap import DotMap


def set_log_level(log_level: str, cfg: DotMap, log_to_file: bool = True):
    """Set the log level for the logger.

    Args:
        log_level (str): The log level to set. Options are 'TRACE','DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
        cfg (DotMap): The configuration object.
        log_to_file (bool): If True, logs will be saved to a file. Default is True.

    Raises:
        ValueError: If the provided log_level is not one of the expected values.
    """
    # Define valid log levels
    valid_log_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

    # Assert that the provided log_level is valid
    assert (
        log_level.upper() in valid_log_levels
    ), f"Invalid log level: {log_level}. Expected one of {valid_log_levels}."

    # Create logs directory in project root (two levels up from utils)
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Remove any existing logger configuration
    logger.remove()

    # Add a new logger configuration for console output
    logger.add(
        sys.stderr,
        colorize=True,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green>|AnomalyMatch-<blue>{level}</blue>| <level>{message}</level>",
    )

    logger.debug(f"Setting LogLevel to {log_level.upper()}")

    # Optionally add logging to a file with improved format
    if log_to_file:
        logger.add(
            os.path.join(logs_dir, "UI_thread.log"),
            rotation="1 MB",
            format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
        )

    # Store the log level in the config
    cfg.log_level = log_level.upper()
