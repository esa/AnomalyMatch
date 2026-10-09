#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Logging level configuration."""

import os
import sys
from datetime import datetime, timezone

from dotmap import DotMap
from loguru import logger

# Stable per-process log path — generated once so repeated set_log_level
# calls reuse the same file instead of creating a new one each time.
_log_file_path: str | None = None

# Sink IDs this module added on its previous call.  Tracked so each call
# removes ONLY its own sinks instead of a blanket ``logger.remove()``.  The
# blanket remove was a real bug: ``set_log_level`` runs after every prediction
# subprocess (Session._launch_prediction_subprocess), so it silently dropped
# the per-session ``session.log`` sink (added once by SessionIOHandler) and the
# UI's live-log widget the moment the first scoring chunk finished — freezing
# ``session.log`` for the rest of a multi-hour, hundred-chunk run.
_own_sink_ids: list[int] = []


def _get_log_path() -> str:
    """Return the log file path, creating the logs directory if needed.

    Uses a timestamped filename so each process gets its own file and
    loguru never needs to rename/rotate (which fails on Windows when
    another process holds the file open).
    """
    global _log_file_path  # noqa: PLW0603
    if _log_file_path is None:
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
        os.makedirs(logs_dir, exist_ok=True)
        stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        _log_file_path = os.path.join(logs_dir, f"UI_thread_{stamp}.log")
    return _log_file_path


def set_log_level(log_level: str, cfg: DotMap, log_to_file: bool = True):
    """Set the log level for the logger.

    Args:
        log_level: The log level to set. Options are 'TRACE', 'DEBUG', 'INFO',
            'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'.
        cfg: The configuration object.
        log_to_file: If True, logs will be saved to a file. Default is True.
    """
    # Define valid log levels
    valid_log_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

    # Assert that the provided log_level is valid
    assert log_level.upper() in valid_log_levels, (
        f"Invalid log level: {log_level}. Expected one of {valid_log_levels}."
    )

    # Remove only the sinks this module added last time (plus loguru's built-in
    # default stderr sink on the first call), leaving externally-registered
    # sinks — the per-session session.log and the UI live-log widget — intact.
    global _own_sink_ids
    for sink_id in _own_sink_ids or [0]:
        try:
            logger.remove(sink_id)
        except ValueError:
            pass  # already removed (e.g. default sink cleared elsewhere)
    _own_sink_ids = []

    # Bypass ipykernel's OutStream (replaces sys.stderr in a kernel): its
    # parent_header walk spins forever when ThreadPoolExecutor ident reuse
    # creates a cycle in _thread_to_parent, hanging the scoring thread.
    console_stream = sys.__stderr__ if sys.__stderr__ is not None else sys.stderr
    _own_sink_ids.append(
        logger.add(
            console_stream,
            colorize=True,
            level=log_level.upper(),
            format="<green>{time:HH:mm:ss}</green>|AnomalyMatch-<blue>{level}</blue>| <level>{message}</level>",
        )
    )

    logger.debug(f"Setting LogLevel to {log_level.upper()}")

    # Per-process timestamped log file — no rotation needed, so no
    # rename that could fail with PermissionError on Windows.
    if log_to_file:
        _own_sink_ids.append(
            logger.add(
                _get_log_path(),
                enqueue=True,
                format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
            )
        )

    # Store the log level in the config
    cfg.log_level = log_level.upper()
