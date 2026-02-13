#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
Entry point for the AnomalyMatch UI.

This module provides the main entry function for starting the UI with a Session.
"""

from loguru import logger

from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widget import Widget


def start_ui(session):
    """Start the AnomalyMatch UI with the given session.

    This is the main entry point for launching the UI. It sets up the
    BackendInterface with the session and creates/displays the Widget.

    Args:
        session: An AnomalyMatch Session instance that has been initialized
                 and has predictions ready for display.

    Returns:
        Widget: The created Widget instance.

    Example:
        >>> import anomaly_match as am
        >>> from anomaly_match_ui import start_ui
        >>>
        >>> cfg = am.get_default_cfg()
        >>> cfg.data_dir = "path/to/images"
        >>> session = am.Session(cfg)
        >>> session.train(cfg)
        >>> widget = start_ui(session)
    """
    logger.debug("Starting AnomalyMatch UI...")

    # Set up the backend interface with the session
    BackendInterface.set_session(session)

    # Create and start the widget
    widget = Widget()
    widget.start()

    logger.debug("AnomalyMatch UI started successfully.")

    return widget
