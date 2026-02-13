#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
AnomalyMatch UI Package - Jupyter notebook interface for anomaly detection.

This package provides the UI components for AnomalyMatch, separated from the core
backend functionality to allow headless operation of the backend.
"""

from anomaly_match_ui.app import start_ui
from anomaly_match_ui.utils.backend_interface import BackendInterface

__all__ = ["start_ui", "BackendInterface"]
