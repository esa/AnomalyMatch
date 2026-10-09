#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""AnomalyMatch: Semi-supervised anomaly detection for astronomical images."""

import subprocess
from pathlib import Path

from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from .data_io.SessionIOHandler import print_session
from .pipeline.session import Session
from .utils.get_default_cfg import get_default_cfg
from .utils.print_cfg import print_cfg
from .utils.set_log_level import set_log_level

__version__ = "2.0.0"


def _get_git_commit() -> str | None:
    """Return the short git commit hash of this checkout, or None if unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip() or None


__commit__ = _get_git_commit()

__all__ = [
    "get_default_cfg",
    "NormalisationMethod",
    "print_cfg",
    "print_session",
    "Session",
    "set_log_level",
]
