#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Prediction result storage and caching.

Provides :class:`AnomalyScoreDB` for SQLite-backed prediction results and
:class:`ImageCache` for LRU image loading.
"""

from anomaly_match.prediction.anomaly_score_db import AnomalyScoreDB, SchemaVersionError
from anomaly_match.prediction.db_location import (
    is_relocated,
    prediction_db_path,
    prepare_local_db,
    session_db_path,
    snapshot_db_to_session,
)
from anomaly_match.prediction.image_cache import ImageCache

__all__ = [
    "AnomalyScoreDB",
    "ImageCache",
    "SchemaVersionError",
    "is_relocated",
    "prediction_db_path",
    "prepare_local_db",
    "session_db_path",
    "snapshot_db_to_session",
]
