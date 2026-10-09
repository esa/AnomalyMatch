#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the small diagnostic helpers in anomaly_match.pipeline.session."""

from __future__ import annotations

from anomaly_match.pipeline.session import _count_db_rows
from anomaly_match.prediction import AnomalyScoreDB


def test_count_db_rows_missing_file(tmp_path):
    """Missing DB must report zero rows (not raise)."""
    assert _count_db_rows(str(tmp_path / "nope.db")) == 0


def test_count_db_rows_reflects_stored_results(tmp_path):
    """After storing results, _count_db_rows reports the current count."""
    db_path = tmp_path / "predictions.db"
    with AnomalyScoreDB(str(db_path)) as db:
        db.store_results([("a.png", 0.1), ("b.png", 0.2), ("c.png", 0.3)])

    assert _count_db_rows(str(db_path)) == 3
