#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for cutana catalogue validation and source counting."""

import pandas as pd
import pytest

from anomaly_match.utils.cutana_stream_utils import cutana_validate_files_and_count_sources


def _make_valid_df(n=50):
    """Create a minimal valid cutana catalogue DataFrame."""
    return pd.DataFrame(
        {
            "SourceID": range(n),
            "RA": [180.0] * n,
            "Dec": [-30.0] * n,
            "fits_file_paths": ["dummy.fits"] * n,
            "diameter_pixel": [10] * n,
        }
    )


@pytest.fixture
def valid_parquet(tmp_path):
    """Write a valid parquet catalogue and return its path."""
    path = tmp_path / "valid.parquet"
    _make_valid_df(100).to_parquet(path, index=False)
    return path


@pytest.fixture
def invalid_parquet(tmp_path):
    """Write a parquet file with wrong columns."""
    path = tmp_path / "invalid.parquet"
    pd.DataFrame({"bad_col": [1, 2, 3]}).to_parquet(path, index=False)
    return path


class TestCutanaValidateFilesAndCountSources:
    def test_valid_parquet_accepted(self, valid_parquet):
        files, total, chunks = cutana_validate_files_and_count_sources(
            [valid_parquet], chunk_size=50
        )
        assert len(files) == 1
        assert total == 100
        assert chunks == 2  # 100 rows / 50 chunk_size

    def test_row_count_from_metadata(self, tmp_path):
        """Row counts should match actual rows without scanning data."""
        paths = []
        for i, n in enumerate([200, 300]):
            p = tmp_path / f"cat_{i}.parquet"
            _make_valid_df(n).to_parquet(p, index=False)
            paths.append(p)

        files, total, chunks = cutana_validate_files_and_count_sources(paths, chunk_size=100)
        assert len(files) == 2
        assert total == 500
        assert chunks == 5  # 200/100 + 300/100

    def test_invalid_columns_rejected(self, invalid_parquet):
        files, total, chunks = cutana_validate_files_and_count_sources(
            [invalid_parquet], chunk_size=100
        )
        assert len(files) == 0
        assert total == 0

    def test_mixed_valid_invalid(self, valid_parquet, invalid_parquet):
        """Invalid file should be skipped, valid file should be counted."""
        files, total, chunks = cutana_validate_files_and_count_sources(
            [invalid_parquet, valid_parquet], chunk_size=100
        )
        assert len(files) == 1
        assert total == 100

    def test_valid_csv_accepted(self, tmp_path):
        path = tmp_path / "valid.csv"
        _make_valid_df(75).to_csv(path, index=False)
        files, total, chunks = cutana_validate_files_and_count_sources([path], chunk_size=50)
        assert len(files) == 1
        assert total == 75
        assert chunks == 2  # 50 + 25
