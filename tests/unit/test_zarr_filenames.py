#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the shared zarr-filename derivation helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import zarr

from anomaly_match.prediction.zarr_filenames import derive_filenames


def _make_zarr(path, n_images: int = 5) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("images", shape=(n_images, 8, 8, 3), chunks=(1, 8, 8, 3), dtype=np.uint8)


def test_original_filename_column_takes_priority(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    _make_zarr(zarr_path, n_images=3)
    pd.DataFrame(
        {
            "original_filename": ["a.png", "b.png", "c.png"],
            "filename": ["x.png", "y.png", "z.png"],
            "source_id": [1, 2, 3],
        }
    ).to_parquet(tmp_path / "store_metadata.parquet", index=False)

    assert derive_filenames(zarr_path) == ["a.png", "b.png", "c.png"]


def test_filename_column_fallback(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    _make_zarr(zarr_path, n_images=2)
    pd.DataFrame({"filename": ["one.png", "two.png"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    assert derive_filenames(zarr_path) == ["one.png", "two.png"]


def test_source_id_fallback_stringified(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    _make_zarr(zarr_path, n_images=2)
    pd.DataFrame({"source_id": [10, 20]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    assert derive_filenames(zarr_path) == ["10", "20"]


def test_generated_names_when_no_metadata(tmp_path):
    zarr_path = tmp_path / "standalone.zarr"
    _make_zarr(zarr_path, n_images=2)

    names = derive_filenames(zarr_path)
    assert names == ["standalone__image_000000", "standalone__image_000001"]


def test_batch_folder_prefix_is_parent_name(tmp_path):
    batch = tmp_path / "batch_042"
    batch.mkdir()
    zarr_path = batch / "images.zarr"
    _make_zarr(zarr_path, n_images=2)

    names = derive_filenames(zarr_path)
    assert names == ["batch_042__image_000000", "batch_042__image_000001"]


def test_batch_folder_prefers_images_metadata_parquet(tmp_path):
    batch = tmp_path / "batch_042"
    batch.mkdir()
    zarr_path = batch / "images.zarr"
    _make_zarr(zarr_path, n_images=2)
    pd.DataFrame({"original_filename": ["A.png", "B.png"]}).to_parquet(
        batch / "images_metadata.parquet", index=False
    )

    assert derive_filenames(zarr_path) == ["A.png", "B.png"]


def test_row_count_mismatch_falls_back_to_generated(tmp_path):
    zarr_path = tmp_path / "store.zarr"
    _make_zarr(zarr_path, n_images=3)
    # metadata has only 2 rows → treat as unreliable
    pd.DataFrame({"filename": ["a", "b"]}).to_parquet(
        tmp_path / "store_metadata.parquet", index=False
    )

    names = derive_filenames(zarr_path)
    assert names == [
        "store__image_000000",
        "store__image_000001",
        "store__image_000002",
    ]


def test_missing_images_array_returns_empty(tmp_path):
    zarr_path = tmp_path / "broken.zarr"
    root = zarr.open_group(str(zarr_path), mode="w")
    root.create_dataset("not_images", shape=(3,), dtype=np.uint8)
    assert derive_filenames(zarr_path) == []


def test_missing_path_returns_empty(tmp_path):
    assert derive_filenames(tmp_path / "does_not_exist.zarr") == []


@pytest.mark.parametrize("n", [0, 1, 10])
def test_length_matches_image_count(tmp_path, n):
    zarr_path = tmp_path / "store.zarr"
    _make_zarr(zarr_path, n_images=n)
    assert len(derive_filenames(zarr_path)) == n
