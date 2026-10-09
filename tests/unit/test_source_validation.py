#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for source_validation module."""

import os

import numpy as np
import pandas as pd
import pytest
import zarr

from anomaly_match.data_io.source_validation import (
    validate_labeled_files,
)


def _is_lfs_pointer(path: str) -> bool:
    """Return True if *path* is a Git LFS pointer, not real content."""
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return f.read(8) == b"version "


# ---------------------------------------------------------------------------
# Fixtures: Image folder
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_folder(tmp_path):
    """Create a temporary image folder with a few dummy files."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ["img_001.png", "img_002.png", "img_003.png"]:
        (folder / name).write_bytes(b"fake image data")
    return str(folder)


# ---------------------------------------------------------------------------
# Fixtures: Zarr
# ---------------------------------------------------------------------------


@pytest.fixture()
def zarr_single_store(tmp_path):
    """Create a single Zarr store with 20 images (no metadata)."""
    store_path = tmp_path / "test_images.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    images = root.create_dataset(
        "images", shape=(20, 64, 64, 3), chunks=(1, 64, 64, 3), dtype=np.uint8
    )
    images[:] = np.random.randint(0, 255, (20, 64, 64, 3), dtype=np.uint8)
    return str(store_path)


@pytest.fixture()
def zarr_single_store_with_metadata(tmp_path):
    """Create a Zarr store with metadata parquet providing filenames."""
    store_path = tmp_path / "data.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    images = root.create_dataset(
        "images", shape=(5, 32, 32, 3), chunks=(1, 32, 32, 3), dtype=np.uint8
    )
    images[:] = np.random.randint(0, 255, (5, 32, 32, 3), dtype=np.uint8)

    filenames = [
        "star_001.fits",
        "star_002.fits",
        "star_003.fits",
        "galaxy_001.fits",
        "galaxy_002.fits",
    ]
    metadata_df = pd.DataFrame({"original_filename": filenames})
    metadata_df.to_parquet(tmp_path / "data_metadata.parquet")
    return str(store_path), filenames


# ---------------------------------------------------------------------------
# Fixtures: Cutana
# ---------------------------------------------------------------------------


@pytest.fixture()
def cutana_csv_catalogue(tmp_path):
    """Create a CSV catalogue with Cutana-style columns."""
    cat_dir = tmp_path / "catalogue"
    cat_dir.mkdir()
    df = pd.DataFrame(
        {
            "SourceID": [f"SRC_{i:04d}" for i in range(100)],
            "RA": np.random.uniform(0, 360, 100),
            "Dec": np.random.uniform(-90, 90, 100),
            "fits_file_paths": ["['fake.fits']"] * 100,
        }
    )
    csv_path = cat_dir / "catalogue.csv"
    df.to_csv(csv_path, index=False)
    return str(cat_dir)


# ---------------------------------------------------------------------------
# Tests: Image folder
# ---------------------------------------------------------------------------


class TestImageFolderValidation:
    """Tests for image folder source validation."""

    def test_all_found(self, image_folder):
        label_df = pd.DataFrame({"id": ["img_001.png", "img_002.png"]})
        result = validate_labeled_files(image_folder, label_df, "image_folder")

        assert set(result.found) == {"img_001.png", "img_002.png"}
        assert result.missing == []
        assert len(result.found_locations) == 2

    def test_some_missing(self, image_folder):
        label_df = pd.DataFrame({"id": ["img_001.png", "nonexistent.png"]})
        result = validate_labeled_files(image_folder, label_df, "image_folder")

        assert result.found == ["img_001.png"]
        assert result.missing == ["nonexistent.png"]

    def test_location_is_filepath(self, image_folder):
        label_df = pd.DataFrame({"id": ["img_001.png"]})
        result = validate_labeled_files(image_folder, label_df, "image_folder")

        filepath = result.found_locations["img_001.png"][0]
        assert os.path.isfile(filepath)


# ---------------------------------------------------------------------------
# Tests: Zarr (generated filenames)
# ---------------------------------------------------------------------------


class TestZarrGeneratedFilenameValidation:
    """Tests for Zarr stores without metadata (generated filenames)."""

    def test_valid_indices(self, zarr_single_store):
        label_df = pd.DataFrame(
            {
                "id": ["test_images__image_000000", "test_images__image_000019"],
            }
        )
        result = validate_labeled_files(zarr_single_store, label_df, "zarr")

        assert len(result.found) == 2
        assert result.missing == []

    def test_out_of_range_index(self, zarr_single_store):
        label_df = pd.DataFrame(
            {
                "id": ["test_images__image_000000", "test_images__image_000099"],
            }
        )
        result = validate_labeled_files(zarr_single_store, label_df, "zarr")

        assert result.found == ["test_images__image_000000"]
        assert result.missing == ["test_images__image_000099"]

    def test_location_has_store_path_and_index(self, zarr_single_store):
        label_df = pd.DataFrame({"id": ["test_images__image_000005"]})
        result = validate_labeled_files(zarr_single_store, label_df, "zarr")

        store_path, idx = result.found_locations["test_images__image_000005"]
        assert store_path == zarr_single_store
        assert idx == 5


# ---------------------------------------------------------------------------
# Tests: Zarr (metadata filenames)
# ---------------------------------------------------------------------------


class TestZarrMetadataFilenameValidation:
    """Tests for Zarr stores with metadata parquet providing filenames."""

    def test_found_by_metadata_name(self, zarr_single_store_with_metadata):
        store_path, filenames = zarr_single_store_with_metadata
        label_df = pd.DataFrame({"id": [filenames[0], filenames[2]]})
        result = validate_labeled_files(store_path, label_df, "zarr")

        assert set(result.found) == {filenames[0], filenames[2]}
        assert result.missing == []

    def test_location_index_matches_metadata_position(self, zarr_single_store_with_metadata):
        store_path, filenames = zarr_single_store_with_metadata
        label_df = pd.DataFrame({"id": [filenames[3]]})
        result = validate_labeled_files(store_path, label_df, "zarr")

        _, idx = result.found_locations[filenames[3]]
        assert idx == 3


# ---------------------------------------------------------------------------
# Tests: Cutana
# ---------------------------------------------------------------------------


class TestCutanaValidation:
    """Tests for Cutana catalogue validation."""

    def test_found_source_ids(self, cutana_csv_catalogue):
        label_df = pd.DataFrame({"id": ["SRC_0000", "SRC_0050", "SRC_0099"]})
        result = validate_labeled_files(cutana_csv_catalogue, label_df, "cutana")

        assert len(result.found) == 3
        assert result.missing == []

    def test_missing_source_ids(self, cutana_csv_catalogue):
        label_df = pd.DataFrame({"id": ["SRC_0000", "SRC_9999"]})
        result = validate_labeled_files(cutana_csv_catalogue, label_df, "cutana")

        assert result.found == ["SRC_0000"]
        assert result.missing == ["SRC_9999"]

    def test_location_has_catalogue_path_and_row(self, cutana_csv_catalogue):
        label_df = pd.DataFrame({"id": ["SRC_0010"]})
        result = validate_labeled_files(cutana_csv_catalogue, label_df, "cutana")

        cat_path, row_idx = result.found_locations["SRC_0010"]
        assert cat_path.endswith(".csv")
        assert row_idx == 10


# ---------------------------------------------------------------------------
# Tests: Edge cases and dispatch
# ---------------------------------------------------------------------------


class TestValidationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_unknown_source_type_raises(self, tmp_path):
        label_df = pd.DataFrame({"id": ["test"]})
        with pytest.raises(ValueError, match="Unknown source type"):
            validate_labeled_files(str(tmp_path), label_df, "unknown_type")

    def test_empty_zarr_directory(self, tmp_path):
        label_df = pd.DataFrame({"id": ["test__image_000000"]})
        result = validate_labeled_files(str(tmp_path), label_df, "zarr")
        assert result.found == []
        assert result.missing == ["test__image_000000"]


# ---------------------------------------------------------------------------
# Tests: Real test data
# ---------------------------------------------------------------------------


class TestWithRealTestData:
    """Tests using actual test data from the repository."""

    def test_zarr_test_data(self):
        zarr_path = os.path.join("tests", "test_data", "zarr", "test_images.zarr")
        if not os.path.exists(zarr_path):
            pytest.skip("Zarr test data not available")

        if _is_lfs_pointer(os.path.join(zarr_path, "zarr.json")):
            pytest.skip("Zarr test data is a Git LFS pointer (run git lfs pull)")

        label_df = pd.read_csv(os.path.join("tests", "test_data", "zarr", "labeled_data.csv"))
        result = validate_labeled_files(zarr_path, label_df, "zarr")

        assert len(result.found) == len(label_df)
        assert result.missing == []

    def test_cutana_test_data(self):
        cat_path = os.path.join("tests", "test_data", "cutana_catalogue")
        if not os.path.exists(cat_path):
            pytest.skip("Cutana test data not available")

        label_df = pd.read_csv(
            os.path.join("tests", "test_data", "cutana_catalogue", "labeled_data.csv")
        )
        result = validate_labeled_files(cat_path, label_df, "cutana")

        assert len(result.found) > 0
        assert len(result.missing) == 0

    def test_image_folder_test_data(self):
        img_path = os.path.join("tests", "test_data", "grayscale")
        if not os.path.exists(img_path):
            pytest.skip("Image test data not available")

        label_df = pd.read_csv(os.path.join("tests", "test_data", "grayscale", "labeled_data.csv"))
        result = validate_labeled_files(img_path, label_df, "image_folder")

        assert len(result.found) > 0
