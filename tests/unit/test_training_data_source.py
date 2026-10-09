#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for TrainingDataSource implementations."""

import copy
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import anomaly_match as am
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.training_data_source import (
    ImageFolderSource,
    _auto_detect_source_type,
    create_training_data_source,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def base_cfg():
    """Base config for image folder tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.normalisation.fits_extension = None
    cfg.label_file = os.path.join("tests", "test_data", "grayscale", "labeled_data.csv")
    cfg.seed = 42
    return cfg


@pytest.fixture()
def cfg(base_cfg):
    """Per-test deep copy."""
    return copy.deepcopy(base_cfg)


@pytest.fixture(scope="module")
def label_df():
    """Label DataFrame matching tests/test_data/labeled_data.csv."""
    return pd.read_csv(os.path.join("tests", "test_data", "grayscale", "labeled_data.csv"))


def _is_lfs_pointer(path: str) -> bool:
    """Return True if *path* is a Git LFS pointer, not real content."""
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as f:
        return f.read(8) == b"version "


@pytest.fixture(scope="module")
def zarr_cfg():
    """Config for Zarr source tests."""
    zarr_json = os.path.join("tests", "test_data", "zarr", "test_images.zarr", "zarr.json")
    if _is_lfs_pointer(zarr_json):
        pytest.skip("Zarr test data is a Git LFS pointer (run git lfs pull)")

    cfg = am.get_default_cfg()
    cfg.data_dir = os.path.join("tests", "test_data", "zarr", "test_images.zarr")
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.normalisation.fits_extension = None
    cfg.label_file = os.path.join("tests", "test_data", "zarr", "labeled_data.csv")
    cfg.seed = 42
    cfg.training_data_source = "zarr"
    cfg = get_fitsbolt_config(cfg)
    return cfg


@pytest.fixture(scope="module")
def zarr_label_df():
    """Label DataFrame for Zarr test data."""
    return pd.read_csv(os.path.join("tests", "test_data", "zarr", "labeled_data.csv"))


@pytest.fixture()
def cutana_cfg():
    """Config for Cutana source tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = os.path.join("tests", "test_data", "cutana_catalogue")
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.normalisation.fits_extension = None
    cfg.label_file = os.path.join("tests", "test_data", "grayscale", "labeled_data.csv")
    cfg.seed = 42
    cfg.training_data_source = "cutana"
    cfg = get_fitsbolt_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoDetectSourceType:
    """Tests for _auto_detect_source_type."""

    def test_image_folder(self):
        assert _auto_detect_source_type("/some/folder") == "image_folder"

    def test_zarr(self):
        assert _auto_detect_source_type("/data/images.zarr") == "zarr"

    def test_cutana_csv(self):
        assert _auto_detect_source_type("/data/catalogue.csv") == "cutana"


class TestCreateTrainingDataSource:
    """Tests for the factory function."""

    def test_default_creates_image_folder(self, cfg):
        source = create_training_data_source(cfg)
        assert isinstance(source, ImageFolderSource)

    def test_unknown_raises(self, cfg):
        cfg.training_data_source = "unknown_type"
        with pytest.raises(ValueError, match="Unknown training data source"):
            create_training_data_source(cfg)


# ---------------------------------------------------------------------------
# ImageFolderSource
# ---------------------------------------------------------------------------


class TestImageFolderSource:
    """Tests for ImageFolderSource."""

    def test_total_count(self, cfg):
        source = ImageFolderSource(cfg)
        assert source.get_total_count() == 19

    def test_get_labeled_images(self, cfg, label_df):
        source = ImageFolderSource(cfg)
        results = source.get_labeled_images(label_df)
        assert len(results) == 10
        for fn, img in results:
            assert isinstance(fn, str)
            assert isinstance(img, np.ndarray)
            assert img.shape == (64, 64, 3)

    def test_get_unlabeled_batch(self, cfg, label_df):
        source = ImageFolderSource(cfg)
        labeled_fns = set(label_df["id"])
        results = source.get_unlabeled_batch(5, exclude=labeled_fns)
        assert len(results) == 5
        for fn, img in results:
            assert fn not in labeled_fns
            assert img.shape == (64, 64, 3)


# ---------------------------------------------------------------------------
# ZarrSource
# ---------------------------------------------------------------------------


class TestZarrSource:
    """Tests for ZarrSource."""

    def test_total_count(self, zarr_cfg):
        from anomaly_match.datasets.zarr_source import ZarrSource

        source = ZarrSource(zarr_cfg)
        assert source.get_total_count() == 100

    def test_get_labeled_images(self, zarr_cfg, zarr_label_df):
        from anomaly_match.datasets.zarr_source import ZarrSource

        source = ZarrSource(zarr_cfg)
        results = source.get_labeled_images(zarr_label_df)
        assert len(results) == 10
        for fn, img in results:
            assert isinstance(fn, str)
            assert isinstance(img, np.ndarray)
            assert img.shape[:2] == (64, 64)

    def test_get_unlabeled_batch(self, zarr_cfg, zarr_label_df):
        from anomaly_match.datasets.zarr_source import ZarrSource

        source = ZarrSource(zarr_cfg)
        labeled_fns = set(zarr_label_df["id"])
        results = source.get_unlabeled_batch(5, exclude=labeled_fns)
        assert len(results) == 5
        for fn, img in results:
            assert fn not in labeled_fns

    def test_metadata_parquet_loaded(self, zarr_cfg):
        """Verify filenames come from the metadata parquet, not generated."""
        from anomaly_match.datasets.zarr_source import ZarrSource

        source = ZarrSource(zarr_cfg)
        assert not source._filenames[0].startswith("test_images__image_")


# ---------------------------------------------------------------------------
# CutanaSource
# ---------------------------------------------------------------------------


class TestCutanaSource:
    """Tests for CutanaSource with mocked StreamingOrchestrator."""

    def test_get_total_count(self, cutana_cfg):
        """Verify total count sums rows across all catalogue files."""
        from anomaly_match.datasets.cutana_source import CutanaSource

        with patch("anomaly_match.datasets.cutana_source.StreamingOrchestrator"):
            source = CutanaSource(cutana_cfg)
        assert source.get_total_count() == 25

    def test_lazy_unlabeled_batch_excludes_labeled_ids(self, cutana_cfg):
        """Lazy sampling forwards the exclude set so labeled ids stay out of the pool.

        Root-cause fix: lazy mode used to pass an empty exclude set, letting labeled
        sources resurface in the unlabeled stream. ``_sample_by_tiles`` loads each
        catalogue in full and filters by ``SourceID``, so exclusion is free.
        """
        from anomaly_match.datasets.cutana_source import CutanaSource

        with patch("anomaly_match.datasets.cutana_source.StreamingOrchestrator"):
            source = CutanaSource(cutana_cfg, lazy=True)

        with patch.object(source, "_sample_by_tiles", return_value=pd.DataFrame()) as mock_sample:
            result = source.get_unlabeled_batch(5, exclude={"labeled_a", "labeled_b"})

        assert result == []
        # exclude_str is the third positional arg of _sample_by_tiles.
        assert mock_sample.call_args.args[2] == {"labeled_a", "labeled_b"}

    def test_get_labeled_images_with_mock(self, cutana_cfg):
        """Verify get_labeled_images filters and streams cutouts."""
        from anomaly_match.datasets.cutana_source import CutanaSource

        with patch("anomaly_match.datasets.cutana_source.StreamingOrchestrator"):
            source = CutanaSource(cutana_cfg)

        label_df = pd.DataFrame(
            {
                "id": ["MockSource_102018211000366", "MockSource_102018211000155"],
                "label": ["anomaly", "normal"],
            }
        )

        with (
            patch.object(source, "_stream_cutouts") as mock_stream,
        ):
            mock_stream.return_value = [
                ("MockSource_102018211000366", np.zeros((64, 64, 3), dtype=np.uint8)),
                ("MockSource_102018211000155", np.zeros((64, 64, 3), dtype=np.uint8)),
            ]
            results = source.get_labeled_images(label_df)

        assert len(results) == 2
        assert results[0][0] == "MockSource_102018211000366"
