#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests: SSL_Dataset with different TrainingDataSource backends."""

import copy
import os

import numpy as np
import pandas as pd
import pytest
import zarr

import anomaly_match as am
from anomaly_match.data_io.labeled_data_cache import LabeledDataCache
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.AnomalyDetectionDataset import AnomalyDetectionDataset
from anomaly_match.datasets.cutana_source import CutanaSource
from anomaly_match.datasets.SSL_Dataset import SSL_Dataset
from anomaly_match.datasets.training_data_source import ImageFolderSource
from anomaly_match.datasets.zarr_source import ZarrSource

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def folder_cfg():
    """Config for image-folder tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = "tests/test_data/grayscale/"
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    cfg.normalisation.fits_extension = None
    cfg.label_file = None
    cfg.seed = 42
    return cfg


@pytest.fixture(scope="module")
def zarr_cfg(tmp_path_factory):
    """Config for Zarr source.

    The fixture generates a fresh ``.zarr`` store plus a labels CSV inside
    a module-scoped temp directory.  Previously this pointed at a
    Git-LFS-tracked fixture under ``tests/test_data/zarr/`` which left
    developers without ``git lfs pull`` staring at opaque JSON decode
    errors from ``zarr.open_group``.
    """
    zarr_dir = tmp_path_factory.mktemp("zarr_fixture")
    zarr_path = zarr_dir / "test_images.zarr"

    n_images = 20
    img_shape = (32, 32, 3)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(n_images, *img_shape), dtype=np.uint8)

    root = zarr.open_group(str(zarr_path), mode="w")
    zarr_images = root.create_dataset(
        "images",
        shape=images.shape,
        chunks=(1, *img_shape),
        dtype=images.dtype,
    )
    zarr_images[...] = images

    filenames = [f"img_{i:03d}.png" for i in range(n_images)]
    metadata_path = zarr_dir / "test_images_metadata.parquet"
    pd.DataFrame({"filename": filenames}).to_parquet(metadata_path, index=False)

    labels_df = pd.DataFrame(
        {
            "id": filenames[:8],
            "label": ["anomaly"] * 4 + ["normal"] * 4,
        }
    )
    label_path = zarr_dir / "labeled_data.csv"
    labels_df.to_csv(label_path, index=False)

    cfg = am.get_default_cfg()
    cfg.data_dir = str(zarr_path)
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    cfg.normalisation.fits_extension = None
    cfg.label_file = str(label_path)
    cfg.seed = 42
    cfg.training_data_source = "zarr"
    cfg = get_fitsbolt_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify existing code paths still work without data_source parameter."""

    def test_anomaly_detection_dataset_no_source(self, folder_cfg):
        cfg = copy.deepcopy(folder_cfg)
        dataset = AnomalyDetectionDataset(cfg)
        assert len(dataset.data_dict) > 0
        assert dataset.split_indices is not None

    def test_ssl_dataset_no_source(self, folder_cfg):
        cfg = copy.deepcopy(folder_cfg)
        ssl = SSL_Dataset(cfg=cfg, train=True)
        lb, ulb = ssl.get_ssl_dset()
        assert len(lb) > 0
        assert len(ulb) > 0


# ---------------------------------------------------------------------------
# ImageFolderSource integration
# ---------------------------------------------------------------------------


class TestSSLDatasetWithImageFolder:
    """SSL_Dataset with explicit ImageFolderSource."""

    def test_ssl_dset_produces_labeled_and_unlabeled(self, folder_cfg):
        cfg = copy.deepcopy(folder_cfg)
        source = ImageFolderSource(cfg)
        ssl = SSL_Dataset(cfg=cfg, train=True, data_source=source)
        lb, ulb = ssl.get_ssl_dset()
        assert len(lb) > 0
        assert len(ulb) > 0

    def test_eval_dset(self, folder_cfg):
        cfg = copy.deepcopy(folder_cfg)
        source = ImageFolderSource(cfg)
        ssl = SSL_Dataset(cfg=cfg, train=False, data_source=source)
        dset = ssl.get_dset()
        assert len(dset) > 0


# ---------------------------------------------------------------------------
# ZarrSource integration
# ---------------------------------------------------------------------------


class TestSSLDatasetWithZarr:
    """SSL_Dataset with ZarrSource."""

    def test_ssl_dset(self, zarr_cfg):
        cfg = copy.deepcopy(zarr_cfg)
        source = ZarrSource(cfg)
        ssl = SSL_Dataset(cfg=cfg, train=True, data_source=source)
        lb, ulb = ssl.get_ssl_dset()
        assert len(lb) > 0
        assert len(ulb) > 0

    def test_eval_dset(self, zarr_cfg):
        cfg = copy.deepcopy(zarr_cfg)
        source = ZarrSource(cfg)
        ssl = SSL_Dataset(cfg=cfg, train=False, data_source=source)
        dset = ssl.get_dset()
        assert len(dset) > 0


# ---------------------------------------------------------------------------
# CutanaSource integration
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cutana_cfg():
    """Config for Cutana source integration tests."""
    cfg = am.get_default_cfg()
    cfg.data_dir = os.path.join("tests", "test_data", "cutana_catalogue")
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.N_to_load = 10
    cfg.num_train_iter = 2
    cfg.test_ratio = 0.5
    cfg.normalisation.fits_extension = None
    cfg.label_file = os.path.join("tests", "test_data", "cutana_catalogue", "labeled_data.csv")
    cfg.seed = 42
    cfg.training_data_source = "cutana"
    cfg = get_fitsbolt_config(cfg)
    return cfg


class TestCutanaSourceIntegration:
    """CutanaSource end-to-end with committed FITS test data."""

    def test_get_labeled_images(self, cutana_cfg):
        """Verify CutanaSource streams real cutouts for labeled sources."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg)
        label_df = pd.read_csv(cfg.label_file)
        results = source.get_labeled_images(label_df)
        assert len(results) == 10
        for fn, img in results:
            assert isinstance(fn, str)
            assert isinstance(img, np.ndarray)
            assert img.shape[0] == 64
            assert img.shape[1] == 64

    def test_get_unlabeled_batch(self, cutana_cfg):
        """Verify CutanaSource streams real unlabeled cutouts."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg)
        label_df = pd.read_csv(cfg.label_file)
        labeled_fns = set(label_df["id"])
        results = source.get_unlabeled_batch(5, exclude=labeled_fns)
        assert 0 < len(results) <= 5
        for fn, img in results:
            assert fn not in labeled_fns
            assert isinstance(img, np.ndarray)

    def test_ssl_dset(self, cutana_cfg):
        """Verify SSL_Dataset works end-to-end with CutanaSource."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg)
        ssl = SSL_Dataset(cfg=cfg, train=True, data_source=source)
        lb, ulb = ssl.get_ssl_dset()
        assert len(lb) > 0
        assert len(ulb) > 0

    def test_lazy_mode_get_labeled_raises(self, cutana_cfg):
        """get_labeled_images not available in lazy mode."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg, lazy=True)
        label_df = pd.read_csv(cfg.label_file)
        with pytest.raises(RuntimeError, match="lazy mode"):
            source.get_labeled_images(label_df)

    def test_lazy_mode_init(self, cutana_cfg):
        """Lazy mode skips O(N) permutation and stores total count."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg, lazy=True)
        assert source._lazy
        assert source.get_total_count() > 0

    def test_detect_num_channels(self, cutana_cfg):
        """detect_num_channels returns configured n_output_channels."""
        cfg = copy.deepcopy(cutana_cfg)
        source = CutanaSource(cfg)
        assert source.detect_num_channels() == 3

    def test_no_catalogues_raises(self, tmp_path):
        """ValueError when no catalogue files are found."""
        cfg = am.get_default_cfg()
        cfg.data_dir = str(tmp_path)
        cfg.normalisation.image_size = [64, 64]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.fits_extension = None
        cfg.seed = 42
        cfg = get_fitsbolt_config(cfg)
        with pytest.raises(ValueError, match="No catalogue files"):
            CutanaSource(cfg)


# ---------------------------------------------------------------------------
# LabeledDataCache integration: Cutana build / append
# ---------------------------------------------------------------------------


class TestLabeledDataCacheCutana:
    """Integration tests for LabeledDataCache with real Cutana FITS data."""

    def test_build_from_cutana(self, cutana_cfg, tmp_path):
        """Build a labeled cache from Cutana catalogue + FITS cutouts."""
        cfg = copy.deepcopy(cutana_cfg)
        label_df = pd.read_csv(cfg.label_file)
        cat_path = os.path.join(cfg.data_dir, "test_catalogue.csv")

        # Build found_locations: source_id -> (catalogue_path, row_index)
        cat_df = pd.read_csv(cat_path)
        found_locations = {}
        for _, row in label_df.iterrows():
            sid = str(row["id"])
            match = cat_df.index[cat_df["SourceID"] == sid]
            if len(match) > 0:
                found_locations[sid] = (cat_path, int(match[0]))

        cache = LabeledDataCache(tmp_path / "cutana_cache")
        cache.build_from_cutana(found_locations, label_df, cfg)

        assert cache.is_populated
        assert cache.source_type == "cutana"
        info = cache.get_cache_info()
        assert info["num_images"] == len(found_locations)
        assert info["source_type"] == "cutana"

        raw_images = cache.get_raw_images()
        assert len(raw_images) == len(found_locations)
        for fn, img in raw_images:
            assert isinstance(fn, str)
            assert isinstance(img, np.ndarray)

    def test_build_from_cutana_empty_locations(self, cutana_cfg, tmp_path):
        """build_from_cutana with no locations produces empty cache."""
        cfg = copy.deepcopy(cutana_cfg)
        cache = LabeledDataCache(tmp_path / "empty_cache")
        cache.build_from_cutana({}, pd.DataFrame(columns=["id", "label"]), cfg)
        assert not cache.is_populated

    def test_append_from_cutana(self, cutana_cfg, tmp_path):
        """Append new Cutana cutouts to an existing cache."""
        cfg = copy.deepcopy(cutana_cfg)
        label_df = pd.read_csv(cfg.label_file)
        cat_path = os.path.join(cfg.data_dir, "test_catalogue.csv")
        cat_df = pd.read_csv(cat_path)

        # Build with first 5 labels
        first_5 = label_df.head(5)
        first_locs = {}
        for _, row in first_5.iterrows():
            sid = str(row["id"])
            match = cat_df.index[cat_df["SourceID"] == sid]
            if len(match) > 0:
                first_locs[sid] = (cat_path, int(match[0]))

        cache = LabeledDataCache(tmp_path / "append_cache")
        cache.build_from_cutana(first_locs, first_5, cfg)
        assert cache.get_cache_info()["num_images"] == len(first_locs)

        # Append remaining labels
        rest = label_df.tail(5)
        rest_locs = {}
        for _, row in rest.iterrows():
            sid = str(row["id"])
            match = cat_df.index[cat_df["SourceID"] == sid]
            if len(match) > 0:
                rest_locs[sid] = (cat_path, int(match[0]))

        cache.append_from_cutana(rest_locs, rest, cfg, label_df)
        assert cache.get_cache_info()["num_images"] == len(first_locs) + len(rest_locs)
