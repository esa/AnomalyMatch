#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for zarr_source module."""

import numpy as np
import pandas as pd
import pytest
import zarr
from dotmap import DotMap

import anomaly_match as am
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.zarr_source import ZarrSource

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_zarr_cfg(data_dir: str) -> DotMap:
    """Build a minimal config pointing at *data_dir*."""
    cfg = am.get_default_cfg()
    cfg.data_dir = data_dir
    cfg.normalisation.image_size = [32, 32]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.fits_extension = None
    cfg.seed = 42
    cfg = get_fitsbolt_config(cfg)
    return cfg


def _create_zarr_store(path, *, n_images=5, shape=(32, 32, 3), dtype=np.uint8):
    """Create a simple zarr store with an 'images' array."""
    root = zarr.open_group(str(path), mode="w")
    arr = root.create_dataset(
        "images",
        shape=(n_images, *shape),
        chunks=(1, *shape),
        dtype=dtype,
    )
    for i in range(n_images):
        arr[i] = np.full(shape, i * 10, dtype=dtype)
    return root


# ---------------------------------------------------------------------------
# Tests: lazy mode
# ---------------------------------------------------------------------------


class TestZarrSourceLazyMode:
    """Lazy mode skips O(N) indexing and uses random sampling."""

    def test_lazy_init_no_filename_index(self, tmp_path):
        """In lazy mode, _name_to_loc and _filenames are None."""
        store_path = tmp_path / "data.zarr"
        _create_zarr_store(store_path, n_images=5)
        cfg = _make_zarr_cfg(str(store_path))

        source = ZarrSource(cfg, lazy=True)

        assert source._name_to_loc is None
        assert source._filenames is None
        assert source._shuffled_indices is None
        assert source.get_total_count() == 5

    def test_lazy_get_labeled_images_raises(self, tmp_path):
        """get_labeled_images() is not available in lazy mode."""
        store_path = tmp_path / "data.zarr"
        _create_zarr_store(store_path, n_images=3)
        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg, lazy=True)

        label_df = pd.DataFrame({"id": ["data__image_000000"], "label": ["anomaly"]})
        with pytest.raises(RuntimeError, match="lazy mode"):
            source.get_labeled_images(label_df)

    def test_lazy_get_unlabeled_batch(self, tmp_path):
        """Lazy mode samples random images without pre-built permutation."""
        store_path = tmp_path / "data.zarr"
        _create_zarr_store(store_path, n_images=10)
        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg, lazy=True)

        results = source.get_unlabeled_batch(3)
        assert len(results) == 3
        for fn, img in results:
            assert isinstance(fn, str)
            assert img.shape[:2] == (32, 32)

    def test_lazy_get_unlabeled_batch_with_exclude_positions(self, tmp_path):
        """Lazy mode excludes specific (store_idx, local_idx) positions."""
        store_path = tmp_path / "data.zarr"
        _create_zarr_store(store_path, n_images=5)
        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg, lazy=True)

        # Exclude all positions to get empty result
        exclude = {(0, i) for i in range(5)}
        results = source.get_unlabeled_batch(3, exclude_positions=exclude)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Tests: multi-store directory
# ---------------------------------------------------------------------------


class TestZarrSourceMultiStore:
    """Parent directory containing multiple .zarr children."""

    def test_multi_store_discovery(self, tmp_path):
        """Discovers multiple .zarr stores in a parent directory."""
        parent = tmp_path / "stores"
        parent.mkdir()
        _create_zarr_store(parent / "batch_a.zarr", n_images=3)
        _create_zarr_store(parent / "batch_b.zarr", n_images=4)
        cfg = _make_zarr_cfg(str(parent))

        source = ZarrSource(cfg)

        assert source.get_total_count() == 7
        assert len(source._stores) == 2

    def test_multi_store_get_unlabeled_batch(self, tmp_path):
        """Unlabeled batch works across multiple stores."""
        parent = tmp_path / "stores"
        parent.mkdir()
        _create_zarr_store(parent / "a.zarr", n_images=3)
        _create_zarr_store(parent / "b.zarr", n_images=3)
        cfg = _make_zarr_cfg(str(parent))
        source = ZarrSource(cfg)

        results = source.get_unlabeled_batch(5)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Tests: error paths and edge cases
# ---------------------------------------------------------------------------


class TestZarrSourceEdgeCases:
    """Error handling and edge cases."""

    def test_no_stores_raises(self, tmp_path):
        """ValueError when no Zarr stores are found."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        cfg = _make_zarr_cfg(str(empty_dir))

        with pytest.raises(ValueError, match="No Zarr stores"):
            ZarrSource(cfg)

    def test_store_without_images_array_skipped(self, tmp_path):
        """Zarr store without 'images' array is skipped."""
        parent = tmp_path / "stores"
        parent.mkdir()

        # Valid store
        _create_zarr_store(parent / "good.zarr", n_images=3)

        # Invalid store: no 'images' array
        bad_path = parent / "bad.zarr"
        root = zarr.open_group(str(bad_path), mode="w")
        root.create_dataset("other_data", shape=(5,), dtype=np.float32)

        cfg = _make_zarr_cfg(str(parent))
        source = ZarrSource(cfg)

        assert source.get_total_count() == 3
        assert len(source._stores) == 1

    def test_detect_num_channels_4d(self, tmp_path):
        """detect_num_channels infers from 4D array shape."""
        store_path = tmp_path / "data.zarr"
        _create_zarr_store(store_path, n_images=2, shape=(32, 32, 3))
        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)

        # Shape is (N=2, H=32, W=32, C=3), so channels = min(32, 3) = 3
        assert source.detect_num_channels() == 3

    def test_detect_num_channels_non_4d(self, tmp_path):
        """detect_num_channels returns None for non-4D arrays."""
        store_path = tmp_path / "data.zarr"
        # Create a 2D image store (grayscale, no channel dim)
        root = zarr.open_group(str(store_path), mode="w")
        root.create_dataset("images", shape=(2, 32, 32), chunks=(1, 32, 32), dtype=np.uint8)

        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)
        assert source.detect_num_channels() is None

    def test_detect_num_channels_empty(self, tmp_path):
        """detect_num_channels returns None for empty stores."""
        store_path = tmp_path / "data.zarr"
        root = zarr.open_group(str(store_path), mode="w")
        root.create_dataset("images", shape=(0, 32, 32, 3), chunks=(1, 32, 32, 3), dtype=np.uint8)

        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)
        assert source.detect_num_channels() is None


# ---------------------------------------------------------------------------
# Tests: metadata / filename loading
# ---------------------------------------------------------------------------


class TestZarrSourceMetadata:
    """Filename loading from various metadata sources."""

    def test_metadata_from_zarr_attrs(self, tmp_path):
        """Filenames loaded from metadata file referenced in zarr attrs."""
        store_path = tmp_path / "data.zarr"
        root = _create_zarr_store(store_path, n_images=3)

        # Write metadata parquet with original_filename column
        meta_path = tmp_path / "my_metadata.parquet"
        pd.DataFrame({"original_filename": ["img_a.png", "img_b.png", "img_c.png"]}).to_parquet(
            meta_path
        )

        # Point zarr attrs to the metadata file (relative path)
        root.attrs["metadata_file"] = str(meta_path)

        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)

        assert source._filenames == ["img_a.png", "img_b.png", "img_c.png"]

    def test_metadata_from_images_metadata_parquet(self, tmp_path):
        """Fallback: images_metadata.parquet next to images.zarr."""
        # Create the "batch folder" layout: <parent>/images.zarr
        batch_dir = tmp_path / "batch01"
        batch_dir.mkdir()
        store_path = batch_dir / "images.zarr"
        _create_zarr_store(store_path, n_images=2)

        # Write images_metadata.parquet next to images.zarr
        pd.DataFrame({"filename": ["file_x.fits", "file_y.fits"]}).to_parquet(
            batch_dir / "images_metadata.parquet"
        )

        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)

        assert source._filenames == ["file_x.fits", "file_y.fits"]

    def test_generated_filenames_fallback(self, tmp_path):
        """Filenames generated from indices when no metadata exists."""
        store_path = tmp_path / "mystore.zarr"
        _create_zarr_store(store_path, n_images=3)
        cfg = _make_zarr_cfg(str(store_path))
        source = ZarrSource(cfg)

        assert source._filenames == [
            "mystore__image_000000",
            "mystore__image_000001",
            "mystore__image_000002",
        ]
