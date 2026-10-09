#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for ImageCache."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from dotmap import DotMap
from PIL import Image

from anomaly_match.prediction.image_cache import ImageCache, _ensure_uint8_hwc


@pytest.fixture()
def dummy_cfg():
    """Minimal config for ImageCache (not used for actual loading in mocked tests)."""
    cfg = DotMap(_dynamic=False)
    cfg.normalisation = DotMap(_dynamic=False)
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    return cfg


@pytest.fixture()
def image_dir(tmp_path):
    """Create a temp directory with a few test PNG images."""
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(tmp_path / f"img_{i:03d}.png")
    return tmp_path


def _mock_load(path, cfg, desc="", show_progress=False):
    """Return a deterministic image based on the filename."""
    return np.full((64, 64, 3), fill_value=42, dtype=np.uint8)


# ── Cache hit / miss ─────────────────────────────────────────────────


class TestCacheHitMiss:
    """Verify cache hit returns same object, miss triggers load."""

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_cache_hit(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=10)
        img1 = cache.get_image("img_000.png")
        img2 = cache.get_image("img_000.png")
        assert img1 is img2
        # Should only have loaded once
        assert mock_load.call_count == 1

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_cache_miss_loads(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=10)
        img = cache.get_image("img_001.png")
        assert img is not None
        assert img.shape == (64, 64, 3)
        assert mock_load.call_count == 1


# ── LRU eviction ─────────────────────────────────────────────────────


class TestLRUEviction:
    """Cache evicts LRU entry when full."""

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_eviction_at_capacity(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=3)
        cache.get_image("img_000.png")
        cache.get_image("img_001.png")
        cache.get_image("img_002.png")
        assert cache.size == 3

        # Inserting a 4th should evict img_000 (LRU)
        cache.get_image("img_003.png")
        assert cache.size == 3
        # img_000 was evicted, accessing it again triggers a reload
        cache.get_image("img_000.png")
        # 5 loads total: 000, 001, 002, 003, 000 again
        assert mock_load.call_count == 5

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_access_promotes_to_mru(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=3)
        cache.get_image("img_000.png")
        cache.get_image("img_001.png")
        cache.get_image("img_002.png")

        # Access img_000 again to promote it (cache hit, no load)
        cache.get_image("img_000.png")
        assert mock_load.call_count == 3  # no new load

        # Now inserting img_003 should evict img_001 (the new LRU)
        cache.get_image("img_003.png")
        assert cache.size == 3

        # img_001 should have been evicted, not img_000
        cache.get_image("img_001.png")
        assert mock_load.call_count == 5  # loaded img_003 + img_001 again


# ── Missing files ────────────────────────────────────────────────────


class TestMissingFiles:
    """get_image returns None for non-existent files."""

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
    )
    def test_missing_file_returns_none(self, mock_load, dummy_cfg, tmp_path):
        cache = ImageCache(dummy_cfg, str(tmp_path), cache_size=10)
        result = cache.get_image("nonexistent.jpg")
        assert result is None
        mock_load.assert_not_called()


# ── Prefetch ─────────────────────────────────────────────────────────


class TestPrefetch:
    """Batch pre-loading."""

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_prefetch(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=10)
        cache.prefetch(["img_000.png", "img_001.png", "img_002.png"])
        assert cache.size == 3
        assert mock_load.call_count == 3

        # Subsequent gets should be cache hits
        cache.get_image("img_000.png")
        assert mock_load.call_count == 3


# ── Clear ────────────────────────────────────────────────────────────


class TestClear:
    """Cache clear drops all entries."""

    @patch(
        "anomaly_match.data_io.load_images.load_and_process_single_wrapper",
        side_effect=_mock_load,
    )
    def test_clear(self, mock_load, dummy_cfg, image_dir):
        cache = ImageCache(dummy_cfg, str(image_dir), cache_size=10)
        cache.get_image("img_000.png")
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0


# ── Format conversion helper ─────────────────────────────────────────


class TestEnsureUint8Hwc:
    """_ensure_uint8_hwc handles various input formats."""

    def test_chw_to_hwc(self):
        chw = np.zeros((3, 64, 64), dtype=np.uint8)
        result = _ensure_uint8_hwc(chw)
        assert result.shape == (64, 64, 3)

    def test_float01_to_uint8(self):
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        result = _ensure_uint8_hwc(img)
        assert result.dtype == np.uint8
        assert result.max() == 128  # skimage img_as_ubyte: round(0.5 * 255) = 128

    def test_float255_to_uint8(self):
        img = np.ones((64, 64, 3), dtype=np.float32) * 200.0
        result = _ensure_uint8_hwc(img)
        assert result.dtype == np.uint8
        assert result.max() == 200

    def test_already_uint8_hwc_passthrough(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = _ensure_uint8_hwc(img)
        assert result is img  # No copy needed
