#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for labeled_data_cache module."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import zarr
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger

import anomaly_match as am
from anomaly_match.data_io.container_loaders import (
    apply_channel_combination_to_cutana_image,
    decode_zarr_image,
)
from anomaly_match.data_io.labeled_data_cache import (
    _CACHE_FORMAT_VERSION,
    _CACHE_READ_TARGET_LINES,
    _CACHE_SHARD_SIZE,
    LABELED_CACHE_RESOLUTION,
    LabeledDataCache,
    _build_raw_extraction_cfg,
    _cache_read_log_stride,
    _compute_extraction_hash,
    _compute_label_csv_hash,
    _current_source_band_count,
    _cutana_hash_params,
    _load_catalogue_filtered,
    _serialise_fits_extension,
)
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.cutana_source import _resolve_extension_names, first_cutana_catalogue
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.normalisation_parameters import EXTRACTION_AFFECTING_NORM_FIELDS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    return tmp_path / "labeled_cache"


@pytest.fixture()
def base_cfg():
    """Minimal config for cache tests."""
    cfg = am.get_default_cfg()
    cfg.normalisation.image_size = [64, 64]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.fits_extension = None
    cfg.fitsbolt_cfg = None
    return cfg


@pytest.fixture()
def zarr_store_with_data(tmp_path):
    """Create a Zarr store with 10 images and return path + filenames."""
    store_path = tmp_path / "source.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    images = root.create_dataset(
        "images",
        shape=(10, 32, 32, 3),
        chunks=(1, 32, 32, 3),
        dtype=np.uint8,
    )
    # Fill with deterministic data so we can verify extraction
    for i in range(10):
        images[i] = np.full((32, 32, 3), i * 25, dtype=np.uint8)

    filenames = [f"source__image_{i:06d}" for i in range(10)]
    return str(store_path), filenames


@pytest.fixture()
def sample_label_df():
    """Label DataFrame for 3 labeled images."""
    return pd.DataFrame(
        {
            "id": ["source__image_000002", "source__image_000005", "source__image_000008"],
            "label": ["anomaly", "normal", "anomaly"],
        }
    )


@pytest.fixture()
def sample_found_locations(zarr_store_with_data, sample_label_df):
    """Validation result locations for the sample labels."""
    store_path, _ = zarr_store_with_data
    return {
        "source__image_000002": (store_path, 2),
        "source__image_000005": (store_path, 5),
        "source__image_000008": (store_path, 8),
    }


# ---------------------------------------------------------------------------
# Tests: LabeledDataCache build (Zarr)
# ---------------------------------------------------------------------------


class TestCacheBuildZarr:
    """Tests for building cache from Zarr sources."""

    def test_build_creates_cache_files(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        assert cache.is_populated
        assert (cache_dir / LabeledDataCache.IMAGES_ZARR).exists()
        assert (cache_dir / LabeledDataCache.METADATA_PARQUET).exists()
        assert (cache_dir / LabeledDataCache.CACHE_INFO_JSON).exists()

    def test_labels_survive_numeric_id_dtype(self, zarr_store_with_data):
        """Numeric ids (int64 from ``pd.read_csv``) must still map to labels.

        Mirrors the Cutana regression: ``found_locations`` keys ids as strings,
        so a label map keyed by the CSV's native int64 dtype misses every
        lookup and would cache an empty label per cutout."""
        store_path, _ = zarr_store_with_data
        found = {"1": (store_path, 1), "2": (store_path, 2)}
        # int64 ids, exactly as pandas infers them from a numeric CSV column.
        label_df = pd.DataFrame({"id": [1, 2], "label": ["anomaly", "normal"]})

        _images, meta = LabeledDataCache._extract_zarr_images(found, label_df)

        by_id = {row["id"]: row["label"] for row in meta}
        assert by_id == {"1": "anomaly", "2": "normal"}

    def test_build_writes_sharded_compressed_array(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Images are packed into shard files and compressed, so reading the
        set over NFS is a few file opens instead of one per cutout."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        arr = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="r")["images"]
        # Per-cutout chunk, but a shard groups multiple chunks into one file.
        assert arr.chunks == (1, *arr.shape[1:])
        # Deterministic shard size: min(_CACHE_SHARD_SIZE, n) — assert it exactly
        # so a regression that drops back to per-chunk files is caught.
        assert arr.shards == (min(_CACHE_SHARD_SIZE, arr.shape[0]), *arr.shape[1:])
        assert arr.compressors  # non-empty → compression is on
        assert cache.get_cache_info()["cache_format_version"] == _CACHE_FORMAT_VERSION

    def test_build_extracts_correct_images(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        results = cache.get_raw_images()
        assert len(results) == 3

        # Image at index 2 should have pixel value 50 (2 * 25)
        for fn, img in results:
            if fn == "source__image_000002":
                assert img[0, 0, 0] == 50
            elif fn == "source__image_000005":
                assert img[0, 0, 0] == 125
            elif fn == "source__image_000008":
                assert img[0, 0, 0] == 200

    def test_build_preserves_metadata(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        meta_df = cache.get_label_df()
        assert len(meta_df) == 3
        assert "id" in meta_df.columns
        assert "id" in meta_df.columns
        assert "label" in meta_df.columns
        assert set(meta_df["id"]) == {
            "source__image_000002",
            "source__image_000005",
            "source__image_000008",
        }

    def test_build_writes_cache_info(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        info = cache.get_cache_info()
        assert info["source_type"] == "zarr"
        assert info["num_images"] == 3
        assert info["image_shape"] == [32, 32, 3]

    def test_build_with_empty_locations(self, cache_dir, base_cfg):
        cache = LabeledDataCache(cache_dir)
        label_df = pd.DataFrame(columns=["id", "label"])
        cache.build_from_zarr({}, label_df, base_cfg)

        assert not cache.is_populated

    def test_build_clears_previous_cache(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)

        # Build twice
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert cache.get_cache_info()["num_images"] == 3

        # Build again with fewer items
        small_locs = {"source__image_000002": sample_found_locations["source__image_000002"]}
        small_df = sample_label_df[sample_label_df["id"] == "source__image_000002"]
        cache.build_from_zarr(small_locs, small_df, base_cfg)

        assert cache.get_cache_info()["num_images"] == 1


# ---------------------------------------------------------------------------
# Tests: LabeledDataCache append (Zarr)
# ---------------------------------------------------------------------------


class TestExtractionHashFieldsStayInSync:
    """The setup-screen preview decides whether its held raw cutouts survive a
    normalisation edit by reading the same constant this hash is keyed by.  If
    they drift, the zoom-out slider silently stops refreshing the preview."""

    def test_hash_params_match_the_shared_constant(self):
        cfg = get_default_cfg()
        assert set(_cutana_hash_params(cfg)) == EXTRACTION_AFFECTING_NORM_FIELDS

    def test_padding_factor_changes_the_hash(self):
        cfg = get_default_cfg()
        cfg.normalisation.cutout_padding_factor = 1.0
        before = _compute_extraction_hash(DataSourceType.CUTANA, cfg)
        cfg.normalisation.cutout_padding_factor = 2.5
        assert _compute_extraction_hash(DataSourceType.CUTANA, cfg) != before


class TestCacheAppendZarr:
    """Tests for appending to an existing Zarr cache."""

    def test_append_adds_new_images(
        self, cache_dir, zarr_store_with_data, sample_found_locations, sample_label_df, base_cfg
    ):
        store_path, _ = zarr_store_with_data
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        # Append one more image
        new_locs = {"source__image_000003": (store_path, 3)}
        new_df = pd.DataFrame(
            {
                "id": ["source__image_000003"],
                "label": ["normal"],
            }
        )
        full_df = pd.concat([sample_label_df, new_df], ignore_index=True)
        cache.append_from_zarr(new_locs, new_df, full_df)

        assert cache.get_cache_info()["num_images"] == 4
        meta_df = cache.get_label_df()
        assert len(meta_df) == 4
        assert "source__image_000003" in meta_df["id"].values

    def test_append_skips_duplicates(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        # Try to append an already-cached item
        dup_locs = {"source__image_000002": sample_found_locations["source__image_000002"]}
        dup_df = sample_label_df[sample_label_df["id"] == "source__image_000002"]
        cache.append_from_zarr(dup_locs, dup_df, sample_label_df)

        # Count should not change
        assert cache.get_cache_info()["num_images"] == 3

    def test_append_to_unpopulated_raises(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        with pytest.raises(RuntimeError, match="unpopulated"):
            cache.append_from_zarr({}, pd.DataFrame(), pd.DataFrame())


# ---------------------------------------------------------------------------
# Tests: Cache invalidation
# ---------------------------------------------------------------------------


class TestCacheInvalidation:
    """Tests for needs_rebuild detection."""

    def test_zarr_never_needs_rebuild(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        # Change normalisation method -- should NOT trigger rebuild for zarr
        base_cfg.normalisation.normalisation_method = "LOG"
        assert not cache.needs_rebuild(base_cfg)

    def test_cutana_rebuild_only_when_image_size_exceeds_stored(self, cache_dir, base_cfg):
        """Cache resize-down is fine; only rebuild when requested > stored."""
        cache = LabeledDataCache(cache_dir)
        # Manually write cache info as if built from cutana at 64 px
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "source_type": "cutana",
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash("cutana", base_cfg),
            "num_images": 1,
            "image_shape": [64, 64, 3],
        }
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        root = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="w")
        root.create_dataset("images", shape=(1, 64, 64, 3), chunks=(1, 64, 64, 3), dtype=np.uint8)
        pd.DataFrame({"id": ["test"], "label": ["anomaly"]}).to_parquet(
            cache_dir / LabeledDataCache.METADATA_PARQUET
        )

        # Same config -- no rebuild
        assert not cache.needs_rebuild(base_cfg)

        # Smaller requested size -- resize down, no rebuild
        base_cfg.normalisation.image_size = [32, 32]
        assert not cache.needs_rebuild(base_cfg)

        # Larger requested size -- rebuild (cache can't upsample)
        base_cfg.normalisation.image_size = [128, 128]
        assert cache.needs_rebuild(base_cfg)

    @pytest.mark.parametrize(
        "fits_extension",
        [None, ["VIS", "NIR-H"], "PRIMARY", ["PRIMARY"], 0, [0, 2], [99]],
    )
    def test_current_source_band_count_matches_extension_resolver(self, base_cfg, fits_extension):
        """The helper never drifts from the resolver the cache build actually uses.

        Finding 1 of the review: a hand-rolled mirror disagreed with
        ``_resolve_extension_names`` for scalar-int / ``PRIMARY`` / out-of-range
        configs, and every disagreement is a permanent rebuild loop.  The fixture
        dir also contains a stray ``labeled_data.csv`` that sorts before the real
        catalogue, so this doubles as the Finding-3 robustness check.
        """
        base_cfg.data_dir = "tests/test_data/cutana_catalogue"
        base_cfg.normalisation.fits_extension = fits_extension

        catalogue_path = first_cutana_catalogue(base_cfg.data_dir)
        assert catalogue_path.endswith("test_catalogue.csv"), "must skip the stray labels CSV"
        expected = len(_resolve_extension_names(base_cfg, catalogue_path=catalogue_path)[0])
        assert _current_source_band_count(base_cfg) == expected

    def test_current_source_band_count_unknown_for_non_catalogue_dir(self, base_cfg):
        """A non-Cutana / missing data_dir yields None (don't rebuild on this signal)."""
        for data_dir in ("tests/test_data/grayscale", "/no/such/dir", None, ""):
            base_cfg.data_dir = data_dir
            assert _current_source_band_count(base_cfg) is None

    def test_cutana_rebuild_on_source_band_count_change(self, cache_dir, base_cfg, monkeypatch):
        """Adding/removing a band invalidates the cache even if config is unchanged.

        Regression for the stale-cache bug: a 3-band cache was built, then a
        NIR-J band was added to the catalogues' ``fits_file_paths`` (config
        untouched), so ``needs_rebuild`` returned False and training stacked
        3-band labeled cutouts against a 4-band unlabeled stream.  The helper is
        covered directly above; here it is stubbed to isolate the comparison.
        """
        cache = LabeledDataCache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "source_type": "cutana",
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash("cutana", base_cfg),
            "num_images": 1,
            "image_shape": [64, 64, 3],
        }
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        root = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="w")
        root.create_dataset("images", shape=(1, 64, 64, 3), chunks=(1, 64, 64, 3), dtype=np.uint8)
        pd.DataFrame({"id": ["test"], "label": ["anomaly"]}).to_parquet(
            cache_dir / LabeledDataCache.METADATA_PARQUET
        )

        stub = "anomaly_match.data_io.labeled_data_cache._current_source_band_count"

        # Source still resolves to 3 bands -> no rebuild.
        monkeypatch.setattr(stub, lambda cfg: 3)
        assert not cache.needs_rebuild(base_cfg)

        # Source now resolves to 4 bands (NIR-J added) -> rebuild.
        monkeypatch.setattr(stub, lambda cfg: 4)
        assert cache.needs_rebuild(base_cfg)

        # Band count unknown (unreadable catalogue) -> don't force a rebuild on
        # this signal alone.
        monkeypatch.setattr(stub, lambda cfg: None)
        assert not cache.needs_rebuild(base_cfg)

    @pytest.mark.parametrize("stored_shape", [[], [64], [64, 64]])
    def test_cutana_short_image_shape_does_not_raise(
        self, cache_dir, base_cfg, monkeypatch, stored_shape
    ):
        """A cached ``image_shape`` with no channel axis must not IndexError.

        The band-count comparison indexes ``stored_shape[2]``; a cache written
        without a channel axis has nothing to compare, so it degrades to "can't
        tell" rather than crashing every ``needs_rebuild`` call.
        """
        cache = LabeledDataCache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "source_type": "cutana",
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash("cutana", base_cfg),
            "num_images": 1,
            "image_shape": stored_shape,
        }
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        root = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="w")
        root.create_dataset("images", shape=(1, 64, 64, 3), chunks=(1, 64, 64, 3), dtype=np.uint8)
        pd.DataFrame({"id": ["test"], "label": ["anomaly"]}).to_parquet(
            cache_dir / LabeledDataCache.METADATA_PARQUET
        )

        # A band count is available, but the stored shape cannot be compared to it.
        monkeypatch.setattr(
            "anomaly_match.data_io.labeled_data_cache._current_source_band_count",
            lambda cfg: 4,
        )
        assert not cache.needs_rebuild(base_cfg)

    def test_cutana_needs_rebuild_on_fits_extension_change(self, cache_dir, base_cfg):
        cache = LabeledDataCache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "source_type": "cutana",
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash("cutana", base_cfg),
            "num_images": 1,
        }
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        root = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="w")
        root.create_dataset("images", shape=(1, 32, 32, 3), chunks=(1, 32, 32, 3), dtype=np.uint8)
        pd.DataFrame({"id": ["test"], "label": ["anomaly"]}).to_parquet(
            cache_dir / LabeledDataCache.METADATA_PARQUET
        )

        assert not cache.needs_rebuild(base_cfg)

        # Change fits_extension
        base_cfg.normalisation.fits_extension = ["VIS", "NIR-H"]
        assert cache.needs_rebuild(base_cfg)

    def test_cutana_no_rebuild_on_normalisation_change(self, cache_dir, base_cfg):
        """The cache stores unnormalised data; normalisation is applied at read
        time, so changing ``normalisation_method`` / ``output_dtype`` must
        NOT invalidate the cache (the whole point of the #392 refactor)."""
        cache = LabeledDataCache(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        info = {
            "source_type": "cutana",
            "cache_format_version": _CACHE_FORMAT_VERSION,
            "extraction_hash": _compute_extraction_hash("cutana", base_cfg),
            "num_images": 1,
            "image_shape": [64, 64, 3],
        }
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        root = zarr.open_group(str(cache_dir / LabeledDataCache.IMAGES_ZARR), mode="w")
        root.create_dataset("images", shape=(1, 64, 64, 3), chunks=(1, 64, 64, 3), dtype=np.uint8)
        pd.DataFrame({"id": ["test"], "label": ["anomaly"]}).to_parquet(
            cache_dir / LabeledDataCache.METADATA_PARQUET
        )

        base_cfg.normalisation.normalisation_method = "ASINH"
        assert not cache.needs_rebuild(base_cfg)

        base_cfg.normalisation.output_dtype = np.float32
        assert not cache.needs_rebuild(base_cfg)

    def test_unpopulated_cache_needs_rebuild(self, cache_dir, base_cfg):
        cache = LabeledDataCache(cache_dir)
        assert cache.needs_rebuild(base_cfg)

    def test_old_format_version_triggers_rebuild(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """A cache written in an older on-disk format rebuilds once, even when
        every extraction parameter is unchanged."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert not cache.needs_rebuild(base_cfg)

        # Simulate a cache built before the sharded format (no/older version).
        info = cache.get_cache_info()
        info["cache_format_version"] = _CACHE_FORMAT_VERSION - 1
        with open(cache_dir / LabeledDataCache.CACHE_INFO_JSON, "w") as f:
            json.dump(info, f)
        assert cache.needs_rebuild(base_cfg)

    def test_rebuild_when_source_matched_more_ids(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Adding a catalogue → more matches → cache must rebuild.

        Regression test for the bug where dropping a third Cutana
        catalogue into ``data_dir`` produced 197 matched IDs while the
        cache still held the original 186; the old ``needs_rebuild``
        only looked at extraction params and the CSV hash (both
        unchanged) and skipped.
        """
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        cached_ids = {str(k) for k in sample_found_locations}
        # Source shows one new id on top of the cached set.
        extra_ids = cached_ids | {"new_extra_id"}
        assert cache.needs_rebuild(base_cfg, found_ids=extra_ids)
        # Same set → no rebuild needed.
        assert not cache.needs_rebuild(base_cfg, found_ids=cached_ids)

    def test_rebuild_when_source_drops_ids(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Removing a catalogue → fewer matches → cache must rebuild."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        cached_ids = {str(k) for k in sample_found_locations}
        # Drop one id from the validated set.
        dropped_one = set(list(cached_ids)[1:])
        assert cache.needs_rebuild(base_cfg, found_ids=dropped_one)


# ---------------------------------------------------------------------------
# Tests: Cache clear
# ---------------------------------------------------------------------------


class TestCacheClear:
    """Tests for clearing the cache."""

    def test_clear_removes_all_files(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert cache.is_populated

        cache.clear()
        assert not cache.is_populated

    def test_clear_on_empty_cache_is_noop(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        cache.clear()  # should not raise
        assert not cache.is_populated


# ---------------------------------------------------------------------------
# Tests: Read operations
# ---------------------------------------------------------------------------


class TestCacheRead:
    """Tests for reading from the cache."""

    def test_get_raw_images_empty_cache(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        assert cache.get_raw_images() == []

    def test_get_label_df_empty_cache(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        df = cache.get_label_df()
        assert len(df) == 0
        assert "id" in df.columns

    def test_get_cache_info_empty(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        assert cache.get_cache_info() == {}

    def test_raw_images_preserve_dtype(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        results = cache.get_raw_images()
        for _, img in results:
            assert img.dtype == np.uint8

    def test_source_type_property(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """source_type property reads from cache_info.json."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert cache.source_type == "zarr"

    def test_get_raw_images_by_id_preserves_order(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Subset read returns the requested ids in the requested order."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        # Stored order is 2, 5, 8; request in reverse.
        requested = ["source__image_000008", "source__image_000002"]
        results = cache.get_raw_images_by_id(requested)

        assert [sid for sid, _ in results] == requested
        # Image at index 2 should have pixel value 50, at index 8 value 200
        by_id = dict(results)
        assert by_id["source__image_000002"][0, 0, 0] == 50
        assert by_id["source__image_000008"][0, 0, 0] == 200

    def test_get_raw_images_by_id_logs_progress(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """The (NFS-bound) subset read emits a 'Loading labeled cache: i/N'
        heartbeat the setup-preview spinner surfaces, so it isn't silent."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        messages: list[str] = []
        sink_id = logger.add(
            lambda m: messages.append(m.record["message"]), level="INFO", format="{message}"
        )
        try:
            cache.get_raw_images_by_id(["source__image_000002", "source__image_000008"])
        finally:
            logger.remove(sink_id)

        progress = [m for m in messages if m.startswith("Loading labeled cache")]
        assert progress, f"no cache-read progress logged: {messages}"
        # Final line reports completion of all requested ids.
        assert progress[-1] == "Loading labeled cache: 2/2 cutouts"


class TestCacheReadLogStride:
    """The heartbeat throttle keeps the cache-read log bounded for any N."""

    @pytest.mark.parametrize(
        ("total", "expected_stride"),
        [
            (1, 1),
            (5, 1),
            (20, 1),
            (30, 2),  # ceil(30/20)=2 — floor//20 would wrongly give 1 here
            (40, 2),
            (100, 5),
            (1000, 50),
        ],
    )
    def test_stride_values(self, total, expected_stride):
        assert _cache_read_log_stride(total) == expected_stride

    @pytest.mark.parametrize("total", [25, 39, 100, 1000, 5000])
    def test_line_count_capped(self, total):
        """At most ~20 heartbeats regardless of N, including the 20<N<40 band."""
        stride = _cache_read_log_stride(total)
        # Replicate the emit rule: log every `stride`th cutout plus the final.
        logged = {done for done in range(1, total + 1) if done % stride == 0 or done == total}
        assert len(logged) <= _CACHE_READ_TARGET_LINES + 1

    def test_get_raw_images_by_id_raises_when_unpopulated(self, cache_dir):
        cache = LabeledDataCache(cache_dir)
        with pytest.raises(RuntimeError, match="unpopulated"):
            cache.get_raw_images_by_id(["anything"])

    def test_get_raw_images_by_id_raises_on_empty_request(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        with pytest.raises(ValueError, match="empty ids"):
            cache.get_raw_images_by_id([])

    def test_get_raw_images_by_id_raises_on_unknown_id(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Callers must pre-filter to known ids; missing entries raise KeyError."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        with pytest.raises(KeyError, match="not in labeled cache"):
            cache.get_raw_images_by_id(["source__image_000002", "nonexistent"])


# ── decode_zarr_image with channel combination ────────────────────


class TestDecodeZarrImageCC:
    """Regression: CC must not crash or double-apply for cached images."""

    def test_cc_applied_to_zarr_image(self):
        """CC is applied as post-processing by decode_zarr_image."""
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [32, 32]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.fits_extension = None
        # CC: Out1=sum, Out2=0, Out3=0
        cfg.normalisation.channel_combination = np.array(
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float
        )
        cfg = get_fitsbolt_config(cfg)

        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        result = decode_zarr_image(img, cfg)

        assert result.shape == (32, 32, 3)
        # Out1 should have signal, Out2/Out3 should be ~0
        assert result[:, :, 1].max() < 5
        assert result[:, :, 2].max() < 5

    def test_cutana_cache_read_applies_cc(self):
        """Cutana cache data (already normalised) only needs channel_combination."""
        cfg = am.get_default_cfg()
        cfg.normalisation.image_size = [32, 32]
        cfg.normalisation.n_output_channels = 3
        cfg.normalisation.fits_extension = None
        cfg.normalisation.channel_combination = np.array(
            [[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=float
        )

        # Simulate a 3-band Cutana cache entry already normalised to uint8.
        img = np.full((32, 32, 3), 100, dtype=np.uint8)
        result = apply_channel_combination_to_cutana_image(img, cfg)

        assert result.shape == (32, 32, 3)
        # Out1 collapses the bands, Out2/Out3 are zeroed by the matrix.
        assert result[:, :, 1].max() == 0
        assert result[:, :, 2].max() == 0


# ---------------------------------------------------------------------------
# Tests: Extraction hash
# ---------------------------------------------------------------------------


class TestBuildRawExtractionCfg:
    """Tests for :func:`_build_raw_extraction_cfg`."""

    def test_pins_resolution_to_cache_floor_when_user_size_smaller(self, base_cfg):
        """User 64 px → cache side = max(384, 64) = 384."""
        base_cfg.normalisation.image_size = [64, 64]
        cfg = _build_raw_extraction_cfg(base_cfg)
        assert cfg.normalisation.image_size == [LABELED_CACHE_RESOLUTION, LABELED_CACHE_RESOLUTION]

    def test_lifts_resolution_when_user_size_exceeds_floor(self, base_cfg):
        """User 512 px → cache side = max(384, 512) = 512."""
        base_cfg.normalisation.image_size = [512, 512]
        cfg = _build_raw_extraction_cfg(base_cfg)
        assert cfg.normalisation.image_size == [512, 512]

    def test_forces_conversion_only_and_float32(self, base_cfg):
        base_cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
        base_cfg.normalisation.output_dtype = np.uint8
        cfg = _build_raw_extraction_cfg(base_cfg)
        assert cfg.normalisation.normalisation_method == NormalisationMethod.CONVERSION_ONLY
        assert cfg.normalisation.output_dtype == np.float32

    def test_does_not_mutate_input_cfg(self, base_cfg):
        base_cfg.normalisation.image_size = [64, 64]
        base_cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
        before = (
            list(base_cfg.normalisation.image_size),
            base_cfg.normalisation.normalisation_method,
        )
        _build_raw_extraction_cfg(base_cfg)
        after = (
            list(base_cfg.normalisation.image_size),
            base_cfg.normalisation.normalisation_method,
        )
        assert before == after

    # Band count is kept consistent with n_output_channels so the case under
    # test is the fits_extension passthrough, not a channel-count mismatch.
    @pytest.mark.parametrize(
        ("fits_extension", "n_output_channels"),
        [(None, 3), (["VIS", "NIR_Y", "NIR_J"], 3), (["VIS"], 1), ([0, 2], 2)],
        ids=["none", "named", "single", "integer_indices"],
    )
    def test_raw_extraction_cfg_preserves_fits_extension(
        self, base_cfg, fits_extension, n_output_channels
    ):
        """The band count ``needs_rebuild`` compares is resolved from the user cfg
        but was stored from the raw-extraction cfg, so the two only agree while
        this helper leaves ``fits_extension`` alone.  If it ever starts rewriting
        it, that comparison never matches and the cache rebuilds on every call.
        """
        base_cfg.normalisation.fits_extension = fits_extension
        base_cfg.normalisation.n_output_channels = n_output_channels
        assert _build_raw_extraction_cfg(base_cfg).normalisation.fits_extension == fits_extension


class TestExtractionHash:
    """Tests for the extraction hash computation."""

    def test_zarr_hash_is_fixed(self, base_cfg):
        h1 = _compute_extraction_hash("zarr", base_cfg)
        base_cfg.normalisation.image_size = [256, 256]
        h2 = _compute_extraction_hash("zarr", base_cfg)
        assert h1 == h2 == "zarr_raw_fixed"

    def test_cutana_hash_stable_across_image_size(self, base_cfg):
        """Resolution is tracked via stored ``image_shape`` (see
        ``needs_rebuild``), not the hash — resize-down must not force a
        rebuild."""
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.image_size = [256, 256]
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 == h2

    def test_cutana_hash_changes_with_fits_extension(self, base_cfg):
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.fits_extension = ["VIS"]
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 != h2

    def test_cutana_hash_stable_across_normalisation_method(self, base_cfg):
        """Normalisation is applied at read time, so the method isn't in the
        extraction hash (the whole point of the #392 refactor)."""
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.normalisation_method = "ASINH"
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 == h2

    def test_cutana_hash_stable_across_output_dtype(self, base_cfg):
        """Output dtype is applied at read time; cache holds float32 raw."""
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.output_dtype = np.float32
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 == h2

    def test_cutana_hash_stable_across_channel_combination(self, base_cfg):
        """channel_combination is applied at read time, so changing it must NOT
        invalidate the cache — cached per-band data is reusable."""
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.channel_combination = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 == h2

    def test_cutana_hash_stable_across_n_output_channels(self, base_cfg):
        """n_output_channels is applied at read time, so changing it must NOT
        invalidate the cache."""
        h1 = _compute_extraction_hash("cutana", base_cfg)
        base_cfg.normalisation.n_output_channels = 1
        h2 = _compute_extraction_hash("cutana", base_cfg)
        assert h1 == h2


# ---------------------------------------------------------------------------
# Tests: source_type property
# ---------------------------------------------------------------------------


class TestSourceType:
    """Tests for the source_type property read path."""

    def test_source_type_empty_when_no_cache(self, cache_dir):
        """source_type returns empty string when cache_info.json is absent."""
        cache = LabeledDataCache(cache_dir)
        assert cache.source_type == ""

    def test_source_type_reads_from_cache_info(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """source_type reads from an existing cache_info.json."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert cache.source_type == "zarr"


class TestExtractCutanaProgress:
    """The Cutana cache build reports cumulative cutout progress."""

    def test_cumulative_progress_logged_per_catalogue(self):
        """Each catalogue logs an overall ``done/total`` count, not just the
        per-catalogue FITS-set noise (one Cutana call covers one catalogue)."""
        found = {
            "a": ("cat1.parquet", 0),
            "b": ("cat1.parquet", 1),
            "c": ("cat2.parquet", 0),
            "d": ("cat2.parquet", 1),
        }
        label_df = pd.DataFrame({"id": ["a", "b", "c", "d"], "label": ["anomaly"] * 4})

        def fake_load(cat_path, source_ids):
            return pd.DataFrame(
                {"SourceID": source_ids, "fits_file_paths": ["x.fits"] * len(source_ids)}
            )

        def fake_stream(cat_df, cfg):
            return [(sid, np.zeros((4, 4, 1), dtype=np.float32)) for sid in cat_df["SourceID"]]

        messages: list[str] = []
        sink_id = logger.add(
            lambda m: messages.append(m.record["message"]), level="INFO", format="{message}"
        )
        try:
            with (
                patch(
                    "anomaly_match.data_io.labeled_data_cache._load_catalogue_filtered",
                    side_effect=fake_load,
                ),
                patch(
                    "anomaly_match.data_io.labeled_data_cache._stream_cutana_cutouts",
                    side_effect=fake_stream,
                ),
            ):
                images, _meta = LabeledDataCache._extract_cutana_images(
                    found, label_df, MagicMock()
                )
        finally:
            logger.remove(sink_id)

        assert len(images) == 4
        progress = [m for m in messages if m.startswith("Extracting labeled data")]
        # One cumulative line per catalogue (advancing the done count), then a
        # final tick that reaches the total.
        assert progress[0].startswith("Extracting labeled data: 0/4 cutouts (catalogue 1/2")
        assert progress[1].startswith("Extracting labeled data: 2/4 cutouts (catalogue 2/2")
        assert progress[-1].startswith("Extracting labeled data: 4/4 cutouts")

    def test_labels_survive_numeric_id_dtype(self):
        """Numeric ids (int64 from ``pd.read_csv``) must still map to labels.

        ``found_locations`` keys ids as strings; if the label lookup keys the
        DataFrame by its native int64 id it misses every string lookup and
        stores an empty label per cutout (which later poisons the append-time
        ``label_csv_hash``).  The extraction must cast to str so the metadata
        carries the real labels."""
        found = {"101": ("cat1.parquet", 0), "202": ("cat1.parquet", 1)}
        # int64 ids, exactly as pandas infers them from a numeric CSV column.
        label_df = pd.DataFrame({"id": [101, 202], "label": ["anomaly", "normal"]})

        def fake_load(cat_path, source_ids):
            return pd.DataFrame({"SourceID": source_ids})

        def fake_stream(cat_df, cfg):
            return [(sid, np.zeros((4, 4, 1), dtype=np.float32)) for sid in cat_df["SourceID"]]

        with (
            patch(
                "anomaly_match.data_io.labeled_data_cache._load_catalogue_filtered",
                side_effect=fake_load,
            ),
            patch(
                "anomaly_match.data_io.labeled_data_cache._stream_cutana_cutouts",
                side_effect=fake_stream,
            ),
        ):
            _images, meta = LabeledDataCache._extract_cutana_images(found, label_df, MagicMock())

        by_id = {row["id"]: row["label"] for row in meta}
        assert by_id == {"101": "anomaly", "202": "normal"}


# ---------------------------------------------------------------------------
# Tests: _serialise_fits_extension edge cases
# ---------------------------------------------------------------------------


class TestSerialiseFitsExtension:
    """Tests for _serialise_fits_extension with various input types."""

    def test_none_passthrough(self):
        assert _serialise_fits_extension(None) is None

    def test_string_passthrough(self):
        assert _serialise_fits_extension("VIS") == "VIS"

    def test_int_passthrough(self):
        assert _serialise_fits_extension(1) == 1

    def test_list_recursive(self):
        result = _serialise_fits_extension(["VIS", "NIR-H"])
        assert result == ["VIS", "NIR-H"]

    def test_np_integer_converted(self):
        """numpy integer types are converted to plain int."""
        val = np.int64(42)
        result = _serialise_fits_extension(val)
        assert result == 42
        assert isinstance(result, int)

    def test_unknown_type_falls_back_to_str(self):
        """Unknown types use str() fallback."""
        result = _serialise_fits_extension(3.14)
        assert result == "3.14"


# ---------------------------------------------------------------------------
# Tests: _load_catalogue_filtered
# ---------------------------------------------------------------------------


class TestLoadCatalogueFiltered:
    """Tests for _load_catalogue_filtered helper."""

    def test_csv_catalogue(self, tmp_path):
        """Filter a CSV catalogue by SourceID."""
        cat_path = tmp_path / "catalogue.csv"
        df = pd.DataFrame(
            {
                "SourceID": ["src_1", "src_2", "src_3", "src_4"],
                "fits_file_paths": ["a.fits", "b.fits", "c.fits", "d.fits"],
            }
        )
        df.to_csv(cat_path, index=False)

        result = _load_catalogue_filtered(str(cat_path), ["src_1", "src_3"])
        assert len(result) == 2
        assert set(result["SourceID"]) == {"src_1", "src_3"}

    def test_parquet_catalogue(self, tmp_path):
        """Filter a parquet catalogue by SourceID."""
        cat_path = tmp_path / "catalogue.parquet"
        df = pd.DataFrame(
            {
                "SourceID": ["src_1", "src_2", "src_3"],
                "fits_file_paths": ["a.fits", "b.fits", "c.fits"],
            }
        )
        df.to_parquet(cat_path, index=False)

        result = _load_catalogue_filtered(str(cat_path), ["src_2"])
        assert len(result) == 1
        assert result["SourceID"].iloc[0] == "src_2"


# ---------------------------------------------------------------------------
# Tests: append_from_cutana error paths
# ---------------------------------------------------------------------------


class TestAppendCutana:
    """Tests for append_from_cutana error handling."""

    def test_append_from_cutana_unpopulated_raises(self, cache_dir):
        """append_from_cutana raises RuntimeError when cache is empty."""
        cache = LabeledDataCache(cache_dir)
        with pytest.raises(RuntimeError, match="unpopulated"):
            cache.append_from_cutana({}, pd.DataFrame(), am.get_default_cfg(), pd.DataFrame())


# ---------------------------------------------------------------------------
# Tests: label-CSV hash (issue #379)
# ---------------------------------------------------------------------------


class TestLabelCsvHash:
    """Tests for the CSV-hash check that lets the persistent cache detect
    when the label file was edited between sessions."""

    def test_hash_empty_df_raises(self):
        with pytest.raises(ValueError, match="empty DataFrame"):
            _compute_label_csv_hash(pd.DataFrame(columns=["id", "label"]))

    def test_hash_stable_on_row_reorder(self, sample_label_df):
        reordered = sample_label_df.iloc[::-1].reset_index(drop=True)
        assert _compute_label_csv_hash(sample_label_df) == _compute_label_csv_hash(reordered)

    def test_hash_changes_on_label_flip(self, sample_label_df):
        h0 = _compute_label_csv_hash(sample_label_df)
        flipped = sample_label_df.copy()
        flipped.loc[0, "label"] = "normal" if flipped.loc[0, "label"] == "anomaly" else "anomaly"
        assert h0 != _compute_label_csv_hash(flipped)

    def test_hash_changes_on_id_add(self, sample_label_df):
        h0 = _compute_label_csv_hash(sample_label_df)
        extended = pd.concat(
            [sample_label_df, pd.DataFrame({"id": ["new_id"], "label": ["normal"]})],
            ignore_index=True,
        )
        assert h0 != _compute_label_csv_hash(extended)

    def test_hash_changes_on_id_remove(self, sample_label_df):
        h0 = _compute_label_csv_hash(sample_label_df)
        removed = sample_label_df.iloc[1:].reset_index(drop=True)
        assert h0 != _compute_label_csv_hash(removed)

    def test_hash_ignored_extra_columns(self, sample_label_df):
        """Hash must be invariant to metadata columns added alongside id/label."""
        augmented = sample_label_df.assign(timestamp="2026-01-01", source="test")
        assert _compute_label_csv_hash(sample_label_df) == _compute_label_csv_hash(augmented)

    def test_cache_stores_csv_hash(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        info = cache.get_cache_info()
        assert info["label_csv_hash"] == _compute_label_csv_hash(sample_label_df)

    def test_needs_rebuild_ignores_csv_when_not_passed(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Back-compat: callers that don't have the df shouldn't get spurious rebuilds."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert not cache.needs_rebuild(base_cfg)

    def test_needs_rebuild_on_csv_mismatch(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)
        assert not cache.needs_rebuild(base_cfg, label_df=sample_label_df)

        flipped = sample_label_df.copy()
        flipped.loc[0, "label"] = "normal" if flipped.loc[0, "label"] == "anomaly" else "anomaly"
        assert cache.needs_rebuild(base_cfg, label_df=flipped)

    def test_hash_from_input_label_df_not_extraction_subset(
        self, cache_dir, sample_found_locations, sample_label_df, base_cfg
    ):
        """Regression: hashes must reflect the *input* label CSV, not the
        extraction-result metadata.  When found_locations is a strict subset
        of label_df (a real-world scenario for Cutana where some cutouts
        silently fail to extract), needs_rebuild(label_df=full_csv) must
        still return False.  Prior behaviour hashed the meta parquet →
        cache rebuilt every session even though neither CSV nor cfg
        had changed."""
        partial_locations = dict(list(sample_found_locations.items())[:-1])
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(partial_locations, sample_label_df, base_cfg)

        # Cache has 2 images (from partial_locations), but was told about 3.
        assert cache.get_cache_info()["num_images"] == 2
        # Full CSV hash should match — the build remembers what it was asked
        # to extract, not what it actually got.
        assert not cache.needs_rebuild(base_cfg, label_df=sample_label_df)

    def test_append_updates_csv_hash(
        self, cache_dir, zarr_store_with_data, sample_found_locations, sample_label_df, base_cfg
    ):
        """After append, cache_info['label_csv_hash'] must match the union of old + new."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        store_path, _ = zarr_store_with_data
        new_label_df = pd.DataFrame({"id": ["source__image_000001"], "label": ["normal"]})
        new_locations = {"source__image_000001": (store_path, 1)}
        combined = pd.concat([sample_label_df, new_label_df], ignore_index=True)
        cache.append_from_zarr(new_locations, new_label_df, combined)

        info = cache.get_cache_info()
        assert info["label_csv_hash"] == _compute_label_csv_hash(combined)
        # And needs_rebuild(combined_df) must now be False
        assert not cache.needs_rebuild(base_cfg, label_df=combined)

    def test_append_hash_tracks_full_csv_not_extraction_subset(
        self, cache_dir, zarr_store_with_data, sample_found_locations, sample_label_df, base_cfg
    ):
        """The append-time hash must reflect the whole CSV, not what was cached.

        When the CSV holds ids the source can't match (so they never enter the
        cache), the stored ``label_csv_hash`` must still equal the full-CSV
        hash that ``needs_rebuild`` recomputes — otherwise the cache is rebuilt
        from scratch on the very next session even though nothing changed."""
        cache = LabeledDataCache(cache_dir)
        cache.build_from_zarr(sample_found_locations, sample_label_df, base_cfg)

        store_path, _ = zarr_store_with_data
        new_label_df = pd.DataFrame({"id": ["source__image_000001"], "label": ["normal"]})
        new_locations = {"source__image_000001": (store_path, 1)}
        # The full CSV also carries an id the source never matched; it is part
        # of the hash but never materialises in the cache metadata.
        full_csv = pd.concat(
            [
                sample_label_df,
                new_label_df,
                pd.DataFrame({"id": ["source__image_999999"], "label": ["anomaly"]}),
            ],
            ignore_index=True,
        )
        cache.append_from_zarr(new_locations, new_label_df, full_csv)

        info = cache.get_cache_info()
        assert info["label_csv_hash"] == _compute_label_csv_hash(full_csv)
        assert not cache.needs_rebuild(base_cfg, label_df=full_csv)
