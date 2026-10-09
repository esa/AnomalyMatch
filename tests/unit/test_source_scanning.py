#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for source_scanning module."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import zarr
from fitsbolt.cfg.create_config import create_config as fb_create_cfg
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from PIL import Image

from anomaly_match.data_io.labeled_data_cache import LabeledDataCache
from anomaly_match.data_io.load_images import (
    get_fitsbolt_config as real_get_fitsbolt_config,
)
from anomaly_match.data_io.source_scanning import (
    DataSourceType,
    SourceScanResult,
    decode_raw_cutouts,
    detect_and_count_prediction_sources,
    detect_prediction_source_type,
    detect_source_channel_count,
    load_cutana_preview,
    load_preview_raw_cutouts,
    load_preview_samples,
    read_label_map,
    scan_and_count_sources,
)
from anomaly_match.utils.get_default_cfg import get_default_cfg


@pytest.fixture(autouse=True)
def _stub_get_fitsbolt_config():
    """Skip fitsbolt-config rebuild for tests that use ``MagicMock`` cfg.

    ``load_preview_samples`` rebuilds ``cfg.fitsbolt_cfg`` from
    ``cfg.normalisation`` so Cutana previews stay current with widget
    changes (regression tests in this file's
    ``TestPreviewRebuildsFitsbolt`` class cover the rebuild itself).
    The other ``load_preview_samples`` cases here mock cfg so the
    rebuild's reads of ``cfg.normalisation.*`` would either crash or
    return MagicMocks that ``fb_create_cfg`` rejects.  The pass-through
    keeps these tests focused on the per-source logic they're actually
    asserting.
    """
    with patch(
        "anomaly_match.data_io.source_scanning.get_fitsbolt_config",
        side_effect=lambda c: c,
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_folder(tmp_path):
    """Create a temp folder with dummy image files."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ["img_001.png", "img_002.png", "img_003.png"]:
        (folder / name).write_bytes(b"fake")
    return str(folder)


@pytest.fixture()
def zarr_folder(tmp_path):
    """Create a temp folder containing a .zarr store with 5 images."""
    folder = tmp_path / "data"
    folder.mkdir()
    zpath = folder / "test.zarr"
    root = zarr.open_group(str(zpath), mode="w")
    root.create_array("images", shape=(5, 32, 32, 3), dtype=np.uint8)
    return str(folder)


@pytest.fixture()
def label_csv(tmp_path):
    """Create a minimal label CSV."""
    path = tmp_path / "labels.csv"
    path.write_text("id,label\nimg_001.png,anomaly\nimg_002.png,normal\n")
    return str(path)


# ---------------------------------------------------------------------------
# scan_and_count_sources
# ---------------------------------------------------------------------------


class TestScanAndCountSources:
    def test_image_folder(self, image_folder):
        result = scan_and_count_sources(image_folder)
        assert result.source_type == DataSourceType.IMAGE_FOLDER
        assert result.count == 3
        assert len(result.image_files) == 3

    def test_zarr_folder(self, zarr_folder):
        result = scan_and_count_sources(zarr_folder)
        assert result.source_type == DataSourceType.ZARR
        assert result.count == 5
        assert result.image_files == []

    def test_single_zarr_store(self, zarr_folder):
        """Passing a .zarr path directly."""
        import os

        zpath = os.path.join(zarr_folder, "test.zarr")
        result = scan_and_count_sources(zpath)
        assert result.source_type == DataSourceType.ZARR
        assert result.count == 5

    def test_empty_folder(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        result = scan_and_count_sources(str(folder))
        assert result.source_type == DataSourceType.IMAGE_FOLDER
        assert result.count == 0
        assert result.image_files == []

    def test_cutana_folder(self, tmp_path):
        """Cutana detection delegates to _is_cutana_catalogue."""
        folder = tmp_path / "cutana"
        folder.mkdir()
        cat = folder / "cat.csv"
        cat.write_text("SourceID,fits_file_paths\n1,/a.fits\n2,/b.fits\n")

        result = scan_and_count_sources(str(folder))
        assert result.source_type == DataSourceType.CUTANA
        assert result.count == 2


# ---------------------------------------------------------------------------
# detect_prediction_source_type / detect_and_count_prediction_sources
# ---------------------------------------------------------------------------


class TestDetectPredictionSourceType:
    """A zarr store must always win over non-catalogue CSV/parquet sidecars
    (label CSVs, zarr metadata parquets), matching the training-side
    `_auto_detect_source_type` priority.
    """

    def test_zarr_wins_over_non_catalogue_csv_and_parquet_sidecars(self, tmp_path):
        zpath = tmp_path / "images.zarr"
        root = zarr.open_group(str(zpath), mode="w")
        root.create_array("images", shape=(5, 32, 32, 3), dtype=np.uint8)

        (tmp_path / "labeled_data.csv").write_text("id,label\na.png,anomaly\n")
        pd.DataFrame({"filename": ["a"], "index": [0]}).to_parquet(
            tmp_path / "images_metadata.parquet", index=False
        )

        assert detect_prediction_source_type(str(tmp_path)) == DataSourceType.ZARR
        file_type, count = detect_and_count_prediction_sources(str(tmp_path))
        assert file_type == DataSourceType.ZARR
        assert count == 5

    def test_real_zarr_fixture_detected_as_zarr_not_cutana(self):
        """`tests/test_data/zarr` ships a labeled_data.csv and a metadata
        parquet alongside the zarr store (the #611 folder) — exactly the
        layout that used to tip detection to CUTANA with count 0.
        """
        folder = os.path.normpath(
            os.path.join(os.path.dirname(__file__), os.pardir, "test_data", "zarr")
        )
        file_type, count = detect_and_count_prediction_sources(folder)
        assert file_type == DataSourceType.ZARR
        assert count == 100

    def test_any_csv_without_zarr_detected_as_cutana_even_if_malformed(self, tmp_path):
        """A malformed catalogue must still come back as Cutana so the real
        validation step downstream raises its specific error instead of
        this function silently reporting an empty image folder.
        """
        (tmp_path / "notes.csv").write_text("a,b\n1,2\n")
        assert detect_prediction_source_type(str(tmp_path)) == DataSourceType.CUTANA

    def test_real_catalogue_without_zarr_detected_as_cutana(self, tmp_path):
        pd.DataFrame({"SourceID": [1, 2], "fits_file_paths": ["/a.fits", "/b.fits"]}).to_csv(
            tmp_path / "catalogue.csv", index=False
        )
        assert detect_prediction_source_type(str(tmp_path)) == DataSourceType.CUTANA

    def test_catalogue_folder_with_fits_tile_detected_as_cutana_not_image_folder(self):
        """`tests/test_data/cutana_catalogue` holds the FITS tile the
        catalogue references alongside the catalogue CSV. A `.fits` file
        counting as an image-folder signal must not pre-empt the catalogue
        file from being found, regardless of `os.listdir` order.
        """
        folder = os.path.normpath(
            os.path.join(os.path.dirname(__file__), os.pardir, "test_data", "cutana_catalogue")
        )
        assert detect_prediction_source_type(folder) == DataSourceType.CUTANA
        file_type, count = detect_and_count_prediction_sources(folder)
        assert file_type == DataSourceType.CUTANA
        assert count == 25

    def test_missing_search_dir_defaults_to_image_folder(self, tmp_path):
        assert (
            detect_prediction_source_type(str(tmp_path / "does_not_exist"))
            == DataSourceType.IMAGE_FOLDER
        )


# ---------------------------------------------------------------------------
# read_label_map
# ---------------------------------------------------------------------------


class TestReadLabelMap:
    def test_valid_csv(self, label_csv):
        result = read_label_map(label_csv)
        assert result == {"img_001.png": "anomaly", "img_002.png": "normal"}

    def test_empty_path_raises(self):
        with pytest.raises(FileNotFoundError):
            read_label_map("")

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_label_map(str(tmp_path / "nope.csv"))

    def test_missing_columns_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("x,y\n1,2\n")
        with pytest.raises(ValueError, match="missing required column"):
            read_label_map(str(path))

    def test_null_entries_raise(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("id,label\na.png,anomaly\n,normal\nc.png,\n")
        with pytest.raises(ValueError, match="null id or label"):
            read_label_map(str(path))

    def test_extra_columns_ignored(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("id,label,extra\na.png,anomaly,foo\nb.png,normal,bar\n")
        result = read_label_map(str(path))
        assert result == {"a.png": "anomaly", "b.png": "normal"}

    def test_numeric_ids_converted_to_str(self, tmp_path):
        path = tmp_path / "labels.csv"
        path.write_text("id,label\n12345,anomaly\n67890,normal\n")
        result = read_label_map(str(path))
        assert "12345" in result
        assert "67890" in result


# ---------------------------------------------------------------------------
# load_preview_samples — image folder
# ---------------------------------------------------------------------------


class TestPreviewImageFolder:
    def test_loads_images(self, tmp_path):
        """Preview returns (name, ndarray) tuples for image files."""
        folder = tmp_path / "imgs"
        folder.mkdir()
        # Create a real tiny PNG-like file that load_and_process_single_wrapper can read
        fake_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)

        cfg = MagicMock()
        with patch(
            "anomaly_match.data_io.source_scanning.load_and_process_single_wrapper",
            return_value=fake_img,
        ):
            result = load_preview_samples(
                cfg,
                str(folder),
                DataSourceType.IMAGE_FOLDER,
                image_files=["a.png", "b.png"],
                max_items=5,
            )

        assert len(result) == 2
        assert result[0][0] == "a.png"
        assert isinstance(result[0][1], np.ndarray)

    def test_chw_transposed_to_hwc(self, tmp_path):
        """CHW images with <=4 channels are transposed to HWC."""
        chw_img = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
        cfg = MagicMock()
        with patch(
            "anomaly_match.data_io.source_scanning.load_and_process_single_wrapper",
            return_value=chw_img,
        ):
            result = load_preview_samples(
                cfg,
                str(tmp_path),
                DataSourceType.IMAGE_FOLDER,
                image_files=["a.png"],
            )
        assert result[0][1].shape == (32, 32, 3)

    def test_max_items_respected(self, tmp_path):
        fake_img = np.zeros((32, 32, 3), dtype=np.uint8)
        cfg = MagicMock()
        files = [f"img_{i}.png" for i in range(20)]
        with patch(
            "anomaly_match.data_io.source_scanning.load_and_process_single_wrapper",
            return_value=fake_img,
        ):
            result = load_preview_samples(
                cfg,
                str(tmp_path),
                DataSourceType.IMAGE_FOLDER,
                image_files=files,
                max_items=3,
            )
        assert len(result) == 3

    def test_empty_file_list(self, tmp_path):
        cfg = MagicMock()
        result = load_preview_samples(
            cfg, str(tmp_path), DataSourceType.IMAGE_FOLDER, image_files=[]
        )
        assert result == []

    def test_load_failure_skipped(self, tmp_path):
        """Files that fail to load are skipped, not raised."""
        cfg = MagicMock()
        with patch(
            "anomaly_match.data_io.source_scanning.load_and_process_single_wrapper",
            side_effect=OSError("bad file"),
        ):
            result = load_preview_samples(
                cfg,
                str(tmp_path),
                DataSourceType.IMAGE_FOLDER,
                image_files=["bad.png"],
            )
        assert result == []


# ---------------------------------------------------------------------------
# load_preview_samples — zarr
# ---------------------------------------------------------------------------


def _zarr_preview_cfg():
    """Real config: the Zarr preview now decodes through ``decode_zarr_image``."""
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [32, 32]
    return real_get_fitsbolt_config(cfg)


class TestPreviewZarr:
    def test_loads_from_zarr(self, zarr_folder):
        """With no metadata parquet, titles fall back to the generated id
        scheme (``{prefix}__image_{idx:06d}``) shared with training/
        validation -- never the raw ``store.zarr[idx]`` index."""
        result = load_preview_samples(
            _zarr_preview_cfg(), zarr_folder, DataSourceType.ZARR, max_items=3
        )
        assert len(result) == 3
        for name, img in result:
            assert name.startswith("test__image_")
            assert isinstance(img, np.ndarray)

    def test_uses_real_source_id_from_metadata(self, tmp_path):
        """When a sidecar metadata parquet exists, preview titles use the
        real source id instead of a generated or positional name."""
        folder = tmp_path / "data"
        folder.mkdir()
        zpath = folder / "cutouts.zarr"
        root = zarr.open_group(str(zpath), mode="w")
        root.create_array("images", shape=(3, 16, 16, 3), dtype=np.uint8)
        pd.DataFrame({"source_id": ["gal_001", "gal_002", "gal_003"]}).to_parquet(
            folder / "cutouts_metadata.parquet"
        )

        cfg = _zarr_preview_cfg()
        result = load_preview_samples(cfg, str(folder), DataSourceType.ZARR, max_items=3)
        names = {name for name, _ in result}
        assert names == {"gal_001", "gal_002", "gal_003"}

    def test_single_zarr_store(self, tmp_path):
        """Passing a .zarr path directly."""
        zpath = tmp_path / "direct.zarr"
        root = zarr.open_group(str(zpath), mode="w")
        root.create_array("images", shape=(2, 16, 16, 3), dtype=np.uint8)

        result = load_preview_samples(
            _zarr_preview_cfg(), str(zpath), DataSourceType.ZARR, max_items=5
        )
        assert len(result) == 2

    def test_source_ids_restricts_to_labeled_subset(self, tmp_path):
        """Passing ``source_ids`` (as the setup screen does once it has
        stratified by label) must load exactly those ids, not a random
        sample of the whole store -- unlabeled cutouts must not appear."""
        folder = tmp_path / "data"
        folder.mkdir()
        zpath = folder / "cutouts.zarr"
        root = zarr.open_group(str(zpath), mode="w")
        root.create_array("images", shape=(5, 16, 16, 3), dtype=np.uint8)
        pd.DataFrame(
            {"source_id": [f"gal_{i:03d}" for i in range(5)]},
        ).to_parquet(folder / "cutouts_metadata.parquet")

        cfg = _zarr_preview_cfg()
        result = load_preview_samples(
            cfg,
            str(folder),
            DataSourceType.ZARR,
            source_ids=["gal_001", "gal_003"],
            max_items=10,
        )
        names = [name for name, _ in result]
        assert names == ["gal_001", "gal_003"]

    def test_source_ids_empty_list_yields_no_preview(self, zarr_folder):
        """An empty (but non-``None``) id list means 'nothing labeled yet'
        -- it must not fall back to sampling the whole store."""
        cfg = _zarr_preview_cfg()
        result = load_preview_samples(
            cfg, zarr_folder, DataSourceType.ZARR, source_ids=[], max_items=5
        )
        assert result == []

    def test_training_label_cache_is_not_previewed(self, tmp_path):
        """The cache next to the label CSV holds labelled cutouts again; the
        sampling preview must show only the real store (#611)."""
        folder = tmp_path / "data"
        folder.mkdir()
        zarr.open_group(str(folder / "cutouts.zarr"), mode="w").create_array(
            "images", shape=(2, 16, 16, 3), dtype=np.uint8
        )
        cache = folder / "labeled_data_cache"
        zarr.open_group(str(cache / "images.zarr"), mode="w").create_array(
            "images", shape=(4, 16, 16, 3), dtype=np.uint8
        )
        (cache / LabeledDataCache.CACHE_INFO_JSON).write_text("{}")

        result = load_preview_samples(
            _zarr_preview_cfg(), str(folder), DataSourceType.ZARR, max_items=10
        )
        assert len(result) == 2
        assert all(name.startswith("cutouts__image_") for name, _ in result)

    def test_max_items_capped(self, zarr_folder):
        result = load_preview_samples(
            _zarr_preview_cfg(), zarr_folder, DataSourceType.ZARR, max_items=2
        )
        assert len(result) == 2


# ---------------------------------------------------------------------------
# load_preview_samples — cutana
# ---------------------------------------------------------------------------


class TestPreviewCutana:
    def test_empty_source_ids(self, tmp_path):
        cfg = MagicMock()
        result = load_preview_samples(cfg, str(tmp_path), DataSourceType.CUTANA, source_ids=[])
        assert result == []

    def test_delegates_to_load_batch_cutouts(self, tmp_path):
        from anomaly_match.prediction.cutana_loader import BatchCutoutResult

        fake_batch = BatchCutoutResult(
            cutouts={"src_1": np.zeros((32, 32, 3), dtype=np.uint8)},
        )
        cfg = MagicMock()
        with patch(
            "anomaly_match.data_io.source_scanning.load_batch_cutouts",
            return_value=fake_batch,
        ) as mock_load:
            result = load_preview_samples(
                cfg,
                str(tmp_path),
                DataSourceType.CUTANA,
                source_ids=["src_1", "src_2"],
                max_items=5,
            )
            mock_load.assert_called_once_with(
                cfg, ["src_1", "src_2"], search_dir=str(tmp_path), stale_check=None
            )
        assert len(result) == 1
        assert result[0][0] == "src_1"


# ---------------------------------------------------------------------------
# load_preview_samples — string source_type
# ---------------------------------------------------------------------------


class TestPreviewStringSourceType:
    def test_accepts_string_source_type(self, tmp_path):
        """source_type can be passed as a string value."""
        fake_img = np.zeros((32, 32, 3), dtype=np.uint8)
        cfg = MagicMock()
        with patch(
            "anomaly_match.data_io.source_scanning.load_and_process_single_wrapper",
            return_value=fake_img,
        ):
            result = load_preview_samples(
                cfg,
                str(tmp_path),
                "image_folder",
                image_files=["x.png"],
            )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# load_preview_samples — fitsbolt_cfg rebuild on every call
# ---------------------------------------------------------------------------


class TestPreviewRebuildsFitsbolt:
    """Setup-screen normalisation widget changes must reach Cutana previews.

    The Cutana decode path reads ``cfg.fitsbolt_cfg`` directly (see
    :func:`anomaly_match.data_io.container_loaders.decode_cutana_raw_images`)
    and does not refresh it from ``cfg.normalisation``.  So
    ``load_preview_samples`` is the entry point that has to rebuild
    ``cfg.fitsbolt_cfg`` before dispatching — otherwise a user toggle
    of ``normalisation_method`` (or any other non-CC field) renders a
    stale preview while the cfg's ``normalisation_method`` says
    otherwise.
    """

    def test_fitsbolt_cfg_refreshed_from_normalisation(self):
        """A preview call rebuilds cfg.fitsbolt_cfg from cfg.normalisation."""
        cfg = get_default_cfg()
        cfg.num_workers = 0
        cfg.normalisation.normalisation_method = NormalisationMethod.LOG

        # Seed a deliberately stale fitsbolt_cfg with the wrong method.
        cfg.fitsbolt_cfg = fb_create_cfg(
            output_dtype=cfg.normalisation.output_dtype,
            size=cfg.normalisation.image_size,
            fits_extension=cfg.normalisation.fits_extension,
            interpolation_order=cfg.normalisation.interpolation_order,
            normalisation_method=NormalisationMethod.CONVERSION_ONLY,
            num_workers=1,
            log_level="WARNING",
            force_dtype=True,
        )
        assert int(cfg.fitsbolt_cfg.normalisation_method) == int(
            NormalisationMethod.CONVERSION_ONLY
        )

        # Override the autouse stub with the real reconstructor for this
        # test only.  Empty input list keeps the call to the entry-point
        # rebuild only — no real decoding work.
        with patch(
            "anomaly_match.data_io.source_scanning.get_fitsbolt_config",
            wraps=real_get_fitsbolt_config,
        ) as wrapped_rebuild:
            load_preview_samples(cfg, "/tmp", DataSourceType.IMAGE_FOLDER, image_files=[])

        wrapped_rebuild.assert_called_once_with(cfg)
        # Stale CONVERSION_ONLY → LOG after the rebuild.
        assert int(cfg.fitsbolt_cfg.normalisation_method) == int(NormalisationMethod.LOG)


# ---------------------------------------------------------------------------
# SourceScanResult dataclass
# ---------------------------------------------------------------------------


class TestSourceScanResult:
    def test_defaults(self):
        r = SourceScanResult()
        assert r.source_type == DataSourceType.IMAGE_FOLDER
        assert r.count == 0
        assert r.image_files == []

    def test_custom_values(self):
        r = SourceScanResult(source_type=DataSourceType.ZARR, count=42, image_files=[])
        assert r.source_type == DataSourceType.ZARR
        assert r.count == 42


# ---------------------------------------------------------------------------
# load_preview_raw_cutouts / decode_raw_cutouts (issue #501)
# ---------------------------------------------------------------------------


def _build_populated_cache(tmp_path):
    """Build a small populated labeled cache and return (cfg, ids).

    Built from a Zarr source (cheap, no Cutana data needed); the preview
    raw-load / re-decode functions are source-type-agnostic readers, so this
    exercises them faithfully.
    """
    store_path = tmp_path / "source.zarr"
    root = zarr.open_group(str(store_path), mode="w")
    images = root.create_dataset(
        "images", shape=(6, 32, 32, 3), chunks=(1, 32, 32, 3), dtype=np.uint8
    )
    for i in range(6):
        images[i] = np.full((32, 32, 3), i * 40, dtype=np.uint8)
    ids = [f"source__image_{i:06d}" for i in range(6)]
    label_df = pd.DataFrame({"id": ids, "label": ["anomaly", "normal"] * 3})
    found = {sid: (str(store_path), i) for i, sid in enumerate(ids)}

    cache_dir = tmp_path / "labeled_cache"
    cfg = get_default_cfg()
    cfg.normalisation.image_size = [16, 16]
    cfg.normalisation.n_output_channels = 3
    cfg.normalisation.fits_extension = None
    cfg.fitsbolt_cfg = None
    LabeledDataCache(cache_dir).build_from_zarr(found, label_df, cfg)
    cfg.labeled_cache_path = str(cache_dir)
    return cfg, ids


class TestLoadPreviewRawCutouts:
    def test_returns_cache_backed_ids_in_order(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        requested = [ids[3], ids[0], ids[5]]
        raw = load_preview_raw_cutouts(cfg, requested, max_items=10)
        assert [sid for sid, _ in raw] == requested
        # Raw arrays are returned at the cache resolution (un-resized to 16).
        assert all(isinstance(arr, np.ndarray) for _, arr in raw)

    def test_subset_when_some_ids_uncached(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        raw = load_preview_raw_cutouts(cfg, [ids[1], "not_in_cache", ids[4]], max_items=10)
        assert [sid for sid, _ in raw] == [ids[1], ids[4]]

    def test_empty_when_no_cache_path(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        cfg.labeled_cache_path = None
        assert load_preview_raw_cutouts(cfg, ids, max_items=10) == []

    def test_empty_when_unpopulated(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        cfg.labeled_cache_path = str(tmp_path / "does_not_exist")
        assert load_preview_raw_cutouts(cfg, ids, max_items=10) == []

    def test_respects_max_items(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        raw = load_preview_raw_cutouts(cfg, ids, max_items=2)
        assert len(raw) == 2


class TestDecodeRawCutouts:
    @pytest.fixture(autouse=True)
    def _use_real_fitsbolt_config(self):
        """Undo the module-level pass-through stub for this class.

        ``decode_raw_cutouts`` must really rebuild ``fitsbolt_cfg`` from
        ``cfg.normalisation`` — that is exactly what makes a normalisation
        change take effect on the in-memory re-decode (#501).
        """
        with patch(
            "anomaly_match.data_io.source_scanning.get_fitsbolt_config",
            real_get_fitsbolt_config,
        ):
            yield

    def test_decodes_to_uint8_hwc_in_order(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        raw = load_preview_raw_cutouts(cfg, ids, max_items=10)
        decoded = decode_raw_cutouts(cfg, raw)
        assert [sid for sid, _ in decoded] == ids
        for _, img in decoded:
            assert img.dtype == np.uint8
            assert img.shape == (16, 16, 3)  # decoded to cfg.image_size

    def test_empty_returns_empty(self, tmp_path):
        cfg, _ = _build_populated_cache(tmp_path)
        assert decode_raw_cutouts(cfg, []) == []

    def test_reflects_normalisation_change(self, tmp_path):
        """Re-decoding the SAME raws with a different norm method changes output
        — this is what makes the in-memory re-decode reflect a norm change."""
        cfg, ids = _build_populated_cache(tmp_path)
        raw = load_preview_raw_cutouts(cfg, ids, max_items=10)

        cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
        conv = decode_raw_cutouts(cfg, raw)
        cfg.normalisation.normalisation_method = NormalisationMethod.ASINH
        asinh = decode_raw_cutouts(cfg, raw)

        assert any(not np.array_equal(a, b) for (_, a), (_, b) in zip(conv, asinh))

    def test_matches_full_preview_load(self, tmp_path):
        """In-memory re-decode equals a fresh cache read + decode for same ids."""
        cfg, ids = _build_populated_cache(tmp_path)
        raw = load_preview_raw_cutouts(cfg, ids, max_items=10)
        redecoded = decode_raw_cutouts(cfg, raw)
        full = load_preview_samples(cfg, "", DataSourceType.CUTANA, source_ids=ids, max_items=10)
        assert [s for s, _ in redecoded] == [s for s, _ in full]
        for (_, a), (_, b) in zip(redecoded, full):
            assert np.array_equal(a, b)


class TestLoadCutanaPreview:
    """`load_cutana_preview` returns decoded images + holdable raws in one read."""

    @pytest.fixture(autouse=True)
    def _use_real_fitsbolt_config(self):
        with patch(
            "anomaly_match.data_io.source_scanning.get_fitsbolt_config",
            real_get_fitsbolt_config,
        ):
            yield

    def test_full_coverage_returns_holdable_raws(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        decoded, holdable = load_cutana_preview(cfg, "", ids, max_items=10)
        assert [sid for sid, _ in decoded] == ids
        # Fully cache-backed → raws are holdable, one per id.
        assert holdable is not None
        assert [sid for sid, _ in holdable] == ids

    def test_partial_coverage_holdable_is_none(self, tmp_path):
        cfg, ids = _build_populated_cache(tmp_path)
        from anomaly_match.prediction.cutana_loader import BatchCutoutResult

        # One id the cache can't serve forces the fallback; the merged result is
        # not re-decodable, so holdable must be None.
        extra = "not_in_cache"
        fake = BatchCutoutResult(cutouts={extra: np.zeros((32, 32, 3), dtype=np.uint8)})
        with patch("anomaly_match.data_io.source_scanning.load_batch_cutouts", return_value=fake):
            decoded, holdable = load_cutana_preview(cfg, "", [*ids, extra], max_items=20)
        assert holdable is None
        assert extra in {sid for sid, _ in decoded}

    def test_empty_ids(self, tmp_path):
        cfg, _ = _build_populated_cache(tmp_path)
        assert load_cutana_preview(cfg, "", [], max_items=10) == ([], None)


class TestDetectSourceChannelCount:
    @pytest.mark.parametrize(
        "shape, expected",
        [((2, 16, 16, 3), 3), ((2, 3, 16, 16), 3), ((2, 16, 16, 1), 1), ((2, 16, 16), 1)],
        ids=["HWC", "CHW", "single-channel", "no-channel-axis"],
    )
    def test_zarr_layouts(self, tmp_path, shape, expected):
        store = tmp_path / "cutouts.zarr"
        zarr.open_group(str(store), mode="w").create_array("images", shape=shape, dtype=np.uint8)
        assert detect_source_channel_count(str(tmp_path), DataSourceType.ZARR) == expected

    def test_image_folder_samples_the_first_image(self, tmp_path):
        Image.new("L", (8, 8)).save(tmp_path / "a.png")
        assert (
            detect_source_channel_count(str(tmp_path), DataSourceType.IMAGE_FOLDER, ["a.png"]) == 1
        )

    def test_other_source_types_are_unknown(self, tmp_path):
        assert detect_source_channel_count(str(tmp_path), DataSourceType.CUTANA) is None
