#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for anomaly_match.prediction.cutana_loader.

Tests the helper functions and early-return paths that don't require
a running Cutana orchestrator or real FITS data.
"""

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from dotmap import DotMap

from anomaly_match.prediction import cutana_loader
from anomaly_match.prediction.anomaly_score_db import AnomalyScoreDB
from anomaly_match.prediction.cutana_loader import (
    _create_batch_cutouts,
    _create_cutout,
    _find_id_column,
    _find_source_row,
    _postprocess_image,
    load_batch_cutouts,
    load_sample_cutouts,
    load_single_cutout,
    load_single_native_cutout,
)
from anomaly_match.prediction.db_location import prediction_db_path

MODULE = "anomaly_match.prediction.cutana_loader"

# ── _find_id_column ─────────────────────────────────────────────


class TestFindIdColumn:
    def test_finds_SourceID(self):
        df = pd.DataFrame({"SourceID": [1], "RA": [0.0]})
        assert _find_id_column(df) == "SourceID"

    def test_finds_source_id(self):
        df = pd.DataFrame({"source_id": [1], "RA": [0.0]})
        assert _find_id_column(df) == "source_id"

    def test_finds_SOURCEID(self):
        df = pd.DataFrame({"SOURCEID": [1], "RA": [0.0]})
        assert _find_id_column(df) == "SOURCEID"

    def test_raises_when_missing(self):
        df = pd.DataFrame({"name": [1], "RA": [0.0]})
        with pytest.raises(ValueError, match="No source ID column"):
            _find_id_column(df)

    def test_prefers_first_match(self):
        """SourceID is preferred over source_id (checked first)."""
        df = pd.DataFrame({"SourceID": [1], "source_id": [2]})
        assert _find_id_column(df) == "SourceID"


# ── _find_source_row ────────────────────────────────────────────


class TestFindSourceRow:
    def test_finds_in_csv(self, tmp_path):
        csv = tmp_path / "cat.csv"
        pd.DataFrame(
            {"SourceID": ["src_001", "src_002"], "RA": [1.0, 2.0], "Dec": [3.0, 4.0]}
        ).to_csv(csv, index=False)

        result = _find_source_row(str(csv), "src_002")
        assert result is not None
        assert len(result) == 1
        assert result.iloc[0]["SourceID"] == "src_002"

    def test_finds_in_parquet(self, tmp_path):
        pq = tmp_path / "cat.parquet"
        pd.DataFrame(
            {"SourceID": ["src_001", "src_002"], "RA": [1.0, 2.0], "Dec": [3.0, 4.0]}
        ).to_parquet(pq, index=False)

        result = _find_source_row(str(pq), "src_001")
        assert result is not None
        assert result.iloc[0]["SourceID"] == "src_001"

    def test_returns_none_when_not_found(self, tmp_path):
        csv = tmp_path / "cat.csv"
        pd.DataFrame({"SourceID": ["src_001"], "RA": [1.0], "Dec": [3.0]}).to_csv(csv, index=False)

        result = _find_source_row(str(csv), "nonexistent")
        assert result is None

    def test_returns_none_when_no_id_column(self, tmp_path):
        csv = tmp_path / "cat.csv"
        pd.DataFrame({"name": ["foo"], "val": [1.0]}).to_csv(csv, index=False)

        result = _find_source_row(str(csv), "foo")
        assert result is None

    def test_matches_numeric_id_as_string(self, tmp_path):
        csv = tmp_path / "cat.csv"
        pd.DataFrame({"SourceID": [12345, 67890], "RA": [1.0, 2.0], "Dec": [3.0, 4.0]}).to_csv(
            csv, index=False
        )

        result = _find_source_row(str(csv), "12345")
        assert result is not None


# ── _postprocess_image ──────────────────────────────────────────


class TestPostprocessImage:
    def test_converts_hwc_uint8(self):
        image = np.random.rand(64, 64, 3).astype(np.float32)
        result = _postprocess_image(image)
        assert result is not None
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_converts_chw_to_hwc(self):
        image = np.random.rand(3, 64, 64).astype(np.float32)
        result = _postprocess_image(image)
        assert result is not None
        assert result.shape == (64, 64, 3)

    def test_handles_grayscale(self):
        image = np.random.rand(64, 64).astype(np.float32)
        result = _postprocess_image(image)
        assert result is not None
        assert result.dtype == np.uint8


# ── load_single_cutout early returns ─────────────────────────────


class TestLoadSingleCutoutEarlyReturns:
    def test_returns_none_no_catalogue_files(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        cfg = DotMap(prediction_search_dir=str(d))
        assert load_single_cutout(cfg, "src_001") is None

    def test_returns_none_source_not_in_catalogue(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {"SourceID": ["other"], "RA": [1.0], "Dec": [2.0], "fits_file_paths": ["x.fits"]}
        ).to_csv(d / "cat.csv", index=False)

        cfg = DotMap(prediction_search_dir=str(d))
        assert load_single_cutout(cfg, "nonexistent") is None


# ── load_batch_cutouts early returns ─────────────────────────────


class TestLoadBatchCutoutsEarlyReturns:
    def test_returns_empty_no_catalogues(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["src_001"])
        assert result.cutouts == {}
        assert result.partial is False

    def test_returns_empty_when_no_matches(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {"SourceID": ["other"], "RA": [1.0], "Dec": [2.0], "fits_file_paths": ["x.fits"]}
        ).to_csv(d / "cat.csv", index=False)

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["nonexistent"])
        assert result.cutouts == {}
        assert result.partial is False


# ── load_sample_cutouts early returns ────────────────────────────


class TestLoadSampleCutoutsEarlyReturns:
    def test_returns_empty_no_catalogues(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        cfg = DotMap(_dynamic=False)
        assert load_sample_cutouts(cfg, str(d)) == []

    def test_returns_empty_for_empty_catalogue(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(columns=["SourceID", "RA", "Dec", "fits_file_paths"]).to_csv(
            d / "cat.csv", index=False
        )

        cfg = DotMap(_dynamic=False)
        assert load_sample_cutouts(cfg, str(d)) == []


# ── _create_batch_cutouts ────────────────────────────────────────


def _fake_image(h=64, w=64, c=3):
    return np.random.randint(0, 255, (h, w, c), dtype=np.uint8).astype(np.float32) / 255.0


def _minimal_cfg(n_output_channels: int = 3) -> DotMap:
    """Build a minimal cfg sufficient for ``apply_channel_combination_to_cutana_image``."""
    cfg = DotMap(_dynamic=False)
    cfg.normalisation = DotMap(_dynamic=False)
    cfg.normalisation.n_output_channels = n_output_channels
    cfg.normalisation.channel_combination = None
    cfg.normalisation.output_dtype = np.uint8
    cfg.fitsbolt_cfg = DotMap(_dynamic=False)
    return cfg


def _batch(cutouts, source_ids):
    """Build a cutana-style batch dict as emitted by create_cutouts_direct."""
    return {
        "cutouts": cutouts,
        "metadata": [{"source_id": sid} for sid in source_ids],
    }


class TestCreateBatchCutouts:
    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_returns_dict_keyed_by_source_id(self, mock_extract):
        img1 = _fake_image()
        img2 = _fake_image()
        mock_extract.return_value = [_batch(np.stack([img1, img2]), ["s1", "s2"])]

        batch_df = pd.DataFrame({"SourceID": ["s1", "s2"], "RA": [1.0, 2.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert set(result.keys()) == {"s1", "s2"}
        assert result["s1"].dtype == np.uint8
        assert result["s2"].dtype == np.uint8

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_delegates_to_direct_path(self, mock_extract):
        mock_extract.return_value = [_batch(np.stack([_fake_image()]), ["s1"])]

        batch_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        mock_extract.assert_called_once()

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_handles_multiple_batches(self, mock_extract):
        """Direct path can split sources across multiple FITS-set batches."""
        mock_extract.return_value = [
            _batch(np.stack([_fake_image()]), ["s1"]),
            _batch(np.stack([_fake_image()]), ["s2"]),
        ]

        batch_df = pd.DataFrame({"SourceID": ["s1", "s2"], "RA": [1.0, 2.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert set(result.keys()) == {"s1", "s2"}

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_handles_empty_result(self, mock_extract):
        mock_extract.return_value = []

        batch_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert result == {}

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_rejects_empty_cutout_images(self, mock_extract):
        """Empty/invalid cutouts are a Cutana contract violation — the
        function logs and returns ``{}`` instead of partially fulfilling the
        request."""
        good_img = _fake_image()
        empty_img = np.array([])
        mock_extract.return_value = [
            _batch(np.array([good_img, empty_img], dtype=object), ["s1", "s2"])
        ]

        batch_df = pd.DataFrame({"SourceID": ["s1", "s2"], "RA": [1.0, 2.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert result == {}

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_returns_empty_on_exception(self, mock_extract):
        mock_extract.side_effect = RuntimeError("direct extract failed")

        batch_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert result == {}

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_rejects_unexpected_source_ids(self, mock_extract):
        """Cutouts for source ids not in *batch_df* are a Cutana contract
        violation — the function logs and returns ``{}`` instead of silently
        dropping the stranger entries."""
        mock_extract.return_value = [
            _batch(np.stack([_fake_image(), _fake_image()]), ["s1", "stranger"])
        ]

        batch_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_batch_cutouts(_minimal_cfg(), batch_df, "SourceID")

        assert result == {}


# ── _create_cutout ───────────────────────────────────────────────


class TestCreateCutout:
    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_returns_processed_image(self, mock_extract):
        img = _fake_image()
        mock_extract.return_value = [_batch(np.stack([img]), ["s1"])]

        source_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_cutout(_minimal_cfg(), source_df)

        assert result is not None
        assert result.dtype == np.uint8

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_returns_none_for_empty_batches(self, mock_extract):
        mock_extract.return_value = []

        source_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_cutout(_minimal_cfg(), source_df)

        assert result is None

    @patch(f"{MODULE}.extract_cutouts_direct")
    def test_returns_none_on_exception(self, mock_extract):
        mock_extract.side_effect = RuntimeError("direct extract failed")

        source_df = pd.DataFrame({"SourceID": ["s1"], "RA": [1.0]})
        result = _create_cutout(_minimal_cfg(), source_df)

        assert result is None


# ── load_single_cutout full path ─────────────────────────────────


class TestLoadSingleCutoutFullPath:
    @patch(f"{MODULE}._create_cutout")
    def test_delegates_to_create_cutout(self, mock_create, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {"SourceID": ["src_001", "src_002"], "RA": [1.0, 2.0], "Dec": [3.0, 4.0]}
        ).to_csv(d / "cat.csv", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_create.return_value = fake_img

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_single_cutout(cfg, "src_001")

        assert result is not None
        np.testing.assert_array_equal(result, fake_img)
        mock_create.assert_called_once()
        # Verify the source_df passed to _create_cutout has the right row
        call_df = mock_create.call_args[0][1]
        assert call_df.iloc[0]["SourceID"] == "src_001"

    @patch(f"{MODULE}._create_cutout")
    def test_warns_when_found_but_no_cutout(self, mock_create, tmp_path):
        """A source present in the catalogue but yielding no cutout must warn
        with the source id + method, not vanish into an opaque None — this is
        the transient/data failure the detail view used to surface as a bare
        "Re-decode returned no image"."""
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame({"SourceID": ["src_001"], "RA": [1.0], "Dec": [3.0]}).to_csv(
            d / "cat.csv", index=False
        )
        mock_create.return_value = None

        cfg = _minimal_cfg()
        cfg.prediction_search_dir = str(d)
        cfg.normalisation.normalisation_method = "LOG"

        messages: list[str] = []
        sink_id = cutana_loader.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = load_single_cutout(cfg, "src_001")
        finally:
            cutana_loader.logger.remove(sink_id)

        assert result is None
        joined = "".join(messages)
        assert "src_001" in joined
        assert "no cutout" in joined
        # The method context is the point of the warning — pin it so a
        # regression that drops it is caught.
        assert "LOG" in joined

    def test_warns_when_source_not_in_any_catalogue(self, tmp_path):
        """A source missing from every catalogue warns (not a silent debug) so
        a stale id in the detail view is visible in the log."""
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame({"SourceID": ["other"], "RA": [1.0], "Dec": [3.0]}).to_csv(
            d / "cat.csv", index=False
        )

        cfg = DotMap(prediction_search_dir=str(d))
        messages: list[str] = []
        sink_id = cutana_loader.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = load_single_cutout(cfg, "missing_id")
        finally:
            cutana_loader.logger.remove(sink_id)

        assert result is None
        joined = "".join(messages)
        assert "missing_id" in joined
        assert "not found" in joined


# ── load_single_native_cutout ────────────────────────────────────


class TestLoadSingleNativeCutout:
    @patch(f"{MODULE}.get_fitsbolt_config")
    @patch(f"{MODULE}.extract_cutouts_direct")
    @patch(f"{MODULE}._find_source_in_catalogues")
    def test_returns_native_float32(self, mock_find, mock_extract, mock_fitsbolt):
        """Returns the native per-band float cutout via do_only_cutout_extraction,
        not a normalised/resized uint8 — that's what the detail screen
        re-normalises in place."""
        mock_find.return_value = (pd.DataFrame({"SourceID": ["s1"]}), "/d/cat.parquet")
        mock_fitsbolt.side_effect = lambda c: c
        native = _fake_image(h=14, w=14, c=4)  # native (small) float32
        mock_extract.return_value = [_batch(np.stack([native]), ["s1"])]

        cfg = _minimal_cfg()
        cfg.prediction_search_dir = "/d"
        out = load_single_native_cutout(cfg, "s1")

        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (14, 14, 4)
        # Native extraction must be requested.
        assert mock_extract.call_args.kwargs["do_only_cutout_extraction"] is True

    @patch(f"{MODULE}._find_source_in_catalogues")
    def test_returns_none_when_source_not_found(self, mock_find):
        mock_find.return_value = (None, None)
        cfg = _minimal_cfg()
        cfg.prediction_search_dir = "/d"
        assert load_single_native_cutout(cfg, "missing") is None

    @patch(f"{MODULE}.get_fitsbolt_config")
    @patch(f"{MODULE}.extract_cutouts_direct")
    @patch(f"{MODULE}._find_source_in_catalogues")
    def test_returns_none_when_no_cutout(self, mock_find, mock_extract, mock_fitsbolt):
        mock_find.return_value = (pd.DataFrame({"SourceID": ["s1"]}), "/d/cat.parquet")
        mock_fitsbolt.side_effect = lambda c: c
        mock_extract.return_value = []
        cfg = _minimal_cfg()
        cfg.prediction_search_dir = "/d"
        assert load_single_native_cutout(cfg, "s1") is None


# ── load_sample_cutouts full path ────────────────────────────────


class TestLoadSampleCutoutsFullPath:
    @patch(f"{MODULE}._create_batch_cutouts")
    def test_samples_and_returns_tuples(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {
                "SourceID": ["s1", "s2", "s3", "s4", "s5"],
                "RA": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Dec": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ).to_csv(d / "cat.csv", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_batch.return_value = {"s1": fake_img, "s3": fake_img, "s5": fake_img}

        cfg = DotMap(_dynamic=False)
        result = load_sample_cutouts(cfg, str(d), n_samples=3)

        # Should return tuples for whichever source IDs were in batch_results
        assert isinstance(result, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in result)
        returned_ids = {r[0] for r in result}
        assert returned_ids.issubset({"s1", "s3", "s5"})

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_reads_parquet_catalogue(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {
                "SourceID": ["s1", "s2"],
                "RA": [1.0, 2.0],
                "Dec": [1.0, 2.0],
                "fits_file_paths": ["['a.fits']"] * 2,
            }
        ).to_parquet(d / "cat.parquet", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_batch.return_value = {"s1": fake_img, "s2": fake_img}

        cfg = DotMap(_dynamic=False)
        result = load_sample_cutouts(cfg, str(d), n_samples=2)

        assert len(result) == 2

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_limits_to_n_samples(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        ids = [f"s{i}" for i in range(20)]
        pd.DataFrame(
            {
                "SourceID": ids,
                "RA": range(20),
                "Dec": range(20),
                "fits_file_paths": ["['a.fits']"] * 20,
            }
        ).to_csv(d / "cat.csv", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_batch.return_value = {sid: fake_img for sid in ids}

        cfg = DotMap(_dynamic=False)
        load_sample_cutouts(cfg, str(d), n_samples=5)

        # batch_df passed to _create_batch_cutouts should have at most 5 rows
        call_df = mock_batch.call_args[0][1]
        assert len(call_df) == 5

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_returns_empty_on_batch_failure(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {"SourceID": ["s1"], "RA": [1.0], "Dec": [1.0], "fits_file_paths": ["['a.fits']"]}
        ).to_csv(d / "cat.csv", index=False)

        mock_batch.side_effect = RuntimeError("batch failed")

        cfg = DotMap(_dynamic=False)
        result = load_sample_cutouts(cfg, str(d), n_samples=1)

        assert result == []


# ── load_batch_cutouts full path ─────────────────────────────────


class TestLoadBatchCutoutsFullPath:
    @patch(f"{MODULE}._create_batch_cutouts")
    def test_filters_catalogue_and_delegates(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {
                "SourceID": ["s1", "s2", "s3"],
                "RA": [1.0, 2.0, 3.0],
                "Dec": [1.0, 2.0, 3.0],
                "fits_file_paths": ["['a.fits']"] * 3,
            }
        ).to_csv(d / "cat.csv", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_batch.return_value = {"s1": fake_img, "s3": fake_img}

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["s1", "s3"])

        assert set(result.cutouts.keys()) == {"s1", "s3"}
        assert result.partial is False
        # Verify only the matching rows were passed
        call_df = mock_batch.call_args[0][1]
        assert set(call_df["SourceID"].tolist()) == {"s1", "s3"}

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_reads_parquet_catalogue(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {
                "SourceID": ["s1", "s2"],
                "RA": [1.0, 2.0],
                "Dec": [1.0, 2.0],
                "fits_file_paths": ["['a.fits']"] * 2,
            }
        ).to_parquet(d / "cat.parquet", index=False)

        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        mock_batch.return_value = {"s1": fake_img}

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["s1"])

        assert "s1" in result.cutouts
        assert result.partial is False

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_returns_partial_on_exception(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        pd.DataFrame(
            {"SourceID": ["s1"], "RA": [1.0], "Dec": [1.0], "fits_file_paths": ["['a.fits']"]}
        ).to_csv(d / "cat.csv", index=False)

        mock_batch.side_effect = RuntimeError("batch failed")

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["s1"])

        assert result.cutouts == {}
        # Mid-walk exception leaves the same partial-truth state as a
        # stale_check bail — flagged so callers don't trust the empty dict
        # as "definitively no matches".
        assert result.partial is True

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_marks_partial_when_stale_check_bails(self, mock_batch, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        # Two catalogues so the bail can fire between them.
        pd.DataFrame(
            {"SourceID": ["s1"], "RA": [1.0], "Dec": [1.0], "fits_file_paths": ["['a.fits']"]}
        ).to_csv(d / "cat1.csv", index=False)
        pd.DataFrame(
            {"SourceID": ["s2"], "RA": [2.0], "Dec": [2.0], "fits_file_paths": ["['b.fits']"]}
        ).to_csv(d / "cat2.csv", index=False)

        mock_batch.return_value = {}

        # Bail before the first catalogue is even processed.
        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, ["s1", "s2"], stale_check=lambda: True)

        assert result.cutouts == {}
        assert result.partial is True

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_parallel_scan_finds_id_in_later_catalogue(self, mock_batch, tmp_path):
        """A source that lives only in a late catalogue is still returned.

        Regression for #444: serial scan + early bail-out used to
        time out when one of the early catalogues was slow.  The
        parallel scan dispatches every catalogue at once, so a slow
        early read can't gate a hit in a fast later one.
        """
        d = tmp_path / "data"
        d.mkdir()
        # Twelve catalogues, only the last one holds the requested id.
        for i in range(12):
            pd.DataFrame(
                {
                    "SourceID": [f"s_cat{i}_a", f"s_cat{i}_b"],
                    "RA": [1.0, 2.0],
                    "Dec": [1.0, 2.0],
                    "fits_file_paths": ["['a.fits']"] * 2,
                }
            ).to_csv(d / f"cat_{i:02d}.csv", index=False)
        target = "s_cat11_a"
        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)

        def _fake_create(_cfg, batch_df, _id_col):
            ids = batch_df["SourceID"].astype(str).tolist()
            return {sid: fake_img for sid in ids if sid == target}

        mock_batch.side_effect = _fake_create

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, [target])

        assert target in result.cutouts

    @patch(f"{MODULE}._create_batch_cutouts")
    def test_parallel_scan_short_circuits_on_full_match(self, mock_batch, tmp_path):
        """Once every requested id has been resolved, queued workers are cancelled.

        We can't deterministically assert that *no* extra work runs
        (the executor may already have N catalogues in flight), but
        the result must reach the caller as soon as the answer is
        complete — i.e. ``_create_batch_cutouts`` is invoked at most
        once per catalogue and the call count is bounded.
        """
        d = tmp_path / "data"
        d.mkdir()
        # 10 catalogues, each unique row set; one holds the only target.
        for i in range(10):
            pd.DataFrame(
                {
                    "SourceID": [f"cat{i}_x"],
                    "RA": [0.0],
                    "Dec": [0.0],
                    "fits_file_paths": ["['a.fits']"],
                }
            ).to_csv(d / f"cat_{i:02d}.csv", index=False)
        target = "cat3_x"
        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)

        def _fake_create(_cfg, batch_df, _id_col):
            ids = batch_df["SourceID"].astype(str).tolist()
            return {sid: fake_img for sid in ids if sid == target}

        mock_batch.side_effect = _fake_create

        cfg = DotMap(prediction_search_dir=str(d))
        result = load_batch_cutouts(cfg, [target])

        assert result.cutouts == {target: pytest.approx(fake_img)}
        # Each catalogue with a row matching `wanted` triggers one
        # `_create_batch_cutouts`; un-matched catalogues skip the call
        # via the empty-batch_df short-circuit.  The target catalogue
        # is hit exactly once.
        match_calls = [c for c in mock_batch.call_args_list if not c.args[1].empty]
        assert len(match_calls) == 1


# ── load_batch_cutouts catalogue-origin index (#506) ─────────────


def _write_catalogue(path, ids):
    """Write a minimal valid Cutana catalogue CSV with the given source ids."""
    pd.DataFrame(
        {
            "SourceID": ids,
            "RA": [1.0] * len(ids),
            "Dec": [2.0] * len(ids),
            "fits_file_paths": ["x.fits"] * len(ids),
        }
    ).to_csv(path, index=False)


def _fake_create_batch(cfg, batch_df, id_col):
    """Stand in for Cutana extraction: one blank cutout per requested row."""
    return {str(sid): np.zeros((4, 4, 3), dtype=np.uint8) for sid in batch_df[id_col]}


class TestLoadBatchCutoutsCatalogueIndex:
    """The gallery loader consults/back-fills the DB catalogue index (#506)."""

    def _build(self, tmp_path):
        search = tmp_path / "search"
        search.mkdir()
        _write_catalogue(search / "cat_a.csv", ["s1"])
        _write_catalogue(search / "cat_b.csv", ["s2"])
        cfg = DotMap(_dynamic=False)
        cfg.prediction_search_dir = str(search)
        cfg.output_dir = str(tmp_path)
        cfg.prediction_db_dir = str(tmp_path)  # pin DB here, no relocation
        return cfg, search

    def _spy_scan(self, scanned):
        real = cutana_loader._scan_one_catalogue

        def spy(cat_file, *args, **kwargs):
            scanned.append(os.path.basename(cat_file))
            return real(cat_file, *args, **kwargs)

        return spy

    def test_uses_index_to_scan_only_recorded_catalogue(self, tmp_path):
        cfg, search = self._build(tmp_path)
        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            db.store_results([("s1", 0.9), ("s2", 0.5)])
            db.record_catalogues({"s1": str(search / "cat_a.csv"), "s2": str(search / "cat_b.csv")})

        scanned: list[str] = []
        with (
            patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch),
            patch(f"{MODULE}._scan_one_catalogue", side_effect=self._spy_scan(scanned)),
        ):
            result = load_batch_cutouts(cfg, ["s1"])

        assert set(result.cutouts) == {"s1"}
        assert scanned == ["cat_a.csv"]  # cat_b skipped via the index

    def test_backfills_index_after_full_scan(self, tmp_path):
        cfg, search = self._build(tmp_path)
        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            db.store_results([("s1", 0.9), ("s2", 0.5)])  # no origins recorded yet

        with patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch):
            load_batch_cutouts(cfg, ["s1", "s2"])

        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            assert db.get_catalogue_map(["s1", "s2"]) == {
                "s1": str(search / "cat_a.csv"),
                "s2": str(search / "cat_b.csv"),
            }

    def test_stale_recorded_catalogue_falls_back_to_full_scan(self, tmp_path):
        cfg, search = self._build(tmp_path)
        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            db.store_results([("s1", 0.9)])
            db.record_catalogues({"s1": str(search / "gone.csv")})  # no longer present

        scanned: list[str] = []
        with (
            patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch),
            patch(f"{MODULE}._scan_one_catalogue", side_effect=self._spy_scan(scanned)),
        ):
            result = load_batch_cutouts(cfg, ["s1"])

        assert set(result.cutouts) == {"s1"}
        assert set(scanned) == {"cat_a.csv", "cat_b.csv"}  # stale path → full scan

    def test_training_preview_path_ignores_index(self, tmp_path):
        """A search_dir other than prediction_search_dir never touches the index."""
        cfg, _ = self._build(tmp_path)
        other = tmp_path / "other"
        other.mkdir()
        _write_catalogue(other / "c.csv", ["s1"])
        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            db.store_results([("s1", 0.9)])

        with (
            patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch),
            patch(f"{MODULE}._run_on_index_db") as index_mock,
        ):
            load_batch_cutouts(cfg, ["s1"], search_dir=str(other))

        # A non-prediction search_dir never reads or writes the index.
        index_mock.assert_not_called()

    def test_missing_db_falls_back_without_error(self, tmp_path):
        """No predictions.db yet → full scan, no crash, cutouts still returned."""
        cfg, _ = self._build(tmp_path)  # DB never created
        with patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch):
            result = load_batch_cutouts(cfg, ["s1"])
        assert set(result.cutouts) == {"s1"}

    def test_index_hit_does_not_rewrite_known_origins(self, tmp_path):
        """On a full index hit every origin is already recorded, so no write
        transaction is opened on the page turn (avoids per-page churn)."""
        cfg, search = self._build(tmp_path)
        with AnomalyScoreDB(prediction_db_path(cfg)) as db:
            db.store_results([("s1", 0.9)])
            db.record_catalogues({"s1": str(search / "cat_a.csv")})

        # Spy on the shared index helper (read + back-fill go through it) while
        # keeping its real behaviour, then assert no call ran with write=True.
        real_run_on_index_db = cutana_loader._run_on_index_db
        index_writes: list[bool] = []

        def _spy_index(cfg, *, write, operation, default):
            index_writes.append(write)
            return real_run_on_index_db(cfg, write=write, operation=operation, default=default)

        with (
            patch(f"{MODULE}._create_batch_cutouts", side_effect=_fake_create_batch),
            patch(f"{MODULE}._run_on_index_db", side_effect=_spy_index),
        ):
            result = load_batch_cutouts(cfg, ["s1"])

        assert set(result.cutouts) == {"s1"}
        assert True not in index_writes, "a full index hit must not open a write transaction"
