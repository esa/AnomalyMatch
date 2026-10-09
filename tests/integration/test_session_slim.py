#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for Session, SessionIOHandler, and SessionTracker.

Focuses on real code paths that increase coverage:
- SessionIOHandler: merge_gallery_labels, labeled data extraction
- SessionTracker: update_labeled_data (complex DataFrame merge with iteration tracking)
- Session: init, remember_current_file, evaluate_all_images error paths, file type detection
"""

import os

import pandas as pd
import pytest

from anomaly_match.data_io.checkpoint_io import save_checkpoint
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.pipeline.session import Session
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.utils.get_default_cfg import get_default_cfg


@pytest.fixture()
def session_cfg(tmp_path):
    """Create a minimal config for Session initialization."""
    cfg = get_default_cfg()
    cfg.output_dir = str(tmp_path / "output")
    cfg.save_dir = str(tmp_path / "sessions")
    cfg.name = "test_session"
    cfg.log_level = "WARNING"
    return cfg


# ── SessionIOHandler.merge_gallery_labels ────────────────────────


class TestMergeGalleryLabels:
    """Test the real merge logic (not via BackendInterface mock)."""

    def test_creates_csv_from_scratch(self, tmp_path):
        io = SessionIOHandler()
        out = str(tmp_path / "labels.csv")
        labels = {"a.png": LABEL_ANOMALY, "b.png": LABEL_NORMAL}

        result = io.merge_gallery_labels(None, labels, out)

        df = pd.read_csv(result)
        assert len(df) == 2
        assert set(df["id"]) == {"a.png", "b.png"}
        assert list(df.columns[:2]) == ["id", "label"]

    def test_merges_with_existing(self, tmp_path):
        existing = str(tmp_path / "existing.csv")
        pd.DataFrame([{"id": "old.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)

        io = SessionIOHandler()
        out = str(tmp_path / "merged.csv")
        result = io.merge_gallery_labels(existing, {"new.png": LABEL_ANOMALY}, out)

        df = pd.read_csv(result)
        assert len(df) == 2
        assert "old.png" in df["id"].values
        assert "new.png" in df["id"].values

    def test_new_labels_override_existing(self, tmp_path):
        existing = str(tmp_path / "existing.csv")
        pd.DataFrame([{"id": "img.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)

        io = SessionIOHandler()
        out = str(tmp_path / "merged.csv")
        io.merge_gallery_labels(existing, {"img.png": LABEL_ANOMALY}, out)

        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["label"] == LABEL_ANOMALY

    def test_filters_invalid_labels(self, tmp_path):
        io = SessionIOHandler()
        out = str(tmp_path / "labels.csv")
        labels = {"good.png": LABEL_ANOMALY, "bad.png": "garbage"}

        io.merge_gallery_labels(None, labels, out)
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["id"] == "good.png"

    def test_corrupt_existing_csv_raises(self, tmp_path):
        """An unusable existing CSV fails loudly rather than dropping the labels.

        Falling back to an empty frame here writes out only the new gallery
        labels, so the user's entire existing label set disappears behind a
        warning and the next training run silently uses the remainder.
        """
        corrupt = str(tmp_path / "corrupt.csv")
        with open(corrupt, "w") as f:
            f.write("not,valid\x00csv\n\x00\x00\x00")

        io = SessionIOHandler()
        out = str(tmp_path / "labels.csv")
        with pytest.raises(ValueError, match="no 'id' column"):
            io.merge_gallery_labels(corrupt, {"a.png": LABEL_ANOMALY}, out)

    def test_empty_new_labels_preserves_existing(self, tmp_path):
        existing = str(tmp_path / "existing.csv")
        pd.DataFrame([{"id": "a.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)

        io = SessionIOHandler()
        out = str(tmp_path / "labels.csv")
        io.merge_gallery_labels(existing, {}, out)

        df = pd.read_csv(out)
        assert len(df) == 1


# ── SessionIOHandler.get_labeled_data_from_checkpoint ────────────


class TestLabeledDataFromCheckpoint:
    def test_extracts_csv_from_checkpoint(self):
        csv_str = "filename,label\na.png,1\nb.png,0\n"
        checkpoint = {"labeled_data_csv": csv_str}

        df = SessionIOHandler.get_labeled_data_from_checkpoint(checkpoint)
        assert len(df) == 2
        assert "a.png" in df["filename"].values

    def test_raises_when_missing(self):
        with pytest.raises(ValueError, match="does not contain labeled data"):
            SessionIOHandler.get_labeled_data_from_checkpoint({})


# ── SessionTracker.update_labeled_data ───────────────────────────


class TestUpdateLabeledData:
    """Complex DataFrame merge preserving iteration info."""

    def test_initializes_from_empty(self):
        tracker = SessionTracker(session_name="test")
        new_df = pd.DataFrame(
            [
                {"id": "a.png", "label": LABEL_ANOMALY},
                {"id": "b.png", "label": LABEL_NORMAL},
            ]
        )

        tracker.update_labeled_data(new_df)
        assert len(tracker.labeled_data_df) == 2
        assert "iteration" in tracker.labeled_data_df.columns
        assert "id" in tracker.labeled_data_df.columns

    def test_merge_preserves_existing_iterations(self):
        tracker = SessionTracker(session_name="test")
        tracker.labeled_data_df = pd.DataFrame(
            [
                {"id": "a.png", "label": LABEL_ANOMALY, "iteration": 0},
                {"id": "b.png", "label": LABEL_NORMAL, "iteration": 1},
            ]
        )

        new_df = pd.DataFrame(
            [
                {"id": "a.png", "label": LABEL_NORMAL},
                {"id": "c.png", "label": LABEL_ANOMALY},
            ]
        )

        tracker.update_labeled_data(new_df)
        result = tracker.labeled_data_df

        a_row = result[result["id"] == "a.png"].iloc[0]
        assert a_row["iteration"] == 0

        c_row = result[result["id"] == "c.png"].iloc[0]
        assert c_row["iteration"] == -1

    def test_deduplicates_keeping_last(self):
        tracker = SessionTracker(session_name="test")
        tracker.labeled_data_df = pd.DataFrame(
            [{"id": "a.png", "label": LABEL_NORMAL, "iteration": 0}]
        )

        new_df = pd.DataFrame([{"id": "a.png", "label": LABEL_ANOMALY, "iteration": -1}])
        tracker.update_labeled_data(new_df)

        result = tracker.labeled_data_df
        assert len(result[result["id"] == "a.png"]) == 1

    def test_adds_iteration_column_if_missing(self):
        tracker = SessionTracker(session_name="test")
        new_df = pd.DataFrame([{"id": "a.png", "label": LABEL_ANOMALY}])
        assert "iteration" not in new_df.columns

        tracker.update_labeled_data(new_df)
        assert "iteration" in tracker.labeled_data_df.columns
        assert tracker.labeled_data_df.iloc[0]["iteration"] == -1


# ── SessionTracker.add_labeled_sample with explicit iteration ────


class TestAddLabeledSample:
    def test_explicit_iteration_number(self):
        tracker = SessionTracker(session_name="test")
        tracker.start_new_session_iteration()

        tracker.add_labeled_sample("a.png", LABEL_ANOMALY, iteration_number=0)

        df = tracker.get_labeled_data_df()
        assert len(df) == 1
        assert df.iloc[0]["iteration"] == 0
        assert df.iloc[0]["id"] == "a.png"

    def test_auto_assigns_current_iteration(self):
        tracker = SessionTracker(session_name="test")
        tracker.start_new_session_iteration()
        tracker.start_new_session_iteration()

        tracker.add_labeled_sample("a.png", LABEL_ANOMALY)

        df = tracker.get_labeled_data_df()
        assert df.iloc[0]["iteration"] == 1

    def test_updates_iteration_counts(self):
        tracker = SessionTracker(session_name="test")
        tracker.start_new_session_iteration()

        tracker.add_labeled_sample("a.png", LABEL_ANOMALY, iteration_number=0)
        tracker.add_labeled_sample("b.png", LABEL_NORMAL, iteration_number=0)
        tracker.add_labeled_sample("c.png", LABEL_ANOMALY, iteration_number=0)

        info = tracker.get_iteration_info(0)
        assert info["num_newly_labeled_anomalous"] == 2
        assert info["num_newly_labeled_nominal"] == 1


# ── SessionTracker.get_session_info edge cases ───────────────────


def test_session_info_legacy_no_iteration_column():
    """When iteration column is missing, all data treated as initial."""
    tracker = SessionTracker(session_name="test")
    tracker.labeled_data_df = pd.DataFrame(
        [
            {"filename": "a.png", "label": LABEL_ANOMALY},
            {"filename": "b.png", "label": LABEL_NORMAL},
        ]
    )

    info = tracker.get_session_info()
    assert info["initial_labeled_samples"] == 2
    assert info["iteration_labeled_samples"] == 0


# ── Session init and remember_current_file ───────────────────────


def test_session_init(session_cfg):
    session = Session(session_cfg)
    assert session.session_tracker is not None
    assert session.cfg is session_cfg


def test_remember_file_dedup(session_cfg):
    session = Session(session_cfg)
    os.makedirs(session.cfg.output_dir, exist_ok=True)

    session.remember_current_file("galaxy_001.fits")
    session.remember_current_file("galaxy_001.fits")
    session.remember_current_file("galaxy_002.fits")

    output_files = [f for f in os.listdir(session.cfg.output_dir) if "remembered" in f]
    df = pd.read_csv(os.path.join(session.cfg.output_dir, output_files[0]))
    assert len(df) == 2


# ── Session.evaluate_all_images error paths ──────────────────────


def test_evaluate_raises_without_model(session_cfg, tmp_path):
    session = Session(session_cfg)
    session.cfg.model_path = str(tmp_path / "nonexistent_model.safetensors")

    with pytest.raises(FileNotFoundError, match="Model not found"):
        session.evaluate_all_images()


def test_evaluate_raises_without_search_dir(session_cfg, tmp_path):
    session = Session(session_cfg)
    model_path = tmp_path / "model.safetensors"
    save_checkpoint({"train_model": {}, "eval_model": {}, "fitsbolt_cfg": None}, model_path)
    session.cfg.model_path = str(model_path)
    session.cfg.prediction_search_dir = None

    with pytest.raises(ValueError, match="No prediction_search_dir"):
        session.evaluate_all_images()


def test_evaluate_raises_when_model_lacks_normalisation(session_cfg, tmp_path):
    """A checkpoint without embedded fitsbolt config must fail hard rather than
    inventing a normalisation pipeline from the current cfg."""
    session = Session(session_cfg)
    model_path = tmp_path / "model.safetensors"
    save_checkpoint({"train_model": {}, "eval_model": {}, "fitsbolt_cfg": None}, model_path)
    session.cfg.model_path = str(model_path)
    search_dir = tmp_path / "search"
    search_dir.mkdir()
    (search_dir / "a.jpg").touch()
    session.cfg.prediction_search_dir = str(search_dir)

    with pytest.raises(ValueError, match="no embedded fitsbolt config"):
        session.evaluate_all_images()


# ── Session._auto_detect_prediction_file_type ────────────────────


class TestAutoDetectFileType:
    def test_image_files(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        d = tmp_path / "images"
        d.mkdir()
        (d / "a.jpg").touch()
        (d / "b.png").touch()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.IMAGE_FOLDER

    def test_zarr_directory(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        d = tmp_path / "data"
        d.mkdir()
        (d / "store.zarr").touch()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.ZARR

    def test_nested_zarr(self, session_cfg, tmp_path):
        """Zarr detected from nested images.zarr inside a directory."""
        session = Session(session_cfg)
        d = tmp_path / "data"
        d.mkdir()
        container = d / "my_dataset"
        container.mkdir()
        (container / "images.zarr").mkdir()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.ZARR

    def test_search_dir_is_the_zarr_store_itself(self, session_cfg, tmp_path):
        """Pointing prediction_search_dir directly at a .zarr store (not its
        parent) must detect ZARR rather than scanning the store's own
        contents ("images"/"zarr.json") as if they were candidate files.
        """
        session = Session(session_cfg)
        store = tmp_path / "test_images.zarr"
        store.mkdir()
        (store / "zarr.json").touch()
        (store / "images").mkdir()
        assert session._auto_detect_prediction_file_type(str(store)) == DataSourceType.ZARR

    def test_stream_parquet(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        d = tmp_path / "catalogs"
        d.mkdir()
        (d / "catalog.parquet").touch()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.CUTANA

    def test_empty_defaults_image(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        d = tmp_path / "empty"
        d.mkdir()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.IMAGE_FOLDER

    def test_nonexistent_defaults_image(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        assert (
            session._auto_detect_prediction_file_type(str(tmp_path / "nope"))
            == DataSourceType.IMAGE_FOLDER
        )

    def test_mixed_prefers_majority(self, session_cfg, tmp_path):
        session = Session(session_cfg)
        d = tmp_path / "mixed"
        d.mkdir()
        (d / "a.jpg").touch()
        (d / "b.jpg").touch()
        (d / "c.png").touch()
        assert session._auto_detect_prediction_file_type(str(d)) == DataSourceType.IMAGE_FOLDER
