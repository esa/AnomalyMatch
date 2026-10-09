#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Integration tests for BackendInterface training-related methods."""

import os
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL, LABEL_REMOVED
from anomaly_match_ui.utils.backend_interface import BackendInterface


@pytest.fixture()
def mock_session(tmp_path):
    """Create a mock session with minimal config for label merging."""
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "tests/test_data/grayscale/"
    session.cfg.output_dir = str(tmp_path)
    session.cfg.label_file = None
    session.cfg.model_path = None
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_train_iter = 200
    session.cfg.toDict.return_value = {"output_dir": str(tmp_path)}
    session.session_io = SessionIOHandler(base_save_path=str(tmp_path / "sessions"))
    BackendInterface.set_session(session)
    yield session
    BackendInterface.set_session(None)


# ── merge_gallery_labels ─────────────────────────────────────────


class TestMergeGalleryLabels:
    """Tests for BackendInterface.merge_gallery_labels."""

    def test_creates_csv_from_empty(self, mock_session, tmp_path):
        labels = {"img1.png": LABEL_ANOMALY, "img2.png": LABEL_NORMAL}
        out = BackendInterface.merge_gallery_labels(labels)
        assert os.path.isfile(out)
        df = pd.read_csv(out)
        assert len(df) == 2
        assert set(df["id"]) == {"img1.png", "img2.png"}

    def test_merges_with_existing_csv(self, mock_session, tmp_path):
        existing = tmp_path / "labels.csv"
        pd.DataFrame([{"id": "old.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)
        mock_session.cfg.label_file = str(existing)

        labels = {"new.png": LABEL_ANOMALY}
        out = BackendInterface.merge_gallery_labels(labels)
        df = pd.read_csv(out)
        assert len(df) == 2
        assert "old.png" in df["id"].values
        assert "new.png" in df["id"].values

    def test_new_labels_override_existing(self, mock_session, tmp_path):
        existing = tmp_path / "labels.csv"
        pd.DataFrame([{"id": "img.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)
        mock_session.cfg.label_file = str(existing)

        # Override with anomaly
        labels = {"img.png": LABEL_ANOMALY}
        out = BackendInterface.merge_gallery_labels(labels)
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["label"] == LABEL_ANOMALY

    def test_empty_labels_preserves_existing(self, mock_session, tmp_path):
        existing = tmp_path / "labels.csv"
        pd.DataFrame([{"id": "img.png", "label": LABEL_NORMAL}]).to_csv(existing, index=False)
        mock_session.cfg.label_file = str(existing)

        out = BackendInterface.merge_gallery_labels({})
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["label"] == LABEL_NORMAL

    def test_custom_output_path(self, mock_session, tmp_path):
        out_path = str(tmp_path / "custom" / "labels.csv")
        labels = {"img.png": LABEL_ANOMALY}
        result = BackendInterface.merge_gallery_labels(labels, output_path=out_path)
        assert result == out_path
        assert os.path.isfile(out_path)

    def test_ignores_invalid_labels(self, mock_session, tmp_path):
        labels = {"good.png": LABEL_ANOMALY, "bad.png": "invalid_label"}
        out = BackendInterface.merge_gallery_labels(labels)
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["id"] == "good.png"

    def test_removed_overrides_existing_label(self, mock_session, tmp_path):
        """Un-labelling a previously-labelled id writes a ``removed`` row that
        supersedes its old label (excluded from the labeled set at build)."""
        existing = tmp_path / "labels.csv"
        pd.DataFrame([{"id": "img.png", "label": LABEL_ANOMALY}]).to_csv(existing, index=False)
        mock_session.cfg.label_file = str(existing)

        out = BackendInterface.merge_gallery_labels({"img.png": LABEL_REMOVED})
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df.iloc[0]["label"] == LABEL_REMOVED

    def test_removed_for_unknown_id_is_dropped(self, mock_session, tmp_path):
        """A ``removed`` row for an id that never carried a label is noise and
        must not be written (anomaly/normal in the same batch still land)."""
        existing = tmp_path / "labels.csv"
        pd.DataFrame([{"id": "img.png", "label": LABEL_ANOMALY}]).to_csv(existing, index=False)
        mock_session.cfg.label_file = str(existing)

        out = BackendInterface.merge_gallery_labels(
            {"ghost.png": LABEL_REMOVED, "img.png": LABEL_NORMAL}
        )
        df = pd.read_csv(out)
        assert set(df["id"]) == {"img.png"}
        assert df.iloc[0]["label"] == LABEL_NORMAL


# ── launch_training_subprocess ───────────────────────────────────


class TestLaunchTrainingSubprocess:
    """Tests for BackendInterface.launch_training_subprocess."""

    def test_raises_without_session(self):
        BackendInterface.set_session(None)
        with pytest.raises(RuntimeError, match="No session set"):
            BackendInterface.launch_training_subprocess()

    def test_applies_num_train_iter_override(self, mock_session):
        mock_session.cfg.num_train_iter = 100
        import unittest.mock

        with unittest.mock.patch("anomaly_match.pipeline.session.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            # Use real Session.launch_training_subprocess instead of MagicMock
            from anomaly_match.pipeline.session import Session

            mock_session.launch_training_subprocess = lambda **kw: (
                Session.launch_training_subprocess(mock_session, **kw)
            )
            BackendInterface.launch_training_subprocess(num_train_iter=300)
        assert mock_session.cfg.num_train_iter == 300

    def test_creates_temp_dir_with_config_and_labels(self, mock_session, tmp_path):
        import unittest.mock

        from anomaly_match.pipeline.session import Session

        mock_session.launch_training_subprocess = lambda **kw: Session.launch_training_subprocess(
            mock_session, **kw
        )

        with unittest.mock.patch("anomaly_match.pipeline.session.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            _proc, temp_dir, progress_file = BackendInterface.launch_training_subprocess()

        assert os.path.isdir(temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "config.pkl"))
        assert os.path.isfile(os.path.join(temp_dir, "labelled_data.csv"))
        assert progress_file == os.path.join(temp_dir, "progress.jsonl")
