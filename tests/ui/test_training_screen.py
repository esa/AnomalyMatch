#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the TrainingScreen state machine, labeling, and retrain flow."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.prediction import AnomalyScoreDB
from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.screens.training_screen import (
    _LABEL_STATE_TO_CSV,
    TrainingState,
    shorten_filename,
)
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widgets.labelable_thumbnail_cell import LabelState

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_session(tmp_path):
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "tests/test_data/grayscale/"
    session.cfg.net = "test-cnn"
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_channels = 3
    session.cfg.num_train_iter = 200
    session.cfg.cutana_stratify_source_size = False
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.prediction_search_dir = None
    session.cfg.output_dir = str(tmp_path)
    session.cfg.label_file = None
    session.session_io = SessionIOHandler(base_save_path=str(tmp_path / "sessions"))
    BackendInterface.set_session(session)
    yield session
    BackendInterface.set_session(None)


@pytest.fixture()
def screen(mock_session):
    """Build a TrainingScreen instance."""
    with patch("IPython.display.display"):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training")
        return app._current_screen


# ── Waiting spinner ─────────────────────────────────────────────


def test_scoring_refused_before_start_stops_the_spinner(screen):
    """An early return before the subprocess starts must not leave it spinning."""
    screen.cfg.model_path = None

    screen._start_scoring()

    assert screen._state == TrainingState.BROWSING
    assert screen.progress.spinner.value == ""


def test_failed_run_stops_the_spinner(screen):
    """A training run that dies before its first iteration ends on BROWSING."""
    screen._set_state(TrainingState.BUSY, phase="training")
    screen.progress.start("Starting training...")
    assert "am-spinner" in screen.progress.spinner.value

    screen._set_state(TrainingState.BROWSING)

    assert screen.progress.spinner.value == ""


# ── State machine ───────────────────────────────────────────────


def test_state_transitions(screen):
    """BROWSING → BUSY(training) → BUSY(scoring) → BROWSING."""
    assert screen._state == TrainingState.BROWSING
    assert screen._stop_btn.disabled is True

    screen._set_state(TrainingState.BUSY, phase="training")
    assert screen._state == TrainingState.BUSY
    assert screen._stop_btn.disabled is False
    assert screen._retrain_btn.disabled is True
    assert "TRAINING" in screen._state_html.value

    screen._set_state(TrainingState.BUSY, phase="scoring")
    assert screen._busy_phase == "scoring"
    assert "SCORING" in screen._state_html.value

    screen._set_state(TrainingState.BROWSING)
    assert screen._retrain_btn.disabled is False
    assert screen._iter_slider.disabled is False
    assert "BROWSING" in screen._state_html.value


# ── Label handling ──────────────────────────────────────────────


def test_label_lifecycle(screen):
    """Store → override → remove labels via _on_label_change."""
    screen._on_label_change("img.png", LabelState.ANOMALY)
    assert screen._labels["img.png"] == LabelState.ANOMALY

    screen._on_label_change("img.png", LabelState.NORMAL)
    assert screen._labels["img.png"] == LabelState.NORMAL

    screen._on_label_change("img.png", LabelState.UNLABELLED)
    assert "img.png" not in screen._labels


def test_label_summary_html(screen):
    screen._on_label_change("a.png", LabelState.ANOMALY)
    screen._on_label_change("b.png", LabelState.NORMAL)
    screen._on_label_change("c.png", LabelState.ANOMALY)

    html = screen._label_summary_html.value
    assert "2 anomaly" in html
    assert "1 normal" in html


def test_label_state_to_csv_mapping():
    assert _LABEL_STATE_TO_CSV[LabelState.ANOMALY] == LABEL_ANOMALY
    assert _LABEL_STATE_TO_CSV[LabelState.NORMAL] == LABEL_NORMAL
    assert LabelState.UNLABELLED not in _LABEL_STATE_TO_CSV


# ── Retrain flow ────────────────────────────────────────────────


def test_retrain_blocked_when_busy(screen):
    screen._set_state(TrainingState.BUSY, phase="training")
    screen._labels = {"img.png": LabelState.ANOMALY}

    with patch.object(screen, "_start_training") as mock_start:
        screen._on_retrain()
        mock_start.assert_not_called()


def test_retrain_with_no_labels_still_starts(screen):
    """Retrain without new labels should start training (e.g. after normalisation change)."""
    screen._labels = {}
    with patch.object(screen, "_start_training") as mock_start:
        screen._on_retrain()
        mock_start.assert_called_once()


def test_retrain_merges_labels_and_restarts(screen, tmp_path):
    screen._labels = {
        "img1.png": LabelState.ANOMALY,
        "img2.png": LabelState.NORMAL,
    }

    merged_csv = str(tmp_path / "merged.csv")
    with (
        patch.object(BackendInterface, "merge_gallery_labels", return_value=merged_csv),
        patch.object(screen, "_start_training") as mock_start,
    ):
        screen._on_retrain()

        call_labels = BackendInterface.merge_gallery_labels.call_args[0][0]
        assert call_labels["img1.png"] == LABEL_ANOMALY
        assert call_labels["img2.png"] == LABEL_NORMAL
        mock_start.assert_called_once()

    assert len(screen._labels) == 0


def test_retrain_recovers_when_merge_raises(screen):
    """A failed label merge must leave the Retrain button usable.

    ``merge_gallery_labels`` raises ``ValueError`` on an unreadable or
    id-less labels CSV.  The handler disables the button up-front and only
    ``_set_state`` re-enables it, so without an ``except`` the click left a
    dead button and a status frozen on "Preparing retrain…".
    """
    screen._labels = {"img1.png": LabelState.ANOMALY}

    with (
        patch.object(
            BackendInterface, "merge_gallery_labels", side_effect=ValueError("no 'id' column")
        ),
        patch.object(screen, "_start_training") as mock_start,
    ):
        screen._on_retrain()

        mock_start.assert_not_called()

    assert screen._state == TrainingState.BROWSING
    assert screen._retrain_btn.disabled is False
    assert screen._retrain_btn.description == "Retrain"
    assert "no 'id' column" in screen.status_html.value
    # Labels stay in the gallery — they were never written to the CSV.
    assert screen._labels == {"img1.png": LabelState.ANOMALY}


# ── Stop ────────────────────────────────────────────────────────


def test_stop_terminates_training_process(screen):
    mock_proc = MagicMock()
    screen._process = mock_proc
    screen._set_state(TrainingState.BUSY, phase="training")

    screen._on_stop()

    mock_proc.terminate.assert_called_once()
    assert screen._state == TrainingState.BROWSING
    assert screen._process is None


def test_stop_scoring_shows_partial(screen, tmp_path):
    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.store_results([("img1.png", 0.8)])
    db.close()
    BackendInterface.open_prediction_monitor(str(db_path), str(tmp_path))

    screen._set_state(TrainingState.BUSY, phase="scoring")
    screen._on_stop()

    assert screen._state == TrainingState.BROWSING
    assert "partial results" in screen.status_html.value


def test_stop_training_shows_immediate_feedback(screen):
    """Stop click must flip the button + status before subprocess teardown (#434)."""
    captured: dict = {}

    def _fake_terminate() -> None:
        captured["btn_disabled"] = screen._stop_btn.disabled
        captured["btn_label"] = screen._stop_btn.description
        captured["status"] = screen.status_html.value

    mock_proc = MagicMock()
    mock_proc.terminate.side_effect = _fake_terminate
    screen._process = mock_proc
    screen._set_state(TrainingState.BUSY, phase="training")

    screen._on_stop()

    assert captured["btn_disabled"] is True
    assert captured["btn_label"] == "Stopping…"
    assert "Stopping training" in captured["status"]
    # After teardown, the canonical label is restored so the next run
    # shows "Stop" again.
    assert screen._stop_btn.description == "Stop"


def test_stop_scoring_shows_immediate_feedback(screen, tmp_path):
    """Stop during scoring updates the button label + banner before the
    blocking thread.join, so the click doesn't appear lost (#434)."""
    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.store_results([("img1.png", 0.8)])
    db.close()
    BackendInterface.open_prediction_monitor(str(db_path), str(tmp_path))

    captured: dict = {}

    def _capture() -> None:
        captured["btn_disabled"] = screen._stop_btn.disabled
        captured["btn_label"] = screen._stop_btn.description
        captured["status"] = screen.status_html.value

    screen._set_state(TrainingState.BUSY, phase="scoring")

    with patch.object(screen, "_cancel_scoring_if_running", side_effect=_capture):
        screen._on_stop()

    assert captured["btn_disabled"] is True
    assert captured["btn_label"] == "Stopping…"
    assert "Cancelling prediction subprocess" in captured["status"]
    assert screen._stop_btn.description == "Stop"


def test_retrain_shows_immediate_feedback(screen):
    """Retrain click greys out the button + sets the banner before the
    blocking _cancel_scoring_if_running call (#434)."""
    captured: dict = {}

    def _capture() -> None:
        captured["btn_disabled"] = screen._retrain_btn.disabled
        captured["btn_label"] = screen._retrain_btn.description
        captured["status"] = screen.status_html.value

    screen._labels = {}
    with (
        patch.object(screen, "_cancel_scoring_if_running", side_effect=_capture),
        patch.object(screen, "_start_training"),
    ):
        screen._on_retrain()

    assert captured["btn_disabled"] is True
    assert captured["btn_label"] == "Preparing retrain…"
    assert "Preparing retrain" in captured["status"]
    # Description is restored to "Retrain" so subsequent BROWSING
    # transitions show the right label.
    assert screen._retrain_btn.description == "Retrain"


# ── Poll restart on re-enter ─────────────────────────────────────


def test_on_enter_restarts_db_polling_when_scoring_in_flight(screen):
    """Returning from the detail screen while scoring is still running must
    restart DB polling — on_leave stops it, so without this the gallery
    would freeze until the next button click."""
    alive_thread = MagicMock()
    alive_thread.is_alive.return_value = True
    screen._prediction_thread = alive_thread
    screen._polling = False
    screen._process = None
    screen._db_path = None

    with (
        patch.object(screen, "_start_polling") as start_polling,
        patch.object(screen, "_start_training_poll") as start_train_poll,
        patch.object(screen, "_start_training"),
    ):
        screen.on_enter()

    start_polling.assert_called_once()
    start_train_poll.assert_not_called()


def test_on_enter_restarts_training_poll_when_subprocess_alive(screen):
    """A live training subprocess (process.poll() is None) gets its progress
    poll restarted on re-enter."""
    running_proc = MagicMock()
    running_proc.poll.return_value = None
    screen._process = running_proc
    screen._training_poll_thread = None
    screen._prediction_thread = None
    screen._db_path = None

    with (
        patch.object(screen, "_start_polling") as start_polling,
        patch.object(screen, "_start_training_poll") as start_train_poll,
        patch.object(screen, "_start_training"),
    ):
        screen.on_enter()

    start_train_poll.assert_called_once()
    start_polling.assert_not_called()


def test_on_enter_restarts_training_poll_when_subprocess_exited_while_away(screen):
    """If training finished while the detail screen was open, poll() returns an
    exit code (not None).  The poll must still restart so the loop can dispatch
    _on_training_complete — otherwise the screen stays stuck in the training
    phase.  _on_training_complete clears _process, so a non-None handle means
    completion was never handled."""
    exited_proc = MagicMock()
    exited_proc.poll.return_value = 0
    screen._process = exited_proc
    screen._training_poll_thread = None
    screen._prediction_thread = None
    screen._db_path = None

    with (
        patch.object(screen, "_start_polling") as start_polling,
        patch.object(screen, "_start_training_poll") as start_train_poll,
        patch.object(screen, "_start_training"),
    ):
        screen.on_enter()

    start_train_poll.assert_called_once()
    start_polling.assert_not_called()


def test_on_enter_does_not_restart_polls_when_idle(screen):
    """With no scoring thread and no live subprocess, neither poll restarts —
    a finished run's static gallery shouldn't spin up background threads."""
    screen._prediction_thread = None
    screen._process = None
    screen._db_path = None

    with (
        patch.object(screen, "_start_polling") as start_polling,
        patch.object(screen, "_start_training_poll") as start_train_poll,
        patch.object(screen, "_start_training"),
    ):
        screen.on_enter()

    start_polling.assert_not_called()
    start_train_poll.assert_not_called()


# ── Progress parsing ────────────────────────────────────────────


def test_on_enter_preserves_user_channels_for_cutana(mock_session):
    """Re-entering the training screen must not reset the user's channel count.

    Regression test: ``apply_cutana_bands`` forced ``n_output_channels``
    to equal ``len(filter_names)``, clobbering the user's single-channel
    choice from the setup screen and producing a cfg/cache mismatch in
    the training subprocess.
    """
    mock_session.cfg.data_dir = "/fake/q1_catalogues"
    mock_session.cfg.normalisation.n_output_channels = 1

    with (
        patch("IPython.display.display"),
        patch(
            "anomaly_match_ui.screens.training_screen._auto_detect_source_type",
            return_value=DataSourceType.CUTANA,
        ),
        patch(
            "anomaly_match_ui.screens.training_screen.detect_cutana_filter_names",
            return_value=["VIS", "NIR-H", "NIR-J", "NIR-Y"],
        ),
        patch("anomaly_match_ui.screens.training_screen.TrainingScreen._start_training"),
    ):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training")
        screen = app._current_screen

        assert screen._norm_widget._n_channels == 1
        cfg = screen._norm_widget.get_normalisation_config()
        assert cfg["n_output_channels"] == 1
        # Extensions should still be synced so the matrix has 4 columns
        assert len(screen._norm_widget._extensions) == 4


def test_handle_progress_training(screen):
    screen._handle_progress_line({"status": "training", "iteration": 50, "total": 200})
    assert screen.progress.progress_bar.value == pytest.approx(0.25)
    assert "50/200" in screen.progress.status_label.value


def test_loading_phase_tqdm_drives_progress_bar(screen):
    """While loading, the decode/stream meters fill the otherwise-dead bar."""
    screen._loading_active = True
    screen.progress.progress_bar.value = 0.0

    screen._on_tqdm_progress(
        {"desc": "Decoding labeled cache", "current": 250, "total": 1000, "timing": "00:30<01:30"}
    )
    assert screen.progress.progress_bar.value == pytest.approx(0.25)
    assert "Decoding labeled cache: 250/1000" in screen.progress.status_label.value


def test_phase_hint_surfaces_in_secondary_caption_not_status_label(screen):
    """Non-tqdm phase lines go to the small secondary caption (cleaned of the
    relayed subprocess loguru header), NOT the batch-progress status label —
    otherwise a gallery catalogue load clobbers "Processing batches: X/Y"."""
    screen.progress.status_label.value = "Processing batches: 71/93 [05:49<01:45]"
    screen._on_phase_hint(
        "[subprocess] 2026-06-22 16:20:47.000 | INFO     | __main__:main:138 - "
        "Warming up GPU for inference (cuDNN autotuning)..."
    )
    assert "Warming up GPU for inference (cuDNN autotuning)..." in screen._preview_hint_html.value
    # Batch progress is left untouched.
    assert screen.progress.status_label.value == "Processing batches: 71/93 [05:49<01:45]"


def test_tqdm_progress_clears_secondary_hint(screen):
    """A live meter tick clears the secondary phase hint so it doesn't linger."""
    screen._preview_hint_html.value = "<span>Loaded 3 FITS files (mmap)</span>"
    screen._on_tqdm_progress(
        {"desc": "Processing batches", "current": 5, "total": 10, "timing": "00:05<00:05"}
    )
    assert screen._preview_hint_html.value == ""


def test_training_iteration_releases_bar_from_loading(screen):
    """The first training line clears the loading flag so iteration progress
    owns the bar; later loading-style tqdm events must not move it."""
    screen._loading_active = True

    screen._handle_progress_line({"status": "training", "iteration": 20, "total": 200})
    assert screen._loading_active is False
    assert screen.progress.progress_bar.value == pytest.approx(0.1)

    # A stray tqdm event after iterations begin must not touch the bar.
    screen._on_tqdm_progress(
        {"desc": "Streaming cutouts", "current": 1, "total": 4, "timing": "00:01<00:03"}
    )
    assert screen.progress.progress_bar.value == pytest.approx(0.1)
    # ...but the status label still tracks the phase for transparency.
    assert "Streaming cutouts: 1/4" in screen.progress.status_label.value


def test_handle_progress_done_sets_model_path(screen, mock_session):
    screen._handle_progress_line(
        {"status": "done", "model_path": "/tmp/model.pth", "elapsed": 42.0}
    )
    assert mock_session.cfg.model_path == "/tmp/model.pth"


def test_handle_progress_size_histogram_renders(screen):
    """A size_histogram progress line renders into the (otherwise empty during
    training) histogram slot and makes it visible."""
    screen._histogram_image.layout.display = "none"
    screen._handle_progress_line(
        {
            "status": "size_histogram",
            "edges": [5.0, 50.0, 100.0, 150.0, 200.0],
            "population_counts": [800, 30, 10, 5],
            "sampled_counts": [10, 5, 2, 1],
            "stratified": True,
            "unit": "pixel",
        }
    )
    assert screen._histogram_image.value  # PNG bytes set
    assert screen._histogram_image.value[:8] == b"\x89PNG\r\n\x1a\n"
    assert screen._histogram_image.layout.display == ""


def test_handle_progress_size_histogram_empty_is_noop(screen):
    """An all-zero histogram leaves the slot hidden rather than drawing blank."""
    screen._histogram_image.layout.display = "none"
    screen._histogram_image.value = b""
    screen._handle_progress_line(
        {
            "status": "size_histogram",
            "edges": [5.0, 10.0, 15.0, 20.0],
            "population_counts": [100, 20, 5],
            "sampled_counts": [0, 0, 0],
            "stratified": False,
            "unit": "pixel",
        }
    )
    assert screen._histogram_image.value == b""
    assert screen._histogram_image.layout.display == "none"


def test_score_histogram_uses_log_y(screen):
    """The training screen scores anomalies, whose distribution is sharply
    peaked, so its score histogram opts into the shared base's log y-axis."""
    assert screen._score_histogram_log_y is True


def test_training_complete_starts_scoring(screen, tmp_path):
    progress_file = str(tmp_path / "progress.jsonl")
    with open(progress_file, "w") as f:
        f.write(json.dumps({"status": "done", "model_path": "/tmp/m.pth", "elapsed": 10.5}) + "\n")

    screen._progress_file = progress_file
    screen._process = None

    with patch.object(screen, "_start_scoring") as mock_score:
        screen._on_training_complete()
        mock_score.assert_called_once()


# ── shorten_filename ────────────────────────────────────────────


class TestShortenFilename:
    def test_short_unchanged(self):
        assert shorten_filename("img.png", 25) == "img.png"

    def test_long_with_extension(self):
        result = shorten_filename("very_long_filename_with_many_characters.fits", 25)
        assert len(result) <= 25
        assert result.endswith(".fits")
        assert "..." in result

    def test_no_extension(self):
        result = shorten_filename("a" * 30, 20)
        assert len(result) <= 20
        assert "..." in result


# ── Size-stratification toggle (Cutana only, applies to retrain) ──


def test_stratify_checkbox_disabled_for_image_folder(screen):
    """The grayscale fixture is an image folder, so the toggle is disabled."""
    assert screen._stratify_checkbox.disabled is True


def test_stratify_checkbox_enabled_for_cutana(mock_session):
    mock_session.cfg.data_dir = "/fake/q1_catalogues"
    with (
        patch("IPython.display.display"),
        patch(
            "anomaly_match_ui.screens.training_screen._auto_detect_source_type",
            return_value=DataSourceType.CUTANA,
        ),
        patch(
            "anomaly_match_ui.screens.training_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ),
        patch("anomaly_match_ui.screens.training_screen.TrainingScreen._start_training"),
    ):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training")
        screen = app._current_screen
    assert screen._stratify_checkbox.disabled is False


def test_on_enter_syncs_stratify_value_from_cfg(mock_session):
    """Regression: screens are cached, so a training screen built before the
    setup checkbox was ticked must still reflect cfg on re-entry — otherwise its
    stale-unchecked box clobbers the setup choice on _start_training."""
    mock_session.cfg.data_dir = "/fake/q1_catalogues"
    with (
        patch("IPython.display.display"),
        patch(
            "anomaly_match_ui.screens.training_screen._auto_detect_source_type",
            return_value=DataSourceType.CUTANA,
        ),
        patch(
            "anomaly_match_ui.screens.training_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ),
        patch("anomaly_match_ui.screens.training_screen.TrainingScreen._start_training"),
    ):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training")
        screen = app._current_screen
        # Screen was built with cfg False → stale unchecked box.
        assert screen._stratify_checkbox.value is False
        # Setup later sets the flag; re-entering must sync the box to True.
        mock_session.cfg.cutana_stratify_source_size = True
        screen.on_enter()
    assert screen._stratify_checkbox.value is True
    assert screen._stratify_checkbox.disabled is False


def test_on_enter_forces_stratify_false_for_non_cutana(mock_session):
    """A stale True must not survive on an image/Zarr source."""
    mock_session.cfg.cutana_stratify_source_size = True  # e.g. left over in cfg
    with patch("IPython.display.display"):
        app = AnomalyMatchApp(mock_session)  # grayscale data_dir → image folder
        app.navigate_to("training")
        screen = app._current_screen
    assert screen._stratify_checkbox.value is False
    assert screen._stratify_checkbox.disabled is True


def _cutana_training_screen(mock_session):
    mock_session.cfg.data_dir = "/fake/q1_catalogues"
    with (
        patch("IPython.display.display"),
        patch(
            "anomaly_match_ui.screens.training_screen._auto_detect_source_type",
            return_value=DataSourceType.CUTANA,
        ),
        patch(
            "anomaly_match_ui.screens.training_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ),
        patch("anomaly_match_ui.screens.training_screen.TrainingScreen._start_training"),
    ):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training")
        yield app, app._current_screen


def test_training_toggle_writes_cfg_immediately(mock_session):
    """The observer persists a toggle to cfg on change, not only at retrain."""
    gen = _cutana_training_screen(mock_session)
    _app, screen = next(gen)
    assert screen._stratify_checkbox.disabled is False
    screen._stratify_checkbox.value = True
    assert mock_session.cfg.cutana_stratify_source_size is True


def test_training_toggle_survives_navigation_roundtrip(mock_session):
    """A toggle made on the training screen must survive a navigate-away-and-back:
    the observer keeps cfg in sync, so on_enter's resync restores it (not reverts)."""
    gen = _cutana_training_screen(mock_session)
    _app, screen = next(gen)
    screen._stratify_checkbox.value = True  # user toggles on the training screen
    with (
        patch(
            "anomaly_match_ui.screens.training_screen._auto_detect_source_type",
            return_value=DataSourceType.CUTANA,
        ),
        patch(
            "anomaly_match_ui.screens.training_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ),
        patch("anomaly_match_ui.screens.training_screen.TrainingScreen._start_training"),
    ):
        screen.on_leave()
        screen.on_enter()  # e.g. returning from the image-detail screen
    assert screen._stratify_checkbox.value is True
    assert mock_session.cfg.cutana_stratify_source_size is True


def _run_start_training(screen):
    with (
        patch.object(
            BackendInterface,
            "launch_training_subprocess",
            return_value=(MagicMock(), "/tmp/x", "/tmp/p.jsonl"),
        ),
        patch.object(screen, "_start_stderr_reader"),
        patch.object(screen, "_start_training_poll"),
    ):
        screen._start_training()


def test_retrain_pushes_stratify_flag(screen, mock_session):
    """An enabled, checked toggle reaches cfg so the retrain subprocess applies it."""
    screen._stratify_checkbox.disabled = False
    screen._stratify_checkbox.value = True
    _run_start_training(screen)
    assert mock_session.cfg.cutana_stratify_source_size is True


def test_retrain_forces_stratify_false_when_disabled(screen, mock_session):
    """A checked-but-disabled toggle (non-Cutana) must not leak True into cfg."""
    screen._stratify_checkbox.disabled = True
    screen._stratify_checkbox.value = True
    _run_start_training(screen)
    assert mock_session.cfg.cutana_stratify_source_size is False
