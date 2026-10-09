#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for the PredictionScreen with polling, gallery, and histogram."""

import json
import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.prediction import AnomalyScoreDB
from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.screens.prediction_screen import PredictionScreen, _format_duration
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widgets.gallery_widget import PAGE_SIZE

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_session():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.5
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.prediction_search_dir = None
    session.cfg.output_dir = "/fake/output"
    return session


def _wait_for_filled_cells(screen, expected, timeout=5):
    """Wait for the background executor to populate gallery cells, then assert.

    Metadata queries and image loads run off the UI thread now (non-blocking
    navigation), so cells fill asynchronously after a refresh. Poll until
    *expected* cells carry a filename (or the timeout elapses) and assert the
    count, returning the filled cells for any further checks.

    Args:
        screen: The prediction screen whose ``gallery._cells`` are populated.
        expected: Number of cells expected to fill.
        timeout: Seconds to wait before giving up.

    Returns:
        The list of filled gallery cells.
    """
    deadline = time.time() + timeout
    while (
        time.time() < deadline
        and sum(c._filename is not None for c in screen.gallery._cells) < expected
    ):
        time.sleep(0.01)

    filled = [c for c in screen.gallery._cells if c._filename is not None]
    assert len(filled) == expected
    return filled


# ── Screen build tests ───────────────────────────────────────────


@patch("IPython.display.display")
def test_prediction_screen_builds(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    assert isinstance(app._current_screen, PredictionScreen)
    assert app._current_screen.widget is not None
    assert app._current_screen.gallery is not None
    assert len(app._current_screen.gallery._cells) == PAGE_SIZE
    assert app._current_screen.start_btn.description == "Start Prediction"


# ── Polling tests ────────────────────────────────────────────────


@patch("IPython.display.display")
def test_polling_thread_starts_and_stops(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    screen._start_polling()
    assert screen._polling is True
    assert screen._poll_thread is not None
    assert screen._poll_thread.is_alive()

    screen._stop_polling()
    assert screen._polling is False


@patch("IPython.display.display")
def test_poll_db_update_reads_count(mock_display, mock_session, tmp_path):
    """_poll_db_update reads count from the DB and updates stats."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.store_results([("img1.png", 0.8), ("img2.png", 0.6), ("img3.png", 0.4)])
    db.close()

    BackendInterface.open_prediction_monitor(str(db_path), str(tmp_path))
    screen._start_time = 1000.0
    screen._last_db_count = 0

    with patch("time.time", return_value=1010.0):
        screen._poll_db_update()

    assert screen._last_db_count == 3
    assert "3" in screen.stats_html.value

    BackendInterface.close_prediction_monitor()


# ── Gallery refresh tests ────────────────────────────────────────


@patch("IPython.display.display")
def test_refresh_gallery_populates_cells(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    for i in range(5):
        db.store_results([(f"img_{i}.png", 0.9 - i * 0.1)])
    db.close()

    BackendInterface.open_prediction_monitor(str(db_path), str(tmp_path))

    screen._refresh_gallery()

    # Metadata + images load on the background executor now (non-blocking nav).
    _wait_for_filled_cells(screen, 5)

    BackendInterface.close_prediction_monitor()


@patch("IPython.display.display")
def test_refresh_reuses_polled_count_without_blocking_count_query(
    mock_display, mock_session, tmp_path
):
    """A page refresh reuses the poll-maintained count instead of a synchronous
    COUNT(*) on the UI thread, which can stall for seconds on a large,
    actively-written DB and freeze the gallery on every page click."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    for i in range(5):
        db.store_results([(f"img_{i}.png", 0.9 - i * 0.1)])
    db.close()
    BackendInterface.open_prediction_monitor(str(db_path), str(tmp_path))

    # The poll loop has already established the count.
    screen._last_db_count = 5

    with patch.object(BackendInterface, "get_prediction_count") as count_mock:
        screen._refresh_gallery()
        count_mock.assert_not_called()

    # The metadata query and image load now run on the background executor so
    # the click never blocks on the DB; wait for that task to populate cells.
    _wait_for_filled_cells(screen, 5)

    BackendInterface.close_prediction_monitor()


# ── Non-blocking navigation & window prefetch ────────────────────


@patch("IPython.display.display")
class TestNonBlockingGallery:
    """The gallery flips pages instantly and warms neighbouring pages."""

    def _screen(self, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("prediction")
        return app._current_screen

    def test_user_refresh_paints_spinners_via_begin_page(self, _mock_display, mock_session):
        """A user navigation flips the page synchronously with spinners
        (``begin_page``) and pushes the DB query + image load to the executor,
        so the click never blocks on the DB."""
        screen = self._screen(mock_session)
        screen._last_db_count = 30
        with (
            patch.object(screen.gallery, "begin_page") as begin_mock,
            patch.object(screen.gallery, "set_pagination") as set_pag_mock,
            patch.object(screen._executor, "submit") as submit_mock,
            patch.object(screen, "_start_prefetch_window") as prefetch_mock,
        ):
            screen._refresh_gallery(user_initiated=True)

        begin_mock.assert_called_once()
        set_pag_mock.assert_not_called()
        # The metadata query + image load are deferred to the executor.
        assert submit_mock.call_args.args[0] == screen._load_page
        prefetch_mock.assert_called_once()

    def test_poll_refresh_updates_counter_without_spinner_flash(self, _mock_display, mock_session):
        """A background poll refresh updates the counter via ``set_pagination``,
        never calls ``begin_page`` (no spinner flash every interval), and never
        kicks off a prefetch — otherwise every poll during active scoring would
        flood the main process with Cutana reads that contend with the scoring
        subprocess for NFS bandwidth."""
        screen = self._screen(mock_session)
        screen._last_db_count = 30
        with (
            patch.object(screen.gallery, "begin_page") as begin_mock,
            patch.object(screen.gallery, "set_pagination") as set_pag_mock,
            patch.object(screen._executor, "submit"),
            patch.object(screen, "_start_prefetch_window") as prefetch_mock,
        ):
            screen._refresh_gallery(user_initiated=False)

        set_pag_mock.assert_called_once()
        begin_mock.assert_not_called()
        prefetch_mock.assert_not_called()

    def test_next_button_click_triggers_prefetch(self, _mock_display, mock_session):
        """Clicking Next through the real gallery button path
        (``_change_page`` -> ``_on_page_change`` -> ``_refresh_gallery``) must
        submit a prefetch anchored at the new page.  Guards the wiring that
        makes look-ahead paging work from silently breaking."""
        screen = self._screen(mock_session)
        screen._last_db_count = 80
        screen._current_page = 0
        screen.gallery._total_pages = 8
        with (
            patch.object(screen, "_start_prefetch_window") as prefetch_mock,
            patch.object(screen._executor, "submit"),
        ):
            screen.gallery._change_page(1)

        prefetch_mock.assert_called_once()
        assert prefetch_mock.call_args.args[0] == 1  # anchored at the newly-shown page

    def test_prefetch_window_targets_forward_biased_pages(self, _mock_display, mock_session):
        """The prefetch warms the current page, five ahead, and one behind, in
        the current sort order — so a burst of Next clicks stays cached."""
        screen = self._screen(mock_session)
        screen._prefetch_generation = 7
        seen_pages = []
        seen_sorts = []

        def fake_results(sort_by, limit, offset):
            seen_sorts.append(sort_by)
            seen_pages.append(offset // limit)
            return []

        with patch.object(BackendInterface, "get_prediction_results", side_effect=fake_results):
            screen._do_prefetch_window(
                page=2, page_size=10, sort_mode="score_asc", count=200, gen=7
            )

        # Current page (2) is skipped — the foreground load owns it; window is
        # the 5 pages ahead plus the one behind.
        assert seen_pages == [3, 4, 5, 6, 7, 1]
        assert set(seen_sorts) == {"score_asc"}

    def test_prefetch_window_skips_out_of_range_pages(self, _mock_display, mock_session):
        """Near the start/end the window clamps to valid pages only."""
        screen = self._screen(mock_session)
        screen._prefetch_generation = 1
        seen_pages = []

        def fake_results(sort_by, limit, offset):
            seen_pages.append(offset // limit)
            return []

        with patch.object(BackendInterface, "get_prediction_results", side_effect=fake_results):
            # page 0 of a 3-page set: no page -1, and 3,4,5 don't exist, so only
            # the two pages ahead (1, 2) are warmed.
            screen._do_prefetch_window(
                page=0, page_size=10, sort_mode="score_desc", count=25, gen=1
            )

        assert seen_pages == [1, 2]

    def test_prefetch_window_bails_on_stale_generation(self, _mock_display, mock_session):
        """A prefetch for a window the user has left aborts before querying."""
        screen = self._screen(mock_session)
        screen._prefetch_generation = 8
        with patch.object(BackendInterface, "get_prediction_results") as query_mock:
            # gen=1 is stale (live generation is 8) → return before any query.
            screen._do_prefetch_window(
                page=0, page_size=10, sort_mode="score_desc", count=200, gen=1
            )
        query_mock.assert_not_called()

    def test_warm_cache_batch_loads_uncached_cutana(self, _mock_display, mock_session):
        """Cutana misses (which return ``None`` from the cache) are batch-loaded
        then inserted, so the neighbouring pages are warm for the next click."""
        screen = self._screen(mock_session)
        screen._file_type = DataSourceType.CUTANA
        screen._prefetch_generation = 3
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch.object(BackendInterface, "load_prediction_image", return_value=None),
            patch.object(
                BackendInterface, "load_cutana_batch", return_value={"a": img, "b": img}
            ) as batch_mock,
            patch.object(BackendInterface, "cache_prediction_image") as cache_mock,
        ):
            screen._warm_cache(["a", "b"], gen=3)

        batch_mock.assert_called_once_with(["a", "b"])
        assert cache_mock.call_count == 2

    def test_warm_cache_skips_already_cached(self, _mock_display, mock_session):
        """Cached filenames are not re-loaded."""
        screen = self._screen(mock_session)
        screen._file_type = DataSourceType.CUTANA
        screen._prefetch_generation = 2
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        with (
            patch.object(BackendInterface, "load_prediction_image", return_value=img),
            patch.object(BackendInterface, "load_cutana_batch") as batch_mock,
        ):
            screen._warm_cache(["a", "b"], gen=2)

        batch_mock.assert_not_called()

    def test_start_prefetch_window_bumps_generation_and_submits(self, _mock_display, mock_session):
        screen = self._screen(mock_session)
        screen._prefetch_generation = 0
        with patch.object(screen._prefetch_executor, "submit") as submit_mock:
            screen._start_prefetch_window(0, "score_desc", 100)
        assert screen._prefetch_generation == 1
        submit_mock.assert_called_once()

    def test_start_prefetch_window_noop_when_no_results(self, _mock_display, mock_session):
        screen = self._screen(mock_session)
        with patch.object(screen._prefetch_executor, "submit") as submit_mock:
            screen._start_prefetch_window(0, "score_desc", 0)
        submit_mock.assert_not_called()

    def test_unavailable_image_resolves_to_no_preview_not_eternal_spinner(
        self, _mock_display, mock_session
    ):
        """An image that can't be loaded must end as "No preview", never as a
        spinner that spins forever — the loading state is only for pending
        loads, not permanent failures."""
        screen = self._screen(mock_session)
        screen._file_type = DataSourceType.IMAGE_FOLDER
        screen._image_load_page = 0
        # Put a cell into the loading (spinner) state as begin_page would.
        screen.gallery._cells[0].mark_loading("missing.png", 0.5)
        assert "am-spinner" in screen.gallery._cells[0]._no_image_html.value

        with patch.object(BackendInterface, "load_prediction_image", return_value=None):
            screen._load_page_images([{"filename": "missing.png", "score": 0.5}], page=0)

        html = screen.gallery._cells[0]._no_image_html.value
        assert "No preview" in html
        assert "am-spinner" not in html


# ── Duration formatting ─────────────────────────────────────────


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(125) == "2m 05s"

    def test_hours(self):
        assert _format_duration(3661) == "1h 01m"


# ── Progress callback tests ──────────────────────────────────────


@patch("IPython.display.display")
def test_progress_completed(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    screen._update_progress(1, 1, completed=True, total_time_str="5m 30s", final_speed=100.0)
    assert screen.progress.progress_bar.value == 1.0


# ── Stop feedback ──────────────────────────────────────────────


@patch("IPython.display.display")
def test_stop_shows_immediate_feedback(mock_display, mock_session):
    """Stop click flips the button label + status banner before signalling
    the subprocess, so the UI doesn't appear frozen (#434)."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    screen._running = True

    captured: dict = {}

    def _fake_request_stop() -> None:
        captured["btn_disabled"] = screen.stop_btn.disabled
        captured["btn_label"] = screen.stop_btn.description
        captured["status"] = screen.status_html.value

    with patch.object(BackendInterface, "request_prediction_stop", side_effect=_fake_request_stop):
        screen._on_stop()

    assert captured["btn_disabled"] is True
    assert captured["btn_label"] == "Stopping…"
    assert "Cancelling prediction subprocess" in captured["status"]
    assert screen._stop_requested is True


# ── Resume loading test ────────────────────────────────────────


@patch("IPython.display.display")
def test_on_enter_loads_from_existing_db(mock_display, mock_session, tmp_path):
    """on_enter auto-detects existing predictions.db and loads results."""
    mock_session.cfg.output_dir = str(tmp_path)
    mock_session.cfg.prediction_search_dir = str(tmp_path)

    db_path = tmp_path / "predictions.db"
    db = AnomalyScoreDB(str(db_path))
    db.store_results([("img1.png", 0.9), ("img2.png", 0.5)])
    db.close()

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("prediction")
    screen = app._current_screen

    assert screen._db_path == str(db_path)
    assert screen._last_db_count == 2

    BackendInterface.close_prediction_monitor()


# ── Profiler stats tests ────────────────────────────────────────


@patch("IPython.display.display")
def test_load_profiler_stats(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    mock_session.cfg.output_dir = str(tmp_path)
    app.navigate_to("prediction")
    screen = app._current_screen

    report = {
        "total_images": 1000,
        "total_wall_clock_s": 25.0,
        "throughput_images_per_sec": 42.5,
        "stages": {
            "io_load": {"total_s": 5.0, "percentage": 33.3},
            "inference": {"total_s": 10.0, "percentage": 66.7},
        },
        "peak_memory": {"rss_mb": 512.0, "gpu_allocated_mb": 256.0, "gpu_reserved_mb": 300.0},
    }
    report_path = os.path.join(str(tmp_path), "performance_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f)

    screen._load_profiler_stats()
    assert "Profiler" in screen.profiler_html.value
    assert "42.5" in screen.profiler_html.value
