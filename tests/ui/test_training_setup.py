#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for training setup screen — edge cases and CSV validation.

Happy-path tests (rendering, auto-validation, navigation) are covered
by browser tests in tests/browser/test_training_setup.py.
"""

import math
import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import zarr
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.data_io.source_scanning import SourceScanResult
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.utils.get_default_cfg import get_default_cfg
from anomaly_match.utils.validate_config import serialisable_config
from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.screens.training_setup_screen import (
    _strip_image_extension,
    _validate_label_csv,
)
from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.chooser_paths import initial_dir, initial_file, resolved_dir
from anomaly_match_ui.widgets.labelable_thumbnail_cell import LabelState

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_session():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.num_train_iter = 200
    session.cfg.cutana_stratify_source_size = False
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [128, 128]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.normalisation.interpolation_order = 1
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.0
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.label_file = None
    session.cfg.prediction_search_dir = None
    session.cfg.output_dir = "/fake/output"
    return session


# ── CSV validation tests ─────────────────────────────────────────


class TestValidateLabelCSV:
    """Tests for the _validate_label_csv helper."""

    def test_valid_csv(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,normal\nc.jpg,removed\n")
        valid, msg, df = _validate_label_csv(str(csv_path))
        assert valid is True
        assert "1 anomaly" in msg
        assert "1 normal" in msg
        assert df is not None
        assert len(df) == 3

    def test_missing_columns(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("name,type\na.jpg,anomaly\n")
        valid, msg, _ = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "Missing columns" in msg

    def test_invalid_labels(self, tmp_path):
        csv_path = tmp_path / "bad_labels.csv"
        csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,weird\n")
        valid, msg, _ = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "Invalid labels" in msg

    def test_unreadable_file(self, tmp_path):
        csv_path = tmp_path / "corrupt.csv"
        csv_path.write_bytes(b"\x00\x01\x02")
        valid, msg, _ = _validate_label_csv(str(csv_path))
        assert isinstance(valid, bool)


# ── Chooser path resolution: persisted state vs cfg (#429) ───────


class TestChooserPathResolution:
    """Covers both ``initial_dir`` and ``initial_file``.

    ``ui_state`` is redirected to tmp_path by the autouse conftest fixture.
    """

    def test_persisted_wins_over_cfg(self, tmp_path):
        """The persisted last-browsed dir is more current than cfg.

        Default cfg values (e.g. ``tests/test_data/grayscale``) would
        otherwise clobber the user's actual recent intent on every
        fresh kernel start.
        """
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        ui_state.set_last_browsed_dir("source_chooser", str(persisted))

        result = initial_dir(str(cfg_dir), purpose="source_chooser")
        assert result == str(persisted)

    def test_cfg_used_when_no_persisted(self, tmp_path):
        """First-run fallback: nothing persisted → use the cfg value."""
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()

        result = initial_dir(str(cfg_dir), purpose="source_chooser")
        assert result == str(cfg_dir)

    def test_persisted_used_when_cfg_missing(self, tmp_path):
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        ui_state.set_last_browsed_dir("source_chooser", str(persisted))

        result = initial_dir(None, purpose="source_chooser")
        assert result == str(persisted)

    def test_unresolved_reports_none_while_initial_dir_falls_back_to_cwd(
        self, tmp_path, monkeypatch
    ):
        """Callers must be able to tell the cwd fallback from a real resolution."""
        monkeypatch.chdir(tmp_path)
        missing = str(tmp_path / "not_created_yet")

        assert resolved_dir(missing, purpose="output_chooser") is None
        assert os.path.realpath(initial_dir(missing, purpose="output_chooser")) == (
            os.path.realpath(tmp_path)
        )

    @pytest.mark.parametrize("persisted_holds_file", [False, True])
    def test_shows_the_configured_file_not_a_same_named_stranger(
        self, tmp_path, persisted_holds_file
    ):
        """The chooser must display the configured file itself, never a namesake.

        With ``persisted_holds_file`` the persisted dir holds a *different*
        ``labeled_data.csv``; without it the dir lacks one entirely.  In both
        cases the configured file wins and is shown in its own directory, so the
        displayed path is the one the screen goes on to load.  Checkpoints make
        the stranger case concrete: ``model_iteration_0.safetensors`` exists in
        every session folder, so opening at the persisted dir and matching on
        name alone would substitute another session's model.
        """
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_label = cfg_dir / "labeled_data.csv"
        cfg_label.write_text("id,label\na.png,anomaly\n")
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        if persisted_holds_file:
            (persisted / "labeled_data.csv").write_text("id,label\nb.png,normal\n")
        ui_state.set_last_browsed_dir("label_chooser", str(persisted))

        directory, filename = initial_file(str(cfg_label), purpose="label_chooser")

        assert directory == str(cfg_dir)
        assert filename == "labeled_data.csv"
        assert os.path.samefile(os.path.join(directory, filename), cfg_label)

    def test_preselects_when_persisted_dir_holds_the_configured_file(self, tmp_path):
        """The common case: the user selected the file, so the persisted dir *is*
        its directory and both rules agree on the pre-selection."""
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_label = cfg_dir / "labeled_data.csv"
        cfg_label.write_text("id,label\na.png,anomaly\n")
        ui_state.set_last_browsed_dir("label_chooser", str(cfg_dir))

        directory, filename = initial_file(str(cfg_label), purpose="label_chooser")

        assert directory == str(cfg_dir)
        assert filename == "labeled_data.csv"

    def test_no_preselection_when_cfg_file_is_gone(self, tmp_path):
        """A deleted cfg path pre-selects nothing, even when the persisted dir
        holds a same-named file: with no configured file to display, the chooser
        falls back to where the user was last browsing and stays unselected."""
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_label = cfg_dir / "labeled_data.csv"  # never created
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        (persisted / "labeled_data.csv").write_text("id,label\nb.png,normal\n")
        ui_state.set_last_browsed_dir("label_chooser", str(persisted))

        directory, filename = initial_file(str(cfg_label), purpose="label_chooser")

        assert directory == str(persisted)
        assert filename == ""

    def test_preselects_cfg_file_when_nothing_persisted(self, tmp_path):
        """First run: cfg file exists in its own dir, so it is pre-selected."""
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_label = cfg_dir / "labeled_data.csv"
        cfg_label.write_text("id,label\na.png,anomaly\n")

        directory, filename = initial_file(str(cfg_label), purpose="label_chooser")

        assert directory == str(cfg_dir)
        assert filename == "labeled_data.csv"

    @pytest.mark.parametrize("cfg_value", ["", None])
    def test_no_preselection_when_cfg_value_is_empty_or_none(
        self, tmp_path, monkeypatch, cfg_value
    ):
        """cfg.metadata_file defaults to None, so None must be handled too."""
        monkeypatch.chdir(tmp_path)
        directory, filename = initial_file(cfg_value, purpose="metadata_chooser")

        # ``getcwd`` resolves symlinks, so compare realpaths — on macOS
        # tmp_path is /var/... while getcwd() reports /private/var/...
        assert os.path.realpath(directory) == os.path.realpath(tmp_path)
        assert filename == ""

    def test_shipped_default_yields_to_remembered_folder(self, tmp_path):
        """The bundled default always exists, so it must not outrank the user's folder."""
        default_label = get_default_cfg().label_file
        persisted = tmp_path / "my_labels"
        persisted.mkdir()
        (persisted / "labeled_data.csv").write_text("id,label\na.png,anomaly\n")
        ui_state.set_last_browsed_dir("label_chooser", str(persisted))

        directory, filename = initial_file(
            default_label, purpose="label_chooser", is_shipped_default=True
        )

        assert directory == str(persisted)
        assert filename == "labeled_data.csv"

    def test_shipped_default_not_composed_into_a_phantom_path(self, tmp_path):
        """A remembered folder without the default's filename selects nothing."""
        persisted = tmp_path / "elsewhere"
        persisted.mkdir()
        ui_state.set_last_browsed_dir("label_chooser", str(persisted))

        directory, filename = initial_file(
            get_default_cfg().label_file, purpose="label_chooser", is_shipped_default=True
        )

        assert directory == str(persisted)
        assert filename == ""

    def test_shipped_default_preselected_when_nothing_remembered(self):
        """First run with the default config still opens on the bundled labels."""
        default_label = get_default_cfg().label_file

        directory, filename = initial_file(
            default_label, purpose="label_chooser", is_shipped_default=True
        )

        assert os.path.join(directory, filename) == default_label

    @patch("IPython.display.display")
    def test_screen_prefers_remembered_labels_over_shipped_default(
        self, _mock_display, mock_session, tmp_path
    ):
        """Screen-level: default-config paths plus a remembered label folder.

        With the bundled test CSV in ``cfg.label_file`` the chooser and
        ``_resolve_label_path`` must both use the user's remembered file, or
        training silently runs on the test labels.
        """
        defaults = get_default_cfg()
        my_labels = tmp_path / "my_labels"
        my_labels.mkdir()
        user_label = my_labels / "labeled_data.csv"
        user_label.write_text("id,label\na.png,anomaly\nb.png,normal\n")
        ui_state.set_last_browsed_dir("label_chooser", str(my_labels))

        mock_session.cfg.data_dir = defaults.data_dir
        mock_session.cfg.label_file = defaults.label_file

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        assert screen._label_chooser.selected == str(user_label)
        assert screen._resolve_label_path() == str(user_label)

    @patch("IPython.display.display")
    def test_screen_displays_the_label_file_it_loads(self, _mock_display, mock_session, tmp_path):
        """Screen-level regression: displayed path == loaded file.

        The helper tests cover the resolution rule; this pins the wiring in
        ``build()``, where the phantom path was composed.  Both earlier
        behaviours failed here: composing persisted + cfg basename displayed a
        file that does not exist, and suppressing the pre-selection left the
        widget empty while ``_resolve_label_path`` still loaded cfg.
        """
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_label = cfg_dir / "labeled_data.csv"
        cfg_label.write_text("id,label\na.png,anomaly\nb.png,normal\n")
        # Persisted dir holds no labeled_data.csv — the pre-fix code composed
        # persisted/labeled_data.csv and displayed it anyway.
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        ui_state.set_last_browsed_dir("label_chooser", str(persisted))

        mock_session.cfg.data_dir = str(cfg_dir)
        mock_session.cfg.label_file = str(cfg_label)

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        assert screen._label_chooser.selected == str(cfg_label)
        assert screen._resolve_label_path() == str(cfg_label)

    @patch("IPython.display.display")
    def test_screen_metadata_row_matches_the_chooser(self, _mock_display, mock_session, tmp_path):
        """The metadata row has no cfg fallback, so it must track the chooser.

        ``_on_start`` writes only what the chooser holds, so a panel line fed
        from cfg would advertise a file the user never selected.
        """
        cfg_dir = tmp_path / "cfg_dir"
        cfg_dir.mkdir()
        cfg_meta = cfg_dir / "metadata.csv"
        cfg_meta.write_text("id,ra,dec\na.png,1.0,2.0\n")
        persisted = tmp_path / "persisted"
        persisted.mkdir()
        ui_state.set_last_browsed_dir("metadata_chooser", str(persisted))

        mock_session.cfg.data_dir = str(cfg_dir)
        mock_session.cfg.metadata_file = str(cfg_meta)

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        assert screen._meta_chooser.selected == str(cfg_meta)
        assert "Metadata: metadata.csv" in screen._render_validation()

    @patch("IPython.display.display")
    def test_full_flow_persists_through_screen_chooser_select(
        self, _mock_display, mock_session, tmp_path
    ):
        """End-to-end integration regression for #429.

        Mimics the user's repro: build the setup screen, drive the
        source chooser the same way ipyfilechooser drives it on a
        Select click, then build a *fresh* screen with a default cfg
        that points at a different folder — confirm the new chooser's
        initial path is the previously-selected one, not the cfg
        default.
        """
        # 1. First session: cfg points at the default test_data path.
        mock_session.cfg.data_dir = str(tmp_path / "test_data" / "grayscale")
        os.makedirs(mock_session.cfg.data_dir, exist_ok=True)

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        # 2. User browses to a different folder and clicks Select.
        q1_data = tmp_path / "q1_data"
        q1_data.mkdir()
        screen._source_chooser._file_chooser._selected_path = str(q1_data)
        screen._source_chooser._file_chooser._selected_filename = ""
        screen._source_chooser._dispatch_callbacks(screen._source_chooser._file_chooser)

        # 3. Persistence captured the selection.
        assert ui_state.get_last_browsed_dir("source_chooser") == str(q1_data)

        # 4. Simulate "restart AnomalyMatch": fresh app/session with
        #    cfg back to its default data_dir.
        mock_session2 = MagicMock()
        mock_session2.cfg = MagicMock()
        mock_session2.cfg.data_dir = str(tmp_path / "test_data" / "grayscale")
        mock_session2.cfg.net = "test-cnn"
        mock_session2.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
        mock_session2.cfg.normalisation.image_size = [128, 128]
        mock_session2.cfg.normalisation.n_output_channels = 3
        mock_session2.cfg.num_channels = 3
        mock_session2.cfg.num_train_iter = 200
        mock_session2.cfg.cutana_stratify_source_size = False
        mock_session2.cfg.metadata_file = None
        mock_session2.cfg.label_file = None
        mock_session2.cfg.output_dir = "/fake/output"

        app2 = AnomalyMatchApp(mock_session2)
        app2.navigate_to("training_setup")
        screen2 = app2._current_screen

        # 5. The fresh chooser opened at the persisted q1_data folder,
        #    not the cfg's default test_data subfolder.
        chooser_initial_path = screen2._source_chooser._file_chooser._default_path
        assert chooser_initial_path == str(q1_data)

        # 6. Auto-validation also walks the persisted dir, not cfg —
        #    otherwise the chooser would show q1_data while the
        #    "Found N images" panel reported the cfg-default scan
        #    against test_data (#429 follow-up).
        assert screen2._source_initial_path == str(q1_data)

    @patch("IPython.display.display")
    def test_resolve_label_path_falls_through_invalid_chooser(
        self, _mock_display, mock_session, tmp_path
    ):
        """Regression test for #429 follow-up.

        ``_read_label_map`` used to short-circuit on
        ``chooser.selected or cfg.label_file`` and never fall through
        to cfg when the chooser held a truthy-but-invalid path (e.g.
        the persisted dir + cfg basename combination resolved to a
        non-existent file).  The new ``_resolve_label_path`` helper
        skips invalid candidates so the cfg fallback can still kick in.
        """
        cfg_csv = tmp_path / "labels.csv"
        cfg_csv.write_text("id,label\nabc,anomaly\nxyz,normal\n", encoding="utf-8")

        mock_session.cfg.label_file = str(cfg_csv)
        mock_session.cfg.data_dir = str(tmp_path)

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        # Simulate the chooser holding a stale, non-existent path.
        screen._label_chooser._file_chooser._selected_path = str(tmp_path)
        screen._label_chooser._file_chooser._selected_filename = "ghost.csv"

        resolved = screen._resolve_label_path()
        assert resolved == str(cfg_csv)


# ── Gallery preview + label edits (#427) ─────────────────────────


class TestSetupGalleryPreview:
    @patch("IPython.display.display")
    def test_gallery_starts_hidden(self, _mock_display, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        assert screen._preview_gallery.widget.layout.display == "none"

    @patch("IPython.display.display")
    def test_gallery_setup_mode_hides_sort_bar(self, _mock_display, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        # The sort bar lives inside the controls bar; setup_mode must hide it.
        controls_bar = screen._preview_gallery.widget.children[0]
        sort_bar = controls_bar.children[0]
        assert sort_bar.layout.display == "none"

    @patch("IPython.display.display")
    def test_zarr_preview_strips_extension_from_display_only(self, _mock_display, mock_session):
        """Zarr ids may carry a pre-conversion extension (e.g. ``.jpeg``);
        the gallery caption must drop it, but the underlying id used for
        labeling/magnify must stay untouched."""
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._detected_source_type = DataSourceType.ZARR
        screen._preview_samples = [("4231499475.jpeg", b"\x89PNG", "")]
        screen._render_preview_page(0)

        cell = screen._preview_gallery._cells[0]
        assert cell._cell._filename == "4231499475.jpeg"
        assert ".jpeg" not in cell._cell._label.value
        assert "4231499475" in cell._cell._label.value

    @patch("IPython.display.display")
    def test_image_folder_preview_keeps_extension(self, _mock_display, mock_session):
        """Image-folder ids are real filenames -- the extension is accurate
        and must still be shown."""
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._detected_source_type = DataSourceType.IMAGE_FOLDER
        screen._preview_samples = [("img_001.png", b"\x89PNG", "")]
        screen._render_preview_page(0)

        cell = screen._preview_gallery._cells[0]
        assert ".png" in cell._cell._label.value

    @patch("IPython.display.display")
    def test_render_preview_page_paginates(self, _mock_display, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        # Seed enough samples to span several pages at the gallery's
        # current effective page size (slider-aware).
        page_size = screen._preview_gallery.effective_page_size
        n_samples = page_size * 3 + 5
        screen._preview_samples = [(f"img_{i}.png", b"\x89PNG", "") for i in range(n_samples)]
        screen._render_preview_page(0)
        assert screen._preview_gallery.widget.layout.display == ""
        # Pagination state — total_pages reflects all samples at the
        # current effective chunk size, not the static PAGE_SIZE.
        assert screen._preview_gallery._total_pages == math.ceil(n_samples / page_size)

        # Next page renders a different slice without re-decoding.
        screen._render_preview_page(1)
        assert screen._preview_gallery._current_page == 1

    @patch("IPython.display.display")
    def test_scale_change_re_pages_anchored_on_first_offset(self, _mock_display, mock_session):
        """Regression for #432.

        When the user changes the column count, the gallery's effective
        page size changes (page size is ``_ROWS * columns``).
        Pagination must re-page at the new chunk size *anchored on the
        first cell of the previous page* — otherwise ``Next`` would skip
        over cells that no longer fall on the same page.
        """
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        # Park the slider at its widest (most columns) so the effective
        # page size is large; seed enough samples for several pages.
        gallery = screen._preview_gallery
        gallery._scale_slider.value = gallery._scale_slider.max
        large_slider_page_size = gallery.effective_page_size
        screen._preview_samples = [
            (f"img_{i}.png", b"\x89PNG", "") for i in range(large_slider_page_size * 4)
        ]
        # User pages past page 0 so the anchor offset is non-zero.
        screen._render_preview_page(2)
        anchor = screen._preview_first_offset
        assert anchor == 2 * large_slider_page_size

        # Drag the slider to the fewest columns.  Effective page size
        # shrinks; the screen must seek to the page containing the
        # anchor cell, not stay at numeric ``_current_page=2`` under a
        # stale chunk size.
        gallery._scale_slider.value = gallery._scale_slider.min
        small_slider_page_size = gallery.effective_page_size
        assert small_slider_page_size < large_slider_page_size
        assert screen._preview_first_offset // small_slider_page_size == gallery._current_page
        # The first rendered cell still sits inside the new page.
        assert (
            gallery._current_page * small_slider_page_size
            <= anchor
            < (gallery._current_page + 1) * small_slider_page_size
        )

    @patch("IPython.display.display")
    def test_label_edit_recorded_in_setup_labels(self, _mock_display, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        # Every touched cell is recorded, including an explicit UNLABELLED —
        # the map is the authority for a cell's displayed state, so an
        # un-label must persist (and override a CSV badge) across pages.
        screen._on_setup_label_change("img_a.png", LabelState.ANOMALY)
        screen._on_setup_label_change("img_b.png", LabelState.NORMAL)
        screen._on_setup_label_change("img_c.png", LabelState.UNLABELLED)

        assert screen._setup_labels == {
            "img_a.png": LabelState.ANOMALY,
            "img_b.png": LabelState.NORMAL,
            "img_c.png": LabelState.UNLABELLED,
        }

    @patch("IPython.display.display")
    def test_unlabelling_is_recorded_not_dropped(self, _mock_display, mock_session):
        """Re-toggling an edit to UNLABELLED keeps the explicit state.

        Dropping it (the old behaviour) let a CSV-labelled cell fall back
        to its original badge on the next page render.
        """
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._on_setup_label_change("img_a.png", LabelState.ANOMALY)
        screen._on_setup_label_change("img_a.png", LabelState.UNLABELLED)
        assert screen._setup_labels["img_a.png"] == LabelState.UNLABELLED

    @patch("IPython.display.display")
    def test_unlabelled_edit_overrides_csv_badge_across_pages(self, _mock_display, mock_session):
        """Regression: un-labelling a CSV-labelled cell must survive paging.

        The cell starts with a CSV ``anomaly`` badge; the user un-labels it,
        pages away and back, and it must still read UNLABELLED rather than
        reverting to the original CSV label.
        """
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._preview_samples = [("csv_anom.png", b"\x89PNG", "anomaly")]
        screen._render_preview_page(0)
        cell = screen._preview_gallery._cells[0]
        assert cell.label_state == LabelState.ANOMALY  # CSV badge shows first

        screen._on_setup_label_change("csv_anom.png", LabelState.UNLABELLED)
        screen._render_preview_page(0)  # page away and back
        assert cell.label_state == LabelState.UNLABELLED

    @patch("IPython.display.display")
    def test_start_maps_unlabelled_edits_to_removed(self, _mock_display, mock_session):
        """Every un-labelled edit is handed to the merge as ``removed``; the
        backend decides which actually persist (it drops a ``removed`` row whose
        id never carried a label), so the UI does no CSV read here."""
        from anomaly_match.datasets.Label import LABEL_REMOVED

        mock_session.cfg.data_dir = "/fake/data"
        mock_session.cfg.label_file = "/fake/labels.csv"

        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._setup_labels = {
            "had_label.png": LabelState.UNLABELLED,
            "never_labelled.png": LabelState.UNLABELLED,
            "now_anom.png": LabelState.ANOMALY,
        }
        with (
            patch.object(
                BackendInterface, "merge_gallery_labels", return_value="/tmp/merged.csv"
            ) as merge,
            patch.object(app, "navigate_to"),
        ):
            screen._on_start()

        merged = merge.call_args[0][0]
        assert merged == {
            "had_label.png": LABEL_REMOVED,
            "never_labelled.png": LABEL_REMOVED,
            "now_anom.png": "anomaly",
        }

    @patch("IPython.display.display")
    def test_render_preview_page_hides_empty_setup_cells(self, _mock_display, mock_session):
        """In setup_mode the gallery hides cleared cells beyond the
        candidate list — no "Waiting for results" placeholders."""
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._preview_samples = [(f"img_{i}.png", b"\x89PNG", "") for i in range(5)]
        screen._render_preview_page(0)

        cells = screen._preview_gallery._cells
        # First 5 cells visible, the rest hidden.
        for i in range(5):
            assert cells[i].widget.layout.display == ""
        for i in range(5, len(cells)):
            assert cells[i].widget.layout.display == "none"

    @patch("IPython.display.display")
    def test_setup_magnify_navigates_to_detail(self, _mock_display, mock_session):
        """Clicking magnify on a setup cell sets _detail_context with
        back_screen=training_setup and navigates to image_detail (#427)."""
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        raw = np.zeros((32, 32, 3), dtype=np.uint8)
        screen._preview_raw_images = {"img_a.png": raw}

        with patch.object(app, "navigate_to") as nav:
            screen._on_setup_magnify("img_a.png")

        assert nav.call_args[0][0] == "image_detail"
        ctx = app._detail_context
        assert ctx["filename"] == "img_a.png"
        assert ctx["back_screen"] == "training_setup"
        assert ctx["image"] is raw

    def test_empty_csv(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("id,label\n")
        valid, msg, df = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "need at least one" in msg
        assert df is not None
        assert len(df) == 0

    def test_single_class_only_anomaly(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,anomaly\n")
        valid, msg, df = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "normal" in msg
        assert df is not None

    def test_single_class_only_normal(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("id,label\na.jpg,normal\nb.jpg,normal\n")
        valid, msg, df = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "anomaly" in msg
        assert df is not None

    def test_valid_csv_with_id_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,normal\n")
        valid, msg, df = _validate_label_csv(str(csv_path))
        assert valid is True
        assert df is not None
        assert "id" in df.columns

    def test_missing_id_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        csv_path.write_text("name,label\na.jpg,anomaly\n")
        valid, msg, _ = _validate_label_csv(str(csv_path))
        assert valid is False
        assert "id" in msg


# ── Edge case tests (not covered by browser tests) ───────────────


@patch("IPython.display.display")
def test_start_button_initially_disabled(mock_display, mock_session):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen
    assert screen._start_btn.disabled is True


@patch("IPython.display.display")
def test_progress_tap_toggles_cutana_logging(mock_display, mock_session):
    """The setup screen opts into cutana's (import-disabled) logger while its
    progress tap is installed, so cutana's cutout/FITS-set hints reach the
    preview spinner, then restores the disabled default on uninstall."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    # on_enter already installed the tap; reset so we can observe a clean cycle.
    screen._uninstall_progress_tap()

    with patch("anomaly_match_ui.screens.training_setup_screen.logger") as mock_logger:
        screen._install_progress_tap()
        mock_logger.enable.assert_called_once_with("cutana")

        # Idempotent install must not enable twice.
        screen._install_progress_tap()
        mock_logger.enable.assert_called_once_with("cutana")

        screen._uninstall_progress_tap()
        mock_logger.disable.assert_called_once_with("cutana")


@patch("IPython.display.display")
def test_source_folder_empty(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    empty_result = SourceScanResult(
        source_type=DataSourceType.IMAGE_FOLDER, count=0, image_files=[]
    )
    with patch(
        "anomaly_match_ui.screens.training_setup_screen.BackendInterface.scan_and_count_sources",
        return_value=empty_result,
    ):
        screen._phase_count_sources(str(empty_dir), stale_check=lambda: False)

    assert screen._source_ok is False
    assert screen._source_count == 0


@patch("IPython.display.display")
def test_label_validation_invalid(mock_display, mock_session, tmp_path):
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col1,col2\nx,y\n")

    screen._validate_label_file(str(csv_path))
    assert screen._labels_ok is False


@patch("IPython.display.display")
def test_phase_validate_labels_caches_result(mock_display, mock_session, tmp_path):
    """Second _phase_validate_labels call with the same inputs must not re-run validation.

    Regression test: norm-only changes triggered a full label re-validation on
    every keystroke, spamming logs and hammering Cutana catalogues.
    """
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,normal\n")

    with patch(
        "anomaly_match_ui.screens.training_setup_screen.BackendInterface"
        ".validate_labels_against_source",
        return_value=("2/2 found", {"a.jpg": ("p", 0), "b.jpg": ("p", 1)}, False),
    ) as mock_validate:
        screen._phase_validate_labels(
            "/fake/source",
            DataSourceType.IMAGE_FOLDER,
            str(csv_path),
            stale_check=lambda: False,
        )
        screen._phase_validate_labels(
            "/fake/source",
            DataSourceType.IMAGE_FOLDER,
            str(csv_path),
            stale_check=lambda: False,
        )

    # Only the first call should hit the backend — second is served from cache
    assert mock_validate.call_count == 1
    assert len(screen._found_source_ids) == 2


@patch("IPython.display.display")
def test_auto_validate_validates_the_chooser_selection(mock_display, mock_session, tmp_path):
    """Re-entry must validate the CSV the panel names, not ``cfg.label_file``.

    The panel renders ``_label_chooser.selected or cfg``, so keying the
    validation on cfg alone showed one filename beside another file's
    anomaly/normal counts once the user picked a CSV without starting a run.
    """
    cfg_dir = tmp_path / "cfg_dir"
    cfg_dir.mkdir()
    cfg_label = cfg_dir / "labeled_data.csv"
    cfg_label.write_text("id,label\na.png,anomaly\n")
    picked_dir = tmp_path / "picked"
    picked_dir.mkdir()
    picked_label = picked_dir / "other_labels.csv"
    picked_label.write_text("id,label\nb.png,normal\n")

    mock_session.cfg.data_dir = str(cfg_dir)
    mock_session.cfg.label_file = str(cfg_label)

    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    # The user picks a different CSV; cfg still holds the old one until Start.
    screen._label_chooser._file_chooser._selected_path = str(picked_dir)
    screen._label_chooser._file_chooser._selected_filename = "other_labels.csv"

    with (
        patch.object(screen, "_start_background_job"),
        patch.object(screen, "_validate_label_file") as validate,
    ):
        screen._auto_validate_signature = None
        screen._auto_validate_from_config()

    validate.assert_called_once_with(str(picked_label))


@patch("IPython.display.display")
def test_auto_validate_skips_on_unchanged_reentry(mock_display, mock_session, tmp_path):
    """Returning to the setup screen with the same source/label/normalisation
    must reuse the held count + preview instead of re-running the full
    count → validate → preview job (the "screen restarts on return" report)."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen
    screen._source_initial_path = str(tmp_path)

    with patch.object(screen, "_start_background_job") as job:
        screen._source_ok = False
        screen._counting = False
        screen._auto_validate_from_config()  # first run: validates + previews
        assert job.call_count == 1

        # Simulate the first validation succeeding, then a pure re-entry with
        # nothing changed — must skip the job.
        screen._source_ok = True
        screen._auto_validate_from_config()
        assert job.call_count == 1


@patch("IPython.display.display")
def test_auto_validate_reruns_when_normalisation_changes(mock_display, mock_session, tmp_path):
    """A normalisation change between entries must re-run the job — the preview
    is decoded with the widget's normalisation, so a stale skip would show the
    old stretch/channel combination."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen
    screen._source_initial_path = str(tmp_path)
    screen._source_ok = True
    screen._counting = False

    with (
        patch.object(screen, "_start_background_job") as job,
        patch.object(
            screen._norm_widget,
            "get_normalisation_config",
            side_effect=[{"m": 1}, {"m": 1}, {"m": 2}],
        ),
    ):
        screen._auto_validate_from_config()  # signature A → runs
        screen._auto_validate_from_config()  # signature A unchanged → skip
        screen._auto_validate_from_config()  # signature B (norm changed) → runs

    assert job.call_count == 2


@patch("IPython.display.display")
def test_source_change_resets_autovalidate_signature(mock_display, mock_session):
    """The production reset wiring: editing the source clears the skip
    signature so the next entry re-validates rather than reusing a stale
    result."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen
    screen._auto_validate_signature = ("stale-signature",)

    with (
        patch.object(screen, "_start_background_job"),
        patch.object(screen, "_refresh_validation"),
    ):
        screen._on_source_change(None)  # no selection → resets, then clears

    assert screen._auto_validate_signature is None


@patch("IPython.display.display")
def test_start_background_job_applies_norm_before_thread_spawn(
    mock_display, mock_session, tmp_path
):
    """Norm overrides must reach cfg synchronously before the background job thread starts.

    Regression test for torn writes across overlapping background jobs:
    each daemon used to call ``_apply_norm_overrides_to_cfg`` itself, so
    two jobs could race on the nested ``setattr`` sequence and leave cfg
    half-written.  ``_start_background_job`` now applies the overrides on
    the caller thread before spawning the daemon, so observers always see
    a consistent config.
    """
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    # Simulate the user reducing output channels to 1
    screen._norm_widget._n_channels = 1
    screen._source_ok = True
    screen._labels_ok = False  # skip Phase 2/3 body — we only test the pre-thread apply

    # Block the daemon from actually running — we only care that the
    # norm apply ran synchronously on this thread before the spawn.
    with patch("threading.Thread"):
        screen._start_background_job(str(tmp_path), skip_counting=True)

    assert mock_session.cfg.normalisation.n_output_channels == 1


@patch("IPython.display.display")
def test_phase_validate_labels_reruns_when_label_file_changes(mock_display, mock_session, tmp_path):
    """Editing the labels CSV must invalidate the validation cache (mtime changes)."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("id,label\na.jpg,anomaly\nb.jpg,normal\n")

    with patch(
        "anomaly_match_ui.screens.training_setup_screen.BackendInterface"
        ".validate_labels_against_source",
        return_value=("ok", {"a.jpg": ("p", 0)}, False),
    ) as mock_validate:
        screen._phase_validate_labels(
            "/fake/source",
            DataSourceType.IMAGE_FOLDER,
            str(csv_path),
            stale_check=lambda: False,
        )
        # Simulate user editing the CSV by bumping mtime
        new_mtime = os.path.getmtime(csv_path) + 10
        os.utime(csv_path, (new_mtime, new_mtime))

        screen._phase_validate_labels(
            "/fake/source",
            DataSourceType.IMAGE_FOLDER,
            str(csv_path),
            stale_check=lambda: False,
        )

    assert mock_validate.call_count == 2


@patch("IPython.display.display")
def test_phase_count_sources_stale_result_does_not_clobber_state(
    mock_display, mock_session, tmp_path
):
    """A slow scan that finishes after its job was superseded must not write state.

    Regression test: when gen=1 scanned a slow image folder and gen=2 had
    already scanned a different Cutana folder, gen=1's late result used to
    clobber ``_detected_source_type`` back to IMAGE_FOLDER.  A subsequent
    skip_counting job (e.g. on normalisation change) then ran the image-
    folder validation path against the Cutana folder and matched 0 labels.
    """
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    # Pretend a newer job has already committed a Cutana result.
    screen._detected_source_type = DataSourceType.CUTANA
    screen._image_files = []
    screen._source_count = 29_951_314
    screen._source_ok = True

    stale_image_folder_result = SourceScanResult(
        source_type=DataSourceType.IMAGE_FOLDER,
        count=19,
        image_files=[str(tmp_path / "a.jpg")],
    )

    with patch(
        "anomaly_match_ui.screens.training_setup_screen.BackendInterface.scan_and_count_sources",
        return_value=stale_image_folder_result,
    ):
        # Job is stale from the moment it starts checking — its result
        # must be discarded rather than overwriting the live state.
        screen._phase_count_sources(str(tmp_path), stale_check=lambda: True)

    assert screen._detected_source_type == DataSourceType.CUTANA, (
        "Stale scan result must not overwrite _detected_source_type"
    )
    assert screen._source_count == 29_951_314
    assert screen._image_files == []


@patch("IPython.display.display")
def test_phase_count_sources_fresh_result_commits_state(mock_display, mock_session, tmp_path):
    """Sanity check: a non-stale scan still applies its result."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    screen = app._current_screen

    (tmp_path / "a.jpg").touch()
    result = SourceScanResult(
        source_type=DataSourceType.IMAGE_FOLDER,
        count=1,
        image_files=[str(tmp_path / "a.jpg")],
    )

    with patch(
        "anomaly_match_ui.screens.training_setup_screen.BackendInterface.scan_and_count_sources",
        return_value=result,
    ):
        screen._phase_count_sources(str(tmp_path), stale_check=lambda: False)

    assert screen._detected_source_type == DataSourceType.IMAGE_FOLDER
    assert screen._source_count == 1
    assert screen._source_ok is True


# ── #501: in-memory raw re-decode on normalisation change ──────────


@patch("IPython.display.display")
class TestNormChangeRedecodeRouting:
    """``_run_norm_refresh`` (the debounced refresh) takes the fast in-memory
    re-decode path only when the preview is fully backed by held raws (#501)."""

    def _screen(self, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        screen._source_ok = True
        screen._suppress_norm_preview = False
        # navigate_to may have kicked off an initial load that set the in-flight
        # flag; clear it so these tests exercise the launch path directly.
        screen._job_in_flight = False
        return screen

    def test_uses_redecode_when_raws_held(self, _disp, mock_session):
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        redecode.assert_called_once()
        full.assert_not_called()

    def test_uses_full_job_when_no_raws(self, _disp, mock_session):
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = None
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        full.assert_called_once()
        assert full.call_args.kwargs["skip_counting"] is True
        redecode.assert_not_called()

    def test_extraction_affecting_edit_forces_full_job(self, _disp, mock_session):
        """Regression: the zoom-out slider appeared to do nothing on a warm cache.

        ``cutout_padding_factor`` changes the field of view Cutana *extracts*, so
        the held raws cannot show it — re-decoding them in memory renders the
        exact same pixels.  Such an edit must drop the raws and take the full
        job, which re-checks the extraction hash and re-reads at the new zoom.
        """
        screen = self._screen(mock_session)
        raws = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._preview_raw_cutouts = raws
        screen._last_norm_config = {"cutout_padding_factor": 1.0}
        with patch.object(
            screen._norm_widget,
            "get_normalisation_config",
            return_value={"cutout_padding_factor": 2.5},
        ):
            with patch.object(screen, "_schedule_norm_refresh"):
                screen._on_norm_change()
        assert screen._norm_refresh_needs_extraction is True

        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        full.assert_called_once()
        redecode.assert_not_called()
        # The stale raws must be released, or the next full job would re-hold them.
        assert screen._preview_raw_cutouts is None
        assert screen._norm_refresh_needs_extraction is False

    def test_real_zoom_out_control_routes_to_full_job(self, _disp, mock_session):
        """End of the chain the stubbed tests can't see: the key the widget
        actually emits must intersect ``EXTRACTION_AFFECTING_NORM_FIELDS``.

        Drives the real control instead of stubbing ``get_normalisation_config``,
        so renaming that key in the widget fails here rather than silently
        bringing the bug back.
        """
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = serialisable_config(
            screen._norm_widget.get_normalisation_config()
        )
        with patch.object(screen, "_schedule_norm_refresh"):
            screen._norm_widget._padding_factor.value = 2.5  # the zoom-out control
        assert screen._norm_refresh_needs_extraction is True

        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        full.assert_called_once()
        redecode.assert_not_called()
        # Asserted here as well as on the stubbed variant: the release matters
        # most on the path a real control actually takes.
        assert screen._preview_raw_cutouts is None

    def test_real_method_control_still_uses_redecode(self, _disp, mock_session):
        """Mirror of the above for a render-only control, driven the same way —
        pins that the fast path is not simply dead for every real edit."""
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = serialisable_config(
            screen._norm_widget.get_normalisation_config()
        )
        with patch.object(screen, "_schedule_norm_refresh"):
            screen._norm_widget._method_dropdown.value = NormalisationMethod.ASINH
        assert screen._norm_refresh_needs_extraction is False

        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        redecode.assert_called_once()
        full.assert_not_called()

    def test_render_only_edit_still_uses_redecode(self, _disp, mock_session):
        """The #501 fast path must survive: a purely render-affecting field
        (normalisation method) still re-decodes the held raws in memory."""
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = {"normalisation_method": "LOG"}
        with patch.object(
            screen._norm_widget,
            "get_normalisation_config",
            return_value={"normalisation_method": "ASINH"},
        ):
            with patch.object(screen, "_schedule_norm_refresh"):
                screen._on_norm_change()
        assert screen._norm_refresh_needs_extraction is False

        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        redecode.assert_called_once()
        full.assert_not_called()

    def test_extraction_flag_survives_a_change_and_revert_mid_drag(self, _disp, mock_session):
        """The flag is set per event, not from the settled diff: nudging the zoom
        out and back still invalidates the raws, because the *cache* read that
        produced them happened before the excursion."""
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = {"cutout_padding_factor": 1.0}
        with patch.object(screen, "_schedule_norm_refresh"):
            for value in (2.0, 1.0):
                with patch.object(
                    screen._norm_widget,
                    "get_normalisation_config",
                    return_value={"cutout_padding_factor": value},
                ):
                    screen._on_norm_change()
        assert screen._norm_refresh_needs_extraction is True

    def test_extraction_flag_set_while_source_is_still_counting(self, _disp, mock_session):
        """The snapshot absorbs the edit before the ``_source_ok`` guard runs.

        ``_on_source_change`` clears ``_source_ok`` and starts a count without
        dropping the held raws, and counting a Cutana catalogue takes seconds.
        A zoom dragged during that window used to update ``_last_norm_config``
        and then return before flagging, so the edit was invisible to every
        later diff: the next render-only edit took the fast path over 1.0x
        pixels while the panel read 2.5x — the reported symptom, one path over.
        """
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = {"cutout_padding_factor": 1.0}
        screen._source_ok = False  # count in flight after a source change

        with (
            patch.object(screen, "_schedule_norm_refresh"),
            patch.object(
                screen._norm_widget,
                "get_normalisation_config",
                return_value={"cutout_padding_factor": 2.5},
            ),
        ):
            screen._on_norm_change()
        assert screen._norm_refresh_needs_extraction is True

        # Count finishes, then a render-only edit: the earlier zoom must still
        # force the full job rather than re-decoding the stale raws.
        screen._source_ok = True
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        full.assert_called_once()
        redecode.assert_not_called()
        assert screen._preview_raw_cutouts is None

    def test_extraction_flag_is_written_under_the_refresh_lock(self, _disp, mock_session):
        """The UI-thread write must take the same lock the timer thread reads under.

        ``_run_norm_refresh`` reads and clears the flag inside
        ``_norm_refresh_lock``.  An unlocked write can land between that read and
        the clear and be dropped, sending the edit it belonged to back to the
        re-decode path over raws extracted at the old zoom.  Held here from the
        main thread, so an unlocked write shows up as the flag flipping while the
        lock is held.
        """
        screen = self._screen(mock_session)
        screen._last_norm_config = {"cutout_padding_factor": 1.0}

        def edit():
            with (
                patch.object(screen, "_schedule_norm_refresh"),
                patch.object(
                    screen._norm_widget,
                    "get_normalisation_config",
                    return_value={"cutout_padding_factor": 2.5},
                ),
            ):
                screen._on_norm_change()

        with screen._norm_refresh_lock:
            worker = threading.Thread(target=edit)
            worker.start()
            worker.join(timeout=0.5)
            assert worker.is_alive(), "the write did not take the lock"
            assert screen._norm_refresh_needs_extraction is False
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert screen._norm_refresh_needs_extraction is True

    def test_deferred_refresh_keeps_the_extraction_flag(self, _disp, mock_session):
        """A refresh deferred behind an in-flight job must not eat the flag.

        ``_run_norm_refresh`` returns early when a job is in flight; that return
        has to happen *before* the flag is read, or the edit loses its
        invalidation and the re-run that ``_on_preview_job_done`` triggers takes
        the re-decode path over raws extracted at the old zoom.
        """
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._last_norm_config = {"cutout_padding_factor": 1.0}
        with (
            patch.object(screen, "_schedule_norm_refresh"),
            patch.object(
                screen._norm_widget,
                "get_normalisation_config",
                return_value={"cutout_padding_factor": 2.5},
            ),
        ):
            screen._on_norm_change()

        screen._job_in_flight = True
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._run_norm_refresh()
        redecode.assert_not_called()
        full.assert_not_called()
        assert screen._refresh_pending is True
        assert screen._norm_refresh_needs_extraction is True

        # The job finishing drains the pending refresh, which must still route
        # to the full job.
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            screen._on_preview_job_done(screen._job_generation)
        full.assert_called_once()
        redecode.assert_not_called()

    def test_norm_change_debounces_into_single_refresh(self, _disp, mock_session):
        """Rapid changes (slider drag) must collapse to one refresh, not one per
        event — otherwise overlapping multi-core decodes thrash the pod."""
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        # Make each _on_norm_change see a genuine diff so it schedules.
        configs = iter([{"image_size": ([64, 64], [80, 80])}] * 5)
        with (
            patch.object(screen, "_run_norm_refresh") as refresh,
            patch(
                "anomaly_match_ui.screens.training_setup_screen.diff_configs",
                side_effect=lambda *a: next(configs, {}),
            ),
            patch.object(screen._norm_widget, "get_normalisation_config", return_value={}),
        ):
            for _ in range(5):
                screen._on_norm_change()  # rapid "drag"
            timer = screen._norm_refresh_timer
            assert timer is not None
            # Only the last timer should be live; let it fire.
            timer.join(2.0)
        refresh.assert_called_once()

    def test_single_flight_defers_while_job_in_flight(self, _disp, mock_session):
        """A refresh arriving while ANY preview job is in flight is deferred, not
        run concurrently; the in-flight job's completion re-runs it once."""
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        with (
            patch.object(screen, "_start_redecode_job") as redecode,
            patch.object(screen, "_start_background_job") as full,
        ):
            # Simulate a preview job (e.g. the initial cache load) already running.
            screen._job_in_flight = True
            screen._job_generation = 5

            screen._run_norm_refresh()
            # Deferred — nothing launched concurrently, re-run queued.
            redecode.assert_not_called()
            full.assert_not_called()
            assert screen._refresh_pending is True

            # The in-flight job (latest generation) completes → drains pending →
            # launches once, now via the fast path (raws are held).
            screen._on_preview_job_done(5)
            redecode.assert_called_once()
            assert screen._refresh_pending is False

    def test_stale_job_completion_does_not_release_slot(self, _disp, mock_session):
        """A superseded job finishing must not hand the slot to a pending refresh
        while the job that replaced it is still running."""
        screen = self._screen(mock_session)
        screen._job_in_flight = True
        screen._job_generation = 7
        screen._refresh_pending = True
        with patch.object(screen, "_run_norm_refresh") as rerun:
            screen._on_preview_job_done(3)  # stale generation
            rerun.assert_not_called()
        assert screen._job_in_flight is True
        assert screen._refresh_pending is True


@patch("IPython.display.display")
class TestRedecodeJob:
    """The re-decode worker decodes held raws and is gen-guarded."""

    def _screen(self, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        return app._current_screen

    def test_applies_decoded_preview(self, _disp, mock_session):
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._job_generation = 7
        decoded = [("id0", np.zeros((4, 4, 3), dtype=np.uint8))]
        with (
            patch(
                "anomaly_match_ui.screens.training_setup_screen.BackendInterface.decode_raw_cutouts",
                return_value=decoded,
            ) as dec,
            patch.object(screen, "_apply_decoded_preview") as apply_,
            patch.object(screen, "_read_label_map", return_value={}),
        ):
            screen._redecode_job(7)
        dec.assert_called_once()
        apply_.assert_called_once_with(decoded, {})

    def test_bails_when_stale(self, _disp, mock_session):
        screen = self._screen(mock_session)
        screen._preview_raw_cutouts = [("id0", np.zeros((4, 4, 3), dtype=np.float32))]
        screen._job_generation = 9  # newer than the gen passed below
        with (
            patch(
                "anomaly_match_ui.screens.training_setup_screen.BackendInterface.decode_raw_cutouts",
            ) as dec,
            patch.object(screen, "_apply_decoded_preview") as apply_,
        ):
            screen._redecode_job(8)
        dec.assert_not_called()
        apply_.assert_not_called()


@patch("IPython.display.display")
class TestPhaseLoadPreviewHold:
    """`_phase_load_preview` holds the raws only when the cache fully backs the
    Cutana preview, and clears them otherwise (#501 review, Finding 5)."""

    def _cutana_screen(self, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        screen._source_ok = True
        screen._detected_source_type = DataSourceType.CUTANA
        screen._found_source_ids = ["id0", "id1"]
        return screen

    def _run(self, screen, holdable):
        decoded = [("id0", np.zeros((4, 4, 3), np.uint8)), ("id1", np.zeros((4, 4, 3), np.uint8))]
        with (
            patch.object(
                screen, "_read_label_map", return_value={"id0": "anomaly", "id1": "normal"}
            ),
            patch(
                "anomaly_match_ui.screens.training_setup_screen.BackendInterface.load_cutana_preview",
                return_value=(decoded, holdable),
            ),
            patch.object(screen, "_apply_decoded_preview") as applied,
        ):
            screen._phase_load_preview(
                "/fake/cutana", DataSourceType.CUTANA, stale_check=lambda: False
            )
        return applied

    def test_holds_raws_on_full_coverage(self, _disp, mock_session):
        screen = self._cutana_screen(mock_session)
        raws = [("id0", np.zeros((8, 8, 4), np.float32)), ("id1", np.zeros((8, 8, 4), np.float32))]
        applied = self._run(screen, holdable=raws)
        assert screen._preview_raw_cutouts is raws
        assert screen._preview_labeled_map == {"id0": "anomaly", "id1": "normal"}
        applied.assert_called_once()

    def test_clears_raws_when_not_holdable(self, _disp, mock_session):
        screen = self._cutana_screen(mock_session)
        # Pretend a previous full preview had held raws.
        screen._preview_raw_cutouts = [("old", np.zeros((8, 8, 4), np.float32))]
        applied = self._run(screen, holdable=None)
        assert screen._preview_raw_cutouts is None
        applied.assert_called_once()


@patch("IPython.display.display")
class TestPhaseLoadPreviewZarrLabelFilter:
    """Zarr previews must only show labeled cutouts, same as image-folder
    and Cutana sources -- not a random sample of the whole store."""

    def test_only_labeled_ids_are_requested(self, _disp, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        screen._source_ok = True
        screen._detected_source_type = DataSourceType.ZARR
        # ``_found_source_ids`` is the output of label validation against
        # the source -- by construction it already contains only ids that
        # are both labeled *and* present in the store (#429 behaviour the
        # Zarr path must reuse rather than falling back to the whole store).
        screen._found_source_ids = ["gal_001", "gal_003"]

        with (
            patch.object(
                screen, "_read_label_map", return_value={"gal_001": "anomaly", "gal_003": "normal"}
            ),
            patch(
                "anomaly_match_ui.screens.training_setup_screen.BackendInterface."
                "load_preview_samples",
                return_value=[],
            ) as load_samples,
        ):
            screen._phase_load_preview("/fake/zarr", DataSourceType.ZARR, stale_check=lambda: False)

        load_samples.assert_called_once()
        passed_ids = load_samples.call_args.kwargs["source_ids"]
        assert set(passed_ids) == {"gal_001", "gal_003"}

    def test_no_labels_requests_empty_list_not_none(self, _disp, mock_session):
        """No label CSV selected -- must request zero ids, not fall back to
        sampling the whole (unlabeled) store."""
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen
        screen._source_ok = True
        screen._detected_source_type = DataSourceType.ZARR
        screen._found_source_ids = []

        with (
            patch.object(screen, "_read_label_map", return_value={}),
            patch(
                "anomaly_match_ui.screens.training_setup_screen.BackendInterface."
                "load_preview_samples",
                return_value=[],
            ) as load_samples,
        ):
            screen._phase_load_preview("/fake/zarr", DataSourceType.ZARR, stale_check=lambda: False)

        load_samples.assert_called_once()
        assert load_samples.call_args.kwargs["source_ids"] == []


# ── Source-size stratification checkbox (Cutana only) ────────────


class TestStratifyCheckbox:
    """The 'Stratify source size per tile' checkbox is Cutana-gated and
    pushes ``cfg.cutana_stratify_source_size`` on Start (PR2 of #size-strat)."""

    @patch("IPython.display.display")
    def test_enabled_only_for_cutana(self, _disp, mock_session):
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._detected_source_type = DataSourceType.IMAGE_FOLDER
        screen._refresh_validation()
        assert screen._stratify_checkbox.disabled is True

        screen._detected_source_type = DataSourceType.CUTANA
        screen._refresh_validation()
        assert screen._stratify_checkbox.disabled is False

    @patch("IPython.display.display")
    def test_on_start_pushes_flag_for_cutana(self, _disp, mock_session):
        mock_session.cfg.data_dir = "/fake/cutana"
        mock_session.cfg.label_file = "/fake/labels.csv"
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._detected_source_type = DataSourceType.CUTANA
        screen._refresh_validation()  # enables the checkbox
        screen._stratify_checkbox.value = True
        with patch.object(app, "navigate_to"):
            screen._on_start()
        assert mock_session.cfg.cutana_stratify_source_size is True

    @patch("IPython.display.display")
    def test_on_start_forces_false_for_non_cutana(self, _disp, mock_session):
        # A True left over from a prior Cutana selection must not leak into an
        # image/Zarr run, where the checkbox is disabled.
        mock_session.cfg.data_dir = "/fake/images"
        mock_session.cfg.label_file = "/fake/labels.csv"
        app = AnomalyMatchApp(mock_session)
        app.navigate_to("training_setup")
        screen = app._current_screen

        screen._detected_source_type = DataSourceType.IMAGE_FOLDER
        screen._refresh_validation()  # disables the checkbox
        screen._stratify_checkbox.value = True  # settable even while disabled
        with patch.object(app, "navigate_to"):
            screen._on_start()
        assert mock_session.cfg.cutana_stratify_source_size is False


# ── Remembered normalisation settings ────────────────────────────


def _build_setup_screen(mock_session):
    """Build a training setup screen the way navigation does."""
    app = AnomalyMatchApp(mock_session)
    app.navigate_to("training_setup")
    return app, app._current_screen


@patch("IPython.display.display")
class TestRememberedNormalisation:
    """Normalisation choices persist across sessions via ``ui_state``.

    ``tests/ui/conftest.py`` redirects ``ui_state`` to ``tmp_path`` for every
    UI test, so these write and read a throwaway state file.
    """

    def test_on_start_persists_settings_and_extensions(self, _disp, mock_session):
        mock_session.cfg.label_file = "/fake/labels.csv"
        app, screen = _build_setup_screen(mock_session)
        screen._norm_widget._method_dropdown.value = NormalisationMethod.ASINH
        screen._norm_widget._resolution_input.value = 224
        screen._norm_widget.set_extensions(["VIS", "NIR-H"])

        with patch.object(app, "navigate_to"):
            screen._on_start()

        record = ui_state.get_normalisation_settings("training_setup")
        assert record["extensions"] == ["VIS", "NIR-H"]
        assert record["settings"]["normalisation_method"] == NormalisationMethod.ASINH
        assert record["settings"]["image_size"] == [224, 224]

    def test_full_flow_start_then_restart(self, _disp, mock_session):
        """Start a run, then rebuild the app: the settings come back.

        Guards the write/read pair together — the state file round-trips
        through JSON, so the enum and matrix types written on Start must be
        the ones the restore path expects to read.
        """
        mock_session.cfg.label_file = "/fake/labels.csv"
        app, screen = _build_setup_screen(mock_session)
        screen._norm_widget._method_dropdown.value = NormalisationMethod.ASINH
        screen._norm_widget._resolution_input.value = 256
        screen._norm_widget._matrix_cells[0][0].value = 0.5
        screen._norm_widget._matrix_cells[0][1].value = 0.5
        with patch.object(app, "navigate_to"):
            screen._on_start()

        _app2, screen2 = _build_setup_screen(mock_session)

        cfg = screen2._norm_widget.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.ASINH
        assert cfg["image_size"] == [256, 256]
        assert np.allclose(cfg["channel_combination"][0], [0.5, 0.5, 0.0])

    def test_fresh_screen_restores_persisted_settings(self, _disp, mock_session):
        """A new kernel picks up the last-used settings, not the cfg defaults."""
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.LOG),
                "image_size": [224, 224],
                "n_output_channels": 3,
                "channel_combination": None,
                "norm_log_calculate_minimum_value": True,
            },
            ["R", "G", "B"],
        )

        _app, screen = _build_setup_screen(mock_session)

        cfg = screen._norm_widget.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.LOG
        assert cfg["image_size"] == [224, 224]
        assert cfg["norm_log_calculate_minimum_value"] is True
        # The restored values are pushed into cfg too, so ``on_enter``'s
        # read-back cannot clobber them with the notebook's defaults.
        assert mock_session.cfg.normalisation.image_size == [224, 224]

    def test_restored_settings_survive_on_enter(self, _disp, mock_session):
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.CONVERSION_ONLY),
                "image_size": [96, 96],
                "n_output_channels": 3,
                "channel_combination": None,
            },
            ["R", "G", "B"],
        )

        _app, screen = _build_setup_screen(mock_session)
        screen.on_enter()

        assert screen._norm_widget.get_normalisation_config()["image_size"] == [96, 96]

    def test_matrix_restored_when_input_channels_match(self, _disp, mock_session):
        matrix = [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.CONVERSION_ONLY),
                "n_output_channels": 3,
                "channel_combination": matrix,
            },
            ["R", "G"],
        )

        _app, screen = _build_setup_screen(mock_session)
        # Widget starts on the default R/G/B headers, so nothing is restored
        # until the columns match what was recorded.
        screen._norm_widget.set_extensions(["R", "G"])
        screen._restore_persisted_normalisation()

        restored = screen._norm_widget.get_normalisation_config()["channel_combination"]
        assert np.allclose(restored, matrix)

    def test_matrix_dropped_when_input_channels_differ(self, _disp, mock_session):
        """A matrix recorded for two bands must not be forced onto a 3-band source."""
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.ASINH),
                "image_size": [200, 200],
                "n_output_channels": 2,
                "channel_combination": [[0.5, 0.5], [1.0, 0.0]],
                "norm_asinh_scale": [0.2, 0.3],
                "norm_asinh_clip": [95.0, 96.0],
            },
            ["VIS", "NIR-H"],
        )

        _app, screen = _build_setup_screen(mock_session)

        cfg = screen._norm_widget.get_normalisation_config()
        # Non-channel settings still come back...
        assert cfg["normalisation_method"] == NormalisationMethod.ASINH
        assert cfg["image_size"] == [200, 200]
        # ...but the channel layout falls back to the widget's own default
        # (3 outputs over the default R/G/B columns → identity → None).
        assert cfg["n_output_channels"] == 3
        assert cfg["channel_combination"] is None
        # The per-channel lists were sized to the dropped channel count, so
        # applying them would have left ch2 on the factory default while ch0/1
        # carried remembered values.
        assert cfg["norm_asinh_scale"] == [0.7, 0.7, 0.7]
        assert cfg["norm_asinh_clip"] == [99.8, 99.8, 99.8]

    @pytest.mark.parametrize("stored_method", [99, "LogStretch", None])
    def test_unusable_stored_method_falls_back_to_defaults(
        self, _disp, mock_session, stored_method
    ):
        """A state file from another version must not make the screen unopenable."""
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": stored_method,
                "image_size": [224, 224],
                "channel_combination": None,
            },
            ["R", "G", "B"],
        )

        _app, screen = _build_setup_screen(mock_session)

        cfg = screen._norm_widget.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.CONVERSION_ONLY
        # The whole record is skipped, not partly applied.
        assert cfg["image_size"] == [128, 128]

    def test_absent_matrix_key_still_restores_the_rest(self, _disp, mock_session):
        ui_state.set_normalisation_settings(
            "training_setup",
            {"normalisation_method": int(NormalisationMethod.ZSCALE), "image_size": [192, 192]},
            ["R", "G", "B"],
        )

        _app, screen = _build_setup_screen(mock_session)

        cfg = screen._norm_widget.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.ZSCALE
        assert cfg["image_size"] == [192, 192]

    def test_cutana_same_bands_keeps_remembered_channels(self, _disp, mock_session):
        """Re-selecting the same catalogue preserves a customised channel count."""
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.CONVERSION_ONLY),
                "n_output_channels": 3,
                "channel_combination": [[1.0], [0.5], [0.25]],
            },
            ["VIS"],
        )

        _app, screen = _build_setup_screen(mock_session)
        with patch(
            "anomaly_match_ui.screens.training_setup_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ):
            screen._apply_cutana_band_info("/fake/cutana")

        cfg = screen._norm_widget.get_normalisation_config()
        assert screen._norm_widget.extensions == ["VIS"]
        # apply_cutana_bands would have collapsed this to one output channel.
        assert cfg["n_output_channels"] == 3
        assert np.allclose(cfg["channel_combination"], [[1.0], [0.5], [0.25]])
        # Band detection runs on the event loop while the job that scheduled it
        # still reads cfg.normalisation from its worker thread, so the restore
        # must hand off to the debounced refresh rather than write cfg here.
        assert screen._norm_refresh_timer is not None

    def test_cutana_unusable_record_still_gets_auto_config(self, _disp, mock_session):
        """Matching bands are not enough — the record has to be applicable too.

        The band names decide nothing on their own: a record written by another
        version can carry a normalisation method this build cannot resolve, and
        the restore then bails. Taking the ``sync_cutana_bands`` branch on the
        strength of the band names alone left the single-band catalogue on a
        3-channel broadcast — no error, just the wrong input shape into the
        model.
        """
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": 99,
                "n_output_channels": 3,
                "channel_combination": [[1.0], [0.5], [0.25]],
            },
            ["VIS"],
        )

        _app, screen = _build_setup_screen(mock_session)
        with patch(
            "anomaly_match_ui.screens.training_setup_screen.detect_cutana_filter_names",
            return_value=["VIS"],
        ):
            screen._apply_cutana_band_info("/fake/cutana")

        cfg = screen._norm_widget.get_normalisation_config()
        assert screen._norm_widget.extensions == ["VIS"]
        # One output channel per detected band — the auto-configuration ran.
        assert cfg["n_output_channels"] == 1
        assert cfg["channel_combination"] is None
        # Nothing was restored, so there is no cfg sync to hand off either.
        assert screen._norm_refresh_timer is None

    def test_cutana_different_bands_falls_back_to_auto_config(self, _disp, mock_session):
        """A catalogue with other bands gets the destructive auto-configuration."""
        ui_state.set_normalisation_settings(
            "training_setup",
            {
                "normalisation_method": int(NormalisationMethod.CONVERSION_ONLY),
                "n_output_channels": 3,
                "channel_combination": [[1.0], [0.5], [0.25]],
            },
            ["VIS"],
        )

        _app, screen = _build_setup_screen(mock_session)
        with patch(
            "anomaly_match_ui.screens.training_setup_screen.detect_cutana_filter_names",
            return_value=["VIS", "NIR-H"],
        ):
            screen._apply_cutana_band_info("/fake/cutana")

        cfg = screen._norm_widget.get_normalisation_config()
        assert screen._norm_widget.extensions == ["VIS", "NIR-H"]
        assert cfg["n_output_channels"] == 2
        assert cfg["channel_combination"] is None


def _zarr_store(path, shape):
    zarr.open_group(str(path), mode="w").create_array("images", shape=shape, dtype=np.uint8)
    return str(path)


@patch("IPython.display.display")
class TestSourceChannelSync:
    """The matrix's input columns follow the selected Zarr store or image folder.

    A matrix left over from a Cutana catalogue (four bands) used to stay in
    place, so a three-channel store failed with "got 3 expected channels, but
    requested 4 output channels".
    """

    def test_zarr_after_cutana_bands_resets_to_store_channels(self, _disp, mock_session, tmp_path):
        _app, screen = _build_setup_screen(mock_session)
        screen._norm_widget.apply_cutana_bands(["VIS", "NIR-H", "NIR-Y", "NIR-J"])
        store = _zarr_store(tmp_path / "cutouts.zarr", (2, 16, 16, 3))

        screen._apply_source_channel_info(store, DataSourceType.ZARR, None)

        cfg = screen._norm_widget.get_normalisation_config()
        assert screen._norm_widget.extensions == ["R", "G", "B"]
        assert cfg["n_output_channels"] == 3
        assert cfg["channel_combination"] is None
        # The running job read the old layout, so a refresh must supersede it.
        assert screen._norm_refresh_timer is not None

    def test_same_channels_keep_a_custom_matrix(self, _disp, mock_session, tmp_path):
        """Rescanning a source with the same channels must not discard the user's matrix."""
        _app, screen = _build_setup_screen(mock_session)
        screen._norm_widget.update_from_config(
            {"n_output_channels": 1, "channel_combination": np.array([[1 / 3, 1 / 3, 1 / 3]])}
        )
        store = _zarr_store(tmp_path / "cutouts.zarr", (2, 16, 16, 3))

        screen._apply_source_channel_info(store, DataSourceType.ZARR, None)

        cfg = screen._norm_widget.get_normalisation_config()
        assert cfg["n_output_channels"] == 1
        assert np.allclose(cfg["channel_combination"], [[1 / 3, 1 / 3, 1 / 3]])

    def test_channel_less_store_gets_one_grey_input(self, _disp, mock_session, tmp_path):
        _app, screen = _build_setup_screen(mock_session)
        store = _zarr_store(tmp_path / "cutouts.zarr", (2, 16, 16))

        screen._apply_source_channel_info(store, DataSourceType.ZARR, None)

        assert screen._norm_widget.extensions == ["Grey"]
        assert screen._norm_widget.get_normalisation_config()["n_output_channels"] == 1


@pytest.mark.parametrize(
    "name, expected",
    [
        ("img_0001.jpeg", "img_0001"),
        ("cutout.PNG", "cutout"),
        ("-592196974511252224", "-592196974511252224"),
        ("J123456.78+123456.7", "J123456.78+123456.7"),
        ("NGC_4038.1", "NGC_4038.1"),
    ],
)
def test_zarr_caption_strips_only_image_extensions(name, expected):
    assert _strip_image_extension(name) == expected


@patch("IPython.display.display")
@pytest.mark.parametrize(
    ("fits_extension", "expected"),
    [
        (None, ["VIS", "NIR-H", "NIR-Y", "NIR-J"]),
        (["VIS", "NIR-Y", "NIR-J"], ["VIS", "NIR-Y", "NIR-J"]),
    ],
)
def test_cutana_matrix_sized_to_extracted_bands(_disp, tmp_path, fits_extension, expected):
    """The channel editor gets one column per band extraction will deliver.

    Unmocked: the band names come from the real catalogue resolver, so a
    notebook-set ``fits_extension`` subset sizes the editor the same way it
    sizes extraction.
    """
    band_paths = str(
        [
            f"/tiles/EUC_MER_BGSUB-MOSAIC-{b}_TILE1_00.00.fits"
            for b in ("VIS", "NIR-H", "NIR-Y", "NIR-J")
        ]
    )
    pd.DataFrame(
        {
            "SourceID": ["s1"],
            "RA": [150.0],
            "Dec": [2.3],
            "diameter_pixel": [64],
            "fits_file_paths": [band_paths],
        }
    ).to_parquet(tmp_path / "q1_source_cat_search_0.parquet", index=False)
    session = MagicMock()
    session.cfg = get_default_cfg()
    # No data_dir: the test drives band detection directly, without the
    # background scan auto-validation would start on the catalogue.
    session.cfg.data_dir = None
    session.cfg.normalisation.fits_extension = fits_extension

    _app, screen = _build_setup_screen(session)
    screen._apply_cutana_band_info(str(tmp_path))

    widget = screen._norm_widget
    assert widget._extensions == expected
    assert all(len(row) == len(expected) for row in widget._matrix_cells)
