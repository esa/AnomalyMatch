#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for ThumbnailCell and GalleryWidget."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream
from anomaly_match_ui.widgets.gallery_widget import PAGE_SIZE, GalleryWidget
from anomaly_match_ui.widgets.thumbnail_cell import ThumbnailCell, _shorten

pytestmark = pytest.mark.ui


# ── ThumbnailCell ────────────────────────────────────────────────


class TestThumbnailCell:
    """Tests for the ThumbnailCell widget."""

    def test_update_shows_image_and_label(self):
        cell = ThumbnailCell()
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        cell.update("test_image.fits", 0.8765, img_bytes)
        assert cell._filename == "test_image.fits"
        assert cell._image.value == img_bytes
        assert "0.8765" in cell._score_html.value
        assert "test_image.fits" in cell._label.value
        assert cell._overlay_bar.layout.display == ""

    def test_clear_resets_cell(self):
        cell = ThumbnailCell()
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        cell.update("image.png", 0.5, img_bytes)
        cell.clear()
        assert cell._filename is None
        assert cell._overlay_bar.layout.display == "none"

    def test_magnify_callback(self):
        callback = MagicMock()
        cell = ThumbnailCell(on_magnify=callback)
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        cell.update("img.png", 0.5, img_bytes)
        cell._handle_magnify(None)
        callback.assert_called_once_with("img.png")

    def test_callbacks_do_nothing_when_cleared(self):
        magnify = MagicMock()
        star = MagicMock()
        cell = ThumbnailCell(on_magnify=magnify, on_star=star)
        cell._handle_magnify(None)
        cell._handle_star(None)
        magnify.assert_not_called()
        star.assert_not_called()

    def test_esasky_button_shown_with_url(self):
        cell = ThumbnailCell()
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )
        cell.update("img.png", 0.5, img_bytes, esasky_url="https://sky.esa.int/test")
        assert cell._esasky_btn.layout.display == ""
        assert cell._esasky_url == "https://sky.esa.int/test"

    def test_mark_loading_shows_spinner_over_hidden_image(self):
        """A loading cell hides the image and shows the square spinner box so
        the pixels can drop in later without a layout jump."""
        cell = ThumbnailCell()
        cell.mark_loading("img.png", 0.42)
        assert cell._filename == "img.png"
        assert "am-spinner" in cell._no_image_html.value
        assert cell._no_image_html.layout.display == ""
        assert cell._image.layout.display == "none"
        # Overlay/score/filename shown so the user can act while it loads.
        assert cell._overlay_bar.layout.display == ""
        assert "0.42" in cell._score_html.value

    def test_mark_loading_without_metadata_hides_overlay(self):
        """The instant page-flip spinner (no filename yet) shows only the
        spinner — no stale overlay from the previous page's cell."""
        cell = ThumbnailCell()
        cell.update("old.png", 0.9, numpy_array_to_byte_stream(np.zeros((8, 8, 3), np.uint8)))
        cell.mark_loading()
        assert cell._filename is None
        assert cell._overlay_bar.layout.display == "none"
        assert "am-spinner" in cell._no_image_html.value

    def test_loading_then_update_swaps_spinner_for_image(self):
        cell = ThumbnailCell()
        cell.mark_loading("img.png", 0.5)
        img_bytes = numpy_array_to_byte_stream(np.zeros((16, 16, 3), np.uint8))
        cell.update("img.png", 0.5, img_bytes)
        assert cell._image.value == img_bytes
        assert cell._image.layout.display == ""
        assert cell._no_image_html.layout.display == "none"

    def test_update_without_image_shows_no_preview_box(self):
        """A genuinely missing image shows the "No preview" box, distinct from
        the loading spinner, so the two states are visually different."""
        cell = ThumbnailCell()
        cell.update("img.png", 0.5, b"")
        assert "No preview" in cell._no_image_html.value
        assert "am-spinner" not in cell._no_image_html.value
        assert cell._no_image_html.layout.display == ""


# ── _shorten ─────────────────────────────────────────────────────


class TestShorten:
    """Tests for the filename shortening utility."""

    def test_short_name_unchanged(self):
        assert _shorten("img.png") == "img.png"

    def test_long_name_truncated(self):
        result = _shorten("a_very_long_filename_that_exceeds_limit.fits", 22)
        assert len(result) <= 22
        assert result.endswith(".fits")
        assert "\u2026" in result


# ── GalleryWidget ────────────────────────────────────────────────


class TestGalleryWidget:
    """Tests for the GalleryWidget."""

    def test_gallery_has_page_size_cells(self):
        gallery = GalleryWidget()
        assert len(gallery._cells) == PAGE_SIZE

    def test_sort_mode_switching(self):
        gallery = GalleryWidget()
        assert gallery.get_sort_mode() == "score_desc"
        gallery._set_sort("score_asc")
        assert gallery.get_sort_mode() == "score_asc"

    def test_sort_change_callback(self):
        callback = MagicMock()
        gallery = GalleryWidget()
        gallery.set_on_sort_change(callback)
        gallery._set_sort("score_asc")
        callback.assert_called_once()

    def test_update_page_fills_cells(self):
        gallery = GalleryWidget()
        results = [{"filename": f"img_{i}.png", "score": 0.9 - i * 0.1} for i in range(3)]

        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )

        def loader(_fn):
            return img_bytes

        gallery.update_page(results, 0, 1, 3, image_loader=loader)

        # First 3 cells should be populated
        for i in range(3):
            assert gallery._cells[i]._filename == f"img_{i}.png"
        # Remaining cells should be cleared
        for i in range(3, 10):
            assert gallery._cells[i]._filename is None

    def test_update_page_clears_excess_cells(self):
        gallery = GalleryWidget()
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )

        # Fill with 10 results first
        full_results = [{"filename": f"img_{i}.png", "score": 0.5} for i in range(10)]
        gallery.update_page(full_results, 0, 1, 10, image_loader=lambda _: img_bytes)

        # Then update with only 2 results
        small_results = [{"filename": f"new_{i}.png", "score": 0.8} for i in range(2)]
        gallery.update_page(small_results, 0, 1, 2, image_loader=lambda _: img_bytes)

        assert gallery._cells[0]._filename == "new_0.png"
        assert gallery._cells[1]._filename == "new_1.png"
        assert gallery._cells[2]._filename is None

    def test_pagination_buttons(self):
        gallery = GalleryWidget()
        img_bytes = numpy_array_to_byte_stream(
            np.zeros((32, 32, 3), dtype=np.uint8),
        )

        # Page 0 of 3 — First and Previous disabled on the first page.
        gallery.update_page([], 0, 3, 27, image_loader=lambda _: img_bytes)
        assert gallery._first_btn.disabled is True
        assert gallery._prev_btn.disabled is True
        assert gallery._next_btn.disabled is False

        # Page 2 of 3 (last page) — First re-enabled, Next disabled.
        gallery.update_page([], 2, 3, 27, image_loader=lambda _: img_bytes)
        assert gallery._first_btn.disabled is False
        assert gallery._prev_btn.disabled is False
        assert gallery._next_btn.disabled is True

    def test_page_change_callback(self):
        callback = MagicMock()
        gallery = GalleryWidget()
        gallery.set_on_page_change(callback)
        gallery._total_pages = 5
        gallery._change_page(1)
        callback.assert_called_once_with(1)

    def test_first_button_jumps_to_page_zero(self):
        callback = MagicMock()
        gallery = GalleryWidget()
        gallery.set_on_page_change(callback)
        gallery._total_pages = 5
        gallery._current_page = 4

        gallery._jump_to_page(0)

        assert gallery._current_page == 0
        callback.assert_called_once_with(0)

    def test_first_button_noop_when_already_first(self):
        callback = MagicMock()
        gallery = GalleryWidget()
        gallery.set_on_page_change(callback)
        gallery._total_pages = 5
        gallery._current_page = 0

        gallery._jump_to_page(0)

        callback.assert_not_called()

    def test_jump_to_page_clamps_out_of_range_target(self):
        """An absolute seek past the last page clamps to the final page so a
        bad target can't drive the gallery off the end of the result set."""
        callback = MagicMock()
        gallery = GalleryWidget()
        gallery.set_on_page_change(callback)
        gallery._total_pages = 5
        gallery._current_page = 0

        gallery._jump_to_page(99)

        assert gallery._current_page == 4
        callback.assert_called_once_with(4)

    def test_scale_slider_sets_fixed_column_count(self):
        """The slider sets a fixed column count, not a cell pixel width.

        Page size is derived from that same count, so the grid and
        pagination can't disagree about how many cells a page holds —
        the disagreement that previously clipped labeled cutouts (#481).
        """
        gallery = GalleryWidget()
        gallery._scale_slider.value = 7
        assert gallery._grid.layout.grid_template_columns == "repeat(7, 1fr)"
        # Page size tracks the column count: _ROWS (2) * 7 columns.
        assert gallery.effective_page_size == 14

    def test_grid_pins_two_row_layout(self):
        """Gallery pages always show exactly two rows of cells.

        An earlier #432 attempt removed the row-clipping CSS to cure
        "blank cells at larger sizes" — but the resulting multi-row
        gallery broke the fixed-height layout other panels depend on.
        Two-row clipping is intentional; pagination compensates.
        """
        gallery = GalleryWidget()
        layout = gallery._grid.layout
        assert layout.grid_template_rows == "repeat(2, auto)"
        assert layout.grid_auto_rows == "0px"
        assert layout.overflow == "hidden"

    def test_setup_mode_hides_sort_bar(self):
        """In setup_mode the sort bar is hidden — score-based sorts have
        no meaning before any prediction has run (#427)."""
        gallery = GalleryWidget(setup_mode=True)
        controls_bar = gallery.widget.children[0]
        sort_bar = controls_bar.children[0]
        assert sort_bar.layout.display == "none"

    def test_default_mode_shows_sort_bar(self):
        gallery = GalleryWidget()
        controls_bar = gallery.widget.children[0]
        sort_bar = controls_bar.children[0]
        assert sort_bar.layout.display in (None, "")

    def test_begin_page_marks_loading_cells_and_updates_pagination(self):
        """``begin_page`` flips the page instantly: the first ``n_loading``
        cells show a spinner, the rest are cleared, and the pagination label
        and buttons update — all without a DB query or image load."""
        gallery = GalleryWidget()
        gallery.begin_page(page=1, total_pages=3, total_count=27, n_loading=4)

        for i in range(4):
            assert "am-spinner" in gallery._cells[i]._no_image_html.value
            assert gallery._cells[i]._filename is None
        for i in range(4, PAGE_SIZE):
            assert gallery._cells[i]._filename is None
            assert "am-spinner" not in gallery._cells[i]._no_image_html.value

        assert "Page 2 of 3" in gallery._page_label.value
        # Middle page: every nav button is enabled.
        assert gallery._first_btn.disabled is False
        assert gallery._next_btn.disabled is False

    def test_set_pagination_updates_label_without_touching_cells(self):
        """A background poll refresh updates the counter/buttons via
        ``set_pagination`` while leaving the visible thumbnails untouched, so
        the page the user is reading doesn't flash a row of spinners."""
        gallery = GalleryWidget()
        img_bytes = numpy_array_to_byte_stream(np.zeros((8, 8, 3), np.uint8))
        results = [{"filename": f"img_{i}.png", "score": 0.5} for i in range(3)]
        gallery.update_page(results, 0, 1, 3, image_loader=lambda _: img_bytes)

        gallery.set_pagination(page=0, total_pages=2, total_count=42)

        assert "Page 1 of 2" in gallery._page_label.value
        assert "42" in gallery._page_label.value
        assert gallery._next_btn.disabled is False
        # Cells are left exactly as they were.
        assert gallery._cells[0]._filename == "img_0.png"
        assert gallery._cells[0]._image.value == img_bytes
