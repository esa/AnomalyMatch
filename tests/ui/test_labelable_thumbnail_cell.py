#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for LabelableThumbnailCell."""

from unittest.mock import MagicMock, patch

import pytest

from anomaly_match_ui.widgets.labelable_thumbnail_cell import (
    LabelableThumbnailCell,
    LabelState,
)

pytestmark = pytest.mark.ui

# 1x1 transparent PNG for tests
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
    b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@patch("IPython.display.display")
class TestLabelableThumbnailCell:
    """Test the labelable thumbnail cell widget."""

    def test_default_state_is_unlabelled(self, _mock_display):
        cell = LabelableThumbnailCell()
        assert cell.label_state == LabelState.UNLABELLED

    def test_update_sets_filename_and_label(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("img001.png", 0.95, _TINY_PNG, label=LabelState.ANOMALY)
        assert cell.filename == "img001.png"
        assert cell.label_state == LabelState.ANOMALY

    def test_update_defaults_to_unlabelled(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("img001.png", 0.5, _TINY_PNG)
        assert cell.label_state == LabelState.UNLABELLED

    def test_click_anomaly_button(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        cell.update("img001.png", 0.8, _TINY_PNG)

        cell._buttons[LabelState.ANOMALY].click()
        assert cell.label_state == LabelState.ANOMALY
        callback.assert_called_once_with("img001.png", LabelState.ANOMALY)

    def test_click_normal_button(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        cell.update("img001.png", 0.8, _TINY_PNG)

        cell._buttons[LabelState.NORMAL].click()
        assert cell.label_state == LabelState.NORMAL
        callback.assert_called_once_with("img001.png", LabelState.NORMAL)

    def test_click_unlabelled_button(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        cell.update("img001.png", 0.8, _TINY_PNG, label=LabelState.ANOMALY)

        cell._buttons[LabelState.UNLABELLED].click()
        assert cell.label_state == LabelState.UNLABELLED
        callback.assert_called_once_with("img001.png", LabelState.UNLABELLED)

    def test_click_on_empty_cell_does_nothing(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        # No update() called — cell is empty
        cell._buttons[LabelState.ANOMALY].click()
        callback.assert_not_called()
        assert cell.label_state == LabelState.UNLABELLED

    def test_set_label_programmatic_no_callback(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        cell.update("img001.png", 0.8, _TINY_PNG)

        cell.set_label(LabelState.NORMAL)
        assert cell.label_state == LabelState.NORMAL
        callback.assert_not_called()

    def test_clear_resets_label(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("img001.png", 0.8, _TINY_PNG, label=LabelState.ANOMALY)
        cell.clear()
        assert cell.label_state == LabelState.UNLABELLED
        assert cell.filename is None

    def test_toggle_bar_hidden_when_cleared(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("img001.png", 0.8, _TINY_PNG)
        assert cell._toggle_bar.layout.display == ""
        cell.clear()
        assert cell._toggle_bar.layout.display == "none"

    def test_mark_loading_keeps_label_toggles_visible(self, _mock_display):
        """While a cell's image loads, its stored label stays selectable so the
        user can label a source without waiting for the thumbnail to render."""
        cell = LabelableThumbnailCell()
        cell.mark_loading("img001.png", 0.8, label=LabelState.ANOMALY)
        assert cell.filename == "img001.png"
        assert cell.label_state == LabelState.ANOMALY
        assert cell._toggle_bar.layout.display == ""
        assert "am-spinner" in cell._cell._no_image_html.value

    def test_mark_loading_without_metadata_hides_toggle_bar(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("old.png", 0.9, _TINY_PNG, label=LabelState.NORMAL)
        cell.mark_loading()
        assert cell.filename is None
        assert cell.label_state == LabelState.UNLABELLED
        assert cell._toggle_bar.layout.display == "none"

    def test_magnify_callback_forwarded(self, _mock_display):
        magnify_cb = MagicMock()
        cell = LabelableThumbnailCell(on_magnify=magnify_cb)
        cell.update("img001.png", 0.8, _TINY_PNG)
        cell._cell._magnify_btn.click()
        magnify_cb.assert_called_once_with("img001.png")

    def test_star_callback_forwarded(self, _mock_display):
        star_cb = MagicMock()
        cell = LabelableThumbnailCell(on_star=star_cb)
        cell.update("img001.png", 0.8, _TINY_PNG)
        cell._cell._star_btn.click()
        star_cb.assert_called_once_with("img001.png")

    def test_active_button_has_colored_style(self, _mock_display):
        cell = LabelableThumbnailCell()
        cell.update("img001.png", 0.8, _TINY_PNG, label=LabelState.ANOMALY)
        # Active button should have its state color
        from anomaly_match_ui.styles import ESA_RED

        assert cell._buttons[LabelState.ANOMALY].style.button_color == ESA_RED
        # Inactive buttons should be dimmed
        assert cell._buttons[LabelState.UNLABELLED].style.button_color == "#333333"
        assert cell._buttons[LabelState.NORMAL].style.button_color == "#333333"

    def test_sequential_label_changes(self, _mock_display):
        callback = MagicMock()
        cell = LabelableThumbnailCell(on_label_change=callback)
        cell.update("img001.png", 0.8, _TINY_PNG)

        cell._buttons[LabelState.ANOMALY].click()
        cell._buttons[LabelState.NORMAL].click()
        cell._buttons[LabelState.UNLABELLED].click()

        assert callback.call_count == 3
        assert cell.label_state == LabelState.UNLABELLED
