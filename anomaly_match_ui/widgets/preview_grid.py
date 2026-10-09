#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Lightweight image preview grid for setup screens."""

from __future__ import annotations

import ipywidgets as widgets

from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT, TEXT_COLOR

_CELL_SIZE = "120px"
_N_CELLS = 20


class PreviewGrid:
    """A 3x3 grid of image thumbnails for previewing source data.

    Unlike the full :class:`GalleryWidget`, this is a read-only grid
    with no pagination, sorting, or interaction — just images and labels.

    Call :meth:`update` with image bytes and filenames to populate,
    or :meth:`show_message` / :meth:`show_spinner` for status feedback.
    """

    def __init__(self) -> None:
        self._cells: list[widgets.VBox] = []
        self._images: list[widgets.Image] = []
        self._labels: list[widgets.HTML] = []
        # Tracks whether the current display is a spinner.  Used to gate
        # background-thread updates via :meth:`update_spinner_text` so a
        # stale log line cannot re-show the spinner after the real preview
        # (or an error message) has already been rendered.
        self._showing_spinner: bool = False
        self._spinner_text: str = ""

        for _ in range(_N_CELLS):
            img = widgets.Image(
                format="png",
                layout=widgets.Layout(
                    width="100%",
                    max_width=_CELL_SIZE,
                    height="auto",
                    object_fit="contain",
                ),
            )
            label = widgets.HTML(
                value="",
                layout=widgets.Layout(
                    width="100%",
                    max_width=_CELL_SIZE,
                    overflow="hidden",
                ),
            )
            cell = widgets.VBox(
                [img, label],
                layout=widgets.Layout(
                    background_color=BG_COLOR,
                    align_items="center",
                    overflow="hidden",
                ),
            )
            self._images.append(img)
            self._labels.append(label)
            self._cells.append(cell)

        # Two-row constraint: `grid_template_rows="repeat(2, auto)"` +
        # `grid_auto_rows="0px"` + `overflow="hidden"` keeps the preview
        # grid at a fixed height regardless of cell count.  Cells that
        # overflow past row 2 are clipped on purpose — see the analogous
        # GalleryWidget rationale (#432 follow-up).
        self._grid = widgets.GridBox(
            self._cells,
            layout=widgets.Layout(
                grid_template_columns=f"repeat(auto-fill, minmax({_CELL_SIZE}, 1fr))",
                grid_template_rows="repeat(2, auto)",
                grid_auto_rows="0px",
                grid_gap="6px",
                padding="8px 0",
                background_color=BG_COLOR,
                overflow="hidden",
            ),
        )

        self._message = widgets.HTML(
            value="",
            layout=widgets.Layout(display="none", padding="12px"),
        )

        self._title = widgets.HTML(
            value=(f'<b style="color:{ESA_BLUE_BRIGHT}; font-size:13px">Preview</b>'),
        )

        self.widget = widgets.VBox(
            [self._title, self._message, self._grid],
            layout=widgets.Layout(background_color=BG_COLOR, overflow="hidden"),
        )
        self.clear()

    def update(self, items: list[tuple[str, bytes]]) -> None:
        """Populate the grid with image data.

        Args:
            items: List of ``(caption_html, png_bytes)`` tuples.
                *caption_html* is rendered as raw HTML under each image.
        """
        self._message.layout.display = "none"
        self._grid.layout.display = ""
        self._showing_spinner = False

        for i, cell in enumerate(self._cells):
            if i < len(items):
                caption_html, img_bytes = items[i]
                self._images[i].value = img_bytes
                self._labels[i].value = (
                    f'<div style="color:{TEXT_COLOR}; font-size:10px;'
                    f' text-align:center;">'
                    f"{caption_html}</div>"
                )
                cell.layout.display = ""
            else:
                self._images[i].value = b""
                self._labels[i].value = ""
                cell.layout.display = "none"

    def show_spinner(self, text: str = "Loading preview...") -> None:
        """Show a spinner with a status message."""
        self._grid.layout.display = "none"
        self._message.layout.display = ""
        self._showing_spinner = True
        self._spinner_text = text
        self._message.value = self._render_spinner_html(text)

    def update_spinner_text(self, text: str) -> None:
        """Replace the spinner caption without altering display mode.

        Safe to call from a background thread. No-op when the grid is
        not currently showing a spinner — prevents stale log messages
        from stomping on a rendered preview or an error state.
        """
        if not self._showing_spinner or text == self._spinner_text:
            return
        self._spinner_text = text
        self._message.value = self._render_spinner_html(text)

    def show_message(self, text: str) -> None:
        """Show a plain text message (e.g. error or empty state)."""
        self._grid.layout.display = "none"
        self._message.layout.display = ""
        self._showing_spinner = False
        self._message.value = (
            f'<div style="color:#888; font-size:12px; text-align:center; padding:8px;">{text}</div>'
        )

    def clear(self) -> None:
        """Hide all cells and messages."""
        self._grid.layout.display = "none"
        self._message.layout.display = "none"
        self._showing_spinner = False

    @staticmethod
    def _render_spinner_html(text: str) -> str:
        return (
            f'<div style="color:#aaa; font-size:12px; text-align:center;">'
            f'<div class="am-spinner" style="width:20px; height:20px;'
            f' border-width:2px; margin:0 auto 6px auto;"></div>'
            f"{text}</div>"
        )
