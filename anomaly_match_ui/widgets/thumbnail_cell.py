#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Single thumbnail cell for the prediction gallery."""

from __future__ import annotations

import os
import webbrowser
from collections.abc import Callable

import ipywidgets as widgets

from anomaly_match_ui.styles import BG_COLOR

_CELL_SIZE = "220px"

# Square placeholders that occupy exactly the box the thumbnail image will
# fill, so swapping a spinner (or "No preview") for the loaded image never
# resizes the cell.  ``padding-bottom:100%`` makes the inner box's height
# track its own width, matching the square cutouts rendered at the grid
# column width — this is what removes the "wrong-size placeholder then jump"
# the gallery used to flash on every page change.
_SPINNER_BOX_HTML = (
    '<div style="position:relative; width:100%; padding-bottom:100%;">'
    '<div style="position:absolute; inset:0; display:flex; align-items:center;'
    ' justify-content:center;">'
    '<div class="am-spinner" style="margin:0;"></div>'
    "</div></div>"
)
_NO_PREVIEW_BOX_HTML = (
    '<div style="position:relative; width:100%; padding-bottom:100%;">'
    '<div style="position:absolute; inset:0; display:flex; align-items:center;'
    " justify-content:center; color:#666; font-size:11px;"
    ' border:1px dashed #444;">No preview</div></div>'
)


class ThumbnailCell:
    """A single image thumbnail with filename, score, and action buttons.

    Displays a 200x200 preview image with an overlay bar containing
    magnify button, score label, and star button.

    Args:
        on_magnify: Callback invoked with the filename when magnify is clicked.
        on_star: Callback invoked with the filename when star is clicked.
    """

    def __init__(
        self,
        on_magnify: Callable[[str], None] | None = None,
        on_star: Callable[[str], None] | None = None,
    ) -> None:
        self._filename: str | None = None
        self._on_magnify = on_magnify
        self._on_star = on_star

        # Image display
        self._image = widgets.Image(
            format="png",
            layout=widgets.Layout(
                width=_CELL_SIZE,
                height=_CELL_SIZE,
                object_fit="contain",
                overflow="hidden",
            ),
        )

        # Placeholder shown when no image is available
        self._no_image_html = widgets.HTML(
            value="",
            layout=widgets.Layout(
                width=_CELL_SIZE,
                height=_CELL_SIZE,
                display="none",
            ),
        )

        # Action buttons + score (overlay bar): [magnify] [score] [esasky?] [star]
        self._magnify_btn = widgets.Button(
            description="",
            icon="search-plus",
            button_style="primary",
            layout=widgets.Layout(width="28px", height="24px", padding="0"),
        )
        self._magnify_btn.on_click(self._handle_magnify)

        self._score_html = widgets.HTML(
            value="",
            layout=widgets.Layout(flex="1 1 auto"),
        )

        self._esasky_btn = widgets.Button(
            description="",
            icon="globe",
            button_style="primary",
            layout=widgets.Layout(width="28px", height="24px", padding="0", display="none"),
        )
        self._esasky_url: str | None = None
        self._esasky_btn.on_click(self._handle_esasky)

        self._star_btn = widgets.Button(
            description="",
            icon="star",
            button_style="primary",
            layout=widgets.Layout(width="28px", height="24px", padding="0"),
        )
        self._star_btn.on_click(self._handle_star)

        # Single overlay row: [magnify] [score] [esasky] [star]
        self._overlay_bar = widgets.HBox(
            [self._magnify_btn, self._score_html, self._esasky_btn, self._star_btn],
            layout=widgets.Layout(
                width=_CELL_SIZE,
                align_items="center",
                gap="2px",
            ),
        )

        # Label (filename only)
        self._label = widgets.HTML(
            value="",
            layout=widgets.Layout(width=_CELL_SIZE, overflow="hidden"),
        )

        # Assemble cell: overlay (buttons+score) → image → filename
        self.widget = widgets.VBox(
            [self._overlay_bar, self._image, self._no_image_html, self._label],
            layout=widgets.Layout(
                width=_CELL_SIZE,
                background_color=BG_COLOR,
                align_items="center",
                overflow="hidden",
            ),
        )

        self.clear()

    def update(
        self,
        filename: str,
        score: float,
        image_bytes: bytes,
        esasky_url: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Refresh the cell with new image data.

        Args:
            filename: The source filename, passed back to ``on_magnify``/
                ``on_star`` and used as the cache key — never altered for
                display.
            score: The anomaly score.
            image_bytes: PNG-encoded image bytes.
            esasky_url: Optional ESASky URL. Shows globe button when set.
            display_name: Caption text to show instead of *filename*, for
                callers whose real id is misleading to show verbatim (e.g.
                a Zarr source id carrying a pre-conversion file extension).
                Defaults to *filename*.
        """
        self._filename = filename
        self._esasky_url = esasky_url
        self._esasky_btn.layout.display = "" if esasky_url else "none"

        self._score_html.value = _score_markup(score)
        self._label.value = _filename_markup(display_name if display_name is not None else filename)
        self._overlay_bar.layout.display = ""

        if image_bytes:
            self._image.value = image_bytes
            self._image.layout.display = ""
            self._no_image_html.layout.display = "none"
        else:
            self._image.value = b""
            self._image.layout.display = "none"
            self._no_image_html.value = _NO_PREVIEW_BOX_HTML
            self._no_image_html.layout.display = ""

    def mark_loading(self, filename: str | None = None, score: float | None = None) -> None:
        """Show a loading spinner in the image slot without resizing the cell.

        The spinner occupies the same square box the image will fill, so
        paging to an uncached image shows an in-place spinner that the
        pixels replace without a layout jump.  When *filename* is known the
        overlay bar (magnify / score / star) is shown too so the user can
        act on the cell while it streams in; a bare call (page just flipped,
        metadata not yet queried) shows only the spinner.

        Args:
            filename: Source filename, or ``None`` if not yet known.
            score: Anomaly score, or ``None`` if not yet known.
        """
        self._filename = filename
        self._esasky_url = None
        self._esasky_btn.layout.display = "none"
        if filename is not None:
            self._score_html.value = _score_markup(score) if score is not None else ""
            self._label.value = _filename_markup(filename)
            self._overlay_bar.layout.display = ""
        else:
            self._score_html.value = ""
            self._label.value = ""
            self._overlay_bar.layout.display = "none"
        self._image.value = b""
        self._image.layout.display = "none"
        self._no_image_html.value = _SPINNER_BOX_HTML
        self._no_image_html.layout.display = ""

    def clear(self) -> None:
        """Reset the cell to an empty placeholder."""
        self._filename = None
        self._esasky_url = None
        self._image.value = b""
        self._score_html.value = ""
        self._label.value = (
            f'<div style="color:#444; font-size:11px; text-align:center;'
            f" display:flex; align-items:center; justify-content:center;"
            f' height:{_CELL_SIZE}; border:1px dashed #333;">Waiting for results\u2026</div>'
        )
        self._overlay_bar.layout.display = "none"
        self._image.layout.display = "none"
        self._no_image_html.layout.display = "none"

    def _handle_esasky(self, _btn: widgets.Button) -> None:
        if self._esasky_url:
            webbrowser.open(self._esasky_url)

    def _handle_magnify(self, _btn: widgets.Button) -> None:
        if self._filename and self._on_magnify:
            self._on_magnify(self._filename)

    def _handle_star(self, _btn: widgets.Button) -> None:
        if self._filename and self._on_star:
            self._on_star(self._filename)


def _score_markup(score: float) -> str:
    """Render the centered score caption shown in a cell's overlay bar.

    Returns:
        HTML string for the score label.
    """
    return f'<div style="color:#ddd; font-size:11px; text-align:center;">Score: {score:.4f}</div>'


def _filename_markup(filename: str) -> str:
    """Render the truncated, centered filename caption for a cell.

    Returns:
        HTML string for the filename label.
    """
    short = _shorten(filename, 26)
    return (
        f'<div style="color:white; font-size:11px; text-align:center;'
        f' white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">'
        f"{short}</div>"
    )


def _shorten(filename: str, max_length: int = 26) -> str:
    """Truncate a filename for display, preserving extension.

    Returns:
        Shortened filename string.
    """
    base = os.path.basename(filename)
    if len(base) <= max_length:
        return base
    name, ext = os.path.splitext(base)
    avail = max_length - len(ext) - 1  # 1 char for ellipsis
    if avail < 4:
        return base[:max_length]
    keep_start = avail * 2 // 3
    keep_end = avail - keep_start
    return name[:keep_start] + "\u2026" + name[-keep_end:] + ext
