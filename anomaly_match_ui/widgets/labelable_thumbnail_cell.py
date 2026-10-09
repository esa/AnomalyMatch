#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Thumbnail cell with a three-way label toggle for the training gallery."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

import ipywidgets as widgets

from anomaly_match_ui.styles import BG_COLOR, ESA_GREEN, ESA_RED, NEUTRAL_GREY
from anomaly_match_ui.widgets.thumbnail_cell import ThumbnailCell


class LabelState(Enum):
    """Three-way label state for a gallery cell."""

    UNLABELLED = "unlabelled"
    ANOMALY = "anomaly"
    NORMAL = "normal"


# Button appearance per state: (description, active_color)
_BUTTON_SPECS: dict[LabelState, tuple[str, str]] = {
    LabelState.ANOMALY: ("Anomaly", ESA_RED),
    LabelState.UNLABELLED: ("Unlabelled", NEUTRAL_GREY),
    LabelState.NORMAL: ("Normal", ESA_GREEN),
}


class LabelableThumbnailCell:
    """Thumbnail cell with Anomaly / Unlabelled / Normal toggle buttons.

    Composes a :class:`ThumbnailCell` and adds a row of three toggle
    buttons below it. Exactly one button is active at a time.

    Args:
        on_magnify: Forwarded to the inner ``ThumbnailCell``.
        on_star: Forwarded to the inner ``ThumbnailCell``.
        on_label_change: Called with ``(filename, new_label)`` when the
            user clicks a label button. Not fired for programmatic changes.
    """

    def __init__(
        self,
        on_magnify: Callable[[str], None] | None = None,
        on_star: Callable[[str], None] | None = None,
        on_label_change: Callable[[str, LabelState], None] | None = None,
    ) -> None:
        self._on_label_change = on_label_change
        self._label_state = LabelState.UNLABELLED

        self._cell = ThumbnailCell(on_magnify=on_magnify, on_star=on_star)

        # Three toggle buttons: [Anomaly] [Unlabelled] [Normal]
        self._buttons: dict[LabelState, widgets.Button] = {}
        btn_widgets: list[widgets.Button] = []
        for state, (desc, _color) in _BUTTON_SPECS.items():
            btn = widgets.Button(
                description=desc,
                layout=widgets.Layout(width="auto", height="26px", padding="0 6px"),
            )
            btn.on_click(self._make_click_handler(state))
            self._buttons[state] = btn
            btn_widgets.append(btn)

        self._toggle_bar = widgets.HBox(
            btn_widgets,
            layout=widgets.Layout(
                justify_content="center",
                gap="4px",
                background_color=BG_COLOR,
            ),
        )
        self._refresh_button_styles()

        # Rearrange layout: filename → overlay_bar → image → toggles.
        # ThumbnailCell default is [overlay_bar, image, no_image_html, label].
        c = self._cell
        # Tighten spacing between filename and overlay bar
        c._label.layout.margin = "0"
        c._overlay_bar.layout.margin = "0"
        # Use 100% width so cells scale with the CSS Grid column width
        c._label.layout.width = "100%"
        c._overlay_bar.layout.width = "100%"
        c._image.layout.width = "100%"
        c._image.layout.height = "auto"
        c._no_image_html.layout.width = "100%"
        # Let the square spinner/no-preview box (padding-bottom:100%) drive
        # the height so it matches the aspect-driven image and the cell keeps
        # a stable size across the loading → loaded swap (no layout jump).
        c._no_image_html.layout.height = "auto"

        self.widget = widgets.VBox(
            [c._label, c._overlay_bar, c._image, c._no_image_html, self._toggle_bar],
            layout=widgets.Layout(
                width="100%",
                # Cap generously rather than tightly: the column count is
                # user-chosen now, so fewer columns on a wide panel are
                # meant to enlarge the cutout for detail.  The cap only
                # stops a single cell from filling an ultra-wide screen.
                max_width="480px",
                background_color=BG_COLOR,
                align_items="center",
                overflow="hidden",
            ),
        )

    # ── Public API ────────────────────────────────────────────

    @property
    def filename(self) -> str | None:
        """The filename currently displayed, or ``None`` if cleared."""
        return self._cell._filename

    @property
    def label_state(self) -> LabelState:
        """The current label state."""
        return self._label_state

    def update(
        self,
        filename: str,
        score: float,
        image_bytes: bytes,
        label: LabelState = LabelState.UNLABELLED,
        esasky_url: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Refresh the cell with new image data and label state.

        Args:
            filename: The source filename.
            score: The anomaly score.
            image_bytes: PNG-encoded image bytes.
            label: Initial label state for this cell.
            esasky_url: Optional ESASky URL.
            display_name: Caption text to show instead of *filename* — see
                :meth:`ThumbnailCell.update`.
        """
        self._cell.update(
            filename, score, image_bytes, esasky_url=esasky_url, display_name=display_name
        )
        self._label_state = label
        self._refresh_button_styles()
        self._toggle_bar.layout.display = ""

    def mark_loading(
        self,
        filename: str | None = None,
        score: float | None = None,
        label: LabelState = LabelState.UNLABELLED,
    ) -> None:
        """Show a loading spinner for the image while metadata stays visible.

        Delegates the spinner to the inner cell and, when *filename* is
        known, keeps the label toggle bar visible with the stored label so
        the user can label the source before its pixels finish streaming
        in. A bare call (page just flipped) hides the toggle bar.

        Args:
            filename: Source filename, or ``None`` if not yet known.
            score: Anomaly score, or ``None`` if not yet known.
            label: Stored label state to reflect on the toggle bar.
        """
        self._cell.mark_loading(filename, score)
        if filename is not None:
            self._label_state = label
            self._refresh_button_styles()
            self._toggle_bar.layout.display = ""
        else:
            self._label_state = LabelState.UNLABELLED
            self._toggle_bar.layout.display = "none"

    def set_label(self, label: LabelState) -> None:
        """Set the label state programmatically (no callback fired).

        Args:
            label: New label state.
        """
        self._label_state = label
        self._refresh_button_styles()

    def clear(self) -> None:
        """Reset the cell to an empty placeholder."""
        self._cell.clear()
        self._label_state = LabelState.UNLABELLED
        self._refresh_button_styles()
        self._toggle_bar.layout.display = "none"

    # ── Internals ─────────────────────────────────────────────

    def _make_click_handler(self, state: LabelState) -> Callable[[widgets.Button], None]:
        def _handler(_btn: widgets.Button) -> None:
            if self._cell._filename is None:
                return
            self._label_state = state
            self._refresh_button_styles()
            if self._on_label_change:
                self._on_label_change(self._cell._filename, state)

        return _handler

    def _refresh_button_styles(self) -> None:
        """Update button colors to reflect the active label state."""
        for state, btn in self._buttons.items():
            _desc, active_color = _BUTTON_SPECS[state]
            if state == self._label_state:
                btn.style.button_color = active_color
                btn.style.font_weight = "bold"
            else:
                btn.style.button_color = "#333333"
                btn.style.font_weight = "normal"
