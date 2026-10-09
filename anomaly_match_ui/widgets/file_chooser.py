#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Styled file/folder chooser wrapper for AnomalyMatch UI."""

from __future__ import annotations

import os
from collections.abc import Callable

import ipywidgets as widgets
from ipyfilechooser import FileChooser
from loguru import logger

from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT, ESA_BLUE_DEEP, TEXT_COLOR
from anomaly_match_ui.utils import ui_state

# Component-specific styles scoped to this widget (not in styles.py)
_CHOOSER_STYLE = f"""
<style>
.am-file-chooser .widget-file-chooser,
.am-file-chooser .filechooser {{
    background-color: {BG_COLOR} !important;
    border: 1px solid {ESA_BLUE_DEEP} !important;
    border-radius: 5px !important;
    max-height: 280px !important;
    overflow: auto !important;
}}

.am-file-chooser select,
.am-file-chooser input {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
    border: 1px solid {ESA_BLUE_DEEP} !important;
    border-radius: 3px !important;
    padding: 6px !important;
    min-height: 28px !important;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}}

.am-file-chooser select:focus,
.am-file-chooser input:focus {{
    outline: none !important;
    border-color: {ESA_BLUE_BRIGHT} !important;
    box-shadow: 0 0 0 2px rgba(0, 161, 222, 0.2) !important;
}}

.am-file-chooser button {{
    background-color: {ESA_BLUE_BRIGHT} !important;
    color: {BG_COLOR} !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 8px 12px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    min-height: 32px !important;
}}

.am-file-chooser button:hover {{
    opacity: 0.85 !important;
}}

.am-file-chooser .filename,
.am-file-chooser label,
.am-file-chooser div,
.am-file-chooser span,
.am-file-chooser p {{
    color: {TEXT_COLOR} !important;
}}

.am-file-chooser select option {{
    background-color: {BG_COLOR} !important;
    color: {TEXT_COLOR} !important;
}}

.am-file-chooser [style*="color: black"],
.am-file-chooser [style*="color: #000"],
.am-file-chooser [style*="color: rgb(0, 0, 0)"] {{
    color: {TEXT_COLOR} !important;
}}

.am-file-chooser {{
    min-height: 100px !important;
    width: 100% !important;
    max-width: 100% !important;
}}

.am-file-chooser,
.am-file-chooser .widget-file-chooser,
.am-file-chooser .filechooser,
.am-file-chooser > *,
.am-file-chooser * {{
    max-width: 100% !important;
    overflow-x: hidden !important;
    box-sizing: border-box !important;
}}
</style>
"""


class AMFileChooser(widgets.VBox):
    """Dark-themed file/folder chooser for AnomalyMatch UI.

    Wraps ``ipyfilechooser.FileChooser`` with ESA-branded dark styling
    scoped via the ``.am-file-chooser`` CSS class.

    Args:
        path: Initial directory to display.
        select_default: Whether to pre-select the default path.
        show_only_dirs: If ``True``, only directories are shown.
        filter_pattern: Glob pattern to filter files (e.g. ``"*.pth"``).
        title: Optional label displayed above the chooser.
        purpose: Stable key identifying this chooser site (e.g.
            ``"model_chooser"``, ``"source_chooser"``).  When set, the
            chooser writes through the parent directory of every
            confirmed selection to :mod:`anomaly_match_ui.utils.ui_state`,
            so that the next time the user opens a chooser with the same
            purpose it can default to where they were last browsing
            instead of the original cfg value's directory (#429).
    """

    _style_widget: widgets.HTML | None = None

    def __init__(
        self,
        path: str = "",
        *,
        filename: str = "",
        select_default: bool = False,
        show_only_dirs: bool = False,
        filter_pattern: str = "",
        title: str = "",
        purpose: str = "",
    ) -> None:
        if not path:
            path = os.getcwd()

        fc_kwargs: dict = {
            "path": path,
            "select_default": select_default,
            "show_only_dirs": show_only_dirs,
        }
        if filename:
            fc_kwargs["filename"] = filename
        if filter_pattern:
            fc_kwargs["filter_pattern"] = filter_pattern

        self._file_chooser = FileChooser(**fc_kwargs)
        self._purpose = purpose
        # ipyfilechooser.FileChooser.register_callback **replaces** the
        # callback rather than appending — so we own a list and dispatch
        # to all registered handlers from a single bound shim.  Without
        # this, every screen-side ``register_callback`` overwrites the
        # persistence hook and the last-browsed dir never gets recorded
        # (#429 follow-up).
        self._callbacks: list[Callable] = []
        self._file_chooser.register_callback(self._dispatch_callbacks)
        if purpose:
            self._callbacks.append(self._persist_browsed_dir)

        # Inject CSS only once across all instances
        if AMFileChooser._style_widget is None:
            AMFileChooser._style_widget = widgets.HTML(value=_CHOOSER_STYLE)

        children: list[widgets.Widget] = [AMFileChooser._style_widget]

        if title:
            children.append(
                widgets.HTML(
                    value=(
                        f'<div style="color:{TEXT_COLOR}; font-size:13px;'
                        f' font-weight:bold; padding:4px 0;">{title}</div>'
                    ),
                )
            )

        children.append(self._file_chooser)

        super().__init__(
            children=children,
            layout=widgets.Layout(
                min_height="120px",
                max_height="280px",
                width="100%",
                max_width="100%",
            ),
        )
        self.add_class("am-file-chooser")

    @property
    def selected(self) -> str | None:
        """Return the currently selected path, or ``None``."""
        return self._file_chooser.selected

    @property
    def selected_filename(self) -> str | None:
        """Return the selected filename component."""
        return self._file_chooser.selected_filename

    def register_callback(self, callback: Callable) -> None:
        """Observe selection changes.

        Multiple callers can register; each is invoked in registration
        order on every selection.  An exception in one does not stop
        the rest from firing.

        Args:
            callback: Called with the underlying ``FileChooser``
                instance whenever the user clicks Select.
        """
        self._callbacks.append(callback)

    def _dispatch_callbacks(self, fc: FileChooser) -> None:
        """Fan out a Select event to every registered callback."""
        for cb in list(self._callbacks):
            try:
                cb(fc)
            except Exception as exc:
                logger.warning("AMFileChooser callback {!r} failed: {}", cb, exc)

    def reset(self, path: str | None = None) -> None:
        """Reset the chooser, optionally to a new *path*.

        Args:
            path: New starting directory. Uses the current directory if ``None``.
        """
        if path:
            self._file_chooser.reset(path)
        else:
            self._file_chooser.reset()

    def _persist_browsed_dir(self, _chooser: FileChooser) -> None:
        """Write the current selection's parent directory to ``ui_state``.

        Registered with the underlying ``FileChooser`` whenever a
        ``purpose`` was supplied — fires every time the user clicks
        Select.  ``ui_state.set_last_browsed_dir`` is best-effort and
        swallows I/O errors.
        """
        selected = self._file_chooser.selected
        if not selected:
            return
        ui_state.set_last_browsed_dir(self._purpose, selected)
