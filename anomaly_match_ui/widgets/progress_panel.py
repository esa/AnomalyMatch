#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Reusable progress bar with status label and ETA tracking."""

from __future__ import annotations

import ipywidgets as widgets

from anomaly_match_ui.styles import BG_COLOR, ESA_GREEN, inline_spinner_html


class ProgressPanel:
    """A progress bar with a status label and a waiting spinner.

    Args:
        bar_color: Initial color of the progress bar.
    """

    def __init__(self, bar_color: str = ESA_GREEN) -> None:
        self.progress_bar = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            layout=widgets.Layout(background_color=BG_COLOR),
            style={"bar_color": bar_color},
        )

        # Shown from start() until the bar first moves: subprocess startup
        # (model load, Cutana orchestrator init, first tile stream) can take
        # minutes with the bar at 0, which otherwise looks like a hang.
        self.spinner = widgets.HTML(value="", layout=widgets.Layout(width="20px"))
        self._waiting = False
        self.progress_bar.observe(self._on_value_change, names="value")

        self.status_label = widgets.Label(
            value="",
            layout=widgets.Layout(background_color=BG_COLOR),
            style={"color": "white"},
        )

    def set_color(self, color: str) -> None:
        """Change the progress bar color.

        Args:
            color: CSS color string.
        """
        self.progress_bar.style = {"bar_color": color}

    def start(self, message: str) -> None:
        """Reset the bar for a new run and spin until the first progress.

        Args:
            message: Status text to display while waiting.
        """
        self.update(0.0, message)
        self._set_waiting(True)

    def stop(self) -> None:
        """End the waiting spinner: the run finished, failed or was stopped."""
        self._set_waiting(False)

    def _set_waiting(self, waiting: bool) -> None:
        self._waiting = waiting
        self.spinner.value = inline_spinner_html() if waiting else ""

    def _on_value_change(self, change: dict) -> None:
        if self._waiting and change["new"] > 0:
            self._set_waiting(False)

    def update(self, value: float, message: str) -> None:
        """Update progress value and status message.

        Args:
            value: Progress fraction between 0.0 and 1.0.
            message: Status text to display.
        """
        self.progress_bar.value = value
        self.status_label.value = message
