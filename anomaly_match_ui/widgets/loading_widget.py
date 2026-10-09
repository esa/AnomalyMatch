#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Loading spinner with AM logo and live log tail."""

from __future__ import annotations

import html as html_lib

import ipywidgets as widgets
from loguru import logger

from anomaly_match_ui.styles import BG_COLOR
from anomaly_match_ui.utils.svg_loader import load_am_logo_base64

_MAX_LOG_LINES = 3


class LoadingWidget:
    """A loading indicator with the AM logo, spinner, and live log tail.

    Shows a centred spinner with the AM logo and the last few log messages
    automatically updating so users can track progress during screen
    transitions and long operations.
    """

    def __init__(self) -> None:
        am_src = load_am_logo_base64()

        self._logo_html = widgets.HTML(
            value=(
                f'<div style="text-align:center;">'
                f'<img src="{am_src}" style="height:120px;" alt="AnomalyMatch"/>'
                f"</div>"
            ),
        )

        self._spinner_html = widgets.HTML(
            value=(
                '<div style="text-align:center;">'
                '<div class="am-spinner" style="margin:16px auto 8px auto;"></div>'
                '<div style="color:#aaa; font-size:13px;">Loading...</div>'
                "</div>"
            ),
        )

        self._log_html = widgets.HTML(
            value=self._render_log_lines([]),
        )

        self.widget = widgets.VBox(
            [self._logo_html, self._spinner_html, self._log_html],
            layout=widgets.Layout(
                display="flex",
                align_items="center",
                justify_content="center",
                min_height="350px",
                background_color=BG_COLOR,
                padding="24px",
            ),
        )

        self._log_lines: list[str] = []
        self._handler_id: int | None = None

    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start capturing log messages into the live tail."""
        self._log_lines.clear()
        self._log_html.value = self._render_log_lines([])
        if self._handler_id is None:
            self._handler_id = logger.add(self._sink, format="{message}", level="DEBUG")

    def stop(self) -> None:
        """Stop capturing log messages."""
        if self._handler_id is not None:
            try:
                logger.remove(self._handler_id)
            except ValueError:
                pass
            self._handler_id = None

    # ------------------------------------------------------------------

    def _sink(self, message: str) -> None:
        text = str(message).rstrip()
        if not text:
            return
        self._log_lines.append(text)
        if len(self._log_lines) > _MAX_LOG_LINES:
            self._log_lines = self._log_lines[-_MAX_LOG_LINES:]
        self._log_html.value = self._render_log_lines(self._log_lines)

    @staticmethod
    def _render_log_lines(lines: list[str]) -> str:
        escaped = [html_lib.escape(ln) for ln in lines]
        joined = "<br>".join(escaped) if escaped else "&nbsp;"
        return (
            f'<div style="color:#666; font-size:11px; font-family:monospace;'
            f" text-align:center; line-height:1.5; min-height:50px;"
            f' margin-top:12px; max-width:600px;">'
            f"{joined}</div>"
        )
