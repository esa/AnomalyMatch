#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Reusable header widget with ESA/AM branding, version, log controls, and help."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as widgets
from ipywidgets import HBox, VBox

from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT, ESA_BLUE_DEEP, TEXT_COLOR
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.svg_loader import get_dual_logo_html

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp

_LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]

_HELP_TEXT = (
    '<div style="color:white; padding:8px; font-size:13px;">'
    "<b>AnomalyMatch Help</b><br><br>"
    "<b>Workflow:</b> Label images &rarr; Train &rarr; Evaluate &rarr; Repeat<br>"
    "<b>Sorting:</b> Rank images by anomaly score, normal score, or distance to mean/median<br>"
    "<b>Transforms:</b> Invert, brightness, contrast, unsharp mask, full resolution<br><br>"
    '<a href="https://anomalymatch.readthedocs.io" target="_blank" '
    'style="color:#00a1de;">Documentation</a>'
    "</div>"
)


class HeaderWidget:
    """Reusable application header bar with ESA/AM branding.

    Supports two modes controlled by constructor arguments:

    * **Main-menu mode** (default): version left, dual logos centred, help right.
    * **Sub-screen mode** (``out_widget`` provided): adds back button, log-level
      dropdown, and log-toggle button.

    Args:
        app: The parent application instance (used for navigation).
        out_widget: The output widget for terminal logs.  ``None`` hides
            log-level / log-toggle controls.
        show_back_button: If ``True``, a back button is shown on the left.
        back_to: Screen name to navigate to when back is pressed.
            Defaults to ``"main_menu"``.
    """

    def __init__(
        self,
        app: AnomalyMatchApp,
        out_widget: widgets.Output | None = None,
        *,
        show_back_button: bool = False,
        back_to: str = "main_menu",
        config_text: str = "",
    ) -> None:
        self._app = app
        self._back_btn: widgets.Button | None = None
        self._build(app, out_widget, show_back_button, back_to, config_text)

    # ------------------------------------------------------------------

    def _build(
        self,
        app: AnomalyMatchApp,
        out_widget: widgets.Output | None,
        show_back_button: bool,
        back_to: str = "main_menu",
        config_text: str = "",
    ) -> None:
        show_log_controls = out_widget is not None

        # ---- left section ----
        left_items: list[widgets.Widget] = []
        if show_back_button:
            label = "\u2190 Back" if back_to != "main_menu" else "\u2190 Menu"
            back_btn = widgets.Button(
                description=label,
                button_style="",
                layout=widgets.Layout(width="80px", height="28px"),
                style={"button_color": ESA_BLUE_BRIGHT, "font_size": "11px"},
            )
            back_btn.on_click(lambda _: self._app.navigate_to(self._back_to))
            self._back_btn = back_btn
            self._back_to = back_to
            left_items.append(back_btn)

        version_label = f"v{BackendInterface.get_version()}"
        commit = BackendInterface.get_git_commit()
        if commit:
            version_label += f" ({commit})"
        version_html = widgets.HTML(
            value=(
                f'<div style="color:{TEXT_COLOR}; font-size:11px; white-space:nowrap;">'
                f"{version_label}</div>"
            ),
            layout=widgets.Layout(width="auto"),
        )
        left_items.append(version_html)

        left_box = HBox(
            left_items,
            layout=widgets.Layout(
                gap="4px",
                align_items="center",
                min_width="100px",
                flex="0 0 auto",
            ),
        )

        # ---- centre: dual logos ----
        logo_html = widgets.HTML(
            value=get_dual_logo_html(),
            layout=widgets.Layout(
                flex="1 1 auto",
            ),
        )

        # ---- right section ----
        help_button, self._help_panel = self._build_help_panel()
        right_items: list[widgets.Widget] = []

        if show_log_controls:
            log_level_dropdown = widgets.Dropdown(
                options=_LOG_LEVELS,
                value="DEBUG",
                layout=widgets.Layout(width="100px"),
                style={"description_width": "0px"},
            )

            def _on_log_level_change(change: dict) -> None:
                BackendInterface.set_log_level(change["new"])

            log_level_dropdown.observe(_on_log_level_change, names="value")
            right_items.append(log_level_dropdown)

        right_items.append(help_button)

        if show_log_controls:
            log_toggle = widgets.ToggleButton(
                value=True,
                description="Log",
                button_style="",
                layout=widgets.Layout(width="50px", height="28px"),
                style={"font_size": "11px"},
            )

            def _toggle_log(change: dict) -> None:
                out_widget.layout.display = "" if change["new"] else "none"

            log_toggle.observe(_toggle_log, names="value")
            right_items.append(log_toggle)

        right_box = HBox(
            right_items,
            layout=widgets.Layout(
                gap="4px",
                justify_content="flex-end",
                min_width="100px",
                flex="0 0 auto",
            ),
        )

        # ---- assemble ----
        header_bar = HBox(
            [left_box, logo_html, right_box],
            layout=widgets.Layout(
                background_color=ESA_BLUE_DEEP,
                padding="6px 12px",
                border_bottom=f"2px solid {ESA_BLUE_BRIGHT}",
                width="100%",
                align_items="center",
                overflow="hidden",
            ),
        )

        # Optional config info bar
        self._config_html = widgets.HTML(
            value="",
            layout=widgets.Layout(
                padding="2px 12px",
                background_color=ESA_BLUE_DEEP,
                display="none",
            ),
        )
        if config_text:
            self.set_config_text(config_text)

        self.widget = VBox(
            [header_bar, self._config_html, self._help_panel],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                width="100%",
                overflow="hidden",
            ),
        )

    # ------------------------------------------------------------------

    def set_back_to(self, screen_name: str) -> None:
        """Re-target the back button at *screen_name* on the next click.

        Image-detail navigation needs this — the same screen serves
        both training and prediction galleries, so the back-button
        target depends on which gallery navigated here (#427 / #434
        follow-up).

        Args:
            screen_name: Screen name to navigate to on back-click.
        """
        self._back_to = screen_name
        if self._back_btn is not None:
            self._back_btn.description = "← Back" if screen_name != "main_menu" else "← Menu"

    def set_config_text(self, text: str) -> None:
        """Update the config info bar below the header.

        Args:
            text: HTML content for the config bar. Empty string hides it.
        """
        if text:
            self._config_html.value = f'<span style="color:#ccc; font-size:11px;">{text}</span>'
            self._config_html.layout.display = ""
        else:
            self._config_html.value = ""
            self._config_html.layout.display = "none"

    @staticmethod
    def _build_help_panel() -> tuple[widgets.Button, widgets.HTML]:
        """Build the help button and collapsible help panel.

        Returns:
            Tuple of (help_button, help_panel).
        """
        help_panel = widgets.HTML(
            value=_HELP_TEXT,
            layout=widgets.Layout(
                display="none",
                background_color=ESA_BLUE_DEEP,
                border=f"1px solid {ESA_BLUE_BRIGHT}",
                padding="8px",
                width="400px",
            ),
        )

        help_button = widgets.Button(
            description="?",
            button_style="info",
            layout=widgets.Layout(width="36px", height="28px"),
        )

        def _toggle_help(_: object) -> None:
            if help_panel.layout.display == "none":
                help_panel.layout.display = "block"
            else:
                help_panel.layout.display = "none"

        help_button.on_click(_toggle_help)
        return help_button, help_panel
