#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Main menu screen with ESA branding, navigation cards, and help."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as widgets
from ipywidgets import HBox, VBox

from anomaly_match_ui.screens.base_screen import BaseScreen
from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT, ESA_BLUE_DEEP, ESA_GREEN
from anomaly_match_ui.utils.svg_loader import load_am_logo_base64, load_icon_svg
from anomaly_match_ui.widgets.header_widget import HeaderWidget

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp


def _make_card(
    title: str,
    description: str,
    icon_svg: str,
    button_color: str,
    on_click: object,
) -> widgets.Widget:
    """Build a navigation card with icon, title, description, and click action.

    Args:
        title: Card heading text.
        description: Short description shown below the heading.
        icon_svg: Inline SVG markup for the card icon.
        button_color: Background color for the card border accent.
        on_click: Callback for the card button.

    Returns:
        A VBox widget representing the card.
    """
    icon_html = widgets.HTML(
        value=f'<div style="text-align:center; padding:8px 0;">{icon_svg}</div>',
    )

    text_html = widgets.HTML(
        value=(
            f'<div style="text-align:center; color:white;">'
            f'<div style="font-size:16px; font-weight:bold; margin-bottom:4px;">{title}</div>'
            f'<div style="font-size:12px; color:#aaa; line-height:1.4;">{description}</div>'
            f"</div>"
        ),
    )

    btn = widgets.Button(
        description=f"Open {title}",
        button_style="primary",
        layout=widgets.Layout(width="100%", height="36px", margin="8px 0 0 0"),
        style={"button_color": button_color, "font_size": "13px"},
    )
    btn.on_click(on_click)

    return VBox(
        [icon_html, text_html, btn],
        layout=widgets.Layout(
            background_color="#0a0a0a",
            border=f"1px solid {button_color}",
            padding="16px",
            width="260px",
            border_radius="8px",
        ),
    )


class MainMenuScreen(BaseScreen):
    """Welcome screen with navigation cards and header.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)

    def build(self) -> widgets.Widget:
        """Build and return the main menu layout.

        Returns:
            The root VBox widget for the main menu screen.
        """
        header = HeaderWidget(self.app)

        # Large AM logo
        am_logo_src = load_am_logo_base64()
        logo_widget = widgets.HTML(
            value=(
                f'<div style="text-align:center; padding:32px 0 8px 0;">'
                f'<img src="{am_logo_src}" style="height:120px;" alt="AnomalyMatch"/>'
                f"</div>"
            ),
        )

        # Navigation cards
        training_card = _make_card(
            title="Training",
            description=(
                "Label images, train the anomaly detection model, "
                "and evaluate results with active learning."
            ),
            icon_svg=load_icon_svg("training_icon.svg"),
            button_color=ESA_GREEN,
            on_click=lambda _: self.navigate_to("training_setup"),
        )

        prediction_card = _make_card(
            title="Prediction",
            description=(
                "Run the trained model on new data to find "
                "anomalies across large image collections."
            ),
            icon_svg=load_icon_svg("prediction_icon.svg"),
            button_color=ESA_BLUE_BRIGHT,
            on_click=lambda _: self.navigate_to("prediction_setup"),
        )

        card_row = HBox(
            [training_card, prediction_card],
            layout=widgets.Layout(
                justify_content="center",
                gap="32px",
                padding="24px 0",
            ),
        )

        return VBox(
            [header.widget, logo_widget, card_row],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                border=f"1px solid {ESA_BLUE_DEEP}",
                min_height="400px",
                width="100%",
            ),
        )

    def on_enter(self) -> None:
        """Refresh the main menu when returning to it."""
        self._widget = None
