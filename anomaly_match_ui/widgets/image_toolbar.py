#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Image transform toolbar: brightness, contrast, invert, unsharp, remember."""

from __future__ import annotations

from typing import TYPE_CHECKING

import ipywidgets as widgets
from ipywidgets import Button, HBox, VBox

from anomaly_match_ui.styles import BG_COLOR, ESA_ORANGE, TEXT_COLOR

if TYPE_CHECKING:
    from anomaly_match_ui.widgets.preview_widget import PreviewWidget


class ImageToolbar:
    """Toolbar of image transform controls wired to a PreviewWidget.

    Args:
        preview: The PreviewWidget whose transforms this toolbar controls.
    """

    def __init__(self, preview: PreviewWidget) -> None:
        self._preview = preview

        # Transform buttons
        self.invert_button = Button(
            description="Invert",
            button_style="success",
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
            style={"button_color": TEXT_COLOR, "text_color": "black"},
        )

        self.restore_button = Button(
            description="Restore",
            button_style="success",
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
            style={"button_color": TEXT_COLOR, "text_color": "black"},
        )

        self.unsharp_button = Button(
            description="Unsharp Mask",
            button_style="success",
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
            style={"button_color": TEXT_COLOR, "text_color": "black"},
        )

        self.remember_button = Button(
            description="Remember",
            button_style="warning",
            layout=widgets.Layout(width="auto", flex="1 1 auto"),
            style={"button_color": ESA_ORANGE},
        )

        # Sliders.  continuous_update=False so each transform fires once on
        # release, not on every intermediate value: re-applying brightness/
        # contrast and re-encoding the displayed image runs on the UI thread, so
        # a continuous drag queued dozens of those and felt laggy.
        self.brightness_slider = widgets.FloatSlider(
            value=1.0,
            min=0.5,
            max=2.0,
            step=0.01,
            description="Brightness",
            continuous_update=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="50%"),
            style={"description_width": "initial", "handle_color": "white"},
        )

        self.contrast_slider = widgets.FloatSlider(
            value=1.0,
            min=0.5,
            max=2.0,
            step=0.01,
            description="Contrast",
            continuous_update=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="50%"),
            style={"description_width": "initial", "handle_color": "white"},
        )

        # RGB channel checkboxes
        channel_label = widgets.HTML(
            value='<div style="color:white; margin-right:5px;">RGB:</div>',
            layout=widgets.Layout(background_color=BG_COLOR, width="35px"),
        )
        self.red_checkbox = widgets.Checkbox(
            value=True,
            description="R",
            indent=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="35px"),
            style={"description_width": "15px"},
        )
        self.green_checkbox = widgets.Checkbox(
            value=True,
            description="G",
            indent=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="35px"),
            style={"description_width": "15px"},
        )
        self.blue_checkbox = widgets.Checkbox(
            value=True,
            description="B",
            indent=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="35px"),
            style={"description_width": "15px"},
        )

        self.channel_controls = HBox(
            [channel_label, self.red_checkbox, self.green_checkbox, self.blue_checkbox],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                width="180px",
                justify_content="flex-start",
            ),
        )

        # Wire internal event handlers
        self.invert_button.on_click(lambda _: self._preview.toggle_invert())
        self.unsharp_button.on_click(lambda _: self._preview.toggle_unsharp_mask())
        self.restore_button.on_click(lambda _: self.restore())

        def _on_brightness_contrast(_: object) -> None:
            self._preview.set_brightness(self.brightness_slider.value)
            self._preview.set_contrast(self.contrast_slider.value)

        self.brightness_slider.observe(_on_brightness_contrast, names="value")
        self.contrast_slider.observe(_on_brightness_contrast, names="value")

        self.red_checkbox.observe(
            lambda c: self._preview.set_rgb_channels(r=c["new"]), names="value"
        )
        self.green_checkbox.observe(
            lambda c: self._preview.set_rgb_channels(g=c["new"]), names="value"
        )
        self.blue_checkbox.observe(
            lambda c: self._preview.set_rgb_channels(b=c["new"]), names="value"
        )

        # Compose layout
        button_row = HBox(
            [
                self.invert_button,
                self.restore_button,
                self.unsharp_button,
                self.remember_button,
            ],
            layout=widgets.Layout(background_color=BG_COLOR, width="600px"),
        )
        slider_row = HBox(
            [self.brightness_slider, self.contrast_slider],
            layout=widgets.Layout(background_color=BG_COLOR),
        )

        self.widget = VBox(
            [button_row, slider_row],
            layout=widgets.Layout(background_color=BG_COLOR),
        )

    def restore(self) -> None:
        """Reset all controls to defaults and restore the preview image."""
        self.brightness_slider.value = 1.0
        self.contrast_slider.value = 1.0
        self.red_checkbox.value = True
        self.green_checkbox.value = True
        self.blue_checkbox.value = True
        self._preview.restore()
