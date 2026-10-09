#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""PreviewWidget: image viewer with transform controls for the detail screen."""

import os

import ipywidgets as widgets
import numpy as np
from ipywidgets import VBox

from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.display_transforms import (
    apply_transforms_ui,
    display_image_normalisation,
)
from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream


class PreviewWidget:
    """Image viewer with transform controls.

    Used by :class:`ImageDetailScreen` to display a single image loaded
    from the prediction cache (via ``display_standalone``).
    """

    def __init__(self):
        self.num_channels = BackendInterface.get_num_channels()

        self.filename_text = widgets.HTML(
            value="",
            layout=widgets.Layout(background_color="black"),
            style={"color": "white"},
        )

        self.image_widget = widgets.Image(
            value=b"",
            width=600,
            height=600,
            layout=widgets.Layout(background_color="black"),
        )

        self.widget = VBox(
            [self.filename_text, self.image_widget],
            layout=widgets.Layout(background_color="black"),
        )

        # Transform states
        self.invert = False
        self.brightness = 1.0
        self.contrast = 1.0
        self.unsharp_mask_applied = False

        # Channel visibility.  display_image_normalisation always collapses to a
        # 3-channel RGB image (picking rgb_mapping for >3-band sources), so the
        # R/G/B toggles act on those three displayed channels regardless of the
        # source's band count.
        self.show_r = True
        self.show_g = True
        self.show_b = True
        self.rgb_mapping = [0, 1, 2] if self.num_channels >= 3 else list(range(self.num_channels))

        # Image data
        self.original_image = None
        self.modified_image = None

    # ── Public API ────────────────────────────────────────────────

    def display_standalone(self, image_array, filename, score):
        """Display an image directly without going through BackendInterface.

        Used by the image detail screen to show a single image loaded
        from the prediction cache.  Current display transforms (invert,
        brightness, contrast, unsharp, channel visibility) are re-applied
        so that a re-decode triggered by changing upstream normalisation
        doesn't desync from the toolbar's toggle state — without this,
        clicking Invert then changing the channel combination would drop
        the inversion while leaving the user with a button that has to
        be clicked twice to re-invert.

        Args:
            image_array: Numpy array (HWC, uint8) of the image.
            filename: The source filename for the label.
            score: The anomaly score for the label.
        """
        rgb_map = self.rgb_mapping if self.num_channels > 3 else None
        self.original_image = display_image_normalisation(image_array, rgb_mapping=rgb_map)
        short = os.path.basename(filename) if filename else ""
        self.filename_text.value = (
            f'<span style="color:white;">Name: {short}</span>'
            f'<span style="float:right; color:white;">Score: {score:.4f}</span>'
        )
        self._apply_transforms_and_display()

    # ── Transform methods ─────────────────────────────────────────

    def restore(self):
        """Restore the current image to its original state."""
        self.invert = False
        self.brightness = 1.0
        self.contrast = 1.0
        self.unsharp_mask_applied = False
        self.show_r = True
        self.show_g = True
        self.show_b = True
        self.modified_image = self.original_image
        # Skip the redraw on a fresh screen where no image has been
        # loaded yet — ``display_standalone`` will paint the real image
        # immediately afterwards.
        if self.original_image is not None:
            self._display_image(self.modified_image)

    def toggle_invert(self):
        """Toggle colour inversion."""
        self.invert = not self.invert
        self._apply_transforms_and_display()

    def toggle_unsharp_mask(self):
        """Toggle unsharp-mask sharpening."""
        self.unsharp_mask_applied = not self.unsharp_mask_applied
        self._apply_transforms_and_display()

    def set_brightness(self, value):
        """Set brightness and update display."""
        self.brightness = value
        self._apply_transforms_and_display()

    def set_contrast(self, value):
        """Set contrast and update display."""
        self.contrast = value
        self._apply_transforms_and_display()

    def set_rgb_channels(self, r=None, g=None, b=None):
        """Set RGB channel visibility."""
        if r is not None:
            self.show_r = r
        if g is not None:
            self.show_g = g
        if b is not None:
            self.show_b = b
        self._apply_transforms_and_display()

    # ── Internal ──────────────────────────────────────────────────

    def _display_image(self, img, filename=None, score=None):
        """Display the given image array in the widget."""
        image_byte_stream = numpy_array_to_byte_stream(np.array(img))
        self.image_widget.value = image_byte_stream

    def _apply_transforms_and_display(self):
        """Apply current transforms and update the display."""
        # Drive channel toggling through show_r/g/b: the displayed image is
        # always 3-channel RGB, so these map directly to it.  (The old
        # channel_visibility override silently ignored the toggles for >3-band
        # sources, so the R/G/B checkboxes did nothing on the detail screen.)
        self.modified_image = apply_transforms_ui(
            self.original_image,
            invert=self.invert,
            brightness=self.brightness,
            contrast=self.contrast,
            unsharp_mask_applied=self.unsharp_mask_applied,
            show_r=self.show_r,
            show_g=self.show_g,
            show_b=self.show_b,
        )
        self._display_image(self.modified_image)
