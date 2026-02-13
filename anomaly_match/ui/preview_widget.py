#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""
PreviewWidget: A self-contained widget for displaying and manipulating preview images.
"""

import os
import numpy as np
import ipywidgets as widgets
from ipywidgets import VBox
from loguru import logger

from anomaly_match.data_io.load_images import load_and_process_single_wrapper
from anomaly_match.image_processing.display_transforms import (
    apply_transforms_ui,
    display_image_normalisation,
)
from anomaly_match.utils.numpy_to_byte_stream import numpy_array_to_byte_stream


class PreviewWidget:
    """A widget component for displaying images with transformation controls."""

    def __init__(self, session):
        """
        Initialize the preview widget.

        Args:
            session: The session object providing image data and configuration.
        """
        self.session = session

        # Create UI elements
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

        # Compose into a VBox
        self.widget = VBox(
            [self.filename_text, self.image_widget],
            layout=widgets.Layout(background_color="black"),
        )

        # Current image index
        self.current_index = 0

        # Transform states
        self.invert = False
        self.brightness = 1.0
        self.contrast = 1.0
        self.unsharp_mask_applied = False
        self.show_r = True
        self.show_g = True
        self.show_b = True
        self.full_resolution_mode = False

        # Image data
        self.original_image = None
        self.modified_image = None

        # Reference to full resolution button (set by parent)
        self._full_res_button = None

    def set_full_res_button(self, button):
        """Set the full resolution button reference for updating its state."""
        self._full_res_button = button

    def set_index(self, index):
        """Set the current image index."""
        self.current_index = index

    def update_display(self):
        """Updates the display of the current image."""
        filename = self.session.filenames[self.current_index]
        score = self.session.scores[self.current_index]
        filepath = os.path.join(self.session.cfg.data_dir, filename)

        # Determine if we need to reload from disk
        needs_reload = (
            self.full_resolution_mode
            or self.session.cfg.normalisation.normalisation_method
            != self.session.cached_image_normalisation_enum
        )

        if needs_reload:
            try:
                size_override = None if self.full_resolution_mode else "default"
                logger.debug(
                    f"Loading image from {filename} (full_res={self.full_resolution_mode})"
                )

                img = load_and_process_single_wrapper(
                    filepath,
                    self.session.cfg,
                    desc="widget loading image",
                    show_progress=False,
                    size_override=size_override,
                )

                self.original_image = display_image_normalisation(img)
            except Exception as e:
                logger.error(f"Error loading image {filepath}: {e}")
                return
        else:
            img = self.session.img_catalog[self.current_index]
            self.original_image = display_image_normalisation(img)

        # Apply transforms
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

        self._display_image(self.modified_image, filename, score)

    def _display_image(self, img, filename=None, score=None):
        """Displays the given PIL image in the widget."""
        image_byte_stream = numpy_array_to_byte_stream(np.array(img))
        self.image_widget.value = image_byte_stream
        self._update_label(filename, score)

    def _update_label(self, filename=None, score=None):
        """Updates the UI label with the current image's filename, score, and label."""
        label_color = "white"
        label_text = "None"
        label = self.session.get_label(self.current_index)
        if label == "anomaly":
            label_color = "red"
            label_text = "Anomalous"
        elif label == "normal":
            label_color = "green"
            label_text = "Nominal"

        # Get counts for anomalies and nominal samples
        normal_count, anomalous_count = self.session.get_label_distribution()

        # Calculate newly annotated samples using cached method
        new_nominal, new_anomalous = self.session.get_active_learning_counts()

        # Format the file name (shortened version)
        fname = self.session.filenames[self.current_index]
        fname_short = os.path.basename(fname)
        if len(fname_short) > 57:
            fname_short = (
                fname_short[:45]
                + "..."
                + fname_short.split(".")[-2][-5:]
                + "."
                + fname_short.split(".")[-1]
            )
        sc = self.session.scores[self.current_index]
        total_len = len(self.session.img_catalog) - 1

        self.filename_text.value = (
            f'<span style="color:white;">'
            f"Name: {fname_short}"
            f"</span>"
            f'<span style="float:right; text-align:right; color:white;">'
            f"Score: {sc:.2f} | Index: {self.current_index}/{total_len}"
            f"</span>"
            f'<br style="clear:both;">'
            f'<span style="color:white">Label: </span>'
            f'<span style="color:{label_color}">{label_text}</span>'
            f'<span style="float:right; text-align:right;">'
            f'<span style="color:red">Anomalies: {anomalous_count}(+{new_anomalous})</span> | '
            f'<span style="color:green">Nominal: {normal_count}(+{new_nominal})</span>'
            f"</span>"
        )

    def update_label_only(self):
        """Updates only the label without reloading the image."""
        self._update_label()

    # ======== Transform Methods ========
    def restore(self):
        """Restores the current image to its original state."""
        self.invert = False
        self.brightness = 1.0
        self.contrast = 1.0
        self.unsharp_mask_applied = False
        self.show_r = True
        self.show_g = True
        self.show_b = True

        self.modified_image = self.original_image
        self._display_image(self.modified_image)

    def toggle_invert(self):
        """Toggles the inversion of the current image."""
        self.invert = not self.invert
        self._apply_transforms_and_display()

    def toggle_unsharp_mask(self):
        """Toggles the application of an unsharp mask."""
        self.unsharp_mask_applied = not self.unsharp_mask_applied
        self._apply_transforms_and_display()

    def set_brightness(self, value):
        """Sets brightness and updates display."""
        self.brightness = value
        self._apply_transforms_and_display()

    def set_contrast(self, value):
        """Sets contrast and updates display."""
        self.contrast = value
        self._apply_transforms_and_display()

    def set_rgb_channels(self, r=None, g=None, b=None):
        """Sets RGB channel visibility."""
        if r is not None:
            self.show_r = r
        if g is not None:
            self.show_g = g
        if b is not None:
            self.show_b = b
        self._apply_transforms_and_display()

    def toggle_full_resolution(self):
        """Toggles full resolution mode and updates the display."""
        self.full_resolution_mode = not self.full_resolution_mode
        self._update_full_res_button()
        self.update_display()

    def reset_full_resolution_mode(self):
        """Resets full resolution mode to preview mode."""
        if self.full_resolution_mode:
            self.full_resolution_mode = False
            self._update_full_res_button()

    def _update_full_res_button(self):
        """Updates the full resolution button appearance."""
        if self._full_res_button is None:
            return
        if self.full_resolution_mode:
            self._full_res_button.description = "Show Preview"
            self._full_res_button.style.button_color = "#17a2b8"
        else:
            self._full_res_button.description = "Show Full Resolution"
            self._full_res_button.style.button_color = "#ffffff"

    def _apply_transforms_and_display(self):
        """Applies current transforms and updates display."""
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
