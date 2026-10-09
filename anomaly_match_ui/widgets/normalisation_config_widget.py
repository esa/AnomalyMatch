#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Normalisation configuration widget for FITS image processing.

Provides UI controls for normalisation method, per-channel parameters,
channel-combination matrix, image resolution, and interpolation order.
"""

from __future__ import annotations

from collections.abc import Callable

import ipywidgets as widgets
import numpy as np
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from ipywidgets import HBox, VBox
from loguru import logger

from anomaly_match.utils.normalisation_parameters import (
    ASINH_CLIP_DEFAULT,
    ASINH_CLIP_MAX,
    ASINH_CLIP_MIN,
    ASINH_SCALE_DEFAULT,
    ASINH_SCALE_MAX,
    ASINH_SCALE_MIN,
    CHANNEL_COMBINATION_MAX,
    CHANNEL_COMBINATION_MIN,
    CUTOUT_PADDING_FACTOR_DEFAULT,
    CUTOUT_PADDING_FACTOR_MAX,
    CUTOUT_PADDING_FACTOR_MIN,
    DEFAULT_IMAGE_EXTENSIONS,
    RESOLUTION_DEFAULT,
    RESOLUTION_MAX,
    RESOLUTION_MIN,
    validate_normalisation,
)
from anomaly_match_ui.styles import (
    BG_COLOR,
    ESA_BLUE_BRIGHT,
    FORM_CONTROL_CSS,
    TEXT_COLOR,
    WIDGET_LAYOUT_KWARGS,
    WIDGET_STYLE,
    WIDGET_WIDE_LAYOUT_KWARGS,
)

_STYLE = WIDGET_STYLE
_LAYOUT = widgets.Layout(**WIDGET_LAYOUT_KWARGS)
_WIDE_LAYOUT = widgets.Layout(**WIDGET_WIDE_LAYOUT_KWARGS)


class NormalisationConfigWidget:
    """Widget for configuring fitsbolt normalisation settings.

    Args:
        n_channels: Initial number of output channels.
        image_size: Initial image size ``[height, width]``.
        extensions: Initial list of FITS extension names for the
            channel-combination matrix column headers.
    """

    def __init__(
        self,
        n_channels: int = 3,
        image_size: list[int] | None = None,
        extensions: list[str] | None = None,
    ) -> None:
        if image_size is None:
            image_size = [RESOLUTION_DEFAULT, RESOLUTION_DEFAULT]
        if extensions is None:
            extensions = list(DEFAULT_IMAGE_EXTENSIONS)

        self._on_change_callback: Callable[[], None] | None = None
        self._extensions = list(extensions)

        # ── Method dropdown ─────────────────────────────────────────
        self._method_dropdown = widgets.Dropdown(
            options=NormalisationMethod.get_options(),
            value=NormalisationMethod.CONVERSION_ONLY,
            description="Method",
            layout=_WIDE_LAYOUT,
            style=_STYLE,
        )
        self._method_dropdown.observe(self._on_method_change, names="value")

        # ── Resolution ──────────────────────────────────────────────
        # A free-entry number box rather than a slider: a slider quantises to
        # its step, which made most resolutions — including the 150 px default
        # — impossible to select by dragging (#472).
        # Own Layout instance (not the shared _WIDE_LAYOUT): show_resolution()
        # toggles this field's layout.display, and _WIDE_LAYOUT is shared with
        # the method dropdown + padding control — mutating it would hide those
        # too (the detail screen's "missing Method dropdown" bug).
        self._resolution_input = widgets.BoundedIntText(
            value=image_size[0],
            min=RESOLUTION_MIN,
            max=RESOLUTION_MAX,
            step=1,
            description="Resolution (px)",
            layout=widgets.Layout(**WIDGET_WIDE_LAYOUT_KWARGS),
            style=_STYLE,
        )

        # ── ASINH parameters (per-channel) ──────────────────────────
        self._asinh_scale_inputs: list[widgets.BoundedFloatText] = []
        self._asinh_clip_inputs: list[widgets.BoundedFloatText] = []
        for i in range(n_channels):
            self._asinh_scale_inputs.append(
                widgets.BoundedFloatText(
                    value=ASINH_SCALE_DEFAULT,
                    min=ASINH_SCALE_MIN,
                    max=ASINH_SCALE_MAX,
                    step=0.1,
                    description=f"Scale ch{i}",
                    layout=widgets.Layout(background_color=BG_COLOR, width="180px"),
                    style=_STYLE,
                )
            )
            self._asinh_clip_inputs.append(
                widgets.BoundedFloatText(
                    value=ASINH_CLIP_DEFAULT,
                    min=ASINH_CLIP_MIN,
                    max=ASINH_CLIP_MAX,
                    step=0.1,
                    description=f"Clip ch{i}",
                    layout=widgets.Layout(background_color=BG_COLOR, width="180px"),
                    style=_STYLE,
                )
            )
        self._asinh_box = VBox(
            [
                widgets.HTML(
                    f"<b style='color:{TEXT_COLOR}'>ASINH Parameters</b>",
                    layout=_LAYOUT,
                ),
                HBox(self._asinh_scale_inputs, layout=_LAYOUT),
                HBox(self._asinh_clip_inputs, layout=_LAYOUT),
            ],
            layout=_LAYOUT,
        )

        # ── LOG parameters ──────────────────────────────────────────
        self._log_calc_min = widgets.Checkbox(
            value=False,
            description="",
            indent=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="auto"),
        )
        self._log_box = VBox(
            [
                widgets.HTML(
                    f"<b style='color:{TEXT_COLOR}'>LOG Parameters</b>",
                    layout=_LAYOUT,
                ),
                HBox(
                    [
                        self._log_calc_min,
                        widgets.HTML(
                            f"<span style='color:{TEXT_COLOR}'>Auto-calculate minimum</span>",
                            layout=_LAYOUT,
                        ),
                    ],
                    layout=_LAYOUT,
                ),
            ],
            layout=_LAYOUT,
        )

        # ── Flux conversion ─────────────────────────────────────────
        # Checkbox is instantiated so get_normalisation_config() keeps
        # surfacing the value, but no corresponding row is placed in any
        # VBox — Cutana only supports Euclid today, where flux conversion
        # should always be on.  Restore a flux row in ``self.norm_section``
        # when non-Euclid data becomes supported (see follow-up issue).
        self._flux_checkbox = widgets.Checkbox(
            value=True,
            description="",
            indent=False,
            layout=widgets.Layout(background_color=BG_COLOR, width="auto"),
        )

        # ── Cutout padding factor (Cutana zoom) ────────────────────
        self._padding_factor = widgets.FloatSlider(
            value=CUTOUT_PADDING_FACTOR_DEFAULT,
            min=CUTOUT_PADDING_FACTOR_MIN,
            max=CUTOUT_PADDING_FACTOR_MAX,
            step=0.25,
            description="Cutout Zoom-out",
            readout=False,
            layout=_WIDE_LAYOUT,
            style={**_STYLE, "handle_color": "white"},
        )
        self._padding_factor_text = widgets.BoundedFloatText(
            value=CUTOUT_PADDING_FACTOR_DEFAULT,
            min=CUTOUT_PADDING_FACTOR_MIN,
            max=CUTOUT_PADDING_FACTOR_MAX,
            step=0.25,
            layout=widgets.Layout(width="70px", background_color=BG_COLOR),
        )
        widgets.link((self._padding_factor, "value"), (self._padding_factor_text, "value"))
        _hidden_layout = widgets.Layout(display="none")
        self._padding_factor_row = HBox(
            [self._padding_factor, self._padding_factor_text],
            layout=_hidden_layout,
        )

        # ── Channel-combination matrix ──────────────────────────────
        self._n_channels = n_channels
        self._matrix_cells: list[list[widgets.BoundedFloatText]] = []
        self._matrix_box = VBox(layout=_LAYOUT)
        self._build_matrix()

        self._add_ch_btn = widgets.Button(
            description="+ Channel",
            button_style="info",
            layout=widgets.Layout(width="auto"),
        )
        self._remove_ch_btn = widgets.Button(
            description="- Channel",
            button_style="warning",
            layout=widgets.Layout(width="auto"),
        )
        self._add_ch_btn.on_click(lambda _: self._add_channel())
        self._remove_ch_btn.on_click(lambda _: self._remove_channel())

        self._channel_controls = HBox(
            [self._add_ch_btn, self._remove_ch_btn],
            layout=_LAYOUT,
        )

        # ── Method-conditional container ────────────────────────────
        self._method_params_box = VBox(layout=_LAYOUT)
        self._update_method_params()

        # ── Grouped sections ─────────────────────────────────────────
        _section_layout = widgets.Layout(
            background_color=BG_COLOR,
            padding="8px",
            gap="6px",
            border=f"1px solid {ESA_BLUE_BRIGHT}",
        )

        self.norm_section = VBox(
            [
                widgets.HTML(
                    f"<b style='color:{ESA_BLUE_BRIGHT}; font-size:14px'>"
                    "Normalisation Configuration</b>",
                    layout=_LAYOUT,
                ),
                self._method_dropdown,
                self._resolution_input,
                self._padding_factor_row,
                self._method_params_box,
            ],
            layout=_section_layout,
        )

        self.channel_section = VBox(
            [
                widgets.HTML(
                    f"<b style='color:{ESA_BLUE_BRIGHT}; font-size:14px'>Channel Combination</b>",
                    layout=_LAYOUT,
                ),
                self._matrix_box,
                self._channel_controls,
            ],
            layout=_section_layout,
        )

        # ── Top-level layout ────────────────────────────────────────
        self.widget = VBox(
            [widgets.HTML(FORM_CONTROL_CSS), self.norm_section, self.channel_section],
            layout=widgets.Layout(background_color=BG_COLOR, gap="8px"),
        )
        self.widget.add_class("am-form-themed")

    # ── Method change ───────────────────────────────────────────────

    def _on_method_change(self, _change: dict) -> None:
        self._update_method_params()

    def _update_method_params(self) -> None:
        """Show/hide parameter boxes based on the selected method."""
        method = self._method_dropdown.value
        children: list[widgets.Widget] = []
        if method == NormalisationMethod.ASINH:
            children.append(self._asinh_box)
        elif method == NormalisationMethod.LOG:
            children.append(self._log_box)
        self._method_params_box.children = children

    # ── Enable / disable ─────────────────────────────────────────────

    def set_disabled(self, disabled: bool) -> None:
        """Enable or disable all interactive controls.

        Args:
            disabled: ``True`` to grey out all controls, ``False`` to enable.
        """
        self._method_dropdown.disabled = disabled
        self._resolution_input.disabled = disabled
        self._flux_checkbox.disabled = disabled
        self._log_calc_min.disabled = disabled
        self._add_ch_btn.disabled = disabled
        self._remove_ch_btn.disabled = disabled
        for inp in self._asinh_scale_inputs:
            inp.disabled = disabled
        for inp in self._asinh_clip_inputs:
            inp.disabled = disabled
        for row in self._matrix_cells:
            for cell in row:
                cell.disabled = disabled

    # ── Change notification ────────────────────────────────────────

    def register_on_change(self, callback: Callable[[], None]) -> None:
        """Register a callback invoked when any setting changes.

        Args:
            callback: Zero-argument callable fired on any widget value change.
        """
        self._on_change_callback = callback
        self._attach_observers_to_all()

    def _attach_observers_to_all(self) -> None:
        """Attach the on-change callback to every interactive widget."""
        cb = self._on_change_callback
        if cb is None:
            return
        handler = lambda _: cb()  # noqa: E731
        for w in self._all_interactive_widgets():
            w.observe(handler, names="value")

    def _attach_observer_to(self, widget: widgets.Widget) -> None:
        """Attach the on-change callback to a single widget."""
        cb = self._on_change_callback
        if cb is None:
            return
        widget.observe(lambda _: cb(), names="value")

    def _all_interactive_widgets(self) -> list[widgets.Widget]:
        """Return all interactive widgets that affect normalisation config."""
        result: list[widgets.Widget] = [
            self._method_dropdown,
            self._resolution_input,
            self._padding_factor,
            self._flux_checkbox,
            self._log_calc_min,
        ]
        result.extend(self._asinh_scale_inputs)
        result.extend(self._asinh_clip_inputs)
        for row in self._matrix_cells:
            result.extend(row)
        return result

    # ── Channel-combination matrix ──────────────────────────────────

    def _build_matrix(self) -> None:
        """Rebuild the channel-combination matrix grid."""
        self._matrix_cells = []
        rows: list[widgets.Widget] = []

        # Header row — columns are input channels, rows are output channels
        header_items: list[widgets.Widget] = [
            widgets.HTML(
                f"<span style='color:{TEXT_COLOR}; width:80px; display:inline-block;"
                f" font-size:11px'>Input →</span>",
                layout=widgets.Layout(width="80px", background_color=BG_COLOR),
            )
        ]
        for ext_name in self._extensions:
            header_items.append(
                widgets.HTML(
                    f"<span style='color:{ESA_BLUE_BRIGHT}; font-size:11px'>{ext_name}</span>",
                    layout=widgets.Layout(
                        width="70px",
                        background_color=BG_COLOR,
                        text_align="center",
                    ),
                )
            )
        rows.append(HBox(header_items, layout=_LAYOUT))

        # Data rows (one per output channel)
        for ch_idx in range(self._n_channels):
            row_cells: list[widgets.BoundedFloatText] = []
            row_items: list[widgets.Widget] = [
                widgets.HTML(
                    f"<span style='color:{TEXT_COLOR}; font-size:11px'>Out {ch_idx + 1}</span>",
                    layout=widgets.Layout(width="80px", background_color=BG_COLOR),
                )
            ]
            for _ext_idx in range(len(self._extensions)):
                # Default: identity mapping when n_in >= n_out,
                # broadcast (all 1.0) when n_in < n_out (e.g. 1 VIS → 3 RGB)
                if len(self._extensions) < self._n_channels:
                    default = 1.0
                else:
                    default = 1.0 if _ext_idx == ch_idx else 0.0
                cell = widgets.BoundedFloatText(
                    value=default,
                    min=CHANNEL_COMBINATION_MIN,
                    max=CHANNEL_COMBINATION_MAX,
                    step=0.1,
                    layout=widgets.Layout(width="70px", background_color=BG_COLOR),
                )
                self._attach_observer_to(cell)
                row_cells.append(cell)
                row_items.append(cell)
            self._matrix_cells.append(row_cells)
            rows.append(HBox(row_items, layout=_LAYOUT))

        self._matrix_box.children = rows

    def _add_channel(self) -> None:
        """Add one output channel row to the matrix."""
        self._n_channels += 1
        # Sync ASINH inputs
        self._asinh_scale_inputs.append(
            widgets.BoundedFloatText(
                value=0.7,
                min=0.01,
                max=100.0,
                step=0.1,
                description=f"Scale ch{self._n_channels - 1}",
                layout=widgets.Layout(background_color=BG_COLOR, width="180px"),
                style=_STYLE,
            )
        )
        self._asinh_clip_inputs.append(
            widgets.BoundedFloatText(
                value=99.8,
                min=0.1,
                max=100.0,
                step=0.1,
                description=f"Clip ch{self._n_channels - 1}",
                layout=widgets.Layout(background_color=BG_COLOR, width="180px"),
                style=_STYLE,
            )
        )
        self._attach_observer_to(self._asinh_scale_inputs[-1])
        self._attach_observer_to(self._asinh_clip_inputs[-1])
        self._asinh_box.children = [
            self._asinh_box.children[0],
            HBox(self._asinh_scale_inputs, layout=_LAYOUT),
            HBox(self._asinh_clip_inputs, layout=_LAYOUT),
        ]
        self._build_matrix()
        if self._on_change_callback:
            self._on_change_callback()

    def _remove_channel(self) -> None:
        """Remove the last output channel row from the matrix."""
        if self._n_channels <= 1:
            return
        self._n_channels -= 1
        self._asinh_scale_inputs.pop()
        self._asinh_clip_inputs.pop()
        self._asinh_box.children = [
            self._asinh_box.children[0],
            HBox(self._asinh_scale_inputs, layout=_LAYOUT),
            HBox(self._asinh_clip_inputs, layout=_LAYOUT),
        ]
        self._build_matrix()
        if self._on_change_callback:
            self._on_change_callback()

    # ── Config extraction / restoration ─────────────────────────────

    def show_cutout_zoom(self, visible: bool = True) -> None:
        """Show or hide the cutout zoom control (only relevant for Cutana)."""
        self._padding_factor_row.layout.display = None if visible else "none"

    def show_resolution(self, visible: bool = True) -> None:
        """Show or hide the output-resolution field.

        The detail view decodes at a fixed high resolution for viewing, so the
        field has no meaning there — hide it rather than let the user pick a
        model-input size while inspecting a single cutout.

        Args:
            visible: ``True`` to show the field, ``False`` to hide it.
        """
        # Mutate only this field's own Layout — never a Layout shared with
        # other controls (see the field's construction).  Log the visibility
        # of the neighbouring method dropdown so a recurrence of the
        # shared-layout bug (dropdown vanishing when the field hides) is
        # visible in the session log rather than only on screen.
        self._resolution_input.layout.display = None if visible else "none"
        logger.debug(
            "NormalisationConfigWidget.show_resolution({}) — resolution.display={!r}, "
            "method_dropdown.display={!r}",
            visible,
            self._resolution_input.layout.display,
            self._method_dropdown.layout.display,
        )

    def apply_cutana_bands(self, filter_names: list[str]) -> None:
        """Auto-configure channel settings from detected Cutana filter names.

        Sets extensions, shows the cutout zoom slider, and adjusts the
        output channel count to match the number of bands.  Use for the
        first-time auto-configuration when the user picks a new Cutana
        source — it is destructive and will overwrite any customised
        ``n_output_channels``.  Call :meth:`sync_cutana_bands` instead
        when the widget already reflects a user's choice.

        Args:
            filter_names: Filter names detected from the catalogue
                (e.g. ``["VIS"]`` or ``["VIS", "NIR-H"]``).
        """
        logger.info("Cutana bands detected: {} ({})", filter_names, len(filter_names))
        self.apply_input_channels(filter_names)
        self.show_cutout_zoom(True)

    def apply_input_channels(self, names: list[str]) -> None:
        """Set the input channels and reset to one output channel per input.

        Destructive, like :meth:`apply_cutana_bands`: the matrix becomes the
        identity for *names*.  Used when a new source has different channels
        from the ones the widget shows.

        Args:
            names: Input-channel names, one per source channel.
        """
        self.set_extensions(names)
        while self._n_channels > len(names):
            self._remove_channel()
        while self._n_channels < len(names):
            self._add_channel()

    def sync_cutana_bands(self, filter_names: list[str]) -> None:
        """Sync extension column headers to the Cutana filter names.

        Unlike :meth:`apply_cutana_bands` this preserves the current
        ``n_output_channels`` — use when re-entering a screen after the
        user has already customised the channel count.

        Args:
            filter_names: Filter names detected from the catalogue.
        """
        self.set_extensions(filter_names)
        self.show_cutout_zoom(True)

    @property
    def extensions(self) -> list[str]:
        """Input-channel names currently shown as matrix column headers."""
        return list(self._extensions)

    def set_extensions(self, extensions: list[str]) -> None:
        """Update the FITS extension names and rebuild the matrix.

        Args:
            extensions: List of extension names for column headers.
        """
        self._extensions = list(extensions)
        self._build_matrix()

    def get_normalisation_config(self) -> dict:
        """Extract current widget state as a config dict.

        Returns:
            Dict with keys matching ``cfg.normalisation.*`` fields.
        """
        size = self._resolution_input.value
        cfg: dict = {
            "normalisation_method": self._method_dropdown.value,
            "image_size": [size, size],
            "n_output_channels": self._n_channels,
            "apply_flux_conversion": self._flux_checkbox.value,
            "cutout_padding_factor": self._padding_factor.value,
        }

        method = self._method_dropdown.value
        if method == NormalisationMethod.ASINH:
            cfg["norm_asinh_scale"] = [w.value for w in self._asinh_scale_inputs]
            cfg["norm_asinh_clip"] = [w.value for w in self._asinh_clip_inputs]
        elif method == NormalisationMethod.LOG:
            cfg["norm_log_calculate_minimum_value"] = self._log_calc_min.value

        # Channel-combination matrix — only emit a real matrix when the user
        # has set a non-trivial mapping.  Identity and broadcast matrices are
        # handled automatically by fitsbolt/cutana, so emitting them would
        # override auto-detection with a potentially wrong shape (e.g. 3×3
        # identity when the actual input has only 1 channel).
        if self._matrix_cells and len(self._extensions) > 0:
            matrix = np.array(
                [[cell.value for cell in row] for row in self._matrix_cells],
                dtype=np.float64,
            )
            n_out, n_in = matrix.shape
            is_identity = n_out == n_in and np.allclose(matrix, np.eye(n_out))
            is_broadcast = n_in == 1 and np.allclose(matrix, np.ones((n_out, 1)))
            cfg["channel_combination"] = None if (is_identity or is_broadcast) else matrix
        else:
            cfg["channel_combination"] = None

        return cfg

    def update_from_config(self, norm_cfg: dict) -> None:
        """Restore widget state from a normalisation config dict.

        Args:
            norm_cfg: Dict with ``cfg.normalisation.*`` fields.
        """
        if "normalisation_method" in norm_cfg:
            self._method_dropdown.value = norm_cfg["normalisation_method"]

        if "image_size" in norm_cfg and norm_cfg["image_size"] is not None:
            size = norm_cfg["image_size"]
            self._resolution_input.value = size[0] if isinstance(size, (list, tuple)) else size

        if "apply_flux_conversion" in norm_cfg:
            self._flux_checkbox.value = norm_cfg["apply_flux_conversion"]

        if "cutout_padding_factor" in norm_cfg:
            self._padding_factor.value = norm_cfg["cutout_padding_factor"]

        if "n_output_channels" in norm_cfg:
            target = norm_cfg["n_output_channels"]
            while self._n_channels < target:
                self._add_channel()
            while self._n_channels > target:
                self._remove_channel()

        if "norm_asinh_scale" in norm_cfg:
            for i, val in enumerate(norm_cfg["norm_asinh_scale"]):
                if i < len(self._asinh_scale_inputs):
                    self._asinh_scale_inputs[i].value = val

        if "norm_asinh_clip" in norm_cfg:
            for i, val in enumerate(norm_cfg["norm_asinh_clip"]):
                if i < len(self._asinh_clip_inputs):
                    self._asinh_clip_inputs[i].value = val

        if "norm_log_calculate_minimum_value" in norm_cfg:
            self._log_calc_min.value = norm_cfg["norm_log_calculate_minimum_value"]

        if "channel_combination" in norm_cfg:
            override = norm_cfg["channel_combination"]
            if override is None:
                # No override pinned by cfg → reset to the same default
                # ``_build_matrix`` would create from scratch (identity
                # for n_in >= n_out, broadcast for n_in < n_out).  Without
                # this, re-opening the detail screen leaves the matrix
                # cells displaying the *previous* visit's selection while
                # the actual decode runs against the cfg default.
                n_in = len(self._extensions)
                for i, row in enumerate(self._matrix_cells):
                    for j, cell in enumerate(row):
                        if n_in < self._n_channels:
                            cell.value = 1.0
                        else:
                            cell.value = 1.0 if j == i else 0.0
            else:
                matrix = np.asarray(override)
                for i, row in enumerate(self._matrix_cells):
                    for j, cell in enumerate(row):
                        if i < matrix.shape[0] and j < matrix.shape[1]:
                            cell.value = float(matrix[i, j])

        self._update_method_params()

        errors = self.validate()
        if errors:
            for err in errors:
                logger.warning("Normalisation config: {}", err)

    def validate(self) -> list[str]:
        """Check current widget values against parameter bounds.

        Delegates to :func:`~anomaly_match.utils.normalisation_parameters.validate_normalisation`
        so the same rules apply whether config comes from the UI or code.

        Returns:
            List of validation error messages, empty if all valid.
        """
        return validate_normalisation(self.get_normalisation_config())
