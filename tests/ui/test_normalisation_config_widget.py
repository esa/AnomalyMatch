#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Tests for NormalisationConfigWidget."""

from unittest.mock import patch

import ipywidgets as widgets
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match_ui.widgets.normalisation_config_widget import NormalisationConfigWidget

pytestmark = pytest.mark.ui


@patch("IPython.display.display")
class TestNormalisationConfigWidget:
    """Test the normalisation configuration widget."""

    def test_default_config(self, _mock_display):
        w = NormalisationConfigWidget()
        cfg = w.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.CONVERSION_ONLY
        assert cfg["image_size"] == [150, 150]
        assert cfg["n_output_channels"] == 3
        assert "interpolation_order" not in cfg

    def test_change_method_to_asinh(self, _mock_display):
        w = NormalisationConfigWidget()
        w._method_dropdown.value = NormalisationMethod.ASINH
        cfg = w.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.ASINH
        assert len(cfg["norm_asinh_scale"]) == 3
        assert len(cfg["norm_asinh_clip"]) == 3
        assert cfg["norm_asinh_scale"][0] == 0.7
        assert cfg["norm_asinh_clip"][0] == 99.8

    def test_change_method_to_log(self, _mock_display):
        w = NormalisationConfigWidget()
        w._method_dropdown.value = NormalisationMethod.LOG
        cfg = w.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.LOG
        assert cfg["norm_log_calculate_minimum_value"] is False

    def test_resolution_change(self, _mock_display):
        """137 is deliberately not a multiple of 16: it is the kind of value the
        old step-16 slider could not reach (#472).

        The type assertion is the actual regression oracle — the quantisation
        was browser-side, so the round-trip below passes against the old widget
        too, as would ``step == 1`` (ipywidgets' own default).
        """
        w = NormalisationConfigWidget()
        assert isinstance(w._resolution_input, widgets.BoundedIntText)
        w._resolution_input.value = 137
        cfg = w.get_normalisation_config()
        assert cfg["image_size"] == [137, 137]

    def test_show_resolution_toggles_field_visibility(self, _mock_display):
        w = NormalisationConfigWidget()
        w.show_resolution(False)
        assert w._resolution_input.layout.display == "none"
        w.show_resolution(True)
        assert w._resolution_input.layout.display is None

    def test_show_resolution_does_not_hide_other_controls(self, _mock_display):
        """Regression: the resolution field must own its Layout, not share the
        module-level _WIDE_LAYOUT with the method dropdown / padding control —
        otherwise hiding it on the detail screen also hid the Method dropdown."""
        w = NormalisationConfigWidget()
        w.show_resolution(False)
        assert w._resolution_input.layout.display == "none"
        # The neighbours that previously shared the same Layout stay visible.
        assert w._method_dropdown.layout.display is None
        assert w._padding_factor.layout.display is None

    def test_channel_combination_identity_is_none(self, _mock_display):
        """Identity matrix (n_in == n_out) is emitted as None (auto-detect)."""
        w = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        cfg = w.get_normalisation_config()
        assert cfg["channel_combination"] is None

    def test_channel_combination_custom_values(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        w._matrix_cells[0][1].value = 0.5
        cfg = w.get_normalisation_config()
        assert cfg["channel_combination"][0, 1] == 0.5

    def test_add_channel(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        w._add_channel()
        assert w._n_channels == 3
        cfg = w.get_normalisation_config()
        assert cfg["n_output_channels"] == 3
        # 3 out × 2 in: broadcast default (all 1.0), which is non-trivial
        assert cfg["channel_combination"].shape == (3, 2)

    def test_remove_channel(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=3, extensions=["a", "b", "c"])
        w._remove_channel()
        assert w._n_channels == 2
        cfg = w.get_normalisation_config()
        assert cfg["n_output_channels"] == 2

    def test_remove_channel_minimum(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=1, extensions=["VIS"])
        w._remove_channel()
        assert w._n_channels == 1

    def test_set_extensions(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=2, extensions=["a", "b"])
        w.set_extensions(["VIS", "NIR", "SWIR"])
        cfg = w.get_normalisation_config()
        # Matrix columns should match new extension count
        assert cfg["channel_combination"].shape == (2, 3)

    def test_sync_cutana_bands_preserves_n_channels(self, _mock_display):
        """sync_cutana_bands must not change the user's n_output_channels.

        Regression test: apply_cutana_bands forced n_channels to match the
        band count, clobbering the user's single-channel choice when
        re-entering the training screen.
        """
        w = NormalisationConfigWidget(n_channels=1, extensions=["R", "G", "B"])
        w.sync_cutana_bands(["VIS", "NIR-H", "NIR-J", "NIR-Y"])
        assert w._n_channels == 1
        cfg = w.get_normalisation_config()
        assert cfg["n_output_channels"] == 1
        # Matrix columns should match the new band count
        assert cfg["channel_combination"].shape == (1, 4)

    def test_apply_cutana_bands_still_forces_n_channels(self, _mock_display):
        """apply_cutana_bands remains the destructive auto-configure helper."""
        w = NormalisationConfigWidget(n_channels=1, extensions=["R", "G", "B"])
        w.apply_cutana_bands(["VIS", "NIR-H", "NIR-J", "NIR-Y"])
        assert w._n_channels == 4

    def test_update_from_config(self, _mock_display):
        w = NormalisationConfigWidget()
        w.update_from_config(
            {
                "normalisation_method": NormalisationMethod.ASINH,
                "image_size": [224, 224],
                "n_output_channels": 2,
                "norm_asinh_scale": [0.5, 0.3],
                "norm_asinh_clip": [95.0, 90.0],
                "apply_flux_conversion": True,
            }
        )
        cfg = w.get_normalisation_config()
        assert cfg["normalisation_method"] == NormalisationMethod.ASINH
        assert cfg["image_size"] == [224, 224]
        assert cfg["n_output_channels"] == 2
        assert cfg["norm_asinh_scale"] == [0.5, 0.3]
        assert cfg["norm_asinh_clip"] == [95.0, 90.0]
        assert cfg["apply_flux_conversion"] is True

    def test_update_from_config_with_none_resets_matrix_to_identity(self, _mock_display):
        """Re-applying a cfg whose ``channel_combination`` is ``None``
        must wipe any user-dialled matrix back to the default.

        Without this, re-opening the image-detail screen still displays
        the previous visit's channel-combo selection even though the
        cfg-default decode is what's actually being rendered.
        """
        w = NormalisationConfigWidget(n_channels=3, extensions=["R", "G", "B"])
        # Simulate a user-dialled off-diagonal selection.
        w._matrix_cells[0][1].value = 0.7
        w._matrix_cells[2][0].value = 0.4

        w.update_from_config({"channel_combination": None})

        # Back to the identity the widget would build from scratch.
        for i, row in enumerate(w._matrix_cells):
            for j, cell in enumerate(row):
                assert cell.value == (1.0 if i == j else 0.0)

    def test_roundtrip_config(self, _mock_display):
        """Extract config, apply to a fresh widget, extract again — should match."""
        w1 = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        w1._method_dropdown.value = NormalisationMethod.ZSCALE
        w1._resolution_input.value = 256
        w1._matrix_cells[0][1].value = 0.3
        cfg1 = w1.get_normalisation_config()

        w2 = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        w2.update_from_config(cfg1)
        cfg2 = w2.get_normalisation_config()

        assert cfg2["normalisation_method"] == cfg1["normalisation_method"]
        assert cfg2["image_size"] == cfg1["image_size"]
        assert cfg2["n_output_channels"] == cfg1["n_output_channels"]

    def test_channel_labels_use_out_prefix(self, _mock_display):
        """Row labels should use 'Out 1', 'Out 2' format."""
        w = NormalisationConfigWidget(n_channels=2, extensions=["VIS", "NIR"])
        # Matrix box rows: [header, row0, row1]
        rows = w._matrix_box.children
        # Row labels are the first child in each data row
        row0_label = rows[1].children[0].value  # HBox children[0] is the label
        row1_label = rows[2].children[0].value
        assert "Out 1" in row0_label
        assert "Out 2" in row1_label

    def test_no_interpolation_in_config(self, _mock_display):
        """interpolation_order should not be in widget config output."""
        w = NormalisationConfigWidget()
        cfg = w.get_normalisation_config()
        assert "interpolation_order" not in cfg

    def test_flux_conversion(self, _mock_display):
        w = NormalisationConfigWidget()
        # Defaults on — Cutana currently only supports Euclid data where
        # flux conversion is always needed for tile-invariant scores.
        assert w.get_normalisation_config()["apply_flux_conversion"] is True
        w._flux_checkbox.value = False
        assert w.get_normalisation_config()["apply_flux_conversion"] is False

    def test_asinh_params_sync_with_channels(self, _mock_display):
        w = NormalisationConfigWidget(n_channels=2)
        assert len(w._asinh_scale_inputs) == 2
        w._add_channel()
        assert len(w._asinh_scale_inputs) == 3
        w._remove_channel()
        assert len(w._asinh_scale_inputs) == 2

    def test_default_extensions_are_rgb(self, _mock_display):
        """Default extension labels should be R, G, B for image inputs."""
        w = NormalisationConfigWidget()
        assert w._extensions == ["R", "G", "B"]

    def test_validate_returns_empty_for_defaults(self, _mock_display):
        """Default config should pass validation."""
        w = NormalisationConfigWidget()
        assert w.validate() == []

    def test_validate_catches_bad_resolution(self, _mock_display):
        """``validate()`` reports an out-of-range ``image_size``.

        Lowering ``min`` first is not incidental: the control clamps, so this
        branch of ``validate()`` is unreachable through the widget and the test
        has to break that invariant to reach it at all.  It guards the shared
        ``validate_normalisation`` bounds rather than anything a user can
        currently trigger here — the clamp that hides it is #590, and once that
        warns instead of silently rewriting the value, this path becomes live.
        """
        w = NormalisationConfigWidget()
        w._resolution_input.min = 1  # Defeat the clamp; see the docstring.
        w._resolution_input.value = 10
        errors = w.validate()
        assert any("image_size" in e for e in errors)

    def test_validate_catches_bad_asinh_scale(self, _mock_display):
        """ASINH scale outside bounds should be reported."""
        w = NormalisationConfigWidget()
        w._method_dropdown.value = NormalisationMethod.ASINH
        w._asinh_scale_inputs[0].min = 0.0  # Bypass widget clamp
        w._asinh_scale_inputs[0].value = 0.0
        errors = w.validate()
        assert any("norm_asinh_scale" in e for e in errors)

    def test_update_from_config_logs_warnings_for_invalid(self, _mock_display):
        """update_from_config should log warnings for out-of-range values."""
        w = NormalisationConfigWidget()
        # BoundedFloatText clamps values, so validation will pass after clamping.
        # This tests that the validation path is exercised without errors.
        w.update_from_config({"image_size": [150, 150]})
        assert w.validate() == []
