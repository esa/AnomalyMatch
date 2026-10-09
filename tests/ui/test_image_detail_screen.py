#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Unit tests for the embedded normalisation widget on the image detail screen.

The full screen is rendered in :mod:`tests.browser.test_image_detail_screen`;
this file pins the re-decode hook contract that's hard to drive from a
browser test (cancel-on-supersede, cfg deep-copy, source-type dispatch).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod

from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match_ui.app import AnomalyMatchApp
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widgets.preview_widget import PreviewWidget

pytestmark = pytest.mark.ui


@pytest.fixture()
def mock_session():
    session = MagicMock()
    session.cfg = MagicMock()
    session.cfg.data_dir = "/fake/data"
    session.cfg.prediction_search_dir = "/fake/data"
    session.cfg.net = "test-cnn"
    session.cfg.N_to_load = 1000
    session.cfg.normalisation.normalisation_method = NormalisationMethod.CONVERSION_ONLY
    session.cfg.normalisation.image_size = [64, 64]
    session.cfg.normalisation.n_output_channels = 3
    session.cfg.normalisation.channel_combination = None
    session.cfg.num_channels = 3
    session.cfg.test_ratio = 0.5
    session.cfg.top_N = 10
    session.cfg.num_eval_iter = 10
    session.cfg.metadata_file = None
    session.cfg.model_path = None
    session.cfg.output_dir = "/fake/output"
    session.cfg.fitsbolt_cfg = None
    return session


@pytest.fixture()
def app(mock_session):
    with patch("IPython.display.display"):
        BackendInterface.set_session(mock_session)
        return AnomalyMatchApp(mock_session)


# ── decode_single_with_normalisation ─────────────────────────────


class TestDecodeSingleWithNormalisation:
    def test_zarr_raises_not_implemented(self, mock_session):
        """Zarr re-decode is deferred (issue #466); reaching the backend is a bug.

        The detail screen short-circuits Zarr in ``_schedule_redecode``
        and surfaces a status message to the user, so this raise only
        fires if a future caller skips that guard — fail loudly rather
        than silently return ``None``.
        """
        BackendInterface.set_session(mock_session)
        with pytest.raises(NotImplementedError, match="not supported"):
            BackendInterface.decode_single_with_normalisation(
                "anything",
                DataSourceType.ZARR,
                {"normalisation_method": NormalisationMethod.LOG},
                search_dir="/fake/data",
            )

    def test_does_not_mutate_session_cfg(self, mock_session):
        """Overrides go into a deep-copy; the live cfg is untouched."""
        BackendInterface.set_session(mock_session)
        original_method = mock_session.cfg.normalisation.normalisation_method
        original_size = list(mock_session.cfg.normalisation.image_size)

        with patch(
            "anomaly_match_ui.utils.backend_interface.load_single_cutout",
            return_value=np.zeros((32, 32, 3), dtype=np.uint8),
        ) as mock_loader:
            BackendInterface.decode_single_with_normalisation(
                "src_001",
                DataSourceType.CUTANA,
                {
                    "normalisation_method": NormalisationMethod.LOG,
                    "image_size": [128, 128],
                },
                search_dir="/fake/data",
            )

        # The loader was called with a *copy* of cfg, so the live cfg
        # must still hold the original values.
        assert mock_session.cfg.normalisation.normalisation_method == original_method
        assert list(mock_session.cfg.normalisation.image_size) == original_size

        # And the copy passed to the loader had the overrides applied.
        loader_cfg = mock_loader.call_args[0][0]
        assert loader_cfg.normalisation.normalisation_method == NormalisationMethod.LOG
        assert list(loader_cfg.normalisation.image_size) == [128, 128]
        # fitsbolt_cfg must be cleared so the override actually takes
        # effect — otherwise the stale build wins.
        assert loader_cfg.fitsbolt_cfg is None

    def test_cutana_dispatches_to_load_single_cutout(self, mock_session):
        BackendInterface.set_session(mock_session)
        fake_img = np.zeros((64, 64, 3), dtype=np.uint8)
        with patch(
            "anomaly_match_ui.utils.backend_interface.load_single_cutout",
            return_value=fake_img,
        ) as mock_loader:
            result = BackendInterface.decode_single_with_normalisation(
                "src_001", DataSourceType.CUTANA, {}, search_dir="/fake/data"
            )
        assert result is fake_img
        assert mock_loader.call_args[0][1] == "src_001"

    def test_cutana_pins_explicit_search_dir(self, mock_session):
        """The caller-supplied search_dir wins over any cfg state.

        Regression for the live bug — TrainingSetupScreen magnify
        previously fell back to a stale ``cfg.data_dir`` (the notebook
        default) because ``cfg.prediction_search_dir`` only gets set on
        Start.  Passing the chooser's current selection through the
        detail context fixes it.
        """
        mock_session.cfg.prediction_search_dir = "/stale/from/notebook"
        mock_session.cfg.data_dir = "/also/stale"
        BackendInterface.set_session(mock_session)
        with patch(
            "anomaly_match_ui.utils.backend_interface.load_single_cutout",
            return_value=np.zeros((32, 32, 3), dtype=np.uint8),
        ) as mock_loader:
            BackendInterface.decode_single_with_normalisation(
                "src_001",
                DataSourceType.CUTANA,
                {},
                search_dir="/chooser/picked/this",
            )
        loader_cfg = mock_loader.call_args[0][0]
        assert loader_cfg.prediction_search_dir == "/chooser/picked/this"

    def test_image_folder_propagates_loader_failure(self, mock_session):
        """A loader exception must propagate so the UI status can surface it."""
        BackendInterface.set_session(mock_session)
        with patch(
            "anomaly_match_ui.utils.backend_interface.load_and_process_single_wrapper",
            side_effect=RuntimeError("nope"),
        ):
            with pytest.raises(RuntimeError, match="nope"):
                BackendInterface.decode_single_with_normalisation(
                    "/abs/path/img.png",
                    DataSourceType.IMAGE_FOLDER,
                    {},
                    search_dir="/fake/data",
                )

    @pytest.mark.parametrize("n_channels", [3, 8])
    def test_image_folder_transposes_chw_to_hwc(self, mock_session, n_channels):
        """Multi-channel loader output (CHW) is rotated to HWC.

        Parametrised over a 3-channel RGB cutout and an 8-channel
        multispectral one: the rotation keys off the configured channel
        count, not a fixed "<= 4" guess, so a >4-channel cutout doesn't
        slip through as a mislabelled HWC array.
        """
        mock_session.cfg.normalisation.n_output_channels = n_channels
        BackendInterface.set_session(mock_session)
        # CHW (n_channels, 64, 64); leading axis matches n_output_channels → rotate.
        chw = np.zeros((n_channels, 64, 64), dtype=np.uint8)
        with patch(
            "anomaly_match_ui.utils.backend_interface.load_and_process_single_wrapper",
            return_value=chw,
        ):
            result = BackendInterface.decode_single_with_normalisation(
                "/abs/path/img.png",
                DataSourceType.IMAGE_FOLDER,
                {},
                search_dir="/fake/data",
            )
        assert result.shape == (64, 64, n_channels)


# ── Raw decode path (decode once, re-normalise in-process) ───────


class TestRawDecodePath:
    def test_decode_raw_with_normalisation_applies_overrides(self, mock_session):
        """Overrides go onto a deep-copy and drive the decode; the live cfg is
        untouched and the held raw array is the one decoded."""
        BackendInterface.set_session(mock_session)
        raw = np.zeros((384, 384, 4), dtype=np.float32)
        decoded = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # Decode is delegated to the shared backend re-decode (#501 path).
        with patch(
            "anomaly_match_ui.utils.backend_interface._decode_raw_cutouts",
            return_value=[("", decoded)],
        ) as mock_decode:
            result = BackendInterface.decode_raw_with_normalisation(
                raw, {"normalisation_method": NormalisationMethod.LOG}
            )

        assert result is decoded
        # Live cfg untouched (override went onto a copy).
        assert (
            mock_session.cfg.normalisation.normalisation_method
            == NormalisationMethod.CONVERSION_ONLY
        )
        passed_cfg, passed_items = mock_decode.call_args[0][0], mock_decode.call_args[0][1]
        assert len(passed_items) == 1 and passed_items[0][1] is raw
        assert passed_cfg.normalisation.normalisation_method == NormalisationMethod.LOG
        assert passed_cfg.fitsbolt_cfg is None

    def test_decode_raw_with_normalisation_none_when_empty(self, mock_session):
        BackendInterface.set_session(mock_session)
        with patch(
            "anomaly_match_ui.utils.backend_interface._decode_raw_cutouts",
            return_value=[],
        ):
            assert (
                BackendInterface.decode_raw_with_normalisation(
                    np.zeros((4, 4, 4), dtype=np.float32), {}
                )
                is None
            )

    def test_load_single_native_cutout_pins_search_dir(self, mock_session):
        """The caller's search_dir wins over the cfg (training-setup flow has
        no prediction_search_dir set)."""
        mock_session.cfg.prediction_search_dir = "/stale"
        BackendInterface.set_session(mock_session)
        with patch(
            "anomaly_match_ui.utils.backend_interface.load_single_native_cutout",
            return_value=np.zeros((14, 14, 4), dtype=np.float32),
        ) as mock_loader:
            BackendInterface.load_single_native_cutout(
                "s1", search_dir="/picked", cutout_padding_factor=1.0
            )
        cfg_arg = mock_loader.call_args[0][0]
        assert cfg_arg.prediction_search_dir == "/picked"
        # The widget's zoom is applied at extraction time (it changes the window).
        assert cfg_arg.normalisation.cutout_padding_factor == 1.0

    def test_detail_decode_reuses_held_raw_across_normalisation_change(self, app):
        """A normalisation-only retune (same zoom) re-normalises the held raw
        without re-reading the catalogue — the core of issue #501.  The decode
        is pinned to the native size."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        native = np.zeros((14, 14, 4), dtype=np.float32)
        decoded = np.zeros((14, 14, 3), dtype=np.uint8)
        with (
            patch.object(
                BackendInterface, "load_single_native_cutout", return_value=native
            ) as mock_native,
            patch.object(
                BackendInterface, "decode_raw_with_normalisation", return_value=decoded
            ) as mock_dec,
        ):
            screen._decode_cutana_from_raw(
                "s1", "/d", {"cutout_padding_factor": 1.0, "normalisation_method": "ASINH"}
            )
            # Same zoom, different normalisation method → reuse the held raw.
            screen._decode_cutana_from_raw(
                "s1", "/d", {"cutout_padding_factor": 1.0, "normalisation_method": "LOG"}
            )

        assert mock_native.call_count == 1  # read once, reused across the retune
        assert mock_dec.call_count == 2  # re-normalised each time
        assert screen._raw_cutout_key == "s1"
        # The decode is pinned to the native cutout's pixel size (no resample).
        assert mock_dec.call_args[0][1]["image_size"] == [14, 14]

    def test_detail_decode_reextracts_on_zoom_change(self, app):
        """Changing the cutout zoom-out widens the extraction window, so it must
        re-read the native cutout (re-normalising the held one can't widen it)."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        with (
            patch.object(
                BackendInterface,
                "load_single_native_cutout",
                side_effect=[
                    np.zeros((14, 14, 4), dtype=np.float32),
                    np.zeros((28, 28, 4), dtype=np.float32),
                ],
            ) as mock_native,
            patch.object(
                BackendInterface,
                "decode_raw_with_normalisation",
                return_value=np.zeros((28, 28, 3), dtype=np.uint8),
            ),
        ):
            screen._decode_cutana_from_raw("s1", "/d", {"cutout_padding_factor": 1.0})
            screen._decode_cutana_from_raw("s1", "/d", {"cutout_padding_factor": 2.0})

        assert mock_native.call_count == 2  # zoom change forces a re-extract
        assert mock_native.call_args.kwargs["cutout_padding_factor"] == 2.0
        assert screen._raw_cutout_zoom == 2.0

    def test_detail_decode_none_when_native_unavailable(self, app):
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        with patch.object(BackendInterface, "load_single_native_cutout", return_value=None):
            assert (
                screen._decode_cutana_from_raw("s1", "/d", {"cutout_padding_factor": 1.0}) is None
            )
        assert screen._raw_cutout is None

    def test_on_enter_schedules_initial_decode_for_cutana(self, app):
        """Opening a Cutana source kicks off the one-time raw read so the view
        upgrades from the gallery thumbnail to the full-detail cutout."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "s1",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.CUTANA,
            "search_dir": "/fake/data",
        }
        with (
            patch(
                "anomaly_match_ui.screens.image_detail_screen.detect_cutana_filter_names",
                return_value=["VIS"],
            ),
            patch.object(screen, "_schedule_redecode") as mock_sched,
        ):
            screen.on_enter()
        mock_sched.assert_called_once()

    def test_on_enter_no_initial_decode_for_image_folder(self, app):
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "/abs/img.png",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.IMAGE_FOLDER,
            "search_dir": "/fake/data",
        }
        with patch.object(screen, "_schedule_redecode") as mock_sched:
            screen.on_enter()
        mock_sched.assert_not_called()


# ── Detail-screen layout ─────────────────────────────────────────


class TestDetailScreenLayout:
    def test_display_transforms_grouped_with_normalisation(self, app):
        """Brightness/contrast sliders + RGB toggles live in the right-hand
        controls column next to the normalisation panel, not as a detached
        strip under the image."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        root = screen.widget

        # The toolbar + RGB controls must not be loose top-level children.
        assert screen._toolbar.widget not in root.children
        assert screen._toolbar.channel_controls not in root.children

        # Find the image|controls HBox and its right-hand column.
        from ipywidgets import HBox

        hbox = next(c for c in root.children if isinstance(c, HBox))
        controls_column = hbox.children[1]
        # Normalisation panel first, then the display-adjustments group.
        assert screen._norm_widget.widget is controls_column.children[0]
        display_section = controls_column.children[1]
        assert screen._toolbar.widget in display_section.children
        assert screen._toolbar.channel_controls in display_section.children


# ── Detail-view resolution ───────────────────────────────────────


class TestDetailViewResolution:
    def test_build_hides_resolution_input(self, app):
        """The output-resolution field makes no sense on the detail view —
        Cutana decodes at the cutout's native pixel size — so build() hides it."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        assert screen._norm_widget._resolution_input.layout.display == "none"

    def test_transform_sliders_are_non_continuous(self, app):
        """A transform+encode on the decoded image is non-trivial, so the
        brightness/contrast sliders fire on release, not on every drag tick —
        otherwise a drag queues dozens of re-encodes on the UI thread."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        assert screen._toolbar.brightness_slider.continuous_update is False
        assert screen._toolbar.contrast_slider.continuous_update is False

    def test_schedule_redecode_keeps_cfg_resolution_for_image_folder(self, app):
        """Image-folder sources are not bumped to the high detail resolution —
        that would only upsample a stored file (and could distort aspect)."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "/abs/img.png",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.IMAGE_FOLDER,
            "search_dir": "/fake/data",
        }
        screen._norm_widget._resolution_input.value = 96

        with patch.object(screen._redecode_executor, "submit") as mock_submit:
            screen._schedule_redecode()

        overrides = mock_submit.call_args[0][3]
        assert overrides["image_size"] == [96, 96]


# ── ImageDetailScreen re-decode supersede ────────────────────────


class TestRedecodeSupersede:
    def test_zarr_skips_redecode_entirely(self, app):
        """Zarr context never schedules a re-decode (out of scope)."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "x__image_000001",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.ZARR,
            "search_dir": "/fake/data",
        }
        screen._redecode_seq = 0

        with patch.object(BackendInterface, "decode_single_with_normalisation") as mock_decode:
            screen._schedule_redecode()
        mock_decode.assert_not_called()
        # Seq must not advance — there's nothing to supersede.
        assert screen._redecode_seq == 0

    def test_supersede_drops_stale_result(self, app):
        """A worker that finishes after a newer schedule must not push its result."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "src_001",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.CUTANA,
            "search_dir": "/fake/data",
        }

        with patch.object(BackendInterface, "decode_single_with_normalisation") as mock_decode:
            mock_decode.return_value = np.zeros((32, 32, 3), dtype=np.uint8)
            stale_seq = screen._redecode_seq + 1
            # Bump seq past stale_seq to simulate a newer change while
            # the worker was running, then run the worker body.
            screen._redecode_seq = stale_seq + 5
            with patch.object(screen._preview, "display_standalone") as mock_display:
                screen._do_redecode(stale_seq, DataSourceType.CUTANA, {})
        mock_display.assert_not_called()

    def test_none_result_surfaces_failure_status(self, app):
        """A real failure (decode returns None, not superseded) must show a
        meaningful status and not paint an image — the failure mode this PR
        replaced the opaque "Re-decode returned no image" for."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "src_001",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.CUTANA,
            "search_dir": "/fake/data",
        }
        screen._redecode_seq = 1

        # Cutana routes through the native-cutout path; a None read is the failure.
        with patch.object(BackendInterface, "load_single_native_cutout", return_value=None):
            with patch.object(screen._preview, "display_standalone") as mock_display:
                screen._do_redecode(
                    1,
                    DataSourceType.CUTANA,
                    {
                        "normalisation_method": NormalisationMethod.LOG,
                        "cutout_padding_factor": 1.0,
                    },
                )

        mock_display.assert_not_called()
        assert "No cutout available" in screen._redecode_status.value


# ── Cutana band sync + channel toggles ───────────────────────────


class TestDetailBandSync:
    def test_on_enter_syncs_cutana_bands(self, app):
        """The channel-combination matrix must take one column per FITS band of
        the source, not the 3 default RGB columns — otherwise a 4-band cutout
        gets an (n_out x 3) matrix that can't be combined (the reshape error)."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "src_001",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.CUTANA,
            "search_dir": "/fake/data",
        }
        bands = ["VIS", "NIR-H", "NIR-J", "NIR-Y"]
        with patch(
            "anomaly_match_ui.screens.image_detail_screen.detect_cutana_filter_names",
            return_value=bands,
        ):
            screen.on_enter()

        assert screen._norm_widget._extensions == bands

    def test_on_enter_skips_band_sync_for_image_folder(self, app):
        """Image-folder sources have no Cutana catalogue to detect bands from."""
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        app._detail_context = {
            "filename": "/abs/img.png",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.IMAGE_FOLDER,
            "search_dir": "/fake/data",
        }
        with patch(
            "anomaly_match_ui.screens.image_detail_screen.detect_cutana_filter_names"
        ) as mock_detect:
            screen.on_enter()
        mock_detect.assert_not_called()


class TestPreviewChannelToggle:
    def test_rgb_toggle_zeroes_displayed_channel_multiband(self, mock_session):
        """Turning off R must blank the red channel of the displayed RGB image
        even for a >3-band source.  This is exactly the case the old
        channel_visibility override broke (it ignored the R/G/B toggles when
        num_channels > 3), so the test forces num_channels = 4."""
        mock_session.cfg.num_channels = 4
        BackendInterface.set_session(mock_session)
        preview = PreviewWidget()
        assert preview.num_channels == 4  # exercises the >3-band branch

        img = np.zeros((8, 8, 4), dtype=np.uint8)
        img[:, :, 0] = 200
        img[:, :, 1] = 100
        img[:, :, 2] = 50
        img[:, :, 3] = 30
        preview.display_standalone(img, "x", 0.5)

        preview.set_rgb_channels(r=False)
        arr = np.array(preview.modified_image)
        assert arr[:, :, 0].max() == 0  # red blanked
        assert arr[:, :, 1].max() > 0  # green untouched


# ── on_enter / display-state hygiene ─────────────────────────────


class TestDetailScreenStateHygiene:
    def test_on_enter_resets_toolbar_display_transforms(self, app):
        """Brightness / contrast / RGB toggles must reset on every visit.

        Without this, a second visit to the detail screen still showed
        the previous visit's slider positions even though the image was
        re-decoded fresh.
        """
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        # Prime the preview with an image so slider observers don't
        # explode trying to enhance ``None`` — in production the screen
        # is always populated before the user can touch a slider.
        screen._preview.display_standalone(np.zeros((32, 32, 3), dtype=np.uint8), "old.png", 0.0)

        # Simulate a prior visit where the user dialled in some
        # transforms.
        screen._toolbar.brightness_slider.value = 1.5
        screen._toolbar.contrast_slider.value = 0.7
        screen._toolbar.red_checkbox.value = False

        app._detail_context = {
            "filename": "img.png",
            "score": 0.5,
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
            "back_screen": "prediction",
            "source_type": DataSourceType.IMAGE_FOLDER,
            "search_dir": "/fake/data",
        }
        screen.on_enter()

        assert screen._toolbar.brightness_slider.value == 1.0
        assert screen._toolbar.contrast_slider.value == 1.0
        assert screen._toolbar.red_checkbox.value is True

    def test_redecode_preserves_invert_toggle(self, app):
        """A re-decode triggered by a norm change must keep the invert state.

        Pre-fix the new image was displayed un-inverted while
        ``preview.invert`` stayed True, leading to the \"have to click
        Invert twice\" behaviour the user reported.
        """
        app.navigate_to("image_detail")
        screen = app._screens["image_detail"]
        # User had inverted the image; the toggle is on.
        screen._preview.invert = True
        screen._preview.original_image = np.zeros((32, 32, 3), dtype=np.uint8)

        with patch.object(screen._preview, "_apply_transforms_and_display") as mock_apply:
            screen._preview.display_standalone(
                np.full((32, 32, 3), 200, dtype=np.uint8), "img.png", 0.5
            )

        # display_standalone must route through _apply_transforms_and_display
        # so the live invert/brightness/contrast state stays in sync
        # with the new image bytes.
        mock_apply.assert_called_once()
        assert screen._preview.invert is True
