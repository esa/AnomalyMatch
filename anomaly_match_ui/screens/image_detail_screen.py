#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Image detail screen — full-size viewer with transforms for a single prediction result."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import ipywidgets as widgets
import numpy as np
from ipywidgets import HBox, VBox
from loguru import logger

from anomaly_match.datasets.cutana_source import detect_cutana_filter_names
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match_ui.screens.base_screen import BaseScreen
from anomaly_match_ui.styles import BG_COLOR, ESA_BLUE_BRIGHT
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.widgets.header_widget import HeaderWidget
from anomaly_match_ui.widgets.image_toolbar import ImageToolbar
from anomaly_match_ui.widgets.normalisation_config_widget import NormalisationConfigWidget
from anomaly_match_ui.widgets.preview_widget import PreviewWidget

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp


class ImageDetailScreen(BaseScreen):
    """Full-screen image viewer with transform controls.

    Reads image data from ``app._detail_context`` which is set by the
    prediction screen before navigating here.  Embeds a
    :class:`NormalisationConfigWidget` next to :class:`ImageToolbar` so
    the user can retune the *upstream* normalisation (method, channel
    combination, resolution, fitsbolt padding) and watch the cutout
    redraw in place — without leaving the screen and without writing
    back to ``cfg.normalisation`` (#443).  The display-only transforms
    (log / zscale / percentile clip) stay on the toolbar.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        # Single-worker so a fast slider drag collapses to "decode the
        # latest"; combined with the seq-based supersede check the
        # widget stays at most one redecode behind the most recent
        # change.
        # Underscore prefix is deliberate (see gallery_screen_base threading
        # docs): the UI conftest joins every ``am-*``-named thread on teardown,
        # and this single worker idles in ``queue.get()`` between redecodes, so
        # an ``am-`` (dash) prefix would eat the full ~2s join timeout on every
        # UI test that opens the detail screen — and leak the worker into the
        # rest of the session, slowing the whole suite.
        self._redecode_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="am_detail_redecode"
        )
        self._redecode_seq: int = 0
        # Normalisation-independent raw float32 cutout for the source on screen,
        # read once per (source, zoom) and re-normalised in-process on every
        # other widget change (#501).  The key (filename + cutout zoom) is what
        # the held raw was extracted for, so a stale hold is never reused — a
        # zoom change widens the extraction and must re-read.
        self._raw_cutout: np.ndarray | None = None
        self._raw_cutout_key: str | None = None
        self._raw_cutout_zoom: float | None = None

    def build(self) -> widgets.Widget:
        """Build and return the image detail layout.

        Returns:
            The root VBox widget for the detail screen.
        """
        self._out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid white",
                height="100px",
                background_color=BG_COLOR,
                overflow="auto",
            ),
        )
        self._header = HeaderWidget(
            self.app, self._out, show_back_button=True, back_to="prediction"
        )

        self._preview = PreviewWidget()
        self._toolbar = ImageToolbar(self._preview)

        cfg = BackendInterface.get_config()
        self._norm_widget = NormalisationConfigWidget(
            n_channels=cfg.normalisation.n_output_channels,
            image_size=list(cfg.normalisation.image_size),
        )
        # First render mirrors the cfg currently in use so the initial
        # display matches what the gallery showed; subsequent edits are
        # detail-screen-local and never written back.
        self._norm_widget.update_from_config(dict(cfg.normalisation))
        # The detail view decodes Cutana cutouts at their native pixel size, so
        # the output-resolution field has no meaning here — hide it.
        self._norm_widget.show_resolution(False)
        self._norm_widget.register_on_change(self._schedule_redecode)

        self._info_html = widgets.HTML(value="")
        self._redecode_status = widgets.HTML(value="")

        # Group the display transforms (brightness/contrast sliders + RGB
        # toggles) with the normalisation controls in the right-hand column
        # instead of as a detached strip under the image — they belong with
        # the other per-view adjustments, next to the cutout they affect.
        display_section = VBox(
            [
                widgets.HTML(
                    f"<b style='color:{ESA_BLUE_BRIGHT}; font-size:14px'>Display Adjustments</b>",
                ),
                self._toolbar.widget,
                self._toolbar.channel_controls,
            ],
            layout=widgets.Layout(background_color=BG_COLOR, gap="6px"),
        )
        controls_column = VBox(
            [self._norm_widget.widget, display_section],
            layout=widgets.Layout(background_color=BG_COLOR, gap="8px"),
        )

        return VBox(
            [
                self._header.widget,
                self._info_html,
                HBox(
                    [
                        VBox(
                            [self._preview.filename_text, self._preview.image_widget],
                            layout=widgets.Layout(background_color=BG_COLOR),
                        ),
                        controls_column,
                    ],
                    layout=widgets.Layout(background_color=BG_COLOR, align_items="flex-start"),
                ),
                self._redecode_status,
                self._out,
            ],
            layout=widgets.Layout(background_color=BG_COLOR),
        )

    def on_enter(self) -> None:
        """Load image from detail context and display it."""
        ctx = self.app._detail_context
        if ctx is None:
            logger.warning("No detail context available")
            return

        filename = ctx["filename"]
        score = ctx["score"]
        image = ctx["image"]
        source_type = ctx["source_type"]
        search_dir = ctx["search_dir"]

        # Re-target the back button at whichever screen navigated here
        # — the same detail screen serves training, prediction, and the
        # training-setup gallery (#427), so a static ``back_to`` would
        # send users to the wrong one.
        self._header.set_back_to(ctx["back_screen"])

        # Sync the channel-combination matrix's columns to the source's actual
        # FITS bands BEFORE restoring cfg values (same order the training screen
        # uses).  Without this the matrix keeps its 3 default RGB columns, so a
        # 4-band Euclid cutout gets an (n_out x 3) matrix that can't combine a
        # 4-band array — the "cannot reshape" re-decode failure.
        if source_type == DataSourceType.CUTANA and search_dir:
            filter_names = detect_cutana_filter_names(search_dir, BackendInterface.get_config())
            logger.debug(
                "Detail on_enter: source={} search_dir={} detected bands={}",
                filename,
                search_dir,
                filter_names,
            )
            if filter_names:
                self._norm_widget.sync_cutana_bands(filter_names)
            else:
                logger.warning(
                    "No Cutana bands detected for {} under {}; channel-combination "
                    "matrix keeps its default columns and a multi-band re-decode may fail.",
                    filename,
                    search_dir,
                )

        # Re-sync the embedded NormalisationConfigWidget with the
        # current cfg so the controls match the active settings each
        # time the screen is opened.  Without this re-sync, navigating
        # between screens with different cfg would leave stale widget
        # values from the previous detail visit.
        cfg = BackendInterface.get_config()
        self._norm_widget.update_from_config(dict(cfg.normalisation))
        # Same reset for the toolbar's display transforms (brightness,
        # contrast, invert, unsharp, RGB visibility) — otherwise a
        # second visit would still show the previous visit's slider
        # positions and toggle states, even though the underlying image
        # has been re-decoded fresh.
        self._toolbar.restore()
        # Reset the supersede counter so a stale future from the
        # previous session can't push its result into this one.
        self._redecode_seq += 1
        self._redecode_status.value = ""

        if image is not None:
            self._preview.display_standalone(image, filename, score)
        else:
            self._info_html.value = (
                '<div style="color:white; font-size:14px; padding:16px;'
                ' text-align:center;">'
                f"<b>{filename}</b><br>"
                f'<span style="color:#aaa;">Score: {score:.4f}</span><br><br>'
                '<span style="color:#888;">Image not available on disk.<br>'
                "Streaming source cutouts are created in memory during prediction.</span>"
                "</div>"
            )

        # Upgrade the gallery thumbnail to the native-resolution cutout: Cutana
        # sources read the native cutout once and then re-normalise in-process,
        # so kick that initial read off now rather than waiting for the user's
        # first widget change.  Image-folder thumbnails are already the file on
        # disk and Zarr re-decode is unsupported, so only Cutana.
        if source_type == DataSourceType.CUTANA:
            self._schedule_redecode()

    def on_leave(self) -> None:
        """Clear the stored image array to free memory."""
        # Bump the seq so any in-flight re-decode won't push its result
        # into a freshly-built detail screen on the next visit.
        self._redecode_seq += 1
        # Drop the held raw cutout so a stack of visits doesn't accumulate them;
        # the next visit reads it fresh.
        self._raw_cutout = None
        self._raw_cutout_key = None
        self._raw_cutout_zoom = None
        self.app._detail_context = None

    # ── Re-decode pipeline ─────────────────────────────────────────

    def _schedule_redecode(self) -> None:
        """Submit a fresh re-decode for the latest widget state.

        Each call increments :attr:`_redecode_seq` and queues a worker
        on the single-thread executor.  Workers ignore their result if
        a newer seq has been issued by the time they finish, so a slider
        drag that fires 20 events ends up displaying only the most
        recent decode.
        """
        ctx = self.app._detail_context
        if ctx is None:
            # Norm-widget callback fired after on_leave already cleared
            # the context — likely a queued event from the previous
            # visit.  Nothing to redecode against; surface so we'd notice
            # if it started firing on legitimate state.
            logger.warning("Re-decode scheduled with no detail context — ignoring stale event")
            return
        source_type = ctx["source_type"]
        if source_type == DataSourceType.ZARR:
            # Zarr stores hold already-post-fitsbolt bytes and the
            # synthetic filename mapping doesn't expose the upstream
            # FITS, so live re-decode is deferred — tracked in #466.
            # Tell the user explicitly so the silent no-op doesn't look
            # like a broken widget.
            self._redecode_status.value = (
                '<span style="color:#fa0; font-size:12px;">'
                "Live re-decode is not yet supported for Zarr sources — "
                "use the toolbar transforms below."
                "</span>"
            )
            return

        self._redecode_seq += 1
        seq = self._redecode_seq
        overrides = self._norm_widget.get_normalisation_config()
        # Cutana sources decode at their native pixel size — _decode_cutana_from_raw
        # pins image_size to the held native cutout's shape so the in-memory decode
        # doesn't resample.  Image-folder sources keep the cfg size (the resolution
        # slider is hidden either way).
        self._redecode_status.value = (
            '<span style="color:#aaa; font-size:12px;">Re-decoding…</span>'
        )
        self._redecode_executor.submit(self._do_redecode, seq, source_type, overrides)

    def _do_redecode(self, seq: int, source_type: DataSourceType, overrides: dict) -> None:
        """Worker body for the re-decode executor."""
        if seq != self._redecode_seq:
            return  # superseded before we even started
        ctx = self.app._detail_context
        if ctx is None:
            # Same race as in _schedule_redecode but on the worker
            # thread: on_leave cleared the context after we were queued.
            logger.warning("Re-decode worker started with no detail context — dropping result")
            return
        filename = ctx["filename"]
        score = ctx["score"]
        search_dir = ctx["search_dir"]
        try:
            if source_type == DataSourceType.CUTANA:
                image = self._decode_cutana_from_raw(filename, search_dir, overrides)
            else:
                image = BackendInterface.decode_single_with_normalisation(
                    filename, source_type, overrides, search_dir=search_dir
                )
        except Exception as exc:
            if seq == self._redecode_seq:
                self._redecode_status.value = (
                    f'<span style="color:#f88; font-size:12px;">Re-decode failed: {exc}</span>'
                )
            logger.warning("Detail re-decode worker failed: {}", exc)
            return

        if seq != self._redecode_seq:
            return  # newer change came in mid-decode; drop this result
        if image is None:
            # Real failure (source unreachable / no cutout), not a superseded
            # decode.  The loader already logged the cause with source + method,
            # so keep this a breadcrumb and tell the user plainly.
            logger.debug(
                "Detail re-decode produced no image for {} (source_type={}, normalisation={}).",
                filename,
                source_type.value,
                overrides["normalisation_method"],
            )
            self._redecode_status.value = (
                '<span style="color:#f88; font-size:12px;">'
                "No cutout available for this source under the current settings — "
                "it may be unreachable or have no data here. See the log for details."
                "</span>"
            )
            return
        self._preview.display_standalone(image, filename, score)
        self._redecode_status.value = ""

    def _decode_cutana_from_raw(
        self, filename: str, search_dir: str, overrides: dict
    ) -> np.ndarray | None:
        """Decode a Cutana source from the held raw cutout, reading it once per zoom.

        The native cutout is read from the catalogue once per (source, zoom):
        re-normalising it in-process serves every method / channel / clip change
        without touching the catalogue, while a "Cutout Zoom-out" change widens
        the extracted window and so re-reads.  Runs on the single-thread
        re-decode executor, so the read and the hold update are serialised with
        no extra locking.

        Args:
            filename: Cutana source id on screen.
            search_dir: Catalogue directory from the detail context.
            overrides: Normalisation overrides from the widget.

        Returns:
            HWC uint8 array, or ``None`` if the raw can't be read / decoded.
        """
        # "Cutout Zoom-out" changes the extracted sky window, not normalisation,
        # so a zoom change must re-extract the native cutout (re-normalising the
        # held one can't widen it).  Key the hold on (source, zoom) and re-read
        # when either changes.
        zoom = overrides["cutout_padding_factor"]
        if (
            self._raw_cutout is None
            or self._raw_cutout_key != filename
            or self._raw_cutout_zoom != zoom
        ):
            raw = BackendInterface.load_single_native_cutout(
                filename, search_dir=search_dir, cutout_padding_factor=zoom
            )
            if raw is None:
                return None
            self._raw_cutout = raw
            self._raw_cutout_key = filename
            self._raw_cutout_zoom = zoom
        # Pin the decode to the held cutout's native size so it normalises +
        # channel-combines in place without resampling — the user sees the
        # source's real pixels.
        overrides = {
            **overrides,
            "image_size": [self._raw_cutout.shape[0], self._raw_cutout.shape[1]],
        }
        return BackendInterface.decode_raw_with_normalisation(self._raw_cutout, overrides)
