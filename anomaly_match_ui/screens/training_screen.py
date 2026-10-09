#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Training screen 2.0 — train → score → browse → retrain loop.

Extends :class:`GalleryScreenBase` with a two-state machine:

- **BUSY** — a background task (training or scoring) is running. The UI
  disables action buttons and shows a progress bar.  The ``_busy_phase``
  attribute tracks which sub-phase is active (``"training"`` or
  ``"scoring"``) for badge colour and stop-button behaviour.
- **BROWSING** — DB-backed gallery with :class:`LabelableThumbnailCell`
  for labelling. User labels and hits "Retrain" to loop back.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from enum import Enum
from typing import TYPE_CHECKING

import ipywidgets as widgets
import numpy as np
from ipywidgets import HBox, VBox
from loguru import logger

from anomaly_match.datasets.cutana_source import detect_cutana_filter_names
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL
from anomaly_match.datasets.training_data_source import DataSourceType, _auto_detect_source_type
from anomaly_match.utils.validate_config import configs_differ
from anomaly_match_ui.screens.gallery_screen_base import (
    GalleryScreenBase,
    _format_duration,
)
from anomaly_match_ui.styles import (
    BG_COLOR,
    ESA_BLUE_BRIGHT,
    ESA_BLUE_DEEP,
    ESA_GREEN,
    ESA_ORANGE,
    ESA_RED,
)
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.plot_utils import render_size_stratification_png
from anomaly_match_ui.utils.progress_tap import LogProgressTap, clean_hint_text
from anomaly_match_ui.utils.scrolling_log_sink import ScreenLogSink
from anomaly_match_ui.widgets.gallery_widget import GalleryWidget
from anomaly_match_ui.widgets.header_widget import HeaderWidget
from anomaly_match_ui.widgets.labelable_thumbnail_cell import LabelableThumbnailCell, LabelState
from anomaly_match_ui.widgets.normalisation_config_widget import NormalisationConfigWidget
from anomaly_match_ui.widgets.progress_panel import ProgressPanel

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp  # lazy: avoid circular import


def shorten_filename(filename: str, max_length: int = 25) -> str:
    """Shorten a filename to fit within the specified maximum length.

    Preserves the extension and shows beginning/end of the basename.

    Args:
        filename: The filename to shorten.
        max_length: Maximum total length of the result.

    Returns:
        Shortened filename if needed, original otherwise.
    """
    if len(filename) <= max_length:
        return filename

    if "." in filename:
        last_dot_idx = filename.rfind(".")
        basename = filename[:last_dot_idx]
        extension = filename[last_dot_idx:]
    else:
        basename = filename
        extension = ""

    ellipsis = "..."
    available = max_length - len(extension) - len(ellipsis)

    if available <= 6:
        return filename[: max_length - 3] + "..."

    start_len = (available * 2) // 3
    end_len = available - start_len
    return basename[:start_len] + ellipsis + basename[-end_len:] + extension


_LABEL_STATE_TO_CSV = {
    LabelState.ANOMALY: LABEL_ANOMALY,
    LabelState.NORMAL: LABEL_NORMAL,
}


class TrainingState(Enum):
    """Two-state machine for the training screen."""

    BUSY = "busy"
    BROWSING = "browsing"


class TrainingScreen(GalleryScreenBase):
    """Training screen with train → score → browse → retrain loop.

    Extends :class:`GalleryScreenBase` to add training subprocess management,
    scoring via prediction subprocess, and a labelable gallery for the browse
    phase.

    Args:
        app: The parent application instance used for navigation.
    """

    # Score distributions from FixMatch are sharply peaked near 0/1, so a linear
    # count axis flattens the sparse mid/high-score tail (where anomalies sit).
    _score_histogram_log_y = True

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        self._state = TrainingState.BROWSING
        self._busy_phase: str = ""  # "training" or "scoring" when BUSY
        self._labels: dict[str, LabelState] = {}
        self._log_sink = ScreenLogSink("TrainingScreen")
        self._process: subprocess.Popen | None = None
        self._progress_file: str = ""
        self._training_poll_thread: threading.Thread | None = None
        self._training_poll_stop = threading.Event()
        self._stderr_thread: threading.Thread | None = None
        self._prediction_thread: threading.Thread | None = None
        self._temp_dir: str | None = None
        self._norm_snapshot: dict = {}
        # Installed on BUSY, removed on BROWSING.  Forks the subprocess
        # log stream into structured progress events that drive the
        # progress bar and status label.
        self._progress_tap: LogProgressTap | None = None
        # True while the training subprocess is still loading datasets —
        # before the first training-iteration line arrives there is no
        # JSON-driven bar value, so the loading-phase tqdm meters (decode,
        # streaming) drive the bar instead of leaving it dead at 0%.
        self._loading_active: bool = False

    # ── Build ────────────────────────────────────────────────────────

    def build(self) -> widgets.Widget:
        """Build and return the training screen layout.

        Returns:
            The root VBox widget for the training screen.
        """
        # Terminal output
        self.out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid white",
                height="200px",
                background_color=BG_COLOR,
                overflow="auto",
            ),
            style={"color": "white"},
        )

        self._header = HeaderWidget(
            self.app, self.out, show_back_button=True, back_to="training_setup"
        )

        # ── Info row ──────────────────────────────────────────────────
        self._state_html = widgets.HTML(
            value=self._render_state_badge(),
            layout=widgets.Layout(flex="0 0 auto", padding="0 8px"),
        )
        self._config_html = widgets.HTML(
            value="",
            layout=widgets.Layout(flex="1 1 auto", padding="0 8px"),
        )
        self._loading_html = widgets.HTML(
            value="",
            layout=widgets.Layout(width="20px"),
        )
        self._countdown_html = widgets.HTML(
            value="",
            layout=widgets.Layout(width="auto", padding="0 4px"),
        )
        # Small, secondary caption next to the refresh spinner for non-tqdm phase
        # hints (Cutana orchestrator init, gallery cutout fetches, etc.).  Kept
        # OUT of the batch-progress ``status_label`` so a preview/catalogue load
        # during scoring can't clobber "Processing batches: X/Y".
        self._preview_hint_html = widgets.HTML(
            value="",
            layout=widgets.Layout(width="auto", padding="0 4px"),
        )
        info_row = HBox(
            [
                self._state_html,
                self._config_html,
                self._loading_html,
                self._preview_hint_html,
                self._countdown_html,
            ],
            layout=widgets.Layout(
                align_items="center",
                padding="2px 12px",
                background_color=ESA_BLUE_DEEP,
                gap="4px",
            ),
        )

        # ── Progress row ──────────────────────────────────────────────
        self.status_html = widgets.HTML(
            value=self._status_text("Ready"),
            layout=widgets.Layout(
                width="auto",
                min_width="140px",
                flex="0 0 auto",
                padding="0 8px 0 0",
            ),
        )

        self.progress = ProgressPanel()
        self.progress.progress_bar.layout.flex = "1 1 auto"
        self.progress.progress_bar.layout.height = "28px"

        self._stop_btn = widgets.Button(
            description="Stop",
            button_style="danger",
            disabled=True,
            layout=widgets.Layout(width="70px", height="28px"),
            style={"button_color": ESA_RED},
        )
        self._stop_btn.on_click(lambda _: self._on_stop())

        progress_row = HBox(
            [
                self.status_html,
                self.progress.spinner,
                self.progress.progress_bar,
                self.progress.status_label,
                self._stop_btn,
            ],
            layout=widgets.Layout(
                gap="8px",
                align_items="center",
                padding="4px 12px",
            ),
        )

        # ── Gallery (left panel) — uses LabelableThumbnailCell ────────
        def _cell_factory(**kwargs: object) -> LabelableThumbnailCell:
            return LabelableThumbnailCell(on_label_change=self._on_label_change, **kwargs)

        self.gallery = GalleryWidget(
            on_magnify=self._on_magnify,
            on_star=self._on_star,
            cell_factory=_cell_factory,
        )
        self.gallery.set_on_sort_change(self._on_sort_change)
        self.gallery.set_on_page_change(self._on_page_change)
        self.gallery.set_on_scale_change(self._on_scale_change)

        left_panel = VBox(
            [self.gallery.widget],
            layout=widgets.Layout(
                flex="1 1 auto",
                min_width="0",
                background_color=BG_COLOR,
            ),
        )

        # ── Right panel — stats, histogram, labels, controls ─────────
        self.stats_html = widgets.HTML(
            value=self._render_stats(0, 0.0, 0.0),
            layout=widgets.Layout(padding="4px 8px"),
        )

        self._histogram_image = widgets.Image(
            format="png",
            layout=widgets.Layout(
                width="300px",
                height="200px",
                object_fit="contain",
                background_color=BG_COLOR,
                display="none",
            ),
        )

        self._label_summary_html = widgets.HTML(
            value=self._render_label_summary(),
            layout=widgets.Layout(padding="4px 8px"),
        )

        # ── Normalisation config ──────────────────────────────────
        norm_cfg = self.cfg.normalisation
        self._norm_widget = NormalisationConfigWidget(
            n_channels=norm_cfg.n_output_channels,
            image_size=list(norm_cfg.image_size),
        )
        self._norm_widget.register_on_change(self._on_norm_change)

        # Training controls
        self._iter_slider = widgets.IntSlider(
            value=self.cfg.num_train_iter if hasattr(self.cfg, "num_train_iter") else 200,
            min=50,
            max=600,
            step=10,
            description="Iterations:",
            style={"description_width": "initial", "handle_color": "white"},
            layout=widgets.Layout(width="280px"),
        )

        # Mirror the setup screen's size-stratification toggle so it's clear it
        # applies to retrains too, and remains adjustable between rounds.  Gated
        # to Cutana sources in ``on_enter`` (only they carry a per-source size).
        # ``on_enter`` owns the value (re-syncs it from cfg each entry); this
        # initial value only matters for standalone construction.
        self._stratify_checkbox = widgets.Checkbox(
            value=bool(self.cfg.cutana_stratify_source_size),
            description="Stratify source size per tile (Cutana)",
            indent=False,
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="360px"),
        )
        # Write through to cfg on change so cfg is always the source of truth:
        # this lets ``on_enter`` re-sync from cfg without discarding a toggle the
        # user made on the training screen (it's not only pushed at retrain).
        self._stratify_checkbox.observe(self._on_stratify_toggle, names="value")

        self._retrain_warning_html = widgets.HTML(
            value="",
            layout=widgets.Layout(padding="0 8px"),
        )

        self._retrain_btn = widgets.Button(
            description="Retrain",
            button_style="primary",
            layout=widgets.Layout(width="120px", height="36px"),
            style={"font_size": "13px"},
        )
        self._retrain_btn.on_click(lambda _: self._on_retrain())

        self._save_labels_btn = widgets.Button(
            description="Save Labels",
            layout=widgets.Layout(width="120px", height="28px"),
        )
        self._save_labels_btn.on_click(lambda _: self._on_save_labels())

        controls = VBox(
            [
                self._iter_slider,
                self._stratify_checkbox,
                self._retrain_warning_html,
                HBox(
                    [self._retrain_btn],
                    layout=widgets.Layout(gap="8px", padding="4px 0"),
                ),
                HBox(
                    [self._save_labels_btn],
                    layout=widgets.Layout(gap="8px", padding="4px 0"),
                ),
            ],
            layout=widgets.Layout(padding="8px"),
        )

        right_panel = VBox(
            [
                widgets.HTML(
                    value=(
                        f'<div style="color:{ESA_BLUE_BRIGHT}; font-size:14px;'
                        f' font-weight:bold; padding:4px 8px;">Statistics</div>'
                    ),
                ),
                self.stats_html,
                self._histogram_image,
                self._label_summary_html,
                self._norm_widget.widget,
                controls,
            ],
            layout=widgets.Layout(
                width="360px",
                flex="0 0 auto",
                background_color=BG_COLOR,
                border_left=f"1px solid {ESA_BLUE_DEEP}",
                padding="0 8px 0 8px",
                overflow_x="hidden",
            ),
        )

        main_content = HBox(
            [left_panel, right_panel],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="8px 0",
                gap="0px",
            ),
        )

        # Action buttons for busy-state toggling
        self._action_buttons = [self._retrain_btn, self._save_labels_btn]

        root = VBox(
            [
                self._header.widget,
                info_row,
                progress_row,
                main_content,
                self.out,
            ],
            layout=widgets.Layout(background_color=BG_COLOR),
        )
        return root

    # ── Lifecycle ────────────────────────────────────────────────────

    def on_enter(self) -> None:
        """Set up terminal output and auto-start training on first entry."""
        BackendInterface.set_terminal_output(self.out)
        self._log_sink.attach(self.out)
        self._update_config_summary()

        # Sync extension column headers for Cutana sources BEFORE restoring
        # widget values, so the channel-combination matrix has the right
        # column count when update_from_config fills in cfg values.  Use
        # sync_cutana_bands (not apply_cutana_bands) so the user's
        # n_output_channels choice from the setup screen is preserved.
        is_cutana = (
            bool(self.cfg.data_dir)
            and _auto_detect_source_type(self.cfg.data_dir) == DataSourceType.CUTANA
        )
        if is_cutana:
            filter_names = detect_cutana_filter_names(self.cfg.data_dir, self.cfg)
            if filter_names:
                self._norm_widget.sync_cutana_bands(filter_names)
            else:
                self._norm_widget.show_cutout_zoom(True)

        # Size stratification only applies to Cutana sources; disable the toggle
        # otherwise so a retrain can't carry it into an image/Zarr run.  Re-sync the
        # value from cfg on every entry: screens are cached, so a training screen
        # built before the setup checkbox was ticked would otherwise keep a stale
        # unchecked box and, on ``_start_training``, clobber cfg back to False —
        # silently dropping the user's setup choice.
        self._stratify_checkbox.disabled = not is_cutana
        self._stratify_checkbox.value = is_cutana and bool(self.cfg.cutana_stratify_source_size)

        # Sync normalisation widget from cfg (setup screen may have changed it)
        self._norm_widget.update_from_config(dict(self.cfg.normalisation))
        self._norm_snapshot = self._norm_widget.get_normalisation_config()
        self._retrain_warning_html.value = ""

        # Reopen DB if returning from detail screen
        if self._db_path and BackendInterface._score_db is None:
            BackendInterface.open_prediction_monitor(self._db_path, self.cfg.data_dir or "")

        # If we have existing results, refresh gallery
        if self._db_path:
            count = BackendInterface.get_prediction_count()
            if count > 0:
                self._last_db_count = count
                self._refresh_gallery()
                if count >= 10:
                    self._update_histogram()

        # on_leave stops both poll threads so a hidden screen doesn't keep
        # refreshing.  Returning from the image-detail screen therefore lands
        # with polling dead: if scoring or training is still in flight the
        # gallery/progress would freeze until the next button click.  Restart
        # whichever work is still running so the screen stays live — the same
        # guard PredictionScreen.on_enter uses for its own poll loop.
        if (
            self._prediction_thread is not None
            and self._prediction_thread.is_alive()
            and not self._polling
        ):
            self._start_polling()
        # Restart the training poll whenever a subprocess handle is still set
        # and its poll thread is gone.  ``_on_training_complete`` clears
        # ``self._process`` once it has dispatched, so a non-None handle means
        # completion hasn't been handled yet — including the case where the
        # subprocess *exited while the detail screen was open*.  Gating on
        # ``poll() is None`` would skip that exited-while-away case, leaving
        # the screen stuck in the training phase because the loop that calls
        # ``_on_training_complete`` never runs again.  The restarted loop
        # detects the already-exited process and dispatches completion itself.
        if self._process is not None and self._training_poll_thread is None:
            self._start_training_poll()

        # Auto-start training on first entry (no results yet, browsing state)
        if self._state == TrainingState.BROWSING and not self._db_path:
            self._start_training()

    def on_leave(self) -> None:
        """Stop polling when navigating away. Keep DB open for detail screen."""
        self._stop_polling()
        self._stop_training_poll()
        self._uninstall_progress_tap()
        self._log_sink.detach()

    def _detail_back_screen(self) -> str:
        """Return the screen name to navigate back to."""
        return "training"

    # ── State machine ────────────────────────────────────────────────

    def _set_state(self, state: TrainingState, phase: str = "") -> None:
        self._state = state
        self._busy_phase = phase
        self._state_html.value = self._render_state_badge()
        is_busy = state == TrainingState.BUSY
        self._set_busy(is_busy)
        self._stop_btn.disabled = not is_busy
        self._retrain_btn.disabled = is_busy
        self._iter_slider.disabled = is_busy
        self._norm_widget.set_disabled(is_busy)

        if is_busy:
            self._install_progress_tap()
        else:
            self._uninstall_progress_tap()
            # Every way a run ends — done, failed, stopped or refused before it
            # started — leaves BUSY here, so the waiting spinner ends with it.
            self.progress.stop()

    def _install_progress_tap(self) -> None:
        """Start capturing tqdm lines from subprocess logs into the progress bar.

        Idempotent — safe to call on every state transition.  The tap is
        a loguru sink that forks log messages to :meth:`_on_tqdm_progress`;
        it does not block or alter the log stream the Output widget shows.
        """
        if self._progress_tap is not None:
            return
        self._progress_tap = LogProgressTap(
            on_tqdm=self._on_tqdm_progress, on_cutana_hint=self._on_phase_hint
        )
        self._progress_tap.install()

    def _uninstall_progress_tap(self) -> None:
        """Remove the progress tap.  Safe to call when none is installed."""
        if self._progress_tap is None:
            return
        self._progress_tap.uninstall()
        self._progress_tap = None

    def _on_tqdm_progress(self, event: dict) -> None:
        """Update the status label from a parsed tqdm event.

        Runs on whatever thread loguru dispatched from (typically the
        subprocess-stdout reader thread); ipywidgets attribute writes
        are safe from non-UI threads.  The status label always tracks
        the tqdm description + counts so the user sees phase transitions
        ("Decoding labeled cache" → "Streaming cutouts" → "Processing
        batches") immediately.

        Touches the progress bar ONLY during the dataset-loading phase
        (``_loading_active``).  Before the first training iteration there
        is no JSON-driven bar value, so the loading meters (decode,
        streaming) fill the bar per phase — the status label names the
        phase, so the reset between meters reads as a new phase rather
        than a regression.  Once training iterations begin the JSON-lines
        file owns the bar (``_loading_active`` is cleared), and during
        scoring the DB-polling path in :class:`GalleryScreenBase` owns
        it; in both cases a per-batch tqdm percentage would misrepresent
        the work left, so the bar is left untouched there.
        """
        desc = event["desc"]
        current = event["current"]
        total = event["total"]
        timing = event["timing"]
        self.progress.status_label.value = f"{desc}: {current}/{total} [{timing}]"
        # A real meter is ticking, so the secondary phase hint (if any) is stale —
        # clear it so it doesn't linger next to live batch progress.
        self._preview_hint_html.value = ""

        # This runs on the subprocess-stderr reader thread while
        # _handle_progress_line clears _loading_active on the poll thread, so a
        # late-buffered loading line can land just after the first training
        # line and momentarily overwrite the iteration value with a stale
        # loading fraction.  The next poll (~1 s) restores it, so the race is
        # benign and cosmetic — not worth a lock for a one-tick flicker.
        if self._loading_active and total > 0:
            self.progress.progress_bar.value = current / total

    def _on_phase_hint(self, text: str) -> None:
        """Show a non-tqdm phase line in the small caption by the refresh spinner.

        Surfaces the otherwise-silent phases — Cutana orchestrator init, model
        load, GPU warmup, the first FITS-tile stream during scoring, and the
        per-page cutout fetches that load Cutana catalogues during gallery
        browsing.  These go in a *secondary* caption, NOT the batch-progress
        ``status_label``: during scoring the gallery fetches cutouts between
        batch ticks, and routing those catalogue-load lines to the main label
        used to clobber "Processing batches: X/Y".  ``_on_tqdm_progress`` clears
        this caption when a real meter ticks.

        Runs on the subprocess-stderr reader thread; ipywidgets attribute
        writes are safe off the UI thread.
        """
        self._preview_hint_html.value = (
            f'<span style="color:#888; font-size:10px;">{clean_hint_text(text)}</span>'
        )

    def _on_stratify_toggle(self, change: dict) -> None:
        """Persist a size-stratification toggle to cfg the moment it changes.

        Keeps ``cfg`` the single source of truth so ``on_enter`` can re-sync the
        box from it without dropping a pending toggle.  Only writes when the box
        is enabled (Cutana source), so a programmatic reset on a non-Cutana
        source can't leak a stale value into cfg.

        Args:
            change: ipywidgets observe payload with the new ``value``.
        """
        if not self._stratify_checkbox.disabled:
            self.cfg.cutana_stratify_source_size = bool(change["new"])

    def _render_state_badge(self) -> str:
        if self._state == TrainingState.BUSY:
            phase_colors = {"training": ESA_BLUE_BRIGHT, "scoring": "#e0a800"}
            color = phase_colors.get(self._busy_phase, "white")
            label = self._busy_phase.upper() or "BUSY"
        else:
            color = ESA_GREEN
            label = "BROWSING"
        return (
            f'<span style="color:{color}; font-size:12px; font-weight:bold;'
            f' padding:2px 8px; border:1px solid {color}; border-radius:4px;">'
            f"{label}</span>"
        )

    # ── TRAINING state ───────────────────────────────────────────────

    def _start_training(self) -> None:
        """Launch training subprocess via BackendInterface."""
        self._set_state(TrainingState.BUSY, phase="training")
        self.progress.set_color(ESA_BLUE_BRIGHT)
        self._set_status("Preparing training...", color=ESA_BLUE_BRIGHT)
        # Hand the bar to the loading-phase tqdm meters until the first
        # training iteration arrives (see _on_tqdm_progress / _handle_progress_line).
        self._loading_active = True

        try:
            # Collect normalisation overrides from widget and apply to cfg
            norm_overrides = self._norm_widget.get_normalisation_config()
            for key, value in norm_overrides.items():
                setattr(self.cfg.normalisation, key, value)
            self._norm_snapshot = dict(norm_overrides)
            self._retrain_warning_html.value = ""

            # Apply the size-stratification toggle to cfg so the retrain
            # subprocess sees the user's current choice.  Forced False when the
            # checkbox is disabled (non-Cutana source) so it can't leak in.
            self.cfg.cutana_stratify_source_size = (
                self._stratify_checkbox.value and not self._stratify_checkbox.disabled
            )

            # Convert gallery LabelState values to CSV label strings
            csv_labels = {
                fn: _LABEL_STATE_TO_CSV[state]
                for fn, state in self._labels.items()
                if state in _LABEL_STATE_TO_CSV
            }

            # BackendInterface handles config serialization, label merging,
            # and subprocess launch.
            self._process, self._temp_dir, self._progress_file = (
                BackendInterface.launch_training_subprocess(
                    num_train_iter=self._iter_slider.value,
                    normalisation_overrides=norm_overrides,
                    gallery_labels=csv_labels,
                )
            )

            self._set_status("Training...", color=ESA_BLUE_BRIGHT)
            self.progress.start("Starting training...")

            # Stream subprocess stderr → Output widget so logs are visible
            self._start_stderr_reader()

            # Start polling the progress file
            self._start_training_poll()

        except Exception as exc:
            logger.error("Failed to start training: {}", exc)
            self._set_status(f"Error: {exc}", color=ESA_RED)
            self._set_state(TrainingState.BROWSING)

    def _start_training_poll(self) -> None:
        self._training_poll_stop.clear()
        self._training_poll_thread = threading.Thread(
            target=self._training_poll_loop, daemon=True, name="am-train-poll"
        )
        self._training_poll_thread.start()

    def _stop_training_poll(self) -> None:
        self._training_poll_stop.set()
        if self._training_poll_thread is not None:
            self._training_poll_thread.join(timeout=3)
            self._training_poll_thread = None

    def _start_stderr_reader(self) -> None:
        """Start a daemon thread that reads subprocess stderr into the Output widget."""
        if self._process is None or self._process.stderr is None:
            return
        self._stderr_thread = threading.Thread(
            target=self._stderr_reader_loop, daemon=True, name="am-train-stderr"
        )
        self._stderr_thread.start()

    def _stderr_reader_loop(self) -> None:
        """Read subprocess stderr line-by-line and relay through loguru.

        Routing every line through ``logger.info("[subprocess] {}", line)``
        — instead of writing directly to the Output widget — lets the
        progress tap (:class:`LogProgressTap`) pick up the subprocess's
        tqdm lines (e.g. ``Streaming cutouts: 3/4``) and Cutana /
        label-cache status hints, so the progress bar and status label
        stay live during the otherwise silent ``Loading datasets...``
        phase before the JSON-lines progress file is first written.
        The Output widget still sees every line because the UI kernel
        has a loguru sink bound to it.
        """
        try:
            for raw_line in iter(self._process.stderr.readline, b""):
                line = raw_line.decode(errors="replace").rstrip()
                if line:
                    logger.info("[subprocess] {}", line)
        except (ValueError, OSError) as exc:
            logger.warning("Stderr reader stopped: {}", exc)

    # Cutana unlabeled streaming can run silently for several minutes while
    # it downloads and parses FITS tiles before the first progress line
    # hits the JSONL file — bump the threshold well past that so the
    # warning fires only on genuine hangs (e.g. subprocess deadlock, SIGSTOP).
    _STALL_WARN_SECONDS = 900

    def _training_poll_loop(self) -> None:
        """Poll the progress file and subprocess status."""
        last_line_count = 0
        no_progress_seconds = 0
        stall_warned = False
        while not self._training_poll_stop.is_set():
            # Read new progress lines
            had_progress = False
            if os.path.isfile(self._progress_file):
                try:
                    with open(self._progress_file, "r") as f:
                        lines = f.readlines()
                    for line in lines[last_line_count:]:
                        line = line.strip()
                        if line:
                            self._handle_progress_line(json.loads(line))
                            had_progress = True
                    last_line_count = len(lines)
                except Exception as exc:
                    logger.warning("Failed to read progress file: {}", exc)

            # Check if process has finished
            if self._process is not None and self._process.poll() is not None:
                rc = self._process.returncode
                if rc == 0:
                    self._on_training_complete()
                else:
                    # Stderr is already streamed to the Output widget by
                    # the _stderr_reader_loop thread — just log the exit code.
                    logger.error("Training subprocess failed (exit code {})", rc)
                    self._set_status(f"Training failed (exit code {rc})", color=ESA_RED)
                    self._set_state(TrainingState.BROWSING)
                return

            # Detect stalled subprocess — log + flag the UI once when the
            # threshold is first crossed, then recover quietly when
            # progress resumes.  Use amber (not red) since this is a
            # soft-warning signal, not a failure.
            if had_progress:
                if stall_warned:
                    logger.info("Training subprocess progress resumed — clearing stall warning")
                    self._set_status("Training...", color=ESA_BLUE_BRIGHT)
                no_progress_seconds = 0
                stall_warned = False
            else:
                no_progress_seconds += 1
            if no_progress_seconds >= self._STALL_WARN_SECONDS and not stall_warned:
                minutes = self._STALL_WARN_SECONDS // 60
                logger.warning(
                    "Training subprocess has reported no progress for {} min — "
                    "still waiting (this is often fine for Cutana streaming).",
                    minutes,
                )
                self._set_status(
                    f"No progress in {minutes} min — still waiting",
                    color=ESA_ORANGE,
                )
                stall_warned = True

            self._training_poll_stop.wait(timeout=1)

    def _handle_progress_line(self, data: dict) -> None:
        """Update UI from a single progress JSON-lines entry."""
        # status is the dispatch key and iteration/total are guaranteed on
        # every training line (see subprocess_scripts/training_process.py), so
        # index them directly per the fail-hard convention — a renamed/missing
        # field should surface as a logged error in the poll loop, not be
        # masked by a default that silently mis-drives the bar.
        status = data["status"]
        if status == "training":
            # Iteration progress drives the bar from here on — release it
            # from the loading-phase tqdm meters.
            self._loading_active = False
            iteration = data["iteration"]
            total = data["total"]
            if total > 0:
                pct = iteration / total
                self.progress.progress_bar.value = pct
                self.progress.status_label.value = (
                    f"Iteration {iteration}/{total} ({pct * 100:.0f}%)"
                )
        elif status == "loading":
            msg = data.get("message", "Loading...")
            self.progress.status_label.value = msg
        elif status == "saving":
            self.progress.status_label.value = data.get("message", "Saving model...")
        elif status == "size_histogram":
            self._render_size_histogram(data)
        elif status == "done":
            model_path = data.get("model_path")
            if model_path:
                self.cfg.model_path = model_path
                logger.info("Model checkpoint: {}", model_path)

    def _render_size_histogram(self, data: dict) -> None:
        """Overlay the sampled unlabeled pool's sizes on the tile population.

        Emitted once by the training subprocess at dataset build (Cutana sources
        only).  The score-histogram slot is empty during the TRAINING phase, so
        we reuse it here; ``_update_histogram`` replaces this with the score
        histogram once scoring produces results.  The population bars vs. the
        sampled step line show whether size stratification flattened the pool.
        Runs on the poll thread — ipywidgets attribute writes are thread-safe.

        Args:
            data: Progress entry with ``edges``, ``population_counts``,
                ``sampled_counts``, ``stratified`` and ``unit``.
        """
        edges = np.asarray(data["edges"])
        population_counts = np.asarray(data["population_counts"])
        sampled_counts = np.asarray(data["sampled_counts"])
        if sampled_counts.sum() == 0:
            return
        try:
            png = render_size_stratification_png(
                edges,
                population_counts,
                sampled_counts,
                population_color=ESA_BLUE_DEEP,
                sampled_color=ESA_ORANGE,
                unit=data["unit"],
                stratified=data["stratified"],
            )
        except Exception as exc:
            logger.warning("Size histogram render failed: {}", exc)
            return
        self._histogram_image.value = png
        self._histogram_image.layout.display = ""

    def _on_training_complete(self) -> None:
        """Transition from training phase → scoring phase."""
        # Extract elapsed time and model path from the last progress entry
        elapsed = 0.0
        if os.path.isfile(self._progress_file):
            try:
                with open(self._progress_file, "r") as f:
                    for line in f:
                        pass
                    data = json.loads(line.strip())
                    elapsed = data.get("elapsed", 0.0)
                    if "model_path" in data:
                        self.cfg.model_path = data["model_path"]
            except Exception as exc:
                logger.warning("Failed to read final progress entry: {}", exc)

        self.progress.update(1.0, f"Training complete in {_format_duration(elapsed)}")
        self.progress.set_color(ESA_GREEN)
        logger.info("Training complete in {}", _format_duration(elapsed))

        self._process = None
        self._start_scoring()

    # ── SCORING state ────────────────────────────────────────────────

    def _start_scoring(self) -> None:
        """Clear stale DB and launch prediction subprocess for scoring."""
        self._set_state(TrainingState.BUSY, phase="scoring")
        # Scoring's bar is driven by DB polling, not the loading meters.
        self._loading_active = False
        self._set_status("Scoring...", color="#e0a800")
        self.progress.set_color("#e0a800")
        self.progress.start("Starting scoring...")

        if not self.cfg.model_path or not os.path.isfile(self.cfg.model_path):
            logger.error("No model checkpoint found — cannot score.")
            self._set_status("No model found", color=ESA_RED)
            self._set_state(TrainingState.BROWSING)
            return

        # Use data_dir as the source folder for scoring
        source_dir = self.cfg.data_dir
        if not source_dir or not os.path.isdir(source_dir):
            logger.error("No source directory configured.")
            self._set_status("No source directory", color=ESA_RED)
            self._set_state(TrainingState.BROWSING)
            return

        # Set prediction_search_dir to point at source folder
        self.cfg.prediction_search_dir = source_dir

        # Detect source type and count for progress tracking
        try:
            self._file_type, self._total_source_count = BackendInterface.detect_and_count_sources(
                source_dir
            )
        except Exception as exc:
            logger.warning("Failed to count sources: {}", exc)
            self._file_type = DataSourceType.IMAGE_FOLDER
            self._total_source_count = 0

        # Delete stale predictions.db — both the live (possibly relocated to
        # local scratch) copy and the durable session snapshot — so a new run
        # starts clean. The two paths coincide when the DB is not relocated.
        stale_dbs = {
            BackendInterface.get_prediction_db_path(),
            BackendInterface.get_session_db_path(),
        }
        if any(os.path.isfile(p) for p in stale_dbs):
            try:
                # Close existing DB monitor
                if BackendInterface._score_db is not None:
                    BackendInterface._score_db.close()
                    BackendInterface._score_db = None
                for base in stale_dbs:
                    for path in (base, base + "-wal", base + "-shm"):
                        if os.path.isfile(path):
                            os.remove(path)
                logger.info("Cleared stale predictions.db")
            except Exception as exc:
                logger.warning("Could not remove old predictions.db: {}", exc)

        # Open fresh DB monitor
        self._db_path = BackendInterface.get_prediction_db_path()
        BackendInterface.open_prediction_monitor(self._db_path, source_dir)
        self._start_time = time.time()
        self._last_db_count = 0
        self._current_page = 0

        # Start DB polling (reuses GalleryScreenBase infrastructure)
        self._start_polling()

        # Launch prediction in background thread
        self._prediction_thread = threading.Thread(
            target=self._run_scoring, daemon=True, name="am-score"
        )
        self._prediction_thread.start()

    def _run_scoring(self) -> None:
        """Run the prediction pipeline (background thread)."""
        self._scoring_error: str | None = None
        try:
            with self.out:
                logger.info("Scoring with model: {}", self.cfg.model_path)
                logger.info("Source dir: {}", self.cfg.prediction_search_dir)
                BackendInterface.evaluate_all_images(top_n=0)
        except KeyboardInterrupt:
            self._scoring_error = "stopped"
            self._set_status("Scoring stopped", color=ESA_RED)
        except Exception as exc:
            self._scoring_error = str(exc)
            self._set_status(f"Scoring error: {exc}", color=ESA_RED)
            logger.error("Scoring failed: {}", exc)
        finally:
            self._stop_polling()
            self._scoring_complete()

    def _scoring_complete(self) -> None:
        """Finalize scoring and transition to BROWSING."""
        count = BackendInterface.get_prediction_count()
        elapsed = time.time() - self._start_time if self._start_time > 0 else 0
        speed = count / elapsed if elapsed > 0 else 0

        self._update_stats_panel(count, speed, elapsed)
        if count >= 10:
            self._update_histogram()
        self._refresh_gallery()

        # Pull skip stats from Session so a partial run (some chunks
        # produced zero rows because the data volume detached mid-run,
        # for instance) surfaces clearly instead of being painted over
        # by a green "Done" bar.  When stats aren't available (older
        # backend, or scoring threw before the loop ran), default to
        # zeros — the aborted/done branch below handles that gracefully.
        try:
            skip_stats = BackendInterface.get_last_run_skip_stats()
        except Exception as exc:
            logger.debug("Could not read skip stats: {}", exc)
            skip_stats = {
                "skipped_chunks": 0,
                "skipped_sources": 0,
                "total_chunks": 0,
                "total_sources": 0,
            }
        skipped_chunks = skip_stats["skipped_chunks"]
        skipped_sources = skip_stats["skipped_sources"]
        total_sources = skip_stats["total_sources"] or self._total_source_count

        # Preserve the red error status set by _run_scoring — otherwise a
        # partial run (e.g. a chunk OOM'd after 733k of 1.1M sources) would
        # overwrite the error with a misleading green "Done".
        aborted = getattr(self, "_scoring_error", None)
        # Real-progress fraction: ``count / max(count, 1)`` was always 1.0,
        # so an aborted run still showed a full bar.  Anchor on the total
        # source count instead (falling back to count if total is unknown,
        # which keeps the bar from overshooting).
        progress_fraction = count / max(total_sources, count, 1)

        if aborted:
            self.progress.update(
                progress_fraction,
                f"{count:,} of {total_sources:,} scored — incomplete ({aborted})",
            )
            self.progress.set_color(ESA_RED)
            logger.warning(
                "Scoring aborted after {:,} images in {} — {}",
                count,
                _format_duration(elapsed),
                aborted,
            )
        elif skipped_chunks > 0:
            # Subprocess succeeded overall but some chunks were skipped
            # (data volume disconnected, etc.).  Make the partial state
            # obvious rather than painting it green.
            self._set_status(
                f"Partial — {count:,} scored, {skipped_sources:,} unscored "
                f"({skipped_chunks} chunk(s) skipped)",
                color=ESA_ORANGE,
            )
            self.progress.update(
                progress_fraction,
                f"{count:,} of {total_sources:,} scored — "
                f"{skipped_chunks} chunk(s) skipped, {skipped_sources:,} sources unscored",
            )
            self.progress.set_color(ESA_ORANGE)
            logger.warning(
                "Scoring partial — {:,} scored, {:,} skipped ({} chunk(s)) in {}",
                count,
                skipped_sources,
                skipped_chunks,
                _format_duration(elapsed),
            )
        else:
            if count > 0:
                self._set_status(f"Done — {count:,} images scored", color=ESA_GREEN)
            else:
                self._set_status("No results found", color=ESA_RED)
            self.progress.update(1.0, f"{count:,} scored — Done")
            self.progress.set_color(ESA_GREEN)
            logger.info("Scoring complete: {:,} images in {}", count, _format_duration(elapsed))

        self._set_state(TrainingState.BROWSING)

    # ── BROWSING state — label handling ──────────────────────────────

    def _on_label_change(self, filename: str, label: LabelState) -> None:
        """Store label change from gallery toggle."""
        if label == LabelState.UNLABELLED:
            self._labels.pop(filename, None)
        else:
            self._labels[filename] = label
        self._label_summary_html.value = self._render_label_summary()

    def _update_cell(self, cell_index: int, filename: str, score: float, img_bytes: bytes) -> None:
        """Inject stored label state when updating gallery cells."""
        stored_label = self._labels.get(filename, LabelState.UNLABELLED)
        self.gallery._cells[cell_index].update(filename, score, img_bytes, label=stored_label)

    def _mark_cell_loading(self, cell_index: int, filename: str, score: float) -> None:
        """Inject stored label state while a gallery cell is loading."""
        stored_label = self._labels.get(filename, LabelState.UNLABELLED)
        self.gallery._cells[cell_index].mark_loading(filename, score, label=stored_label)

    # ── Normalisation change detection ──────────────────────────────

    def _on_norm_change(self) -> None:
        """Apply normalisation changes to cfg and refresh gallery previews."""
        current = self._norm_widget.get_normalisation_config()

        # Show retrain warning if settings differ from last training run
        if self._norm_snapshot and configs_differ(self._norm_snapshot, current):
            self._retrain_warning_html.value = (
                f'<div style="color:{ESA_ORANGE}; font-size:12px; padding:2px 0;">'
                f"&#9888; Normalisation changed — retrain to apply new settings.</div>"
            )
        elif self._norm_snapshot:
            self._retrain_warning_html.value = ""

        # Apply overrides to cfg so gallery images re-load with new settings
        for key, value in current.items():
            setattr(self.cfg.normalisation, key, value)

        # Clear image cache and refresh gallery so thumbnails reflect the change
        if self._state == TrainingState.BROWSING:
            BackendInterface.clear_image_cache()
            self._refresh_gallery()

    # ── Actions ──────────────────────────────────────────────────────

    def _on_retrain(self) -> None:
        """Merge labels via BackendInterface and start a new training cycle."""
        if self._state != TrainingState.BROWSING:
            return

        # Show preparing state up-front.  _cancel_scoring_if_running can
        # block this handler for up to 20 s while it waits on the scoring
        # thread to join after SIGTERMing the subprocess; without
        # immediate feedback the UI looks frozen and users assume the
        # click was lost (#434).
        self._retrain_btn.disabled = True
        self._retrain_btn.description = "Preparing retrain…"
        self._set_status("Preparing retrain — finishing current batch…", color=ESA_BLUE_BRIGHT)

        try:
            # Defensive cancel: if scoring was stopped via the old path
            # (which only turned off DB polling and left the subprocess
            # holding the GPU), the scoring thread may still be alive
            # here.  Launching training now would double-book the GPU
            # and OOM.  request_stop is idempotent and instant when
            # nothing's running.
            self._cancel_scoring_if_running()

            n_labeled = len(self._labels)
            n_anom = sum(1 for v in self._labels.values() if v == LabelState.ANOMALY)
            n_nom = sum(1 for v in self._labels.values() if v == LabelState.NORMAL)

            if n_labeled > 0:
                logger.info(
                    "Retraining with {} new labels ({} anomaly, {} normal)",
                    n_labeled,
                    n_anom,
                    n_nom,
                )

                # Convert LabelState → CSV label strings and merge via BackendInterface
                csv_labels = {
                    fn: _LABEL_STATE_TO_CSV[state]
                    for fn, state in self._labels.items()
                    if state in _LABEL_STATE_TO_CSV
                }
                merged_path = BackendInterface.merge_gallery_labels(csv_labels)
                self.cfg.label_file = merged_path

                # Update labeled data cache with new labels (container sources only)
                self._update_labeled_cache(csv_labels)

                # Clear gallery labels (they're now in the CSV)
                self._labels.clear()
                self._label_summary_html.value = self._render_label_summary()
            else:
                logger.info("Retraining with current settings (no new labels)")

            self._start_training()
        except Exception as exc:
            # The button was disabled up-front and only _set_state re-enables
            # it, which is otherwise reached solely via _start_training.  So
            # anything raised before that point (an unreadable labels CSV makes
            # merge_gallery_labels raise ValueError) would leave the Retrain
            # button dead and the status frozen on "Preparing retrain…", with
            # the traceback swallowed by ipywidgets.  Recover to BROWSING and
            # surface the reason, mirroring _start_training's own handler.
            logger.error("Failed to start retrain: {}", exc)
            self._set_status(f"Retrain failed: {exc}", color=ESA_RED)
            self._set_state(TrainingState.BROWSING)
        finally:
            # Restore the canonical label so the next BROWSING transition
            # shows "Retrain" again.  _set_state owns the disabled flag.
            self._retrain_btn.description = "Retrain"

    def _update_labeled_cache(self, csv_labels: dict[str, str]) -> None:
        """Update the labeled data cache with newly labeled items from the gallery.

        Delegates all I/O to :meth:`BackendInterface.update_labeled_cache`.

        Args:
            csv_labels: Dict mapping ``id -> csv_label_string`` for newly
                labeled items.
        """
        try:
            BackendInterface.update_labeled_cache(csv_labels)
        except Exception as exc:
            logger.warning("Failed to update labeled cache: {}", exc)

    def _on_stop(self) -> None:
        """Stop the current subprocess and transition to BROWSING."""
        if self._state != TrainingState.BUSY:
            return

        # Show stopping state up-front so the user can see the click landed.
        # Subprocess teardown blocks the click handler for up to 5–20 s while
        # we wait for SIGTERM + thread.join.  Without immediate feedback the
        # whole UI looks frozen and users assume the click was lost (#434).
        self._stop_btn.disabled = True
        self._stop_btn.description = "Stopping…"
        if self._busy_phase == "training":
            self._set_status("Stopping training — saving checkpoint…", color=ESA_RED)
        else:
            self._set_status(
                "Cancelling prediction subprocess — waiting for current batch…",
                color=ESA_RED,
            )

        try:
            if self._busy_phase == "training":
                self._stop_training_poll()
                if self._process is not None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                    self._process = None
                self._set_status("Training stopped", color=ESA_RED)

            elif self._busy_phase == "scoring":
                self._stop_polling()
                # SIGTERM the subprocess and wait for the scoring thread to
                # finish its cleanup — otherwise the prediction process keeps
                # running on GPU until the next chunk boundary, and a
                # subsequent Retrain OOMs.
                self._cancel_scoring_if_running()
                count = BackendInterface.get_prediction_count()
                self._set_status(f"Stopped — {count:,} partial results", color=ESA_RED)
                self.progress.set_color(ESA_RED)
        finally:
            self._stop_btn.description = "Stop"

        self._set_state(TrainingState.BROWSING)

    def _cancel_scoring_if_running(self) -> None:
        """Terminate any live prediction subprocess and join the scoring thread.

        Idempotent — a no-op when no prediction is in flight.  Called
        both from :meth:`_on_stop` (scoring branch) and defensively from
        :meth:`_on_retrain`, since old stop flows left the subprocess
        holding the GPU while flipping the UI back to BROWSING.
        """
        # ``_prediction_thread`` is None when scoring has never been started
        # this session, and ``not is_alive()`` when the previous scoring run
        # has already finished — neither needs cancelling.
        thread = self._prediction_thread
        if thread is None or not thread.is_alive():
            return
        try:
            BackendInterface.request_prediction_stop()
        except Exception as exc:
            # Don't let a Stop/Retrain click die on a backend error — log
            # and fall through to join so we at least try to wait it out.
            logger.warning("request_prediction_stop failed: {}", exc)
        thread.join(timeout=20)
        if thread.is_alive():
            logger.warning(
                "Scoring thread still alive after cancel — continuing anyway; GPU may be busy."
            )

    def _on_save_labels(self) -> None:
        """Save merged labels to the session directory via BackendInterface."""
        try:
            csv_labels = {
                fn: _LABEL_STATE_TO_CSV[state]
                for fn, state in self._labels.items()
                if state in _LABEL_STATE_TO_CSV
            }
            out_path = BackendInterface.merge_gallery_labels(csv_labels)
            self.cfg.label_file = out_path
            logger.info("Labels saved to: {}", out_path)
            self._set_status("Labels saved", color=ESA_GREEN)
        except Exception as exc:
            logger.error("Failed to save labels: {}", exc)
            self._set_status(f"Save failed: {exc}", color=ESA_RED)

    # ── Rendering helpers ────────────────────────────────────────────

    def _render_label_summary(self) -> str:
        n_anom = sum(1 for v in self._labels.values() if v == LabelState.ANOMALY)
        n_nom = sum(1 for v in self._labels.values() if v == LabelState.NORMAL)
        total = n_anom + n_nom
        return (
            f'<div style="color:white; font-size:12px; padding:4px 8px;">'
            f"<b>New labels:</b> {total} "
            f'(<span style="color:{ESA_RED};">{n_anom} anomaly</span>, '
            f'<span style="color:{ESA_GREEN};">{n_nom} normal</span>)'
            f"</div>"
        )

    def _update_config_summary(self) -> None:
        source = os.path.basename((self.cfg.data_dir or "").rstrip(os.sep)) or "?"
        model = os.path.basename(self.cfg.model_path) if self.cfg.model_path else "none"
        labels = os.path.basename(self.cfg.label_file) if self.cfg.label_file else "?"
        self._config_html.value = (
            f'<span style="color:#ccc; font-size:11px;">'
            f"<b>Source:</b> {source} &nbsp;|&nbsp; "
            f"<b>Model:</b> {model} &nbsp;|&nbsp; "
            f"<b>Labels:</b> {labels}"
            f"</span>"
        )
