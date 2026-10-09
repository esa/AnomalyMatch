#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Prediction screen — live gallery with polling, histogram, and profiler stats."""

from __future__ import annotations

import glob as glob_module
import json
import os
import threading
import time
from typing import TYPE_CHECKING

import ipywidgets as widgets
from ipywidgets import HBox, VBox
from loguru import logger

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
from anomaly_match_ui.utils.scrolling_log_sink import ScreenLogSink
from anomaly_match_ui.widgets.gallery_widget import GalleryWidget
from anomaly_match_ui.widgets.header_widget import HeaderWidget
from anomaly_match_ui.widgets.progress_panel import ProgressPanel

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp

_DEFAULT_POLL_INTERVAL = 15

_INTERVAL_OPTIONS = [
    ("5s", 5),
    ("15s", 15),
    ("1m", 60),
    ("10m", 600),
]


class PredictionScreen(GalleryScreenBase):
    """Prediction screen with live gallery, histogram, and profiler stats.

    Polls the AnomalyScoreDB while the subprocess runs, updating a paginated
    thumbnail gallery, score histogram, and statistics panel in real time.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        self._running = False
        self._stop_requested = False
        self._prediction_thread: threading.Thread | None = None
        self._current_chunk: int = 0
        self._total_chunks: int = 0
        self._log_sink = ScreenLogSink("PredictionScreen")

    # ── Build ────────────────────────────────────────────────────────

    def build(self) -> widgets.Widget:
        """Build and return the prediction screen layout.

        Returns:
            The root VBox widget for the prediction screen.
        """
        # Terminal output (styled like training screen)
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
            self.app, self.out, show_back_button=True, back_to="prediction_setup"
        )

        # ── Info/control row ─────────────────────────────────────────
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
        self._interval_dropdown = widgets.Dropdown(
            options=_INTERVAL_OPTIONS,
            value=_DEFAULT_POLL_INTERVAL,
            layout=widgets.Layout(width="60px", height="24px"),
            style={"description_width": "0px"},
        )
        info_row = HBox(
            [self._config_html, self._loading_html, self._countdown_html, self._interval_dropdown],
            layout=widgets.Layout(
                align_items="center",
                padding="2px 12px",
                background_color=ESA_BLUE_DEEP,
                gap="4px",
            ),
        )

        # ── Progress row (status + bar + buttons) ────────────────────
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

        self.start_btn = widgets.Button(
            description="Start Prediction",
            button_style="primary",
            layout=widgets.Layout(width="150px", height="28px"),
        )
        self.start_btn.on_click(lambda _: self._on_start())

        self.stop_btn = widgets.Button(
            description="Stop",
            button_style="danger",
            disabled=True,
            layout=widgets.Layout(width="70px", height="28px"),
            style={"button_color": ESA_RED},
        )
        self.stop_btn.on_click(lambda _: self._on_stop())

        progress_row = HBox(
            [
                self.status_html,
                self.progress.spinner,
                self.progress.progress_bar,
                self.progress.status_label,
                self.start_btn,
                self.stop_btn,
            ],
            layout=widgets.Layout(
                gap="8px",
                align_items="center",
                padding="4px 12px",
            ),
        )

        # ── Gallery (left panel) ─────────────────────────────────────
        self.gallery = GalleryWidget(
            on_magnify=self._on_magnify,
            on_star=self._on_star,
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

        # ── Right panel — stats + histogram ──────────────────────────
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

        self.profiler_html = widgets.HTML(
            value="",
            layout=widgets.Layout(padding="4px 8px"),
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
                self.profiler_html,
            ],
            layout=widgets.Layout(
                width="320px",
                flex="0 0 auto",
                background_color=BG_COLOR,
                border_left=f"1px solid {ESA_BLUE_DEEP}",
                padding="0 0 0 8px",
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

        # Collect action buttons for busy-state toggling
        self._action_buttons = [self.start_btn]

        # Assemble layout
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
        """Set up terminal output and refresh config summary."""
        BackendInterface.set_terminal_output(self.out)
        self._log_sink.attach(self.out)
        self._update_config_summary()

        # Reopen DB monitor if returning from detail screen or on resume
        # TODO(#324): add public accessor
        if self._db_path and BackendInterface._score_db is None:
            search_dir = self.cfg.prediction_search_dir or ""
            BackendInterface.open_prediction_monitor(self._db_path, search_dir)

        # Auto-detect DB on first entry (resume scenario)
        if not self._db_path and self.cfg.output_dir:
            candidate = BackendInterface.get_prediction_db_path()
            if os.path.exists(candidate):
                self._db_path = candidate
                search_dir = self.cfg.prediction_search_dir or ""
                BackendInterface.open_prediction_monitor(self._db_path, search_dir)

                # Load total source count for progress display
                if self._total_source_count == 0:
                    try:
                        self._file_type, self._total_source_count = (
                            BackendInterface.detect_and_count_sources(
                                self.cfg.prediction_search_dir
                            )
                        )
                    except Exception as exc:
                        logger.debug("Failed to detect total sources on resume: %s", exc)

        # Restart polling if prediction is still running
        if self._running and not self._polling:
            self._start_polling()

        # Refresh gallery, stats, histogram, and profiler when we have results
        if self._db_path:
            count = BackendInterface.get_prediction_count()
            if count > 0:
                self._last_db_count = count
                self._refresh_gallery()

                elapsed = time.time() - self._start_time if self._start_time > 0 else 0
                speed = count / elapsed if elapsed > 0 else 0
                self._update_stats_panel(count, speed, elapsed)
                if count >= 10:
                    self._update_histogram()

            # Reload profiler stats if prediction finished
            if not self._running:
                self._load_profiler_stats()

    def on_leave(self) -> None:
        """Stop polling when navigating away. Keep DB open for detail screen."""
        self._stop_polling()
        self._log_sink.detach()

    # ── Actions ──────────────────────────────────────────────────────

    def _on_start(self) -> None:
        """Launch the prediction pipeline with live DB polling."""
        self._set_busy(True)
        self._running = True
        self._stop_requested = False
        self.stop_btn.disabled = False

        # Validate paths
        if not self.cfg.model_path or not os.path.exists(self.cfg.model_path):
            self._set_status("Model path not found", color=ESA_RED)
            self._set_busy(False)
            self._running = False
            return

        if not self.cfg.prediction_search_dir or not os.path.isdir(self.cfg.prediction_search_dir):
            self._set_status("Search directory not found", color=ESA_RED)
            self._set_busy(False)
            self._running = False
            return

        self.progress.set_color(ESA_BLUE_BRIGHT)
        self._set_status("Running prediction...", color=ESA_BLUE_BRIGHT)

        # Detect total source count for ETA
        try:
            self._file_type, self._total_source_count = BackendInterface.detect_and_count_sources(
                self.cfg.prediction_search_dir
            )
        except Exception as exc:
            logger.debug("Failed to detect total source count: %s", exc)
            self._total_source_count = 0

        if self._total_source_count > 0:
            self.progress.start(f"0 / {self._total_source_count:,}")
        else:
            self.progress.start("Starting prediction...")

        # Open DB and cache for polling via BackendInterface
        self._db_path = BackendInterface.get_prediction_db_path()
        BackendInterface.open_prediction_monitor(self._db_path, self.cfg.prediction_search_dir)
        self._start_time = time.time()
        self._last_db_count = 0
        self._current_page = 0
        self._current_chunk = 0
        self._total_chunks = 0
        self._start_polling()

        # Run prediction in a background thread so the UI stays responsive
        self._prediction_thread = threading.Thread(
            target=self._run_prediction, daemon=True, name="am-predict"
        )
        self._prediction_thread.start()

    def _run_prediction(self) -> None:
        """Execute the prediction pipeline (runs in a background thread)."""
        try:
            with self.out:
                logger.info(f"Model: {self.cfg.model_path}")
                logger.info(f"Search dir: {self.cfg.prediction_search_dir}")
                logger.info(f"Output dir: {self.cfg.output_dir}")

                BackendInterface.evaluate_all_images(
                    top_n=0,
                    progress_callback=self._update_progress,
                )

        except KeyboardInterrupt:
            self._set_status("Prediction stopped by user", color=ESA_RED)
            with self.out:
                logger.info("Prediction stopped by user")
        except Exception as exc:
            self._set_status(f"Error: {exc}", color=ESA_RED)
            with self.out:
                logger.error(f"Prediction failed: {exc}")
        finally:
            self._stop_polling()
            self._final_display()
            self._running = False
            self._stop_requested = False
            self.stop_btn.disabled = True
            # Restore the canonical "Stop" label so the next run starts clean.
            # _on_stop swaps it to "Stopping…" for immediate feedback (#434).
            self.stop_btn.description = "Stop"
            self.progress.stop()
            self._set_busy(False)

    def _on_stop(self) -> None:
        """Request the prediction to stop — now and not just between chunks.

        Sets ``_stop_requested`` (which the progress-callback reads to keep
        the next subprocess from starting) and also SIGTERMs the currently
        running subprocess via :meth:`BackendInterface.request_prediction_stop`.
        Without the SIGTERM, the current chunk kept running to completion —
        minutes of scoring the user explicitly asked to cancel.
        """
        if not self._running:
            return
        self._stop_requested = True
        self.stop_btn.disabled = True
        # Reflect the click in the button label too — the subprocess can
        # take seconds to release after SIGTERM, and a greyed-out "Stop"
        # alone reads as "did the click register?" (#434).
        self.stop_btn.description = "Stopping…"
        self._set_status(
            "Cancelling prediction subprocess — waiting for current batch…",
            color=ESA_RED,
        )
        logger.info("Stop requested — signalling prediction subprocess")
        try:
            BackendInterface.request_prediction_stop()
        except Exception as exc:
            logger.warning("request_prediction_stop failed: {}", exc)

    # ── Hooks ────────────────────────────────────────────────────────

    def _get_poll_interval(self) -> int:
        """Use the dropdown value for poll interval."""
        return self._interval_dropdown.value

    def _update_poll_progress(self, count: int, speed: float, elapsed: float) -> None:
        """Update progress bar with chunk tracking."""
        if self._total_source_count > 0:
            pct = min(count / self._total_source_count, 1.0)
            self.progress.progress_bar.value = pct
            remaining = self._total_source_count - count
            eta = remaining / speed if speed > 0 else 0
            chunk_str = ""
            if self._total_chunks > 0:
                chunk_str = f" | chunk {self._current_chunk}/{self._total_chunks}"
            self.progress.status_label.value = (
                f"{count:,} / {self._total_source_count:,} "
                f"({pct * 100:.1f}%){chunk_str} | {speed:.1f} img/s | ETA: {_format_duration(eta)}"
            )

    # ── Final display ────────────────────────────────────────────────

    def _final_display(self) -> None:
        """Final UI update after prediction completes."""
        count = BackendInterface.get_prediction_count()
        elapsed = time.time() - self._start_time
        speed = count / elapsed if elapsed > 0 else 0
        self._update_stats_panel(count, speed, elapsed)
        if count >= 10:
            self._update_histogram()
        self._refresh_gallery()

        try:
            skip_stats = BackendInterface.get_last_run_skip_stats()
        except Exception as exc:
            logger.debug("Could not read skip stats: {}", exc)
            skip_stats = {"skipped_chunks": 0, "skipped_sources": 0}
        skipped_chunks = skip_stats["skipped_chunks"]
        skipped_sources = skip_stats["skipped_sources"]

        if skipped_chunks > 0:
            self._set_status(
                f"Partial — {count:,} scored, {skipped_sources:,} unscored "
                f"({skipped_chunks} chunk(s) skipped — see log for cause)",
                color=ESA_ORANGE,
            )
            if self._total_source_count > 0:
                pct = count / self._total_source_count
                self.progress.update(
                    pct,
                    f"{count:,} / {self._total_source_count:,} scored — "
                    f"{skipped_chunks} chunk(s) skipped",
                )
            else:
                self.progress.update(0.0, f"{skipped_chunks} chunk(s) skipped")
            self.progress.set_color(ESA_ORANGE)
        elif count > 0:
            try:
                lo, hi = BackendInterface.get_prediction_score_range()
                self._set_status(
                    f"Done — {count:,} results, scores [{lo:.4f}, {hi:.4f}]",
                    color=ESA_GREEN,
                )
            except ValueError:
                logger.debug("Score range unavailable")
                self._set_status(f"Done — {count:,} results", color=ESA_GREEN)
            if self._total_source_count > 0:
                self.progress.update(1.0, f"{count:,} / {self._total_source_count:,} — Done")
            else:
                self.progress.update(1.0, "Done")
            self.progress.set_color(ESA_GREEN)
        else:
            self._set_status("No results found", color=ESA_RED)
            if self._total_source_count > 0:
                self.progress.update(0.0, f"0 / {self._total_source_count:,} — No results")
            else:
                self.progress.update(0.0, "No results")
            self.progress.set_color(ESA_RED)

        # Load profiler stats
        self._load_profiler_stats()

    # ── Progress callback ────────────────────────────────────────────

    def _update_progress(self, current: int, total: int, **kwargs: object) -> None:
        """Update the progress bar and status label.

        Called between subprocess launches. Raises ``KeyboardInterrupt``
        if stop was requested, preventing the next subprocess from starting.

        Args:
            current: Chunk index (1-based).
            total: Total number of chunks/files.
            **kwargs: Extra keyword args from Session.

        Raises:
            KeyboardInterrupt: If stop was requested by the user.
        """
        if self._stop_requested:
            raise KeyboardInterrupt("Prediction stopped by user")

        # Track chunk progress
        self._current_chunk = current
        self._total_chunks = total

        if kwargs.get("completed"):
            total_time = kwargs.get("total_time_str", "")
            final_speed = kwargs.get("final_speed", 0)
            msg = "Prediction complete"
            if total_time:
                msg += f" in {total_time}"
            if final_speed:
                msg += f" ({final_speed:.1f} img/s)"
            self.progress.update(1.0, msg)
            self.progress.set_color(ESA_GREEN)
            return

        if kwargs.get("results_updated"):
            return

        progress_pct = kwargs.get("progress_percent")
        if progress_pct is not None:
            speed = kwargs.get("images_per_second")
            eta = kwargs.get("eta_str")

            # When we have total_source_count, use count/total format
            if self._total_source_count > 0:
                count = BackendInterface.get_prediction_count()
                pct = count / self._total_source_count if self._total_source_count > 0 else 0
                self.progress.progress_bar.value = pct

                parts = [f"{count:,} / {self._total_source_count:,} ({pct * 100:.1f}%)"]
                if total > 0:
                    parts.append(f"chunk {current}/{total}")
                if speed:
                    parts.append(f"{speed:.1f} img/s")
                if eta:
                    parts.append(f"ETA: {eta}")
                self.progress.status_label.value = " | ".join(parts)
            else:
                self.progress.progress_bar.value = progress_pct / 100
                parts = [f"Progress: {progress_pct:.1f}%"]
                if total > 0:
                    parts.append(f"chunk {current}/{total}")
                if speed:
                    parts.append(f"{speed:.1f} img/s")
                if eta:
                    parts.append(f"ETA: {eta}")
                self.progress.status_label.value = " | ".join(parts)
        elif total > 0 and self._total_source_count == 0:
            pct = current / total * 100
            self.progress.progress_bar.value = current / total
            self.progress.status_label.value = f"Chunk {current}/{total} ({pct:.0f}%)"

    # ── Prediction-specific helpers ──────────────────────────────────

    def _update_config_summary(self) -> None:
        model = os.path.basename(self.cfg.model_path) if self.cfg.model_path else "?"
        search = os.path.basename((self.cfg.prediction_search_dir or "").rstrip(os.sep)) or "?"
        output = os.path.basename((self.cfg.output_dir or "").rstrip(os.sep)) or "?"
        self._config_html.value = (
            f'<span style="color:#ccc; font-size:11px;">'
            f"<b>Model:</b> {model} &nbsp;|&nbsp; "
            f"<b>Search:</b> {search} &nbsp;|&nbsp; "
            f"<b>Output:</b> {output}"
            f"</span>"
        )

    # ── Profiler stats ───────────────────────────────────────────────

    def _load_profiler_stats(self) -> None:
        report_path = os.path.join(self.cfg.output_dir, "performance_report.json")

        # Also search subdirectories (session may use a timestamped subdir)
        if not os.path.exists(report_path) and os.path.isdir(self.cfg.output_dir):
            for name in sorted(os.listdir(self.cfg.output_dir), reverse=True):
                candidate = os.path.join(self.cfg.output_dir, name, "performance_report.json")
                if os.path.exists(candidate):
                    report_path = candidate
                    break

        # If merged report not found, try reading partial reports directly
        if not os.path.exists(report_path):
            report_path = self._find_partial_report()
            if report_path is None:
                return

        try:
            with open(report_path) as f:
                report = json.load(f)

            lines = [
                f'<div style="color:{ESA_BLUE_BRIGHT}; font-size:13px;'
                f' font-weight:bold; padding:4px 0;">Profiler Stats</div>'
            ]

            # Total wall clock time
            wall_clock = report.get("total_wall_clock_s", 0)
            if wall_clock > 0:
                lines.append(
                    f'<span style="color:#aaa;">Wall clock:</span> {_format_duration(wall_clock)}'
                )

            total_images = report.get("total_images", 0)
            if total_images > 0:
                lines.append(f'<span style="color:#aaa;">Total images:</span> {total_images:,}')

            # Show per-stage breakdown if available
            stages = report.get("stages", {})
            for stage_name, stage_data in stages.items():
                total_time = stage_data.get("total_s", 0)
                pct = stage_data.get("percentage", 0)
                if total_time > 0:
                    lines.append(
                        f'<span style="color:#aaa;">{stage_name}:</span> '
                        f"{_format_duration(total_time)} ({pct:.0f}%)"
                    )

            throughput = report.get("throughput_images_per_sec")
            if throughput:
                lines.append(f'<span style="color:#aaa;">Throughput:</span> {throughput:.1f} img/s')

            peak_mem = report.get("peak_memory", {})
            rss = peak_mem.get("rss_mb", 0)
            gpu = peak_mem.get("gpu_allocated_mb", 0)
            if rss > 0:
                lines.append(f'<span style="color:#aaa;">Peak RAM:</span> {rss:.0f} MB')
            if gpu > 0:
                lines.append(f'<span style="color:#aaa;">Peak GPU:</span> {gpu:.0f} MB')

            self.profiler_html.value = (
                '<div style="color:white; font-size:12px; line-height:1.6;">'
                + "<br>".join(lines)
                + "</div>"
            )
        except Exception as exc:
            logger.warning(f"Failed to load profiler stats: {exc}")

    def _find_partial_report(self) -> str | None:
        """Search for partial profiler reports when merged report is missing.

        Returns:
            Path to the most recent partial report, or ``None``.
        """
        pattern = os.path.join(self.cfg.output_dir, "performance_partial_*.json")
        partials = sorted(glob_module.glob(pattern))
        if partials:
            return partials[-1]

        # Also check subdirectories
        if os.path.isdir(self.cfg.output_dir):
            for name in sorted(os.listdir(self.cfg.output_dir), reverse=True):
                subdir = os.path.join(self.cfg.output_dir, name)
                if os.path.isdir(subdir):
                    pattern = os.path.join(subdir, "performance_partial_*.json")
                    partials = sorted(glob_module.glob(pattern))
                    if partials:
                        return partials[-1]
        return None
