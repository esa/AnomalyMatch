#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Base class for screens with a DB-backed paginated gallery.

Provides shared infrastructure for gallery refresh, DB polling, score
histogram, stats panel, image detail navigation, and prefetching.
Subclasses implement :meth:`build` to compose their own layout and
override hooks for custom progress display and poll intervals.

Threading model
---------------
**Foreground executor** (``_executor``, ``thread_name_prefix="am"``,
``max_workers=4``):
    User-driven, latency-sensitive tasks.  Sized so a click can run
    even when other foreground work is in flight.

    - *page-load* — loads thumbnail images for the current gallery page
    - *detail-load* — loads a full image on magnify click

**Prefetch executor** (``_prefetch_executor``,
``thread_name_prefix="am_prefetch"``, ``max_workers=1``):
    Background work that must never starve the foreground lane.  A
    wedged Cutana parquet read here can't queue ahead of clicks (#444).

    - *prefetch* — warms the image cache for the pages around the
      current one (window prefetch), cancel-on-navigate

    The underscore in the prefix is deliberate: the UI conftest joins
    every ``am-*``-named thread on test teardown to keep ``am-poll``
    from racing the SQLite GC at interpreter shutdown, and an idle
    ``ThreadPoolExecutor`` worker blocked in ``queue.get()`` would
    eat the full join timeout (~2s) per test under that prefix.

**Managed threads** (explicit ``threading.Thread``):
    Long-running threads that need ``Event``-based cancellation.

    - *poll* (``am-poll``) — polls the DB every N seconds until stopped
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import ipywidgets as widgets
import numpy as np
from loguru import logger

from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match_ui.screens.base_screen import BaseScreen
from anomaly_match_ui.styles import ESA_BLUE_BRIGHT, ESA_RED, inline_spinner_html
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream
from anomaly_match_ui.utils.plot_utils import render_histogram_png

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp

_DEFAULT_POLL_INTERVAL = 15

# How many pages ahead (forward-biased, plus one behind) to warm into the
# image cache after each navigation so a subsequent Next/Prev almost never
# has to cold-load.  Five pages of thumbnails is a small memory footprint
# against the LRU cache and comfortably covers a burst of paging clicks.
_PREFETCH_PAGES = 5

_SPINNER_HTML = inline_spinner_html(size_px=12)


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Args:
        seconds: Number of seconds.

    Returns:
        Human-readable string like ``"2m 05s"`` or ``"1h 01m"``.
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins:02d}m"


class GalleryScreenBase(BaseScreen):
    """Base for screens with a DB-backed paginated image gallery.

    Provides gallery refresh, DB polling, histogram, stats panel,
    image detail navigation, prefetching, and common UI helpers.
    Subclasses must implement :meth:`build` and may override hooks
    like :meth:`_get_poll_interval` and :meth:`_update_poll_progress`.

    Subclasses **must** assign the following attributes in their
    :meth:`build` before returning:

    - ``self.gallery`` — a :class:`GalleryWidget`
    - ``self.stats_html`` — :class:`widgets.HTML` for statistics
    - ``self._histogram_image`` — :class:`widgets.Image` for the histogram
    - ``self.status_html`` — :class:`widgets.HTML` for status text
    - ``self.progress`` — a :class:`ProgressPanel`
    - ``self.out`` — :class:`widgets.Output` for terminal logging

    Args:
        app: The parent application instance used for navigation.
    """

    # Whether the score histogram uses a log y-axis.  Off by default (the
    # prediction screen keeps a linear count axis); the training screen flips it
    # on so the sharply-peaked score distribution doesn't bury the high-score
    # anomaly tail in a flat line.
    _score_histogram_log_y: bool = False

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        self.cfg = BackendInterface.get_config()

        # Gallery state
        self._current_page: int = 0
        # Absolute offset of the first visible cell.  Tracked alongside
        # ``_current_page`` so a slider-driven re-page (effective page
        # size shrinks when cells widen) can land the user near the
        # same item rather than at a stale page index (#432).
        self._gallery_first_offset: int = 0
        self._image_load_page: int = -1
        self._detail_loading: bool = False

        # Polling state
        self._polling: bool = False
        self._poll_stop = threading.Event()
        self._poll_thread: threading.Thread | None = None

        # User-driven foreground work: page-load and detail-load.  Sized
        # so a click for the next page or the magnify view doesn't sit
        # behind an in-flight Cutana batch read (#444).  Keeping it
        # separate from the prefetch lane below means a slow parquet
        # walk on the prefetch side can't starve a click that happens
        # while it's running.
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="am")
        # Background prefetch lane — single-threaded so a wedged Cutana
        # read here can never queue ahead of foreground work.  Earlier
        # versions shared a single 3-worker pool and a stuck prefetch
        # could hold 1/3 of the slots indefinitely, leaving foreground
        # clicks blocked behind page-loads queued after it.
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="am_prefetch"
        )
        self._prefetch_future: Future | None = None
        # Bumped on every navigation / sort change / poll so a prefetch task
        # warming a now-stale window bails out between pages the moment the
        # user moves, keeping the single prefetch worker on the live view.
        self._prefetch_generation: int = 0

        # DB / source state
        self._db_path: str = ""
        self._start_time: float = 0.0
        self._last_db_count: int = 0
        self._total_source_count: int = 0
        self._file_type: DataSourceType = DataSourceType.IMAGE_FOLDER

        # UI widgets set by subclass build()
        self._loading_html = widgets.HTML()
        self._countdown_html = widgets.HTML()
        self._action_buttons: list[widgets.Button] = []

    # ── Gallery ──────────────────────────────────────────────────────

    def _refresh_gallery(self, user_initiated: bool = True) -> None:
        """Reload the current gallery page without blocking the UI thread.

        Pages at the gallery's *effective* page size (slider-aware) so
        widening the slider doesn't make ``Next`` skip past cells that
        just got clipped past row 2 (#432).

        On a user navigation the page flips *immediately* — spinner cells
        and the updated pagination counter are painted synchronously — and
        the DB metadata query **and** image decode both run on the
        background executor (:meth:`_load_page`).  Previously the metadata
        ``SELECT ... ORDER BY ... LIMIT/OFFSET`` ran on the UI thread and
        could stall a click for seconds on a large, actively-written DB
        (deep pages, ``random``/``*_dist`` sorts, WAL contention) — the
        same reason ``COUNT(*)`` was already moved off-thread.

        Args:
            user_initiated: ``True`` for a click/sort/slider change — show
                spinners at once for clear feedback.  ``False`` for a
                background poll refresh — update the counter quietly and
                swap images in place, so the visible page doesn't flash a
                full row of spinners every poll interval.
        """
        sort_mode = self.gallery.get_sort_mode()
        # Reuse the count the poll loop already maintains instead of issuing a
        # COUNT(*) on the UI thread (see the WAL-contention note above).  Fall
        # back to a direct count only before the first poll populated it.
        count = self._last_db_count or BackendInterface.get_prediction_count()
        page_size = self.gallery.effective_page_size
        total_pages = max(1, math.ceil(count / page_size))

        if self._current_page >= total_pages:
            self._current_page = total_pages - 1

        page = self._current_page
        offset = page * page_size
        n_cells = max(0, min(page_size, count - offset))

        self._gallery_first_offset = offset
        # Publish the target page before spawning background work so a stale
        # in-flight load for the previous page sees the mismatch and aborts.
        self._image_load_page = page

        if user_initiated:
            self.gallery.begin_page(page, total_pages, count, n_cells)
        else:
            self.gallery.set_pagination(page, total_pages, count)

        self._set_loading(True)
        self._executor.submit(self._load_page, sort_mode, page, page_size, offset, user_initiated)
        # Warm the surrounding pages so the next click usually hits the cache —
        # but ONLY on user navigation.  A background poll refresh fires every
        # interval while scoring is still producing results; re-running the
        # window prefetch each time floods the main process with Cutana cutout
        # reads (each spinning up worker processes + shared memory) that contend
        # with the scoring subprocess's own FITS streaming for NFS bandwidth.
        # Look-ahead only helps when the user is actually paging; a poll refresh
        # just repaints the current page in place.
        if user_initiated:
            self._start_prefetch_window(page, sort_mode, count)

    def _load_page(
        self,
        sort_mode: str,
        page: int,
        page_size: int,
        offset: int,
        show_spinner: bool,
    ) -> None:
        """Background: query page metadata, then stream in thumbnail images.

        Runs entirely off the UI thread so a slow ``ORDER BY``/``OFFSET``
        query can't freeze the click that triggered it.  On a user
        navigation (*show_spinner*) each cell is first shown with its
        filename/score over a spinner, so the metadata appears the instant
        the query returns and the pixels fill in as they load.

        Args:
            sort_mode: Active sort key.
            page: Zero-based page index this load is for.
            page_size: Effective cells per page.
            offset: Row offset for the DB query.
            show_spinner: Whether to paint per-cell spinners with metadata
                before loading images (user navigation) or update images in
                place (background poll refresh).
        """
        try:
            results = BackendInterface.get_prediction_results(
                sort_by=sort_mode, limit=page_size, offset=offset
            )
            if page != self._image_load_page:
                return
            if show_spinner:
                for i, r in enumerate(results):
                    if page != self._image_load_page:
                        return
                    self._mark_cell_loading(i, r["filename"], r["score"])
            self._load_page_images(results, page)
        except Exception as exc:
            logger.opt(exception=True).error("Gallery page-load failed: {}", exc)
        finally:
            self._set_loading(False)

    def _mark_cell_loading(self, cell_index: int, filename: str, score: float) -> None:
        """Show *filename*/score over a loading spinner in a single cell.

        Subclasses override to inject extra per-cell state (e.g. the stored
        label) — the loading-state analogue of :meth:`_update_cell`.
        """
        self.gallery._cells[cell_index].mark_loading(filename, score)

    def _update_cell(self, cell_index: int, filename: str, score: float, img_bytes: bytes) -> None:
        """Update a single gallery cell with image data.

        Subclasses can override to inject extra state (e.g. label).
        """
        self.gallery._cells[cell_index].update(filename, score, img_bytes)

    def _load_page_images(self, results: list[dict], page: int) -> None:
        """Load thumbnail images in the background and update cells.

        Loads cached images immediately, then batch-loads uncached ones
        via Cutana or container reading in a single call.
        """
        # Runs on the page-load executor — exceptions land on the dropped
        # Future and disappear unless we log them here. The bug in the past:
        # a single mis-typed _file_type made every page load a no-op silently.
        try:
            # Cells that received a decoded image.  Anything left out at the
            # end resolves to an explicit "No preview" so a genuinely
            # unavailable image never spins forever (the loading spinner is
            # only for *pending* loads).
            filled: set[int] = set()
            uncached: list[tuple[int, dict]] = []

            for i, r in enumerate(results):
                if page != self._image_load_page:
                    return
                fn = r["filename"]
                image = BackendInterface.load_prediction_image(fn)
                if image is not None:
                    img_bytes = numpy_array_to_byte_stream(image, normalize=True)
                    if page == self._image_load_page:
                        self._update_cell(i, fn, r["score"], img_bytes)
                        filled.add(i)
                else:
                    uncached.append((i, r))

            # Cached cells were painted above for responsiveness; now resolve
            # the misses in one batched read (Cutana/Zarr) and fill them in.
            # A source is exactly one of Cutana / Zarr / image-folder, so at
            # most one batch path runs; image-folder misses are terminal (the
            # on-disk decode already ran in the loop above).
            if uncached:
                batch = self._batch_load_source([r["filename"] for _, r in uncached])
                for i, r in uncached:
                    if page != self._image_load_page:
                        return
                    image = batch.get(r["filename"])
                    if image is not None:
                        img_bytes = numpy_array_to_byte_stream(image, normalize=True)
                        if page == self._image_load_page:
                            self._update_cell(i, r["filename"], r["score"], img_bytes)
                            filled.add(i)

            unresolved = [(i, r) for i, r in enumerate(results) if i not in filled]
            for i, r in unresolved:
                if page != self._image_load_page:
                    return
                self._update_cell(i, r["filename"], r["score"], b"")
            if unresolved:
                logger.debug(
                    "Gallery page-load: {} image(s) unavailable (file_type={!r}); "
                    "shown as 'No preview'.",
                    len(unresolved),
                    self._file_type,
                )
        except Exception as exc:
            logger.opt(exception=True).error("Gallery image-load failed: {}", exc)

    def _start_prefetch_window(self, page: int, sort_mode: str, count: int) -> None:
        """Warm the image cache for the pages around *page* in *sort_mode*.

        Bumps :attr:`_prefetch_generation` so any prefetch still running for
        a previous window bails out between pages (it checks the token), then
        submits a fresh task to the single-worker prefetch lane.  That lane is
        deliberately separate from the foreground executor so a slow Cutana
        parquet walk here can never queue ahead of a click (#444).

        Args:
            page: Page the user is currently on.
            sort_mode: Active sort key to prefetch in.
            count: Total result count (0 means nothing to prefetch).
        """
        if count <= 0:
            return
        self._prefetch_generation += 1
        gen = self._prefetch_generation
        page_size = self.gallery.effective_page_size
        # Drop a still-queued prior prefetch so rapid paging doesn't build a
        # backlog on the single worker; one already running bails on its own
        # via the generation check above.  cancel() is a no-op once started.
        if self._prefetch_future is not None:
            self._prefetch_future.cancel()
        self._prefetch_future = self._prefetch_executor.submit(
            self._do_prefetch_window, page, page_size, sort_mode, count, gen
        )

    def _do_prefetch_window(
        self, page: int, page_size: int, sort_mode: str, count: int, gen: int
    ) -> None:
        """Pre-load images for the pages surrounding *page* into the cache.

        Warms *_PREFETCH_PAGES* pages ahead (``Next`` is the common action)
        plus the one page behind, in the current sort order.  Between pages it
        checks *gen* against the live generation and bails the instant the
        user navigates or re-sorts, so the single prefetch worker never keeps
        grinding on a window the user has already left.
        """
        total_pages = max(1, math.ceil(count / page_size))
        # Start at the *next* page: the current page is already being loaded by
        # the foreground `_load_page`, so re-warming it here would only risk a
        # duplicate batch read racing that load.  One page back covers a Prev
        # after paging forward.
        targets = [page + delta for delta in range(1, _PREFETCH_PAGES + 1)]
        targets.append(page - 1)
        in_range = [t for t in targets if 0 <= t < total_pages]
        # Observability: prefetch is background + invisible, so log which pages
        # it warms.  Enable DEBUG logging to confirm it fires on navigation.
        logger.debug(
            "Prefetch: warming pages {} ({} sort) around page {}", in_range, sort_mode, page
        )
        for target_page in in_range:
            if gen != self._prefetch_generation:
                return
            try:
                results = BackendInterface.get_prediction_results(
                    sort_by=sort_mode, limit=page_size, offset=target_page * page_size
                )
            except Exception as exc:
                logger.debug("Prefetch query failed: {}", exc)
                return
            self._warm_cache([r["filename"] for r in results], gen)

    def _warm_cache(self, filenames: list[str], gen: int) -> None:
        """Load any uncached *filenames* into the prediction image cache.

        For image folders a cache miss decodes and caches in one call, so the
        miss check itself warms the cache; Cutana/Zarr misses return ``None``
        and are batch-loaded then inserted.  Aborts early if *gen* is stale.

        Args:
            filenames: Source filenames to ensure are cached.
            gen: Prefetch generation this batch belongs to.
        """
        uncached = []
        for fn in filenames:
            if gen != self._prefetch_generation:
                return
            if BackendInterface.load_prediction_image(fn) is None:
                uncached.append(fn)
        self._batch_load_source(uncached)

    def _batch_load_source(self, filenames: list[str]) -> dict[str, np.ndarray]:
        """Batch-load and cache *filenames* via the current source's bulk path.

        Cutana and Zarr sources resolve cache misses in a single batched read;
        image folders have no bulk path (their per-file on-disk decode already
        ran on the cache-miss check), so this returns empty for them.  Shared
        by the foreground page load and the background prefetch so the
        source-dispatch ladder lives in exactly one place.

        Args:
            filenames: Uncached source filenames to fetch.

        Returns:
            ``{filename: image}`` for those that resolved — each cached as a
            side effect; unresolved filenames are absent from the mapping.
        """
        if not filenames:
            return {}
        if self._file_type == DataSourceType.CUTANA:
            batch = BackendInterface.load_cutana_batch(filenames)
        elif self._file_type == DataSourceType.ZARR:
            batch = BackendInterface.load_container_images(filenames)
        else:
            return {}
        for fn, image in batch.items():
            BackendInterface.cache_prediction_image(fn, image)
        return batch

    def _on_magnify(self, filename: str) -> None:
        """Open the image detail screen for the given filename (non-blocking)."""
        if self._detail_loading:
            return
        self._detail_loading = True
        self._set_status("Loading image...", color=ESA_BLUE_BRIGHT)
        self._executor.submit(self._load_and_navigate_detail, filename)

    def _load_and_navigate_detail(self, filename: str) -> None:
        """Load an image in a background thread and navigate to detail view."""
        try:
            image = BackendInterface.load_prediction_image(filename)

            if image is None and self._file_type == DataSourceType.CUTANA:
                image = BackendInterface.load_cutana_cutout(filename)

            if image is None and self._file_type == DataSourceType.ZARR:
                batch = BackendInterface.load_container_images([filename])
                image = batch.get(filename)

            score = 0.0
            result = BackendInterface.get_prediction_result(filename)
            if result is not None:
                score = result["score"]

            self.app._detail_context = {
                "filename": filename,
                "score": score,
                "image": image,
                "back_screen": self._detail_back_screen(),
                # Re-decode hook in :class:`ImageDetailScreen` needs the
                # source type to pick the right loader path (#443).
                "source_type": self._file_type,
                # Training/prediction commits the source folder into
                # ``cfg.prediction_search_dir`` on start, but pass it
                # explicitly so the re-decode path doesn't have to
                # reach back into cfg state that may have been
                # invalidated by a later screen.
                "search_dir": self.cfg.prediction_search_dir,
            }
            self._set_status("Ready", color="white")
            self.navigate_to("image_detail")
        except Exception as exc:
            self._set_status(f"Failed to load image: {exc}", color=ESA_RED)
            logger.warning(f"Magnify failed for {filename}: {exc}")
        finally:
            self._detail_loading = False

    def _detail_back_screen(self) -> str:
        """Return the screen name to navigate back to from image detail.

        Override in subclasses that register under a different name.

        Returns:
            Screen name string.
        """
        return "prediction"

    def _on_star(self, filename: str) -> None:
        """Remember the given file."""
        try:
            BackendInterface.remember_file(filename)
            logger.info(f"Remembered: {filename}")
        except Exception as exc:
            logger.warning(f"Could not remember {filename}: {exc}")

    def _on_sort_change(self) -> None:
        """Handle sort mode change from gallery controls."""
        self._current_page = 0
        self._refresh_gallery()

    def _on_page_change(self, page: int) -> None:
        """Handle page change from gallery pagination."""
        self._current_page = page
        self._refresh_gallery()

    def _on_scale_change(self) -> None:
        """Handle slider change — re-page at the new effective chunk size.

        Anchored on :attr:`_gallery_first_offset` so the user lands on
        the page that still contains the cell they were looking at,
        rather than the same numeric ``_current_page`` under a stale
        chunk size (#432).
        """
        page_size = self.gallery.effective_page_size
        if page_size <= 0:
            return
        self._current_page = self._gallery_first_offset // page_size
        self._refresh_gallery()

    # ── Polling ──────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        self._polling = True
        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="am-poll")
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        while self._polling:
            try:
                self._set_loading(True)
                self._poll_db_update()
            except Exception as exc:
                logger.debug("Poll iteration failed: %s", exc)
            finally:
                self._set_loading(False)

            interval = self._get_poll_interval()
            for remaining in range(interval, 0, -1):
                if not self._polling:
                    self._update_countdown("")
                    return
                self._update_countdown(f"Refresh in {remaining}s")
                self._poll_stop.wait(timeout=1)
                if self._poll_stop.is_set():
                    self._update_countdown("")
                    return

    def _poll_db_update(self) -> None:
        count = BackendInterface.get_prediction_count()
        if count == self._last_db_count:
            return
        self._last_db_count = count

        elapsed = time.time() - self._start_time
        speed = count / elapsed if elapsed > 0 else 0
        self._update_stats_panel(count, speed, elapsed)
        self._update_poll_progress(count, speed, elapsed)

        if count >= 10:
            self._update_histogram()

        # Background refresh: update the counter and swap in fresh images
        # without flashing a row of spinners on the page the user is reading.
        # This also (re-)warms the prefetch window around the current page.
        self._refresh_gallery(user_initiated=False)

    def _stop_polling(self) -> None:
        self._polling = False
        self._poll_stop.set()
        # Signal page-load tasks to abort via page mismatch
        self._image_load_page = -1
        # Bump the prefetch generation so any in-flight window warm bails out
        # at its next page boundary instead of grinding on after teardown.
        self._prefetch_generation += 1
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=2)
            self._poll_thread = None
        if self._prefetch_future is not None:
            self._prefetch_future.cancel()
            self._prefetch_future = None
        self._update_countdown("")
        self._set_loading(False)

    # ── Hooks for subclasses ─────────────────────────────────────────

    def _get_poll_interval(self) -> int:
        """Return the polling interval in seconds.

        Override to use a dynamic value (e.g. from a dropdown widget).

        Returns:
            Polling interval in seconds.
        """
        return _DEFAULT_POLL_INTERVAL

    def _update_poll_progress(self, count: int, speed: float, elapsed: float) -> None:
        """Update the progress bar during a poll cycle.

        Override to add extra information (e.g. chunk tracking).

        Args:
            count: Current number of results in the DB.
            speed: Processing speed in images/second.
            elapsed: Elapsed time in seconds.
        """
        if self._total_source_count > 0:
            pct = min(count / self._total_source_count, 1.0)
            self.progress.progress_bar.value = pct
            remaining = self._total_source_count - count
            eta = remaining / speed if speed > 0 else 0
            self.progress.status_label.value = (
                f"{count:,} / {self._total_source_count:,} "
                f"({pct * 100:.1f}%) | {speed:.1f} img/s | ETA: {_format_duration(eta)}"
            )

    # ── Stats panel ──────────────────────────────────────────────────

    def _update_stats_panel(self, count: int, speed: float, elapsed: float) -> None:
        self.stats_html.value = self._render_stats(count, speed, elapsed)

    def _render_stats(self, count: int, speed: float, elapsed: float) -> str:
        lines = [f"Processed: <b>{count:,}</b> images"]

        if self._total_source_count > 0:
            lines.append(f"Total sources: <b>{self._total_source_count:,}</b>")

        if count > 0:
            try:
                lo, hi = BackendInterface.get_prediction_score_range()
                lines.append(f"Score range: [{lo:.4f}, {hi:.4f}]")
            except ValueError:
                logger.debug("Score range not available yet")

        if speed > 0:
            lines.append(f"Speed: {speed:.1f} img/s")

        if elapsed > 0:
            lines.append(f"Elapsed: {_format_duration(elapsed)}")

        if self._total_source_count > 0 and count > 0 and speed > 0:
            remaining = self._total_source_count - count
            if remaining > 0:
                eta_seconds = remaining / speed
                lines.append(f"ETA: <b>{_format_duration(eta_seconds)}</b>")
            else:
                lines.append("ETA: <b>Done</b>")

        return (
            '<div style="color:white; font-size:12px; line-height:1.6;">'
            + "<br>".join(lines)
            + "</div>"
        )

    # ── Histogram ────────────────────────────────────────────────────

    def _update_histogram(self) -> None:
        try:
            counts, edges = BackendInterface.get_prediction_histogram(bins=30)
        except Exception as exc:
            logger.warning("Histogram generation failed: %s", exc)
            return

        if counts.sum() == 0:
            return

        try:
            png_bytes = render_histogram_png(
                counts, edges, bar_color=ESA_BLUE_BRIGHT, log_y=self._score_histogram_log_y
            )
            self._histogram_image.value = png_bytes
            self._histogram_image.layout.display = ""
        except Exception as exc:
            logger.debug(f"Histogram render failed: {exc}")

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_busy(self, busy: bool) -> None:
        for btn in self._action_buttons:
            btn.disabled = busy

    def _set_status(self, text: str, color: str = "white") -> None:
        self.status_html.value = self._status_text(text, color)

    def _set_loading(self, loading: bool) -> None:
        """Show or hide the loading spinner."""
        self._loading_html.value = _SPINNER_HTML if loading else ""

    def _update_countdown(self, text: str) -> None:
        """Update the refresh countdown display."""
        if text:
            self._countdown_html.value = f'<span style="color:#888; font-size:10px;">{text}</span>'
        else:
            self._countdown_html.value = ""

    @staticmethod
    def _status_text(text: str, color: str = "white") -> str:
        """Render a status message as styled HTML.

        Args:
            text: The status message.
            color: CSS color for the text.

        Returns:
            HTML string for the status widget.
        """
        return f'<div style="color:{color}; font-size:14px; padding:4px 0;">{text}</div>'
