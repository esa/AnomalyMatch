#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Training setup screen — choose source folder, labels CSV, and model checkpoint."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import TYPE_CHECKING

import ipywidgets as widgets
import numpy as np
import pandas as pd
from fitsbolt import SUPPORTED_IMAGE_EXTENSIONS
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from ipywidgets import HBox, VBox
from loguru import logger

from anomaly_match.data_io.load_images import load_and_process_single_wrapper
from anomaly_match.datasets.cutana_source import detect_cutana_filter_names
from anomaly_match.datasets.Label import (
    LABEL_ANOMALY,
    LABEL_NORMAL,
    LABEL_REMOVED,
    VALID_CSV_LABELS,
)
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.utils.normalisation_parameters import (
    DEFAULT_IMAGE_EXTENSIONS,
    EXTRACTION_AFFECTING_NORM_FIELDS,
)
from anomaly_match.utils.validate_config import diff_configs, serialisable_config
from anomaly_match_ui.screens.base_screen import BaseScreen
from anomaly_match_ui.styles import (
    BG_COLOR,
    ESA_BLUE_BRIGHT,
    ESA_BLUE_DEEP,
    ESA_GREEN,
    ESA_RED,
    inline_spinner_html,
)
from anomaly_match_ui.utils import ui_state
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.chooser_paths import initial_dir, initial_file
from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream
from anomaly_match_ui.utils.progress_tap import LogProgressTap, clean_hint_text
from anomaly_match_ui.utils.scrolling_log_sink import ScreenLogSink
from anomaly_match_ui.widgets.file_chooser import AMFileChooser
from anomaly_match_ui.widgets.gallery_widget import GalleryWidget
from anomaly_match_ui.widgets.header_widget import HeaderWidget
from anomaly_match_ui.widgets.labelable_thumbnail_cell import LabelableThumbnailCell, LabelState
from anomaly_match_ui.widgets.normalisation_config_widget import NormalisationConfigWidget
from anomaly_match_ui.widgets.preview_grid import PreviewGrid

# Cap for the candidate list we pre-stratify and decode for the preview
# gallery.  Pagination at PAGE_SIZE per page walks this cap so the user
# gets multiple pages of labelable cutouts without us having to thread
# offset through ``load_preview_samples`` for every page turn (#427).
#
# Cap applied uniformly to every source type.  For Cutana the raw cutouts behind
# these cells are held in memory and re-decoded in place on a normalisation
# change (#501); the raws are ~2.4 MB each (384x384x4 float32), so this also
# bounds that hold (~2.4 GB at 1000).
PREVIEW_MAX_CELLS = 1000

# Quiet period before a normalisation edit triggers a preview refresh.  Collapses a
# burst of edits — a zoom-out slider drag, or several fields changed in a row —
# into a single decode of the settled values.
_NORM_REFRESH_DEBOUNCE_SECONDS = 0.3

# ``ui_state`` key under which this screen remembers its normalisation choices.
_NORM_STATE_PURPOSE = "training_setup"

# Map from CSV label string back to LabelState for round-tripping
# existing labels through the labelable cells.
_CSV_TO_LABEL_STATE = {LABEL_ANOMALY: LabelState.ANOMALY, LABEL_NORMAL: LabelState.NORMAL}

# Inverse: LabelState → CSV string for setup-time edits we merge on Start.
_LABEL_STATE_TO_CSV = {LabelState.ANOMALY: LABEL_ANOMALY, LabelState.NORMAL: LABEL_NORMAL}

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp  # lazy: avoid circular import

_CHECK = f'<span style="color:{ESA_GREEN};">&#10003;</span>'
_CROSS = f'<span style="color:{ESA_RED};">&#10007;</span>'
_SPINNER_INLINE = inline_spinner_html(margin="0 6px 0 0")
_INFO = f'<span style="color:{ESA_BLUE_BRIGHT};">&#8505;</span>'

_REQUIRED_CSV_COLUMNS = {"id", "label"}


def _status_line(icon: str, text: str) -> str:
    return f'<div style="color:white; font-size:13px; padding:2px 0;">{icon} {text}</div>'


def _strip_image_extension(name: str) -> str:
    """Drop a trailing image-file extension from a Zarr source id for display.

    Only real image extensions go: astronomical names such as
    ``J123456.78+123456.7`` contain dots that ``os.path.splitext`` would cut.

    Returns:
        *name* without a ``.png``/``.jpeg``/... suffix, otherwise unchanged.
    """
    base, ext = os.path.splitext(name)
    return base if ext.lower() in SUPPORTED_IMAGE_EXTENSIONS else name


def _input_channel_names(n_channels: int) -> list[str]:
    """Column names for a source with *n_channels* channels per image.

    Returns:
        ``R/G/B`` for three channels, ``Grey`` for one, else ``Ch 1..N``.
    """
    if n_channels == len(DEFAULT_IMAGE_EXTENSIONS):
        return list(DEFAULT_IMAGE_EXTENSIONS)
    if n_channels == 1:
        return ["Grey"]
    return [f"Ch {i + 1}" for i in range(n_channels)]


def _stratify_by_label(
    candidates: list[str],
    labeled_map: dict[str, str],
    *,
    key,
    max_items: int,
) -> list[str]:
    """Interleave candidates by label class so the preview shows both kinds.

    Label CSVs are typically heavily skewed toward ``normal`` (often
    20:1+), so taking the first ``max_items`` entries usually yields an
    all-``normal`` preview.  Split by class, then interleave so the first
    slots alternate between anomaly and normal.

    Args:
        candidates: Ordered list of candidate ids/filenames.
        labeled_map: ``{id: csv_label_string}`` from ``read_label_map``.
        key: Callable mapping a candidate to its key in ``labeled_map``.
        max_items: Maximum number of items to return.

    Returns:
        Up to ``max_items`` candidates with labels evenly represented.
    """
    buckets: dict[str, list[str]] = {LABEL_ANOMALY: [], LABEL_NORMAL: []}
    other: list[str] = []
    for c in candidates:
        lbl = labeled_map.get(key(c), "")
        if lbl in buckets:
            buckets[lbl].append(c)
        else:
            other.append(c)

    result: list[str] = []
    iters = [iter(buckets[LABEL_ANOMALY]), iter(buckets[LABEL_NORMAL]), iter(other)]
    while len(result) < max_items and any(iters):
        next_iters = []
        for it in iters:
            if len(result) >= max_items:
                break
            item = next(it, None)
            if item is not None:
                result.append(item)
                next_iters.append(it)
        iters = next_iters
    return result


def _restorable_normalisation(record: dict, extensions: list[str]) -> dict:
    """Turn a persisted normalisation record into widget-ready settings.

    JSON round-tripping flattens the method enum to an int and the
    channel-combination matrix to nested lists; both are restored to the
    types the widget and ``cfg.normalisation`` expect.

    The matrix — and everything sized to the output-channel count it implies —
    is dropped whenever the recorded input-channel names differ from
    *extensions*.  A different catalogue can expose a different band set (e.g.
    ``["VIS"]`` instead of ``["VIS", "NIR-H"]``), and a matrix shaped for the
    previous bands would silently combine the wrong columns rather than fail
    loudly.

    The state file is user-editable and may have been written by another
    version, so an unusable record is logged and skipped rather than allowed
    to break the screen it is read from.

    Args:
        record: Record from :func:`ui_state.get_normalisation_settings`.
        extensions: Input-channel names the widget currently shows.

    Returns:
        Settings dict for ``NormalisationConfigWidget.update_from_config``,
        or an empty dict when the record cannot be applied.
    """
    settings = dict(record["settings"])
    if "normalisation_method" not in settings:
        logger.warning("Ignoring remembered normalisation settings — no normalisation_method")
        return {}
    try:
        settings["normalisation_method"] = NormalisationMethod(settings["normalisation_method"])
    except (ValueError, TypeError):
        logger.warning(
            "Ignoring remembered normalisation settings — {!r} is not a known normalisation method",
            settings["normalisation_method"],
        )
        return {}

    matrix = settings.get("channel_combination")
    if record["extensions"] != list(extensions):
        logger.info(
            "Input channels changed ({} → {}) — keeping remembered normalisation but "
            "resetting the channel layout to its default",
            record["extensions"],
            list(extensions),
        )
        # The per-channel ASINH lists are sized to the dropped channel count;
        # applying them onto a differently-sized widget would leave the extra
        # channels on factory defaults and the rest on remembered values.
        for key in (
            "channel_combination",
            "n_output_channels",
            "norm_asinh_scale",
            "norm_asinh_clip",
        ):
            settings.pop(key, None)
    elif isinstance(matrix, list):
        settings["channel_combination"] = np.asarray(matrix, dtype=np.float64)
    return settings


def _validate_label_csv(path: str) -> tuple[bool, str, pd.DataFrame | None]:
    """Validate a labelled_data CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (is_valid, message, dataframe_or_None).
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"Cannot read CSV: {exc}", None

    missing = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        return False, f"Missing columns: {', '.join(sorted(missing))}", None

    invalid_labels = set(df["label"].unique()) - VALID_CSV_LABELS
    if invalid_labels:
        return False, f"Invalid labels: {', '.join(sorted(invalid_labels))}", None

    n_anom = (df["label"] == LABEL_ANOMALY).sum()
    n_norm = (df["label"] == LABEL_NORMAL).sum()
    n_rem = (df["label"] == LABEL_REMOVED).sum()
    summary = f"{n_anom} anomaly, {n_norm} normal, {n_rem} removed"

    if n_anom == 0 or n_norm == 0:
        missing = "anomaly" if n_anom == 0 else "normal"
        return (
            False,
            f"{summary} — need at least one {missing} label to train",
            df,
        )

    return True, summary, df


class TrainingSetupScreen(BaseScreen):
    """Setup screen for configuring a training run.

    Provides file choosers for the source folder, label CSV, and optional
    metadata CSV, plus normalisation configuration.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        self._cfg = BackendInterface.get_config()
        self._source_ok = False
        self._labels_ok = False
        self._source_count: int | None = None
        self._counting = False
        self._counting_status: str = "Counting images..."
        self._image_files: list[str] = []
        self._label_message: str = ""
        self._label_validation_message: str = ""
        self._detected_source_type: DataSourceType = DataSourceType.IMAGE_FOLDER
        self._found_source_ids: list[str] = []
        self._suppress_norm_preview: bool = False
        # Monotonic counter — incremented each time a new background job
        # starts.  Any running job whose generation doesn't match bails out.
        self._job_generation: int = 0
        # Cache the most recent label-validation result so norm-only changes
        # don't re-run expensive ID-matching against the data source.
        # Keyed on (source_dir, source_type, label_path, label_mtime).
        self._validation_cache: dict | None = None
        # Signature of the inputs the last on_enter auto-validation ran against
        # (source path, label file, normalisation).  Lets a pure back-navigation
        # — e.g. returning from the image-detail screen with nothing changed —
        # reuse the held count + preview instead of re-running the full
        # count → validate → preview job.  Reset on explicit source/label edits.
        self._auto_validate_signature: tuple | None = None
        # Mirror Cutana's INFO-level progress messages into the preview
        # spinner caption so the user sees what's happening while Cutana
        # extracts cutouts.  Installed/removed on screen enter/leave.
        self._progress_tap: LogProgressTap | None = None
        # Last norm config seen by ``_on_norm_change`` — deduplicates the
        # observer cascade when ``update_from_config`` (or
        # ``_add_channel``/``_remove_channel``) assigns several widgets
        # in a row, each firing ``observe``.  Without this, a single
        # screen-enter could spawn ten or more background jobs.
        self._last_norm_config: dict | None = None
        # In-memory store for label edits made on this screen.  Holds
        # only the user's setup-time toggles; existing CSV labels are
        # *not* mirrored here — round-tripping happens at render time
        # via the labeled_map argument to ``_render_preview_items``.
        # On Start Training we merge ``_setup_labels`` into the chosen
        # CSV via ``BackendInterface.merge_gallery_labels`` so the
        # training subprocess sees the up-to-date label set (#427).
        self._setup_labels: dict[str, LabelState] = {}
        # Decoded preview samples for the current source, paged by the
        # gallery.  Tuples of ``(name, png_bytes, csv_label_str)`` so the
        # gallery can render the existing label badge on each cell.
        self._preview_samples: list[tuple[str, bytes, str]] = []
        # Absolute offset of the first cell currently rendered into the
        # gallery.  Tracked in addition to the gallery's ``_current_page``
        # so slider changes (which alter the effective page size) can
        # land the user back near the same cell instead of the same
        # *page index* under a stale chunk size (#432).
        self._preview_first_offset: int = 0
        # Numpy-array versions of the preview samples, keyed by
        # filename — needed for the detail screen, which expects raw
        # HWC uint8 arrays (PreviewWidget.display_standalone) rather
        # than the PNG bytes the gallery cell displays.
        self._preview_raw_images: dict[str, np.ndarray] = {}
        # Normalisation-independent raw float32 cutouts behind the current
        # Cutana preview, in gallery order.  Held so a normalisation change
        # re-decodes them in-process instead of re-reading the labeled cache
        # over NFS and spawning a fresh count+validate+read job (#501).  None
        # whenever the preview isn't fully cache-backed (image folder, Zarr, or
        # a Cutana cache miss) — those fall back to the full reload path.
        self._preview_raw_cutouts: list[tuple[str, np.ndarray]] | None = None
        # Label map captured alongside the held raws so the in-memory re-decode
        # (#501) can render label badges without re-reading the CSV over NFS —
        # the label map can't change during a normalisation tweak (editing the
        # CSV chooser triggers a separate full reload).
        self._preview_labeled_map: dict[str, str] = {}
        # Serialises the nested setattr sequence in
        # ``_apply_norm_overrides_to_cfg`` so the UI-thread invariant
        # survives a future refactor that lets a worker thread call it.
        # ipykernel already serialises widget callbacks today, but that's
        # convention not contract — without this lock a debounced observer
        # firing from a background timer would silently re-introduce the
        # torn-write race against ``self._cfg.normalisation``.
        self._norm_apply_lock = threading.Lock()
        # Debounce timer + guard for coalescing rapid normalisation edits (e.g.
        # a zoom-out slider drag) into a single preview refresh.
        self._norm_refresh_timer: threading.Timer | None = None
        self._norm_refresh_lock = threading.Lock()
        # Set when a debounced edit touched an EXTRACTION_AFFECTING_NORM_FIELD.
        # Those change the pixels Cutana extracts, so the held raws are stale and
        # the in-memory re-decode below cannot show the change (#592:
        # the zoom-out slider appeared to do nothing on a warm cache).
        self._norm_refresh_needs_extraction = False
        # Single-flight guard for preview jobs.  ``_job_in_flight`` is True while
        # ANY preview job runs (initial load, source/label change, norm refresh).
        # A norm change arriving while one is in flight sets ``_refresh_pending``
        # and the in-flight job's completion re-runs the refresh once — by then
        # the raws are held, so it takes the fast in-memory path.  This stops a
        # norm change from spawning a *concurrent* cache read on top of a still-
        # running initial load (the raws aren't held until that load finishes).
        self._job_in_flight = False
        self._refresh_pending = False
        self._log_sink = ScreenLogSink("TrainingSetupScreen")

    # ── Build ────────────────────────────────────────────────────────

    def build(self) -> widgets.Widget:
        """Build and return the training setup layout.

        Returns:
            The root VBox widget for the setup screen.
        """
        self._out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid white",
                height="120px",
                background_color=BG_COLOR,
                overflow="auto",
            ),
        )
        header = HeaderWidget(self.app, self._out, show_back_button=True)

        # ── Source folder chooser ─────────────────────────────────
        # Resolve the source-chooser starting dir once so auto-validation
        # scans the same folder the chooser displays.  When persistence
        # picks a different dir than cfg.data_dir, _auto_validate_from_config
        # would otherwise walk cfg.data_dir (the test-data default) while
        # the chooser shows the persisted dir, and the user gets
        # mismatched "Source folder = q1_data / Found 19 images" reports.
        self._source_initial_path = initial_dir(self._cfg.data_dir, purpose="source_chooser")
        source_has_default = bool(
            self._source_initial_path and os.path.isdir(self._source_initial_path)
        )
        self._source_chooser = AMFileChooser(
            path=self._source_initial_path,
            show_only_dirs=True,
            select_default=source_has_default,
            title="Source folder",
            purpose="source_chooser",
        )
        self._source_chooser.register_callback(self._on_source_change)

        # ── Label CSV chooser ─────────────────────────────────────
        label_dir, label_name = initial_file(
            self._cfg.label_file,
            purpose="label_chooser",
            is_shipped_default=BackendInterface.is_shipped_default_path(
                "label_file", self._cfg.label_file
            ),
        )
        self._label_chooser = AMFileChooser(
            path=label_dir,
            filename=label_name,
            select_default=bool(label_name),
            filter_pattern="*.csv",
            title="Labelled data CSV",
            purpose="label_chooser",
        )
        self._label_chooser.register_callback(self._on_label_change)

        # ── Metadata CSV chooser (optional) ───────────────────────
        meta_dir, meta_name = initial_file(
            self._cfg.metadata_file,
            purpose="metadata_chooser",
            is_shipped_default=BackendInterface.is_shipped_default_path(
                "metadata_file", self._cfg.metadata_file
            ),
        )
        self._meta_chooser = AMFileChooser(
            path=meta_dir,
            filename=meta_name,
            select_default=bool(meta_name),
            filter_pattern="*.csv",
            title="Metadata CSV (optional)",
            purpose="metadata_chooser",
        )

        # ── Preview grid ──────────────────────────────────────────
        # Keep PreviewGrid for the spinner / error-message panel — its
        # status / message HTML is exactly what we want for the
        # "Loading preview…" + "Select a label CSV" affordances.
        # The actual cell rendering moves to GalleryWidget so cells get
        # pagination + per-cell label toggles (#427).
        self._preview_grid = PreviewGrid()
        # Hide the static cell grid that ships with PreviewGrid — we
        # render cells through GalleryWidget instead.
        self._preview_grid._grid.layout.display = "none"

        def _setup_cell_factory(**kwargs: object) -> LabelableThumbnailCell:
            # The factory is called with on_magnify / on_star kwargs by
            # GalleryWidget; override on_magnify with our own handler so
            # the magnifying-glass button on a setup-screen cell opens
            # the image detail view (the gallery's default on_magnify
            # is None at setup time).
            kwargs["on_magnify"] = self._on_setup_magnify
            return LabelableThumbnailCell(on_label_change=self._on_setup_label_change, **kwargs)

        self._preview_gallery = GalleryWidget(
            setup_mode=True,
            cell_factory=_setup_cell_factory,
        )
        self._preview_gallery.set_on_page_change(self._on_preview_page_change)
        # Slider widens cells → fewer fit in two rows → effective page
        # size shrinks → re-render the current page so paging walks the
        # new chunk size and the user doesn't skip cells (#432).
        self._preview_gallery.set_on_scale_change(self._on_preview_scale_change)
        # Hide until we have items.
        self._preview_gallery.widget.layout.display = "none"

        # Wrap the three PreviewGrid status-affording methods so any
        # spinner / message / clear also hides the cell gallery — the
        # two panels share the same screen real estate and only one
        # should be visible at a time.
        _orig_show_spinner = self._preview_grid.show_spinner
        _orig_show_message = self._preview_grid.show_message
        _orig_clear = self._preview_grid.clear

        def _show_spinner(*args: object, **kwargs: object) -> None:
            self._preview_gallery.widget.layout.display = "none"
            _orig_show_spinner(*args, **kwargs)

        def _show_message(*args: object, **kwargs: object) -> None:
            self._preview_gallery.widget.layout.display = "none"
            _orig_show_message(*args, **kwargs)

        def _clear() -> None:
            self._preview_gallery.widget.layout.display = "none"
            _orig_clear()

        self._preview_grid.show_spinner = _show_spinner  # type: ignore[method-assign]
        self._preview_grid.show_message = _show_message  # type: ignore[method-assign]
        self._preview_grid.clear = _clear  # type: ignore[method-assign]

        # ── Normalisation config ─────────────────────────────────
        norm_cfg = self._cfg.normalisation
        self._norm_widget = NormalisationConfigWidget(
            n_channels=norm_cfg.n_output_channels,
            image_size=list(norm_cfg.image_size),
        )
        # Restore before registering the observer so the widget assignments
        # can't spawn a preview job, and before ``on_enter`` reads
        # ``cfg.normalisation`` back into the widget.  Same rationale as
        # ``initial_dir``: what the user last ran with is more current than a
        # notebook's cfg defaults, which would otherwise clobber it on every
        # fresh kernel.  Nothing is reading cfg yet at build time, so pushing
        # the restored values across is safe here.
        if self._restore_persisted_normalisation():
            self._apply_norm_overrides_to_cfg()
        self._norm_widget.register_on_change(self._on_norm_change)
        # Seed the dedup snapshot with the widget's initial state so the
        # observer cascade from build-time widget assignments is not
        # mistaken for a real user change.
        self._last_norm_config = serialisable_config(self._norm_widget.get_normalisation_config())

        # ── Training iterations ───────────────────────────────────
        self._iter_slider = widgets.IntSlider(
            value=self._cfg.num_train_iter,
            min=50,
            max=600,
            step=10,
            description="Iterations:",
            style={"description_width": "initial", "handle_color": "white"},
            layout=widgets.Layout(width="320px"),
        )

        # ── Source-size stratification (Cutana only) ──────────────
        # Euclid Q1/DR1 sources skew heavily small.  When the unlabelled source
        # is a Cutana catalogue (which carries each source's diameter), sample
        # the unlabelled training pool ~uniform in size so tiny sources don't
        # dominate.  Disabled for image/Zarr sources, which expose no per-source
        # size — ``_refresh_validation`` enables it once a Cutana source is
        # detected.
        self._stratify_checkbox = widgets.Checkbox(
            value=self._cfg.cutana_stratify_source_size,
            description="Stratify source size per tile (Cutana)",
            indent=False,
            disabled=True,
            style={"description_width": "initial"},
            layout=widgets.Layout(width="360px"),
        )

        # ── Validation panel + start button ───────────────────────
        self._validation_html = widgets.HTML(
            value=self._render_validation(),
            layout=widgets.Layout(padding="8px 0"),
        )

        self._start_btn = widgets.Button(
            description="Start Training",
            button_style="success",
            disabled=True,
            layout=widgets.Layout(width="200px", height="40px"),
            style={"font_size": "14px"},
        )
        self._start_btn.on_click(lambda _: self._on_start())

        # ── Layout ────────────────────────────────────────────────
        # Left: file choosers (constrained so preview gets space)
        left_panel = VBox(
            [self._source_chooser, self._label_chooser, self._meta_chooser],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="12px",
                gap="4px",
                flex="0 1 auto",
                max_width="420px",
            ),
        )

        # Centre: image preview (flex so it grows with window width)
        centre_panel = VBox(
            [self._preview_grid.widget, self._preview_gallery.widget],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="8px",
                flex="1 1 auto",
                min_width="140px",
                border_left=f"1px solid {ESA_BLUE_DEEP}",
                border_right=f"1px solid {ESA_BLUE_DEEP}",
            ),
        )

        # Right: normalisation config + iterations + validation + start
        right_panel = VBox(
            [
                self._norm_widget.norm_section,
                self._norm_widget.channel_section,
                self._iter_slider,
                self._stratify_checkbox,
                self._validation_html,
                self._start_btn,
            ],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="8px 12px",
                gap="8px",
                width="380px",
                flex="0 0 auto",
            ),
        )

        form = HBox(
            [left_panel, centre_panel, right_panel],
            layout=widgets.Layout(
                background_color=BG_COLOR,
            ),
        )

        return VBox(
            [
                header.widget,
                form,
                self._out,
            ],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                border=f"1px solid {ESA_BLUE_DEEP}",
                width="100%",
            ),
        )

    # ── Lifecycle ────────────────────────────────────────────────────

    def on_enter(self) -> None:
        """Set up terminal output and auto-validate from config."""
        BackendInterface.set_terminal_output(self._out)
        self._log_sink.attach(self._out)
        self._install_progress_tap()
        self._auto_validate_from_config()
        # update_from_config() writes many widget values in sequence; each
        # assignment fires the value-observer callback, and without
        # suppression that would spawn a separate background job per field
        # (setup-screen gens jumped from 1 to 14 on a single re-enter).
        self._suppress_norm_preview = True
        try:
            self._norm_widget.update_from_config(dict(self._cfg.normalisation))
        finally:
            self._suppress_norm_preview = False
        # Resync the dedup snapshot so the next genuine user change
        # registers as a diff against the post-restore widget state.
        self._last_norm_config = serialisable_config(self._norm_widget.get_normalisation_config())

    def on_leave(self) -> None:
        """Remove the progress tap and widget sink when navigating away."""
        with self._norm_refresh_lock:
            if self._norm_refresh_timer is not None:
                self._norm_refresh_timer.cancel()
                self._norm_refresh_timer = None
        self._uninstall_progress_tap()
        self._log_sink.detach()

    def _install_progress_tap(self) -> None:
        """Start mirroring Cutana hints into the preview spinner caption.

        Cutana disables its own loguru namespace at import (so it stays
        quiet for apps that don't opt in), so its progress lines — the
        cutout/FITS-set hints the spinner is built to show — never reach
        the tap until we enable it.  Enable it here for the lifetime of
        the setup screen and restore the disabled default on leave.
        """
        if self._progress_tap is not None:
            return
        logger.enable("cutana")
        self._progress_tap = LogProgressTap(on_cutana_hint=self._on_cutana_hint)
        self._progress_tap.install()

    def _uninstall_progress_tap(self) -> None:
        if self._progress_tap is None:
            return
        self._progress_tap.uninstall()
        self._progress_tap = None
        # Restore Cutana's import-time default so its logs don't keep flowing
        # into other screens' output once we leave setup.  This assumes the
        # setup screen is the sole owner of the "cutana" namespace toggle
        # (true today — it's the only enable("cutana") in the app); if another
        # site ever enables it deliberately, this would need to snapshot and
        # restore the prior state instead of unconditionally disabling.
        logger.disable("cutana")

    def _on_cutana_hint(self, text: str) -> None:
        """Update the preview spinner caption with a Cutana status line.

        No-op when the preview grid is not currently showing a spinner,
        so the hint cannot stomp on a finished preview or an error
        message.  Runs on loguru's dispatch thread; ipywidgets attribute
        writes are thread-safe.

        Setup-screen hints are emitted in this UI kernel and carry no
        ``[subprocess]`` header, so ``clean_hint_text`` is a no-op here —
        but routing through it keeps caption cleaning uniform with the
        training screen and stays correct if a future hint is ever relayed.
        """
        self._preview_grid.update_spinner_text(clean_hint_text(text))

    # ── Auto-validate ────────────────────────────────────────────────

    def _auto_validate_from_config(self) -> None:
        """Pre-validate fields that already have values in cfg.

        Skips the full count → validate → preview job when re-entering the
        screen with nothing changed (e.g. returning from the image-detail
        screen): the widgets keep their state and the preview grid keeps its
        cutouts, so re-running would only blank and redo identical work.

        The skip signature keys on the label file + its mtime (an external edit
        at the same path must re-validate) and the normalisation widget's
        config (what the preview is decoded with).  A source change is detected
        not by the signature but by the ``_auto_validate_signature = None`` reset
        in :meth:`_on_source_change` / :meth:`_on_label_change`.  The skip only
        fires once the source has counted successfully (``_source_ok``) and no
        count is in flight.
        """
        # Resolve as the panel does, so a CSV picked in the chooser but not yet
        # committed to cfg is the one validated: keying on cfg here reported one
        # filename in the panel with another file's anomaly/normal counts.
        label_file = self._resolve_label_path()
        label_mtime = os.path.getmtime(label_file) if label_file else None
        signature = (
            label_file,
            label_mtime,
            serialisable_config(self._norm_widget.get_normalisation_config()),
        )
        if signature == self._auto_validate_signature and self._source_ok and not self._counting:
            logger.debug(
                "Setup re-entry with unchanged label + normalisation — "
                "reusing cached validation + preview instead of re-running the job"
            )
            return
        self._auto_validate_signature = signature

        if label_file:
            self._validate_label_file(label_file)

        # Scan the chooser's initial path, not ``cfg.data_dir`` directly.
        # ``initial_dir`` may have resolved to a persisted last-browsed
        # dir that overrides cfg's default — using cfg here would walk a
        # different directory than the chooser is showing on the left
        # side of the screen (#429 follow-up).
        if self._source_initial_path and os.path.isdir(self._source_initial_path):
            self._start_background_job(self._source_initial_path)

    # ── File chooser callbacks ───────────────────────────────────────

    def _on_source_change(self, _chooser: object) -> None:
        # Explicit edit: drop the auto-validate skip-signature so the next
        # screen entry re-validates rather than reusing a now-stale result.
        self._auto_validate_signature = None
        path = self._source_chooser.selected
        if path and (os.path.isdir(path) or os.path.isfile(path)):
            self._source_ok = False
            self._source_count = None
            self._image_files = []
            self._sample_info: dict[str, object] | None = None
            self._start_background_job(path)
        else:
            self._source_ok = False
            self._source_count = None
            self._counting = False
            self._image_files = []
            self._sample_info = None
            self._preview_grid.clear()
            self._refresh_validation()

    def _on_label_change(self, _chooser: object) -> None:
        self._auto_validate_signature = None
        path = self._label_chooser.selected
        if path and os.path.isfile(path):
            self._found_source_ids = []
            self._validate_label_file(path)
            # Re-run the full background job so validation and preview
            # happen on a single thread, sequentially.
            if self._source_ok:
                source = self._source_chooser.selected or self._cfg.data_dir
                if source:
                    self._start_background_job(source, skip_counting=True)
            else:
                self._preview_grid.clear()
        else:
            self._labels_ok = False
            self._label_message = ""
            self._label_validation_message = ""
            self._refresh_validation()

    def _validate_label_file(self, path: str) -> None:
        """Validate the label CSV."""
        valid, message, _df = _validate_label_csv(path)
        self._labels_ok = valid
        self._label_message = message
        self._refresh_validation()

    # ── Background job — single thread for count + validate + preview ─

    def _apply_norm_overrides_to_cfg(self) -> None:
        """Push the current widget normalisation state into ``self._cfg``.

        Called before any phase that depends on up-to-date normalisation
        (cache build, preview load).  Idempotent.

        Thread-safe: holds ``self._norm_apply_lock`` across the setattr
        sequence so concurrent callers can't observe a half-written
        ``cfg.normalisation`` (the bug this PR is fixing for the UI-thread
        case — the lock keeps the invariant intact even if a future
        refactor lets a worker thread call this).
        """
        norm_overrides = self._norm_widget.get_normalisation_config()
        with self._norm_apply_lock:
            for key, value in norm_overrides.items():
                setattr(self._cfg.normalisation, key, value)

    def _restore_persisted_normalisation(self, record: dict | None = None) -> bool:
        """Seed the normalisation widget from the last settings a run started with.

        Touches the widget only.  Callers are responsible for getting the
        restored values into ``cfg.normalisation``, because how that is done
        safely depends on whether a background job is currently reading it.

        Args:
            record: A record already read from ``ui_state``, to avoid a second
                disk read (and the window in which the two could disagree).
                Read here when omitted.

        Returns:
            ``True`` when remembered settings were applied.
        """
        if record is None:
            record = ui_state.get_normalisation_settings(_NORM_STATE_PURPOSE)
        if record is None:
            return False
        settings = _restorable_normalisation(record, self._norm_widget.extensions)
        if not settings:
            return False
        self._apply_restored_normalisation(settings, record["extensions"])
        return True

    def _apply_restored_normalisation(self, settings: dict, extensions: list[str]) -> None:
        """Push already-converted remembered *settings* into the widget.

        Split out from :meth:`_restore_persisted_normalisation` so a caller that
        has to know whether the record is usable *before* it touches the widget
        (the Cutana band path) can convert first and apply after, rather than
        committing to a branch and discovering the record was unusable later.

        Args:
            settings: Output of :func:`_restorable_normalisation` — non-empty.
            extensions: Input-channel names the record was stored against, for
                the log line.
        """
        # ``update_from_config`` assigns many widgets in sequence and each
        # assignment fires the value observer; suppress the preview jobs that
        # cascade would otherwise spawn.
        self._suppress_norm_preview = True
        try:
            self._norm_widget.update_from_config(settings)
        finally:
            self._suppress_norm_preview = False
        self._last_norm_config = serialisable_config(self._norm_widget.get_normalisation_config())
        logger.info("Restored last-used normalisation settings for {}", extensions)

    def _start_background_job(self, folder: str, *, skip_counting: bool = False) -> None:
        """Start (or restart) the single background job for this folder.

        Any previously running job is cancelled via the generation counter.
        The job runs count → validate → preview sequentially on ONE thread.

        Args:
            folder: Source data directory.
            skip_counting: If True, skip the counting step (source already
                counted) and go straight to validation + preview.
        """
        with self._norm_refresh_lock:
            self._job_generation += 1
            generation = self._job_generation
            self._job_in_flight = True

        # Sync widget state into cfg on the UI thread BEFORE spawning the
        # daemon.  Previously every daemon called
        # ``_apply_norm_overrides_to_cfg`` itself, so two overlapping jobs
        # raced on the nested setattr sequence against self._cfg and
        # produced a half-written config — cutana's next catalogue read
        # then deadlocked on inconsistent state.  All reads happen from
        # worker threads; doing the writes here guarantees torn-write
        # free observation.
        self._apply_norm_overrides_to_cfg()

        if not skip_counting:
            self._counting = True
            self._counting_status = "Scanning sources..."
        self._preview_grid.show_spinner("Loading preview...")
        self._refresh_validation()

        logger.info(
            "Starting background job gen={} for {} (skip_counting={})",
            generation,
            folder,
            skip_counting,
        )
        threading.Thread(
            target=self._background_job,
            args=(folder, generation, skip_counting),
            daemon=True,
            name=f"setup-job-{generation}",
        ).start()

    def _background_job(self, folder: str, generation: int, skip_counting: bool) -> None:
        """Single background thread: count sources → validate labels → load preview.

        This is the ONLY method that runs background work for the setup
        screen.  No other methods spawn threads.  If a newer job has been
        requested (generation mismatch), this one bails out immediately.

        Args:
            folder: Source data directory.
            generation: Job generation for cancellation.
            skip_counting: Skip counting (already done).
        """

        def _stale() -> bool:
            return self._job_generation != generation

        try:
            # ── Phase 1: Count sources ──────────────────────────────
            if not skip_counting:
                self._phase_count_sources(folder, _stale)
                if _stale():
                    logger.debug("Job gen={} cancelled after counting", generation)
                    return

            # Normalisation was already synced to cfg on the UI thread by
            # ``_start_background_job`` before this daemon spawned — doing
            # it here again from a worker thread used to race with other
            # workers on nested DotMap setattrs and produce a half-written
            # config that deadlocked cutana's next catalogue read.

            # ── Phase 2: Validate labels against source ─────────────
            source_type = self._detected_source_type
            if self._source_ok and self._labels_ok:
                # Use the same resolver as ``_read_label_map`` so phase 2
                # validates the same CSV phase 3 will read — otherwise an
                # invalid chooser selection could short-circuit ``or`` and
                # leave the label-vs-source check using a different file
                # than the preview gallery (#429 follow-up).
                label_path = self._resolve_label_path()
                if label_path is not None:
                    if _stale():
                        return
                    self._phase_validate_labels(folder, source_type, label_path, _stale)

            if _stale():
                return

            # ── Phase 3: Load preview images ────────────────────────
            self._phase_load_preview(folder, source_type, _stale)

        except Exception as exc:
            logger.opt(exception=True).error(
                "Background job gen={} failed: {}",
                generation,
                exc,
            )
            if not _stale():
                self._preview_grid.show_message(f"Setup failed: {exc}")
        finally:
            self._on_preview_job_done(generation)

    def _phase_count_sources(self, folder: str, stale_check: callable) -> None:
        """Phase 1: Detect source type and count items.

        Args:
            folder: Source data directory.
            stale_check: Callable returning True if this job is cancelled.
        """
        result = BackendInterface.scan_and_count_sources(folder)

        # A slow scan (e.g. a large image folder) can outlive the job that
        # spawned it — by the time we get here, a newer job may already have
        # scanned a different folder and committed its result.  Bailing
        # before writing ``self._detected_source_type`` avoids clobbering
        # the newer job's state; a stale gen=1 IMAGE_FOLDER result used to
        # stomp on a live gen=2 CUTANA result, and later skip_counting
        # jobs then validated against the wrong source type.
        if stale_check():
            logger.debug("Stale scan result for {} — not applying", folder)
            return

        self._detected_source_type = result.source_type
        self._image_files = result.image_files
        logger.info("Found {:,} sources ({}) in {}", result.count, result.source_type, folder)
        if result.source_type == DataSourceType.CUTANA and result.count > 0:
            self._apply_cutana_band_info(folder)
        elif result.count > 0:
            self._apply_source_channel_info(folder, result.source_type, result.image_files)

        self._source_count = result.count
        self._source_ok = self._source_count > 0
        self._counting = False

        # Probe a sample image for metadata shown in validation panel
        if (
            self._source_ok
            and self._image_files
            and self._detected_source_type == DataSourceType.IMAGE_FOLDER
        ):
            self._probe_sample_image(folder, self._image_files[0])

        self._refresh_validation()

    def _apply_cutana_band_info(self, folder: str) -> None:
        """Auto-configure channel settings from Cutana catalogue bands.

        Both band detection (parquet I/O) and widget modification must run
        on the kernel event loop — the lazy ``cutana.catalogue_preprocessor``
        import deadlocks when executed from a background thread in Voila
        (Python's import lock interacts badly with the kernel event loop).

        Safe to call from any thread: if an event loop is running, the
        work is scheduled there; otherwise it runs inline (tests).
        """

        def _detect_and_apply() -> None:
            try:
                filter_names = detect_cutana_filter_names(folder, self._cfg)
                if filter_names:
                    logger.info("Cutana bands detected: {}", filter_names)
                    # A remembered channel-combination matrix only describes the
                    # band set it was built against, so only treat the widget as
                    # already reflecting the user's choice when the catalogue
                    # exposes exactly those bands again.  Otherwise fall back to
                    # the destructive auto-configuration (one output channel per
                    # detected band).
                    record = ui_state.get_normalisation_settings(_NORM_STATE_PURPOSE)
                    # Convert the record BEFORE choosing a band helper.  Matching
                    # band names alone are not enough to commit to the
                    # non-destructive branch: the record can still turn out to be
                    # unusable (an unknown normalisation method from another
                    # version), and then nothing restores the channel layout that
                    # ``sync_cutana_bands`` deliberately left alone — a
                    # single-band catalogue would keep a 3-channel broadcast and
                    # feed the model the wrong input shape, silently.
                    bands_match = record is not None and record["extensions"] == list(filter_names)
                    settings = (
                        _restorable_normalisation(record, filter_names) if bands_match else {}
                    )
                    remembered = bool(settings)
                    # Suppress preview refresh during programmatic widget
                    # changes — the counting thread triggers preview separately.
                    self._suppress_norm_preview = True
                    try:
                        if remembered:
                            self._norm_widget.sync_cutana_bands(filter_names)
                        else:
                            self._norm_widget.apply_cutana_bands(filter_names)
                    finally:
                        self._suppress_norm_preview = False
                    if remembered:
                        # Both band helpers rebuild the matrix from defaults, so
                        # the remembered cell values have to go back in after.
                        self._apply_restored_normalisation(settings, record["extensions"])
                        # We are on the event loop while the job that scheduled
                        # us is still reading cfg.normalisation from its worker
                        # thread, so the restored values must NOT be written
                        # there from here.  Route through the debounced refresh
                        # instead: it supersedes the running job (or defers
                        # until it finishes) and syncs cfg from that job's own
                        # thread, keeping the single-writer invariant.
                        self._schedule_norm_refresh()
            except Exception as exc:
                logger.opt(exception=True).warning("Cutana band detection failed: {}", exc)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_detect_and_apply)
                return
        except RuntimeError:
            pass
        # No running event loop (e.g. plain Python tests) — run inline
        _detect_and_apply()

    def _apply_source_channel_info(
        self, folder: str, source_type: DataSourceType, image_files: list[str] | None
    ) -> None:
        """Size the matrix's input columns to a Zarr store's or image folder's channels.

        Without this the widget kept whatever columns the previous source or
        the remembered settings left (e.g. four Cutana bands), and fitsbolt
        then refused to turn the store's three channels into four outputs.

        The channel count is read on the calling (background) thread; the
        widget is updated on the kernel event loop, like the Cutana path.

        Args:
            folder: Source directory.
            source_type: Detected source type.
            image_files: Image filenames for image-folder sources.
        """
        try:
            n_channels = BackendInterface.detect_source_channel_count(
                folder, source_type, image_files
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Channel detection failed for {}: {}", folder, exc)
            return
        if not n_channels:
            return
        names = _input_channel_names(n_channels)

        def _apply() -> None:
            # Same columns as before: keep the user's matrix as it is.
            if names == self._norm_widget.extensions:
                return
            record = ui_state.get_normalisation_settings(_NORM_STATE_PURPOSE)
            bands_match = record is not None and record["extensions"] == names
            settings = _restorable_normalisation(record, names) if bands_match else {}
            self._suppress_norm_preview = True
            try:
                self._norm_widget.apply_input_channels(names)
                self._norm_widget.show_cutout_zoom(False)
            finally:
                self._suppress_norm_preview = False
            if settings:
                self._apply_restored_normalisation(settings, record["extensions"])
            logger.info("Input channels set to {} for {}", names, folder)
            # The running job read cfg before this change; supersede it so the
            # preview decodes with the new channel layout.
            self._schedule_norm_refresh()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_apply)
                return
        except RuntimeError:
            pass
        _apply()

    # ── Phase 2: Validate labels against source ─────────────────────

    def _phase_validate_labels(
        self,
        source_dir: str,
        source_type: str,
        label_path: str,
        stale_check: callable,
    ) -> None:
        """Validate labeled files against the data source and build cache.

        Runs as part of the background job. Delegates all I/O to
        BackendInterface.

        Args:
            source_dir: Path to the data directory.
            source_type: Detected source type string.
            label_path: Path to the label CSV.
            stale_check: Callable returning True if this job is cancelled.
                Checked before the expensive cache build to avoid two
                concurrent jobs racing on the cache directory.
        """
        # Skip ID-matching if we already have a fresh result for this exact
        # (source, label) combination — label matching depends only on IDs,
        # not on normalisation, so norm-only changes must not re-show the
        # "Validating labels..." spinner (it flashes for no reason and
        # blanks the preview grid).  The cache build below still runs with
        # its own freshness guard (LabeledDataCache.needs_rebuild).
        try:
            label_mtime = os.path.getmtime(label_path)
        except OSError:
            label_mtime = 0.0
        cache_key = (source_dir, source_type, label_path, label_mtime)

        cached = self._validation_cache
        if cached and cached.get("key") == cache_key:
            message = cached["message"]
            found_locations = cached["found_locations"]
        else:
            self._preview_grid.show_spinner("Validating labels against source...")
            try:
                message, found_locations, partial = BackendInterface.validate_labels_against_source(
                    source_dir,
                    source_type,
                    label_path,
                    stale_check=stale_check,
                )
            except Exception as exc:
                logger.opt(exception=True).error("Label validation failed: {}", exc)
                self._label_validation_message = f"Validation error: {exc}"
                self._preview_grid.show_message(f"Label validation failed: {exc}")
                self._refresh_validation()
                return
            # The stale_check inside ``_validate_cutana`` short-circuits
            # the per-catalogue loop and returns a partial
            # ``found_locations``.  Caching or building a labeled cache
            # from that would silently drop labels (e.g. only 473 of
            # 1097 labels materialised in the cache, then training ran
            # on a fraction of the data).  Bail out now — the newer job
            # that triggered our staleness will validate fully.
            if partial:
                logger.debug(
                    "Validation bailed partway ({} matched so far); "
                    "skipping cache update — newer job will redo this",
                    len(found_locations),
                )
                return
            self._validation_cache = {
                "key": cache_key,
                "message": message,
                "found_locations": found_locations,
            }
            logger.info("Label validation: {} IDs found in source", len(found_locations))

        self._label_validation_message = message
        self._found_source_ids = list(found_locations.keys())
        self._refresh_validation()

        # Build labeled data cache for container sources.  The build is a
        # no-op when the cache is already current (see
        # LabeledDataCache.needs_rebuild), so re-calling it after a
        # norm change only rebuilds when extraction parameters changed.
        if source_type in (DataSourceType.ZARR, DataSourceType.CUTANA) and found_locations:
            if stale_check():
                logger.debug("Skipping cache build — newer job is queued")
                return
            try:
                BackendInterface.build_labeled_cache(
                    found_locations, label_path, source_type, stale_check=stale_check
                )
            except Exception as exc:
                logger.opt(exception=True).warning("Failed to build labeled cache: {}", exc)

    # ── Image info probe ─────────────────────────────────────────────

    def _probe_sample_image(self, folder: str, fname: str) -> None:
        """Load one image to extract resolution, channels, dtype, and value range."""
        filepath = fname if os.path.isabs(fname) else os.path.join(folder, fname)
        try:
            img = load_and_process_single_wrapper(
                filepath,
                self._cfg,
                desc="probe",
                show_progress=False,
            )
            if img.ndim == 3 and img.shape[0] <= 4:
                # CHW → HWC for display
                img = img.transpose(1, 2, 0)
            h, w = img.shape[:2]
            ch = img.shape[2] if img.ndim == 3 else 1
            self._sample_info = {
                "resolution": f"{w}\u00d7{h}",
                "channels": ch,
                "dtype": str(img.dtype),
                "range": f"[{np.nanmin(img):.3g}, {np.nanmax(img):.3g}]",
            }
        except Exception as exc:
            logger.debug(f"Image probe failed for {fname}: {exc}")
            self._sample_info = None

    # ── Phase 3: Load preview images ────────────────────────────────

    def _resolve_label_path(self) -> str | None:
        """Return the first candidate label CSV that resolves to a real file.

        Tries the chooser's current selection first, then ``cfg.label_file``.
        ``or`` short-circuits on the first *truthy* value, but the chooser
        can hold a non-empty selection that no longer points at an existing
        file (the CSV was deleted or moved after it was selected) — that
        used to make ``_read_label_map`` return ``{}`` even though
        ``cfg.label_file`` was a perfectly readable CSV, which silently
        broke the preview gallery and the label-vs-source check (#429
        follow-up).

        Returns:
            Path to the first existing label CSV among the candidates,
            or ``None`` if neither resolves.
        """
        for candidate in (self._label_chooser.selected, self._cfg.label_file):
            if candidate and os.path.isfile(candidate):
                return candidate
        return None

    def _read_label_map(self) -> dict[str, str]:
        """Read the selected label CSV and return an ``{id: label}`` mapping.

        Returns:
            Mapping from sample id to label string, or empty dict if neither
            the chooser nor the cfg points at an existing CSV.
        """
        label_path = self._resolve_label_path()
        if label_path is None:
            logger.debug(
                "No label path resolved (chooser={!r}, cfg={!r}); "
                "preview will show 'Select a label CSV…'",
                self._label_chooser.selected,
                self._cfg.label_file,
            )
            return {}
        return BackendInterface.read_label_map(label_path)

    def _phase_load_preview(self, folder: str, source_type: str, stale_check: callable) -> None:
        """Load preview images (runs on the background job thread).

        Always updates the preview grid on completion — either with images,
        a message, or an error.  Never leaves the spinner stuck.

        Args:
            folder: Path to the source data directory.
            source_type: One of ``"image_folder"``, ``"zarr"``, ``"cutana"``.
            stale_check: Callable returning True if this job is cancelled.
        """
        if not self._source_ok or not folder:
            self._preview_grid.clear()
            return

        # Normalisation overrides are already applied to cfg at the start
        # of _background_job — no need to redo it here.
        self._preview_grid.show_spinner("Loading preview...")

        labeled_map = self._read_label_map()

        # For Cutana, require labels to know which cutouts to preview
        if source_type == DataSourceType.CUTANA and not labeled_map:
            self._preview_grid.show_message("Select a label CSV to preview labeled cutouts")
            return

        # Select which items to preview — stratify by label so the grid
        # shows a mix of anomaly and normal examples instead of whatever
        # happens to come first in the catalogue (label distributions are
        # usually very skewed towards normal).
        image_files: list[str] | None = None
        source_ids: list[str] | None = None
        if source_type == DataSourceType.IMAGE_FOLDER:
            candidates = [f for f in self._image_files if os.path.basename(f) in labeled_map]
            image_files = _stratify_by_label(
                candidates, labeled_map, key=os.path.basename, max_items=PREVIEW_MAX_CELLS
            )
        elif source_type in (DataSourceType.CUTANA, DataSourceType.ZARR):
            candidates = self._found_source_ids or list(labeled_map.keys())
            source_ids = _stratify_by_label(
                candidates, labeled_map, key=str, max_items=PREVIEW_MAX_CELLS
            )

        # Cutana fast path: one cache read returns the decoded preview *and* the
        # raw (normalisation-independent) cutouts.  When the cache fully covers
        # the set we hold the raws (and the label map) so a later normalisation
        # change re-decodes them in-process — no cache read, no re-count, no
        # re-validation (#501).  ``load_cutana_preview`` returns ``None`` raws
        # for a partial/fallback preview, so we just don't hold in that case.
        if source_type == DataSourceType.CUTANA and source_ids:
            try:
                decoded, holdable = BackendInterface.load_cutana_preview(
                    folder, source_ids, max_items=PREVIEW_MAX_CELLS, stale_check=stale_check
                )
            except Exception as exc:
                logger.opt(exception=True).error("Preview loading failed: {}", exc)
                if not stale_check():
                    self._preview_grid.show_message(f"Preview failed: {exc}")
                return
            # Hold the raws even if this job was superseded mid-read: they are
            # normalisation-independent, so the superseding job (or the next
            # norm change) can re-decode them via the fast path instead of
            # re-reading the cache.  Gating this behind the stale check meant a
            # drag of rapid changes never persisted the hold — every job bailed
            # here before setting it, so each norm change fell back to a full
            # cache read (#501 follow-up).  ``holdable`` is ``None`` only on
            # genuine partial coverage (never staleness), so assigning it
            # unconditionally also correctly clears a stale hold.
            self._preview_raw_cutouts = holdable
            self._preview_labeled_map = labeled_map
            if stale_check():
                return
            self._apply_decoded_preview(decoded, labeled_map)
            return

        # General path: image folder or Zarr.  Cutouts here are decoded with
        # normalisation baked in (or read from disk), so there are no
        # norm-independent raws to hold — a later norm change goes through the
        # full reload.
        self._preview_raw_cutouts = None
        try:
            raw_samples = BackendInterface.load_preview_samples(
                folder,
                source_type,
                image_files=image_files,
                source_ids=source_ids,
                max_items=PREVIEW_MAX_CELLS,
                stale_check=stale_check,
            )
        except Exception as exc:
            logger.opt(exception=True).error("Preview loading failed: {}", exc)
            if not stale_check():
                self._preview_grid.show_message(f"Preview failed: {exc}")
            return

        if stale_check():
            return

        self._apply_decoded_preview(raw_samples, labeled_map)

    def _apply_decoded_preview(
        self, decoded: list[tuple[str, np.ndarray]], labeled_map: dict[str, str]
    ) -> None:
        """Populate the preview caches from decoded samples and render page 0.

        Shared by the full preview load and the #501 in-memory re-decode path.

        Args:
            decoded: ``(name, uint8_hwc)`` pairs in gallery order.
            labeled_map: ``{id: csv_label}`` used to render existing label
                badges on each cell.
        """
        # Cache decoded samples with their CSV labels so the gallery can
        # paginate over them without redecoding on every page turn.  The raw
        # HWC arrays are kept separately keyed by filename so the magnify
        # handler can hand them to the detail screen.
        self._preview_samples = [
            (name, numpy_array_to_byte_stream(img), labeled_map.get(name, ""))
            for name, img in decoded
        ]
        self._preview_raw_images = {name: img for name, img in decoded}

        if self._preview_samples:
            self._render_preview_page(0)
        else:
            self._preview_gallery.widget.layout.display = "none"
            self._preview_grid.show_message("No preview images could be loaded")

    def _on_norm_change(self) -> None:
        """Re-render the preview when normalisation settings actually change.

        ``ipywidgets`` observers fire per-widget, so a call to
        ``update_from_config`` (or widget construction that assigns several
        sliders in a row) would otherwise spawn a background job per
        field.  Compare the current norm config against the snapshot we
        took last time this fired; only start a new job if the dict
        genuinely differs.
        """
        if self._suppress_norm_preview:
            return
        current = self._norm_widget.get_normalisation_config()
        diff = diff_configs(self._last_norm_config, current)
        if not diff:
            return
        self._last_norm_config = serialisable_config(current)
        # Flag here, above the guards below: the snapshot assignment above has
        # already absorbed this edit, so any early return past this point loses
        # it for every later diff.  Under ``_norm_refresh_lock`` because
        # ``_run_norm_refresh`` reads and clears the flag under it from the timer
        # thread — an unlocked write can land between that read and clear and be
        # dropped, which routes the edit it belonged to back to the re-decode
        # path over stale raws.  ``_run_norm_refresh`` re-checks the source and
        # folder itself, so a flag set with nothing selected simply waits.
        extraction_keys = set(diff) & EXTRACTION_AFFECTING_NORM_FIELDS
        if extraction_keys:
            with self._norm_refresh_lock:
                self._norm_refresh_needs_extraction = True
        if not self._source_ok:
            return
        folder = self._source_chooser.selected or self._cfg.data_dir
        if not folder:
            return
        # Debounce: dragging the image-size slider fires a change per step, and
        # each refresh kicks off a multi-core decode.  Without coalescing, a
        # single drag spawned a pile of overlapping decodes that thrashed the pod
        # (the generation counter discards their *results* but not the CPU they
        # burn).  Schedule the refresh after a short quiet period so a drag
        # collapses into one decode of the settled values.
        logger.info(
            "Normalisation changed: {} — scheduling preview refresh (re-extract={})",
            diff,
            bool(extraction_keys),
        )
        self._schedule_norm_refresh()

    def _schedule_norm_refresh(self) -> None:
        """(Re)arm the debounce timer that refreshes the preview after norm edits."""
        with self._norm_refresh_lock:
            if self._norm_refresh_timer is not None:
                self._norm_refresh_timer.cancel()
            self._norm_refresh_timer = threading.Timer(
                _NORM_REFRESH_DEBOUNCE_SECONDS, self._run_norm_refresh
            )
            self._norm_refresh_timer.daemon = True
            self._norm_refresh_timer.start()

    def _run_norm_refresh(self) -> None:
        """Fire the debounced preview refresh once normalisation edits settle.

        Single-flight: only one refresh job runs at a time.  If one is already
        running, mark a re-run as pending and return — the in-flight job's
        completion callback re-runs once with the now-settled cfg (and, by then,
        held raws → the fast path).  Without this, a slider drag whose pauses
        exceed the debounce window started overlapping full cache reads before
        the first finished holding the raws, piling up concurrent NFS reads
        (#501 follow-up).

        Runs on the timer thread; ``_apply_norm_overrides_to_cfg`` (called inside
        the launched job) takes ``_norm_apply_lock`` so the cfg writes stay
        torn-write free even off the UI thread.
        """
        if not self._source_ok:
            return
        folder = self._source_chooser.selected or self._cfg.data_dir
        if not folder:
            return
        # Read *and* clear the extraction flag under the same lock the UI thread
        # sets it under.  Outside it, an edit landing between the two would have
        # its flag cleared without ever routing to the full job — silently
        # reinstating the bug this method exists to fix.
        with self._norm_refresh_lock:
            if self._job_in_flight:
                # A preview job (initial load, source/label change, or an earlier
                # refresh) is still running.  Defer: its completion re-runs us
                # once, by which point the raws are held → the fast path below.
                self._refresh_pending = True
                return
            needs_extraction = self._norm_refresh_needs_extraction
            self._norm_refresh_needs_extraction = False
        # Fast path: the current preview is fully backed by raw cutouts held in
        # memory, so re-decode them with the new normalisation in-process — no
        # cache read, no re-count, no re-validation (#501).  Only valid for
        # fields that change how raws are *rendered*; an extraction-affecting
        # edit invalidates them.  Anything else (image folder, Zarr, Cutana cache
        # miss, re-extraction needed) falls back to the full job.
        # Both ``_start_*`` set ``_job_in_flight``; the job's completion callback
        # (:meth:`_on_preview_job_done`) releases it and drains a pending refresh.
        if self._preview_raw_cutouts is not None and not needs_extraction:
            self._start_redecode_job()
        else:
            # Re-extraction needed (or nothing held): drop the stale raws so the
            # job re-reads them.  ``source_scanning.load_preview_raw_cutouts``
            # re-checks the extraction hash and returns [] when it moved, which
            # sends the read back to the catalogue with the new settings.
            self._preview_raw_cutouts = None
            self._start_background_job(folder, skip_counting=True)

    def _on_preview_job_done(self, generation: int) -> None:
        """Release the in-flight slot when a preview job ends; drain a pending refresh.

        Only the *latest* generation clears the slot — a superseded job
        finishing must not hand the slot to a norm refresh while the job that
        replaced it is still running.

        Args:
            generation: The finishing job's generation.
        """
        with self._norm_refresh_lock:
            if generation != self._job_generation:
                return
            self._job_in_flight = False
            pending = self._refresh_pending
            self._refresh_pending = False
        if pending:
            self._run_norm_refresh()

    def _start_redecode_job(self) -> None:
        """Re-decode the held preview cutouts after a normalisation change (#501).

        Reuses the generation counter so an in-flight decode (or a slower full
        job) is superseded, but skips counting, validation and the cache read —
        the raw cutouts are already in memory and normalisation-independent.
        """
        with self._norm_refresh_lock:
            self._job_generation += 1
            generation = self._job_generation
            self._job_in_flight = True
        # Sync the widget state into cfg on the UI thread before the worker
        # rebuilds fitsbolt_cfg from it (same torn-write guard as
        # ``_start_background_job``).
        self._apply_norm_overrides_to_cfg()
        self._preview_grid.show_spinner("Applying normalisation...")
        threading.Thread(
            target=self._redecode_job,
            args=(generation,),
            daemon=True,
            name=f"setup-redecode-{generation}",
        ).start()

    def _redecode_job(self, generation: int) -> None:
        """Background worker: re-decode held raw cutouts and re-render.

        Args:
            generation: Job generation for cancellation; bails out if a newer
                job (full or re-decode) has since been requested.
        """

        def _stale() -> bool:
            return self._job_generation != generation

        try:
            raw_cutouts = self._preview_raw_cutouts
            if not raw_cutouts or _stale():
                return
            decoded = BackendInterface.decode_raw_cutouts(raw_cutouts)
            if _stale():
                return
            # Reuse the label map captured when the raws were held — it can't
            # change during a norm tweak, so this avoids an NFS CSV re-read (#501
            # review).
            self._apply_decoded_preview(decoded, self._preview_labeled_map)
        except Exception as exc:
            logger.opt(exception=True).error("Preview re-decode gen={} failed: {}", generation, exc)
            if not _stale():
                self._preview_grid.show_message(f"Preview failed: {exc}")
        finally:
            self._on_preview_job_done(generation)

    # ── Validation display ───────────────────────────────────────────

    def _render_validation(self) -> str:
        lines: list[str] = []

        # Source folder
        if self._counting:
            lines.append(_status_line(_SPINNER_INLINE, self._counting_status))
        elif self._source_ok and self._source_count is not None:
            source_path = self._source_chooser.selected or self._cfg.data_dir or ""
            short = os.path.basename(source_path.rstrip(os.sep)) if source_path else ""
            stype = self._detected_source_type
            type_label = {
                DataSourceType.IMAGE_FOLDER: "images",
                DataSourceType.ZARR: "images (Zarr)",
                DataSourceType.CUTANA: "sources (Cutana catalogue)",
            }.get(stype, "sources")
            lines.append(
                _status_line(_CHECK, f"Found {self._source_count:,} {type_label} in {short}")
            )
            info = getattr(self, "_sample_info", None)
            if info:
                details = (
                    f"{info['resolution']} px, {info['channels']} ch,"
                    f" {info['dtype']}, values {info['range']}"
                )
                lines.append(_status_line("&nbsp;&nbsp;", details))
        else:
            msg = "Select a source folder (images, Zarr, or catalogue)"
            if self._source_count == 0:
                msg = "No images or catalogues found in selected folder"
            lines.append(_status_line(_CROSS, msg))

        # Label CSV
        if self._labels_ok:
            label_path = self._label_chooser.selected or self._cfg.label_file or ""
            short = os.path.basename(label_path) if label_path else ""
            lines.append(_status_line(_CHECK, f"Labels: {short} ({self._label_message})"))
            # Label-vs-source validation
            if self._label_validation_message:
                has_missing = "missing" in self._label_validation_message
                icon = _CROSS if has_missing else _CHECK
                lines.append(_status_line(icon, self._label_validation_message))
        else:
            if self._label_message:
                lines.append(_status_line(_CROSS, f"Labels: {self._label_message}"))
            else:
                lines.append(_status_line(_CROSS, "Select a labelled_data.csv file"))

        # Optional metadata.  No cfg fallback: ``initial_file`` pre-selects a
        # configured metadata file, so the chooser already holds it, and
        # ``_on_start`` writes only what the chooser holds.  Reporting a cfg
        # value the widget does not show would put the panel back out of step
        # with the selection.
        meta_path = self._meta_chooser.selected
        if meta_path and os.path.isfile(meta_path):
            short = os.path.basename(meta_path)
            lines.append(_status_line(_INFO, f"Metadata: {short}"))

        return (
            f'<div style="background:{ESA_BLUE_DEEP}; border-radius:6px;'
            f' padding:10px 14px; margin:4px 0;">' + "".join(lines) + "</div>"
        )

    def _refresh_validation(self) -> None:
        self._validation_html.value = self._render_validation()
        all_ok = self._source_ok and self._labels_ok
        self._start_btn.disabled = not all_ok
        # Size stratification reads the per-source diameter from the Cutana
        # catalogue, so it only applies to Cutana sources — image/Zarr sources
        # carry no size column.  Enable the checkbox only when one is detected.
        self._stratify_checkbox.disabled = self._detected_source_type != DataSourceType.CUTANA

    # ── Start action ─────────────────────────────────────────────────

    def _on_start(self) -> None:
        """Write validated config and navigate to training screen."""
        try:
            source = self._source_chooser.selected or self._cfg.data_dir
            label = self._label_chooser.selected or self._cfg.label_file

            self._cfg.data_dir = os.path.normpath(source)
            self._cfg.label_file = os.path.normpath(label)

            meta = self._meta_chooser.selected
            if meta:
                self._cfg.metadata_file = os.path.normpath(meta)

            # Apply normalisation overrides to shared cfg
            self._apply_norm_overrides_to_cfg()

            # Remember them for the next session.  Starting a run is the point
            # the user commits to these settings — persisting on every widget
            # edit would rewrite the state file on each step of a slider drag.
            ui_state.set_normalisation_settings(
                _NORM_STATE_PURPOSE,
                serialisable_config(self._norm_widget.get_normalisation_config()),
                self._norm_widget.extensions,
            )

            # Push slider value into cfg so the training screen's auto-start
            # picks up the user's choice before the first run (and its own
            # slider initialises to the same value).
            self._cfg.num_train_iter = self._iter_slider.value

            # Size stratification is Cutana-only.  Gate on the detected source
            # type (the source of truth) rather than the checkbox's disabled
            # state, so a stale True from a prior Cutana selection can't leak into
            # an image/Zarr run regardless of widget-refresh ordering.
            self._cfg.cutana_stratify_source_size = (
                self._stratify_checkbox.value
                and self._detected_source_type == DataSourceType.CUTANA
            )

            # Merge any setup-time label edits into a fresh CSV before
            # navigating, so the training subprocess sees the up-to-date
            # set on its first iteration (#427).  ``merge_gallery_labels``
            # writes a timestamped sibling file and returns its path; we
            # point the cfg at that path so training picks it up.
            if self._setup_labels:
                # An UNLABELLED edit removes a label the user no longer wants,
                # so it becomes a ``removed`` row (excluded from the labeled set
                # at dataset build).  ``merge_gallery_labels`` drops a ``removed``
                # row whose id never carried a label, so mapping every UNLABELLED
                # here adds no noise and keeps that policy in the backend.
                gallery_csv_labels: dict[str, str] = {}
                for fn, state in self._setup_labels.items():
                    if state == LabelState.UNLABELLED:
                        gallery_csv_labels[fn] = LABEL_REMOVED
                    elif state in _LABEL_STATE_TO_CSV:
                        gallery_csv_labels[fn] = _LABEL_STATE_TO_CSV[state]
                if gallery_csv_labels:
                    try:
                        merged_path = BackendInterface.merge_gallery_labels(gallery_csv_labels)
                        self._cfg.label_file = merged_path
                        logger.info(
                            "Merged {} setup-screen label edit(s) into {}",
                            len(gallery_csv_labels),
                            merged_path,
                        )
                    except Exception as exc:
                        logger.warning("Could not merge setup-screen labels: {}", exc)

            logger.info(f"Source dir: {self._cfg.data_dir}")
            logger.info(f"Label file: {self._cfg.label_file}")
            logger.info(f"Training iterations: {self._cfg.num_train_iter}")

            self.navigate_to("training")
        except Exception as exc:
            logger.opt(exception=True).error("Failed to navigate to training: {}", exc)

    # ── Preview gallery integration ──────────────────────────────────

    def _render_preview_page(self, page: int) -> None:
        """Render *page* of the cached preview samples in the gallery.

        Pagination walks at the gallery's *effective* page size, not the
        static :data:`PAGE_SIZE` — when the user widens the slider, the
        two-row viewport holds fewer cells, so we re-page at the smaller
        chunk size to avoid silently skipping the cells the slider just
        clipped past row 2 (#432).

        Args:
            page: Zero-based page index.
        """
        page_size = self._preview_gallery.effective_page_size
        total = len(self._preview_samples)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = max(0, min(page, total_pages - 1))
        start = page * page_size
        end = min(start + page_size, total)
        page_slice = self._preview_samples[start:end]
        self._preview_first_offset = start

        # Zarr ids may carry a pre-conversion filename extension (e.g.
        # ".jpeg") from whatever format the cutouts were originally
        # ingested from — the store itself holds decoded pixel arrays, so
        # showing that extension in the gallery misleadingly implies the
        # file still exists in that format. Strip it for display only;
        # ``name`` (used for labeling and the magnify lookup) is untouched.
        strip_extension = self._detected_source_type == DataSourceType.ZARR
        results = [
            {
                "filename": name,
                "score": 0.0,
                "display": _strip_image_extension(name) if strip_extension else name,
            }
            for name, _, _ in page_slice
        ]
        bytes_by_name = {name: png for name, png, _ in page_slice}

        def _image_loader(filename: str) -> bytes:
            return bytes_by_name.get(filename, b"")

        # Hide the spinner / message panel and show the gallery.
        self._preview_grid.clear()
        self._preview_gallery.widget.layout.display = ""

        self._preview_gallery.update_page(
            results=results,
            page=page,
            total_pages=total_pages,
            total_count=total,
            image_loader=_image_loader,
        )

        # Round-trip existing CSV labels + setup-time edits into the
        # cells so the user sees the current label state even after
        # paging away and back.  A setup-time edit (membership in
        # ``_setup_labels``) wins over the CSV badge; the membership test
        # (not ``.get(...) or csv``) is what makes a recorded UNLABELLED
        # stick — ``_on_setup_label_change`` no longer drops it, so it must
        # take precedence over the cell's original CSV label here.
        for cell, (name, _png, csv_lbl) in zip(self._preview_gallery._cells, page_slice):
            if name in self._setup_labels:
                state = self._setup_labels[name]
            else:
                state = _CSV_TO_LABEL_STATE.get(csv_lbl, LabelState.UNLABELLED)
            if isinstance(cell, LabelableThumbnailCell):
                cell.set_label(state)

    def _on_preview_page_change(self, page: int) -> None:
        """Gallery page-change callback — re-render with the new page index."""
        self._render_preview_page(page)

    def _on_preview_scale_change(self) -> None:
        """Slider-change callback — re-page at the new effective page size.

        Anchored on :attr:`_preview_first_offset` so the user lands on
        the page that still contains the cell they were looking at,
        rather than the same numeric ``_current_page`` under a stale
        chunk size.
        """
        page_size = self._preview_gallery.effective_page_size
        new_page = self._preview_first_offset // page_size if page_size else 0
        self._render_preview_page(new_page)

    def _on_setup_label_change(self, filename: str, state: LabelState) -> None:
        """Record a label edit made on the setup-screen gallery.

        Edits live in :attr:`_setup_labels` until ``Start Training`` is
        clicked, at which point they're merged into a fresh CSV via
        ``BackendInterface.merge_gallery_labels``.  Writing through to
        the chosen ``labelled_data.csv`` immediately would force the
        chooser to re-point at a renamed file mid-edit, which breaks
        the user's mental model — the merge happens once on Start.

        An explicit ``UNLABELLED`` is recorded too (not popped): the map
        is the authority for the displayed state of every touched cell,
        so a deliberate un-label must override the cell's CSV badge on
        re-render.  Popping it instead let the original CSV label
        re-appear when the user paged away and back.
        """
        self._setup_labels[filename] = state

    def _on_setup_magnify(self, filename: str) -> None:
        """Open the image detail screen for a setup-gallery cutout.

        The detail screen reads ``_detail_context`` for the image to
        display and the back-screen target.  Setup-time magnify uses
        the raw arrays we already decoded into :attr:`_preview_raw_images`
        so no extra I/O is needed.
        """
        image = self._preview_raw_images.get(filename)
        if image is None:
            logger.warning("No raw image cached for {} — magnify ignored", filename)
            return
        self.app._detail_context = {
            "filename": filename,
            "score": 0.0,
            "image": image,
            "back_screen": "training_setup",
            # Re-decode hook in :class:`ImageDetailScreen` needs the
            # source type to pick the right loader path (#443).
            "source_type": self._detected_source_type,
            # The chooser's current selection is the only source-of-truth
            # for "where the catalogues live" until the user clicks Start
            # — ``cfg.data_dir`` only updates in ``_on_start``, so the
            # backend would otherwise fall back to a stale notebook
            # default.
            "search_dir": self._source_chooser.selected or self._cfg.data_dir,
        }
        self.navigate_to("image_detail")
