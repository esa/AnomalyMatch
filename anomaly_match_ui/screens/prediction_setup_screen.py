#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Prediction setup screen — choose model, search folder, and output folder."""

from __future__ import annotations

import os
import random
import threading
from typing import TYPE_CHECKING

import ipywidgets as widgets
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from ipywidgets import HBox, VBox
from loguru import logger

from anomaly_match.data_io.find_images_in_folder import get_image_names_from_folder
from anomaly_match.data_io.load_images import load_and_process_single_wrapper
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.image_processing.image_utils import ensure_uint8_hwc
from anomaly_match.prediction import SchemaVersionError
from anomaly_match.prediction.anomaly_score_db import compute_model_sha256
from anomaly_match.prediction.cutana_loader import load_sample_cutouts
from anomaly_match_ui.screens.base_screen import BaseScreen
from anomaly_match_ui.styles import (
    BG_COLOR,
    ESA_BLUE_BRIGHT,
    ESA_BLUE_DEEP,
    ESA_GREEN,
    ESA_RED,
    inline_spinner_html,
)
from anomaly_match_ui.utils.backend_interface import BackendInterface
from anomaly_match_ui.utils.chooser_paths import initial_dir, initial_file, resolved_dir
from anomaly_match_ui.utils.image_utils import numpy_array_to_byte_stream
from anomaly_match_ui.utils.scrolling_log_sink import ScreenLogSink
from anomaly_match_ui.widgets.file_chooser import AMFileChooser
from anomaly_match_ui.widgets.header_widget import HeaderWidget
from anomaly_match_ui.widgets.preview_grid import PreviewGrid

if TYPE_CHECKING:
    from anomaly_match_ui.app import AnomalyMatchApp

_CHECK = f'<span style="color:{ESA_GREEN};">&#10003;</span>'
_CROSS = f'<span style="color:{ESA_RED};">&#10007;</span>'
_SPINNER_INLINE = inline_spinner_html(margin="0 6px 0 0")

_INFO = f'<span style="color:{ESA_BLUE_BRIGHT};">&#8505;</span>'
_WARN = '<span style="color:orange;">&#9888;</span>'

_FILE_TYPE_LABELS = {
    DataSourceType.IMAGE_FOLDER: "images",
    DataSourceType.CUTANA: "catalogue sources",
    DataSourceType.ZARR: "Zarr stores",
}


def _status_line(icon: str, text: str) -> str:
    return f'<div style="color:white; font-size:13px; padding:2px 0;">{icon} {text}</div>'


class PredictionSetupScreen(BaseScreen):
    """Setup screen for configuring a prediction run.

    Provides file choosers for the model checkpoint, search folder,
    and output folder.  Validates selections and counts source images
    in a background thread before enabling the start button.

    Args:
        app: The parent application instance used for navigation.
    """

    def __init__(self, app: AnomalyMatchApp) -> None:
        super().__init__(app)
        self._cfg = BackendInterface.get_config()
        self._model_ok = False
        self._search_ok = False
        self._output_ok = False
        self._source_count: int | None = None
        self._search_type: DataSourceType | None = None
        self._counting = False
        self._counting_status: str = "Counting sources..."
        self._resume_count: int | None = None
        self._resume_compatible: bool | None = None
        self._resume_message: str = ""
        self._image_files: list[str] = []
        self._cutana_preview_error: str = ""
        # Normalisation summary read from the selected model checkpoint.  The
        # model is the source of truth — prediction overrides its decode
        # pipeline from the checkpoint — so the screen shows this read-only
        # rather than letting the user pick settings that could desync the run.
        self._model_norm: dict | None = None
        # Set when reading the selected model's metadata raised, so the overview
        # can say "model unreadable" rather than "select a model".
        self._model_norm_error: str | None = None
        self._log_sink = ScreenLogSink("PredictionSetupScreen")

    # ── Build ────────────────────────────────────────────────────────

    def build(self) -> widgets.Widget:
        """Build and return the prediction setup layout.

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

        # ── Model chooser ────────────────────────────────────────
        model_dir, model_name = initial_file(
            self._cfg.model_path,
            purpose="model_chooser",
            is_shipped_default=BackendInterface.is_shipped_default_path(
                "model_path", self._cfg.model_path
            ),
        )
        self._model_chooser = AMFileChooser(
            path=model_dir,
            filename=model_name,
            select_default=bool(model_name),
            filter_pattern="*.safetensors",
            title="Model checkpoint",
            purpose="model_chooser",
        )
        self._model_chooser.register_callback(self._on_model_change)

        # ── Search folder chooser ────────────────────────────────
        # Resolve once so auto-validation counts the same dir the
        # chooser displays — see TrainingSetupScreen.build for the
        # rationale (#429 follow-up).
        self._search_initial_path = initial_dir(
            self._cfg.prediction_search_dir, purpose="search_chooser"
        )
        search_has_default = bool(
            self._search_initial_path and os.path.isdir(self._search_initial_path)
        )
        self._search_chooser = AMFileChooser(
            path=self._search_initial_path,
            show_only_dirs=True,
            select_default=search_has_default,
            title="Search folder (images to scan)",
            purpose="search_chooser",
        )
        self._search_chooser.register_callback(self._on_search_change)

        # ── Output folder chooser ────────────────────────────────
        # Decide on the same path the chooser displays, as the search chooser
        # above does: deriving the flag from cfg while showing the resolved dir
        # leaves Start disabled next to a perfectly valid folder.  Only a real
        # resolution may be pre-selected: the cwd fallback is merely where the
        # chooser opens, and selecting it would make ``_on_start`` write the
        # cwd over the configured ``cfg.output_dir``.
        self._output_resolved_path = resolved_dir(self._cfg.output_dir, purpose="output_chooser")
        self._output_chooser = AMFileChooser(
            path=self._output_resolved_path or os.getcwd(),
            show_only_dirs=True,
            select_default=self._output_resolved_path is not None,
            title="Output folder (results destination)",
            purpose="output_chooser",
        )
        self._output_chooser.register_callback(self._on_output_change)

        # ── Preview grid ──────────────────────────────────────────
        self._preview_grid = PreviewGrid()

        # ── Normalisation overview (read-only) ────────────────────
        # Prediction must reuse the model's training normalisation (the
        # subprocess overrides its decode pipeline from the checkpoint), so
        # this is a read-only summary, not an editable control.  Populated
        # when a model is selected.
        self._norm_overview = widgets.HTML(value=self._render_norm_overview())

        # ── Validation panel + start button ──────────────────────
        self._validation_html = widgets.HTML(
            value=self._render_validation(),
            layout=widgets.Layout(padding="8px 0"),
        )

        self._start_btn = widgets.Button(
            description="Start Prediction",
            button_style="success",
            disabled=True,
            layout=widgets.Layout(width="200px", height="40px"),
            style={"font_size": "14px"},
        )
        self._start_btn.on_click(lambda _: self._on_start())

        # ── Layout ───────────────────────────────────────────────
        # Left: file choosers (constrained so preview gets space)
        left_panel = VBox(
            [
                self._model_chooser,
                self._search_chooser,
                self._output_chooser,
            ],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="12px",
                gap="8px",
                flex="0 1 auto",
                max_width="420px",
            ),
        )

        # Centre: image preview (flex so it grows with window width)
        centre_panel = VBox(
            [self._preview_grid.widget],
            layout=widgets.Layout(
                background_color=BG_COLOR,
                padding="8px",
                flex="1 1 auto",
                min_width="140px",
                border_left=f"1px solid {ESA_BLUE_DEEP}",
                border_right=f"1px solid {ESA_BLUE_DEEP}",
            ),
        )

        # Right: normalisation overview + validation + start
        right_panel = VBox(
            [
                self._norm_overview,
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
            layout=widgets.Layout(background_color=BG_COLOR),
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

    # ── Lifecycle ────────────────────────────────────────────────

    def on_enter(self) -> None:
        """Set up terminal output and auto-validate from config."""
        BackendInterface.set_terminal_output(self._out)
        self._log_sink.attach(self._out)
        self._auto_validate_from_config()

    def on_leave(self) -> None:
        """Remove the widget sink when navigating away."""
        self._log_sink.detach()

    # ── Auto-validate ────────────────────────────────────────────

    def _auto_validate_from_config(self) -> None:
        """Pre-validate fields that already have values in cfg."""
        if self._cfg.model_path and os.path.isfile(self._cfg.model_path):
            self._model_ok = self._cfg.model_path.endswith(".safetensors")
            self._apply_model_normalisation()
            self._refresh_validation()

        # Count the chooser's initial path, not ``cfg.prediction_search_dir``
        # directly — persistence may have overridden cfg, in which case
        # cfg points at one folder but the chooser shows another (#429
        # follow-up).
        if self._search_initial_path and os.path.isdir(self._search_initial_path):
            self._counting = True
            self._refresh_validation()
            threading.Thread(
                target=self._count_sources,
                args=(self._search_initial_path,),
                daemon=True,
            ).start()

        # A persisted dir counts even without a cfg value, since the chooser
        # shows it selected.  A configured ``output_dir`` that does not exist
        # yet also counts: ``_on_start`` falls back to it and the run creates it.
        if self._output_resolved_path is not None or self._cfg.output_dir:
            self._output_ok = True
            self._refresh_validation()
            self._check_resume()

    # ── File chooser callbacks ───────────────────────────────────

    def _on_model_change(self, _chooser: object) -> None:
        path = self._model_chooser.selected
        self._model_ok = bool(path and os.path.isfile(path) and path.endswith(".safetensors"))
        self._apply_model_normalisation()
        self._refresh_validation()
        self._check_resume()

    def _on_search_change(self, _chooser: object) -> None:
        path = self._search_chooser.selected
        if path and os.path.isdir(path):
            self._search_ok = False
            self._source_count = None
            self._counting = True
            self._image_files = []
            self._preview_grid.clear()
            self._refresh_validation()
            threading.Thread(target=self._count_sources, args=(path,), daemon=True).start()
        else:
            self._search_ok = False
            self._source_count = None
            self._counting = False
            self._image_files = []
            self._preview_grid.clear()
            self._refresh_validation()

    def _on_output_change(self, _chooser: object) -> None:
        path = self._output_chooser.selected
        self._output_ok = bool(path)
        self._refresh_validation()
        self._check_resume()

    # ── Source counting (background thread) ──────────────────────

    def _count_sources(self, folder: str) -> None:
        """Count sources in *folder* on a background thread.

        For image-type sources, also collects file paths for the
        preview gallery.
        """
        self._counting_status = "Detecting source type..."
        self._refresh_validation()

        try:
            logger.info(f"Scanning {folder}...")
            file_type, count = BackendInterface.detect_and_count_prediction_sources(folder)
            self._search_type = file_type
            logger.info(
                f"Detected {count:,} {_FILE_TYPE_LABELS.get(file_type, 'sources')} ({file_type.value})"
            )

            # Collect image file paths for preview
            if file_type == DataSourceType.IMAGE_FOLDER and count > 0:
                self._counting_status = "Collecting image file paths..."
                self._refresh_validation()

                self._image_files = get_image_names_from_folder(folder, recursive=True)
        except Exception as exc:
            logger.warning(f"Source count failed: {exc}")
            count = 0
            self._search_type = None

        self._source_count = count
        self._search_ok = count > 0
        self._counting = False
        self._refresh_validation()

        # Render preview gallery for all source types
        if self._search_ok:
            self._refresh_preview()
        else:
            self._preview_grid.clear()

    # ── Preview grid ──────────────────────────────────────────────

    def _refresh_preview(self) -> None:
        """Load sample images for all source types.

        Decoding uses ``cfg.normalisation``, which is mirrored from the
        selected model (:meth:`_apply_model_normalisation`) — so the preview
        reflects the same settings inference will run with.
        """
        folder = self._search_chooser.selected or self._cfg.prediction_search_dir
        if not self._search_ok or not folder:
            self._preview_grid.clear()
            return

        self._preview_grid.show_spinner("Loading preview...")
        threading.Thread(
            target=self._load_preview,
            args=(folder, self._search_type),
            daemon=True,
        ).start()

    def _load_preview(self, folder: str, source_type: DataSourceType | None) -> None:
        """Load preview images on a background thread for any source type."""
        items: list[tuple[str, bytes]] = []
        try:
            if source_type == DataSourceType.IMAGE_FOLDER and self._image_files:
                sample = self._image_files[:9]
                if len(self._image_files) > 9:
                    sample = random.sample(self._image_files, 9)
                for fname in sample:
                    filepath = fname if os.path.isabs(fname) else os.path.join(folder, fname)
                    try:
                        img = load_and_process_single_wrapper(
                            filepath, self._cfg, desc="preview", show_progress=False
                        )
                        if img.ndim == 3 and img.shape[0] <= 4:
                            img = img.transpose(1, 2, 0)
                        items.append((os.path.basename(fname), numpy_array_to_byte_stream(img)))
                    except Exception as exc:
                        logger.debug(f"Preview load failed for {fname}: {exc}")

            elif source_type == DataSourceType.CUTANA:
                samples = load_sample_cutouts(self._cfg, folder, n_samples=9)
                for source_id, image in samples:
                    img = ensure_uint8_hwc(image)
                    short = source_id if len(source_id) <= 18 else source_id[:15] + "\u2026"
                    items.append((short, numpy_array_to_byte_stream(img)))

            elif source_type == DataSourceType.ZARR:
                # The training-setup decoder, so normalisation and
                # channel_combination apply as at inference (#621).
                samples = BackendInterface.load_preview_samples(
                    folder, DataSourceType.ZARR, max_items=9
                )
                for name, image in samples:
                    items.append((name, numpy_array_to_byte_stream(image)))

        except Exception as exc:
            logger.info(f"Preview failed: {exc}")

        if items:
            self._preview_grid.update(items)
        else:
            self._preview_grid.show_message("No preview images could be loaded")

    # ── Normalisation overview (read-only, model-sourced) ─────────

    def _apply_model_normalisation(self) -> None:
        """Mirror the selected model's embedded normalisation onto cfg.

        The prediction subprocess decodes images with the model's embedded
        fitsbolt config (see ``load_model``), so the model — not a UI choice —
        is the source of truth.  Mirroring ``image_size`` / method / channels
        onto ``cfg.normalisation`` keeps batch sizing, the preview and the DB
        run metadata consistent with what inference actually runs, and feeds
        the read-only overview.
        """
        self._model_norm = None
        self._model_norm_error = None
        model_path = self._model_chooser.selected or self._cfg.model_path
        if not (model_path and os.path.isfile(model_path) and model_path.endswith(".safetensors")):
            self._refresh_norm_overview()
            return
        try:
            summary = BackendInterface.read_model_normalisation(model_path)
        except Exception as exc:
            logger.warning("Could not read normalisation from model {}: {}", model_path, exc)
            self._model_norm_error = str(exc)
            self._refresh_norm_overview()
            return

        self._model_norm = summary
        # A checkpoint without embedded normalisation can't be mirrored (and
        # prediction would reject it anyway); surface that in the overview
        # rather than clobbering cfg with Nones.
        if summary["image_size"] is not None:
            # Sync all three normalisation surfaces (fitsbolt_cfg,
            # cfg.normalisation fields, channel_combination) from the checkpoint
            # — the same write the prediction subprocess does.  Mirroring only
            # the cfg.normalisation fields left cfg.fitsbolt_cfg stale at the
            # pickled default (CONVERSION_ONLY), and the Cutana preview's
            # fitsbolt/normalisation agreement guard then crashed the setup
            # screen on resume (the ASINH-vs-CONVERSION_ONLY ValueError).
            #
            # The sync can itself raise on a corrupt checkpoint (e.g. a
            # channel_combination whose row count disagrees with
            # n_output_channels).  Surface that through the same
            # _model_norm_error path the header read above uses rather than
            # letting it escape the model-chooser observe callback and leave
            # the overview stale.
            try:
                BackendInterface.sync_normalisation_from_model(model_path)
            except Exception as exc:
                logger.warning("Could not sync normalisation from model {}: {}", model_path, exc)
                self._model_norm_error = str(exc)
                self._refresh_norm_overview()
                return
            if self._search_ok:
                self._refresh_preview()
        self._refresh_norm_overview()

    def _render_norm_overview(self) -> str:
        """Render the read-only normalisation summary HTML.

        Sourced from the selected model checkpoint — there is nothing for the
        user to set, since mismatched settings would only desync the run from
        the model it must reuse.
        """
        norm = self._model_norm
        if self._model_norm_error is not None:
            body = _status_line(
                _WARN, f"Could not read model normalisation: {self._model_norm_error}"
            )
        elif norm is None:
            body = _status_line(_INFO, "Select a model to view its normalisation settings")
        elif norm["image_size"] is None:
            body = _status_line(
                _WARN,
                "Model has no embedded normalisation metadata — retrain to enable prediction",
            )
        else:
            # image_size is not None here, so read_model_normalisation decoded a
            # real fitsbolt config and the method is a NormalisationMethod.
            method_name = norm["normalisation_method"].name
            size = norm["image_size"]
            body = "".join(
                [
                    _status_line(_CHECK, f"Resolution: {size[0]}×{size[1]} px"),
                    _status_line(_CHECK, f"Method: {method_name}"),
                    _status_line(_CHECK, f"Output channels: {norm['n_output_channels']}"),
                ]
            )
        return (
            f'<div style="background:{ESA_BLUE_DEEP}; border-radius:6px;'
            f' padding:10px 14px; margin:4px 0;">'
            f'<div style="color:{ESA_BLUE_BRIGHT}; font-size:14px; font-weight:bold;'
            f' padding-bottom:4px;">Normalisation (from model)</div>' + body + "</div>"
        )

    def _refresh_norm_overview(self) -> None:
        self._norm_overview.value = self._render_norm_overview()

    # ── Resume detection ─────────────────────────────────────────

    def _check_resume(self) -> None:
        """Check for an existing predictions.db and reconcile settings.

        When the output folder points at a DB produced by an earlier
        run, its ``run_metadata`` is the source of truth for the
        accumulated scores.  Rather than reject a setting mismatch
        (the prior behaviour), we *apply* the stored normalisation
        onto cfg — the user resumes with the exact settings the DB
        needs.  These match the model's own normalisation (shown in the
        read-only overview), since a resume requires the model sha256 to
        match the one that produced the DB.

        The one thing we cannot auto-reconcile is the model checkpoint:
        a mismatched sha256 means the scores in the DB are from a
        different model entirely, and the user has to pick the right
        one themselves.  Schema-version drift is also a hard error —
        the DB must be deleted and re-run.
        """
        self._resume_count = None
        self._resume_compatible = None
        self._resume_message = ""

        output_path = self._output_chooser.selected or self._cfg.output_dir
        if not output_path:
            self._refresh_validation()
            return

        db_path = os.path.join(output_path, "predictions.db")
        if not os.path.isfile(db_path):
            self._refresh_validation()
            return

        try:
            count, stored = BackendInterface.load_prediction_db_state(db_path)
        except SchemaVersionError as exc:
            self._resume_count = 0
            self._resume_compatible = False
            self._resume_message = str(exc)
            self._refresh_validation()
            return
        except Exception as exc:
            # Re-running every chooser change, so a transient read error
            # (UI polling thread mid-checkpoint, unreadable file, etc.)
            # must not crash the widget.  Surface a short message so the
            # user knows resume info is unavailable; details go to the log.
            logger.warning("Failed to read prediction DB at {}: {}", db_path, exc)
            self._resume_compatible = False
            self._resume_message = f"Could not read existing DB: {exc}"
            self._refresh_validation()
            return

        if count == 0 or not stored:
            self._refresh_validation()
            return

        # Model identity: cannot auto-reconcile.  If the user selected
        # a model whose bytes don't match the DB's, block and explain.
        # ``model_sha256`` and ``model_path`` are written by every PR-420+
        # subprocess; direct dict access is a load-bearing assertion that
        # the DB has the schema we expect.
        model_path = self._model_chooser.selected or self._cfg.model_path
        if model_path and os.path.isfile(model_path):
            current_sha = compute_model_sha256(model_path)
            if current_sha != stored["model_sha256"]:
                self._resume_count = count
                self._resume_compatible = False
                self._resume_message = (
                    f"DB was created with a different model ({stored['model_path']})."
                )
                self._refresh_validation()
                return

        # Everything else the DB records (image_size, normalisation_method,
        # n_output_channels) is authoritative — apply to the widget/cfg
        # so the user's next click starts a matching resume.
        applied = self._apply_db_settings(stored)
        self._resume_count = count
        self._resume_compatible = True
        self._resume_message = applied
        self._refresh_validation()

    def _apply_db_settings(self, stored: dict) -> str:
        """Mirror stored normalisation settings onto cfg.

        Args:
            stored: Metadata dict as returned by
                :meth:`AnomalyScoreDB.get_metadata`.  Must contain
                ``image_size``, ``normalisation_method`` and
                ``n_output_channels`` — guaranteed by the schema written
                in :func:`build_compat_metadata`.

        Returns:
            A human-readable summary of what was applied.
        """
        size = stored["image_size"]
        # NormalisationMethod is stored via ``str(NormalisationMethod)``,
        # which yields the integer enum value as a string ("2") — decode
        # back to the enum.  A corrupt DB raises here instead of silently
        # falling through to a wrong default.
        method = NormalisationMethod(int(stored["normalisation_method"]))
        n = stored["n_output_channels"]

        self._cfg.normalisation.image_size = size
        self._cfg.normalisation.normalisation_method = method
        self._cfg.normalisation.n_output_channels = n

        return f"image_size={size}, method={method.name}, channels={n}"

    # ── Validation display ───────────────────────────────────────

    def _render_validation(self) -> str:
        lines: list[str] = []

        # Model
        if self._model_ok:
            model_path = self._model_chooser.selected or self._cfg.model_path or ""
            short_model = os.path.basename(model_path) if model_path else ""
            lines.append(_status_line(_CHECK, f"Model: {short_model}"))
        else:
            lines.append(_status_line(_CROSS, "Select a model checkpoint (.safetensors)"))

        # Search folder
        if self._counting:
            lines.append(_status_line(_SPINNER_INLINE, self._counting_status))
        elif self._search_ok and self._source_count is not None:
            type_label = _FILE_TYPE_LABELS.get(self._search_type, "sources")
            search_path = self._search_chooser.selected or self._cfg.prediction_search_dir or ""
            short_search = os.path.basename(search_path.rstrip(os.sep)) if search_path else ""
            lines.append(
                _status_line(_CHECK, f"Found {self._source_count:,} {type_label} in {short_search}")
            )
        else:
            msg = "Select a folder containing images or catalogues"
            if self._source_count == 0:
                msg = "No supported files found in selected folder"
            lines.append(_status_line(_CROSS, msg))

        # Output folder
        if self._output_ok:
            output_path = self._output_chooser.selected or self._cfg.output_dir or ""
            short_output = os.path.basename(output_path.rstrip(os.sep)) if output_path else ""
            lines.append(_status_line(_CHECK, f"Output: {short_output}"))
        else:
            lines.append(_status_line(_CROSS, "Select an output folder"))

        # Source type info for non-image sources
        if (
            self._search_ok
            and self._search_type
            and self._search_type != DataSourceType.IMAGE_FOLDER
            and not self._counting
        ):
            type_notes = {
                DataSourceType.CUTANA: "Cutouts will be created during prediction via Cutana",
                DataSourceType.ZARR: "Images will be read from Zarr stores during prediction",
            }
            note = type_notes.get(self._search_type, "")
            if note:
                lines.append(_status_line(_INFO, note))

        # Cutana preview error
        if self._cutana_preview_error:
            lines.append(_status_line(_WARN, self._cutana_preview_error))

        # Resume detection
        if self._resume_count is not None and self._resume_count > 0:
            if self._resume_compatible:
                msg = f"Resume from {self._resume_count:,} existing results"
                if self._resume_message:
                    msg += f" — settings loaded from DB ({self._resume_message})"
                lines.append(_status_line(_INFO, msg))
            elif self._resume_compatible is False:
                lines.append(
                    _status_line(
                        _CROSS,
                        f"Existing DB incompatible: {self._resume_message}",
                    )
                )

        return (
            f'<div style="background:{ESA_BLUE_DEEP}; border-radius:6px;'
            f' padding:10px 14px; margin:4px 0;">' + "".join(lines) + "</div>"
        )

    def _refresh_validation(self) -> None:
        self._validation_html.value = self._render_validation()
        all_ok = self._model_ok and self._search_ok and self._output_ok
        # An incompatible existing DB must block start — delete the file
        # or pick a different output folder.
        incompatible = self._resume_compatible is False
        # A selected model whose normalisation can't be read (unreadable file or
        # no embedded metadata) would be rejected by ``load_model`` after the
        # subprocess spawns — block here so the failure surfaces at setup.
        model_unusable = self._model_ok and (
            self._model_norm is None or self._model_norm["image_size"] is None
        )
        self._start_btn.disabled = (not all_ok) or incompatible or model_unusable

        if (
            all_ok
            and self._resume_compatible
            and self._resume_count is not None
            and self._resume_count > 0
        ):
            self._start_btn.description = f"Continue prediction ({self._resume_count:,} done)"
        else:
            self._start_btn.description = "Start Prediction"

    # ── Start action ─────────────────────────────────────────────

    def _on_start(self) -> None:
        """Write validated paths to config and navigate to prediction screen."""
        # Fall back to cfg values when choosers haven't been used (auto-validated)
        model = self._model_chooser.selected or self._cfg.model_path
        search = self._search_chooser.selected or self._cfg.prediction_search_dir
        output = self._output_chooser.selected or self._cfg.output_dir

        self._cfg.model_path = os.path.normpath(model)
        self._cfg.prediction_search_dir = os.path.normpath(search)
        self._cfg.output_dir = os.path.normpath(output)

        # Normalisation is already mirrored onto cfg from the selected model
        # (and, on resume, reconciled against the DB) — nothing to apply here.

        logger.info(f"Model: {self._cfg.model_path}")
        logger.info(f"Search dir: {self._cfg.prediction_search_dir}")
        logger.info(f"Output dir: {self._cfg.output_dir}")

        self.navigate_to("prediction")
