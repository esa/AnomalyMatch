#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""Session orchestrator for the AnomalyMatch workflow.

The Session is a lightweight config holder and subprocess orchestrator.
Training runs via ``subprocess_scripts/training_process.py``, scoring via
``subprocess_scripts/prediction_process*.py``.  Labels are managed as CSV files
through :class:`~anomaly_match.data_io.SessionIOHandler.SessionIOHandler`.
"""

from __future__ import annotations

import datetime
import os
import pickle
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import torch
import zarr
from fitsbolt import SUPPORTED_IMAGE_EXTENSIONS
from fitsbolt.normalisation.NormalisationMethod import NormalisationMethod
from loguru import logger

from anomaly_match.data_io.checkpoint_io import (
    read_checkpoint_normalisation,
    sync_normalisation_from_checkpoint,
)
from anomaly_match.data_io.SessionIOHandler import SessionIOHandler
from anomaly_match.data_io.source_scanning import (
    detect_prediction_source_type,
    find_prediction_zarr_stores,
)
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.pipeline.SessionTracker import SessionTracker
from anomaly_match.prediction import (
    AnomalyScoreDB,
    SchemaVersionError,
    prediction_db_path,
    prepare_local_db,
    snapshot_db_to_session,
)
from anomaly_match.prediction.cutana_loader import _find_id_column
from anomaly_match.prediction.zarr_filenames import derive_filenames as _derive_zarr_filenames
from anomaly_match.utils.cutana_stream_utils import (
    cutana_buffer_generator,
    cutana_validate_files_and_count_sources,
)
from anomaly_match.utils.prediction_profiler import PredictionProfiler
from anomaly_match.utils.print_cfg import print_cfg
from anomaly_match.utils.set_log_level import set_log_level
from anomaly_match.utils.validate_config import validate_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from dotmap import DotMap
    from ipywidgets import Output


class Session:
    """Config holder and subprocess orchestrator for AnomalyMatch.

    Training and scoring run as subprocesses — the Session no longer
    holds an in-memory model, datasets, or prediction arrays.

    Args:
        cfg: Configuration for the session.
    """

    def __init__(self, cfg: DotMap) -> None:
        logger.debug("Initializing session")
        self.session_start = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if cfg.log_level is not None:
            set_log_level(cfg.log_level, cfg)
        if cfg.log_level in ["TRACE", "DEBUG"]:
            print_cfg(cfg)

        # Initialize session tracking
        session_name = getattr(cfg, "name", None)
        self.session_tracker = SessionTracker(session_name=session_name)
        self.session_io = SessionIOHandler()

        # Update config paths to use centralized session directory BEFORE validation
        self.session_io.update_config_paths_for_session(cfg, self.session_tracker)

        validate_config(cfg)

        self.cfg = cfg
        self.out = None

        # Prediction-cancellation state. The UI thread sets _prediction_stop_requested
        # (via request_prediction_stop) and the chunk loop in evaluate_all_images
        # checks it between subprocesses. _current_prediction_process holds the
        # live Popen so request_prediction_stop can SIGTERM it directly — without
        # this, Stop/Retrain could only skip the *next* chunk while the current
        # subprocess kept holding the GPU, which OOM'd a subsequent retrain.
        self._current_prediction_process: subprocess.Popen | None = None
        self._prediction_stop_requested = threading.Event()
        self._prediction_process_lock = threading.Lock()

        # Skip stats from the most recent evaluate_all_images() call.  When
        # a chunk subprocess produces zero rows (e.g. data volume detached
        # so every Cutana cutout came back empty) we skip the chunk and
        # continue rather than aborting the whole multi-hour run.  The UI
        # reads these after the run to surface a partial-completion warning
        # instead of pretending everything succeeded.
        self.last_run_skipped_chunks: int = 0
        self.last_run_skipped_sources: int = 0
        self.last_run_total_chunks: int = 0
        self.last_run_total_sources: int = 0

    # ── Terminal output ──────────────────────────────────────────────

    def set_terminal_out(self, out: Output) -> None:
        """Set the ipywidgets Output used by :meth:`remember_current_file`.

        Screens manage their own widget log sinks through ``_setup_logging``,
        so this method does not touch loguru.

        Args:
            out: The ipywidgets Output widget.
        """
        self.out = out

    # ── Remembered files ─────────────────────────────────────────────

    def remember_current_file(self, filename):
        """Remember the current file by appending it to a CSV if not already present."""
        with self.out if self.out is not None else nullcontext():
            os.makedirs(self.cfg.output_dir, exist_ok=True)

            output_file = os.path.join(
                self.cfg.output_dir,
                f"{self.cfg.name}_{self.session_start}_remembered_files.csv",
            )

            if os.path.exists(output_file):
                df = pd.read_csv(output_file)
                if filename in df["filename"].values:
                    logger.debug("File {} already in remembered files", filename)
                    return
            else:
                df = pd.DataFrame(columns=["filename", "timestamp"])

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = pd.DataFrame({"filename": [filename], "timestamp": [current_time]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(output_file, index=False)

            logger.info("Remembered file {}", filename)

    # ── Prediction subprocess orchestration ──────────────────────────

    def request_prediction_stop(self, timeout: float = 15.0) -> None:
        """Ask the current prediction run to stop gracefully.

        Sets a flag that prevents :meth:`evaluate_all_images` from
        launching any further chunk subprocesses, and sends ``SIGTERM``
        to the subprocess currently running (if any).  The subprocess's
        :func:`install_shutdown_handler` catches the signal, finishes the
        in-flight batch — which commits its scores to ``predictions.db``
        — and exits cleanly.

        Idempotent: safe to call when no prediction is running, or
        multiple times in a row.  Required before launching a new
        training subprocess on the same GPU: without it, the prediction
        subprocess keeps holding VRAM and the new training OOMs.

        Args:
            timeout: Seconds to wait for the SIGTERM'd subprocess to
                exit before escalating to ``SIGKILL``.  The shutdown
                handler flushes per-batch, so ``timeout`` needs to cover
                one batch of inference — 15s is comfortable at default
                batch sizes.
        """
        self._prediction_stop_requested.set()
        with self._prediction_process_lock:
            process = self._current_prediction_process
        # Idempotent contract: ``process`` is None when no prediction is in
        # flight (e.g. Stop clicked between chunks), and ``poll() is not None``
        # when the subprocess already exited on its own (e.g. last chunk just
        # finished).  Both are legitimate "nothing to terminate" states.
        if process is None or process.poll() is not None:
            return
        logger.info(
            "Requesting graceful shutdown of prediction subprocess (pid={})",
            process.pid,
        )
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Prediction subprocess {} did not exit within {}s of SIGTERM — sending SIGKILL",
                process.pid,
                timeout,
            )
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("Prediction subprocess {} ignored SIGKILL", process.pid)

    def run_pipeline(
        self,
        temp_config_path: str,
        input_path: str,
        top_N: int,
        file_type: DataSourceType | None = None,
    ) -> None:
        """Run the appropriate prediction subprocess based on file type.

        Args:
            temp_config_path: Path to the temporary config file for the subprocess.
            input_path: Path to input data (file or directory).
            top_N: Number of top predictions to keep.
            file_type: Data source override.  Auto-detected if None.

        Raises:
            FileNotFoundError: If the prediction script is not found.
            ValueError: If the file type is unsupported.
            RuntimeError: If the prediction subprocess exits with a
                non-zero return code (including signal-based termination
                such as SIGKILL from OOM).
        """
        if file_type is None:
            if os.path.isfile(input_path):
                _, ext = os.path.splitext(input_path.lower())
                extension_map = {
                    ".zarr": DataSourceType.ZARR,
                    ".txt": DataSourceType.IMAGE_FOLDER,
                    ".parquet": DataSourceType.CUTANA,
                    ".csv": DataSourceType.CUTANA,
                }
                file_type = extension_map.get(ext, DataSourceType.IMAGE_FOLDER)
            else:
                file_type = self._auto_detect_prediction_file_type(input_path)

        script_map = {
            DataSourceType.IMAGE_FOLDER: "prediction_process.py",
            DataSourceType.ZARR: "prediction_process_zarr.py",
            DataSourceType.CUTANA: "prediction_process_cutana.py",
        }

        script = script_map.get(file_type)
        if not script:
            raise ValueError(f"Unsupported prediction file type: {file_type}")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        script_path = os.path.join(root_dir, "subprocess_scripts", script)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found at expected path: {script_path}")

        if file_type == DataSourceType.IMAGE_FOLDER:
            temp_file_list = os.path.join("tmp", f"{self.cfg.save_file}_file_list.txt")
            with open(temp_file_list, "w") as f:
                f.write(input_path)
            cmd = [sys.executable, script_path, temp_config_path, temp_file_list, str(top_N)]
        else:
            cmd = [sys.executable, script_path, temp_config_path, input_path, str(top_N)]

        logger.info("Launching prediction subprocess: {}", script)
        # Stream subprocess output to our logger so progress is visible
        # in the Voila output widget.  PYTHONUNBUFFERED forces line-buffered
        # output so log lines appear immediately instead of waiting for
        # the 8KB pipe buffer to fill.
        #
        # Put the repo root and subprocess_scripts/ on PYTHONPATH so the script
        # can import anomaly_match (and prediction_utils) when the package is not
        # pip-installed.  The training launcher already does this; without the
        # same here, prediction fails at import time on a bare source checkout
        # even though training works.
        scripts_dir = os.path.join(root_dir, "subprocess_scripts")
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join([root_dir, scripts_dir])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        with self._prediction_process_lock:
            self._current_prediction_process = process
        try:
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    logger.info("[subprocess] {}", line)
            process.wait()
        finally:
            with self._prediction_process_lock:
                self._current_prediction_process = None

        set_log_level(self.cfg.log_level, self.cfg)

        if process.returncode != 0:
            # If we asked the subprocess to stop (SIGTERM from
            # request_prediction_stop), a nonzero return code is expected
            # — don't raise, just return so the chunk loop can break on
            # the stop flag.  SIGTERM-terminated subprocesses have already
            # flushed their current batch to DB (see install_shutdown_handler).
            if self._prediction_stop_requested.is_set():
                logger.info(
                    "Prediction subprocess exited after stop request (code {})",
                    process.returncode,
                )
                return

            # Raise so the caller (evaluate_all_images) aborts the chunk loop
            # instead of silently continuing.  A negative return code
            # indicates a signal (e.g. -9 = SIGKILL, usually OOM).
            signal_hint = ""
            if process.returncode < 0:
                signal_hint = (
                    f" (killed by signal {-process.returncode} — commonly SIGKILL from an OOM kill)"
                )
            raise RuntimeError(
                f"Prediction subprocess {script} exited with code "
                f"{process.returncode}{signal_hint}. "
                f"See {self.cfg.output_dir}/subprocess_logs/ and the "
                f"prediction_*.log next to the scored checkpoint for the full traceback."
            )

    def evaluate_all_images(
        self, top_N: int = 1000, progress_callback: Callable | None = None
    ) -> None:
        """Launch prediction subprocesses to score all images in the search directory.

        Args:
            top_N: Number of top-scoring images to keep.
            progress_callback: Optional callback for progress updates.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If no cutana-compatible files are found.
            ValueError: If no prediction search directory is configured or file type
                is unsupported.
            KeyError: If a stream chunk is missing the required ``source_id``
                column needed for resume filtering.
        """
        logger.info("Evaluating all images")

        # Fresh run — clear any stop flag left over from a previous call so
        # we don't immediately bail out of this one.
        self._prediction_stop_requested.clear()
        self.last_run_skipped_chunks = 0
        self.last_run_skipped_sources = 0
        self.last_run_total_chunks = 0
        self.last_run_total_sources = 0

        if not os.path.exists(self.cfg.model_path):
            error_msg = (
                f"Model not found at {self.cfg.model_path}. "
                "Please train and save a model before running predictions."
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Validate the search dir before reading the checkpoint — cheap input
        # validation precedes model I/O.
        if not self.cfg.prediction_search_dir:
            error_msg = (
                "No prediction_search_dir configured. "
                "Please set cfg.prediction_search_dir to a directory containing "
                "images, Zarr files, or Cutana buffer files."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # The model checkpoint is the single source of truth for normalisation.
        # Always sync from it (not just when fitsbolt_cfg is unset): the Cutana
        # orchestrator resolution and channel mixing read cfg.normalisation while
        # inference decode reads cfg.fitsbolt_cfg, and a stale cfg left over from
        # training — or a different model selected later in the same session —
        # would otherwise feed the model images it was never trained on.
        model_fitsbolt_cfg, model_channel_combination = read_checkpoint_normalisation(
            self.cfg.model_path
        )
        if not sync_normalisation_from_checkpoint(
            self.cfg, model_fitsbolt_cfg, model_channel_combination
        ):
            # Fail hard: the checkpoint must carry the normalisation it was
            # trained with.  Building a config from the current settings would
            # invent an unverifiable pipeline and silently score on the wrong
            # resolution/normalisation — retrain to embed it instead.
            error_msg = (
                f"Model checkpoint {self.cfg.model_path} has no embedded fitsbolt "
                "config; it cannot be used for prediction. Retrain the model so its "
                "normalisation settings are saved with it."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        logger.debug(
            "Synced normalisation from model checkpoint for prediction: "
            "image_size={}, method={}, channels={}",
            self.cfg.normalisation.image_size,
            self.cfg.normalisation.normalisation_method,
            self.cfg.normalisation.n_output_channels,
        )

        detected_file_type = self._auto_detect_prediction_file_type(self.cfg.prediction_search_dir)

        # Check for Cutana + MIDTONES incompatibility
        if detected_file_type == DataSourceType.CUTANA:
            if self.cfg.normalisation.normalisation_method == NormalisationMethod.MIDTONES:
                error_msg = (
                    "MIDTONES normalisation is not supported for Cutana streaming predictions. "
                    "Please use CONVERSION_ONLY, LOG, ZSCALE, or ASINH."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Define supported file extensions
        supported_extensions = {
            DataSourceType.IMAGE_FOLDER: SUPPORTED_IMAGE_EXTENSIONS,
            DataSourceType.ZARR: [".zarr"],
            DataSourceType.CUTANA: [".csv", ".parquet"],
        }

        pattern = supported_extensions.get(detected_file_type)
        if not pattern:
            raise ValueError(f"Unsupported prediction file type: {detected_file_type}")

        # Get all matching files
        input_files = []
        if detected_file_type == DataSourceType.ZARR:
            input_files = [
                path for _name, path in find_prediction_zarr_stores(self.cfg.prediction_search_dir)
            ]
        else:
            for f in os.listdir(self.cfg.prediction_search_dir):
                file_path = os.path.join(self.cfg.prediction_search_dir, f)
                file_ext = os.path.splitext(f.lower())[1]
                if file_ext in pattern:
                    input_files.append(file_path)

        total_images = 0
        processed_images = 0
        start_time = time.time()

        # Count total images
        logger.info(
            "Counting sources in {} ({} detected type: {})...",
            self.cfg.prediction_search_dir,
            len(input_files),
            detected_file_type,
        )
        if detected_file_type != DataSourceType.CUTANA:
            for input_file in input_files:
                try:
                    if detected_file_type == DataSourceType.ZARR:
                        root = zarr.open_group(input_file, mode="r")
                        if "images" in root:
                            total_images += root["images"].shape[0]
                        else:
                            logger.warning("No 'images' array found in Zarr file {}", input_file)
                    else:
                        total_images += 1
                except Exception as e:
                    logger.warning("Error counting images in {}: {}", input_file, str(e))
        else:
            logger.info("Validating files against cutana")
            input_files, total_images, total_chunks = cutana_validate_files_and_count_sources(
                input_files, chunk_size=self.cfg.subprocess_buffer_size
            )
            if not input_files:
                msg = "All found files are not compatible with cutana"
                logger.error(msg)
                raise RuntimeError(msg)

        num_files = len(input_files)
        logger.info("Found total of {:,} images to process in {} files", total_images, num_files)

        # Relocate the live DB to local scratch (and seed it from the session
        # snapshot) before any read/write, so resume filtering and the writer
        # subprocesses all target the same file. No-op on a local session dir.
        prepare_local_db(self.cfg)

        # Pre-filter resume: drop fully-scored inputs at dispatch time so
        # we never spawn a subprocess whose only job is to open the DB,
        # discover it has nothing to do, and exit.  Stream chunks can't
        # be filtered until they materialise, so they're handled inside
        # the per-chunk loop below.
        processed = self._load_processed_filenames()
        if processed and detected_file_type == DataSourceType.IMAGE_FOLDER:
            before = len(input_files)
            input_files = [p for p in input_files if str(p) not in processed]
            dropped = before - len(input_files)
            if dropped:
                total_images -= dropped
                logger.info(
                    "Resume: dropping {:,} already-scored image path(s) from dispatch",
                    dropped,
                )
        elif processed and detected_file_type == DataSourceType.ZARR:
            kept: list[str] = []
            dropped_files = 0
            dropped_images = 0
            for zarr_path in input_files:
                names = _derive_zarr_filenames(zarr_path)
                if names and all(str(n) in processed for n in names):
                    dropped_files += 1
                    dropped_images += len(names)
                else:
                    kept.append(zarr_path)
            if dropped_files:
                input_files = kept
                total_images -= dropped_images
                logger.info(
                    "Resume: dropping {} fully-scored Zarr file(s) ({:,} images) from dispatch",
                    dropped_files,
                    dropped_images,
                )

        # Nothing to score — avoid launching a subprocess with an empty file list.
        if total_images == 0 or not input_files:
            logger.warning("No images found to process in {}", self.cfg.prediction_search_dir)
            return

        # Group image files
        if detected_file_type == DataSourceType.IMAGE_FOLDER:
            total_input_images = total_images
            group_size = 10000
            grouped_files = (
                [input_files]
                if len(input_files) <= group_size
                else [
                    input_files[i : i + group_size] for i in range(0, len(input_files), group_size)
                ]
            )
            tmp_dir = Path("tmp")
            tmp_dir.mkdir(exist_ok=True)

            input_files = []
            for idx, group in enumerate(grouped_files):
                path = tmp_dir / f"evaluate_all_images_grouped_{idx}.txt"
                path.write_text("\n".join(group))
                input_files.append(str(path))
            num_files = len(input_files)
            logger.debug(
                "Created {} group{} for {} images",
                len(input_files),
                "s" if len(input_files) != 1 else "",
                total_input_images,
            )
        elif detected_file_type == DataSourceType.CUTANA:
            cutana_buffer_path = Path("tmp") / ".cutana_buffer.parquet"
            logger.info(
                "Streaming {:,} sources in {} chunks of {:,}",
                total_images,
                total_chunks,
                self.cfg.subprocess_buffer_size,
            )
            input_files = cutana_buffer_generator(
                files=input_files,
                buffer_path=cutana_buffer_path,
                chunk_size=self.cfg.subprocess_buffer_size,
            )
            num_files = total_chunks

        profiler_coordinator = PredictionProfiler.Coordinator(self.cfg.output_dir)

        for file_idx, input_file in enumerate(input_files):
            if self._prediction_stop_requested.is_set():
                logger.info(
                    "Prediction stop requested — skipping {}/{} remaining chunk(s)",
                    num_files - file_idx,
                    num_files,
                )
                break
            if detected_file_type == DataSourceType.ZARR:
                try:
                    root = zarr.open_group(input_file, mode="r")
                    if "images" in root:
                        num_items = root["images"].shape[0]
                    else:
                        logger.warning("No 'images' array found in Zarr file {}", input_file)
                        num_items = 0
                except Exception as e:
                    logger.error("Error reading Zarr file {}: {}", input_file, e)
                    num_items = 0
            elif detected_file_type == DataSourceType.CUTANA:
                if str(input_file).endswith(".parquet"):
                    chunk_df = pd.read_parquet(input_file)
                else:
                    chunk_df = pd.read_csv(input_file)
                num_items = len(chunk_df)
                # Stream chunks materialise per-iteration, so the resume
                # pre-filter runs here rather than up front like image/zarr.
                # The id column may use any of the recognised Cutana spellings
                # (the data uses ``SourceID``); a genuinely missing column is a
                # schema bug.
                if processed:
                    try:
                        id_column = _find_id_column(chunk_df)
                    except ValueError as exc:
                        raise KeyError(
                            f"Stream chunk {input_file} has no source-id column; "
                            f"got {list(chunk_df.columns)}."
                        ) from exc
                    chunk_ids = {str(sid) for sid in chunk_df[id_column].tolist()}
                    if chunk_ids and chunk_ids.issubset(processed):
                        logger.info(
                            "Resume: chunk {}/{} is fully scored ({:,} sources) "
                            "— skipping subprocess spawn",
                            file_idx + 1,
                            num_files,
                            num_items,
                        )
                        total_images -= num_items
                        continue
                del chunk_df
            else:
                if str(input_file).endswith(".txt"):
                    with open(input_file, "r") as f:
                        num_items = len(f.readlines())
                else:
                    num_items = 1

            # Progress tracking — emit an overall run-status line at *every*
            # subprocess spawn (including the first) so the session log always
            # carries the whole-run picture: chunks and sources done/todo,
            # elapsed runtime, throughput, and ETA.  Previously the first chunk
            # logged only a bare "Processing N images" line (speed/ETA need a
            # completed chunk to measure), so a run inspected early — or one
            # that only reached chunk 1 — showed no overall status at all.
            elapsed_time = time.time() - start_time
            runtime_str = str(datetime.timedelta(seconds=int(elapsed_time)))
            progress_percent = (processed_images / total_images * 100) if total_images else 0.0
            images_per_second = processed_images / elapsed_time if elapsed_time > 0 else 0.0
            # A usable rate needs a completed chunk *and* measurable elapsed time;
            # images_per_second > 0 already implies processed_images > 0.  Gate the
            # log line and the progress callback on this single condition so they
            # can't disagree in the elapsed_time <= 0 corner (where the callback
            # would otherwise forward images_per_second=0.0 / eta "unknown" while
            # the log says "measuring").
            have_rate = images_per_second > 0
            if have_rate:
                eta_seconds = (total_images - processed_images) / images_per_second
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                speed_str = f"{images_per_second:.1f} img/s"
            else:
                # No completed chunk yet — no rate to extrapolate from.
                eta_str = "unknown"
                speed_str = "measuring"

            logger.info(
                "Run status: chunk {}/{} | {:,}/{:,} sources ({:.1f}%) | "
                "runtime {} | {} | ETA {} | now processing {:,} in {}",
                file_idx + 1,
                num_files,
                processed_images,
                total_images,
                progress_percent,
                runtime_str,
                speed_str,
                eta_str,
                num_items,
                os.path.basename(str(input_file)),
            )

            if progress_callback:
                if have_rate:
                    progress_callback(
                        file_idx + 1,
                        num_files,
                        batch_update=True,
                        eta_str=eta_str,
                        progress_percent=progress_percent,
                        images_per_second=images_per_second,
                    )
                else:
                    progress_callback(file_idx + 1, num_files, batch_update=True)

            # Serialize config and launch subprocess
            temp_config = self.cfg.toDict()
            if not os.path.exists(self.cfg.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.cfg.model_path}. "
                    "Please ensure you have saved the model before running predictions."
                )

            temp_config_path = os.path.join("tmp", f"{self.cfg.save_file}_config.pkl")
            os.makedirs("tmp", exist_ok=True)
            with open(temp_config_path, "wb") as f:
                pickle.dump(temp_config, f)
            logger.debug("Temporary config saved to {}", temp_config_path)

            os.makedirs(self.cfg.output_dir, exist_ok=True)

            profiler_coordinator.record_subprocess_gap()

            # Measure row count before/after so we can detect chunks that
            # exit cleanly but produce no predictions (e.g. an orchestrator
            # returning zero batches, or silent failures upstream).
            db_path = prediction_db_path(self.cfg)
            rows_before = _count_db_rows(db_path)

            chunk_failed = False
            try:
                self.run_pipeline(temp_config_path, input_file, top_N, detected_file_type)
            except RuntimeError as exc:
                # Don't abort the whole multi-hour run on one bad chunk.
                # Common cause is the data volume detaching mid-run, which
                # the subprocess surfaces as a non-zero exit; future chunks
                # may target tiles that are still reachable.
                logger.warning(
                    "Chunk {}/{} subprocess failed — {}.  Skipping and continuing "
                    "with remaining chunks.",
                    file_idx + 1,
                    num_files,
                    exc,
                )
                chunk_failed = True
            profiler_coordinator.mark_subprocess_complete()

            rows_after = _count_db_rows(db_path)
            rows_added = rows_after - rows_before
            logger.info(
                "Chunk {}/{} finished — {:,} new rows (total {:,})",
                file_idx + 1,
                num_files,
                rows_added,
                rows_after,
            )

            # Snapshot the relocated DB back to the (durable) session dir after
            # each chunk so a pod restart loses at most one chunk. No-op when the
            # DB already lives in the session dir.
            snapshot_db_to_session(self.cfg)
            if chunk_failed or rows_added == 0:
                # Either the subprocess errored out or it ran cleanly but
                # produced nothing (e.g. empty cutouts from unreachable
                # FITS tiles).  Both count as a skipped chunk.
                self.last_run_skipped_chunks += 1
                self.last_run_skipped_sources += num_items
                if not chunk_failed:
                    logger.warning(
                        "Chunk {}/{} produced zero predictions ({:,} sources skipped). "
                        "Subprocess exited cleanly — see {}/subprocess_logs/ for details.",
                        file_idx + 1,
                        num_files,
                        num_items,
                        self.cfg.output_dir,
                    )
            if not os.path.exists(db_path):
                logger.error(
                    "Output database not found. Prediction process might have failed. "
                    "On Datalabs, the process may have exceeded the RAM allocation. "
                    "Please check logs in the folder <anomaly_match/logs>."
                )
            elif progress_callback:
                progress_callback(file_idx + 1, num_files, results_updated=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            processed_images += num_items

        # Final statistics
        total_time = time.time() - start_time
        if total_time > 0 and processed_images > 0:
            final_speed = processed_images / total_time
            time_str = str(datetime.timedelta(seconds=int(total_time)))
            logger.success(
                "Completed processing {:,} images in {} ({:.1f} img/s)",
                processed_images,
                time_str,
                final_speed,
            )

            if progress_callback:
                progress_callback(
                    num_files,
                    num_files,
                    batch_update=True,
                    completed=True,
                    total_time_str=time_str,
                    final_speed=final_speed,
                )
        else:
            logger.warning("No images were processed or processing time was too short")

        self.last_run_total_chunks = num_files
        self.last_run_total_sources = total_images

        # Surface partial completion clearly — the UI reads
        # ``last_run_skipped_*`` to paint an orange (not green) status
        # when chunks were skipped, so a run where every chunk failed
        # ends up with "0 / N scored, N chunks skipped" instead of a
        # misleading "Done" bar.  The summary log here also tells the
        # user where to look without aborting downstream cleanup.
        if self.last_run_skipped_chunks > 0:
            logger.warning(
                "Run finished with {}/{} chunk(s) skipped — ~{:,} source(s) unscored. "
                "See per-chunk warnings above for the cause (commonly: data volume "
                "detached mid-run, FITS tiles unreachable).",
                self.last_run_skipped_chunks,
                num_files,
                self.last_run_skipped_sources,
            )

        profiler_coordinator.finalize()
        logger.info("Processed {} files with {} format", num_files, detected_file_type)

    # ── Session info ─────────────────────────────────────────────────

    def get_session_info(self) -> dict:
        """Get session information from the session tracker.

        Returns:
            Session information dictionary.
        """
        return self.session_tracker.get_session_info()

    def get_iteration_info(self, iteration_number: int | None = None) -> dict:
        """Get iteration information from the session tracker.

        Args:
            iteration_number: Specific iteration to retrieve. Defaults to latest.

        Returns:
            Iteration information dictionary.
        """
        return self.session_tracker.get_iteration_info(iteration_number)

    def save_session(self) -> str:
        """Save the complete session using SessionIOHandler.

        Returns:
            Path where the session was saved.
        """
        return self.session_io.save_session(self.session_tracker, cfg=self.cfg)

    # ── Internal helpers ─────────────────────────────────────────────

    def _load_processed_filenames(self) -> set[str]:
        """Read the set of filenames already scored into ``predictions.db``.

        Used to pre-filter dispatch so subprocesses don't spawn to do
        no-op skip passes.  Returns an empty set when the DB is missing,
        unreadable, or written by an incompatible schema — the
        subprocess gate will surface that error properly on its own.

        Returns:
            Set of filenames already present in ``predictions.db``.
        """
        db_path = prediction_db_path(self.cfg)
        if not os.path.exists(db_path):
            return set()
        try:
            with AnomalyScoreDB(db_path) as db:
                return db.get_processed_filenames()
        except SchemaVersionError as exc:
            logger.warning("Predictions DB at {} is unreadable: {}", db_path, exc)
            return set()
        except Exception as exc:
            logger.warning("Could not read predictions DB at {}: {}", db_path, exc)
            return set()

    def _auto_detect_prediction_file_type(self, search_dir: str) -> DataSourceType:
        """Auto-detect prediction file type based on files in the directory.

        Args:
            search_dir: Directory to scan for prediction files, or a direct
                path to a single store (e.g. a ``.zarr`` directory).

        Returns:
            Detected source-type enum member.
        """
        if not search_dir or not os.path.exists(search_dir):
            logger.warning(
                "Search directory {} does not exist, defaulting to image-folder file type",
                search_dir,
            )
            return DataSourceType.IMAGE_FOLDER

        detected = detect_prediction_source_type(search_dir)
        logger.info("Auto-detected prediction file type: {}", detected.value)
        return detected


def _count_db_rows(db_path: str) -> int:
    """Return ``MAX(id)`` from predictions.db as a row-count proxy.

    Used to detect chunks that exit cleanly but produce no predictions.
    Returns ``MAX(id)`` (the autoincremented ``INTEGER PRIMARY KEY``)
    instead of ``COUNT(*)`` so the diagnostic stays O(log N) on the
    rowid index — predictions DBs grow into the hundreds of millions
    of rows, where a full ``COUNT(*)`` scan would cost minutes.

    Single-writer assumption: the per-chunk delta computed from this
    value is only meaningful while ``evaluate_all_images`` runs chunks
    sequentially.  If chunk subprocesses are ever parallelised, replace
    this global delta with a per-subprocess count reported by the
    writer itself (see issue #391).

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        ``MAX(id)`` of the ``results`` table, or 0 if the DB or table
        isn't present.  Equals total inserted rows for an append-only
        table with autoincrementing rowids.
    """
    if not os.path.exists(db_path):
        return 0
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            row = conn.execute("SELECT COALESCE(MAX(id), 0) FROM results").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error:
        return 0
