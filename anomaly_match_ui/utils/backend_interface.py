#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""BackendInterface: Single interface for UI-backend communication.

This module provides a static interface for the UI components to interact with
the backend Session without direct imports, enabling clean separation between
the UI and backend packages.
"""

from __future__ import annotations

import contextlib
import copy
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import zarr
from filelock import FileLock, Timeout
from loguru import logger

import anomaly_match
from anomaly_match.data_io.checkpoint_io import (
    read_checkpoint_normalisation,
    read_model_normalisation,
    sync_normalisation_from_checkpoint,
)
from anomaly_match.data_io.container_loaders import decode_zarr_gallery_images
from anomaly_match.data_io.labeled_data_cache import (
    LabeledDataCache,
    is_labeled_cache_populated,
)
from anomaly_match.data_io.load_images import load_and_process_single_wrapper
from anomaly_match.data_io.source_scanning import (
    SourceScanResult,
)
from anomaly_match.data_io.source_scanning import (
    decode_raw_cutouts as _decode_raw_cutouts,
)
from anomaly_match.data_io.source_scanning import (
    detect_and_count_prediction_sources as _detect_and_count_prediction_sources,
)
from anomaly_match.data_io.source_scanning import (
    detect_source_channel_count as _detect_source_channel_count,
)
from anomaly_match.data_io.source_scanning import (
    find_prediction_zarr_stores as _find_prediction_zarr_stores,
)
from anomaly_match.data_io.source_scanning import (
    load_cutana_preview as _load_cutana_preview,
)
from anomaly_match.data_io.source_scanning import (
    load_preview_samples as _load_preview_samples,
)
from anomaly_match.data_io.source_scanning import (
    read_label_map as _read_label_map,
)
from anomaly_match.data_io.source_scanning import (
    scan_and_count_sources as _scan_and_count_sources,
)
from anomaly_match.data_io.source_validation import validate_labeled_files
from anomaly_match.datasets.Label import LABEL_ANOMALY, LABEL_NORMAL
from anomaly_match.datasets.training_data_source import DataSourceType
from anomaly_match.prediction import (
    AnomalyScoreDB,
    ImageCache,
    prediction_db_path,
    session_db_path,
)
from anomaly_match.prediction.cutana_loader import (
    load_batch_cutouts,
    load_single_cutout,
    load_single_native_cutout,
)
from anomaly_match.prediction.zarr_filenames import derive_filenames as _derive_zarr_filenames
from anomaly_match.utils.get_default_cfg import (
    is_shipped_default_path as _is_shipped_default_path,
)
from anomaly_match.utils.set_log_level import set_log_level as _set_log_level_impl

if TYPE_CHECKING:
    from anomaly_match.pipeline.session import Session


# Seconds to wait for a peer process / thread to finish building the
# labeled cache before we bail out.  One minute is plenty for typical
# catalogues; large Cutana builds that exceed it will raise Timeout,
# which the caller surfaces as a clear error instead of silently racing.
_LABELED_CACHE_LOCK_TIMEOUT = 60


def _labeled_cache_lock(cache_dir: str) -> FileLock:
    """Cross-process lock for a labeled-cache directory.

    Uses ``{cache_dir}.lock`` as a sibling file so the lock works even
    before the cache itself has been created.

    Args:
        cache_dir: Directory that holds / will hold the cache.

    Returns:
        Unacquired :class:`FileLock` — caller must use ``with``.
    """
    parent = os.path.dirname(cache_dir) or "."
    os.makedirs(parent, exist_ok=True)
    return FileLock(f"{cache_dir}.lock", timeout=_LABELED_CACHE_LOCK_TIMEOUT)


@contextlib.contextmanager
def _acquire_or_bail(cache_dir: str, stale_check: Callable[[], bool] | None):
    """Acquire the cache lock while polling *stale_check*.

    The setup screen spawns one background job per slider tick, and
    every job tries to acquire the cache lock.  Without this wrapper a
    later job would block on the lock for a full 60 s while an earlier
    (still-running) job rebuilds — then raise ``Timeout`` even though
    the cache is about to become fresh.  This helper polls
    *stale_check* every second while we wait, so superseded jobs bail
    out silently instead of queuing up a storm of timeouts.

    Yields:
        The acquired :class:`FileLock` on success, or ``None`` when the
        job was marked stale before we could acquire.

    Raises:
        Timeout: Only if we waited the full
            :data:`_LABELED_CACHE_LOCK_TIMEOUT` seconds without becoming
            stale — genuinely indicating a stuck peer.
    """
    parent = os.path.dirname(cache_dir) or "."
    os.makedirs(parent, exist_ok=True)
    lock = FileLock(f"{cache_dir}.lock")
    deadline = time.monotonic() + _LABELED_CACHE_LOCK_TIMEOUT
    acquired = False
    try:
        while True:
            if stale_check is not None and stale_check():
                yield None
                return
            try:
                lock.acquire(timeout=1)
                acquired = True
                break
            except Timeout:
                if time.monotonic() >= deadline:
                    raise
        yield lock
    finally:
        if acquired:
            lock.release()


def _derive_labeled_cache_path(label_file: str | None, output_dir: str) -> str:
    """Return the labeled-cache directory for *label_file*.

    Preferred location is ``{label_file.parent}/{label_file.stem}_cache``
    so the cache persists across sessions as long as the same CSV is
    reused.  Falls back to ``{output_dir}/labeled_cache`` when
    *label_file* is unset or its parent is unwritable (e.g. read-only
    share) — determined by a probe-file write, not :func:`os.access`,
    which lies on network mounts.

    Args:
        label_file: Absolute or relative path to the label CSV, or
            ``None``.
        output_dir: Session output directory used for the fallback.

    Returns:
        Absolute path to the cache directory (may or may not exist yet).
    """
    fallback = os.path.join(output_dir, "labeled_cache")
    if not label_file:
        return fallback
    label_path = os.path.abspath(label_file)
    parent = os.path.dirname(label_path) or "."
    stem, _ = os.path.splitext(os.path.basename(label_path))
    preferred = os.path.join(parent, f"{stem}_cache")
    try:
        os.makedirs(parent, exist_ok=True)
        probe_fd, probe_path = tempfile.mkstemp(prefix=".am_cache_probe_", dir=parent)
        os.close(probe_fd)
        os.remove(probe_path)
    except OSError as exc:
        logger.warning(
            "Cannot write labeled cache next to label file ({}: {}); falling back to {}",
            parent,
            exc,
            fallback,
        )
        return fallback
    return preferred


class BackendInterface:
    """Single interface for UI-backend communication.

    This static class wraps all Session methods needed by the UI components,
    providing a clean interface that decouples the UI from the backend implementation.
    """

    _session = None

    # ========== Session Lifecycle ==========

    @staticmethod
    def set_session(session: Session) -> None:
        """Set the session instance to use.

        Args:
            session: The Session instance to interact with.
        """
        BackendInterface._session = session

    @staticmethod
    def get_session() -> Session | None:
        """Get the current session instance.

        Returns:
            The current Session instance, or None if not set.
        """
        return BackendInterface._session

    @staticmethod
    def _check_session() -> None:
        """Check that a session is set.

        Raises:
            RuntimeError: If no session has been set via ``set_session()``.
        """
        if BackendInterface._session is None:
            raise RuntimeError("No session set. Call BackendInterface.set_session() first.")

    # ========== Configuration ==========

    @staticmethod
    def get_config() -> Any:
        """Get the session configuration.

        Returns:
            The session configuration object (DotMap).
        """
        BackendInterface._check_session()
        return BackendInterface._session.cfg

    @staticmethod
    def get_num_channels() -> int:
        """Get the number of channels in the images.

        Returns:
            int: Number of image channels.
        """
        BackendInterface._check_session()
        return BackendInterface._session.cfg.num_channels

    @staticmethod
    def get_prediction_db_path(output_dir: str | None = None) -> str:
        """Return the live ``predictions.db`` path for the active session.

        Resolves any local-disk relocation (NFS sessions) centrally, so the UI
        poller reads the same file the prediction subprocess writes.

        Args:
            output_dir: Optional session-dir override (e.g. a chooser selection);
                defaults to the session's ``output_dir``.

        Returns:
            Path to the live ``predictions.db``.
        """
        BackendInterface._check_session()
        return prediction_db_path(BackendInterface._session.cfg, output_dir=output_dir)

    @staticmethod
    def get_session_db_path(output_dir: str | None = None) -> str:
        """Return the durable session-dir ``predictions.db`` path.

        This is the snapshot copy that survives a relocated run; it equals
        :meth:`get_prediction_db_path` when the DB is not relocated.

        Args:
            output_dir: Optional session-dir override; defaults to the session's
                ``output_dir``.

        Returns:
            Path to ``predictions.db`` in the session directory.
        """
        BackendInterface._check_session()
        return session_db_path(BackendInterface._session.cfg, output_dir=output_dir)

    # ========== Training Subprocess ==========

    @staticmethod
    def launch_training_subprocess(
        num_train_iter: int | None = None,
        normalisation_overrides: dict[str, Any] | None = None,
        gallery_labels: dict[str, str] | None = None,
    ) -> tuple[subprocess.Popen, str, str]:
        """Serialize config and labels, then launch the training subprocess.

        Prepares a temporary directory with the pickled config and merged
        labels CSV, then spawns ``subprocess_scripts/training_process.py`` as a
        process.  The caller monitors progress via the returned
        progress-file path and :class:`~subprocess.Popen` handle.

        Args:
            num_train_iter: Override for ``cfg.num_train_iter``.
            normalisation_overrides: Dict of normalisation config overrides
                (keys matching ``cfg.normalisation.*``).
            gallery_labels: Dict mapping ``filename → csv_label_string``
                (e.g. ``LABEL_ANOMALY`` / ``LABEL_NORMAL``) for new labels
                from the UI gallery to merge with the existing CSV.

        Returns:
            ``(process, temp_dir, progress_file)`` — the Popen handle, the
            path to the temporary directory, and the path to the JSON-lines
            progress file the subprocess writes to.

        Raises:
            FileNotFoundError: If the training script cannot be found.
        """
        BackendInterface._check_session()
        session = BackendInterface._session
        cfg = session.cfg

        # Apply overrides
        if num_train_iter is not None:
            cfg.num_train_iter = num_train_iter
        if normalisation_overrides:
            for key, value in normalisation_overrides.items():
                setattr(cfg.normalisation, key, value)

        # Advance session iteration counter and create per-iteration output
        # subdirectory so each train cycle gets its own model, labels, and log.
        iteration = session.session_tracker.start_new_session_iteration()
        iter_dir = os.path.join(cfg.output_dir, f"iteration_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        cfg.model_path = os.path.join(iter_dir, "model.safetensors")

        # Prepare temp directory with config pickle, labels CSV, progress file
        temp_dir = tempfile.mkdtemp(prefix="am_train_")
        config_path = os.path.join(temp_dir, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(cfg.toDict(), f)

        labels_csv_path = os.path.join(temp_dir, "labelled_data.csv")
        BackendInterface.merge_gallery_labels(gallery_labels or {}, output_path=labels_csv_path)

        progress_file = os.path.join(temp_dir, "progress.jsonl")

        # Locate training script
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "subprocess_scripts",
        )
        script = os.path.join(scripts_dir, "training_process.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"Training script not found: {script}")

        cmd = [sys.executable, script, config_path, labels_csv_path, progress_file]

        # Pass labeled data cache path if available (for container sources).
        # Verify the cache is actually populated (all three files exist),
        # not just that the directory was created.  Also rebuild it if
        # extraction-affecting parameters have drifted since it was built —
        # otherwise the training subprocess would see cached cutouts at one
        # resolution while the unlabeled stream produces another, and
        # BasicDataset.stack would fail with a shape mismatch.
        labeled_cache_path = cfg.labeled_cache_path
        if labeled_cache_path and is_labeled_cache_populated(labeled_cache_path):
            BackendInterface._refresh_labeled_cache_if_stale(labeled_cache_path, cfg)
            cmd.extend(["--labeled-cache", labeled_cache_path])

            # Drop labels whose id didn't make it into the labeled cache —
            # without this, AnomalyDetectionDataset.update_labels logs one
            # "not found in dataset" warning per orphaned row and can flood
            # the training screen on catalogues where only a fraction of
            # labelled sources were surfaced by Cutana.  The cache has
            # already been validated+refreshed above so its id set is the
            # ground truth for what training will actually see.
            cache = LabeledDataCache(labeled_cache_path)
            valid_ids = set(cache.get_label_df()["id"].astype(str))
            kept, dropped = session.session_io.filter_labels_csv_to_ids(labels_csv_path, valid_ids)
            if dropped:
                logger.info(
                    "Dropped {} label(s) missing from the data source "
                    "before launching training ({} kept)",
                    dropped,
                    kept,
                )

        logger.info("Launching training subprocess: {}", " ".join(cmd))

        # Ensure repo root and scripts/ are on PYTHONPATH so the
        # subprocess can import anomaly_match and prediction_utils.
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = {**os.environ}
        extra = os.pathsep.join([repo_root, scripts_dir])
        env["PYTHONPATH"] = extra + os.pathsep + env.get("PYTHONPATH", "")

        # stdout → DEVNULL: tqdm progress bars write there and would fill
        # the pipe buffer causing the subprocess to block.  All useful output
        # goes elsewhere (loguru → training.log, progress → .jsonl, logs → stderr).
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
        return process, temp_dir, progress_file

    @staticmethod
    def _refresh_labeled_cache_if_stale(cache_path: str, cfg: Any) -> None:
        """Rebuild the labeled cache when extraction-affecting params have drifted.

        The training screen lets the user change ``image_size``,
        ``normalisation_method``, ``fits_extension`` and other extraction
        parameters after validation, without re-triggering the setup
        screen's background job.  Without this refresh the cache would
        stay baked at the old parameters while the unlabeled stream uses
        the new cfg, causing a shape or dtype mismatch when
        ``BasicDataset.__init__`` stacks the two together.

        Cache rebuild is only triggered when extraction-affecting
        parameters change — ``channel_combination`` and
        ``n_output_channels`` are applied at read time, so changing them
        does **not** trigger a rebuild.

        Args:
            cache_path: Path to the existing labeled cache directory.
            cfg: Current session config.
        """
        source_dir = cfg.data_dir
        label_path = cfg.label_file
        if not source_dir or not label_path or not os.path.isfile(label_path):
            logger.warning(
                "Labeled cache appears stale but source_dir/label_file are unset — "
                "cannot rebuild from launch_training_subprocess"
            )
            return

        label_df = pd.read_csv(label_path)
        cache = LabeledDataCache(cache_path)
        if not cache.needs_rebuild(cfg, label_df=label_df):
            return

        source_type = cache.source_type
        if source_type not in (DataSourceType.ZARR, DataSourceType.CUTANA):
            return

        logger.info("Labeled cache is stale (extraction params or label CSV changed) — rebuilding")
        result = validate_labeled_files(source_dir, label_df, source_type)
        with _labeled_cache_lock(cache_path):
            cache.build(result.found_locations, label_df, source_type, cfg)

    @staticmethod
    def merge_gallery_labels(
        new_labels: dict[str, str],
        output_path: str | None = None,
    ) -> str:
        """Merge new gallery labels with the existing label CSV and save.

        Delegates I/O to :meth:`SessionIOHandler.merge_gallery_labels`.

        Args:
            new_labels: Dict mapping ``id → csv_label_string``
                (values should be :data:`LABEL_ANOMALY`,
                :data:`LABEL_NORMAL` or :data:`LABEL_REMOVED`).
            output_path: Where to write the merged CSV.  Defaults to
                ``cfg.output_dir / labelled_data.csv``.

        Returns:
            Absolute path to the written CSV file.
        """
        BackendInterface._check_session()
        session = BackendInterface._session
        cfg = session.cfg

        if output_path is None:
            output_path = os.path.join(cfg.output_dir, "labelled_data.csv")

        return session.session_io.merge_gallery_labels(
            existing_label_file=cfg.label_file,
            new_labels=new_labels,
            output_path=output_path,
        )

    # ========== Label Validation & Cache ==========

    @staticmethod
    def validate_labels_against_source(
        source_dir: str,
        source_type: str,
        label_path: str,
        *,
        stale_check=None,
    ) -> tuple[str, dict[str, tuple], bool]:
        """Validate labeled IDs against the data source.

        Delegates to :func:`~anomaly_match.data_io.source_validation.validate_labeled_files`
        and formats the result as a human-readable message.

        Args:
            source_dir: Path to the data directory.
            source_type: A :class:`DataSourceType` value.
            label_path: Path to the label CSV.
            stale_check: Optional zero-arg callable returning ``True``
                when this request has been superseded by a newer
                background job (see the setup screen's job-generation
                tracking).  Forwarded into the Cutana validation loop so
                older jobs bail between catalogues instead of driving
                through all N.

        Returns:
            Tuple of ``(status_message, found_locations, partial)``.  When
            ``partial`` is ``True`` the validator bailed mid-loop and the
            other two fields reflect only the catalogues processed so far —
            callers must not cache them or build a labeled cache from them.
        """
        label_df = pd.read_csv(label_path)
        result = validate_labeled_files(source_dir, label_df, source_type, stale_check=stale_check)

        if result.partial:
            # Counts are misleading when the loop bailed early — surface the
            # supersession explicitly instead of fabricating a "{n} missing"
            # number that includes unscanned catalogues.
            message = "Validation interrupted (newer job pending)"
            return message, result.found_locations, True

        n_found = len(result.found)
        n_missing = len(result.missing)

        # Break down found/missing by training-relevant labels so the user
        # sees whether a silent source drop wiped out their anomaly
        # examples vs. the normal ones — the raw "N found" headline hides
        # that, and anomaly loss is far more damaging than normal loss.
        label_map = dict(zip(label_df["id"].astype(str), label_df["label"].astype(str)))

        def _breakdown(ids):
            anom = sum(1 for i in ids if label_map.get(str(i)) == LABEL_ANOMALY)
            norm = sum(1 for i in ids if label_map.get(str(i)) == LABEL_NORMAL)
            return anom, norm

        found_anom, found_norm = _breakdown(result.found)

        if n_missing == 0:
            message = (
                f"All {n_found} labeled files found ({found_anom} anomaly, {found_norm} normal)"
            )
        else:
            missing_anom, missing_norm = _breakdown(result.missing)
            missing_preview = ", ".join(result.missing[:3])
            if n_missing > 3:
                missing_preview += f" (+{n_missing - 3} more)"
            message = (
                f"{n_found} found ({found_anom} anomaly, {found_norm} normal), "
                f"{n_missing} missing ({missing_anom} anomaly, {missing_norm} normal): "
                f"{missing_preview}"
            )

        return message, result.found_locations, False

    @staticmethod
    def build_labeled_cache(
        found_locations: dict[str, tuple],
        label_path: str,
        source_type: str,
        stale_check: Callable[[], bool] | None = None,
    ) -> str | None:
        """Build a labeled data cache for container sources (Zarr/Cutana).

        Delegates to :meth:`LabeledDataCache.build`.

        Args:
            found_locations: Validated locations from
                :meth:`validate_labels_against_source`.
            label_path: Path to the label CSV.
            source_type: A :class:`DataSourceType` value.
            stale_check: Optional zero-arg callable returning ``True``
                when a newer job has superseded this one.  When set, the
                call bails out (returning ``None``) instead of queuing
                behind a concurrent build — avoids a storm of 60 s
                timeouts when the user drags a slider (one job per
                tick).

        Returns:
            Path to the cache directory, ``None`` if source type is not
            a container type, or ``None`` when *stale_check* signalled
            the job was superseded.

        Raises:
            RuntimeError: If another process holds the cache lock past
                :data:`_LABELED_CACHE_LOCK_TIMEOUT`.
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg

        if source_type not in (DataSourceType.ZARR, DataSourceType.CUTANA):
            return None

        cache_dir = _derive_labeled_cache_path(label_path, cfg.output_dir)
        label_df = pd.read_csv(label_path)
        # Filelock serialises concurrent builds both across threads and
        # across processes — matters when the cache now lives next to the
        # label CSV and a second session may open the same one.  clear()
        # does shutil.rmtree, so racing the zarr directory tree would
        # corrupt one of the builds.
        try:
            with _acquire_or_bail(cache_dir, stale_check) as lock:
                if lock is None:
                    logger.debug(
                        "Labeled cache build cancelled — newer job queued while "
                        "waiting for {}.lock",
                        cache_dir,
                    )
                    return None
                cache = LabeledDataCache(cache_dir)
                # Skip rebuild when the cache is already current (always
                # true for Zarr after the first build; true for Cutana
                # until any extraction-affecting parameter or the label
                # CSV itself changes).
                found_ids = {str(k) for k in found_locations}
                if cache.is_populated and not cache.needs_rebuild(
                    cfg, label_df=label_df, found_ids=found_ids
                ):
                    logger.debug("Labeled cache is up to date — skipping rebuild")
                    cfg.labeled_cache_path = cache_dir
                    return cache_dir
                cache.build(found_locations, label_df, source_type, cfg)
        except Timeout as exc:
            raise RuntimeError(
                f"Timed out after {_LABELED_CACHE_LOCK_TIMEOUT}s waiting for the "
                f"labeled cache lock at {cache_dir}.lock — another process or "
                "session may be rebuilding the same cache.  If no peer is "
                "running, delete the stale .lock file."
            ) from exc

        cfg.labeled_cache_path = cache_dir
        return cache_dir

    @staticmethod
    def update_labeled_cache(new_labels: dict[str, str]) -> None:
        """Update the labeled data cache after gallery labeling.

        Delegates to :meth:`LabeledDataCache.update`.

        Args:
            new_labels: Dict mapping ``id -> csv_label_string``.
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg

        cache_path = cfg.labeled_cache_path
        if not cache_path:
            return

        cache = LabeledDataCache(cache_path)
        if not cache.is_populated:
            return

        with _labeled_cache_lock(cache_path):
            cache.update(new_labels, cfg)

    # ========== Prediction/Evaluation ==========

    @staticmethod
    def evaluate_all_images(top_n: int, progress_callback: Callable | None = None) -> None:
        """Evaluate all images in the prediction search directory.

        Args:
            top_n: Number of top images to keep.
            progress_callback: Optional callback for progress updates.
        """
        BackendInterface._check_session()
        BackendInterface._session.evaluate_all_images(
            top_N=top_n, progress_callback=progress_callback
        )

    @staticmethod
    def get_last_run_skip_stats() -> dict:
        """Return how many chunks/sources were skipped on the last evaluate_all_images.

        Used by scoring screens to display a partial-completion warning when
        some chunks failed (typically: data volume detached mid-run so Cutana
        couldn't reach the FITS tiles).  Stats are reset at the start of each
        ``evaluate_all_images`` call.

        Returns:
            Dict with ``skipped_chunks``, ``skipped_sources``,
            ``total_chunks``, ``total_sources``.
        """
        BackendInterface._check_session()
        session = BackendInterface._session
        return {
            "skipped_chunks": session.last_run_skipped_chunks,
            "skipped_sources": session.last_run_skipped_sources,
            "total_chunks": session.last_run_total_chunks,
            "total_sources": session.last_run_total_sources,
        }

    @staticmethod
    def request_prediction_stop(timeout: float = 15.0) -> None:
        """Request a graceful cancel of the currently running prediction.

        SIGTERMs the prediction subprocess (if any) and sets the flag that
        tells the chunk loop not to spawn the next one.  Idempotent — safe
        to call when no prediction is running.  Callers that care about
        the GPU being free (e.g. retrain) should also ``join`` the thread
        running ``evaluate_all_images`` after calling this.

        Args:
            timeout: Seconds to wait for SIGTERM to take effect before
                escalating to SIGKILL.
        """
        BackendInterface._check_session()
        BackendInterface._session.request_prediction_stop(timeout=timeout)

    # ========== Utilities ==========

    @staticmethod
    def set_log_level(log_level: str) -> None:
        """Set the log level via the backend utility.

        Args:
            log_level: The log level to set (e.g. 'DEBUG', 'INFO').
        """
        BackendInterface._check_session()
        _set_log_level_impl(log_level, BackendInterface._session.cfg)

    @staticmethod
    def set_terminal_output(output_widget: Any) -> None:
        """Set the terminal output widget for logging.

        Args:
            output_widget: The output widget for terminal logging.
        """
        BackendInterface._check_session()
        BackendInterface._session.set_terminal_out(output_widget)

    @staticmethod
    def remember_current_file(filename: str) -> None:
        """Remember the current file by appending it to a CSV.

        Args:
            filename: The filename to remember.
        """
        BackendInterface._check_session()
        BackendInterface._session.remember_current_file(filename)

    @staticmethod
    def remember_file(filename: str) -> None:
        """Remember a file by name, without requiring a catalog index.

        Delegates to ``remember_current_file`` on the session.

        Args:
            filename: The filename to remember.
        """
        BackendInterface._check_session()
        BackendInterface._session.remember_current_file(filename)

    # ========== Prediction Monitoring ==========

    _score_db = None
    _prediction_cache = None

    @staticmethod
    def open_prediction_monitor(db_path: str, search_dir: str) -> None:
        """Open an AnomalyScoreDB and ImageCache for live polling.

        Args:
            db_path: Path to the SQLite prediction database.
            search_dir: Root directory for loading source images.
        """
        BackendInterface._check_session()
        # The DB may live on relocated local scratch whose dir the writer has not
        # created yet; ensure it exists so opening the reader can't race-fail.
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        BackendInterface._score_db = AnomalyScoreDB(db_path)
        BackendInterface._prediction_cache = ImageCache(BackendInterface._session.cfg, search_dir)

    @staticmethod
    def close_prediction_monitor() -> None:
        """Close the score DB and cache.

        Sets the reference to None before closing so that daemon threads
        (poll/prefetch) see None and bail out instead of querying a
        connection that is being closed — avoids a C-level segfault.
        """
        db = BackendInterface._score_db
        BackendInterface._score_db = None
        BackendInterface._prediction_cache = None
        if db is not None:
            db.close()

    @staticmethod
    def get_prediction_count() -> int:
        """Return total number of stored prediction results.

        Returns:
            Row count, or 0 if no DB is open.
        """
        db = BackendInterface._score_db
        if db is None:
            return 0
        return db.get_count()

    @staticmethod
    def get_prediction_results(
        sort_by: str = "score_desc", limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Query prediction results with sorting and pagination.

        Args:
            sort_by: Sort key (e.g. ``"score_desc"``).
            limit: Maximum rows.
            offset: Rows to skip.

        Returns:
            List of result dicts.
        """
        db = BackendInterface._score_db
        if db is None:
            return []
        return db.get_results(sort_by=sort_by, limit=limit, offset=offset)

    @staticmethod
    def get_prediction_score_range() -> tuple[float, float]:
        """Return ``(min_score, max_score)`` across all results.

        Returns:
            Tuple of min and max score.

        Raises:
            ValueError: If no results exist.
        """
        db = BackendInterface._score_db
        if db is None:
            raise ValueError("No prediction DB open")
        return db.get_score_range()

    @staticmethod
    def get_prediction_histogram(
        bins: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute a score histogram from stored results.

        Args:
            bins: Number of histogram bins.

        Returns:
            ``(counts, bin_edges)`` arrays.
        """
        db = BackendInterface._score_db
        if db is None:
            return np.zeros(bins, dtype=np.int64), np.linspace(0, 1, bins + 1)
        return db.get_score_histogram(bins=bins)

    @staticmethod
    def get_prediction_result(filename: str) -> dict | None:
        """Look up a single prediction result by filename.

        Args:
            filename: Source filename.

        Returns:
            Result dict or ``None``.
        """
        db = BackendInterface._score_db
        if db is None:
            return None
        return db.get_result_by_filename(filename)

    @staticmethod
    def load_prediction_image(filename: str) -> np.ndarray | None:
        """Load a source image via the prediction image cache.

        Args:
            filename: Source filename.

        Returns:
            HWC uint8 numpy array, or ``None``.
        """
        if BackendInterface._prediction_cache is None:
            return None
        return BackendInterface._prediction_cache.get_image(filename)

    @staticmethod
    def cache_prediction_image(filename: str, image: np.ndarray) -> None:
        """Manually insert an image into the prediction cache.

        Useful for caching images loaded via alternative paths (e.g.
        Cutana cutouts) so subsequent lookups hit the cache.

        Args:
            filename: Cache key.
            image: HWC uint8 numpy array.
        """
        if BackendInterface._prediction_cache is not None:
            BackendInterface._prediction_cache.put(filename, image)

    @staticmethod
    def clear_image_cache() -> None:
        """Drop all cached images so they are re-loaded with current settings."""
        if BackendInterface._prediction_cache is not None:
            BackendInterface._prediction_cache.clear()

    # ========== Source Scanning ==========

    @staticmethod
    def scan_and_count_sources(folder: str) -> SourceScanResult:
        """Detect source type and count sources in a folder.

        Args:
            folder: Path to the search directory.

        Returns:
            A :class:`SourceScanResult` with type, count, and file list.
        """
        return _scan_and_count_sources(folder)

    @staticmethod
    def detect_and_count_sources(folder: str) -> tuple[DataSourceType, int]:
        """Detect source type and count sources in a folder.

        Args:
            folder: Path to the search directory.

        Returns:
            Tuple of ``(source_type, source_count)``.
        """
        result = _scan_and_count_sources(folder)
        return result.source_type, result.count

    @staticmethod
    def detect_source_channel_count(
        folder: str, source_type: DataSourceType, image_files: list[str] | None = None
    ) -> int | None:
        """Return the per-image channel count of a Zarr store or image folder.

        Delegates to :func:`anomaly_match.data_io.source_scanning.detect_source_channel_count`.

        Args:
            folder: Source directory or single Zarr store.
            source_type: The detected source type.
            image_files: Image filenames for image-folder sources.

        Returns:
            The channel count, or ``None`` when it cannot be read.
        """
        return _detect_source_channel_count(folder, source_type, image_files)

    @staticmethod
    def detect_and_count_prediction_sources(folder: str) -> tuple[DataSourceType, int]:
        """Detect prediction source type and count items in a folder.

        Args:
            folder: Directory to scan, or a direct path to a single store.

        Returns:
            Tuple of ``(source_type, source_count)``.
        """
        return _detect_and_count_prediction_sources(folder)

    @staticmethod
    def is_shipped_default_path(key: str, value: str | None) -> bool:
        """Return whether *value* is the bundled default path for config *key*.

        Delegates to :func:`anomaly_match.utils.get_default_cfg.is_shipped_default_path`.

        Args:
            key: Top-level config key, e.g. ``"label_file"``.
            value: The configured path, or ``None``.

        Returns:
            ``True`` when *value* is the shipped default.
        """
        return _is_shipped_default_path(key, value)

    @staticmethod
    def read_label_map(label_path: str) -> dict[str, str]:
        """Read a label CSV and return an ``{id: label}`` mapping.

        Delegates to :func:`anomaly_match.data_io.source_scanning.read_label_map`,
        which raises ``FileNotFoundError`` for missing paths and ``ValueError``
        for malformed content — callers must guard against the "no label
        selected" UI state before invoking.

        Args:
            label_path: Path to an existing label CSV file.

        Returns:
            Mapping from sample id to label string.
        """
        return _read_label_map(label_path)

    @staticmethod
    def load_preview_samples(
        folder: str,
        source_type: str,
        *,
        image_files: list[str] | None = None,
        source_ids: list[str] | None = None,
        max_items: int = 9,
        offset: int = 0,
        stale_check=None,
    ) -> list[tuple[str, np.ndarray]]:
        """Load sample images for preview display.

        Args:
            folder: Path to the data directory.
            source_type: Detected source type value string.
            image_files: Files to load for image_folder sources.
            source_ids: Source IDs to load for Cutana sources.
            max_items: Maximum number of preview samples to return.
            offset: Skip this many items before loading ``max_items``.
                Used by paged preview galleries (#427).
            stale_check: Optional zero-arg callable returning ``True``
                when this preview request has been superseded.  Forwarded
                into the Cutana fallback loop so older setup-screen jobs
                bail out between catalogues the moment a newer one is
                queued.

        Returns:
            List of ``(name, image_array)`` tuples (HWC uint8).
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg
        return _load_preview_samples(
            cfg,
            folder,
            source_type,
            image_files=image_files,
            source_ids=source_ids,
            max_items=max_items,
            offset=offset,
            stale_check=stale_check,
        )

    @staticmethod
    def load_cutana_preview(
        folder: str, source_ids: list[str], *, max_items: int, stale_check=None
    ) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]] | None]:
        """Load Cutana preview cutouts plus the raws worth holding (issue #501).

        Reads the labeled cache once and returns both the decoded preview images
        and — when the cache fully covered the request — the raw arrays the setup
        screen holds to re-decode on a normalisation change without re-reading.

        Args:
            folder: Source catalogue directory (fallback search root).
            source_ids: Candidate ids, already stratified by label.
            max_items: Cap on preview grid size.
            stale_check: Optional zero-arg callable returning ``True`` when this
                request has been superseded.

        Returns:
            ``(decoded, holdable_raws)``; *holdable_raws* is ``None`` unless the
            cache fully backed the request.
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg
        return _load_cutana_preview(cfg, folder, source_ids, max_items, stale_check=stale_check)

    @staticmethod
    def decode_raw_cutouts(
        raw_items: list[tuple[str, np.ndarray]],
    ) -> list[tuple[str, np.ndarray]]:
        """Apply the current normalisation to held raw cutouts (issue #501).

        Rebuilds ``fitsbolt_cfg`` from the session's current normalisation
        settings and decodes the raw arrays to display-ready HWC uint8.  The
        UI calls this on a normalisation change to refresh the preview from the
        in-memory raws without any cache read.

        Args:
            raw_items: ``(source_id, raw_float32_hwc)`` pairs previously returned
                by :meth:`load_cutana_preview`.

        Returns:
            List of ``(source_id, uint8_hwc)`` tuples in input order.
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg
        return _decode_raw_cutouts(cfg, raw_items)

    @staticmethod
    def load_prediction_db_state(db_path: str) -> tuple[int, dict]:
        """Return the row count and stored metadata for an existing DB.

        The caller uses this to decide whether to present a resume and,
        if so, which stored normalisation settings to apply to the
        setup widgets.  Compatibility comparison lives on the caller
        side — we hand back the raw stored metadata here, not a
        compatible/incompatible verdict.

        ``AnomalyScoreDB`` propagates ``SchemaVersionError`` when the
        file's schema doesn't match; callers handle it by surfacing a
        "delete and re-run" message to the user.

        Args:
            db_path: Path to the ``predictions.db`` file.

        Returns:
            Tuple of ``(count, metadata)``.  Returns ``(0, {})`` if the
            file does not exist.
        """
        if not os.path.exists(db_path):
            return 0, {}
        with AnomalyScoreDB(db_path) as db:
            return db.get_count(), db.get_metadata()

    @staticmethod
    def read_model_normalisation(model_path: str) -> dict:
        """Read the normalisation settings embedded in a model checkpoint.

        Thin wrapper over
        :func:`anomaly_match.data_io.checkpoint_io.read_model_normalisation`.
        Prediction overrides its decode pipeline from the checkpoint, so these
        settings — not a user choice — are what inference runs; the prediction
        setup screen reads them to show a read-only overview.

        Args:
            model_path: Path to the ``.safetensors`` checkpoint.

        Returns:
            Summary dict (see the backend function); normalisation fields are
            ``None`` for checkpoints without embedded metadata.
        """
        return read_model_normalisation(model_path)

    @staticmethod
    def sync_normalisation_from_model(model_path: str) -> bool:
        """Mirror a checkpoint's full normalisation onto the session config.

        The prediction setup screen previews Cutana cutouts before the
        prediction subprocess runs, and the Cutana orchestrator validates that
        ``cfg.fitsbolt_cfg`` and ``cfg.normalisation`` agree.  Mirroring only the
        display-shaped ``cfg.normalisation`` fields (as
        :meth:`read_model_normalisation` feeds) leaves ``cfg.fitsbolt_cfg`` stale
        at the pickled default (CONVERSION_ONLY), so the preview crashes on the
        disagreement guard.  This writes all three normalisation surfaces —
        ``fitsbolt_cfg``, the ``cfg.normalisation`` fields and
        ``channel_combination`` — exactly as the prediction subprocess does via
        :func:`~anomaly_match.data_io.checkpoint_io.sync_normalisation_from_checkpoint`,
        keeping the preview decode in lock-step with inference.

        Args:
            model_path: Path to the ``.safetensors`` checkpoint.

        Returns:
            ``True`` when the checkpoint carried an embedded fitsbolt config and
            it was applied; ``False`` for a legacy checkpoint without one (the
            caller keeps showing the read-only overview and lets the prediction
            subprocess raise the definitive "retrain to embed normalisation"
            error).
        """
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg
        model_fitsbolt_cfg, model_channel_combination = read_checkpoint_normalisation(model_path)
        return sync_normalisation_from_checkpoint(
            cfg, model_fitsbolt_cfg, model_channel_combination
        )

    @staticmethod
    def load_cutana_cutout(source_id: str) -> np.ndarray | None:
        """Create a single cutout for a streaming source via Cutana.

        Uses the session config to locate the catalogue and create
        a cutout on demand.

        Args:
            source_id: The source identifier from the prediction DB.

        Returns:
            HWC uint8 numpy array, or ``None`` if loading fails.
        """
        BackendInterface._check_session()
        return load_single_cutout(BackendInterface._session.cfg, source_id)

    @staticmethod
    def load_cutana_batch(source_ids: list[str]) -> dict[str, np.ndarray]:
        """Create cutouts for multiple sources in a single batch.

        Much more efficient than calling ``load_cutana_cutout`` per source
        since only one Cutana orchestrator is created.

        Args:
            source_ids: List of source identifiers from the prediction DB.

        Returns:
            Dict mapping source_id to HWC uint8 numpy array.  Gallery
            callers don't pass ``stale_check``, so the underlying
            :class:`BatchCutoutResult` is always non-partial here and we
            unwrap straight to the cutouts dict.
        """
        if not source_ids:
            return {}
        BackendInterface._check_session()
        return load_batch_cutouts(BackendInterface._session.cfg, source_ids).cutouts

    # ========== Container Image Loading ==========

    @staticmethod
    def load_container_images(filenames: list[str]) -> dict[str, np.ndarray]:
        """Load and decode images from Zarr containers by filename.

        Handles synthetic filenames from the prediction DB:
        Zarr: parses ``prefix__image_NNNNNN`` to extract array indices.
        Pixels are decoded with the session's normalisation and
        ``channel_combination`` via :func:`decode_zarr_gallery_images`.

        Args:
            filenames: List of filenames from the prediction DB.

        Returns:
            Dict mapping filename to decoded HWC image.
        """
        if not filenames:
            return {}
        BackendInterface._check_session()
        cfg = BackendInterface._session.cfg
        search_dir = cfg.prediction_search_dir
        if not search_dir or not os.path.isdir(search_dir):
            return {}

        raw = _load_images_from_zarr(search_dir, set(filenames))
        return decode_zarr_gallery_images(raw, cfg)

    @staticmethod
    def decode_single_with_normalisation(
        filename: str,
        source_type: DataSourceType,
        norm_overrides: dict,
        *,
        search_dir: str,
    ) -> np.ndarray | None:
        """Re-decode a single source under one-shot normalisation overrides.

        Used by :class:`ImageDetailScreen` to redraw the displayed image
        after the user tweaks normalisation knobs without leaving the
        screen.  The session config is **never** mutated — overrides go
        into a deep-copy so subsequent training/prediction runs keep the
        committed cfg (#443).  ``cfg.fitsbolt_cfg`` is cleared before
        the call so the loader rebuilds it from the override values
        rather than re-using the stale build.

        Args:
            filename: Source identifier (filename for image-folder /
                Zarr, source-id for Cutana) as stored in the prediction
                DB.
            source_type: Which loader path to use for the re-decode.
            norm_overrides: A mapping of ``cfg.normalisation`` field
                names to their override values, e.g. as produced by
                :meth:`NormalisationConfigWidget.get_normalisation_config`.
            search_dir: Folder holding the catalogues / image files for
                this source.  Passed through ``_detail_context`` from
                whichever screen launched the detail view so we don't
                depend on ``cfg.prediction_search_dir`` (unset until
                Start is clicked in the training-setup flow) or
                ``cfg.data_dir`` (stale until then).

        Returns:
            HWC uint8 array under the override settings.  For Cutana
            sources, ``None`` when the source id is missing from every
            catalogue (propagated from
            :func:`~anomaly_match.prediction.cutana_loader.load_single_cutout`).

        Raises:
            NotImplementedError: ``source_type`` is :attr:`DataSourceType.ZARR`.
                Zarr stores hold already-post-fitsbolt bytes; live
                re-decode is deferred (issue #466).  The detail screen
                catches Zarr before it reaches the backend.
        """
        BackendInterface._check_session()
        cfg = copy.deepcopy(BackendInterface._session.cfg)
        for key, value in norm_overrides.items():
            if hasattr(cfg.normalisation, key):
                setattr(cfg.normalisation, key, value)
        # Drop the cached fitsbolt_cfg so the loader rebuilds from the
        # override normalisation values; otherwise the stale build wins
        # and the override is silently ignored.
        cfg.fitsbolt_cfg = None

        # Pin the loader-visible search dir to whatever the caller
        # passed in — the cfg copy's prediction_search_dir may be unset
        # (training-setup flow before Start) or stale, and the
        # cutana / image-folder loaders both key off this attribute.
        cfg.prediction_search_dir = search_dir

        if source_type == DataSourceType.CUTANA:
            return load_single_cutout(cfg, filename)
        if source_type == DataSourceType.IMAGE_FOLDER:
            filepath = filename if os.path.isabs(filename) else os.path.join(search_dir, filename)
            img = load_and_process_single_wrapper(
                filepath, cfg, desc="detail-redecode", show_progress=False
            )
            # The loader returns multi-channel images channel-first (CHW)
            # but PreviewWidget expects channel-last (HWC).  Detect the CHW
            # layout by matching the leading axis against the configured
            # output channel count rather than a fixed "<= 4" guess — users
            # can request more than 4 channels (multispectral), so a magic
            # cap would silently fail to rotate those.  Same approach as
            # :func:`~anomaly_match.data_io.container_loaders.decode_cutana_raw_images`.
            n_channels = cfg.normalisation.n_output_channels
            if img.ndim == 3 and img.shape[0] == n_channels and img.shape[2] != n_channels:
                img = img.transpose(1, 2, 0)
            return img
        # Zarr: stored cutouts are already post-fitsbolt; re-decoding
        # them with new normalisation requires re-reading the upstream
        # FITS, which the Zarr filename mapping doesn't expose.  Tracked
        # in issue #466.  The detail screen surfaces this to the user
        # via :meth:`ImageDetailScreen._schedule_redecode`, so reaching
        # the backend with a Zarr source_type is itself a bug.
        raise NotImplementedError(
            f"Detail re-decode is not supported for source_type={source_type.value}"
        )

    @staticmethod
    def load_single_native_cutout(
        source_id: str, *, search_dir: str, cutout_padding_factor: float
    ) -> np.ndarray | None:
        """Read one native-resolution raw cutout for the detail view.

        The detail screen holds this normalisation-independent per-band float
        array and re-normalises it in-process on every widget change via
        :meth:`decode_raw_with_normalisation`, so the catalogue/tile is read
        once per (source, zoom) (issue #501).  The cutout comes back at its true
        native pixel size (Cutana ``do_only_cutout_extraction``), so the user
        inspects real pixels rather than a fixed-resolution resample.

        Args:
            source_id: The Cutana source id stored in the prediction DB.
            search_dir: Catalogue directory passed through the detail context —
                the cfg's ``prediction_search_dir`` may be unset (training-setup
                flow) or stale, exactly as for
                :meth:`decode_single_with_normalisation`.
            cutout_padding_factor: The "Cutout Zoom-out" the widget currently
                shows.  Unlike the normalisation knobs this changes the *extracted*
                sky window (and hence the native pixel size), so it must be applied
                at extraction time — re-normalising a held raw can't widen it.

        Returns:
            HWC float32 array (one channel per FITS band) at native resolution,
            or ``None`` when the source can't be located or Cutana produces no
            cutout.
        """
        BackendInterface._check_session()
        cfg = copy.deepcopy(BackendInterface._session.cfg)
        cfg.prediction_search_dir = search_dir
        cfg.normalisation.cutout_padding_factor = cutout_padding_factor
        return load_single_native_cutout(cfg, source_id)

    @staticmethod
    def decode_raw_with_normalisation(
        raw_image: np.ndarray, norm_overrides: dict
    ) -> np.ndarray | None:
        """Apply one-shot normalisation overrides to a held raw cutout.

        In-memory counterpart to :meth:`decode_single_with_normalisation` for
        Cutana: instead of re-reading the catalogue and re-extracting, it
        re-normalises the raw float32 array the detail screen already holds.
        The session config is never mutated — overrides go into a deep-copy
        (the ``hasattr`` guard mirrors :meth:`decode_single_with_normalisation`)
        and the actual decode is delegated to the same backend re-decode the
        #501 preview path uses, so the result is identical to a fresh
        :func:`load_single_cutout` at the same settings.

        Args:
            raw_image: HWC float32 cutout from :meth:`load_single_native_cutout`.
            norm_overrides: ``cfg.normalisation`` field overrides, as produced
                by :meth:`NormalisationConfigWidget.get_normalisation_config`.

        Returns:
            HWC uint8 array under the override settings, or ``None`` if the
            decode produced nothing.
        """
        BackendInterface._check_session()
        cfg = copy.deepcopy(BackendInterface._session.cfg)
        for key, value in norm_overrides.items():
            if hasattr(cfg.normalisation, key):
                setattr(cfg.normalisation, key, value)
        cfg.fitsbolt_cfg = None
        decoded = _decode_raw_cutouts(cfg, [("", raw_image)])
        if not decoded:
            return None
        return decoded[0][1]

    @staticmethod
    def get_version() -> str:
        """Get the anomaly_match version.

        Returns:
            str: The version string.
        """
        return anomaly_match.__version__

    @staticmethod
    def get_git_commit() -> str | None:
        """Get the git commit the UI is running from.

        Returns:
            The abbreviated commit hash, or ``None`` when it can't be resolved
            (e.g. an installed wheel with no ``.git`` directory).
        """
        return anomaly_match.__commit__


# search_dir -> {filename: (zarr_path, index)}, and the set of store paths
# already folded into that map. Chunked prediction output has one
# `batch_*/images.zarr` + metadata parquet per batch, so re-deriving
# filenames for every store on every gallery page is hundreds of parquet
# reads per page on NFS. Keyed by search_dir (fixed for a prediction run) and
# extended incrementally so stores written mid-run are still picked up.
_ZARR_NAME_INDEX_CACHE: dict[str, dict[str, tuple[str, int]]] = {}
_ZARR_NAME_INDEX_KNOWN_STORES: dict[str, set[str]] = {}
# Names still missing after a rebuild, with the monotonic time they were given
# up on. Without this, a stale DB row (e.g. from a deleted batch) would re-read
# every parquet on every gallery page. A store written later needs no reset:
# _get_zarr_name_index adds its names incrementally, so they resolve without a
# rebuild. Entries expire after _ZARR_NAME_RETRY_SECONDS so a parquet read that
# failed during the rebuild too (transient NFS error) does not blank those
# images for the whole kernel.
_ZARR_NAME_INDEX_RETRIED: dict[str, dict[str, float]] = {}
_ZARR_NAME_RETRY_SECONDS = 300.0
# Gallery prefetch and magnify call load_container_images from executor
# threads; the caches above are mutated in place, so serialise access.
_ZARR_NAME_INDEX_LOCK = threading.Lock()


def _get_zarr_name_index(search_dir: str) -> dict[str, tuple[str, int]]:
    """Return the filename -> (store_path, index) map for *search_dir*.

    Builds the map once per search_dir by reading each store's metadata
    parquet, then only reads stores not seen before on later calls. Callers
    must hold ``_ZARR_NAME_INDEX_LOCK``.

    Returns:
        Mapping from filename to the (zarr store path, array index) that
        produces it.
    """
    name_index = _ZARR_NAME_INDEX_CACHE.setdefault(search_dir, {})
    known_stores = _ZARR_NAME_INDEX_KNOWN_STORES.setdefault(search_dir, set())

    for _name, zarr_path in _find_prediction_zarr_stores(search_dir):
        if zarr_path in known_stores:
            continue
        names = _derive_zarr_filenames(zarr_path)
        if not names:
            # Empty means the store couldn't be opened or read this time
            # (e.g. a transient NFS hiccup) — don't mark it known, so the
            # next call retries instead of permanently blanking it for the
            # rest of the kernel's lifetime.
            continue
        known_stores.add(zarr_path)
        for idx, fn in enumerate(names):
            name_index[fn] = (zarr_path, idx)

    return name_index


def _load_images_from_zarr(search_dir: str, wanted: set[str]) -> dict[str, np.ndarray]:
    """Load images from Zarr containers within *search_dir*.

    Args:
        search_dir: Directory containing Zarr stores.
        wanted: Set of filenames to load.

    Returns:
        Dict mapping filename to the untouched store pixels; callers decode them.
    """
    results: dict[str, np.ndarray] = {}
    if not wanted:
        return results

    with _ZARR_NAME_INDEX_LOCK:
        name_index = _get_zarr_name_index(search_dir)
        retried = _ZARR_NAME_INDEX_RETRIED.setdefault(search_dir, {})
        now = time.monotonic()
        missing = {
            fn
            for fn in wanted - name_index.keys()
            if fn not in retried or now - retried[fn] > _ZARR_NAME_RETRY_SECONDS
        }
        if missing:
            # A wanted name missing from the index means some store was marked
            # known with wrong names — e.g. its metadata parquet failed to read
            # once and derive_filenames fell back to generated names, which are
            # non-empty so _get_zarr_name_index couldn't detect the failure.
            # Drop the known-stores cache for this search_dir and rebuild once.
            # A store rewritten in place is picked up the same way, once any
            # earlier give-up on its names has expired.
            _ZARR_NAME_INDEX_KNOWN_STORES.pop(search_dir, None)
            name_index = _get_zarr_name_index(search_dir)
            given_up = missing - name_index.keys()
            if given_up:
                logger.warning(
                    "{} gallery name(s) match no Zarr store in {} (e.g. {}); retrying in {:.0f} s",
                    len(given_up),
                    search_dir,
                    sorted(given_up)[:3],
                    _ZARR_NAME_RETRY_SECONDS,
                )
                retried.update(dict.fromkeys(given_up, now))

        by_store: dict[str, list[tuple[str, int]]] = {}
        for fn in wanted:
            location = name_index.get(fn)
            if location is not None:
                by_store.setdefault(location[0], []).append((fn, location[1]))

    for zarr_path, matches in by_store.items():
        try:
            root = zarr.open_group(zarr_path, mode="r")
            arr = root["images"]
            for fn, idx in matches:
                # Untouched store pixels: load_container_images decodes them
                # exactly as ZarrSource does for training.
                results[fn] = np.asarray(arr[idx])
        except Exception as exc:
            logger.warning("Failed to read zarr store {}: {}", zarr_path, exc)
            continue

    return results
