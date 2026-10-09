#   Copyright (c) European Space Agency, 2025.
#
#   This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#   is part of this source code package. No part of the package, including
#   this file, may be copied, modified, propagated, or distributed except according to
#   the terms contained in the file 'LICENCE.txt'.
"""On-demand cutout creation via Cutana for streaming sources.

Used by the prediction UI to create cutouts for the image detail
view and preview gallery when the original cutout was created in-memory
during prediction and no file exists on disk.
"""

from __future__ import annotations

import copy
import os
import sqlite3
from collections.abc import Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotmap import DotMap
from loguru import logger

from anomaly_match.data_io.container_loaders import apply_channel_combination_to_cutana_image
from anomaly_match.data_io.labeled_data_cache import _read_parquet_filtered
from anomaly_match.data_io.load_images import get_fitsbolt_config
from anomaly_match.datasets.cutana_source import (
    _is_cutana_catalogue,
    _iter_batch_items,
    extract_cutouts_direct,
)
from anomaly_match.image_processing.image_utils import ensure_uint8_hwc
from anomaly_match.prediction.anomaly_score_db import AnomalyScoreDB, SchemaVersionError
from anomaly_match.prediction.db_location import prediction_db_path

# Concurrency cap for the per-catalogue parquet/CSV scan in
# :func:`load_batch_cutouts`.  IO-bound, so threading is fine; size
# matches typical Datalabs disk parallelism (#444).  Capped so we
# don't drown the UI's host process when a search dir has many
# catalogues but the actual hits live in only one.
_CATALOGUE_SCAN_WORKERS = 8

# Catalogue-index lookups (#506) open the DB read-only, so they take no write
# lock; a short busy_timeout still lets a momentarily-locked WAL settle without
# the gallery thread stalling on a slow scoring writer.
_INDEX_READ_BUSY_TIMEOUT_MS = 2000


@dataclass
class BatchCutoutResult:
    """Cutouts loaded by :func:`load_batch_cutouts`, plus the bail-out flag.

    Attributes:
        cutouts: Map from ``source_id`` to HWC uint8 cutout.  Missing sources
            are simply absent from the dict.
        partial: ``True`` when ``stale_check`` reported the request was
            superseded mid-loop and at least one catalogue was skipped.
            Callers that require complete results (e.g. cache builds, training
            warmup) MUST treat ``cutouts`` as untrustworthy when this is set.
            UI previews can ignore it — a missing thumbnail just renders as a
            placeholder until the next refresh.
    """

    cutouts: dict[str, np.ndarray] = field(default_factory=dict)
    partial: bool = False


def _find_source_in_catalogues(
    cfg: DotMap, search_dir: str, source_id: str
) -> tuple[Any, str | None]:
    """Locate the catalogue row for *source_id*, consulting the #506 index.

    Single-source counterpart to the per-catalogue scan in
    :func:`load_batch_cutouts`: when the predictions DB records which catalogue
    holds the source it is tried first (and the rest kept as fallback in case
    the folder was rebuilt), so opening a detail view no longer globs every
    catalogue.  A first-time hit is back-filled into the index.  Without the
    index it falls back to scanning all catalogues in directory order.

    Args:
        cfg: Full prediction config (locates the catalogues and the index DB).
        search_dir: Catalogue directory (``cfg.prediction_search_dir``).
        source_id: The source identifier stored in the prediction DB.

    Returns:
        ``(row_df, catalogue_path)`` for the matching one-row DataFrame, or
        ``(None, None)`` when the source can't be located (a warning is logged).
    """
    if not search_dir or not os.path.isdir(search_dir):
        logger.warning("prediction_search_dir is not set or does not exist: {}", search_dir)
        return None, None

    cat_files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.lower().endswith((".csv", ".parquet"))
    ]
    if not cat_files:
        logger.warning("No catalogue files (.csv/.parquet) found in {}", search_dir)
        return None, None

    # #506 index: try the recorded catalogue first, then the rest as fallback.
    scan_order = cat_files
    use_index = _is_prediction_search_dir(cfg, search_dir)
    recorded_cat = None
    if use_index:
        index_map = _run_on_index_db(
            cfg,
            write=False,
            operation=lambda db: db.get_catalogue_map([str(source_id)]),
            default={},
        )
        recorded_cat = index_map.get(str(source_id))
        if recorded_cat and recorded_cat in set(cat_files):
            scan_order = [recorded_cat] + [c for c in cat_files if c != recorded_cat]

    for cat_file in scan_order:
        row_df = _find_source_row(cat_file, source_id)
        if row_df is not None:
            # Back-fill the index when the origin isn't already recorded so the
            # next open jumps straight to this catalogue (best-effort).
            if use_index and recorded_cat != cat_file:
                _run_on_index_db(
                    cfg,
                    write=True,
                    operation=lambda db: db.record_catalogues({str(source_id): cat_file}),
                    default=None,
                )
            return row_df, cat_file

    logger.warning(
        "Source {} not found in any of the {} catalogue file(s) under {} — "
        "it cannot be re-decoded for the detail view.",
        source_id,
        len(cat_files),
        search_dir,
    )
    return None, None


def load_single_cutout(cfg: DotMap, source_id: str) -> np.ndarray | None:
    """Create a single normalised cutout for a streaming source using Cutana.

    Locates *source_id* in the catalogues under ``cfg.prediction_search_dir``
    (via the #506 catalogue index when available) and runs Cutana to produce a
    display-ready cutout under the config's normalisation.

    Args:
        cfg: Full prediction config with normalisation settings.
        source_id: The source identifier stored in the prediction DB.

    Returns:
        HWC uint8 numpy array, or ``None`` if loading fails.
    """
    row_df, cat_file = _find_source_in_catalogues(cfg, cfg.prediction_search_dir, source_id)
    if row_df is None:
        return None
    image = _create_cutout(cfg, row_df)
    if image is None:
        # Found but no cutout (distinct from "not found"): surface source id +
        # method so a detail-view recurrence is traceable.
        logger.warning(
            "Source {} found in {} but produced no cutout under normalisation {} — "
            "tile unreachable, no data, or decode error.",
            source_id,
            os.path.basename(cat_file),
            cfg.normalisation.normalisation_method,
        )
    return image


def load_single_native_cutout(cfg: DotMap, source_id: str) -> np.ndarray | None:
    """Extract one cutout at its *native* pixel size for the detail view.

    Returns a normalisation-independent per-band float cutout the detail view
    re-normalises in-process (so retuning normalisation never re-reads the
    catalogue/tile).  Cutana is run with ``do_only_cutout_extraction`` so it
    skips the resize to a fixed target resolution — the cutout comes back at its
    true native pixel size (e.g. 14x14 for a small Euclid source), letting the
    user inspect real pixels rather than an interpolated resample.

    Args:
        cfg: Full prediction config (locates the catalogues / index DB).
        source_id: The source identifier stored in the prediction DB.

    Returns:
        HWC float32 array (one channel per FITS band) at native resolution, or
        ``None`` if the source can't be located or Cutana produces no cutout.
    """
    row_df, cat_file = _find_source_in_catalogues(cfg, cfg.prediction_search_dir, source_id)
    if row_df is None:
        return None

    native_cfg = copy.deepcopy(cfg)
    native_cfg.fitsbolt_cfg = None
    native_cfg = get_fitsbolt_config(native_cfg)
    try:
        batches = extract_cutouts_direct(row_df, native_cfg, do_only_cutout_extraction=True)
        items = (item for batch in batches for item in _iter_batch_items(batch))
        first = next(items, None)
        if first is None:
            logger.warning(
                "Source {} found in {} but Cutana produced no native cutout — "
                "tile unreachable or no data at that position.",
                source_id,
                os.path.basename(cat_file),
            )
            return None
        _sid, native_img = first
        logger.debug("Native cutout for {}: shape={}", source_id, native_img.shape)
        return native_img
    except Exception as exc:
        logger.warning("Native cutout extraction failed for source {}: {}", source_id, exc)
        return None


def load_sample_cutouts(
    cfg: DotMap, search_dir: str, n_samples: int = 9
) -> list[tuple[str, np.ndarray]]:
    """Create a few sample cutouts from the catalogue for preview.

    Args:
        cfg: Full prediction config with normalisation settings.
        search_dir: Path to the directory containing catalogue files.
        n_samples: Maximum number of sample cutouts to create.

    Returns:
        List of ``(source_id, image_array)`` tuples. May be shorter
        than *n_samples* if some cutouts fail to load.
    """
    cat_files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.lower().endswith((".csv", ".parquet"))
        and _is_cutana_catalogue(Path(os.path.join(search_dir, f)))
    ]
    if not cat_files:
        logger.warning("No Cutana catalogue files found in {}", search_dir)
        return []

    try:
        cat_file = cat_files[0]
        if cat_file.endswith(".parquet"):
            df = pd.read_parquet(cat_file)
        else:
            df = pd.read_csv(cat_file)

        n = min(n_samples, len(df))
        if n == 0:
            return []

        sample_df = df.sample(n=n, random_state=42)

        id_col = _find_id_column(sample_df)

        # Use batch mode for efficiency (single orchestrator call)
        batch_results = _create_batch_cutouts(cfg, sample_df, id_col)

        results = []
        for _, row in sample_df.iterrows():
            sid = str(row[id_col])
            if sid in batch_results:
                results.append((sid, batch_results[sid]))

        return results

    except Exception:
        logger.opt(exception=True).warning("Cutana sample cutouts failed")
        return []


def load_batch_cutouts(
    cfg: DotMap,
    source_ids: list[str],
    *,
    search_dir: str | None = None,
    stale_check=None,
) -> BatchCutoutResult:
    """Create cutouts for multiple sources by scanning catalogues in parallel.

    Dispatches one worker per catalogue file across a thread pool
    (:func:`_scan_one_catalogue`).  Each worker reads its catalogue,
    filters to the requested ids, and builds cutouts for the matching
    rows in a single batch orchestrator call.  The parent thread merges
    results as workers complete and cancels queued workers once every
    requested id has resolved, so fast catalogues that hold the wanted
    ids return before a slow read finishes.  Far more efficient than
    calling ``load_single_cutout`` per source, and the parallel scan
    keeps one slow parquet read from freezing the whole walk (#444).

    Args:
        cfg: Full prediction config with normalisation settings.
        source_ids: List of source identifiers from the prediction DB.
        search_dir: Directory containing catalogue files.  Falls back to
            ``cfg.prediction_search_dir`` when not provided.
        stale_check: Optional zero-arg callable returning ``True`` when
            this request has been superseded by a newer one.  Checked
            once per catalogue so a stale preview bails out without
            driving through all N parquets.  Without this, five rapid
            norm changes caused five overlapping ``load_batch_cutouts``
            runs across the same 54 catalogues, deadlocking pyarrow on
            contested memmap'd reads.

    Returns:
        :class:`BatchCutoutResult` with the loaded cutouts dict plus a
        ``partial`` flag.  ``partial=True`` means the scan did not run to
        completion over every catalogue — either ``stale_check`` fired
        and the remaining workers were cancelled, or one or more
        catalogue reads raised.  In both cases the cutouts dict may be
        missing entries, so consumers that require complete results must
        inspect this flag rather than treat the dict as authoritative.
    """
    if not source_ids:
        return BatchCutoutResult()

    if search_dir is None:
        search_dir = cfg.prediction_search_dir
    if not search_dir or not os.path.isdir(search_dir):
        logger.warning("prediction_search_dir is not set or does not exist: {}", search_dir)
        return BatchCutoutResult()

    # Filter to actual Cutana catalogues (must have SourceID + fits_file_paths)
    cat_files = [
        os.path.join(search_dir, f)
        for f in os.listdir(search_dir)
        if f.lower().endswith((".csv", ".parquet"))
        and _is_cutana_catalogue(Path(os.path.join(search_dir, f)))
    ]
    if not cat_files:
        logger.warning("No catalogue files found in {} for batch cutouts", search_dir)
        return BatchCutoutResult()

    # Walk every discovered catalogue — a single source_ids list may span
    # multiple catalogue files, and picking only the first one would drop
    # matches in the rest (previously caused the setup-screen preview to
    # never show anomalies when anomaly IDs happened to live in a later
    # catalogue file).  Reads run in parallel because the bottleneck is
    # IO, not CPU: when a concurrent prediction subprocess saturates disk
    # bandwidth on one catalogue, that file alone could take minutes —
    # serialising the scan extended UI freezes to ~5 minutes (#444).
    # Parallelism turns worst-case ``n_cat × slow_read`` into
    # ``ceil(n_cat / workers) × slow_read``, and fast catalogues that
    # hold the requested ids return immediately so the user sees results
    # before the slow read finishes.
    wanted = {str(s) for s in source_ids}

    # Catalogue-origin index (#506): when the predictions DB already records
    # which catalogue holds each requested source, scan only those files
    # instead of every catalogue on the page.  The index is keyed to the
    # prediction run, so only the prediction gallery (search_dir ==
    # prediction_search_dir) uses it; other callers — e.g. the training-setup
    # preview, whose search_dir is the training folder — fall through to the
    # full scan.
    cat_files_to_scan = cat_files
    # Origins already recorded for the requested ids; used both to narrow the
    # scan and, at the end, to write back only genuinely-new mappings.
    recorded: dict[str, str] = {}
    use_index = _is_prediction_search_dir(cfg, search_dir)
    if use_index:
        cat_files_set = set(cat_files)
        index_map = _run_on_index_db(
            cfg,
            write=False,
            operation=lambda db: db.get_catalogue_map(list(wanted)),
            default={},
        )
        # Drop ids whose recorded catalogue no longer exists (folder changed):
        # they must be relocated by a full scan, not pinned to a stale path.
        recorded = {sid: cat for sid, cat in index_map.items() if cat in cat_files_set}
        if recorded and recorded.keys() >= wanted:
            known = set(recorded.values())
            cat_files_to_scan = [c for c in cat_files if c in known]
            logger.debug(
                "Catalogue index hit: scanning {}/{} catalogue(s) for {} id(s)",
                len(cat_files_to_scan),
                len(cat_files),
                len(wanted),
            )

    results: dict[str, np.ndarray] = {}
    # source_id -> catalogue path it was found in, recorded back into the DB so
    # the next page load can skip straight to the right catalogue.
    origin: dict[str, str] = {}
    n_cat = len(cat_files_to_scan)
    workers = min(_CATALOGUE_SCAN_WORKERS, n_cat)
    partial = False
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cutana-cat") as ex:
            # Each worker gets its own immutable snapshot of `wanted` taken
            # at submit time.  The parent thread is the sole writer to
            # `wanted` and `results` (via the `as_completed` loop below),
            # so no locks are needed — workers only read their snapshot
            # and return a result dict by value.  In-flight workers may do
            # redundant matching for ids a peer has already produced, but
            # that's harmless and avoids the lock overhead.
            futures = {
                ex.submit(
                    _scan_one_catalogue,
                    cat_file,
                    i,
                    n_cat,
                    cfg,
                    set(wanted),
                    stale_check,
                ): cat_file
                for i, cat_file in enumerate(cat_files_to_scan, start=1)
            }
            for fut in as_completed(futures):
                if stale_check is not None and stale_check():
                    logger.info("Preview load superseded — cancelling catalogue scan")
                    partial = True
                    for f in futures:
                        f.cancel()
                    break
                try:
                    cat_results = fut.result()
                except CancelledError:
                    # A future cancelled by a previous iteration of this
                    # loop still surfaces here via `as_completed`; its
                    # `.result()` raises CancelledError.  Skip it so we
                    # don't spuriously flip `partial` or log a misleading
                    # "Catalogue read failed".
                    continue
                except Exception as exc:
                    logger.warning("Catalogue read failed: {}", exc)
                    # A skipped catalogue means the cutouts dict cannot be
                    # treated as authoritative — cache builds and warmup must
                    # see partial=True so they don't lock in an empty result.
                    partial = True
                    continue
                if not cat_results:
                    continue
                results.update(cat_results)
                for sid in cat_results:
                    origin[sid] = futures[fut]
                wanted.difference_update(cat_results.keys())
                if not wanted:
                    # All requested ids found — cancel queued workers so
                    # we don't keep paying for unrelated parquet reads.
                    # Workers already running will finish (their reads
                    # may even contribute extra hits, which is harmless).
                    for f in futures:
                        f.cancel()
                    break

        # Best-effort: remember where each source lived so later pages skip
        # the full scan.  Only write origins the index doesn't already hold —
        # on an index hit every origin is already recorded, so re-writing them
        # would open a contended write transaction on every page turn for no
        # gain.  Never let an index write break the preview.
        if use_index:
            new_origins = {sid: cat for sid, cat in origin.items() if recorded.get(sid) != cat}
            if new_origins:
                _run_on_index_db(
                    cfg,
                    write=True,
                    operation=lambda db: db.record_catalogues(new_origins),
                    default=None,
                )
        return BatchCutoutResult(cutouts=results, partial=partial)

    except Exception as exc:
        logger.warning("Batch cutout loading failed: {}", exc)
        # An exception mid-walk leaves the same partial-truth situation as a
        # stale_check bail — flag it so callers don't treat the dict as
        # complete.
        return BatchCutoutResult(cutouts=results, partial=True)


def _is_prediction_search_dir(cfg: DotMap, search_dir: str) -> bool:
    """Whether *search_dir* is the run's prediction search directory (#506).

    The catalogue-origin index is keyed to ``cfg.prediction_search_dir``, so only
    loads over that directory may consult or write it; the training-setup preview
    (a different folder) must fall through to the full scan.

    Returns:
        ``True`` when *search_dir* resolves to ``cfg.prediction_search_dir``.
    """
    prediction_dir = cfg.prediction_search_dir
    if not prediction_dir or not search_dir:
        return False
    return os.path.normpath(search_dir) == os.path.normpath(str(prediction_dir))


@lru_cache(maxsize=64)
def _warn_index_fault_once(message: str) -> None:
    """Warn that the catalogue index is unavailable, deduplicated per message.

    The index is best-effort (the loader still works via the full scan), but a
    persistent fault would otherwise warn on every gallery page. ``lru_cache``
    logs the first occurrence of each distinct message and no-ops repeats,
    avoiding spam without a module-level flag.
    """
    logger.warning(message)


def _run_on_index_db(
    cfg: DotMap, *, write: bool, operation: Callable[[AnomalyScoreDB], Any], default: Any
) -> Any:
    """Open the predictions DB for the catalogue index and run *operation*.

    Best-effort and shared by the read and back-fill helpers (#506).  A missing
    DB, incompatible schema, or transient WAL lock from the live scoring writer
    yields *default* so the loader falls back to the full catalogue scan; an
    unexpected fault warns once.  Reads open ``read_only`` (no schema-version
    write lock) and writes give up immediately under contention rather than
    block the gallery thread.

    Args:
        cfg: Prediction config locating the predictions DB.
        write: ``True`` for the back-fill write, ``False`` for the lookup.
        operation: Callable invoked with the open DB; its result is returned.
        default: Value returned when the DB is unavailable or the op fails.

    Returns:
        ``operation``'s result, or *default* on any unavailability/fault.
    """
    # Locating the DB is config-level: a cfg that can't resolve a predictions
    # DB (e.g. a minimal/headless cfg with no output dir) simply has no index,
    # so degrade to the full scan rather than propagate a path-resolution error.
    try:
        db_path = prediction_db_path(cfg)
    except Exception as exc:
        logger.debug("Catalogue index path unresolved; full scan fallback: {}", exc)
        return default
    if not os.path.exists(db_path):
        return default

    try:
        # Writers give up immediately (busy_timeout 0) so a back-fill never
        # blocks the gallery behind the scoring writer; reads open read-only
        # and take no write lock at all.
        with AnomalyScoreDB(
            db_path,
            read_only=not write,
            busy_timeout_ms=0 if write else _INDEX_READ_BUSY_TIMEOUT_MS,
        ) as db:
            return operation(db)
    except SchemaVersionError as exc:
        # An incompatible-schema DB silently costs the catalogue index and
        # drops us to a full scan; say so once rather than leave it unexplained.
        _warn_index_fault_once(f"Catalogue index unavailable (full scan fallback): {exc}")
        return default
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower():
            # Expected under concurrent scoring — the back-fill just lands on a
            # later page; nothing to warn about.
            logger.debug("Catalogue index busy (scoring writer active): {}", exc)
        else:
            _warn_index_fault_once(f"Catalogue index unavailable (full scan fallback): {exc}")
        return default
    except sqlite3.Error as exc:
        # Other DB-layer errors are best-effort failures: degrade to the full
        # scan and warn once.  A non-DB exception (e.g. a logic bug in the
        # operation) is deliberately *not* caught here so it surfaces instead
        # of being permanently masked by the once-per-message warn cache.
        _warn_index_fault_once(f"Catalogue index unavailable (full scan fallback): {exc}")
        return default


def _scan_one_catalogue(
    cat_file: str,
    index: int,
    total: int,
    cfg: DotMap,
    wanted: set[str],
    stale_check: Callable[[], bool] | None,
) -> dict[str, np.ndarray]:
    """Read one catalogue and build cutouts for any requested ids.

    Runs on the catalogue-scan thread pool inside
    :func:`load_batch_cutouts`.

    Args:
        cat_file: Path to the parquet/CSV catalogue to read.
        index: 1-based catalogue index for the spinner-tap log line.
        total: Total catalogue count for the same log line.
        cfg: Prediction config forwarded to :func:`_create_batch_cutouts`.
        wanted: Immutable snapshot of the source ids the parent was
            searching for at submit time.  Owned by this worker; the
            parent never mutates it.  Workers may do redundant matching
            for ids a peer has already produced, which is harmless.
        stale_check: Optional zero-arg callable returning ``True`` when
            this preview request has been superseded.  Checked before
            the catalogue read.

    Returns:
        Mapping ``{source_id: HWC uint8 array}`` for the rows in this
        catalogue that matched ``wanted``.  Empty when the read is
        stale, the catalogue holds none of the requested ids, or the
        read fails.
    """
    if stale_check is not None and stale_check():
        return {}

    logger.info(
        "Loading preview: reading catalogue {}/{} ({}) for {} id(s)",
        index,
        total,
        os.path.basename(cat_file),
        len(wanted),
    )
    if cat_file.endswith(".parquet"):
        # Push the SourceID filter down to pyarrow so row groups that
        # don't contain any requested id are skipped at I/O time —
        # preview always asks for a handful of ids, so this turns a
        # full-catalogue scan into a sub-second read.
        df = _read_parquet_filtered(Path(cat_file), wanted)
    else:
        df = pd.read_csv(cat_file)

    id_col = _find_id_column(df)
    mask = df[id_col].astype(str).isin(wanted)
    batch_df = df[mask]
    if batch_df.empty:
        return {}

    return _create_batch_cutouts(cfg, batch_df, id_col)


def _create_batch_cutouts(cfg: DotMap, batch_df, id_col: str) -> dict[str, np.ndarray]:
    """Create cutouts from a multi-row catalogue DataFrame.

    Uses Cutana's in-process ``create_cutouts_direct`` path.  The callers of
    this function are all UI-side preview flows (image-detail, sample grid,
    batch preview) that request at most a handful of sources, so the direct
    path is always the right choice here — streaming orchestrators are only
    worthwhile in the training / full-prediction pipelines where catalogues
    are orders of magnitude larger and go through the dedicated subprocess
    scripts instead.

    Args:
        cfg: Full prediction config.
        batch_df: DataFrame with catalogue rows.
        id_col: Name of the source ID column.

    Returns:
        Dict mapping source_id to HWC uint8 image.

    Raises:
        RuntimeError: If Cutana returns a cutout for a source id outside
            *batch_df*, returns metadata without a matching cutout array, or
            returns an empty/invalid cutout.  These are Cutana API contract
            violations — the caller's ``try/except`` downgrades them to a
            warning so the UI stays responsive, but they must not be silently
            dropped from the result dict.
    """
    try:
        if cfg.fitsbolt_cfg is None:
            cfg = get_fitsbolt_config(cfg)

        batches = extract_cutouts_direct(batch_df, cfg)

        results: dict[str, np.ndarray] = {}
        expected_ids = set(batch_df[id_col].astype(str))
        for batch in batches:
            for sid, raw_img in _iter_batch_items(batch):
                if sid not in expected_ids:
                    raise RuntimeError(
                        f"Cutana returned cutout for unexpected source_id {sid!r}; "
                        f"requested ids were {sorted(expected_ids)}"
                    )
                combined = apply_channel_combination_to_cutana_image(raw_img, cfg)
                image = _postprocess_image(combined)
                if image is None:
                    raise RuntimeError(f"Cutana returned empty/invalid cutout for source {sid!r}")
                results[sid] = image

        return results

    except Exception:
        logger.opt(exception=True).warning("Batch cutout creation failed")
        return {}


def _postprocess_image(image: np.ndarray) -> np.ndarray | None:
    """Convert a raw cutout to HWC uint8.

    Args:
        image: Raw cutout array from Cutana.

    Returns:
        HWC uint8 numpy array, or ``None`` if the image is empty.
    """
    if image is None or image.size == 0:
        logger.warning("Empty cutout array from Cutana — corrupt or missing FITS tiles?")
        return None

    return ensure_uint8_hwc(image)


def _find_id_column(df) -> str:
    """Find the source ID column name in a catalogue DataFrame.

    Args:
        df: Catalogue DataFrame.

    Returns:
        Column name string.

    Raises:
        ValueError: If no recognised source ID column is found.
    """
    for col in ("SourceID", "source_id", "SOURCE_ID", "SOURCEID"):
        if col in df.columns:
            return col
    raise ValueError(
        f"No source ID column found in catalogue. "
        f"Expected one of (SourceID, source_id, SOURCE_ID, SOURCEID), "
        f"got columns: {list(df.columns)}"
    )


def _find_source_row(cat_file: str, source_id: str):
    """Find a row matching *source_id* in the given catalogue file.

    Args:
        cat_file: Path to a CSV or Parquet catalogue.
        source_id: Source identifier to match.

    Returns:
        Single-row DataFrame or ``None`` if not found.
    """
    if cat_file.endswith(".parquet"):
        df = pd.read_parquet(cat_file)
    else:
        df = pd.read_csv(cat_file)

    try:
        id_col = _find_id_column(df)
    except ValueError:
        logger.debug("No source ID column in catalogue {}", cat_file)
        return None

    match = df[df[id_col].astype(str) == str(source_id)]
    if not match.empty:
        return match.head(1)

    return None


def _create_cutout(cfg: DotMap, source_df) -> np.ndarray | None:
    """Create a cutout from a single-row catalogue using Cutana.

    Uses Cutana's in-process ``create_cutouts_direct`` — a single-source
    request has no reason to pay the orchestrator subprocess cost.

    Args:
        cfg: Full prediction config.
        source_df: Single-row DataFrame with catalogue metadata.

    Returns:
        HWC uint8 numpy array, or ``None`` on failure.
    """
    try:
        if cfg.fitsbolt_cfg is None:
            cfg = get_fitsbolt_config(cfg)

        batches = extract_cutouts_direct(source_df, cfg)
        items = (item for batch in batches for item in _iter_batch_items(batch))
        first = next(items, None)
        if first is None:
            # Clean run, no cutout → tile unreachable or no data there.
            # debug-level breadcrumb; load_single_cutout emits the warning.
            logger.debug(
                "Cutana produced no cutout for the requested source — its tile "
                "is likely unreachable or has no data at that position."
            )
            return None
        _sid, raw_img = first
        combined = apply_channel_combination_to_cutana_image(raw_img, cfg)
        return _postprocess_image(combined)

    except Exception as exc:
        logger.warning("Cutana single-cutout load failed: {}", exc)
        return None
